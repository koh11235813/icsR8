"""廊下座標系 1D Gaussian Process radio map + セグメント階層推定。

doc/improvement_methods_note.txt 手法3（廊下 arc-length s 上での RSSI GP モデル）
＋ 手法16（segment-first, coordinate-second）= doc/mid_report/main.tex §3.2。

各 (ap_name, band) 鍵の rssi_median を弧長 s の関数として Matérn-3/2 GP で
モデル化する（壁越しの疑似相関を避けるため Euclidean ではなく廊下 geodesic ＝
弧長差を距離に用いる）。推定はまず廊下セグメント（C/C2/C3）を分類し、その
セグメント範囲内の s グリッド上で GP 尤度の事後平均を取る 2 段構成。
"""

from __future__ import annotations

from math import hypot
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve, solve_triangular
from sklearn.linear_model import LogisticRegression

from icsr8.constants import CORRIDOR_SEGMENTS, RANDOM_SEED
from icsr8.corridor import arclength_to_xy, segment_of, xy_to_arclength
from icsr8.fingerprint import ap_band_fingerprint
from icsr8.methods import register
from icsr8.methods.base import Method

# Why not reimplement the WKNN pivot: centered_fp._build_matrix は
# ap_band_fingerprint → (ap_name, band) 和集合への pivot → NON_DETECT_DBM 埋めと
# いう WKNN 特徴行列を既に構築済みで、query 側の鍵整列（train 鍵への reindex）まで
# 面倒を見る。セグメント分類器の特徴行列はこれと同一なので再利用する（契約が
# 明示的に許可する sibling private import）。
from icsr8.methods.centered_fp import _build_matrix

# 手法3/16 のデフォルトハイパーパラメータグリッド。
_DEFAULT_LENGTH_GRID: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0)
_DEFAULT_SIGMA_F_GRID: tuple[float, ...] = (2.0, 5.0, 10.0)
_DEFAULT_SIGMA_N_GRID: tuple[float, ...] = (1.0, 2.0, 3.0)

_SEGMENT_NAMES: tuple[str, ...] = ("C", "C2", "C3")


def _segment_ranges() -> dict[str, tuple[float, float]]:
    # Why not hardcode {C:(0,32),...}: 弧長の区分境界は corridor.py と同じ
    # CORRIDOR_SEGMENTS の累積長から導出し、廊下定義が変わっても drift しない。
    ranges: dict[str, tuple[float, float]] = {}
    cum = 0.0
    for name, ((ax, ay), (bx, by)) in zip(_SEGMENT_NAMES, CORRIDOR_SEGMENTS):
        seg_len = hypot(bx - ax, by - ay)
        ranges[name] = (cum, cum + seg_len)
        cum += seg_len
    return ranges


SEGMENT_RANGES: dict[str, tuple[float, float]] = _segment_ranges()

_SQRT3 = np.sqrt(3.0)
_LOG_2PI = float(np.log(2.0 * np.pi))


def _matern32(d, sigma_f: float, length: float):
    """Matérn-3/2 カーネル k(d)=σ_f²(1+√3 d/ℓ)exp(−√3 d/ℓ)、d は弧長距離。"""
    r = _SQRT3 * np.abs(np.asarray(d, dtype=float)) / length
    return sigma_f**2 * (1.0 + r) * np.exp(-r)


class _GP(NamedTuple):
    s_train: np.ndarray
    chol: np.ndarray  # lower Cholesky of K + σ_n² I
    alpha: np.ndarray  # (K + σ_n² I)⁻¹ (y − y_mean)
    sigma_f: float
    length: float
    sigma_n: float
    y_mean: float


def _fit_gp(
    s,
    y,
    *,
    length_grid: tuple[float, ...],
    sigma_f_grid: tuple[float, ...],
    sigma_n_grid: tuple[float, ...],
    jitter: float = 1e-9,
) -> _GP:
    """厳密 GP 回帰を fit し、log marginal likelihood 最大のハイパーパラメータを選ぶ。"""
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(s)
    # Why not a GLS-profiled mean per (length, sigma_f, sigma_n) candidate: y_mean
    # はここで一度だけ固定する標本平均（算術平均）で、共分散選択より前に決める。
    # n≤59・このカーネル族では候補ごとに平均を re-estimate してもランキング差は
    # 無視できるほど小さく、前処理として平均を固定した方が全候補の LML（yc の
    # 定義）を同じ基準で比較できる。意図的な簡略化として明記する。
    y_mean = float(y.mean())
    yc = y - y_mean
    dist = np.abs(s[:, None] - s[None, :])
    eye = np.eye(n)

    best: _GP | None = None
    best_lml = -np.inf
    # Why iterate (length, sigma_f, sigma_n) with strict `>`: LML の同点は反復順で
    # 最初のもの（最小 ℓ→最小 σ_f→最小 σ_n）を残す決定的タイブレーク。
    for length in length_grid:
        for sigma_f in sigma_f_grid:
            base_k = _matern32(dist, sigma_f, length)
            for sigma_n in sigma_n_grid:
                k_noise = base_k + (sigma_n**2 + jitter) * eye
                try:
                    chol = np.linalg.cholesky(k_noise)
                except np.linalg.LinAlgError:
                    continue
                alpha = cho_solve((chol, True), yc)
                lml = (
                    -0.5 * float(yc @ alpha)
                    - float(np.log(np.diag(chol)).sum())
                    - 0.5 * n * _LOG_2PI
                )
                if lml > best_lml:
                    best_lml = lml
                    best = _GP(s, chol, alpha, float(sigma_f), float(length), float(sigma_n), y_mean)

    if best is None:
        raise ValueError(
            "GP Cholesky failed for every hyperparameter combination; "
            "increase jitter or σ_n grid"
        )
    return best


def _gp_posterior(gp: _GP, s_query) -> tuple[np.ndarray, np.ndarray]:
    """GP 事後平均 μ(s)=y_mean+k_*ᵀα と潜在分散 v(s)=k(0)−k_*ᵀK⁻¹k_*（雑音を含まない）。"""
    s_query = np.asarray(s_query, dtype=float)
    dist = np.abs(s_query[:, None] - gp.s_train[None, :])
    k_star = _matern32(dist, gp.sigma_f, gp.length)  # (m, n)
    mu = gp.y_mean + k_star @ gp.alpha
    v_solve = solve_triangular(gp.chol, k_star.T, lower=True)  # (n, m)
    v = gp.sigma_f**2 - np.sum(v_solve**2, axis=0)
    v = np.maximum(v, 0.0)
    return mu, v


@register
class GpCorridor(Method):
    name = "gp_corridor"
    uses_geometry = False

    def __init__(
        self,
        *,
        grid_step: float = 0.5,
        min_locations: int = 5,
        length_grid: tuple[float, ...] | None = None,
        sigma_f_grid: tuple[float, ...] | None = None,
        sigma_n_grid: tuple[float, ...] | None = None,
    ) -> None:
        self.grid_step = grid_step
        self.min_locations = min_locations
        self.length_grid = tuple(length_grid) if length_grid is not None else _DEFAULT_LENGTH_GRID
        self.sigma_f_grid = tuple(sigma_f_grid) if sigma_f_grid is not None else _DEFAULT_SIGMA_F_GRID
        self.sigma_n_grid = tuple(sigma_n_grid) if sigma_n_grid is not None else _DEFAULT_SIGMA_N_GRID

        # 学習後に公表される属性（最終レポートが集計する）。
        self.gp_params: dict[tuple[str, str], dict[str, float]] = {}
        self.segment_train_accuracy: float | None = None
        self.fallback_count: int = 0
        self.last_predictions_: pd.DataFrame | None = None

        self._gps: dict[tuple[str, str], _GP] = {}
        self._clf: LogisticRegression | None = None
        self._clf_keys: list[tuple[str, str]] | None = None

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "GpCorridor":
        # Why not use ap_coords: uses_geometry=False — 廊下弧長と (ap_name, band)
        # 指紋のみで動く。ap_coords は共通シグネチャ充足のためだけに受け取り無視する。
        del ap_coords

        coords = location_coords.set_index("location_p")
        s_by_loc = {
            int(loc): xy_to_arclength(float(row.x), float(row.y))
            for loc, row in coords.iterrows()
        }

        # 各 (ap_name, band) 鍵の GP: ≥ min_locations 地点で検出された鍵のみ fit。
        ab = ap_band_fingerprint(train_scans)
        self._gps = {}
        self.gp_params = {}
        for (ap_name, band), grp in ab.groupby(["ap_name", "band"], sort=True):
            if grp["location_p"].nunique() < self.min_locations:
                continue
            s = grp["location_p"].map(s_by_loc).to_numpy(dtype=float)
            y = grp["rssi_median"].to_numpy(dtype=float)
            gp = _fit_gp(
                s, y,
                length_grid=self.length_grid,
                sigma_f_grid=self.sigma_f_grid,
                sigma_n_grid=self.sigma_n_grid,
            )
            key = (str(ap_name), str(band))
            self._gps[key] = gp
            self.gp_params[key] = {
                "length": gp.length,
                "sigma_f": gp.sigma_f,
                "sigma_n": gp.sigma_n,
            }

        # セグメント分類器（手法16）: WKNN 特徴行列 → segment_of ラベル。
        keys, locs, raw, _ = _build_matrix(train_scans)
        self._clf_keys = keys
        labels = [
            segment_of(float(coords.loc[loc, "x"]), float(coords.loc[loc, "y"]))
            for loc in locs
        ]
        self._clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED).fit(raw, labels)
        self.segment_train_accuracy = float(self._clf.score(raw, labels))
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        if self._clf is None:
            raise RuntimeError("fit() must be called before predict()")

        # クエリ各地点の検出鍵 → rssi_median（自身のスキャンのみ由来）。
        ab_q = ap_band_fingerprint(test_scans)
        detected_by_loc: dict[int, dict[tuple[str, str], float]] = {}
        for loc, grp in ab_q.groupby("location_p"):
            detected_by_loc[int(loc)] = {
                (str(ap), str(band)): float(r)
                for ap, band, r in zip(grp["ap_name"], grp["band"], grp["rssi_median"])
            }

        # セグメント分類（自身のスキャンのみから作った特徴ベクトル）。
        _, q_locs, q_raw, _ = _build_matrix(test_scans, keys=self._clf_keys)
        seg_pred = self._clf.predict(q_raw)

        # 各セグメントの s グリッドと、GP 事後（雑音込み分散）を 1 度だけ前計算する。
        seg_grid: dict[str, np.ndarray] = {}
        for name, (lo, hi) in SEGMENT_RANGES.items():
            n_pts = int(round((hi - lo) / self.grid_step)) + 1
            seg_grid[name] = np.linspace(lo, hi, n_pts)
        gp_post: dict[str, dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]] = {}
        for name, grid in seg_grid.items():
            gp_post[name] = {}
            for key, gp in self._gps.items():
                mu, v = _gp_posterior(gp, grid)
                var = np.maximum(v + gp.sigma_n**2, 1e-9)
                gp_post[name][key] = (mu, var)

        self.fallback_count = 0
        rows = []
        diag = []
        for i, loc in enumerate(q_locs):
            loc = int(loc)
            z = str(seg_pred[i])
            lo, hi = SEGMENT_RANGES[z]
            grid = seg_grid[z]
            detected = detected_by_loc.get(loc, {})
            usable = [k for k in detected if k in gp_post[z]]

            if not usable:
                # Why fallback to segment midpoint: どの検出鍵も GP を持たないと
                # 尤度が定義できないため、予測セグメントの中点で埋める。
                self.fallback_count += 1
                s_hat = 0.5 * (lo + hi)
            else:
                loglik = np.zeros(len(grid))
                for key in usable:
                    mu, var = gp_post[z][key]
                    r = detected[key]
                    loglik += -0.5 * np.log(2.0 * np.pi * var) - 0.5 * (r - mu) ** 2 / var
                # Why softmax posterior mean within the segment (not a global grid):
                # セグメント段が L 字コーナーでの質量分裂を防ぐ目的（手法7 発展形）。
                w = np.exp(loglik - loglik.max())
                w /= w.sum()
                s_hat = float((w * grid).sum())

            x, y = arclength_to_xy(s_hat)
            rows.append({"location_p": loc, "x": x, "y": y})
            diag.append({"location_p": loc, "segment": z, "s_hat": s_hat})

        self.last_predictions_ = pd.DataFrame(diag)
        return pd.DataFrame(rows)
