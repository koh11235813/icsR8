"""Centered / Relative RSSI fingerprinting（手法4, doc/improvement_methods_note.txt）。

モデル r_{s,a} = μ_{l,a} + δ_s + ε の δ_s（端末 AGC・身体遮蔽等によるスキャン
全体のオフセット）を、そのスキャンで観測された鍵集合の中央値で推定して除去する
（doc/mid_report/main.tex §3.3 の δ̂ 成分の単体版）。

特徴空間は学習データの (ap_name, band) 鍵の和集合で固定し、raw 距離と centered
距離を λ で線形補間する:
    d² = λ · Σ_{a∈union}(r-μ)²  +  (1-λ) · mean_{a∈O_q∩O_l}(r̃-μ̃)²
raw 項は鍵和集合（-100 埋め）での二乗和で、欠測そのものが持つ censored な情報を
保持する。centered 項は両側が観測した鍵 O_q∩O_l のみでの平均二乗差（和でなく
平均: 重なり数が異なる候補間で比較可能にするため）で、共通観測鍵が 3 未満の候補は
raw 項を代用する（+inf にすると候補を静かに全滅させ pool を空にしうるため）。
λ は inner CV（location 単位 5-fold）で {0.0, 0.25, 0.5, 0.75, 1.0} から選択する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from icsr8.constants import NON_DETECT_DBM, RANDOM_SEED
from icsr8.fingerprint import ap_band_fingerprint
from icsr8.methods import register
from icsr8.methods.base import Method
from icsr8.protocols import iter_inner_cv

LAMBDA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
K_NEAREST = 3
MIN_COMMON = 3


def _build_matrix_masked(
    scans: pd.DataFrame, keys: list[tuple[str, str]] | None = None
) -> tuple[list[tuple[str, str]], list[int], np.ndarray, np.ndarray, np.ndarray]:
    """`_build_matrix` に観測マスクを足した版（centered 項の共通観測判定に使う）。

    Returns:
        keys, locs, raw (n_locs, n_keys), centered (n_locs, n_keys),
        observed (n_locs, n_keys, bool) — その location が実際に観測した鍵。
    """
    ab = ap_band_fingerprint(scans, ap_coords=None)
    if keys is None:
        keys = sorted(set(zip(ab["ap_name"], ab["band"])))

    pivot = ab.pivot_table(index="location_p", columns=["ap_name", "band"], values="rssi_median")
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(keys))
    # Why capture the mask before fillna: NON_DETECT_DBM (-100) は実測値と衝突
    # しうるので「観測されたか」は埋める前の NaN 有無からしか判定できない。
    observed = pivot.notna().to_numpy()
    pivot = pivot.fillna(NON_DETECT_DBM)
    locs = pivot.index.tolist()
    raw = pivot.to_numpy(dtype=float)

    # Why not median over the fixed `keys` (or over -100 fills): δ_s は「その
    # location/scan が実際に観測した鍵集合 O_s」の中央値でしか定義できない
    # （手法4）。`keys` に含まれる未観測の鍵や NON_DETECT_DBM 埋めを混ぜると
    # δ̂ が観測事実と無関係な定数で汚染される。
    own_median = ab.groupby("location_p")["rssi_median"].median().reindex(locs).to_numpy()
    centered = raw - own_median[:, None]
    return keys, locs, raw, centered, observed


def _build_matrix(
    scans: pd.DataFrame, keys: list[tuple[str, str]] | None = None
) -> tuple[list[tuple[str, str]], list[int], np.ndarray, np.ndarray]:
    """(ap_name, band) 特徴行列を作る。

    `keys=None` なら scans 自身の (ap_name, band) 和集合を特徴空間とする
    （fit 時の DB 構築）。`keys` を与えると query 側をその固定空間へ整列する
    （predict / inner CV の query 構築）。どちらの場合も中央値によるオフセット
    除去は「その location 自身が観測した鍵」に対してのみ行う。

    Returns:
        keys, locs, raw (n_locs, n_keys), centered (n_locs, n_keys)
    """
    keys, locs, raw, centered, _ = _build_matrix_masked(scans, keys)
    return keys, locs, raw, centered


def _distance_sq(
    raw_q: np.ndarray,
    cen_q: np.ndarray,
    obs_q: np.ndarray,
    raw_db: np.ndarray,
    cen_db: np.ndarray,
    obs_db: np.ndarray,
    lam: float,
) -> np.ndarray:
    # raw λ 項: 鍵和集合全体（-100 埋め）での二乗距離。一方が非検出で他方が
    # 検出、という組み合わせ自体が「その鍵で信号強度差がある」ことを示す
    # censored な情報なので、鍵和集合全体で合算する。
    raw_term = np.sum((raw_db - raw_q) ** 2, axis=1)

    # centered (1-λ) 項: 両側が観測した鍵 O_q∩O_l のみでの平均二乗差。-100 埋めを
    # 混ぜると λ=0 のオフセット不変性が壊れる（欠測鍵の centered 値が own_median の
    # シフトでずれるため、G1）。
    common = obs_db & obs_q[None, :]
    n_common = common.sum(axis=1)
    d_cen = np.where(common, cen_db - cen_q, 0.0)
    with np.errstate(invalid="ignore"):
        mean_cen = np.sum(d_cen**2, axis=1) / n_common
    # Why not +inf when |O_q∩O_l|<3: 候補を静かに全滅させ pool を空にしうる。
    # raw 項を代用すれば候補は残り、順位付けは生きる。
    cen_term = np.where(n_common >= MIN_COMMON, mean_cen, raw_term)

    return lam * raw_term + (1.0 - lam) * cen_term


def _weighted_centroid(dist_sq: np.ndarray, db_xy: np.ndarray) -> tuple[float, float]:
    # Why K=3 (固定・非チューニング): ベースライン WCL の top-3 と条件を揃え、
    # centered_fp をその改良として比較可能にする。
    order = np.argsort(dist_sq, kind="stable")[:K_NEAREST]
    w = 1.0 / (dist_sq[order] + 1e-9)
    top_xy = db_xy[order]
    x = float((w * top_xy[:, 0]).sum() / w.sum())
    y = float((w * top_xy[:, 1]).sum() / w.sum())
    return x, y


def _select_lambda(train_scans: pd.DataFrame, location_coords: pd.DataFrame) -> float:
    coords = location_coords.set_index("location_p")
    errors: dict[float, list[float]] = {lam: [] for lam in LAMBDA_GRID}

    for inner_train, inner_val in iter_inner_cv(train_scans, k=5, seed=RANDOM_SEED):
        keys, locs, raw_db, cen_db, obs_db = _build_matrix_masked(inner_train)
        _, val_locs, raw_val, cen_val, obs_val = _build_matrix_masked(inner_val, keys=keys)
        db_xy = coords.loc[locs, ["x", "y"]].to_numpy(dtype=float)
        truth_xy = coords.loc[val_locs, ["x", "y"]].to_numpy(dtype=float)

        for lam in LAMBDA_GRID:
            for i in range(len(val_locs)):
                dist_sq = _distance_sq(
                    raw_val[i], cen_val[i], obs_val[i], raw_db, cen_db, obs_db, lam
                )
                x, y = _weighted_centroid(dist_sq, db_xy)
                errors[lam].append(float(np.hypot(x - truth_xy[i, 0], y - truth_xy[i, 1])))

    mean_errors = {lam: float(np.mean(v)) for lam, v in errors.items() if v}
    # Why iterate λ descending with a strict `<`: argmin のタイは大きい λ を
    # 優先する契約なので、大きい方から走査し「厳密に」小さい誤差のときだけ
    # 更新すれば、同点は最初に見つかった（＝最大の）λ が残る。
    best_lambda = LAMBDA_GRID[-1]
    best_error = float("inf")
    for lam in sorted(LAMBDA_GRID, reverse=True):
        err = mean_errors.get(lam, float("inf"))
        if err < best_error:
            best_error = err
            best_lambda = lam
    return best_lambda


@register
class CenteredFP(Method):
    name = "centered_fp"
    uses_geometry = False

    def __init__(self, *, lambda_: float | None = None) -> None:
        self.lambda_ = lambda_
        self.selected_lambda: float | None = None
        self._keys: list[tuple[str, str]] | None = None
        self._raw_db: np.ndarray | None = None
        self._cen_db: np.ndarray | None = None
        self._obs_db: np.ndarray | None = None
        self._db_xy: np.ndarray | None = None

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "CenteredFP":
        # Why not use ap_coords: uses_geometry=False — centered_fp は (ap_name,
        # band) の識別子のみで動く純指紋手法で、座標未知 AP も含め全 AP を使う。
        # ap_coords は Method 共通シグネチャを満たすためだけに受け取り無視する。
        del ap_coords

        if self.lambda_ is not None:
            self.selected_lambda = self.lambda_
        else:
            self.selected_lambda = _select_lambda(train_scans, location_coords)

        keys, locs, raw_db, cen_db, obs_db = _build_matrix_masked(train_scans)
        coords = location_coords.set_index("location_p")
        self._keys = keys
        self._raw_db = raw_db
        self._cen_db = cen_db
        self._obs_db = obs_db
        self._db_xy = coords.loc[locs, ["x", "y"]].to_numpy(dtype=float)
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        if self.selected_lambda is None:
            raise RuntimeError("fit() must be called before predict()")

        _, locs, raw_q, cen_q, obs_q = _build_matrix_masked(test_scans, keys=self._keys)
        rows = []
        for i, loc in enumerate(locs):
            dist_sq = _distance_sq(
                raw_q[i], cen_q[i], obs_q[i],
                self._raw_db, self._cen_db, self._obs_db, self.selected_lambda,
            )
            x, y = _weighted_centroid(dist_sq, self._db_xy)
            rows.append({"location_p": int(loc), "x": x, "y": y})
        return pd.DataFrame(rows)
