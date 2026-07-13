"""RSSI + 検出率の複合 fingerprint 距離（doc/improvement_methods_note.txt 手法19）。

各 train 地点 l と鍵 k=(ap_name, band) について _tier4.location_feature_stats の
μ_{l,k}/σ_{l,k}（MIN_COUNT フロア済み）・q̂_{l,k} を学習し、query 側は自身の
median RSSI r と Beta(1,1) 平滑化検出率 d̂ = (n_q_detect+1)/(n_q_scans+2) を求めて
距離を測る（note 手法19 の Σ_a：全 train 鍵で和を取る）:

    d(q,l) = Σ_{k∈train鍵} (r_k-μ_{l,k})²/(σ_{l,k}²+ε) / #train鍵
             + η · Σ_{k∈train鍵} (d̂_k-q̂_{l,k})² / #train鍵

未検出鍵の r/μ は NON_DETECT_DBM、未適格 σ は SIGMA_MIN_DB で埋める（下記
Why-not「双方検出だけに絞らない」参照）。

K 近傍・重み付け centroid は _tier4.knn_estimate に委譲する。η・k・weighting は
train_scans のみを使う inner CV（location 単位 5-fold）でグリッド選択する。

Why not doc の φ_{l,a}=[μ,median,σ,q,log(1+C)] をそのまま距離へ組み込む:
    median は μ と同じ量（location_feature_stats の μ は既に rssi_median）を
    2 度距離へ加えるだけの重複項になる。log(1+C) の C（検出回数）が運ぶ情報は
    q̂=(n_detect+1)/(n_scans+2) の Beta(1,1) 平滑化に既に集約済みで、log(1+C) を
    さらに加えると同じ検出回数の証拠を 2 系統（q̂ の項と log(1+C) の項）で
    二重に数えてしまう。距離式は q̂ 由来の 1 項のみで検出回数の情報を表す。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from icsr8.constants import NON_DETECT_DBM, SIGMA_MIN_DB
from icsr8.methods import register
from icsr8.methods.base import Method
from icsr8.methods._tier4 import (
    FeatureStats,
    knn_estimate,
    location_feature_stats,
    query_feature_stats,
    select_by_inner_cv,
)

ETA_GRID: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
K_GRID: tuple[int, ...] = (3, 5)
WEIGHTING_GRID: tuple[str, ...] = ("inv", "inv_sq")

# ε = SIGMA_MIN_DB²（契約の式をそのまま実装。σ は既に SIGMA_MIN_DB でフロア
# 済みなので二重フロアではあるが、契約が明示する式を変えない）。
_EPS = SIGMA_MIN_DB**2

Key = tuple[str, str]

_CANDIDATES: list[tuple[float, int, str]] = [
    (eta, k, weighting)
    for eta in ETA_GRID
    for k in K_GRID
    for weighting in WEIGHTING_GRID
]


def _query_vectors(
    query_scans: pd.DataFrame, keys: list[Key]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """location_p -> (r_vec, dhat_vec)。どちらも train 鍵空間 `keys` に整列する。

    r_vec は未検出鍵で NaN、dhat_vec は Beta(1,1) 平滑化検出率で全鍵に定義される
    （未検出鍵は n_detect=0 として評価される）。整列・pivot は共有の
    _tier4.query_feature_stats に委譲し、ここでは Beta 平滑化のみ担う。
    """
    qs = query_feature_stats(query_scans, keys)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, loc in enumerate(qs.locs):
        r_vec = qs.median[i]  # NaN preserved for undetected keys
        dhat_vec = (qs.n_detect[i].astype(float) + 1.0) / (qs.n_scans[i] + 2.0)
        out[loc] = (r_vec, dhat_vec)
    return out


def _distances(
    stats: FeatureStats, r_vec: np.ndarray, dhat_vec: np.ndarray, eta: float
) -> np.ndarray:
    """query 1 地点に対する全 train 地点への d(q,l) を返す（stats.mu.index 順）。"""
    mu = stats.mu.to_numpy(dtype=float)
    sigma = stats.sigma.to_numpy(dtype=float)
    qhat = stats.qhat.to_numpy(dtype=float)
    n_locs, n_keys = mu.shape

    # Why keep the n_keys==0 guard: term1/det とも /n_keys で割るので、train 鍵空間が
    #   空だと 0/0=NaN。鍵ゼロなら全候補の情報量が等しく、同距離 0 を返す
    #   （knn_estimate の stable sort が決定的に順序を決める）。
    if n_keys == 0:
        return np.zeros(n_locs)

    # Why not 双方検出だけに絞る + /#双方検出 で割る（旧実装）: それは note 手法19 の
    #   Σ_a（全 AP で和）から逸脱し、実データで崩壊させる。共通検出鍵だけを 1 鍵あたり
    #   平均へ正規化すると、query と鍵を 1 個しか共有しない遠方地点でもその 1 鍵が
    #   たまたま一致すれば term1≈0 になり、18 鍵を共有する真の地点（平均残差>0）に
    #   勝ってしまう。実測: F→B の全 catastrophic failure で当選近傍の共通検出数=1、
    #   Ave 22.7 m。fisher_wknn/wknn と同じく未検出鍵を NON_DETECT_DBM で埋めて
    #   全鍵で和を取り「query が強く見る AP を見ない候補」に罰則を課すと Ave≈1.7 m。
    #   未適格 σ（n_detect<MIN_COUNT）は SIGMA_MIN_DB で埋める。/n_keys は候補間で
    #   一定なので順位に無影響（スケールを term1/det 間で揃えるためだけの定数除算）。
    mu_f = np.where(np.isnan(mu), NON_DETECT_DBM, mu)

    # Why SIGMA_MIN_DB for the n_detect<MIN_COUNT NaN sigma (not a wider fallback):
    #   location_feature_stats は n_detect<MIN_COUNT の鍵の σ を NaN のまま返す
    #   （生の std がわずか 1〜2 サンプルで不安定なため）。それをここでも
    #   SIGMA_MIN_DB（= elig 側の σ フロア値そのもの）で埋めるのは、少数観測を
    #   「高精度」扱いする統計的過信になり得るが、非検出を NON_DETECT_DBM で
    #   埋める上の mu_f と同じ support-mismatch penalty の思想（wknn/fisher_wknn
    #   の MIN_COUNT フロアと整合）で意図した設計。効果は大きい: 典型的な検出
    #   RSSI(~-50dB) の鍵が片側だけ非検出で NON_DETECT_DBM(-100) 埋めになると、
    #   その 1 鍵だけで (50)^2/(SIGMA_MIN_DB**2+_EPS=2) = 1250 となり、η 項が
    #   取り得る最大寄与（det<=1 なので eta<=4.0 grid 上限）を軽く支配する
    #   スケールだと認識の上で採用。保守的な pooled variance（近傍地点や全体
    #   分散からの縮小推定）への置換は future work。
    sig_f = np.where(np.isnan(sigma), SIGMA_MIN_DB, sigma)
    r_f = np.where(np.isnan(r_vec), NON_DETECT_DBM, r_vec)

    term1 = ((r_f[None, :] - mu_f) ** 2 / (sig_f**2 + _EPS)).sum(axis=1) / n_keys
    det = ((dhat_vec[None, :] - qhat) ** 2).sum(axis=1) / n_keys
    return term1 + eta * det


def _predict_from_stats(
    stats: FeatureStats,
    keys: list[Key],
    location_coords: pd.DataFrame,
    query_scans: pd.DataFrame,
    eta: float,
    k: int,
    weighting: str,
) -> pd.DataFrame:
    locs = list(stats.mu.index)
    coords = (
        location_coords.set_index("location_p")
        .reindex(locs)[["x", "y"]]
        .to_numpy(dtype=float)
    )
    qvecs = _query_vectors(query_scans, keys)

    rows = []
    for loc in sorted(qvecs):
        r_vec, dhat_vec = qvecs[loc]
        dists = _distances(stats, r_vec, dhat_vec, eta)
        x, y = knn_estimate(dists, coords, k, weighting)
        rows.append({"location_p": loc, "x": x, "y": y})
    return pd.DataFrame(rows)


def _fit_predict_candidate(
    train_scans: pd.DataFrame,
    val_scans: pd.DataFrame,
    location_coords: pd.DataFrame,
    cand: tuple[float, int, str],
) -> pd.DataFrame:
    """select_by_inner_cv 用の fit_predict。fold ごとに統計を再構築し、
    inner CV の validation 地点が前処理統計に混入しないことを構造で保証する。"""
    eta, k, weighting = cand
    stats = location_feature_stats(train_scans)
    keys = list(stats.mu.columns)
    return _predict_from_stats(stats, keys, location_coords, val_scans, eta, k, weighting)


@register
class JointFp(Method):
    name = "joint_fp"
    uses_geometry = False

    def __init__(
        self,
        *,
        eta: float | None = None,
        k: int | None = None,
        weighting: str | None = None,
    ) -> None:
        given = (eta is not None, k is not None, weighting is not None)
        if any(given) and not all(given):
            raise ValueError("eta/k/weighting must be given together or all omitted")
        self.eta = eta
        self.k = k
        self.weighting = weighting
        self.selected_eta: float | None = None
        self.selected_k: int | None = None
        self.selected_weighting: str | None = None
        self._stats: FeatureStats | None = None
        self._keys: list[Key] | None = None
        self._location_coords: pd.DataFrame | None = None
        self.diagnostics_: dict = {}

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "JointFp":
        # Why not consume ap_coords: uses_geometry=False — 鍵 (ap_name, band) の
        #   識別子と指紋統計のみで動く純指紋手法。共通シグネチャを満たすため
        #   受け取って無視する。
        del ap_coords
        self._stats = location_feature_stats(train_scans)
        self._keys = list(self._stats.mu.columns)
        self._location_coords = location_coords

        if self.eta is not None:
            self.selected_eta = self.eta
            self.selected_k = self.k
            self.selected_weighting = self.weighting
            cv_scores = None
        else:
            best, scores = select_by_inner_cv(
                train_scans,
                location_coords,
                _CANDIDATES,
                _fit_predict_candidate,
                k=5,
            )
            self.selected_eta, self.selected_k, self.selected_weighting = best
            cv_scores = {
                f"eta={c[0]},k={c[1]},w={c[2]}": v for c, v in scores.items()
            }

        self.diagnostics_ = {
            "selected_eta": self.selected_eta,
            "selected_k": self.selected_k,
            "selected_weighting": self.selected_weighting,
            "cv_scores": cv_scores,
        }
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        if self._stats is None:
            raise RuntimeError("fit() must be called before predict()")
        return _predict_from_stats(
            self._stats,
            self._keys,
            self._location_coords,
            test_scans,
            self.selected_eta,
            self.selected_k,
            self.selected_weighting,
        )
