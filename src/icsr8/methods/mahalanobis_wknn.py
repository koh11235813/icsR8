"""Mahalanobis WKNN（Tier4 #14）: Ledoit-Wolf shrinkage 共分散版 WKNN。

特徴は地点別 mu 行列（NON_DETECT fill、icsr8.methods._tier4 と共有）。距離を
plain L2 から Mahalanobis d²=(r-mu)^T Σ̂^{-1} (r-mu) に置換する。cov_mode="total"
は地点別 mu の LedoitWolf 共分散、cov_mode="within" は地点等重みの残差共分散 S̄
に LedoitWolf の shrinkage 係数を適用したもの（生の標本共分散は地点数 <= 特徴
次元で必ず特異になるため）。cov_mode・K・weighting は inner CV
（icsr8.methods._tier4.select_by_inner_cv、地点単位 5-fold）で選択する。

Why not 明示的逆行列 (np.linalg.inv): 逆行列を経由すると数値誤差が Σ の
条件数に対して不安定に増幅されうる。scipy.linalg.cho_factor/cho_solve は
Cholesky 分解を 1 回だけ行い、以降の距離計算は三角行列 solve に落ちるため、
逆行列を明示的に持たずに済み、非対称化などの浮動小数点崩れも起きない。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from sklearn.covariance import LedoitWolf, ledoit_wolf_shrinkage

from icsr8.fingerprint import band_of
from icsr8.methods import register
from icsr8.methods.base import Method
from icsr8.methods._tier4 import (
    dense_matrix,
    knn_estimate,
    location_feature_stats,
    safe_cho_factor,
    select_by_inner_cv,
)

_COV_MODES: tuple[str, ...] = ("within", "total")
_K_GRID: tuple[int, ...] = (3, 5)
_WEIGHTING_GRID: tuple[str, ...] = ("inv", "inv_sq")


class _Candidate(NamedTuple):
    cov_mode: str
    k: int
    weighting: str


_GRID: tuple[_Candidate, ...] = tuple(
    _Candidate(cov_mode, k, weighting)
    for cov_mode in _COV_MODES
    for k in _K_GRID
    for weighting in _WEIGHTING_GRID
)


def _scan_level_matrix(
    scans: pd.DataFrame, keys: list[tuple[str, str]]
) -> tuple[np.ndarray, np.ndarray]:
    """(location_p, count) 粒度の密行列。'within' 共分散の残差母集団を作る。

    Why not ap_band_fingerprint を再利用: それは location 粒度に集約済みで
    scan 単位の残差を作れない。count を追加の pivot キーにするだけで足りるので
    最小限の変形として別関数にする。
    """
    work = scans.assign(band=scans["frequency"].map(band_of))
    pivot = work.pivot_table(
        index=["location_p", "count"], columns=["ap_name", "band"],
        values="rssi", aggfunc="median",
    )
    filled, _ = dense_matrix(pivot, keys=keys)
    locs = np.array([idx[0] for idx in pivot.index], dtype=int)
    return filled, locs


def _mean_within_covariance(
    scan_mat: np.ndarray,
    scan_locs: np.ndarray,
    mu_dense: np.ndarray,
    locs_order: list[int],
) -> np.ndarray:
    """地点別残差外積平均 S_l を等重み平均した S̄ を返す。

    S_l = (1/n_l) Σ_i (r_i − μ_l)(r_i − μ_l)ᵀ（μ_l は地点 l の中央値特徴）。
    地点ごとに平均を取ってから地点間で等重み平均するので、scan 数の多い地点が
    共分散を支配しない（学習単位＝地点、の契約遵守）。
    """
    loc_pos = {loc: i for i, loc in enumerate(locs_order)}
    per_loc: list[np.ndarray] = []
    for loc in locs_order:
        rows = scan_locs == loc
        n_l = int(rows.sum())
        if n_l == 0:
            continue
        resid = scan_mat[rows] - mu_dense[loc_pos[loc]]
        per_loc.append((resid.T @ resid) / n_l)
    return np.mean(per_loc, axis=0)


def _estimate_covariance(
    scans: pd.DataFrame,
    mu_dense: np.ndarray,
    locs_order: list[int],
    keys: list[tuple[str, str]],
    cov_mode: str,
) -> np.ndarray:
    if cov_mode == "total":
        return LedoitWolf().fit(mu_dense).covariance_
    if cov_mode == "within":
        scan_mat, scan_locs = _scan_level_matrix(scans, keys)
        loc_pos = {loc: i for i, loc in enumerate(locs_order)}
        row_idx = np.array([loc_pos[loc] for loc in scan_locs])
        residual = scan_mat - mu_dense[row_idx]

        s_bar = _mean_within_covariance(scan_mat, scan_locs, mu_dense, locs_order)
        # Why apply a LedoitWolf *scalar* shrinkage to the location-equal-weighted
        # S̄, rather than LedoitWolf().fit(residual).covariance_ directly: fitting on
        # the raw scan matrix weights scan-heavy locations more, breaking the
        # per-location learning-unit contract (F1). We keep S̄ as the covariance
        # structure and borrow only the shrinkage *coefficient* from the scan
        # residuals — a scalar whose dependence on scan multiplicity distorts the
        # unit far less than a full scan-weighted covariance would (documented
        # compromise: applying that λ to S̄'s own tr(S̄)/p·I target is approximate).
        lam = float(ledoit_wolf_shrinkage(residual))
        p = s_bar.shape[0]
        mu_target = float(np.trace(s_bar)) / p
        return (1.0 - lam) * s_bar + lam * mu_target * np.eye(p)
    raise ValueError(f"unknown cov_mode: {cov_mode!r}")


def _mahalanobis_sq(diffs: np.ndarray, cho) -> np.ndarray:
    solved = cho_solve(cho, diffs.T)
    return np.einsum("ij,ji->i", diffs, solved)


def _predict_with_covariance(
    mu_db: np.ndarray,
    ref_xy: np.ndarray,
    query_mat: np.ndarray,
    query_locs: list[int],
    cho,
    k: int,
    weighting: str,
) -> pd.DataFrame:
    rows = []
    for i, loc in enumerate(query_locs):
        diffs = mu_db - query_mat[i]
        d2 = _mahalanobis_sq(diffs, cho)
        # Why clip at 0: Cholesky solve 経由でも丸め誤差で d2 が僅かに負に
        # なりうる（理論上は Σ̂ が SPD なので非負）。sqrt の domain error を防ぐ。
        d2 = np.clip(d2, 0.0, None)
        x, y = knn_estimate(np.sqrt(d2), ref_xy, k=k, weighting=weighting)
        rows.append({"location_p": int(loc), "x": x, "y": y})
    return pd.DataFrame(rows)


def _cv_fit_predict(
    inner_train: pd.DataFrame,
    inner_val: pd.DataFrame,
    inner_train_coords: pd.DataFrame,
    cand: _Candidate,
) -> pd.DataFrame:
    stats = location_feature_stats(inner_train)
    mu_dense, keys = dense_matrix(stats.mu)
    locs_order = list(stats.mu.index)

    coords = inner_train_coords.set_index("location_p")
    ref_xy = coords.loc[locs_order, ["x", "y"]].to_numpy(dtype=float)

    sigma = _estimate_covariance(inner_train, mu_dense, locs_order, keys, cand.cov_mode)
    cho = safe_cho_factor(sigma)

    val_stats = location_feature_stats(inner_val)
    val_mu, _ = dense_matrix(val_stats.mu, keys=keys)
    val_locs = list(val_stats.mu.index)

    return _predict_with_covariance(mu_dense, ref_xy, val_mu, val_locs, cho, cand.k, cand.weighting)


@register
class MahalanobisWknn(Method):
    name = "mahalanobis_wknn"
    uses_geometry = False

    def __init__(self) -> None:
        self.selected_cov_mode: str | None = None
        self.selected_k: int | None = None
        self.selected_weighting: str | None = None
        self.diagnostics_: dict | None = None
        self._keys: list[tuple[str, str]] | None = None
        self._mu_db: np.ndarray | None = None
        self._ref_xy: np.ndarray | None = None
        self._cho = None

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "MahalanobisWknn":
        # Why not consume ap_coords: uses_geometry=False — 指紋 (ap_name, band)
        # のみで動く。座標未知 AP も含め全 AP を使ってよい。
        del ap_coords

        best, scores = select_by_inner_cv(
            train_scans, location_coords, list(_GRID), _cv_fit_predict, k=5
        )
        self.selected_cov_mode, self.selected_k, self.selected_weighting = best

        stats = location_feature_stats(train_scans)
        mu_dense, keys = dense_matrix(stats.mu)
        locs_order = list(stats.mu.index)
        coords = location_coords.set_index("location_p")

        self._keys = keys
        self._mu_db = mu_dense
        self._ref_xy = coords.loc[locs_order, ["x", "y"]].to_numpy(dtype=float)
        sigma = _estimate_covariance(
            train_scans, mu_dense, locs_order, keys, self.selected_cov_mode
        )
        self._cho = safe_cho_factor(sigma)

        self.diagnostics_ = {
            "selected_cov_mode": self.selected_cov_mode,
            "selected_k": self.selected_k,
            "selected_weighting": self.selected_weighting,
            "cv_scores": {
                f"{c.cov_mode}_k{c.k}_{c.weighting}": v for c, v in scores.items()
            },
        }
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        if self._cho is None:
            raise RuntimeError("fit() must be called before predict()")
        stats = location_feature_stats(test_scans)
        query_mat, _ = dense_matrix(stats.mu, keys=self._keys)
        query_locs = list(stats.mu.index)
        return _predict_with_covariance(
            self._mu_db, self._ref_xy, query_mat, query_locs,
            self._cho, self.selected_k, self.selected_weighting,
        )
