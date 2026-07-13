"""Fisher スコア AP 選択 → WKNN（Tier4 手法#13）。

key=(ap_name, band) ごとに F = Var_l(mu_l) / mean_l(sigma_l^2) を計算し、
上位 M key に制限した特徴空間で通常の重み付き kNN を行う。M / k / weighting は
inner CV（location 単位 5-fold, icsr8.protocols.iter_inner_cv）でグリッド選択する。

Why not mRMR（相互情報量による冗長性抑制）: 59 地点しかない本データセットでは
mutual information の推定自体が分散過大で不安定になり、Fisher score のような
分散比よりかえって選択を不安定化させる。単変量 Fisher score のみで十分な
差別化が得られるため採用しない。

Why not band 排他（AP1 台につき最良 band のみ残す等）: ap_band_fingerprint が
BSSID 違いの重複 SSID を既に band 単位へ集約済みであり、2.4G/5G/6G は同一 AP
でも別 key として独立に検出特性・分散を持つ。band を人為的に間引くと、Fisher
score が正しく高評価するはずの安定 band を落としかねない。
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from icsr8.methods import register
from icsr8.methods.base import Method
from icsr8.methods._tier4 import (
    Key,
    dense_matrix,
    knn_estimate,
    location_feature_stats,
    select_by_inner_cv,
)

M_GRID: tuple[object, ...] = (8, 16, 24, 32, 48, "all")
K_GRID: tuple[int, ...] = (3, 5)
WEIGHTING_GRID: tuple[str, ...] = ("inv", "inv_sq")

# finite sigma（n_detect >= MIN_COUNT で floor 済み）の地点数がこれ未満の key は
# 分散推定が不安定なため Fisher score 対象外。分子 Var も同じ地点集合で計算する
# （F6: eligibility と sigma 母集団の地点 mask を一致させる）。
_MIN_DETECT_LOCS = 3

Candidate = tuple[object, int, str]  # (M, k, weighting)

# Why itertools.product over nested loops: グリッド走査順 (M 昇順 -> k 昇順 ->
# weighting 昇順) をそのまま select_by_inner_cv のタイブレーク（先頭優先）に
# 使うため、宣言順を明示的に固定する。
_CANDIDATES: list[Candidate] = list(itertools.product(M_GRID, K_GRID, WEIGHTING_GRID))


def _fisher_scores(stats) -> dict[Key, float]:
    """適格 key の Fisher score を返す（finite sigma 地点数 < 3 の key は除外）。

    Why one mask for both numerator and denominator: mu だけ検出地点全体で分散を
    取ると、σ が未定義（scan 1–2 回）の地点の不安定な median が分子だけを歪め、
    分子と分母が別の地点母集団を測ることになる。eligibility・Var・mean(σ²) を
    全て「finite sigma の地点集合」に統一する。
    """
    mu = stats.mu
    sigma = stats.sigma
    scores: dict[Key, float] = {}
    for key in mu.columns:
        sigma_col = sigma[key].to_numpy(dtype=float)
        eligible = ~np.isnan(sigma_col)
        if int(eligible.sum()) < _MIN_DETECT_LOCS:
            continue
        mu_col = mu[key].to_numpy(dtype=float)
        var_mu = float(np.var(mu_col[eligible]))  # ddof=0: sigma 側の population std と揃える
        mean_sigma_sq = float(np.mean(sigma_col[eligible] ** 2))
        scores[key] = var_mu / mean_sigma_sq
    return scores


def _select_keys(stats, m: object) -> list[Key]:
    """F 降順で上位 M key を選ぶ（同点は ap_name, band 昇順で決定的に解決）。"""
    scores = _fisher_scores(stats)
    ranked = sorted(scores, key=lambda key: (-scores[key], key[0], key[1]))
    if m == "all":
        return ranked
    return ranked[: int(m)]


def _predict_from_db(
    db_matrix: np.ndarray,
    db_xy: np.ndarray,
    query_matrix: np.ndarray,
    query_locs: list[int],
    k: int,
    weighting: str,
) -> pd.DataFrame:
    rows = []
    for i, loc in enumerate(query_locs):
        dist = np.linalg.norm(db_matrix - query_matrix[i], axis=1)
        x, y = knn_estimate(dist, db_xy, k, weighting)
        rows.append({"location_p": int(loc), "x": x, "y": y})
    return pd.DataFrame(rows)


def _fit_predict(
    inner_train: pd.DataFrame,
    inner_val: pd.DataFrame,
    inner_train_coords: pd.DataFrame,
    cand: Candidate,
) -> pd.DataFrame:
    m, k, weighting = cand
    train_stats = location_feature_stats(inner_train)
    keys = _select_keys(train_stats, m)
    db_matrix, keys = dense_matrix(train_stats.mu, keys=keys)
    db_locs = list(train_stats.mu.index)
    coords = inner_train_coords.set_index("location_p")
    db_xy = coords.loc[db_locs, ["x", "y"]].to_numpy(dtype=float)

    val_stats = location_feature_stats(inner_val)
    val_matrix, _ = dense_matrix(val_stats.mu, keys=keys)
    val_locs = list(val_stats.mu.index)

    return _predict_from_db(db_matrix, db_xy, val_matrix, val_locs, k, weighting)


@register
class FisherWknn(Method):
    name = "fisher_wknn"
    uses_geometry = False

    def __init__(self) -> None:
        self.diagnostics_: dict = {}
        self.selected_m: object | None = None
        self.selected_k: int | None = None
        self.selected_weighting: str | None = None
        self._keys: list[Key] | None = None
        self._db_matrix: np.ndarray | None = None
        self._db_xy: np.ndarray | None = None
        self._db_locs: list[int] | None = None

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "FisherWknn":
        # Why not consume ap_coords: uses_geometry=False -- Fisher score は
        #   fingerprint 統計のみで定義され、座標未知 AP も含め全 key を評価対象にできる。
        del ap_coords

        best, scores = select_by_inner_cv(
            train_scans, location_coords, _CANDIDATES, _fit_predict, k=5
        )
        self.selected_m, self.selected_k, self.selected_weighting = best

        stats = location_feature_stats(train_scans)
        keys = _select_keys(stats, self.selected_m)
        db_matrix, keys = dense_matrix(stats.mu, keys=keys)
        db_locs = list(stats.mu.index)
        coords = location_coords.set_index("location_p")

        self._keys = keys
        self._db_matrix = db_matrix
        self._db_locs = db_locs
        self._db_xy = coords.loc[db_locs, ["x", "y"]].to_numpy(dtype=float)

        self.diagnostics_ = {
            "selected_M": self.selected_m,
            "selected_k": self.selected_k,
            "selected_weighting": self.selected_weighting,
            "n_selected_keys": len(keys),
            "cv_best_score": scores[best],
        }
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        if self._db_matrix is None:
            raise RuntimeError("fit() must be called before predict()")

        query_stats = location_feature_stats(test_scans)
        query_matrix, _ = dense_matrix(query_stats.mu, keys=self._keys)
        query_locs = list(query_stats.mu.index)

        return _predict_from_db(
            self._db_matrix, self._db_xy, query_matrix, query_locs,
            self.selected_k, self.selected_weighting,
        )
