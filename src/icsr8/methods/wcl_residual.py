"""WCL 残差学習（doc/improvement_methods_note.txt 手法18, Tier 4 #18）。

baseline WCL の弧長推定 s_wcl を、[fold 内標準化した地点別 μ 行列, s_wcl] を
特徴量とする Ridge 回帰の残差 g(・) で補正する: ŝ = clip(s_wcl + g(・))。
目標は s_true − s_wcl（WCL が既に説明できない誤差のみを学習）。

Why not fit residual on raw (x, y): 廊下は 1 次元弧長で完全に表現できる
（corridor.py）。2 次元残差より自由度が少なく、廊下外の非物理な予測を
構造的に排除できる。

Why not skip WCL and regress s_true directly on μ 行列 だけ: WCL 自体が既に
強い幾何情報（top-3 AP 重心）を持つ。s_wcl を特徴に含めることで Ridge は
「WCL のどこがどれだけ間違っているか」だけを学習すればよくなり、59 標本の
小データで μ 行列単独回帰より過学習しにくい。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from icsr8.corridor import arclength_to_xy, xy_to_arclength
from icsr8.estimators import estimate_wcl
from icsr8.fingerprint import candidate_medians, reproduction_fingerprint
from icsr8.methods import register
from icsr8.methods.base import Method
from icsr8.methods._tier4 import clip_arclength, dense_matrix, location_feature_stats, select_by_inner_cv

ALPHA_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)

# Why a floor (not raise) on zero-variance columns: real RSSI data essentially
# never has an exactly constant key across locations, but the synthetic small
# fixtures used in tests can. Flooring keeps standardization well-defined
# without silently dropping a column (feature-selection is out of scope here).
_STD_EPS = 1e-9


class _FeatureSpace(NamedTuple):
    keys: list[tuple[str, str]]
    mu_mean: np.ndarray
    mu_std: np.ndarray


def _train_feature_stats(scans: pd.DataFrame) -> _FeatureSpace:
    """μ 行列の列（(ap_name, band) 鍵）と標準化統計を `scans` のみから構築する。

    Why a module-level function (not inlined): inner CV の各 fold で
    再構築される単位を明示し、テストが標準化統計の scope を直接検証できる
    ようにする（呼び出し引数の location 集合を spy できる）。
    """
    stats = location_feature_stats(scans)
    mat, keys = dense_matrix(stats.mu)
    mu_mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    mu_std = np.where(std < _STD_EPS, 1.0, std)
    return _FeatureSpace(keys=keys, mu_mean=mu_mean, mu_std=mu_std)


def _wcl_arclength(scans: pd.DataFrame, ap_coords: pd.DataFrame) -> pd.DataFrame:
    """baseline WCL の (x, y) を弧長へ射影する（doc 手法7と同じ `xy_to_arclength`）。

    Why self-query (scans as both DB and query): WCL は学習を持たない
    memoryless 推定器（top-3 AP の重心）なので、地点自身の scans から直接
    計算できる（baselines._WCL.predict と同じ呼び出し列）。
    """
    fp = reproduction_fingerprint(candidate_medians(scans, ap_coords))
    est = estimate_wcl(fp)
    s = [xy_to_arclength(float(x), float(y)) for x, y in zip(est["x"], est["y"])]
    return pd.DataFrame({"location_p": est["location_p"].to_numpy(), "s_wcl": s})


def _feature_matrix(
    scans: pd.DataFrame, ap_coords: pd.DataFrame, space: _FeatureSpace
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """地点でソートした (locs, X, s_wcl) を返す。X = [標準化μ, s_wcl]。"""
    stats = location_feature_stats(scans)
    mat, _ = dense_matrix(stats.mu, keys=space.keys)
    z = (mat - space.mu_mean) / space.mu_std
    mu_df = pd.DataFrame(z, index=stats.mu.index)

    wcl = _wcl_arclength(scans, ap_coords).set_index("location_p")
    merged = mu_df.join(wcl, how="inner").sort_index()

    locs = [int(loc) for loc in merged.index]
    s_wcl = merged["s_wcl"].to_numpy(dtype=float)
    X = np.concatenate(
        [merged.drop(columns="s_wcl").to_numpy(dtype=float), s_wcl[:, None]], axis=1
    )
    return locs, X, s_wcl


def _true_arclength(location_coords: pd.DataFrame, locs: list[int]) -> np.ndarray:
    coords = location_coords.set_index("location_p")
    return np.array(
        [xy_to_arclength(float(coords.loc[loc, "x"]), float(coords.loc[loc, "y"])) for loc in locs]
    )


def _fit_predict(
    train_scans: pd.DataFrame,
    query_scans: pd.DataFrame,
    train_coords: pd.DataFrame,
    alpha: float,
    ap_coords: pd.DataFrame,
) -> pd.DataFrame:
    """1 fold 分の fit+predict。inner CV の候補評価と最終 predict の両方が使う。"""
    space = _train_feature_stats(train_scans)
    locs_tr, X_tr, s_wcl_tr = _feature_matrix(train_scans, ap_coords, space)
    y_tr = _true_arclength(train_coords, locs_tr) - s_wcl_tr
    model = Ridge(alpha=alpha).fit(X_tr, y_tr)

    locs_q, X_q, s_wcl_q = _feature_matrix(query_scans, ap_coords, space)
    s_hat = clip_arclength(s_wcl_q + model.predict(X_q))
    xy = [arclength_to_xy(float(s)) for s in s_hat]
    return pd.DataFrame(
        {"location_p": locs_q, "x": [p[0] for p in xy], "y": [p[1] for p in xy]}
    )


@register
class WclResidual(Method):
    name = "wcl_residual"
    uses_geometry = True

    def __init__(self, *, alpha: float | None = None) -> None:
        # Why require explicit alpha to skip CV (mirrors wknn's k/weighting):
        # unit tests need a deterministic, cheap fit without running the full
        # inner CV grid.
        self.alpha = alpha
        self.selected_alpha: float | None = None
        self.diagnostics_: dict | None = None
        self._ap_coords: pd.DataFrame | None = None
        self._space: _FeatureSpace | None = None
        self._model: Ridge | None = None

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "WclResidual":
        self._ap_coords = ap_coords

        if self.alpha is not None:
            self.selected_alpha = self.alpha
            cv_scores: dict[float, float] = {}
        else:
            # Why select_by_inner_cv (not a hand-rolled loop): it already
            # encodes the location-unit 5-fold / seed=0 / leak-free contract
            # (icsr8.methods._tier4, shared by every Tier 4 method).
            self.selected_alpha, cv_scores = select_by_inner_cv(
                train_scans,
                location_coords,
                list(ALPHA_GRID),
                lambda itr, ival, itrc, cand: _fit_predict(itr, ival, itrc, cand, ap_coords),
                k=5,
            )

        # Why not reuse _fit_predict for the final model: this branch must
        # persist (space, model) as instance state for a later, separate
        # predict() call; _fit_predict returns only a prediction DataFrame.
        self._space = _train_feature_stats(train_scans)
        locs, X, s_wcl = _feature_matrix(train_scans, ap_coords, self._space)
        y = _true_arclength(location_coords, locs) - s_wcl
        self._model = Ridge(alpha=self.selected_alpha).fit(X, y)

        self.diagnostics_ = {
            "selected_alpha": self.selected_alpha,
            "alpha_grid": list(ALPHA_GRID),
            "cv_scores": {str(k): v for k, v in cv_scores.items()},
            "n_train_locations": len(locs),
        }
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        if self._model is None or self._space is None:
            raise RuntimeError("fit() must be called before predict()")

        locs, X, s_wcl = _feature_matrix(test_scans, self._ap_coords, self._space)
        s_hat = clip_arclength(s_wcl + self._model.predict(X))
        xy = [arclength_to_xy(float(s)) for s in s_hat]
        return pd.DataFrame(
            {"location_p": locs, "x": [p[0] for p in xy], "y": [p[1] for p in xy]}
        )
