"""PLS 回帰による廊下弧長 (arc-length) 直接回帰（doc/improvement_methods_note.txt 手法15）。

地点別 (ap_name, band) μ 特徴（icsr8.methods._tier4.location_feature_stats）を
PLSRegression で廊下弧長 s = xy_to_arclength(x, y) へ直接回帰する。廊下折れ線が
1 次元多様体であることを利用し、59 次元超の特徴を少数の潜在成分に圧縮してから
1 スカラー s を予測する（WKNN の座標平均より外挿に強い動機）。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from icsr8.corridor import arclength_to_xy, xy_to_arclength
from icsr8.methods import register
from icsr8.methods.base import Method
from icsr8.methods._tier4 import (
    clip_arclength,
    dense_matrix,
    location_feature_stats,
    select_by_inner_cv,
)

N_COMPONENTS_GRID: tuple[int, ...] = (2, 3, 4, 6, 8, 10)

# Why not eps-based threshold: 列は dense_matrix の -100 埋め混じりの実測値な
# ので、丸め誤差由来の偽の非ゼロ分散はほぼ生じない。厳密 0 判定で十分。
_ZERO_VAR_EPS = 0.0


class _InterceptOnly:
    """intercept-only fallback（F5）: 常に fold 内平均弧長を返す決定的モデル。

    Why not raise instead: keys が空（全列無分散）や学習地点 <2 の縮退 fold は
    inner CV の分割や合成データで正当に発生しうる。fold を落とすより、情報ゼロ時
    の最良定数（平均弧長）を返して CV スコアに参加させる方が候補比較を歪めない。
    """

    def __init__(self, s_mean: float) -> None:
        self.s_mean = float(s_mean)

    def predict(self, X) -> np.ndarray:
        return np.full((np.asarray(X).shape[0], 1), self.s_mean)


class _FittedPls(NamedTuple):
    keys: list[tuple[str, str]]
    mean: np.ndarray
    std: np.ndarray
    pls: PLSRegression | _InterceptOnly
    n_components_actual: int
    locs: list[int]


def _fit_train_matrix(
    scans: pd.DataFrame,
) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """train 側特徴構築: μ 行列 → dense 化 → 無分散列除去 → train 統計で標準化。

    Returns: keys(除去後), mean, std, x_std, locs
    """
    mu = location_feature_stats(scans).mu
    raw, all_keys = dense_matrix(mu)
    locs = mu.index.tolist()
    col_std = raw.std(axis=0)
    keep = col_std > _ZERO_VAR_EPS
    keys = [k for k, m in zip(all_keys, keep) if m]
    kept = raw[:, keep]
    mean = kept.mean(axis=0)
    std = kept.std(axis=0)
    x_std = (kept - mean) / std
    return keys, mean, std, x_std, locs


def _transform_matrix(
    scans: pd.DataFrame,
    keys: list[tuple[str, str]],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    """query 側特徴構築: train の keys/mean/std で固定整列・標準化する。"""
    mu = location_feature_stats(scans).mu
    raw, _ = dense_matrix(mu, keys=keys)
    locs = mu.index.tolist()
    x_std = (raw - mean) / std
    return locs, x_std


def _fit_pls(
    scans: pd.DataFrame, location_coords: pd.DataFrame, n_components: int
) -> _FittedPls:
    keys, mean, std, x_std, locs = _fit_train_matrix(scans)
    coords = location_coords.set_index("location_p")
    s_train = np.array(
        [
            xy_to_arclength(float(coords.loc[loc, "x"]), float(coords.loc[loc, "y"]))
            for loc in locs
        ]
    )
    # 縮退 fold（keys 空 or 学習地点 <2）は PLS が定義できない → intercept-only。
    n_comp = min(n_components, len(locs) - 1, len(keys))
    if n_comp < 1:
        return _FittedPls(keys, mean, std, _InterceptOnly(s_train.mean()), 0, locs)
    pls = PLSRegression(n_components=n_comp, scale=False)
    pls.fit(x_std, s_train)
    return _FittedPls(keys, mean, std, pls, n_comp, locs)


def _predict_s(fitted: _FittedPls, scans: pd.DataFrame) -> tuple[list[int], np.ndarray]:
    locs, x_std = _transform_matrix(scans, fitted.keys, fitted.mean, fitted.std)
    s = clip_arclength(np.asarray(fitted.pls.predict(x_std)).ravel())
    return locs, s


def _s_to_frame(locs: list[int], s: np.ndarray) -> pd.DataFrame:
    xy = [arclength_to_xy(float(v)) for v in s]
    return pd.DataFrame(
        {
            "location_p": [int(loc) for loc in locs],
            "x": [p[0] for p in xy],
            "y": [p[1] for p in xy],
        }
    )


def _fit_predict_candidate(
    inner_train: pd.DataFrame,
    inner_val: pd.DataFrame,
    inner_train_coords: pd.DataFrame,
    cand: int,
) -> pd.DataFrame:
    fitted = _fit_pls(inner_train, inner_train_coords, cand)
    locs, s = _predict_s(fitted, inner_val)
    return _s_to_frame(locs, s)


@register
class PlsCorridor(Method):
    name = "pls_corridor"
    uses_geometry = False

    def __init__(
        self,
        *,
        n_components: int | None = None,
        component_grid: tuple[int, ...] | None = None,
    ) -> None:
        self.n_components = n_components
        self.component_grid = (
            tuple(component_grid) if component_grid is not None else N_COMPONENTS_GRID
        )
        self.diagnostics_: dict | None = None
        self._fitted: _FittedPls | None = None

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "PlsCorridor":
        # Why not use ap_coords: uses_geometry=False — (ap_name, band) 指紋と
        # 廊下弧長のみで動く。共通シグネチャ充足のためだけに受け取り無視する。
        del ap_coords

        if self.n_components is not None:
            best_c, scores = self.n_components, {}
        else:
            best_c, scores = select_by_inner_cv(
                train_scans,
                location_coords,
                list(self.component_grid),
                _fit_predict_candidate,
                k=5,
            )

        self._fitted = _fit_pls(train_scans, location_coords, best_c)
        self.diagnostics_ = {
            "component_grid": list(self.component_grid),
            "selected_n_components_candidate": best_c,
            "selected_n_components_actual": self._fitted.n_components_actual,
            "cv_scores": scores,
            "n_features_after_drop": len(self._fitted.keys),
            "n_train_locations": len(self._fitted.locs),
        }
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        if self._fitted is None:
            raise RuntimeError("fit() must be called before predict()")
        locs, s = _predict_s(self._fitted, test_scans)
        return _s_to_frame(locs, s)
