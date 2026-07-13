"""累積確率の順序回帰 → arc-length（doc/improvement_methods_note.txt 手法17）。

train 地点の弧長 s の分位点を閾値 c_1<...<c_m とし、各閾値 k で二値ロジット
1[s_l > c_k] を fold 内標準化した地点別 μ 行列に fit する（m ∈ {8,12}・
C ∈ {0.1,1,10} は inner CV で共同選択）。predict では各閾値の
p_k = P(s>c_k) を isotonic 回帰で非増加に射影したのち、生存関数の階段積分

    E[s] = s_min + Σ_{k=0}^{m} p_k・(c_{k+1} − c_k),
           p_0 := 1,  c_0 := s_min,  c_{m+1} := s_max

で期待弧長を再構成する（[s_min, c_1) 区間は「学習範囲内では必ず s > s_min」
という定義から p_0=1 を閉じ、[c_m, s_max] 区間は p_m が覆う）。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from icsr8.constants import RANDOM_SEED
from icsr8.corridor import arclength_to_xy, xy_to_arclength
from icsr8.methods import register
from icsr8.methods.base import Method
from icsr8.methods._tier4 import (
    clip_arclength,
    dense_matrix,
    location_feature_stats,
    select_by_inner_cv,
)

_M_GRID: tuple[int, ...] = (8, 12)
_C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0)
_STD_EPS = 1e-9


def _isotonic_nonincreasing(p) -> np.ndarray:
    """独立フィットの p_1..p_m を非増加列へ射影する。"""
    p = np.asarray(p, dtype=float)
    x = np.arange(len(p), dtype=float)
    iso = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0, out_of_bounds="clip")
    return iso.fit_transform(x, p)


class _ThresholdModel(NamedTuple):
    thresholds: np.ndarray  # c_1..c_m, ascending
    keys: list[tuple[str, str]]
    mean: np.ndarray
    std: np.ndarray
    s_min: float
    s_max: float
    classifiers: list  # len == m; LogisticRegression or float (degenerate fallback)


def _fit_ordinal_model(
    scans: pd.DataFrame, location_coords: pd.DataFrame, m: int, C: float
) -> _ThresholdModel:
    stats = location_feature_stats(scans)
    mat, keys = dense_matrix(stats.mu)
    locs = stats.mu.index.tolist()

    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std_safe = np.where(std < _STD_EPS, 1.0, std)
    Xz = (mat - mean) / std_safe

    coords = location_coords.set_index("location_p")
    s_vals = np.array([
        xy_to_arclength(float(coords.loc[loc, "x"]), float(coords.loc[loc, "y"]))
        for loc in locs
    ])
    s_min = float(s_vals.min())
    s_max = float(s_vals.max())

    levels = [k / (m + 1) for k in range(1, m + 1)]
    thresholds = np.quantile(s_vals, levels)

    classifiers: list = []
    for c_k in thresholds:
        y = (s_vals > c_k).astype(int)
        if y.min() == y.max():
            # Why not fit LogisticRegression on a single-class y: sklearn
            # raises on a degenerate label vector, and a constant P(s>c_k) is
            # the only information this quantile carries in that case anyway.
            classifiers.append(float(y[0]))
            continue
        # Why not a dedicated proportional-odds (cumulative-link) model:
        # a shared-slope ordinal regressor would need a new dependency
        # (statsmodels/mord) and, at ~n≤59 (fewer per inner-CV fold), the
        # gain of a common slope across m thresholds over independently
        # L2-regularized logits is marginal. Independent logits + isotonic
        # projection recovers monotonicity using sklearn (already a
        # dependency) without committing to the proportional-odds assumption.
        # Why not pass penalty="l2" explicitly: it is sklearn's default and
        # this sklearn version (>=1.8) deprecates the explicit spelling in
        # favor of l1_ratio; the fit is L2-regularized either way.
        clf = LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_SEED)
        clf.fit(Xz, y)
        classifiers.append(clf)

    return _ThresholdModel(
        thresholds=thresholds, keys=keys, mean=mean, std=std_safe,
        s_min=s_min, s_max=s_max, classifiers=classifiers,
    )


def _predict_s_hat(model: _ThresholdModel, scans: pd.DataFrame) -> dict[int, float]:
    stats = location_feature_stats(scans)
    mat, _ = dense_matrix(stats.mu, keys=model.keys)
    locs = stats.mu.index.tolist()
    Xz = (mat - model.mean) / model.std

    m = len(model.classifiers)
    raw_p = np.empty((len(locs), m), dtype=float)
    for k, clf in enumerate(model.classifiers):
        if isinstance(clf, float):
            raw_p[:, k] = clf
        else:
            col = list(clf.classes_).index(1)
            raw_p[:, k] = clf.predict_proba(Xz)[:, col]

    c = model.thresholds
    widths = np.empty(m + 1)
    widths[0] = c[0] - model.s_min
    widths[1:m] = c[1:] - c[:-1]
    widths[m] = model.s_max - c[-1]

    result: dict[int, float] = {}
    for i, loc in enumerate(locs):
        p_proj = _isotonic_nonincreasing(raw_p[i])
        p_full = np.concatenate(([1.0], p_proj))
        s_hat = model.s_min + float(np.dot(p_full, widths))
        result[int(loc)] = clip_arclength(s_hat)
    return result


def _s_hat_to_xy_rows(s_hats: dict[int, float]) -> pd.DataFrame:
    rows = []
    for loc, s in s_hats.items():
        x, y = arclength_to_xy(s)
        rows.append({"location_p": loc, "x": x, "y": y})
    return pd.DataFrame(rows)


def _fit_predict_candidate(inner_train, inner_val, inner_train_coords, cand):
    m, C = cand
    model = _fit_ordinal_model(inner_train, inner_train_coords, m, C)
    return _s_hat_to_xy_rows(_predict_s_hat(model, inner_val))


@register
class OrdinalCorridor(Method):
    name = "ordinal_corridor"
    uses_geometry = False

    def __init__(self, *, m: int | None = None, C: float | None = None) -> None:
        # Why all-or-none (F11, JointFp と同契約): 片方だけの指定は「残りをどの
        # グリッドから選ぶか」が未定義で、黙って CV に落ちると指定が無視される。
        given = (m is not None, C is not None)
        if any(given) and not all(given):
            raise ValueError("m/C must be given together or both omitted")
        self.m = m
        self.C = C
        self.diagnostics_: dict = {}
        self._model: _ThresholdModel | None = None

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "OrdinalCorridor":
        # Why not use ap_coords: uses_geometry=False — 弧長 s と (ap_name,
        # band) 指紋のみで動く。ap_coords は共通シグネチャ充足のためだけに
        # 受け取り無視する。
        del ap_coords

        if self.m is not None and self.C is not None:
            selected_m, selected_C, scores = self.m, self.C, {}
        else:
            candidates = [(m, C) for m in _M_GRID for C in _C_GRID]
            (selected_m, selected_C), scores = select_by_inner_cv(
                train_scans, location_coords, candidates, _fit_predict_candidate,
            )

        self._model = _fit_ordinal_model(train_scans, location_coords, selected_m, selected_C)
        self.diagnostics_ = {
            "selected_m": selected_m,
            "selected_C": selected_C,
            "cv_scores": scores,
            "n_degenerate_thresholds": sum(
                isinstance(clf, float) for clf in self._model.classifiers
            ),
        }
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("fit() must be called before predict()")
        return _s_hat_to_xy_rows(_predict_s_hat(self._model, test_scans))
