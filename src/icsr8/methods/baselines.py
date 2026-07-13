"""PBL/CLA/WCL のレジストリアダプタ。

ベースラインは学習を持たないため fit は ap_coords を保持するだけ。
predict は test_scans から再現指紋を作り、フリーズ済み estimate_* へ渡す。
"""

from __future__ import annotations

import pandas as pd

from icsr8.estimators import estimate_cla, estimate_pbl, estimate_wcl
from icsr8.fingerprint import candidate_medians, reproduction_fingerprint
from icsr8.methods import register
from icsr8.methods.base import Method


class _BaselineMethod(Method):
    uses_geometry = True
    # Why-not (staticmethod でラップ): 素の関数を class 属性に置くと instance
    #   経由アクセスで bound method 化し fp より前に self が渡ってしまう。
    _estimator = staticmethod(estimate_pbl)

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "_BaselineMethod":
        # Why not consume location_coords: ベースラインは学習を持たないため参照点
        #   座標を必要としない。統一シグネチャを満たすために受け取るだけで無視する。
        self._ap_coords = ap_coords
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        fp = reproduction_fingerprint(candidate_medians(test_scans, self._ap_coords))
        return self._estimator(fp)


@register
class _PBL(_BaselineMethod):
    name = "pbl"
    _estimator = staticmethod(estimate_pbl)


@register
class _CLA(_BaselineMethod):
    name = "cla"
    _estimator = staticmethod(estimate_cla)


@register
class _WCL(_BaselineMethod):
    name = "wcl"
    _estimator = staticmethod(estimate_wcl)
