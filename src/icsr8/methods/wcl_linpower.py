"""WCL variant using linear-power mean instead of median for per-scan aggregation.

RSSI_agg = 10·log10(mean(10^(r_i/10)))

Why-not (extending _BaselineMethod vs. clean class):
    The parent _BaselineMethod expects candidate_medians() pipeline.
    For linear_power aggregation, we need candidate_aggregate(..., aggregation="linear_power")
    instead. Sharing code via inheritance would require parameterizing _BaselineMethod's
    _estimator + aggregation, creating a leaky abstraction. A clean standalone class
    is simpler and keeps the pattern explicit and localized.
"""

from __future__ import annotations

import pandas as pd

from icsr8.estimators import estimate_wcl
from icsr8.fingerprint import candidate_aggregate, reproduction_fingerprint
from icsr8.methods import register
from icsr8.methods.base import Method


@register
class _WCLLinpower(Method):
    name = "wcl_linpower"
    uses_geometry = True

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "_WCLLinpower":
        # Why not consume location_coords: ベースラインは学習を持たないため参照点
        #   座標を必要としない。統一シグネチャを満たすために受け取るだけで無視する。
        self._ap_coords = ap_coords
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        # Use linear-power aggregation instead of median
        candidates = candidate_aggregate(
            test_scans,
            self._ap_coords,
            aggregation="linear_power",
        )
        fp = reproduction_fingerprint(candidates)
        return estimate_wcl(fp)
