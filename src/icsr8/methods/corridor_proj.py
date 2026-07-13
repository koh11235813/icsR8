"""廊下射影の後処理（doc/improvement_methods_note.txt 手法7）。

推定 (x, y) を icsr8.corridor.project_to_corridor で廊下の折れ線へ射影する。
apply_corridor_projection は任意メソッドの出力に適用できる純関数として Phase-4
harness から再利用され、"wcl_corridor" は baseline WCL にこれを適用した
登録メソッドとして提供する。
"""

from __future__ import annotations

import pandas as pd

from icsr8.corridor import project_to_corridor
from icsr8.estimators import estimate_wcl
from icsr8.fingerprint import candidate_medians, reproduction_fingerprint
from icsr8.methods import register
from icsr8.methods.base import Method


def apply_corridor_projection(estimates: pd.DataFrame) -> pd.DataFrame:
    """estimates の各 (x, y) を廊下上へ射影する。他の列・行順は保持する。"""
    projected = estimates.copy()
    xy = [project_to_corridor(x, y) for x, y in zip(estimates["x"], estimates["y"])]
    projected["x"] = [px for px, _ in xy]
    projected["y"] = [py for _, py in xy]
    return projected


@register
class WclCorridor(Method):
    name = "wcl_corridor"
    uses_geometry = True

    def __init__(self) -> None:
        self._ap_coords: pd.DataFrame | None = None

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "WclCorridor":
        # Why not consume location_coords: 廊下射影は事後処理であり、baseline WCL
        #   自体が学習を持たないため参照点座標は不要。
        self._ap_coords = ap_coords
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        if self._ap_coords is None:
            raise RuntimeError("fit() must be called before predict()")
        fp = reproduction_fingerprint(candidate_medians(test_scans, self._ap_coords))
        return apply_corridor_projection(estimate_wcl(fp))
