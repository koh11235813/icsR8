"""WCL 改善手法10: 廊下から離れた部屋内設置 AP を候補プールから除外。

AP-C0-3F-04 は座標 (-1.9, -11.2) で廊下外に設置され、局所的な信号強度異常が
測定誤差を増大させる。本手法はこの AP を reproduction_fingerprint 前に候補から
フィルタし、候補集約の重複排除時点では既に消滅している状態を保証する。
"""

from __future__ import annotations

import pandas as pd

from icsr8.constants import BLACKLIST_APS
from icsr8.estimators import estimate_wcl
from icsr8.fingerprint import candidate_medians, reproduction_fingerprint
from icsr8.methods import register
from icsr8.methods.base import Method


@register
class WCLBlacklist(Method):
    name = "wcl_blacklist"
    uses_geometry = True

    def __init__(self, **kwargs) -> None:
        # Why not accept kwargs: 基底 Method の fit/predict シグネチャに合わせるため
        # のみ。本手法は追加パラメータを持たない。
        pass

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "WCLBlacklist":
        # Why not consume location_coords: ベースラインと同様、学習を持たないため
        # 参照点座標を必要としない。
        self._ap_coords = ap_coords
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        # Step 1: 候補を作成（全 AP を対象）
        candidates = candidate_medians(test_scans, self._ap_coords)

        # Step 2: ブラックリスト AP を除外
        filtered_candidates = candidates[
            ~candidates["ap_name"].isin(BLACKLIST_APS)
        ]

        # Step 3: reproduction_fingerprint で物理 AP 単位に集約
        fp = reproduction_fingerprint(filtered_candidates)

        # Step 4: WCL 推定
        return estimate_wcl(fp)
