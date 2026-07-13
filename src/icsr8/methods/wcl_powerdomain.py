"""WCL powerdomain 変種 — 数学的に baseline WCL と等価な Tier 3 ablation。

ベースライン WCL の重み: w = 10^((rssi - rssi_min_top3) / 10)
wcl_powerdomain の重み:  w = 10^(rssi / 10)

数学的等価性:
    正規化ステップ Σ(w·P) / Σ(w) では、定数因子 10^(-rssi_min/10) が
    分子分母の両方に現れるため相殺される。つまり:
        Σ(10^((rssi-rssi_min)/10) · P) / Σ(10^((rssi-rssi_min)/10))
      = Σ(10^(rssi/10) · 10^(-rssi_min/10) · P) / Σ(10^(rssi/10) · 10^(-rssi_min/10))
      = Σ(10^(rssi/10) · P) / Σ(10^(rssi/10))  ← 定数因子が相殺
    よって両者は同一の推定値を返す。

ただし式が違うので浮動小数点演算の順序の違いで微小な誤差が生じる可能性がある。

本メソッドは Tier 3 (doc/improvement_methods_note.txt 手法8) の
等価性検証用に保持される。改善は期待されない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from icsr8.estimators import select_top_k
from icsr8.fingerprint import candidate_medians, reproduction_fingerprint
from icsr8.methods import register
from icsr8.methods.base import Method


def _wcl_powerdomain_one(fp: pd.DataFrame, **kw) -> tuple[float, float]:
    """WCL with powerdomain weights: w = 10^(rssi/10) (no min-subtraction)."""
    if len(fp) < 3:
        loc = fp["location_p"].iloc[0] if len(fp) else "?"
        raise ValueError(
            f"WCL requires 3 candidates per location; "
            f"location_p={loc} has only {len(fp)}"
        )

    top = select_top_k(fp, 3, **kw)
    # Why not subtract rssi_min: constant factor cancels in normalization.
    #   We use w = 10^(rssi/10) directly to show the equivalence to baseline WCL.
    weights = np.power(10.0, top["rssi_median"].to_numpy() / 10.0)
    wsum = weights.sum()
    x = float((weights * top["x"].to_numpy()).sum() / wsum)
    y = float((weights * top["y"].to_numpy()).sum() / wsum)
    return x, y


def estimate_wcl_powerdomain(fp: pd.DataFrame, **kw) -> pd.DataFrame:
    """Apply _wcl_powerdomain_one per location_p."""
    rows = []
    for loc, group in fp.groupby("location_p", sort=True):
        x, y = _wcl_powerdomain_one(group, **kw)
        rows.append({"location_p": int(loc), "x": x, "y": y})
    return pd.DataFrame(rows)


@register
class WCLPowerdomain(Method):
    """WCL Powerdomain variant (mathematically identical to baseline WCL).

    Uses w = 10^(rssi/10) directly instead of w = 10^((rssi-rssi_min)/10).
    The constant factor 10^(-rssi_min/10) cancels in the normalization step,
    producing identical estimates.

    Tier 3 ablation to verify equivalence; no improvement is claimed.
    """

    name = "wcl_powerdomain"
    uses_geometry = True

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "WCLPowerdomain":
        # Why not consume location_coords: no learning; adapter only.
        self._ap_coords = ap_coords
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        fp = reproduction_fingerprint(candidate_medians(test_scans, self._ap_coords))
        return estimate_wcl_powerdomain(fp)
