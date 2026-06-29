"""Raw scans から候補 (candidate) を作る集約層。

`candidate_medians`
    1 行 = (location_p, ap_name, ssid, frequency) の RSSI 中央値 + AP 座標。
    `restrict_to_known_aps=True` で 3F の既知座標 AP のみ残す。

`reproduction_fingerprint`
    公表ベースライン値の再現用に物理 AP 単位へ集約する。
    1. 名前から AP 棟 (wing) を抽出し、`allowed_wings` に含まれるものだけ残す。
       公表ベースラインは C 棟群（C0 / C2 / C3）のみで C1 棟 AP を除外している
       （仕様書に明記なし、estimation_result_C3F.xlsx の P1 CLA = (20.0, 0.3) が
       AP-C0-3F-01/02/03 の centroid であることから判明）。
    2. location_p × ap_name で最強の rssi_median を取り、1 物理 AP 1 行に正規化。
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

DEFAULT_REPRODUCTION_WINGS: frozenset[str] = frozenset({"C0", "C2", "C3"})


def _wing_of(ap_name: str) -> str:
    return ap_name.split("-", 2)[1]


def candidate_medians(
    scans: pd.DataFrame,
    ap_coords: pd.DataFrame,
    *,
    restrict_to_known_aps: bool = True,
) -> pd.DataFrame:
    grouped = (
        scans.groupby(["location_p", "ap_name", "ssid", "frequency"], as_index=False)
        .agg(rssi_median=("rssi", "median"))
    )

    join = grouped.merge(
        ap_coords[["ap_name", "x", "y"]],
        on="ap_name",
        how="left" if not restrict_to_known_aps else "inner",
    )

    return join[["location_p", "ap_name", "ssid", "frequency", "rssi_median", "x", "y"]]


def reproduction_fingerprint(
    candidates: pd.DataFrame,
    *,
    allowed_wings: Iterable[str] = DEFAULT_REPRODUCTION_WINGS,
) -> pd.DataFrame:
    allowed = set(allowed_wings)
    wing = candidates["ap_name"].map(_wing_of)
    filtered = candidates.loc[wing.isin(allowed)]
    # Sort explicitly before drop_duplicates so the variant-tie policy is visible
    # (matches select_top_k's tie order: freq asc, ssid asc).
    ordered = filtered.sort_values(
        ["location_p", "ap_name", "rssi_median", "frequency", "ssid"],
        ascending=[True, True, False, True, True],
        kind="stable",
    )
    return ordered.drop_duplicates(subset=["location_p", "ap_name"]).reset_index(drop=True)
