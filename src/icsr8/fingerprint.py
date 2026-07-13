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

import numpy as np
import pandas as pd

from icsr8.constants import BAND_BOUNDARIES_MHZ

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


def band_of(frequency_mhz: int) -> str:
    for band, (low, high) in BAND_BOUNDARIES_MHZ.items():
        if low <= frequency_mhz <= high:
            return band
    raise ValueError(f"frequency {frequency_mhz} MHz is outside all known bands")


def detailed_fingerprint(
    scans: pd.DataFrame,
    ap_coords: pd.DataFrame | None = None,
) -> pd.DataFrame:
    work = scans.assign(_linear=np.power(10.0, scans["rssi"] / 10.0))
    grouped = work.groupby(
        ["location_p", "ap_name", "ssid", "frequency"], as_index=False
    ).agg(
        n_detect=("rssi", "size"),
        rssi_median=("rssi", "median"),
        rssi_mean_db=("rssi", "mean"),
        rssi_std=("rssi", lambda s: s.std(ddof=0)),
        _linear_mean=("_linear", "mean"),
    )
    grouped["detection_rate"] = grouped["n_detect"] / 10.0
    grouped["rssi_mean_linear_dbm"] = 10.0 * np.log10(grouped["_linear_mean"])
    grouped["band"] = grouped["frequency"].map(band_of)
    grouped = grouped.drop(columns="_linear_mean")

    columns = [
        "location_p", "ap_name", "ssid", "frequency", "band",
        "n_detect", "detection_rate", "rssi_median", "rssi_mean_db",
        "rssi_mean_linear_dbm", "rssi_std",
    ]
    if ap_coords is not None:
        grouped = grouped.merge(
            ap_coords[["ap_name", "floor", "x", "y"]],
            on="ap_name",
            how="left",
        )
        columns = columns + ["floor", "x", "y"]
    return grouped[columns]


def ap_band_fingerprint(
    scans: pd.DataFrame,
    ap_coords: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """物理 AP × band 単位へ集約した指紋を作る。

    detailed_fingerprint が (location_p, ap_name, ssid, frequency) で分割するのに
    対し、tutwifi/tutwifi2025 の SSID 違いや同一 band 内のチャネル違いを 1 行へ
    束ねる。下流の各手法が個別に再集約する必要をなくす。
    """
    work = scans.assign(
        band=scans["frequency"].map(band_of),
        _linear=np.power(10.0, scans["rssi"] / 10.0),
    )
    grouped = work.groupby(
        ["location_p", "ap_name", "band"], as_index=False
    ).agg(
        # Why not size(): 行数はスキャン数ではない。同一 scan (count) 内で同じ
        #   物理 AP-band が 2 つの SSID から見えることがあり、size() は 2 と数えて
        #   しまう。distinct な scan index (count) の nunique で 1 回と数える。
        n_detect=("count", "nunique"),
        rssi_median=("rssi", "median"),
        rssi_mean_db=("rssi", "mean"),
        rssi_std=("rssi", lambda s: s.std(ddof=0)),
        _linear_mean=("_linear", "mean"),
    )
    grouped["detection_rate"] = grouped["n_detect"] / 10.0
    grouped["rssi_mean_linear_dbm"] = 10.0 * np.log10(grouped["_linear_mean"])
    grouped = grouped.drop(columns="_linear_mean")

    columns = [
        "location_p", "ap_name", "band",
        "n_detect", "detection_rate", "rssi_median", "rssi_mean_db",
        "rssi_mean_linear_dbm", "rssi_std",
    ]
    if ap_coords is not None:
        grouped = grouped.merge(
            ap_coords[["ap_name", "floor", "x", "y"]],
            on="ap_name",
            how="left",
        )
        columns = columns + ["floor", "x", "y"]
    return grouped[columns]


def candidate_aggregate(
    scans: pd.DataFrame,
    ap_coords: pd.DataFrame,
    aggregation: str = "median",
    *,
    restrict_to_known_aps: bool = True,
) -> pd.DataFrame:
    group_keys = ["location_p", "ap_name", "ssid", "frequency"]
    if aggregation == "median":
        grouped = scans.groupby(group_keys, as_index=False).agg(
            rssi_median=("rssi", "median")
        )
    elif aggregation == "dbm_mean":
        grouped = scans.groupby(group_keys, as_index=False).agg(
            rssi_median=("rssi", "mean")
        )
    elif aggregation == "linear_power":
        work = scans.assign(_linear=np.power(10.0, scans["rssi"] / 10.0))
        grouped = work.groupby(group_keys, as_index=False).agg(
            _linear_mean=("_linear", "mean")
        )
        grouped["rssi_median"] = 10.0 * np.log10(grouped["_linear_mean"])
        grouped = grouped.drop(columns="_linear_mean")
    else:
        raise ValueError(f"unknown aggregation: {aggregation!r}")

    join = grouped.merge(
        ap_coords[["ap_name", "x", "y"]],
        on="ap_name",
        how="left" if not restrict_to_known_aps else "inner",
    )

    # Why not rename rssi_median per aggregation kind: keeping this column name
    # makes select_top_k/estimators drop-in compatible with any aggregation;
    # renaming would fork every downstream consumer for no functional gain.
    return join[["location_p", "ap_name", "ssid", "frequency", "rssi_median", "x", "y"]]
