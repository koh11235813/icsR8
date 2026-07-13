from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.fingerprint import (
    ap_band_fingerprint,
    band_of,
    candidate_aggregate,
    candidate_medians,
    detailed_fingerprint,
)
from icsr8.io import load_ap_coords, load_raw_scans


def _scans_for_one_ap(location_p, ap_name, ssid, frequency, rssis):
    return pd.DataFrame(
        [
            {
                "location_p": location_p,
                "ssid": ssid,
                "rssi": r,
                "frequency": frequency,
                "count": i,
                "ap_name": ap_name,
                "direction": "forward",
            }
            for i, r in enumerate(rssis)
        ]
    )


@pytest.mark.parametrize(
    "freq,expected",
    [(2412, "2.4G"), (5180, "5G"), (5975, "6G"), (6135, "6G")],
)
def test_band_of_classifies_known_frequencies(freq, expected):
    assert band_of(freq) == expected


def test_band_of_raises_outside_all_ranges():
    with pytest.raises(ValueError):
        band_of(3000)


def test_detailed_fingerprint_rssi_mean_linear_dbm_matches_power_domain_average():
    scans = _scans_for_one_ap(1, "AP-3F-A", "tutwifi", 2412, [-40, -50, -60])
    expected = 10.0 * np.log10(np.mean([1e-4, 1e-5, 1e-6]))

    fp = detailed_fingerprint(scans)

    assert fp.iloc[0]["rssi_mean_linear_dbm"] == pytest.approx(expected, abs=1e-9)
    assert expected == pytest.approx(-44.32, abs=0.01)


def test_detailed_fingerprint_on_real_forward_scans(rawdata_root: Path, dataset_dir: Path):
    scans = load_raw_scans("forward", rawdata_root)
    ap_coords = load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")

    detailed = detailed_fingerprint(scans, ap_coords)

    assert (detailed["n_detect"] <= 10).all()
    assert (detailed["n_detect"] >= 1).all()
    assert ((detailed["detection_rate"] > 0) & (detailed["detection_rate"] <= 1)).all()

    known_names = set(ap_coords["ap_name"])
    restricted = detailed.loc[detailed["ap_name"].isin(known_names)]
    candidates = candidate_medians(scans, ap_coords, restrict_to_known_aps=True)
    assert len(restricted) == len(candidates)


def test_ap_band_fingerprint_counts_shared_scan_once():
    # 同一 scan (count=0) 内で 1 つの物理 AP-band が 2 つの SSID から見える。
    # n_detect は scan を 1 回だけ数える（size() なら 2 になってしまう）。
    scans = pd.DataFrame([
        {"location_p": 1, "ssid": "tutwifi", "rssi": -40, "frequency": 5180,
         "count": 0, "ap_name": "AP-C0-3F-01", "direction": "forward"},
        {"location_p": 1, "ssid": "tutwifi2025", "rssi": -42, "frequency": 5200,
         "count": 0, "ap_name": "AP-C0-3F-01", "direction": "forward"},
    ])

    fp = ap_band_fingerprint(scans)

    assert len(fp) == 1
    row = fp.iloc[0]
    assert row["band"] == "5G"
    assert row["n_detect"] == 1
    assert row["detection_rate"] == pytest.approx(0.1)
    # 統計は全 variant 行（2 行）で計算される。
    assert row["rssi_median"] == pytest.approx(-41.0)
    assert row["rssi_std"] == pytest.approx(1.0)


def test_ap_band_fingerprint_on_real_forward_scans(rawdata_root: Path, dataset_dir: Path):
    scans = load_raw_scans("forward", rawdata_root)
    ap_coords = load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")

    band_fp = ap_band_fingerprint(scans, ap_coords)
    detailed = detailed_fingerprint(scans, ap_coords)

    assert (band_fp["n_detect"] <= 10).all()
    assert ((band_fp["detection_rate"] > 0) & (band_fp["detection_rate"] <= 1)).all()
    # band への集約は (ssid, frequency) より粗いので群数は増えない。
    assert len(band_fp) <= len(detailed)
    assert {"floor", "x", "y"}.issubset(band_fp.columns)


def test_candidate_aggregate_median_matches_candidate_medians(
    rawdata_root: Path, dataset_dir: Path
):
    scans = load_raw_scans("forward", rawdata_root)
    ap_coords = load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")

    sort_cols = ["location_p", "ap_name", "ssid", "frequency"]
    expected = candidate_medians(scans, ap_coords).sort_values(sort_cols).reset_index(drop=True)
    actual = (
        candidate_aggregate(scans, ap_coords, aggregation="median")
        .sort_values(sort_cols)
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_candidate_aggregate_linear_power_differs_and_stays_in_range(
    rawdata_root: Path, dataset_dir: Path
):
    scans = load_raw_scans("forward", rawdata_root)
    ap_coords = load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")

    median_cand = candidate_aggregate(scans, ap_coords, aggregation="median")
    linear_cand = candidate_aggregate(scans, ap_coords, aggregation="linear_power")

    merged = median_cand.merge(
        linear_cand,
        on=["location_p", "ap_name", "ssid", "frequency"],
        suffixes=("_median", "_linear"),
    )
    assert (merged["rssi_median_median"] != merged["rssi_median_linear"]).any()
    assert linear_cand["rssi_median"].between(-100, 0).all()
