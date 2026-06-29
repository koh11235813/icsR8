from pathlib import Path

import pytest

from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans


def test_load_ap_coords_handles_bom_and_returns_13_3F_rows(dataset_dir: Path):
    df = load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")

    assert len(df) == 13
    assert set(df.columns) == {"ap_name", "floor", "x", "y", "z"}
    assert (df["floor"] == 3).all()
    assert df["ap_name"].is_unique
    assert not df["ap_name"].iloc[0].startswith("﻿")


def test_load_location_coords_returns_59_rows_no_default_filter(dataset_dir: Path):
    df = load_location_coords(dataset_dir / "location_coordinate_C.csv")

    assert len(df) == 59
    assert set(df.columns) == {
        "floor", "building", "location_p", "x", "y", "z_floor", "z_device",
    }
    assert set(df["building"].unique()) == {"C", "C2", "C3"}
    assert sorted(df["location_p"].tolist()) == list(range(1, 60))


def test_load_raw_scans_forward_loads_59_files_count_0_to_9(rawdata_root: Path):
    df = load_raw_scans("forward", rawdata_root)

    assert set(df.columns) == {
        "location_p", "ssid", "rssi", "frequency", "count", "ap_name", "direction",
    }
    assert df["location_p"].nunique() == 59
    assert sorted(df["location_p"].unique().tolist()) == list(range(1, 60))
    assert df["count"].min() == 0
    assert df["count"].max() == 9
    assert (df["direction"] == "forward").all()


def test_load_raw_scans_backward_uses_093_prefix(rawdata_root: Path):
    df = load_raw_scans("backward", rawdata_root)

    assert df["location_p"].nunique() == 59
    assert (df["direction"] == "backward").all()


def test_load_raw_scans_keeps_non_3F_aps_pre_filter(rawdata_root: Path):
    """IO layer must not silently drop off-floor APs; the fingerprint layer filters."""
    df = load_raw_scans("forward", rawdata_root)
    ap_names = set(df["ap_name"].unique())

    has_3F = any(name.endswith("-3F-01") or "-3F-" in name for name in ap_names)
    has_off_floor = any(
        "-2F-" in name or "-4F-" in name or "-5F-" in name for name in ap_names
    )
    assert has_3F
    assert has_off_floor, "loader stripped non-3F APs — that responsibility belongs to fingerprint.py"


def test_load_raw_scans_rejects_unknown_direction(rawdata_root: Path):
    with pytest.raises(ValueError):
        load_raw_scans("sideways", rawdata_root)  # type: ignore[arg-type]
