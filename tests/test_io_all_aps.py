from pathlib import Path

import pytest

from icsr8.io import load_ap_coords, load_ap_coords_all


@pytest.fixture()
def all_ap_csv(repo_root: Path) -> Path:
    return repo_root / "data" / "dataset_r0701" / "AP_coordinate_C_All.csv"


def test_load_ap_coords_all_returns_67_rows(all_ap_csv: Path):
    df = load_ap_coords_all(all_ap_csv)

    assert len(df) == 67
    assert set(df.columns) == {"ap_name", "floor", "x", "y"}
    assert df["ap_name"].is_unique


def test_load_ap_coords_all_is_superset_of_baseline_13_aps(
    dataset_dir: Path, all_ap_csv: Path
):
    baseline = load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")
    all_aps = load_ap_coords_all(all_ap_csv)

    assert set(baseline["ap_name"]).issubset(set(all_aps["ap_name"]))


def test_load_ap_coords_all_baseline_coordinates_match(
    dataset_dir: Path, all_ap_csv: Path
):
    baseline = load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv").set_index("ap_name")
    all_aps = load_ap_coords_all(all_ap_csv).set_index("ap_name")

    shared = all_aps.loc[baseline.index]
    assert (baseline["x"] - shared["x"]).abs().max() <= 0.01
    assert (baseline["y"] - shared["y"]).abs().max() <= 0.01
