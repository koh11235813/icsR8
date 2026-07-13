"""wcl_linpower method tests.

Linear-power aggregation variant of WCL: RSSI_agg = 10·log10(mean(10^(r_i/10)))
instead of median.

Tests:
1. Single scan row → linear-power agg equals that value
2. 3-scan group [-40,-50,-60] → 10*log10((1e-4+1e-5+1e-6)/3)
3. Real forward data: estimates differ from baseline "wcl" at ≥1 location
4. Smoke test per contract
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.fingerprint import candidate_aggregate
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import run_method
from icsr8.protocols import iter_protocol_a


# --- session fixtures --------------------------------------------------------

@pytest.fixture(scope="session")
def ap_coords(dataset_dir: Path) -> pd.DataFrame:
    return load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")


@pytest.fixture(scope="session")
def location_coords(dataset_dir: Path) -> pd.DataFrame:
    return load_location_coords(dataset_dir / "location_coordinate_C.csv")


# --- T1: Single scan row (linear-power agg = that value) ---------------------

def test_wcl_linpower_single_scan(ap_coords):
    """Single scan with RSSI -40 dBm → linear-power agg = -40 dBm."""
    # Create synthetic scans: 1 location, 1 AP, 1 scan
    scans = pd.DataFrame({
        "location_p": [1],
        "ap_name": ["AP-C0-3F-01"],
        "ssid": ["tutwifi"],
        "frequency": [2412],
        "rssi": [-40],
        "count": [0],  # scan index
    })

    # Aggregate with linear_power
    candidates = candidate_aggregate(scans, ap_coords, aggregation="linear_power")

    # Single value → mean = value → 10*log10(10^(-40/10)) = -40
    assert len(candidates) > 0
    rssi_agg = candidates.iloc[0]["rssi_median"]
    assert rssi_agg == pytest.approx(-40.0, abs=0.01)


# --- T2: 3-scan group [-40, -50, -60] → expected linear-power value --------

def test_wcl_linpower_three_scans():
    """Three RSSI values [-40, -50, -60] dBm.

    Expected: 10*log10((10^-4 + 10^-5 + 10^-6) / 3)
             = 10*log10((0.0001 + 0.00001 + 0.000001) / 3)
             = 10*log10(0.000037) ≈ -44.31 dBm
    """
    rssi_values = [-40, -50, -60]
    linear_values = [10 ** (r / 10) for r in rssi_values]
    expected_linear_mean = np.mean(linear_values)
    expected_rssi_agg = 10 * np.log10(expected_linear_mean)

    # Create scans: same location, same AP, 3 different scan indices
    scans = pd.DataFrame({
        "location_p": [1, 1, 1],
        "ap_name": ["AP-C0-3F-01", "AP-C0-3F-01", "AP-C0-3F-01"],
        "ssid": ["tutwifi", "tutwifi", "tutwifi"],
        "frequency": [2412, 2412, 2412],
        "rssi": rssi_values,
        "count": [0, 1, 2],  # different scan indices
    })

    ap_coords = pd.DataFrame({
        "ap_name": ["AP-C0-3F-01"],
        "x": [10.0],
        "y": [20.0],
    })

    candidates = candidate_aggregate(scans, ap_coords, aggregation="linear_power")
    rssi_agg = candidates.iloc[0]["rssi_median"]

    assert rssi_agg == pytest.approx(expected_rssi_agg, abs=0.01)


# --- T3: Real forward data - estimate differs from baseline WCL at ≥1 loc ---

@pytest.fixture(scope="session")
def wcl_linpower_estimates(rawdata_root: Path, ap_coords, location_coords):
    """Run wcl_linpower on forward data via protocol A."""
    scans_f = load_raw_scans("forward", rawdata_root)
    scans_b = load_raw_scans("backward", rawdata_root)

    folds = iter_protocol_a(scans_f, scans_b)
    fold = folds[0]  # forward_to_backward fold

    return run_method(
        "wcl_linpower",
        fold.train_scans,
        fold.test_scans,
        ap_coords,
        location_coords,
    )


@pytest.fixture(scope="session")
def baseline_wcl_estimates(rawdata_root: Path, ap_coords, location_coords):
    """Run baseline wcl on forward data via protocol A for comparison."""
    scans_f = load_raw_scans("forward", rawdata_root)
    scans_b = load_raw_scans("backward", rawdata_root)

    folds = iter_protocol_a(scans_f, scans_b)
    fold = folds[0]  # forward_to_backward fold

    return run_method(
        "wcl",
        fold.train_scans,
        fold.test_scans,
        ap_coords,
        location_coords,
    )


def test_wcl_linpower_differs_from_baseline_wcl(
    wcl_linpower_estimates, baseline_wcl_estimates
):
    """wcl_linpower estimates must differ from baseline wcl at ≥1 location."""
    # Compare all locations
    diff_x = (wcl_linpower_estimates["x"] - baseline_wcl_estimates["x"]).abs()
    diff_y = (wcl_linpower_estimates["y"] - baseline_wcl_estimates["y"]).abs()

    # Check that at least one location differs
    any_diff = (diff_x > 0.01) | (diff_y > 0.01)
    assert any_diff.any(), \
        "wcl_linpower must differ from wcl at ≥1 location; all estimates identical"


# --- T4: Smoke test per contract (59 rows, [location_p, x, y], no NaN) ------

def test_wcl_linpower_smoke(rawdata_root: Path, ap_coords, location_coords):
    """Smoke test: fold structure, output shape, no NaN."""
    scans_f = load_raw_scans("forward", rawdata_root)
    scans_b = load_raw_scans("backward", rawdata_root)

    fold = iter_protocol_a(scans_f, scans_b)[0]
    est = run_method(
        "wcl_linpower",
        fold.train_scans,
        fold.test_scans,
        ap_coords,
        location_coords,
    )

    # 59 rows (all test locations)
    assert len(est) == 59, f"expected 59 rows, got {len(est)}"

    # Required columns
    assert set(est.columns) >= {"location_p", "x", "y"}, \
        f"missing required columns in {est.columns.tolist()}"

    # No NaN
    assert not est[["location_p", "x", "y"]].isna().any().any(), \
        "wcl_linpower output contains NaN values"
