"""wcl_powerdomain メソッドの等価性テスト。

WCL (Weighted Centroid Localization) の重みを w = 10^(rssi/10) と定義する。
ベースラインの w = 10^((rssi - rssi_min) / 10) と異なるように見えるが、
正規化ステップで定数因子 10^(-rssi_min/10) が相殺され、数学的に等価である。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.estimators import select_top_k
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import run_method
from icsr8.protocols import iter_protocol_a


@pytest.fixture(scope="module")
def ap13(dataset_dir: Path) -> pd.DataFrame:
    return load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")


@pytest.fixture(scope="module")
def loc_coords(dataset_dir: Path) -> pd.DataFrame:
    return load_location_coords(dataset_dir / "location_coordinate_C.csv")[
        ["location_p", "x", "y"]
    ]


@pytest.fixture(scope="module")
def scans(rawdata_root: Path) -> dict[str, pd.DataFrame]:
    return {
        "forward": load_raw_scans("forward", rawdata_root),
        "backward": load_raw_scans("backward", rawdata_root),
    }


@pytest.fixture(scope="module")
def folds(scans):
    """Protocol A folds (forward→backward, backward→forward)."""
    return iter_protocol_a(scans["forward"], scans["backward"])


# --- Test 1: Smoke test (basic output shape and validity) ---

@pytest.mark.parametrize("fold_idx", [0, 1])
def test_smoke_test(fold_idx, folds, ap13, loc_coords):
    """Verify wcl_powerdomain produces 59 locations with [location_p, x, y], no NaN."""
    fold = folds[fold_idx]
    est = run_method(
        "wcl_powerdomain",
        train_scans=fold.train_scans,
        test_scans=fold.test_scans,
        ap_coords=ap13,
        location_coords=loc_coords,
    )

    assert list(est.columns) == ["location_p", "x", "y"], f"Got columns: {list(est.columns)}"
    assert len(est) == 59, f"Expected 59 rows, got {len(est)}"
    assert not est.isnull().any().any(), "Found NaN values in output"
    assert est["location_p"].nunique() == 59, "Not all 59 locations present"
    # All coordinates should be numeric and non-infinite
    assert est["x"].dtype in [np.float32, np.float64]
    assert est["y"].dtype in [np.float32, np.float64]
    assert np.isfinite(est["x"]).all() and np.isfinite(est["y"]).all()


# --- Test 2: Equivalence with baseline WCL on real data ---

@pytest.mark.parametrize("fold_idx", [0, 1])
def test_equivalence_with_wcl(fold_idx, folds, ap13, loc_coords):
    """wcl_powerdomain and wcl must produce identical estimates (max diff < 1e-9)."""
    fold = folds[fold_idx]

    pd_est = run_method(
        "wcl_powerdomain",
        train_scans=fold.train_scans,
        test_scans=fold.test_scans,
        ap_coords=ap13,
        location_coords=loc_coords,
    ).sort_values("location_p").reset_index(drop=True)

    wcl_est = run_method(
        "wcl",
        train_scans=fold.train_scans,
        test_scans=fold.test_scans,
        ap_coords=ap13,
        location_coords=loc_coords,
    ).sort_values("location_p").reset_index(drop=True)

    # Verify location_p sequences match
    assert pd_est["location_p"].tolist() == wcl_est["location_p"].tolist()

    # Compute max absolute differences
    dx_max = (pd_est["x"] - wcl_est["x"]).abs().max()
    dy_max = (pd_est["y"] - wcl_est["y"]).abs().max()

    assert dx_max < 1e-9, f"x coords differ by up to {dx_max} (tolerance 1e-9)"
    assert dy_max < 1e-9, f"y coords differ by up to {dy_max} (tolerance 1e-9)"


# --- Test 3: Synthetic weight equivalence ---

def test_synthetic_weight_equivalence(ap13):
    """Show that normalized powerdomain weights equal normalized baseline weights.

    Create a minimal 3-candidate fingerprint and compute weights both ways,
    proving that Σ(w_pd * coord) / Σ(w_pd) = Σ(w_base * coord) / Σ(w_base).
    """
    # Synthetic fingerprint: 1 location, 3 candidates with known rssi values
    fp = pd.DataFrame({
        "location_p": [1, 1, 1],
        "ap_name": ["AP1", "AP2", "AP3"],
        "rssi_median": [-50.0, -60.0, -70.0],  # Test rssi values
        "x": [100.0, 200.0, 300.0],  # Candidate coordinates
        "y": [50.0, 100.0, 150.0],
        "frequency": [2400, 2400, 2400],
        "ssid": ["net", "net", "net"],
        "n_detect": [10, 10, 10],
        "detection_rate": [1.0, 1.0, 1.0],
        "rssi_std": [1.0, 1.0, 1.0],
        "rssi_mean_linear_dbm": [1.0, 1.0, 1.0],
    })

    top = select_top_k(fp, 3)
    rssi_vals = top["rssi_median"].to_numpy()
    x_vals = top["x"].to_numpy()
    y_vals = top["y"].to_numpy()

    # Baseline weights: w = 10^((rssi - min) / 10)
    rssi_min = rssi_vals.min()
    w_base = np.power(10.0, (rssi_vals - rssi_min) / 10.0)

    # Powerdomain weights: w_pd = 10^(rssi / 10)
    w_pd = np.power(10.0, rssi_vals / 10.0)

    # Powerdomain weights normalized by factor of 10^(-rssi_min/10)
    # should equal baseline weights
    norm_factor = np.power(10.0, -rssi_min / 10.0)
    w_pd_normalized = w_pd * norm_factor

    # The normalized weights must match baseline weights
    assert np.allclose(w_pd_normalized, w_base, rtol=1e-14, atol=1e-14), \
        f"Normalized weights mismatch:\nw_base: {w_base}\nw_pd_norm: {w_pd_normalized}"

    # Weighted averages must be identical
    x_base = (w_base * x_vals).sum() / w_base.sum()
    x_pd = (w_pd * x_vals).sum() / w_pd.sum()

    y_base = (w_base * y_vals).sum() / w_base.sum()
    y_pd = (w_pd * y_vals).sum() / w_pd.sum()

    # Verify equivalence (accounting for floating-point precision)
    assert np.isclose(x_base, x_pd, rtol=1e-14, atol=1e-14), \
        f"x estimates differ: base={x_base}, pd={x_pd}"
    assert np.isclose(y_base, y_pd, rtol=1e-14, atol=1e-14), \
        f"y estimates differ: base={y_base}, pd={y_pd}"
