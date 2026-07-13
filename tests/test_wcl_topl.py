"""Tests for wcl_topl method (top-L AP variant of WCL).

Tests cover:
1. L=3 equivalence to baseline WCL
2. L="all" uses all available APs at each location
3. fit(L=None) auto-selects best L from {3,4,5,7,"all"}
4. Smoke test: protocol A fold runs end-to-end
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.estimators import estimate_wcl
from icsr8.fingerprint import candidate_medians, reproduction_fingerprint
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import run_method
from icsr8.protocols import iter_protocol_a


@pytest.fixture(scope="module")
def ap_coords(dataset_dir: Path) -> pd.DataFrame:
    return load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")


@pytest.fixture(scope="module")
def location_coords(dataset_dir: Path) -> pd.DataFrame:
    df = load_location_coords(dataset_dir / "location_coordinate_C.csv")
    return df[["location_p", "x", "y"]]


@pytest.fixture(scope="module")
def scans_f(rawdata_root: Path) -> pd.DataFrame:
    return load_raw_scans("forward", rawdata_root)


@pytest.fixture(scope="module")
def scans_b(rawdata_root: Path) -> pd.DataFrame:
    return load_raw_scans("backward", rawdata_root)


# --- Test 1: L=3 equals baseline WCL ---

def test_wcl_topl_L3_equals_baseline(scans_f, ap_coords, location_coords):
    """Verify L=3 produces same results as baseline WCL (max abs diff < 1e-12)."""
    fp = reproduction_fingerprint(candidate_medians(scans_f, ap_coords))
    baseline = estimate_wcl(fp)

    est = run_method("wcl_topl", scans_f, scans_f, ap_coords, location_coords, L=3)

    # Sort both by location_p for comparison
    baseline_sorted = baseline.sort_values("location_p").reset_index(drop=True)
    est_sorted = est.sort_values("location_p").reset_index(drop=True)

    # Check location_p matches
    assert baseline_sorted["location_p"].tolist() == est_sorted["location_p"].tolist()

    # Check coordinates are equivalent within numerical precision
    assert (baseline_sorted["x"] - est_sorted["x"]).abs().max() < 1e-12, \
        f"x diff: {(baseline_sorted['x'] - est_sorted['x']).abs().max()}"
    assert (baseline_sorted["y"] - est_sorted["y"]).abs().max() < 1e-12, \
        f"y diff: {(baseline_sorted['y'] - est_sorted['y']).abs().max()}"


# --- Test 2: L="all" matches an independently computed all-candidate centroid ---

def test_wcl_topl_L_all_matches_independent_centroid_and_differs_from_L3():
    """L='all' の推定値が numpy で独立計算した全候補セントロイドと一致し、
    L=3 のセントロイドとは異なることを検証する（>3 AP の合成データ）。
    """
    ap_names = ["AP-C0-3F-1", "AP-C2-3F-2", "AP-C3-3F-3", "AP-C0-3F-4", "AP-C2-3F-5"]
    xs = [0.0, 10.0, 20.0, 30.0, 40.0]
    ys = [0.0, 0.0, 0.0, 0.0, 0.0]
    rssis = [-40, -45, -50, -55, -60]  # strictly decreasing strength

    scans = pd.concat(
        [
            pd.DataFrame({
                "location_p": [1] * 10,
                "ap_name": [name] * 10,
                "ssid": ["tutwifi"] * 10,
                "frequency": [2400] * 10,
                "rssi": [rssi] * 10,
                "count": list(range(10)),
            })
            for name, rssi in zip(ap_names, rssis)
        ],
        ignore_index=True,
    )

    ap_coords = pd.DataFrame({
        "ap_name": ap_names,
        "x": xs,
        "y": ys,
        "floor": [3] * 5,
    })
    location_coords = pd.DataFrame({"location_p": [1], "x": [0.0], "y": [0.0]})

    est_all = run_method("wcl_topl", scans, scans, ap_coords, location_coords, L="all")
    est_3 = run_method("wcl_topl", scans, scans, ap_coords, location_coords, L=3)

    fp = reproduction_fingerprint(candidate_medians(scans, ap_coords))
    loc_fp = fp[fp["location_p"] == 1]
    assert len(loc_fp) == 5  # sanity: all 5 synthetic APs survived dedup

    rssi_min_all = loc_fp["rssi_median"].min()
    w_all = np.power(10.0, (loc_fp["rssi_median"].to_numpy() - rssi_min_all) / 10.0)
    expected_x_all = float((w_all * loc_fp["x"].to_numpy()).sum() / w_all.sum())
    expected_y_all = float((w_all * loc_fp["y"].to_numpy()).sum() / w_all.sum())

    assert est_all["x"].iloc[0] == pytest.approx(expected_x_all, abs=1e-9)
    assert est_all["y"].iloc[0] == pytest.approx(expected_y_all, abs=1e-9)

    # >3 candidates with monotonically decreasing rssi means L="all" pulls the
    # centroid toward the weaker/farther APs that L=3 would have dropped.
    diff = np.hypot(
        est_all["x"].iloc[0] - est_3["x"].iloc[0],
        est_all["y"].iloc[0] - est_3["y"].iloc[0],
    )
    assert diff > 1e-6


# --- Test 3: fit(L=None) selects the argmin train L2 error over the candidates ---

def test_wcl_topl_fit_L_none_selects_argmin(scans_f, ap_coords, location_coords):
    """fit(L=None) の selected_L が、公開 API (run_method) で候補ごとに再計算した
    train L2 誤差の argmin と一致することを検証する（同点は候補順で早い方が勝つ）。
    """
    from icsr8.methods.wcl_topl import WCLTopL

    method = WCLTopL(L=None)
    method.fit(scans_f, ap_coords, location_coords)

    errors: dict = {}
    for candidate_L in [3, 4, 5, 7, "all"]:
        est = run_method("wcl_topl", scans_f, scans_f, ap_coords, location_coords, L=candidate_L)
        merged = est.merge(location_coords, on="location_p", suffixes=("_est", "_truth"))
        l2 = np.hypot(
            merged["x_est"] - merged["x_truth"],
            merged["y_est"] - merged["y_truth"],
        )
        errors[candidate_L] = l2.mean()

    # dict preserves [3,4,5,7,"all"] insertion order; min() picks the first
    # minimal entry on ties, matching fit()'s strict "<" comparison.
    best_L = min(errors, key=lambda k: errors[k])

    assert method.selected_L == best_L


# --- Test 4: fit + predict works with selected L ---

def test_wcl_topl_fit_then_predict(scans_f, scans_b, ap_coords, location_coords):
    """Verify fit(L=None) + predict works and returns valid estimates."""
    from icsr8.methods.wcl_topl import WCLTopL

    method = WCLTopL(L=None)
    method.fit(scans_f, ap_coords, location_coords)

    # Predict on different data (backward scans)
    est = method.predict(scans_b)

    # Should be a DataFrame with [location_p, x, y]
    assert isinstance(est, pd.DataFrame)
    assert list(est.columns) == ["location_p", "x", "y"]
    assert len(est) > 0
    assert not est["x"].isna().any()
    assert not est["y"].isna().any()


# --- Test 5: Smoke test (protocol A) ---

def test_wcl_topl_smoke_protocol_a(scans_f, scans_b, ap_coords, location_coords):
    """Smoke test: run wcl_topl on protocol A fold, verify output structure."""
    fold = list(iter_protocol_a(scans_f, scans_b))[0]

    est = run_method(
        "wcl_topl",
        fold.train_scans,
        fold.test_scans,
        ap_coords,
        location_coords,
    )

    # Verify output structure
    assert isinstance(est, pd.DataFrame)
    assert list(est.columns) == ["location_p", "x", "y"]
    assert len(est) == 59, f"Expected 59 locations, got {len(est)}"
    assert not est["x"].isna().any(), "Found NaN in x"
    assert not est["y"].isna().any(), "Found NaN in y"
    assert est["location_p"].nunique() == 59, "Should have 59 unique locations"


# --- Test 6: invalid L is rejected at construction ---

def test_wcl_topl_invalid_L_raises():
    """L が許可集合 {3,4,5,7,"all",None} 以外だと ValueError。"""
    from icsr8.methods.wcl_topl import WCLTopL

    with pytest.raises(ValueError):
        WCLTopL(L=6)
