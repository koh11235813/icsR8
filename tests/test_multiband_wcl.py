"""Tests for multiband_wcl method (Tier 2, 手法6: 周波数帯分離 + Multi-Band Fusion).

Covers:
  - Single-band data collapses to baseline WCL exactly (fusion is a no-op with 1 band)
  - Fusion arithmetic: inverse-MSE weighted average over per-band estimates
  - G3: fusion weight derives from per-band POSITION MSE (not dB residual sigma)
  - Path-loss fit recovers known (P0, alpha) from noiseless synthetic data
  - Real-data smoke + divergence from baseline WCL
  - G6: no band has >=3 candidates -> deterministic fallback to pooled baseline WCL
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.estimators import estimate_wcl
from icsr8.fingerprint import band_of, candidate_medians, reproduction_fingerprint
from icsr8.io import load_ap_coords, load_location_coords
from icsr8.methods import run_method
from icsr8.methods.multiband_wcl import MultibandWcl, _fuse_band_estimates
from icsr8.protocols import iter_protocol_a


# --- fixtures ----------------------------------------------------------------

@pytest.fixture(scope="session")
def ap_coords(dataset_dir: Path) -> pd.DataFrame:
    return load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")


@pytest.fixture(scope="session")
def location_coords(dataset_dir: Path) -> pd.DataFrame:
    return load_location_coords(dataset_dir / "location_coordinate_C.csv")


@pytest.fixture(scope="session")
def protocol_a_fold(rawdata_root: Path):
    """First fold of Protocol A (forward train, backward test)."""
    from icsr8.io import load_raw_scans as _load_raw_scans

    scans_forward = _load_raw_scans("forward", rawdata_root)
    scans_backward = _load_raw_scans("backward", rawdata_root)
    folds = list(iter_protocol_a(scans_forward, scans_backward))
    return folds[0]


# --- 1: Synthetic single-band data equals baseline WCL exactly ---------------

def test_multiband_wcl_single_band_equals_baseline():
    """When all training/test detections fall in one band, fusion has only one
    band to average over so it must reduce to plain baseline WCL.
    """
    ap_coords_syn = pd.DataFrame({
        "ap_name": ["AP-C0-3F-A", "AP-C2-3F-B", "AP-C3-3F-C"],
        "x": [0.0, 10.0, 5.0],
        "y": [0.0, 0.0, 10.0],
        "floor": [3, 3, 3],
    })

    train_locations = [(1, 1.0, 1.0), (2, 8.0, 1.0), (3, 4.0, 8.0), (4, 6.0, 3.0)]
    location_coords_syn = pd.DataFrame(
        train_locations, columns=["location_p", "x", "y"]
    )

    train_rows = []
    for loc_p, lx, ly in train_locations:
        for ap_name, ax, ay in [
            ("AP-C0-3F-A", 0.0, 0.0),
            ("AP-C2-3F-B", 10.0, 0.0),
            ("AP-C3-3F-C", 5.0, 10.0),
        ]:
            d = max(((lx - ax) ** 2 + (ly - ay) ** 2) ** 0.5, 0.5)
            rssi = -30.0 - 20.0 * np.log10(d)
            train_rows.append({
                "location_p": loc_p, "ap_name": ap_name, "ssid": "tutwifi",
                "frequency": 2412, "rssi": rssi, "count": 0,
            })
    train_scans = pd.DataFrame(train_rows)

    test_rows = [
        {"location_p": 99, "ap_name": "AP-C0-3F-A", "ssid": "tutwifi",
         "frequency": 2412, "rssi": -45.0, "count": 0},
        {"location_p": 99, "ap_name": "AP-C2-3F-B", "ssid": "tutwifi",
         "frequency": 2412, "rssi": -55.0, "count": 0},
        {"location_p": 99, "ap_name": "AP-C3-3F-C", "ssid": "tutwifi",
         "frequency": 2412, "rssi": -60.0, "count": 0},
    ]
    test_scans = pd.DataFrame(test_rows)

    baseline_fp = reproduction_fingerprint(candidate_medians(test_scans, ap_coords_syn))
    baseline_est = estimate_wcl(baseline_fp)

    multiband_est = run_method(
        "multiband_wcl", train_scans, test_scans, ap_coords_syn, location_coords_syn
    )

    assert multiband_est["x"].iloc[0] == pytest.approx(baseline_est["x"].iloc[0], abs=1e-9)
    assert multiband_est["y"].iloc[0] == pytest.approx(baseline_est["y"].iloc[0], abs=1e-9)


# --- 2: Fusion unit test (pure inverse-variance average) ----------------------

def test_fuse_band_estimates_inverse_mse_weighted_average():
    """Two bands with known (x_b, y_b) and position MSE fuse to the hand-computed
    inverse-MSE weighted average: w_b = 1/max(mse_b, 0.01).
    """
    band_estimates = {"2.4G": (10.0, 20.0), "5G": (14.0, 24.0)}
    # mse_2.4G = 0.25 -> w = 4.0 ; mse_5G = 1.0 -> w = 1.0
    band_weights = {"2.4G": 4.0, "5G": 1.0}

    x, y = _fuse_band_estimates(band_estimates, band_weights)

    expected_x = (4.0 * 10.0 + 1.0 * 14.0) / 5.0
    expected_y = (4.0 * 20.0 + 1.0 * 24.0) / 5.0
    assert x == pytest.approx(expected_x)
    assert y == pytest.approx(expected_y)


def test_fuse_band_estimates_no_weight_returns_none():
    """A band with no path-loss fit contributes weight 0; if every producing
    band has weight 0, fusion is undefined and predict() must fall back.
    """
    band_estimates = {"2.4G": (10.0, 20.0)}
    band_weights: dict[str, float] = {}
    assert _fuse_band_estimates(band_estimates, band_weights) is None


# --- G3: fusion weight derives from per-band POSITION MSE, not dB residual ---

def _two_band_train():
    """3 AP を 2.4G/5G 両帯で持つ 4 地点。5G は AP-C0 を +15dB バイアスして WCL を
    偏らせ、5G の位置 MSE を 2.4G より大きくする。ノイズなしなので dB 残差 σ は両帯
    とも下限 (SIGMA_MIN_DB) に張り付き、σ ベースだと重みが同じになってしまう。"""
    aps = [("AP-C0-3F-A", 0.0, 0.0), ("AP-C2-3F-B", 10.0, 0.0), ("AP-C3-3F-C", 5.0, 10.0)]
    ap_coords = pd.DataFrame({
        "ap_name": [a[0] for a in aps], "x": [a[1] for a in aps],
        "y": [a[2] for a in aps], "floor": [3, 3, 3],
    })
    train_locs = [(1, 2.0, 2.0), (2, 8.0, 2.0), (3, 5.0, 7.0), (4, 5.0, 4.0)]
    location_coords = pd.DataFrame(train_locs, columns=["location_p", "x", "y"])
    rows = []
    for loc_p, lx, ly in train_locs:
        for ap_name, ax, ay in aps:
            d = max(((lx - ax) ** 2 + (ly - ay) ** 2) ** 0.5, 0.5)
            base = -30.0 - 20.0 * np.log10(d)
            for freq, bias in [(2412, 0.0), (5180, 15.0 if ap_name == "AP-C0-3F-A" else 0.0)]:
                for count in range(3):
                    rows.append({
                        "location_p": loc_p, "ap_name": ap_name, "ssid": "tutwifi",
                        "frequency": freq, "rssi": base + bias, "count": count,
                    })
    return pd.DataFrame(rows), ap_coords, location_coords


def test_multiband_wcl_band_weights_from_position_mse():
    train_scans, ap_coords, location_coords = _two_band_train()
    method = MultibandWcl().fit(train_scans, ap_coords, location_coords)

    # Independent per-band position MSE: run that band's WCL on each train
    # location and average the squared L2 error against truth.
    candidates = candidate_medians(train_scans, ap_coords).assign(
        band=lambda d: d["frequency"].map(band_of)
    )
    truth = location_coords.set_index("location_p")
    expected_mse: dict[str, float] = {}
    for band in ("2.4G", "5G"):
        band_cand = candidates[candidates["band"] == band]
        sq = []
        for loc_p, group in band_cand.groupby("location_p"):
            est = estimate_wcl(reproduction_fingerprint(group))
            ex, ey = float(est["x"].iloc[0]), float(est["y"].iloc[0])
            tx, ty = float(truth.loc[loc_p, "x"]), float(truth.loc[loc_p, "y"])
            sq.append((ex - tx) ** 2 + (ey - ty) ** 2)
        expected_mse[band] = float(np.mean(sq))

    # Weight == 1/max(mse, 0.01) (fails on σ-based code: both weights would be 1.0).
    for band in ("2.4G", "5G"):
        assert method.band_weights[band] == pytest.approx(1.0 / max(expected_mse[band], 0.01))
        assert method.band_mse[band] == pytest.approx(expected_mse[band])

    # The biased 5G band must be down-weighted relative to 2.4G.
    assert method.band_weights["5G"] < method.band_weights["2.4G"]
    # σ_b is retained as a diagnostic (both are tabulated in the report).
    assert set(method.band_sigma) >= {"2.4G", "5G"}


# --- 3: Path-loss fit recovers known (P0, alpha) from noiseless data ---------

def test_multiband_wcl_pathloss_fit_recovers_known_params():
    """Noiseless synthetic RSSI generated with P0=-30, alpha=2.0 along a line
    of known distances must be recovered by the least-squares fit in fit().
    """
    ap_coords_syn = pd.DataFrame({
        "ap_name": ["AP-C0-3F-A"], "x": [0.0], "y": [0.0], "floor": [3],
    })

    distances = [1.0, 2.0, 4.0, 8.0]
    p0_true, alpha_true = -30.0, 2.0
    location_coords_syn = pd.DataFrame({
        "location_p": [1, 2, 3, 4],
        "x": distances,
        "y": [0.0] * 4,
    })

    rows = []
    for loc_p, d in zip([1, 2, 3, 4], distances):
        rssi = p0_true - 10.0 * alpha_true * np.log10(d)
        for count in range(10):
            rows.append({
                "location_p": loc_p, "ap_name": "AP-C0-3F-A", "ssid": "tutwifi",
                "frequency": 2412, "rssi": rssi, "count": count,
            })
    train_scans = pd.DataFrame(rows)

    method = MultibandWcl().fit(train_scans, ap_coords_syn, location_coords_syn)

    fit_params = method.path_loss_fits[("AP-C0-3F-A", "2.4G")]
    assert fit_params["alpha"] == pytest.approx(alpha_true, abs=0.05)
    assert fit_params["P0"] == pytest.approx(p0_true, abs=0.1)


# --- 4: Real-data smoke, divergence from baseline WCL ------------------------

def test_multiband_wcl_real_data_differs_from_baseline(
    protocol_a_fold, ap_coords, location_coords
):
    fold = protocol_a_fold
    train_location_coords = location_coords[
        location_coords["location_p"].isin(fold.train_scans["location_p"].unique())
    ]

    method = MultibandWcl()
    method.fit(fold.train_scans, ap_coords, train_location_coords)
    est = method.predict(fold.test_scans)

    baseline_est = run_method(
        "wcl", fold.train_scans, fold.test_scans, ap_coords, location_coords
    )

    diff_x = (est["x"] - baseline_est["x"]).abs()
    diff_y = (est["y"] - baseline_est["y"]).abs()
    max_diff = np.maximum(diff_x, diff_y).max()

    assert max_diff > 0.05, f"Expected difference from baseline, got max_diff={max_diff}"


# --- G6: fallback to pooled WCL when no single band has >=3 candidates --------

def test_multiband_wcl_fallback_to_pooled_wcl_when_no_band_has_three():
    """G6: 2.4G に 2 AP + 5G に 2 AP の test 地点は、どの帯も候補<3 で帯別 WCL を
    出せない。融合不能なので pooled 13-AP baseline WCL に fallback し、
    fallback_count==1、推定は独立計算した pooled WCL 座標に厳密一致する。"""
    aps = [
        ("AP-C0-3F-A", 0.0, 0.0, 2412),
        ("AP-C2-3F-B", 10.0, 0.0, 2412),
        ("AP-C3-3F-C", 5.0, 10.0, 5180),
        ("AP-C0-3F-D", 3.0, 7.0, 5180),
    ]
    ap_coords = pd.DataFrame({
        "ap_name": [a[0] for a in aps], "x": [a[1] for a in aps],
        "y": [a[2] for a in aps], "floor": [3] * 4,
    })

    # Minimal train (fit must run; weights are irrelevant to the fallback path).
    train_locs = [(1, 2.0, 2.0), (2, 8.0, 3.0), (3, 4.0, 6.0)]
    location_coords = pd.DataFrame(train_locs, columns=["location_p", "x", "y"])
    train_rows = []
    for loc_p, lx, ly in train_locs:
        for ap_name, ax, ay, freq in aps:
            d = max(((lx - ax) ** 2 + (ly - ay) ** 2) ** 0.5, 0.5)
            train_rows.append({
                "location_p": loc_p, "ap_name": ap_name, "ssid": "tutwifi",
                "frequency": freq, "rssi": -30.0 - 20.0 * np.log10(d), "count": 0,
            })
    train_scans = pd.DataFrame(train_rows)

    test_scans = pd.DataFrame([
        {"location_p": 99, "ap_name": "AP-C0-3F-A", "ssid": "tutwifi", "frequency": 2412, "rssi": -45.0, "count": 0},
        {"location_p": 99, "ap_name": "AP-C2-3F-B", "ssid": "tutwifi", "frequency": 2412, "rssi": -55.0, "count": 0},
        {"location_p": 99, "ap_name": "AP-C3-3F-C", "ssid": "tutwifi", "frequency": 5180, "rssi": -50.0, "count": 0},
        {"location_p": 99, "ap_name": "AP-C0-3F-D", "ssid": "tutwifi", "frequency": 5180, "rssi": -60.0, "count": 0},
    ])

    method = MultibandWcl().fit(train_scans, ap_coords, location_coords)
    est = method.predict(test_scans)

    assert method.fallback_count == 1

    # Independently computed pooled baseline WCL over all 4 APs.
    pooled_fp = reproduction_fingerprint(candidate_medians(test_scans, ap_coords))
    pooled_est = estimate_wcl(pooled_fp)

    assert est["x"].iloc[0] == pytest.approx(pooled_est["x"].iloc[0], abs=1e-9)
    assert est["y"].iloc[0] == pytest.approx(pooled_est["y"].iloc[0], abs=1e-9)


# --- 5: Smoke test (contract) -------------------------------------------------

def test_multiband_wcl_smoke(protocol_a_fold, ap_coords, location_coords):
    """Mandatory smoke test: basic functionality and contract compliance."""
    fold = protocol_a_fold

    est = run_method(
        "multiband_wcl",
        fold.train_scans, fold.test_scans,
        ap_coords, location_coords
    )

    assert len(est) == 59, f"Expected 59 rows, got {len(est)}"
    assert set(est.columns) == {"location_p", "x", "y"}, \
        f"Expected columns {{location_p, x, y}}, got {set(est.columns)}"
    assert not est.isna().any().any(), "Found NaN values in estimate"
    assert est["location_p"].min() == 1
    assert est["location_p"].max() == 59
    assert np.isfinite(est[["x", "y"]]).all().all(), "Found non-finite coordinates"
