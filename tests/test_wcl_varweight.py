"""WCL with variance + detection-count confidence weighting.

Tests for wcl_varweight method (Tier 3, hand法11):
  - Variance-based downweighting: unstable APs (high rssi_std, low n_detect) → lower weight
  - sigma_ref = median(rssi_std) over training ap_band_fingerprint, floored at SIGMA_MIN_DB
  - Per-location weight: w = baseline_w / (1 + (sigma_q/sigma_ref)²) * min(n_detect/10, 1.0)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.constants import SIGMA_MIN_DB
from icsr8.fingerprint import ap_band_fingerprint, band_of, candidate_medians, reproduction_fingerprint
from icsr8.io import load_ap_coords, load_location_coords
from icsr8.methods import run_method
from icsr8.protocols import iter_protocol_a


# --- shared expected-formula helpers (independent reimplementation for asserts) ---

def _sigma_ref_from_train(train_scans: pd.DataFrame) -> float:
    """fit() と同じ計算: 学習データの ap_band_fingerprint.rssi_std の中央値、
    SIGMA_MIN_DB でフロア処理。
    """
    ab_fp = ap_band_fingerprint(train_scans, ap_coords=None)
    return max(float(ab_fp["rssi_std"].median()), SIGMA_MIN_DB)


def _varweight_weight(
    rssi_median: float, rssi_min: float, sigma_q: float, sigma_ref: float, n_detect: int
) -> float:
    """モジュールdocstringに記載の重み式をテスト側で独立実装したもの。"""
    w_base = 10.0 ** ((rssi_median - rssi_min) / 10.0)
    variance_factor = 1.0 + (sigma_q / sigma_ref) ** 2
    detection_factor = min(n_detect / 10.0, 1.0)
    return w_base / variance_factor * detection_factor


def _expected_varweight_estimate(
    test_scans: pd.DataFrame, ap_coords: pd.DataFrame, loc_p: int, sigma_ref: float
) -> tuple[float, float]:
    """predict() を呼ばずに、公表済みの重み式から (x, y) を独立に再計算する。

    σ_q / n_detect は勝者 variant の band に対応する ap_band_fingerprint 行
    からのみ取る（F1: 物理AP全体の pooled 値ではない）。
    """
    fp = reproduction_fingerprint(candidate_medians(test_scans, ap_coords))
    loc_fp = fp[fp["location_p"] == loc_p]

    ab_fp = ap_band_fingerprint(test_scans, ap_coords=None)
    ab_fp = ab_fp[ab_fp["location_p"] == loc_p]

    rssi_min = loc_fp["rssi_median"].min()
    weights = []
    for _, row in loc_fp.iterrows():
        band = band_of(row["frequency"])
        match = ab_fp[(ab_fp["ap_name"] == row["ap_name"]) & (ab_fp["band"] == band)]
        sigma_q = float(match["rssi_std"].iloc[0])
        n_detect = int(match["n_detect"].iloc[0])
        weights.append(_varweight_weight(row["rssi_median"], rssi_min, sigma_q, sigma_ref, n_detect))

    weights = np.array(weights)
    x = float((weights * loc_fp["x"].to_numpy()).sum() / weights.sum())
    y = float((weights * loc_fp["y"].to_numpy()).sum() / weights.sum())
    return x, y


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


# --- 1: Synthetic two-AP case (variance downweighting) -----------------------

def test_wcl_varweight_synthetic_high_variance_downweighted():
    """High-variance AP should have strictly lower weight in wcl_varweight vs baseline WCL.

    Setup:
      - Three APs at (0,0), (10,0), and (5,10)
      - Training location 1: All APs stable (std=0)
      - Test location 2: AP1 and AP3 stable, AP2 high variance
        → AP2's instability should reduce its weight vs baseline

    docstring 記載の重み式から独立に計算した厳密値 (atol 1e-9) との一致に加え、
    「不安定な AP から baseline より遠ざかる」ことも確認する（F7: 従来は
    "baseline と違う" としか検証しておらず、式の正しさを保証していなかった）。
    """
    # Training data for location 1 (stable) - use valid wing names for reproduction_fingerprint
    train = pd.DataFrame({
        "location_p": [1] * 30,
        "ap_name": ["AP-C0-3F-TEST1"] * 10 + ["AP-C2-3F-TEST2"] * 10 + ["AP-C3-3F-TEST3"] * 10,
        "ssid": ["test"] * 30,
        "frequency": [2400] * 30,
        "rssi": [-40] * 10 + [-70] * 10 + [-50] * 10,
        "count": list(range(10)) + list(range(10)) + list(range(10)),
    })

    # Test query at location 2 - AP2 varies significantly (high variance)
    # AP1: stable -40 dBm (std ≈ 0)
    # AP2: varies -60 to -80 dBm with median -70 (std ≈ 7.4)
    # AP3: stable -50 dBm (std ≈ 0)
    ap2_rssi = [-60, -62, -65, -68, -70, -72, -75, -78, -80, -65]  # High variance
    test = pd.concat([
        pd.DataFrame({
            "location_p": [2] * 10,
            "ap_name": ["AP-C0-3F-TEST1"] * 10,
            "ssid": ["test"] * 10,
            "frequency": [2400] * 10,
            "rssi": [-40] * 10,  # Stable
            "count": list(range(10)),
        }),
        pd.DataFrame({
            "location_p": [2] * 10,
            "ap_name": ["AP-C2-3F-TEST2"] * 10,
            "ssid": ["test"] * 10,
            "frequency": [2400] * 10,
            "rssi": ap2_rssi,  # High variance
            "count": list(range(10)),
        }),
        pd.DataFrame({
            "location_p": [2] * 10,
            "ap_name": ["AP-C3-3F-TEST3"] * 10,
            "ssid": ["test"] * 10,
            "frequency": [2400] * 10,
            "rssi": [-50] * 10,  # Stable
            "count": list(range(10)),
        })
    ], ignore_index=True)

    # AP coordinates - use valid wing names
    ap_coords = pd.DataFrame({
        "ap_name": ["AP-C0-3F-TEST1", "AP-C2-3F-TEST2", "AP-C3-3F-TEST3"],
        "x": [0.0, 10.0, 5.0],
        "y": [0.0, 0.0, 10.0],
        "floor": [3, 3, 3],
    })

    location_coords = pd.DataFrame({
        "location_p": [1],
        "x": [5.0],
        "y": [0.0],
    })

    # Get baseline WCL estimates for reference
    from icsr8.estimators import estimate_wcl

    test_cands = candidate_medians(test, ap_coords)
    test_fp = reproduction_fingerprint(test_cands)

    baseline_est = estimate_wcl(test_fp)

    # Get wcl_varweight estimates
    varweight_est = run_method(
        "wcl_varweight",
        train, test,
        ap_coords, location_coords
    )

    sigma_ref = _sigma_ref_from_train(train)
    expected_x, expected_y = _expected_varweight_estimate(test, ap_coords, 2, sigma_ref)

    assert varweight_est["x"].iloc[0] == pytest.approx(expected_x, abs=1e-9)
    assert varweight_est["y"].iloc[0] == pytest.approx(expected_y, abs=1e-9)

    # AP2 (the high-variance AP, at (10, 0)) should be pulled toward less
    # strongly than under baseline WCL, i.e. the estimate should end up
    # farther from AP2 than baseline WCL's estimate.
    ap2_xy = np.array([10.0, 0.0])
    varweight_dist = np.hypot(
        varweight_est["x"].iloc[0] - ap2_xy[0], varweight_est["y"].iloc[0] - ap2_xy[1]
    )
    baseline_dist = np.hypot(
        baseline_est["x"].iloc[0] - ap2_xy[0], baseline_est["y"].iloc[0] - ap2_xy[1]
    )
    assert varweight_dist > baseline_dist


# --- 2: n_detect=10 and sigma_q=0 → weight multiplier exactly 1 --------------

def test_wcl_varweight_perfect_detection_equals_baseline():
    """全AP が 10 scan 全てで検出され (n_detect=10) 分散ゼロ (sigma_q=0) のとき、
    重み倍率が厳密に 1 になり baseline WCL と一致することを検証する。

    Why not 1行/AP (count=[0]) で組む: n_detect が意図せず 1 になり、weight の
    共通因子 0.1 が分子分母でキャンセルして「たまたま」テストが通ってしまう
    (F3)。10 scan 分の行を distinct count で与えないと n_detect=10 を実際には
    検証できない。
    """
    # Training: single location with stable APs - use valid wing names
    train = pd.DataFrame({
        "location_p": [1] * 30,
        "ap_name": ["AP-C0-3F-TEST1"] * 10 + ["AP-C2-3F-TEST2"] * 10 + ["AP-C3-3F-TEST3"] * 10,
        "ssid": ["test"] * 30,
        "frequency": [2400] * 30,
        "rssi": [-50] * 10 + [-60] * 10 + [-55] * 10,
        "count": list(range(10)) + list(range(10)) + list(range(10)),
    })

    # Test: 10 rows/AP, distinct counts 0..9, constant rssi per AP
    # (perfect detection, zero variance).
    test = pd.concat([
        pd.DataFrame({
            "location_p": [2] * 10,
            "ap_name": ["AP-C0-3F-TEST1"] * 10,
            "ssid": ["test"] * 10,
            "frequency": [2400] * 10,
            "rssi": [-50] * 10,
            "count": list(range(10)),
        }),
        pd.DataFrame({
            "location_p": [2] * 10,
            "ap_name": ["AP-C2-3F-TEST2"] * 10,
            "ssid": ["test"] * 10,
            "frequency": [2400] * 10,
            "rssi": [-60] * 10,
            "count": list(range(10)),
        }),
        pd.DataFrame({
            "location_p": [2] * 10,
            "ap_name": ["AP-C3-3F-TEST3"] * 10,
            "ssid": ["test"] * 10,
            "frequency": [2400] * 10,
            "rssi": [-55] * 10,
            "count": list(range(10)),
        })
    ], ignore_index=True)

    ap_coords = pd.DataFrame({
        "ap_name": ["AP-C0-3F-TEST1", "AP-C2-3F-TEST2", "AP-C3-3F-TEST3"],
        "x": [0.0, 10.0, 5.0],
        "y": [0.0, 0.0, 10.0],
        "floor": [3, 3, 3],
    })

    location_coords = pd.DataFrame({
        "location_p": [1],
        "x": [5.0],
        "y": [0.0],
    })

    from icsr8.estimators import estimate_wcl

    test_cands = candidate_medians(test, ap_coords)
    test_fp = reproduction_fingerprint(test_cands)

    baseline_est = estimate_wcl(test_fp)

    varweight_est = run_method(
        "wcl_varweight",
        train, test,
        ap_coords, location_coords
    )

    # Weight multiplier is exactly 1 (variance_factor=1, detection_factor=1),
    # so the estimate must equal baseline WCL to numerical precision, not the
    # loose 0.1 m tolerance the accidental-pass version used.
    assert varweight_est["x"].iloc[0] == pytest.approx(baseline_est["x"].iloc[0], abs=1e-9)
    assert varweight_est["y"].iloc[0] == pytest.approx(baseline_est["y"].iloc[0], abs=1e-9)


# --- 3: sigma_ref floor at SIGMA_MIN_DB (constant training data) -----

def test_wcl_varweight_constant_rssi_no_divbyzero():
    """When training data has constant RSSI (all std=0),
    sigma_ref should floor to SIGMA_MIN_DB and avoid div-by-zero.
    """
    # Training: constant RSSI (all measurements identical) - use valid wing names
    train = pd.DataFrame({
        "location_p": [1] * 30,
        "ap_name": ["AP-C0-3F-TEST1"] * 10 + ["AP-C2-3F-TEST2"] * 10 + ["AP-C3-3F-TEST3"] * 10,
        "ssid": ["test"] * 30,
        "frequency": [2400] * 30,
        "rssi": [-50] * 10 + [-60] * 10 + [-55] * 10,
        "count": list(range(10)) + list(range(10)) + list(range(10)),
    })

    test = pd.concat([
        pd.DataFrame({
            "location_p": [2],
            "ap_name": ["AP-C0-3F-TEST1"],
            "ssid": ["test"],
            "frequency": [2400],
            "rssi": [-50],
            "count": [0],
        }),
        pd.DataFrame({
            "location_p": [2],
            "ap_name": ["AP-C2-3F-TEST2"],
            "ssid": ["test"],
            "frequency": [2400],
            "rssi": [-60],
            "count": [0],
        }),
        pd.DataFrame({
            "location_p": [2],
            "ap_name": ["AP-C3-3F-TEST3"],
            "ssid": ["test"],
            "frequency": [2400],
            "rssi": [-55],
            "count": [0],
        })
    ], ignore_index=True)

    ap_coords = pd.DataFrame({
        "ap_name": ["AP-C0-3F-TEST1", "AP-C2-3F-TEST2", "AP-C3-3F-TEST3"],
        "x": [0.0, 10.0, 5.0],
        "y": [0.0, 0.0, 10.0],
        "floor": [3, 3, 3],
    })

    location_coords = pd.DataFrame({
        "location_p": [1],
        "x": [5.0],
        "y": [0.0],
    })

    # Should not raise div-by-zero error
    varweight_est = run_method(
        "wcl_varweight",
        train, test,
        ap_coords, location_coords
    )

    # Should return valid finite coordinates
    assert np.isfinite(varweight_est["x"].iloc[0])
    assert np.isfinite(varweight_est["y"].iloc[0])


# --- 4: Real-data: estimates differ from baseline at ≥1 location --------

def test_wcl_varweight_differs_from_baseline_real_data(
    protocol_a_fold, ap_coords, location_coords
):
    """Real-world protocol A: wcl_varweight should produce different estimates
    than baseline WCL at least at one location (due to variance/detection weighting).
    """
    fold = protocol_a_fold

    baseline_est = run_method(
        "wcl",
        fold.train_scans, fold.test_scans,
        ap_coords, location_coords
    )

    varweight_est = run_method(
        "wcl_varweight",
        fold.train_scans, fold.test_scans,
        ap_coords, location_coords
    )

    # At least some locations should differ
    diff_x = (varweight_est["x"] - baseline_est["x"]).abs()
    diff_y = (varweight_est["y"] - baseline_est["y"]).abs()
    max_diff = np.maximum(diff_x, diff_y).max()

    # Require at least 0.05m difference in at least one location
    assert max_diff > 0.05, f"Expected difference from baseline, but max_diff={max_diff}"


# --- 5: Smoke test (contract) -----------------------------------------------

def test_wcl_varweight_smoke(protocol_a_fold, ap_coords, location_coords):
    """Mandatory smoke test: basic functionality and contract compliance.

    Per contract:
      - fold.train_scans and fold.test_scans loaded successfully
      - Method produces output with 59 rows, columns [location_p, x, y]
      - No NaN values
      - location_p matches expected 1–59 range
    """
    fold = protocol_a_fold

    est = run_method(
        "wcl_varweight",
        fold.train_scans, fold.test_scans,
        ap_coords, location_coords
    )

    # Check shape and columns
    assert len(est) == 59, f"Expected 59 rows, got {len(est)}"
    assert set(est.columns) == {"location_p", "x", "y"}, \
        f"Expected columns {{location_p, x, y}}, got {set(est.columns)}"

    # Check for NaN
    assert not est.isna().any().any(), "Found NaN values in estimate"

    # Check location_p range
    assert est["location_p"].min() == 1, f"Min location_p should be 1, got {est['location_p'].min()}"
    assert est["location_p"].max() == 59, f"Max location_p should be 59, got {est['location_p'].max()}"

    # Check all are finite
    assert np.isfinite(est[["x", "y"]]).all().all(), "Found non-finite coordinates"


# --- 6: <3 candidates raises, parity with baseline WCL -----------------------

def test_wcl_varweight_requires_three_candidates():
    """候補が3未満の地点では baseline "wcl" と同様に ValueError を送出する。

    Baseline WCL は _require_three で <3 候補を弾くが、wcl_varweight は
    ガードがなく黙って1-2候補で推定してしまっていた（重大度: MED, F2）。
    """
    train = pd.DataFrame({
        "location_p": [1] * 20,
        "ap_name": ["AP-C0-3F-TEST1"] * 10 + ["AP-C2-3F-TEST2"] * 10,
        "ssid": ["test"] * 20,
        "frequency": [2400] * 20,
        "rssi": [-50] * 10 + [-60] * 10,
        "count": list(range(10)) * 2,
    })

    test = pd.concat([
        pd.DataFrame({
            "location_p": [2],
            "ap_name": ["AP-C0-3F-TEST1"],
            "ssid": ["test"],
            "frequency": [2400],
            "rssi": [-50],
            "count": [0],
        }),
        pd.DataFrame({
            "location_p": [2],
            "ap_name": ["AP-C2-3F-TEST2"],
            "ssid": ["test"],
            "frequency": [2400],
            "rssi": [-60],
            "count": [0],
        }),
    ], ignore_index=True)

    ap_coords = pd.DataFrame({
        "ap_name": ["AP-C0-3F-TEST1", "AP-C2-3F-TEST2"],
        "x": [0.0, 10.0],
        "y": [0.0, 0.0],
        "floor": [3, 3],
    })

    location_coords = pd.DataFrame({"location_p": [1], "x": [5.0], "y": [0.0]})

    with pytest.raises(ValueError):
        run_method("wcl_varweight", train, test, ap_coords, location_coords)

    # Baseline "wcl" raises the same way on the identical <3-candidate input.
    with pytest.raises(ValueError):
        run_method("wcl", train, test, ap_coords, location_coords)


# --- 7: sigma_q/n_detect must come from the winning variant's band only (F1) --

def test_wcl_varweight_sigma_q_not_pooled_across_bands():
    """multi-band 物理AP で、σ_q/n_detect が (勝者variantの) band 単位で
    取られ、他 band を pooling してリークしないことを検証する。

    AP-B は 2.4G (安定, 弱い分散) と 5G (荒れた分散だが中央値は弱い) の
    2 variant を持つ。reproduction_fingerprint は 2.4G を勝者として残す。
    pooled 実装 (修正前) だと 5G の巨大な分散が sigma_q に混入し、AP-B の
    重みを不当に下げてしまう。
    """
    train = pd.DataFrame({
        "location_p": [1] * 30,
        "ap_name": ["AP-C0-3F-A"] * 10 + ["AP-C2-3F-B"] * 10 + ["AP-C3-3F-C"] * 10,
        "ssid": ["tutwifi"] * 30,
        "frequency": [2400] * 30,
        "rssi": [-50] * 10 + [-55] * 10 + [-60] * 10,
        "count": list(range(10)) * 3,
    })

    ap_a = pd.DataFrame({
        "location_p": [2] * 10,
        "ap_name": ["AP-C0-3F-A"] * 10,
        "ssid": ["tutwifi"] * 10,
        "frequency": [2400] * 10,
        "rssi": [-50] * 10,
        "count": list(range(10)),
    })
    # AP-B winning variant: 2.4G, stronger median (-45), small variance.
    ap_b_24g = pd.DataFrame({
        "location_p": [2] * 10,
        "ap_name": ["AP-C2-3F-B"] * 10,
        "ssid": ["tutwifi"] * 10,
        "frequency": [2400] * 10,
        "rssi": [-45, -45, -46, -44, -45, -45, -46, -44, -45, -45],
        "count": list(range(10)),
    })
    # AP-B losing variant: 5G, weaker median (-75), wildly noisy.
    ap_b_5g = pd.DataFrame({
        "location_p": [2] * 10,
        "ap_name": ["AP-C2-3F-B"] * 10,
        "ssid": ["tutwifi2025"] * 10,
        "frequency": [5180] * 10,
        "rssi": [-60, -90, -65, -85, -60, -90, -65, -85, -60, -90],
        "count": list(range(10)),
    })
    ap_c = pd.DataFrame({
        "location_p": [2] * 10,
        "ap_name": ["AP-C3-3F-C"] * 10,
        "ssid": ["tutwifi"] * 10,
        "frequency": [2400] * 10,
        "rssi": [-60] * 10,
        "count": list(range(10)),
    })
    test = pd.concat([ap_a, ap_b_24g, ap_b_5g, ap_c], ignore_index=True)

    ap_coords = pd.DataFrame({
        "ap_name": ["AP-C0-3F-A", "AP-C2-3F-B", "AP-C3-3F-C"],
        "x": [0.0, 10.0, 5.0],
        "y": [0.0, 0.0, 10.0],
        "floor": [3, 3, 3],
    })
    location_coords = pd.DataFrame({"location_p": [1], "x": [5.0], "y": [0.0]})

    # Sanity: reproduction_fingerprint keeps the 2.4G variant as AP-B's winner.
    loc_fp = reproduction_fingerprint(candidate_medians(test, ap_coords))
    loc_fp = loc_fp[loc_fp["location_p"] == 2]
    assert len(loc_fp) == 3
    b_row = loc_fp[loc_fp["ap_name"] == "AP-C2-3F-B"].iloc[0]
    assert b_row["frequency"] == 2400

    est = run_method("wcl_varweight", train, test, ap_coords, location_coords)

    sigma_ref = _sigma_ref_from_train(train)
    expected_x, expected_y = _expected_varweight_estimate(test, ap_coords, 2, sigma_ref)

    assert est["x"].iloc[0] == pytest.approx(expected_x, abs=1e-9)
    assert est["y"].iloc[0] == pytest.approx(expected_y, abs=1e-9)
