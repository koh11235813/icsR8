"""Tests for wcl_blacklist method: WCL with blacklisted APs removed from candidates."""

from pathlib import Path

import pandas as pd
import pytest

from icsr8.constants import BLACKLIST_APS
from icsr8.estimators import estimate_with_trace
from icsr8.fingerprint import candidate_medians, reproduction_fingerprint
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import run_method
from icsr8.protocols import iter_protocol_a


# --- fixtures ----------------------------------------------------------------

@pytest.fixture(scope="session")
def ap_coords(dataset_dir: Path) -> pd.DataFrame:
    return load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")


@pytest.fixture(scope="session")
def location_coords(dataset_dir: Path) -> pd.DataFrame:
    df = load_location_coords(dataset_dir / "location_coordinate_C.csv")
    return df[["location_p", "x", "y"]]


@pytest.fixture(scope="session")
def folds(rawdata_root: Path):
    """Protocol A folds: (forward->backward, backward->forward)."""
    return list(iter_protocol_a(
        load_raw_scans("forward", rawdata_root),
        load_raw_scans("backward", rawdata_root)
    ))


# --- test 1: production path matches baseline WCL with blacklisted AP removed ---

def test_wcl_blacklist_matches_wcl_without_blacklisted_ap():
    """run_method("wcl_blacklist", ...) を、baseline "wcl" を同じ scans から
    blacklist AP の行だけ手で除いたもので実行した結果と一致させ、blacklist AP
    を含めたままの baseline "wcl" とは異なることを確認する。

    Why not 旧テスト (candidate_medians/reproduction_fingerprint を直接呼んで
    フィルタを手で再現): production の run_method 経路を一切通らず、
    wcl_blacklist.predict() 自体にバグがあっても検出できなかった (F6)。
    """
    # AP-C0-3F-04 (blacklist 対象) を最強信号にして、廊下外配置でも合成データ上
    # top-3 に選ばれる状況を作る。他に 3 AP を用意し、除外後も >=3 候補を残す。
    ap_names = ["AP-C0-3F-04", "AP-C2-3F-B", "AP-C3-3F-C", "AP-C0-3F-D"]
    xs = [-1.9, 10.0, 20.0, 30.0]
    ys = [-11.2, 0.0, 0.0, 0.0]
    rssis = [-30, -50, -55, -60]  # AP-C0-3F-04 strongest

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
        "floor": [3] * 4,
    })
    location_coords = pd.DataFrame({"location_p": [1], "x": [0.0], "y": [0.0]})

    blacklist_est = run_method("wcl_blacklist", scans, scans, ap_coords, location_coords)
    baseline_included = run_method("wcl", scans, scans, ap_coords, location_coords)

    scans_excluded = scans[~scans["ap_name"].isin(BLACKLIST_APS)]
    baseline_excluded = run_method("wcl", scans_excluded, scans_excluded, ap_coords, location_coords)

    assert blacklist_est["x"].iloc[0] == pytest.approx(baseline_excluded["x"].iloc[0], abs=1e-9)
    assert blacklist_est["y"].iloc[0] == pytest.approx(baseline_excluded["y"].iloc[0], abs=1e-9)

    assert (
        blacklist_est["x"].iloc[0] != pytest.approx(baseline_included["x"].iloc[0], abs=1e-9)
        or blacklist_est["y"].iloc[0] != pytest.approx(baseline_included["y"].iloc[0], abs=1e-9)
    )


# --- test 2: estimates differ from baseline at ≥1 location & match elsewhere ----

def test_wcl_blacklist_differs_from_baseline(ap_coords: pd.DataFrame, location_coords: pd.DataFrame, folds):
    """Verify that wcl_blacklist estimates differ from baseline WCL at ≥1 location
    where AP-C0-3F-04 was in baseline top-3, and are identical elsewhere.
    """
    fold = folds[0]

    # Run baseline and method
    baseline_est, baseline_trace = estimate_with_trace(
        reproduction_fingerprint(candidate_medians(fold.test_scans, ap_coords)),
        "wcl"
    )
    method_est = run_method(
        "wcl_blacklist",
        fold.train_scans,
        fold.test_scans,
        ap_coords,
        location_coords
    )

    # Sort both by location_p for comparison
    baseline_est = baseline_est.sort_values("location_p").reset_index(drop=True)
    method_est = method_est.sort_values("location_p").reset_index(drop=True)

    # Verify location_p match
    assert baseline_est["location_p"].tolist() == method_est["location_p"].tolist()

    # For each location, check if AP-C0-3F-04 was in baseline top-3
    differs_at_least_once = False
    for idx, (_, baseline_row) in enumerate(baseline_est.iterrows()):
        loc_p = baseline_row["location_p"]
        method_row = method_est.iloc[idx]

        # Extract baseline top-3 for this location
        baseline_top3 = baseline_trace[
            baseline_trace["location_p"] == loc_p
        ].head(3)

        baseline_has_blacklist = baseline_top3["ap_name"].isin(BLACKLIST_APS).any()

        # Compute difference
        dx = abs(baseline_row["x"] - method_row["x"])
        dy = abs(baseline_row["y"] - method_row["y"])
        diff = (dx ** 2 + dy ** 2) ** 0.5  # L2 distance

        if baseline_has_blacklist:
            # Estimates SHOULD differ when blacklist AP was in top-3
            # (not strict: might be the same by chance, but we expect at least 1 diff)
            pass
        else:
            # Estimates MUST be identical when blacklist AP was NOT in top-3
            assert diff < 1e-6, \
                f"location_p={loc_p}: estimates differ but blacklist AP not in top-3; " \
                f"baseline=({baseline_row['x']:.3f}, {baseline_row['y']:.3f}), " \
                f"method=({method_row['x']:.3f}, {method_row['y']:.3f})"

        if baseline_has_blacklist and diff > 1e-6:
            differs_at_least_once = True

    assert differs_at_least_once, \
        "wcl_blacklist should differ from baseline at ≥1 location where " \
        "AP-C0-3F-04 was in top-3"


# --- test 3: smoke test (contract requirement) -------------------------------

def test_wcl_blacklist_smoke_test(ap_coords: pd.DataFrame, location_coords: pd.DataFrame, folds):
    """Smoke test per contract: 59 rows, [location_p, x, y] columns, no NaN."""
    fold = folds[0]

    est = run_method(
        "wcl_blacklist",
        fold.train_scans,
        fold.test_scans,
        ap_coords,
        location_coords
    )

    # Verify shape and columns
    assert len(est) == 59, f"expected 59 rows, got {len(est)}"
    assert list(est.columns) == ["location_p", "x", "y"], \
        f"expected columns [location_p, x, y], got {list(est.columns)}"

    # Verify no NaN
    assert not est.isna().any().any(), "found NaN values in estimates"

    # Verify location_p range
    assert est["location_p"].min() == 1 and est["location_p"].max() == 59
