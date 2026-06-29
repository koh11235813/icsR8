"""Reproduce the published PBL/CLA/WCL baseline values.

5a tracer-bullet: per-method estimates at P1 (forward).
5b full reproduction:  all 59 locations × forward+backward × 3 methods, atol=0.05.
5c summary stats:      Ave/Max/Std vs. doc Table 1.
"""

from pathlib import Path

import pandas as pd
import pytest

from icsr8.estimators import estimate_cla, estimate_pbl, estimate_wcl
from icsr8.evaluate import l2_errors, summary
from icsr8.fingerprint import candidate_medians, reproduction_fingerprint
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.types import Direction

METHODS = {"pbl": estimate_pbl, "cla": estimate_cla, "wcl": estimate_wcl}

# Published Table 1 (doc/icsR8_text.txt §3.2) - Std uses ddof=0
DOC_TABLE_1 = {
    "forward": {
        "pbl": {"Ave": 4.38, "Max": 13.6, "Std": 2.82},
        "cla": {"Ave": 8.07, "Max": 24.2, "Std": 5.33},
        "wcl": {"Ave": 3.57, "Max": 11.9, "Std": 2.42},
    },
    "backward": {
        "pbl": {"Ave": 4.52, "Max": 15.6, "Std": 3.14},
        "cla": {"Ave": 7.02, "Max": 18.0, "Std": 4.22},
        "wcl": {"Ave": 3.51, "Max": 12.2, "Std": 2.54},
    },
}


# --- session fixtures --------------------------------------------------------

@pytest.fixture(scope="session")
def ap_coords(dataset_dir: Path) -> pd.DataFrame:
    return load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")


@pytest.fixture(scope="session")
def truth(dataset_dir: Path) -> pd.DataFrame:
    df = load_location_coords(dataset_dir / "location_coordinate_C.csv")
    return df[["location_p", "x", "y"]]


@pytest.fixture(scope="session")
def fp_forward(rawdata_root: Path, ap_coords: pd.DataFrame) -> pd.DataFrame:
    scans = load_raw_scans("forward", rawdata_root)
    return reproduction_fingerprint(candidate_medians(scans, ap_coords))


@pytest.fixture(scope="session")
def fp_backward(rawdata_root: Path, ap_coords: pd.DataFrame) -> pd.DataFrame:
    scans = load_raw_scans("backward", rawdata_root)
    return reproduction_fingerprint(candidate_medians(scans, ap_coords))


@pytest.fixture(scope="session")
def fingerprints(fp_forward, fp_backward) -> dict[Direction, pd.DataFrame]:
    return {"forward": fp_forward, "backward": fp_backward}


@pytest.fixture(scope="session")
def oracles(fixtures_dir: Path) -> dict[Direction, pd.DataFrame]:
    return {
        "forward": pd.read_csv(fixtures_dir / "baseline_forward.csv"),
        "backward": pd.read_csv(fixtures_dir / "baseline_backward.csv"),
    }


# --- 5a: P1 tracer-bullet ----------------------------------------------------

@pytest.mark.parametrize("method", ["pbl", "cla", "wcl"])
def test_reproduce_baseline_forward_P1(method, fp_forward, oracles):
    estimator = METHODS[method]
    fp_p1 = fp_forward[fp_forward["location_p"] == 1]
    out = estimator(fp_p1).iloc[0]
    expected = oracles["forward"].loc[oracles["forward"]["location_p"] == 1].iloc[0]
    assert out["x"] == pytest.approx(expected[f"{method}_x"], abs=0.01)
    assert out["y"] == pytest.approx(expected[f"{method}_y"], abs=0.01)


# --- 5b: all-locations reproduction -----------------------------------------

@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("method", ["pbl", "cla", "wcl"])
def test_reproduce_baseline_all_locations(direction, method, fingerprints, oracles):
    fp = fingerprints[direction]
    est = METHODS[method](fp).sort_values("location_p").reset_index(drop=True)
    oracle = oracles[direction].sort_values("location_p").reset_index(drop=True)

    # Guard against silent location_p mismatches before doing column subtraction.
    assert est["location_p"].tolist() == oracle["location_p"].tolist(), \
        f"location_p sequence mismatch for {direction}/{method}"

    diffs_x = (est["x"] - oracle[f"{method}_x"]).abs()
    diffs_y = (est["y"] - oracle[f"{method}_y"]).abs()
    bad = (diffs_x > 1e-6) | (diffs_y > 1e-6)

    if bad.any():
        offenders = pd.DataFrame({
            "location_p": est["location_p"],
            "est_x": est["x"],
            "est_y": est["y"],
            "oracle_x": oracle[f"{method}_x"],
            "oracle_y": oracle[f"{method}_y"],
            "dx": diffs_x,
            "dy": diffs_y,
        }).loc[bad]
        pytest.fail(
            f"{direction}/{method}: {int(bad.sum())} locations deviate > 1e-6m\n"
            f"{offenders.to_string(index=False)}"
        )


# --- 5c: summary statistics -------------------------------------------------

@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("method", ["pbl", "cla", "wcl"])
def test_summary_matches_doc_table1(direction, method, fingerprints, truth):
    fp = fingerprints[direction]
    est = METHODS[method](fp)
    err = l2_errors(est, truth)
    s = summary(err["error"])
    expected = DOC_TABLE_1[direction][method]
    assert s["Ave"] == pytest.approx(expected["Ave"], abs=0.01)
    assert s["Max"] == pytest.approx(expected["Max"], abs=0.05)
    assert s["Std"] == pytest.approx(expected["Std"], abs=0.01)
