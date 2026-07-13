"""廊下射影の後処理関数 apply_corridor_projection と wcl_corridor（手法7）。

doc/improvement_methods_note.txt 手法7: 推定位置を廊下の折れ線へ射影する
汎用後処理。Phase-4 harness が任意メソッドの出力へ適用できる純関数として、
また "wcl_corridor" 登録メソッド（baseline WCL + 射影）として提供する。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.io import load_ap_coords, load_location_coords
from icsr8.methods import run_method
from icsr8.methods.corridor_proj import apply_corridor_projection
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


# --- 1: fixed-point (on-corridor estimates unchanged) ------------------------

@pytest.mark.parametrize(
    "xy",
    [
        (32.0, 0.0),  # vertex
        (0.0, 0.0),  # vertex (corner)
        (0.0, 56.0),  # vertex (corner)
        (28.0, 56.0),  # vertex
        (16.0, 0.0),  # mid-segment C
        (0.0, 28.0),  # mid-segment C2
        (14.0, 56.0),  # mid-segment C3
    ],
)
def test_apply_corridor_projection_fixed_point(xy):
    est = pd.DataFrame({"location_p": [1], "x": [xy[0]], "y": [xy[1]]})
    out = apply_corridor_projection(est)
    assert out["x"].iloc[0] == pytest.approx(xy[0], abs=1e-9)
    assert out["y"].iloc[0] == pytest.approx(xy[1], abs=1e-9)


# --- 2: note's P17 example ----------------------------------------------------

def test_apply_corridor_projection_wcl_p17_example():
    est = pd.DataFrame({"location_p": [17], "x": [-0.10], "y": [11.16]})
    out = apply_corridor_projection(est)
    assert out["x"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert out["y"].iloc[0] == pytest.approx(11.16, abs=1e-9)


# --- 3: column/shape preservation incl. location_p order ---------------------

def test_apply_corridor_projection_preserves_columns_shape_and_order():
    est = pd.DataFrame(
        {
            "location_p": [3, 1, 2],
            "x": [-0.10, 32.0, 100.0],
            "y": [11.16, 0.0, 56.0],
        }
    )
    out = apply_corridor_projection(est)
    assert list(out.columns) == list(est.columns)
    assert len(out) == len(est)
    assert list(out["location_p"]) == [3, 1, 2]


def test_apply_corridor_projection_preserves_extra_columns_and_index():
    est = pd.DataFrame(
        {"location_p": [1, 2], "x": [32.0, -0.10], "y": [0.0, 11.16], "method": ["wcl", "wcl"]},
        index=[10, 20],
    )
    out = apply_corridor_projection(est)
    assert list(out.columns) == list(est.columns)
    assert list(out.index) == [10, 20]
    assert list(out["method"]) == ["wcl", "wcl"]


# --- 4: real-data equivalence: wcl_corridor == apply_corridor_projection(wcl) -

def test_wcl_corridor_equals_projected_baseline_wcl(
    protocol_a_fold, ap_coords, location_coords
):
    fold = protocol_a_fold

    baseline_est = run_method(
        "wcl", fold.train_scans, fold.test_scans, ap_coords, location_coords
    )
    projected_baseline = apply_corridor_projection(baseline_est)

    corridor_est = run_method(
        "wcl_corridor", fold.train_scans, fold.test_scans, ap_coords, location_coords
    )

    pd.testing.assert_frame_equal(
        corridor_est.reset_index(drop=True), projected_baseline.reset_index(drop=True)
    )


# --- 5: Smoke test (contract) -------------------------------------------------

def test_wcl_corridor_smoke(protocol_a_fold, ap_coords, location_coords):
    fold = protocol_a_fold

    est = run_method(
        "wcl_corridor", fold.train_scans, fold.test_scans, ap_coords, location_coords
    )

    assert len(est) == 59
    assert set(est.columns) == {"location_p", "x", "y"}
    assert not est.isna().any().any()
    assert np.isfinite(est[["x", "y"]]).all().all()
