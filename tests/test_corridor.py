import pandas as pd
import pytest

from icsr8.corridor import (
    arclength_to_xy,
    assert_locations_on_corridor,
    geodesic_distance,
    project_to_corridor,
    segment_of,
    xy_to_arclength,
)
from icsr8.io import load_location_coords


@pytest.mark.parametrize(
    "xy, s",
    [
        ((32.0, 0.0), 0.0),
        ((0.0, 0.0), 32.0),
        ((0.0, 56.0), 88.0),
        ((28.0, 56.0), 116.0),
    ],
)
def test_vertices_map_to_expected_arclength(xy, s):
    assert xy_to_arclength(*xy) == pytest.approx(s, abs=1e-9)


@pytest.mark.parametrize(
    "xy, s",
    [
        ((16.0, 0.0), 16.0),
        ((0.0, 28.0), 60.0),
        ((14.0, 56.0), 102.0),
    ],
)
def test_mid_segment_points(xy, s):
    assert xy_to_arclength(*xy) == pytest.approx(s, abs=1e-9)


def test_corner_tie_break_earlier_segment_wins():
    assert segment_of(0.0, 0.0) == "C"
    assert segment_of(0.0, 56.0) == "C2"


def test_off_corridor_projection_wcl_p17():
    px, py = project_to_corridor(-0.10, 11.16)
    assert (px, py) == pytest.approx((0.0, 11.16), abs=1e-9)
    assert xy_to_arclength(-0.10, 11.16) == pytest.approx(43.16, abs=1e-9)


@pytest.mark.parametrize(
    "xy",
    [
        (32.0, 0.0),
        (16.0, 0.0),
        (0.0, 0.0),
        (0.0, 28.0),
        (0.0, 56.0),
        (14.0, 56.0),
        (28.0, 56.0),
    ],
)
def test_round_trip_for_on_corridor_points(xy):
    back = arclength_to_xy(xy_to_arclength(*xy))
    assert back == pytest.approx(xy, abs=1e-9)


def test_arclength_clamps_out_of_range():
    assert arclength_to_xy(-5.0) == pytest.approx((32.0, 0.0), abs=1e-9)
    assert arclength_to_xy(200.0) == pytest.approx((28.0, 56.0), abs=1e-9)


def test_geodesic_distance_end_to_end():
    assert geodesic_distance((32.0, 0.0), (28.0, 56.0)) == pytest.approx(116.0, abs=1e-9)


def test_real_locations_lie_on_corridor(dataset_dir):
    loc = load_location_coords(dataset_dir / "location_coordinate_C.csv")
    assert_locations_on_corridor(loc, tol=0.5)
    p1 = loc.loc[loc["location_p"] == 1].iloc[0]
    assert xy_to_arclength(p1["x"], p1["y"]) == pytest.approx(0.0, abs=1e-9)


def test_non_finite_input_raises():
    nan = float("nan")
    with pytest.raises(ValueError):
        xy_to_arclength(nan, 0.0)
    with pytest.raises(ValueError):
        project_to_corridor(0.0, nan)
    with pytest.raises(ValueError):
        segment_of(float("inf"), 0.0)
    with pytest.raises(ValueError):
        geodesic_distance((nan, 0.0), (0.0, 0.0))


def test_assert_locations_reports_nan_row_as_offender():
    loc = pd.DataFrame([
        {"location_p": 1, "x": 32.0, "y": 0.0},
        {"location_p": 2, "x": float("nan"), "y": 5.0},
    ])
    with pytest.raises(AssertionError) as exc:
        assert_locations_on_corridor(loc)
    assert "2" in str(exc.value)
