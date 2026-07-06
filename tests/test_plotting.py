import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from icsr8.plotting import plot_error_by_position, plot_estimate_map


def test_plot_error_by_position_draws_input_series():
    errors = pd.DataFrame([
        {"location_p": 1, "error": 2.0},
        {"location_p": 2, "error": 5.0},
        {"location_p": 3, "error": 1.5},
    ])

    ax = plot_error_by_position(errors)

    assert len(ax.lines) == 1
    xdata, ydata = ax.lines[0].get_data()
    assert list(xdata) == [1, 2, 3]
    assert list(ydata) == pytest.approx([2.0, 5.0, 1.5])
    assert ax.get_xlabel() == "location_p"
    assert ax.get_ylabel() == "error"


def test_plot_error_by_position_sorts_by_location_p():
    errors = pd.DataFrame([
        {"location_p": 3, "error": 1.5},
        {"location_p": 1, "error": 2.0},
        {"location_p": 2, "error": 5.0},
    ])

    ax = plot_error_by_position(errors)

    xdata, ydata = ax.lines[0].get_data()
    assert list(xdata) == [1, 2, 3]
    assert list(ydata) == pytest.approx([2.0, 5.0, 1.5])


def test_plot_error_by_position_label_and_overlay():
    errors_a = pd.DataFrame([{"location_p": 1, "error": 2.0}, {"location_p": 2, "error": 3.0}])
    errors_b = pd.DataFrame([{"location_p": 1, "error": 4.0}, {"location_p": 2, "error": 1.0}])

    ax = plot_error_by_position(errors_a, label="forward")
    plot_error_by_position(errors_b, ax=ax, label="backward")

    assert len(ax.lines) == 2
    assert ax.lines[0].get_label() == "forward"
    assert ax.lines[1].get_label() == "backward"
    _, ydata_b = ax.lines[1].get_data()
    assert list(ydata_b) == pytest.approx([4.0, 1.0])


def test_plot_estimate_map_scatters_true_and_estimated_positions():
    estimates = pd.DataFrame([
        {"location_p": 1, "x": 1.0, "y": 1.0},
        {"location_p": 2, "x": 4.0, "y": 2.0},
    ])
    truth = pd.DataFrame([
        {"location_p": 1, "x": 0.0, "y": 0.0},
        {"location_p": 2, "x": 5.0, "y": 2.0},
    ])

    ax = plot_estimate_map(estimates, truth)

    assert len(ax.collections) == 2
    true_offsets = ax.collections[0].get_offsets()
    est_offsets = ax.collections[1].get_offsets()
    np.testing.assert_allclose(true_offsets, [[0.0, 0.0], [5.0, 2.0]])
    np.testing.assert_allclose(est_offsets, [[1.0, 1.0], [4.0, 2.0]])


def test_plot_estimate_map_includes_ap_coords_when_given():
    estimates = pd.DataFrame([{"location_p": 1, "x": 1.0, "y": 1.0}])
    truth = pd.DataFrame([{"location_p": 1, "x": 0.0, "y": 0.0}])
    ap_coords = pd.DataFrame([{"ap_name": "AP-C0-3F-01", "x": 30.1, "y": 1.0}])

    ax = plot_estimate_map(estimates, truth, ap_coords=ap_coords)

    assert len(ax.collections) == 3
    ap_offsets = ax.collections[2].get_offsets()
    np.testing.assert_allclose(ap_offsets, [[30.1, 1.0]])


def test_plot_estimate_map_reuses_given_axes():
    estimates = pd.DataFrame([{"location_p": 1, "x": 1.0, "y": 1.0}])
    truth = pd.DataFrame([{"location_p": 1, "x": 0.0, "y": 0.0}])

    fig, ax = plt.subplots()
    out = plot_estimate_map(estimates, truth, ax=ax)

    assert out is ax
