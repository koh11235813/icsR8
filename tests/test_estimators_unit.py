import numpy as np
import pandas as pd
import pytest

from icsr8.estimators import (
    estimate_cla,
    estimate_pbl,
    estimate_wcl,
    select_top_k,
)


@pytest.fixture()
def toy_fp() -> pd.DataFrame:
    """4 candidate APs at location 1, hand-computable expected outputs."""
    return pd.DataFrame(
        [
            {"location_p": 1, "ap_name": "A", "ssid": "tutwifi",
             "frequency": 2412, "rssi_median": -50, "x": 0.0, "y": 0.0},
            {"location_p": 1, "ap_name": "B", "ssid": "tutwifi",
             "frequency": 2412, "rssi_median": -60, "x": 10.0, "y": 0.0},
            {"location_p": 1, "ap_name": "C", "ssid": "tutwifi",
             "frequency": 2412, "rssi_median": -70, "x": 0.0, "y": 10.0},
            {"location_p": 1, "ap_name": "D", "ssid": "tutwifi",
             "frequency": 2412, "rssi_median": -80, "x": 100.0, "y": 100.0},
        ]
    )


def test_pbl_picks_strongest_ap(toy_fp):
    out = estimate_pbl(toy_fp)
    assert len(out) == 1
    row = out.iloc[0]
    assert (row["x"], row["y"]) == (0.0, 0.0)


def test_cla_returns_centroid_of_top3(toy_fp):
    out = estimate_cla(toy_fp)
    row = out.iloc[0]
    assert row["x"] == pytest.approx(10.0 / 3)
    assert row["y"] == pytest.approx(10.0 / 3)


def test_wcl_weight_formula_against_hand_calc(toy_fp):
    out = estimate_wcl(toy_fp)
    row = out.iloc[0]
    # weights = 10^((-50+70)/10), 10^((-60+70)/10), 10^0 = 100, 10, 1
    assert row["x"] == pytest.approx(100 / 111)
    assert row["y"] == pytest.approx(10 / 111)


def test_select_top_k_deterministic_tie_break_by_frequency_then_ssid_then_ap_name():
    fp = pd.DataFrame(
        [
            {"location_p": 1, "ap_name": "C", "ssid": "tutwifi", "frequency": 5180,
             "rssi_median": -50, "x": 0.0, "y": 0.0},
            {"location_p": 1, "ap_name": "A", "ssid": "tutwifi2025", "frequency": 2412,
             "rssi_median": -50, "x": 1.0, "y": 0.0},
            {"location_p": 1, "ap_name": "B", "ssid": "tutwifi", "frequency": 2412,
             "rssi_median": -50, "x": 2.0, "y": 0.0},
            {"location_p": 1, "ap_name": "D", "ssid": "tutwifi", "frequency": 5180,
             "rssi_median": -50, "x": 3.0, "y": 0.0},
        ]
    )
    top = select_top_k(fp, k=3)
    # Primary tie: freq asc → 2412 group {A, B} first. Within 2412 ssid asc:
    #   tutwifi=B before tutwifi2025=A. Then freq=5180 ssid=tutwifi: C (ap_name asc) before D.
    assert top["ap_name"].tolist() == ["B", "A", "C"]


def test_select_top_k_random_with_seed_is_reproducible():
    fp = pd.DataFrame(
        [
            {"location_p": 1, "ap_name": chr(65 + i), "ssid": "s", "frequency": 2412,
             "rssi_median": -50, "x": float(i), "y": 0.0}
            for i in range(6)
        ]
    )
    a = select_top_k(fp, k=3, tie_break="random", rng=np.random.default_rng(0))
    b = select_top_k(fp, k=3, tie_break="random", rng=np.random.default_rng(0))
    assert a["ap_name"].tolist() == b["ap_name"].tolist()
