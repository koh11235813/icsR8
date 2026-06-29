"""Negative tests for guards added in response to code review."""

import pandas as pd
import pytest

from icsr8.estimators import estimate_cla, estimate_pbl, estimate_wcl
from icsr8.evaluate import l2_errors


def _fp(rows):
    return pd.DataFrame(
        [
            {"location_p": 1, "ap_name": ap, "ssid": "tutwifi", "frequency": 2412,
             "rssi_median": r, "x": x, "y": y}
            for ap, r, x, y in rows
        ]
    )


def test_l2_errors_raises_when_locations_differ():
    est = pd.DataFrame([{"location_p": 1, "x": 0.0, "y": 0.0}])
    truth = pd.DataFrame([{"location_p": 2, "x": 0.0, "y": 0.0}])
    with pytest.raises(ValueError, match="location_p"):
        l2_errors(est, truth)


def test_cla_raises_when_fewer_than_three_candidates():
    fp = _fp([("A", -50, 0.0, 0.0), ("B", -60, 10.0, 0.0)])
    with pytest.raises(ValueError, match="CLA requires 3"):
        estimate_cla(fp)


def test_wcl_raises_when_fewer_than_three_candidates():
    fp = _fp([("A", -50, 0.0, 0.0), ("B", -60, 10.0, 0.0)])
    with pytest.raises(ValueError, match="WCL requires 3"):
        estimate_wcl(fp)


def test_pbl_works_with_single_candidate():
    fp = _fp([("A", -50, 1.0, 2.0)])
    out = estimate_pbl(fp).iloc[0]
    assert (out["x"], out["y"]) == (1.0, 2.0)
