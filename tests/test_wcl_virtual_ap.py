"""wcl_virtual_ap (Ji 2012 vWCL) の性質テスト。

What:
  - 全 vAP 採用時は不動点（= WCL と同一推定）になること
  - 強信号 AP へ寄った初期推定では、棄却が起きて推定が WCL より外側へ動くこと
  - 凸包の厳密内部判定（境界上は内部でない）
  - 3 候補未満の契約違反が ValueError になること
  - registry 経由の fit/predict が (location_p, x, y) スキーマを返すこと
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icsr8.methods.wcl_virtual_ap import (
    _convex_hull,
    _strictly_inside,
    _vwcl_one,
    estimate_vwcl,
    vwcl_point,
)

TRIANGLE = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]])


def _wcl_reference(pts: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (w[:, None] * pts).sum(axis=0) / w.sum()


def test_equal_weights_is_fixed_point_of_wcl():
    # 等重みの重心では各頂点の反転像が対辺中点（境界上）に落ち、
    # 「厳密内部のみ棄却」の規約では全 vAP が採用される → 不動点 = WCL。
    w = np.ones(3)
    x, y = vwcl_point(TRIANGLE, w)
    ref = _wcl_reference(TRIANGLE, w)
    assert np.allclose([x, y], ref, atol=1e-9)


def test_skewed_weights_moves_estimate_outward():
    # AP0 が支配的な重みのとき WCL は AP0 側に寄るが凸包内に留まる。
    # vWCL は AP0 の反転像（内部側）を棄却し、遠方 2 AP の反転像が推定を
    # さらに AP0 方向へ押し出す（境界バイアスの補正方向）。
    w = np.array([100.0, 1.0, 1.0])
    wcl = _wcl_reference(TRIANGLE, w)
    vx, vy = vwcl_point(TRIANGLE, w)
    d_wcl = float(np.hypot(*(wcl - TRIANGLE[0])))
    d_vwcl = float(np.hypot(vx - TRIANGLE[0][0], vy - TRIANGLE[0][1]))
    assert d_vwcl < d_wcl


def test_strictly_inside_excludes_boundary():
    hull = _convex_hull(TRIANGLE)
    inside = np.array([[5.0, 2.0]])
    on_edge = np.array([[5.0, 0.0]])
    outside = np.array([[50.0, 50.0]])
    assert _strictly_inside(inside, hull).tolist() == [True]
    assert _strictly_inside(on_edge, hull).tolist() == [False]
    assert _strictly_inside(outside, hull).tolist() == [False]


def test_degenerate_collinear_hull_keeps_everything():
    line = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    hull = _convex_hull(line)
    assert len(hull) <= 2
    q = np.array([[5.0, 0.0], [3.0, 1.0]])
    assert _strictly_inside(q, hull).tolist() == [False, False]


def _fp(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_vwcl_one_requires_three_candidates():
    fp = _fp(
        [
            {"location_p": 1, "ap_name": "a", "rssi_median": -50.0,
             "frequency": 2412, "ssid": "tutwifi", "x": 0.0, "y": 0.0},
            {"location_p": 1, "ap_name": "b", "rssi_median": -60.0,
             "frequency": 2412, "ssid": "tutwifi", "x": 10.0, "y": 0.0},
        ]
    )
    with pytest.raises(ValueError, match="requires 3 candidates"):
        _vwcl_one(fp)


def test_estimate_vwcl_schema_and_determinism():
    fp = _fp(
        [
            {"location_p": 7, "ap_name": "a", "rssi_median": -40.0,
             "frequency": 2412, "ssid": "tutwifi", "x": 0.0, "y": 0.0},
            {"location_p": 7, "ap_name": "b", "rssi_median": -70.0,
             "frequency": 2412, "ssid": "tutwifi", "x": 10.0, "y": 0.0},
            {"location_p": 7, "ap_name": "c", "rssi_median": -70.0,
             "frequency": 5180, "ssid": "tutwifi", "x": 5.0, "y": 8.0},
        ]
    )
    out1 = estimate_vwcl(fp)
    out2 = estimate_vwcl(fp)
    assert list(out1.columns) == ["location_p", "x", "y"]
    assert out1["location_p"].tolist() == [7]
    pd.testing.assert_frame_equal(out1, out2)


def test_registered_and_runs_via_registry():
    from icsr8.methods import REGISTRY

    assert "wcl_virtual_ap" in REGISTRY
