"""L 字廊下の 1 次元弧長座標系（doc/improvement_methods_note.txt 手法7）。

折れ線 C→C2→C3 上の点を弧長 s∈[0, 116] で表す。s=0 が (32,0)、
s=116 が (28,56)。角部（測定点が壁越しに近接する箇所）で Euclidean 距離が
実際の廊下移動距離を過小評価する問題を、弧長差 |s_p − s_q| で置き換える。
"""

from __future__ import annotations

from math import hypot, isfinite

import pandas as pd

from icsr8.constants import CORRIDOR_SEGMENTS

_SEGMENT_NAMES: tuple[str, ...] = ("C", "C2", "C3")
# 退化区分（長さ 0）は _project の t 計算で 0 除算を招くため import 時に弾く。
assert all(
    hypot(bx - ax, by - ay) > 0.0 for (ax, ay), (bx, by) in CORRIDOR_SEGMENTS
), "every CORRIDOR_SEGMENT must have nonzero length"
_TOTAL_LENGTH: float = sum(
    hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in CORRIDOR_SEGMENTS
)


def _project(x: float, y: float) -> tuple[float, tuple[float, float], float, str]:
    if not (isfinite(x) and isfinite(y)):
        raise ValueError(f"non-finite coordinate: x={x!r}, y={y!r}")
    best: tuple[float, tuple[float, float], float, str] | None = None
    cum = 0.0
    for name, ((ax, ay), (bx, by)) in zip(_SEGMENT_NAMES, CORRIDOR_SEGMENTS):
        abx, aby = bx - ax, by - ay
        seg_len = hypot(abx, aby)
        t = ((x - ax) * abx + (y - ay) * aby) / (seg_len * seg_len)
        t = min(max(t, 0.0), 1.0)
        cx, cy = ax + t * abx, ay + t * aby
        dist = hypot(x - cx, y - cy)
        arc = cum + t * seg_len
        # Why not <=: 同距離なら先行区分（C < C2 < C3）を残す決定的タイブレーク。
        if best is None or dist < best[2]:
            best = (arc, (cx, cy), dist, name)
        cum += seg_len
    assert best is not None
    return best


def xy_to_arclength(x: float, y: float) -> float:
    return _project(x, y)[0]


def arclength_to_xy(s: float) -> tuple[float, float]:
    s = min(max(s, 0.0), _TOTAL_LENGTH)
    cum = 0.0
    for (ax, ay), (bx, by) in CORRIDOR_SEGMENTS:
        seg_len = hypot(bx - ax, by - ay)
        if s <= cum + seg_len:
            t = (s - cum) / seg_len
            return (ax + t * (bx - ax), ay + t * (by - ay))
        cum += seg_len
    return CORRIDOR_SEGMENTS[-1][1]


def project_to_corridor(x: float, y: float) -> tuple[float, float]:
    return _project(x, y)[1]


def segment_of(x: float, y: float) -> str:
    return _project(x, y)[3]


def geodesic_distance(p: tuple[float, float], q: tuple[float, float]) -> float:
    return abs(xy_to_arclength(*p) - xy_to_arclength(*q))


def assert_locations_on_corridor(loc_df: pd.DataFrame, tol: float = 0.5) -> None:
    # Why not just `_project(...)[2] > tol`: 非有限座標は _project が ValueError を
    # 投げてしまう上、`NaN > tol` は False なので静かに通過する。非有限行は明示的に
    # 違反として計上する。
    offenders = [
        row.location_p
        for row in loc_df.itertuples()
        if not (isfinite(row.x) and isfinite(row.y)) or _project(row.x, row.y)[2] > tol
    ]
    if offenders:
        raise AssertionError(
            f"locations off corridor (orthogonal distance > {tol} m): {offenders}"
        )
