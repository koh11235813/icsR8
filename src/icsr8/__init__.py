"""Indoor localization baseline library for the tutwifi RSSI dataset.

Glossary
--------
scan
    1 サンプル = 単一 location_p × 単一 AP × 単一 (SSID, frequency) の 10 計測中 1 つ。
candidate
    (location_p, ap_name, ssid, frequency) 単位の 1 行 = 中央値 RSSI + AP 座標。
fingerprint
    1 つの location_p に紐づく candidate の集合。
estimate
    1 つの location_p に対する 2D 推定座標 (x, y)。
truth
    既知の真位置座標。

Quick usage
-----------
>>> from icsr8 import (
...     load_ap_coords, load_location_coords, load_raw_scans,
...     candidate_medians, reproduction_fingerprint,
...     estimate_wcl, l2_errors, summary,
... )
>>> ap = load_ap_coords("data/dataset/AP_coordinate_C3F.csv")
>>> truth = load_location_coords("data/dataset/location_coordinate_C.csv")
>>> scans = load_raw_scans("forward", "data/rawdata")
>>> fp = reproduction_fingerprint(candidate_medians(scans, ap))
>>> est = estimate_wcl(fp)
>>> err = l2_errors(est, truth[["location_p", "x", "y"]])
>>> summary(err["error"])  # → {"Ave": 3.569, "Max": 11.85, ...}

Method registry は明示指定を強制するため top-level に再輸出しない。
>>> from icsr8.methods import run_method, available_methods
"""

from icsr8 import methods
from icsr8.corridor import (
    arclength_to_xy,
    geodesic_distance,
    project_to_corridor,
    segment_of,
    xy_to_arclength,
)
from icsr8.estimators import (
    estimate_cla,
    estimate_pbl,
    estimate_wcl,
    estimate_with_trace,
    select_top_k,
)
from icsr8.evaluate import (
    bootstrap_ci_paired,
    errors_ledger,
    l2_errors,
    percentiles,
    summary,
    within_ratio,
)
from icsr8.fingerprint import (
    DEFAULT_REPRODUCTION_WINGS,
    band_of,
    candidate_aggregate,
    candidate_medians,
    detailed_fingerprint,
    reproduction_fingerprint,
)
from icsr8.io import (
    load_ap_coords,
    load_ap_coords_all,
    load_location_coords,
    load_raw_scans,
)
from icsr8.protocols import iter_inner_cv, iter_lolo, iter_protocol_a
from icsr8.types import Direction

# Why not `run_method` を top-level に再輸出しない:
#   呼び出し側にレジストリ (icsr8.methods) の明示を強制し、どの推定器が
#   registry 経由で走ったかを import 文から追跡可能に保つ。
__all__ = [
    "Direction",
    "DEFAULT_REPRODUCTION_WINGS",
    "methods",
    "load_ap_coords",
    "load_ap_coords_all",
    "load_location_coords",
    "load_raw_scans",
    "candidate_medians",
    "candidate_aggregate",
    "detailed_fingerprint",
    "band_of",
    "reproduction_fingerprint",
    "select_top_k",
    "estimate_pbl",
    "estimate_cla",
    "estimate_wcl",
    "estimate_with_trace",
    "l2_errors",
    "summary",
    "percentiles",
    "within_ratio",
    "errors_ledger",
    "bootstrap_ci_paired",
    "xy_to_arclength",
    "arclength_to_xy",
    "project_to_corridor",
    "segment_of",
    "geodesic_distance",
    "iter_protocol_a",
    "iter_lolo",
    "iter_inner_cv",
]
