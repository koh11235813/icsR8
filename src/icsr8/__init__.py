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
"""

from icsr8.estimators import (
    estimate_cla,
    estimate_pbl,
    estimate_wcl,
    estimate_with_trace,
    select_top_k,
)
from icsr8.evaluate import l2_errors, summary
from icsr8.fingerprint import (
    DEFAULT_REPRODUCTION_WINGS,
    candidate_medians,
    reproduction_fingerprint,
)
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.types import Direction

__all__ = [
    "Direction",
    "DEFAULT_REPRODUCTION_WINGS",
    "load_ap_coords",
    "load_location_coords",
    "load_raw_scans",
    "candidate_medians",
    "reproduction_fingerprint",
    "select_top_k",
    "estimate_pbl",
    "estimate_cla",
    "estimate_wcl",
    "estimate_with_trace",
    "l2_errors",
    "summary",
]
