"""CSV ローダ群。BOM 付き utf-8 と相対パス禁止を一元化する。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from icsr8.types import Direction

_RAW_DIR_BY_DIRECTION = {
    "forward": "rawdata_C3F-F",
    "backward": "rawdata_C3F-B",
}

_AP_COORD_RENAME = {
    "AP_Name": "ap_name",
    "Floor": "floor",
    "x": "x",
    "y": "y",
    "z (AP_height)": "z",
}

_LOCATION_RENAME = {
    "Floor": "floor",
    "Building": "building",
    "Location index P": "location_p",
    "x": "x",
    "y": "y",
    "z (floor)": "z_floor",
    "z (device_height)": "z_device",
}


def load_ap_coords(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns=_AP_COORD_RENAME)
    return df[list(_AP_COORD_RENAME.values())]


def load_location_coords(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns=_LOCATION_RENAME)
    return df[list(_LOCATION_RENAME.values())]


def load_raw_scans(direction: Direction, root: str | Path) -> pd.DataFrame:
    if direction not in _RAW_DIR_BY_DIRECTION:
        raise ValueError(
            f"unknown direction: {direction!r}; expected 'forward' or 'backward'"
        )
    folder = Path(root) / _RAW_DIR_BY_DIRECTION[direction]
    files = sorted(folder.glob("*_C3F_*.csv"))
    if not files:
        raise FileNotFoundError(f"no scan CSVs found under {folder}")

    frames = [pd.read_csv(f, encoding="utf-8-sig") for f in files]
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(
        columns={
            "SSID": "ssid",
            "RSSI": "rssi",
            "Frequency": "frequency",
            "Count": "count",
            "Position": "location_p",
            "AP_Name": "ap_name",
        }
    )
    raw["direction"] = direction

    if not raw["count"].between(0, 9).all():
        bad = raw.loc[~raw["count"].between(0, 9), "count"].unique().tolist()
        raise ValueError(f"Count column out of range 0..9: {bad}")

    return raw[
        ["location_p", "ssid", "rssi", "frequency", "count", "ap_name", "direction"]
    ]
