"""PBL / CLA / WCL 推定器と共通の top-k 選択器。

WCL の重み（doc/icsR8_text.txt §3.1）:
    w_j = 10 ** ((rssi_j - rssi_min_of_top3) / 10)

Tie-break:
    仕様書は「random」だが、再現性確保 + 公表値再現のため決定的にする。
    `tie_break="frequency"` (デフォルト) は以下の優先順:
        1. rssi_median 降順
        2. frequency 昇順    (低周波が強信号として優先される傾向に合致)
        3. ssid 昇順         (tutwifi < tutwifi2025)
        4. ap_name 昇順
    estimation_result_C3F.xlsx で観測された 5 件 (P19/P30/P35/P43/P49) の
    tie が全てこの規則で公表値と一致することを実測で確認している。
    乱択モードは ``tie_break="random"`` で opt-in。
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

TieBreak = Literal["frequency", "random"]


def select_top_k(
    fingerprint: pd.DataFrame,
    k: int,
    *,
    tie_break: TieBreak = "frequency",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    if tie_break == "frequency":
        ordered = fingerprint.sort_values(
            ["rssi_median", "frequency", "ssid", "ap_name"],
            ascending=[False, True, True, True],
            kind="stable",
        )
    elif tie_break == "random":
        if rng is None:
            rng = np.random.default_rng()
        ordered = fingerprint.assign(
            _jitter=rng.random(len(fingerprint))
        ).sort_values(
            ["rssi_median", "_jitter"],
            ascending=[False, True],
            kind="stable",
        ).drop(columns="_jitter")
    else:
        raise ValueError(f"unknown tie_break: {tie_break!r}")

    return ordered.head(k).reset_index(drop=True)


def _apply_per_location(fp: pd.DataFrame, fn, **kw) -> pd.DataFrame:
    rows = []
    for loc, group in fp.groupby("location_p", sort=True):
        x, y = fn(group, **kw)
        rows.append({"location_p": int(loc), "x": x, "y": y})
    return pd.DataFrame(rows)


def _pbl_one(fp: pd.DataFrame, **kw) -> tuple[float, float]:
    top = select_top_k(fp, 1, **kw)
    return float(top["x"].iloc[0]), float(top["y"].iloc[0])


def _require_three(fp: pd.DataFrame, method: str) -> None:
    if len(fp) < 3:
        loc = fp["location_p"].iloc[0] if len(fp) else "?"
        raise ValueError(
            f"{method.upper()} requires 3 candidates per location; "
            f"location_p={loc} has only {len(fp)}"
        )


def _cla_one(fp: pd.DataFrame, **kw) -> tuple[float, float]:
    _require_three(fp, "cla")
    top = select_top_k(fp, 3, **kw)
    return float(top["x"].mean()), float(top["y"].mean())


def _wcl_one(fp: pd.DataFrame, **kw) -> tuple[float, float]:
    _require_three(fp, "wcl")
    top = select_top_k(fp, 3, **kw)
    rssi_min = top["rssi_median"].min()
    weights = np.power(10.0, (top["rssi_median"].to_numpy() - rssi_min) / 10.0)
    wsum = weights.sum()
    x = float((weights * top["x"].to_numpy()).sum() / wsum)
    y = float((weights * top["y"].to_numpy()).sum() / wsum)
    return x, y


def estimate_pbl(fp: pd.DataFrame, **kw) -> pd.DataFrame:
    return _apply_per_location(fp, _pbl_one, **kw)


def estimate_cla(fp: pd.DataFrame, **kw) -> pd.DataFrame:
    return _apply_per_location(fp, _cla_one, **kw)


def estimate_wcl(fp: pd.DataFrame, **kw) -> pd.DataFrame:
    return _apply_per_location(fp, _wcl_one, **kw)


_METHODS = {"pbl": _pbl_one, "cla": _cla_one, "wcl": _wcl_one}


def estimate_with_trace(
    fp: pd.DataFrame,
    method: Literal["pbl", "cla", "wcl"],
    **kw,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (estimates_df, trace_df) — trace_df shows the top-k rows that
    fed each per-location estimate, with computed WCL weights when applicable.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method: {method!r}")
    k = 1 if method == "pbl" else 3

    est_rows = []
    trace_rows = []
    for loc, group in fp.groupby("location_p", sort=True):
        top = select_top_k(group, k, **kw)
        if method == "wcl":
            rssi_min = top["rssi_median"].min()
            w = np.power(10.0, (top["rssi_median"].to_numpy() - rssi_min) / 10.0)
            x = float((w * top["x"].to_numpy()).sum() / w.sum())
            y = float((w * top["y"].to_numpy()).sum() / w.sum())
            top = top.assign(weight=w)
        elif method == "cla":
            x = float(top["x"].mean())
            y = float(top["y"].mean())
            top = top.assign(weight=1.0)
        else:
            x = float(top["x"].iloc[0])
            y = float(top["y"].iloc[0])
            top = top.assign(weight=1.0)
        est_rows.append({"location_p": int(loc), "x": x, "y": y})
        trace_rows.append(top.assign(location_p=int(loc)))

    return pd.DataFrame(est_rows), pd.concat(trace_rows, ignore_index=True)
