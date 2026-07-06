"""matplotlib による誤差・推定位置の可視化。

`plot_error_by_position` は測定点ごとの誤差を折れ線で描く。`ax` を渡して
繰り返し呼べば forward/backward や複数手法を 1 枚に重ね描きできる。

`plot_estimate_map` は真の位置と推定位置（任意で AP 座標）を平面上に散布する。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


def plot_error_by_position(
    errors: pd.DataFrame,
    *,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots()

    ordered = errors.sort_values("location_p")
    ax.plot(ordered["location_p"], ordered["error"], marker="o", label=label)
    ax.set_xlabel("location_p")
    ax.set_ylabel("error")
    if label is not None:
        ax.legend()
    return ax


def plot_estimate_map(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    ap_coords: pd.DataFrame | None = None,
    ax: Axes | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(truth["x"], truth["y"], marker="o", label="true")
    ax.scatter(estimates["x"], estimates["y"], marker="x", label="estimate")
    if ap_coords is not None:
        ax.scatter(ap_coords["x"], ap_coords["y"], marker="^", label="AP")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    return ax
