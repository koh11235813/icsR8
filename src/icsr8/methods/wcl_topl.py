"""WCL 一般化: 上位 L 本の AP を選択する拡張（手法9）。

ベースラインの WCL は固定で上位3APを使用。このモジュールは L ∈ {3,4,5,7,"all"}
または自動選択（L=None）に対応する。

自動選択時の fit() では、訓練データに対して各候補 L で位置推定を行い、
ground truth との L2 誤差で最適な L を選択する。
パラメータが「上位K個の選択」のみで、フィッティングを持たないため、
訓練誤差評価＝選択基準であり、外側の protocol で未知データテストするため leakage がない。

Why-not (nested CV なし):
  top-L WCL は fitted parameter を持たないため、訓練データでの誤差評価が
  そのまま L 選択の criteria になる。外側の eval protocol が
  train/test を分離するので、提案法自体の内側に leakage はない。
"""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
import pandas as pd

from icsr8.estimators import select_top_k
from icsr8.fingerprint import candidate_medians, reproduction_fingerprint
from icsr8.methods import register
from icsr8.methods.base import Method


TopLValue = Literal[3, 4, 5, 7, "all"]


def _wcl_topl_one(
    fp: pd.DataFrame, k: int, **kw
) -> tuple[float, float]:
    """WCL estimate for a single location using top-k APs.

    Args:
        fp: per-location fingerprint DataFrame (columns: location_p, x, y, rssi_median, ...)
        k: number of top APs to select
        **kw: passed to select_top_k (tie_break, rng)

    Returns:
        (x, y) tuple of estimated coordinates

    Raises:
        ValueError: if len(fp) < 3 after selecting k
    """
    if len(fp) < k:
        k = len(fp)
    if k < 3:
        raise ValueError(
            f"WCL requires >= 3 APs; location_p={fp['location_p'].iloc[0]} has only {k}"
        )
    top = select_top_k(fp, k, **kw)
    rssi_min = top["rssi_median"].min()
    weights = np.power(10.0, (top["rssi_median"].to_numpy() - rssi_min) / 10.0)
    wsum = weights.sum()
    x = float((weights * top["x"].to_numpy()).sum() / wsum)
    y = float((weights * top["y"].to_numpy()).sum() / wsum)
    return x, y


def _estimate_wcl_topl(
    fp: pd.DataFrame, k: int, **kw
) -> pd.DataFrame:
    """Estimate locations using top-k AP weighting (generalized WCL).

    Args:
        fp: full fingerprint DataFrame (all locations, all APs)
        k: number of top APs to select per location
        **kw: passed to select_top_k

    Returns:
        DataFrame with [location_p, x, y] estimates
    """
    rows = []
    for loc, group in fp.groupby("location_p", sort=True):
        x, y = _wcl_topl_one(group, k, **kw)
        rows.append({"location_p": int(loc), "x": x, "y": y})
    return pd.DataFrame(rows)


def _l2_error(
    est: pd.DataFrame, truth: pd.DataFrame
) -> float:
    """Compute mean L2 error between estimates and ground truth.

    Both must have location_p sorted.
    """
    est = est.sort_values("location_p").reset_index(drop=True)
    truth = truth.sort_values("location_p").reset_index(drop=True)
    assert est["location_p"].tolist() == truth["location_p"].tolist(), \
        "location_p mismatch"
    dx = est["x"].to_numpy() - truth["x"].to_numpy()
    dy = est["y"].to_numpy() - truth["y"].to_numpy()
    l2_errors = np.sqrt(dx**2 + dy**2)
    return float(l2_errors.mean())


@register
class WCLTopL(Method):
    """WCL variant with configurable top-L AP selection.

    If L is None, fit() auto-selects best L from {3, 4, 5, 7, "all"}.
    If L is specified, predict() uses that value directly.
    """

    name: ClassVar[str] = "wcl_topl"
    uses_geometry: ClassVar[bool] = True

    def __init__(self, *, L: TopLValue | None = None) -> None:
        """Initialize with fixed or auto-select L.

        Args:
            L: Number of top APs to use. If None, fit() will auto-select.

        Raises:
            ValueError: if L is not one of {3, 4, 5, 7, "all", None}.
        """
        # Why not accept arbitrary L: fit()'s auto-select loop and the docstring
        # both hard-code the candidate set {3,4,5,7,"all"}; silently accepting
        # e.g. L=6 would run predict() without ever having validated it.
        if L not in (3, 4, 5, 7, "all", None):
            raise ValueError(f"L must be one of 3, 4, 5, 7, 'all', None; got {L!r}")
        self.L = L
        self.selected_L: TopLValue | None = None
        self._ap_coords: pd.DataFrame | None = None

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> WCLTopL:
        """Learn the method on training data.

        If L is None, select best L from {3, 4, 5, 7, "all"} by training-set error.
        Otherwise, store ap_coords for predict().

        Args:
            train_scans: training scan data
            ap_coords: AP coordinate table (13 APs, 3F only)
            location_coords: ground truth [location_p, x, y] for TRAINING locations only

        Returns:
            self
        """
        self._ap_coords = ap_coords

        if self.L is None:
            # Auto-select best L
            fp = reproduction_fingerprint(candidate_medians(train_scans, ap_coords))
            train_coords = location_coords[["location_p", "x", "y"]]

            best_L: TopLValue | None = None
            best_error = float("inf")

            for candidate_L in [3, 4, 5, 7, "all"]:
                try:
                    est = _estimate_wcl_topl(fp, candidate_L if candidate_L != "all" else len(fp))
                    error = _l2_error(est, train_coords)
                    if error < best_error:
                        best_error = error
                        best_L = candidate_L
                except ValueError:
                    # Location doesn't have enough APs
                    continue

            self.selected_L = best_L if best_L is not None else 3
        else:
            self.selected_L = self.L

        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        """Estimate locations on test data using selected/fixed L.

        Args:
            test_scans: test scan data

        Returns:
            DataFrame with [location_p, x, y]
        """
        assert self._ap_coords is not None, "fit() not called"

        fp = reproduction_fingerprint(candidate_medians(test_scans, self._ap_coords))

        # Determine k from selected_L
        L = self.selected_L if self.selected_L is not None else self.L
        if L == "all":
            # For "all", we need to use all APs at each location
            # So we pass a very large k that will be clamped to len(fp_loc)
            k = 10000
        else:
            k = int(L)

        return _estimate_wcl_topl(fp, k)
