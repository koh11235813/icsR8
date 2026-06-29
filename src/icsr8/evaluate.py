"""L2 誤差と要約統計。

`summary` の `ddof` デフォルトは 0（population）。これは公表ベースライン
（doc Table 1 と estimation_result_C3F.xlsx の Std 値）に整合する選択。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def l2_errors(estimates: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    est_locs = set(estimates["location_p"])
    truth_locs = set(truth["location_p"])
    only_est = est_locs - truth_locs
    only_truth = truth_locs - est_locs
    if only_est or only_truth:
        raise ValueError(
            "estimates and truth must cover the same location_p set; "
            f"only in estimates={sorted(only_est)}, only in truth={sorted(only_truth)}"
        )

    merged = estimates.merge(
        truth[["location_p", "x", "y"]].rename(columns={"x": "true_x", "y": "true_y"}),
        on="location_p",
        how="inner",
        validate="one_to_one",
    ).rename(columns={"x": "est_x", "y": "est_y"})
    merged["error"] = np.hypot(merged["est_x"] - merged["true_x"],
                               merged["est_y"] - merged["true_y"])
    return merged[["location_p", "est_x", "est_y", "true_x", "true_y", "error"]]


def summary(errors: pd.Series, *, ddof: int = 0) -> dict[str, float]:
    arr = np.asarray(errors, dtype=float)
    return {
        "Ave": float(arr.mean()),
        "Max": float(arr.max()),
        "Min": float(arr.min()),
        "Std": float(arr.std(ddof=ddof)),
        "Var": float(arr.var(ddof=ddof)),
    }
