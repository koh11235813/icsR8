"""L2 誤差と要約統計、および手法比較用の追加指標（percentiles / within_ratio /
errors_ledger / bootstrap_ci_paired）。

`summary` の `ddof` デフォルトは 0（population）。これは公表ベースライン
（doc Table 1 と estimation_result_C3F.xlsx の Std 値）に整合する選択。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from icsr8.constants import RANDOM_SEED


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


def percentiles(errors: pd.Series, qs: tuple[int, ...] = (50, 75, 90)) -> dict[str, float]:
    arr = np.asarray(errors, dtype=float)
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}


def within_ratio(errors: pd.Series, threshold: float = 2.0) -> float:
    arr = np.asarray(errors, dtype=float)
    return float(np.mean(arr <= threshold))


def errors_ledger(estimates: pd.DataFrame, truth: pd.DataFrame, method_name: str) -> pd.DataFrame:
    # Why not RangeIndex + location_p column: bootstrap_ci_paired が index の
    # 一致だけを確認して positional に減算するため、location_p を index に据えて
    # 昇順ソートしておかないと、行順が異なる 2 台帳が別 location の誤差を
    # 無警告でペアリングしてしまう。index="location_p"（昇順）でペアリング安全に。
    merged = l2_errors(estimates, truth)
    return (
        merged.assign(method=method_name)
        .set_index("location_p")
        .sort_index()[["method", "error"]]
    )


def _resolve_stat(stat: str | Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
    if stat == "mean":
        return lambda a: float(np.mean(a))
    if stat == "median":
        return lambda a: float(np.median(a))
    if callable(stat):
        return stat
    raise ValueError(f"unknown stat: {stat!r}")


def bootstrap_ci_paired(
    errors_a: pd.Series,
    errors_b: pd.Series | None = None,
    stat: str | Callable[[np.ndarray], float] = "mean",
    B: int = 1000,
    seed: int = RANDOM_SEED,
    level: float = 0.95,
) -> dict[str, float]:
    if len(errors_a) == 0:
        raise ValueError("errors_a must be non-empty")
    if B < 1:
        raise ValueError(f"B must be >= 1; got {B}")
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1); got {level}")
    fn = _resolve_stat(stat)

    if errors_b is not None:
        # Why not align by index: silently reindexing could pair errors from
        # different location_p rows without the caller noticing; fail loudly.
        if list(errors_a.index) != list(errors_b.index):
            raise ValueError(
                "errors_a and errors_b must have identical index content and order"
            )
        # Why not `errors_a - errors_b` directly: pandas aligns by index label,
        # which silently fans out into a Cartesian product on duplicate labels
        # even though the equality check above passed; numpy subtracts positionally.
        data = errors_a.to_numpy(dtype=float) - errors_b.to_numpy(dtype=float)
    else:
        data = np.asarray(errors_a, dtype=float)

    point = fn(data)
    n = len(data)
    rng = np.random.default_rng(seed)
    # Why one upfront (B, n) index matrix: guarantees the same resample set
    # is reused across all downstream calls with the same seed (determinism).
    resample_idx = rng.integers(0, n, size=(B, n))
    boot = np.array([fn(data[row]) for row in resample_idx], dtype=float)

    alpha = (1.0 - level) / 2.0
    lo = float(np.percentile(boot, alpha * 100.0))
    hi = float(np.percentile(boot, (1.0 - alpha) * 100.0))

    return {"stat": float(point), "lo": lo, "hi": hi, "B": B, "level": level}
