"""評価プロトコル用の分割 iterator 群。

Protocol A（方向間 train→test）, LOLO（Leave-One-Location-Out）,
inner CV（location 単位の k-fold）を scan DataFrame 上の純関数として提供する。
ファイル IO は行わない。scans の読み込みは呼び出し側が icsr8.io で行うこと。

リーク契約:
    集計統計 (μ, σ, 検出率, GP fit) は fit に渡された train_scans のみから
    計算すること。これらの iterator は held-out location の行が train_scans に
    混入しないことを保証する。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
import pandas as pd

from icsr8.constants import RANDOM_SEED


class Fold(NamedTuple):
    name: str
    train_scans: pd.DataFrame
    test_scans: pd.DataFrame


class LoloFold(NamedTuple):
    held_out: int
    train_scans: pd.DataFrame
    test_scans: pd.DataFrame


def iter_protocol_a(
    scans_forward: pd.DataFrame, scans_backward: pd.DataFrame
) -> list[Fold]:
    return [
        Fold("forward_to_backward", scans_forward, scans_backward),
        Fold("backward_to_forward", scans_backward, scans_forward),
    ]


def iter_lolo(
    train_pool: pd.DataFrame, test_pool: pd.DataFrame | None = None
) -> Iterator[LoloFold]:
    train_locs = sorted(set(train_pool["location_p"]))
    if len(train_locs) < 2:
        raise ValueError(
            "iter_lolo needs >= 2 locations in train_pool "
            f"(training set would be empty); got {len(train_locs)}"
        )
    # Why not lazy check inside the generator: test_pool 不一致は設定ミスなので
    # 反復開始を待たず即時に失敗させ、呼び出し側の debug を容易にする。
    if test_pool is not None:
        if set(train_pool["location_p"]) != set(test_pool["location_p"]):
            raise ValueError(
                "train_pool and test_pool must cover identical location sets"
            )

    source = train_pool if test_pool is None else test_pool

    def _gen() -> Iterator[LoloFold]:
        for loc in train_locs:
            train = train_pool[train_pool["location_p"] != loc]
            test = source[source["location_p"] == loc]
            yield LoloFold(int(loc), train, test)

    return _gen()


def iter_inner_cv(
    scans: pd.DataFrame, k: int = 5, seed: int = RANDOM_SEED
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    # Why not lazy check: 本関数を素の generator にすると引数不正が最初の反復まで
    # 顕在化しない。検証を外側で先に済ませ、生成のみを内部 generator に委譲する。
    if scans.empty:
        raise ValueError("scans must be non-empty")
    locations = np.array(sorted(set(scans["location_p"])))
    n_locations = len(locations)
    if not (2 <= k <= n_locations):
        raise ValueError(
            f"k must satisfy 2 <= k <= n_locations={n_locations}; got k={k}"
        )
    permuted = np.random.default_rng(seed).permutation(locations)

    def _gen() -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        for val_locs in np.array_split(permuted, k):
            val_set = set(val_locs.tolist())
            val = scans[scans["location_p"].isin(val_set)]
            train = scans[~scans["location_p"].isin(val_set)]
            yield train, val

    return _gen()
