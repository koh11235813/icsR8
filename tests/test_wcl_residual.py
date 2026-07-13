"""wcl_residual（WCL 残差学習, Tier 4 手法18）のテスト。

Tests:
  1. 残差 0（ridge を monkeypatch で全ゼロ出力に固定）のとき、出力が
     apply_corridor_projection(estimate_wcl(...)) = wcl_corridor と一致し、
     かつ素の WCL（廊下射影なし）とは一致しないことを両方向で確認する。
  2. inner CV: 標準化統計 (_train_feature_stats) が fold ごとに再構築され、
     各 CV 呼び出しが inner_val の地点を除いた真部分集合のみを見て、最後の
     1 回（最終 refit）だけ train 全地点を見ることを spy で固定する。
  3. 廊下 assert: 実データ 1 fold の予測が全て廊下上（射影距離 <=1e-6）かつ
     弧長 s ∈ [0, 116]。
  4. run_method 経由の外側リーク: LOLO fold で held_out 地点が fit の
     location_coords に届かないことを spy で確認する。
  5. Protocol A 1 fold smoke。
  6. LOLO smoke（3 地点, islice）。
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.corridor import _TOTAL_LENGTH, assert_locations_on_corridor, xy_to_arclength
from icsr8.estimators import estimate_wcl
from icsr8.fingerprint import candidate_medians, reproduction_fingerprint
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import run_method
from icsr8.methods.corridor_proj import apply_corridor_projection
from icsr8.constants import RANDOM_SEED
from icsr8.methods.wcl_residual import ALPHA_GRID, WclResidual
from icsr8.protocols import iter_inner_cv, iter_lolo, iter_protocol_a


# --- fixtures ------------------------------------------------------------

@pytest.fixture(scope="module")
def ap_coords(dataset_dir: Path) -> pd.DataFrame:
    return load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")


@pytest.fixture(scope="module")
def location_coords(dataset_dir: Path) -> pd.DataFrame:
    df = load_location_coords(dataset_dir / "location_coordinate_C.csv")
    return df[["location_p", "x", "y"]]


@pytest.fixture(scope="module")
def scans_f(rawdata_root: Path) -> pd.DataFrame:
    return load_raw_scans("forward", rawdata_root)


@pytest.fixture(scope="module")
def scans_b(rawdata_root: Path) -> pd.DataFrame:
    return load_raw_scans("backward", rawdata_root)


@pytest.fixture(scope="module")
def protocol_a_fold(scans_f, scans_b):
    return iter_protocol_a(scans_f, scans_b)[0]


# --- 1: zero-residual == wcl_corridor, != raw wcl -------------------------

def test_zero_residual_equals_wcl_corridor_not_raw_wcl(
    protocol_a_fold, ap_coords, location_coords, monkeypatch
):
    fold = protocol_a_fold
    method = WclResidual(alpha=1.0).fit(fold.train_scans, ap_coords, location_coords)
    monkeypatch.setattr(
        method._model, "predict", lambda X: np.zeros(X.shape[0])
    )
    est = method.predict(fold.test_scans).sort_values("location_p").reset_index(drop=True)

    raw_fp = reproduction_fingerprint(candidate_medians(fold.test_scans, ap_coords))
    raw_wcl = estimate_wcl(raw_fp).sort_values("location_p").reset_index(drop=True)
    expected_corridor = apply_corridor_projection(raw_wcl)

    pd.testing.assert_frame_equal(
        est[["location_p", "x", "y"]], expected_corridor[["location_p", "x", "y"]],
        check_exact=False, atol=1e-9,
    )

    # Guard against a bug that skips the corridor projection entirely: the
    # zero-residual output must actually differ from raw (unprojected) WCL
    # at some location, otherwise this test would pass even if projection
    # were silently dropped.
    diff = np.maximum(
        (est["x"] - raw_wcl["x"]).abs(), (est["y"] - raw_wcl["y"]).abs()
    )
    assert diff.max() > 1e-6


# --- 2: fold-internal standardization + inner-CV leak spy -----------------

def test_inner_cv_standardization_is_fold_internal(scans_f, ap_coords, location_coords, monkeypatch):
    import icsr8.methods.wcl_residual as wcl_residual_mod

    calls: list[frozenset[int]] = []
    orig = wcl_residual_mod._train_feature_stats

    def spy(scans):
        calls.append(frozenset(scans["location_p"].unique()))
        return orig(scans)

    monkeypatch.setattr(wcl_residual_mod, "_train_feature_stats", spy)

    WclResidual().fit(scans_f, ap_coords, location_coords)

    full_locs = frozenset(scans_f["location_p"].unique())
    k = 5
    assert len(calls) == k * len(ALPHA_GRID) + 1

    # F7: each inner-CV call's location set must equal the EXACT complement of
    # its fold's validation set under iter_inner_cv(seed=RANDOM_SEED) — not
    # merely be a strict subset (which even a wrongly-seeded split satisfies).
    folds = list(iter_inner_cv(scans_f, k=k, seed=RANDOM_SEED))
    idx = 0
    for _, inner_val in folds:
        val_locs = frozenset(inner_val["location_p"].unique())
        expected = full_locs - val_locs
        for _ in range(len(ALPHA_GRID)):
            assert calls[idx] == expected
            idx += 1
    # The final refit call (after CV selects alpha) uses the full train set.
    assert calls[idx] == full_locs


# --- 3: corridor assertion (projection distance + arclength bounds) -------

def test_predictions_on_corridor_and_arclength_in_range(
    protocol_a_fold, ap_coords, location_coords
):
    fold = protocol_a_fold
    est = run_method(
        "wcl_residual", fold.train_scans, fold.test_scans, ap_coords, location_coords
    )

    assert_locations_on_corridor(est, tol=1e-6)

    s = np.array([xy_to_arclength(x, y) for x, y in zip(est["x"], est["y"])])
    assert np.all(s >= 0.0 - 1e-9)
    assert np.all(s <= _TOTAL_LENGTH + 1e-9)


# --- 4: run_method leak spy (outer test location never reaches fit) -------

def test_run_method_held_out_location_not_seen_by_fit(scans_f, scans_b, ap_coords, location_coords, monkeypatch):
    from icsr8.methods.wcl_residual import WclResidual as _WclResidual

    fold = next(islice(iter_lolo(scans_f, scans_b), 1))

    seen: dict[str, set[int]] = {}
    orig_fit = _WclResidual.fit

    def spy_fit(self, train_scans, ap_coords, location_coords):
        seen["locs"] = set(location_coords["location_p"])
        return orig_fit(self, train_scans, ap_coords, location_coords)

    monkeypatch.setattr(_WclResidual, "fit", spy_fit)

    run_method(
        "wcl_residual", fold.train_scans, fold.test_scans, ap_coords, location_coords
    )

    assert fold.held_out not in seen["locs"]


# --- 5: Protocol A smoke ---------------------------------------------------

def test_wcl_residual_smoke(protocol_a_fold, ap_coords, location_coords):
    fold = protocol_a_fold
    est = run_method(
        "wcl_residual", fold.train_scans, fold.test_scans, ap_coords, location_coords
    )

    assert len(est) == 59
    assert set(est.columns) == {"location_p", "x", "y"}
    assert not est.isna().any().any()
    assert np.isfinite(est[["x", "y"]]).all().all()


# --- 6: LOLO smoke (3 locations, islice) -----------------------------------

def test_wcl_residual_lolo_smoke(scans_f, scans_b, ap_coords, location_coords):
    for fold in islice(iter_lolo(scans_f, scans_b), 3):
        est = run_method(
            "wcl_residual", fold.train_scans, fold.test_scans, ap_coords, location_coords
        )
        assert list(est.columns) == ["location_p", "x", "y"]
        assert len(est) == 1
        assert est["location_p"].iloc[0] == fold.held_out
        assert np.isfinite(est[["x", "y"]]).all().all()
