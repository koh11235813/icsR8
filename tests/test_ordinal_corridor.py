"""ordinal_corridor（手法17: 累積確率の順序回帰 → arc-length）のテスト。

Tests:
  1. isotonic 射影: 手書き PAV 参照実装と一致し、出力が非増加。
  2. 完全分離可能な合成データ（弧長に単調な単一 AP 信号）で held-out 地点の
     予測順序が真の弧長順序を復元する。
  3. 退化閾値: 全訓練地点が同一弧長点に載ると閾値超過クラスが単一になり、
     fallback の定数確率（float, LogisticRegression でない）を使う。
  4. inner CV リーク spy: _fit_ordinal_model の閾値/標準化統計が、渡された
     train 部分集合のみから計算され、held-out 地点を含む全体分布とは一致しない。
  5. 廊下 assert: 予測点の廊下への射影距離 <= 1e-6、s ∈ [0, 116]。
  6. Protocol A 1 fold smoke（契約必須）+ diagnostics_ の必須キー。
  7. LOLO smoke（3 地点、iter_lolo を islice）で NaN/shape 崩れ無し。
"""

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.constants import RANDOM_SEED
from icsr8.corridor import _TOTAL_LENGTH, project_to_corridor, xy_to_arclength
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import run_method
from icsr8.methods.ordinal_corridor import (
    OrdinalCorridor,
    _fit_ordinal_model,
    _isotonic_nonincreasing,
)
from icsr8.protocols import iter_inner_cv, iter_lolo, iter_protocol_a


# --- fixtures --------------------------------------------------------------

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


def _make_scan_rows(location_p: int, ap_rssi: dict[str, float]) -> pd.DataFrame:
    aps = list(ap_rssi)
    return pd.DataFrame({
        "location_p": [location_p] * len(aps),
        "ssid": ["test"] * len(aps),
        "rssi": [ap_rssi[a] for a in aps],
        "frequency": [2400] * len(aps),
        "count": [0] * len(aps),
        "ap_name": aps,
    })


# --- 1: isotonic projection vs hand-written PAV reference -------------------

def _pav_nonincreasing(y: np.ndarray) -> np.ndarray:
    """L2-optimal 非増加 step function（教科書 PAV, in-test 独立実装）。"""
    blocks = [[float(v), 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        avg_i = blocks[i][0] / blocks[i][1]
        avg_next = blocks[i + 1][0] / blocks[i + 1][1]
        if avg_i < avg_next:
            merged = [blocks[i][0] + blocks[i + 1][0], blocks[i][1] + blocks[i + 1][1]]
            blocks[i:i + 2] = [merged]
            i = max(i - 1, 0)
        else:
            i += 1
    out: list[float] = []
    for total, count in blocks:
        out.extend([total / count] * count)
    return np.array(out)


def test_isotonic_projection_matches_pav_reference_and_nonincreasing():
    raw = np.array([0.2, 0.9, 0.5, 0.1, 0.6])
    expected = _pav_nonincreasing(raw)
    got = _isotonic_nonincreasing(raw)

    assert got == pytest.approx(expected, abs=1e-9)
    assert np.all(np.diff(got) <= 1e-9)


def test_isotonic_projection_already_monotone_is_identity():
    raw = np.array([0.9, 0.7, 0.5, 0.2])
    got = _isotonic_nonincreasing(raw)
    assert got == pytest.approx(raw, abs=1e-9)


# --- 2: perfectly separable synthetic data recovers location ordering ------

def test_recovers_ordering_on_separable_synthetic_data():
    # 16 locations along corridor segment C (y=0): s = 32 - x, single AP whose
    # RSSI decreases monotonically and strongly with x -> thresholds should be
    # trivially separable by a linear logistic decision boundary.
    xs = np.linspace(0.0, 30.0, 16)
    locs = list(range(1, 17))
    frames = [_make_scan_rows(loc, {"AP-A": -30.0 - 3.0 * x}) for loc, x in zip(locs, xs)]
    scans = pd.concat(frames, ignore_index=True)
    coords = pd.DataFrame({"location_p": locs, "x": xs, "y": [0.0] * 16})

    train_idx = list(range(0, 16, 2))  # even indices -> train
    test_idx = list(range(1, 16, 2))   # odd indices -> held out
    train_locs = [locs[i] for i in train_idx]
    test_locs = [locs[i] for i in test_idx]

    train_scans = scans[scans["location_p"].isin(train_locs)]
    train_coords = coords[coords["location_p"].isin(train_locs)]
    test_scans = scans[scans["location_p"].isin(test_locs)]

    method = OrdinalCorridor(m=4, C=10.0).fit(train_scans, None, train_coords)
    est = method.predict(test_scans).set_index("location_p")

    true_s = {loc: xy_to_arclength(float(x), 0.0) for loc, x in zip(locs, xs)}
    pred_s = {loc: xy_to_arclength(float(est.loc[loc, "x"]), float(est.loc[loc, "y"])) for loc in test_locs}

    true_order = sorted(test_locs, key=lambda loc: true_s[loc])
    pred_order = sorted(test_locs, key=lambda loc: pred_s[loc])
    assert pred_order == true_order


# --- 3: degenerate threshold fallback ---------------------------------------

def test_degenerate_threshold_falls_back_to_constant_probability():
    # 5 locations all mapped to the same corridor point (32, 0) -> identical
    # arc length s=0 for every training location, so any interior quantile
    # threshold has zero locations strictly above it (single-class label).
    locs = [1, 2, 3, 4, 5]
    frames = [_make_scan_rows(loc, {"AP-A": -40.0 - loc}) for loc in locs]
    scans = pd.concat(frames, ignore_index=True)
    coords = pd.DataFrame({"location_p": locs, "x": [32.0] * 5, "y": [0.0] * 5})

    model = _fit_ordinal_model(scans, coords, m=1, C=1.0)

    assert len(model.classifiers) == 1
    assert isinstance(model.classifiers[0], float)
    assert model.classifiers[0] == pytest.approx(0.0)

    method = OrdinalCorridor(m=1, C=1.0).fit(scans, None, coords)
    query = _make_scan_rows(99, {"AP-A": -50.0})
    est = method.predict(query)

    assert len(est) == 1
    assert np.isfinite(est[["x", "y"]].to_numpy()).all()
    # s_min == s_max == 0 here, so the predicted point must sit at s=0.
    assert xy_to_arclength(float(est["x"].iloc[0]), float(est["y"].iloc[0])) == pytest.approx(0.0)


# --- 4: inner-CV leak spy ----------------------------------------------------

def test_fit_ordinal_model_thresholds_use_only_given_subset():
    # 6 locations along segment C: s = 32 - x. Fit on the first 4 only; the
    # resulting thresholds must match quantiles of those 4 s-values alone,
    # not the full 6-location distribution (which would signal a leak of the
    # held-out locations into preprocessing statistics).
    xs = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0]
    locs = list(range(1, 7))
    frames = [_make_scan_rows(loc, {"AP-A": -30.0 - x}) for loc, x in zip(locs, xs)]
    scans = pd.concat(frames, ignore_index=True)
    coords = pd.DataFrame({"location_p": locs, "x": xs, "y": [0.0] * 6})

    train_locs = locs[:4]
    train_scans = scans[scans["location_p"].isin(train_locs)]
    train_coords = coords[coords["location_p"].isin(train_locs)]

    model = _fit_ordinal_model(train_scans, train_coords, m=2, C=1.0)

    s_train_only = np.array(sorted(32.0 - x for x in xs[:4]))
    expected = np.quantile(s_train_only, [1.0 / 3.0, 2.0 / 3.0])
    assert model.thresholds == pytest.approx(expected)

    s_all = np.array(sorted(32.0 - x for x in xs))
    leaked = np.quantile(s_all, [1.0 / 3.0, 2.0 / 3.0])
    assert not np.allclose(model.thresholds, leaked)


# --- 4c: inner-CV model fits see exactly the seeded inner_train sets (F7) ----

def test_inner_cv_model_fits_match_seeded_folds(monkeypatch):
    # The spy records what _fit_ordinal_model (thresholds + standardization
    # stats) sees on the REAL select_by_inner_cv path and compares it 1:1 with
    # iter_inner_cv(seed=RANDOM_SEED): per fold, one call per (m, C) candidate
    # with the exact inner_train set, then the full train set for the refit.
    import icsr8.methods.ordinal_corridor as oc_mod

    xs = np.linspace(0.0, 30.0, 10)
    locs = list(range(1, 11))
    frames = [_make_scan_rows(loc, {"AP-A": -30.0 - 3.0 * x}) for loc, x in zip(locs, xs)]
    train_scans = pd.concat(frames, ignore_index=True)
    coords = pd.DataFrame({"location_p": locs, "x": xs, "y": [0.0] * 10})

    calls: list[frozenset[int]] = []
    real = oc_mod._fit_ordinal_model

    def spy(scans, location_coords, m, C):
        calls.append(frozenset(int(x) for x in scans["location_p"].unique()))
        return real(scans, location_coords, m, C)

    monkeypatch.setattr(oc_mod, "_fit_ordinal_model", spy)

    OrdinalCorridor().fit(train_scans, None, coords)

    folds = list(iter_inner_cv(train_scans, k=5, seed=RANDOM_SEED))
    n_cand = len(oc_mod._M_GRID) * len(oc_mod._C_GRID)
    full = frozenset(locs)
    expected: list[frozenset[int]] = []
    for tr, _ in folds:
        expected += [frozenset(int(x) for x in tr["location_p"].unique())] * n_cand
    expected.append(full)
    assert calls == expected
    for (_, val), got in zip(folds, calls[:: n_cand]):
        val_locs = frozenset(int(x) for x in val["location_p"].unique())
        assert got == full - val_locs  # exact complement, not merely disjoint


# --- 4b: constructor all-or-none param contract (F11) ------------------------

def test_constructor_requires_m_and_C_together():
    OrdinalCorridor()  # both omitted -> CV path, ok
    OrdinalCorridor(m=8, C=1.0)  # both given -> pinned, ok
    with pytest.raises(ValueError):
        OrdinalCorridor(m=8)
    with pytest.raises(ValueError):
        OrdinalCorridor(C=1.0)


# --- 5: corridor assert ------------------------------------------------------

def test_predictions_are_on_corridor(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]
    est = run_method(
        "ordinal_corridor",
        fold.train_scans, fold.test_scans,
        ap_coords, location_coords,
    )

    for row in est.itertuples():
        px, py = project_to_corridor(row.x, row.y)
        dist = math.hypot(row.x - px, row.y - py)
        assert dist <= 1e-6
        s = xy_to_arclength(row.x, row.y)
        assert 0.0 <= s <= _TOTAL_LENGTH


# --- 6: Protocol A smoke + diagnostics_ --------------------------------------

def test_protocol_a_smoke_and_diagnostics(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]

    method = OrdinalCorridor().fit(fold.train_scans, ap_coords, location_coords)
    est = method.predict(fold.test_scans)

    assert len(est) == 59
    assert set(est.columns) == {"location_p", "x", "y"}
    assert not est.isna().any().any()
    assert np.isfinite(est[["x", "y"]]).all().all()

    diag = method.diagnostics_
    assert diag["selected_m"] in {8, 12}
    assert diag["selected_C"] in {0.1, 1.0, 10.0}
    assert "cv_scores" in diag
    assert "n_degenerate_thresholds" in diag


# --- 7: LOLO smoke ------------------------------------------------------------

def test_lolo_smoke(scans_f, ap_coords, location_coords):
    # m/C fixed (skip inner CV) to keep 3 folds fast; the CV path itself is
    # already exercised by test_protocol_a_smoke_and_diagnostics.
    for fold in itertools.islice(iter_lolo(scans_f), 3):
        train_coords = location_coords[
            location_coords["location_p"].isin(fold.train_scans["location_p"].unique())
        ]
        method = OrdinalCorridor(m=8, C=1.0).fit(fold.train_scans, ap_coords, train_coords)
        est = method.predict(fold.test_scans)

        assert len(est) == 1
        assert set(est.columns) == {"location_p", "x", "y"}
        assert not est.isna().any().any()
        assert np.isfinite(est[["x", "y"]]).all().all()
        assert est["location_p"].iloc[0] == fold.held_out
