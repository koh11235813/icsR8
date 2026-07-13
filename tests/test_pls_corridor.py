"""pls_corridor（手法15 PLS 回帰による廊下弧長直接回帰）のテスト。

Tests:
  1. 無分散列除去: train 側で全地点同一値の AP 鍵が特徴から落ちる。
  2. inner CV leak spy: _fit_predict_candidate の予測が inner_val の他 location
     の値に依存しない（train 側統計のみで前処理・fit していることの直接証拠）。
  3. outer leak: fit() の diagnostics_ が渡された train location 数のみを反映する
     （run_method 経由で train 部分集合を渡す構図）。
  4. 線形合成 radio map から弧長 s をほぼ厳密に復元する。
  5. clip: PLS 出力を差し替えたスタブで [0,116] 範囲外予測を注入し、クリップが
     効くことを直接確認する。
  6. 廊下射影距離 <=1e-6 かつ s in [0,116]（実データ 1 fold）。
  7. Protocol A 1 fold + LOLO 3 地点 smoke で NaN/shape 崩れ無し。
  8. 契約必須の generic smoke test。
"""

from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.corridor import _TOTAL_LENGTH, project_to_corridor, xy_to_arclength, arclength_to_xy
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import run_method
from icsr8.methods.pls_corridor import (
    _FittedPls,
    _fit_pls,
    _fit_predict_candidate,
    _fit_train_matrix,
    _predict_s,
    PlsCorridor,
)
from icsr8.constants import RANDOM_SEED
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


def _scan_rows(location_p, ap_name, rssi_list, freq=2400, start=0):
    return [
        {
            "location_p": location_p,
            "ssid": "s",
            "rssi": float(v),
            "frequency": freq,
            "count": start + c,
            "ap_name": ap_name,
        }
        for c, v in enumerate(rssi_list)
    ]


def _make_scans(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _coords(locs: list[int], xy: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"location_p": locs, "x": [p[0] for p in xy], "y": [p[1] for p in xy]}
    )


# --- 1: zero-variance column dropped --------------------------------------

def test_zero_variance_column_dropped():
    # AP-CONST is -55 dBm at every location -> zero variance, must be dropped.
    # AP-VAR varies across locations -> kept.
    rows = []
    rows += _scan_rows(1, "AP-CONST", [-55.0] * 10)
    rows += _scan_rows(1, "AP-VAR", [-40.0] * 10)
    rows += _scan_rows(2, "AP-CONST", [-55.0] * 10)
    rows += _scan_rows(2, "AP-VAR", [-60.0] * 10)
    rows += _scan_rows(3, "AP-CONST", [-55.0] * 10)
    rows += _scan_rows(3, "AP-VAR", [-80.0] * 10)
    scans = _make_scans(rows)

    keys, mean, std, x_std, locs = _fit_train_matrix(scans)

    assert ("AP-CONST", "2.4G") not in keys
    assert ("AP-VAR", "2.4G") in keys
    assert locs == [1, 2, 3]
    assert x_std.shape == (3, 1)
    # standardized column has zero mean, unit population std by construction
    assert x_std.mean() == pytest.approx(0.0, abs=1e-9)
    assert x_std.std() == pytest.approx(1.0, abs=1e-9)


# --- 1b: all-zero-variance / degenerate folds fall back deterministically (F5)

def test_all_zero_variance_columns_intercept_only_fallback():
    # Every key constant across locations -> all columns dropped. The fit must
    # not crash: it degrades to an intercept-only model predicting the mean
    # train arc-length.
    rows = []
    for loc in (1, 2, 3):
        rows += _scan_rows(loc, "AP-CONST", [-55.0] * 10)
        rows += _scan_rows(loc, "AP-CONST2", [-65.0] * 10)
    scans = _make_scans(rows)
    train_s = [0.0, 10.0, 20.0]
    coords = _coords([1, 2, 3], [arclength_to_xy(s) for s in train_s])

    fitted = _fit_pls(scans, coords, n_components=3)
    assert fitted.keys == []

    query = _make_scans(_scan_rows(9, "AP-CONST", [-55.0] * 10))
    locs, s = _predict_s(fitted, query)
    assert locs == [9]
    assert s[0] == pytest.approx(np.mean(train_s))

    # Determinism: two fits give identical predictions.
    fitted2 = _fit_pls(scans, coords, n_components=3)
    _, s2 = _predict_s(fitted2, query)
    assert s2[0] == s[0]


def test_single_train_location_intercept_only_fallback():
    # <2 train locations: PLS cannot be posed (0 components); fall back to the
    # single location's arc-length.
    rows = _scan_rows(1, "AP-A", [-50.0] * 10)
    scans = _make_scans(rows)
    coords = _coords([1], [arclength_to_xy(15.0)])

    fitted = _fit_pls(scans, coords, n_components=2)
    query = _make_scans(_scan_rows(9, "AP-A", [-48.0] * 10))
    locs, s = _predict_s(fitted, query)
    assert locs == [9]
    assert s[0] == pytest.approx(15.0)


# --- 2: inner CV leak spy --------------------------------------------------

def test_inner_cv_predictions_independent_of_other_val_locations():
    # inner_train: 6 locations spread along the corridor, 2 informative AP keys
    # with a clean linear relation to s (so PLS fit is well-posed).
    train_locs = [1, 2, 3, 4, 5, 6]
    train_s = [0.0, 8.0, 20.0, 40.0, 70.0, 100.0]
    train_xy = [arclength_to_xy(s) for s in train_s]
    rows = []
    for loc, s in zip(train_locs, train_s):
        rows += _scan_rows(loc, "AP-A", [-40.0 - 0.3 * s] * 10)
        rows += _scan_rows(loc, "AP-B", [-90.0 + 0.4 * s] * 10)
    inner_train = _make_scans(rows)
    inner_train_coords = _coords(train_locs, train_xy)

    # Two variants of inner_val: location 99 is IDENTICAL in both; location 98
    # differs wildly (canary). If preprocessing/fit ever depended on inner_val
    # content, location 99's prediction would differ between the two calls.
    def _val(loc98_ap_a, loc98_ap_b):
        rows = []
        rows += _scan_rows(99, "AP-A", [-40.0 - 0.3 * 30.0] * 10)
        rows += _scan_rows(99, "AP-B", [-90.0 + 0.4 * 30.0] * 10)
        rows += _scan_rows(98, "AP-A", [loc98_ap_a] * 10)
        rows += _scan_rows(98, "AP-B", [loc98_ap_b] * 10)
        return _make_scans(rows)

    val_1 = _val(-40.0, -90.0)
    val_2 = _val(-999.0, 999.0)  # extreme canary values

    pred_1 = _fit_predict_candidate(inner_train, val_1, inner_train_coords, 2)
    pred_2 = _fit_predict_candidate(inner_train, val_2, inner_train_coords, 2)

    row_1 = pred_1.set_index("location_p").loc[99]
    row_2 = pred_2.set_index("location_p").loc[99]
    assert row_1["x"] == pytest.approx(row_2["x"], abs=1e-9)
    assert row_1["y"] == pytest.approx(row_2["y"], abs=1e-9)


# --- 2b: inner-CV train stats see exactly the seeded inner_train sets (F7) --

def test_inner_cv_train_stats_match_seeded_folds(monkeypatch):
    # 10 locations, 2 informative APs linear in s -> PLS fit well-posed in
    # every inner fold. The spy records what the train-side preprocessing
    # (_fit_train_matrix: standardization stats) sees on the REAL
    # select_by_inner_cv path and compares it 1:1 with iter_inner_cv(seed=
    # RANDOM_SEED): per fold the exact inner_train set (= complement of that
    # fold's validation set), then the full train set for the final refit.
    import icsr8.methods.pls_corridor as pls_mod

    locs = list(range(1, 11))
    s_vals = [2.0 + 10.0 * i for i in range(10)]
    rows = []
    for loc, s in zip(locs, s_vals):
        rows += _scan_rows(loc, "AP-A", [-40.0 - 0.3 * s] * 10)
        rows += _scan_rows(loc, "AP-B", [-90.0 + 0.4 * s] * 10)
    train_scans = _make_scans(rows)
    coords = _coords(locs, [arclength_to_xy(s) for s in s_vals])

    calls: list[frozenset[int]] = []
    real = pls_mod._fit_train_matrix

    def spy(scans):
        calls.append(frozenset(int(x) for x in scans["location_p"].unique()))
        return real(scans)

    monkeypatch.setattr(pls_mod, "_fit_train_matrix", spy)

    PlsCorridor(component_grid=(2,)).fit(train_scans, None, coords)

    folds = list(iter_inner_cv(train_scans, k=5, seed=RANDOM_SEED))
    full = frozenset(locs)
    expected = [
        frozenset(int(x) for x in tr["location_p"].unique()) for tr, _ in folds
    ] + [full]
    assert calls == expected
    for (_, val), got in zip(folds, calls):
        val_locs = frozenset(int(x) for x in val["location_p"].unique())
        assert got == full - val_locs  # exact complement, not merely disjoint


# --- 3: outer leak (diagnostics reflect only the passed-in train subset) ---

def test_fit_diagnostics_reflect_only_train_subset(scans_f, ap_coords, location_coords):
    train_locs = sorted(scans_f["location_p"].unique())[:30]
    train_scans = scans_f[scans_f["location_p"].isin(train_locs)]
    train_coords = location_coords[location_coords["location_p"].isin(train_locs)]

    method = PlsCorridor(n_components=2).fit(train_scans, ap_coords, train_coords)

    assert method.diagnostics_["n_train_locations"] == 30
    assert set(method._fitted.locs) == set(train_locs)


# --- 3b: diagnostics carry a real inner-CV score per candidate -------------

def test_fit_diagnostics_cv_scores_populated(scans_f, ap_coords, location_coords):
    method = PlsCorridor().fit(scans_f, ap_coords, location_coords)

    assert set(method.diagnostics_["cv_scores"].keys()) == set(
        method.diagnostics_["component_grid"]
    )
    assert all(np.isfinite(v) for v in method.diagnostics_["cv_scores"].values())
    assert method.diagnostics_["selected_n_components_candidate"] in method.diagnostics_[
        "component_grid"
    ]


# --- 4: linear synthetic radio map recovers arc-length ---------------------

def test_linear_synthetic_recovers_arclength():
    rng = np.random.default_rng(0)
    n = 40
    s_all = np.sort(rng.uniform(2.0, 114.0, size=n))
    locs = list(range(1, n + 1))
    xy = [arclength_to_xy(float(s)) for s in s_all]

    rows = []
    # Several AP keys, each a distinct linear function of s (noise-free).
    slopes = [(-40.0, -0.5), (-90.0, 0.6), (-70.0, 0.2), (-30.0, -0.8)]
    for loc, s in zip(locs, s_all):
        for i, (b, m) in enumerate(slopes):
            rows += _scan_rows(loc, f"AP-{i}", [b + m * s] * 10)
    scans = _make_scans(rows)
    coords = _coords(locs, xy)

    n_train = 30
    train_locs, test_locs = locs[:n_train], locs[n_train:]
    train_scans = scans[scans["location_p"].isin(train_locs)]
    test_scans = scans[scans["location_p"].isin(test_locs)]
    train_coords = coords[coords["location_p"].isin(train_locs)]

    method = PlsCorridor(n_components=3).fit(train_scans, ap_coords=None, location_coords=train_coords)
    est = method.predict(test_scans)

    truth = coords.set_index("location_p")
    errs = []
    for row in est.itertuples():
        tx, ty = truth.loc[row.location_p, ["x", "y"]]
        errs.append(float(np.hypot(row.x - tx, row.y - ty)))
    max_err = max(errs)
    print(f"\nlinear synthetic max positional error = {max_err:.4f} m")
    assert max_err < 1.0


# --- 5: clip is effective on out-of-range predictions -----------------------

class _StubPls:
    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def predict(self, X):
        return self._values.reshape(-1, 1)


def test_clip_effective_on_out_of_range_prediction():
    keys = [("AP-A", "2.4G")]
    mean = np.array([-50.0])
    std = np.array([5.0])
    locs = [1, 2]
    fitted_low = _FittedPls(keys, mean, std, _StubPls(np.array([-500.0, -500.0])), 1, locs)
    fitted_high = _FittedPls(keys, mean, std, _StubPls(np.array([9999.0, 9999.0])), 1, locs)

    scans = _make_scans(_scan_rows(1, "AP-A", [-50.0] * 10) + _scan_rows(2, "AP-A", [-55.0] * 10))

    locs_low, s_low = _predict_s(fitted_low, scans)
    locs_high, s_high = _predict_s(fitted_high, scans)

    assert np.all(s_low >= 0.0) and np.all(s_low <= _TOTAL_LENGTH)
    assert np.all(s_high >= 0.0) and np.all(s_high <= _TOTAL_LENGTH)
    assert np.allclose(s_low, 0.0)
    assert np.allclose(s_high, _TOTAL_LENGTH)


# --- 6: corridor projection distance + arc-length bounds (real fold) -------

def test_corridor_projection_and_bounds_real_fold(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]
    est = run_method(
        "pls_corridor",
        fold.train_scans, fold.test_scans,
        ap_coords, location_coords,
    )

    for row in est.itertuples():
        px, py = project_to_corridor(row.x, row.y)
        dist = float(np.hypot(row.x - px, row.y - py))
        assert dist <= 1e-6
        s = xy_to_arclength(row.x, row.y)
        assert 0.0 <= s <= _TOTAL_LENGTH


# --- 7: Protocol A + LOLO smoke ---------------------------------------------

def test_protocol_a_and_lolo_smoke(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]
    est_a = run_method(
        "pls_corridor",
        fold.train_scans, fold.test_scans,
        ap_coords, location_coords,
    )
    assert len(est_a) == 59
    assert set(est_a.columns) == {"location_p", "x", "y"}
    assert not est_a.isna().any().any()
    assert np.isfinite(est_a[["x", "y"]]).all().all()

    for lolo in islice(iter_lolo(scans_f), 3):
        est = run_method(
            "pls_corridor",
            lolo.train_scans, lolo.test_scans,
            ap_coords, location_coords,
        )
        assert len(est) == 1
        assert set(est.columns) == {"location_p", "x", "y"}
        assert not est.isna().any().any()
        assert np.isfinite(est[["x", "y"]]).all().all()


# --- 8: smoke test (contract) ------------------------------------------------

def test_pls_corridor_smoke(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]

    est = run_method(
        "pls_corridor",
        fold.train_scans, fold.test_scans,
        ap_coords, location_coords,
    )

    assert len(est) == 59
    assert set(est.columns) == {"location_p", "x", "y"}
    assert not est.isna().any().any()
    assert np.isfinite(est[["x", "y"]]).all().all()
