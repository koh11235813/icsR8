"""gp_augmented_wknn（#20 GP radio map による仮想 fingerprint 拡張 WKNN）のテスト。

Tests:
  1. w_virt=0（仮想無効化フラグ）で素の WKNN（inv_sq, 固定 k）に一致。
  2. 仮想参照点が全て廊下上（射影距離 ≤ 1e-6, s ∈ [0, 116]）。
  3. 低 q̂ 領域の key が合成データで NON_DETECT になる。
  4. fit 時間（GP fit）が diagnostics_ に記録される。
  5. Protocol A 1 fold smoke（run_method 経由, shape/NaN 無し）。
  6. inner CV で validation 地点が前処理統計に混入しない（location_feature_stats spy）。
  7. outer test 地点が fit に届かない（run_method 経由）＋ LOLO 3 地点 smoke。
"""

import time
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import icsr8.methods.gp_augmented_wknn as mod
from icsr8.constants import NON_DETECT_DBM
from icsr8.corridor import _TOTAL_LENGTH, _project, arclength_to_xy
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import run_method
from icsr8.methods._tier4 import dense_matrix, knn_estimate, location_feature_stats
from icsr8.methods.gp_augmented_wknn import (
    GpAugmentedWknn,
    _query_matrix,
    _select_hyperparams,
)
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


def _scan_rows(location_p: int, ap_rssi: dict[str, float], count: int = 0) -> pd.DataFrame:
    aps = list(ap_rssi)
    return pd.DataFrame({
        "location_p": [location_p] * len(aps),
        "ssid": ["test"] * len(aps),
        "rssi": [ap_rssi[a] for a in aps],
        "frequency": [2400] * len(aps),
        "count": [count] * len(aps),
        "ap_name": aps,
    })


# --- 1: w_virt=0 == plain WKNN (inv_sq, fixed k) -------------------------

def test_wvirt_zero_matches_plain_wknn(scans_f, ap_coords, location_coords):
    k = 5
    method = GpAugmentedWknn(delta=4.0, w_virt=0.0, k=k).fit(
        scans_f, ap_coords, location_coords
    )
    est = method.predict(scans_f).set_index("location_p")

    # No virtual points when disabled.
    assert method._virtual_xy.shape[0] == 0

    # Independent plain WKNN over the 59 real reference points only.
    stats = location_feature_stats(scans_f)
    real_mat, keys = dense_matrix(stats.mu)
    coords = location_coords.set_index("location_p")
    real_xy = coords.loc[list(stats.mu.index), ["x", "y"]].to_numpy(float)
    q_locs, q_mat = _query_matrix(scans_f, keys)

    for i, loc in enumerate(q_locs):
        dists = np.linalg.norm(real_mat - q_mat[i], axis=1)
        x, y = knn_estimate(dists, real_xy, k, "inv_sq")
        assert est.loc[loc, "x"] == pytest.approx(x)
        assert est.loc[loc, "y"] == pytest.approx(y)


# --- 2: virtual reference points lie on the corridor ---------------------

def test_virtual_refs_on_corridor(scans_f, ap_coords, location_coords):
    method = GpAugmentedWknn(delta=2.0, w_virt=1.0, k=5).fit(
        scans_f, ap_coords, location_coords
    )
    vs = method._virtual_s
    vx = method._virtual_xy
    assert vx.shape[0] > 0
    assert vs.min() >= 0.0
    assert vs.max() <= _TOTAL_LENGTH + 1e-9

    for x, y in vx:
        assert _project(float(x), float(y))[2] <= 1e-6


# --- 3: low-q̂ region gated to NON_DETECT ---------------------------------

def test_low_qhat_region_non_detect(ap_coords):
    # AP-X detected only in the s∈[0,4] region; AP-Y detected everywhere.
    s_by_loc = {1: 0.0, 2: 2.0, 3: 4.0, 4: 40.0, 5: 50.0, 6: 60.0}
    coords_rows = {loc: arclength_to_xy(s) for loc, s in s_by_loc.items()}
    location_coords = pd.DataFrame({
        "location_p": list(coords_rows),
        "x": [coords_rows[loc][0] for loc in coords_rows],
        "y": [coords_rows[loc][1] for loc in coords_rows],
    })

    train = pd.concat([
        _scan_rows(1, {"AP-X": -40.0, "AP-Y": -55.0}),
        _scan_rows(2, {"AP-X": -42.0, "AP-Y": -53.0}),
        _scan_rows(3, {"AP-X": -44.0, "AP-Y": -51.0}),
        _scan_rows(4, {"AP-Y": -50.0}),
        _scan_rows(5, {"AP-Y": -48.0}),
        _scan_rows(6, {"AP-Y": -46.0}),
    ], ignore_index=True)

    method = GpAugmentedWknn(delta=4.0, w_virt=1.0, k=5).fit(
        train, ap_coords, location_coords
    )

    keys = method._keys
    jx = keys.index(("AP-X", "2.4G"))
    vmat = method._virtual_matrix
    vs = method._virtual_s

    i_far = int(np.argmin(np.abs(vs - 60.0)))
    i_near = int(np.argmin(np.abs(vs - 0.0)))
    assert vmat[i_far, jx] == NON_DETECT_DBM
    assert vmat[i_near, jx] != NON_DETECT_DBM


# --- 3b: constructor all-or-none param contract (F11) --------------------

def test_constructor_requires_all_or_none_params():
    GpAugmentedWknn()  # all omitted -> CV path, ok
    GpAugmentedWknn(delta=2.0, w_virt=0.5, k=5)  # all given -> pinned, ok
    with pytest.raises(ValueError):
        GpAugmentedWknn(delta=2.0)
    with pytest.raises(ValueError):
        GpAugmentedWknn(w_virt=0.5, k=5)


# --- 4: diagnostics_ are deterministic (no wall-clock, F10) --------------

def test_diagnostics_have_no_wallclock_and_are_deterministic(scans_f, ap_coords, location_coords):
    method = GpAugmentedWknn(delta=4.0, w_virt=1.0, k=5).fit(
        scans_f, ap_coords, location_coords
    )
    # F10: perf_counter-based keys removed so diagnostics_ stay deterministic.
    assert "gp_fit_seconds" not in method.diagnostics_
    assert "fit_seconds" not in method.diagnostics_
    assert method.diagnostics_["selected_delta"] == 4.0
    assert method.diagnostics_["selected_w_virt"] == 1.0
    assert method.diagnostics_["selected_k"] == 5
    assert method.diagnostics_["n_gp_keys"] > 0

    # Structural diagnostics reproduce exactly across identical fits.
    again = GpAugmentedWknn(delta=4.0, w_virt=1.0, k=5).fit(
        scans_f, ap_coords, location_coords
    )
    assert method.diagnostics_ == again.diagnostics_


# --- 4b: inner-CV GP fit work is reduced ≥5x (F2) ------------------------

def test_inner_cv_gp_fit_work_reduced(monkeypatch, scans_f, location_coords):
    from icsr8.methods.gp_augmented_wknn import (
        _DEFAULT_LENGTH_GRID,
        _DEFAULT_SIGMA_F_GRID,
        _DEFAULT_SIGMA_N_GRID,
    )

    keep = sorted(scans_f["location_p"].unique())[:22]
    sub = scans_f[scans_f["location_p"].isin(keep)]
    sub_coords = location_coords[location_coords["location_p"].isin(keep)]

    full = (
        len(_DEFAULT_LENGTH_GRID)
        * len(_DEFAULT_SIGMA_F_GRID)
        * len(_DEFAULT_SIGMA_N_GRID)
    )
    real_fit = mod._fit_gp
    tally = {"work": 0, "calls": 0, "grid_sizes": []}

    def spy(s, y, *, length_grid, sigma_f_grid, sigma_n_grid, **kw):
        # Count candidate-Cholesky work: one Cholesky per (ℓ, σ_f, σ_n) combo.
        g = len(length_grid) * len(sigma_f_grid) * len(sigma_n_grid)
        tally["work"] += g
        tally["calls"] += 1
        tally["grid_sizes"].append(g)
        return real_fit(
            s, y,
            length_grid=length_grid,
            sigma_f_grid=sigma_f_grid,
            sigma_n_grid=sigma_n_grid,
            **kw,
        )

    monkeypatch.setattr(mod, "_fit_gp", spy)

    GpAugmentedWknn().fit(sub, None, sub_coords)  # full CV path (no pinned params)

    # Baseline = the current all-full-grid behavior: the same number of GP fits,
    # each over the full 45-candidate grid. The fix must cut total candidate work
    # to <= 1/5 of that by using a reduced grid inside inner CV.
    baseline = tally["calls"] * full
    assert tally["work"] <= baseline / 5
    # Structure: inner-CV fits use a reduced grid; the final radio-map fit uses
    # the full grid.
    assert min(tally["grid_sizes"]) < full
    assert max(tally["grid_sizes"]) == full


# --- 4c: re-fit resets stale virtual points + zero-weight guard (F8) ------

def test_refit_resets_and_updates_virtual_points(scans_f, ap_coords, location_coords):
    all_locs = sorted(scans_f["location_p"].unique())
    keep_a, keep_b = all_locs[:25], all_locs[25:50]
    sub_a = scans_f[scans_f["location_p"].isin(keep_a)]
    coords_a = location_coords[location_coords["location_p"].isin(keep_a)]
    sub_b = scans_f[scans_f["location_p"].isin(keep_b)]
    coords_b = location_coords[location_coords["location_p"].isin(keep_b)]

    method = GpAugmentedWknn(delta=4.0, w_virt=1.0, k=5)
    method.fit(sub_a, ap_coords, coords_a)
    virt_a = method._virtual_matrix.copy()
    assert virt_a.shape[0] > 0

    # Re-fitting the same instance on different train data rebuilds (does not
    # reuse) the virtual radio map.
    method.fit(sub_b, ap_coords, coords_b)
    virt_b = method._virtual_matrix
    assert virt_b.shape[0] > 0
    assert virt_a.shape != virt_b.shape or not np.allclose(virt_a, virt_b)

    # Turning virtual points off and re-fitting must clear ALL stale arrays,
    # not leave the previous fit's virtual points behind.
    method.w_virt = 0.0
    method.fit(sub_b, ap_coords, coords_b)
    assert method._virtual_matrix.shape[0] == 0
    assert method._virtual_xy.shape[0] == 0
    assert method._virtual_s.shape[0] == 0


def test_augmented_estimate_zero_total_weight_no_nan():
    from icsr8.methods.gp_augmented_wknn import _augmented_estimate

    # All reference points are virtual with w_virt=0 -> every neighbor weight is
    # 0 -> total weight 0. The vote must fall back to a finite centroid, not NaN.
    real_mat = np.empty((0, 2))
    real_xy = np.empty((0, 2))
    virt_mat = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    virt_xy = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    q = np.array([0.5, 0.5])

    x, y = _augmented_estimate(q, real_mat, real_xy, virt_mat, virt_xy, k=2, w_virt=0.0)
    assert np.isfinite(x) and np.isfinite(y)


# --- 5: Protocol A one-fold smoke (contract) -----------------------------

def test_protocol_a_smoke(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]
    t0 = time.monotonic()
    est = run_method(
        "gp_augmented_wknn",
        fold.train_scans, fold.test_scans,
        ap_coords, location_coords,
    )
    print(f"\ngp_augmented_wknn fit(CV)+predict one fold = {time.monotonic() - t0:.1f} s")

    assert len(est) == 59
    assert set(est.columns) == {"location_p", "x", "y"}
    assert not est.isna().any().any()
    assert np.isfinite(est[["x", "y"]]).all().all()


# --- 6: inner CV does not leak validation locations into train stats -----

def test_inner_cv_no_validation_leak(monkeypatch, scans_f, location_coords):
    keep = sorted(scans_f["location_p"].unique())[:25]
    sub = scans_f[scans_f["location_p"].isin(keep)]
    sub_coords = location_coords[location_coords["location_p"].isin(keep)]

    seen: list[frozenset[int]] = []
    real_fn = mod.location_feature_stats

    def spy(scans):
        seen.append(frozenset(int(loc) for loc in scans["location_p"].unique()))
        return real_fn(scans)

    monkeypatch.setattr(mod, "location_feature_stats", spy)
    _select_hyperparams(sub, sub_coords)

    expected_train = {
        frozenset(int(loc) for loc in tr["location_p"].unique())
        for tr, _ in iter_inner_cv(sub, k=5, seed=0)
    }
    assert len(seen) >= 5
    # Every training-side stats build equals a fold's inner_train set (its
    # complementary validation locations are structurally excluded).
    assert set(seen) == expected_train


# --- 7: outer test locations never reach fit + LOLO smoke ----------------

def test_outer_leak_and_lolo_smoke(scans_f, ap_coords, location_coords, monkeypatch):
    keep = sorted(scans_f["location_p"].unique())[:30]
    held = sorted(set(scans_f["location_p"].unique()) - set(keep))
    train30 = scans_f[scans_f["location_p"].isin(keep)]

    # F7: spy on the radio-map construction inside fit (_build_train_model ->
    # location_feature_stats) and compare the exact location set it sees with
    # the train subset — held-out locations must never reach it.
    seen: list[frozenset[int]] = []
    real_fn = mod.location_feature_stats

    def spy(scans):
        seen.append(frozenset(int(loc) for loc in scans["location_p"].unique()))
        return real_fn(scans)

    monkeypatch.setattr(mod, "location_feature_stats", spy)

    # run_method with the full 59-location coords must still filter to train.
    est = run_method(
        "gp_augmented_wknn",
        train30, scans_f[scans_f["location_p"].isin(held)],
        ap_coords, location_coords,
        delta=4.0, w_virt=1.0, k=5,
    )
    assert set(est["location_p"]) == set(held)
    assert seen and all(s == frozenset(keep) for s in seen)

    # LOLO smoke: 3 held-out locations, exact fit set + no NaN / shape corruption.
    for fold in islice(iter_lolo(scans_f), 3):
        seen.clear()
        out = run_method(
            "gp_augmented_wknn",
            fold.train_scans, fold.test_scans,
            ap_coords, location_coords,
            delta=4.0, w_virt=0.5, k=5,
        )
        expected = frozenset(int(loc) for loc in fold.train_scans["location_p"].unique())
        assert seen and all(s == expected for s in seen)
        assert fold.held_out not in frozenset().union(*seen)
        assert len(out) == 1
        assert not out.isna().any().any()
        assert np.isfinite(out[["x", "y"]]).all().all()
