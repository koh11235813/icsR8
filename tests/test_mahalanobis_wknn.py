"""mahalanobis_wknn（Tier 4 #14: Ledoit-Wolf shrinkage Mahalanobis WKNN）のテスト。

Tests:
  1. Sigma=I を注入すると Mahalanobis 距離が plain L2 の重み付き centroid と
     厳密一致する（1e-9）。
  2. 実データ由来の LedoitWolf 共分散（within/total 双方）で cho_factor が
     例外なく成功する（SPD）。
  3. fit() の選択ハイパーパラメータと predict() 出力が決定論的（2 回一致）。
  4. fit() 完了時に self.diagnostics_ が選択値と CV スコアを保持する。
  5. leak spy: inner CV の各 fold で validation 地点の scan 行が
     _estimate_covariance に渡らない（cov_mode 双方）。
  6. leak spy: LOLO で held-out location が run_method 経由の fit に届かない
     （DB 地点数が held-out を除いた train 地点数と一致）。
  7. Protocol A 1 fold smoke（registry 経由）。
  8. LOLO smoke（3 地点、iter_lolo を islice）。
"""

from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from icsr8.constants import RANDOM_SEED
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import run_method
from icsr8.methods._tier4 import (
    dense_matrix,
    knn_estimate,
    location_feature_stats,
    safe_cho_factor,
)
from icsr8.methods.mahalanobis_wknn import (
    _GRID,
    MahalanobisWknn,
    _estimate_covariance,
    _predict_with_covariance,
)
from icsr8.protocols import iter_inner_cv, iter_lolo, iter_protocol_a


# --- fixtures ----------------------------------------------------------------

@pytest.fixture(scope="module")
def ap_coords(dataset_dir: Path) -> pd.DataFrame:
    return load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")


@pytest.fixture(scope="module")
def location_coords(dataset_dir: Path) -> pd.DataFrame:
    return load_location_coords(dataset_dir / "location_coordinate_C.csv")[
        ["location_p", "x", "y"]
    ]


@pytest.fixture(scope="module")
def scans_f(rawdata_root: Path) -> pd.DataFrame:
    return load_raw_scans("forward", rawdata_root)


@pytest.fixture(scope="module")
def scans_b(rawdata_root: Path) -> pd.DataFrame:
    return load_raw_scans("backward", rawdata_root)


def _scan_rows(location_p: int, ap_rssi: dict[str, list[float]]) -> pd.DataFrame:
    """1 location 分の scan 行を作る。値は AP ごとの scan 系列（count 昇順）。"""
    rows = []
    for ap, series in ap_rssi.items():
        for count, rssi in enumerate(series):
            rows.append({
                "location_p": location_p,
                "ssid": "test",
                "rssi": rssi,
                "frequency": 2400,
                "count": count,
                "ap_name": ap,
            })
    return pd.DataFrame(rows)


# --- 1: Sigma=I matches plain L2 weighted centroid (1e-9) --------------------

def test_identity_covariance_matches_l2():
    mu_db = np.array([[-40.0, -60.0], [-70.0, -45.0], [-55.0, -55.0]])
    ref_xy = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
    query = np.array([[-42.0, -58.0]])
    query_locs = [99]

    cho = cho_factor(np.eye(2))
    est = _predict_with_covariance(mu_db, ref_xy, query, query_locs, cho, k=2, weighting="inv_sq")

    # Independent L2 computation via the shared knn_estimate helper.
    dists = np.linalg.norm(mu_db - query[0], axis=1)
    exp_x, exp_y = knn_estimate(dists, ref_xy, k=2, weighting="inv_sq")

    assert est.loc[0, "x"] == pytest.approx(exp_x, abs=1e-9)
    assert est.loc[0, "y"] == pytest.approx(exp_y, abs=1e-9)


# --- 1b: within covariance is location-equal-weighted (F1) -------------------

def test_within_covariance_equal_weight_across_locations():
    from sklearn.covariance import LedoitWolf

    from icsr8.methods.mahalanobis_wknn import (
        _mean_within_covariance,
        _scan_level_matrix,
    )

    rng = np.random.default_rng(0)
    base_a1 = rng.normal(-50.0, 6.0, 20)
    base_a2 = rng.normal(-60.0, 4.0, 20)
    base_b1 = np.array([-55.0, -57.0])
    base_b2 = np.array([-62.0, -64.0])

    def build(copies_a: int) -> pd.DataFrame:
        rows = []
        count = 0
        for _ in range(copies_a):
            for v1, v2 in zip(base_a1, base_a2):
                rows.append({"location_p": 1, "ssid": "s", "rssi": float(v1), "frequency": 2400, "count": count, "ap_name": "AP-1"})
                rows.append({"location_p": 1, "ssid": "s", "rssi": float(v2), "frequency": 2400, "count": count, "ap_name": "AP-2"})
                count += 1
        for j, (v1, v2) in enumerate(zip(base_b1, base_b2)):
            rows.append({"location_p": 2, "ssid": "s", "rssi": float(v1), "frequency": 2400, "count": j, "ap_name": "AP-1"})
            rows.append({"location_p": 2, "ssid": "s", "rssi": float(v2), "frequency": 2400, "count": j, "ap_name": "AP-2"})
        return pd.DataFrame(rows)

    def inputs(scans):
        stats = location_feature_stats(scans)
        mu_dense, keys = dense_matrix(stats.mu)
        locs_order = list(stats.mu.index)
        scan_mat, scan_locs = _scan_level_matrix(scans, keys)
        return mu_dense, locs_order, scan_mat, scan_locs

    mu1, lo1, sm1, sl1 = inputs(build(1))
    mu2, lo2, sm2, sl2 = inputs(build(2))

    # Duplicating location A's scans leaves the location-equal-weighted S̄ unchanged.
    s_bar_1 = _mean_within_covariance(sm1, sl1, mu1, lo1)
    s_bar_2 = _mean_within_covariance(sm2, sl2, mu2, lo2)
    assert np.allclose(s_bar_1, s_bar_2, atol=1e-9)

    # Negative control: the buggy scan-level LedoitWolf covariance IS distorted by
    # scan multiplicity (proves the equal-weight fix is what makes S̄ invariant).
    def scan_resid(mu_dense, locs_order, scan_mat, scan_locs):
        loc_pos = {loc: i for i, loc in enumerate(locs_order)}
        row_idx = np.array([loc_pos[loc] for loc in scan_locs])
        return scan_mat - mu_dense[row_idx]

    lw1 = LedoitWolf().fit(scan_resid(mu1, lo1, sm1, sl1)).covariance_
    lw2 = LedoitWolf().fit(scan_resid(mu2, lo2, sm2, sl2)).covariance_
    assert not np.allclose(lw1, lw2, atol=1e-6)


# --- 2: LedoitWolf covariance is SPD (cho_factor succeeds) -------------------

def test_ledoitwolf_covariance_spd_both_modes(scans_f):
    stats = location_feature_stats(scans_f)
    mu_dense, keys = dense_matrix(stats.mu)
    locs_order = list(stats.mu.index)

    for cov_mode in ("within", "total"):
        sigma = _estimate_covariance(scans_f, mu_dense, locs_order, keys, cov_mode)
        assert sigma.shape == (len(keys), len(keys))
        try:
            cho = cho_factor(sigma)
        except LinAlgError:
            pytest.fail(f"cho_factor failed for cov_mode={cov_mode!r} (not SPD)")
        # Solve should also be well-behaved (no NaN/inf).
        solved = cho_solve(cho, np.eye(len(keys)))
        assert np.isfinite(solved).all()


# --- 2b: degenerate (constant-feature) covariance still factors (F4) ---------

def test_degenerate_constant_features_covariance_factors():
    # All scans identical at every location -> zero-variance features -> both
    # within and total covariance collapse to a singular (all-zero) matrix.
    # The SPD guard must let cho_factor succeed and prediction stay finite.
    rows = []
    for loc in (1, 2, 3, 4):
        rows.append(_scan_rows(loc, {"AP-1": [-50.0] * 6, "AP-2": [-60.0] * 6}))
    scans = pd.concat(rows, ignore_index=True)
    stats = location_feature_stats(scans)
    mu_dense, keys = dense_matrix(stats.mu)
    locs_order = list(stats.mu.index)

    for cov_mode in ("within", "total"):
        sigma = _estimate_covariance(scans, mu_dense, locs_order, keys, cov_mode)
        cho = safe_cho_factor(sigma)  # must not raise despite singular sigma
        solved = cho_solve(cho, np.eye(len(keys)))
        assert np.isfinite(solved).all()


# --- 3: determinism -----------------------------------------------------------

def test_fit_predict_deterministic(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]

    m1 = MahalanobisWknn().fit(fold.train_scans, ap_coords, location_coords)
    m2 = MahalanobisWknn().fit(fold.train_scans, ap_coords, location_coords)

    assert m1.selected_cov_mode == m2.selected_cov_mode
    assert m1.selected_k == m2.selected_k
    assert m1.selected_weighting == m2.selected_weighting

    est1 = m1.predict(fold.test_scans)
    est2 = m2.predict(fold.test_scans)
    pd.testing.assert_frame_equal(
        est1.sort_values("location_p").reset_index(drop=True),
        est2.sort_values("location_p").reset_index(drop=True),
    )


# --- 4: diagnostics_ populated -------------------------------------------------

def test_diagnostics_dict(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]
    method = MahalanobisWknn().fit(fold.train_scans, ap_coords, location_coords)

    assert isinstance(method.diagnostics_, dict)
    assert method.diagnostics_["selected_cov_mode"] == method.selected_cov_mode
    assert method.diagnostics_["selected_k"] == method.selected_k
    assert method.diagnostics_["selected_weighting"] == method.selected_weighting
    assert len(method.diagnostics_["cv_scores"]) == len(_GRID)
    assert all(np.isfinite(v) for v in method.diagnostics_["cv_scores"].values())


# --- 5: leak spy — inner CV covariance never sees validation-fold scans ------

def test_inner_cv_covariance_excludes_validation_locations(scans_f, ap_coords, location_coords, monkeypatch):
    import icsr8.methods.mahalanobis_wknn as mw

    recorded: list[tuple[str, set[int]]] = []
    original = mw._estimate_covariance

    def spy(scans_arg, mu_dense, locs_order, keys, cov_mode):
        recorded.append((cov_mode, set(int(x) for x in scans_arg["location_p"].unique())))
        return original(scans_arg, mu_dense, locs_order, keys, cov_mode)

    monkeypatch.setattr(mw, "_estimate_covariance", spy)

    MahalanobisWknn().fit(scans_f, ap_coords, location_coords)

    n_candidates = len(_GRID)
    folds = list(iter_inner_cv(scans_f, k=5, seed=RANDOM_SEED))
    assert len(recorded) >= n_candidates * len(folds)

    idx = 0
    for _, val in folds:
        val_locs = set(int(x) for x in val["location_p"].unique())
        for _ in range(n_candidates):
            _, seen_locs = recorded[idx]
            assert seen_locs.isdisjoint(val_locs), (
                f"inner CV covariance leaked validation locations: {seen_locs & val_locs}"
            )
            idx += 1


# --- 6: leak spy — LOLO held-out location excluded from fit (via run_method) --

def test_lolo_held_out_excluded_from_fit(scans_f, ap_coords, location_coords, monkeypatch):
    # F7: the spy rides the REAL production path (run_method -> fit) and
    # compares the exact location set reaching fit against the LOLO complement.
    import icsr8.methods.mahalanobis_wknn as mw

    fold = next(iter(iter_lolo(scans_f)))
    expected = set(int(x) for x in location_coords["location_p"]) - {fold.held_out}

    seen: dict[str, object] = {}
    real_fit = mw.MahalanobisWknn.fit

    def spy_fit(self, train_scans, ap_coords_arg, location_coords_arg):
        seen["fit_coords"] = set(int(x) for x in location_coords_arg["location_p"])
        result = real_fit(self, train_scans, ap_coords_arg, location_coords_arg)
        seen["db_rows"] = self._mu_db.shape[0]
        return result

    monkeypatch.setattr(mw.MahalanobisWknn, "fit", spy_fit)

    est = run_method("mahalanobis_wknn", fold.train_scans, fold.test_scans, ap_coords, location_coords)
    assert set(est["location_p"]) == {fold.held_out}
    assert np.isfinite(est[["x", "y"]]).all().all()

    assert seen["fit_coords"] == expected
    assert fold.held_out not in seen["fit_coords"]
    assert seen["db_rows"] == len(expected)
    assert fold.held_out not in list(location_feature_stats(fold.train_scans).mu.index)


# --- 7: Protocol A smoke ------------------------------------------------------

def test_protocol_a_smoke(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]
    est = run_method(
        "mahalanobis_wknn", fold.train_scans, fold.test_scans, ap_coords, location_coords
    )
    assert len(est) == 59
    assert set(est.columns) == {"location_p", "x", "y"}
    assert not est.isna().any().any()
    assert np.isfinite(est[["x", "y"]]).all().all()


# --- 8: LOLO smoke (3 locations) ----------------------------------------------

def test_lolo_smoke_three_locations(scans_f, ap_coords, location_coords):
    for fold in islice(iter_lolo(scans_f), 3):
        est = run_method(
            "mahalanobis_wknn", fold.train_scans, fold.test_scans, ap_coords, location_coords
        )
        assert len(est) == 1
        assert set(est.columns) == {"location_p", "x", "y"}
        assert not est.isna().any().any()
        assert np.isfinite(est[["x", "y"]]).all().all()
