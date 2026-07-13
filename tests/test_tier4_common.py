"""Tier 4 共通基盤 (icsr8.methods._tier4) のテスト。

Tests:
  1. location_feature_stats: shape / MultiIndex / mu は未検出のみ NaN
     （n_detect=2 でも mu は有限、sigma は NaN）。
  2. qhat: Beta(1,1) 平滑化が地点ごとのスキャン数を分母に使う
     （10 scan → /12、8 scan → /10）。
  3. sigma: n_detect>=MIN_COUNT で max(std, SIGMA_MIN_DB)、未満で NaN。
  4. dense_matrix: 列整列・NaN 埋め・keys 並べ替え。
  5. knn_estimate: uniform / inv / inv_sq の手計算一致、ゼロ距離ガード、
     同距離の stable sort 決定性。
  6. select_by_inner_cv: fold 数・pooled 平均・タイ先頭優先・決定性・
     validation 地点が inner_train_coords に混入しない（leak spy）。
  7. clip_arclength: 廊下全長への境界クリップ（scalar/array）。
  8. 実データ smoke。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.constants import NON_DETECT_DBM, SIGMA_MIN_DB
from icsr8.corridor import _TOTAL_LENGTH
from icsr8.io import load_raw_scans
from scipy.linalg import cho_solve

from icsr8.methods._tier4 import (
    clip_arclength,
    dense_matrix,
    knn_estimate,
    location_feature_stats,
    query_feature_stats,
    safe_cho_factor,
    select_by_inner_cv,
)

KEY_A = ("AP-A", "2.4G")
KEY_B = ("AP-B", "2.4G")


def _make_scans(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


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


@pytest.fixture
def two_loc_scans() -> pd.DataFrame:
    rows = []
    # loc 1: AP-A 10 scans all -50 (std 0 -> floored), AP-B 2 scans (ineligible)
    rows += _scan_rows(1, "AP-A", [-50.0] * 10)
    rows += _scan_rows(1, "AP-B", [-60.0, -62.0])
    # loc 2: AP-A 8 scans alternating (std 5), AP-B undetected
    rows += _scan_rows(2, "AP-A", [-40.0, -50.0] * 4)
    return _make_scans(rows)


# --- 1 & 3: location_feature_stats shape / mu-NaN-rule / sigma floor --------

def test_stats_shape_and_index(two_loc_scans):
    stats = location_feature_stats(two_loc_scans)
    assert stats.mu.shape == (2, 2)
    assert list(stats.mu.index) == [1, 2]
    assert isinstance(stats.mu.columns, pd.MultiIndex)
    assert list(stats.mu.columns) == [KEY_A, KEY_B]
    assert stats.sigma.shape == stats.qhat.shape == stats.n_detect.shape == (2, 2)


def test_mu_nan_only_for_undetected(two_loc_scans):
    stats = location_feature_stats(two_loc_scans)
    # n_detect=2 (< MIN_COUNT) but detected -> mu finite (pins task rule, not studentt's)
    assert stats.mu.loc[1, KEY_B] == pytest.approx(-61.0)
    # undetected -> mu NaN
    assert np.isnan(stats.mu.loc[2, KEY_B])
    assert stats.n_detect.loc[2, KEY_B] == 0
    assert stats.n_detect.loc[1, KEY_A] == 10


def test_sigma_floor_and_nan(two_loc_scans):
    stats = location_feature_stats(two_loc_scans)
    # n_detect=2 < MIN_COUNT -> sigma NaN even though mu finite
    assert np.isnan(stats.sigma.loc[1, KEY_B])
    # std 0 floored to SIGMA_MIN_DB
    assert stats.sigma.loc[1, KEY_A] == pytest.approx(SIGMA_MIN_DB)
    # std 5 preserved (> floor)
    assert stats.sigma.loc[2, KEY_A] == pytest.approx(5.0)


# --- 2: qhat Beta(1,1) with per-location n_scans ----------------------------

def test_qhat_per_location_scan_count(two_loc_scans):
    stats = location_feature_stats(two_loc_scans)
    # loc 1: 10 scans -> denom 12
    assert stats.qhat.loc[1, KEY_A] == pytest.approx(11.0 / 12.0)
    assert stats.qhat.loc[1, KEY_B] == pytest.approx(3.0 / 12.0)
    # loc 2: 8 scans -> denom 10 (discriminates per-location vs hardcoded 10)
    assert stats.qhat.loc[2, KEY_A] == pytest.approx(9.0 / 10.0)
    # undetected at loc 2 -> (0+1)/10
    assert stats.qhat.loc[2, KEY_B] == pytest.approx(1.0 / 10.0)


# --- 4: dense_matrix --------------------------------------------------------

def test_dense_matrix_fill_and_order(two_loc_scans):
    stats = location_feature_stats(two_loc_scans)
    mat, keys = dense_matrix(stats.mu)
    assert keys == [KEY_A, KEY_B]
    assert mat.shape == (2, 2)
    # undetected cell filled
    assert mat[1, 1] == pytest.approx(NON_DETECT_DBM)
    assert mat[0, 0] == pytest.approx(-50.0)


def test_dense_matrix_key_reorder(two_loc_scans):
    stats = location_feature_stats(two_loc_scans)
    mat, keys = dense_matrix(stats.mu, keys=[KEY_B, KEY_A])
    assert keys == [KEY_B, KEY_A]
    # loc1 AP-A now column 1
    assert mat[0, 1] == pytest.approx(-50.0)


# --- 5: knn_estimate --------------------------------------------------------

REF_XY = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])


def test_knn_uniform():
    dists = np.array([1.0, 2.0, 3.0, 4.0])
    x, y = knn_estimate(dists, REF_XY, k=2, weighting="uniform")
    assert (x, y) == pytest.approx((5.0, 0.0))


def test_knn_inv():
    eps = 1e-9
    dists = np.array([1.0, 2.0, 3.0, 4.0])
    x, y = knn_estimate(dists, REF_XY, k=2, weighting="inv")
    w0, w1 = 1.0 / (1.0 + eps), 1.0 / (2.0 + eps)
    exp_x = (w0 * 0.0 + w1 * 10.0) / (w0 + w1)
    assert x == pytest.approx(exp_x)
    assert y == pytest.approx(0.0)


def test_knn_inv_sq():
    eps = 1e-9
    dists = np.array([1.0, 2.0, 3.0, 4.0])
    x, y = knn_estimate(dists, REF_XY, k=2, weighting="inv_sq")
    w0, w1 = 1.0 / (1.0 + eps) ** 2, 1.0 / (2.0 + eps) ** 2
    exp_x = (w0 * 0.0 + w1 * 10.0) / (w0 + w1)
    assert x == pytest.approx(exp_x)


def test_knn_zero_distance_guard():
    ref = np.array([[3.0, 4.0], [100.0, 100.0]])
    dists = np.array([0.0, 5.0])
    x, y = knn_estimate(dists, ref, k=2, weighting="inv")
    assert (x, y) == pytest.approx((3.0, 4.0), abs=1e-3)


def test_knn_stable_sort_ties():
    ref = np.array([[0.0, 0.0], [6.0, 0.0], [9.0, 9.0]])
    dists = np.array([2.0, 2.0, 2.0])
    x, y = knn_estimate(dists, ref, k=2, weighting="uniform")
    # stable sort keeps original order -> first two rows
    assert (x, y) == pytest.approx((3.0, 0.0))


def test_knn_unknown_weighting():
    with pytest.raises(ValueError):
        knn_estimate(np.array([1.0]), REF_XY[:1], k=1, weighting="bogus")


# --- 6: select_by_inner_cv --------------------------------------------------

def _cv_scans(n_loc: int) -> pd.DataFrame:
    rows = []
    for p in range(1, n_loc + 1):
        rows += _scan_rows(p, "AP-A", [-50.0] * 10)
    return _make_scans(rows)


def _coords_x_equals_p(n_loc: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"location_p": list(range(1, n_loc + 1)),
         "x": [float(p) for p in range(1, n_loc + 1)],
         "y": [0.0] * n_loc}
    )


def test_cv_pooled_mean():
    # 6 locations, coords (p, 0); predict (0,0) -> per-location error = p.
    # pooled mean over all 6 locations = 3.5 regardless of fold sizes.
    scans = _cv_scans(6)
    coords = _coords_x_equals_p(6)

    def fit_predict(itrain, ival, itrain_coords, cand):
        vlocs = sorted(ival["location_p"].unique())
        return pd.DataFrame({"location_p": vlocs, "x": 0.0, "y": 0.0})

    best, scores = select_by_inner_cv(scans, coords, [0], fit_predict, k=5)
    assert best == 0
    assert scores[0] == pytest.approx(3.5)


def test_cv_fold_count_and_leak():
    scans = _cv_scans(6)
    coords = _coords_x_equals_p(6)
    calls = []

    def fit_predict(itrain, ival, itrain_coords, cand):
        train_locs = set(itrain_coords["location_p"])
        val_locs = set(ival["location_p"].unique())
        calls.append((train_locs, val_locs))
        # leak contract: validation locations never reach fit's coords
        assert train_locs.isdisjoint(val_locs)
        return pd.DataFrame({"location_p": sorted(val_locs), "x": 0.0, "y": 0.0})

    select_by_inner_cv(scans, coords, [0], fit_predict, k=5)
    assert len(calls) == 5  # k folds x 1 candidate


def test_cv_tie_prefers_first_candidate():
    scans = _cv_scans(6)
    coords = _coords_x_equals_p(6)

    def fit_predict(itrain, ival, itrain_coords, cand):
        # candidate ignored -> identical error for all candidates
        vlocs = sorted(ival["location_p"].unique())
        return pd.DataFrame({"location_p": vlocs, "x": 0.0, "y": 0.0})

    best, scores = select_by_inner_cv(scans, coords, [9, 4], fit_predict, k=5)
    assert best == 9
    assert scores[9] == pytest.approx(scores[4])


def test_cv_deterministic():
    scans = _cv_scans(6)
    coords = _coords_x_equals_p(6)

    def fit_predict(itrain, ival, itrain_coords, cand):
        vlocs = sorted(ival["location_p"].unique())
        # predict (cand, 0) -> error |p - cand|
        return pd.DataFrame({"location_p": vlocs, "x": float(cand), "y": 0.0})

    r1 = select_by_inner_cv(scans, coords, [2, 4, 6], fit_predict, k=5)
    r2 = select_by_inner_cv(scans, coords, [2, 4, 6], fit_predict, k=5)
    assert r1[0] == r2[0]
    assert r1[1] == r2[1]


# --- 7: clip_arclength ------------------------------------------------------

def test_clip_scalar_bounds():
    assert clip_arclength(-5.0) == 0.0
    assert clip_arclength(_TOTAL_LENGTH + 10.0) == pytest.approx(_TOTAL_LENGTH)
    assert clip_arclength(50.0) == pytest.approx(50.0)
    assert isinstance(clip_arclength(50.0), float)


def test_clip_array():
    out = clip_arclength(np.array([-1.0, 50.0, _TOTAL_LENGTH + 100.0]))
    assert isinstance(out, np.ndarray)
    assert out[0] == 0.0
    assert out[1] == pytest.approx(50.0)
    assert out[2] == pytest.approx(_TOTAL_LENGTH)


# --- 10: safe_cho_factor (F4 SPD guard) -------------------------------------

def test_safe_cho_factor_identity_is_exact():
    cho = safe_cho_factor(np.eye(3))
    solved = cho_solve(cho, np.eye(3))
    assert np.allclose(solved, np.eye(3), atol=1e-12)


def test_safe_cho_factor_recovers_singular_matrix():
    # Fully degenerate (all-zero) covariance: raw cho_factor fails; the
    # trace-scaled jitter retry must recover a finite, usable factor.
    cho = safe_cho_factor(np.zeros((4, 4)))
    solved = cho_solve(cho, np.eye(4))
    assert np.isfinite(solved).all()


def test_safe_cho_factor_symmetrizes_input():
    # Asymmetric input whose symmetric part is SPD -> factors the symmetrization.
    a = np.array([[2.0, 1.0], [0.0, 2.0]])
    cho = safe_cho_factor(a)
    solved = cho_solve(cho, np.eye(2))
    assert np.isfinite(solved).all()


# --- 9: query_feature_stats (F9 shared query feature path) ------------------

def test_query_feature_stats_nan_and_counts(two_loc_scans):
    qs = query_feature_stats(two_loc_scans, [KEY_A, KEY_B])
    assert qs.locs == [1, 2]
    assert qs.median.shape == qs.n_detect.shape == (2, 2)
    # loc1 AP-A median -50; AP-B median -61 (2 scans, still detected -> finite)
    assert qs.median[0, 0] == pytest.approx(-50.0)
    assert qs.median[0, 1] == pytest.approx(-61.0)
    # loc2 AP-B undetected -> NaN preserved (query path never fills NON_DETECT)
    assert np.isnan(qs.median[1, 1])
    assert qs.n_detect[0, 0] == 10
    assert qs.n_detect[1, 1] == 0
    # n_scans per location (distinct counts): loc1=10, loc2=8
    assert qs.n_scans[0] == 10
    assert qs.n_scans[1] == 8


def test_query_feature_stats_key_alignment_and_order():
    rows = _scan_rows(1, "AP-A", [-50.0] * 4) + _scan_rows(1, "AP-B", [-60.0] * 4)
    scans = _make_scans(rows)
    # reversed key order reorders columns deterministically
    qs = query_feature_stats(scans, [KEY_B, KEY_A])
    assert qs.median[0, 0] == pytest.approx(-60.0)  # KEY_B first
    assert qs.median[0, 1] == pytest.approx(-50.0)
    # a train key absent from the query yields an all-NaN / zero-detect column
    qs2 = query_feature_stats(scans, [KEY_A, ("AP-Z", "2.4G")])
    assert qs2.median[0, 0] == pytest.approx(-50.0)
    assert np.isnan(qs2.median[0, 1])
    assert qs2.n_detect[0, 1] == 0


def test_query_feature_stats_is_shared_by_joint_and_gp():
    # F9: both methods route their query features through this single helper
    # rather than reimplementing pivot/align/smoothing each.
    import icsr8.methods.gp_augmented_wknn as ga
    import icsr8.methods.joint_fp as jf

    assert jf.query_feature_stats is query_feature_stats
    assert ga.query_feature_stats is query_feature_stats


# --- 8: real-data smoke -----------------------------------------------------

def test_real_data_smoke(rawdata_root: Path):
    scans = load_raw_scans("forward", rawdata_root)
    stats = location_feature_stats(scans)
    assert stats.mu.shape[0] == 59
    assert stats.mu.shape == stats.sigma.shape == stats.qhat.shape
    # qhat is a probability in (0, 1]
    q = stats.qhat.to_numpy()
    assert np.all(q > 0.0) and np.all(q <= 1.0)
    # sigma floored where present
    s = stats.sigma.to_numpy()
    assert np.all(s[~np.isnan(s)] >= SIGMA_MIN_DB)
    mat, keys = dense_matrix(stats.mu)
    assert mat.shape == (59, len(keys))
    assert not np.isnan(mat).any()
