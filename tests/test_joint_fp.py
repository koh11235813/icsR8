"""joint_fp（RSSI + 検出率の複合 fingerprint 距離, doc/improvement_methods_note.txt 手法19）。

Tests:
  1. η=0 退化: d(q,l) が全 train 鍵の分散重み付き L2（NON_DETECT fill,
     /#train鍵 正規化）に一致する（手計算値と比較）。
  2. 同一 fingerprint（自地点の scans をそのまま query に replay）で d=0。
  3. 検出パターンだけが異なる 2 候補: μ/σ が同一で検出数のみ異なるとき、
     NON_DETECT fill 経由で η=0 でも区別され、η>0 で差がさらに開く。
  4. _query_vectors: r_vec は未検出鍵で NaN、dhat_vec は Beta(1,1) 平滑化で
     全 train 鍵に定義される。
  5. コンストラクタ契約: eta/k/weighting は揃って与えるか揃って省略するか。
  6. リーク guard (outer): fit() の統計は train_scans の地点のみから作られる。
  7. リーク guard (inner CV): fold ごとに再構築される統計が、その fold の
     validation 地点を一切含まない。
  8. inner CV 選択の決定性 + グリッド所属（実データ）。
  9. diagnostics_ の契約必須フィールド。
  10. Protocol A 1 fold smoke（契約必須, run_method 経由, パラメータ固定で高速化）。
  11. LOLO smoke（3 fold, iter_lolo を islice, パラメータ固定で高速化）。
  12. 実データ精度回帰: Protocol A forward→backward 1 fold で Ave < 5 m
     （distance 設計の崩壊を実スケールで検出する）。
  13. LOW-3a: query と 1 鍵だけ共有し完全一致する遠方候補は、多数鍵を共有し
     小さな残差を持つ近傍候補に負ける（現行の全鍵和 NON_DETECT fill 式）。
     旧 common-detected-only 正規化式を同じ query/stats に対して手計算すると
     逆転する（遠方候補が勝つ）ことを old_formula で直接示す — _distances の
     Why-not コメントが記録する崩壊経路の単体再現。
  14. LOW-3b: μ/σ を同一にした 2 候補（term1=0）で q̂ のみ異ならせ、η 項の
     寄与が手計算値 eta*(dhat-qhat)**2/n_keys と一致する（η の効果の直接固定）。
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import run_method
from icsr8.methods._tier4 import location_feature_stats
from icsr8.methods.joint_fp import (
    ETA_GRID,
    K_GRID,
    WEIGHTING_GRID,
    JointFp,
    _EPS,
    _distances,
    _query_vectors,
)
from icsr8.protocols import iter_lolo, iter_protocol_a

KEY_A = ("AP-A", "2.4G")
KEY_B = ("AP-B", "2.4G")
KEY_Z = ("AP-Z", "2.4G")


# --- shared synthetic-data helpers -------------------------------------------

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


# --- fixtures (real data) -----------------------------------------------------

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


# --- 1: eta=0 degenerates to normalized variance-weighted L2 -----------------

def test_eta_zero_degenerates_to_variance_weighted_l2():
    train = pd.concat([
        _make_scans(_scan_rows(1, "AP-A", [-50.0] * 10)),
        _make_scans(_scan_rows(1, "AP-B", [-60.0] * 10)),
        _make_scans(_scan_rows(2, "AP-A", [-55.0] * 10)),
        _make_scans(_scan_rows(2, "AP-B", [-65.0] * 10)),
    ], ignore_index=True)

    stats = location_feature_stats(train)
    keys = list(stats.mu.columns)
    assert keys == [KEY_A, KEY_B]

    # query: r_A=-50 (matches loc1 exactly), r_B=-63 (off from both).
    query = pd.concat([
        _make_scans(_scan_rows(99, "AP-A", [-50.0] * 10)),
        _make_scans(_scan_rows(99, "AP-B", [-63.0] * 10)),
    ], ignore_index=True)
    qvecs = _query_vectors(query, keys)
    r_vec, dhat_vec = qvecs[99]

    dists = _distances(stats, r_vec, dhat_vec, eta=0.0)

    # sigma floors to SIGMA_MIN_DB=1.0 everywhere (raw std=0 in every group),
    # so sigma**2+eps == 2.0 for every (loc, key).
    hand_loc1 = ((-50.0 - (-50.0)) ** 2 + (-63.0 - (-60.0)) ** 2) / 2.0 / 2
    hand_loc2 = ((-50.0 - (-55.0)) ** 2 + (-63.0 - (-65.0)) ** 2) / 2.0 / 2

    locs = list(stats.mu.index)
    assert dists[locs.index(1)] == pytest.approx(hand_loc1, abs=1e-9)
    assert dists[locs.index(2)] == pytest.approx(hand_loc2, abs=1e-9)


# --- 2: identical fingerprint replay -> distance 0 ----------------------------

def test_identical_fingerprint_replay_gives_zero_distance():
    train = pd.concat([
        _make_scans(_scan_rows(1, "AP-A", [-50.0, -48.0, -52.0, -49.0, -51.0])),
        _make_scans(_scan_rows(1, "AP-B", [-60.0, -61.0, -59.0, -60.0, -60.0])),
        _make_scans(_scan_rows(2, "AP-A", [-70.0] * 5)),
        _make_scans(_scan_rows(2, "AP-B", [-40.0] * 5)),
    ], ignore_index=True)

    stats = location_feature_stats(train)
    keys = list(stats.mu.columns)
    locs = list(stats.mu.index)

    # replay loc1's own scans verbatim as the query.
    query = train[train["location_p"] == 1].copy()
    query["location_p"] = 99
    qvecs = _query_vectors(query, keys)
    r_vec, dhat_vec = qvecs[99]

    for eta in (0.0, 1.0, 4.0):
        dists = _distances(stats, r_vec, dhat_vec, eta=eta)
        assert dists[locs.index(1)] == pytest.approx(0.0, abs=1e-9)
        assert dists[locs.index(2)] > 1.0


# --- 3: detection-pattern-only difference, distinguished by eta>0 ------------

def test_detection_pattern_only_difference_distinguished_by_eta():
    # loc1: A/B fully detected (10/10). loc2: A/B detected 3/10, padded with
    # AP-Z (10/10, absent at loc1) so both locations share n_scans=10.
    train = pd.concat([
        _make_scans(_scan_rows(1, "AP-A", [-50.0] * 10)),
        _make_scans(_scan_rows(1, "AP-B", [-60.0] * 10)),
        _make_scans(_scan_rows(2, "AP-A", [-50.0] * 3)),
        _make_scans(_scan_rows(2, "AP-B", [-60.0] * 3)),
        _make_scans(_scan_rows(2, "AP-Z", [-70.0] * 10)),
    ], ignore_index=True)

    stats = location_feature_stats(train)
    keys = list(stats.mu.columns)
    assert set(keys) == {KEY_A, KEY_B, KEY_Z}
    locs = list(stats.mu.index)

    # mu/sigma identical at both locations for A and B (std=0 -> floor).
    assert stats.mu.loc[1, KEY_A] == stats.mu.loc[2, KEY_A] == pytest.approx(-50.0)
    assert stats.mu.loc[1, KEY_B] == stats.mu.loc[2, KEY_B] == pytest.approx(-60.0)

    # query detects only A/B, fully (10/10), never Z.
    query = pd.concat([
        _make_scans(_scan_rows(99, "AP-A", [-50.0] * 10)),
        _make_scans(_scan_rows(99, "AP-B", [-60.0] * 10)),
    ], ignore_index=True)
    qvecs = _query_vectors(query, keys)
    r_vec, dhat_vec = qvecs[99]

    i1, i2 = locs.index(1), locs.index(2)

    dists0 = _distances(stats, r_vec, dhat_vec, eta=0.0)
    assert dists0[i1] == pytest.approx(0.0, abs=1e-9)
    # eta=0 already separates the two: loc2's Z (mu=-70, query undetected ->
    # r filled with NON_DETECT_DBM=-100) contributes (-100+70)^2/(1+1)=450,
    # normalized by #train keys=3 -> 150.
    assert dists0[i2] == pytest.approx(150.0, abs=1e-9)

    dists_pos = _distances(stats, r_vec, dhat_vec, eta=1.0)
    assert dists_pos[i1] == pytest.approx(0.0, abs=1e-9)
    assert dists_pos[i2] > dists0[i2]  # eta>0 widens the gap via the q-hat term


# --- 3b: zero-overlap candidates must not win with distance 0 ----------------

def test_zero_overlap_candidate_does_not_beat_dense_match():
    # loc1 detects A/B (matching the query well); loc2 detects only Z (zero
    # common keys with the query). Old bug: common-only masking gave loc2 an
    # RSSI term of 0, so the sparsest candidate won outright at eta=0. Fixed:
    # NON_DETECT fill makes every detection mismatch a large finite penalty.
    train = pd.concat([
        _make_scans(_scan_rows(1, "AP-A", [-50.0] * 10)),
        _make_scans(_scan_rows(1, "AP-B", [-60.0] * 10)),
        _make_scans(_scan_rows(2, "AP-Z", [-70.0] * 10)),
    ], ignore_index=True)

    stats = location_feature_stats(train)
    keys = list(stats.mu.columns)
    locs = list(stats.mu.index)

    # query matches loc1 imperfectly (so its distance is > 0) and never sees Z.
    query = pd.concat([
        _make_scans(_scan_rows(99, "AP-A", [-52.0] * 10)),
        _make_scans(_scan_rows(99, "AP-B", [-63.0] * 10)),
    ], ignore_index=True)
    r_vec, dhat_vec = _query_vectors(query, keys)[99]

    dists = _distances(stats, r_vec, dhat_vec, eta=0.0)
    i1, i2 = locs.index(1), locs.index(2)
    assert dists[i1] > 0.0
    assert np.isfinite(dists[i2])
    # hand value: loc2 = [(-52+100)^2 + (-63+100)^2 + (-100+70)^2] / 2 / 3
    assert dists[i2] == pytest.approx((2304.0 + 1369.0 + 900.0) / 2.0 / 3.0, abs=1e-9)
    assert dists[i1] < dists[i2]


def test_all_candidates_zero_overlap_stays_finite_and_ranked():
    # Query detects only a key absent from the train key space -> zero common
    # detection with EVERY candidate. NON_DETECT fill keeps every distance
    # finite (no NaN/inf) and still ranks candidates by fingerprint mismatch.
    train = pd.concat([
        _make_scans(_scan_rows(1, "AP-A", [-50.0] * 10)),
        _make_scans(_scan_rows(2, "AP-A", [-55.0] * 3) + _scan_rows(2, "AP-B", [-70.0] * 10)),
    ], ignore_index=True)

    stats = location_feature_stats(train)
    keys = list(stats.mu.columns)
    locs = list(stats.mu.index)

    query = _make_scans(_scan_rows(99, "AP-OTHER", [-40.0] * 10))
    r_vec, dhat_vec = _query_vectors(query, keys)[99]
    assert np.isnan(r_vec).all()

    dists = _distances(stats, r_vec, dhat_vec, eta=1.0)
    assert np.isfinite(dists).all()
    # loc1 (A only) vs loc2 (A weak + B strong): with the query at fill level
    # (-100 everywhere), loc2 accumulates more mismatch (two detected keys and
    # a higher qhat_B), so loc1 ranks closer.
    assert dists[locs.index(1)] < dists[locs.index(2)]


def test_empty_train_key_space_yields_finite_distances():
    # 0 train keys -> the eta term's /n_keys must not become 0/0=NaN.
    from icsr8.methods._tier4 import FeatureStats

    empty_cols = pd.MultiIndex.from_tuples([], names=["ap_name", "band"])
    empty = pd.DataFrame(
        index=pd.Index([1, 2], name="location_p"), columns=empty_cols, dtype=float
    )
    stats = FeatureStats(
        mu=empty, sigma=empty.copy(), qhat=empty.copy(), n_detect=empty.copy(),
    )
    dists = _distances(stats, np.array([]), np.array([]), eta=1.0)
    assert dists.shape == (2,)
    assert np.isfinite(dists).all()


# --- 4: _query_vectors NaN / Beta(1,1) contract -------------------------------

def test_query_vectors_nan_and_beta_smoothing():
    query = _make_scans(_scan_rows(1, "AP-A", [-50.0] * 4, start=0))
    keys = [KEY_A, KEY_B]
    qvecs = _query_vectors(query, keys)
    r_vec, dhat_vec = qvecs[1]

    assert r_vec[0] == pytest.approx(-50.0)
    assert np.isnan(r_vec[1])  # AP-B never detected
    # n_scans = 4 distinct counts (0..3), n_detect_A=4, n_detect_B=0
    assert dhat_vec[0] == pytest.approx(5.0 / 6.0)
    assert dhat_vec[1] == pytest.approx(1.0 / 6.0)


def test_query_vectors_location_with_no_detections_is_all_nan_r():
    # location present in query_scans but with a detected key outside `keys`.
    query = _make_scans(_scan_rows(1, "AP-OTHER", [-50.0] * 3))
    keys = [KEY_A]
    qvecs = _query_vectors(query, keys)
    r_vec, dhat_vec = qvecs[1]
    assert np.isnan(r_vec[0])
    assert dhat_vec[0] == pytest.approx(1.0 / 5.0)  # n_detect=0, n_scans=3


# --- 5: constructor contract ---------------------------------------------------

def test_constructor_requires_all_or_none():
    JointFp()  # all omitted: ok
    JointFp(eta=1.0, k=3, weighting="inv")  # all given: ok
    with pytest.raises(ValueError):
        JointFp(eta=1.0)
    with pytest.raises(ValueError):
        JointFp(k=3, weighting="inv")


# --- 6: outer leak guard (fit derives stats only from train_scans) -----------

def test_fit_stats_derive_only_from_train_scans(scans_f, location_coords):
    keep = sorted(scans_f["location_p"].unique())[:10]
    train_subset = scans_f[scans_f["location_p"].isin(keep)]
    coords_subset = location_coords[location_coords["location_p"].isin(keep)]

    method = JointFp(eta=1.0, k=3, weighting="inv").fit(
        train_subset, pd.DataFrame(), coords_subset
    )
    assert set(int(x) for x in method._stats.mu.index) == set(int(x) for x in keep)


def test_run_method_leak_guard_joint_fp(monkeypatch, scans_f, ap_coords, location_coords):
    """outer leak guard, run_method 経由: run_method が location_coords を
    train_scans の地点へ絞ってから fit() に渡すので、fit() に届く地点集合は
    test-only 地点を一切含まない（held-out 地点は全 59 地点の truth の一部で、
    train30 には無いことを直接確認する）。"""
    keep = sorted(scans_f["location_p"].unique())[:30]
    train30 = scans_f[scans_f["location_p"].isin(keep)]
    assert set(location_coords["location_p"]) > set(keep)  # truth has 59, train has 30

    import icsr8.methods.joint_fp as joint_fp_mod
    seen: dict[str, set[int]] = {}
    real_fit = joint_fp_mod.JointFp.fit

    def spy_fit(self, train_scans, ap_coords, location_coords):
        seen["locations"] = set(int(x) for x in location_coords["location_p"])
        return real_fit(self, train_scans, ap_coords, location_coords)

    monkeypatch.setattr(joint_fp_mod.JointFp, "fit", spy_fit)

    run_method(
        "joint_fp",
        train_scans=train30, test_scans=scans_f,
        ap_coords=ap_coords, location_coords=location_coords,
        eta=1.0, k=3, weighting="inv",
    )

    assert seen["locations"] == set(keep)


# --- 7: inner CV leak guard (validation locations never reach the fold's stats)

def test_inner_cv_fold_stats_match_seeded_folds(monkeypatch):
    # F7: the spy rides the REAL path (JointFp.fit -> select_by_inner_cv ->
    # _fit_predict_candidate) and compares what the fold statistics see 1:1
    # with iter_inner_cv(seed=RANDOM_SEED): first the full-train stats built by
    # fit(), then per fold one call per candidate with the exact inner_train
    # set (= complement of that fold's validation locations).
    from icsr8.constants import RANDOM_SEED
    from icsr8.protocols import iter_inner_cv
    import icsr8.methods.joint_fp as joint_fp_mod

    train = pd.concat(
        [_make_scans(_scan_rows(p, "AP-A", [-50.0 - p] * 10)) for p in range(1, 11)],
        ignore_index=True,
    )
    coords = pd.DataFrame({
        "location_p": list(range(1, 11)),
        "x": [float(p) for p in range(1, 11)],
        "y": [0.0] * 10,
    })

    calls: list[frozenset[int]] = []
    real_stats = joint_fp_mod.location_feature_stats

    def spy_stats(scans):
        calls.append(frozenset(int(x) for x in scans["location_p"].unique()))
        return real_stats(scans)

    monkeypatch.setattr(joint_fp_mod, "location_feature_stats", spy_stats)

    JointFp().fit(train, pd.DataFrame(), coords)

    folds = list(iter_inner_cv(train, k=5, seed=RANDOM_SEED))
    n_cand = len(joint_fp_mod._CANDIDATES)
    full = frozenset(range(1, 11))
    expected: list[frozenset[int]] = [full]  # fit() builds final stats first
    for tr, _ in folds:
        expected += [frozenset(int(x) for x in tr["location_p"].unique())] * n_cand
    assert calls == expected
    for (_, val), got in zip(folds, calls[1:: n_cand]):
        val_locs = frozenset(int(x) for x in val["location_p"].unique())
        assert got == full - val_locs  # exact complement, not merely disjoint


# --- 8: inner CV selection determinism + grid membership (real data) ---------

def test_inner_cv_selection_deterministic(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]
    method_a = JointFp().fit(fold.train_scans, ap_coords, location_coords)
    method_b = JointFp().fit(fold.train_scans, ap_coords, location_coords)

    assert method_a.selected_eta in ETA_GRID
    assert method_a.selected_k in K_GRID
    assert method_a.selected_weighting in WEIGHTING_GRID
    assert method_a.selected_eta == method_b.selected_eta
    assert method_a.selected_k == method_b.selected_k
    assert method_a.selected_weighting == method_b.selected_weighting
    assert len(method_a.diagnostics_["cv_scores"]) == len(ETA_GRID) * len(K_GRID) * len(WEIGHTING_GRID)


# --- 9: diagnostics_ contract --------------------------------------------------

def test_diagnostics_contract():
    train = pd.concat(
        [_make_scans(_scan_rows(p, "AP-A", [-50.0 - p] * 10)) for p in range(1, 7)],
        ignore_index=True,
    )
    coords = pd.DataFrame({
        "location_p": list(range(1, 7)),
        "x": [float(p) for p in range(1, 7)],
        "y": [0.0] * 6,
    })
    method = JointFp(eta=2.0, k=5, weighting="inv_sq").fit(train, pd.DataFrame(), coords)
    assert method.diagnostics_["selected_eta"] == 2.0
    assert method.diagnostics_["selected_k"] == 5
    assert method.diagnostics_["selected_weighting"] == "inv_sq"
    assert method.diagnostics_["cv_scores"] is None  # pinned params -> no CV ran


# --- 10: Protocol A smoke (contract, pinned params for speed) ----------------

def test_joint_fp_protocol_a_smoke(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]

    est = run_method(
        "joint_fp",
        fold.train_scans, fold.test_scans,
        ap_coords, location_coords,
        eta=1.0, k=3, weighting="inv",
    )

    assert len(est) == 59
    assert set(est.columns) == {"location_p", "x", "y"}
    assert not est.isna().any().any()
    assert np.isfinite(est[["x", "y"]]).all().all()


# --- 11: LOLO smoke (3 folds via islice, pinned params for speed) ------------

def test_joint_fp_lolo_smoke(scans_f, ap_coords, location_coords):
    for fold in islice(iter_lolo(scans_f), 3):
        est = run_method(
            "joint_fp",
            fold.train_scans, fold.test_scans,
            ap_coords, location_coords,
            eta=0.5, k=3, weighting="inv_sq",
        )
        assert len(est) == 1
        assert set(est.columns) == {"location_p", "x", "y"}
        assert not est.isna().any().any()
        assert np.isfinite(est[["x", "y"]]).all().all()
        assert int(est["location_p"].iloc[0]) == fold.held_out


# --- 12: real-data accuracy regression (Protocol A forward->backward) --------

def test_joint_fp_protocol_a_accuracy_regression(scans_f, scans_b, ap_coords, location_coords):
    # Guards against distance-design collapse at real scale. The synthetic unit
    # tests missed the original bug because their fixtures were dense: every
    # candidate shared (nearly) all keys with the query, so the buggy
    # "common-detected-only sum / #common" happened to equal the correct
    # full-key-space formula. Only real data has sparse heterogeneous detection
    # patterns, where a far-away candidate sharing exactly ONE well-matching
    # key with the query got a near-zero normalized distance and beat the true
    # location (~18 shared keys, nonzero mean residual): Ave was 22.7 m vs the
    # ~1.7 m this asserts. Params pinned (no inner CV) to keep runtime low; the
    # collapse was param-independent, so any grid point catches a regression.
    fold = iter_protocol_a(scans_f, scans_b)[0]

    est = run_method(
        "joint_fp",
        fold.train_scans, fold.test_scans,
        ap_coords, location_coords,
        eta=1.0, k=3, weighting="inv",
    )

    truth = location_coords.set_index("location_p")
    merged = est.set_index("location_p").join(truth, lsuffix="_est")
    errors = np.hypot(merged["x_est"] - merged["x"], merged["y_est"] - merged["y"])
    assert len(errors) == 59
    assert float(errors.mean()) < 5.0


# --- 13: LOW-3a sparse-but-perfect-match far candidate must not beat a dense
# near neighbor (unit reproduction of the collapse path recorded in the
# Why-not comment above the sig_f/mu_f fill in _distances) -------------------

def test_low3a_sparse_far_candidate_does_not_beat_dense_near_neighbor():
    # loc1 ("near"): detects A/B/C/D, all close to the query (residual 5dB each).
    # loc2 ("far"): detects only E, which matches the query PERFECTLY.
    train = pd.concat([
        _make_scans(_scan_rows(1, "AP-A", [-50.0] * 10)),
        _make_scans(_scan_rows(1, "AP-B", [-52.0] * 10)),
        _make_scans(_scan_rows(1, "AP-C", [-48.0] * 10)),
        _make_scans(_scan_rows(1, "AP-D", [-55.0] * 10)),
        _make_scans(_scan_rows(2, "AP-E", [-50.0] * 10)),
    ], ignore_index=True)

    stats = location_feature_stats(train)
    keys = list(stats.mu.columns)
    locs = list(stats.mu.index)
    assert set(keys) == {
        ("AP-A", "2.4G"), ("AP-B", "2.4G"), ("AP-C", "2.4G"),
        ("AP-D", "2.4G"), ("AP-E", "2.4G"),
    }

    # query detects A-D with a small residual against loc1, and E exactly
    # matching loc2 -- the "1 shared key, perfectly matched" bug trigger.
    query = pd.concat([
        _make_scans(_scan_rows(99, "AP-A", [-55.0] * 10)),
        _make_scans(_scan_rows(99, "AP-B", [-57.0] * 10)),
        _make_scans(_scan_rows(99, "AP-C", [-53.0] * 10)),
        _make_scans(_scan_rows(99, "AP-D", [-60.0] * 10)),
        _make_scans(_scan_rows(99, "AP-E", [-50.0] * 10)),
    ], ignore_index=True)
    r_vec, dhat_vec = _query_vectors(query, keys)[99]

    i_near, i_far = locs.index(1), locs.index(2)

    # current implementation (full key-space sum, NON_DETECT fill): the dense
    # near neighbor wins despite its nonzero per-key residuals.
    dists = _distances(stats, r_vec, dhat_vec, eta=0.0)
    assert dists[i_near] == pytest.approx(260.0, abs=1e-9)
    assert dists[i_far] == pytest.approx(768.3, abs=1e-9)
    assert dists[i_near] < dists[i_far]

    # old (pre-fix) formula: sum only over keys BOTH query and location detect,
    # normalized by that common-key count -- no NON_DETECT fill at all. Applied
    # by hand to the exact same stats/query, it flips the ranking: the far
    # candidate's single common key matches with residual 0, so it "wins" with
    # distance 0 while the near candidate carries its (small) nonzero residual.
    # This is exactly the catastrophic-failure path the Why-not comment in
    # _distances records for the real F->B run (common-detected count=1).
    mu = stats.mu.to_numpy(dtype=float)
    sigma = stats.sigma.to_numpy(dtype=float)

    def old_formula(row: int) -> float:
        common = ~np.isnan(mu[row]) & ~np.isnan(r_vec)
        assert common.sum() > 0
        resid = (r_vec[common] - mu[row][common]) ** 2 / (sigma[row][common] ** 2 + _EPS)
        return float(resid.sum() / common.sum())

    old_near = old_formula(i_near)
    old_far = old_formula(i_far)
    assert old_near == pytest.approx(12.5, abs=1e-9)
    assert old_far == pytest.approx(0.0, abs=1e-9)
    assert old_far < old_near  # old logic would have picked the wrong (far) winner


# --- 14: LOW-3b eta term contribution pinned to a hand calc ------------------

def test_low3b_eta_term_contribution_matches_hand_calc():
    # mu/sigma identical at both locations and equal to the query's r -> term1
    # is 0 for both candidates regardless of eta, isolating the eta*det term.
    from icsr8.methods._tier4 import FeatureStats

    col_index = pd.MultiIndex.from_tuples([("AP-A", "2.4G")], names=["ap_name", "band"])
    idx = pd.Index([1, 2], name="location_p")
    mu = pd.DataFrame([[-50.0], [-50.0]], index=idx, columns=col_index)
    sigma = pd.DataFrame([[1.0], [1.0]], index=idx, columns=col_index)
    qhat = pd.DataFrame([[0.5], [0.9]], index=idx, columns=col_index)
    n_detect = pd.DataFrame([[10], [10]], index=idx, columns=col_index)
    stats = FeatureStats(mu=mu, sigma=sigma, qhat=qhat, n_detect=n_detect)

    r_vec = np.array([-50.0])   # matches mu exactly at both locations
    dhat_vec = np.array([0.5])  # matches loc1's qhat exactly; differs from loc2's

    eta = 2.0
    dists = _distances(stats, r_vec, dhat_vec, eta=eta)

    det_loc1 = (0.5 - 0.5) ** 2 / 1
    det_loc2 = (0.5 - 0.9) ** 2 / 1
    assert dists[0] == pytest.approx(eta * det_loc1, abs=1e-9)
    assert dists[1] == pytest.approx(eta * det_loc2, abs=1e-9)
    assert dists[0] == pytest.approx(0.0, abs=1e-9)
    assert dists[1] == pytest.approx(eta * 0.16, abs=1e-9)
