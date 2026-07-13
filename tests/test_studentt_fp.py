"""studentt_fp（手法2 Student-t + 手法4 オフセット除去 + 手法11 分散重み）のテスト。

Tests:
  1. _log_t_density が scipy.stats.t.logpdf(x, df=ν, loc=μ+δ, scale=σ) に一致する。
  2. q̂ の Beta(1,1) 平滑化: n_detect=0→1/12, n_detect=10→11/12。
  3. オフセット不変性: 全鍵検出・全鍵 eligible の合成データで、全 query rssi に +7 dB
     しても推定が 1e-9 以内で不変（δ̂ がシフトを吸収する）。
  4. 数値衛生: 実 Protocol-A fold で logp_l が全て有限、posterior が 1 に和する。
  5. 自己認識サニティ: TRAIN scans 自身を query すると 59 地点の 80% 以上で
     posterior argmax が自地点に一致する。
  6. 契約必須の smoke test。
  7. eligibility 境界: n_detect=2（ineligible, Bernoulli のみ）vs n_detect=3
     （eligible, t 項が効く）で、2 DB location 間の logp 差を手計算し、t 項が
     後者でのみスコアに影響することを確認する。
  8. candidate-specific δ̂: μ が定数シフトで異なる 2 DB location に対し、
     δ̂ が候補ごとに異なる query を投げ、posterior が正解 location を優先する
     ことを手計算で確認する。
  9. all-ineligible 経路: 全鍵 n_detect<3 のモデルで predict が NaN なく走り、
     posterior が 1 に和する。
  10. ν 選択の決定性: 実データで 2 回連続 fit しても同じ selected_nu になる。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from icsr8.constants import SIGMA_MIN_DB
from icsr8.fingerprint import ap_band_fingerprint
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import run_method
from icsr8.methods.studentt_fp import (
    StudentTFP,
    _build_model,
    _log_t_density,
    _query_vector,
    _score,
    _softmax,
)
from icsr8.protocols import iter_protocol_a


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


def _make_scans(location_p: int, key_values: dict[str, list[float]], freq: int = 2400):
    """1 location の scan 行を作る。key_values[ap] = 各 scan(count) の rssi 列。"""
    rows = []
    for ap, vals in key_values.items():
        for c, v in enumerate(vals):
            rows.append({
                "location_p": location_p, "ssid": "s", "rssi": float(v),
                "frequency": freq, "count": c, "ap_name": ap,
            })
    return pd.DataFrame(rows)


# --- 1: closed-form log-t equals scipy ------------------------------------

def test_log_t_density_matches_scipy():
    xs = np.linspace(-8.0, 8.0, 33)
    for nu in (3, 5, 10):
        for mu, delta, sigma in [(0.0, 0.0, 1.0), (-55.0, 2.5, 4.0), (10.0, -3.0, 0.7)]:
            got = _log_t_density(xs, nu, mu + delta, sigma)
            expected = stats.t.logpdf(xs, df=nu, loc=mu + delta, scale=sigma)
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


# --- 2: q̂ Beta(1,1) smoothing --------------------------------------------

def test_qhat_smoothing():
    train = pd.concat([
        _make_scans(1, {"AP-A": [-40.0] * 10}),   # loc1: A n_detect=10, B 未検出
        _make_scans(2, {"AP-B": [-50.0] * 10}),   # loc2: B n_detect=10, A 未検出
    ], ignore_index=True)
    coords = pd.DataFrame({"location_p": [1, 2], "x": [0.0, 10.0], "y": [0.0, 0.0]})

    model = _build_model(train, coords)
    ja = model.keys.index(("AP-A", "2.4G"))
    jb = model.keys.index(("AP-B", "2.4G"))
    i1 = model.locs.index(1)

    assert model.q[i1, ja] == pytest.approx(11.0 / 12.0)   # n_detect=10
    assert model.q[i1, jb] == pytest.approx(1.0 / 12.0)    # n_detect=0


# --- 3: offset invariance (δ̂ absorbs a uniform shift) --------------------

def test_offset_invariance_full_detection():
    train = pd.concat([
        _make_scans(1, {"AP-A": [-40.0] * 10, "AP-B": [-60.0] * 10, "AP-C": [-50.0] * 10}),
        _make_scans(2, {"AP-A": [-70.0] * 10, "AP-B": [-45.0] * 10, "AP-C": [-55.0] * 10}),
        _make_scans(3, {"AP-A": [-55.0] * 10, "AP-B": [-55.0] * 10, "AP-C": [-42.0] * 10}),
    ], ignore_index=True)
    coords = pd.DataFrame({
        "location_p": [1, 2, 3], "x": [0.0, 10.0, 5.0], "y": [0.0, 0.0, 10.0],
    })
    ap_coords = pd.DataFrame({"ap_name": [], "x": [], "y": []})

    base = _make_scans(99, {"AP-A": [-42.0], "AP-B": [-58.0], "AP-C": [-49.0]})
    shifted = _make_scans(99, {"AP-A": [-35.0], "AP-B": [-51.0], "AP-C": [-42.0]})  # +7 dB

    method = StudentTFP(nu=5).fit(train, ap_coords, coords)
    est_base = method.predict(base)
    est_shift = method.predict(shifted)

    assert est_base["x"].iloc[0] == pytest.approx(est_shift["x"].iloc[0], abs=1e-9)
    assert est_base["y"].iloc[0] == pytest.approx(est_shift["y"].iloc[0], abs=1e-9)


# --- 4: numerical hygiene on a real fold ----------------------------------

def test_numerical_hygiene(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]
    method = StudentTFP(nu=5).fit(fold.train_scans, ap_coords, location_coords)
    method.predict(fold.test_scans)

    assert method.last_logp
    for logp in method.last_logp.values():
        assert np.isfinite(logp).all()
    for post in method.last_posterior.values():
        assert post.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.isfinite(post).all()


# --- 5: self-recognition sanity -------------------------------------------

def test_self_recognition(scans_f, ap_coords, location_coords):
    method = StudentTFP(nu=5).fit(scans_f, ap_coords, location_coords)
    method.predict(scans_f)

    hits = sum(1 for loc, m in method.last_map_locations.items() if m == loc)
    assert hits / len(method.last_map_locations) >= 0.8


# --- 6: smoke test (contract) ---------------------------------------------

def test_studentt_fp_smoke(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]

    est = run_method(
        "studentt_fp",
        fold.train_scans, fold.test_scans,
        ap_coords, location_coords,
    )

    assert len(est) == 59
    assert set(est.columns) == {"location_p", "x", "y"}
    assert not est.isna().any().any()
    assert np.isfinite(est[["x", "y"]]).all().all()


# --- 7: eligibility boundary (n_detect=2 vs n_detect=3) --------------------

def test_eligibility_boundary_t_term_activates_at_min_count():
    # 2 keys, 2 DB locations, distinct mu per key/location. n_detect is set
    # identically across both locations for both keys, so q_hat (and hence
    # the Bernoulli term) is exactly equal for loc1/loc2 in both scenarios --
    # any logp difference is attributable entirely to the t-term.
    coords = pd.DataFrame({"location_p": [1, 2], "x": [0.0, 10.0], "y": [0.0, 0.0]})
    query_df = _make_scans(99, {"AP-A": [-42.0], "AP-B": [-52.0]})
    q_ab = ap_band_fingerprint(query_df, ap_coords=None)

    # -- ineligible: n_detect=2 < MIN_COUNT=3, t-term must not apply at all.
    train_inelig = pd.concat([
        _make_scans(1, {"AP-A": [-40.0] * 2, "AP-B": [-50.0] * 2}),
        _make_scans(2, {"AP-A": [-46.0] * 2, "AP-B": [-53.0] * 2}),
    ], ignore_index=True)
    model_inelig = _build_model(train_inelig, coords)
    assert not model_inelig.elig.any()
    d, r = _query_vector(model_inelig, q_ab)
    logp_inelig = _score(model_inelig, d, r, nu=5)
    # Bernoulli-only: q_hat identical at both locations -> logp identical.
    assert logp_inelig[0] == pytest.approx(logp_inelig[1], abs=1e-12)

    # -- eligible: n_detect=3 == MIN_COUNT, t-term active for both keys.
    train_elig = pd.concat([
        _make_scans(1, {"AP-A": [-40.0] * 3, "AP-B": [-50.0] * 3}),
        _make_scans(2, {"AP-A": [-46.0] * 3, "AP-B": [-53.0] * 3}),
    ], ignore_index=True)
    model_elig = _build_model(train_elig, coords)
    assert model_elig.elig.all()
    d, r = _query_vector(model_elig, q_ab)
    logp_elig = _score(model_elig, d, r, nu=5)

    # Hand-computed: sigma floors to SIGMA_MIN_DB (raw std=0 within each
    # location/key group), so both keys' beta = 1/(1+(sigma_bar/sigma_ref)^2)
    # = 1/(1+1) = 0.5 (sigma_bar == sigma_ref == SIGMA_MIN_DB for both keys).
    beta = 0.5
    mu1 = {"A": -40.0, "B": -50.0}
    mu2 = {"A": -46.0, "B": -53.0}
    rA, rB = -42.0, -52.0
    delta1 = float(np.median([rA - mu1["A"], rB - mu1["B"]]))
    delta2 = float(np.median([rA - mu2["A"], rB - mu2["B"]]))
    hand_logp_diff = beta * (
        _log_t_density(rA, 5, mu1["A"] + delta1, SIGMA_MIN_DB)
        + _log_t_density(rB, 5, mu1["B"] + delta1, SIGMA_MIN_DB)
        - _log_t_density(rA, 5, mu2["A"] + delta2, SIGMA_MIN_DB)
        - _log_t_density(rB, 5, mu2["B"] + delta2, SIGMA_MIN_DB)
    )
    assert logp_elig[0] - logp_elig[1] == pytest.approx(hand_logp_diff, abs=1e-9)
    # And the t-term must actually move the score (not degenerate to 0).
    assert abs(hand_logp_diff) > 1.0


# --- 8: candidate-specific delta-hat picks the correct location ------------

def test_candidate_specific_delta_prefers_correct_location():
    # loc1 and loc2 differ by non-constant per-key shifts (A: +10, B: +15),
    # so delta-hat (median of per-key residuals) is genuinely candidate-
    # specific rather than a scan-wide constant that cancels identically.
    coords = pd.DataFrame({"location_p": [1, 2], "x": [0.0, 10.0], "y": [0.0, 0.0]})
    train = pd.concat([
        _make_scans(1, {"AP-A": [-40.0] * 3, "AP-B": [-50.0] * 3}),
        _make_scans(2, {"AP-A": [-30.0] * 3, "AP-B": [-35.0] * 3}),
    ], ignore_index=True)
    # Query = loc1's fingerprint shifted by a uniform +3 dB (device AGC-like
    # offset): delta-hat recovers this shift exactly for loc1 (peak t-density)
    # but only partially for loc2 (residual mismatch, lower t-density).
    query_df = _make_scans(99, {"AP-A": [-37.0], "AP-B": [-47.0]})
    q_ab = ap_band_fingerprint(query_df, ap_coords=None)

    model = _build_model(train, coords)
    d, r = _query_vector(model, q_ab)
    logp = _score(model, d, r, nu=5)

    beta = 0.5  # both keys: sigma_bar == sigma_ref == SIGMA_MIN_DB
    rA, rB = -37.0, -47.0
    delta1 = float(np.median([rA - (-40.0), rB - (-50.0)]))
    delta2 = float(np.median([rA - (-30.0), rB - (-35.0)]))
    assert delta1 != pytest.approx(delta2)  # candidate-specific, as designed

    hand_logp1 = beta * (
        _log_t_density(rA, 5, -40.0 + delta1, SIGMA_MIN_DB)
        + _log_t_density(rB, 5, -50.0 + delta1, SIGMA_MIN_DB)
    )
    hand_logp2 = beta * (
        _log_t_density(rA, 5, -30.0 + delta2, SIGMA_MIN_DB)
        + _log_t_density(rB, 5, -35.0 + delta2, SIGMA_MIN_DB)
    )
    # Bernoulli term is identical at both locations (same n_detect for both
    # keys), so it cancels in the difference; logp[0]-logp[1] == t-term diff.
    assert logp[0] - logp[1] == pytest.approx(hand_logp1 - hand_logp2, abs=1e-9)

    post = _softmax(logp)
    assert post[0] > post[1]
    assert post[0] > 0.8  # loc1 is the constructed correct match


# --- 9: all-ineligible path (no key reaches MIN_COUNT) ---------------------

def test_all_ineligible_predict_hygiene():
    train = pd.concat([
        _make_scans(1, {"AP-A": [-40.0] * 2, "AP-B": [-50.0] * 2}),
        _make_scans(2, {"AP-A": [-46.0] * 2, "AP-B": [-53.0] * 2}),
    ], ignore_index=True)
    coords = pd.DataFrame({"location_p": [1, 2], "x": [0.0, 10.0], "y": [0.0, 0.0]})
    ap_coords = pd.DataFrame({"ap_name": [], "x": [], "y": []})
    query = _make_scans(99, {"AP-A": [-42.0], "AP-B": [-52.0]})

    method = StudentTFP(nu=5).fit(train, ap_coords, coords)
    assert not method._model.elig.any()
    est = method.predict(query)

    assert not est.isna().any().any()
    assert np.isfinite(est[["x", "y"]]).all().all()
    for logp in method.last_logp.values():
        assert np.isfinite(logp).all()
    for post in method.last_posterior.values():
        assert post.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.isfinite(post).all()


# --- 10: nu selection determinism (real data) -------------------------------

def test_select_nu_deterministic(scans_f, ap_coords, location_coords):
    method_a = StudentTFP().fit(scans_f, ap_coords, location_coords)
    method_b = StudentTFP().fit(scans_f, ap_coords, location_coords)

    assert method_a.selected_nu in {3, 5, 10}
    assert method_a.selected_nu == method_b.selected_nu
