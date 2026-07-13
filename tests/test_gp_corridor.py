"""gp_corridor（手法3 廊下座標系 GP radio map + 手法16 セグメント階層推定）のテスト。

Tests:
  1. Matérn-3/2 カーネルの解析値: k(0)=σ_f²、k(ℓ) は in-test 閉形式に一致。
  2. 無雑音補間: 12 点の滑らかな y を σ_n≈0 で GP fit すると訓練点で y を再現
     （潜在分散は σ_n=1e-3 スケールで <1e-2 に収束）。
  3. LML グリッド選択: in-test の独立 numpy Cholesky 実装で 2 候補の log marginal
     likelihood を計算し、_fit_gp が LML の高い方を選ぶことを確認する。
  4. セグメント分類器の自己精度が実 forward データで ≥ 0.8。
  5. held-out セグメント汎化: 全 3 セグメントに分散する 12 地点を学習から除外し、
     予測座標の segment_of が真のセグメントに ≥70% 一致する。
  6. fallback 経路: query が GP を持たない鍵しか検出しないと、推定は予測
     セグメントの中点になり fallback_count が 1 増える。
  7. 事後分布 hygiene: 実 fold で ŝ が予測セグメント範囲内・有限（全 59 地点）。
  8. ランタイムガード: 1 Protocol-A fold の fit+predict が 120 s 未満。
  9. 契約必須の smoke test。
"""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.corridor import arclength_to_xy, segment_of
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import run_method
from icsr8.methods.gp_corridor import (
    SEGMENT_RANGES,
    GpCorridor,
    _fit_gp,
    _gp_posterior,
    _matern32,
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


# --- 1: Matérn-3/2 kernel analytic ---------------------------------------

def test_matern32_analytic():
    sigma_f, length = 5.0, 8.0

    # k(0) = σ_f²
    assert _matern32(0.0, sigma_f, length) == pytest.approx(sigma_f**2)

    # k(ℓ) = σ_f²·(1+√3)·exp(-√3), closed form computed here in-test.
    expected = sigma_f**2 * (1.0 + math.sqrt(3.0)) * math.exp(-math.sqrt(3.0))
    assert _matern32(length, sigma_f, length) == pytest.approx(expected)

    # array input: elementwise, monotone decreasing with |d|.
    d = np.array([0.0, 4.0, 8.0, 16.0])
    k = _matern32(d, sigma_f, length)
    assert k[0] == pytest.approx(sigma_f**2)
    assert np.all(np.diff(k) < 0.0)


# --- 2: noiseless interpolation reproduces train targets -----------------

def test_gp_noiseless_interpolation_reproduces_train():
    s = np.linspace(0.0, 30.0, 12)
    y = 3.0 * np.sin(s / 5.0) + 2.0

    gp = _fit_gp(
        s, y,
        length_grid=(8.0,),
        sigma_f_grid=(5.0,),
        sigma_n_grid=(1e-3,),
    )
    mu, v = _gp_posterior(gp, s)

    # With σ_n → 0 the GP posterior mean interpolates the training targets.
    assert np.max(np.abs(mu - y)) < 1e-2
    # Latent variance at (near-)train points collapses toward 0. Tightened from
    # the original <0.5 bound: at σ_n=1e-3 the actual max is ~1e-6, so <1e-2 is
    # still generous while being a substantive tightening (not a fluke).
    assert np.all(v >= -1e-9)
    assert np.max(v) < 1e-2


# --- 3: LML grid selection (independent Cholesky computation) ------------

def _independent_lml(s: np.ndarray, y: np.ndarray, sigma_f: float, length: float,
                      sigma_n: float, jitter: float = 1e-9) -> float:
    """Log marginal likelihood, computed from scratch with np.linalg.cholesky
    (not scipy's cho_solve/solve_triangular that _fit_gp uses internally)."""
    dist = np.abs(s[:, None] - s[None, :])
    k = _matern32(dist, sigma_f, length)
    n = len(s)
    k_noisy = k + (sigma_n**2 + jitter) * np.eye(n)
    chol = np.linalg.cholesky(k_noisy)
    yc = y - y.mean()
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, yc))
    return float(
        -0.5 * yc @ alpha - np.sum(np.log(np.diag(chol))) - 0.5 * n * math.log(2.0 * math.pi)
    )


def test_fit_gp_selects_higher_lml_candidate():
    # Tiny, non-smooth synthetic series: length=2.0 and length=20.0 give
    # visibly different fits, so the two candidates are not near a tie.
    s = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
    y = np.array([1.0, 2.5, 1.0, -1.0, 0.5])
    sigma_f, sigma_n = 3.0, 1.0

    lml_short = _independent_lml(s, y, sigma_f, 2.0, sigma_n)
    lml_long = _independent_lml(s, y, sigma_f, 20.0, sigma_n)
    assert lml_short != pytest.approx(lml_long)
    winner = 2.0 if lml_short > lml_long else 20.0

    # Restrict the module's grids to exactly these two length candidates
    # (single-value sigma_f/sigma_n grids) so _fit_gp's selection is a clean
    # binary choice between the two independently-scored candidates.
    gp = _fit_gp(s, y, length_grid=(2.0, 20.0), sigma_f_grid=(sigma_f,), sigma_n_grid=(sigma_n,))
    assert gp.length == pytest.approx(winner)


# --- 4: segment classifier self-accuracy ---------------------------------

def test_segment_classifier_self_accuracy(scans_f, ap_coords, location_coords):
    method = GpCorridor().fit(scans_f, ap_coords, location_coords)
    acc = method.segment_train_accuracy
    print(f"\nsegment classifier self-accuracy (forward) = {acc:.3f}")
    assert acc >= 0.8


# --- 5: held-out segment generalization -----------------------------------

def test_held_out_segment_generalization(scans_f, ap_coords, location_coords):
    # 12 locations spread across all three segments (4 from C, 5 from C2,
    # 3 from C3), held out entirely from fit; the remaining ~47 train the
    # GPs + segment classifier.
    coords = location_coords.set_index("location_p")
    by_segment: dict[str, list[int]] = {"C": [], "C2": [], "C3": []}
    for loc, row in coords.iterrows():
        by_segment[segment_of(float(row.x), float(row.y))].append(int(loc))

    held_out = by_segment["C"][::4][:4] + by_segment["C2"][::5][:5] + by_segment["C3"][::4][:3]
    assert len(held_out) == 12
    train_locs = sorted(set(coords.index) - set(held_out))

    train_scans = scans_f[scans_f["location_p"].isin(train_locs)]
    test_scans = scans_f[scans_f["location_p"].isin(held_out)]
    train_coords = location_coords[location_coords["location_p"].isin(train_locs)]

    method = GpCorridor().fit(train_scans, ap_coords, train_coords)
    est = method.predict(test_scans)

    hits = 0
    for row in est.itertuples():
        pred_segment = segment_of(row.x, row.y)
        true_segment = segment_of(*coords.loc[row.location_p, ["x", "y"]])
        hits += pred_segment == true_segment
    match_rate = hits / len(est)
    print(f"\nheld-out segment match rate (12 locations) = {match_rate:.3f}")
    assert match_rate >= 0.7


# --- 6: fallback path when no detected key has a GP -----------------------

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


def test_predict_fallback_to_segment_midpoint():
    # 5 synthetic locations, keys A/B only, split across two segments (C, C2)
    # so the segment classifier has >=2 classes to fit.
    train = pd.concat([
        _make_scan_rows(1, {"AP-A": -40.0, "AP-B": -55.0}),
        _make_scan_rows(2, {"AP-A": -42.0, "AP-B": -53.0}),
        _make_scan_rows(3, {"AP-A": -44.0, "AP-B": -51.0}),
        _make_scan_rows(4, {"AP-A": -60.0, "AP-B": -30.0}),
        _make_scan_rows(5, {"AP-A": -62.0, "AP-B": -28.0}),
    ], ignore_index=True)
    location_coords = pd.DataFrame({
        "location_p": [1, 2, 3, 4, 5],
        "x": [2.0, 6.0, 10.0, 0.0, 0.0],
        "y": [0.0, 0.0, 0.0, 2.0, 6.0],
    })
    ap_coords = pd.DataFrame({"ap_name": ["AP-A", "AP-B"], "x": [0.0, 0.0], "y": [0.0, 0.0]})

    method = GpCorridor().fit(train, ap_coords, location_coords)
    assert set(method._gps) == {("AP-A", "2.4G"), ("AP-B", "2.4G")}

    # Query detects only AP-C: a key with no fitted GP, so `usable` is empty
    # regardless of which segment the classifier predicts.
    query = _make_scan_rows(99, {"AP-C": -50.0})
    est = method.predict(query)

    assert method.fallback_count == 1
    diag = method.last_predictions_.iloc[0]
    lo, hi = SEGMENT_RANGES[diag["segment"]]
    expected_mid = 0.5 * (lo + hi)
    assert diag["s_hat"] == pytest.approx(expected_mid)
    expected_x, expected_y = arclength_to_xy(expected_mid)
    assert est["x"].iloc[0] == pytest.approx(expected_x)
    assert est["y"].iloc[0] == pytest.approx(expected_y)


# --- 7: posterior hygiene on a real fold ---------------------------------

def test_posterior_hygiene_one_fold(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]
    method = GpCorridor().fit(fold.train_scans, ap_coords, location_coords)
    est = method.predict(fold.test_scans)

    diag = method.last_predictions_
    assert len(diag) == 59

    # ŝ finite and inside the predicted segment's arc range for every location.
    for row in diag.itertuples():
        lo, hi = SEGMENT_RANGES[row.segment]
        assert math.isfinite(row.s_hat)
        assert lo - 1e-6 <= row.s_hat <= hi + 1e-6

    # Substantive check: for interior ŝ, the returned (x, y) projects back to
    # the predicted segment (skip corner rows where s∈{32,88} tie-break is ambiguous).
    merged = est.merge(diag, on="location_p")
    for row in merged.itertuples():
        near_corner = min(abs(row.s_hat - 32.0), abs(row.s_hat - 88.0)) < 0.5
        if near_corner:
            continue
        assert segment_of(row.x, row.y) == row.segment


# --- 8: runtime guard ----------------------------------------------------

def test_runtime_under_120s(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]

    t0 = time.monotonic()
    method = GpCorridor().fit(fold.train_scans, ap_coords, location_coords)
    method.predict(fold.test_scans)
    elapsed = time.monotonic() - t0

    print(f"\ngp_corridor fit+predict one fold = {elapsed:.2f} s")
    assert elapsed < 120.0


# --- 9: smoke test (contract) --------------------------------------------

def test_gp_corridor_smoke(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]

    est = run_method(
        "gp_corridor",
        fold.train_scans, fold.test_scans,
        ap_coords, location_coords,
    )

    assert len(est) == 59
    assert set(est.columns) == {"location_p", "x", "y"}
    assert not est.isna().any().any()
    assert np.isfinite(est[["x", "y"]]).all().all()
