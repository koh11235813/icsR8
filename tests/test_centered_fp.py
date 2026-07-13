"""centered_fp（手法4: Centered/Relative RSSI）のテスト。

Tests:
  1. λ=1 は独立実装した raw-only WKNN centroid と一致する。
  2. λ=0 (centered) はクエリ全体への定数オフセット加算に不変、λ=1 (raw) は不変でない。
  G4. λ=0 は欠測鍵が混在しても観測鍵への +7dB オフセットに不変（G1 修正の要）。
  G5. 地点別セッションオフセットで centered が raw を上回るとき λ≤0.25 が選ばれ、
      再 fit で決定的に同じ λ になる。
  3. fit() が {0.0, 0.25, 0.5, 0.75, 1.0} から λ を選び selected_lambda に格納する。
  4. λ=1 で TRAIN scans 自身を query すると各 location がほぼ自分自身に一致する。
  5. 契約必須の smoke test。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import run_method
from icsr8.methods.centered_fp import CenteredFP
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


def _make_scan_rows(location_p: int, ap_rssi: dict[str, float]) -> pd.DataFrame:
    """1 location 分の scan 行を作る（各 AP 1 回検出、frequency は 2.4G 帯固定）。"""
    aps = list(ap_rssi)
    return pd.DataFrame({
        "location_p": [location_p] * len(aps),
        "ssid": ["test"] * len(aps),
        "rssi": [ap_rssi[a] for a in aps],
        "frequency": [2400] * len(aps),
        "count": [0] * len(aps),
        "ap_name": aps,
    })


@pytest.fixture
def tiny_db():
    """3 DB locations, 3 keys, 全鍵が全 location で観測される（欠測なし）。

    鍵を 3 にしているのは centered 項の共通観測しきい値 (MIN_COMMON=3) を満たし、
    λ=0 が raw 代用に落ちず本来の centered 経路を通るようにするため。
    """
    train = pd.concat([
        _make_scan_rows(1, {"AP-A": -40.0, "AP-B": -60.0, "AP-C": -50.0}),
        _make_scan_rows(2, {"AP-A": -70.0, "AP-B": -45.0, "AP-C": -55.0}),
        _make_scan_rows(3, {"AP-A": -55.0, "AP-B": -55.0, "AP-C": -48.0}),
    ], ignore_index=True)
    location_coords = pd.DataFrame({
        "location_p": [1, 2, 3],
        "x": [0.0, 10.0, 5.0],
        "y": [0.0, 0.0, 10.0],
    })
    ap_coords = pd.DataFrame({
        "ap_name": ["AP-A", "AP-B", "AP-C"], "x": [0.0, 0.0, 0.0], "y": [0.0, 0.0, 0.0],
    })
    return train, location_coords, ap_coords


@pytest.fixture
def sparse_db():
    """欠測鍵を含む DB（G1/G4）。

    鍵和集合は A/B/C/D/E。E は query と loc1 で jointly-missing、loc2 は D を欠く
    ため query の D は one-sided-missing。各候補と query の共通観測鍵は 3 以上
    （centered 項が raw 代用に落ちない）。
    """
    train = pd.concat([
        _make_scan_rows(1, {"AP-A": -40.0, "AP-B": -50.0, "AP-C": -55.0, "AP-D": -60.0}),
        _make_scan_rows(2, {"AP-A": -58.0, "AP-B": -52.0, "AP-C": -48.0, "AP-E": -62.0}),
        _make_scan_rows(3, {"AP-A": -50.0, "AP-B": -45.0, "AP-C": -52.0, "AP-D": -58.0, "AP-E": -55.0}),
    ], ignore_index=True)
    location_coords = pd.DataFrame({
        "location_p": [1, 2, 3], "x": [0.0, 10.0, 5.0], "y": [0.0, 0.0, 10.0],
    })
    ap_coords = pd.DataFrame({
        "ap_name": ["AP-A", "AP-B", "AP-C", "AP-D", "AP-E"], "x": [0.0] * 5, "y": [0.0] * 5,
    })
    return train, location_coords, ap_coords


# --- 1: λ=1 equals an independently-computed raw-only WKNN centroid ------

def test_lambda1_equals_independent_raw_wknn(tiny_db):
    train, location_coords, ap_coords = tiny_db
    query = _make_scan_rows(99, {"AP-A": -42.0, "AP-B": -58.0, "AP-C": -51.0})

    method = CenteredFP(lambda_=1.0)
    method.fit(train, ap_coords, location_coords)
    est = method.predict(query)

    # Independent raw-distance K=3 weighted-centroid computation (no icsr8 reuse).
    db = {
        1: (-40.0, -60.0, -50.0, 0.0, 0.0),
        2: (-70.0, -45.0, -55.0, 10.0, 0.0),
        3: (-55.0, -55.0, -48.0, 5.0, 10.0),
    }
    q = (-42.0, -58.0, -51.0)
    weights = {}
    for loc, (a, b, c, x, y) in db.items():
        d2 = (q[0] - a) ** 2 + (q[1] - b) ** 2 + (q[2] - c) ** 2
        weights[loc] = 1.0 / (d2 + 1e-9)
    wsum = sum(weights.values())
    expected_x = sum(w * db[loc][3] for loc, w in weights.items()) / wsum
    expected_y = sum(w * db[loc][4] for loc, w in weights.items()) / wsum

    assert est.loc[est["location_p"] == 99, "x"].iloc[0] == pytest.approx(expected_x, abs=1e-9)
    assert est.loc[est["location_p"] == 99, "y"].iloc[0] == pytest.approx(expected_y, abs=1e-9)


# --- 2: offset invariance for λ=0, sensitivity for λ=1 --------------------

def test_offset_invariance_centered_vs_raw(tiny_db):
    train, location_coords, ap_coords = tiny_db
    base_query = _make_scan_rows(99, {"AP-A": -42.0, "AP-B": -58.0, "AP-C": -51.0})
    shifted_query = _make_scan_rows(99, {"AP-A": -35.0, "AP-B": -51.0, "AP-C": -44.0})  # +7 dB to every rssi

    centered = CenteredFP(lambda_=0.0)
    centered.fit(train, ap_coords, location_coords)
    est_base_c = centered.predict(base_query)
    est_shift_c = centered.predict(shifted_query)

    raw = CenteredFP(lambda_=1.0)
    raw.fit(train, ap_coords, location_coords)
    est_base_r = raw.predict(base_query)
    est_shift_r = raw.predict(shifted_query)

    assert est_base_c["x"].iloc[0] == pytest.approx(est_shift_c["x"].iloc[0], abs=1e-9)
    assert est_base_c["y"].iloc[0] == pytest.approx(est_shift_c["y"].iloc[0], abs=1e-9)

    assert est_base_r["x"].iloc[0] != pytest.approx(est_shift_r["x"].iloc[0], abs=1e-9)


# --- G4: λ=0 offset invariance holds even with missing keys present -------

def test_lambda0_offset_invariant_with_missing_keys(sparse_db):
    """G1: +7 dB を query の観測鍵のみに足しても λ=0 推定は不変。

    欠測鍵を -100 埋めしたまま centered 差を鍵和集合で合算する旧実装では、
    観測鍵への一律オフセットが own_median を動かし、-100 埋め鍵の centered 値を
    ずらして距離が変わる → 推定が動く（G1 修正前は red）。
    """
    train, location_coords, ap_coords = sparse_db
    base_q = _make_scan_rows(99, {"AP-A": -42.0, "AP-B": -49.0, "AP-C": -54.0, "AP-D": -59.0})
    # +7 dB to every OBSERVED key; AP-E stays missing on the query side.
    shift_q = _make_scan_rows(99, {"AP-A": -35.0, "AP-B": -42.0, "AP-C": -47.0, "AP-D": -52.0})

    method = CenteredFP(lambda_=0.0)
    method.fit(train, ap_coords, location_coords)
    est_base = method.predict(base_q)
    est_shift = method.predict(shift_q)

    assert est_base["x"].iloc[0] == pytest.approx(est_shift["x"].iloc[0], abs=1e-9)
    assert est_base["y"].iloc[0] == pytest.approx(est_shift["y"].iloc[0], abs=1e-9)


# --- G5: session offsets make centered win -> λ selection favours low λ ---

def _offset_corrupted_train():
    """8 地点を一直線に並べ距離依存の指紋を与え、各地点の scan 全体に別々の定数
    オフセット (+12 / -15 dB) を足す。raw 距離はオフセットで撹乱されるが centered
    距離はそれを除去するので、centered が明確に勝つ（λ が小さく選ばれる）はず。"""
    ap_x = {"AP1": 0.0, "AP2": 20.0, "AP3": 40.0, "AP4": 60.0, "AP5": 80.0}
    n = 8
    frames = []
    for i in range(n):
        lx = 10.0 * i
        offset = 12.0 if i % 2 == 0 else -15.0
        rssi = {ap: -30.0 - 0.8 * abs(lx - ax) + offset for ap, ax in ap_x.items()}
        frames.append(_make_scan_rows(i + 1, rssi))
    train = pd.concat(frames, ignore_index=True)
    location_coords = pd.DataFrame({
        "location_p": [i + 1 for i in range(n)],
        "x": [10.0 * i for i in range(n)],
        "y": [0.0] * n,
    })
    ap_coords = pd.DataFrame({
        "ap_name": list(ap_x), "x": list(ap_x.values()), "y": [0.0] * len(ap_x),
    })
    return train, location_coords, ap_coords


def test_fit_selects_low_lambda_under_session_offsets():
    train, location_coords, ap_coords = _offset_corrupted_train()

    method = CenteredFP(lambda_=None).fit(train, ap_coords, location_coords)
    # centered (低 λ) が raw を明確に上回るので grid membership より強い主張ができる。
    assert method.selected_lambda <= 0.25

    # Determinism: 同一 seed の inner CV は再 fit しても同じ λ を選ぶ。
    method2 = CenteredFP(lambda_=None).fit(train, ap_coords, location_coords)
    assert method.selected_lambda == method2.selected_lambda


# --- 3: fit selects λ from the grid and stores selected_lambda -----------

def test_fit_selects_lambda_from_grid(scans_f, ap_coords, location_coords):
    method = CenteredFP(lambda_=None)
    method.fit(scans_f, ap_coords, location_coords)

    assert method.selected_lambda in {0.0, 0.25, 0.5, 0.75, 1.0}


# --- 4: self-recognition sanity (λ=1, query with TRAIN scans itself) -----

def test_self_recognition_lambda1_mean_error_under_1m(scans_f, ap_coords, location_coords):
    est = run_method("centered_fp", scans_f, scans_f, ap_coords, location_coords, lambda_=1.0)

    truth = location_coords.set_index("location_p")
    merged = est.set_index("location_p").join(truth, lsuffix="_est", rsuffix="_truth")
    errors = np.hypot(merged["x_est"] - merged["x_truth"], merged["y_est"] - merged["y_truth"])

    assert errors.mean() < 1.0


# --- 5: smoke test (contract) --------------------------------------------

def test_centered_fp_smoke(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]

    est = run_method(
        "centered_fp",
        fold.train_scans, fold.test_scans,
        ap_coords, location_coords,
    )

    assert len(est) == 59
    assert set(est.columns) == {"location_p", "x", "y"}
    assert not est.isna().any().any()
    assert np.isfinite(est[["x", "y"]]).all().all()
