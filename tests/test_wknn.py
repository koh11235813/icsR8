"""wknn（手法1: Weighted K-Nearest-Neighbor Fingerprinting）のテスト。

Tests:
  1. self-query: TRAIN scans を k=1 で query すると各 location が自分自身の
     座標に厳密一致する（distance-0 self-match）。
  2. hand-computed synthetic: 3 DB locations, 既知 query vector に対し
     k=2, weighting="inv_sq" の centroid が numpy 独自計算と一致する。
  3. CV selection: fit(k=None) が selected_k / selected_weighting をグリッド
     内に格納し、再度 fit しても同じ組が選ばれる（決定性）。
  4. CV argmin + tie-break proof: 2 クラスタ（各 location が別クラスタの
     全 location から遠い）の合成データで K=1 が inner-CV 誤差を最小化する
     ことを構成的に保証し、fit(k=None) が (K=1, "uniform") を選ぶことを
     確認する（K=1 では重み方式が結果に影響しないため 3 方式が厳密同点になり、
     グリッド走査順のタイブレークで "uniform" が残ることも同時に証明する）。
  5. 重複ベクトルの決定性: 2 DB location が完全に同一の指紋を持つとき、
     それと一致する query の k=1 予測が 2 回の predict で厳密に一致する。
  6. 契約必須の smoke test。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import run_method
from icsr8.methods.wknn import Wknn
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
    """3 DB locations, 2 keys, 全鍵が全 location で観測される（欠測なし）。"""
    train = pd.concat([
        _make_scan_rows(1, {"AP-A": -40.0, "AP-B": -60.0}),
        _make_scan_rows(2, {"AP-A": -70.0, "AP-B": -45.0}),
        _make_scan_rows(3, {"AP-A": -55.0, "AP-B": -55.0}),
    ], ignore_index=True)
    location_coords = pd.DataFrame({
        "location_p": [1, 2, 3],
        "x": [0.0, 10.0, 5.0],
        "y": [0.0, 0.0, 10.0],
    })
    ap_coords = pd.DataFrame({"ap_name": ["AP-A", "AP-B"], "x": [0.0, 0.0], "y": [0.0, 0.0]})
    return train, location_coords, ap_coords


# --- 1: self-query, k=1 exact self-match ----------------------------------

def test_self_query_k1_exact_match(scans_f, ap_coords, location_coords):
    method = Wknn(k=1, weighting="uniform")
    method.fit(scans_f, ap_coords, location_coords)
    est = method.predict(scans_f)

    truth = location_coords.set_index("location_p")
    merged = est.set_index("location_p").join(truth, lsuffix="_est", rsuffix="_truth")

    assert np.allclose(merged["x_est"], merged["x_truth"], atol=1e-9)
    assert np.allclose(merged["y_est"], merged["y_truth"], atol=1e-9)


# --- 2: hand-computed synthetic centroid (k=2, inv_sq) --------------------

def test_hand_computed_k2_inv_sq(tiny_db):
    train, location_coords, ap_coords = tiny_db
    query = _make_scan_rows(99, {"AP-A": -42.0, "AP-B": -58.0})

    method = Wknn(k=2, weighting="inv_sq")
    method.fit(train, ap_coords, location_coords)
    est = method.predict(query)

    # Independent k=2 inv_sq weighted-centroid computation (no icsr8 reuse).
    db = {1: (-40.0, -60.0, 0.0, 0.0), 2: (-70.0, -45.0, 10.0, 0.0), 3: (-55.0, -55.0, 5.0, 10.0)}
    q = np.array([-42.0, -58.0])
    dists = {}
    for loc, (a, b, x, y) in db.items():
        dists[loc] = float(np.linalg.norm(q - np.array([a, b])))
    nearest_two = sorted(dists, key=lambda loc: dists[loc])[:2]
    weights = {loc: 1.0 / (dists[loc] ** 2 + 1e-9) for loc in nearest_two}
    wsum = sum(weights.values())
    expected_x = sum(w * db[loc][2] for loc, w in weights.items()) / wsum
    expected_y = sum(w * db[loc][3] for loc, w in weights.items()) / wsum

    assert est.loc[est["location_p"] == 99, "x"].iloc[0] == pytest.approx(expected_x, abs=1e-9)
    assert est.loc[est["location_p"] == 99, "y"].iloc[0] == pytest.approx(expected_y, abs=1e-9)


# --- 3: CV selects (K, weighting) from the grid, deterministically -------

def test_fit_selects_hyperparams_deterministically(scans_f, ap_coords, location_coords):
    method_a = Wknn()
    method_a.fit(scans_f, ap_coords, location_coords)
    method_b = Wknn()
    method_b.fit(scans_f, ap_coords, location_coords)

    assert method_a.selected_k in {1, 3, 5, 7}
    assert method_a.selected_weighting in {"uniform", "inv", "inv_sq"}
    assert method_a.selected_k == method_b.selected_k
    assert method_a.selected_weighting == method_b.selected_weighting


# --- 4: CV argmin + tie-break proof (K=1, "uniform") ----------------------

def test_cv_selects_k1_uniform_by_construction():
    # Two clusters, well separated in *feature* space (RSSI), 5 total
    # locations so inner CV (k=5) is exact leave-one-out for every location
    # (5 folds of size 1, regardless of permutation seed). Cluster A = {1,2},
    # cluster B = {3,4,5}, each with asymmetric within-cluster spacing (both
    # in RSSI offset and in physical position) so no even blend of same-
    # cluster neighbors can coincidentally reproduce a held-out point better
    # than its single nearest neighbor.
    #
    # Cross-cluster feature distance (~85 dB, since AP-A/AP-B alternate
    # between ~-40..-50 and the -100 dBm non-detect fill) dwarfs within-
    # cluster distance (4-10 dB), so for every held-out location the K=1
    # neighbor is always same-cluster (small physical error), while K>=3
    # is forced to blend in at least one far cluster point (large physical
    # error). At K=1 a single neighbor's weight always normalizes to 1
    # regardless of the weighting formula, so uniform/inv/inv_sq tie exactly
    # -- the grid's ascending iteration order (uniform first) then decides.
    train = pd.concat([
        _make_scan_rows(1, {"AP-A": -40.0}),
        _make_scan_rows(2, {"AP-A": -45.0}),
        _make_scan_rows(3, {"AP-B": -40.0}),
        _make_scan_rows(4, {"AP-B": -44.0}),
        _make_scan_rows(5, {"AP-B": -50.0}),
    ], ignore_index=True)
    location_coords = pd.DataFrame({
        "location_p": [1, 2, 3, 4, 5],
        "x": [0.0, 3.0, 1000.0, 1005.0, 1030.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    ap_coords = pd.DataFrame({"ap_name": ["AP-A", "AP-B"], "x": [0.0, 1000.0], "y": [0.0, 0.0]})

    method = Wknn()
    method.fit(train, ap_coords, location_coords)

    assert method.selected_k == 1
    assert method.selected_weighting == "uniform"


# --- 5: duplicate-vector determinism --------------------------------------

def test_duplicate_fingerprint_deterministic():
    # Two DB locations with IDENTICAL fingerprints: a k=1 query that matches
    # both exactly (distance 0 for both) must resolve the tie the same way
    # every time (stable sort keeps the earlier-indexed DB row).
    train = pd.concat([
        _make_scan_rows(1, {"AP-A": -50.0, "AP-B": -60.0}),
        _make_scan_rows(2, {"AP-A": -50.0, "AP-B": -60.0}),
    ], ignore_index=True)
    location_coords = pd.DataFrame({
        "location_p": [1, 2],
        "x": [0.0, 100.0],
        "y": [0.0, 0.0],
    })
    ap_coords = pd.DataFrame({"ap_name": ["AP-A", "AP-B"], "x": [0.0, 0.0], "y": [0.0, 0.0]})
    query = _make_scan_rows(99, {"AP-A": -50.0, "AP-B": -60.0})

    method = Wknn(k=1, weighting="uniform").fit(train, ap_coords, location_coords)
    est_first = method.predict(query)
    est_second = method.predict(query)

    assert est_first["x"].iloc[0] == pytest.approx(est_second["x"].iloc[0])
    assert est_first["y"].iloc[0] == pytest.approx(est_second["y"].iloc[0])
    # Documented behavior: the earlier location_p (stable sort on tied
    # distance-0 rows) wins, i.e. location 1's coordinates, not the mean.
    assert est_first["x"].iloc[0] == pytest.approx(0.0)
    assert est_first["y"].iloc[0] == pytest.approx(0.0)


# --- 6: smoke test (contract) --------------------------------------------

def test_wknn_smoke(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]

    est = run_method(
        "wknn",
        fold.train_scans, fold.test_scans,
        ap_coords, location_coords,
    )

    assert len(est) == 59
    assert set(est.columns) == {"location_p", "x", "y"}
    assert not est.isna().any().any()
    assert np.isfinite(est[["x", "y"]]).all().all()
