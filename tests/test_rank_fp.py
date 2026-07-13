"""順位ベース Fingerprinting（rank_fp, Tier2 手法5）。

Tests:
  1. 完全一致指紋の自己クエリ: λ=0 で d_rank=0 となり、推定位置がその地点の
     真の座標にほぼ一致する。
  2. 加法オフセット不変性: クエリ全AP に +7dB しても λ=0 の推定はビット同一。
  3. 4-AP の手書き例で footrule(=d_rank, λ=0 では d_hybrid と一致) を厳密検証。
     完全同値のタイに対する average rank も検証する。
  G2. 共通観測鍵<3 で候補が空のとき raw Euclidean fallback で有限推定を返し、
      rank_fallback_count に回数を記録する（NaN centroid 回避）。
  G5. 地点別セッションオフセットで rank が raw を上回るとき λ≤0.25 が選ばれ、
      再 fit で決定的に同じ λ になる。
  4. λ グリッド選択: fit 後に選ばれた λ が候補集合に含まれる。
  5. 契約必須の smoke test。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.io import load_ap_coords, load_location_coords
from icsr8.methods import run_method
from icsr8.methods.rank_fp import _LAMBDA_GRID, RankFp, _build_db, _hybrid_distances, _rerank
from icsr8.protocols import iter_protocol_a


# --- fixtures ----------------------------------------------------------------

@pytest.fixture(scope="session")
def ap_coords(dataset_dir: Path) -> pd.DataFrame:
    return load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")


@pytest.fixture(scope="session")
def location_coords(dataset_dir: Path) -> pd.DataFrame:
    return load_location_coords(dataset_dir / "location_coordinate_C.csv")


@pytest.fixture(scope="session")
def protocol_a_fold(rawdata_root: Path):
    """First fold of Protocol A (forward train, backward test)."""
    from icsr8.io import load_raw_scans as _load_raw_scans

    scans_forward = _load_raw_scans("forward", rawdata_root)
    scans_backward = _load_raw_scans("backward", rawdata_root)
    folds = list(iter_protocol_a(scans_forward, scans_backward))
    return folds[0]


def _scan_rows(location_p: int, rssi_by_ap: dict[str, float]) -> pd.DataFrame:
    """1 AP = 1 行の最小 scan テーブル（frequency=2412 -> band 2.4G 固定）。"""
    return pd.DataFrame({
        "location_p": [location_p] * len(rssi_by_ap),
        "ap_name": list(rssi_by_ap.keys()),
        "ssid": ["test"] * len(rssi_by_ap),
        "frequency": [2412] * len(rssi_by_ap),
        "rssi": list(rssi_by_ap.values()),
        "count": [0] * len(rssi_by_ap),
    })


# --- synthetic 5-location DB (distinct rank permutations per location) -------

_APS = ["AP1", "AP2", "AP3", "AP4", "AP5"]

# Each location has a distinct strongest-to-weakest AP order (cyclic shift),
# so only a genuinely identical fingerprint can achieve d_rank=0 against it.
_LOC_ORDER = {
    1: ["AP1", "AP2", "AP3", "AP4", "AP5"],
    2: ["AP2", "AP3", "AP4", "AP5", "AP1"],
    3: ["AP3", "AP4", "AP5", "AP1", "AP2"],
    4: ["AP4", "AP5", "AP1", "AP2", "AP3"],
    5: ["AP5", "AP1", "AP2", "AP3", "AP4"],
}


def _synthetic_train_scans() -> pd.DataFrame:
    frames = []
    for loc, order in _LOC_ORDER.items():
        rssi_by_ap = {ap: -40.0 - 5.0 * rank for rank, ap in enumerate(order)}
        frames.append(_scan_rows(loc, rssi_by_ap))
    return pd.concat(frames, ignore_index=True)


def _synthetic_location_coords() -> pd.DataFrame:
    return pd.DataFrame({
        "location_p": list(_LOC_ORDER.keys()),
        "x": [0.0, 10.0, 20.0, 30.0, 40.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0],
    })


# --- 1: self-query at λ=0 recovers own position -------------------------------

def test_rank_fp_self_query_lambda0_recovers_own_position():
    train_scans = _synthetic_train_scans()
    loc_coords = _synthetic_location_coords()
    db = _build_db(train_scans)

    # Location 1's own fingerprint replayed as a query -> identical to DB entry.
    rssi_by_ap = {ap: -40.0 - 5.0 * rank for rank, ap in enumerate(_LOC_ORDER[1])}
    query = {(ap, "2.4G"): rssi for ap, rssi in rssi_by_ap.items()}

    distances = _hybrid_distances(query, db, lam=0.0)
    assert distances[1] == pytest.approx(0.0, abs=1e-12)
    assert all(d > 0.0 for loc, d in distances.items() if loc != 1)

    from icsr8.methods.rank_fp import _weighted_centroid

    x, y = _weighted_centroid(distances, loc_coords)
    # d_rank=0 -> weight 1/eps ~ 1e9, dwarfing any competitor (d_rank in (0,1]
    # -> weight <= 1e6), so the centroid collapses onto location 1's coords.
    assert x == pytest.approx(0.0, abs=1e-3)
    assert y == pytest.approx(0.0, abs=1e-3)


# --- 2: additive-offset invariance at λ=0 (bit-identical) --------------------

def test_rank_fp_offset_invariance_lambda0_bit_identical():
    from icsr8.methods.rank_fp import _predict_db

    train_scans = _synthetic_train_scans()
    loc_coords = _synthetic_location_coords()
    db = _build_db(train_scans)

    rssi_by_ap = {ap: -40.0 - 5.0 * rank for rank, ap in enumerate(_LOC_ORDER[3])}
    query = _scan_rows(200, rssi_by_ap)
    query_offset = _scan_rows(200, {ap: r + 7.0 for ap, r in rssi_by_ap.items()})

    est, _ = _predict_db(db, loc_coords, query, lam=0.0)
    est_offset, _ = _predict_db(db, loc_coords, query_offset, lam=0.0)

    pd.testing.assert_frame_equal(est, est_offset)


# --- 3: hand-computed 4-AP footrule + tie-average rank ------------------------

def test_rank_fp_footrule_hand_computed_exact():
    query = {
        ("AP1", "2.4G"): -40.0,
        ("AP2", "2.4G"): -50.0,
        ("AP3", "2.4G"): -60.0,
        ("AP4", "2.4G"): -70.0,
    }
    db = {
        1: {
            ("AP1", "2.4G"): -45.0,
            ("AP2", "2.4G"): -42.0,
            ("AP3", "2.4G"): -65.0,
            ("AP4", "2.4G"): -68.0,
        }
    }
    # rank_q (strongest=1): AP1=1, AP2=2, AP3=3, AP4=4
    # rank_l (strongest=1): AP1=2, AP2=1, AP3=3, AP4=4
    # footrule = |1-2|+|2-1|+|3-3|+|4-4| = 2 ; normalizer = floor(4**2/2) = 8
    distances = _hybrid_distances(query, db, lam=0.0)
    assert distances[1] == pytest.approx(2 / 8, abs=1e-12)


def test_rank_fp_rerank_average_ties():
    ranks = _rerank(np.array([-40.0, -40.0, -60.0, -70.0]))
    assert ranks.tolist() == pytest.approx([1.5, 1.5, 3.0, 4.0])


# --- G2: empty candidate set falls back to raw Euclidean (no NaN) ------------

def test_rank_fp_empty_candidates_fallback_to_raw_euclidean():
    """G2: query がどの DB 地点とも共通観測鍵<3 なら hybrid 候補が空になり、
    従来は (NaN, NaN) centroid を返していた。raw-RSSI Euclidean fallback で必ず
    有限推定を返し、回数を public 属性 rank_fallback_count に記録する。"""
    # 5 DB locations (inner CV k=5 を満たす)、各 4 鍵 A/B/C/D を観測。
    db_rssi = {
        1: {"AP-A": -40.0, "AP-B": -42.0, "AP-C": -80.0, "AP-D": -82.0},
        2: {"AP-A": -80.0, "AP-B": -82.0, "AP-C": -40.0, "AP-D": -42.0},
        3: {"AP-A": -60.0, "AP-B": -62.0, "AP-C": -60.0, "AP-D": -62.0},
        4: {"AP-A": -70.0, "AP-B": -72.0, "AP-C": -50.0, "AP-D": -52.0},
        5: {"AP-A": -50.0, "AP-B": -52.0, "AP-C": -70.0, "AP-D": -72.0},
    }
    train = pd.concat([_scan_rows(loc, r) for loc, r in db_rssi.items()], ignore_index=True)
    coords = {1: (0.0, 0.0), 2: (10.0, 0.0), 3: (5.0, 10.0), 4: (20.0, 0.0), 5: (0.0, 20.0)}
    loc_coords = pd.DataFrame({
        "location_p": list(coords), "x": [c[0] for c in coords.values()],
        "y": [c[1] for c in coords.values()],
    })
    ap_coords = pd.DataFrame({
        "ap_name": ["AP-A", "AP-B", "AP-C", "AP-D"], "x": [0.0] * 4, "y": [0.0] * 4,
    })

    # Query observes only 2 keys -> overlap with every DB location is 2 (<3).
    query = _scan_rows(99, {"AP-A": -41.0, "AP-B": -43.0})

    method = RankFp().fit(train, ap_coords, loc_coords)
    est = method.predict(query)

    assert np.isfinite(est[["x", "y"]].to_numpy()).all()
    assert method.rank_fallback_count == 1

    # Independent raw-Euclidean K=3 weighted centroid over the full key union
    # (query's unobserved C/D filled with NON_DETECT_DBM = -100).
    keys = ["AP-A", "AP-B", "AP-C", "AP-D"]
    q = {"AP-A": -41.0, "AP-B": -43.0}
    d2 = {loc: sum((q.get(k, -100.0) - r[k]) ** 2 for k in keys) for loc, r in db_rssi.items()}
    nearest = sorted(d2, key=d2.get)[:3]
    w = {loc: 1.0 / (d2[loc] + 1e-9) for loc in nearest}
    wsum = sum(w.values())
    exp_x = sum(w[loc] * coords[loc][0] for loc in nearest) / wsum
    exp_y = sum(w[loc] * coords[loc][1] for loc in nearest) / wsum

    assert est["x"].iloc[0] == pytest.approx(exp_x, abs=1e-9)
    assert est["y"].iloc[0] == pytest.approx(exp_y, abs=1e-9)


# --- G5: session offsets make rank distance win -> λ selection favours low λ -

def _offset_corrupted_train():
    """8 地点を一直線に並べ距離依存の指紋を与え、各地点の scan 全体に別々の定数
    オフセット (+12 / -15 dB) を足す。順位は加法オフセットに完全不変なので rank
    距離 (λ=0) は真の空間構造を保つが、raw 距離 (λ=1) はオフセットで撹乱される。"""
    ap_x = {"AP1": 0.0, "AP2": 20.0, "AP3": 40.0, "AP4": 60.0, "AP5": 80.0}
    n = 8
    frames = []
    for i in range(n):
        lx = 10.0 * i
        offset = 12.0 if i % 2 == 0 else -15.0
        rssi = {ap: -30.0 - 0.8 * abs(lx - ax) + offset for ap, ax in ap_x.items()}
        frames.append(_scan_rows(i + 1, rssi))
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


def test_rank_fp_selects_low_lambda_under_session_offsets():
    train, location_coords, ap_coords = _offset_corrupted_train()

    method = RankFp().fit(train, ap_coords, location_coords)
    # rank (低 λ) が raw を明確に上回るので grid membership より強い主張ができる。
    assert method._lambda <= 0.25

    # Determinism: 同一 seed の inner CV は再 fit しても同じ λ を選ぶ。
    method2 = RankFp().fit(train, ap_coords, location_coords)
    assert method._lambda == method2._lambda


# --- 4: λ grid selection stored ----------------------------------------------

def test_rank_fp_lambda_selected_from_grid(protocol_a_fold, ap_coords, location_coords):
    fold = protocol_a_fold
    method = RankFp().fit(fold.train_scans, ap_coords, location_coords)
    assert method._lambda in _LAMBDA_GRID


# --- 5: smoke test (contract) -------------------------------------------------

def test_rank_fp_smoke(protocol_a_fold, ap_coords, location_coords):
    fold = protocol_a_fold

    est = run_method(
        "rank_fp",
        fold.train_scans, fold.test_scans,
        ap_coords, location_coords,
    )

    assert len(est) == 59, f"Expected 59 rows, got {len(est)}"
    assert set(est.columns) == {"location_p", "x", "y"}, \
        f"Expected columns {{location_p, x, y}}, got {set(est.columns)}"
    assert not est.isna().any().any(), "Found NaN values in estimate"
    assert est["location_p"].min() == 1
    assert est["location_p"].max() == 59
    assert np.isfinite(est[["x", "y"]]).all().all(), "Found non-finite coordinates"
