"""順位ベース Fingerprinting（doc/improvement_methods_note.txt 手法5）。

AP強度の絶対値ではなく順位（強い順）を使うため、加法オフセット（AP出力変動や
端末側キャリブレーション差）に不変。共通AP集合 O 上で Spearman footrule 距離
d_rank と RSSI二乗距離 d_rssi を λ で混合し、K=3 最近傍の逆距離二乗重み付き
centroid で推定する。λ は train_scans のみを使う inner CV (k=5) でグリッド選択
する（test data には一切触れない）。

どの DB 地点とも共通観測鍵 |O_q∩O_l| が 3 未満で候補集合が空になる場合は、鍵和集合
（-100 埋め）上の raw-RSSI Euclidean 距離に fallback して NaN centroid を回避する
（発動回数は RankFp.rank_fallback_count に記録、inner CV 採点にも同一経路が効く）。

Why not 事前計算した global rank vector を保持する: d_rank は共通鍵集合 O 内での
再順位付けを要求する（O が異なれば global rank は比較不能）。生の rssi_median を
保持しておけば O ごとの再順位付けはそこから毎回導出でき、global rank vector を
別途持つ意味がない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from icsr8.constants import NON_DETECT_DBM, RANDOM_SEED
from icsr8.evaluate import l2_errors
from icsr8.fingerprint import ap_band_fingerprint
from icsr8.methods import register
from icsr8.methods.base import Method
from icsr8.protocols import iter_inner_cv

_LAMBDA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
_K_NEIGHBORS = 3
_MIN_OVERLAP = 3
_EPS = 1e-9

Key = tuple[str, str]  # (ap_name, band)
Db = dict[int, dict[Key, float]]


def _build_db(scans: pd.DataFrame) -> Db:
    """location_p -> {(ap_name, band): rssi_median}（クエリ地点自身の scan のみ使用可能なので predict 側も同じ関数を使う）。"""
    ab_fp = ap_band_fingerprint(scans, ap_coords=None)
    db: Db = {}
    for loc, group in ab_fp.groupby("location_p"):
        db[int(loc)] = dict(zip(zip(group["ap_name"], group["band"]), group["rssi_median"]))
    return db


def _rerank(values: np.ndarray) -> np.ndarray:
    """rank 1 = 最強（RSSI最大）。完全同値は average rank。"""
    return rankdata(-values, method="average")


def _hybrid_distances(query: dict[Key, float], db: Db, lam: float) -> dict[int, float]:
    """クエリ 1 地点分の d_hybrid を DB 各地点について返す（|O|<3 の地点は除外 = +inf 相当）。"""
    d_rank: dict[int, float] = {}
    d_rssi_raw: dict[int, float] = {}
    query_keys = query.keys()
    for loc, fp in db.items():
        common = sorted(query_keys & fp.keys())
        if len(common) < _MIN_OVERLAP:
            continue
        q_vals = np.array([query[k] for k in common])
        l_vals = np.array([fp[k] for k in common])
        footrule = float(np.abs(_rerank(q_vals) - _rerank(l_vals)).sum())
        normalizer = (len(common) ** 2) // 2
        d_rank[loc] = footrule / normalizer
        d_rssi_raw[loc] = float(np.mean((q_vals - l_vals) ** 2))

    if not d_rank:
        return {}

    # Why normalize per-query (not globally): dB^2 と rank 単位は無関係な尺度なので、
    # λ が意味を持つにはこのクエリの候補集合内での相対スケールに揃える必要がある。
    max_rssi = max(d_rssi_raw.values())
    return {
        loc: lam * (d_rssi_raw[loc] / max_rssi if max_rssi > 0 else 0.0) + (1.0 - lam) * d_rank[loc]
        for loc in d_rank
    }


def _raw_euclidean_distances(query: dict[Key, float], db: Db) -> dict[int, float]:
    """全 DB 地点との raw-RSSI Euclidean 距離（鍵和集合を NON_DETECT_DBM で埋める）。

    どの DB 地点とも共通観測鍵が 3 未満で hybrid 候補が空になったときの fallback。
    NaN centroid を避けるため、必ず全地点に有限距離を与える（inner CV の λ 採点に
    NaN を到達させない）。返す距離は Euclidean（sqrt）で、_weighted_centroid が
    二乗して逆距離二乗重みにする点は hybrid 経路と揃う。
    """
    union = set(query)
    for fp in db.values():
        union |= fp.keys()
    keys = sorted(union)
    q_vec = np.array([query.get(k, NON_DETECT_DBM) for k in keys])
    return {
        loc: float(np.sqrt(np.sum((q_vec - np.array([fp.get(k, NON_DETECT_DBM) for k in keys])) ** 2)))
        for loc, fp in db.items()
    }


def _weighted_centroid(distances: dict[int, float], location_coords: pd.DataFrame) -> tuple[float, float]:
    nearest = sorted(distances.items(), key=lambda kv: kv[1])[:_K_NEIGHBORS]
    coords = location_coords.set_index("location_p")
    weights = np.array([1.0 / (d * d + _EPS) for _, d in nearest])
    xs = np.array([coords.loc[loc, "x"] for loc, _ in nearest])
    ys = np.array([coords.loc[loc, "y"] for loc, _ in nearest])
    wsum = weights.sum()
    return float((weights * xs).sum() / wsum), float((weights * ys).sum() / wsum)


def _predict_db(
    db: Db, location_coords: pd.DataFrame, query_scans: pd.DataFrame, lam: float
) -> tuple[pd.DataFrame, int]:
    """推定 DataFrame と fallback 発動回数を返す（predict / inner CV 共通経路）。"""
    ab_fp = ap_band_fingerprint(query_scans, ap_coords=None)
    rows = []
    fallback_count = 0
    for loc_q, group in ab_fp.groupby("location_p"):
        query = dict(zip(zip(group["ap_name"], group["band"]), group["rssi_median"]))
        distances = _hybrid_distances(query, db, lam)
        if not distances:
            # Why not (NaN, NaN): 候補が空だと逆距離重みの分母が 0 になり centroid が
            #   NaN 化する。raw Euclidean に落として必ず有限推定を返す。
            distances = _raw_euclidean_distances(query, db)
            fallback_count += 1
        x, y = _weighted_centroid(distances, location_coords)
        rows.append({"location_p": int(loc_q), "x": x, "y": y})
    return pd.DataFrame(rows), fallback_count


@register
class RankFp(Method):
    name = "rank_fp"
    uses_geometry = False

    def __init__(self) -> None:
        self._db: Db | None = None
        self._location_coords: pd.DataFrame | None = None
        self._lambda: float | None = None
        # Why public: raw-Euclidean fallback がクエリ何地点で発動したかは report 面
        #   の診断値（cf. multiband_wcl.fallback_count）で、fit/predict 後も読める。
        self.rank_fallback_count = 0

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "RankFp":
        # Why not consume ap_coords: uses_geometry=False は指紋（ランキング）のみで
        #   動くメソッドであり、座標未知APも含め全APのfingerprintを利用してよい。
        self._db = _build_db(train_scans)
        self._location_coords = location_coords
        self._lambda = self._select_lambda(train_scans, location_coords)
        return self

    def _select_lambda(self, train_scans: pd.DataFrame, location_coords: pd.DataFrame) -> float:
        best_lambda = _LAMBDA_GRID[0]
        best_err = np.inf
        for lam in _LAMBDA_GRID:
            fold_errors = []
            for cv_train, cv_val in iter_inner_cv(train_scans, k=5, seed=RANDOM_SEED):
                cv_db = _build_db(cv_train)
                est, _ = _predict_db(cv_db, location_coords, cv_val, lam)
                truth = location_coords[location_coords["location_p"].isin(est["location_p"])]
                fold_errors.append(l2_errors(est, truth)["error"])
            mean_err = float(pd.concat(fold_errors).mean())
            # Why <= (not <): 昇順グリッドを前から辿るので、同値なら後発（より大きい
            #   λ）が best を上書きする = 契約の "tie -> larger λ" を素直に実現する。
            if mean_err <= best_err:
                best_err = mean_err
                best_lambda = lam
        return best_lambda

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        if self._db is None or self._location_coords is None or self._lambda is None:
            raise RuntimeError("fit() must be called before predict()")
        est, self.rank_fallback_count = _predict_db(
            self._db, self._location_coords, test_scans, self._lambda
        )
        return est
