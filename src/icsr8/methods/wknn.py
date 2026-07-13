"""Weighted K-Nearest-Neighbor Fingerprinting（doc/improvement_methods_note.txt 手法1）。

特徴空間は学習データの (ap_name, band) 鍵の和集合で固定し、非検出は NON_DETECT_DBM
で埋める。K 近傍の逆距離（または逆距離二乗）重み付き centroid で座標を推定する:
    T = Σ(w_j · P_j) / Σ(w_j)
K ∈ {1,3,5,7} と weighting ∈ {"uniform","inv","inv_sq"} は inner CV
（location 単位 5-fold）でグリッド選択する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from icsr8.constants import NON_DETECT_DBM, RANDOM_SEED
from icsr8.evaluate import l2_errors
from icsr8.fingerprint import ap_band_fingerprint
from icsr8.methods import register
from icsr8.methods.base import Method
from icsr8.protocols import iter_inner_cv

K_GRID: tuple[int, ...] = (1, 3, 5, 7)
WEIGHTING_GRID: tuple[str, ...] = ("uniform", "inv", "inv_sq")


def _build_matrix(
    scans: pd.DataFrame, keys: list[tuple[str, str]] | None = None
) -> tuple[list[tuple[str, str]], list[int], np.ndarray]:
    """(ap_name, band) 特徴行列を作る。

    `keys=None` なら scans 自身の (ap_name, band) 和集合を特徴空間とする
    （fit 時の DB 構築）。`keys` を与えると query 側をその固定空間へ整列する
    （predict / inner CV の query 構築）。

    Returns:
        keys, locs, matrix (n_locs, n_keys)
    """
    ab = ap_band_fingerprint(scans, ap_coords=None)
    if keys is None:
        keys = sorted(set(zip(ab["ap_name"], ab["band"])))

    pivot = ab.pivot_table(index="location_p", columns=["ap_name", "band"], values="rssi_median")
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(keys)).fillna(NON_DETECT_DBM)
    locs = pivot.index.tolist()
    matrix = pivot.to_numpy(dtype=float)
    return keys, locs, matrix


def _weighted_centroid(
    dist: np.ndarray, db_xy: np.ndarray, k: int, weighting: str
) -> tuple[float, float]:
    order = np.argsort(dist, kind="stable")[:k]
    d = dist[order]
    if weighting == "uniform":
        w = np.ones_like(d)
    elif weighting == "inv":
        w = 1.0 / (d + 1e-9)
    elif weighting == "inv_sq":
        w = 1.0 / (d**2 + 1e-9)
    else:
        raise ValueError(f"unknown weighting: {weighting!r}")
    top_xy = db_xy[order]
    x = float((w * top_xy[:, 0]).sum() / w.sum())
    y = float((w * top_xy[:, 1]).sum() / w.sum())
    return x, y


def _select_hyperparams(
    train_scans: pd.DataFrame, location_coords: pd.DataFrame
) -> tuple[int, str]:
    coords = location_coords.set_index("location_p")[["x", "y"]]

    # Why precompute per-fold matrices once: K x weighting は 12 通りあるが、
    # 特徴行列自体はグリッドに依存しない。fold ごとに 12 回再構築するのは無駄。
    fold_data = []
    for inner_train, inner_val in iter_inner_cv(train_scans, k=5, seed=RANDOM_SEED):
        keys, locs_db, mat_db = _build_matrix(inner_train)
        _, locs_val, mat_val = _build_matrix(inner_val, keys=keys)
        db_xy = coords.loc[locs_db].to_numpy(dtype=float)
        truth = location_coords[location_coords["location_p"].isin(locs_val)]
        fold_data.append((mat_db, db_xy, locs_val, mat_val, truth))

    # Why iterate K ascending then weighting in grid order with strict `<`:
    # 契約のタイブレーク（K 小優先 → uniform < inv < inv_sq）を、先勝ちの走査順で
    # そのまま実現する。
    best: tuple[int, str] | None = None
    best_err = float("inf")
    for k in K_GRID:
        for weighting in WEIGHTING_GRID:
            fold_errors = []
            for mat_db, db_xy, locs_val, mat_val, truth in fold_data:
                rows = []
                for i, loc in enumerate(locs_val):
                    dist = np.linalg.norm(mat_db - mat_val[i], axis=1)
                    x, y = _weighted_centroid(dist, db_xy, k, weighting)
                    rows.append({"location_p": int(loc), "x": x, "y": y})
                est = pd.DataFrame(rows)
                fold_errors.append(l2_errors(est, truth)["error"])
            mean_err = float(pd.concat(fold_errors).mean())
            if mean_err < best_err:
                best_err = mean_err
                best = (k, weighting)
    assert best is not None
    return best


@register
class Wknn(Method):
    name = "wknn"
    uses_geometry = False

    def __init__(self, *, k: int | None = None, weighting: str | None = None) -> None:
        # Why require both to skip CV (not either alone): 部分指定だと「残り
        # 半分は何のグリッドから選ぶのか」が未定義になる。テスト容易性のための
        # 明示上書きは両方揃えた場合のみ有効とする。
        self.k = k
        self.weighting = weighting
        self.selected_k: int | None = None
        self.selected_weighting: str | None = None
        self._keys: list[tuple[str, str]] | None = None
        self._db_matrix: np.ndarray | None = None
        self._db_xy: np.ndarray | None = None

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "Wknn":
        # Why not use ap_coords: uses_geometry=False — wknn は (ap_name, band)
        # の識別子のみで動く純指紋手法で、座標未知 AP も含め全 AP を使う。
        del ap_coords

        if self.k is not None and self.weighting is not None:
            self.selected_k = self.k
            self.selected_weighting = self.weighting
        else:
            self.selected_k, self.selected_weighting = _select_hyperparams(
                train_scans, location_coords
            )

        keys, locs, matrix = _build_matrix(train_scans)
        coords = location_coords.set_index("location_p")
        self._keys = keys
        self._db_matrix = matrix
        self._db_xy = coords.loc[locs, ["x", "y"]].to_numpy(dtype=float)
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        if self.selected_k is None or self.selected_weighting is None:
            raise RuntimeError("fit() must be called before predict()")

        _, locs, matrix = _build_matrix(test_scans, keys=self._keys)
        rows = []
        for i, loc in enumerate(locs):
            dist = np.linalg.norm(self._db_matrix - matrix[i], axis=1)
            x, y = _weighted_centroid(dist, self._db_xy, self.selected_k, self.selected_weighting)
            rows.append({"location_p": int(loc), "x": x, "y": y})
        return pd.DataFrame(rows)
