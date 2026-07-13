"""GP radio map による仮想 fingerprint 拡張 WKNN（#20, Tier 4）。

各 (ap_name, band) 鍵の RSSI を廊下弧長 s の 1D Gaussian Process でモデル化し、
一定間隔 Δ の s グリッド上に仮想参照点を合成する。仮想点の特徴 μ_key(s) は
検出率 q̂(s) < 0.5 の領域で NON_DETECT へゲートする。query は実測参照点と
仮想参照点の合同集合へ L2 近傍投票し、仮想点の重みは係数 w_virt で減衰する。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from icsr8.constants import NON_DETECT_DBM, RANDOM_SEED
from icsr8.corridor import _TOTAL_LENGTH, arclength_to_xy, xy_to_arclength
from icsr8.methods import register
from icsr8.methods.base import Method

# Why not reimplement Matérn-3/2 + grid-LML + Cholesky: gp_corridor がその GP 部品を
# 既に厳密実装しており（k(0)=σ_f²・jitter 付き Cholesky・LML タイブレーク）、契約が
# 明示的に許可する sibling private import で再利用する。同一設計の再実装は診断困難な
# 数値差を生むだけで利得がない。
from icsr8.methods.gp_corridor import (
    _DEFAULT_LENGTH_GRID,
    _DEFAULT_SIGMA_F_GRID,
    _DEFAULT_SIGMA_N_GRID,
    _fit_gp,
    _gp_posterior,
)
from icsr8.methods._tier4 import (
    dense_matrix,
    location_feature_stats,
    query_feature_stats,
)
from icsr8.protocols import iter_inner_cv

Key = tuple[str, str]

DELTA_GRID: tuple[float, ...] = (2.0, 4.0)
WVIRT_GRID: tuple[float, ...] = (0.25, 0.5, 1.0)
K_GRID: tuple[int, ...] = (5, 7)

# Why a reduced GP grid inside inner CV (F2): inner CV ranks only (Δ, w_virt, k),
# which are independent of the GP kernel hyperparameters (the GP radio map is a
# preprocessing step shared by every candidate). A single mid-scale GP config is
# enough to rank those candidates cheaply; the final radio map is still fit on the
# full grid for accuracy. This cuts the dominant per-fold GP Cholesky work by the
# full/reduced grid-size ratio (45 → 1 candidate) without touching gp_corridor's
# _fit_gp (whose per-key hyperparameter search we deliberately do not reimplement).
_INNER_LENGTH_GRID: tuple[float, ...] = (8.0,)
_INNER_SIGMA_F_GRID: tuple[float, ...] = (5.0,)
_INNER_SIGMA_N_GRID: tuple[float, ...] = (2.0,)

_MIN_DETECT_LOCS = 3
_QHAT_GATE = 0.5
# Why match _tier4.knn_estimate's additive guard exactly: w_virt=0 で仮想点を除いた
# 経路が同ヘルパの inv_sq 重みとビット一致する必要があるため（等価性テスト）。
_KNN_EPS = 1e-9


def _query_matrix(
    scans: pd.DataFrame, keys: list[Key]
) -> tuple[list[int], np.ndarray]:
    """query scans を学習鍵空間 `keys` 上の median 特徴行列へ整列（NON_DETECT 埋め）。

    整列・pivot は共有の _tier4.query_feature_stats に委譲し、ここでは未検出 NaN を
    NON_DETECT へ埋める WKNN 特徴化のみ担う。
    """
    qs = query_feature_stats(scans, keys)
    mat = np.where(np.isnan(qs.median), NON_DETECT_DBM, qs.median)
    return qs.locs, mat


class _TrainModel(NamedTuple):
    keys: list[Key]
    key_idx: dict[Key, int]
    real_mat: np.ndarray          # (n_loc, n_key) median, NON_DETECT 埋め
    real_xy: np.ndarray           # (n_loc, 2)
    real_locs: list[int]
    gps: dict[Key, object]        # 鍵 → 学習済み _GP（≥3 地点で検出された鍵のみ）
    gate: dict[Key, tuple[np.ndarray, np.ndarray]]  # 鍵 → (s_sorted, q̂_sorted)


def _build_train_model(
    train_scans: pd.DataFrame,
    location_coords: pd.DataFrame,
    *,
    length_grid: tuple[float, ...] = _DEFAULT_LENGTH_GRID,
    sigma_f_grid: tuple[float, ...] = _DEFAULT_SIGMA_F_GRID,
    sigma_n_grid: tuple[float, ...] = _DEFAULT_SIGMA_N_GRID,
) -> _TrainModel:
    stats = location_feature_stats(train_scans)
    keys: list[Key] = [tuple(c) for c in stats.mu.columns]
    key_idx = {k: j for j, k in enumerate(keys)}
    real_locs = [int(loc) for loc in stats.mu.index]

    coords = location_coords.set_index("location_p")
    real_xy = coords.loc[real_locs, ["x", "y"]].to_numpy(dtype=float)
    real_mat, _ = dense_matrix(stats.mu, keys=keys)

    s_all = np.array(
        [xy_to_arclength(float(coords.loc[loc, "x"]), float(coords.loc[loc, "y"]))
         for loc in real_locs],
        dtype=float,
    )
    order = np.argsort(s_all, kind="stable")
    s_sorted = s_all[order]

    gps: dict[Key, object] = {}
    gate: dict[Key, tuple[np.ndarray, np.ndarray]] = {}
    for key in keys:
        nd = stats.n_detect[key].to_numpy(dtype=float)
        det = nd > 0
        if int(det.sum()) < _MIN_DETECT_LOCS:
            continue
        y = stats.mu[key].to_numpy(dtype=float)
        gp = _fit_gp(
            s_all[det], y[det],
            length_grid=length_grid,
            sigma_f_grid=sigma_f_grid,
            sigma_n_grid=sigma_n_grid,
        )
        gps[key] = gp
        # Why gate on q̂ over ALL train locations (not only detected ones): 未検出
        # 地点の低 q̂ こそ AP の検出フットプリント境界を表す。GP μ は検出領域で
        # 外挿を続けるため、境界外の仮想点を NON_DETECT に落とす gate が必要。
        q = stats.qhat[key].to_numpy(dtype=float)[order]
        gate[key] = (s_sorted, q)

    return _TrainModel(
        keys=keys, key_idx=key_idx, real_mat=real_mat, real_xy=real_xy,
        real_locs=real_locs, gps=gps, gate=gate,
    )


def _virtual_refs(
    model: _TrainModel, delta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """間隔 Δ の s グリッド上に仮想参照 (特徴行列, xy, s) を合成する。"""
    # Why linspace over np.arange: 端点 0 と _TOTAL_LENGTH を厳密に含み、s > 116 の
    # はみ出しも取りこぼしも生じない（arange は浮動小数の累積で両端を壊す）。
    n_pts = int(round(_TOTAL_LENGTH / delta)) + 1
    s_grid = np.linspace(0.0, _TOTAL_LENGTH, n_pts)
    virt_xy = np.array([arclength_to_xy(float(s)) for s in s_grid], dtype=float)

    virt = np.full((len(s_grid), len(model.keys)), NON_DETECT_DBM, dtype=float)
    for key, gp in model.gps.items():
        mu, _ = _gp_posterior(gp, s_grid)
        s_sorted, q_sorted = model.gate[key]
        q = np.interp(s_grid, s_sorted, q_sorted)
        virt[:, model.key_idx[key]] = np.where(q < _QHAT_GATE, NON_DETECT_DBM, mu)
    return virt, virt_xy, s_grid


def _augmented_estimate(
    q_vec: np.ndarray,
    real_mat: np.ndarray,
    real_xy: np.ndarray,
    virt_mat: np.ndarray,
    virt_xy: np.ndarray,
    k: int,
    w_virt: float,
) -> tuple[float, float]:
    """実測＋仮想の合同集合への L2 近傍を重み w·w_virt^{is_virtual} で加重平均する。"""
    if virt_mat.shape[0] == 0:
        ref_mat, ref_xy = real_mat, real_xy
        factor = np.ones(real_mat.shape[0])
    else:
        ref_mat = np.vstack([real_mat, virt_mat])
        ref_xy = np.vstack([real_xy, virt_xy])
        factor = np.concatenate([
            np.ones(real_mat.shape[0]),
            np.full(virt_mat.shape[0], w_virt),
        ])

    dists = np.linalg.norm(ref_mat - q_vec, axis=1)
    order = np.argsort(dists, kind="stable")[:k]
    w = factor[order] / (dists[order] + _KNN_EPS) ** 2
    total = w.sum()
    # Why a uniform fallback on zero total weight: 近傍の係数が全て 0（例: w_virt=0
    #   で近傍が全て仮想点）だと w/w.sum() が 0/0=NaN になる。等重み centroid へ
    #   退避して有限値を保証する（この分岐は防御的で、通常経路では total>0）。
    w = np.full(len(w), 1.0 / len(w)) if total <= 0.0 else w / total
    xy = ref_xy[order]
    return float(w @ xy[:, 0]), float(w @ xy[:, 1])


def _select_hyperparams(
    train_scans: pd.DataFrame, location_coords: pd.DataFrame
) -> tuple[tuple[float, float, int], dict]:
    """inner CV の pooled 平均 L2 argmin で (Δ, w_virt, k) を選ぶ。"""
    coords = location_coords.set_index("location_p")
    candidates = [
        (delta, w_virt, k)
        for delta in DELTA_GRID
        for w_virt in WVIRT_GRID
        for k in K_GRID
    ]
    errs: dict[tuple[float, float, int], list[float]] = {c: [] for c in candidates}

    for inner_train, inner_val in iter_inner_cv(train_scans, k=5, seed=RANDOM_SEED):
        # Why fit GPs once per fold and reuse across (Δ, w_virt, k): GP fit は候補
        # ハイパーに依存しない前処理で、候補ごとに再学習しても同一結果になる。
        # inner_train のみから構築するのでリークは生じない。Why the reduced GP grid
        # here (not the full one used in fit): see _INNER_*_GRID (F2 コスト削減)。
        model = _build_train_model(
            inner_train, location_coords,
            length_grid=_INNER_LENGTH_GRID,
            sigma_f_grid=_INNER_SIGMA_F_GRID,
            sigma_n_grid=_INNER_SIGMA_N_GRID,
        )
        q_locs, q_mat = _query_matrix(inner_val, model.keys)
        truth = np.array(
            [[coords.loc[loc, "x"], coords.loc[loc, "y"]] for loc in q_locs],
            dtype=float,
        )
        for delta in DELTA_GRID:
            virt_mat, virt_xy, _ = _virtual_refs(model, delta)
            for w_virt in WVIRT_GRID:
                for k in K_GRID:
                    for i in range(len(q_locs)):
                        x, y = _augmented_estimate(
                            q_mat[i], model.real_mat, model.real_xy,
                            virt_mat, virt_xy, k, w_virt,
                        )
                        errs[(delta, w_virt, k)].append(
                            float(np.hypot(x - truth[i, 0], y - truth[i, 1]))
                        )

    scores = {c: (float(np.mean(errs[c])) if errs[c] else float("inf")) for c in candidates}
    # Why tie-break by candidate index: Δ 昇順 → w_virt 昇順 → k 昇順の走査順で
    # 最初の（最小 Δ・最小 w_virt・最小 k）を残す決定的タイブレーク。
    best = min(range(len(candidates)), key=lambda i: (scores[candidates[i]], i))
    return candidates[best], scores


@register
class GpAugmentedWknn(Method):
    name = "gp_augmented_wknn"
    uses_geometry = False

    def __init__(
        self,
        *,
        delta: float | None = None,
        w_virt: float | None = None,
        k: int | None = None,
    ) -> None:
        # Why all-or-none, raising on partial specs (F11, JointFp と同契約):
        # 部分指定は「残りをどのグリッドから選ぶか」が未定義で、黙って CV に
        # 落ちると指定が無視される。三つ揃った場合のみ CV をスキップする。
        given = (delta is not None, w_virt is not None, k is not None)
        if any(given) and not all(given):
            raise ValueError("delta/w_virt/k must be given together or all omitted")
        self.delta = delta
        self.w_virt = w_virt
        self.k = k

        self.selected_delta: float | None = None
        self.selected_w_virt: float | None = None
        self.selected_k: int | None = None
        self.diagnostics_: dict = {}

        self._model: _TrainModel | None = None
        self._keys: list[Key] | None = None
        self._real_locs: list[int] | None = None
        self._virtual_matrix: np.ndarray = np.empty((0, 0))
        self._virtual_xy: np.ndarray = np.empty((0, 2))
        self._virtual_s: np.ndarray = np.empty((0,))

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "GpAugmentedWknn":
        # Why not use ap_coords: uses_geometry=False — 廊下弧長と (ap_name, band)
        # 指紋のみで動く。ap_coords は共通シグネチャ充足のためだけに受け取り無視。
        del ap_coords

        # Why reset every virtual array at fit entry (F8): 同一 instance を別の
        #   train で再 fit したとき、今回 selected_w_virt=0 なら下の populate 分岐が
        #   走らず、前回 fit の仮想点が残留して古い radio map で予測してしまう。
        self._virtual_matrix = np.empty((0, 0))
        self._virtual_xy = np.empty((0, 2))
        self._virtual_s = np.empty((0,))

        if self.delta is not None and self.w_virt is not None and self.k is not None:
            selected = (float(self.delta), float(self.w_virt), int(self.k))
            scores: dict = {}
        else:
            selected, scores = _select_hyperparams(train_scans, location_coords)
        self.selected_delta, self.selected_w_virt, self.selected_k = selected

        model = _build_train_model(train_scans, location_coords)
        self._model = model
        self._keys = model.keys
        self._real_locs = model.real_locs

        # Why not append virtual points when w_virt == 0: w_virt=0 は「仮想無効化
        # フラグ」であり零重み係数ではない。零重みでも仮想点が KNN の近傍枠を占め、
        # 実測近傍を静かに押し出して素の WKNN と一致しなくなる。合同集合から
        # 完全に除外して初めて素の WKNN と等価になる。
        if self.selected_w_virt > 0.0:
            virt, virt_xy, s_grid = _virtual_refs(model, self.selected_delta)
            self._virtual_matrix = virt
            self._virtual_xy = virt_xy
            self._virtual_s = s_grid

        # Why not record wall-clock timings (F10): perf_counter は実行環境ごとに
        #   ぶれ、diagnostics_ の決定性を壊す。所要計測は外部（scripts の診断計測）へ
        #   委ね、ここには再現可能な構造量のみ残す。
        self.diagnostics_ = {
            "selected_delta": self.selected_delta,
            "selected_w_virt": self.selected_w_virt,
            "selected_k": self.selected_k,
            "cv_scores": scores,
            "n_gp_keys": len(model.gps),
            "n_real_refs": len(model.real_locs),
            "n_virtual_refs": int(self._virtual_xy.shape[0]),
        }
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        if self._model is None or self.selected_k is None:
            raise RuntimeError("fit() must be called before predict()")

        q_locs, q_mat = _query_matrix(test_scans, self._keys)
        rows = []
        for i, loc in enumerate(q_locs):
            x, y = _augmented_estimate(
                q_mat[i], self._model.real_mat, self._model.real_xy,
                self._virtual_matrix, self._virtual_xy,
                self.selected_k, float(self.selected_w_virt),
            )
            rows.append({"location_p": int(loc), "x": x, "y": y})
        return pd.DataFrame(rows)
