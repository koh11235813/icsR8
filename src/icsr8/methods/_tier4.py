"""Tier 4 手法群が共有する純ユーティリティ。

特徴統計 (μ/σ/q̂/n)・密行列化・重み付き kNN・inner CV ハイパー選択・
弧長クリップを 1 か所に集約し、各手法での複製を防ぐ。Method サブクラスを
含まないため methods の自動探索はこの module を無害に通過する。
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, cho_factor

from icsr8.constants import MIN_COUNT, NON_DETECT_DBM, RANDOM_SEED, SIGMA_MIN_DB
from icsr8.corridor import _TOTAL_LENGTH
from icsr8.fingerprint import ap_band_fingerprint
from icsr8.protocols import iter_inner_cv

# Why not 1/max(d,eps): 加算ガードは全点で連続な重みを与え、複数のゼロ距離が
#   あっても同点として均等按分できる（max ガードは 1 点へ全質量を寄せる）。
_KNN_EPS = 1e-9

Key = tuple[str, str]


class FeatureStats(NamedTuple):
    mu: pd.DataFrame
    sigma: pd.DataFrame
    qhat: pd.DataFrame
    n_detect: pd.DataFrame


def location_feature_stats(scans: pd.DataFrame) -> FeatureStats:
    """地点 × (ap_name, band) の μ/σ/q̂/n を DataFrame 群として返す。"""
    ab = ap_band_fingerprint(scans, ap_coords=None)
    keys = sorted(set(zip(ab["ap_name"], ab["band"])))
    col_index = pd.MultiIndex.from_tuples(keys, names=["ap_name", "band"])
    locs = sorted(int(loc) for loc in scans["location_p"].unique())

    def _pivot(value: str) -> pd.DataFrame:
        p = ab.pivot_table(
            index="location_p", columns=["ap_name", "band"], values=value
        )
        out = p.reindex(index=locs, columns=col_index)
        out.index.name = "location_p"
        return out

    n_detect = _pivot("n_detect").fillna(0.0).astype(int)
    mu = _pivot("rssi_median")
    sigma_raw = _pivot("rssi_std")

    # q̂ = (n_detect+1)/(n_scans+2)。n_scans は地点ごとの distinct scan 数
    # （studentt_fp の Beta(1,1) 平滑化を地点別分母へ一般化）。
    n_scans = (
        scans.groupby("location_p")["count"].nunique().reindex(locs).to_numpy(float)
    )
    qhat = pd.DataFrame(
        (n_detect.to_numpy(float) + 1.0) / (n_scans[:, None] + 2.0),
        index=mu.index,
        columns=col_index,
    )

    elig = n_detect.to_numpy() >= MIN_COUNT
    sigma = pd.DataFrame(
        np.where(elig, np.maximum(sigma_raw.to_numpy(float), SIGMA_MIN_DB), np.nan),
        index=mu.index,
        columns=col_index,
    )
    return FeatureStats(mu=mu, sigma=sigma, qhat=qhat, n_detect=n_detect)


class QueryStats(NamedTuple):
    locs: list[int]
    median: np.ndarray  # (n_loc, n_key) rssi_median, NaN preserved for undetected
    n_detect: np.ndarray  # (n_loc, n_key) int detection counts
    n_scans: np.ndarray  # (n_loc,) distinct scan count per location


def query_feature_stats(scans: pd.DataFrame, keys: list[Key]) -> QueryStats:
    """query scans を train 鍵空間 `keys` へ整列した median / n_detect / n_scans。

    joint_fp と gp_augmented_wknn が共有する唯一の query 特徴経路（各自の
    ap_band_fingerprint→pivot→整列 の再実装を廃す）。median は未検出鍵で NaN を
    保持し、NON_DETECT 埋め（両手法）や Beta 平滑化検出率（joint_fp）の扱いは
    各手法が自分で選ぶ。

    Why not route through location_feature_stats: それは train 側統計（μ/σ/q̂）の
    構築関数で、inner CV の leak-spy が「train 前処理呼び出しだけ」を捕捉できるよう
    query 経路とは別名にしておく必要がある（validation fold を混入させない）。
    """
    ab = ap_band_fingerprint(scans, ap_coords=None)
    col_index = pd.MultiIndex.from_tuples(keys, names=["ap_name", "band"])
    locs = sorted(int(loc) for loc in scans["location_p"].unique())

    med = ab.pivot_table(
        index="location_p", columns=["ap_name", "band"], values="rssi_median"
    ).reindex(index=locs, columns=col_index)
    nd = (
        ab.pivot_table(
            index="location_p", columns=["ap_name", "band"], values="n_detect"
        )
        .reindex(index=locs, columns=col_index)
        .fillna(0.0)
    )
    n_scans = (
        scans.groupby("location_p")["count"].nunique().reindex(locs).to_numpy(float)
    )
    return QueryStats(
        locs=locs,
        median=med.to_numpy(dtype=float),
        n_detect=nd.to_numpy(dtype=int),
        n_scans=n_scans,
    )


def dense_matrix(
    mu: pd.DataFrame,
    keys: list[Key] | None = None,
    fill: float = NON_DETECT_DBM,
) -> tuple[np.ndarray, list[Key]]:
    """μ を列整列した密行列へ。未検出 (NaN) は `fill` で埋める。"""
    if keys is None:
        keys = list(mu.columns)
    arr = mu.reindex(columns=keys).to_numpy(dtype=float)
    return np.where(np.isnan(arr), fill, arr), keys


def knn_estimate(
    dists: np.ndarray,
    ref_xy: np.ndarray,
    k: int,
    weighting: str,
) -> tuple[float, float]:
    """距離ベクトルから最近傍 k 点の重み付き座標平均を返す。"""
    dists = np.asarray(dists, dtype=float)
    ref_xy = np.asarray(ref_xy, dtype=float)
    order = np.argsort(dists, kind="stable")[:k]
    d = dists[order]
    xy = ref_xy[order]

    if weighting == "uniform":
        w = np.ones(len(d))
    elif weighting == "inv":
        w = 1.0 / (d + _KNN_EPS)
    elif weighting == "inv_sq":
        w = 1.0 / (d + _KNN_EPS) ** 2
    else:
        raise ValueError(f"unknown weighting: {weighting!r}")

    w = w / w.sum()
    return float(w @ xy[:, 0]), float(w @ xy[:, 1])


def select_by_inner_cv(
    train_scans: pd.DataFrame,
    location_coords: pd.DataFrame,
    candidates: list,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, object], pd.DataFrame],
    k: int = 5,
) -> tuple[object, dict]:
    """inner CV の pooled 平均 L2 誤差で候補を選ぶ。タイは候補先頭優先。"""
    coords = location_coords.set_index("location_p")
    # Why not dict keyed by candidate: 集計中は index で貯め、candidate の
    #   hashability に依存しない。scores 構築時のみ candidate を key にする。
    errs: list[list[float]] = [[] for _ in candidates]

    for inner_train, inner_val in iter_inner_cv(
        train_scans, k=k, seed=RANDOM_SEED
    ):
        train_locs = inner_train["location_p"].unique()
        inner_train_coords = location_coords[
            location_coords["location_p"].isin(train_locs)
        ]
        for i, cand in enumerate(candidates):
            pred = fit_predict(inner_train, inner_val, inner_train_coords, cand)
            for row in pred.itertuples(index=False):
                tx, ty = coords.loc[row.location_p, ["x", "y"]]
                errs[i].append(float(np.hypot(row.x - tx, row.y - ty)))

    mean_err = [float(np.mean(e)) if e else float("inf") for e in errs]
    best_i = min(range(len(candidates)), key=lambda i: (mean_err[i], i))
    scores = {candidates[i]: mean_err[i] for i in range(len(candidates))}
    return candidates[best_i], scores


def safe_cho_factor(sigma: np.ndarray, max_tries: int = 8):
    """対称化 + trace-scaled 対角 jitter を有界回数だけ増やしつつ Cholesky する。

    まず jitter=0 で試し、SPD なら摂動なしで返す（well-conditioned 行列はビット
    完全）。失敗したら対角へ trace(Σ)/p 相対の jitter を幾何級数で加えて再試行し、
    縮退（全特徴一定で Σ̂=0 等）でも SPD へ持ち上げる。

    Why trace-relative (not absolute) jitter: 特徴スケールに依存せず相対的に同じ
    だけ対角を持ち上げる。Σ̂=0 の完全縮退では trace=0 になるので、その場合のみ
    絶対スケール 1.0 へ切り替える。
    """
    a = np.asarray(sigma, dtype=float)
    a = 0.5 * (a + a.T)
    p = a.shape[0]
    trace = float(np.trace(a))
    scale = trace / p if trace > 0.0 else 1.0

    jitter = 0.0
    for i in range(max_tries):
        try:
            return cho_factor(a + jitter * np.eye(p))
        except LinAlgError:
            jitter = scale * 10.0 ** (i - max_tries + 4)
    raise LinAlgError(
        f"safe_cho_factor: matrix not SPD after {max_tries} jitter retries"
    )


def clip_arclength(s):
    """弧長を [0, 廊下全長] にクリップする（scalar/array 両対応）。"""
    clipped = np.clip(s, 0.0, _TOTAL_LENGTH)
    if np.ndim(s) == 0:
        return float(clipped)
    return clipped
