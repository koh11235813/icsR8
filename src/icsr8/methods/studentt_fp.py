"""確率的 fingerprinting（Student-t 尤度 + Bernoulli 検出 + 共通オフセット除去 + 分散重み）。

doc/improvement_methods_note.txt 手法2 + 手法4 + 手法11 = doc/mid_report/main.tex §3.3。

各 train 地点 l と鍵 k=(ap_name, band) について、RSSI を Student-t（外れ値に頑健）、
AP 検出/非検出を Bernoulli でモデル化し、query の posterior 平均で座標を推定する:
    log p(l|q) = Σ_k [ D_k·log q̂_{l,k} + β_k·1[k∈R]·log t_ν(r_k; μ_{l,k}+δ̂, σ_{l,k})
                       + (1-D_k)·log(1-q̂_{l,k}) ]
- q̂_{l,k} = (n_detect+1)/(10+2)  … Beta(1,1) 平滑化。未検出鍵は 1/12（手法2）。
- δ̂ = median_{k∈R}(r_k − μ_{l,k})  … 端末 AGC・身体遮蔽等の scan 全体オフセット（手法4）。
- β_k = 1/(1+(σ̄_k/σ_ref)²)  … 分散が大きい鍵を下方重み付けする信頼度（手法11）。
- t 項は n_detect ≥ MIN_COUNT の eligible 鍵のみ（μ/σ が不安定な鍵は Bernoulli 項のみ残す）。
- ν ∈ {3,5,10} を location 単位 5-fold inner CV の pooled L2 argmin で選択。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.special import gammaln

from icsr8.constants import MIN_COUNT, RANDOM_SEED, SIGMA_MIN_DB
from icsr8.fingerprint import ap_band_fingerprint
from icsr8.methods import register
from icsr8.methods.base import Method
from icsr8.protocols import iter_inner_cv

NU_GRID: tuple[int, ...] = (3, 5, 10)

# Beta(1,1) 平滑化: q̂ = (n_detect + 1) / (10 + 2)。10 = 1 地点あたりの scan 数。
_N_SCANS = 10
_BETA_DENOM = _N_SCANS + 2

Key = tuple[str, str]


class _Model(NamedTuple):
    keys: list[Key]        # 学習鍵の和集合（sorted (ap_name, band)）
    locs: list[int]        # 学習地点（sorted location_p）
    xy: np.ndarray         # (n_loc, 2) 学習地点の実測座標
    q: np.ndarray          # (n_loc, n_key) 検出確率 q̂
    log_q: np.ndarray      # (n_loc, n_key) log q̂
    log_1mq: np.ndarray    # (n_loc, n_key) log(1 - q̂)
    mu: np.ndarray         # (n_loc, n_key) RSSI 中央値、非 eligible は NaN
    sigma: np.ndarray      # (n_loc, n_key) RSSI 標準偏差（SIGMA_MIN_DB でフロア）、非 eligible は NaN
    elig: np.ndarray       # (n_loc, n_key) bool: n_detect >= MIN_COUNT
    beta: np.ndarray       # (n_key,) 分散ベース信頼度重み


def _log_t_density(x, df, loc, scale):
    """Student-t 対数密度の閉形式。scipy.stats.t.logpdf(x, df, loc, scale) に一致する。

    Why not exp() 経由: posterior は log 領域で合算し、softmax の直前まで exp を
    取らない契約（数値安定性）。gammaln + log1p で桁落ちを避ける。
    """
    z = (x - loc) / scale
    log_norm = gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0) - 0.5 * np.log(df * np.pi)
    return log_norm - np.log(scale) - ((df + 1.0) / 2.0) * np.log1p(z * z / df)


def _softmax(logp: np.ndarray) -> np.ndarray:
    e = np.exp(logp - logp.max())
    return e / e.sum()


def _build_model(train_scans: pd.DataFrame, location_coords: pd.DataFrame) -> _Model:
    """ν 非依存の DB（q̂/μ/σ/eligibility/β）を学習データから構築する。"""
    ab = ap_band_fingerprint(train_scans, ap_coords=None)
    keys = sorted(set(zip(ab["ap_name"], ab["band"])))
    locs = sorted(int(loc) for loc in train_scans["location_p"].unique())
    col_index = pd.MultiIndex.from_tuples(keys)

    def _pivot(value: str) -> np.ndarray:
        p = ab.pivot_table(index="location_p", columns=["ap_name", "band"], values=value)
        return p.reindex(index=locs, columns=col_index).to_numpy(dtype=float)

    n_det = np.nan_to_num(_pivot("n_detect"), nan=0.0)  # 未観測鍵は n_detect=0
    mu = _pivot("rssi_median")                          # 未観測は NaN
    sigma_raw = _pivot("rssi_std")

    # q̂: Beta(1,1) 平滑化。未観測(n_detect=0)→1/12、全検出(n_detect=10)→11/12。
    q = (n_det + 1.0) / _BETA_DENOM
    elig = n_det >= MIN_COUNT
    # t 項に使うのは eligible 鍵のみ。σ を SIGMA_MIN_DB でフロアし、非 eligible は NaN。
    sigma = np.where(elig, np.maximum(sigma_raw, SIGMA_MIN_DB), np.nan)
    mu = np.where(elig, mu, np.nan)

    # β_k = 1/(1+(σ̄_k/σ_ref)²)。σ̄_k は key ごとの eligible σ 中央値、
    # σ_ref は全 eligible σ の中央値（SIGMA_MIN_DB フロアは冗長だが belt-and-braces）。
    beta = np.zeros(len(keys))
    elig_sigma = sigma[elig]
    if elig_sigma.size:
        sigma_ref = max(float(np.median(elig_sigma)), SIGMA_MIN_DB)
        for j in range(len(keys)):
            col = sigma[elig[:, j], j]
            if col.size:
                sigma_bar = float(np.median(col))
                beta[j] = 1.0 / (1.0 + (sigma_bar / sigma_ref) ** 2)
    # eligible 地点を持たない鍵は β=0 のまま。scoring では R（eligible∩検出）に
    # 入らないので β は参照されず、0 埋めは万一の NaN 混入を防ぐ防御。

    xy = location_coords.set_index("location_p").loc[locs, ["x", "y"]].to_numpy(dtype=float)
    return _Model(keys, locs, xy, q, np.log(q), np.log1p(-q), mu, sigma, elig, beta)


def _query_vector(model: _Model, group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """query 1 地点の (検出フラグ d, RSSI r) を学習鍵空間上に整列する。"""
    key_to_idx = {k: j for j, k in enumerate(model.keys)}
    d = np.zeros(len(model.keys), dtype=bool)
    r = np.zeros(len(model.keys), dtype=float)
    for ap, band, med in zip(group["ap_name"], group["band"], group["rssi_median"]):
        j = key_to_idx.get((ap, band))
        # Why: 学習鍵の和集合に無い query 鍵は μ/σ/q̂ のモデルを持たないので無視する。
        if j is not None:
            d[j] = True
            r[j] = med
    return d, r


def _score(model: _Model, d: np.ndarray, r: np.ndarray, nu: float) -> np.ndarray:
    """query (d, r) に対する全学習地点の非正規化対数事後 logp_l を返す。"""
    # Bernoulli 項は全鍵の和: 検出→log q̂、非検出→log(1-q̂)（δ 非依存なので一括計算）。
    logp = np.where(d, model.log_q, model.log_1mq).sum(axis=1)
    for i in range(len(model.locs)):
        rmask = d & model.elig[i]  # R = D_q ∩ eligible@l
        if not rmask.any():
            continue
        mu_i = model.mu[i, rmask]
        delta = float(np.median(r[rmask] - mu_i))  # 共通オフセット δ̂（R が空なら 0＝この分岐に来ない）
        # Why not weighting the Bernoulli term too: doc/mid_report/main.tex §3.3 の
        # log p(s|q) = Σ β_{a,b}[D·log t_ν(...) + log p(D|s)] は β を角括弧全体、
        # すなわち t 項と Bernoulli 項の両方に掛ける形で書かれている。だがここでは
        # β を連続 RSSI 証拠（t 項）にのみ適用する。検出/非検出は count evidence
        # であり、その信頼度は既に Beta(1,1) 平滑化（q̂）で表現済み。RSSI 分散ベース
        # の β をさらに Bernoulli 項へ掛けると、検出回数由来の信頼度と RSSI 分散
        # 由来の信頼度という単位の異なる量を混同する。最終レポートではこの
        # main.tex からの仕様変更を明記する。
        t_terms = model.beta[rmask] * _log_t_density(r[rmask], nu, mu_i + delta, model.sigma[i, rmask])
        logp[i] += t_terms.sum()
    return logp


def _select_nu(train_scans: pd.DataFrame, location_coords: pd.DataFrame) -> int:
    coords = location_coords.set_index("location_p")
    errors: dict[int, list[float]] = {nu: [] for nu in NU_GRID}

    for inner_train, inner_val in iter_inner_cv(train_scans, k=5, seed=RANDOM_SEED):
        model = _build_model(inner_train, location_coords)
        ab_val = ap_band_fingerprint(inner_val, ap_coords=None)
        for loc_p, group in ab_val.groupby("location_p", sort=True):
            d, r = _query_vector(model, group)
            tx, ty = coords.loc[loc_p, ["x", "y"]]
            for nu in NU_GRID:
                post = _softmax(_score(model, d, r, nu))
                ex = float(post @ model.xy[:, 0])
                ey = float(post @ model.xy[:, 1])
                errors[nu].append(float(np.hypot(ex - tx, ey - ty)))

    mean_err = {nu: float(np.mean(errors[nu])) for nu in NU_GRID}
    # Why: pooled L2 argmin。同点は abs(ν-5) で中央値 5 を優先（契約）、
    # 3-vs-10 の同点（両者とも 5 より良い）は小さい ν（重い裾）を採る。
    return min(NU_GRID, key=lambda nu: (mean_err[nu], abs(nu - 5), nu))


@register
class StudentTFP(Method):
    name = "studentt_fp"
    uses_geometry = False

    def __init__(self, *, nu: int | None = None) -> None:
        self.nu = nu
        self.selected_nu: int | None = None
        self._model: _Model | None = None
        self.last_logp: dict[int, np.ndarray] = {}
        self.last_posterior: dict[int, np.ndarray] = {}
        self.last_map_locations: dict[int, int] = {}

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "StudentTFP":
        # Why not use ap_coords: uses_geometry=False — 鍵 (ap_name, band) の識別子と
        # 指紋統計のみで動く純指紋手法。共通シグネチャを満たすため受け取り無視する。
        del ap_coords
        if self.nu is not None:
            self.selected_nu = int(self.nu)
        else:
            self.selected_nu = _select_nu(train_scans, location_coords)
        self._model = _build_model(train_scans, location_coords)
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        if self._model is None or self.selected_nu is None:
            raise RuntimeError("fit() must be called before predict()")

        model = self._model
        ab = ap_band_fingerprint(test_scans, ap_coords=None)
        rows = []
        self.last_logp = {}
        self.last_posterior = {}
        self.last_map_locations = {}
        for loc_p, group in ab.groupby("location_p", sort=True):
            d, r = _query_vector(model, group)
            logp = _score(model, d, r, self.selected_nu)
            post = _softmax(logp)
            x = float(post @ model.xy[:, 0])
            y = float(post @ model.xy[:, 1])
            rows.append({"location_p": int(loc_p), "x": x, "y": y})
            # posterior 平均で座標推定（Why not MAP: 2 m グリッドへ量子化してしまう）。
            # MAP 地点は診断用に保存する。
            self.last_logp[int(loc_p)] = logp
            self.last_posterior[int(loc_p)] = post
            self.last_map_locations[int(loc_p)] = int(model.locs[int(np.argmax(logp))])
        return pd.DataFrame(rows)
