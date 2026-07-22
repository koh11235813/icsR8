"""Ji 2012 vWCL: 境界効果対策の virtual AP 補正 WCL（付録実験用・本文非掲載）。

Ji, Cho, Kim, Lee, Park, "Improving the Positioning Accuracy using Virtual
Access Points in the Border Area" (IPIN 2012) の vWCL を凍結ベースライン WCL の
上に適用する:

  1. 凍結 WCL と同一の top-3 選択・重み w=10^((r-r_min)/10) で初期推定 P を得る
  2. 実 AP 座標を P について点対称に反転した仮想 AP を生成する
     （vAP_j = 2P - rAP_j。原論文の仮定に従い RSSI・重みは実 AP と同値）
  3. 実 AP 凸包の厳密内部に落ちた仮想 AP は棄却する
     （中心への引き込みを再増悪させるため。境界上は「内部でない」扱い）
  4. 実 + 採用仮想 AP の重み付き重心で P を更新し、収束まで 2-4 を反復する
     （原論文 V 節はシミュレーションで 5-10 回収束と報告。本データでは最大
     50 回強を要する地点があるため防御的上限 100 回・許容誤差 1e-9 m とする）

全仮想 AP が採用された場合、重み対称性 Σw(2P-AP) + ΣwAP = 2PΣw により更新は
不動点となり WCL と一致する。棄却が起きる場合、重みを実 AP から継承する本適用
では更新式が「棄却された実 AP の重み付き重心」と現在点の凸結合に簡約されるため、
推定は実 AP 凸包内（境界を含む）に留まったまま棄却元 AP（典型的には最強 AP）側へ
収縮する（凸包の外へは出ない）。細長い廊下型の top-3 凸包では最強 AP の反転像のみが
棄却されやすく、この収縮は WCL の空間平均を失わせる方向に働く。

学習データを一切使わない（wcl と同じ情報クラス）ため fit は ap_coords を保持
するのみで、リークは構造的に生じない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from icsr8.estimators import select_top_k
from icsr8.fingerprint import candidate_medians, reproduction_fingerprint
from icsr8.methods import register
from icsr8.methods.base import Method

# Why 100 (not the paper's 5-10): 実データ 59 地点中 30 以上で 10 回超・最大 50 回強を
# 要した。契約は「収束まで」であり、100 は観測最大の約 2 倍の防御的上限。
MAX_ITER = 100
TOL = 1e-9
_EPS = 1e-9


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain。CCW の頂点列を返す（共線・重複は退化として縮む）。"""
    uniq = np.unique(pts, axis=0)
    if len(uniq) <= 2:
        return uniq
    order = np.lexsort((uniq[:, 1], uniq[:, 0]))
    p = uniq[order]

    def _cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for pt in p:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    upper: list[np.ndarray] = []
    for pt in p[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)
    return np.asarray(lower[:-1] + upper[:-1])


def _strictly_inside(queries: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """各 query が hull の厳密内部にあるか。退化 hull（頂点<3）は常に False。"""
    if len(hull) < 3:
        return np.zeros(len(queries), dtype=bool)
    a = hull
    b = np.roll(hull, -1, axis=0)
    edge = b - a
    rel = queries[:, None, :] - a[None, :, :]
    cross = edge[None, :, 0] * rel[:, :, 1] - edge[None, :, 1] * rel[:, :, 0]
    return (cross > _EPS).all(axis=1)


def vwcl_point(
    pts: np.ndarray,
    weights: np.ndarray,
    *,
    max_iter: int = MAX_ITER,
    tol: float = TOL,
) -> tuple[float, float]:
    """実 AP 座標と重みから vWCL 推定点を返す純関数（テスト対象の核）。"""
    pts = np.asarray(pts, dtype=float)
    weights = np.asarray(weights, dtype=float)
    wsum = weights.sum()
    p = (weights[:, None] * pts).sum(axis=0) / wsum
    hull = _convex_hull(pts)
    for _ in range(max_iter):
        vpts = 2.0 * p - pts
        keep = ~_strictly_inside(vpts, hull)
        all_pts = np.vstack([pts, vpts[keep]])
        all_w = np.concatenate([weights, weights[keep]])
        p_new = (all_w[:, None] * all_pts).sum(axis=0) / all_w.sum()
        if float(np.hypot(*(p_new - p))) < tol:
            p = p_new
            break
        p = p_new
    return float(p[0]), float(p[1])


def _vwcl_one(fp: pd.DataFrame, **kw) -> tuple[float, float]:
    if len(fp) < 3:
        loc = fp["location_p"].iloc[0] if len(fp) else "?"
        raise ValueError(
            "WCL_VIRTUAL_AP requires 3 candidates per location; "
            f"location_p={loc} has only {len(fp)}"
        )
    top = select_top_k(fp, 3, **kw)
    rssi_min = top["rssi_median"].min()
    weights = np.power(10.0, (top["rssi_median"].to_numpy() - rssi_min) / 10.0)
    pts = top[["x", "y"]].to_numpy(dtype=float)
    return vwcl_point(pts, weights)


def estimate_vwcl(fp: pd.DataFrame, **kw) -> pd.DataFrame:
    rows = []
    for loc, group in fp.groupby("location_p", sort=True):
        x, y = _vwcl_one(group, **kw)
        rows.append({"location_p": int(loc), "x": x, "y": y})
    return pd.DataFrame(rows)


@register
class WCLVirtualAP(Method):
    name = "wcl_virtual_ap"
    uses_geometry = True

    def __init__(self, **kwargs) -> None:
        # Why not accept kwargs: 基底 Method の統一シグネチャに合わせるためのみ。
        # 本手法は追加パラメータを持たない（max_iter/tol はモジュール定数）。
        pass

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "WCLVirtualAP":
        # Why not consume location_coords: ベースラインと同様、学習を持たないため
        # 参照点座標を必要としない。
        self._ap_coords = ap_coords
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        fp = reproduction_fingerprint(candidate_medians(test_scans, self._ap_coords))
        return estimate_vwcl(fp)
