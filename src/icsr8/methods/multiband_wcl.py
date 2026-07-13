"""周波数帯分離 + Multi-Band Fusion（手法6, doc/mid_report/main.tex §3.1）。

フリスの伝達公式より 2.4/5/6 GHz 帯は経路損失が最大 ~20·log10(f/f0) dB 異なるため、
帯域を混ぜたまま単一の WCL に投入すると重みが歪む。本手法は帯域ごとに独立な
対数距離減衰モデル rssi = P0 - 10α·log10(d) を学習データから最小二乗フィットし、
各帯の WCL を train 地点に対し実行して得た位置 MSE の逆数（w_b = 1/max(mse_b, 0.01)）を
融合重みとして帯域別 WCL 推定を統合する。dB 残差 σ_b は同じ dB 誤差でも帯ごとに位置
誤差が異なり位置不確かさの代理として不適なため、診断値としてのみ保持する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from icsr8.constants import SIGMA_MIN_DB
from icsr8.estimators import estimate_wcl
from icsr8.fingerprint import (
    ap_band_fingerprint,
    band_of,
    candidate_medians,
    reproduction_fingerprint,
)
from icsr8.methods import register
from icsr8.methods.base import Method


def _fuse_band_estimates(
    band_estimates: dict[str, tuple[float, float]],
    band_weights: dict[str, float],
) -> tuple[float, float] | None:
    """逆分散重み付き平均。重みの合計が 0 以下なら融合不能として None を返す。"""
    wsum = sum(band_weights.get(band, 0.0) for band in band_estimates)
    if wsum <= 0:
        return None
    x = sum(band_weights.get(b, 0.0) * xy[0] for b, xy in band_estimates.items()) / wsum
    y = sum(band_weights.get(b, 0.0) * xy[1] for b, xy in band_estimates.items()) / wsum
    return x, y


def _band_position_mse(
    train_scans: pd.DataFrame, ap_coords: pd.DataFrame, location_coords: pd.DataFrame
) -> tuple[dict[str, float], dict[str, float]]:
    """各 band の WCL 位置 MSE（train 上、リークなし）と融合重み 1/max(mse,0.01)。

    各帯の WCL を train 地点に対し実行し、真値との二乗L2誤差の平均を重みの基礎に
    する。dB 残差 σ_b は帯ごとに勾配・幾何が異なるため位置不確かさの代理として不適。
    使える train 地点が 3 未満の帯は MSE 推定が不安定なので重み 0（融合から除外）。
    """
    candidates = candidate_medians(train_scans, ap_coords)
    candidates = candidates.assign(band=candidates["frequency"].map(band_of))
    truth = location_coords.set_index("location_p")

    per_band_sqerr: dict[str, list[float]] = {}
    for band, band_cand in candidates.groupby("band"):
        for loc_p, group in band_cand.groupby("location_p"):
            if loc_p not in truth.index:
                continue
            band_fp = reproduction_fingerprint(group)
            if len(band_fp) < 3:
                continue
            est = estimate_wcl(band_fp)
            ex, ey = float(est["x"].iloc[0]), float(est["y"].iloc[0])
            tx, ty = float(truth.loc[loc_p, "x"]), float(truth.loc[loc_p, "y"])
            per_band_sqerr.setdefault(str(band), []).append((ex - tx) ** 2 + (ey - ty) ** 2)

    band_mse: dict[str, float] = {}
    band_weights: dict[str, float] = {}
    for band, sq in per_band_sqerr.items():
        band_mse[band] = float(np.mean(sq))
        # Why weight 0 for <3 usable train locations: 過少標本の MSE は不安定で、
        #   逆数を取ると誤って過大な重みを生む。
        band_weights[band] = 1.0 / max(band_mse[band], 0.01) if len(sq) >= 3 else 0.0
    return band_mse, band_weights


@register
class MultibandWcl(Method):
    name = "multiband_wcl"
    uses_geometry = True

    def __init__(self) -> None:
        self._ap_coords: pd.DataFrame | None = None
        # Why not underscore-prefix these: fallback_count/path_loss_fits/band_sigma/
        #   band_mse/band_weights are report-facing state (cf. WCLTopL.selected_L), not
        #   pure plumbing, so they stay publicly readable after fit()/predict().
        self.path_loss_fits: dict[tuple[str, str], dict[str, float]] = {}
        self.band_sigma: dict[str, float] = {}
        self.band_mse: dict[str, float] = {}
        self.band_weights: dict[str, float] = {}
        self.fallback_count = 0

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "MultibandWcl":
        self._ap_coords = ap_coords

        known_ap_names = set(ap_coords["ap_name"])
        ab_fp = ap_band_fingerprint(train_scans, ap_coords=None)
        ab_fp = ab_fp[ab_fp["ap_name"].isin(known_ap_names)]

        loc_xy = location_coords[["location_p", "x", "y"]].rename(
            columns={"x": "loc_x", "y": "loc_y"}
        )
        ap_xy = ap_coords[["ap_name", "x", "y"]].rename(
            columns={"x": "ap_x", "y": "ap_y"}
        )
        merged = ab_fp.merge(loc_xy, on="location_p").merge(ap_xy, on="ap_name")
        # Why clip at 0.5 m: log10(d) diverges as d→0, but no AP-device separation
        #   in this corridor deployment is realistically sub-0.5 m.
        distance = np.hypot(merged["loc_x"] - merged["ap_x"], merged["loc_y"] - merged["ap_y"])
        merged = merged.assign(distance=distance.clip(lower=0.5))

        path_loss_fits: dict[tuple[str, str], dict[str, float]] = {}
        band_residuals: dict[str, list[np.ndarray]] = {}
        for (ap_name, band), group in merged.groupby(["ap_name", "band"]):
            if len(group) < 3:
                continue
            log_d = np.log10(group["distance"].to_numpy())
            rssi = group["rssi_median"].to_numpy()
            design = np.column_stack([np.ones_like(log_d), log_d])
            coef, *_ = np.linalg.lstsq(design, rssi, rcond=None)
            p0, neg_10_alpha = coef
            path_loss_fits[(ap_name, band)] = {
                "P0": float(p0), "alpha": float(-neg_10_alpha / 10.0),
            }
            band_residuals.setdefault(band, []).append(rssi - design @ coef)

        # Why keep σ_b after switching weights to MSE: report が σ_b と mse_b の
        #   両方を並べて比較するため、診断値として保持する（融合には使わない）。
        band_sigma: dict[str, float] = {}
        for band, residual_list in band_residuals.items():
            pooled = np.concatenate(residual_list)
            band_sigma[band] = max(float(np.std(pooled, ddof=0)), SIGMA_MIN_DB)

        band_mse, band_weights = _band_position_mse(train_scans, ap_coords, location_coords)

        self.path_loss_fits = path_loss_fits
        self.band_sigma = band_sigma
        self.band_mse = band_mse
        self.band_weights = band_weights
        self.fallback_count = 0
        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        if self._ap_coords is None:
            raise RuntimeError("fit() must be called before predict()")

        candidates = candidate_medians(test_scans, self._ap_coords)
        candidates = candidates.assign(band=candidates["frequency"].map(band_of))

        self.fallback_count = 0
        rows = []
        for loc_p, loc_candidates in candidates.groupby("location_p", sort=True):
            band_estimates: dict[str, tuple[float, float]] = {}
            for band in sorted(loc_candidates["band"].unique()):
                band_fp = reproduction_fingerprint(
                    loc_candidates[loc_candidates["band"] == band]
                )
                if len(band_fp) >= 3:
                    est = estimate_wcl(band_fp)
                    band_estimates[band] = (float(est["x"].iloc[0]), float(est["y"].iloc[0]))

            fused = _fuse_band_estimates(band_estimates, self.band_weights)
            if fused is None:
                # Why-not distinguish "no band had >=3 candidates" from "every
                #   producing band had weight 0" (no training fit): both mean no
                #   usable weighted estimate exists, so both fall back identically
                #   to pooled 13-AP baseline WCL.
                fp = reproduction_fingerprint(loc_candidates)
                est = estimate_wcl(fp)
                x, y = float(est["x"].iloc[0]), float(est["y"].iloc[0])
                self.fallback_count += 1
            else:
                x, y = fused

            rows.append({"location_p": int(loc_p), "x": x, "y": y})

        return pd.DataFrame(rows)
