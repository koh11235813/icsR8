"""WCL with variance + detection-count confidence weighting.

分散と検出率による信頼度を考慮したWCL改良。不安定なAP（高分散、低検出率）
の重みを下方修正する。手法11（doc/improvement_methods_note.txt）。

重み関数:
  w = 10^((rssi_median − rssi_min_top3)/10)
      / (1 + (sigma_q/sigma_ref)²)
      * min(n_detect/10, 1.0)

  - sigma_q: クエリ点での、reproduction_fingerprint が残した勝者 variant の
             (ap_name, band) 単位の RSSI標準偏差
  - n_detect: 同じ (ap_name, band) の 10 scan中検出数
  - sigma_ref: 学習データ全体の ap_band_fingerprint から rssi_std の中央値、
              SIGMA_MIN_DB でフロア処理

Why-not (物理AP全体で pooling する):
  band 間の中央値差は安定したオフセット（例: 2.4G と 5G で数十dB）であり、
  これを pooled std に混ぜると「時間的な不安定さ」として誤検出してしまう。
  同様に他 band の検出数を pooled n_detect に足すと、未検出の band が検出済み
  に見えてしまう。band 単位で取ることでこの漏れ込みを防ぐ（F1）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from icsr8.constants import SIGMA_MIN_DB
from icsr8.estimators import _require_three, select_top_k
from icsr8.fingerprint import ap_band_fingerprint, band_of, candidate_medians, reproduction_fingerprint
from icsr8.methods import register
from icsr8.methods.base import Method


@register
class WclVarweight(Method):
    name = "wcl_varweight"
    uses_geometry = True

    def __init__(self) -> None:
        self._ap_coords: pd.DataFrame | None = None
        self._sigma_ref: float | None = None

    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "WclVarweight":
        """σ_ref を学習データから計算する。

        Args:
            train_scans: [location_p, ssid, rssi, frequency, count, ap_name, direction]
            ap_coords: [ap_name, x, y, ...]
            location_coords: [location_p, x, y] of training locations only

        Stores:
            _sigma_ref: median(rssi_std) from ap_band_fingerprint, floored at SIGMA_MIN_DB
        """
        self._ap_coords = ap_coords

        # Compute ap_band_fingerprint (groups by location_p, ap_name, band)
        ab_fp = ap_band_fingerprint(train_scans, ap_coords=None)

        # Extract rssi_std column and compute median, floored at SIGMA_MIN_DB
        sigma_vals = ab_fp["rssi_std"].to_numpy()
        sigma_median = float(np.median(sigma_vals))
        self._sigma_ref = max(sigma_median, SIGMA_MIN_DB)

        return self

    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        """Apply variance + detection-count weighting to WCL.

        Args:
            test_scans: Raw RSSI measurements at test locations

        Returns:
            DataFrame with columns [location_p, x, y]
        """
        if self._ap_coords is None or self._sigma_ref is None:
            raise RuntimeError("fit() must be called before predict()")

        # Get reproduction fingerprint (AP-level, one row per physical AP per location)
        cands = candidate_medians(test_scans, self._ap_coords)
        fp = reproduction_fingerprint(cands)

        # Why: σ_q/n_detect を (ap_name, band) 単位で引くために、テストデータ
        # 全体を一度だけ ap_band_fingerprint に通しておく（F1）。
        ab_fp = ap_band_fingerprint(test_scans, ap_coords=None)

        # Estimate per location
        rows = []
        for loc_p, group in fp.groupby("location_p", sort=True):
            # Why: baseline WCL は _require_three で <3 候補を弾くが、この派生
            # 手法は同じ選抜元 (top-3) を使う以上、同じ契約を守らないと
            # 1-2候補で黙って推定してしまう（F2）。
            _require_three(group, "wcl_varweight")

            # Select top-3 APs (same as baseline WCL)
            top = select_top_k(group, 3)

            # Baseline WCL weights
            rssi_min_top3 = top["rssi_median"].min()
            w_base = np.power(10.0, (top["rssi_median"].to_numpy() - rssi_min_top3) / 10.0)

            loc_ab_fp = ab_fp[ab_fp["location_p"] == loc_p]

            # Apply variance and detection-count adjustments for each top AP
            w_adj = []
            for i, ap_row in enumerate(top.itertuples()):
                # Why: fp の frequency は reproduction_fingerprint が残した勝者
                # variant のものなので、band もそれ由来で決める。物理AP全体を
                # pooling すると別 band の分散/検出数が漏れ込む（F1）。
                band = band_of(ap_row.frequency)
                match = loc_ab_fp[
                    (loc_ab_fp["ap_name"] == ap_row.ap_name) & (loc_ab_fp["band"] == band)
                ]

                if len(match) > 0:
                    sigma_q = float(match["rssi_std"].iloc[0])
                    n_detect = int(match["n_detect"].iloc[0])
                else:
                    # Why not sigma_ref フォールバック: (ap_name, band) が
                    # 見つからないのは勝者 variant の周波数と band 分類が
                    # 食い違う理論上あり得ないケースのみ。保守的に重みゼロにする。
                    sigma_q = 0.0
                    n_detect = 0

                # Weight adjustment: w = w_base / (1 + (sigma_q/sigma_ref)²) * min(n_detect/10, 1.0)
                variance_factor = 1.0 + (sigma_q / self._sigma_ref) ** 2
                detection_factor = min(n_detect / 10.0, 1.0)
                w_adj.append(w_base[i] / variance_factor * detection_factor)

            w_adj = np.array(w_adj)
            wsum = w_adj.sum()

            x = float((w_adj * top["x"].to_numpy()).sum() / wsum)
            y = float((w_adj * top["y"].to_numpy()).sum() / wsum)

            rows.append({"location_p": int(loc_p), "x": x, "y": y})

        return pd.DataFrame(rows)
