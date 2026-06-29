# icsR8

豊橋技術科学大学 学内無線 LAN (tutwifi / tutwifi2025) の RSSI を用いた
屋内位置推定の研究プロジェクト。

本ライブラリ (`icsr8`) は基準方式 (PBL / CLA / WCL) を生 RSSI データから
再計算し、`doc/icsR8_text.txt` Table 1 の公表ベースライン値を再現する。
改善手法 (WKNN, 確率的 FP, GP radio map 等) は別途実装予定。

## セットアップ

```bash
uv sync --all-groups
```

## 使い方

```python
from icsr8 import (
    load_ap_coords, load_location_coords, load_raw_scans,
    candidate_medians, reproduction_fingerprint,
    estimate_pbl, estimate_cla, estimate_wcl,
    l2_errors, summary,
)

ap = load_ap_coords("data/dataset/AP_coordinate_C3F.csv")
truth = load_location_coords("data/dataset/location_coordinate_C.csv")[["location_p", "x", "y"]]
scans = load_raw_scans("forward", "data/rawdata")

fp = reproduction_fingerprint(candidate_medians(scans, ap))
err = l2_errors(estimate_wcl(fp), truth)
print(summary(err["error"]))
# → {"Ave": 3.5685, "Max": 11.85, "Min": 0.469, "Std": 2.423, "Var": 5.871}
```

## テスト

```bash
uv run pytest
```

公表値再現テストは `tests/test_reproduce_baseline.py` に集中している。
Oracle CSV は `tests/fixtures/` にコミット済み。再生成する場合は:

```bash
uv run python scripts/extract_baseline_fixtures.py
```

## 公表ベースライン再現の前提

仕様書 (`doc/icsR8_text.txt` §3.1) を literal に実装してもベースラインは
再現できない。`icsr8` は以下の暗黙の前処理を `reproduction_fingerprint` /
`select_top_k` に明示的に encode している:

1. **3F-AP 既知座標** (`AP_coordinate_C3F.csv` の 13 件) のみ候補化。
2. **C 棟群 (`C0` / `C2` / `C3`) のみ採用**。`AP-C1-3F-*` (C1 棟) は除外。
   これは仕様書に明記されていないが、`estimation_result_C3F.xlsx` の P1 CLA
   が AP-C0-3F-01/02/03 の centroid (20.0, 0.3) になることから判明した運用。
3. **物理 AP 単位で集約**: (SSID, frequency) バリアントから最強の rssi_median
   を取り、1 物理 AP につき 1 行に正規化。
4. **Tie-break**: rssi_median 降順 → frequency 昇順 → ssid 昇順 → ap_name 昇順。
   仕様書の "random" 指示とは異なるが、公表ベースライン 5 件の tie 事象
   (P19/P30/P35/P43/P49) を全て再現する決定的規則として採用。
5. **Std/Var は ddof=0**。

## モジュール構成

```
src/icsr8/
  io.py            CSV ローダ (BOM 対応、相対パス禁止)
  fingerprint.py   candidate 集約 + 再現用前処理
  estimators.py    PBL / CLA / WCL + select_top_k
  evaluate.py      L2 誤差 + summary
  types.py         Direction, Candidate, Estimate
```
