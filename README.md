# icsR8

豊橋技術科学大学 学内無線 LAN (tutwifi / tutwifi2025) の RSSI を用いた
屋内位置推定の研究プロジェクト。

本ライブラリ (`icsr8`) は基準方式 (PBL / CLA / WCL) を生 RSSI データから
再計算して `doc/icsR8_text.txt` Table 1 の公表ベースライン値を再現し、
その凍結ベースライン上に改善手法 19+1 種 (Tier 1–4 + 追試) を実装・評価する。
主提案手法 `gp_corridor` (廊下弧長 1D Gaussian Process radio map) は
LOLO 評価で平均誤差 **0.72 m** (≤2 m 率 90%) を達成し、目標の 2 m 未満をクリアした。

## リポジトリ構造

```
.
├── src/icsr8/            推定ライブラリ本体（詳細は「モジュール構成」）
│   └── methods/          手法レジストリ（1 ファイル = 1 手法）
├── scripts/              評価・検証・再生成の CLI
├── tests/                pytest（374 テスト。公表値再現・リーク契約・凍結ガード含む）
├── data/                 測定データ（*.zip が原本。展開ディレクトリは直接編集しない）
├── doc/
│   ├── final_report/     最終報告書 (LuaLaTeX)。tables/ と figures/ は生成物
│   ├── slides/           発表スライド (Beamer) + ナレーション台本 narration.md
│   ├── mid_report/       中間報告書
│   └── pdf/              課題ガイダンス・課題テキスト等の配布資料
└── results/
    ├── *.csv             本文 Tier 1–3（15 手法）の凍結成果物
    ├── tier4/            Tier 4（付録 A、7 手法）の隔離成果物
    └── extra/            追試（vWCL 等）の隔離成果物
```

## セットアップ

```bash
uv sync --all-groups
```

## 凍結契約（最重要）

本文の比較基準を守るため、以下の 6 ファイルは**凍結**されている
（根拠テスト: `tests/test_harness_tier4.py::test_run_tier4_refuses_frozen_output`）:

- `results/protocol_a.csv` / `results/lolo_ledger.csv` / `results/lolo_summary.csv`
- `doc/final_report/tables/protocol_a.tex` / `doc/final_report/tables/lolo.tex`
- `doc/final_report/figures/cdf_lolo.pdf`

手編集は禁止。再生成する場合も**本文 15 手法を明示した**下記コマンドのみを使う。
評価パイプラインは決定的（seed 固定）なので、正しく再生成すれば HEAD と byte 一致する。

## Tier ごとの評価手順

### 本文 Tier 1–3（15 手法・凍結成果物の再生成）

```bash
uv run python scripts/run_all_methods.py --methods \
  centered_fp,cla,gp_corridor,multiband_wcl,pbl,rank_fp,studentt_fp,wcl,wcl_blacklist,wcl_corridor,wcl_linpower,wcl_powerdomain,wcl_topl,wcl_varweight,wknn
```

> **警告**: `--methods` を省略してはならない。レジストリは `src/icsr8/methods/` を
> 自動探索するため、省略すると登録済みの Tier 4 以降の手法まで掃引し、
> 凍結成果物（本文の表・CSV・図）を上書きしてしまう（2026-07-14 に実際に発生した事故。
> 詳細は `docs/adr/0001-freeze-main-body-artifacts.md`）。

### Tier 4（付録 A・隔離評価）

```bash
uv run python scripts/run_tier4.py            # 既定: TIER4_METHODS の 7 手法
uv run python scripts/run_tier4.py --smoke    # 代役手法+9地点で1分以内の動作確認
```

出力は `results/tier4/`・`doc/final_report/tables/tier4_*.tex`・
`doc/final_report/figures/cdf_lolo_tier4.pdf` に隔離される。
凍結 6 ファイルへの書き込みはハーネス側のガードが `ValueError` で拒否する。
参照手法 (`wcl`, `gp_corridor`) が自動で併走し、`delta_vs_*` 列と 95% CI が付く。

### 追試・新手法（例: vWCL）

```bash
uv run python scripts/run_tier4.py --methods wcl_virtual_ap \
  --output results/extra --tables-dir results/extra --figures-dir results/extra
```

> **警告**: `--tables-dir` / `--figures-dir` も必ず専用ディレクトリへ差し替えること。
> 省略すると既定の `doc/final_report/` 配下に書き、コミット済みの
> `tier4_*.tex`（付録 A の表）を少数手法版で上書きしてしまう。

### 診断値・検証ゲート

```bash
uv run python scripts/dump_method_diagnostics.py   # results/method_diagnostics.csv 再生成
uv run pytest                                      # 374 テスト
uv run python scripts/verify_report.py             # 表数値・診断値・TeX参照パスの整合検証
```

`verify_report.py` は CSV から表 TeX 断片を byte 単位で再構成して照合し、
`main.tex` の `\input` / `\includegraphics` 参照パスの実在も検査する。
**コード・結果・文書のどれを変更した後でも、この 2 つのゲートを必ず通すこと。**

## 新手法の追加手順

1. `src/icsr8/methods/<name>.py` を新規作成（1 ファイル = 1 手法）。
   `Method` を継承し `@register` を付け、`name` と `uses_geometry`
   （AP 座標を幾何的に消費するか）を宣言する
2. `fit(train_scans, ap_coords, location_coords)` は **train の情報のみ**を使う。
   リークは `run_method`（`src/icsr8/methods/__init__.py`）が構造的に防止し、
   spy テスト `test_iter_lolo_leakage_contract_spy` が契約を検証している
3. `tests/test_<name>.py` を追加（性質テスト + `run_method` 経由の e2e）
4. 上記「追試・新手法」の隔離コマンドで評価（凍結成果物には触れない）
5. `uv run pytest` と `scripts/verify_report.py` の両ゲートを通す

## ドキュメントビルド

```bash
(cd doc/final_report && latexmk -lualatex main.tex)   # 最終報告書（2段組・6部構成）
(cd doc/slides       && latexmk -lualatex main.tex)   # 発表スライド（Beamer）
```

`*.pdf` は gitignore されておりコミットされない（表 TeX 断片と CSV が正）。
スライドのナレーション台本は `doc/slides/narration.md`（約 10 分配分付き）。

## 使い方（ライブラリ API）

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

任意の登録手法は統一エントリで実行できる:

```python
from icsr8.methods import run_method, available_methods
est = run_method("gp_corridor", train_scans, test_scans, ap, truth)
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
  io.py             CSV ローダ (BOM 対応、相対パス禁止)
  fingerprint.py    candidate 集約 + 再現用前処理（wing フィルタ・物理AP集約）
  estimators.py     PBL / CLA / WCL + select_top_k（凍結。編集禁止）
  evaluate.py       L2 誤差 + summary
  corridor.py       廊下弧長 (arc-length) 変換
  protocols.py      Protocol A / LOLO の分割 iterator（リーク構造防止の要）
  harness.py        本文評価ハーネス（CSV・図・表 TeX の一括生成）
  harness_tier4.py  隔離評価ハーネス（凍結ガード・参照手法併走・delta CI）
  plotting.py       CDF / ヒートマップ描画
  constants.py      seed・ブラックリスト AP 等の定数
  types.py          Direction, Candidate, Estimate
  methods/          手法レジストリ（@register で自動登録）
    base.py           Method 抽象基底（fit/predict、uses_geometry）
    baselines.py      pbl / cla / wcl（凍結推定器へのアダプタ）
    wknn.py gp_corridor.py studentt_fp.py          Tier 1
    centered_fp.py rank_fp.py                      Tier 2
    corridor_proj.py multiband_wcl.py wcl_*.py     Tier 3（WCL 改良系）
    fisher_wknn.py mahalanobis_wknn.py pls_corridor.py
    ordinal_corridor.py wcl_residual.py joint_fp.py
    gp_augmented_wknn.py                           Tier 4（付録 A）
    wcl_virtual_ap.py                              追試（Ji 2012 vWCL、results/extra）
scripts/
  run_all_methods.py           本文 15 手法の評価（--methods 必須）
  run_tier4.py                 隔離評価 CLI（Tier 4 / 追試）
  verify_report.py             表数値・診断値・TeX 参照の整合検証ゲート
  dump_method_diagnostics.py   本文引用の診断値 CSV 再生成
  extract_baseline_fixtures.py テスト用 Oracle CSV 再生成
```
