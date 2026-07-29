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
├── tests/                pytest（509 テスト。公表値再現・リーク契約・凍結ガード含む）
├── data/                 測定データ（*.zip が原本。展開ディレクトリは直接編集しない）
├── doc/
│   ├── final_report/     最終報告書 (LuaLaTeX)。tables/ と figures/ は生成物
│   ├── slides/           発表スライド (Beamer) + ナレーション台本 narration.md
│   ├── mid_report/       中間報告書
│   └── pdf/              課題ガイダンス・課題テキスト等の配布資料
└── results/              CSV は git 管理下（ULP drift 有り）。追試 extra/ のみ gitignore
    ├── *.csv             本文 Tier 1–3 の生成 CSV（git 管理、ULP drift 有り）
    ├── tier4/            Tier 4（付録 A、7 手法）の成果物（CSV は git 管理、ULP drift 有り）
    └── extra/            追試（vWCL 等）の隔離成果物（CSV は再生成物・gitignore）
```

## セットアップ

```bash
uv sync --all-groups
```

## 凍結契約（最重要）

本文・付録 A の比較基準を守るため、以下の**表 TeX 4 本 + 図 PDF 5 本 = 9 ファイル**
（main body 6 + 付録 A 3）は**凍結**されている
（根拠: `src/icsr8/harness_tier4.FROZEN_OUTPUT_PATHS`。
テスト: `tests/test_harness_tier4.py::test_frozen_output_paths_is_nine_files` /
`test_run_tier4_refuses_frozen_output`）:

main body 6:
- `doc/final_report/tables/protocol_a.tex` / `doc/final_report/tables/lolo.tex`
- `doc/final_report/figures/cdf_lolo.pdf`
- `doc/final_report/figures/cdf_protocol_a_forward_to_backward.pdf`
- `doc/final_report/figures/cdf_protocol_a_backward_to_forward.pdf`
- `doc/final_report/figures/segment_heatmap.pdf`

付録 A 3:
- `doc/final_report/tables/tier4_protocol_a.tex` / `doc/final_report/tables/tier4_lolo.tex`
- `doc/final_report/figures/cdf_lolo_tier4.pdf`

**sanctioned writer 契約**: これらのパスへ書けるのは `icsr8.report.regenerate_main_body()` /
`icsr8.report.regenerate_appendix_a()`（`src/icsr8/harness_tier4._guard_frozen` の
allowlist に登録された 2 関数）だけ。他のあらゆる呼び出し元（追試用
`scripts/run_experimental_tier4.py`・hand-edit・ad-hoc スクリプト）は
`ValueError` で構造的に reject される — 「凍結パスへの書き込みは blocklist で
塞ぐ」のではなく「そもそも sanctioned writer 以外は書けない」という allowlist
契約（詳細: `docs/adr/0004-deep-module-freeze-invariant.md`）。

手編集は禁止。再生成する場合も `scripts/regenerate_main_body.py` /
`scripts/regenerate_appendix_a.py`（いずれも引数ゼロ）のみを使う。
表 TeX は `%.2f` を通すため OS 中立で HEAD と byte 一致する想定、
図 PDF は視覚差なし前提で `076bec5` 以降 git 管理下。

`results/*.csv` は本文値の元になる中間 CSV で git 管理下にあるが、
Mac(Accelerate) と Linux(OpenBLAS) の BLAS 実装差で ULP レベルにドリフトするため
byte 一致契約の対象外である。報告書本文の数値は `%.2f` を通すのでこの drift の
影響を受けない。歴史的経緯は `docs/adr/0001-freeze-main-body-artifacts.md`
（Superseded — 現行契約は `docs/adr/0004-deep-module-freeze-invariant.md`）。

## Tier ごとの評価手順

### 本文 Tier 1–3（15 手法・CSV・表 TeX/図 PDF・診断値の再生成、引数ゼロ）

`results/*.csv` は git 管理下にあるため、フレッシュクローンでも生成済みの状態で
存在する。CSV を再生成したい場合のみ、以下のコマンドを回す
（凍結対象の表 TeX と図 PDF・`results/method_diagnostics.csv` も同時に再生される。
HEAD と byte 一致するはず）。

```bash
uv run python scripts/regenerate_main_body.py
```

このコマンドは引数を取らない。本文 15 手法・出力先のいずれも選べない設計
そのものが、2026-07-14 に実際に発生した contamination incident（`--methods`
省略でレジストリを自動探索し Tier 4 手法まで巻き込んだ事故。詳細は
`docs/adr/0001-freeze-main-body-artifacts.md`）の再発を構造的に防ぐ
（sanctioned writer として `src/icsr8/report.py` に実装。README §凍結契約参照）。

### Tier 4（付録 A・隔離評価、引数ゼロ）

```bash
uv run python scripts/regenerate_appendix_a.py
```

出力は `results/tier4/`・`doc/final_report/tables/tier4_*.tex`・
`doc/final_report/figures/cdf_lolo_tier4.pdf` に隔離される。TIER4_METHODS の
7 手法固定・引数ゼロ。参照手法 (`wcl`, `gp_corridor`) が自動で併走し、
`delta_vs_*` 列と 95% CI が付く。

### 追試・新手法（例: vWCL）

```bash
uv run python scripts/run_experimental_tier4.py --methods wcl_virtual_ap \
  --output results/extra/vwcl --tables-dir results/extra/vwcl --figures-dir results/extra/vwcl
uv run python scripts/run_experimental_tier4.py --methods wcl_virtual_ap --smoke \
  --output /tmp/smoke --tables-dir /tmp/smoke --figures-dir /tmp/smoke   # 配管確認用
```

`--methods` / `--output` / `--tables-dir` / `--figures-dir` は全て required
（省略は argparse エラーで即座に失敗する）。この CLI は sanctioned writer では
ない（`icsr8.report.regenerate_*` を経由しない）ため、`--tables-dir` /
`--figures-dir` に凍結ディレクトリを指定すると `_guard_frozen` が
`ValueError` で reject する。

### 検証ゲート

```bash
uv run pytest                                      # 509 テスト
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

本文 PDF・図 PDF は `076bec5 add: pdf contents` 以降 git 管理下で、
ビルド済み成果物が最終の版管理対象となる（`git pull` で他マシンのビルド結果を取得可能）。
スライドのナレーション台本は `doc/slides/narration.md`（約 10 分配分付き）。

## Google Colab

icsR8 を Google Drive に置いて Colab から実行する手順は
`docs/COLAB.md` にまとめてある（セットアップセル・結果生成・TeXLive
ビルド・成果物の書き戻し）。アーキテクチャ上の決定は
`docs/adr/0003-colab-bootstrap-isolation.md` を参照。

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
  harness.py        本文評価ハーネス（CSV・図・表 TeX の一括生成。sweep 本体の純関数群）
  harness_tier4.py  隔離評価ハーネス（sweep 本体・delta CI・_guard_frozen allowlist）
  report.py         再生成 API（sanctioned writer）。regenerate_main_body() /
                     regenerate_appendix_a()（共に引数ゼロ）。harness.py /
                     harness_tier4.py を組み立てる上位モジュール
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
  regenerate_main_body.py      本文 15 手法 + 診断値の再生成（sanctioned writer、引数ゼロ）
  regenerate_appendix_a.py     付録 A（Tier 4）7 手法の再生成（sanctioned writer、引数ゼロ）
  run_experimental_tier4.py    追試・新手法専用 CLI（--methods/--output/--tables-dir/
                                --figures-dir 全 required。凍結パスは書けない）
  verify_report.py             表数値・診断値・TeX 参照の整合検証ゲート
  extract_baseline_fixtures.py テスト用 Oracle CSV 再生成
```
