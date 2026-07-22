# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For architecture, project structure, command usage, and configuration reference, see README.md.

PLAN.md and MEMO.md are ephemeral scratch space for jotting in-progress plans/memos; durable decisions belong in CONTEXT.md, this file, or docs/adr/.

## Write a program based on the Unix philosophy

- Write programs that do one thing and do it well.
- Write programs to work together.
- Write programs to handle text streams, because that is a universal interface.

## Interaction contract

- If requirements are ambiguous or underspecified, stop and ask 1–3 targeted questions before proceeding.
- Before making any irreversible change (deletes, migrations, dependency upgrades, infra changes), ask for explicit confirmation.
- Never assume environment details (OS, shell, package manager, project conventions). Ask or infer only from repo evidence.
- Start each task by restating: Goal, Non-goals, Constraints, Success criteria (brief).
- When multiple approaches exist, present 2 options with tradeoffs, then ask which to take.

## Comment & Context Policy

Write comments generously — treat them as first-class documentation, not noise. Current AI models benefit from heavy inline context; so do human readers six months later.

In addition to comments, write function and class descriptions and intentions in docstring.

- Always comment intent, not mechanics. Explain why a block exists, what invariant it protects, or what would break without it. Don't restate what the code does — explain what it means.
- Record fix provenance inline. When code exists because of a specific bug or incident, leave a dated note: `# 2026-05-12 crash fix: bare ifconfig omits netmask → classful /8 on Class A`. This is the kind of context that git blame buries and developers lose.
- Keep context close to code. A comment explaining a constraint belongs next to the line it constrains, not in a separate design doc. If someone reads the function, they should see the warning without leaving the file.
- Don't write comments that rot. Avoid referencing ticket numbers, PR links, or caller names ("used by X") — those change. Describe the constraint the code enforces; that outlives the ticket.

## Notice

Separate functions into separate files by type, and do not recreate existing functions in the execution script. If you need to edit them, edit the existing function and check that the modifications have been made. Make functions as flexible as possible by using variables.

# Development

このリポジトリを変更する際の注意点。いずれも実際に事故った・事故りかけた経緯から成文化している（経緯の詳細は docs/adr/）。

## 1. 凍結契約 — 触ってはいけない 6 ファイル

`results/{protocol_a,lolo_ledger,lolo_summary}.csv`・`doc/final_report/tables/{lolo,protocol_a}.tex`・`doc/final_report/figures/cdf_lolo.pdf` は本文の凍結成果物。**手編集・無断再生成は禁止**。再生成が必要なときは README「Tier ごとの評価手順」の本文 15 手法明示コマンドのみを使う。パイプラインは決定的（seed=0）なので、正しい再生成は HEAD と byte 一致する — 一致しなければ何かを壊している。

## 2. `run_all_methods.py` を `--methods` 無しで実行しない

# 2026-07-14 汚染事故: レジストリは `src/icsr8/methods/` を自動探索するため、`@register` 付きの手法を追加した時点で無指定実行の掃引対象が増える。無指定で実行した結果、Tier 4 の 7 手法が凍結ファイルに混入した。新手法を追加したら、この地雷は**さらに**大きくなっていることを忘れない。

## 3. 隔離評価は 3 点セットで

新手法・追試の評価は `run_tier4.py --methods <name> --output <dir> --tables-dir <dir> --figures-dir <dir>` と**出力 3 系統すべて**を専用ディレクトリ（例: `results/extra/`）へ向ける。`--tables-dir`/`--figures-dir` を省略すると既定の `doc/final_report/` 配下に書き、コミット済みの付録 A 表 `tier4_*.tex` を上書きする。

## 4. 変更後は必ず 2 つのゲートを通す

```bash
uv run pytest                           # 374 テスト（凍結ガード・リーク契約・公表値再現を含む）
uv run python scripts/verify_report.py  # CSV↔表TeX の byte 照合 + main.tex 参照パス実在検査
```

コード・結果・LaTeX のどれを触った後でも両方実行する。片方だけでは足りない（pytest は文書と表の整合を見ないし、verify_report は挙動を見ない）。

## 5. 手法追加の規約

- 1 module = 1 method（`src/icsr8/methods/<name>.py`）、`@register`、`name` と `uses_geometry` を宣言
- `fit` に test の情報を渡さない。リーク防止は `run_method`（`methods/__init__.py`）が train 地点への座標フィルタで構造的に保証しており、spy テスト `test_iter_lolo_leakage_contract_spy` が契約を固定している。この保証を迂回する直接呼び出しを書かない
- 反復アルゴリズムは「収束まで」を契約とし、上限は観測最大の余裕をもって設定して根拠をコメントに残す（# 2026-07-22 vWCL: 論文の想定 5-10 回に対し実データは最大 53 回を要した）
- 性質テスト＋`run_method` 経由の e2e テストを `tests/test_<name>.py` に追加

## 6. LaTeX の契約

- ビルドは LuaLaTeX + latexmk（`doc/*/. latexmkrc`）。報告書は `ltjsarticle[twocolumn]`、スライドは beamer + luatexja。pLaTeX 系クラス（ieicej 等）は混ぜない
- `doc/final_report/main.tex` 内の `\input{tables/...}` と `\includegraphics{...figures/...}` の**パス文字列は verify_report.py の検査対象**。セクションを再編してもパス文字列は一字も変えない
- 数値は表生成器の `%.2f` 出力を `\input` するのが原則。本文プロースに数値を書くときは表・CSV と厳密一致させ、独自の丸め直しをしない

## 7. 公開リポジトリのプレースホルダ

`[GROUP_NAME]` / `[AUTHOR_NAME]` は公開リポジトリ用のプレースホルダ。実名・グループ名を埋めた状態でコミットしない（提出用 PDF はローカルで埋めてビルドする）。

## 8. 再現性を壊さない

seed=0・bootstrap B=1000・決定的 tie-break（rssi_median 降順 → frequency 昇順 → ssid 昇順 → ap_name 昇順。公表値の tie 事象 P19/P30/P35/P43/P49 を再現する規則）は結果の同一性の根幹。これらを変更する提案は、公表値再現テストが壊れることを意味する。

## 9. データの原本

`data/*.zip` が原本。展開ディレクトリ（`data/dataset/` 等）を直接編集しない。fixtures の再生成は `scripts/extract_baseline_fixtures.py` 経由で行う。
