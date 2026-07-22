# ADR 0001: 本文成果物の凍結と評価系統の隔離

- Status: Accepted
- Date: 2026-07-13（凍結）/ 2026-07-14（ガード整備）/ 2026-07-22（追試系統の追加）

## Context

最終報告書の本文は Tier 1–3 の 15 手法で結論を述べる。本文の表・CSV・図が
後続の実験（Tier 4 以降の追加手法）によって書き換わると、本文プロース・
統計的主張（CI・有意性）と成果物が乖離し、報告の自己整合が壊れる。

実際に 2026-07-14、`scripts/run_all_methods.py` を `--methods` 無指定で再実行した際、
レジストリの自動探索により登録済み Tier 4 の 7 手法が掃引対象に入り、
本文の凍結成果物 5 ファイルへ混入する事故が発生した（コミット前に検出・復元済み）。

## Decision

1. 以下の 6 ファイルを**凍結成果物**とする:
   `results/{protocol_a,lolo_ledger,lolo_summary}.csv`、
   `doc/final_report/tables/{lolo,protocol_a}.tex`、
   `doc/final_report/figures/cdf_lolo.pdf`
2. 追加手法の評価は**別系統に隔離**する: Tier 4 は `results/tier4/` +
   `tier4_*` 固有名、追試は `results/extra/` 等の専用ディレクトリ
   （`run_tier4.py` の `--output/--tables-dir/--figures-dir` 3 点差し替え）
3. 隔離ハーネス（`icsr8.harness_tier4.run_tier4`）は凍結 6 ファイルへの書き込みを
   `ValueError` で拒否し、契約をテスト
   `test_run_tier4_refuses_frozen_output` で固定する
4. 本文 15 手法の再生成は `--methods` 明示のみ許可。パイプラインは決定的
   （seed=0・決定的 tie-break）であり、正しい再生成は byte 一致で検証できる
5. 文書との整合は `scripts/verify_report.py`（CSV↔表 TeX の byte 照合 +
   `main.tex` 参照パス実在検査）で機械的に検証する

## Consequences

- 良: 本文の統計的主張と成果物の対応が構造的に保護され、追加実験を
  いつでも安全に行える（vWCL 追試は本決定に従い `results/extra/` で実施）
- 良: 「再生成 = byte 一致」という強い検証手段が使える
- 悪: `run_all_methods.py` の無指定実行という自然な操作が地雷として残る
  （ガードは tier4 ハーネス側のみ）。運用ルールで補っており、CLAUDE.md と
  README に警告を明記している
- 悪: 新手法追加のたびに出力先 3 点の指定が必要になり、コマンドが長い
