# ADR 0004: Deep module 化と allowlist 凍結契約

- Status: Accepted
- Date: 2026-07-29

## Context

**(i) `run_all_methods.py` の命名 footgun**: 2026-07-14、`scripts/run_all_methods.py`
を `--methods` 無指定で再実行した際、レジストリの自動探索により登録済み Tier 4
の 7 手法が掃引対象に入り、本文の凍結成果物へ混入する事故が発生した
（ADR-0001 参照）。名前は「全部走らせろ」だが、正しい使い方は「必ず
`--methods` で 15 個列挙する」——名前と挙動が乖離した CLI を、凍結ガードという
external な blocklist で patch していた。ガードは効いていたが、事故の温床
そのもの（「正しい呼び方を覚えておく」という運用ルール）は残り続けた。

**(ii) CSV drift の cross-OS 問題**: `results/*.csv` は Mac(Accelerate) と
Linux(OpenBLAS) の BLAS 実装差で ULP レベルにドリフトする。2026-07-23 に
一度 CSV を byte 一致契約から外して gitignore の再生成物としたが、これは
Colab 上の notebook validation のたびに毎回 8 分 + 45〜60 分 + 12 秒の
再生成を要求する結果になり、検証の敷居を上げていた。2026-07-29 の Commit 1
で CSV を git 管理に戻し（drift は `%.2f` の手前で吸収されるため実害が無い
ことを明記）、この問題自体は解消済み。本 ADR はその上に積む deep module 化を
記録する。

**(iii) 深いモジュール化の設計思想**: (i) の根本原因は「狭いインターフェースの
裏に十分な実装を隠せていない」ことにある。`run_all_methods.py` は
`--methods`・`--output`・`--tables-dir`・`--figures-dir` という 4 つの選択肢を
呼び出し側に開放していた。選択肢が開いている限り、「正しい選び方を運用ルールで
徹底する」以外の防御線が作れない。選択肢そのものを閉じてしまえば
（本文 15 手法・出力先を固定した引数ゼロ関数にすれば）、間違った呼び方を
する余地自体が消える。

## Decision

**Freeze invariant の再定式化**（1 行契約）:

> `main.tex` から `\input` / `\includegraphics` で参照される凍結対象ファイルは、
> `icsr8.report.regenerate_main_body()` および `icsr8.report.regenerate_appendix_a()`
> **のみ**が書ける（sanctioned writer）。`_guard_frozen()` は blocklist ではなく
> allowlist として実装され、これら 2 関数以外からの書き込み
> （`run_experimental_tier4.py`、hand-edit、ad-hoc スクリプト）は全て
> `ValueError` で reject する。

具体的な実装:

1. `src/icsr8/report.py`（新規）に `MAIN_BODY_METHODS`（15 手法・順序込み
   literal）/ `APPENDIX_A_METHODS`（`harness_tier4.TIER4_METHODS` の
   re-export）と、`regenerate_main_body()` / `regenerate_appendix_a()`
   （共に完全引数ゼロ）を実装する。出力先は repo root からの固定パスで、
   呼び出し側が選ぶ余地を持たない。
2. `harness_tier4._guard_frozen(targets, *, writer_id=None)` を allowlist 化。
   `writer_id` が `_SANCTIONED_WRITERS = {"icsr8.report.regenerate_main_body",
   "icsr8.report.regenerate_appendix_a"}` に含まれなければ、凍結パスへの
   書き込みを構造的に reject する。`writer_id=None`（デフォルト）は常に
   非 sanctioned 扱い。
3. `FROZEN_OUTPUT_PATHS` を 9 ファイル（main body 6 + 付録 A 3）に拡張する。
   旧 blocklist は `run_tier4()` 経由の書き込みしか守れず、`tier4_*.tex` /
   `cdf_lolo_tier4.pdf`（付録 A 分）は一度もこのリストに含まれたことがなく
   構造的に無防御だった。allowlist 化と同時にこの穴を塞ぐ。
4. 旧 `scripts/{run_all_methods,run_tier4,dump_method_diagnostics}.py` を削除し、
   `scripts/regenerate_main_body.py` / `scripts/regenerate_appendix_a.py`
   （引数ゼロの薄い CLI）と `scripts/run_experimental_tier4.py`
   （`--methods`/`--output`/`--tables-dir`/`--figures-dir` 全 required、
   sanctioned writer ではない）に置き換える。
5. **（2026-07-29 Codex review finding 1 対応）** `_guard_frozen` の呼び出しを
   `harness.make_figures` / `harness.make_tex_tables` /
   `harness_tier4.make_figures_tier4` / `harness_tier4.make_tex_tables_tier4`
   という実際に書き込みを行う低レベル関数自身の冒頭へ押し下げる。導入当初は
   `report.regenerate_main_body()` が呼び出し側で `harness.make_figures` /
   `make_tex_tables` を包んでガードしていたが、これらの関数は
   `harness.py` を直接 import して呼ぶ経路には無防備だった
   （`harness_tier4.run_tier4()` の冒頭ガードは同様に維持しつつ、
   `make_figures_tier4` / `make_tex_tables_tier4` にも同じガードを追加した
   — belt-and-suspenders）。各関数は `writer_id: str | None = None`
   キーワード引数を受け取り、`report.py` の 2 関数がそれぞれの
   sanctioned writer 識別子を渡す。

## Consequences

- **(i) 新 CLI 3 本の構造**: 「正しい再生成」が唯一の自然な操作になった
  （`regenerate_main_body.py` / `regenerate_appendix_a.py` は選択肢が無いので
  間違えようがない）。追試・新手法評価は `run_experimental_tier4.py` に
  一本化され、全出力チャネルが required なので省略事故が argparse レベルで
  起きなくなった。
- **(ii) 追試の footgun が完全に消滅**: 2026-07-14 のような
  「`--methods` を省略すると Tier 4 まで巻き込む」事故は、`--methods` を
  required にした時点で構造的に起こり得ない。凍結パスへの書き込みも
  writer_id ベースの allowlist で reject されるため、「正しいフラグの
  組み合わせを覚えておく」という運用ルールへの依存が消えた。
- **(iii) `verify_report.py` の CSV↔TeX byte 検査は drift 検出の safety net
  として残る**: allowlist は「誰が書けるか」を守るものであり、「書いた内容が
  正しいか」は別の関心事。`scripts/verify_report.py` の CSV↔TeX byte 照合と
  `main.tex` 参照パス実在検査は、deep module 化後も独立したゲートとして機能し
  続ける（`uv run pytest` と合わせて必ず両方通す。CLAUDE.md §3 参照）。
- 悪: `report.py` は `harness.py` / `harness_tier4.py` の両方に依存する
  上位モジュールになり、依存グラフが一段深くなった。ただし `harness.py` /
  `harness_tier4.py` 自体のテスト可能な純関数群は変更していないため、
  既存のユニットテスト資産はそのまま有効。
- **(iv) `writer_id` は Python の言語機能では偽装を防げない自己申告文字列で
  ある**（2026-07-29 Codex review finding 1 対応で判明・明記）: `_guard_frozen`
  は動的スタック検査をしない。悪意ある in-process caller が
  `writer_id="icsr8.report.regenerate_main_body"` という文字列を手で書けば、
  `harness.make_figures` / `harness.make_tex_tables` /
  `harness_tier4.make_figures_tier4` / `harness_tier4.make_tex_tables_tier4`
  を直接呼んで凍結パスへ書き込むことは Python レベルでは止められない。
  この契約が実際に守るのは (i) **意図しない書き込み**（`writer_id` を渡し忘れる
  新規 CLI・ad-hoc スクリプト・harness.py の低レベル関数を無警戒に直接呼ぶ
  コード — デフォルト `writer_id=None` は常に非 sanctioned なので、これらは
  何もしなくても reject される）と (ii) **凍結対象追加時の保護漏れ**（allowlist
  方式なので、新しい書き手が増えても `writer_id` を明示的に渡さない限り
  自動的に reject される。旧 blocklist 方式で `tier4_*.tex` /
  `cdf_lolo_tier4.pdf` が一度も `FROZEN_OUTPUT_PATHS` に載らず無防備だった
  ような穴を、新しい書き手を追加するたびに手動で塞ぐ必要が無くなる）の
  2 点である。悪意ある in-process caller や hand-edit そのものは Python
  レベルでは防げないため、code review と CI（`uv run pytest` +
  `scripts/verify_report.py`）で catch する運用に依存する。
