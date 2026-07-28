# ADR 0003: Colab bootstrap の隔離設計（パッケージ外配置・compile/exec・read-only Drive）

- Status: Accepted
- Date: 2026-07-28

## Context

icsR8 を Google Drive に置いた状態で Colab から実行し、`notebooks/` の
5 冊（baseline_reproduction, tier1〜4_methods）と `doc/final_report/` /
`doc/slides/` の再現性を確保したい。再現の主張は独立 2 本:
(a) 数値パイプライン（`results/colab/tables/*.tex` が tracked と diff 一致、
`results/colab/figures/*.pdf` が絶対パスで存在・parse 可能）、
(b) LaTeX（tracked スナップショットが Colab の LuaLaTeX + 和文フォントで
コンパイルできる）。

Google Drive は複数マシン・複数セッションから共有されるため、Drive 上の
リポフォルダそのものを read-only として扱いたい（誤って書き換える・
競合する変更を招かない）。一方で `src/icsr8/__init__.py` は
corridor/estimators/evaluate/fingerprint/methods 等を全 import し、
`methods/__init__.py` はさらに全 method module を自動 import するため、
**icsr8 パッケージのどの import も「全体ロード」を誘発する**。したがって
「作業コピーを作ってから初めて icsr8 を import する」という順序を守るには、
bootstrap 自体が icsr8 に依存できない。

## Decision

1. **パッケージ外配置（import-before-stage の回避）**: `scripts/colab_bootstrap.py`
   は `src/icsr8` パッケージの外に置く。モジュールのトップレベルは stdlib
   のみに限定し、`google.colab`/`matplotlib`/`subprocess`/`pandas` は使用する
   関数の内部でだけ import する。これにより、bootstrap 自体を読み込む行為が
   icsr8 の全体 import を誘発することはない。
2. **compile/exec ロード（`.pyc` 非生成）**: notebook セルは
   `scripts/colab_bootstrap.py` を `import` 文ではなく `read_text()` →
   `compile()` → `exec()` でロードする。`import` は成功時に
   `__pycache__/*.pyc` を書き込みうるが、Drive 上のリポは read-only 前提
   なので、そこにバイトコードキャッシュを残さない選択をした。
3. **Drive read-only + 作業コピー（stage）**: 実処理はすべて `/content` 配下の
   作業コピー上で行う。Drive 側の SHA は本 ADR の対象レイヤでは保証しない
   （read-only 監査 manifest による pre/post 一致確認は 実機検証手順（検証ドライバ）のスコープであり、`audit_manifest()` として実装だけは本モジュールに含む）。
4. **env 権威順位**: `ICSR8_REPO_SOURCE` が設定されていれば、そのパスが
   唯一の正。sentinel（`pyproject.toml` / `src/icsr8` / `data/dataset`）が
   不成立なら mount も自動探索もせず即エラーにする（検証時の provenance
   保証）。未設定時のみ `DRIVE_REPO_DIR`（セル定数）→ 不成立なら
   `google.colab.drive.mount` 後の自動探索（`MyDrive/icsR8`・
   `Shareddrives/*/icsR8`・`Shareddrives/*/*/icsR8`）に進む。候補が
   0 件・複数件は列挙付きエラーで停止する（曖昧な自動選択をしない）。
5. **full=正規 results・smoke=隔離ディレクトリの出力分離**: full 実行の CSV は
   `verify_report.py`/`dump_method_diagnostics.py`/notebook がハードコードする
   正規 `results/`・`results/tier4/` に書く（gitignore 済みの再生成物なので
   作業コピー内で上書きしても安全）。tables/figures は凍結ファイルを絶対に
   汚さないよう `results/colab/{tables,figures}` に隔離する。smoke は両
   パイプラインが同名 CSV を書き合うため、`--output`/`--tables-dir`/
   `--figures-dir` の 3 チャンネル全部を `results/colab/smoke/{main_body,tier4}/`
   に向ける。
6. **Colab 限定 chdir**: `activate()` は `sys.path` 先頭挿入・`PYTHONPATH`
   前置・`os.chdir()` を行う。`chdir` は Colab 実行時にのみ意味を持つ副作用
   であり、`bootstrap()` は非 Colab 環境では完全に no-op（cwd から
   `data/dataset` を持つ祖先を返すだけ）にして、ローカル Jupyter 等で誤って
   呼ばれても何も壊さないようにする。
7. **フォント = Colab 限定 rcParams**: `install_cjk_fonts()` は Colab 限定
   （`apt-get install fonts-noto-cjk` + `matplotlib.font_manager.addfont` +
   `rcParams["font.family"]` 先頭挿入）。**`matplotlib.pyplot` は import
   しない** — pyplot の初 import で backend が確定してしまい、
   `harness.py:18` が import 時に `setdefault("MPLBACKEND", "Agg")` する
   前提と衝突しうるため。素の `matplotlib.font_manager` の import は backend
   を確定させないので安全。
8. **latexmk は外部 outdir・`-c`/`-C` 禁止**: `doc/final_report/` と
   `doc/slides/` それぞれを cwd にして `latexmk -lualatex
   -outdir=/content/latex_build/<name> main.tex` を個別実行する。`-outdir`
   は絶対パスでリポ外に固定し、事前に `mkdir` する。`latexmk -c`/`-C` は
   使わない（`doc/slides/main.{aux,fls,log,...}` は latexmk のビルド副産物
   だが tracked であり、`-c`/`-C` で消してしまうと意図しない差分になる）。
9. **ランタイムフォールバック廃止**: 「bootstrap が使えない場合は手作業で
   何とかする」といった暗黙のフォールバック経路は用意しない。代わりに
   `docs/COLAB.md` に「緊急用セル（bootstrap 不在時）」を掲載する。緊急セルは
   通常セルの fail-closed 規律（env 権威・guard・所有権マーカー staging・
   例外は握り潰さない）に従う **compact サブセット**であり、digest 再検証・
   refresh・退避を持たない代わりに**既存 workdir を一切 reuse しない**
   （存在したら案内付きで停止 — 弱い検査で古い/別ソースの workdir を
   有効化して provenance を壊さないため。2026-07-29 改訂）。
   `tests/test_colab_bootstrap.py` の `CELL_SOURCES` パラメータ化スイートで
   通常セルと**恒久的に同期テスト**し、ドキュメントとテストが分離して
   ドリフトする事故を構造的に防ぐ。
10. **staging 安全境界（guard/manifest/退避）**: `guard_paths()` は
    source/workdir が同一・双方向の包含関係・`/content/drive` 配下・
    （Colab 上では）`/content` 直下以外を fail-closed で拒否する。
    `stage_working_copy()` は所有権マーカー（`.icsr8_stage.json` に
    magic/schema/source/digest を記録）で「既存 workdir が icsr8 の管理下か」
    を判定し、マーカー欠落・破損は**退避もせず即時拒否**する（正常に作られた
    workdir は atomic rename 由来なのでマーカー無しにはならない、という
    不変条件に依拠する）。source が更新された場合や `refresh=True` の場合は
    既存 workdir を `.old-<uuid4hex>` へ退避してから新規 stage する
    （削除しない・既存の `.old-*` にも触れない）。

## Consequences

- 良: Drive 上のリポを一度も書き換えずに Colab から icsR8 を実行できる。
  bootstrap 自体が icsr8 を import しないため、stage 前に誤って全体ロードが
  走る事故が構造的に起きない
- 良: env 権威順位により、検証時に「どのソースから実行したか」が常に
  一意に確定する（曖昧な自動選択が無い）
- 良: 所有権マーカー方式により、workdir の再利用/退避/拒否が「icsr8 が
  作ったかどうか」という一貫した基準で判定され、無関係な既存ディレクトリを
  誤って壊すリスクが構造的に低い
- 良: 緊急用セルを共有サブセットの契約テスト + 緊急セル固有の検査
  （成功系完走・no-reuse・already-imported）に通すことで、ドキュメントの
  記載が実装からドリフトしたまま放置される事故を機械的に検出できる
- 悪: bootstrap のトップレベルを stdlib のみに保つ制約上、
  `TIER4_METHODS` は `src/icsr8/harness_tier4.TIER4_METHODS` を import
  できずに literal ミラーせざるを得ない。ドリフトは
  `tests/test_colab_bootstrap.py::test_tier4_methods_matches_harness_tier4_exact_order`
  でのみ検出される（本モジュール自身は検出しない）
- 悪: 緊急用セルは通常セルの機能を完全網羅しない compact 版（reuse 時の
  digest 再検証・refresh・retire を持たない）。bootstrap 不在時の最小限の
  代替という位置づけであり、通常運用では常に通常セル（`scripts/colab_bootstrap.py`
  経由）を使うべきである
- 中立: read-only 監査 manifest（`audit_manifest()`、EXCLUDES 非適用）は
  本モジュールに実装済みだが、Drive 側の pre/post 一致確認そのものは
  実機検証手順（検証ドライバ）のスコープであり、本 ADR は「実装が存在すること」までを
  決定事項とする

## Amendment (2026-07-29): 実機検証で確定した契約修正

実 Colab VM 8 セッションでの実機検証（2026-07-28〜29）で以下を確定・修正した。

11. **staged-integrity 検査から `results/` を除外**: `run_tier4.py` は正規
    動作として git tracked な `results/tier4/run_tier4.log` を上書きする。
    `results/` はパイプライン所有の可変領域であり、整合性検査の対象に
    含めるとスイープ実行後の 2 冊目の notebook で reuse が拒否され、
    生成結果もろとも作業コピーが再作成されてしまう（実機で発生）。
    source 側の鮮度判定は source manifest digest 比較が引き続き担う。
12. **テストの環境密閉性**: `tests/conftest.py` の autouse fixture が
    `ICSR8_REPO_SOURCE` を削除し `ICSR8_WORKDIR` をテスト専用 tmp に
    **強制設定**する。削除だけでは既定値 `/content/icsr8_work` に落ち、
    /content が書き込み可能な実 Colab では本物の作業コピーを破壊する
    （実機で発生。ローカル macOS では / が書き込み不可のため顕在化しない）。
13. **非 Colab ホスト前提テストの skip**: `google.colab` が本物として
    import できる環境では、Colab をスタブで模擬する clean-subprocess
    テスト 3 本（MPLBACKEND 純度・ソースツリー不変・セル成功系）は
    前提が成立しないため `requires_non_colab_host` マーカーで skip する。
    skip してもスタブ対象の分岐カバレッジは維持されるが、実 Colab 上での
    source 不変性の機械的証明は別問題である（Amendment 16 参照）。
14. **`stage_working_copy()` 自身が構造ガードを強制する**（2026-07-29）:
    公開契約の関数として、bootstrap() を経由しない直接
    呼び出しでも `guard_paths(..., on_colab=False)` を冒頭で必ず実行する
    （同一パス・双方向包含・/content/drive 配下の拒否）。workdir を
    /content 直下に制限する Colab 運用ポリシー（guard 検査 5）だけは
    入口である bootstrap() の責務とする。
15. **単一ファイル構成は CLAUDE.md 分割規約への明示的例外**:
    `scripts/colab_bootstrap.py` が discovery・manifest・staging・フォント・
    argv・validator・CLI を 1 ファイルに収めるのは、「セルから compile/exec
    で単体ロードできる（= import 連鎖も .pyc も発生しない）」という本 ADR
    の決定 1〜2 の前提条件であるため。この例外は本モジュールに限る
    （テストファイルには適用されない）。
16. **実機検証の未達 2 点（残余リスクとして開示）**: ①除外なし
    `audit_manifest` の initial-vs-final 等値による source read-only の
    形式証明は、無料枠 VM が持続高負荷 30〜80 分で回収されるため実機で
    完了していない（実 Drive はどのセッションでも mount 自体行っておらず、
    Drive 保全はマーカー記録から自明に成立）。②実 Google Drive の
    mount／Shared Drive 自動探索は E2E 未実施（env 注入とスタブテストのみ）。
    このため Amendment 13 の skip は「スタブ模擬テストとしての検証力」は
    維持するが、**実 Colab 上での source 不変性の機械的証明はこの 2 点の
    分だけ弱い**ことを明記する。
