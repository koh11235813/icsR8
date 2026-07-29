# Google Colab での実行手順

icsR8 を Google Drive に置いた状態で Google Colab から実行するための
E2E 手順。アーキテクチャ上の決定と根拠は
`docs/adr/0003-colab-bootstrap-isolation.md` を参照。実装の正は
`scripts/colab_bootstrap.py`（このモジュールの docstring・関数 docstring が
一次情報。本書はその使い方ガイド）。

再現の主張は独立 2 本:

- **(a) 数値パイプライン**: `regenerate_main_body.py` / `regenerate_appendix_a.py`
  が作業コピー内の `doc/final_report/tables/*.tex` / `figures/*.pdf` へ直接書き、
  それが tracked な HEAD と diff 一致する（2026-07-29 allowlist 化以降、
  `results/colab/` への隔離出力は廃止 — sanctioned writer が凍結パス自体へ
  書く設計になったため、隔離してから比較する中間ステップが不要になった）
- **(b) LaTeX**: tracked スナップショットが Colab の LuaLaTeX + 和文フォントで
  コンパイルできる

Drive 上のリポは常に **read-only** として扱う。すべての実処理は
`/content` 配下の作業コピー（stage）上で行う。

## 1. セットアップセル

5 冊の notebook（`notebooks/baseline_reproduction.ipynb` ほか）の**最初の
code セル**には、共通の Colab bootstrap セルが挿入済みである。ローカル
Jupyter で開いた場合は `import google.colab` が `ImportError` になり、
何もせず通過する（no-op）。Colab 上では以下を行う:

1. `ICSR8_REPO_SOURCE` 環境変数が設定されていれば、そのパスを**唯一の正**
   として扱う。sentinel（`pyproject.toml` / `src/icsr8` / `data/dataset`）が
   不成立なら、mount も自動探索もせず即座にエラーで停止する。
2. 未設定なら、セル冒頭の定数 `DRIVE_REPO_DIR`（既定
   `/content/drive/MyDrive/icsR8`）を試す。
3. それも不成立なら `google.colab.drive.mount("/content/drive")` して
   `MyDrive/icsR8`・`Shareddrives/*/icsR8`・`Shareddrives/*/*/icsR8` を
   自動探索する。候補が 0 件または複数件なら列挙付きエラーで停止する
   （曖昧な自動選択はしない）。
4. 見つけたリポの `scripts/colab_bootstrap.py` を `compile()`/`exec()` で
   ロードし（`import` ではない — `__pycache__/*.pyc` を Drive に書かないため）、
   `bootstrap(_src)` を呼ぶ。

### `DRIVE_REPO_DIR` の編集

自分の Drive 上のリポの実際の配置パスがセルの既定値と異なる場合は、
セル冒頭の

```python
DRIVE_REPO_DIR = "/content/drive/MyDrive/icsR8"  # 置き場所が違うならここだけ編集
```

の行だけを書き換える。セルの他の行は編集しない（fail-closed 分岐が
崩れる）。

### リンク共有フォルダの MyDrive ショートカット注意

他人と共有された Google Drive フォルダ（「リンクを知っている全員」等で
共有されたフォルダ）は、**そのままでは** `drive.mount()` 後の
`MyDrive/icsR8` パスに現れない。共有フォルダは Drive UI 上で
「マイドライブへのショートカットを追加」を実行して、自分の MyDrive 直下に
ショートカットとして配置しておく必要がある。これを忘れると自動探索が
候補 0 件でエラーになる（`DRIVE_REPO_DIR` を共有フォルダの実パス
`/content/drive/Shareddrives/...` や、ショートカット経由のパスへ手動で
書き換えれば回避できる）。

## 2. 依存関係の準備と preflight（`--check`）

**依存修復は本処理（結果生成・LaTeX）より必ず前に済ませる。** 順序を
崩すと、`--check` が矛盾した状態（一部の依存だけ新しい）を green と
誤判定しうる。

1. **どちらの入口かで手順が分かれる**（2026-07-29 記述修正: 旧版は
   「セットアップセルの直後に `install_fonts=False` で呼び直す」としていたが、
   セットアップセル自身が既定 `install_fonts=True` で bootstrap を実行済みの
   ため後出しでは意味がない）:
   - **notebook 利用者（通常）**: セットアップセルを実行した時点でフォント
     導入まで完了している。matplotlib を upgrade する場合（下記 3.）は、
     upgrade 後に **kernel を restart してからセルを再実行**する（古い
     matplotlib モジュールが kernel に残ったままにしないため）。以降の
     手順で使う `root` は、セットアップセル実行後の cwd がそのまま作業
     コピーなので `root = Path.cwd()` で得る。
   - **ヘッドレス/スクリプト駆動（検証ドライバ等）**: セットアップセルを
     使わず、次の自己完結スニペット（copy-paste 可能。restart 後の再実行にも
     このまま使う）で `_ns` を構築して呼ぶ。依存修復を済ませてから必要なら
     フォント導入（`_ns["install_cjk_fonts"]()`）を行う:

     ```python
     import os
     from pathlib import Path

     source = Path(os.environ["ICSR8_REPO_SOURCE"]).resolve()
     _bs = source / "scripts" / "colab_bootstrap.py"
     _ns = {"__name__": "colab_bootstrap", "__file__": str(_bs)}
     exec(compile(_bs.read_text(encoding="utf-8"), str(_bs), "exec"), _ns)
     root = _ns["bootstrap"](source, install_fonts=False)
     ```

2. **`pytest>=8` を確保する**: `colab_bootstrap.py --check` が検査する
   pyproject.toml の 5 依存（matplotlib/numpy/pandas/scikit-learn/scipy）に
   `pytest` は含まれない（`pytest` は `[dependency-groups].dev` のみに
   ある）。`--check` は内部で `tests/test_reproduce_baseline.py` を
   `pytest` 経由で実行するため、`--check` の前に個別に確保しておく。
   `pip install` は既に条件を満たしていれば何もしない（idempotent）ので、
   常にこのフォールバックを打っておけばよい:

   ```bash
   !pip install -q "pytest>=8"
   ```

3. **5 依存のフロアを確認する**: `_check_dependency_floors` は
   pyproject.toml の `>=` 指定と実際にインストール済みのバージョンを
   比較し、不足があれば問題メッセージのリストを返す（空 = 全部満たす）。

   ```python
   problems = _ns["_check_dependency_floors"](root / "pyproject.toml")
   for p in problems:
       print("[preflight] NG:", p)
   ```

4. **不足していれば pip で修復する**（不足したパッケージだけを対象に、
   `--check` が出力するメッセージの通り floor 以上へ上げる）:

   ```bash
   !pip install -q --upgrade "matplotlib>=3.11.0"  # 例: 実際に NG だったものだけ
   ```

5. **matplotlib を upgrade した場合は kernel restart が必須**（Colab メニュー
   の Runtime > Restart runtime。または `os.kill(os.getpid(), 9)`）。
   matplotlib の backend はプロセス内で一度確定すると変えられないため、
   font 設定より前に upgrade するか、upgrade 後にプロセスごと作り直す
   必要がある。

6. **kernel restart 後は環境変数の再注入が必要**: `ICSR8_REPO_SOURCE` /
   `ICSR8_WORKDIR` 等の `os.environ` への設定は、restart で作られる
   新しい Python プロセスには引き継がれない。restart 直後に、
   - notebook 利用者: env 設定セル（使っていれば）→ **セットアップセルを
     そのまま再実行**する（既定 `install_fonts=True` のまま。stage 済み
     マーカーが有効なら reuse、apt は冪等なので再実行は速い）。
   - ヘッドレス: restart で `_ns` や Python 変数もすべて消えるため、
     env 再設定のうえ、手順 1 のヘッドレス用スニペットを**そのまま再実行**
     する（stage 済みマーカーが有効なら reuse されるので速い）。

7. **`--check` を実行する**（Python バージョン ≥3.11・5 依存フロア・
   `tests/test_reproduce_baseline.py` を順に確認する。全部通れば返り値 0）:

   ```bash
   !python scripts/colab_bootstrap.py --check
   ```

8. **`--check` が green になったら本処理へ**。フォントは、notebook 利用者は
   セットアップセルが導入済みなので何もしない。ヘッドレス
   （`install_fonts=False` で来た場合）のみ、ここで導入する:

   ```python
   _ns["install_cjk_fonts"]()
   ```

## 3. 結果生成（引数ゼロの sanctioned writer 経由のみ）

**このセクションは省略可能**: `results/*.csv` は 2026-07-29 以降 git 管理下にあり、
fresh clone 直後（Drive リポ・ローカル clone のどちらでも）から常に生成済みの
状態で存在する。`notebooks/tier{1,2,3,4}_methods.ipynb` の数値照合セルは
これらを読むだけなので、以下の regen コマンドを 1 度も実行しなくても Run All が
そのまま通る。以下は `results/*.csv` や凍結成果物を明示的に再生成したい場合
（コード変更を検証する等）のみ必要な手順。

**2026-07-29 allowlist 化**: 旧 `run_all_methods.py` / `run_tier4.py` と、それらの
argv ヘルパー `colab_bootstrap.run_all_methods_argv()` / `run_tier4_argv()` は
削除された。凍結成果物へ書けるのは `icsr8.report.regenerate_main_body()` /
`icsr8.report.regenerate_appendix_a()`（`src/icsr8/report.py`、共に**完全引数
ゼロ**）だけという allowlist 契約に一本化された（`src/icsr8/harness_tier4.py`
の `_guard_frozen` / `_SANCTIONED_WRITERS`。詳細は
`docs/adr/0004-deep-module-freeze-invariant.md`）。引数が無いので
「`--methods` を打ち間違える」「出力先フラグを省略する」という
2026-07-14 contamination incident 型の事故はそもそも起こり得ない。

Colab に `uv` は入っていない。`!python` （素の python 実行）で起動し、
`uv run` は使わない。

### 本文 Tier 1–3（15 手法 + 診断値・full）

```bash
!python scripts/regenerate_main_body.py
```

引数は取らない。CSV・表 TeX・図 PDF・`results/method_diagnostics.csv` は
すべて正規パス（`results/`・`doc/final_report/{tables,figures}`）へ
直接書かれる（tracked ファイルと byte 一致するはずなので、旧版のように
`results/colab/` へ隔離する必要が無くなった）。

### Tier 4（7 手法・full）

```bash
!python scripts/regenerate_appendix_a.py
```

引数は取らない。CSV は `results/tier4/`、表 TeX/図 PDF は
`doc/final_report/tables/tier4_*.tex` / `doc/final_report/figures/cdf_lolo_tier4.pdf`
へ直接書かれる。

### 追試・新手法・配管確認（`run_experimental_tier4.py`）

`regenerate_main_body.py` / `regenerate_appendix_a.py` は sanctioned writer で
固定 7/15 手法しか回せない。追試・新手法の評価や「パイプラインが最後まで
例外なく通るか」の配管確認には `scripts/run_experimental_tier4.py` を使う。
`--methods` / `--output` / `--tables-dir` / `--figures-dir` は全て required
（省略すると argparse がその場でエラーにする）。この CLI は sanctioned writer
ではないため、`--tables-dir`/`--figures-dir` に凍結ディレクトリ
（`doc/final_report/tables` 等）を指定すると `_guard_frozen` が `ValueError`
で reject する——隔離ディレクトリ以外を指すこと自体が構造的に不可能になっている。

```bash
!python scripts/run_experimental_tier4.py \
  --methods wcl_virtual_ap --output results/extra/vwcl \
  --tables-dir results/extra/vwcl --figures-dir results/extra/vwcl
# 配管確認（B=100・地点サブサンプル、本文照合には使えない）:
!python scripts/run_experimental_tier4.py \
  --methods wcl_virtual_ap --smoke \
  --output /tmp/smoke --tables-dir /tmp/smoke --figures-dir /tmp/smoke
```

`--smoke` は B と地点サブサンプルだけを制御し、`--methods` はどちらの場合も
そのまま使う（旧 `run_tier4.py` は `--smoke` が `--methods` を無条件に上書き
していたが、required 化に伴いこの footgun は無くした）。smoke 出力は
行数・地点サブセットが full と異なるため、notebook の数値照合セル
（公表値・凍結成果物との一致確認）には使えない——配管確認専用。

**regen 後に次の notebook を開いても作業コピーは作り直されない**:
`regenerate_main_body.py` / `regenerate_appendix_a.py` は staged 作業コピー
（`/content` 上）の `doc/final_report/{tables,figures}` を直接書き換える。
`colab_bootstrap.stage_working_copy()` の再利用時整合性検査は、この 2
ディレクトリ（`results/` と同様にパイプライン所有領域）を検査対象から除外する
（2026-07-30 修正。以前は除外が `results/` のみだったため、regen 直後に別の
notebook を Run All すると「staged tree が Drive 上の source と一致しない」と
誤判定され、作業コピーが `.old-*` へ退避されて生成物ごと失われていた）。
`doc/final_report/main.tex` 等（tables/figures 以外）は引き続き検査対象。

## 4. LaTeX ビルド（Colab 上の TeXLive）

### TeXLive 一式の導入

```bash
!apt-get update  # 2026-07-29 実機事故対応: Colab イメージの apt インデックスは
                 # 古く、security ポケットの deb が 404 になる（ruby3.0 で実際に
                 # 発生）。install の前に必ずインデックスを最新化する。
!apt-get install -y --no-install-recommends \
  texlive-luatex texlive-lang-japanese texlive-latex-extra \
  texlive-fonts-recommended latexmk fonts-noto-cjk
```

apt 管理の TeXLive を使う。**`tlmgr` を混用しない**（apt のパッケージ
データベースと `tlmgr` のそれは別管理であり、混ぜると依存関係が壊れる）。

### preflight: クラス/パッケージ疎通確認（3 コマンド個別）

以下の 3 つを**個別に**実行し、いずれも非空の出力（解決パス）が返ることを
確認する。1 コマンドにまとめない — どれが失敗したかを切り分けるため。

```bash
!kpsewhich ltjsarticle.cls
!kpsewhich luatexja.sty
!kpsewhich luatexja-preset.sty
```

### HaranoAji フォントの解決（順序固定）

`doc/slides/main.tex` は `luatexja-preset[haranoaji]` を使う。以下の順序で
確認する（順序を変えない — luaotfload のフォント DB が stale なまま
`--find` すると誤って「解決できない」と判定してしまうため、まず DB 更新を
先に行う）:

1. `luaotfload-tool --update --force`（フォント DB が stale な可能性を
   先に潰す）
2. `luaotfload-tool --find="HaranoAjiMincho-Regular"`（**exit code ではなく
   解決パスが非空であること**で判定する — luaotfload はエラー時にも 0
   を返すことがある）
3. 手順 2 で解決パスが空なら `apt-get install fonts-haranoaji`
4. 再度 `luaotfload-tool --update --force`
5. 再度 `luaotfload-tool --find="HaranoAjiMincho-Regular"` で非空を確認

```bash
!luaotfload-tool --update --force
!luaotfload-tool --find="HaranoAjiMincho-Regular"
# 上の出力が空なら:
!apt-get install -y --no-install-recommends fonts-haranoaji
!luaotfload-tool --update --force
!luaotfload-tool --find="HaranoAjiMincho-Regular"
```

### `latexmk` 実行（各 doc ディレクトリを cwd に、outdir はリポ外の絶対パス）

`doc/final_report/` と `doc/slides/` それぞれを cwd にして個別に実行する。
`-outdir` は**絶対パス**でリポ外（例 `/content/latex_build/<name>`）を指定し、
実行前にディレクトリを用意する。**`latexmk -c` / `-C` は使わない**
（`doc/slides/main.{aux,fls,log,...}` は latexmk のビルド副産物だが tracked
であり、`-c`/`-C` はそれらを削除してしまう）。

```bash
!mkdir -p /content/latex_build/final_report /content/latex_build/slides

!cd doc/final_report && latexmk -lualatex -outdir=/content/latex_build/final_report main.tex
!cd doc/slides       && latexmk -lualatex -outdir=/content/latex_build/slides main.tex
```

## 5. 成果物の書き戻し（Drive・リポ外のみ）

生成物（`results/`・`doc/final_report/{tables,figures}` の CSV/TeX/PDF、
`/content/latex_build/{final_report,slides}` の PDF 等）は、**Drive 上の
リポフォルダの外側**にのみコピーする（例
`MyDrive/icsR8_colab_output/`）。リポフォルダ自体は read-only 前提であり、
そこへ生成物を書き戻すと「Drive のリポが実は変更されている」状態になり
provenance が壊れる（ここでいう「リポフォルダ」は Drive 上の source を指す。
regenerate_main_body.py 等が書くのは `/content` 上の作業コピーであり、
read-only 前提を破らない — ADR-0003 参照）。

```python
import shutil
from pathlib import Path

root = Path.cwd()  # セットアップセル実行後の cwd = 作業コピー（§2 参照）
out = Path("/content/drive/MyDrive/icsR8_colab_output")
out.mkdir(parents=True, exist_ok=True)
shutil.copytree(root / "results", out / "results", dirs_exist_ok=True)
shutil.copytree(root / "doc" / "final_report" / "tables", out / "tables", dirs_exist_ok=True)
shutil.copytree(root / "doc" / "final_report" / "figures", out / "figures", dirs_exist_ok=True)
shutil.copytree(Path("/content/latex_build"), out / "latex_build", dirs_exist_ok=True)
```

### `.old-*` の手動掃除

`stage_working_copy` は source が更新された（または `refresh=True` の）
再実行のたびに、古い作業コピーを削除せず
`<workdir名>.old-<uuid4hex>` として `workdir` の兄弟ディレクトリへ退避する
（誤って有効なコピーを消さないための安全策）。これらは自動では消えない。
`/content` の容量を圧迫してきたら、必要なくなった `.old-*` ディレクトリを
手動で確認のうえ削除する:

```bash
!ls -la /content/*.old-*
!rm -rf /content/icsr8_work.old-<確認した具体的な hex>
```

## 緊急用セル（bootstrap 不在時）

`scripts/colab_bootstrap.py` を Drive 上のリポから読み込めない場合
（ファイルが壊れている・転送に失敗した・リポの更新前に急ぎ実行したい、等）
のための自己完結版。通常セル + bootstrap の **compact サブセット**であり、
検査系（env 権威・guard・symlink 拒否・mid-copy 検出・staged 照合・
already-imported ガード）は本体と同等に持つが、状態遷移系（reuse・refresh・
退避）は持たず、**既存 workdir があれば fail-closed で停止する**。
ローカル/非 Colab では no-op。`tests/test_colab_bootstrap.py` の
`CELL_SOURCES` パラメータ化スイートがこのセル本文を本ファイルから直接抽出し、
**共有サブセットのシナリオ**（env 不正・候補 0 件・staging 到達）+
**緊急セル固有の検査**（成功系完走・no-reuse・already-imported）を
走らせている（乖離検出）。

```python
# --- 緊急用セル（bootstrap 不在時のみ使う自己完結版）------------------------
# scripts/colab_bootstrap.py 自体が読めない/壊れている場合の代替。
# 通常セル + bootstrap の fail-closed 規律の compact サブセットだが、
# staging の安全検査は本体と同等に持つ:
#   env 権威 / sentinel / already-imported ガード / guard（同一・包含双方向・
#   drive 配下・/content 直下制限）/ source 側 marker 衝突拒否 /
#   symlink・特殊ファイル拒否 / mid-copy 変異検出（F_before == F_after）/
#   staged manifest == F_before / 既存 workdir は一切 reuse しない。
# 省くのは reuse・refresh・退避（= 状態遷移系）だけで、検査系は省かない。
# ローカル/非 Colab では no-op。
DRIVE_REPO_DIR = "/content/drive/MyDrive/icsR8"  # 置き場所が違うならここだけ編集

try:
    import google.colab  # noqa: F401
    _ON_COLAB = True
except ImportError:
    _ON_COLAB = False

if _ON_COLAB:  # ここから先は fail-closed（例外はそのまま停止）
    import json
    import os
    import shutil
    import stat as _stat
    import sys
    import tempfile
    from pathlib import Path

    _MARKER_FILENAME = ".icsr8_stage.json"
    _MARKER_MAGIC = "icsr8-colab-stage"
    _EXCLUDES = {".git", ".venv", "__pycache__", ".ipynb_checkpoints",
                 ".pytest_cache", ".ruff_cache"}

    def _is_repo(p):
        p = Path(p)
        return ((p / "pyproject.toml").is_file()
                and (p / "src" / "icsr8").is_dir()
                and (p / "data" / "dataset").is_dir())

    _env = os.environ.get("ICSR8_REPO_SOURCE")
    if _env is not None:  # 明示指定は唯一の正: 探索・mount へ落ちない
        _src = Path(_env)
        if not _is_repo(_src):
            raise FileNotFoundError(
                f"ICSR8_REPO_SOURCE={_env} はリポとして不成立（sentinel 欠落）。"
                "転送・展開の失敗を疑って。自動探索へのフォールバックはしない。")
    else:
        _src = Path(DRIVE_REPO_DIR)
        if not _is_repo(_src):
            from google.colab import drive
            drive.mount("/content/drive")
            _dr = Path("/content/drive")
            _cands = sorted({str(c) for c in (
                _src, _dr / "MyDrive/icsR8",
                *(_dr / "Shareddrives").glob("*/icsR8"),
                *(_dr / "Shareddrives").glob("*/*/icsR8"),
            ) if _is_repo(c)})
            if len(_cands) != 1:
                raise FileNotFoundError(
                    f"リポ候補が {len(_cands)} 件: {_cands} — DRIVE_REPO_DIR を"
                    "実配置へ編集して再実行して（リンク共有フォルダは MyDrive への"
                    "ショートカット追加が必要な場合がある。docs/COLAB.md 参照）。")
            _src = Path(_cands[0])
    _src = _src.resolve()

    _workdir = Path(os.environ.get("ICSR8_WORKDIR", "/content/icsr8_work")).resolve()

    # --- guard（fail-closed。同一パス/包含双方向/drive 配下/`/content` 直下制限）
    if _src == _workdir:
        raise ValueError(f"source と workdir が同一パス: {_src}")
    if _workdir.is_relative_to(_src):
        raise ValueError(f"workdir が source の配下にある: {_workdir}")
    if _src.is_relative_to(_workdir):
        raise ValueError(f"source が workdir の配下にある: {_src}")
    if _workdir.is_relative_to(Path("/content/drive")):
        raise ValueError(f"workdir が /content/drive 配下: {_workdir}")
    # /content 直下制限。ICSR8_EMERGENCY_ALLOW_NONCONTENT_WORKDIR=1 は
    # 同期テストが成功系を tmp で完走させるための**明示 opt-in seam**
    # （既定は fail-closed。暗黙の test-mode 推測はしない）。
    if os.environ.get("ICSR8_EMERGENCY_ALLOW_NONCONTENT_WORKDIR") != "1":
        if _workdir.parent != Path("/content") or _workdir.name in {"drive", "evidence"}:
            raise ValueError(
                f"Colab 上の workdir は /content 直下の安全な子に制限される: {_workdir}")

    # already-imported ガード: 旧パス由来の icsr8 が sys.modules に残ったまま
    # 新 stage を sys.path に前置しても、Python は旧モジュールを返し続ける
    # （見かけと違うコードを実行する provenance 事故）。staging・activate の
    # 前であれば発火位置として十分（通常経路の bootstrap() と同じ順序）。
    if any(m == "icsr8" or m.startswith("icsr8.") for m in sys.modules):
        raise RuntimeError(
            "icsr8 が既に import 済み。kernel を restart してからこのセルを再実行して。")

    # source 直下に marker があるのは source/workdir 取り違えの兆候（本体と同じ拒否）
    if (_src / _MARKER_FILENAME).exists():
        raise ValueError(
            f"source 直下に {_MARKER_FILENAME} が存在する: {_src} — source と "
            "workdir を取り違えていないか確認して。staging を拒否する。")

    def _iter_source_files(root):
        # symlink・特殊ファイルは fail-closed で拒否（本体と同じ契約。
        # 追跡コピーで source 外の bytes を staging しない）。
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            for d in list(dirnames):
                if (Path(dirpath) / d).is_symlink():
                    raise ValueError(f"source 内の symlink ディレクトリを拒否: {Path(dirpath) / d}")
            dirnames[:] = sorted(d for d in dirnames if d not in _EXCLUDES)
            for name in sorted(filenames):
                if name in _EXCLUDES or name.endswith(".pyc"):
                    continue
                p = Path(dirpath) / name
                st = p.lstat()
                if _stat.S_ISLNK(st.st_mode):
                    raise ValueError(f"source 内の symlink を拒否: {p}")
                if not _stat.S_ISREG(st.st_mode):
                    raise ValueError(f"source 内の非通常ファイルを拒否: {p}")
                yield p

    def _sha256_file(path):
        import hashlib
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _manifest(root):
        return {str(f.relative_to(root)): _sha256_file(f) for f in _iter_source_files(root)}

    def _manifest_digest(m):
        import hashlib
        canonical = json.dumps(m, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # --- staging: tmp-sibling copytree → 三方一致検査 → marker → promote。
    # 既存 workdir は reuse しない（digest 再検証・退避を持たない compact 版で
    # 弱い検査の reuse を許すと provenance を壊すため。fail-closed で停止）。
    if _workdir.exists():
        raise RuntimeError(
            f"{_workdir} が既に存在する。緊急セルは既存作業コピーを再利用しない。"
            "通常のセットアップセル（scripts/colab_bootstrap.py 経由）を使うか、"
            f"中身を確認のうえ手動で退避 (`!mv {_workdir} {_workdir}.manual-backup`) "
            "してから再実行して。")
    _f_before = _manifest(_src)
    _workdir.parent.mkdir(parents=True, exist_ok=True)
    _tmp = Path(tempfile.mkdtemp(dir=str(_workdir.parent), prefix="icsr8_stage_"))
    _staged = _tmp / "repo"
    for _f in _iter_source_files(_src):
        _rel = _f.relative_to(_src)
        _target = _staged / _rel
        _target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_f, _target)
    # mid-copy 変異検出: コピー前後の source manifest が一致すること
    if _manifest(_src) != _f_before:
        raise RuntimeError("コピー中に source が変化した（mid-copy mutation）。再実行して。")
    # staged 完全性: staged manifest が F_before と一致すること
    if _manifest(_staged) != _f_before:
        raise RuntimeError("staged tree が source manifest と一致しない。再実行して。")
    (_staged / _MARKER_FILENAME).write_text(
        json.dumps({"magic": _MARKER_MAGIC, "schema": 1,
                    "source": str(_src), "digest": _manifest_digest(_f_before)}),
        encoding="utf-8")
    os.replace(_staged, _workdir)
    shutil.rmtree(_tmp, ignore_errors=True)

    # --- activate: staged src を**必ず最優先**にする（既存出現を除去して
    # 先頭へ 1 回だけ挿入。途中に居座った場合でも他ツリーを先に import しない）
    _src_dir = str((_workdir / "src").resolve())
    sys.path[:] = [p for p in sys.path if p != _src_dir]
    sys.path.insert(0, _src_dir)
    _existing = os.environ.get("PYTHONPATH", "")
    _parts = [p for p in (_existing.split(os.pathsep) if _existing else []) if p != _src_dir]
    os.environ["PYTHONPATH"] = os.pathsep.join([_src_dir, *_parts]) if _parts else _src_dir
    os.chdir(_workdir)
    print(f"[emergency-cell] staged: {_workdir}")
```

## 6. 実機検証で確認済みの注意点（2026-07-29）

実 Colab VM での E2E 検証（8 セッション）から得た運用知見。

- **macOS からの転送で AppleDouble（`._*`）を混入させない**: macOS の
  `tar`/Finder 圧縮は既定で `._ファイル名` を同梱し、Linux 展開後に
  `glob("*.csv")` がバイナリごみを拾って `UnicodeDecodeError`（0xa3 等）で
  落ちる。tar 作成時は `COPYFILE_DISABLE=1 tar ...` を使うこと。混入して
  しまった場合は `find . -name '._*' -delete` で除去してから実行する。
- **matplotlib は preinstall 3.10 系で floor（>=3.11）未満**（実測）。
  `--check` は正しく NG を出す。§2 の手順どおり先に
  `pip install -q "matplotlib>=3.11"` を実施すること。なお図の再生成自体は
  3.10 でも tracked と同一の見た目・表 TeX はバイト一致だった。
- **無料枠 VM は持続高負荷 30〜80 分でサーバー側に回収されることがある**
  （CPU/GPU プール双方で観測）。full スイープ・tier4 notebook のような
  長時間ジョブは、切断に備えて完了直後に成果物（results/ 等）を Drive の
  リポ外フォルダへ退避するのが安全。セッションが失われても、正規パスの
  CSV を作業コピーへ書き戻せば notebook の照合セルはそのまま通る。
- **`verify_report.py` は `results/method_diagnostics.csv` も要求する**。
  2026-07-29 以降 `regenerate_main_body.py` が末尾で自動的に書くため、
  別コマンドを個別に打つ必要は無くなった（旧 `dump_method_diagnostics.py`
  は削除済み）。
- **colab-cli（google-colab-cli 0.6.0）利用者向け**: 依存の
  `jupyter-kernel-client` が 1.0.0 で API 破壊（`KernelClient` 廃止）。
  exec が `AttributeError` で全滅する場合は
  `uv pip install --python <tool-python> 'jupyter-kernel-client<1'` で
  0.x に戻すこと（上流にピンが入るまでの暫定措置）。
- **本書の検証で未達の 2 点（残余リスク）**: ①除外なし監査 manifest の
  initial-vs-final 等値による「ソース read-only」の形式証明は VM 回収により
  未完了（実 Drive はどのセッションでも mount していないため Drive 保全は
  記録上自明）。②実 Drive の mount／Shared Drive 自動探索の E2E は未実施
  （env 注入・スタブテストのみ）。初回に実 Drive で Run All する際は、
  実行前後で Drive 側リポの更新日時が変化していないことを目視確認すると良い。
