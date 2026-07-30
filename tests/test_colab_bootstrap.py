"""``scripts/colab_bootstrap.py`` の契約テスト。

このモジュールは icsr8 パッケージの**外**（scripts/）にあり、通常の
``import`` 経路が無いため、``importlib.util.spec_from_file_location`` で
明示的にロードする（compile/exec ではなく import にしているのは、テスト側は
sys.modules 汚染チェックの対象外であり、素直な import の方がフィクスチャ
再利用しやすいため。本番のノートブックセル側が compile/exec を使うことは
別途「ソースツリー不変」テストで固定する）。

検査する契約の背景は docs/adr/0003-colab-bootstrap-isolation.md（10 決定 +
Amendment）を参照。セクション見出しで対応関係を明記する。
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_PATH = REPO_ROOT / "scripts" / "colab_bootstrap.py"

# 2026-07-29 実機事故対応: 実 Colab VM 上では `google.colab` が本物として
# import できるため、「非 Colab ホスト」を前提にした clean-subprocess テスト
# （子プロセス内で bootstrap() の非 Colab 分岐や /content 書き込み不可を
# 期待するもの）は成立しない。特に成功系セルテストは実 /content に本当に
# ステージして検証用の実作業コピーを退避・破壊した。実 Colab では skip する
# （skip してもスタブ対象の分岐カバレッジは維持されるが、実 Colab 上の
# source 不変性の機械的証明は別問題 — ADR-0003 Amendment 16 参照）。
def _real_colab_available() -> bool:
    """本物の google.colab が import 可能か（find_spec は親パッケージ
    `google` 不在のローカルで ModuleNotFoundError を投げるため包む）。"""
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


requires_non_colab_host = pytest.mark.skipif(
    _real_colab_available(),
    reason="非 Colab ホスト前提の clean-subprocess テスト（実 Colab では本物の "
           "google.colab が子プロセスに見え、非 Colab 分岐/権限前提が成立しない）",
)


def _load_colab_bootstrap():
    """scripts/colab_bootstrap.py を import する（scripts/ はパッケージではない
    ので通常の import 文が使えない）。呼び出しごとに新しいモジュール
    オブジェクトを作る（テスト間で状態を共有しない）。
    """
    spec = importlib.util.spec_from_file_location("colab_bootstrap", BOOTSTRAP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def cb():
    """colab_bootstrap モジュールを毎テスト新規ロードして返す。

    モジュール自体はグローバル状態を持たない（sys.path/PYTHONPATH/cwd の
    副作用は呼んだ関数側が起こすものであり、import 自体は無害）。それでも
    毎回リロードするのは、テストが誤って前のテストの副作用（例えば
    monkeypatch 済みの sys.modules エントリ）を踏まないようにするため。
    """
    return _load_colab_bootstrap()


# ---------------------------------------------------------------------------
# セクション: notebook セットアップセル（notebooks/*.ipynb 第1 code セルの
# 全文と一致するセル）。notebooks 5 冊への挿入済みセルが正なので、ここでは
# notebooks/baseline_reproduction.ipynb を単一の真実源として読み込む
# （tests/test_notebooks_colab_invariants.py が 5 冊の setup セル source
# 完全一致を固定しているため、baseline 1 冊を読めば残り 4 冊も同一と保証される）。
# ---------------------------------------------------------------------------


def _read_setup_cell_source() -> str:
    """notebooks/baseline_reproduction.ipynb の最初の code セルの source を
    文字列として返す。nbformat の source は「行ごとの文字列のリスト（末尾行
    以外は \\n 終端）」なので "".join で結合するだけで元テキストが復元できる。
    """
    nb_path = REPO_ROOT / "notebooks" / "baseline_reproduction.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    first_code_cell = next(c for c in nb["cells"] if c["cell_type"] == "code")
    return "".join(first_code_cell["source"])


SETUP_CELL_SOURCE = _read_setup_cell_source()


def _read_emergency_cell_source() -> str:
    """docs/COLAB.md の「## 緊急用セル（bootstrap 不在時）」見出し直後にある
    最初の ```python フェンス内の全文を抽出する（stdlib 文字列処理のみ。
    Markdown パーサは使わない — 依存を増やさないため）。

    見出しが見つからない、見出し後にフェンスが無い、フェンス内容が空、
    抽出したソースが compile() できない、のいずれでも AssertionError にする
    （空文字列が偶然 CELL_SOURCES に混入して全パラメータが vacuous に
    パスしてしまう事故を防ぐ — 契約スイートの共通化が無意味になる最悪の
    退行なので、ここで確実に落とす）。
    """
    heading = "## 緊急用セル（bootstrap 不在時）"
    text = (REPO_ROOT / "docs" / "COLAB.md").read_text(encoding="utf-8")
    heading_pos = text.find(heading)
    assert heading_pos != -1, f"docs/COLAB.md に見出し {heading!r} が無い"
    after_heading = text[heading_pos + len(heading) :]

    fence_start = after_heading.find("```python\n")
    assert fence_start != -1, "見出し後に ```python フェンスが無い"
    body_start = fence_start + len("```python\n")
    fence_end = after_heading.find("```", body_start)
    assert fence_end != -1, "```python フェンスが閉じられていない"

    source = after_heading[body_start:fence_end]
    assert source.strip(), "緊急用セルの抽出結果が空"
    compile(source, "<emergency_cell>", "exec")  # 構文エラーなら即失敗させる
    return source


EMERGENCY_CELL_SOURCE = _read_emergency_cell_source()


# ---------------------------------------------------------------------------
# 合成リポツリー helper（実リポをコピーしない — 498MB の .venv を避け、
# staging/manifest テストを高速・決定的にする）
# ---------------------------------------------------------------------------


def _make_fake_repo(root: Path, *, extra_files: dict[str, str] | None = None) -> Path:
    """sentinel 3 点（pyproject.toml / src/icsr8 / data/dataset）を満たす
    最小の合成リポを `root` 配下に作り、そのパスを返す。

    `extra_files` は {relpath: content} で追加ファイルを注入できる
    （staging/manifest テストで内容変更・ファイル追加を模擬するために使う）。
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname = \"fake\"\n", encoding="utf-8")
    (root / "src" / "icsr8").mkdir(parents=True, exist_ok=True)
    (root / "src" / "icsr8" / "__init__.py").write_text(
        "__file__marker__ = 'fake-icsr8-package'\n", encoding="utf-8"
    )
    (root / "data" / "dataset").mkdir(parents=True, exist_ok=True)
    (root / "data" / "dataset" / "placeholder.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    if extra_files:
        for rel, content in extra_files.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
    return root


@pytest.fixture()
def fake_source(tmp_path: Path) -> Path:
    """sentinel を満たす合成 source リポを 1 つ作って返す（各テストで独立の
    tmp_path 配下 — pytest がテストごとに新しい tmp_path を発行するため、
    テスト間で共有・汚染されない）。
    """
    return _make_fake_repo(tmp_path / "source_repo")


class _FakeDrive:
    """`google.colab.drive` の最小スタブ。`mount` が呼ばれたかどうかを
    記録する（env 権威テストで「mount は呼ばれていない」ことを確認するため）。
    """

    def __init__(self) -> None:
        self.mount_calls: list[str] = []

    def mount(self, path: str) -> None:
        self.mount_calls.append(path)


def _install_fake_google_colab(monkeypatch: pytest.MonkeyPatch) -> _FakeDrive:
    """`sys.modules` に `google` / `google.colab` / `google.colab.drive` の
    スタブを差し込み、`is_colab()` が True を返すようにする。

    `monkeypatch.setitem(sys.modules, ...)` を使うことでテスト終了時に
    自動で元に戻る（他テストへ漏れない）。
    """
    fake_drive = _FakeDrive()

    fake_google = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("google", loader=None)
    )
    fake_colab = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("google.colab", loader=None)
    )
    fake_colab_drive = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("google.colab.drive", loader=None)
    )
    fake_colab_drive.mount = fake_drive.mount
    fake_colab.drive = fake_colab_drive
    fake_google.colab = fake_colab

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)
    monkeypatch.setitem(sys.modules, "google.colab.drive", fake_colab_drive)
    return fake_drive


# ===========================================================================
# 凍結ファイル保護 manifest（PROTECTED_PATHS）
#
# 2026-07-29 allowlist 化: scripts/run_all_methods.py / scripts/run_tier4.py と
# それらの argv 生成ヘルパー colab_bootstrap.run_all_methods_argv() /
# run_tier4_argv() は削除された（scripts/{regenerate_main_body,
# regenerate_appendix_a,run_experimental_tier4}.py に置き換え。前2つは
# 引数ゼロ、後者は --methods/--output/--tables-dir/--figures-dir 全 required）。
# これに伴い colab_bootstrap.MAIN_BODY_METHODS/TIER4_METHODS ミラーと、
# argv 契約テスト・15/7 手法検証テストは丸ごと不要になった（凍結ファイル保護は
# harness_tier4._SANCTIONED_WRITERS の allowlist に一本化 —
# tests/test_harness_tier4.py 参照）。
#
# PROTECTED_PATHS（doc/final_report・doc/slides の tracked 全ファイル）自体は
# colab_bootstrap の staging 保護（作業コピー中の tracked ファイル一覧）とは
# 独立の関心事として維持する。
# ===========================================================================

PROTECTED_PATHS: frozenset[Path] = frozenset(
    (REPO_ROOT / rel).resolve()
    for rel in (
        "doc/final_report/.latexmkrc",
        "doc/final_report/figures/cdf_lolo_tier4.pdf",
        "doc/final_report/figures/cdf_lolo.pdf",
        "doc/final_report/figures/cdf_protocol_a_backward_to_forward.pdf",
        "doc/final_report/figures/cdf_protocol_a_forward_to_backward.pdf",
        "doc/final_report/figures/segment_heatmap.pdf",
        "doc/final_report/main.pdf",
        "doc/final_report/main.tex",
        "doc/final_report/main.txt",
        "doc/final_report/tables/lolo.tex",
        "doc/final_report/tables/protocol_a.tex",
        "doc/final_report/tables/tier4_lolo.tex",
        "doc/final_report/tables/tier4_protocol_a.tex",
        "doc/slides/.latexmkrc",
        "doc/slides/main.aux",
        "doc/slides/main.fdb_latexmk",
        "doc/slides/main.fls",
        "doc/slides/main.log",
        "doc/slides/main.nav",
        "doc/slides/main.out",
        "doc/slides/main.pdf",
        "doc/slides/main.snm",
        "doc/slides/main.tex",
        "doc/slides/main.toc",
        "doc/slides/narration.md",
    )
)


def test_protected_paths_is_24_files():
    # 定数を死んだコードにしない最小の生存確認（argv 契約テスト削除後も
    # PROTECTED_PATHS 自体は独立の関心事として維持する契約 — 上のコメント参照）。
    # 25 = doc/final_report 13（.latexmkrc・main.{pdf,tex,txt}・tables 4・
    # figures 6）+ doc/slides 12（.latexmkrc・main.{aux,fdb_latexmk,fls,log,
    # nav,out,pdf,snm,tex,toc}・narration.md）。
    assert len(PROTECTED_PATHS) == 25


# ===========================================================================
# env 権威: `ICSR8_REPO_SOURCE` 設定時はそのパスが唯一の正。sentinel 不成立なら
# mount も探索もせず即エラー（ADR-0003 決定 4「env 権威順位」）。
# ===========================================================================


def test_env_source_valid_is_used_without_mount(cb, fake_source, monkeypatch):
    fake_drive = _install_fake_google_colab(monkeypatch)
    monkeypatch.setenv("ICSR8_REPO_SOURCE", str(fake_source))

    resolved = cb.find_repo_source(None)

    assert resolved == fake_source
    assert fake_drive.mount_calls == []


def test_env_source_invalid_errors_without_mount(cb, tmp_path, monkeypatch):
    """sentinel 不成立の ICSR8_REPO_SOURCE は mount へフォールバックせず、
    即エラーにする（唯一の正の権威。曖昧な自動探索への迂回を禁止する）。
    """
    fake_drive = _install_fake_google_colab(monkeypatch)
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    monkeypatch.setenv("ICSR8_REPO_SOURCE", str(not_a_repo))

    with pytest.raises(FileNotFoundError):
        cb.find_repo_source(None)

    assert fake_drive.mount_calls == []  # mount が一切呼ばれていないことが核心


def test_env_source_unset_falls_back_to_preferred_without_mount(cb, fake_source, monkeypatch):
    monkeypatch.delenv("ICSR8_REPO_SOURCE", raising=False)
    fake_drive = _install_fake_google_colab(monkeypatch)

    resolved = cb.find_repo_source(fake_source)

    assert resolved == fake_source
    assert fake_drive.mount_calls == []


def test_env_source_zero_candidates_raises_after_mount(cb, monkeypatch):
    """preferred 不成立・DRIVE_REPO_DIR 相当も不成立で、mount 後の自動探索で
    候補 0 件なら列挙付きエラーになる（列挙は空リスト）。
    """
    monkeypatch.delenv("ICSR8_REPO_SOURCE", raising=False)
    fake_drive = _install_fake_google_colab(monkeypatch)

    with pytest.raises(FileNotFoundError, match=r"リポ候補が 0 件"):
        cb.find_repo_source(None)

    assert fake_drive.mount_calls == ["/content/drive"]


# ===========================================================================
# guard_paths: 同一/包含（双方向）/drive 配下 workdir/symlink 経由/無関係
# ディレクトリ保全/Colab workdir の /content 制限（ADR-0003 決定 10。
# /content 制限は明示注入 seam（on_colab 引数）経由で検査する）。
# ===========================================================================


def test_guard_paths_rejects_identical_path(tmp_path, cb):
    same = tmp_path / "repo"
    same.mkdir()
    with pytest.raises(ValueError, match="同一パス"):
        cb.guard_paths(same, same, on_colab=False)


def test_guard_paths_rejects_workdir_under_source(tmp_path, cb):
    source = tmp_path / "repo"
    source.mkdir()
    workdir = source / "work"
    with pytest.raises(ValueError, match="source の配下"):
        cb.guard_paths(source, workdir, on_colab=False)


def test_guard_paths_rejects_source_under_workdir(tmp_path, cb):
    workdir = tmp_path / "work"
    workdir.mkdir()
    source = workdir / "repo"
    with pytest.raises(ValueError, match="workdir の配下"):
        cb.guard_paths(source, workdir, on_colab=False)


def test_guard_paths_rejects_workdir_under_content_drive(tmp_path, cb):
    source = tmp_path / "repo"
    source.mkdir()
    workdir = Path("/content/drive/MyDrive/icsr8_work")
    with pytest.raises(ValueError, match="/content/drive"):
        cb.guard_paths(source, workdir, on_colab=False)


def test_guard_paths_accepts_unrelated_paths_off_colab(tmp_path, cb):
    source = tmp_path / "repo"
    source.mkdir()
    workdir = tmp_path / "work"
    cb.guard_paths(source, workdir, on_colab=False)  # 例外が飛ばなければ OK


def test_guard_paths_symlink_mediated_overlap_is_rejected(tmp_path, cb):
    """workdir が symlink 経由で source 配下を指す事故を resolve() が潰し、
    通常の「source の配下」拒否に落ちることを確認する。
    """
    source = tmp_path / "repo"
    (source / "nested").mkdir(parents=True)
    link = tmp_path / "link_to_repo"
    link.symlink_to(source, target_is_directory=True)
    workdir_via_symlink = link / "nested"

    with pytest.raises(ValueError, match="source の配下"):
        cb.guard_paths(source, workdir_via_symlink, on_colab=False)


def test_guard_paths_symlink_mediated_source_equal_to_workdir_is_rejected(tmp_path, cb):
    """source 自体が symlink で、その実体が workdir と一致するケースも
    resolve() で同一パスに潰れて拒否されることを確認する。
    """
    real_dir = tmp_path / "real_repo"
    real_dir.mkdir()
    source_link = tmp_path / "source_via_link"
    source_link.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="同一パス"):
        cb.guard_paths(source_link, real_dir, on_colab=False)


def test_guard_paths_unrelated_directory_fully_preserved(tmp_path, cb):
    """guard_paths 自体は検査のみで副作用が無いこと（呼び出し前後で無関係な
    既存ディレクトリの path・内容が完全に維持される）を固定する。
    """
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("do-not-touch", encoding="utf-8")
    before = (unrelated / "keep.txt").read_bytes()

    source = tmp_path / "repo"
    source.mkdir()
    workdir = tmp_path / "work"
    cb.guard_paths(source, workdir, on_colab=False)

    assert unrelated.is_dir()
    assert (unrelated / "keep.txt").read_bytes() == before


def test_guard_paths_on_colab_requires_direct_child_of_content(tmp_path, cb):
    source = tmp_path / "repo"
    source.mkdir()
    nested_workdir = Path("/content/nested/work")
    with pytest.raises(ValueError, match="/content 直下"):
        cb.guard_paths(source, nested_workdir, on_colab=True)


def test_guard_paths_on_colab_rejects_reserved_children(tmp_path, cb):
    source = tmp_path / "repo"
    source.mkdir()
    with pytest.raises(ValueError, match="/content 直下"):
        cb.guard_paths(source, Path("/content/evidence"), on_colab=True)


def test_guard_paths_on_colab_accepts_direct_child_of_content(tmp_path, cb):
    source = tmp_path / "repo"
    source.mkdir()
    cb.guard_paths(source, Path("/content/icsr8_work"), on_colab=True)  # 例外なし = OK


def test_guard_paths_on_colab_seam_is_explicit_not_env_derived(tmp_path, cb, monkeypatch):
    """on_colab は明示注入 seam。tmp_path や PYTEST 環境変数を見て暗黙に
    test-mode を推測しているのではないことを、is_colab() を強制的に True へ
    monkeypatch した状態でも on_colab=False を渡せば /content 制限が
    掛からないことで確認する。
    """
    monkeypatch.setattr(cb, "is_colab", lambda: True)
    source = tmp_path / "repo"
    source.mkdir()
    workdir = tmp_path / "work"  # /content 配下ではない
    cb.guard_paths(source, workdir, on_colab=False)  # 明示 False が is_colab() より優先


# ===========================================================================
# staging（所有権マーカープロトコル）: ADR-0003 決定 10 + Amendment 14 の
# 全分岐。合成ツリーで検査する（実リポをコピーしない）。
# ===========================================================================


def _old_backups(workdir: Path) -> list[Path]:
    """workdir と同階層にある `.old-<hex>` 退避先を列挙する。"""
    return sorted(p for p in workdir.parent.glob(f"{workdir.name}.old-*") if p.is_dir())


def _read_marker_raw(root: Path, marker_filename: str) -> dict:
    return json.loads((root / marker_filename).read_text(encoding="utf-8"))


def test_stage_first_time_creates_valid_marker(fake_source, tmp_path, cb):
    workdir = tmp_path / "work"
    result = cb.stage_working_copy(fake_source, workdir)

    assert result == workdir
    assert (workdir / "src" / "icsr8" / "__init__.py").is_file()
    marker = _read_marker_raw(workdir, cb.MARKER_FILENAME)
    assert marker["magic"] == cb.MARKER_MAGIC
    assert marker["schema"] == cb.MARKER_SCHEMA
    assert marker["source"] == str(fake_source)
    assert re.match(r"^[0-9a-f]{64}$", marker["digest"])
    assert _old_backups(workdir) == []


def test_stage_valid_reuse_returns_same_path_without_retiring(fake_source, tmp_path, cb):
    workdir = tmp_path / "work"
    first = cb.stage_working_copy(fake_source, workdir)
    marker_before = _read_marker_raw(workdir, cb.MARKER_FILENAME)

    second = cb.stage_working_copy(fake_source, workdir)  # refresh=False, digest 一致

    assert second == first == workdir
    assert _read_marker_raw(workdir, cb.MARKER_FILENAME) == marker_before  # 再 stage されていない
    # 「返り値が同じパス」だけでは新規 stage でも成り立つ discriminating でない
    # 判定になってしまう（advisor 指摘）。.old-* が一切増えていないことこそが
    # 「実際に reuse され、restage されなかった」ことの決定的な証拠。
    assert _old_backups(workdir) == []


def test_stage_stale_source_triggers_restage(fake_source, tmp_path, cb):
    workdir = tmp_path / "work"
    cb.stage_working_copy(fake_source, workdir)

    # source 側にファイルを追加して digest を変える。
    (fake_source / "data" / "dataset" / "new_location.csv").write_text(
        "a,b\n9,9\n", encoding="utf-8"
    )

    result = cb.stage_working_copy(fake_source, workdir)

    assert result == workdir
    assert (workdir / "data" / "dataset" / "new_location.csv").is_file()
    backups = _old_backups(workdir)
    assert len(backups) == 1
    assert re.match(rf"^{re.escape(workdir.name)}\.old-[0-9a-f]{{32}}$", backups[0].name)


def test_stage_refresh_true_forces_restage_even_if_digest_matches(fake_source, tmp_path, cb):
    workdir = tmp_path / "work"
    cb.stage_working_copy(fake_source, workdir)

    result = cb.stage_working_copy(fake_source, workdir, refresh=True)

    assert result == workdir
    assert len(_old_backups(workdir)) == 1


def test_stage_marker_missing_is_refused_as_not_owned_without_retiring(tmp_path, cb, fake_source):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "some_unrelated_file.txt").write_text("pre-existing", encoding="utf-8")

    with pytest.raises(RuntimeError, match="非所有"):
        cb.stage_working_copy(fake_source, workdir)

    # 退避もしない契約: workdir はそのまま、.old-* も作られない。
    assert (workdir / "some_unrelated_file.txt").read_text(encoding="utf-8") == "pre-existing"
    assert _old_backups(workdir) == []


def test_stage_marker_missing_refusal_bypasses_refresh(tmp_path, cb, fake_source):
    """refresh=True でもマーカー欠落は迂回できない（ADR-0003 決定 10 のコア主張）。"""
    workdir = tmp_path / "work"
    workdir.mkdir()

    with pytest.raises(RuntimeError, match="非所有"):
        cb.stage_working_copy(fake_source, workdir, refresh=True)

    assert _old_backups(workdir) == []


@pytest.mark.parametrize(
    "corrupt_marker",
    [
        pytest.param("not json at all", id="not-json"),
        pytest.param(json.dumps(["array", "not", "dict"]), id="not-a-dict"),
        pytest.param(json.dumps({"magic": "wrong-magic", "schema": 1, "source": "/x", "digest": "a" * 64}), id="bad-magic"),
        pytest.param(json.dumps({"magic": "icsr8-colab-stage", "schema": 2, "source": "/x", "digest": "a" * 64}), id="bad-schema-value"),
        pytest.param(json.dumps({"magic": "icsr8-colab-stage", "schema": True, "source": "/x", "digest": "a" * 64}), id="schema-is-bool-not-int"),
        pytest.param(json.dumps({"magic": "icsr8-colab-stage", "schema": 1, "source": "", "digest": "a" * 64}), id="empty-source"),
        pytest.param(json.dumps({"magic": "icsr8-colab-stage", "schema": 1, "digest": "a" * 64}), id="missing-source"),
        pytest.param(json.dumps({"magic": "icsr8-colab-stage", "schema": 1, "source": "/x", "digest": "not-hex"}), id="digest-not-hex"),
        pytest.param(json.dumps({"magic": "icsr8-colab-stage", "schema": 1, "source": "/x", "digest": "a" * 63}), id="digest-wrong-length"),
        pytest.param(json.dumps({"magic": "icsr8-colab-stage", "schema": 1, "source": "/x", "digest": "A" * 64}), id="digest-uppercase-not-hex"),
    ],
)
def test_stage_marker_corrupt_field_type_is_refused_as_not_owned(
    tmp_path, cb, fake_source, corrupt_marker
):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / cb.MARKER_FILENAME).write_text(corrupt_marker, encoding="utf-8")

    with pytest.raises(RuntimeError, match="非所有"):
        cb.stage_working_copy(fake_source, workdir)

    assert _old_backups(workdir) == []  # corrupt でも退避しない


def test_stage_marker_symlink_is_refused_as_not_owned(tmp_path, cb, fake_source):
    """マーカーパス自体が symlink（差し替え攻撃/事故の想定）だと lstat 判定で
    「通常ファイルではない」として corrupt/非所有扱いになる。
    """
    workdir = tmp_path / "work"
    workdir.mkdir()
    real_marker_elsewhere = tmp_path / "elsewhere_marker.json"
    real_marker_elsewhere.write_text(
        json.dumps({"magic": cb.MARKER_MAGIC, "schema": 1, "source": str(fake_source), "digest": "a" * 64}),
        encoding="utf-8",
    )
    (workdir / cb.MARKER_FILENAME).symlink_to(real_marker_elsewhere)

    with pytest.raises(RuntimeError, match="非所有"):
        cb.stage_working_copy(fake_source, workdir)


def test_stage_source_root_marker_collision_is_rejected(fake_source, tmp_path, cb):
    """source root に既に .icsr8_stage.json がある場合は staging 自体を拒否する
    （source と workdir を取り違えた事故の検出）。
    """
    (fake_source / cb.MARKER_FILENAME).write_text('{"magic": "x"}', encoding="utf-8")
    workdir = tmp_path / "work"

    with pytest.raises(ValueError, match=re.escape(cb.MARKER_FILENAME)):
        cb.stage_working_copy(fake_source, workdir)

    assert not workdir.exists()  # 新規 stage にすら進んでいない


def test_stage_mid_copy_mutation_is_detected_and_workdir_untouched(
    fake_source, tmp_path, cb, monkeypatch
):
    """copytree 実行中に source が変異したケースを合成する: `_copytree_excluding`
    を「本来のコピーをした後、source に 1 ファイル追加してから戻る」ラッパーに
    差し替え、F_before != F_after を強制的に発生させる。
    """
    workdir = tmp_path / "work"
    real_copytree = cb._copytree_excluding

    def mutating_copytree(source, dest):
        real_copytree(source, dest)
        (Path(source) / "data" / "dataset" / "mutated_during_copy.csv").write_text(
            "x,y\n0,0\n", encoding="utf-8"
        )

    monkeypatch.setattr(cb, "_copytree_excluding", mutating_copytree)

    with pytest.raises(RuntimeError, match="mid-copy 変異検出"):
        cb.stage_working_copy(fake_source, workdir)

    assert not workdir.exists()  # workdir は一切作られていない（promote に進んでいない）
    # tmp ラッパーはフォレンジック用に残置される契約（削除しない）。
    leftover = list(tmp_path.glob("icsr8_stage_*"))
    assert len(leftover) == 1


def test_stage_copy_failure_leaves_canonical_workdir_untouched(
    fake_source, tmp_path, cb, monkeypatch
):
    """copytree 自体が例外を送出するケース（EXDEV 相当）: 事前に存在する
    workdir の内容が一切変更されないことを固定する。
    """
    workdir = tmp_path / "work"
    cb.stage_working_copy(fake_source, workdir)  # 事前に正常な stage を作っておく
    canonical_manifest_before = cb.source_manifest(workdir)

    def failing_copytree(source, dest):
        raise OSError("synthetic EXDEV-like copy failure")

    monkeypatch.setattr(cb, "_copytree_excluding", failing_copytree)

    with pytest.raises(OSError, match="synthetic EXDEV"):
        cb.stage_working_copy(fake_source, workdir, refresh=True)

    # 失敗は copytree の時点（retire より前）で起きるので、canonical workdir は
    # 退避すらされず元のまま残る。
    assert cb.source_manifest(workdir) == canonical_manifest_before
    assert _old_backups(workdir) == []


def test_stage_promote_failure_rolls_back_retired_backup(fake_source, tmp_path, cb, monkeypatch):
    """refresh トランザクション順序（ADR-0003 決定 10）: ①所有権確認 ②sibling tmp
    構築+検証 ③旧 workdir を UUID backup へ rename ④verified tmp を promote
    ⑤promote 失敗時は backup を自動 rollback。ここでは os.replace の「promote」
    呼び出し（src.name == 'repo' かつ dst == workdir）だけを失敗させ、直前の
    retire（dst == backup）は実際の os.replace に委譲することで、rollback だけを
    単離して検証する。
    """
    workdir = tmp_path / "work"
    cb.stage_working_copy(fake_source, workdir)
    manifest_before = cb.source_manifest(workdir)

    real_replace = os.replace

    def flaky_replace(src, dst):
        # promote 呼び出しの識別: src がステージ済み wrapper 内の "repo" で、
        # dst が最終 workdir そのもの。retire 呼び出し（dst はランダムな
        # ".old-<uuid>" backup）とはここで区別できる。
        if Path(src).name == "repo" and Path(dst) == workdir:
            raise OSError("synthetic promote failure")
        return real_replace(src, dst)

    monkeypatch.setattr(cb.os, "replace", flaky_replace)

    # source を変えて refresh 対象にする（同一 digest だと reuse 経路に入り
    # _stage_new 自体が呼ばれない）。
    (fake_source / "data" / "dataset" / "new_for_promote_failure.csv").write_text(
        "a,b\n1,1\n", encoding="utf-8"
    )

    with pytest.raises(OSError, match="synthetic promote failure"):
        cb.stage_working_copy(fake_source, workdir, refresh=True)

    # rollback により workdir は「元の内容」のまま存在し続ける（retire→promote
    # 失敗→rollback の一連の rename が完了している）。
    assert workdir.is_dir()
    assert cb.source_manifest(workdir) == manifest_before
    # rollback 後は backup 名が残っていない（元の名前に戻されている）。
    assert _old_backups(workdir) == []


def test_stage_old_backup_retirement_preserves_prior_backups(fake_source, tmp_path, cb):
    """`.old-<uuid>` 退避は都度新しい uuid を使うため、複数回の stale-restage を
    経ても過去の退避先を破壊しない（契約: 既存 .old-* は触らない）。
    """
    workdir = tmp_path / "work"
    cb.stage_working_copy(fake_source, workdir)

    (fake_source / "data" / "dataset" / "change_1.csv").write_text("1\n", encoding="utf-8")
    cb.stage_working_copy(fake_source, workdir)
    first_backups = _old_backups(workdir)
    assert len(first_backups) == 1

    (fake_source / "data" / "dataset" / "change_2.csv").write_text("2\n", encoding="utf-8")
    cb.stage_working_copy(fake_source, workdir)
    second_backups = _old_backups(workdir)

    assert len(second_backups) == 2
    assert set(first_backups) <= set(second_backups)  # 1 回目の退避先が消えていない


def test_stage_reuse_refuses_when_staged_file_deleted(fake_source, tmp_path, cb):
    """再利用前の staged-integrity 検査: source manifest に載っているファイルが
    staged 側から消えていたら（source は変わっていないので digest は一致する
    ままでも）reuse を拒否する。
    """
    workdir = tmp_path / "work"
    cb.stage_working_copy(fake_source, workdir)
    (workdir / "src" / "icsr8" / "__init__.py").unlink()

    with pytest.raises(RuntimeError, match="整合性検査に失敗"):
        cb.stage_working_copy(fake_source, workdir)


def test_stage_reuse_refuses_when_staged_file_modified(fake_source, tmp_path, cb):
    workdir = tmp_path / "work"
    cb.stage_working_copy(fake_source, workdir)
    (workdir / "src" / "icsr8" / "__init__.py").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="整合性検査に失敗"):
        cb.stage_working_copy(fake_source, workdir)


def test_stage_reuse_tolerates_derived_files_added_to_staged_tree(fake_source, tmp_path, cb):
    """staged 側に results/** のような派生物が追加されているだけなら reuse を
    妨げない（source 由来ファイルの欠落/改変だけを拒否する契約）。
    """
    workdir = tmp_path / "work"
    cb.stage_working_copy(fake_source, workdir)
    (workdir / "results").mkdir()
    (workdir / "results" / "protocol_a.csv").write_text("method,fold\n", encoding="utf-8")

    result = cb.stage_working_copy(fake_source, workdir)

    assert result == workdir
    assert _old_backups(workdir) == []  # reuse であって restage ではない
    assert (workdir / "results" / "protocol_a.csv").is_file()  # 派生物は消されない


def test_stage_reuse_tolerates_regenerated_report_tables_and_figures(tmp_path, cb):
    """staged 側の doc/final_report/{tables,figures} がパイプライン実行で
    上書きされても reuse を妨げない（doc/final_report 直下の main.tex 等は
    引き続き検査対象）。

    2026-07-30 review pivot: docs/COLAB.md の regen 手順は
    `regenerate_main_body.py` / `regenerate_appendix_a.py`（sanctioned writer）を
    staged 作業コピー上で直接実行する。旧実装は results/ 以外を検査対象に
    含めていたため、regen 直後の 2 回目の bootstrap 呼び出しで
    「staged tree が source と一致しない」という誤検出が起きていた
    （tables/figures は sanctioned writer の正規動作で変わるのが前提）。
    """
    source = _make_fake_repo(
        tmp_path / "source",
        extra_files={
            "doc/final_report/tables/protocol_a.tex": "orig-table\n",
            "doc/final_report/figures/cdf_lolo.pdf": "orig-pdf\n",
            "doc/final_report/main.tex": "orig-main\n",
        },
    )
    workdir = tmp_path / "work"
    cb.stage_working_copy(source, workdir)

    # sanctioned writer（regenerate_main_body 等）による正規の書き換えを模擬
    (workdir / "doc" / "final_report" / "tables" / "protocol_a.tex").write_text(
        "regenerated-table\n", encoding="utf-8"
    )
    (workdir / "doc" / "final_report" / "figures" / "cdf_lolo.pdf").write_text(
        "regenerated-pdf\n", encoding="utf-8"
    )

    result = cb.stage_working_copy(source, workdir)

    assert result == workdir
    assert _old_backups(workdir) == []  # reuse であって restage ではない
    assert (workdir / "doc" / "final_report" / "tables" / "protocol_a.tex").read_text(
        encoding="utf-8"
    ) == "regenerated-table\n"

    # doc/final_report/tables・figures 以外（main.tex 等）の改変は引き続き拒否
    # されること（除外が tables/figures に限定される回帰確認）。
    (workdir / "doc" / "final_report" / "main.tex").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="整合性検査に失敗"):
        cb.stage_working_copy(source, workdir)


def test_stage_reuse_tolerates_pipeline_mutated_tracked_results(fake_source, tmp_path, cb):
    """source 側に tracked な results/ 配下ファイル（例: run_tier4.log）が
    ある状態で、staged 側のそれがパイプライン実行で上書きされても reuse を
    妨げない。

    # 2026-07-29 実機事故対応: run_tier4.py は正規動作として tracked な
    # results/tier4/run_tier4.log を上書きする。旧実装は整合性検査で
    # これを「source 由来ファイルの改変」として拒否したため、スイープ後の
    # 2 冊目の notebook で作業コピーが作り直され生成結果が失われた。
    # results/ はパイプライン所有領域として検査対象外にする。
    """
    (fake_source / "results" / "tier4").mkdir(parents=True)
    (fake_source / "results" / "tier4" / "run_tier4.log").write_text("orig\n", encoding="utf-8")
    workdir = tmp_path / "work"
    cb.stage_working_copy(fake_source, workdir)

    # パイプラインが staged 側の tracked ログを上書きした状況を再現
    (workdir / "results" / "tier4" / "run_tier4.log").write_text("rewritten by pipeline\n", encoding="utf-8")

    result = cb.stage_working_copy(fake_source, workdir)

    assert result == workdir
    assert _old_backups(workdir) == []  # reuse であって restage ではない
    # 上書きされたログはそのまま維持される（reuse は staged 側を書き換えない）
    assert (workdir / "results" / "tier4" / "run_tier4.log").read_text(encoding="utf-8") == "rewritten by pipeline\n"
    # src/ 側の改変は引き続き拒否されること（除外が results/ に限定される回帰確認）
    (workdir / "src" / "icsr8" / "__init__.py").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="整合性検査に失敗"):
        cb.stage_working_copy(fake_source, workdir)


# ===========================================================================
# audit_manifest: read-only 監査用（EXCLUDES 適用なし）。.pyc 追加・entry 削除・
# 型変化・symlink/FIFO を必ず検出する（ADR-0003 決定 3・Amendment 16）。
# ===========================================================================


def test_audit_manifest_detects_pycache_pyc_addition(tmp_path, cb):
    """staging/reuse manifest（source_manifest）は __pycache__/*.pyc を
    EXCLUDES で無視するが、audit_manifest は無視しない — この非対称性こそが
    「除外対象が新規生成されていないか」を検出できる理由。
    """
    root = _make_fake_repo(tmp_path / "audited")
    before = cb.audit_manifest(root)
    assert not any("__pycache__" in k for k in before)

    pycache = root / "src" / "icsr8" / "__pycache__"
    pycache.mkdir()
    (pycache / "__init__.cpython-311.pyc").write_bytes(b"\x00\x01fakepyc")

    after = cb.audit_manifest(root)
    assert after != before
    pyc_key = "src/icsr8/__pycache__/__init__.cpython-311.pyc"
    assert pyc_key in after
    assert after[pyc_key][0] == "file"


def test_audit_manifest_detects_entry_removal(tmp_path, cb):
    root = _make_fake_repo(tmp_path / "audited")
    before = cb.audit_manifest(root)

    (root / "data" / "dataset" / "placeholder.csv").unlink()

    after = cb.audit_manifest(root)
    assert "data/dataset/placeholder.csv" in before
    assert "data/dataset/placeholder.csv" not in after
    assert after != before


def test_audit_manifest_detects_file_to_directory_type_change(tmp_path, cb):
    root = _make_fake_repo(tmp_path / "audited")
    before = cb.audit_manifest(root)
    assert before["pyproject.toml"][0] == "file"

    (root / "pyproject.toml").unlink()
    (root / "pyproject.toml").mkdir()  # file -> dir への型変化

    after = cb.audit_manifest(root)
    assert after["pyproject.toml"][0] == "dir"
    assert after["pyproject.toml"] != before["pyproject.toml"]


def test_audit_manifest_records_symlink_as_symlink_kind_without_raising(tmp_path, cb):
    """audit_manifest は source_manifest と違い symlink を**拒否しない**
    （記録するだけ）— 目的が「変化を検出すること」であって「安全に copy
    できるか」ではないため。
    """
    root = _make_fake_repo(tmp_path / "audited")
    target = root / "data" / "dataset" / "placeholder.csv"
    link = root / "data" / "dataset" / "link_to_placeholder.csv"
    link.symlink_to(target)

    manifest = cb.audit_manifest(root)

    assert manifest["data/dataset/link_to_placeholder.csv"] == ("symlink", None)


def test_audit_manifest_records_fifo_as_other_kind_without_raising(tmp_path, cb):
    root = _make_fake_repo(tmp_path / "audited")
    fifo_path = root / "data" / "dataset" / "a_fifo"
    os.mkfifo(fifo_path)

    manifest = cb.audit_manifest(root)

    assert manifest["data/dataset/a_fifo"] == ("other", None)


def test_source_manifest_rejects_symlink_in_source_tree(tmp_path, cb):
    """一方 source_manifest（staging/reuse 用）は symlink を発見したら
    ValueError で拒否する — audit_manifest とは逆の設計意図（ADR-0003 決定 10
    対比: こちらは「安全に copy できる状態か」を検査する）。
    """
    root = _make_fake_repo(tmp_path / "src_repo")
    target = root / "data" / "dataset" / "placeholder.csv"
    link = root / "data" / "dataset" / "link_to_placeholder.csv"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        cb.source_manifest(root)


def test_source_manifest_rejects_fifo_in_source_tree(tmp_path, cb):
    root = _make_fake_repo(tmp_path / "src_repo")
    os.mkfifo(root / "data" / "dataset" / "a_fifo")

    with pytest.raises(ValueError, match="通常ファイル"):
        cb.source_manifest(root)


# ===========================================================================
# クリーン subprocess 群: MPLBACKEND 純度・ソースツリー不変(.pyc 回帰)・
# 子プロセスでの icsr8.__file__ が staged src 配下を指すこと
#
# いずれも「実際の pytest プロセス自体を汚染しない」ことが必須要件 — activate()
# は sys.path/os.environ/cwd を書き換える副作用を持つため、これを直接この
# テストプロセス内で呼ぶとテスト間で状態が漏れる。そのため必ず子プロセス側で
# 呼び、親側は stdout の内容だけを検査する。
# ===========================================================================


def _run_python_script(script: str, args: list[str], *, cwd: Path | None = None, env: dict | None = None):
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@requires_non_colab_host
def test_mplbackend_purity_in_clean_subprocess(tmp_path, cb):
    """compile/exec ロード + 非 Colab `bootstrap()` 呼び出しが、MPLBACKEND を
    一切 setdefault しないこと、icsr8*/matplotlib.pyplot を import しないことを
    クリーンな子プロセスで確認する。

    harness.py:18 は import 時に `MPLBACKEND=Agg` を setdefault するので、もし
    colab_bootstrap 経由で icsr8 が import されてしまえば MPLBACKEND が
    設定されてしまう — このテストはその回帰を検出する。
    """
    root = _make_fake_repo(tmp_path / "cwd_repo")  # bootstrap() の非 Colab no-op 探索対象
    script = textwrap.dedent(
        """
        import sys, os
        assert "MPLBACKEND" not in os.environ, "test setup leaked MPLBACKEND"
        bootstrap_path = sys.argv[1]
        ns = {"__name__": "colab_bootstrap", "__file__": bootstrap_path}
        src = open(bootstrap_path, encoding="utf-8").read()
        exec(compile(src, bootstrap_path, "exec"), ns)
        root = ns["bootstrap"]()
        assert "MPLBACKEND" not in os.environ, "bootstrap() must not import icsr8/pyplot off-Colab"
        bad = [m for m in sys.modules if m == "icsr8" or m.startswith("icsr8.") or "matplotlib" in m]
        assert bad == [], f"unexpected modules: {bad}"
        print("OK", root)
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    result = _run_python_script(script, [str(BOOTSTRAP_PATH)], cwd=root, env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().startswith("OK")


@requires_non_colab_host
def test_source_tree_unchanged_after_cell_loader_path(tmp_path, cb):
    """ノートブックセルの compile/exec ロード経路を実行した前後で、synthetic
    ソースツリーの audit_manifest（除外なし）が完全不変であること —
    `__pycache__/*.pyc` が新規生成されていないことを含む（import と違い
    compile/exec は disk にバイトコードキャッシュを書かないための regression
    テスト）。
    """
    synthetic_root = _make_fake_repo(tmp_path / "synthetic_notebook_cwd")
    scripts_dir = synthetic_root / "scripts"
    scripts_dir.mkdir()
    staged_bootstrap = scripts_dir / "colab_bootstrap.py"
    staged_bootstrap.write_text(BOOTSTRAP_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    before = cb.audit_manifest(synthetic_root)

    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        cwd = Path(sys.argv[1])
        bootstrap_path = cwd / "scripts" / "colab_bootstrap.py"
        ns = {"__name__": "colab_bootstrap", "__file__": str(bootstrap_path)}
        exec(compile(bootstrap_path.read_text(encoding="utf-8"), str(bootstrap_path), "exec"), ns)
        ns["bootstrap"]()  # 非 Colab: cwd から data/dataset 祖先を返すだけの no-op
        print("OK")
        """
    )
    result = _run_python_script(script, [str(synthetic_root)], cwd=synthetic_root)
    assert result.returncode == 0, result.stdout + result.stderr

    after = cb.audit_manifest(synthetic_root)
    assert after == before, "compile/exec ロード経路がソースツリーに副作用を残した"


def test_child_process_icsr8_file_resolves_under_staged_src(tmp_path, cb):
    """`activate()` が設定する PYTHONPATH は `!python` 経由の子プロセスへ
    引き継がれ、その子プロセスの `import icsr8` が staged src 配下を解決する
    ことを二重の subprocess（outer が activate() を呼び、outer が起動する
    grandchild が実際に import する）で確認する。

    activate() 自体は sys.path/os.environ/cwd を書き換える副作用を持つため、
    このテストプロセス内で直接呼ばず、必ず outer subprocess の中だけで呼ぶ。
    """
    staged_root = _make_fake_repo(tmp_path / "staged")
    (staged_root / "src" / "icsr8" / "__init__.py").write_text(
        "MARKER = 'this-is-the-staged-package'\n", encoding="utf-8"
    )

    outer_script = textwrap.dedent(
        """
        import sys, os, subprocess
        from pathlib import Path

        staged_root = Path(sys.argv[1])
        bootstrap_path = Path(sys.argv[2])

        ns = {"__name__": "colab_bootstrap", "__file__": str(bootstrap_path)}
        exec(compile(bootstrap_path.read_text(encoding="utf-8"), str(bootstrap_path), "exec"), ns)
        ns["activate"](staged_root)

        grandchild = subprocess.run(
            [sys.executable, "-c", "import icsr8, sys; print(icsr8.__file__)"],
            env=os.environ, capture_output=True, text=True,
        )
        assert grandchild.returncode == 0, grandchild.stdout + grandchild.stderr
        print(grandchild.stdout.strip())
        """
    )
    clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = _run_python_script(
        outer_script, [str(staged_root), str(BOOTSTRAP_PATH)], env=clean_env
    )

    assert result.returncode == 0, result.stdout + result.stderr
    reported_file = result.stdout.strip()
    assert reported_file.startswith(str((staged_root / "src").resolve()))


# ===========================================================================
# activate 冪等性・already-imported ガード（契約:
# `name == "icsr8" or name.startswith("icsr8.")` を prefix 込みで検査）。
#
# `activate()` はこのプロセスの sys.path/os.environ/cwd を実際に書き換える
# ため、各テストはテスト自身で monkeypatch により確実に元へ戻す
# （monkeypatch.chdir / monkeypatch.setenv 相当を手動で行い、finally で
# 復元する）。
# ===========================================================================


@pytest.fixture()
def isolated_activate_state(monkeypatch):
    """activate() が汚す sys.path / PYTHONPATH / cwd をテスト終了時に必ず
    元へ戻すためのフィクスチャ。monkeypatch の自動ロールバックに乗せる。
    """
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)
    yield
    os.chdir(original_cwd)
    sys.path[:] = original_sys_path


def test_activate_idempotent_no_duplicate_sys_path_or_pythonpath(
    tmp_path, cb, isolated_activate_state, monkeypatch
):
    repo_root = _make_fake_repo(tmp_path / "repo_for_activate")
    monkeypatch.delenv("PYTHONPATH", raising=False)

    cb.activate(repo_root)
    cb.activate(repo_root)  # 2 回目

    src_dir = str((repo_root / "src").resolve())
    assert sys.path.count(src_dir) == 1
    pythonpath_parts = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    assert pythonpath_parts.count(src_dir) == 1
    assert Path.cwd() == repo_root.resolve()


def test_activate_prepends_to_existing_pythonpath_without_duplication(
    tmp_path, cb, isolated_activate_state, monkeypatch
):
    repo_root = _make_fake_repo(tmp_path / "repo_for_activate2")
    preexisting = str(tmp_path / "some_other_path")
    monkeypatch.setenv("PYTHONPATH", preexisting)

    cb.activate(repo_root)

    src_dir = str((repo_root / "src").resolve())
    parts = os.environ["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == src_dir  # 前置
    assert preexisting in parts  # 既存分は消えていない

    cb.activate(repo_root)  # 再度呼んでも重複しない
    parts_after = os.environ["PYTHONPATH"].split(os.pathsep)
    assert parts_after.count(src_dir) == 1


def test_already_imported_guard_rejects_icsr8_itself(cb, monkeypatch):
    monkeypatch.setitem(sys.modules, "icsr8", object())
    monkeypatch.setattr(cb, "is_colab", lambda: True)

    with pytest.raises(RuntimeError, match="既にこのプロセスに import 済み"):
        cb.bootstrap(Path("/unused"))


def test_already_imported_guard_rejects_icsr8_submodule_child_only_stub(cb, monkeypatch):
    """already-imported ガードの prefix 検査の回帰テスト: `sys.modules` に `"icsr8"` 本体は無く
    `"icsr8.foo"` という子モジュールだけが残っているケース（例えば以前の
    `from icsr8 import foo` の残骸）でもガードが発火すること。
    """
    monkeypatch.delitem(sys.modules, "icsr8", raising=False)
    monkeypatch.setitem(sys.modules, "icsr8.foo", object())
    monkeypatch.setattr(cb, "is_colab", lambda: True)

    with pytest.raises(RuntimeError, match="既にこのプロセスに import 済み"):
        cb.bootstrap(Path("/unused"))


def test_already_imported_guard_does_not_false_positive_on_similar_prefix(cb, monkeypatch):
    """`"icsr8x"` のような単なる前方一致（`.` 区切りでない）はガード対象外
    であることを固定する（`name.startswith("icsr8.")` が正しく `.` を要求する）。

    テストプロセスは同一プロセス内で他のテストが既に `icsr8.methods` 等を
    import 済みのことがあるため、`icsr8`/`icsr8.*` に一致する既存エントリを
    一旦すべて monkeypatch で除去してから検査する（monkeypatch がテスト終了時
    に元へ戻すので、他テストへは漏れない）。
    """
    for name in list(sys.modules):
        if name == "icsr8" or name.startswith("icsr8."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "icsr8x_unrelated", object())
    # ガードは通過するので、その先の find_repo_source が実際に到達し、
    # google.colab.drive の mount 経由で候補 0 件になって落ちることを利用して
    # 「already-imported ガードそのものは通過した」ことを確認する。
    _install_fake_google_colab(monkeypatch)
    monkeypatch.delenv("ICSR8_REPO_SOURCE", raising=False)

    with pytest.raises(FileNotFoundError, match=r"リポ候補が 0 件"):
        cb.bootstrap(Path("/definitely/not/a/repo"))


# ===========================================================================
# フロア比較関数（数値ドット prefix の簡易規約）の単体テスト
# （--check の依存フロア照合の前提となる数値 prefix 比較規約）。
# ===========================================================================


@pytest.mark.parametrize(
    "version_str, expected",
    [
        ("3.11.0", (3, 11, 0)),
        ("2.1", (2, 1)),
        ("3.11.0rc1", (3, 11, 0)),
        ("1.26", (1, 26)),
        ("1", (1,)),
        ("2024.1.0.post1", (2024, 1, 0)),
    ],
)
def test_numeric_dot_prefix(cb, version_str, expected):
    assert cb._numeric_dot_prefix(version_str) == expected


def test_numeric_dot_prefix_non_numeric_first_segment_is_empty_tuple(cb):
    assert cb._numeric_dot_prefix("rc1") == ()


@pytest.mark.parametrize(
    "requirement, expected",
    [
        ("matplotlib>=3.11.0", ("matplotlib", (3, 11, 0))),
        ("numpy>=1.26", ("numpy", (1, 26))),
        ("scikit-learn>=1.4", ("scikit-learn", (1, 4))),
    ],
)
def test_parse_floor_requirement_valid(cb, requirement, expected):
    assert cb._parse_floor_requirement(requirement) == expected


@pytest.mark.parametrize(
    "requirement",
    ["matplotlib==3.11.0", "matplotlib<=3.11.0", "matplotlib"],
)
def test_parse_floor_requirement_non_ge_returns_none(cb, requirement):
    assert cb._parse_floor_requirement(requirement) is None


def test_check_dependency_floors_against_real_pyproject_has_no_problems(cb):
    """開発環境自体は pyproject.toml のフロアを満たしているはず（`uv sync`
    済みの前提）。実 pyproject.toml を読む契約自体の smoke テスト。
    """
    problems = cb._check_dependency_floors(REPO_ROOT / "pyproject.toml")
    assert problems == []


def test_check_dependency_floors_detects_unmet_floor(cb, tmp_path, monkeypatch):
    fake_pyproject = tmp_path / "pyproject.toml"
    fake_pyproject.write_text(
        textwrap.dedent(
            """
            [project]
            dependencies = ["numpy>=999.0.0"]
            """
        ),
        encoding="utf-8",
    )

    problems = cb._check_dependency_floors(fake_pyproject)

    assert len(problems) == 1
    assert "numpy" in problems[0]
    assert "999.0.0" in problems[0]


def test_check_dependency_floors_detects_missing_package(cb, tmp_path):
    fake_pyproject = tmp_path / "pyproject.toml"
    fake_pyproject.write_text(
        textwrap.dedent(
            """
            [project]
            dependencies = ["definitely-not-installed-pkg>=1.0"]
            """
        ),
        encoding="utf-8",
    )

    problems = cb._check_dependency_floors(fake_pyproject)

    assert len(problems) == 1
    assert "未インストール" in problems[0]


# ===========================================================================
# validate_colab_outputs 契約テスト（合成 CSV。ファイル別 literal 契約）:
# per-file 一意キー・重複検出・failed フラグ・有限値検査（数値列を明示列挙）。
# ===========================================================================


def _write_csv(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")


def test_validate_colab_outputs_happy_path_returns_facts(tmp_path, cb):
    _write_csv(
        tmp_path / "protocol_a.csv",
        """
        method,fold,ave,failed
        wcl,forward_to_backward,3.5,False
        wcl,backward_to_forward,3.6,False
        gp_corridor,forward_to_backward,3.1,False
        """,
    )

    facts = cb.validate_colab_outputs(tmp_path)

    assert facts["protocol_a.csv"]["rows"] == 3
    assert facts["protocol_a.csv"]["methods"] == ["gp_corridor", "wcl"]
    assert "method" in facts["protocol_a.csv"]["columns"]


def test_validate_colab_outputs_ignores_files_not_in_key_column_map(tmp_path, cb):
    _write_csv(tmp_path / "unrelated_file.csv", "a,b\n1,2\n")
    facts = cb.validate_colab_outputs(tmp_path)
    assert facts == {}


def test_validate_colab_outputs_detects_duplicate_composite_key(tmp_path, cb):
    _write_csv(
        tmp_path / "lolo_summary.csv",
        """
        method,ave
        wcl,3.5
        wcl,3.6
        """,
    )

    with pytest.raises(ValueError, match="重複している"):
        cb.validate_colab_outputs(tmp_path)


def test_validate_colab_outputs_detects_missing_key_columns(tmp_path, cb):
    _write_csv(tmp_path / "lolo_summary.csv", "not_method,ave\nwcl,3.5\n")

    with pytest.raises(ValueError, match="複合キー列が無い"):
        cb.validate_colab_outputs(tmp_path)


def test_validate_colab_outputs_detects_failed_true_row(tmp_path, cb):
    _write_csv(
        tmp_path / "protocol_a_ledger.csv",
        """
        method,fold,location_p,failed
        wcl,forward_to_backward,P1,False
        wcl,forward_to_backward,P2,True
        """,
    )

    with pytest.raises(ValueError, match="failed=True"):
        cb.validate_colab_outputs(tmp_path)


def test_validate_colab_outputs_numeric_columns_detects_non_finite(tmp_path, cb):
    _write_csv(
        tmp_path / "lolo_ledger.csv",
        """
        method,held_out,error
        wcl,P1,1.0
        wcl,P2,inf
        """,
    )

    with pytest.raises(ValueError, match="非有限値"):
        cb.validate_colab_outputs(tmp_path, numeric_columns={"lolo_ledger.csv": ("error",)})


def test_validate_colab_outputs_numeric_columns_detects_nan(tmp_path, cb):
    _write_csv(
        tmp_path / "lolo_ledger.csv",
        """
        method,held_out,error
        wcl,P1,1.0
        wcl,P2,
        """,
    )

    with pytest.raises(ValueError, match="非有限値"):
        cb.validate_colab_outputs(tmp_path, numeric_columns={"lolo_ledger.csv": ("error",)})


def test_validate_colab_outputs_numeric_columns_passes_when_all_finite(tmp_path, cb):
    _write_csv(
        tmp_path / "lolo_ledger.csv",
        """
        method,held_out,error
        wcl,P1,1.0
        wcl,P2,2.5
        """,
    )

    facts = cb.validate_colab_outputs(tmp_path, numeric_columns={"lolo_ledger.csv": ("error",)})
    assert facts["lolo_ledger.csv"]["rows"] == 2


def test_validate_colab_outputs_numeric_columns_missing_column_raises(tmp_path, cb):
    _write_csv(tmp_path / "lolo_ledger.csv", "method,held_out\nwcl,P1\n")

    with pytest.raises(ValueError, match="数値列指定だが列が無い"):
        cb.validate_colab_outputs(tmp_path, numeric_columns={"lolo_ledger.csv": ("error",)})


def test_validate_colab_outputs_diagnostics_mixed_type_value_needs_explicit_numeric_columns(
    tmp_path, cb
):
    """tier4 diagnostics.csv は value 列が文字列/数値混在なので、有限性検査の
    対象は呼び出し側が明示した列だけに限る契約。value 自体は
    検査対象に含めなければ非数値混在でも通る。
    """
    _write_csv(
        tmp_path / "diagnostics.csv",
        """
        protocol,fold,method,key,value
        protocol_a,f1,gp_corridor,segment_label,corridor_A
        protocol_a,f1,gp_corridor,n_iter,7
        """,
    )

    facts = cb.validate_colab_outputs(tmp_path)  # numeric_columns 省略 = value は検査しない
    assert facts["diagnostics.csv"]["rows"] == 2


def test_validate_colab_outputs_composite_key_unique_across_all_five_known_files(tmp_path, cb):
    """5 種の既知ファイルすべてが同時に存在しても独立に検査されること。"""
    _write_csv(tmp_path / "protocol_a.csv", "method,fold,ave\nwcl,f1,1.0\n")
    _write_csv(tmp_path / "protocol_a_ledger.csv", "method,fold,location_p\nwcl,f1,P1\n")
    _write_csv(tmp_path / "lolo_ledger.csv", "method,held_out\nwcl,P1\n")
    _write_csv(tmp_path / "lolo_summary.csv", "method,ave\nwcl,1.0\n")
    _write_csv(tmp_path / "diagnostics.csv", "protocol,fold,method,key,value\npa,f1,wcl,n,1\n")

    facts = cb.validate_colab_outputs(tmp_path)
    assert set(facts) == {
        "protocol_a.csv", "protocol_a_ledger.csv", "lolo_ledger.csv",
        "lolo_summary.csv", "diagnostics.csv",
    }


# ===========================================================================
# 本文 15 手法オラクル: テスト側 literal で固定する（本体定数から生成しない。
# icsr8.report.MAIN_BODY_METHODS 等から逆算すると自己参照になり、本体側の
# バグをテストが見逃してしまうため独立に書き下す）。
#
# 2026-07-29 allowlist 化: colab の argv 生成ヘルパーが無くなったため、旧
# SMOKE_ORACLE_*_CSV/TEX/PDF（basename 個数の smoke 契約）と、それらに依存した
# 4 テストは削除した。results/*.csv は Commit 1 以降 git 管理下にあり
# フレッシュクローンでも常に存在するため、以下のテストは skip 不要になった。
# ===========================================================================

# 本文 15 手法（README の Tier1-3 fenced コマンド由来 — icsr8.report.MAIN_BODY_METHODS
# からの逆算ではなく、ここで独立にもう一度書き下す）。
MAIN_BODY_METHODS_ORACLE = (
    "centered_fp", "cla", "gp_corridor", "multiband_wcl", "pbl", "rank_fp",
    "studentt_fp", "wcl", "wcl_blacklist", "wcl_corridor", "wcl_linpower",
    "wcl_powerdomain", "wcl_topl", "wcl_varweight", "wknn",
)

# 本文 15 手法の期待行数（テスト側独立 literal — 本体定数から生成しない契約）。
FULL_ROW_COUNTS_ORACLE = {
    "protocol_a.csv": 30,
    "protocol_a_ledger.csv": 1770,
    "lolo_ledger.csv": 885,
    "lolo_summary.csv": 15,
}

_RESULTS_DIR = REPO_ROOT / "results"


def test_full_main_body_results_match_expected_row_counts_and_method_set(cb):
    """commit 済み `results/*.csv` に対して、テスト側 literal 契約が明示する
    行数・15 手法 exact・複合キー一意・failed 全 False・有限値を
    validate_colab_outputs 経由で検査する。

    2026-07-29: results/*.csv は Commit 1 以降 git 管理下でフレッシュクローン
    にも常に存在するため、以前あった「不在なら skip」の skipif は削除した。
    """
    facts = cb.validate_colab_outputs(
        _RESULTS_DIR,
        numeric_columns={
            "protocol_a.csv": ("ave", "median", "p90", "max", "std"),
            "lolo_ledger.csv": ("error",),
            "lolo_summary.csv": ("ave", "median", "p90"),
        },
    )

    for filename, expected_rows in FULL_ROW_COUNTS_ORACLE.items():
        assert facts[filename]["rows"] == expected_rows, filename

    assert facts["protocol_a.csv"]["methods"] == sorted(MAIN_BODY_METHODS_ORACLE)
    assert facts["lolo_summary.csv"]["methods"] == sorted(MAIN_BODY_METHODS_ORACLE)


# ===========================================================================
# 契約スイートの共通化: 通常セル（SETUP_CELL_SOURCE）と、docs/COLAB.md から
# 抽出される緊急用セル（EMERGENCY_CELL_SOURCE）を**同一 parameterized
# スイート**に通す（ADR-0003 決定 9: セル本文と実装のドリフト防止）。
# 緊急用セルは通常セルと乖離しうる独立実装（bootstrap 不在時の自己完結版）
# なので、同じテストを両方に通すことでドリフトを検出する。
# ===========================================================================

CELL_SOURCES = [
    pytest.param(SETUP_CELL_SOURCE, id="notebook_setup"),
    pytest.param(EMERGENCY_CELL_SOURCE, id="emergency_cell"),
]


def _build_colab_stub_preamble(mount_log_path: Path) -> str:
    """clean subprocess の中で `google.colab`/`google.colab.drive` スタブを
    sys.modules に差し込む python ソースを生成する。`drive.mount` が呼ばれた
    引数を `mount_log_path` へ JSON で書き出す（サブプロセス終了後に親側から
    「mount が呼ばれたか」を検証するための唯一の手段 — プロセスを跨ぐと
    通常の呼び出し記録オブジェクトは共有できない）。
    """
    return textwrap.dedent(
        f"""
        import sys, json, importlib.util
        from pathlib import Path

        _mount_calls = []

        def _record_mount(path):
            _mount_calls.append(path)
            Path(r{str(mount_log_path)!r}).write_text(json.dumps(_mount_calls), encoding="utf-8")

        _fake_google = importlib.util.module_from_spec(importlib.util.spec_from_loader("google", loader=None))
        _fake_colab = importlib.util.module_from_spec(importlib.util.spec_from_loader("google.colab", loader=None))
        _fake_drive = importlib.util.module_from_spec(importlib.util.spec_from_loader("google.colab.drive", loader=None))
        _fake_drive.mount = _record_mount
        _fake_colab.drive = _fake_drive
        _fake_google.colab = _fake_colab
        sys.modules["google"] = _fake_google
        sys.modules["google.colab"] = _fake_colab
        sys.modules["google.colab.drive"] = _fake_drive
        """
    )


@pytest.mark.parametrize("cell_source", CELL_SOURCES)
def test_cell_invalid_env_source_errors_without_mount(cell_source, tmp_path, cb, monkeypatch):
    """無効な ICSR8_REPO_SOURCE は、セル自身の inline sentinel チェックで
    bootstrap 呼び出し前に即エラーになる — mount は一切呼ばれない。この分岐は
    icsr8/colab_bootstrap.py のどちらにも到達しないため in-process で安全に
    検査できる（失敗系は bootstrap 呼び出し前に停止するため in-process stub で可）。
    """
    fake_drive = _install_fake_google_colab(monkeypatch)
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    monkeypatch.setenv("ICSR8_REPO_SOURCE", str(not_a_repo))

    with pytest.raises(FileNotFoundError, match="sentinel 欠落"):
        exec(compile(cell_source, "<cell>", "exec"), {"__name__": "__main__"})

    assert fake_drive.mount_calls == []


@pytest.mark.parametrize("cell_source", CELL_SOURCES)
def test_cell_env_unset_and_default_drive_repo_dir_missing_raises_zero_candidates(
    cell_source, cb, monkeypatch
):
    """env 未設定・既定の DRIVE_REPO_DIR（セル定数）も不成立の場合、mount 後の
    自動探索で候補 0 件エラーになる。この分岐も bootstrap 呼び出し前に完結する。
    """
    monkeypatch.delenv("ICSR8_REPO_SOURCE", raising=False)
    fake_drive = _install_fake_google_colab(monkeypatch)

    with pytest.raises(FileNotFoundError, match=r"0 件"):
        exec(compile(cell_source, "<cell>", "exec"), {"__name__": "__main__"})

    assert fake_drive.mount_calls == ["/content/drive"]


@pytest.mark.parametrize("cell_source", CELL_SOURCES)
@requires_non_colab_host
def test_cell_valid_env_source_reaches_staging_without_mount_in_clean_subprocess(
    cell_source, fake_source, tmp_path, cb
):
    """成功系（valid ソースで stage まで到達）はクリーン subprocess で実行する
    契約（pytest 自体が既に icsr8 系モジュールを import 済みのため、in-process
    では必ず already-imported ガードに当たってしまう）。

    このテストマシンには実 Colab の `/content` が存在せず書き込み権限も無い
    （`os.access("/", os.W_OK)` が False — 確認済み）ため、"stage まで到達"の
    意味を「already-imported ガード通過 → find_repo_source が ICSR8_REPO_SOURCE
    を mount 無しで採用 → guard_paths 通過 → 実際にステージ処理(`_stage_new`)
    が `/content` 直下へのディレクトリ作成を試みて PermissionError で止まる」
    ところまでとする。これは実 Colab（/content が書き込み可能）であれば
    そのまま成功していたはずの地点を正確に切り出した検査であり、mount が
    一度も呼ばれていないことと合わせて「env 権威 → stage 到達」の経路全体を
    実際に通したことの証拠になる。
    """
    # セルは `_src/scripts/colab_bootstrap.py` の実在を要求する
    # （このファイル自体を compile/exec でロードする対象のため）。
    scripts_dir = fake_source / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "colab_bootstrap.py").write_text(
        BOOTSTRAP_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )

    mount_log = tmp_path / "mount_calls.json"
    preamble = _build_colab_stub_preamble(mount_log)
    script = preamble + "\n" + cell_source

    env = {k: v for k, v in os.environ.items() if not k.startswith("ICSR8_")}
    env["ICSR8_REPO_SOURCE"] = str(fake_source)

    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60
    )

    assert result.returncode != 0  # 実マシンには /content が無いのでここで止まる
    assert not mount_log.exists(), "ICSR8_REPO_SOURCE 設定時に mount が呼ばれてはいけない"
    assert "/content" in result.stderr
    # 弱い検査への注意: guard_paths の拒否メッセージにも "/content" という
    # 文字列が含まれる（"Colab 上の workdir は /content 直下の…に制限される"）。
    # returncode!=0 と "/content" in stderr だけでは「guard_paths に拒否された」
    # ケースと区別が付かない。ValueError（guard_paths が投げる型）ではなく
    # 実際に `_stage_new` の `workdir.parent.mkdir(...)` まで到達して OSError で
    # 止まったことを、トレースバックの発生フレームと例外型で判別する。
    assert "ValueError" not in result.stderr, "guard_paths に拒否されている（staging まで届いていない）"
    assert "in mkdir" in result.stderr or "os.mkdir" in result.stderr
    # マシンにより「作れない」の具体的な理由が違う（権限不足 vs read-only fs）。
    # どちらも「/content 配下への実書き込みを試みた」ことの証拠として等価。
    assert any(marker in result.stderr for marker in ("PermissionError", "Errno 13", "Errno 30", "Read-only file system"))


# ---------------------------------------------------------------------------
# 2026-07-29 追加契約: 直接呼び出しの構造ガード・marker 厳格化・緊急セル no-reuse
# （背景は docs/adr/0003-colab-bootstrap-isolation.md Amendment 14 と決定 9）
# ---------------------------------------------------------------------------


def test_stage_working_copy_direct_call_enforces_containment_guards(fake_source, tmp_path, cb):
    """公開契約の stage_working_copy() は bootstrap() を経由しない直接呼び出し
    でも構造ガードを自前で強制する（Amendment 14）。source 内 workdir を許すと
    sibling tmp が source 内に作られ自己再帰コピーになる。"""
    # workdir が source の配下（未作成パスでも resolve(strict=False) で検出）
    with pytest.raises(ValueError, match="配下"):
        cb.stage_working_copy(fake_source, fake_source / "new-work")
    # source が workdir の配下（意味的に逆転した設定ミス）
    workdir = tmp_path / "outer-work"
    workdir.mkdir()
    with pytest.raises(ValueError, match="配下"):
        cb.stage_working_copy(workdir / "inner-src", workdir)
    # 同一パス
    with pytest.raises(ValueError, match="同一"):
        cb.stage_working_copy(fake_source, fake_source)


def test_stage_working_copy_direct_call_rejects_drive_workdir(fake_source, cb):
    """/content/drive 配下の workdir は直接呼び出しでも拒否される
    （Drive read-only 契約を公開関数の通常呼び出しで破れないこと）。"""
    with pytest.raises(ValueError, match="drive"):
        cb.stage_working_copy(fake_source, Path("/content/drive/MyDrive/whatever"))


def test_marker_with_relative_source_path_is_not_owned(fake_source, tmp_path, cb):
    """marker の source は絶対パス文字列であること（ADR-0003 決定 10）。
    相対パスの marker は corrupt/非所有として reuse も退避もされず拒否される。"""
    workdir = tmp_path / "work"
    cb.stage_working_copy(fake_source, workdir)
    marker_path = workdir / cb.MARKER_FILENAME
    data = json.loads(marker_path.read_text(encoding="utf-8"))
    data["source"] = "relative/path/to/repo"
    marker_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="非所有"):
        cb.stage_working_copy(fake_source, workdir)


def test_emergency_cell_has_no_reuse_branch_and_refuses_existing_workdir():
    """緊急セルは既存 workdir を一切 reuse しない契約（ADR-0003 決定 9・
    2026-07-29 改訂）の**ソース形状**を固定する: (1) 拒否メッセージが存在する
    (2) marker 読み取りによる reuse 分岐が存在しない。実行時の拒否挙動は
    test_emergency_cell_refusal_of_existing_workdir_is_dynamic が seam 経由で
    動的に検査しており、本テストはその補助（dead-branch 化の検出）。"""
    src = EMERGENCY_CELL_SOURCE
    assert "再利用しない" in src, "既存 workdir 拒否の契約文言が緊急セルから消えている"
    assert "_read_marker" not in src, "緊急セルに marker ベースの reuse 分岐が復活している"
    # 拒否は RuntimeError（fail-closed）で行う
    assert "RuntimeError" in src


def test_emergency_cell_full_success_path_in_clean_subprocess(fake_source, tmp_path):
    """緊急セルの成功系を**実際に完走**させて固定する（ADR-0003 決定 9）。

    ICSR8_EMERGENCY_ALLOW_NONCONTENT_WORKDIR=1 はセル内に明示された
    テスト専用 opt-in seam（既定 fail-closed）。これにより /content が無い
    ホストでも staging → marker → activate まで到達できる。クリーン
    subprocess で実行するのは already-imported ガード（本 pytest プロセスは
    icsr8 を import 済み）を正当に通過するため。実 Colab 上でも
    ICSR8_REPO_SOURCE 設定済みなら mount せず同様に完走する（skip 不要）。
    """
    mount_log = tmp_path / "mount_calls.json"
    preamble = _build_colab_stub_preamble(mount_log)
    workdir = tmp_path / "emergency_work"
    epilogue = "\nimport os as _os\nprint('CWD=' + _os.getcwd())\nprint('PP=' + _os.environ.get('PYTHONPATH', ''))\n"
    script = preamble + "\n" + EMERGENCY_CELL_SOURCE + epilogue

    env = {k: v for k, v in os.environ.items() if not k.startswith("ICSR8_")}
    env["ICSR8_REPO_SOURCE"] = str(fake_source)
    env["ICSR8_WORKDIR"] = str(workdir)
    env["ICSR8_EMERGENCY_ALLOW_NONCONTENT_WORKDIR"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not mount_log.exists(), "ICSR8_REPO_SOURCE 設定時に mount が呼ばれてはいけない"
    # staging 完了: marker の全フィールドが valid
    marker = json.loads((workdir / ".icsr8_stage.json").read_text(encoding="utf-8"))
    assert marker["magic"] == "icsr8-colab-stage"
    assert marker["schema"] == 1
    assert os.path.isabs(marker["source"])
    assert re.fullmatch(r"[0-9a-f]{64}", marker["digest"])
    # staged 内容が source と一致（sentinel の代表 2 点で確認）
    assert (workdir / "pyproject.toml").read_bytes() == (fake_source / "pyproject.toml").read_bytes()
    assert (workdir / "src" / "icsr8").is_dir()
    # activate 完了: chdir + PYTHONPATH 前置
    assert f"CWD={workdir}" in result.stdout
    assert str(workdir / "src") in result.stdout.split("PP=")[-1]


def test_emergency_cell_refuses_when_icsr8_already_imported(fake_source, tmp_path):
    """緊急セルの already-imported ガード: 旧 icsr8 が sys.modules に残った
    kernel では staging せず kernel restart を要求する（provenance 事故防止）。"""
    mount_log = tmp_path / "mount_calls.json"
    preamble = _build_colab_stub_preamble(mount_log)
    inject = (
        "\nimport sys, types\n"
        "sys.modules['icsr8'] = types.ModuleType('icsr8')\n"
        "sys.modules['icsr8.harness'] = types.ModuleType('icsr8.harness')\n"
    )
    script = preamble + inject + "\n" + EMERGENCY_CELL_SOURCE

    env = {k: v for k, v in os.environ.items() if not k.startswith("ICSR8_")}
    env["ICSR8_REPO_SOURCE"] = str(fake_source)
    env["ICSR8_WORKDIR"] = str(tmp_path / "never_created")
    env["ICSR8_EMERGENCY_ALLOW_NONCONTENT_WORKDIR"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60
    )

    assert result.returncode != 0
    assert "restart" in result.stderr
    assert not (tmp_path / "never_created").exists(), "ガードは staging より前に発火する契約"


def test_activate_promotes_staged_src_to_front_even_if_already_present(fake_source, tmp_path, cb, monkeypatch):
    """activate() の契約は「staged src を必ず最優先」。既に sys.path/PYTHONPATH
    の**途中**に居る場合も、既存出現を除去して先頭へ挿入し直す。

    # 2026-07-29: 旧実装は「存在すれば何もしない」だったため、staged src より
    # 前に別ツリーが並ぶと子プロセスがそちらを先に import し得た（provenance 欠陥）。
    """
    workdir = tmp_path / "work"
    cb.stage_working_copy(fake_source, workdir)
    src_dir = str((workdir / "src").resolve())
    other = str(tmp_path / "other_src")

    original_sys_path = list(sys.path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([other, src_dir]))
    monkeypatch.chdir(tmp_path)
    try:
        sys.path[:] = [other, src_dir, *original_sys_path]
        cb.activate(workdir)
        assert sys.path[0] == src_dir
        assert sys.path.count(src_dir) == 1
        parts = os.environ["PYTHONPATH"].split(os.pathsep)
        assert parts[0] == src_dir
        assert parts.count(src_dir) == 1
        assert other in parts  # 他エントリは保持（除去するのは staged src の重複だけ）
        assert other in sys.path  # sys.path 側の他エントリも保持
    finally:
        sys.path[:] = original_sys_path


def test_emergency_cell_refusal_of_existing_workdir_is_dynamic(fake_source, tmp_path):
    """緊急セルの no-reuse を**実行して**固定する（static substring 検査の補完）。
    seam（ICSR8_EMERGENCY_ALLOW_NONCONTENT_WORKDIR=1）により拒否分岐へ
    ローカルでも到達できる。既存 workdir は path も内容も完全不変であること。"""
    mount_log = tmp_path / "mount_calls.json"
    preamble = _build_colab_stub_preamble(mount_log)
    script = preamble + "\n" + EMERGENCY_CELL_SOURCE

    workdir = tmp_path / "preexisting_work"
    workdir.mkdir()
    sentinel = workdir / "sentinel.txt"
    sentinel.write_text("do not touch\n", encoding="utf-8")

    env = {k: v for k, v in os.environ.items() if not k.startswith("ICSR8_")}
    env["ICSR8_REPO_SOURCE"] = str(fake_source)
    env["ICSR8_WORKDIR"] = str(workdir)
    env["ICSR8_EMERGENCY_ALLOW_NONCONTENT_WORKDIR"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60
    )

    assert result.returncode != 0
    assert "再利用しない" in result.stderr
    assert sorted(p.name for p in workdir.iterdir()) == ["sentinel.txt"]
    assert sentinel.read_text(encoding="utf-8") == "do not touch\n"


def test_emergency_cell_rejects_symlink_in_source(fake_source, tmp_path):
    """緊急セルの symlink 拒否を実行して固定する（COLAB.md の「検査系は本体と
    同等」という durable な主張の regression テスト）。"""
    mount_log = tmp_path / "mount_calls.json"
    preamble = _build_colab_stub_preamble(mount_log)
    script = preamble + "\n" + EMERGENCY_CELL_SOURCE

    outside = tmp_path / "outside.txt"
    outside.write_text("outside bytes\n", encoding="utf-8")
    (fake_source / "sneaky_link.txt").symlink_to(outside)

    workdir = tmp_path / "never_staged"
    env = {k: v for k, v in os.environ.items() if not k.startswith("ICSR8_")}
    env["ICSR8_REPO_SOURCE"] = str(fake_source)
    env["ICSR8_WORKDIR"] = str(workdir)
    env["ICSR8_EMERGENCY_ALLOW_NONCONTENT_WORKDIR"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60
    )

    assert result.returncode != 0
    assert "symlink" in result.stderr
    assert not workdir.exists()


def test_emergency_cell_detects_induced_mid_copy_mutation(fake_source, tmp_path):
    """緊急セルの mid-copy 変異検出を、builtins.open のラッパーで**決定的に**
    誘発して固定する: 特定ファイルの読み出し内容を open のたびに変える →
    F_before と copy 後の再 manifest が一致しない → RuntimeError。"""
    mount_log = tmp_path / "mount_calls.json"
    preamble = _build_colab_stub_preamble(mount_log)
    mutant = fake_source / "mutant.txt"
    mutant.write_text("gen-0\n", encoding="utf-8")
    inject = (
        "\nimport builtins\n"
        "_orig_open = builtins.open\n"
        "_counter = {'n': 0}\n"
        "def _mutating_open(file, *a, **kw):\n"
        "    import io, os\n"
        "    if isinstance(file, (str, bytes, os.PathLike)) and str(file).endswith('mutant.txt') and (not a or 'r' in str(a[0])):\n"
        "        _counter['n'] += 1\n"
        "        return io.BytesIO(f'gen-{_counter[\"n\"]}\\n'.encode())\n"
        "    return _orig_open(file, *a, **kw)\n"
        "builtins.open = _mutating_open\n"
    )
    script = preamble + inject + "\n" + EMERGENCY_CELL_SOURCE

    workdir = tmp_path / "mutation_work"
    env = {k: v for k, v in os.environ.items() if not k.startswith("ICSR8_")}
    env["ICSR8_REPO_SOURCE"] = str(fake_source)
    env["ICSR8_WORKDIR"] = str(workdir)
    env["ICSR8_EMERGENCY_ALLOW_NONCONTENT_WORKDIR"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60
    )

    assert result.returncode != 0
    assert "mid-copy" in result.stderr or "変化した" in result.stderr
    assert not workdir.exists(), "検出時は promote 前なので workdir は作られない"
