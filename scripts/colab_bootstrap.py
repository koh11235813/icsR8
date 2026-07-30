"""icsR8 を Google Colab 上で動かすための自己完結ブートストラップ。

このモジュールは ``src/icsr8`` パッケージの**外**に置かれている。理由は
``src/icsr8/__init__.py`` が corridor/estimators/evaluate/fingerprint/methods
等をまとめて import し、``methods/__init__.py`` はさらに全 method module を
自動 import するため、icsr8 パッケージのどの import も「全体ロード」を
誘発してしまうこと。Drive 上のリポは read-only 前提で扱いたいので、
「作業コピー（stage）を作ってから初めて icsr8 を import する」という順序を
守る必要があり、そのためには bootstrap 自体が icsr8 に依存できない。

そのため本モジュールのトップレベルは **stdlib のみ** に限定する。
``google.colab`` / ``matplotlib`` / ``subprocess`` / ``pandas`` はすべて
使用する関数の内部でだけ import する。``icsr8`` はこのモジュールのどこからも
import しない（stage 完了前に import してしまうと read-only 前提が壊れる）。

Colab のノートブックセルは ``compile()``/``exec()`` でこのファイルをロードする
（``import`` ではない）。これは ``__pycache__/*.pyc`` を Drive 上のリポフォルダに
書き込まないための意図的な選択。詳細は docs/COLAB.md と
docs/adr/0003-colab-bootstrap-isolation.md を参照。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

#: staging/reuse manifest 用の除外ディレクトリ名（source_manifest と
#: _copytree_excluding の両方に共通適用する。片方だけ除外すると
#: 「manifest には無いのに copy には残る」ようなズレが生じるため、
#: 除外規則は _iter_source_files に一本化してここから参照する）。
#: "*.pyc" はディレクトリ名ではなくファイル拡張子なので別途 _iter_source_files
#: 内でサフィックス判定する。
EXCLUDES: frozenset[str] = frozenset(
    {".git", ".venv", "__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".ruff_cache"}
)

#: リポジトリ判定の sentinel 3 点。(相対パス, 種別) の組で、種別は
#: "file"（is_file）/ "dir"（is_dir）。3 点すべて満たすディレクトリのみを
#: icsR8 リポとして受理する（README/CLAUDE.md が定義する契約）。
SENTINELS: tuple[tuple[str, str], ...] = (
    ("pyproject.toml", "file"),
    ("src/icsr8", "dir"),
    ("data/dataset", "dir"),
)

#: 所有権マーカーファイル名。source root 直下にこの名前のファイルが
#: 既に存在する場合、staging を拒否する（衝突検出。ADR-0003 Amendment 参照）。
MARKER_FILENAME = ".icsr8_stage.json"
MARKER_MAGIC = "icsr8-colab-stage"
MARKER_SCHEMA = 1
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

#: Colab 上で workdir の直接の親として許可されるディレクトリ。
_CONTENT_ROOT = Path("/content")
#: /content 直下でも workdir として使うと危険な予約名（Drive read-only 前提を
#: 壊す・evidence 収集ディレクトリと衝突する、等）。
_RESERVED_CONTENT_CHILDREN = frozenset({"drive", "evidence"})


# ---------------------------------------------------------------------------
# ランタイム判定・リポ探索
# ---------------------------------------------------------------------------


def is_colab() -> bool:
    """Colab ランタイム上で実行されているかを判定する。

    ``google.colab`` は Colab ランタイムにのみ存在するパッケージなので、
    import 可否だけで判定できる。matplotlib 同様、このモジュールの
    トップレベルでは import せずここでだけ試みる。
    """
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def looks_like_repo(p: Path) -> bool:
    """`p` が icsR8 リポジトリのルートに見えるかを sentinel 3 点で判定する。

    3 点すべて（pyproject.toml が file・src/icsr8 と data/dataset が dir）を
    満たすときのみ True。1 点でも欠けたら False（部分的な展開・壊れた
    コピーを弾く）。
    """
    p = Path(p)
    for rel, kind in SENTINELS:
        target = p / rel
        if kind == "file":
            if not target.is_file():
                return False
        elif kind == "dir":
            if not target.is_dir():
                return False
        else:  # pragma: no cover - SENTINELS は上で固定済み、防御的分岐
            return False
    return True


def find_repo_source(preferred: Path | None) -> Path:
    """icsR8 リポのソースパスを権威順位に従って解決する。

    優先順位:
    1. 環境変数 ``ICSR8_REPO_SOURCE`` が設定されている場合、そのパスが
       **唯一の正**。sentinel が不成立なら mount も自動探索もせず即エラー
       にする（検証時の provenance 保証 — 「どこから読んだか」を曖昧にしない）。
    2. 未設定の場合、`preferred`（呼び出し側が渡す既定パス。ノートブックの
       ``DRIVE_REPO_DIR`` セル定数に相当）が sentinel を満たせばそれを使う。
    3. `preferred` が未指定または不成立なら、Colab の ``google.colab.drive``
       で Drive をマウントし、MyDrive/icsR8・Shareddrives/*/icsR8・
       Shareddrives/*/*/icsR8 を自動探索する。候補が 0 件または複数件なら
       列挙付きエラーにする（曖昧な自動選択をしない）。
    """
    env_source = os.environ.get("ICSR8_REPO_SOURCE")
    if env_source is not None:
        candidate = Path(env_source)
        if not looks_like_repo(candidate):
            raise FileNotFoundError(
                f"ICSR8_REPO_SOURCE={env_source} はリポとして不成立（sentinel 欠落: "
                f"pyproject.toml / src/icsr8 / data/dataset のいずれかが無い）。"
                "転送・展開の失敗を疑って。env 設定時は mount も自動探索もしない。"
            )
        return candidate

    if preferred is not None and looks_like_repo(Path(preferred)):
        return Path(preferred)

    from google.colab import drive  # Colab 限定。preferred 不成立時のみ到達する。

    drive.mount("/content/drive")
    drive_root = Path("/content/drive")
    candidates = sorted(
        {
            str(c)
            for c in (
                Path(preferred) if preferred is not None else None,
                drive_root / "MyDrive" / "icsR8",
                *(drive_root / "Shareddrives").glob("*/icsR8"),
                *(drive_root / "Shareddrives").glob("*/*/icsR8"),
            )
            if c is not None and looks_like_repo(c)
        }
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"リポ候補が {len(candidates)} 件: {candidates} — "
            "DRIVE_REPO_DIR を実配置へ編集して再実行して（リンク共有フォルダは "
            "MyDrive へのショートカット追加が必要な場合がある。docs/COLAB.md 参照）。"
        )
    return Path(candidates[0])


# ---------------------------------------------------------------------------
# manifest（staging/reuse 用と read-only 監査用の 2 種）
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """通常ファイル 1 個の sha256 16進 digest。大きいファイルでもメモリに全体を
    載せないようチャンク読みする（data/rawdata 配下は数十 MB の CSV を含む）。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_source_files(root: Path):
    """`root` 配下を EXCLUDES 適用でスキャンし、通常ファイルの絶対 Path を
    relpath 昇順で yield する唯一の実装。

    source_manifest と _copytree_excluding の両方がここを呼ぶ — 除外規則
    （EXCLUDES ディレクトリ名・*.pyc サフィックス）を 2 箇所に別々に書くと
    「manifest には無いのに copy には残る」ようなズレが生じ得るため、
    共有点をここ 1 箇所に絞っている。

    symlink・特殊ファイル（FIFO/socket/device 等）は EXCLUDES 対象で
    スキップされない限りここで ValueError にする。現リポに tracked symlink は
    無いが、将来紛れ込んだ場合の防御として manifest 計算・copy の両方を
    ここで一律に拒否する。
    """
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        kept_dirnames = []
        for name in dirnames:
            if name in EXCLUDES:
                continue
            full = Path(dirpath) / name
            st = full.lstat()
            if stat.S_ISLNK(st.st_mode):
                raise ValueError(f"symlink ディレクトリは非対応: {full}")
            kept_dirnames.append(name)
        dirnames[:] = kept_dirnames  # os.walk はこの in-place 変更で降下先を絞る

        for name in sorted(filenames):
            if name in EXCLUDES or name.endswith(".pyc"):
                continue
            full = Path(dirpath) / name
            st = full.lstat()
            if stat.S_ISLNK(st.st_mode):
                raise ValueError(f"symlink ファイルは非対応: {full}")
            if not stat.S_ISREG(st.st_mode):
                raise ValueError(f"通常ファイル/ディレクトリ以外は非対応: {full}")
            yield full


def source_manifest(root: Path) -> dict[str, str]:
    """staging/reuse 判定用の manifest。EXCLUDES を適用し、relpath(posix) ->
    sha256 の辞書を relpath 昇順で返す。
    """
    root = Path(root)
    manifest = {full.relative_to(root).as_posix(): _sha256_file(full) for full in _iter_source_files(root)}
    return dict(sorted(manifest.items()))


def manifest_digest(manifest: dict[str, str]) -> str:
    """manifest 全体の単一 digest。所有権マーカーの `digest` フィールドに使う。

    キー順に依存しないよう ``sort_keys=True`` の canonical JSON へシリアライズ
    してから sha256 する（manifest 自体は既に sorted dict だが、呼び出し側が
    順不同の dict を渡しても結果が安定するようにするための二重の安全策）。
    """
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audit_manifest(root: Path) -> dict[str, tuple[str, str | None]]:
    """read-only 監査用の manifest（実機検証 専用）。**EXCLUDES を適用しない**。

    lstat（symlink を追跡しない）で全 entry を走査し、
    relpath -> (entry種別, 通常ファイルなら sha256 / それ以外は None) を返す。
    entry 種別は "file" / "dir" / "symlink" / "other" のいずれか。

    staging/reuse manifest（source_manifest）と違い、こちらは symlink や
    特殊ファイルを見つけても **拒否しない**（記録するだけ）。目的が逆だから:
    source_manifest は「安全に copy できる状態か」を検査するが、
    audit_manifest は「Drive 側が read-only のまま保たれているか」
    （.pyc 等の除外対象が新規生成されていないか、entry が消えていないか）を
    展開直後と全工程終了後で比較するためのものなので、型変化そのものを
    検出できる必要がある。
    """
    root = Path(root)
    result: dict[str, tuple[str, str | None]] = {}

    def _classify(st: os.stat_result) -> str:
        if stat.S_ISREG(st.st_mode):
            return "file"
        if stat.S_ISDIR(st.st_mode):
            return "dir"
        if stat.S_ISLNK(st.st_mode):
            return "symlink"
        return "other"

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in dirnames:
            full = Path(dirpath) / name
            rel = full.relative_to(root).as_posix()
            result[rel] = (_classify(full.lstat()), None)
        for name in sorted(filenames):
            full = Path(dirpath) / name
            rel = full.relative_to(root).as_posix()
            st = full.lstat()
            kind = _classify(st)
            result[rel] = (kind, _sha256_file(full) if kind == "file" else None)
    return dict(sorted(result.items()))


def _copytree_excluding(source: Path, dest: Path) -> None:
    """EXCLUDES 適用済みのディレクトリツリーコピー。

    stdlib の ``shutil.copytree(ignore=...)`` は symlink を安全に「拒否」
    できない（``symlinks=False`` は追跡してコピー、``True`` は symlink を
    複製するだけで、どちらも本契約が要求する「拒否」にならない）。そのため
    _iter_source_files の単一実装を手動で walk し、通常ファイルだけを
    ``shutil.copy2`` する。除外規則・symlink 拒否を source_manifest と
    完全に共有できる。
    """
    source = Path(source)
    dest = Path(dest)
    for full in _iter_source_files(source):
        rel = full.relative_to(source)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(full, target)


# ---------------------------------------------------------------------------
# パスガード（fail-closed）
# ---------------------------------------------------------------------------


def guard_paths(source: Path, workdir: Path, *, on_colab: bool | None = None) -> None:
    """source/workdir の組み合わせが安全かを fail-closed で検査する。

    以下のいずれかに該当したら即 ValueError にする:
    1. resolve() 後の source と workdir が同一パス
    2. workdir が source の配下（source を workdir としてステージすると
       source 自体を書き換えてしまう）
    3. source が workdir の配下（意味的に逆転した設定ミス）
    4. workdir が ``/content/drive`` 配下（Drive は read-only 前提。
       そこへ書き込み用の作業コピーを作ってはいけない）
    5. `on_colab` が真のとき、workdir が ``/content`` の直接の子でない、
       または予約名（drive/evidence）である

    `on_colab` はテスト用の**明示注入 seam**（省略時のみ `is_colab()` を
    使う）。暗黙の env 判定（tmp_path や PYTEST_* を見て test-mode を
    推測すること）は禁止 — ローカルテストは `on_colab=False` を明示的に
    渡すことで /content 制限をバイパスする。

    symlink は resolve() が辿って実体パスへ潰すため、symlink 経由で
    このガードを迂回することはできない。
    """
    if on_colab is None:
        on_colab = is_colab()

    resolved_source = Path(source).resolve(strict=False)
    resolved_workdir = Path(workdir).resolve(strict=False)

    if resolved_source == resolved_workdir:
        raise ValueError(f"source と workdir が同一パス: {resolved_source}")

    if resolved_workdir.is_relative_to(resolved_source):
        raise ValueError(
            f"workdir が source の配下にある: {resolved_workdir} は "
            f"{resolved_source} の子孫（source を書き換えてしまうため拒否）"
        )

    if resolved_source.is_relative_to(resolved_workdir):
        raise ValueError(
            f"source が workdir の配下にある: {resolved_source} は "
            f"{resolved_workdir} の子孫（設定が逆転している疑い）"
        )

    drive_root = Path("/content/drive")
    if resolved_workdir.is_relative_to(drive_root):
        raise ValueError(
            f"workdir が /content/drive 配下: {resolved_workdir} — "
            "Drive は read-only 前提なので作業コピーを Drive 側に置けない。"
        )

    if on_colab:
        if (
            resolved_workdir.parent != _CONTENT_ROOT
            or resolved_workdir.name in _RESERVED_CONTENT_CHILDREN
        ):
            raise ValueError(
                f"Colab 上の workdir は /content 直下の安全な子に制限される: "
                f"{resolved_workdir}（予約名 {sorted(_RESERVED_CONTENT_CHILDREN)} も不可）"
            )


# ---------------------------------------------------------------------------
# staging（所有権マーカープロトコル）
# ---------------------------------------------------------------------------


def _read_marker(marker_path: Path) -> dict | None:
    """所有権マーカーを読んで全フィールド検証する。1 つでも欠落・型不正なら
    None（= 非所有/corrupt として扱う。呼び出し側で「拒否」に倒す）。

    マーカー valid の条件は（ADR-0003 決定 10）
    - marker_path がシンボリックリンクではない通常ファイル（lstat で判定。
      symlink 経由でマーカーを差し替える攻撃/事故を防ぐ）
    - JSON として parse でき、dict である
    - magic == "icsr8-colab-stage"（完全一致）
    - schema == 1（int。bool は int のサブクラスなので明示的に除外 —
      True/False が 1/0 と等価判定されて誤って valid になる事故を防ぐ）
    - source が非空 str
    - digest が 64 桁の小文字16進文字列
    """
    try:
        st = marker_path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None  # symlink・特殊ファイルは corrupt 扱い

    try:
        raw = marker_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None
    if data.get("magic") != MARKER_MAGIC:
        return None
    schema = data.get("schema")
    if not isinstance(schema, int) or isinstance(schema, bool) or schema != MARKER_SCHEMA:
        return None
    source_field = data.get("source")
    # source は絶対パス文字列であること（marker validity は「既存 workdir を
    # 退避してよい」という破壊的権限の根拠なので、相対パスや空文字は
    # corrupt/非所有として扱う）。
    if not isinstance(source_field, str) or not os.path.isabs(source_field):
        return None
    digest_field = data.get("digest")
    if not isinstance(digest_field, str) or not _HEX64_RE.match(digest_field):
        return None
    return data


def _stage_new(source: Path, workdir: Path, *, retire_existing: bool) -> Path:
    """新規 stage を「sibling tmp に完全構築+検証してから promote」する
    トランザクションの本体。

    手順（ADR-0003 決定 10 のトランザクション順序）:
    1. workdir.parent を用意し、``tempfile.mkdtemp(dir=workdir.parent)`` で
       同一 filesystem 上の sibling ディレクトリを作る
       （``os.replace`` の atomicity は同一 filesystem が前提）。
    2. F_before = source_manifest(source)
    3. tmp/repo へ copytree
    4. F_after = source_manifest(source)。F_before != F_after なら
       「コピー中に source が変異した」として即エラー（tmp は残置 —
       rollback/フォレンジック用に消さない。EXDEV/コピー失敗時も同様）。
    5. source_manifest(tmp/repo) が F_before の**全項目を含み値が一致する**か
       検査（reuse 時の「派生物許容」と同じ判定にするため、ここでも
       等価ではなく部分集合チェックにしておく — 新規 stage では通常
       完全一致するはずだが、判定ロジックを 1 箇所に共通化する）。
    6. 所有権マーカーを書く。
    7. retire_existing なら既存 workdir を ``.old-<uuid4hex>`` へ退避してから
       promote。promote が失敗したら退避を自動 rollback（workdir を元の
       名前に戻す）。retire_existing でなければそのまま promote。

    どの段階で失敗しても canonical workdir は「元の状態」のまま保たれ、
    tmp ディレクトリだけが残る（削除しない — テストで固定された契約）。
    """
    workdir = Path(workdir)
    workdir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=str(workdir.parent), prefix="icsr8_stage_"))
    staged_repo = tmp_dir / "repo"

    f_before = source_manifest(source)
    _copytree_excluding(source, staged_repo)
    f_after = source_manifest(source)
    if f_before != f_after:
        raise RuntimeError(
            f"mid-copy 変異検出: source ({source}) がコピー中に変化した。"
            f"workdir は変更していない。調査用に {tmp_dir} を残置する。"
        )

    staged_manifest = source_manifest(staged_repo)
    mismatched = [k for k, v in f_before.items() if staged_manifest.get(k) != v]
    if mismatched:
        raise RuntimeError(
            f"staged tree が source manifest と不一致: {mismatched[:5]} 等。"
            f"workdir は変更していない。調査用に {tmp_dir} を残置する。"
        )

    digest = manifest_digest(f_before)
    marker = {
        "magic": MARKER_MAGIC,
        "schema": MARKER_SCHEMA,
        "source": str(source),
        "digest": digest,
    }
    (staged_repo / MARKER_FILENAME).write_text(json.dumps(marker), encoding="utf-8")

    backup = None
    if retire_existing:
        backup = workdir.parent / f"{workdir.name}.old-{uuid.uuid4().hex}"
        os.replace(workdir, backup)  # 退避（削除ではなく rename）

    try:
        os.replace(staged_repo, workdir)
    except OSError:
        if backup is not None:
            os.replace(backup, workdir)  # promote 失敗 → 自動 rollback
        raise

    shutil.rmtree(tmp_dir, ignore_errors=True)  # 成功時のみ: 空になった wrapper を掃除
    return workdir


def stage_working_copy(source: Path, workdir: Path, *, refresh: bool = False) -> Path:
    """source を workdir へ作業コピーとして stage する（所有権マーカー方式）。

    - source root に既に ``.icsr8_stage.json`` が存在する場合は即拒否する
      （マーカー衝突検出。source を workdir と取り違えた
      事故を防ぐ）。
    - workdir が存在しない場合は新規 stage。
    - workdir が存在し、マーカーが**欠落または破損**している場合は
      「非所有」として即時拒否する（退避もしない。`refresh=True` でも
      迂回できない — 正常に作られた workdir は atomic rename 由来なので
      マーカー無しにはならないはずであり、これは「icsr8 が管理していない
      既存ディレクトリ」を誤って上書きする事故を防ぐガード）。
    - マーカーが valid で、source の現在の digest/source と一致し、
      `refresh=False` なら**再利用**する。ただし再利用前に staged tree の
      整合性を検査する: source manifest に載っている全ファイルが staged
      側に存在し hash が一致すること（``results/**`` 等の派生物が
      staged 側に追加されているのは許容 — source 由来ファイルの欠落・
      改変だけを拒否する）。
    - マーカーが valid でも digest/source が不一致、または `refresh=True`
      の場合は、既存 workdir を ``.old-<uuid4hex>`` へ退避してから
      新規 stage する（_stage_new のトランザクションに委譲）。
    """
    source = Path(source).resolve(strict=False)
    workdir = Path(workdir).resolve(strict=False)

    # 2026-07-29: 本関数は公開契約なので、bootstrap() を
    # 経由しない直接呼び出しでも構造的安全性（同一パス・双方向包含・
    # /content/drive 配下 workdir の拒否）を自前で強制する。source 内に
    # tmp を作って自己再帰コピーする事故は、この 1 行が防ぐ。
    # on_colab=False の理由: guard_paths の検査 1〜4（構造検査）は環境に
    # 依存せず常に適用すべき一方、検査 5（workdir を /content 直下に制限する
    # Colab 運用ポリシー）は入口である bootstrap() が自身の guard_paths 呼び出しで
    # 担う。ここで is_colab() を使うと、実 Colab 上で走る staging の
    # トランザクション系テスト（tmp_path を使う）が全滅するため。
    guard_paths(source, workdir, on_colab=False)

    source_marker_path = source / MARKER_FILENAME
    if source_marker_path.exists() or source_marker_path.is_symlink():
        raise ValueError(
            f"source root に {MARKER_FILENAME} が既に存在する: {source_marker_path} — "
            "source と workdir を取り違えていないか確認して。staging を拒否する。"
        )

    if not workdir.exists():
        return _stage_new(source, workdir, retire_existing=False)

    marker = _read_marker(workdir / MARKER_FILENAME)
    if marker is None:
        raise RuntimeError(
            f"{workdir} は所有権マーカーが欠落/破損しており「非所有」として扱う。"
            "icsr8 が作った作業コピーではない可能性が高いため、退避もせず拒否する。"
            "refresh=True でも迂回できない。手動で調査・退避してから再実行して。"
        )

    current_manifest = source_manifest(source)
    current_digest = manifest_digest(current_manifest)
    same_source = marker["source"] == str(source)
    same_digest = marker["digest"] == current_digest

    if same_source and same_digest and not refresh:
        staged_manifest = source_manifest(workdir)
        # 2026-07-29 実機事故対応: 評価パイプライン（regenerate_appendix_a 等）は
        # 正規動作として tracked な results/tier4/run_tier4.log を上書きする。
        # results/ はパイプライン所有の可変領域なので整合性検査の対象から外す（外さないと、
        # スイープ実行後の 2 冊目の notebook で bootstrap の再利用が拒否され、
        # 生成済み results もろとも作業コピーが作り直されてしまう）。
        # 2026-07-30 review pivot: docs/COLAB.md の regen 手順
        # （regenerate_main_body.py / regenerate_appendix_a.py）は staged 作業
        # コピー上の doc/final_report/{tables,figures} も正規動作として書き換える
        # （sanctioned writer — src/icsr8/report.py 参照）。results/ と同じ理由で
        # この 2 ディレクトリも整合性検査の対象から外す。main.tex・main.pdf・
        # .latexmkrc 等（doc/final_report/ 直下や tables/figures 以外）は
        # regen パイプラインが触らない tracked ファイルなので対象外化しない
        # （改変を検出できる状態を維持する）。
        # source の鮮度判定は上の digest 比較（source 側のみ）が引き続き担う。
        _PIPELINE_OWNED_PREFIXES = (
            "results/",
            "doc/final_report/tables/",
            "doc/final_report/figures/",
        )
        mismatched = [
            k for k, v in current_manifest.items()
            if not k.startswith(_PIPELINE_OWNED_PREFIXES) and staged_manifest.get(k) != v
        ]
        if mismatched:
            raise RuntimeError(
                f"再利用前の staged tree 整合性検査に失敗: {mismatched[:5]} 等が "
                f"source 側と一致しない。{workdir} を調査して。"
            )
        return workdir

    return _stage_new(source, workdir, retire_existing=True)


# ---------------------------------------------------------------------------
# activate / フォント / bootstrap 本体
# ---------------------------------------------------------------------------


def activate(repo_root: Path) -> None:
    """staged リポを実行対象として有効化する: sys.path 先頭に src を挿入し、
    ``PYTHONPATH`` にも同じパスを前置し、cwd を repo_root へ変更する。

    ``!python`` で起動する子プロセスは sys.path を継承しないため、
    ``PYTHONPATH`` の前置が別途必要（sys.path 挿入だけでは効かない）。

    冪等かつ**常に先頭**: staged src が既に sys.path/PYTHONPATH の途中に
    入っている場合も、既存の出現を除去してから先頭へ 1 回だけ挿入し直す。
    # 2026-07-29: 「存在すれば何もしない」旧実装は、staged src より前に別の
    # ツリー（例: 別リポの src）があると子プロセスがそちらを先に import する
    # provenance 欠陥だった。activate の契約は「staged src を必ず最優先」。
    """
    repo_root = Path(repo_root)
    src_dir = str((repo_root / "src").resolve(strict=False))

    sys.path[:] = [p for p in sys.path if p != src_dir]
    sys.path.insert(0, src_dir)

    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in (existing.split(os.pathsep) if existing else []) if p != src_dir]
    os.environ["PYTHONPATH"] = os.pathsep.join([src_dir, *parts]) if parts else src_dir

    os.chdir(repo_root)


def install_cjk_fonts() -> None:
    """Colab 上で日本語（CJK）フォントを matplotlib に登録する。Colab 以外では
    no-op。

    ``matplotlib.font_manager`` だけを使い、**``matplotlib.pyplot`` は
    import しない**。pyplot を最初に import した瞬間に backend が確定して
    しまうため（harness.py:18 が import 時に ``MPLBACKEND=Agg`` を
    setdefault しているのはこれの回避策で、ここで pyplot を触ると
    その前提条件と衝突しうる）。素の matplotlib/font_manager の import は
    backend を確定させないので安全。tier1 notebook の ``%matplotlib inline``
    が二重の防御になっている。
    """
    if not is_colab():
        return

    import subprocess

    subprocess.run(
        ["apt-get", "-qq", "install", "-y", "--no-install-recommends", "fonts-noto-cjk"],
        check=True,
    )

    import matplotlib
    import matplotlib.font_manager as font_manager

    noto_cjk_paths = [
        p
        for p in (
            font_manager.findSystemFonts(fontext="ttf")
            + font_manager.findSystemFonts(fontext="otf")
        )
        if "NotoSansCJK" in p or "NotoSerifCJK" in p
    ]
    for path in noto_cjk_paths:
        font_manager.fontManager.addfont(path)

    if noto_cjk_paths:
        added_names = list(
            dict.fromkeys(font_manager.FontProperties(fname=p).get_name() for p in noto_cjk_paths)
        )
        current = matplotlib.rcParams.get("font.family", [])
        if isinstance(current, str):
            current = [current]
        # CJK フォントを候補の先頭に差し込む。既存設定は消さず後ろに残す
        # （非日本語グラフの見た目を変えないため）。
        matplotlib.rcParams["font.family"] = added_names + [c for c in current if c not in added_names]


def _find_repo_root_from_cwd() -> Path:
    """非 Colab 環境向けの純粋 no-op 探索: cwd から上へ辿り、
    ``data/dataset`` を持つ最初の祖先を返す。副作用（stage/chdir/fonts 等）は
    一切起こさない。
    """
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "data" / "dataset").is_dir():
            return candidate
    raise FileNotFoundError(
        "data/dataset を持つ祖先ディレクトリが見つからない — "
        "icsR8 リポジトリ内（またはその子孫ディレクトリ）で実行して。"
    )


def bootstrap(
    source: Path | None = None,
    *,
    workdir: Path | None = None,
    install_fonts: bool = True,
    refresh: bool = False,
) -> Path:
    """Colab 互換化のエントリポイント。

    非 Colab: 副作用ゼロ。cwd から ``data/dataset`` を持つ祖先を探して
    返すだけ（ローカル Jupyter 等で誤って呼ばれても何も壊さない）。

    Colab: 以下の順で実行し、実行基準となる root を返す。
    1. already-imported ガード: ``sys.modules`` に ``icsr8`` または
       ``icsr8.`` で始まる名前が残っていたら RuntimeError（refresh で
       ファイルを差し替えても、既に import 済みのオブジェクトはメモリ上に
       残ったままになるため。kernel restart を促すメッセージにする）。
    2. `find_repo_source` でソース解決
    3. `guard_paths` で安全性検査
    4. `stage_working_copy` で作業コピーを用意
    5. `activate` で sys.path/PYTHONPATH/chdir
    6. `install_fonts` が真なら `install_cjk_fonts`
    """
    if not is_colab():
        return _find_repo_root_from_cwd()

    for name in list(sys.modules):
        if name == "icsr8" or name.startswith("icsr8."):
            raise RuntimeError(
                "icsr8 は既にこのプロセスに import 済み。refresh でファイルを"
                "差し替えても import 済みオブジェクトはメモリ上に残るため、"
                "ランタイムを再起動（Runtime > Restart runtime）してから"
                "もう一度 bootstrap を実行して。"
            )

    resolved_source = find_repo_source(source)
    resolved_workdir = Path(workdir) if workdir is not None else Path(
        os.environ.get("ICSR8_WORKDIR", "/content/icsr8_work")
    )
    guard_paths(resolved_source, resolved_workdir)
    staged_root = stage_working_copy(resolved_source, resolved_workdir, refresh=refresh)
    activate(staged_root)
    if install_fonts:
        install_cjk_fonts()
    return staged_root


#: ファイルごとの一意キー列（検証契約の literal 定義）。diagnostics.csv は tier4 側の
#: 出力で ``(protocol, fold, method, key)`` が複合キーになる。
_OUTPUT_KEY_COLUMNS: dict[str, tuple[str, ...]] = {
    "protocol_a.csv": ("method", "fold"),
    "protocol_a_ledger.csv": ("method", "fold", "location_p"),
    "lolo_ledger.csv": ("method", "held_out"),
    "lolo_summary.csv": ("method",),
    "diagnostics.csv": ("protocol", "fold", "method", "key"),
}


def validate_colab_outputs(
    directory: Path, *, numeric_columns: dict[str, tuple[str, ...]] | None = None
) -> dict[str, dict]:
    """`directory` 直下にある既知の CSV ファイルを検査し、事実を返す。

    このモジュールと tests/test_colab_bootstrap.py・実機検証 の実行ドライバが
    **同一実装**を呼ぶことで検査ロジックの乖離を防ぐ。

    行う検査:
    - `_OUTPUT_KEY_COLUMNS` に載っているファイルが存在すれば、その複合キー
      列がファイル内で一意であることを強制する（重複していたら ValueError。
      重複したキーの内容もメッセージに含める）。
    - ``failed`` 列があれば全行 False であることを強制する（本文/tier4 の
      fail-soft 契約: 手法が失敗しても行は残るが `failed=True` になる。
      smoke/full ともに「1 件も失敗していない」ことを検査したい）。
    - `numeric_columns` で明示された列（ファイル名 -> 列名のタプル）は
      全値が有限（NaN/inf でない）であることを強制する。tier4 の
      diagnostics.csv は ``value`` 列が文字列と数値の混在型なので、
      「数値であるべき列」を dtype から自動判定できず、呼び出し側が
      明示列挙する契約にしている。

    行数・method 集合・期待ファイル集合などの**具体的な数値**は意図的に
    ここでは決め打ちしない（本体定数からオラクルを作ると自己参照的に
    なるため、そこは呼び出し側/テストの literal で判定する）。この関数は
    観測事実（行数・列名・method 集合）を返すので、呼び出し側がそれを
    期待値と突き合わせる。
    """
    import math

    import pandas as pd  # lazy: モジュールトップは stdlib のみに保つ契約

    directory = Path(directory)
    facts: dict[str, dict] = {}

    for filename, key_cols in _OUTPUT_KEY_COLUMNS.items():
        path = directory / filename
        if not path.is_file():
            continue

        df = pd.read_csv(path)

        missing_key_cols = [c for c in key_cols if c not in df.columns]
        if missing_key_cols:
            raise ValueError(f"{filename}: 複合キー列が無い: {missing_key_cols}")

        dup_mask = df.duplicated(subset=list(key_cols), keep=False)
        if dup_mask.any():
            dup_rows = df.loc[dup_mask, list(key_cols)].drop_duplicates().to_dict("records")
            raise ValueError(f"{filename}: 複合キー {key_cols} が重複している: {dup_rows}")

        if "failed" in df.columns and df["failed"].any():
            failed_rows = df.loc[df["failed"].astype(bool), list(key_cols)].to_dict("records")
            raise ValueError(f"{filename}: failed=True の行がある: {failed_rows}")

        if numeric_columns and filename in numeric_columns:
            for col in numeric_columns[filename]:
                if col not in df.columns:
                    raise ValueError(f"{filename}: 数値列指定だが列が無い: {col}")
                is_finite = df[col].map(
                    lambda v: v is not None and not (isinstance(v, float) and math.isnan(v)) and math.isfinite(float(v))
                )
                if not is_finite.all():
                    bad = df.loc[~is_finite, list(key_cols) + [col]].to_dict("records")
                    raise ValueError(f"{filename}: 列 {col} に非有限値がある: {bad[:5]} 等")

        facts[filename] = {
            "rows": len(df),
            "columns": list(df.columns),
            "methods": sorted(df["method"].unique().tolist()) if "method" in df.columns else None,
        }

    return facts


# ---------------------------------------------------------------------------
# --check 用: 依存フロア比較（数値ドット prefix の簡易規約）
# ---------------------------------------------------------------------------


def _numeric_dot_prefix(version_str: str) -> tuple[int, ...]:
    """バージョン文字列の先頭から連続する数値セグメントだけを
    ``tuple[int, ...]`` にする簡易 PEP440 もどきの比較規約（既知の制約:
    厳密な PEP440 準拠ではない。'rc1'/'.post1' 等の非数値 suffix に
    出会った時点で打ち切る）。

    例: "3.11.0" -> (3, 11, 0)。"2.1" -> (2, 1)。"3.11.0rc1" -> (3, 11, 0)。
    """
    prefix: list[int] = []
    for segment in version_str.split("."):
        match = re.match(r"\d+", segment)
        if not match:
            break
        prefix.append(int(match.group()))
    return tuple(prefix)


def _parse_floor_requirement(requirement: str) -> tuple[str, tuple[int, ...]] | None:
    """pyproject.toml の dependencies 要素（例 "matplotlib>=3.11.0"）から
    (パッケージ名, floor tuple) を抽出する。">=" 以外の演算子や比較演算子が
    無い要素は None を返す（5 依存は全て ">=" 単一指定である前提。将来
    別演算子が混ざったら黙ってスキップせず気づけるよう None を明示的に返す）。
    """
    match = re.match(r"^([A-Za-z0-9_.\-]+)\s*>=\s*([0-9][0-9A-Za-z.\-]*)", requirement)
    if not match:
        return None
    name, floor_str = match.group(1), match.group(2)
    return name, _numeric_dot_prefix(floor_str)


def _check_dependency_floors(pyproject_path: Path) -> list[str]:
    """pyproject.toml の [project].dependencies（5 件）のフロアと、
    実際に import 可能な配布パッケージのインストール済みバージョンを比較する。

    stdlib のみを使う（``tomllib`` は 3.11+ 標準ライブラリ、
    ``importlib.metadata`` も標準ライブラリ）。返り値は問題メッセージの
    リスト（空 = 全部フロアを満たす）。
    """
    import tomllib
    from importlib import metadata

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = data["project"]["dependencies"]

    problems: list[str] = []
    for requirement in dependencies:
        parsed = _parse_floor_requirement(requirement)
        if parsed is None:
            continue
        package, floor = parsed
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            problems.append(
                f"{package}: 未インストール（floor {'.'.join(map(str, floor))} 以上が必要）"
            )
            continue
        if _numeric_dot_prefix(installed) < floor:
            problems.append(
                f"{package}: installed {installed} < floor {'.'.join(map(str, floor))}"
            )
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``python scripts/colab_bootstrap.py [--source PATH] [--check]``。

    ``--check`` は以下を順に確認する（依存修復は --check の
    前に済ませておく運用を前提とするため、ここではフロア不足を検出したら
    修復せず報告するだけに留める）:
    1. ``sys.version_info >= (3, 11)``
    2. pyproject.toml の 5 依存のフロア照合（stdlib のみ・数値ドット prefix
       の簡易規約）
    3. 上記が通れば ``tests/test_reproduce_baseline.py`` を pytest で実行

    ``--source`` 指定時は `bootstrap` にそのまま渡す（Colab 外では
    `bootstrap` 自体が副作用ゼロの no-op になる）。
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="icsR8 Colab bootstrap — リポの stage・活性化・依存/テスト検査"
    )
    parser.add_argument(
        "--source", default=None, help="リポの明示ソースパス（省略時は env/自動探索に従う）"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Python バージョン・依存フロア・tests/test_reproduce_baseline.py を確認する",
    )
    args = parser.parse_args(argv)

    if args.check:
        problems: list[str] = []
        if sys.version_info < (3, 11):
            problems.append(
                f"Python {sys.version_info.major}.{sys.version_info.minor} — "
                "3.11 以上が必要"
            )

        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        problems.extend(_check_dependency_floors(pyproject_path))

        if problems:
            for problem in problems:
                print(f"[check] NG: {problem}")
            print(
                "[check] matplotlib を upgrade する場合は、フォント設定より前に"
                "実施するか、実施後に kernel を restart してから続行して。"
            )
            return 1

        print("[check] Python バージョン・依存フロア OK")
        test_path = (
            Path(__file__).resolve().parents[1] / "tests" / "test_reproduce_baseline.py"
        )

        import subprocess

        result = subprocess.run([sys.executable, "-m", "pytest", str(test_path), "-q"])
        return result.returncode

    source_arg = Path(args.source) if args.source is not None else None
    root = bootstrap(source_arg)
    print(f"[colab_bootstrap] root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
