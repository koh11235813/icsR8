"""notebooks/*.ipynb の Colab bootstrap セル不変条件の契約テスト。

stdlib ``json`` のみで notebook を読む（nbformat パッケージには依存しない —
本テストは JSON としての形状のみを検査すれば十分で、依存を増やす理由が無い）。

背景・不変条件の設計判断は docs/adr/0003-colab-bootstrap-isolation.md を参照。

tests/test_colab_bootstrap.py の ``SETUP_CELL_SOURCE`` は
notebooks/baseline_reproduction.ipynb を単一の真実源として読み込む。
ここでの「5 冊完全一致」検査が、その単一真実源化の前提——
baseline を読めば残り 4 冊も同じ内容である——を保証している。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# 設計判断（ADR-0003 決定 9）: フォールバック無しの共通ブートストラップ + 同一セル
# 1 個を 5 冊に複製する。順序はユーザー向け README/COLAB.md の記載順に合わせる。
NOTEBOOKS = (
    "baseline_reproduction.ipynb",
    "tier1_methods.ipynb",
    "tier2_methods.ipynb",
    "tier3_methods.ipynb",
    "tier4_methods.ipynb",
)


def _load_notebook(name: str) -> dict:
    path = REPO_ROOT / "notebooks" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _first_code_cell_index(cells: list[dict]) -> int:
    """最初の code セルの index を返す（先頭の markdown セル群はそのまま残す
    契約なので、単純に cells[0] を仮定できない）。"""
    return next(i for i, c in enumerate(cells) if c["cell_type"] == "code")


def _cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_first_code_cell_is_the_bootstrap_setup_cell(name):
    """各冊の最初の code セルが Colab bootstrap セルであること。

    baseline_reproduction は改修前、第 1 code セルで直接 `from icsr8...` する
    構成だった（裏取り済み事実）ため、setup セルがそれより前に無いと Colab 上で
    import エラーになる。5 冊とも「最初の code セル」に固定するのはこのため。
    """
    nb = _load_notebook(name)
    idx = _first_code_cell_index(nb["cells"])
    source = _cell_source(nb["cells"][idx])
    assert "DRIVE_REPO_DIR" in source
    assert "colab_bootstrap" in source


def test_setup_cell_source_identical_across_all_five_notebooks():
    """5 冊の setup セルは同一セル 1 個を複製したもの（ADR-0003 決定 9）。

    tests/test_colab_bootstrap.py の SETUP_CELL_SOURCE は
    notebooks/baseline_reproduction.ipynb だけを読む単一真実源方式なので、
    この一致性が崩れると他 4 冊の contract テストが実質的に無検査になる。
    """
    sources = {}
    for name in NOTEBOOKS:
        nb = _load_notebook(name)
        idx = _first_code_cell_index(nb["cells"])
        sources[name] = _cell_source(nb["cells"][idx])
    unique_sources = set(sources.values())
    assert len(unique_sources) == 1, f"setup セルの source が notebook 間で不一致: {sources.keys()}"


def test_setup_cell_precedes_tier1_mplbackend_cell():
    """tier1_methods.ipynb では MPLBACKEND を inline に固定するセルより setup
    セルが先行しなければならない。

    harness.py は import 時に `os.environ.setdefault("MPLBACKEND", "Agg")` する
    ため、tier1 の MPLBACKEND セルは icsr8 import 前に inline backend を強制する
    不変条件を持つ（裏取り済み事実）。setup セルが「最初の code セル」であれば
    自動的に成立するが、この不変条件は独立にも固定しておく。
    """
    nb = _load_notebook("tier1_methods.ipynb")
    cells = nb["cells"]
    setup_idx = _first_code_cell_index(cells)
    mplbackend_idx = next(
        i
        for i, c in enumerate(cells)
        if c["cell_type"] == "code" and "MPLBACKEND" in _cell_source(c)
    )
    assert setup_idx < mplbackend_idx


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_setup_cell_has_no_stale_outputs_or_execution_count(name):
    """挿入された setup セルは未実行状態でコミットする契約。

    既存セルは outputs/実測タイミングを意図的に保持してコミットする方針
    （裏取り済み事実）だが、新規挿入する setup セルはそれとは区別され、
    outputs==[] ∧ execution_count==null でなければならない（クリーン状態でコミットする契約）。
    """
    nb = _load_notebook(name)
    idx = _first_code_cell_index(nb["cells"])
    cell = nb["cells"][idx]
    assert cell["outputs"] == []
    assert cell["execution_count"] is None


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_os_chdir_call_confined_to_setup_cell(name):
    """os.chdir(...) の実呼び出しは setup セル以外のどのセルにも存在しない。

    検索対象を呼び出し構文 "os.chdir(" に限定する。複数の tier notebook の
    markdown/code セルには「os.chdir は使わない」という既存の説明コメントが
    含まれており、素の部分文字列 "os.chdir" で検査すると、その既存ドキュメント
    セルを誤検出してしまう（setup セル自身も colab_bootstrap.activate() 内で
    間接的に chdir するだけで、セル文面には呼び出し構文が現れない）。
    """
    nb = _load_notebook(name)
    cells = nb["cells"]
    setup_idx = _first_code_cell_index(cells)
    for i, cell in enumerate(cells):
        if i == setup_idx:
            continue
        assert "os.chdir(" not in _cell_source(cell), f"cell {i} calls os.chdir"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_no_cell_writes_files(name):
    """notebook は disk に何も書かない契約（裏取り済み事実）を固定する。

    to_csv(/savefig( の呼び出し構文がどのセルにも無いことをピンする —
    将来どちらかの notebook が結果を勝手に書き出す退行を検出する。
    """
    nb = _load_notebook(name)
    for i, cell in enumerate(nb["cells"]):
        src = _cell_source(cell)
        assert ".to_csv(" not in src, f"cell {i} writes a csv file"
        assert ".savefig(" not in src, f"cell {i} writes a figure file"
