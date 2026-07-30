"""verify_report.py の実行時 failure 経路をテストする。

scripts/ は package ではないので import 経由の unit test は sys.path 経由で行う。
既存 tests/test_harness_tier4.py::test_frozen_pdf_hashes_manifest_covers_frozen_pdfs
は allowlist と json 内容の静的一致だけを pin しているため、verify_report.py 側の
実際の control flow (特に `--skip-pdf-hash` の early-return 経路と manifest 不在
failure の interaction) が回帰したときに検出できない。ここでは runtime を実際に
呼んで failure list に期待メッセージが積まれることを assert する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"


@pytest.fixture
def verify_report_module(monkeypatch):
    """scripts/verify_report を fresh module として import し、failures を空に初期化する。

    verify_report は module-level `failures` list を持つ side-effect based の設計
    なので、テスト間の状態汚染を避けるため fixture で毎回 reload する。
    """
    monkeypatch.syspath_prepend(str(_SCRIPTS))
    # 既に import 済みなら reload して failures を空に戻す
    if "verify_report" in sys.modules:
        del sys.modules["verify_report"]
    import verify_report  # noqa: PLC0415 - dynamic import が意図
    yield verify_report
    # test 後の cleanup: 他 test が偶発的に import しても失敗しないよう module を捨てる
    if "verify_report" in sys.modules:
        del sys.modules["verify_report"]


def test_verify_pdf_hashes_absent_manifest_records_failure(
    verify_report_module, monkeypatch, tmp_path
):
    """manifest ファイル不在時、`--skip-pdf-hash` 経由でも failure が記録される。

    # 2026-07-30: 旧実装は verify_pdf_hashes(skip=True) の early-return が
    # manifest 存在チェックより前にあり、「manifest 削除 + skip」で silent pass
    # する経路になっていた。存在判定を _verify_pdf_hash_manifest_coverage() 側
    # (single source of truth) に一本化した後の回帰テスト — 実際に verify_report
    # module の runtime を呼び、failures list に期待メッセージが積まれることまで
    # pin する (test_frozen_pdf_hashes_manifest_covers_frozen_pdfs は静的な
    # allowlist↔json 一致しか見ないので、この経路は独立に検査する必要がある)。
    """
    # manifest を存在しない path に差し替える (実ファイルは触らない)
    fake_manifest = tmp_path / "nonexistent_frozen_pdf_hashes.json"
    monkeypatch.setattr(verify_report_module, "FROZEN_PDF_HASHES_JSON", fake_manifest)

    verify_report_module.verify_pdf_hashes(skip=True)

    assert any(
        "hash 一覧 json が存在しない" in msg
        for msg in verify_report_module.failures
    ), f"expected absent-manifest failure to be recorded, got: {verify_report_module.failures}"


def test_verify_pdf_hashes_absent_manifest_without_skip_also_fails(
    verify_report_module, monkeypatch, tmp_path
):
    """manifest ファイル不在時、`--skip-pdf-hash` を渡さなくても failure 記録される。

    上の test は skip=True 経路 (silent pass 対象だった) を pin する。こちらは
    skip=False (通常経路) でも同じ failure 記録に落ちることを pin し、両経路が
    coverage 関数の single source of truth に集約されていることを確認する。
    """
    fake_manifest = tmp_path / "nonexistent_frozen_pdf_hashes.json"
    monkeypatch.setattr(verify_report_module, "FROZEN_PDF_HASHES_JSON", fake_manifest)

    verify_report_module.verify_pdf_hashes(skip=False)

    assert any(
        "hash 一覧 json が存在しない" in msg
        for msg in verify_report_module.failures
    ), f"expected absent-manifest failure to be recorded, got: {verify_report_module.failures}"
