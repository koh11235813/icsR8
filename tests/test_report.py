"""icsr8.report（本文 Tier 1–3 + 付録 A の再生成 API）の契約テスト。

sanctioned writer が凍結パスへ実際に書けることの e2e 相当検証は
tests/test_harness_tier4.py::test_regenerate_main_body_writes_frozen_paths が
既に持つ（_guard_frozen を本物のまま・real repo root で通す構成）。ここでは
それと重複しない 3 つの契約に絞る:

1. MAIN_BODY_METHODS / APPENDIX_A_METHODS の形（個数・重複無し・registry 登録済み）
2. regenerate_main_body() が正しい引数（methods リスト・seed・writer_id）で
   run_protocol_a/run_lolo/make_figures/make_tex_tables/
   _write_method_diagnostics を呼ぶこと（orchestration の配線ミス — 例えば
   method リストを渡し忘れる・順序を混同する・writer_id を渡し忘れる、
   といった誤りを検出する）
3. regenerate_appendix_a() が run_tier4() へ正しい引数（methods・references・
   出力先 3 ディレクトリ・seed・B・writer_id）で配線されていること
"""

from __future__ import annotations

import icsr8.report as report
from icsr8.constants import RANDOM_SEED
from icsr8.methods import REGISTRY


def test_main_body_methods_is_15():
    assert len(report.MAIN_BODY_METHODS) == 15
    assert len(set(report.MAIN_BODY_METHODS)) == 15  # 重複が無い


def test_appendix_a_methods_is_7():
    assert len(report.APPENDIX_A_METHODS) == 7
    assert len(set(report.APPENDIX_A_METHODS)) == 7


def test_main_body_and_appendix_a_methods_are_disjoint():
    # 本文と付録 A は別評価系統（README §凍結契約）なので手法集合も排他のはず。
    assert set(report.MAIN_BODY_METHODS).isdisjoint(set(report.APPENDIX_A_METHODS))


def test_main_body_methods_match_registered():
    # MAIN_BODY_METHODS の全要素が icsr8.methods.REGISTRY に登録済みであること
    # （タイプミスや削除済みモジュール名の残存を検出する）。
    assert set(report.MAIN_BODY_METHODS) <= set(REGISTRY)


def test_appendix_a_methods_match_registered():
    assert set(report.APPENDIX_A_METHODS) <= set(REGISTRY)


def test_appendix_a_methods_is_harness_tier4_mirror():
    # report.APPENDIX_A_METHODS は harness_tier4.TIER4_METHODS を re-export した
    # ものであり、literal 複製ではない（単一の定義元）ことを固定する。
    import icsr8.harness_tier4 as harness_tier4

    assert list(report.APPENDIX_A_METHODS) == list(harness_tier4.TIER4_METHODS)


def test_regenerate_main_body_smoke(monkeypatch):
    """regenerate_main_body() の配線を検証する unit test（実 sweep は回さない）。

    実データロード・重い計算（run_protocol_a/run_lolo/make_figures/
    make_tex_tables/診断値書き出し）を全て monkeypatch で no-op に差し替え、
    正しい引数（本文 15 手法のリスト・RANDOM_SEED）が渡ることだけを検証する。
    実 sweep（Colab 実測 ~8 分）を pytest で毎回回すのは非現実的なので、
    ここでは「呼び出し配線が正しいか」に検査範囲を絞る。
    """
    captured: dict = {}

    class _NoWriteFrame:
        def to_csv(self, path, index=False):  # noqa: ARG002
            pass

    def _fake_run_protocol_a(methods, scans_f, scans_b, ap13, truth, seed, B):
        captured["protocol_a_methods"] = methods
        captured["protocol_a_seed"] = seed
        captured["protocol_a_B"] = B
        return _NoWriteFrame(), _NoWriteFrame()

    def _fake_run_lolo(methods, scans_f, scans_b, ap13, truth, seed):
        captured["lolo_methods"] = methods
        captured["lolo_seed"] = seed
        return _NoWriteFrame(), _NoWriteFrame()

    def _fake_make_figures(*args, **kwargs):
        captured["make_figures_called"] = True
        captured["make_figures_writer_id"] = kwargs.get("writer_id")
        return []

    def _fake_make_tex_tables(*args, **kwargs):
        captured["make_tex_tables_called"] = True
        captured["make_tex_tables_writer_id"] = kwargs.get("writer_id")
        return []

    def _fake_write_diag(scans_f, ap13, truth):
        captured["write_diagnostics_called"] = True

    monkeypatch.setattr(report, "_load_data", lambda data_root: (None, None, None, None))
    monkeypatch.setattr(report, "run_protocol_a", _fake_run_protocol_a)
    monkeypatch.setattr(report, "run_lolo", _fake_run_lolo)
    monkeypatch.setattr(report, "make_figures", _fake_make_figures)
    monkeypatch.setattr(report, "make_tex_tables", _fake_make_tex_tables)
    monkeypatch.setattr(report, "_write_method_diagnostics", _fake_write_diag)

    report.regenerate_main_body()

    assert captured["protocol_a_methods"] == list(report.MAIN_BODY_METHODS)
    assert captured["lolo_methods"] == list(report.MAIN_BODY_METHODS)
    assert captured["protocol_a_seed"] == RANDOM_SEED
    assert captured["lolo_seed"] == RANDOM_SEED
    assert captured["protocol_a_B"] == 1000
    assert captured["make_figures_called"]
    assert captured["make_tex_tables_called"]
    assert captured["write_diagnostics_called"]
    # Codex review finding 1（allowlist 押し下げ）: make_figures/make_tex_tables
    # 自身が凍結ガードを持つようになったので、regenerate_main_body() が
    # sanctioned writer 識別子を正しく渡していることも配線契約に含める。
    assert captured["make_figures_writer_id"] == "icsr8.report.regenerate_main_body"
    assert captured["make_tex_tables_writer_id"] == "icsr8.report.regenerate_main_body"


def test_regenerate_appendix_a_wiring(monkeypatch):
    """regenerate_appendix_a() の配線を検証する unit test（実 sweep は回さない）。

    report.py は `from icsr8.harness_tier4 import ... run_tier4` で名前を束縛
    しているため、icsr8.harness_tier4 側を monkeypatch しても report モジュール
    内の呼び出しには反映されない（test_regenerate_main_body_smoke と同じ理由）。
    report モジュールの `run_tier4` 属性を直接差し替えて配線だけを検証する。
    """
    captured: dict = {}

    def _fake_run_tier4(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(report, "_load_data", lambda data_root: (None, None, None, None))
    monkeypatch.setattr(report, "run_tier4", _fake_run_tier4)

    report.regenerate_appendix_a()

    assert captured["methods"] == list(report.APPENDIX_A_METHODS)
    assert captured["references"] == report.REFERENCE_METHODS
    assert captured["output_dir"] == report._REPO_ROOT / "results" / "tier4"
    assert captured["tables_dir"] == report._REPO_ROOT / "doc" / "final_report" / "tables"
    assert captured["figures_dir"] == report._REPO_ROOT / "doc" / "final_report" / "figures"
    assert captured["seed"] == RANDOM_SEED
    assert captured["B"] == 1000
    assert captured["writer_id"] == "icsr8.report.regenerate_appendix_a"
