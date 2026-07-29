"""icsr8.report（本文 Tier 1–3 + 付録 A の再生成 API）の契約テスト。

sanctioned writer が凍結パスへ実際に書けることの e2e 相当検証は
tests/test_harness_tier4.py::test_regenerate_main_body_writes_frozen_paths が
既に持つ（_guard_frozen を本物のまま・real repo root で通す構成）。ここでは
それと重複しない 2 つの契約に絞る:

1. MAIN_BODY_METHODS / APPENDIX_A_METHODS の形（個数・重複無し・registry 登録済み）
2. regenerate_main_body() が正しい引数（methods リスト・seed）で
   run_protocol_a/run_lolo/_write_method_diagnostics を呼ぶこと（orchestration の
   配線ミス — 例えば method リストを渡し忘れる・順序を混同する、といった
   誤りを検出する）
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
        return []

    def _fake_make_tex_tables(*args, **kwargs):
        captured["make_tex_tables_called"] = True
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
