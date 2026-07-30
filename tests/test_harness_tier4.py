"""Tier 4 専用評価ハーネス（icsr8.harness_tier4）のテスト。

Tier 4 の 7 手法は並行実装中のため、ここでは存在に依存せず既存の速い手法
（wcl / wcl_corridor）を代役に使う。references は本番同一の
["wcl", "gp_corridor"]（サブサンプルを segment 層化することで gp_corridor が
少数地点でも fit できる）。

スキーマ契約（列リスト）は harness_tier4 から import せずここに literal で
直書きする。Why not import: 実装側の列変更をテストが無警告で追従してしまう。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.corridor import segment_of
from icsr8.harness_tier4 import (
    REFERENCE_METHODS,
    TIER4_METHODS,
    _collect_diagnostics,
    _protocol_row,
    _tex_escape,
    lolo_summary_columns,
    make_figures_tier4,
    make_tex_tables_tier4,
    paired_delta_ci,
    protocol_a_columns,
    run_lolo_tier4,
    run_protocol_a_tier4,
    run_tier4,
    subsample_scans,
)
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import REGISTRY
from icsr8.methods.base import Method

STANDIN_METHODS = ["wcl", "wcl_corridor"]
STANDIN_REFERENCES = ["wcl", "gp_corridor"]  # 本番同一（smoke スキーマ乖離防止）
SMOKE_N_LOC = 9
SMOKE_B = 50

# --- スキーマ契約（literal 直書き。実装から import しない）--------------------

PROTOCOL_A_COLUMNS = [
    "method", "fold", "ave", "median", "p90", "within_2m", "max", "std",
    "ci_lo", "ci_hi",
    "delta_vs_wcl", "delta_vs_wcl_lo", "delta_vs_wcl_hi",
    "delta_vs_gp_corridor", "delta_vs_gp_corridor_lo", "delta_vs_gp_corridor_hi",
    "status",
]
LOLO_SUMMARY_COLUMNS = [
    "method", "ave", "median", "p90", "within_2m",
    "delta_vs_wcl", "delta_vs_wcl_lo", "delta_vs_wcl_hi",
    "delta_vs_gp_corridor", "delta_vs_gp_corridor_lo", "delta_vs_gp_corridor_hi",
    "status",
]
LOLO_LEDGER_COLUMNS = ["method", "held_out", "error", "true_x", "true_y"]
DIAG_COLUMNS = ["protocol", "fold", "method", "key", "value"]


# --- ダミー Method 群（registry へは monkeypatch.setitem で一時登録）----------


def _est_df(locs) -> pd.DataFrame:
    locs = list(locs)
    return pd.DataFrame(
        {"location_p": locs, "x": [0.0] * len(locs), "y": [0.0] * len(locs)}
    )


class _StubBase(Method):
    uses_geometry = False

    def fit(self, train_scans, ap_coords, location_coords):
        self._train_locs = sorted(int(v) for v in location_coords["location_p"])
        return self


class _WrongLocMethod(_StubBase):
    """held_out ではなく train の先頭地点を予測として返す（HIGH-2 検証用）。"""

    name = "_t4_wrongloc"

    def predict(self, test_scans):
        return _est_df([self._train_locs[0]])


class _MultiLocMethod(_StubBase):
    """1 fold で 2 地点を返す（HIGH-2 検証用）。"""

    name = "_t4_multiloc"

    def predict(self, test_scans):
        return _est_df(self._train_locs[:2])


class _EmptyEstMethod(_StubBase):
    """空の予測を返す（MED-4 fail-soft 検証用）。"""

    name = "_t4_empty"

    def predict(self, test_scans):
        return pd.DataFrame(
            {"location_p": pd.Series([], dtype="int64"), "x": [], "y": []}
        )


class _DiagMethod(_StubBase):
    """diagnostics_ を持ち全 test 地点を予測する（診断 long-form 検証用）。"""

    name = "_t4_diag"

    def fit(self, train_scans, ap_coords, location_coords):
        super().fit(train_scans, ap_coords, location_coords)
        self.diagnostics_ = {"alpha": 1.5, "beta": "x"}
        return self

    def predict(self, test_scans):
        return _est_df(sorted(set(int(v) for v in test_scans["location_p"])))


def _make_fail_on(fail_loc: int) -> type[Method]:
    class _FailOn(_StubBase):
        name = "_t4_failon"

        def predict(self, test_scans):
            locs = sorted(set(int(v) for v in test_scans["location_p"]))
            if fail_loc in locs:
                raise RuntimeError(f"boom on location {fail_loc}")
            return _est_df(locs)

    return _FailOn


# --- 純関数ユニット（実データ不要）------------------------------------------


def _mk_ledger(errors_by_loc: dict[int, float]) -> pd.DataFrame:
    df = pd.DataFrame({"error": list(errors_by_loc.values())}, index=list(errors_by_loc))
    df.index.name = "location_p"
    return df


def test_paired_delta_ci_full_pairing():
    full = {1, 2, 3, 4}
    a = _mk_ledger({1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0})
    b = _mk_ledger({1: 0.5, 2: 1.5, 3: 2.5, 4: 3.5})
    out = paired_delta_ci(a, b, full, seed=0, B=200)
    assert out["paired"] is True
    assert np.isfinite(out["stat"]) and np.isfinite(out["lo"]) and np.isfinite(out["hi"])
    # 各差分は +0.5 の定数なので stat は 0.5、CI も 0.5 に潰れる。
    assert out["stat"] == pytest.approx(0.5)


def test_paired_delta_ci_incomplete_pairing_returns_nan():
    full = {1, 2, 3, 4}
    a = _mk_ledger({1: 1.0, 2: 2.0, 3: 3.0})  # 地点 4 を欠く
    b = _mk_ledger({1: 0.5, 2: 1.5, 3: 2.5, 4: 3.5})
    out = paired_delta_ci(a, b, full, seed=0, B=200)
    assert out["paired"] is False
    for k in ("stat", "lo", "hi"):
        assert np.isnan(out[k]), f"{k} should be NaN on incomplete pairing"


def test_paired_delta_ci_duplicate_index_not_paired():
    # set 比較だけでは重複 index が完全ペアリングを装える（MED-3）。
    full = {1, 2, 3}
    a = pd.DataFrame({"error": [1.0, 2.0, 2.5, 3.0]}, index=[1, 2, 2, 3])
    a.index.name = "location_p"
    b = _mk_ledger({1: 0.5, 2: 1.5, 3: 2.5})
    out = paired_delta_ci(a, b, full, seed=0, B=200)
    assert out["paired"] is False
    for k in ("stat", "lo", "hi"):
        assert np.isnan(out[k])


def test_paired_delta_ci_order_insensitive():
    # 行順が違っても location_p で整列して同一結果になる（MED-8a）。
    full = {1, 2, 3, 4}
    a_sorted = _mk_ledger({1: 1.0, 2: 5.0, 3: 3.0, 4: 4.0})
    a_scrambled = a_sorted.loc[[3, 1, 4, 2]]
    b = _mk_ledger({1: 0.5, 2: 1.5, 3: 2.5, 4: 3.5})
    out1 = paired_delta_ci(a_sorted, b, full, seed=0, B=200)
    out2 = paired_delta_ci(a_scrambled, b, full, seed=0, B=200)
    assert out1["paired"] and out2["paired"]
    assert out1["stat"] == out2["stat"]
    assert out1["lo"] == out2["lo"] and out1["hi"] == out2["hi"]


def test_protocol_row_pairing_failure_sets_nan_and_status():
    full = {1, 2, 3, 4}
    ledgers = {
        "m": _mk_ledger({1: 1.0, 2: 2.0, 3: 3.0}),  # 地点 4 欠落 → ペアリング破綻
        "wcl": _mk_ledger({1: 0.5, 2: 1.5, 3: 2.5, 4: 3.5}),
    }
    row = _protocol_row("m", "forward_to_backward", ledgers, ["wcl"], full, seed=0, B=100)
    assert row["status"] == "pairing_failed"
    assert np.isnan(row["delta_vs_wcl"])
    assert np.isnan(row["delta_vs_wcl_lo"])
    assert np.isnan(row["delta_vs_wcl_hi"])


def test_protocol_row_self_delta_is_exactly_zero():
    full = {1, 2, 3, 4}
    ledgers = {"wcl": _mk_ledger({1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0})}
    row = _protocol_row("wcl", "forward_to_backward", ledgers, ["wcl"], full, seed=0, B=100)
    assert row["status"] == "ok"
    assert row["delta_vs_wcl"] == 0.0
    assert row["delta_vs_wcl_lo"] == 0.0
    assert row["delta_vs_wcl_hi"] == 0.0


def test_protocol_row_failed_method():
    full = {1, 2, 3}
    ledgers = {"wcl": _mk_ledger({1: 1.0, 2: 2.0, 3: 3.0})}
    row = _protocol_row("m", "forward_to_backward", ledgers, ["wcl"], full, seed=0, B=100)
    assert row["status"] == "failed"
    assert np.isnan(row["ave"])
    assert np.isnan(row["delta_vs_wcl"])


def test_protocol_row_empty_ledger_is_failed_not_crash():
    # 空 ledger の summary()/bootstrap は例外を投げるが、境界内で failed に畳む（MED-4）。
    full = {1, 2, 3}
    empty = pd.DataFrame({"error": pd.Series([], dtype=float)})
    empty.index.name = "location_p"
    ledgers = {"m": empty, "wcl": _mk_ledger({1: 1.0, 2: 2.0, 3: 3.0})}
    row = _protocol_row("m", "forward_to_backward", ledgers, ["wcl"], full, seed=0, B=100)
    assert row["status"] == "failed"
    assert np.isnan(row["ave"])
    assert np.isnan(row["delta_vs_wcl"])


def test_schema_functions_match_literal_contract():
    assert protocol_a_columns(["wcl", "gp_corridor"]) == PROTOCOL_A_COLUMNS
    assert lolo_summary_columns(["wcl", "gp_corridor"]) == LOLO_SUMMARY_COLUMNS


def test_collect_diagnostics_long_form_rows():
    class _Fake:
        diagnostics_ = {"selected_k": 5, "note": "x"}

    rows = _collect_diagnostics(_Fake(), protocol="lolo", fold=42, method_name="fisher_wknn")
    assert len(rows) == 2
    assert set(rows[0]) == set(DIAG_COLUMNS)
    r0 = rows[0]
    assert r0["protocol"] == "lolo"
    assert r0["fold"] == 42
    assert r0["method"] == "fisher_wknn"
    assert {r["key"] for r in rows} == {"selected_k", "note"}


def test_collect_diagnostics_absent_is_empty():
    assert _collect_diagnostics(object(), protocol="protocol_a", fold="f", method_name="m") == []


def test_tier4_constants():
    assert len(TIER4_METHODS) == 7
    assert REFERENCE_METHODS == ["wcl", "gp_corridor"]


def test_tex_escape_specials():
    # LOW-7: _ 以外の LaTeX 特殊文字もエスケープする。
    assert _tex_escape("a&b_c%d#e") == r"a\&b\_c\%d\#e"
    assert _tex_escape("x{y}$z") == r"x\{y\}\$z"
    assert _tex_escape("p~q^r") == r"p\textasciitilde{}q\textasciicircum{}r"
    assert _tex_escape("back\\slash") == r"back\textbackslash{}slash"


def test_tex_fragment_escapes_special_method_name(tmp_path):
    results = pd.DataFrame(
        [{
            "method": "bad&name_1", "fold": "forward_to_backward",
            "ave": 1.0, "median": 1.0, "p90": 1.0, "within_2m": 1.0,
            "max": 1.0, "std": 0.0, "ci_lo": 1.0, "ci_hi": 1.0,
            "delta_vs_wcl": 0.1, "delta_vs_wcl_lo": 0.0, "delta_vs_wcl_hi": 0.2,
            "status": "ok",
        }]
    )
    paths = make_tex_tables_tier4(results, None, ["wcl"], tmp_path)
    text = paths[0].read_text(encoding="utf-8")
    assert r"bad\&name\_1" in text


# --- 実データ fixtures --------------------------------------------------------


@pytest.fixture(scope="session")
def ap13(dataset_dir: Path) -> pd.DataFrame:
    return load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")


@pytest.fixture(scope="session")
def truth(dataset_dir: Path) -> pd.DataFrame:
    return load_location_coords(dataset_dir / "location_coordinate_C.csv")[
        ["location_p", "x", "y"]
    ]


@pytest.fixture(scope="session")
def scans_f(rawdata_root: Path) -> pd.DataFrame:
    return load_raw_scans("forward", rawdata_root)


@pytest.fixture(scope="session")
def scans_b(rawdata_root: Path) -> pd.DataFrame:
    return load_raw_scans("backward", rawdata_root)


@pytest.fixture(scope="session")
def small(scans_f, scans_b, truth):
    return subsample_scans(scans_f, scans_b, truth, SMOKE_N_LOC)


def test_subsample_spans_all_segments(small):
    # MED-5: サブサンプルが 3 segment 全てを跨ぐこと（gp_corridor fit 可能条件）。
    _sf, _sb, tr = small
    segs = {segment_of(float(x), float(y)) for x, y in zip(tr["x"], tr["y"])}
    assert segs == {"C", "C2", "C3"}


# --- Protocol A（実データ・代役手法 + 正式 references）------------------------


@pytest.fixture(scope="session")
def proto_run(small, ap13):
    sf, sb, tr = small
    return run_protocol_a_tier4(
        STANDIN_METHODS, STANDIN_REFERENCES, sf, sb, ap13, tr, seed=0, B=SMOKE_B
    )


def test_protocol_a_schema_literal(proto_run):
    results, _ledger, _diag = proto_run
    assert list(results.columns) == PROTOCOL_A_COLUMNS


def test_protocol_a_two_delta_columns_finite(proto_run):
    results, _ledger, _diag = proto_run
    row = results[results["method"] == "wcl_corridor"]
    assert len(row) == 2  # 2 fold
    assert np.isfinite(row["delta_vs_wcl"]).all()
    assert np.isfinite(row["delta_vs_gp_corridor"]).all()


def test_protocol_a_real_reference_runs_and_self_delta_zero(proto_run):
    # MED-5: smoke 構成でも gp_corridor が fit でき、基準マークの対象行が実在する。
    results, _ledger, _diag = proto_run
    gp = results[results["method"] == "gp_corridor"]
    assert len(gp) == 2
    assert (gp["status"] == "ok").all()
    assert (gp["delta_vs_gp_corridor"] == 0.0).all()


def test_protocol_a_self_delta_zero_in_run(proto_run):
    results, _ledger, _diag = proto_run
    wcl = results[results["method"] == "wcl"]
    assert (wcl["delta_vs_wcl"] == 0.0).all()


def test_protocol_a_status_ok(proto_run):
    results, _ledger, _diag = proto_run
    assert (results["status"] == "ok").all()


def test_protocol_a_diagnostics_collected_from_method(small, ap13, monkeypatch):
    # MED-8b: diagnostics_ を持つ手法から必ず long-form 行が回収される。
    monkeypatch.setitem(REGISTRY, "_t4_diag", _DiagMethod)
    sf, sb, tr = small
    _results, _ledger, diag = run_protocol_a_tier4(
        ["_t4_diag"], ["wcl"], sf, sb, ap13, tr, seed=0, B=SMOKE_B
    )
    assert list(diag.columns) == DIAG_COLUMNS
    sub = diag[diag["method"] == "_t4_diag"]
    assert len(sub) == 2 * 2  # 2 fold × 2 keys
    assert set(sub["key"]) == {"alpha", "beta"}
    assert set(sub["protocol"]) == {"protocol_a"}


def test_protocol_a_empty_prediction_is_fail_soft(small, ap13, monkeypatch):
    # MED-4: 空予測の手法が居ても run 全体は落ちず、当該行だけ failed。
    monkeypatch.setitem(REGISTRY, "_t4_empty", _EmptyEstMethod)
    sf, sb, tr = small
    results, _ledger, _diag = run_protocol_a_tier4(
        ["_t4_empty"], ["wcl"], sf, sb, ap13, tr, seed=0, B=SMOKE_B
    )
    bad = results[results["method"] == "_t4_empty"]
    assert (bad["status"] == "failed").all()
    assert bad["ave"].isna().all()
    good = results[results["method"] == "wcl"]
    assert (good["status"] == "ok").all()


# --- LOLO ---------------------------------------------------------------------


@pytest.fixture(scope="session")
def lolo_run(small, ap13):
    sf, sb, tr = small
    return run_lolo_tier4(
        STANDIN_METHODS, STANDIN_REFERENCES, sf, sb, ap13, tr, seed=0, B=SMOKE_B
    )


def test_lolo_ledger_schema(lolo_run):
    ledger, _summary, _diag = lolo_run
    assert list(ledger.columns) == LOLO_LEDGER_COLUMNS
    # N_LOC folds × union(methods, references) = 3 手法（wcl は重複排除）
    n_methods = len(dict.fromkeys(STANDIN_METHODS + STANDIN_REFERENCES))
    assert len(ledger) == SMOKE_N_LOC * n_methods


def test_lolo_summary_schema_literal_and_deltas(lolo_run):
    _ledger, summary, _diag = lolo_run
    assert list(summary.columns) == LOLO_SUMMARY_COLUMNS
    wcl = summary[summary["method"] == "wcl"]
    assert (wcl["delta_vs_wcl"] == 0.0).all()
    corr = summary[summary["method"] == "wcl_corridor"]
    assert np.isfinite(corr["delta_vs_wcl"]).all()
    assert np.isfinite(corr["delta_vs_gp_corridor"]).all()
    assert (summary["status"] == "ok").all()


def test_lolo_wrong_location_prediction_is_nan(small, ap13, monkeypatch):
    # HIGH-2: held_out と異なる地点の予測は「先頭行採用」で誤差化せず NaN 失敗にする。
    monkeypatch.setitem(REGISTRY, "_t4_wrongloc", _WrongLocMethod)
    sf, sb, tr = small
    ledger, summary, _diag = run_lolo_tier4(
        ["_t4_wrongloc"], ["wcl"], sf, sb, ap13, tr, seed=0, B=SMOKE_B
    )
    bad = ledger[ledger["method"] == "_t4_wrongloc"]
    assert bad["error"].isna().all()
    row = summary[summary["method"] == "_t4_wrongloc"].iloc[0]
    assert row["status"] == "pairing_failed"
    assert np.isnan(row["delta_vs_wcl"])


def test_lolo_multi_location_prediction_is_nan(small, ap13, monkeypatch):
    # HIGH-2: 複数地点の予測も held_out 1 件との厳密一致違反として NaN。
    monkeypatch.setitem(REGISTRY, "_t4_multiloc", _MultiLocMethod)
    sf, sb, tr = small
    ledger, _summary, _diag = run_lolo_tier4(
        ["_t4_multiloc"], [], sf, sb, ap13, tr, seed=0, B=SMOKE_B
    )
    bad = ledger[ledger["method"] == "_t4_multiloc"]
    assert bad["error"].isna().all()


def test_lolo_partial_failure_sets_pairing_failed(small, ap13, monkeypatch):
    # MED-8a: 1 fold 欠落で held_out 整列が破綻 → delta NaN + pairing_failed。
    sf, sb, tr = small
    fail_loc = int(sorted(tr["location_p"])[0])
    monkeypatch.setitem(REGISTRY, "_t4_failon", _make_fail_on(fail_loc))
    ledger, summary, _diag = run_lolo_tier4(
        ["_t4_failon"], ["wcl"], sf, sb, ap13, tr, seed=0, B=SMOKE_B
    )
    bad = ledger[ledger["method"] == "_t4_failon"]
    assert bad.loc[bad["held_out"] == fail_loc, "error"].isna().all()
    assert np.isfinite(bad.loc[bad["held_out"] != fail_loc, "error"]).all()
    row = summary[summary["method"] == "_t4_failon"].iloc[0]
    assert row["status"] == "pairing_failed"
    assert np.isnan(row["delta_vs_wcl"])


def test_lolo_diagnostics_fold_is_held_out(small, ap13, monkeypatch):
    # MED-8b: LOLO 診断の fold 列は held_out 地点そのもの。
    monkeypatch.setitem(REGISTRY, "_t4_diag", _DiagMethod)
    sf, sb, tr = small
    _ledger, _summary, diag = run_lolo_tier4(
        ["_t4_diag"], [], sf, sb, ap13, tr, seed=0, B=SMOKE_B
    )
    assert list(diag.columns) == DIAG_COLUMNS
    sub = diag[diag["method"] == "_t4_diag"]
    assert len(sub) == SMOKE_N_LOC * 2  # folds × 2 keys
    assert set(sub["fold"]) == set(int(v) for v in tr["location_p"])
    assert set(sub["protocol"]) == {"lolo"}


def test_lolo_full_59_folds_real_data(scans_f, scans_b, ap13, truth):
    # MED-8d: 実データで 59 fold × 1 手法の台帳行数と held_out 全域被覆を固定。
    ledger, summary, _diag = run_lolo_tier4(
        ["wcl"], ["wcl"], scans_f, scans_b, ap13, truth, seed=0, B=20
    )
    assert len(ledger) == 59
    assert sorted(ledger["held_out"]) == sorted(int(v) for v in truth["location_p"])
    assert (summary["status"] == "ok").all()


# --- e2e / 出力先限定・凍結保護 ------------------------------------------------


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _frozen_sentinels(repo_root: Path) -> list[Path]:
    # main body 6（表 TeX 2 + 図 PDF 4）+ 付録 A 3（表 TeX 2 + 図 PDF 1）= 9。
    # harness_tier4.FROZEN_OUTPUT_PATHS と 1:1 対応する（ここでは import せず
    # literal で複製する — 実装側の凍結リスト変更をテストが無警告で追従しないため）。
    return [
        repo_root / "doc" / "final_report" / "tables" / "protocol_a.tex",
        repo_root / "doc" / "final_report" / "tables" / "lolo.tex",
        repo_root / "doc" / "final_report" / "figures" / "cdf_protocol_a_forward_to_backward.pdf",
        repo_root / "doc" / "final_report" / "figures" / "cdf_protocol_a_backward_to_forward.pdf",
        repo_root / "doc" / "final_report" / "figures" / "cdf_lolo.pdf",
        repo_root / "doc" / "final_report" / "figures" / "segment_heatmap.pdf",
        repo_root / "doc" / "final_report" / "tables" / "tier4_protocol_a.tex",
        repo_root / "doc" / "final_report" / "tables" / "tier4_lolo.tex",
        repo_root / "doc" / "final_report" / "figures" / "cdf_lolo_tier4.pdf",
    ]


def test_frozen_output_paths_is_nine_files(repo_root):
    # 2026-07-29 allowlist 化: 旧 blocklist は results/*.csv 4 本を含む 6 ファイル
    # だった（Commit 1 で CSV を対象外化）。allowlist 化では、以前は
    # FROZEN_OUTPUT_PATHS に一度も載ったことがなく構造的に無防御だった付録 A の
    # 3 ファイル（tier4_*.tex 2 + cdf_lolo_tier4.pdf）も対象に加え、計 9 とする。
    from icsr8.harness_tier4 import FROZEN_OUTPUT_PATHS

    assert FROZEN_OUTPUT_PATHS == {p.resolve() for p in _frozen_sentinels(repo_root)}
    assert len(FROZEN_OUTPUT_PATHS) == 9


def test_frozen_pdf_hashes_manifest_covers_frozen_pdfs(repo_root):
    # 2026-07-30 codex round2 NEW-1: scripts/verify_report.py の
    # _verify_pdf_hash_manifest_coverage() が実装する set 完全一致契約を、
    # 独立した経路（scripts/ を import せず、FROZEN_OUTPUT_PATHS と json を
    # 直接読む）で pin する。verify_report.py 側のロジックが将来壊れても、
    # この pytest は json ファイルと allowlist 定数を直接突き合わせるため
    # 検出できる。
    from icsr8.harness_tier4 import FROZEN_OUTPUT_PATHS

    frozen_pdfs = {
        str(p.relative_to(repo_root)) for p in FROZEN_OUTPUT_PATHS if p.suffix == ".pdf"
    }
    manifest = json.loads(
        (repo_root / "scripts" / "frozen_pdf_hashes.json").read_text(encoding="utf-8")
    )
    paths = [e["path"] for e in manifest]
    assert set(paths) == frozen_pdfs
    assert len(paths) == len(set(paths))  # 重複エントリが無い


def test_run_tier4_refuses_frozen_output(repo_root):
    # HIGH-1 + 2026-07-29 allowlist 化: writer_id を渡さない（= 実験用 CLI /
    # hand-edit と同じ立場）呼び出しは、凍結パスを 1 つでも含む targets を
    # ValueError で拒否する。sanctioned writer（icsr8.report.regenerate_*）で
    # なければ通らないことがこのテストの核心。
    from icsr8.harness_tier4 import _guard_frozen

    sentinels = _frozen_sentinels(repo_root)
    before = {p: _sha(p) for p in sentinels}
    with pytest.raises(ValueError, match="frozen"):
        _guard_frozen(
            [repo_root / "doc" / "final_report" / "tables" / "protocol_a.tex"],
            writer_id=None,
        )
    assert {p: _sha(p) for p in sentinels} == before


def test_guard_frozen_rejects_unknown_writer_id(repo_root):
    # allowlist は正確な文字列一致で判定する。似た名前や部分一致では通さない
    # （偽装耐性 — _SANCTIONED_WRITERS の docstring が明記する契約）。
    from icsr8.harness_tier4 import _guard_frozen

    with pytest.raises(ValueError, match="frozen"):
        _guard_frozen(
            [repo_root / "doc" / "final_report" / "tables" / "protocol_a.tex"],
            writer_id="icsr8.report.regenerate_main_body_typo",
        )


def test_make_figures_tier4_refuses_frozen_output(repo_root):
    # Codex review finding 1: make_figures_tier4 自身が凍結ガードを持つことを、
    # run_tier4() を経由しない直接呼び出しで検証する（押し下げ前は run_tier4()
    # のガードだけが頼りで、この関数を直接 import して呼ぶ経路が無防備だった）。
    figures_dir = repo_root / "doc" / "final_report" / "figures"
    sentinel = figures_dir / "cdf_lolo_tier4.pdf"
    before = _sha(sentinel)

    with pytest.raises(ValueError, match="frozen"):
        make_figures_tier4(pd.DataFrame(columns=["method", "error"]), figures_dir)

    assert _sha(sentinel) == before


def test_make_tex_tables_tier4_refuses_frozen_output(repo_root):
    tables_dir = repo_root / "doc" / "final_report" / "tables"
    sentinels = [tables_dir / "tier4_protocol_a.tex", tables_dir / "tier4_lolo.tex"]
    before = {p: _sha(p) for p in sentinels}

    with pytest.raises(ValueError, match="frozen"):
        make_tex_tables_tier4(
            pd.DataFrame(columns=["method", "fold"]), None, ["wcl"], tables_dir
        )

    assert {p: _sha(p) for p in sentinels} == before


def test_regenerate_main_body_writes_frozen_paths(monkeypatch, repo_root):
    # 2026-07-29: sanctioned writer（icsr8.report.regenerate_main_body）は
    # 凍結パスへの _guard_frozen を通過できることを検証する。report.py は
    # `from icsr8.harness import run_protocol_a, ...` で名前を束縛しているため、
    # icsr8.harness 側を monkeypatch しても report モジュール内の呼び出しには
    # 反映されない — report モジュールの属性を直接差し替える必要がある。
    #
    # _guard_frozen は本物のまま・real repo root のまま呼ぶ（そうしないと
    # 「sanctioned writer だから通った」のか「凍結パス扱いされずスキップされた
    # だけ」なのか区別できないテストになってしまう）。一方 CSV/TeX/PDF への実
    # 書き込みは commit 済みの tracked ファイルを破壊しうるため、実計算関数を
    # 「to_csv が no-op なスタブオブジェクト」を返すフェイクに差し替えて防ぐ。
    import icsr8.report as report

    calls: list[str] = []

    class _NoWriteFrame:
        """pandas.DataFrame の代わりに渡すスタブ。to_csv を no-op にして
        実リポジトリの commit 済み CSV を書き換えないようにする。"""

        def to_csv(self, path, index=False):  # noqa: ARG002 - シグネチャ合わせ
            calls.append(f"to_csv:{Path(path).name}")

    def _fake_run_protocol_a(*args, **kwargs):
        calls.append("run_protocol_a")
        return _NoWriteFrame(), _NoWriteFrame()

    def _fake_run_lolo(*args, **kwargs):
        calls.append("run_lolo")
        return _NoWriteFrame(), _NoWriteFrame()

    def _fake_make_figures(*args, **kwargs):
        calls.append("make_figures")
        return []

    def _fake_make_tex_tables(*args, **kwargs):
        calls.append("make_tex_tables")
        return []

    def _fake_write_diag(*args, **kwargs):
        calls.append("write_diagnostics")

    monkeypatch.setattr(report, "_load_data", lambda data_root: (None, None, None, None))
    monkeypatch.setattr(report, "run_protocol_a", _fake_run_protocol_a)
    monkeypatch.setattr(report, "run_lolo", _fake_run_lolo)
    monkeypatch.setattr(report, "make_figures", _fake_make_figures)
    monkeypatch.setattr(report, "make_tex_tables", _fake_make_tex_tables)
    monkeypatch.setattr(report, "_write_method_diagnostics", _fake_write_diag)

    # 例外なく完走すれば sanctioned writer として _guard_frozen（実物・実 repo
    # root）を通過した証拠。writer_id=None なら test_run_tier4_refuses_frozen_output/
    # test_guard_frozen_rejects_unknown_writer_id が ValueError になることを固定
    # 済みなので、この 2 テストと合わせて allowlist の両側が検証される。
    report.regenerate_main_body()

    assert calls == [
        "run_protocol_a",
        "to_csv:protocol_a.csv",
        "to_csv:protocol_a_ledger.csv",
        "run_lolo",
        "to_csv:lolo_ledger.csv",
        "to_csv:lolo_summary.csv",
        "make_figures",
        "make_tex_tables",
        "write_diagnostics",
    ]


def test_run_tier4_confines_outputs(small, ap13, repo_root, tmp_path):
    # MED-8c: tmp_path 全走査 = 返却パス集合、かつ凍結ファイルの hash 不変。
    sf, sb, tr = small
    sentinels = _frozen_sentinels(repo_root)
    before = {p: _sha(p) for p in sentinels}

    outdir = tmp_path / "results" / "tier4"
    paths = run_tier4(
        scans_f=sf, scans_b=sb, ap13=ap13, truth=tr,
        methods=STANDIN_METHODS, references=STANDIN_REFERENCES,
        output_dir=outdir,
        tables_dir=tmp_path / "tables", figures_dir=tmp_path / "figures",
        seed=0, B=SMOKE_B,
    )
    assert paths, "run_tier4 は書き出したパス群を返すこと"
    found = {p.resolve() for p in tmp_path.rglob("*") if p.is_file()}
    returned = {Path(p).resolve() for p in paths.values()}
    assert found == returned, "返却パス以外のファイルが書かれている/生成漏れがある"

    written = {Path(p).name for p in paths.values()}
    assert written == {
        "protocol_a.csv", "lolo_ledger.csv", "lolo_summary.csv", "diagnostics.csv",
        "tier4_protocol_a.tex", "tier4_lolo.tex", "cdf_lolo_tier4.pdf",
    }
    assert {p: _sha(p) for p in sentinels} == before


def test_make_figures_tier4_deterministic(lolo_run, tmp_path):
    # MED-6: 同一入力から 2 回描画した PDF が byte 一致（CreationDate 非依存）。
    ledger, _summary, _diag = lolo_run
    p1 = make_figures_tier4(ledger, tmp_path / "a")[0]
    time.sleep(1.1)  # CreationDate の秒が変わる状況を強制する
    p2 = make_figures_tier4(ledger, tmp_path / "b")[0]
    assert p1.read_bytes() == p2.read_bytes()


def test_run_experimental_tier4_refuses_frozen_paths(repo_root, tmp_path):
    # 2026-07-29 allowlist e2e: scripts/run_experimental_tier4.py は writer_id
    # を渡さない非 sanctioned CLI なので、--tables-dir に凍結ディレクトリを指定
    # すると run_tier4() 内部の _guard_frozen が ValueError で reject するはず。
    # サブプロセスで実 CLI を起動し、実プロセス境界を越えても契約が保たれる
    # ことを確認する（_guard_frozen はデータ読み込み・argparse の直後、重い
    # sweep 開始前に呼ばれるため、このテストは高速に失敗する）。
    import os

    sentinels = _frozen_sentinels(repo_root)
    before = {p: _sha(p) for p in sentinels}

    # tests/test_colab_bootstrap.py の他テストが同一 pytest セッション内で
    # os.environ["PYTHONPATH"] を実プロセスに残す場合がある（colab_bootstrap の
    # activate() 副作用）。継承すると子プロセスが偽の src ツリーを先に見つけて
    # `import icsr8.constants` に失敗しうるため、PYTHONPATH を明示的に落とす。
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_experimental_tier4.py"),
            "--methods", "wcl",
            "--output", str(tmp_path / "out"),
            "--tables-dir", str(repo_root / "doc" / "final_report" / "tables"),
            "--figures-dir", str(tmp_path / "figures"),
            "--dataset-root", str(repo_root / "data"),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0, (
        f"expected non-zero exit; stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "frozen" in result.stderr, f"stderr did not mention 'frozen': {result.stderr!r}"
    assert {p: _sha(p) for p in sentinels} == before, "凍結ファイルが書き換わっている"
