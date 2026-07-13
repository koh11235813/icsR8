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
    return [
        repo_root / "results" / "protocol_a.csv",
        repo_root / "results" / "lolo_ledger.csv",
        repo_root / "results" / "lolo_summary.csv",
        repo_root / "doc" / "final_report" / "tables" / "protocol_a.tex",
        repo_root / "doc" / "final_report" / "tables" / "lolo.tex",
        repo_root / "doc" / "final_report" / "figures" / "cdf_lolo.pdf",
    ]


def test_run_tier4_refuses_frozen_output(small, ap13, repo_root, tmp_path):
    # HIGH-1: 凍結済み results/*.csv を指す output は書き込み前に拒否する。
    sf, sb, tr = small
    sentinels = _frozen_sentinels(repo_root)
    before = {p: _sha(p) for p in sentinels}
    with pytest.raises(ValueError, match="frozen"):
        run_tier4(
            scans_f=sf, scans_b=sb, ap13=ap13, truth=tr,
            methods=STANDIN_METHODS, references=STANDIN_REFERENCES,
            output_dir=repo_root / "results",
            tables_dir=tmp_path / "t", figures_dir=tmp_path / "f",
            seed=0, B=SMOKE_B,
        )
    assert {p: _sha(p) for p in sentinels} == before
    # ガードは実行前に働く → tmp 側にも何も書かれていない。
    assert not list(tmp_path.rglob("*"))


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
