"""評価ハーネス（icsr8.harness）の実データスモークテスト。

本命の重い sweep は scripts/run_all_methods.py が担う。ここでは
安定手法 wcl / pbl だけを使い、fold 数を絞った軽量経路で
CSV スキーマ・行数・delta 契約・成果物ファイル生成を固定する。

期待カラムは harness からは import せず、ここに直書きする。
Why not import: スキーマを二重に持つことで、harness 側の
カラム変更をテストが無警告で追従してしまうのを防ぐ。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from icsr8.harness import make_figures, make_tex_tables, run_lolo, run_protocol_a
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans

# --- スキーマ契約（harness からは import しない）-----------------------------

PROTOCOL_A_COLUMNS = [
    "method", "fold", "ave", "median", "p75", "p90", "max", "std",
    "within_2m", "within_4m", "ci_lo", "ci_hi",
    "delta_vs_wcl", "delta_lo", "delta_hi", "failed",
]
LOLO_LEDGER_COLUMNS = ["method", "held_out", "error", "true_x", "true_y"]
LOLO_SUMMARY_COLUMNS = ["method", "ave", "median", "p90", "within_2m"]

SMOKE_METHODS = ["wcl", "pbl"]
SMOKE_B = 50
SMOKE_MAX_FOLDS = 2


# --- session fixtures（scan 読み込みは 1 回だけ）-----------------------------

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
def protocol_a_run(scans_f, scans_b, ap13, truth):
    return run_protocol_a(SMOKE_METHODS, scans_f, scans_b, ap13, truth, seed=0, B=SMOKE_B)


@pytest.fixture(scope="session")
def lolo_run(scans_f, scans_b, ap13, truth):
    return run_lolo(
        SMOKE_METHODS, scans_f, scans_b, ap13, truth, seed=0, max_folds=SMOKE_MAX_FOLDS
    )


# --- Protocol A -------------------------------------------------------------

def test_protocol_a_schema(protocol_a_run):
    results, _ledgers = protocol_a_run
    assert list(results.columns) == PROTOCOL_A_COLUMNS


def test_protocol_a_row_count(protocol_a_run):
    results, _ledgers = protocol_a_run
    # 2 fold × 2 method
    assert len(results) == 2 * len(SMOKE_METHODS)


def test_protocol_a_no_failures(protocol_a_run):
    results, _ledgers = protocol_a_run
    assert not results["failed"].any()


def test_protocol_a_wcl_delta_is_zero(protocol_a_run):
    results, _ledgers = protocol_a_run
    wcl_rows = results[results["method"] == "wcl"]
    assert len(wcl_rows) == 2
    for col in ("delta_vs_wcl", "delta_lo", "delta_hi"):
        assert (wcl_rows[col] == 0.0).all()


def test_protocol_a_ledger_columns(protocol_a_run):
    _results, ledgers = protocol_a_run
    assert list(ledgers.columns) == ["method", "fold", "location_p", "error", "true_x", "true_y"]
    # 59 location × 2 fold × 2 method
    assert len(ledgers) == 59 * 2 * len(SMOKE_METHODS)


# --- LOLO -------------------------------------------------------------------

def test_lolo_ledger_schema(lolo_run):
    ledger, _summary = lolo_run
    assert list(ledger.columns) == LOLO_LEDGER_COLUMNS


def test_lolo_ledger_row_count(lolo_run):
    ledger, _summary = lolo_run
    # max_folds × method
    assert len(ledger) == SMOKE_MAX_FOLDS * len(SMOKE_METHODS)


def test_lolo_summary_schema(lolo_run):
    _ledger, summary = lolo_run
    assert list(summary.columns) == LOLO_SUMMARY_COLUMNS
    assert set(summary["method"]) == set(SMOKE_METHODS)


# --- figures ----------------------------------------------------------------

def test_make_figures_creates_pdfs(protocol_a_run, lolo_run, tmp_path):
    _results, pa_ledgers = protocol_a_run
    lolo_ledger, _summary = lolo_run
    outdir = tmp_path / "figures"
    created = make_figures({"protocol_a": pa_ledgers, "lolo": lolo_ledger}, outdir)
    assert created, "make_figures は生成ファイルパスを返すこと"
    for p in created:
        assert p.exists()
    names = {p.name for p in created}
    assert "cdf_lolo.pdf" in names
    assert "segment_heatmap.pdf" in names


# --- tex tables -------------------------------------------------------------

def test_make_tex_tables_creates_fragments(protocol_a_run, lolo_run, tmp_path):
    results, _ledgers = protocol_a_run
    _ledger, lolo_summary = lolo_run
    outdir = tmp_path / "tables"
    created = make_tex_tables(results, lolo_summary, outdir)
    names = {p.name for p in created}
    assert "protocol_a.tex" in names
    assert "lolo.tex" in names

    proto_tex = (outdir / "protocol_a.tex").read_text(encoding="utf-8")
    assert r"\begin{tabular}" in proto_tex
    for method in SMOKE_METHODS:
        assert method in proto_tex
