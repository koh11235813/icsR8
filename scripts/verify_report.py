"""最終報告の整合性検証（表数値・参照パス）。

results/*.csv の各セルを表生成器と同一の関数（icsr8.harness の _label/_fmt/
_tex_escape）で再フォーマットし、doc/final_report/tables/*.tex の各行を byte 単位で
再構成できることを確認する。加えて main.tex の \\input / \\includegraphics 先が全て
実在することを確かめる。全て通れば exit 0、1 つでも失敗すれば exit 1。

使用例:
    uv run python scripts/verify_report.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

try:
    from icsr8.harness import _POWERDOMAIN_NOTE, _fmt, _label, _tex_escape
except ImportError:  # editable install 未実施でも動くよう src を通す
    sys.path.insert(0, str(ROOT / "src"))
    from icsr8.harness import _POWERDOMAIN_NOTE, _fmt, _label, _tex_escape

REPORT_DIR = ROOT / "doc" / "final_report"
MAIN_TEX = REPORT_DIR / "main.tex"
DIAGNOSTICS_CSV = ROOT / "results" / "method_diagnostics.csv"
TIER4_DIR = ROOT / "results" / "tier4"

# (header, csv-column) — harness の _protocol_a_tex / _lolo_tex と同一順序。
PROTO_COLS = ["ave", "median", "p90", "max", "within_2m", "delta_vs_wcl"]
LOLO_COLS = ["ave", "median", "p90", "within_2m"]

# main.tex §3 が引用する診断値 (method, key) -> results/method_diagnostics.csv の
# value 列から main.tex に載るべき部分文字列を作るフォーマッタ。
# scripts/dump_method_diagnostics.py の出力値が変われば main.tex 側の文言も
# 追随しているべき、という契約をここで検証する。
_DIAG_FORMATTERS = {
    ("wknn", "selected_k"): lambda v: f"K{{=}}{int(float(v))}",
    ("wknn", "selected_weighting"): lambda v: f"weighting={_tex_escape(str(v))}",
    ("gp_corridor", "segment_train_accuracy"): lambda v: f"train accuracy {float(v):.2f}",
    ("gp_corridor", "n_total_keys"): lambda v: f"{int(float(v))} キー中",
    ("gp_corridor", "n_gp_keys"): lambda v: f"{int(float(v))} キーをモデル化",
    ("studentt_fp", "selected_nu"): lambda v: f"\\nu{{=}}{int(float(v))}",
    ("centered_fp", "selected_lambda"): lambda v: f"\\lambda{{=}}{float(v):.2f}",
    ("rank_fp", "selected_lambda"): lambda v: f"\\lambda{{=}}{float(v):.2f}",
}

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def _fragment_data_lines(path: Path) -> set[str]:
    """tabular 断片から本文データ行（末尾 \\\\、ヘッダ除く）を集合で返す。"""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return {ln for ln in lines if ln.endswith(r"\\") and not ln.startswith("Method")}


def _proto_line(row: pd.Series) -> str:
    cells = [_label(row["method"]), _tex_escape(str(row["fold"]))]
    cells += [_fmt(row[c]) for c in PROTO_COLS]
    return " & ".join(cells) + r" \\"


def _lolo_line(row: pd.Series) -> str:
    cells = [_label(row["method"])] + [_fmt(row[c]) for c in LOLO_COLS]
    return " & ".join(cells) + r" \\"


def verify_table(csv_path: Path, tex_path: Path, line_fn) -> None:
    df = pd.read_csv(csv_path)
    frag = _fragment_data_lines(tex_path)
    expected = {line_fn(r) for _, r in df.iterrows()}

    # 行数一致（重複が無い前提で len を突き合わせる）。
    check(
        len(expected) == len(df),
        f"{csv_path.name}: 再構成行が重複 ({len(expected)} unique vs {len(df)} rows)",
    )
    check(
        len(frag) == len(df),
        f"{tex_path.name}: データ行数 {len(frag)} != CSV 行数 {len(df)}",
    )
    # 全数値行が断片に一致する形で存在するか。
    for line in expected:
        check(line in frag, f"{tex_path.name}: 一致行が無い -> {line}")

    # powerdomain † 脚注（両 CSV に wcl_powerdomain を含む）。
    if "wcl_powerdomain" in set(df["method"]):
        note = _POWERDOMAIN_NOTE.strip()
        got = [ln.strip() for ln in tex_path.read_text(encoding="utf-8").splitlines()]
        check(note in got, f"{tex_path.name}: powerdomain † 脚注が無い")


def verify_diagnostics() -> None:
    """main.tex が引用する診断値が results/method_diagnostics.csv と一致するか検証する。"""
    if not DIAGNOSTICS_CSV.exists():
        failures.append(f"{DIAGNOSTICS_CSV.name}: 診断 CSV が存在しない")
        return
    diag = pd.read_csv(DIAGNOSTICS_CSV)
    text = MAIN_TEX.read_text(encoding="utf-8")
    for (method, key), fmt in _DIAG_FORMATTERS.items():
        rows = diag[(diag["method"] == method) & (diag["key"] == key)]
        check(
            len(rows) == 1,
            f"{DIAGNOSTICS_CSV.name}: 行が無い/重複している -> {method}/{key} ({len(rows)} 件)",
        )
        if len(rows) != 1:
            continue
        expected = fmt(rows["value"].iloc[0])
        check(
            expected in text,
            f"main.tex: 診断値の引用が CSV と不一致 -> {method}/{key} は {expected!r} を含むべき",
        )


def verify_paths() -> None:
    text = MAIN_TEX.read_text(encoding="utf-8")
    refs = re.findall(r"\\input\{([^}]+)\}", text)
    refs += re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    check(len(refs) >= 6, f"main.tex: 参照が想定より少ない ({len(refs)} 個)")
    for rel in refs:
        p = (REPORT_DIR / rel).resolve()
        check(p.exists(), f"main.tex: 参照先が存在しない -> {rel}")


def verify_tier4() -> None:
    """付録 Tier 4 の CSV・表・図と main.tex の整合を検証する（本文検証への加算）。

    protocol_a 18 行 / lolo_ledger 531 行 / lolo_summary 9 行、手法集合の完全一致、
    delta_vs_{wcl,gp_corridor} 列の存在と有限性、参照手法の自己 delta=0、status 全 ok、
    main.tex が tier4 表 2 つ・図 1 つを参照していること、tier4_*.tex 断片が CSV から
    バイト単位で再構成できることを確認する。
    """
    from icsr8.harness_tier4 import (  # noqa: PLC0415 - 加算モジュールの局所 import
        REFERENCE_METHODS,
        TIER4_METHODS,
        _lolo_tex,
        _order_by_lolo,
        _protocol_tex,
    )

    n0 = len(failures)
    proto_csv = TIER4_DIR / "protocol_a.csv"
    ledger_csv = TIER4_DIR / "lolo_ledger.csv"
    summary_csv = TIER4_DIR / "lolo_summary.csv"
    for p in (proto_csv, ledger_csv, summary_csv):
        if not p.exists():
            failures.append(f"tier4/{p.name}: CSV が存在しない")
    if not (proto_csv.exists() and ledger_csv.exists() and summary_csv.exists()):
        return

    proto = pd.read_csv(proto_csv)
    ledger = pd.read_csv(ledger_csv)
    summ = pd.read_csv(summary_csv)
    expected_methods = set(TIER4_METHODS) | set(REFERENCE_METHODS)  # 7 + 2 = 9

    check(len(proto) == 18, f"tier4 protocol_a.csv: 行数 {len(proto)} != 18")
    check(len(ledger) == 531, f"tier4 lolo_ledger.csv: 行数 {len(ledger)} != 531")
    check(len(summ) == 9, f"tier4 lolo_summary.csv: 行数 {len(summ)} != 9")

    check(set(proto["method"]) == expected_methods,
          f"tier4 protocol_a: 手法集合不一致 -> {sorted(set(proto['method']))}")
    check(set(summ["method"]) == expected_methods,
          f"tier4 lolo_summary: 手法集合不一致 -> {sorted(set(summ['method']))}")
    check(set(ledger["method"]) == expected_methods,
          f"tier4 lolo_ledger: 手法集合不一致 -> {sorted(set(ledger['method']))}")

    # delta 列の存在・有限性、status 全 ok、参照手法の自己 delta=0。
    for df, name in ((proto, "protocol_a"), (summ, "lolo_summary")):
        check((df["status"] == "ok").all(), f"tier4 {name}: status に ok 以外が存在")
        for ref in REFERENCE_METHODS:
            col = f"delta_vs_{ref}"
            if col not in df.columns:
                failures.append(f"tier4 {name}: {col} 列が無い")
                continue
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            check(bool(np.isfinite(vals).all()), f"tier4 {name}: {col} に非有限値")
            self_delta = df.loc[df["method"] == ref, col].to_numpy(dtype=float)
            check(bool((self_delta == 0.0).all()),
                  f"tier4 {name}: 参照 {ref} の自己 delta が 0 でない")

    # codex 最終レビュー反映: 総行数だけでなく method 単位の行数・一意キー・
    # 有意差判断に使う CI 4 列（*_lo/*_hi）まで契約として固定する。
    check(bool(proto.groupby("method").size().eq(2).all()),
          "tier4 protocol_a: method ごとの fold 行数が 2 でない")
    check(len(proto[["method", "fold"]].drop_duplicates()) == 18,
          "tier4 protocol_a: (method, fold) キーに重複")
    check(bool(summ["method"].is_unique), "tier4 lolo_summary: method に重複")
    check(bool(ledger.groupby("method").size().eq(59).all()),
          "tier4 lolo_ledger: method ごとの fold 数が 59 でない")
    check(len(ledger[["method", "held_out"]].drop_duplicates()) == 531,
          "tier4 lolo_ledger: (method, held_out) キーに重複")
    for df, name in ((proto, "protocol_a"), (summ, "lolo_summary")):
        for ref in REFERENCE_METHODS:
            for suffix in ("_lo", "_hi"):
                col = f"delta_vs_{ref}{suffix}"
                if col not in df.columns:
                    failures.append(f"tier4 {name}: {col} 列が無い")
                    continue
                vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                check(bool(np.isfinite(vals).all()), f"tier4 {name}: {col} に非有限値")
                self_ci = df.loc[df["method"] == ref, col].to_numpy(dtype=float)
                check(bool((self_ci == 0.0).all()),
                      f"tier4 {name}: 参照 {ref} の自己 CI 境界が 0 でない")

    # main.tex が tier4 表 2 つ・図 1 つを \input / \includegraphics していること。
    text = MAIN_TEX.read_text(encoding="utf-8")
    for frag in (r"\input{tables/tier4_protocol_a.tex}",
                 r"\input{tables/tier4_lolo.tex}"):
        check(frag in text, f"main.tex: {frag} を \\input していない")
    check("cdf_lolo_tier4.pdf" in text,
          "main.tex: figures/cdf_lolo_tier4.pdf を \\includegraphics していない")

    # tier4_*.tex 断片が CSV から（表生成器と同一関数で）バイト再構成できること。
    present = list(dict.fromkeys(proto["method"].tolist()))
    order = _order_by_lolo(summ, present)
    checks = (
        (_protocol_tex(proto, REFERENCE_METHODS, order), "tier4_protocol_a.tex"),
        (_lolo_tex(summ, REFERENCE_METHODS, order), "tier4_lolo.tex"),
    )
    for expected_tex, fname in checks:
        got = (REPORT_DIR / "tables" / fname).read_text(encoding="utf-8")
        check(expected_tex == got, f"{fname}: CSV からの再構成と不一致")

    if len(failures) == n0:
        print("[verify_report] tier4 OK: protocol_a 18 行 / lolo_ledger 531 行 / "
              "lolo_summary 9 行、手法集合 9・delta 有限・status ok・"
              "tier4_*.tex が CSV と一致。")


def main() -> int:
    verify_table(ROOT / "results" / "protocol_a.csv",
                 REPORT_DIR / "tables" / "protocol_a.tex", _proto_line)
    verify_table(ROOT / "results" / "lolo_summary.csv",
                 REPORT_DIR / "tables" / "lolo.tex", _lolo_line)
    verify_paths()
    verify_diagnostics()
    verify_tier4()

    if failures:
        print(f"[verify_report] FAIL ({len(failures)} 件)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[verify_report] OK: 表数値 (protocol_a 30 行 / lolo 15 行)、"
          "診断値 (method_diagnostics.csv) と "
          "main.tex の \\input/\\includegraphics 参照が全て整合。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
