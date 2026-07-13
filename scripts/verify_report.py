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


def main() -> int:
    verify_table(ROOT / "results" / "protocol_a.csv",
                 REPORT_DIR / "tables" / "protocol_a.tex", _proto_line)
    verify_table(ROOT / "results" / "lolo_summary.csv",
                 REPORT_DIR / "tables" / "lolo.tex", _lolo_line)
    verify_paths()
    verify_diagnostics()

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
