"""Extract baseline reference values from estimation_result_C3F.xlsx
to CSV fixtures committed to the repo.

Run once after a fresh clone (only when the XLSX changes):
    uv run --group oracle python scripts/extract_baseline_fixtures.py

Output:
    tests/fixtures/baseline_forward.csv
    tests/fixtures/baseline_backward.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import openpyxl

REPO = Path(__file__).resolve().parents[1]
XLSX = REPO / "data" / "dataset" / "estimation_result_C3F.xlsx"
OUT_DIR = REPO / "tests" / "fixtures"

SHEETS = {
    "forward": "C3F順方向",
    "backward": "C3F逆方向",
}

# 0-indexed column offsets of each method block in the XLSX
COLS = {
    "pbl": {"loc": 0, "x": 1, "y": 2, "true_x": 3, "true_y": 4, "err": 5},
    "cla": {"loc": 8, "x": 9, "y": 10, "true_x": 11, "true_y": 12, "err": 13},
    "wcl": {"loc": 16, "x": 17, "y": 18, "true_x": 19, "true_y": 20, "err": 21},
}

DATA_FIRST_ROW = 4  # 1-indexed
N_LOCATIONS = 59


def extract_sheet(ws) -> list[dict]:
    rows = list(ws.iter_rows(min_row=DATA_FIRST_ROW,
                             max_row=DATA_FIRST_ROW + N_LOCATIONS - 1,
                             values_only=True))
    out = []
    for expected_loc, r in enumerate(rows, start=1):
        loc = r[COLS["pbl"]["loc"]]
        assert loc == expected_loc, f"unexpected location order: got {loc}, want {expected_loc}"
        assert r[COLS["cla"]["loc"]] == loc, f"CLA row mismatch at {loc}"
        assert r[COLS["wcl"]["loc"]] == loc, f"WCL row mismatch at {loc}"
        true_x = r[COLS["pbl"]["true_x"]]
        true_y = r[COLS["pbl"]["true_y"]]
        # ground truth must agree across method blocks
        assert r[COLS["cla"]["true_x"]] == true_x
        assert r[COLS["cla"]["true_y"]] == true_y
        assert r[COLS["wcl"]["true_x"]] == true_x
        assert r[COLS["wcl"]["true_y"]] == true_y

        out.append({
            "location_p": int(loc),
            "true_x": float(true_x),
            "true_y": float(true_y),
            "pbl_x": float(r[COLS["pbl"]["x"]]),
            "pbl_y": float(r[COLS["pbl"]["y"]]),
            "pbl_error": float(r[COLS["pbl"]["err"]]),
            "cla_x": float(r[COLS["cla"]["x"]]),
            "cla_y": float(r[COLS["cla"]["y"]]),
            "cla_error": float(r[COLS["cla"]["err"]]),
            "wcl_x": float(r[COLS["wcl"]["x"]]),
            "wcl_y": float(r[COLS["wcl"]["y"]]),
            "wcl_error": float(r[COLS["wcl"]["err"]]),
        })
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    for direction, sheet_name in SHEETS.items():
        ws = wb[sheet_name]
        rows = extract_sheet(ws)
        out_path = OUT_DIR / f"baseline_{direction}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
