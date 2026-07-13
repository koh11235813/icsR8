"""AP_coordinate_C_All.xlsx (全館 67 AP 分) を CSV へ変換する。

Run once after a fresh clone (only when the XLSX changes):
    uv run --with openpyxl python scripts/dump_ap_coord_all_csv.py

Output:
    data/dataset_r0701/AP_coordinate_C_All.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import openpyxl

REPO = Path(__file__).resolve().parents[1]
XLSX = REPO / "data" / "dataset_r0701" / "AP_coordinate_C_All.xlsx"
OUT_CSV = REPO / "data" / "dataset_r0701" / "AP_coordinate_C_All.csv"
SHEET_NAME = "AP_coordinate_C_All"


def main() -> None:
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb[SHEET_NAME]
    rows = list(ws.iter_rows(values_only=True))
    header, data = rows[0], rows[1:]

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)
    print(f"Wrote {len(data)} rows -> {OUT_CSV.relative_to(REPO)}")


if __name__ == "__main__":
    main()
