"""全手法の評価 sweep を回す薄い CLI（本体は icsr8.harness）。

Protocol A と LOLO を回し、CSV・図（CDF/segment ヒートマップ）・TeX 断片を書き出し、
最後に要約表を stdout へ出す。個々の手法の失敗ではプロセスを落とさず、ハーネス自体の
バグ（import 不能・想定外例外）でのみ非ゼロ終了する。

使用例:
    uv run python scripts/run_all_methods.py --smoke --output /tmp/icsr8_smoke_results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from icsr8.harness import (
    make_figures,
    make_tex_tables,
    resolve_available_methods,
    run_lolo,
    run_protocol_a,
)
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="icsR8 evaluation harness sweep")
    p.add_argument("--dataset-root", default="data")
    p.add_argument("--output", default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--methods",
        default=None,
        help="comma-separated method names (default: all available)",
    )
    p.add_argument("--smoke", action="store_true", help="wcl,wcl_corridor + max_folds=3 + B=100")
    p.add_argument("--skip-lolo", action="store_true")
    p.add_argument("--figures-dir", default="doc/final_report/figures")
    p.add_argument("--tables-dir", default="doc/final_report/tables")
    return p.parse_args(argv)


def _resolve_methods(args: argparse.Namespace) -> list[str]:
    if args.smoke:
        return ["wcl", "wcl_corridor"]
    if args.methods:
        return [m.strip() for m in args.methods.split(",") if m.strip()]
    return resolve_available_methods()


def _print_summary(results: pd.DataFrame, lolo_summary: pd.DataFrame | None) -> None:
    # Protocol-A ave は fold 平均に畳む。表示順は LOLO ave 昇順（無ければ protoA ave）。
    proto = results.groupby("method")["ave"].mean().rename("protoA_ave")
    if lolo_summary is not None and not lolo_summary.empty:
        merged = lolo_summary.set_index("method").join(proto, how="outer")
        merged = merged.sort_values("ave", na_position="last")
        header = f"{'method':<18}{'protoA_ave':>12}{'lolo_ave':>10}{'lolo_within2m':>15}"
        print("\n=== Summary (sorted by LOLO ave) ===")
        print(header)
        for method, row in merged.iterrows():
            print(
                f"{method:<18}{_num(row.get('protoA_ave')):>12}"
                f"{_num(row.get('ave')):>10}{_num(row.get('within_2m')):>15}"
            )
    else:
        merged = proto.sort_values(na_position="last")
        print("\n=== Summary (sorted by Protocol-A ave; LOLO skipped) ===")
        print(f"{'method':<18}{'protoA_ave':>12}")
        for method, value in merged.items():
            print(f"{method:<18}{_num(value):>12}")


def _num(value) -> str:
    return f"{value:.2f}" if value is not None and np.isfinite(value) else "--"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    methods = _resolve_methods(args)
    B = 100 if args.smoke else 1000
    max_folds = 3 if args.smoke else None

    root = Path(args.dataset_root)
    ap13 = load_ap_coords(root / "dataset" / "AP_coordinate_C3F.csv")
    truth = load_location_coords(root / "dataset" / "location_coordinate_C.csv")[
        ["location_p", "x", "y"]
    ]
    scans_f = load_raw_scans("forward", root / "rawdata")
    scans_b = load_raw_scans("backward", root / "rawdata")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"[cli] methods={methods} seed={args.seed} B={B} smoke={args.smoke}")
    results, pa_ledgers = run_protocol_a(
        methods, scans_f, scans_b, ap13, truth, seed=args.seed, B=B
    )
    results.to_csv(output / "protocol_a.csv", index=False)

    lolo_ledger = None
    lolo_summary = None
    if not args.skip_lolo:
        lolo_ledger, lolo_summary = run_lolo(
            methods, scans_f, scans_b, ap13, truth, seed=args.seed, max_folds=max_folds
        )
        lolo_ledger.to_csv(output / "lolo_ledger.csv", index=False)
        lolo_summary.to_csv(output / "lolo_summary.csv", index=False)

    make_figures({"protocol_a": pa_ledgers, "lolo": lolo_ledger}, args.figures_dir)
    make_tex_tables(results, lolo_summary, args.tables_dir)

    _print_summary(results, lolo_summary)
    print(f"\n[cli] wrote CSVs to {output}/, figures to {args.figures_dir}/, "
          f"tables to {args.tables_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
