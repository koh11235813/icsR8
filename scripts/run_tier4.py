"""Tier 4 の 7 手法を別経路で評価する薄い CLI（本体は icsr8.harness_tier4）。

Protocol A と LOLO を回し、results/tier4/ と doc/final_report の tier4_*.tex・
cdf_lolo_tier4.pdf を書き出す。個々の手法の失敗ではプロセスを落とさない
（fail-soft）。既存 results/*.csv・doc/final_report の凍結成果物には一切触れない。

使用例:
    uv run python scripts/run_tier4.py --dataset-root data --output results/tier4
    uv run python scripts/run_tier4.py --smoke --output /tmp/tier4_smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

from icsr8.constants import RANDOM_SEED
from icsr8.harness_tier4 import (
    REFERENCE_METHODS,
    TIER4_METHODS,
    run_tier4,
    subsample_scans,
)
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans

# smoke は代役手法 + segment 層化 9 地点。subsample_scans が 3 segment を跨いで
# 抜くため gp_corridor（segment 分類器が 2 クラス以上必須）も fit でき、
# references を本番と同一に保ったまま delta_vs_gp_corridor 列と基準マークを検証できる。
SMOKE_METHODS = ["wcl", "wcl_corridor"]
SMOKE_N_LOC = 9


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="icsR8 Tier 4 evaluation harness")
    p.add_argument("--dataset-root", default="data")
    p.add_argument("--output", default="results/tier4")
    p.add_argument("--tables-dir", default="doc/final_report/tables")
    p.add_argument("--figures-dir", default="doc/final_report/figures")
    p.add_argument("--methods", default=None,
                   help="comma-separated method names (default: TIER4_METHODS)")
    p.add_argument("--smoke", action="store_true",
                   help="代役手法 + 地点サブサンプルで 1 分以内に回す")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    root = Path(args.dataset_root)
    ap13 = load_ap_coords(root / "dataset" / "AP_coordinate_C3F.csv")
    truth = load_location_coords(root / "dataset" / "location_coordinate_C.csv")[
        ["location_p", "x", "y"]
    ]
    scans_f = load_raw_scans("forward", root / "rawdata")
    scans_b = load_raw_scans("backward", root / "rawdata")

    references = REFERENCE_METHODS
    if args.smoke:
        methods = SMOKE_METHODS
        B = 100
        scans_f, scans_b, truth = subsample_scans(scans_f, scans_b, truth, SMOKE_N_LOC)
    elif args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        B = 1000
    else:
        methods = TIER4_METHODS
        B = 1000

    print(f"[cli] methods={methods} references={references} "
          f"seed={RANDOM_SEED} B={B} smoke={args.smoke}")
    written = run_tier4(
        scans_f=scans_f,
        scans_b=scans_b,
        ap13=ap13,
        truth=truth,
        methods=methods,
        references=references,
        output_dir=args.output,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        seed=RANDOM_SEED,
        B=B,
    )
    print("[cli] wrote:")
    for key, path in written.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
