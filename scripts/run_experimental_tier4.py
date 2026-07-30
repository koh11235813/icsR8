"""追試専用 Tier 4 評価 CLI（本体は icsr8.harness_tier4.run_tier4）。

sanctioned writer ではない（icsr8.harness_tier4.run_tier4 へ writer_id を渡さない）
ため、--tables-dir/--figures-dir が凍結パス（doc/final_report 配下の 9 ファイル、
harness_tier4.FROZEN_OUTPUT_PATHS）と衝突すれば _guard_frozen が ValueError で
reject する（run_tier4 冒頭のガードに加え、2026-07-29 以降は
make_figures_tier4/make_tex_tables_tier4 自身の冒頭でも同じガードが働く
belt-and-suspenders — 2026-07-29 review pivot）。
--methods/--output/--tables-dir/--figures-dir は全て required —
省略時に既定の doc/final_report を汚す事故（2026-07-14 contamination incident）を、
argparse レベルで構造的に不可能にする。

# 2026-07-29: 旧 run_tier4.py は --smoke が --methods の中身を無条件に上書きして
# いた（args.smoke を args.methods より先に評価する分岐）。--methods を required
# にした以上、「必ず渡した引数がこっそり無視される」footgun になるため、この CLI
# では --smoke と --methods を独立に扱う: --smoke は B と地点サブサンプルだけを
# 制御し、--methods はどちらの場合も常にそのまま使う。

使用例:
    uv run python scripts/run_experimental_tier4.py \\
        --methods wcl_virtual_ap --output results/extra/vwcl \\
        --tables-dir results/extra/vwcl --figures-dir results/extra/vwcl
    uv run python scripts/run_experimental_tier4.py \\
        --methods wcl_virtual_ap --smoke --output /tmp/x --tables-dir /tmp/x --figures-dir /tmp/x
"""

from __future__ import annotations

import argparse
from pathlib import Path

from icsr8.constants import RANDOM_SEED
from icsr8.harness_tier4 import REFERENCE_METHODS, run_tier4, subsample_scans
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans

# smoke は segment 層化 9 地点サブサンプル（gp_corridor の segment 分類器が
# 2 クラス以上必須なため、3 segment を跨いで抜く subsample_scans を使う）。
SMOKE_N_LOC = 9


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """`--methods`/`--output`/`--tables-dir`/`--figures-dir` を全て required にする。

    この CLI は sanctioned writer ではないので、どれか 1 つでも省略できると
    「意図せず既定のディレクトリへ書いてしまう」余地が argparse レベルで
    残ってしまう（2026-07-14 contamination incident と同じ形の事故）。required
    化はその余地そのものを消す構造的な防御であり、実行時チェックではない。
    """
    p = argparse.ArgumentParser(
        description="icsR8 experimental Tier 4 evaluation（追試・新手法専用、隔離出力）"
    )
    p.add_argument("--dataset-root", default="data")
    p.add_argument("--output", required=True)
    p.add_argument("--tables-dir", required=True)
    p.add_argument("--figures-dir", required=True)
    p.add_argument(
        "--methods", required=True, help="comma-separated method names"
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="B=100 + segment 層化サブサンプルで高速化（--methods はそのまま使う）",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """追試・新手法・配管確認用の Tier 4 評価を実行し、指定した隔離出力先へのみ書く。

    `run_tier4()` に writer_id を渡さないため、`--tables-dir`/`--figures-dir` に
    凍結ディレクトリ（doc/final_report 配下）を指定すると `_guard_frozen` が
    ValueError で拒否する契約を、この CLI の呼び出し側では一切迂回しない。
    """
    args = _parse_args(argv)

    root = Path(args.dataset_root)
    ap13 = load_ap_coords(root / "dataset" / "AP_coordinate_C3F.csv")
    truth = load_location_coords(root / "dataset" / "location_coordinate_C.csv")[
        ["location_p", "x", "y"]
    ]
    scans_f = load_raw_scans("forward", root / "rawdata")
    scans_b = load_raw_scans("backward", root / "rawdata")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    references = REFERENCE_METHODS
    B = 1000
    if args.smoke:
        B = 100
        scans_f, scans_b, truth = subsample_scans(scans_f, scans_b, truth, SMOKE_N_LOC)

    print(f"[cli] methods={methods} references={references} "
          f"seed={RANDOM_SEED} B={B} smoke={args.smoke}")
    # writer_id を渡さない（デフォルト None）→ 凍結パスへの書き込みは
    # harness_tier4._guard_frozen が構造的に reject する。
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
