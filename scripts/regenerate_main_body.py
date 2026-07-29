"""本文 Tier 1–3（15 手法）+ method_diagnostics.csv を再生成する引数ゼロ CLI。

実体は icsr8.report.regenerate_main_body()（sanctioned writer）。凍結対象
（doc/final_report/tables/{protocol_a,lolo}.tex・figures/{cdf_protocol_a_*,
cdf_lolo,segment_heatmap}.pdf）へ書けるのはこの関数だけ。旧 run_all_methods.py
と違い、手法リストも出力先も引数で選べない（選べる余地自体が
2026-07-14 contamination incident のような事故の温床だった）。

使用例:
    uv run python scripts/regenerate_main_body.py
"""

from __future__ import annotations

from icsr8.report import regenerate_main_body


def main() -> int:
    regenerate_main_body()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
