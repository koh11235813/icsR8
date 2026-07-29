"""付録 A（Tier 4、7 手法）を再生成する引数ゼロ CLI。

実体は icsr8.report.regenerate_appendix_a()（sanctioned writer）。凍結対象
（doc/final_report/tables/tier4_{protocol_a,lolo}.tex・
figures/cdf_lolo_tier4.pdf）へ書けるのはこの関数だけ。追試・新手法の評価には
これではなく scripts/run_experimental_tier4.py（隔離出力先）を使う。

使用例:
    uv run python scripts/regenerate_appendix_a.py
"""

from __future__ import annotations

from icsr8.report import regenerate_appendix_a


def main() -> int:
    """引数を一切受け取らない薄い CLI エントリポイント。

    追試・新手法用の隔離出力（run_experimental_tier4.py）と役割を分けるため、
    出力先を選べる引数を意図的に持たない。
    """
    regenerate_appendix_a()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
