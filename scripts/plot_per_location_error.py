#!/usr/bin/env python
"""LOLO ledger の「地点ごとの誤差」を可視化する単機能スクリプト.

本文の CDF 図は誤差の分布形しか見せないため，どの地点で手法が破綻するか
（廊下セグメントの端・L 字コーナーなど位置依存の失敗）が読めない．
本スクリプトは held_out 地点 ID を横軸に取り，手法ごとの誤差を折れ線で
並べることで，その位置依存性を直接見えるようにする．

凍結契約の回避: 出力先は既定で results/extra/ であり，doc/final_report/figures/
（凍結 PDF 群）には一切書き込まない．
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# GUI バックエンドが先に初期化されていても落ちないよう強制的に Agg にする
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# scripts/ をスクリプト実行すると sys.path[0] に入るので直接 import できる。
# 配色定義を二重管理しないための共有（値をいじるなら plot_error_vs_s.py 側）
from plot_error_vs_s import PALETTES, method_colors  # noqa: E402

from icsr8.corridor import segment_of  # noqa: E402


def plot_per_location_error(
    ledger: pd.DataFrame, methods: list[str], path: Path,
    palette: str | None = None,
) -> None:
    """held_out 地点 ID 軸で手法別の誤差折れ線を描き，PDF/PNG として保存する.

    Args:
        ledger: lolo_ledger.csv 相当（method/held_out/error/true_x/true_y）．
        methods: 描画する手法名．先頭ほど強調して描く（主役を先頭に置く想定）．
        path: 出力ファイルパス．拡張子で形式が決まる．
        palette: PALETTES のキー．None なら色を指定せず matplotlib の既定
            カラーサイクルに任せる（本スクリプト初版の挙動を温存するため，
            配色指定は明示的なオプトインにしている）．
    """
    fig, ax = plt.subplots(figsize=(11, 4.2))

    # セグメント境界に薄い帯を敷く: 誤差の山がセグメント端に寄るかを目視するため
    geo = (
        ledger[["held_out", "true_x", "true_y"]]
        .drop_duplicates("held_out")
        .sort_values("held_out")
    )
    geo["segment"] = [segment_of(x, y) for x, y in zip(geo.true_x, geo.true_y)]
    shade = {s: c for s, c in zip(sorted(geo.segment.unique()), ["#f2f2f2", "#ffffff", "#e8eef5"])}
    for seg, grp in geo.groupby("segment"):
        ax.axvspan(grp.held_out.min() - 0.5, grp.held_out.max() + 0.5,
                   color=shade[seg], zorder=0)

    colors = method_colors(methods, palette) if palette else {}
    for i, m in enumerate(methods):
        sub = ledger[ledger.method == m].sort_values("held_out")
        # palette 未指定なら color を渡さない（matplotlib の既定サイクルに委ねる）。
        # 線色と点色は同一にする方針なので 1 手法 1 色で足りる
        style = {}
        if m in colors:
            style = dict(color=colors[m], markerfacecolor=colors[m],
                         markeredgecolor=colors[m])
        ax.plot(sub.held_out, sub.error, marker="o", markersize=3,
                lw=2.0 if i == 0 else 1.0,
                alpha=1.0 if i == 0 else 0.55,
                label=f"{m} (ave {sub.error.mean():.2f} m)", zorder=3 - i,
                **style)

    ax.axhline(2.0, color="crimson", ls="--", lw=1.0, zorder=2)
    ax.text(0.6, 2.05, "target 2 m", color="crimson", fontsize=8, va="bottom")

    ax.set_xlabel("held-out location ID")
    ax.set_ylabel("LOLO error [m]")
    ax.set_title("Per-location LOLO error (train = forward minus held-out, test = backward)")
    ax.set_xlim(0.5, ledger.held_out.max() + 0.5)
    ax.margins(y=0.02)
    ax.legend(fontsize=8, ncol=2)

    # セグメント名は軸の内側上端に置く。軸の外（上端より上）に出すとタイトルと
    # 重なって両方読めなくなるため、transform で軸相対に固定する
    for seg, grp in geo.groupby("segment"):
        ax.text(grp.held_out.median(), 0.98, seg,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color="#888")

    ax.grid(axis="y", alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, default=Path("results/lolo_ledger.csv"))
    ap.add_argument("--methods", nargs="+",
                    default=["gp_corridor", "wknn", "wcl"],
                    help="描画する手法（先頭が主役として強調される）")
    ap.add_argument("--out", type=Path,
                    default=Path("results/extra/per_location_error.pdf"))
    # フラグ自体を省略 -> None（matplotlib 既定色、初版と同じ図）
    # --palettes だけ書く   -> []（全 5 配色を出力）
    # --palettes v1_... v3_ -> 指定された配色のみ
    ap.add_argument("--palettes", nargs="*", choices=sorted(PALETTES), default=None,
                    metavar="NAME",
                    help="配色版を出力する。値を省くと全5配色。"
                         "フラグ自体を省くと matplotlib の既定色")
    args = ap.parse_args()

    ledger = pd.read_csv(args.ledger)

    if args.palettes is None:
        plot_per_location_error(ledger, args.methods, args.out)
        print(f"wrote {args.out}")
        return

    # 配色名はファイル名にだけ入れる（図中には出さない。報告書に貼ったとき
    # "[v1_blue_vermillion]" が残っていると事故になるため）
    for pname in args.palettes or sorted(PALETTES):
        out = args.out.with_name(f"{args.out.stem}_{pname}{args.out.suffix}")
        plot_per_location_error(ledger, args.methods, out, palette=pname)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
