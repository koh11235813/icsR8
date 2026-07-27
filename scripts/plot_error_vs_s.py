#!/usr/bin/env python
"""LOLO 誤差を「廊下弧長」軸で見る図を，配色違いで一括生成する.

姉妹スクリプト plot_per_location_error.py は横軸が held_out 地点 ID という抽象量で，
「廊下のどのあたりで壊れているか」が読者に伝わりにくい．本スクリプトは
横軸を廊下弧長 s [m]（入口からの道のり），縦軸を誤差 [m] に取り，
最終報告書の図として使える形にする．

強調の付け方は plot_palette_variants.py と同一（主役ペアだけ配色色，脇役は
グレー階調）で，配色定義 PALETTES / method_colors はこのモジュールが持ち，
plot_palette_variants.py 側が import して共有する．二重管理を避けるため
色をいじるならここ 1 箇所だけを直せばよい．

凍結契約: 出力先は results/extra/ 配下に限定し，doc/final_report/figures/ の
凍結 PDF 群には絶対に書き込まない．
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# 呼び出し側が GUI バックエンドを初期化済みでも落ちないよう強制する
# （harness_tier4.make_figures_tier4 と同じ理由・同じ対処）
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icsr8.corridor import segment_of, xy_to_arclength  # noqa: E402

# 主役ペアの配色候補．(gp_corridor 色, wcl 色) の順．
# 線色と点色は同一にする方針なので，1手法あたり1色しか要らない．
# いずれも Okabe-Ito 系を基礎にしており，色覚多様性と白黒印刷時の
# 明度差の両方を確保している．
PALETTES: dict[str, tuple[str, str]] = {
    "v1_blue_vermillion": ("#0072B2", "#D55E00"),
    "v2_teal_magenta": ("#009E73", "#CC79A7"),
    "v3_navy_amber": ("#1F3B73", "#E69F00"),
    "v4_ink_crimson": ("#2B2B2B", "#C1272D"),
    "v5_purple_green": ("#6A3D9A", "#33A02C"),
}

# 主役2手法．PALETTES の色はこの2つにだけ割り当てる
PAIR_METHODS = ("gp_corridor", "wcl")

TARGET_ERROR_M = 2.0  # 先行研究 3.57 m に対する本研究の目標値


def method_colors(methods: list[str], palette: str) -> dict[str, str]:
    """主役ペアに配色色を，脇役に淡いグレー階調を割り当てる.

    脇役を単色にすると 13 本が完全に重なって判別不能になるため，
    渡された順（ave 昇順を想定）に明度を変えたグレーへ散らす（良い手法ほど濃い）．

    Args:
        methods: 描画する手法名．ave 昇順で渡すこと．
        palette: PALETTES のキー．

    Returns:
        {手法名: matplotlib が解釈できる色文字列}．
    """
    lead = dict(zip(PAIR_METHODS, PALETTES[palette]))
    others = [m for m in methods if m not in lead]
    # 0.35（濃い灰）〜0.78（薄い灰）。1.0 は白で背景に消えるので使わない
    levels = np.linspace(0.35, 0.78, max(len(others), 1))
    colors = {m: str(round(float(v), 3)) for m, v in zip(others, levels)}
    colors.update(lead)
    return colors


def _with_arclength(ledger: pd.DataFrame) -> pd.DataFrame:
    """台帳に廊下弧長 s [m] 列を足す.

    # 2026-07-27 軸の是正: 当初は横軸に true_x を使っていたが，廊下が L 字で
    # セグメント C2 が x=0 に沿う縦棒区間のため，59 地点中 29 地点が true_x=0 に
    # 潰れて 1 地点 1 点に定まらなかった（C と C3 も同じ x 範囲を共有し全 x が二重）。
    # 弧長 s は L 字を 1 本に伸ばした「入口からの道のり」なので 59 地点すべてが
    # 一意になり，かつ歩行経路順に単調増加する。詳細は docs/adr/0002 を参照。
    """
    df = ledger.copy()
    df["s"] = [xy_to_arclength(x, y) for x, y in zip(df.true_x, df.true_y)]
    df["segment"] = [segment_of(x, y) for x, y in zip(df.true_x, df.true_y)]
    return df


def plot_error_vs_s(
    ledger: pd.DataFrame, palette: str, path: Path,
    ymax: float | None = None, title: str = "LOLO error vs corridor arc length",
) -> None:
    """廊下弧長（横軸）× 誤差（縦軸）の折れ線図を保存する.

    Args:
        ledger: lolo_ledger.csv 相当（method/held_out/error/true_x/true_y）．
        palette: PALETTES のキー．
        path: 出力パス．拡張子で形式が決まる（.pdf / .png）．
        ymax: 縦軸の上限 [m]．cla など一部手法が 30 m 超の外れ値を出すため，
            素の自動スケールだと主役 gp_corridor（最大 3.5 m 程度）と wcl の
            対比が図の下端に潰れて読めない．上限を切ると脇役の線は画面外へ
            出るが，主役の対比を見せるのが本図の目的なので許容する．
            切り捨てた点数は図中に明記して，読者が「外れ値が無い」と
            誤読しないようにする．
    """
    # 凡例は ave 昇順（良い手法が上）— 既存 harness._plot_cdf と同じ順序契約
    order = ledger.groupby("method")["error"].mean().sort_values().index.tolist()
    colors = method_colors(order, palette)

    df = _with_arclength(ledger)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # 廊下セグメントの境界を薄い帯で示す。弧長軸では L 字の各直線区間が
    # そのまま連続した s 区間になるので，誤差の山がどの区間に属するか読める
    # 台帳により地点キーの列名が異なる（LOLO: held_out / Protocol A: location_p）ので
    # s 自体で重複を落とす。s は地点と 1:1 なのでキー列に依存せずに済む
    geo = df.drop_duplicates("s").sort_values("s")
    # 隣接区間で濃淡を交互にしないと境界が見えず，帯を敷く意味が無くなる
    band = ("#f0f0f0", "#ffffff")
    for k, (seg, grp) in enumerate(geo.groupby("segment", sort=True)):
        ax.axvspan(grp.s.min(), grp.s.max(), color=band[k % 2], zorder=0)
        ax.text(grp.s.median(), 1.0, seg, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color="#888", zorder=1)

    # 脇役から描いて主役を最後に重ねる（主役が必ず手前に来るように）
    for method in reversed(order):
        sub = df[df.method == method].sort_values("s")
        if sub.empty:
            continue
        is_lead = method in PAIR_METHODS
        color = colors[method]
        # s は歩行経路順に単調増加するので，分断せず 1 本の折れ線で描ける
        ax.plot(
            sub.s, sub.error,
            marker="o",
            markersize=3.5 if is_lead else 2.5,
            color=color,             # 線色と
            markerfacecolor=color,   # 点色を一致させる
            markeredgecolor=color,
            lw=2.2 if is_lead else 1.0,
            alpha=1.0 if is_lead else 0.7,
            zorder=5 if is_lead else 2,
            label=f"{method} ({sub.error.mean():.2f} m)",
        )

    ax.axhline(TARGET_ERROR_M, color="crimson", ls="--", lw=1.0, zorder=1)
    ax.text(ax.get_xlim()[0], TARGET_ERROR_M, " target 2 m",
            color="crimson", fontsize=8, va="bottom", ha="left")

    ax.set_xlabel("corridor arc length  s [m]")
    ax.set_ylabel("LOLO error [m]")
    # 配色名はタイトルに出さない（ファイル名で識別できるし，報告書に貼るとき
    # "[v1_blue_vermillion]" が残っていると事故になる）
    ax.set_title(title, fontsize=10)
    ax.set_xlim(df.s.min(), df.s.max())
    ax.set_ylim(bottom=0, top=ymax)
    if ymax is not None:
        # 画面外へ出た点を明示する。黙って切ると図が外れ値の存在を隠すことになる
        n_clipped = int((ledger["error"] > ymax).sum())
        if n_clipped:
            # 左上に置く: 右上は凡例（少数手法時は図内配置）と衝突する
            ax.text(0.012, 0.985,
                    f"{n_clipped} points > {ymax:g} m not shown",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=7, color="#555")
    ax.grid(alpha=0.3)
    # 描画は「脇役→主役」順だが，凡例は ave 昇順（良い手法が上）に並べ替える。
    # matplotlib は描画順で凡例を集めるため，明示的に並べ直さないと最悪の手法が
    # 先頭に来てしまう。手法数が多いので凡例は図の外へ逃がす
    handles = {lbl.split(" (")[0]: h
               for h, lbl in zip(*ax.get_legend_handles_labels())}
    labels = {lbl.split(" (")[0]: lbl for lbl in ax.get_legend_handles_labels()[1]}
    # 少数手法なら図内に収める（図外に逃がすと余白ばかりの間延びした図になる）
    outside = len(order) > 5
    ax.legend([handles[m] for m in order if m in handles],
              [labels[m] for m in order if m in labels],
              fontsize=8 if not outside else 7, ncol=1,
              loc="center left" if outside else "upper right",
              bbox_to_anchor=(1.01, 0.5) if outside else None)

    path.parent.mkdir(parents=True, exist_ok=True)
    # CreationDate を潰して同一入力なら同一バイト列になるようにする
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, default=Path("results/lolo_ledger.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("results/extra/error_vs_s"))
    ap.add_argument("--palette", choices=sorted(PALETTES), default=None,
                    help="省略時は全5配色を生成")
    ap.add_argument("--ymax", type=float, default=12.0,
                    help="縦軸上限 [m]（0 以下で自動スケール）")
    # 15 手法すべてを重ねると中央帯（1〜4 m）がグレー線で埋まり，本図の主張である
    # gp_corridor と wcl の対比が潜ってしまう。既定は主役2本＋参照1本に絞る
    ap.add_argument("--methods", default="gp_corridor,wcl,wknn",
                    help="カンマ区切り。'all' で台帳の全手法")
    args = ap.parse_args()

    ymax = args.ymax if args.ymax > 0 else None
    ledger = pd.read_csv(args.ledger)
    if args.methods != "all":
        wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
        ledger = ledger[ledger.method.isin(wanted)]
    # Protocol A 台帳は fold 列（forward_to_backward / backward_to_forward）を持つので
    # 方向ごとに 1 枚に分ける。両方向を混ぜると身体遮蔽・保持方向による
    # 方向依存オフセットが平均で打ち消され、本図で見たい差が消える。
    # LOLO 台帳には fold 列が無い（train=往路・test=復路と定義が固定されており
    # 方向の分割が構造上存在しない）ので、その場合は 1 枚だけ出す。
    if "fold" in ledger.columns:
        groups = [(str(f), g) for f, g in ledger.groupby("fold", sort=True)]
    else:
        groups = [("", ledger)]

    for pname in (args.palette,) if args.palette else sorted(PALETTES):
        for ext in ("pdf", "png"):
            for fold, grp in groups:
                suffix = f"_{fold}" if fold else ""
                title = (f"Protocol A ({fold}) error vs corridor arc length"
                         if fold else "LOLO error vs corridor arc length")
                out = args.outdir / f"error_vs_s{suffix}_{pname}.{ext}"
                plot_error_vs_s(grp, pname, out, ymax=ymax, title=title)
                print(f"wrote {out}")


if __name__ == "__main__":
    main()
