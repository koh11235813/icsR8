#!/usr/bin/env python
"""最終報告書の図（CDF・セグメントヒートマップ）を配色違いで再描画する.

harness.make_figures / harness_tier4.make_figures_tier4 は matplotlib の既定
カラーサイクルをそのまま使うため，15 手法が同じ太さ・似た色で重なり，
「gp_corridor が主役」という本文の主張が図から読み取れない．
本スクリプトは同じ台帳から同じ図を描き直しつつ，

  - 主役ペア（gp_corridor / wcl）だけを配色パターンの色で強調
  - 残りの手法は淡いグレー階調へ退避

という強弱を付ける．配色は plot_error_vs_s.py の PALETTES を共有し，
散布図側と報告書の図で色が食い違わないようにしている．

凍結契約: 出力は results/extra/ 配下のみ．doc/final_report/figures/ には
一切書き込まない（凍結 PDF の再生成は README の専用手順でしか行わない）．
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# scripts/ をスクリプト実行すると sys.path[0] に入るので直接 import できる。
# 配色定義を二重管理しないための共有（値をいじるなら plot_error_vs_s.py 側）
from plot_error_vs_s import PAIR_METHODS, PALETTES, method_colors  # noqa: E402

from icsr8.corridor import segment_of  # noqa: E402

# ヒートマップは 2 色ペアでは表現できないため，配色パターンごとに
# 連続カラーマップを対応付ける。順序は PALETTES と揃えてある
HEATMAP_CMAPS: dict[str, str] = {
    "v1_blue_vermillion": "viridis",
    "v2_teal_magenta": "YlGnBu",
    "v3_navy_amber": "cividis",
    "v4_ink_crimson": "magma",
    "v5_purple_green": "BuPu",
}

_SEGMENTS = ("C", "C2", "C3")


def plot_cdf(ledger: pd.DataFrame, title: str, palette: str, path: Path) -> None:
    """誤差の経験 CDF を，主役ペアだけ強調して描く.

    Args:
        ledger: [method, error] を含む台帳．
        title: 図タイトル．
        palette: PALETTES のキー．
        path: 出力パス．
    """
    # 凡例は ave 昇順（良い手法が上）— 既存 harness._plot_cdf と同じ順序契約
    order = ledger.groupby("method")["error"].mean().sort_values().index.tolist()
    colors = method_colors(order, palette)

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for method in order:
        errs = np.sort(
            ledger.loc[ledger["method"] == method, "error"]
            .dropna().to_numpy(dtype=float)
        )
        if len(errs) == 0:
            continue
        y = np.arange(1, len(errs) + 1) / len(errs)
        is_lead = method in PAIR_METHODS
        ax.plot(errs, y, color=colors[method],
                lw=2.2 if is_lead else 1.0,
                alpha=1.0 if is_lead else 0.75,
                zorder=3 if is_lead else 2,
                label=method)

    ax.axvline(2.0, color="crimson", ls="--", lw=1.0, zorder=1)
    ax.set_xlabel("error [m]")
    ax.set_ylabel("fraction <= x")
    ax.set_ylim(0.0, 1.0)
    # 配色名はタイトルに出さない（識別はファイル名側で足りる）
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def plot_segment_heatmap(ledger: pd.DataFrame, palette: str, path: Path) -> None:
    """(手法 × 廊下セグメント) 別の平均誤差ヒートマップを描く.

    既存 harness._plot_segment_heatmap と同じ集計だが，行を ave 昇順に並べ替え，
    セル注記の文字色をセルの明暗に応じて反転させて可読性を確保する．
    """
    df = ledger.copy()
    # 非有限座標は失敗 fold なので静かに落とす（既存 _safe_segment と同じ方針）
    df["segment"] = [
        segment_of(x, y) if np.isfinite(x) and np.isfinite(y) else None
        for x, y in zip(df["true_x"], df["true_y"])
    ]
    methods = df.groupby("method")["error"].mean().sort_values().index.tolist()

    matrix = np.full((len(methods), len(_SEGMENTS)), np.nan)
    for i, m in enumerate(methods):
        for j, seg in enumerate(_SEGMENTS):
            vals = df.loc[(df["method"] == m) & (df["segment"] == seg), "error"]
            vals = vals.to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                matrix[i, j] = vals.mean()

    cmap = HEATMAP_CMAPS[palette]
    fig, ax = plt.subplots(figsize=(4.6, 0.34 * len(methods) + 1.6))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(_SEGMENTS)), _SEGMENTS)
    ax.set_yticks(range(len(methods)), methods, fontsize=8)
    ax.set_xlabel("segment")
    ax.set_title("mean error per (method x segment) [m]", fontsize=9)

    # 注記色をセル明度で切り替える。固定の白文字だと明るいカラーマップ
    # （cividis/YlGnBu の上端）で数字が飛んで読めなくなる
    lo, hi = np.nanmin(matrix), np.nanmax(matrix)
    for i in range(len(methods)):
        for j in range(len(_SEGMENTS)):
            if not np.isfinite(matrix[i, j]):
                continue
            norm = (matrix[i, j] - lo) / (hi - lo) if hi > lo else 0.5
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="w" if norm < 0.6 else "k")
    fig.colorbar(im, ax=ax, label="mean error [m]")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, default=Path("results/lolo_ledger.csv"))
    ap.add_argument("--tier4-ledger", type=Path,
                    default=Path("results/tier4/lolo_ledger.csv"))
    ap.add_argument("--protocol-a-ledger", type=Path,
                    default=Path("results/extra/pa_run/protocol_a_ledger.csv"),
                    help="run_all_methods.py が書き出す地点別台帳（fold 列を含む）")
    ap.add_argument("--outdir", type=Path, default=Path("results/extra/palettes"))
    ap.add_argument("--palette", choices=sorted(PALETTES), default=None,
                    help="省略時は全5配色")
    args = ap.parse_args()

    # 2026-07-23 の凍結契約刷新で results/*.csv は gitignore 対象の再生成物に
    # なった。クリーンな作業ツリーでは台帳が 1 つも無いのが正常なので、
    # 欠けている入力は例外にせず「その図だけ飛ばす」で扱う。
    def _load(path: Path, label: str) -> pd.DataFrame | None:
        if path.exists():
            return pd.read_csv(path)
        print(f"[skip] {label}: {path} が無い（scripts/run_all_methods.py で再生成）")
        return None

    lolo = _load(args.ledger, "LOLO")
    tier4 = _load(args.tier4_ledger, "Tier 4")
    pa = _load(args.protocol_a_ledger, "Protocol A")

    for pname in (args.palette,) if args.palette else sorted(PALETTES):
        for ext in ("pdf", "png"):
            if lolo is not None:
                out = args.outdir / f"cdf_lolo_{pname}.{ext}"
                plot_cdf(lolo, "LOLO", pname, out)
                print(f"wrote {out}")

                out = args.outdir / f"segment_heatmap_{pname}.{ext}"
                plot_segment_heatmap(lolo, pname, out)
                print(f"wrote {out}")

            if tier4 is not None:
                out = args.outdir / f"cdf_lolo_tier4_{pname}.{ext}"
                plot_cdf(tier4, "LOLO (Tier 4)", pname, out)
                print(f"wrote {out}")

            if pa is not None:
                # 凍結図と同じく fold ごとに 1 枚（両方向を混ぜると
                # 方向依存オフセットが平均で消えてしまう）
                for fold, grp in pa.groupby("fold", sort=True):
                    out = args.outdir / f"cdf_protocol_a_{fold}_{pname}.{ext}"
                    plot_cdf(grp, f"Protocol A: {fold}", pname, out)
                    print(f"wrote {out}")


if __name__ == "__main__":
    main()
