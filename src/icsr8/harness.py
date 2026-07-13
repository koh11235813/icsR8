"""評価ハーネス（Protocol A / LOLO の sweep・図表生成）の import 可能なコア。

scripts/run_all_methods.py は本モジュールの薄い CLI ラッパにすぎない。
sweep 本体・図・TeX 断片の生成をここに集約し、テスト可能な純関数に保つ。

手法一覧は実行時に icsr8.methods.available_methods() から解決する（本モジュールは
今日の手法名を一切ハードコードしない）。並行開発中は methods/*.py の一部が
まだ壊れていることがあるため、icsr8.methods の import は遅延・リトライ付きで行う。
"""

from __future__ import annotations

import os

# Why not make_figures 内で matplotlib.use("Agg"): pytest が別モジュール経由で
# pyplot を先に import 済みだと use() は警告付き no-op になる。最初期に環境変数を
# 立て、以降どの matplotlib import も Agg（GUI 非依存）で固定する。
os.environ.setdefault("MPLBACKEND", "Agg")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import ModuleType  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icsr8.constants import RANDOM_SEED  # noqa: E402
from icsr8.corridor import segment_of  # noqa: E402
from icsr8.evaluate import (  # noqa: E402
    bootstrap_ci_paired,
    errors_ledger,
    percentiles,
    summary,
    within_ratio,
)
from icsr8.protocols import iter_lolo, iter_protocol_a  # noqa: E402

# --- スキーマ契約 ------------------------------------------------------------

PROTOCOL_A_RESULT_COLUMNS = [
    "method", "fold", "ave", "median", "p75", "p90", "max", "std",
    "within_2m", "within_4m", "ci_lo", "ci_hi",
    "delta_vs_wcl", "delta_lo", "delta_hi", "failed",
]
PROTOCOL_A_LEDGER_COLUMNS = ["method", "fold", "location_p", "error", "true_x", "true_y"]
LOLO_LEDGER_COLUMNS = ["method", "held_out", "error", "true_x", "true_y"]
LOLO_SUMMARY_COLUMNS = ["method", "ave", "median", "p90", "within_2m"]

_SEGMENTS: tuple[str, ...] = ("C", "C2", "C3")

_METHODS_IMPORT_RETRIES = 5
_METHODS_IMPORT_WAIT_S = 30.0


# --- methods パッケージの遅延・リトライ import -------------------------------

def _get_methods_module() -> ModuleType:
    """icsr8.methods を遅延 import する（並行開発中の ImportError を吸収）。

    Why not top-level import: 兄弟エージェントが未完成の methods/*.py を置くと
    pkgutil 自動探索が ImportError を投げ、`import icsr8.harness` ごと巻き添えに
    してテスト収集まで失敗させる。ここで囲えば harness の import 自体は常に成功する。
    Why retry-with-sleep: 未完成モジュールは数十秒で修正される見込みなので、
    30 秒間隔で最大 5 回まで再試行する。成功後は sys.modules にキャッシュされ、
    2 回目以降の呼び出しは即座に返る（全 methods が正常な通常時は一切 sleep しない）。
    """
    last: Exception | None = None
    for attempt in range(1, _METHODS_IMPORT_RETRIES + 1):
        try:
            import icsr8.methods as methods_mod

            return methods_mod
        except ImportError as exc:
            last = exc
            if attempt < _METHODS_IMPORT_RETRIES:
                print(
                    f"[harness] icsr8.methods import failed "
                    f"(attempt {attempt}/{_METHODS_IMPORT_RETRIES}): {exc}; "
                    f"retrying in {_METHODS_IMPORT_WAIT_S:.0f}s",
                    file=sys.stderr,
                )
                time.sleep(_METHODS_IMPORT_WAIT_S)
    raise ImportError(
        f"icsr8.methods import failed after {_METHODS_IMPORT_RETRIES} attempts"
    ) from last


def resolve_available_methods() -> list[str]:
    """実行時に登録済み手法名を昇順で返す（CLI の --methods 既定値解決に使う）。"""
    return _get_methods_module().available_methods()


# --- 誤差 → 指標 -------------------------------------------------------------

def _error_metrics(errors: pd.Series, *, seed: int, B: int) -> dict[str, float]:
    """1 fold・1 手法の誤差ベクトルから metric 群を計算する。"""
    stats = summary(errors)
    pct = percentiles(errors, (50, 75, 90))
    ci = bootstrap_ci_paired(errors, stat="mean", B=B, seed=seed)
    return {
        "ave": stats["Ave"],
        "median": pct["p50"],
        "p75": pct["p75"],
        "p90": pct["p90"],
        "max": stats["Max"],
        "std": stats["Std"],
        "within_2m": within_ratio(errors, 2.0),
        "within_4m": within_ratio(errors, 4.0),
        "ci_lo": ci["lo"],
        "ci_hi": ci["hi"],
    }


def _paired_delta(
    method_led: pd.DataFrame, wcl_led: pd.DataFrame, *, seed: int, B: int
) -> dict[str, float]:
    """手法と wcl の同一 location をペアにした差分 bootstrap（mean(method - wcl)）。"""
    # Why not 全 index が一致する前提: 手法が一部 location を欠落させても
    # ペアリング破綻で例外を投げないよう、共通 location（昇順）に揃えてから渡す。
    common = sorted(set(method_led.index) & set(wcl_led.index))
    a = method_led.loc[common, "error"]
    b = wcl_led.loc[common, "error"]
    return bootstrap_ci_paired(a, b, stat="mean", B=B, seed=seed)


# --- Protocol A --------------------------------------------------------------

def run_protocol_a(
    methods: list[str],
    scans_f: pd.DataFrame,
    scans_b: pd.DataFrame,
    ap13: pd.DataFrame,
    truth: pd.DataFrame,
    seed: int = RANDOM_SEED,
    B: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Protocol A（2 fold: forward↔backward）× 各手法の metric 行と誤差台帳を返す。

    AP 座標は uses_geometry の真偽に依らず全手法へ 13-AP 3F 表 `ap13` を一様に渡す。
    Why not 分岐: 指紋のみの手法（uses_geometry=False）は ap13 を単に無視するため、
    一様呼び出しで台帳生成側の分岐を消せる。

    Why not fail-fast: 1 手法が fold で例外を投げても sweep は続行し、その行を NaN と
    failed=True で埋める。8 時間の夜間 sweep を 1 手法の失敗で全損させないため。
    """
    run_method = _get_methods_module().run_method

    result_rows: list[dict] = []
    ledger_frames: list[pd.DataFrame] = []

    for fold in iter_protocol_a(scans_f, scans_b):
        # pass 1: 各手法の誤差台帳を収集（失敗は failures に記録）
        ledgers: dict[str, pd.DataFrame] = {}
        failures: set[str] = set()
        for method in methods:
            try:
                est = run_method(method, fold.train_scans, fold.test_scans, ap13, truth)
                truth_fold = truth[truth["location_p"].isin(est["location_p"])]
                led = errors_ledger(est, truth_fold, method)
            except Exception as exc:  # noqa: BLE001 - sweep を殺さない
                failures.add(method)
                print(
                    f"[harness] protocol_a fold={fold.name} method={method} FAILED: {exc}",
                    file=sys.stderr,
                )
                continue
            ledgers[method] = led
            enriched = led.reset_index().merge(
                truth_fold.rename(columns={"x": "true_x", "y": "true_y"}),
                on="location_p",
                how="left",
            )
            enriched["fold"] = fold.name
            ledger_frames.append(enriched[PROTOCOL_A_LEDGER_COLUMNS])

        wcl_led = ledgers.get("wcl")

        # pass 2: metric 行（delta 参照は上で確定した同 fold の wcl 台帳）
        for method in methods:
            if method in failures:
                row = {col: np.nan for col in PROTOCOL_A_RESULT_COLUMNS}
                row.update({"method": method, "fold": fold.name, "failed": True})
                result_rows.append(row)
                continue

            errors = ledgers[method]["error"]
            row: dict = {"method": method, "fold": fold.name, "failed": False}
            row.update(_error_metrics(errors, seed=seed, B=B))
            if method == "wcl":
                row["delta_vs_wcl"] = row["delta_lo"] = row["delta_hi"] = 0.0
            elif wcl_led is not None:
                delta = _paired_delta(ledgers[method], wcl_led, seed=seed, B=B)
                row["delta_vs_wcl"] = delta["stat"]
                row["delta_lo"] = delta["lo"]
                row["delta_hi"] = delta["hi"]
            else:
                # wcl が methods に無い / 当該 fold で失敗 → 基準が無く delta 不能
                row["delta_vs_wcl"] = row["delta_lo"] = row["delta_hi"] = np.nan
            result_rows.append(row)

    results = pd.DataFrame(result_rows, columns=PROTOCOL_A_RESULT_COLUMNS)
    ledgers_df = (
        pd.concat(ledger_frames, ignore_index=True)
        if ledger_frames
        else pd.DataFrame(columns=PROTOCOL_A_LEDGER_COLUMNS)
    )
    return results, ledgers_df


# --- LOLO --------------------------------------------------------------------

def run_lolo(
    methods: list[str],
    scans_train_pool: pd.DataFrame,
    scans_test_pool: pd.DataFrame,
    ap13: pd.DataFrame,
    truth: pd.DataFrame,
    seed: int = RANDOM_SEED,
    max_folds: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-One-Location-Out sweep。台帳（1 fold=1 誤差）とプール要約を返す。

    train = forward プールから held-out location を除いた 58 地点、
    test = 同 location の backward スキャン。空間汎化（未学習地点）と方向汎化
    （学習と逆向き）を同時に測る設定。59 fold を各手法で回す。

    `seed` は将来 bootstrap を足す際の再現性のために受け取るが、プール要約は
    決定的統計（平均・中央値・分位・比率）のみで現状は消費しない。
    """
    run_method = _get_methods_module().run_method

    folds = list(iter_lolo(scans_train_pool, scans_test_pool))
    if max_folds is not None:
        folds = folds[:max_folds]
    n = len(folds)

    ledger_rows: list[dict] = []
    for method in methods:
        for i, fold in enumerate(folds, start=1):
            print(
                f"[harness] lolo {method} fold {i}/{n} (held_out={fold.held_out})",
                file=sys.stderr,
            )
            try:
                est = run_method(method, fold.train_scans, fold.test_scans, ap13, truth)
                truth_fold = truth[truth["location_p"].isin(est["location_p"])]
                led = errors_ledger(est, truth_fold, method)
                error = float(led["error"].iloc[0]) if len(led) else np.nan
            except Exception as exc:  # noqa: BLE001 - sweep を殺さない
                print(
                    f"[harness] lolo {method} fold {i}/{n} FAILED: {exc}",
                    file=sys.stderr,
                )
                error = np.nan
            true = truth[truth["location_p"] == fold.held_out]
            ledger_rows.append(
                {
                    "method": method,
                    "held_out": fold.held_out,
                    "error": error,
                    "true_x": float(true["x"].iloc[0]) if len(true) else np.nan,
                    "true_y": float(true["y"].iloc[0]) if len(true) else np.nan,
                }
            )

    ledger = pd.DataFrame(ledger_rows, columns=LOLO_LEDGER_COLUMNS)

    summary_rows: list[dict] = []
    for method in methods:
        errs = ledger.loc[ledger["method"] == method, "error"].to_numpy(dtype=float)
        finite = errs[np.isfinite(errs)]
        if len(finite) == 0:
            summary_rows.append(
                {"method": method, "ave": np.nan, "median": np.nan,
                 "p90": np.nan, "within_2m": np.nan}
            )
        else:
            summary_rows.append(
                {
                    "method": method,
                    "ave": float(finite.mean()),
                    "median": float(np.median(finite)),
                    "p90": float(np.percentile(finite, 90)),
                    "within_2m": float(np.mean(finite <= 2.0)),
                }
            )
    summary_df = pd.DataFrame(summary_rows, columns=LOLO_SUMMARY_COLUMNS)
    return ledger, summary_df


# --- 図 ----------------------------------------------------------------------

def make_figures(ledgers: dict[str, pd.DataFrame], outdir: str | Path) -> list[Path]:
    """CDF（Protocol-A fold ごと + LOLO）と segment 別ヒートマップを PDF 出力する。

    `ledgers` は {"protocol_a": 長形式台帳, "lolo": 長形式台帳} の dict。
    各台帳は [method, error, true_x, true_y] を含むこと（Protocol-A は fold 列も）。
    生成した PDF パスのリストを返す。
    """
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    pa = ledgers.get("protocol_a")
    lolo = ledgers.get("lolo")

    if pa is not None and not pa.empty:
        for fold, grp in pa.groupby("fold", sort=True):
            path = outdir / f"cdf_protocol_a_{fold}.pdf"
            _plot_cdf(grp, f"Protocol A: {fold}", path, plt)
            created.append(path)

    if lolo is not None and not lolo.empty:
        path = outdir / "cdf_lolo.pdf"
        _plot_cdf(lolo, "LOLO", path, plt)
        created.append(path)

    # Why not 両方を混ぜる: segment ヒートマップの出所を 1 つに固定する。空間汎化を
    # 見たいので LOLO を優先し、--skip-lolo で不在なら Protocol-A プールに退避。
    seg_source = lolo if (lolo is not None and not lolo.empty) else pa
    if seg_source is not None and not seg_source.empty:
        path = outdir / "segment_heatmap.pdf"
        _plot_segment_heatmap(seg_source, path, plt)
        created.append(path)

    return created


def _plot_cdf(ledger: pd.DataFrame, title: str, path: Path, plt) -> None:
    # legend は ave 昇順（良い手法が上に来る）
    order = ledger.groupby("method")["error"].mean().sort_values().index.tolist()
    fig, ax = plt.subplots()
    for method in order:
        errs = np.sort(
            ledger.loc[ledger["method"] == method, "error"].dropna().to_numpy(dtype=float)
        )
        if len(errs) == 0:
            continue
        y = np.arange(1, len(errs) + 1) / len(errs)
        ax.plot(errs, y, label=method)
    ax.set_xlabel("error [m]")
    ax.set_ylabel("fraction <= x")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _safe_segment(x: float, y: float) -> str | None:
    # Why not segment_of を直接呼ぶ: 非有限座標で ValueError を投げるため、
    # 失敗 fold（true 座標欠落）を None として集計から静かに落とせるよう包む。
    if np.isfinite(x) and np.isfinite(y):
        return segment_of(x, y)
    return None


def _plot_segment_heatmap(ledger: pd.DataFrame, path: Path, plt) -> None:
    df = ledger.copy()
    df["segment"] = [_safe_segment(x, y) for x, y in zip(df["true_x"], df["true_y"])]
    methods = sorted(df["method"].unique())
    matrix = np.full((len(methods), len(_SEGMENTS)), np.nan)
    for i, m in enumerate(methods):
        for j, seg in enumerate(_SEGMENTS):
            vals = df.loc[
                (df["method"] == m) & (df["segment"] == seg), "error"
            ].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                matrix[i, j] = vals.mean()

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(_SEGMENTS)))
    ax.set_xticklabels(_SEGMENTS)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("segment")
    ax.set_ylabel("method")
    ax.set_title("mean error per (method x segment) [m]")
    for i in range(len(methods)):
        for j in range(len(_SEGMENTS)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax, label="mean error [m]")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# --- TeX 断片 ----------------------------------------------------------------

def make_tex_tables(
    results: pd.DataFrame,
    lolo_summary: pd.DataFrame | None,
    outdir: str | Path,
) -> list[Path]:
    """booktabs 風の tabular 断片（\\documentclass 無し）を 2 つ書き出す。

    行は LOLO ave 昇順に並べる（LOLO が無ければ Protocol-A ave 昇順に退避）。
    数値は %.2f。wcl_powerdomain 行には短剣符 †（数学的に WCL と等価）を付す。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    order = _method_order_by_lolo(results, lolo_summary)

    proto_path = outdir / "protocol_a.tex"
    proto_path.write_text(_protocol_a_tex(results, order), encoding="utf-8")

    lolo_path = outdir / "lolo.tex"
    lolo_path.write_text(_lolo_tex(lolo_summary, order), encoding="utf-8")

    return [proto_path, lolo_path]


def _method_order_by_lolo(
    results: pd.DataFrame, lolo_summary: pd.DataFrame | None
) -> list[str]:
    if lolo_summary is not None and not lolo_summary.empty:
        return lolo_summary.sort_values("ave", na_position="last")["method"].tolist()
    agg = results.groupby("method")["ave"].mean().sort_values(na_position="last")
    return agg.index.tolist()


def _order_present(order: list[str], present: list[str]) -> list[str]:
    # order に載らない手法（LOLO を回さなかった等）も落とさず末尾に付ける。
    return [m for m in order if m in present] + [m for m in present if m not in order]


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _fmt(value) -> str:
    return f"{value:.2f}" if value is not None and np.isfinite(value) else "--"


def _label(method: str) -> str:
    label = _tex_escape(method)
    if method == "wcl_powerdomain":
        label += r"$^\dagger$"
    return label


_POWERDOMAIN_NOTE = r"\par\footnotesize $^\dagger$ wcl\_powerdomain mathematically $\equiv$ WCL."


def _protocol_a_tex(results: pd.DataFrame, order: list[str]) -> str:
    cols = [
        ("Ave", "ave"), ("Median", "median"), ("P90", "p90"), ("Max", "max"),
        ("Within 2m", "within_2m"), (r"$\Delta$WCL", "delta_vs_wcl"),
    ]
    present = list(dict.fromkeys(results["method"].tolist()))
    ordered = _order_present(order, present)

    lines = [
        r"\begin{tabular}{ll" + "r" * len(cols) + "}",
        r"\toprule",
        "Method & Fold & " + " & ".join(h for h, _ in cols) + r" \\",
        r"\midrule",
    ]
    for method in ordered:
        rows = results[results["method"] == method].sort_values("fold")
        for _, r in rows.iterrows():
            cells = [_label(method), _tex_escape(str(r["fold"]))]
            cells += [_fmt(r[key]) for _, key in cols]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if "wcl_powerdomain" in present:
        lines.append(_POWERDOMAIN_NOTE)
    return "\n".join(lines) + "\n"


def _lolo_tex(lolo_summary: pd.DataFrame | None, order: list[str]) -> str:
    if lolo_summary is None or lolo_summary.empty:
        return "% LOLO summary unavailable (--skip-lolo)\n"

    cols = [("Ave", "ave"), ("Median", "median"), ("P90", "p90"), ("Within 2m", "within_2m")]
    present = list(dict.fromkeys(lolo_summary["method"].tolist()))
    ordered = _order_present(order, present)
    indexed = lolo_summary.set_index("method")

    lines = [
        r"\begin{tabular}{l" + "r" * len(cols) + "}",
        r"\toprule",
        "Method & " + " & ".join(h for h, _ in cols) + r" \\",
        r"\midrule",
    ]
    for method in ordered:
        r = indexed.loc[method]
        cells = [_label(method)] + [_fmt(r[key]) for _, key in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if "wcl_powerdomain" in present:
        lines.append(_POWERDOMAIN_NOTE)
    return "\n".join(lines) + "\n"
