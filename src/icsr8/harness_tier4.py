"""Tier 4 手法群の評価ハーネス（Protocol A / LOLO を別経路で回す）。

既存 harness.py + run_all_methods.py が出力する results/*.csv・doc/final_report の
表・図は凍結済みなので、Tier 4 の 7 手法はここで独立に評価し results/tier4/ と
tier4_*.tex にのみ書き出す。scripts/run_tier4.py は本モジュールの薄い CLI。

Why not 既存 harness を拡張する: 既存出力（protocol_a.csv 30 行等）は
scripts/verify_report.py が main.tex とバイト一致で固定しており、行の増減や
delta 列追加はその契約を壊す。Tier 4 は「主要比較基準 gp_corridor に対する
2 本の delta CI」「完全ペアリング契約」「診断 long-form」という別スキーマを
要求するため、既存関数を再利用しつつ別モジュールに分離する。
"""

from __future__ import annotations

import os

# Why not make_figures 内で use("Agg"): 先に pyplot が import 済みだと use() は
# 警告付き no-op になる。最初期に環境変数で Agg（GUI 非依存）へ固定する。
os.environ.setdefault("MPLBACKEND", "Agg")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icsr8.constants import RANDOM_SEED  # noqa: E402
from icsr8.evaluate import (  # noqa: E402
    bootstrap_ci_paired,
    errors_ledger,
    percentiles,
    summary,
    within_ratio,
)

from icsr8.corridor import segment_of  # noqa: E402

# Why not run_method を使う: run_method は fit 済み method を破棄して est のみ返す
# ため、各 fit 後の method.diagnostics_ を回収できない。診断 ledger 生成のために
# fit/predict をここで自前に組み、leak-safe フィルタ（run_method と同一）を複製する。
from icsr8.harness import _get_methods_module  # noqa: E402
from icsr8.protocols import iter_lolo, iter_protocol_a  # noqa: E402

# --- 凍結成果物ガード ----------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]

# scripts/verify_report.py が main.tex と一致固定している既存成果物。
# Why not results/ ディレクトリ丸ごと拒否: results/tier4/ など凍結外の新設先まで
# 塞いでしまう。凍結はファイル単位の契約なのでパス単位で列挙する。
FROZEN_OUTPUT_PATHS: frozenset[Path] = frozenset(
    (_REPO_ROOT / rel).resolve()
    for rel in (
        "results/protocol_a.csv",
        "results/lolo_ledger.csv",
        "results/lolo_summary.csv",
        "results/method_diagnostics.csv",
        "doc/final_report/tables/protocol_a.tex",
        "doc/final_report/tables/lolo.tex",
        "doc/final_report/figures/cdf_protocol_a_forward_to_backward.pdf",
        "doc/final_report/figures/cdf_protocol_a_backward_to_forward.pdf",
        "doc/final_report/figures/cdf_lolo.pdf",
        "doc/final_report/figures/segment_heatmap.pdf",
    )
)


def _guard_frozen(targets: list[Path]) -> None:
    # Why not 書き込み直前に個別チェック: sweep 完走後に一部だけ書けて落ちると
    # 半端な成果物が残る。全ターゲットを実行前に resolve して一括拒否する。
    hits = sorted(str(p) for p in targets if p.resolve() in FROZEN_OUTPUT_PATHS)
    if hits:
        raise ValueError(f"refusing to overwrite frozen outputs: {hits}")

# --- 定数（手法名）-----------------------------------------------------------

TIER4_METHODS = [
    "fisher_wknn",
    "mahalanobis_wknn",
    "pls_corridor",
    "ordinal_corridor",
    "wcl_residual",
    "joint_fp",
    "gp_augmented_wknn",
]
REFERENCE_METHODS = ["wcl", "gp_corridor"]

LOLO_LEDGER_COLUMNS = ["method", "held_out", "error", "true_x", "true_y"]
DIAG_COLUMNS = ["protocol", "fold", "method", "key", "value"]


# --- スキーマ（reference 名に依存して動的生成）-------------------------------

def protocol_a_columns(references: list[str]) -> list[str]:
    cols = ["method", "fold", "ave", "median", "p90", "within_2m", "max", "std",
            "ci_lo", "ci_hi"]
    for ref in references:
        cols += [f"delta_vs_{ref}", f"delta_vs_{ref}_lo", f"delta_vs_{ref}_hi"]
    cols += ["status"]
    return cols


def lolo_summary_columns(references: list[str]) -> list[str]:
    cols = ["method", "ave", "median", "p90", "within_2m"]
    for ref in references:
        cols += [f"delta_vs_{ref}", f"delta_vs_{ref}_lo", f"delta_vs_{ref}_hi"]
    cols += ["status"]
    return cols


# --- fit/predict（診断回収のため run_method を使わない）----------------------

def _fit_predict(
    name: str,
    train_scans: pd.DataFrame,
    test_scans: pd.DataFrame,
    ap_coords: pd.DataFrame,
    location_coords: pd.DataFrame,
):
    """run_method と同一の leak-safe フィルタで fit/predict し (est, method) を返す。"""
    registry = _get_methods_module().REGISTRY
    if name not in registry:
        raise ValueError(f"unknown method: {name!r}")
    # Why not caller を信じる: test 地点座標を fit に渡さないことを構造で保証する
    # （run_method と同じ契約。座標欠落・重複も契約違反として弾く）。
    train_location_coords = location_coords[
        location_coords["location_p"].isin(train_scans["location_p"].unique())
    ]
    dup_mask = train_location_coords["location_p"].duplicated()
    if dup_mask.any():
        dups = sorted(train_location_coords.loc[dup_mask, "location_p"].unique())
        raise ValueError(f"duplicate location_p in location_coords: {dups}")
    missing = sorted(
        set(train_scans["location_p"].unique())
        - set(train_location_coords["location_p"])
    )
    if missing:
        raise ValueError(f"location_coords missing train locations: {missing}")
    method = registry[name](**{}).fit(train_scans, ap_coords, train_location_coords)
    est = method.predict(test_scans)
    return est, method


def _collect_diagnostics(
    method_obj, *, protocol: str, fold, method_name: str
) -> list[dict]:
    """method.diagnostics_（dict、無ければ空）を long-form 行に展開する。"""
    diag = getattr(method_obj, "diagnostics_", None)
    if not isinstance(diag, dict):
        return []
    return [
        {"protocol": protocol, "fold": fold, "method": method_name,
         "key": str(k), "value": v}
        for k, v in diag.items()
    ]


# --- ペアリング契約 + delta ---------------------------------------------------

def paired_delta_ci(
    method_led: pd.DataFrame,
    ref_led: pd.DataFrame,
    full_locations,
    *,
    seed: int,
    B: int,
) -> dict:
    """mean(method - ref) の paired bootstrap CI を返す。

    完全ペアリング契約: method と ref の双方が full_locations と「完全一致」する
    地点集合を持つときのみ delta を計算する。片方でも欠落・過剰があれば
    intersection で誤魔化さず paired=False（stat/lo/hi は NaN）を返す。

    Why not set 比較のみ: 重複 index は set では full と一致して見えるのに
    .loc 整列で行が fan-out し、誤ったペアを bootstrap に流し込む。一意性と
    行数の一致も契約に含める。
    """
    full = set(full_locations)
    not_paired = {"stat": np.nan, "lo": np.nan, "hi": np.nan, "paired": False}
    if not (method_led.index.is_unique and ref_led.index.is_unique):
        return not_paired
    if len(method_led.index) != len(full) or len(ref_led.index) != len(full):
        return not_paired
    if set(method_led.index) != full or set(ref_led.index) != full:
        return not_paired
    common = sorted(full)
    a = method_led.loc[common, "error"]
    b = ref_led.loc[common, "error"]
    ci = bootstrap_ci_paired(a, b, stat="mean", B=B, seed=seed)
    return {"stat": ci["stat"], "lo": ci["lo"], "hi": ci["hi"], "paired": True}


def _empty_metrics() -> dict:
    return {k: np.nan for k in
            ("ave", "median", "p90", "within_2m", "max", "std", "ci_lo", "ci_hi")}


def _metrics(errors: pd.Series, *, seed: int, B: int) -> dict:
    stats = summary(errors)
    pct = percentiles(errors, (50, 90))
    ci = bootstrap_ci_paired(errors, stat="mean", B=B, seed=seed)
    return {
        "ave": stats["Ave"],
        "median": pct["p50"],
        "p90": pct["p90"],
        "within_2m": within_ratio(errors, 2.0),
        "max": stats["Max"],
        "std": stats["Std"],
        "ci_lo": ci["lo"],
        "ci_hi": ci["hi"],
    }


def _protocol_row(
    method: str,
    fold,
    ledgers: dict[str, pd.DataFrame],
    references: list[str],
    full_locations,
    *,
    seed: int,
    B: int,
) -> dict:
    """1 手法・1 fold の metric 行（delta 2 本 + status）を組む。

    method 台帳が無い（fit 失敗）→ status='failed'、全指標 NaN。空台帳など
    指標計算自体の例外も failed に畳む（fail-soft の境界を行生成まで広げる）。
    delta は各 reference に対し完全ペアリングを検査。破綻したら当該 delta を NaN、
    status='pairing_failed'。self（method==ref）は厳密に 0（構造的特別扱い）。
    """
    led = ledgers.get(method)
    if led is None:
        return _failed_row(method, fold, references)

    row: dict = {"method": method, "fold": fold}
    # Why not 指標計算を裸で呼ぶ: 空 ledger は summary()/bootstrap が例外を投げ、
    # 1 手法の失敗が sweep 全体を殺す。行生成も fail-soft 境界に含める。
    try:
        row.update(_metrics(led["error"], seed=seed, B=B))
    except Exception as exc:  # noqa: BLE001 - sweep を殺さない
        print(
            f"[tier4] metrics fold={fold} method={method} FAILED: {exc}",
            file=sys.stderr,
        )
        return _failed_row(method, fold, references)

    status = "ok"
    for ref in references:
        if method == ref:
            row[f"delta_vs_{ref}"] = 0.0
            row[f"delta_vs_{ref}_lo"] = 0.0
            row[f"delta_vs_{ref}_hi"] = 0.0
            continue
        ref_led = ledgers.get(ref)
        if ref_led is None:
            d = {"stat": np.nan, "lo": np.nan, "hi": np.nan, "paired": False}
        else:
            try:
                d = paired_delta_ci(led, ref_led, full_locations, seed=seed, B=B)
            except Exception as exc:  # noqa: BLE001 - sweep を殺さない
                print(
                    f"[tier4] delta fold={fold} method={method} vs {ref} "
                    f"FAILED: {exc}",
                    file=sys.stderr,
                )
                d = {"stat": np.nan, "lo": np.nan, "hi": np.nan, "paired": False}
        row[f"delta_vs_{ref}"] = d["stat"]
        row[f"delta_vs_{ref}_lo"] = d["lo"]
        row[f"delta_vs_{ref}_hi"] = d["hi"]
        if not d["paired"]:
            status = "pairing_failed"
    row["status"] = status
    return row


def _failed_row(method: str, fold, references: list[str]) -> dict:
    row: dict = {"method": method, "fold": fold}
    row.update(_empty_metrics())
    for ref in references:
        row[f"delta_vs_{ref}"] = np.nan
        row[f"delta_vs_{ref}_lo"] = np.nan
        row[f"delta_vs_{ref}_hi"] = np.nan
    row["status"] = "failed"
    return row


def _union(methods: list[str], references: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for m in list(methods) + list(references):
        seen.setdefault(m, None)
    return list(seen)


# --- Protocol A --------------------------------------------------------------

def run_protocol_a_tier4(
    methods: list[str],
    references: list[str],
    scans_f: pd.DataFrame,
    scans_b: pd.DataFrame,
    ap13: pd.DataFrame,
    truth: pd.DataFrame,
    seed: int = RANDOM_SEED,
    B: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """2 fold × (methods ∪ references) の metric 行・誤差台帳・診断台帳を返す。

    Why not fail-fast: 1 手法が fold で例外を投げても sweep は続行し、その行を
    status='failed'・指標 NaN で埋める（fail-soft）。
    """
    all_methods = _union(methods, references)
    result_rows: list[dict] = []
    ledger_frames: list[pd.DataFrame] = []
    diag_rows: list[dict] = []

    for fold in iter_protocol_a(scans_f, scans_b):
        full_locs = set(fold.test_scans["location_p"].unique())
        ledgers: dict[str, pd.DataFrame] = {}
        for method in all_methods:
            try:
                est, obj = _fit_predict(
                    method, fold.train_scans, fold.test_scans, ap13, truth
                )
                truth_fold = truth[truth["location_p"].isin(est["location_p"])]
                led = errors_ledger(est, truth_fold, method)
            except Exception as exc:  # noqa: BLE001 - sweep を殺さない
                import traceback
                print(
                    f"[tier4] protocol_a fold={fold.name} method={method} FAILED: {exc}",
                    file=sys.stderr,
                )
                traceback.print_exc()
                continue
            ledgers[method] = led
            diag_rows += _collect_diagnostics(
                obj, protocol="protocol_a", fold=fold.name, method_name=method
            )
            enriched = led.reset_index().merge(
                truth_fold.rename(columns={"x": "true_x", "y": "true_y"}),
                on="location_p", how="left",
            )
            enriched["fold"] = fold.name
            ledger_frames.append(
                enriched[["method", "fold", "location_p", "error", "true_x", "true_y"]]
            )

        for method in all_methods:
            result_rows.append(
                _protocol_row(method, fold.name, ledgers, references,
                              full_locs, seed=seed, B=B)
            )

    cols = protocol_a_columns(references)
    results = pd.DataFrame(result_rows, columns=cols)
    ledger_cols = ["method", "fold", "location_p", "error", "true_x", "true_y"]
    ledgers_df = (
        pd.concat(ledger_frames, ignore_index=True)
        if ledger_frames else pd.DataFrame(columns=ledger_cols)
    )
    diag_df = pd.DataFrame(diag_rows, columns=DIAG_COLUMNS)
    return results, ledgers_df, diag_df


# --- LOLO --------------------------------------------------------------------

def run_lolo_tier4(
    methods: list[str],
    references: list[str],
    scans_train_pool: pd.DataFrame,
    scans_test_pool: pd.DataFrame,
    ap13: pd.DataFrame,
    truth: pd.DataFrame,
    seed: int = RANDOM_SEED,
    B: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """LOLO sweep。per-location 台帳・集約 summary（delta 2 本付き）・診断台帳を返す。

    delta は held_out で厳密整列した全 fold 組から paired bootstrap する。ある手法が
    1 fold でも失敗（NaN）すると finite held_out 集合が全体と一致せずペアリング破綻
    → 当該 delta NaN・status='pairing_failed'。
    """
    all_methods = _union(methods, references)
    folds = list(iter_lolo(scans_train_pool, scans_test_pool))
    n = len(folds)
    full_held = set(f.held_out for f in folds)

    ledger_rows: list[dict] = []
    diag_rows: list[dict] = []
    for method in all_methods:
        for i, fold in enumerate(folds, start=1):
            print(
                f"[tier4] lolo {method} fold {i}/{n} (held_out={fold.held_out})",
                file=sys.stderr,
            )
            try:
                est, obj = _fit_predict(
                    method, fold.train_scans, fold.test_scans, ap13, truth
                )
                truth_fold = truth[truth["location_p"].isin(est["location_p"])]
                led = errors_ledger(est, truth_fold, method)
                # Why not 先頭行を無条件採用: 手法が held-out 以外や複数地点を
                # 返しても気づけず、誤った誤差が台帳に載る。index が正確に
                # [held_out] であることを要求し、違反はこの fold の失敗（NaN）扱い。
                if list(led.index) != [fold.held_out]:
                    raise ValueError(
                        f"prediction must cover exactly [{fold.held_out}]; "
                        f"got {list(led.index)}"
                    )
                error = float(led["error"].iloc[0])
                diag_rows += _collect_diagnostics(
                    obj, protocol="lolo", fold=fold.held_out, method_name=method
                )
            except Exception as exc:  # noqa: BLE001 - sweep を殺さない
                import traceback
                print(
                    f"[tier4] lolo {method} fold {i}/{n} FAILED: {exc}",
                    file=sys.stderr,
                )
                traceback.print_exc()
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

    # method -> held_out 指標の Series（finite のみ、index=held_out）
    def _finite_led(method: str) -> pd.DataFrame:
        sub = ledger[ledger["method"] == method][["held_out", "error"]].copy()
        sub = sub[np.isfinite(sub["error"])].set_index("held_out")
        return sub

    summary_rows: list[dict] = []
    for method in all_methods:
        m_led = _finite_led(method)
        errs = m_led["error"].to_numpy(dtype=float)
        row: dict = {"method": method}
        if len(errs) == 0:
            row.update({"ave": np.nan, "median": np.nan, "p90": np.nan,
                        "within_2m": np.nan})
        else:
            row.update({
                "ave": float(errs.mean()),
                "median": float(np.median(errs)),
                "p90": float(np.percentile(errs, 90)),
                "within_2m": float(np.mean(errs <= 2.0)),
            })
        status = "ok"
        for ref in references:
            if method == ref:
                row[f"delta_vs_{ref}"] = 0.0
                row[f"delta_vs_{ref}_lo"] = 0.0
                row[f"delta_vs_{ref}_hi"] = 0.0
                continue
            r_led = _finite_led(ref)
            d = paired_delta_ci(m_led, r_led, full_held, seed=seed, B=B)
            row[f"delta_vs_{ref}"] = d["stat"]
            row[f"delta_vs_{ref}_lo"] = d["lo"]
            row[f"delta_vs_{ref}_hi"] = d["hi"]
            if not d["paired"]:
                status = "pairing_failed"
        row["status"] = status
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows, columns=lolo_summary_columns(references))
    diag_df = pd.DataFrame(diag_rows, columns=DIAG_COLUMNS)
    return ledger, summary_df, diag_df


# --- 図 ----------------------------------------------------------------------

def make_figures_tier4(lolo_ledger: pd.DataFrame, outdir: str | Path) -> list[Path]:
    """9 手法（7 Tier4 + 2 reference）の LOLO CDF を 1 枚出力する。"""
    # Why not 環境変数 setdefault 頼み: 呼び出し前に別コードが GUI backend で
    # pyplot を初期化済みだと env は効かない。use(force=True) で確実に Agg へ。
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "cdf_lolo_tier4.pdf"
    _plot_cdf_tier4(lolo_ledger, "LOLO (Tier 4)", path, plt)
    return [path]


def _plot_cdf_tier4(ledger: pd.DataFrame, title: str, path: Path, plt) -> None:
    # Why not harness._plot_cdf を再利用: 同一描画だが savefig に CreationDate を
    # 埋めるため PDF が実行時刻依存になる。凍結中の harness は編集できないので、
    # metadata で CreationDate を落とす決定的版をここに持つ。
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
    fig.savefig(
        path, format="pdf", bbox_inches="tight", metadata={"CreationDate": None}
    )
    plt.close(fig)


# --- TeX 断片 ----------------------------------------------------------------

# 主要比較基準（gp_corridor）の脚注マーク。既存 harness の powerdomain † と同流儀で
# 特定手法名にハードコードする（reference が代役の場合は付かない）。
_REFERENCE_MARK = r"$^{\ast}$"
_REFERENCE_NOTE = r"\par\footnotesize $^{\ast}$ gp\_corridor: 主要比較基準。"


# Why not str.replace の連鎖: backslash を先に置換するとその置換結果の { } を
# 後段が再置換して壊す。1 パスの文字単位変換なら順序問題が構造的に消える。
_TEX_ESCAPES: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "$": r"\$",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _tex_escape(text: str) -> str:
    return "".join(_TEX_ESCAPES.get(c, c) for c in str(text))


def _fmt(value) -> str:
    return f"{value:.2f}" if value is not None and np.isfinite(value) else "--"


def _tier4_label(method: str) -> str:
    label = _tex_escape(method)
    if method == "gp_corridor":
        label += _REFERENCE_MARK
    return label


def _order_by_lolo(lolo_summary: pd.DataFrame | None, present: list[str]) -> list[str]:
    if lolo_summary is not None and not lolo_summary.empty:
        ordered = lolo_summary.sort_values("ave", na_position="last")["method"].tolist()
    else:
        ordered = []
    return [m for m in ordered if m in present] + [m for m in present if m not in ordered]


def make_tex_tables_tier4(
    results: pd.DataFrame,
    lolo_summary: pd.DataFrame | None,
    references: list[str],
    outdir: str | Path,
) -> list[Path]:
    """booktabs 風 tabular 断片（tier4_protocol_a.tex / tier4_lolo.tex）を書く。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    present = list(dict.fromkeys(results["method"].tolist()))
    order = _order_by_lolo(lolo_summary, present)

    proto_path = outdir / "tier4_protocol_a.tex"
    proto_path.write_text(_protocol_tex(results, references, order), encoding="utf-8")

    lolo_path = outdir / "tier4_lolo.tex"
    lolo_path.write_text(_lolo_tex(lolo_summary, references, order), encoding="utf-8")
    return [proto_path, lolo_path]


def _protocol_tex(results: pd.DataFrame, references: list[str], order: list[str]) -> str:
    metric_cols = [("Ave", "ave"), ("Median", "median"), ("P90", "p90"),
                   ("Within 2m", "within_2m"), ("Max", "max"), ("Std", "std")]
    delta_cols = [(rf"$\Delta${_tex_escape(ref)}", f"delta_vs_{ref}")
                  for ref in references]
    cols = metric_cols + delta_cols
    ncol = len(cols)

    lines = [
        r"\begin{tabular}{ll" + "r" * ncol + "}",
        r"\toprule",
        "Method & Fold & " + " & ".join(h for h, _ in cols) + r" \\",
        r"\midrule",
    ]
    for method in order:
        rows = results[results["method"] == method].sort_values("fold")
        for _, r in rows.iterrows():
            cells = [_tier4_label(method), _tex_escape(str(r["fold"]))]
            cells += [_fmt(r[key]) for _, key in cols]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if "gp_corridor" in set(results["method"]):
        lines.append(_REFERENCE_NOTE)
    return "\n".join(lines) + "\n"


def _lolo_tex(
    lolo_summary: pd.DataFrame | None, references: list[str], order: list[str]
) -> str:
    if lolo_summary is None or lolo_summary.empty:
        return "% LOLO summary unavailable\n"
    metric_cols = [("Ave", "ave"), ("Median", "median"), ("P90", "p90"),
                   ("Within 2m", "within_2m")]
    delta_cols = [(rf"$\Delta${_tex_escape(ref)}", f"delta_vs_{ref}")
                  for ref in references]
    cols = metric_cols + delta_cols
    indexed = lolo_summary.set_index("method")
    present = [m for m in order if m in indexed.index]

    lines = [
        r"\begin{tabular}{l" + "r" * len(cols) + "}",
        r"\toprule",
        "Method & " + " & ".join(h for h, _ in cols) + r" \\",
        r"\midrule",
    ]
    for method in present:
        r = indexed.loc[method]
        cells = [_tier4_label(method)] + [_fmt(r[key]) for _, key in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if "gp_corridor" in set(lolo_summary["method"]):
        lines.append(_REFERENCE_NOTE)
    return "\n".join(lines) + "\n"


# --- サブサンプル（--smoke 用）-----------------------------------------------

def subsample_scans(
    scans_f: pd.DataFrame,
    scans_b: pd.DataFrame,
    truth: pd.DataFrame,
    n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """segment 層化（round-robin）で n 地点に絞る（smoke 高速化）。

    Why not 先頭 n 連続地点: 連続地点は単一 segment に固まり、gp_corridor の
    segment 分類器（2 クラス以上必須）が fit できない。segment を跨いで抜けば
    smoke でも正式 reference（gp_corridor）を含む本番同一スキーマで回せる。
    """
    common = sorted(set(scans_f["location_p"]) & set(scans_b["location_p"]))
    truth_idx = truth.set_index("location_p")
    by_segment: dict[str, list] = {}
    for loc in common:
        if loc not in truth_idx.index:
            continue
        seg = segment_of(float(truth_idx.at[loc, "x"]), float(truth_idx.at[loc, "y"]))
        by_segment.setdefault(seg, []).append(loc)

    queues = [by_segment[seg] for seg in sorted(by_segment)]
    keep: set = set()
    i = 0
    while len(keep) < n and any(queues):
        q = queues[i % len(queues)]
        if q:
            keep.add(q.pop(0))
        i += 1

    sf = scans_f[scans_f["location_p"].isin(keep)].reset_index(drop=True)
    sb = scans_b[scans_b["location_p"].isin(keep)].reset_index(drop=True)
    tr = truth[truth["location_p"].isin(keep)].reset_index(drop=True)
    return sf, sb, tr


# --- オーケストレーション（出力先は全て引数で明示）--------------------------

def run_tier4(
    *,
    scans_f: pd.DataFrame,
    scans_b: pd.DataFrame,
    ap13: pd.DataFrame,
    truth: pd.DataFrame,
    methods: list[str],
    references: list[str],
    output_dir: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
    seed: int = RANDOM_SEED,
    B: int = 1000,
    skip_lolo: bool = False,
) -> dict[str, Path]:
    """全評価を回し、指定 3 ディレクトリにのみ書き出す。書いたパス dict を返す。

    凍結成果物（FROZEN_OUTPUT_PATHS）と衝突するターゲットが 1 つでもあれば、
    sweep 実行・ディレクトリ作成より前に ValueError で拒否する。
    """
    output_dir = Path(output_dir)
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    _guard_frozen(
        [output_dir / name for name in (
            "protocol_a.csv", "lolo_ledger.csv", "lolo_summary.csv", "diagnostics.csv",
        )]
        + [tables_dir / "tier4_protocol_a.tex", tables_dir / "tier4_lolo.tex"]
        + [figures_dir / "cdf_lolo_tier4.pdf"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    results, _pa_ledger, pa_diag = run_protocol_a_tier4(
        methods, references, scans_f, scans_b, ap13, truth, seed=seed, B=B
    )
    p = output_dir / "protocol_a.csv"
    results.to_csv(p, index=False)
    written["protocol_a"] = p

    lolo_summary = None
    diag = pa_diag
    if not skip_lolo:
        lolo_ledger, lolo_summary, lolo_diag = run_lolo_tier4(
            methods, references, scans_f, scans_b, ap13, truth, seed=seed, B=B
        )
        p = output_dir / "lolo_ledger.csv"
        lolo_ledger.to_csv(p, index=False)
        written["lolo_ledger"] = p
        p = output_dir / "lolo_summary.csv"
        lolo_summary.to_csv(p, index=False)
        written["lolo_summary"] = p
        diag = pd.concat([pa_diag, lolo_diag], ignore_index=True)

        fig_paths = make_figures_tier4(lolo_ledger, figures_dir)
        written["figure"] = fig_paths[0]

    p = output_dir / "diagnostics.csv"
    diag.to_csv(p, index=False)
    written["diagnostics"] = p

    tex_paths = make_tex_tables_tier4(results, lolo_summary, references, tables_dir)
    written["tex_protocol_a"] = tex_paths[0]
    written["tex_lolo"] = tex_paths[1]

    return written
