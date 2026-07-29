"""本文 Tier 1–3 と付録 A の再生成 API（sanctioned writer）。

凍結対象ファイル（`doc/final_report/main.tex` から `\\input` / `\\includegraphics`
で参照される表 TeX + 図 PDF、`harness_tier4.FROZEN_OUTPUT_PATHS` に列挙）は、
このモジュールの `regenerate_main_body()` / `regenerate_appendix_a()` **のみ**が
書ける。他の writer（`scripts/run_experimental_tier4.py`・hand-edit・ad-hoc
スクリプト）は `harness_tier4._guard_frozen()` が `ValueError` で reject する
（旧 blocklist 契約から allowlist 契約への置き換え。背景は
`docs/adr/0004-deep-module-freeze-invariant.md`）。

2026-07-14 の contamination incident（`scripts/run_all_methods.py --methods`
省略時にレジストリの自動探索で Tier 4 手法まで巻き込んだ事故）は、
「正しい再生成コマンドを覚えておく」という運用ルールで塞いでいた。
この 2 関数は完全引数ゼロ設計にすることで、そもそも「手法リストを打ち間違える」
「出力先フラグを省略する」余地自体を無くす（deep module: 狭いインターフェースの
裏に広い実装を隠し、呼び出し側の選択肢を意図的に削る）。

Repo root は本ファイルの位置（`src/icsr8/report.py`）から
`Path(__file__).resolve().parents[2]` で解決する。CI/test で出力先を差し替えたい
場合は、`run_protocol_a` 等の重い計算本体を monkeypatch する（`tests/test_report.py`
参照。repo root 自体を差し替える必要はない — 本モジュールは常に実リポジトリの
`results/`・`doc/final_report/` に対して動作する契約）。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from icsr8.constants import RANDOM_SEED
from icsr8.fingerprint import ap_band_fingerprint
from icsr8.harness import make_figures, make_tex_tables, run_lolo, run_protocol_a
from icsr8.harness_tier4 import REFERENCE_METHODS, TIER4_METHODS, run_tier4
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import REGISTRY

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: sanctioned writer 識別子（harness_tier4._SANCTIONED_WRITERS と対で管理する）。
_WRITER_MAIN_BODY = "icsr8.report.regenerate_main_body"
_WRITER_APPENDIX_A = "icsr8.report.regenerate_appendix_a"

# 本文 Tier 1–3（README「Tier ごとの評価手順」の --methods 列挙・順序と一致すること
# — 2026-07-14 contamination incident の再発防止として、本文手法は常にこの
# literal・この順序でのみ評価する契約。test_report.py がこの契約を固定する）。
MAIN_BODY_METHODS: tuple[str, ...] = (
    "centered_fp",
    "cla",
    "gp_corridor",
    "multiband_wcl",
    "pbl",
    "rank_fp",
    "studentt_fp",
    "wcl",
    "wcl_blacklist",
    "wcl_corridor",
    "wcl_linpower",
    "wcl_powerdomain",
    "wcl_topl",
    "wcl_varweight",
    "wknn",
)

# 付録 A（Tier 4）の 7 手法。harness_tier4.TIER4_METHODS をそのまま再エクスポート
# する（値を literal で複製すると 2 箇所がドリフトしうるため、単一の定義元を
# harness_tier4 に置き、ここは参照するだけに留める）。
APPENDIX_A_METHODS: tuple[str, ...] = tuple(TIER4_METHODS)


def _load_data(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """`data_root`（通常 `<repo>/data`）から評価に必要な 4 テーブルを読む。"""
    ap13 = load_ap_coords(data_root / "dataset" / "AP_coordinate_C3F.csv")
    truth = load_location_coords(data_root / "dataset" / "location_coordinate_C.csv")[
        ["location_p", "x", "y"]
    ]
    scans_f = load_raw_scans("forward", data_root / "rawdata")
    scans_b = load_raw_scans("backward", data_root / "rawdata")
    return scans_f, scans_b, ap13, truth


def _diag_rows(method: str, values: dict[str, object]) -> list[dict]:
    """1 手法分の {key: value} を `results/method_diagnostics.csv` の
    long-form 行（method/key/value）へ展開する小さな整形ヘルパー。

    診断値は手法ごとにキー集合が異なる（wknn は selected_k、gp_corridor は
    fallback_count 等）ため、wide 形式だと大半のセルが空になる。long-form に
    統一しておけば `_write_method_diagnostics` が手法を跨いで単純に concat
    できる。
    """
    return [{"method": method, "key": k, "value": v} for k, v in values.items()]


def _write_method_diagnostics(
    scans_f: pd.DataFrame, ap13: pd.DataFrame, truth: pd.DataFrame
) -> None:
    """最終報告 §3 が引用する手法別ハイパーパラメータ・診断値を
    `results/method_diagnostics.csv` へ書く（旧 `scripts/dump_method_diagnostics.py`
    を吸収）。

    wknn / gp_corridor / studentt_fp / centered_fp / rank_fp を FULL forward
    プール（train_scans=forward の 59 地点全て、location_coords=59 地点の真値）
    で fit する。gp_corridor のみ fallback_count を得るため fit 後に同じ
    forward プールへ self-predict する（held-out test pool が無いため。
    train 集合そのものへの自己予測であり評価指標ではなく診断専用）。
    REGISTRY 経由で手法クラスへアクセスする（run_method は診断値
    diagnostics_ を破棄してしまうため使わない）。
    """
    rows: list[dict] = []

    wknn = REGISTRY["wknn"]().fit(scans_f, ap13, truth)
    rows += _diag_rows(
        "wknn",
        {"selected_k": wknn.selected_k, "selected_weighting": wknn.selected_weighting},
    )

    gp = REGISTRY["gp_corridor"]().fit(scans_f, ap13, truth)
    gp.predict(scans_f)  # self-predict only to populate fallback_count（診断専用）
    n_total_keys = ap_band_fingerprint(scans_f).groupby(["ap_name", "band"]).ngroups
    rows += _diag_rows(
        "gp_corridor",
        {
            "segment_train_accuracy": gp.segment_train_accuracy,
            "n_gp_keys": len(gp.gp_params),
            "n_total_keys": n_total_keys,
            "fallback_count": gp.fallback_count,
        },
    )

    st = REGISTRY["studentt_fp"]().fit(scans_f, ap13, truth)
    rows += _diag_rows("studentt_fp", {"selected_nu": st.selected_nu})

    cfp = REGISTRY["centered_fp"]().fit(scans_f, ap13, truth)
    rows += _diag_rows("centered_fp", {"selected_lambda": cfp.selected_lambda})

    # rank_fp only exposes the selected mixing weight as a private attribute
    # (RankFp._lambda); there is no public alias, so we read it directly here.
    rfp = REGISTRY["rank_fp"]().fit(scans_f, ap13, truth)
    rows += _diag_rows("rank_fp", {"selected_lambda": rfp._lambda})

    df = pd.DataFrame(rows, columns=["method", "key", "value"])
    out = _REPO_ROOT / "results" / "method_diagnostics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"[report] wrote {out}")


def regenerate_main_body() -> None:
    """本文 15 手法 + method_diagnostics.csv を再生成する（引数ゼロ）。

    出力:
      `results/{protocol_a,protocol_a_ledger,lolo_ledger,lolo_summary,
      method_diagnostics}.csv`、
      `doc/final_report/tables/{protocol_a,lolo}.tex`、
      `doc/final_report/figures/{cdf_lolo,cdf_protocol_a_forward_to_backward,
      cdf_protocol_a_backward_to_forward,segment_heatmap}.pdf`。

    このモジュールが `icsr8.report.regenerate_main_body` として
    `harness_tier4._SANCTIONED_WRITERS` に登録された sanctioned writer。
    `_guard_frozen` は 2026-07-29 以降 `harness.make_figures` /
    `harness.make_tex_tables` 自身の冒頭に押し下げられている（Codex review
    finding 1）ため、この関数は `writer_id=_WRITER_MAIN_BODY` をそれらへ渡す
    だけでよい。
    """
    scans_f, scans_b, ap13, truth = _load_data(_REPO_ROOT / "data")

    output_dir = _REPO_ROOT / "results"
    tables_dir = _REPO_ROOT / "doc" / "final_report" / "tables"
    figures_dir = _REPO_ROOT / "doc" / "final_report" / "figures"

    methods = list(MAIN_BODY_METHODS)
    output_dir.mkdir(parents=True, exist_ok=True)

    results, pa_ledgers = run_protocol_a(
        methods, scans_f, scans_b, ap13, truth, seed=RANDOM_SEED, B=1000
    )
    results.to_csv(output_dir / "protocol_a.csv", index=False)
    pa_ledgers.to_csv(output_dir / "protocol_a_ledger.csv", index=False)

    lolo_ledger, lolo_summary = run_lolo(
        methods, scans_f, scans_b, ap13, truth, seed=RANDOM_SEED
    )
    lolo_ledger.to_csv(output_dir / "lolo_ledger.csv", index=False)
    lolo_summary.to_csv(output_dir / "lolo_summary.csv", index=False)

    make_figures(
        {"protocol_a": pa_ledgers, "lolo": lolo_ledger}, figures_dir, writer_id=_WRITER_MAIN_BODY
    )
    make_tex_tables(results, lolo_summary, tables_dir, writer_id=_WRITER_MAIN_BODY)

    _write_method_diagnostics(scans_f, ap13, truth)


def regenerate_appendix_a() -> None:
    """付録 A の 7 手法を再生成する（引数ゼロ）。

    出力:
      `results/tier4/{protocol_a,lolo_ledger,lolo_summary,diagnostics}.csv`、
      `doc/final_report/tables/tier4_{protocol_a,lolo}.tex`、
      `doc/final_report/figures/cdf_lolo_tier4.pdf`。

    `icsr8.report.regenerate_appendix_a` として sanctioned writer 登録済み。
    実際の凍結ガードは `harness_tier4.run_tier4` 内部で行う（writer_id を
    ここから渡すことで通す）。
    """
    scans_f, scans_b, ap13, truth = _load_data(_REPO_ROOT / "data")

    output_dir = _REPO_ROOT / "results" / "tier4"
    tables_dir = _REPO_ROOT / "doc" / "final_report" / "tables"
    figures_dir = _REPO_ROOT / "doc" / "final_report" / "figures"

    run_tier4(
        scans_f=scans_f,
        scans_b=scans_b,
        ap13=ap13,
        truth=truth,
        methods=list(APPENDIX_A_METHODS),
        references=REFERENCE_METHODS,
        output_dir=output_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        seed=RANDOM_SEED,
        B=1000,
        writer_id=_WRITER_APPENDIX_A,
    )
