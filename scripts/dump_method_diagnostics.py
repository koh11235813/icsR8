"""最終報告 §3 が引用する手法別ハイパーパラメータ・診断値のダンプ。

wknn / gp_corridor / studentt_fp / centered_fp / rank_fp を FULL forward プール
（train_scans=forward の 59 地点全て、location_coords=59 地点の真値）で fit し、
report が本文中で言及する診断値を results/method_diagnostics.csv へ
[method, key, value] で書き出す。gp_corridor のみ fallback_count を得るため
fit 後に同じ forward プールへ self-predict する（held-out test pool が無いため。
train 集合そのものへの自己予測であり評価指標ではなく診断専用）。

使用例:
    uv run python scripts/dump_method_diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

try:
    from icsr8.fingerprint import ap_band_fingerprint
    from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
    from icsr8.methods import REGISTRY
except ImportError:  # editable install 未実施でも動くよう src を通す
    sys.path.insert(0, str(ROOT / "src"))
    from icsr8.fingerprint import ap_band_fingerprint
    from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
    from icsr8.methods import REGISTRY

OUTPUT = ROOT / "results" / "method_diagnostics.csv"


def _load_full_forward_pool() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = ROOT / "data"
    ap13 = load_ap_coords(root / "dataset" / "AP_coordinate_C3F.csv")
    truth = load_location_coords(root / "dataset" / "location_coordinate_C.csv")[
        ["location_p", "x", "y"]
    ]
    scans_f = load_raw_scans("forward", root / "rawdata")
    return scans_f, ap13, truth


def _rows(method: str, values: dict[str, object]) -> list[dict]:
    return [{"method": method, "key": k, "value": v} for k, v in values.items()]


def main() -> int:
    scans_f, ap13, truth = _load_full_forward_pool()
    rows: list[dict] = []

    wknn = REGISTRY["wknn"]().fit(scans_f, ap13, truth)
    rows += _rows(
        "wknn",
        {"selected_k": wknn.selected_k, "selected_weighting": wknn.selected_weighting},
    )

    gp = REGISTRY["gp_corridor"]().fit(scans_f, ap13, truth)
    gp.predict(scans_f)  # self-predict only to populate fallback_count (診断専用)
    n_total_keys = ap_band_fingerprint(scans_f).groupby(["ap_name", "band"]).ngroups
    rows += _rows(
        "gp_corridor",
        {
            "segment_train_accuracy": gp.segment_train_accuracy,
            "n_gp_keys": len(gp.gp_params),
            "n_total_keys": n_total_keys,
            "fallback_count": gp.fallback_count,
        },
    )

    st = REGISTRY["studentt_fp"]().fit(scans_f, ap13, truth)
    rows += _rows("studentt_fp", {"selected_nu": st.selected_nu})

    cfp = REGISTRY["centered_fp"]().fit(scans_f, ap13, truth)
    rows += _rows("centered_fp", {"selected_lambda": cfp.selected_lambda})

    # rank_fp only exposes the selected mixing weight as a private attribute
    # (RankFp._lambda); there is no public alias, so we read it directly here.
    rfp = REGISTRY["rank_fp"]().fit(scans_f, ap13, truth)
    rows += _rows("rank_fp", {"selected_lambda": rfp._lambda})

    df = pd.DataFrame(rows, columns=["method", "key", "value"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(df.to_string(index=False))
    print(f"\n[dump_method_diagnostics] wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
