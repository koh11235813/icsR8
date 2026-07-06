"""Generate notebooks/baseline_reproduction.ipynb from source.

Run:
    uv run --group notebook python scripts/generate_baseline_notebook.py
    uv run --group notebook jupyter nbconvert --to notebook --execute --inplace notebooks/baseline_reproduction.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
OUT_PATH = REPO / "notebooks" / "baseline_reproduction.ipynb"

CELLS = [
    ("markdown", """\
# icsR8 baseline reproduction (PBL / CLA / WCL)

Reproduces the published Table 1 values (`doc/icsR8_text.txt` §3.2) for both
scan directions using the `icsr8` library, and plots per-position error curves
and an estimate-vs-truth map."""),
    ("code", """\
from pathlib import Path

import pandas as pd

from icsr8.estimators import estimate_cla, estimate_pbl, estimate_wcl
from icsr8.evaluate import l2_errors, summary
from icsr8.fingerprint import candidate_medians, reproduction_fingerprint
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.plotting import plot_error_by_position, plot_estimate_map

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATASET_DIR = REPO / "data" / "dataset"
RAWDATA_ROOT = REPO / "data" / "rawdata"

METHODS = {"pbl": estimate_pbl, "cla": estimate_cla, "wcl": estimate_wcl}

# Published Table 1 (doc/icsR8_text.txt §3.2) - Std uses ddof=0
DOC_TABLE_1 = {
    "forward": {
        "pbl": {"Ave": 4.38, "Max": 13.6, "Std": 2.82},
        "cla": {"Ave": 8.07, "Max": 24.2, "Std": 5.33},
        "wcl": {"Ave": 3.57, "Max": 11.9, "Std": 2.42},
    },
    "backward": {
        "pbl": {"Ave": 4.52, "Max": 15.6, "Std": 3.14},
        "cla": {"Ave": 7.02, "Max": 18.0, "Std": 4.22},
        "wcl": {"Ave": 3.51, "Max": 12.2, "Std": 2.54},
    },
}"""),
    ("markdown", "## Load data and build per-direction fingerprints"),
    ("code", """\
ap_coords = load_ap_coords(DATASET_DIR / "AP_coordinate_C3F.csv")
truth = load_location_coords(DATASET_DIR / "location_coordinate_C.csv")[["location_p", "x", "y"]]

fingerprints = {}
for direction in ("forward", "backward"):
    scans = load_raw_scans(direction, RAWDATA_ROOT)
    fingerprints[direction] = reproduction_fingerprint(candidate_medians(scans, ap_coords))

fingerprints["forward"].head()"""),
    ("markdown", "## Estimate PBL / CLA / WCL for both directions and compare to the published baseline"),
    ("code", """\
results = {}
rows = []
for direction, fp in fingerprints.items():
    for method_name, estimator in METHODS.items():
        est = estimator(fp)
        err = l2_errors(est, truth)
        stats = summary(err["error"])
        results[(direction, method_name)] = {"estimates": est, "errors": err, "stats": stats}

        expected = DOC_TABLE_1[direction][method_name]
        rows.append({
            "direction": direction,
            "method": method_name,
            "Ave": stats["Ave"], "Ave_doc": expected["Ave"],
            "Max": stats["Max"], "Max_doc": expected["Max"],
            "Std": stats["Std"], "Std_doc": expected["Std"],
        })

comparison = pd.DataFrame(rows)
comparison"""),
    ("markdown", "## Error-by-position curves (per method, forward vs. backward overlaid)"),
    ("code", """\
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, method_name in zip(axes, METHODS):
    plot_error_by_position(results[("forward", method_name)]["errors"], ax=ax, label="forward")
    plot_error_by_position(results[("backward", method_name)]["errors"], ax=ax, label="backward")
    ax.set_title(method_name.upper())
fig.tight_layout()"""),
    ("markdown", "## Estimate-vs-truth map (WCL, forward)"),
    ("code", """\
plot_estimate_map(
    results[("forward", "wcl")]["estimates"],
    truth,
    ap_coords=ap_coords,
)"""),
]


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    for cell_type, source in CELLS:
        if cell_type == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(source))
        else:
            nb.cells.append(nbf.v4.new_code_cell(source))
    return nb


def main() -> None:
    nb = build_notebook()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT_PATH)
    print(f"Wrote {OUT_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
