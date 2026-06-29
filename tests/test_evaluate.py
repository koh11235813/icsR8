from pathlib import Path

import pandas as pd
import pytest

from icsr8.evaluate import l2_errors, summary


def test_l2_errors_against_known_distances():
    est = pd.DataFrame([{"location_p": 1, "x": 3.0, "y": 0.0},
                        {"location_p": 2, "x": 0.0, "y": 4.0}])
    truth = pd.DataFrame([{"location_p": 1, "x": 0.0, "y": 0.0},
                          {"location_p": 2, "x": 3.0, "y": 0.0}])

    out = l2_errors(est, truth)

    out = out.sort_values("location_p").reset_index(drop=True)
    assert out["error"].tolist() == pytest.approx([3.0, 5.0])
    assert set(out.columns) == {"location_p", "est_x", "est_y", "true_x", "true_y", "error"}


def test_summary_ddof0_matches_published_backward_pbl(fixtures_dir: Path):
    oracle = pd.read_csv(fixtures_dir / "baseline_backward.csv")
    s = summary(oracle["pbl_error"])

    # published doc Table 1 逆方向 PBL: Ave 4.52, Max 15.6, Std 3.14
    assert s["Ave"] == pytest.approx(4.52, abs=0.01)
    assert s["Max"] == pytest.approx(15.6, abs=0.05)
    assert s["Std"] == pytest.approx(3.14, abs=0.01)
    assert set(s.keys()) == {"Ave", "Max", "Min", "Std", "Var"}


def test_summary_ddof1_diverges_from_published_std(fixtures_dir: Path):
    """Documents the ddof choice: ddof=1 does NOT reproduce the published Std."""
    oracle = pd.read_csv(fixtures_dir / "baseline_backward.csv")
    s1 = summary(oracle["pbl_error"], ddof=1)
    # Sample std differs noticeably from population std 3.14
    assert abs(s1["Std"] - 3.14) > 0.01
