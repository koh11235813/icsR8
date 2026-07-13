import numpy as np
import pandas as pd
import pytest

from icsr8.constants import RANDOM_SEED
from icsr8.evaluate import bootstrap_ci_paired, errors_ledger, percentiles, within_ratio


def test_percentiles_linear_interpolation():
    arr = np.arange(1, 101)  # 1..100
    out = percentiles(arr, qs=(50, 75, 90))

    assert out["p50"] == pytest.approx(np.percentile(arr, 50))
    assert out["p75"] == pytest.approx(np.percentile(arr, 75))
    assert out["p90"] == pytest.approx(np.percentile(arr, 90))
    assert out["p50"] == pytest.approx(50.5)
    assert out["p90"] == pytest.approx(90.1)


def test_within_ratio_fraction_within_threshold():
    assert within_ratio([1, 2, 3, 4], 2.0) == pytest.approx(0.5)


def test_errors_ledger_columns_and_row_count():
    estimates = pd.DataFrame([
        {"location_p": 1, "x": 0.0, "y": 0.0},
        {"location_p": 2, "x": 1.0, "y": 1.0},
        {"location_p": 3, "x": 2.0, "y": 2.0},
    ])
    truth = pd.DataFrame([
        {"location_p": 1, "x": 0.0, "y": 0.0},
        {"location_p": 2, "x": 0.0, "y": 0.0},
        {"location_p": 3, "x": 0.0, "y": 0.0},
    ])

    ledger = errors_ledger(estimates, truth, "pbl")

    assert list(ledger.columns) == ["method", "error"]
    assert ledger.index.name == "location_p"
    assert len(ledger) == 3
    assert (ledger["method"] == "pbl").all()
    assert list(ledger.index) == [1, 2, 3]
    assert ledger["error"].tolist() == pytest.approx([0.0, np.sqrt(2), np.sqrt(8)])


def test_errors_ledger_59_rows_on_real_data(dataset_dir, rawdata_root):
    from icsr8.estimators import estimate_pbl
    from icsr8.fingerprint import candidate_medians, reproduction_fingerprint
    from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans

    ap_coords = load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")
    loc_coords = load_location_coords(dataset_dir / "location_coordinate_C.csv")
    scans = load_raw_scans("forward", rawdata_root)
    candidates = candidate_medians(scans, ap_coords)
    fp = reproduction_fingerprint(candidates)
    estimates = estimate_pbl(fp)

    ledger = errors_ledger(estimates, loc_coords, "pbl")

    assert len(ledger) == 59
    assert list(ledger.columns) == ["method", "error"]
    assert ledger.index.name == "location_p"


def test_errors_ledger_sorted_index_makes_pairing_safe():
    # 同一 estimates を別々にシャッフルして 2 台帳を作る。errors_ledger が
    # location_p 昇順に正規化するので、bootstrap_ci_paired の index 一致チェックが
    # 通り、同じ誤差同士（差 0）がペアリングされる。
    estimates = pd.DataFrame([
        {"location_p": 3, "x": 2.0, "y": 2.0},
        {"location_p": 1, "x": 0.0, "y": 0.0},
        {"location_p": 2, "x": 1.0, "y": 1.0},
    ])
    truth = pd.DataFrame([
        {"location_p": 1, "x": 0.0, "y": 0.0},
        {"location_p": 2, "x": 0.0, "y": 0.0},
        {"location_p": 3, "x": 0.0, "y": 0.0},
    ])

    ledger_a = errors_ledger(estimates.iloc[[0, 1, 2]], truth, "a")
    ledger_b = errors_ledger(estimates.iloc[[2, 0, 1]], truth, "b")

    assert list(ledger_a.index) == list(ledger_b.index) == [1, 2, 3]
    # 台帳は同じ誤差集合なので、正しくペアリングされれば差の統計は厳密に 0。
    out = bootstrap_ci_paired(ledger_a["error"], ledger_b["error"], seed=0)
    assert out["stat"] == pytest.approx(0.0)


def test_bootstrap_ci_paired_single_series_constant_vector_lo_eq_hi_eq_stat():
    errors_a = pd.Series([5.0, 5.0, 5.0, 5.0], index=[1, 2, 3, 4])

    out = bootstrap_ci_paired(errors_a, seed=0)

    assert out["stat"] == pytest.approx(5.0)
    assert out["lo"] == pytest.approx(5.0)
    assert out["hi"] == pytest.approx(5.0)


def test_bootstrap_ci_paired_delta_is_exactly_minus_one():
    errors_a = pd.Series([1.0, 3.0, 7.0, 2.5], index=[10, 20, 30, 40])
    errors_b = errors_a + 1.0

    out = bootstrap_ci_paired(errors_a, errors_b, seed=0)

    assert out["stat"] == pytest.approx(-1.0)
    assert out["lo"] == pytest.approx(-1.0)
    assert out["hi"] == pytest.approx(-1.0)


def test_bootstrap_ci_paired_same_seed_reproducible():
    errors_a = pd.Series([1.0, 5.0, 2.0, 9.0, 3.5, 6.2, 0.5], index=range(7))

    out1 = bootstrap_ci_paired(errors_a, seed=42)
    out2 = bootstrap_ci_paired(errors_a, seed=42)

    assert out1 == out2


def test_bootstrap_ci_paired_different_seed_generally_differs():
    errors_a = pd.Series([1.0, 5.0, 2.0, 9.0, 3.5, 6.2, 0.5], index=range(7))

    out1 = bootstrap_ci_paired(errors_a, seed=1)
    out2 = bootstrap_ci_paired(errors_a, seed=2)

    assert (out1["lo"], out1["hi"]) != (out2["lo"], out2["hi"])


def test_bootstrap_ci_paired_default_seed_is_random_seed_constant():
    errors_a = pd.Series([1.0, 5.0, 2.0, 9.0], index=range(4))

    out_default = bootstrap_ci_paired(errors_a)
    out_explicit = bootstrap_ci_paired(errors_a, seed=RANDOM_SEED)

    assert out_default == out_explicit


def test_bootstrap_ci_paired_raises_on_different_content():
    errors_a = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3])
    errors_b = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 4])

    with pytest.raises(ValueError):
        bootstrap_ci_paired(errors_a, errors_b)


def test_bootstrap_ci_paired_raises_on_different_order():
    errors_a = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3])
    errors_b = pd.Series([1.0, 2.0, 3.0], index=[3, 2, 1])

    with pytest.raises(ValueError):
        bootstrap_ci_paired(errors_a, errors_b)


def test_bootstrap_ci_paired_duplicate_index_subtracts_positionally():
    # Guards against pandas' index-aligned subtraction, which would fan out
    # duplicate labels into a Cartesian product even though the index equality
    # check (content + order) passes for [1, 1, 2] vs [1, 1, 2].
    errors_a = pd.Series([5.0, 8.0, 3.0], index=[1, 1, 2])
    errors_b = pd.Series([2.0, 2.0, 2.0], index=[1, 1, 2])

    out = bootstrap_ci_paired(errors_a, errors_b, seed=0)

    assert out["stat"] == pytest.approx(np.mean([3.0, 6.0, 1.0]))


def test_bootstrap_ci_paired_callable_stat():
    errors_a = pd.Series(np.arange(1, 21, dtype=float), index=range(20))

    out = bootstrap_ci_paired(errors_a, stat=lambda a: np.percentile(a, 90), seed=0)

    assert out["stat"] == pytest.approx(np.percentile(errors_a.to_numpy(), 90))
    assert out["lo"] <= out["stat"] <= out["hi"]


def test_bootstrap_ci_paired_return_shape():
    errors_a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=range(5))

    out = bootstrap_ci_paired(errors_a, B=200, level=0.9, seed=0)

    assert set(out.keys()) == {"stat", "lo", "hi", "B", "level"}
    assert out["B"] == 200
    assert out["level"] == 0.9
    assert out["lo"] <= out["stat"] <= out["hi"]


def test_bootstrap_ci_paired_raises_on_empty_errors():
    with pytest.raises(ValueError):
        bootstrap_ci_paired(pd.Series([], dtype=float))


def test_bootstrap_ci_paired_raises_on_non_positive_B():
    with pytest.raises(ValueError):
        bootstrap_ci_paired(pd.Series([1.0, 2.0, 3.0]), B=0)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
def test_bootstrap_ci_paired_raises_on_level_out_of_range(level):
    with pytest.raises(ValueError):
        bootstrap_ci_paired(pd.Series([1.0, 2.0, 3.0]), level=level)
