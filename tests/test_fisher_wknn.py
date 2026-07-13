"""fisher_wknn（Tier4 手法#13: Fisher スコア AP 選択 → WKNN）のテスト。

Tests:
  1. Fisher スコア: 識別的で安定な key (location 間で mu 分散大 / 検出内 sigma 小)
     が、平坦でノイジーな key (mu 分散小 / sigma 大) より高スコアを得る。
  2. 検出地点数 < 3 の key は _fisher_scores から除外される。
  3. 全 key 同分散でも NaN が出ない（有限スコアのみ返る）。
  4. inner CV: fit() が選択した (M, k, weighting) をグリッド内から選び、
     diagnostics_ に記録する。
  5. リーク spy: inner CV 中の Fisher key 選択が inner_train 地点のみを参照し
     inner_val 地点と disjoint（module 内 _select_keys を monkeypatch で計装）。
  6. リーク spy: run_method 経由で outer test 地点が fit に届かない
     （db 側の学習地点が train 地点のみに限定される）。
  7. Protocol A 1 fold smoke（実データ）。
  8. LOLO smoke（3 地点、iter_lolo を islice）。
  9. 契約必須の全体 smoke test（59 地点、NaN/shape 崩れなし）。
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icsr8.constants import RANDOM_SEED
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import run_method
from icsr8.methods import fisher_wknn as fw
from icsr8.methods.fisher_wknn import FisherWknn, M_GRID, K_GRID, WEIGHTING_GRID
from icsr8.protocols import iter_inner_cv, iter_lolo, iter_protocol_a


# --- fixtures ------------------------------------------------------------

@pytest.fixture(scope="session")
def ap_coords(dataset_dir: Path) -> pd.DataFrame:
    return load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")


@pytest.fixture(scope="session")
def location_coords(dataset_dir: Path) -> pd.DataFrame:
    return load_location_coords(dataset_dir / "location_coordinate_C.csv")[
        ["location_p", "x", "y"]
    ]


@pytest.fixture(scope="session")
def scans_f(rawdata_root: Path) -> pd.DataFrame:
    return load_raw_scans("forward", rawdata_root)


@pytest.fixture(scope="session")
def scans_b(rawdata_root: Path) -> pd.DataFrame:
    return load_raw_scans("backward", rawdata_root)


@pytest.fixture(scope="session")
def protocol_a_fold(scans_f, scans_b):
    return iter_protocol_a(scans_f, scans_b)[0]


def _scan_rows(location_p: int, ap_name: str, rssi_list: list[float], freq: int = 2400, start: int = 0):
    return [
        {
            "location_p": location_p,
            "ssid": "s",
            "rssi": float(v),
            "frequency": freq,
            "count": start + c,
            "ap_name": ap_name,
        }
        for c, v in enumerate(rssi_list)
    ]


def _make_scans(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --- 1: discriminative & stable key scores higher than flat & noisy ------

def test_fisher_score_discriminative_key_beats_flat_noisy_key():
    rows = []
    # AP-DISCRIM: mu varies strongly across 5 locations, sigma pinned at floor
    # (std 0 -> SIGMA_MIN_DB) -> large Var_l(mu) / small mean(sigma^2).
    discrim_mu = {1: -40.0, 2: -50.0, 3: -60.0, 4: -70.0, 5: -80.0}
    for loc, mu in discrim_mu.items():
        rows += _scan_rows(loc, "AP-DISCRIM", [mu] * 10)
    # AP-FLAT: same median at every location, but noisy scans (large std)
    # -> ~zero Var_l(mu) / large mean(sigma^2).
    rng = np.random.default_rng(0)
    for loc in discrim_mu:
        noisy = (-60.0 + rng.normal(0.0, 8.0, size=10)).tolist()
        rows += _scan_rows(loc, "AP-FLAT", noisy)

    stats = fw.location_feature_stats(_make_scans(rows))
    scores = fw._fisher_scores(stats)

    assert scores[("AP-DISCRIM", "2.4G")] > scores[("AP-FLAT", "2.4G")]


# --- 2: keys detected at < 3 locations are excluded -----------------------

def test_fisher_score_excludes_keys_detected_below_three_locations():
    rows = []
    # AP-RARE detected at only 2 of 5 locations -> ineligible.
    for loc in (1, 2, 3, 4, 5):
        rows += _scan_rows(loc, "AP-COMMON", [-50.0 - loc] * 10)
    rows += _scan_rows(1, "AP-RARE", [-55.0] * 10)
    rows += _scan_rows(2, "AP-RARE", [-56.0] * 10)

    stats = fw.location_feature_stats(_make_scans(rows))
    scores = fw._fisher_scores(stats)

    assert ("AP-COMMON", "2.4G") in scores
    assert ("AP-RARE", "2.4G") not in scores


# --- 2b: eligibility and sigma population share one location mask (F6) ------

def test_key_with_sparse_scans_everywhere_is_excluded_not_nan():
    # AP-THIN is detected at 4 locations but with only 2 scans each -> sigma is
    # NaN at every location (n_detect < MIN_COUNT). The key must be excluded by
    # the eligibility rule (finite-sigma locations < 3), not crash or emit NaN.
    rows = []
    for loc in (1, 2, 3, 4):
        rows += _scan_rows(loc, "AP-BASE", [-50.0 - loc] * 10)
        rows += _scan_rows(loc, "AP-THIN", [-60.0 - loc] * 2)
    stats = fw.location_feature_stats(_make_scans(rows))
    scores = fw._fisher_scores(stats)

    assert ("AP-BASE", "2.4G") in scores
    assert ("AP-THIN", "2.4G") not in scores
    assert all(np.isfinite(v) for v in scores.values())


def test_key_with_fewer_than_three_finite_sigma_locations_is_excluded():
    # AP-MIXED is detected at 4 locations (mu finite at all four) but has >=
    # MIN_COUNT scans at only 2 of them (finite sigma at 2 < 3). Eligibility is
    # defined on the finite-sigma population, so the key must be excluded even
    # though its detection count (4) passes the old mu-based rule.
    rows = []
    for loc in (1, 2, 3, 4):
        rows += _scan_rows(loc, "AP-BASE", [-50.0 - loc] * 10)
    rows += _scan_rows(1, "AP-MIXED", [-60.0] * 10)
    rows += _scan_rows(2, "AP-MIXED", [-62.0] * 10)
    rows += _scan_rows(3, "AP-MIXED", [-64.0] * 2)
    rows += _scan_rows(4, "AP-MIXED", [-66.0] * 2)
    stats = fw.location_feature_stats(_make_scans(rows))
    scores = fw._fisher_scores(stats)

    assert ("AP-MIXED", "2.4G") not in scores


def test_numerator_variance_uses_finite_sigma_location_mask():
    # AP-PART: finite sigma at locations 1-3 (10 scans), detected-but-thin at
    # locations 4-5 (2 scans, extreme mu). The Fisher numerator must use ONLY
    # the finite-sigma locations {1,2,3}; including 4-5's mu values would
    # inflate Var far beyond the hand-computed value.
    rows = []
    for loc in (1, 2, 3, 4, 5):
        rows += _scan_rows(loc, "AP-BASE", [-50.0 - loc] * 10)
    part_mu = {1: -40.0, 2: -50.0, 3: -60.0}
    for loc, m in part_mu.items():
        rows += _scan_rows(loc, "AP-PART", [m] * 10)
    rows += _scan_rows(4, "AP-PART", [-90.0] * 2)
    rows += _scan_rows(5, "AP-PART", [-20.0] * 2)
    stats = fw.location_feature_stats(_make_scans(rows))
    scores = fw._fisher_scores(stats)

    key = ("AP-PART", "2.4G")
    assert key in scores
    # sigma floors to SIGMA_MIN_DB=1 at each eligible location (std=0), so
    # denominator = 1; numerator = population var of {-40,-50,-60} = 200/3.
    expected = float(np.var([-40.0, -50.0, -60.0])) / 1.0
    assert scores[key] == pytest.approx(expected)


# --- 3: equal-variance keys never yield NaN --------------------------------

def test_fisher_score_equal_variance_keys_no_nan():
    rows = []
    for name in ("AP-X", "AP-Y", "AP-Z"):
        for loc, mu in zip((1, 2, 3, 4), (-40.0, -50.0, -60.0, -70.0)):
            rows += _scan_rows(loc, name, [mu] * 10)

    stats = fw.location_feature_stats(_make_scans(rows))
    scores = fw._fisher_scores(stats)

    assert len(scores) == 3
    assert all(np.isfinite(v) for v in scores.values())
    assert not any(np.isnan(v) for v in scores.values())


# --- 4: inner CV selects hyperparams from the documented grid --------------

def test_fit_selects_hyperparams_from_grid(protocol_a_fold, ap_coords, location_coords):
    fold = protocol_a_fold
    method = FisherWknn()
    method.fit(fold.train_scans, ap_coords, location_coords[
        location_coords["location_p"].isin(fold.train_scans["location_p"].unique())
    ])

    assert method.selected_m in M_GRID
    assert method.selected_k in K_GRID
    assert method.selected_weighting in WEIGHTING_GRID
    assert method.diagnostics_["selected_M"] == method.selected_m
    assert method.diagnostics_["selected_k"] == method.selected_k
    assert method.diagnostics_["selected_weighting"] == method.selected_weighting
    assert np.isfinite(method.diagnostics_["cv_best_score"])


# --- 5: leak spy -- inner CV Fisher selection never touches val locations -

def _synthetic_cv_scans(n_loc: int = 15) -> pd.DataFrame:
    rows = []
    for loc in range(1, n_loc + 1):
        for i, name in enumerate(("AP-A", "AP-B", "AP-C")):
            mu = -40.0 - 3.0 * loc - 5.0 * i
            rows += _scan_rows(loc, name, [mu] * 10)
    return _make_scans(rows)


def _synthetic_cv_coords(n_loc: int = 15) -> pd.DataFrame:
    return pd.DataFrame({
        "location_p": list(range(1, n_loc + 1)),
        "x": [float(p) for p in range(1, n_loc + 1)],
        "y": [0.0] * n_loc,
    })


def test_inner_cv_fisher_selection_is_leakproof(monkeypatch):
    train_scans = _synthetic_cv_scans()
    location_coords_syn = _synthetic_cv_coords()

    calls: list[set[int]] = []
    real_select_keys = fw._select_keys

    def spy_select_keys(stats, m):
        calls.append(set(int(loc) for loc in stats.mu.index))
        return real_select_keys(stats, m)

    monkeypatch.setattr(fw, "_select_keys", spy_select_keys)

    method = FisherWknn()
    method.fit(train_scans, None, location_coords_syn)

    n_candidates = len(fw._CANDIDATES)
    # 5 inner folds x n_candidates calls during CV, plus 1 final full-train call.
    assert len(calls) == 5 * n_candidates + 1

    folds = list(iter_inner_cv(train_scans, k=5, seed=RANDOM_SEED))
    idx = 0
    for inner_train, inner_val in folds:
        train_locs = set(int(loc) for loc in inner_train["location_p"].unique())
        val_locs = set(int(loc) for loc in inner_val["location_p"].unique())
        for _ in range(n_candidates):
            call_locs = calls[idx]
            assert call_locs == train_locs
            assert call_locs.isdisjoint(val_locs)
            idx += 1

    assert calls[-1] == set(int(loc) for loc in train_scans["location_p"].unique())


# --- 6: leak spy -- outer test location never reaches fit (via run_method) -

def test_outer_test_location_excluded_from_fit_db(scans_f, scans_b, ap_coords, location_coords, monkeypatch):
    # F7: the spy rides the REAL production path (run_method -> fit) and
    # compares the exact location set reaching fit against the train subset.
    fold = iter_protocol_a(scans_f, scans_b)[0]
    keep = sorted(fold.train_scans["location_p"].unique())[:30]
    train30 = fold.train_scans[fold.train_scans["location_p"].isin(keep)]
    held = set(int(x) for x in location_coords["location_p"]) - set(keep)
    assert held  # truth has 59, train has 30

    seen: dict[str, set[int]] = {}
    real_fit = fw.FisherWknn.fit

    def spy_fit(self, train_scans, ap_coords_arg, location_coords_arg):
        seen["fit_coords"] = set(int(x) for x in location_coords_arg["location_p"])
        result = real_fit(self, train_scans, ap_coords_arg, location_coords_arg)
        seen["db_locs"] = set(int(loc) for loc in self._db_locs)
        return result

    monkeypatch.setattr(fw.FisherWknn, "fit", spy_fit)

    est = run_method("fisher_wknn", train30, fold.test_scans, ap_coords, location_coords)

    assert seen["fit_coords"] == set(keep)
    assert seen["fit_coords"].isdisjoint(held)
    assert seen["db_locs"] == set(keep)
    assert len(est) == 59


# --- 7: Protocol A 1 fold smoke --------------------------------------------

def test_protocol_a_smoke(protocol_a_fold, ap_coords, location_coords):
    fold = protocol_a_fold
    est = run_method(
        "fisher_wknn", fold.train_scans, fold.test_scans, ap_coords, location_coords
    )

    assert len(est) == 59
    assert set(est.columns) == {"location_p", "x", "y"}
    assert not est.isna().any().any()
    assert np.isfinite(est[["x", "y"]]).all().all()


# --- 8: LOLO smoke (3 locations) --------------------------------------------

def test_lolo_smoke(scans_f, location_coords, ap_coords):
    for fold in islice(iter_lolo(scans_f), 3):
        est = run_method(
            "fisher_wknn", fold.train_scans, fold.test_scans, ap_coords, location_coords
        )
        assert len(est) == 1
        assert set(est.columns) == {"location_p", "x", "y"}
        assert not est.isna().any().any()
        assert np.isfinite(est[["x", "y"]]).all().all()
        assert int(est["location_p"].iloc[0]) == fold.held_out


# --- 9: mandatory full contract smoke ---------------------------------------

def test_fisher_wknn_smoke(scans_f, scans_b, ap_coords, location_coords):
    fold = iter_protocol_a(scans_f, scans_b)[0]

    est = run_method(
        "fisher_wknn",
        fold.train_scans, fold.test_scans,
        ap_coords, location_coords,
    )

    assert len(est) == 59
    assert set(est.columns) == {"location_p", "x", "y"}
    assert not est.isna().any().any()
    assert est["location_p"].min() == 1
    assert est["location_p"].max() == 59
    assert np.isfinite(est[["x", "y"]]).all().all()
