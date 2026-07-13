import pandas as pd
import pytest

from icsr8.io import load_raw_scans
from icsr8.protocols import Fold, LoloFold, iter_inner_cv, iter_lolo, iter_protocol_a


@pytest.fixture(scope="module")
def scans_forward(rawdata_root):
    return load_raw_scans("forward", rawdata_root)


@pytest.fixture(scope="module")
def scans_backward(rawdata_root):
    return load_raw_scans("backward", rawdata_root)


class FitSpy:
    """集計統計を fit する側の最小スタブ。train に渡された location_p を記録する。"""

    def __init__(self):
        self.seen: set[int] = set()

    def fit(self, train_scans):
        self.seen = set(train_scans["location_p"])


def test_iter_protocol_a_returns_two_named_folds(scans_forward, scans_backward):
    folds = iter_protocol_a(scans_forward, scans_backward)

    assert [f.name for f in folds] == ["forward_to_backward", "backward_to_forward"]
    assert all(isinstance(f, Fold) for f in folds)

    fwd_to_bwd, bwd_to_fwd = folds
    assert fwd_to_bwd.train_scans is scans_forward
    assert fwd_to_bwd.test_scans is scans_backward
    assert bwd_to_fwd.train_scans is scans_backward
    assert bwd_to_fwd.test_scans is scans_forward


def test_iter_lolo_single_pool_yields_one_fold_per_location(scans_forward):
    all_locs = set(scans_forward["location_p"])
    folds = list(iter_lolo(scans_forward))

    assert len(folds) == 59
    assert [f.held_out for f in folds] == sorted(all_locs)

    for fold in folds:
        assert isinstance(fold, LoloFold)
        assert isinstance(fold.held_out, int)
        train_locs = set(fold.train_scans["location_p"])
        test_locs = set(fold.test_scans["location_p"])

        assert fold.held_out not in train_locs
        assert test_locs == {fold.held_out}
        assert train_locs | test_locs == all_locs
        assert len(fold.train_scans) + len(fold.test_scans) == len(scans_forward)


def test_iter_lolo_test_rows_come_from_test_pool(scans_forward, scans_backward):
    for fold in iter_lolo(scans_forward, scans_backward):
        assert set(fold.test_scans["direction"]) == {"backward"}
        assert set(fold.train_scans["direction"]) == {"forward"}
        assert set(fold.test_scans["location_p"]) == {fold.held_out}
        assert fold.held_out not in set(fold.train_scans["location_p"])


def test_iter_lolo_mismatched_location_sets_raises(scans_forward, scans_backward):
    dropped = scans_backward[scans_backward["location_p"] != 1]

    with pytest.raises(ValueError):
        iter_lolo(scans_forward, dropped)


def test_iter_inner_cv_partitions_locations(scans_forward):
    all_locs = set(scans_forward["location_p"])
    folds = list(iter_inner_cv(scans_forward, k=5, seed=0))

    assert len(folds) == 5

    val_sets = [set(val["location_p"]) for _, val in folds]

    union = set().union(*val_sets)
    assert union == all_locs

    for i, a in enumerate(val_sets):
        for b in val_sets[i + 1 :]:
            assert a.isdisjoint(b)

    for train, val in folds:
        assert set(train["location_p"]).isdisjoint(set(val["location_p"]))
        assert set(train["location_p"]) | set(val["location_p"]) == all_locs


def test_iter_inner_cv_is_deterministic_per_seed(scans_forward):
    def partition(seed):
        return tuple(
            frozenset(val["location_p"]) for _, val in iter_inner_cv(scans_forward, k=5, seed=seed)
        )

    assert partition(0) == partition(0)
    assert partition(0) != partition(1)


def test_iter_lolo_leakage_contract_spy(scans_forward):
    for fold in iter_lolo(scans_forward):
        spy = FitSpy()
        spy.fit(fold.train_scans)
        assert fold.held_out not in spy.seen


def test_iter_lolo_single_location_raises():
    single = pd.DataFrame({"location_p": [5, 5, 5]})
    with pytest.raises(ValueError):
        iter_lolo(single)


def test_iter_inner_cv_raises_on_empty_scans():
    with pytest.raises(ValueError):
        iter_inner_cv(pd.DataFrame({"location_p": []}))


@pytest.mark.parametrize("k", [1, 60])
def test_iter_inner_cv_raises_on_k_out_of_range(scans_forward, k):
    # 59 locations: k=1 は下限違反、k=60 は上限違反（n_locations 超過）。
    with pytest.raises(ValueError):
        iter_inner_cv(scans_forward, k=k)
