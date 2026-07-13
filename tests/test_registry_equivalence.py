"""メソッドレジストリ (icsr8.methods) の等価性テスト。

run_method 経由の PBL/CLA/WCL が、既存の estimate_* 直呼びと
ビット等価であることを固定する。レジストリはフリーズ済みベースラインへ
薄いアダプタを被せるだけなので、公表値の再現性を壊さないことが最重要。
"""

import importlib
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

import icsr8.methods
from icsr8.estimators import (
    estimate_cla,
    estimate_pbl,
    estimate_wcl,
    estimate_with_trace,
)
from icsr8.fingerprint import candidate_medians, reproduction_fingerprint
from icsr8.io import load_ap_coords, load_location_coords, load_raw_scans
from icsr8.methods import available_methods, register, run_method
from icsr8.methods.base import Method

DIRECT = {"pbl": estimate_pbl, "cla": estimate_cla, "wcl": estimate_wcl}


@pytest.fixture(scope="module")
def ap13(dataset_dir: Path) -> pd.DataFrame:
    return load_ap_coords(dataset_dir / "AP_coordinate_C3F.csv")


@pytest.fixture(scope="module")
def loc_coords(dataset_dir: Path) -> pd.DataFrame:
    return load_location_coords(dataset_dir / "location_coordinate_C.csv")[
        ["location_p", "x", "y"]
    ]


@pytest.fixture(scope="module")
def scans(rawdata_root: Path) -> dict[str, pd.DataFrame]:
    return {
        "forward": load_raw_scans("forward", rawdata_root),
        "backward": load_raw_scans("backward", rawdata_root),
    }


def _direct_estimate(method: str, direction_scans: pd.DataFrame, ap13: pd.DataFrame):
    fp = reproduction_fingerprint(candidate_medians(direction_scans, ap13))
    return DIRECT[method](fp).sort_values("location_p").reset_index(drop=True)


# --- 1: run_method == estimate_* direct (both directions) --------------------

@pytest.mark.parametrize("method", ["pbl", "cla", "wcl"])
@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_run_method_equals_direct(method, direction, scans, ap13, loc_coords):
    other = "backward" if direction == "forward" else "forward"
    got = run_method(
        method,
        train_scans=scans[other],
        test_scans=scans[direction],
        ap_coords=ap13,
        location_coords=loc_coords,
    )
    assert list(got.columns) == ["location_p", "x", "y"]
    got = got.sort_values("location_p").reset_index(drop=True)
    expected = _direct_estimate(method, scans[direction], ap13)

    assert got["location_p"].tolist() == expected["location_p"].tolist()
    assert (got["x"] - expected["x"]).abs().max() < 1e-12
    assert (got["y"] - expected["y"]).abs().max() < 1e-12


# --- 2: pin existing estimate_with_trace API --------------------------------

@pytest.mark.parametrize("method", ["pbl", "cla", "wcl"])
def test_estimate_with_trace_matches_estimate(method, scans, ap13):
    fp = reproduction_fingerprint(candidate_medians(scans["forward"], ap13))
    traced = estimate_with_trace(fp, method)[0].sort_values("location_p").reset_index(drop=True)
    direct = DIRECT[method](fp).sort_values("location_p").reset_index(drop=True)

    assert traced["location_p"].tolist() == direct["location_p"].tolist()
    assert traced["x"].tolist() == direct["x"].tolist()
    assert traced["y"].tolist() == direct["y"].tolist()


# --- 3: unknown method name --------------------------------------------------

def test_run_method_unknown_name(scans, ap13, loc_coords):
    with pytest.raises(ValueError) as exc:
        run_method(
            "nope",
            train_scans=scans["forward"],
            test_scans=scans["forward"],
            ap_coords=ap13,
            location_coords=loc_coords,
        )
    assert "wcl" in str(exc.value)


# --- 4: duplicate registration ----------------------------------------------

def test_duplicate_name_raises():
    with pytest.raises(ValueError):

        @register
        class _Dup(Method):
            name = "wcl"
            uses_geometry = True

            def fit(self, train_scans, ap_coords):
                return self

            def predict(self, test_scans):
                return pd.DataFrame(columns=["location_p", "x", "y"])


# --- 5: registry contents ----------------------------------------------------

def test_available_methods():
    # Why not exact equality: 将来のメソッド追加で必ず壊れる。ベースライン 3 種が
    # 部分集合として含まれることだけを固定する。
    assert {"pbl", "cla", "wcl"} <= set(available_methods())


# --- 6: run_method filters location_coords to train (leakage guard) ----------

def test_run_method_filters_location_coords_to_train(scans, ap13, loc_coords):
    seen: dict[str, set[int]] = {}

    class _LeakSpy(Method):
        name = "_leak_spy"
        uses_geometry: ClassVar[bool] = False

        def fit(self, train_scans, ap_coords, location_coords):
            seen["locations"] = set(location_coords["location_p"])
            return self

        def predict(self, test_scans):
            return pd.DataFrame(columns=["location_p", "x", "y"])

    register(_LeakSpy)

    keep = sorted(scans["forward"]["location_p"].unique())[:30]
    train30 = scans["forward"][scans["forward"]["location_p"].isin(keep)]
    assert set(loc_coords["location_p"]) > set(keep)  # truth has all 59, train has 30

    run_method(
        "_leak_spy",
        train_scans=train30,
        test_scans=scans["forward"],
        ap_coords=ap13,
        location_coords=loc_coords,
    )

    assert seen["locations"] == set(keep)


# --- 6b: run_method rejects broken location_coords ---------------------------

def test_run_method_raises_on_missing_train_coords(scans, ap13, loc_coords):
    missing_one = loc_coords[loc_coords["location_p"] != 1]
    with pytest.raises(ValueError, match="missing"):
        run_method(
            "wcl",
            train_scans=scans["backward"],
            test_scans=scans["forward"],
            ap_coords=ap13,
            location_coords=missing_one,
        )


def test_run_method_raises_on_duplicate_coords(scans, ap13, loc_coords):
    dup = pd.concat([loc_coords, loc_coords.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        run_method(
            "wcl",
            train_scans=scans["backward"],
            test_scans=scans["forward"],
            ap_coords=ap13,
            location_coords=dup,
        )


# --- 7: registry reload-safety ------------------------------------------------

def test_register_same_class_idempotent():
    cls = icsr8.methods.REGISTRY["pbl"]
    assert register(cls) is cls  # 同一クラスの再登録は冪等（例外を投げない）


def test_registry_survives_reload():
    importlib.reload(icsr8.methods)
    importlib.reload(icsr8.methods)
    assert {"pbl", "cla", "wcl"} <= set(icsr8.methods.available_methods())
