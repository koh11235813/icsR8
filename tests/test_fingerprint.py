import pandas as pd
import pytest

from icsr8.fingerprint import candidate_medians, reproduction_fingerprint


@pytest.fixture()
def ap_coords():
    return pd.DataFrame(
        [
            {"ap_name": "AP-3F-A", "floor": 3, "x": 0.0, "y": 0.0, "z": 9.7},
            {"ap_name": "AP-3F-B", "floor": 3, "x": 10.0, "y": 0.0, "z": 9.7},
        ]
    )


def _scans_for_one_ap(location_p, ap_name, ssid, frequency, rssis):
    return [
        {
            "location_p": location_p,
            "ssid": ssid,
            "rssi": r,
            "frequency": frequency,
            "count": i,
            "ap_name": ap_name,
            "direction": "forward",
        }
        for i, r in enumerate(rssis)
    ]


def test_candidate_medians_uses_median_not_mean(ap_coords):
    rssis = [-90, -50, -50, -50, -50, -50, -50, -50, -50, -50]  # mean -54, median -50
    scans = pd.DataFrame(_scans_for_one_ap(1, "AP-3F-A", "tutwifi", 2412, rssis))

    fp = candidate_medians(scans, ap_coords)

    assert len(fp) == 1
    assert fp.iloc[0]["rssi_median"] == -50


def test_candidate_medians_keeps_per_ssid_freq_rows(ap_coords):
    scans = pd.DataFrame(
        _scans_for_one_ap(1, "AP-3F-A", "tutwifi", 2412, [-40] * 10)
        + _scans_for_one_ap(1, "AP-3F-A", "tutwifi2025", 2412, [-45] * 10)
        + _scans_for_one_ap(1, "AP-3F-A", "tutwifi", 5220, [-55] * 10)
    )

    fp = candidate_medians(scans, ap_coords)

    assert len(fp) == 3
    key = fp.set_index(["ap_name", "ssid", "frequency"])["rssi_median"]
    assert key[("AP-3F-A", "tutwifi", 2412)] == -40
    assert key[("AP-3F-A", "tutwifi2025", 2412)] == -45
    assert key[("AP-3F-A", "tutwifi", 5220)] == -55


def test_candidate_medians_drops_non_3F_aps_when_restricted(ap_coords):
    scans = pd.DataFrame(
        _scans_for_one_ap(1, "AP-3F-A", "tutwifi", 2412, [-40] * 10)
        + _scans_for_one_ap(1, "AP-4F-X", "tutwifi", 2412, [-30] * 10)
    )

    fp = candidate_medians(scans, ap_coords, restrict_to_known_aps=True)

    assert set(fp["ap_name"]) == {"AP-3F-A"}


def test_candidate_medians_joins_xy_from_ap_coords(ap_coords):
    scans = pd.DataFrame(_scans_for_one_ap(1, "AP-3F-B", "tutwifi", 2412, [-50] * 10))

    fp = candidate_medians(scans, ap_coords)

    row = fp.iloc[0]
    assert row["x"] == 10.0
    assert row["y"] == 0.0


def test_reproduction_fingerprint_collapses_per_physical_ap(ap_coords):
    scans = pd.DataFrame(
        _scans_for_one_ap(1, "AP-3F-A", "tutwifi", 2412, [-60] * 10)
        + _scans_for_one_ap(1, "AP-3F-A", "tutwifi2025", 2412, [-55] * 10)
        + _scans_for_one_ap(1, "AP-3F-A", "tutwifi", 5220, [-50] * 10)
        + _scans_for_one_ap(1, "AP-3F-B", "tutwifi", 2412, [-70] * 10)
    )
    candidates = candidate_medians(scans, ap_coords)

    fp = reproduction_fingerprint(candidates, allowed_wings={"3F"})

    assert set(fp["ap_name"]) == {"AP-3F-A", "AP-3F-B"}
    row_a = fp.loc[fp["ap_name"] == "AP-3F-A"].iloc[0]
    assert row_a["rssi_median"] == -50  # strongest variant wins
    row_b = fp.loc[fp["ap_name"] == "AP-3F-B"].iloc[0]
    assert row_b["rssi_median"] == -70


def test_reproduction_fingerprint_excludes_c1_wing_by_default():
    """公表ベースラインは C 棟群 (C0/C2/C3) のみで C1 棟 AP を除外する。"""
    cand = pd.DataFrame(
        [
            {"location_p": 1, "ap_name": "AP-C0-3F-01", "ssid": "tutwifi",
             "frequency": 2412, "rssi_median": -50, "x": 30.0, "y": 1.0},
            {"location_p": 1, "ap_name": "AP-C1-3F-01", "ssid": "tutwifi",
             "frequency": 2412, "rssi_median": -45, "x": 31.5, "y": 15.8},
        ]
    )
    fp = reproduction_fingerprint(cand)
    assert set(fp["ap_name"]) == {"AP-C0-3F-01"}
