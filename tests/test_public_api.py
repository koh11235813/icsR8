"""トップレベル ``icsr8`` の公開 API 面（Phase 0 統合）を検査する。

新 API がパッケージ直下に露出しているか、``run_method`` が意図通り露出して
いないか（呼び出し側にレジストリの明示を強制）、``icsr8.methods`` が
サブパッケージとして到達可能かを守る。
"""

from __future__ import annotations

import icsr8

_NEW_EXPORTS = (
    "xy_to_arclength",
    "arclength_to_xy",
    "project_to_corridor",
    "segment_of",
    "geodesic_distance",
    "iter_protocol_a",
    "iter_lolo",
    "iter_inner_cv",
    "percentiles",
    "within_ratio",
    "bootstrap_ci_paired",
    "errors_ledger",
    "band_of",
    "detailed_fingerprint",
    "candidate_aggregate",
    "load_ap_coords_all",
)


def test_new_exports_are_top_level() -> None:
    for name in _NEW_EXPORTS:
        assert hasattr(icsr8, name), name
        assert name in icsr8.__all__, name


def test_run_method_not_top_level() -> None:
    assert not hasattr(icsr8, "run_method")
    assert "run_method" not in icsr8.__all__


def test_methods_subpackage_importable() -> None:
    import icsr8.methods

    assert "run_method" in dir(icsr8.methods)
    assert "methods" in icsr8.__all__
