"""メソッドレジストリ。全推定器へ fit/predict の統一エントリを与える。

Why-not (estimators.py を書き換えない):
    フリーズ済みの PBL/CLA/WCL に一切触れないことで公表値再現へのリスクを
    ゼロにする。run_method が新しい統一エントリであり、baselines.py の
    アダプタ経由でのみ既存推定器を呼ぶ。
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pandas as pd

from icsr8.methods.base import Method

REGISTRY: dict[str, type[Method]] = {}


def register(cls: type[Method]) -> type[Method]:
    existing = REGISTRY.get(cls.name)
    # Why not always raise on a taken name: モジュール reload は同じ decorator を
    #   「同一クラス」に再適用するので冪等でなければならない。__module__ +
    #   __qualname__ が一致すれば同一定義とみなし上書きする。別クラスが既存名を
    #   奪う場合のみ本物の衝突として ValueError。
    if existing is not None and (
        existing.__module__ != cls.__module__
        or existing.__qualname__ != cls.__qualname__
    ):
        raise ValueError(f"method name already registered: {cls.name!r}")
    REGISTRY[cls.name] = cls
    return cls


def available_methods() -> list[str]:
    return sorted(REGISTRY)


def run_method(
    name: str,
    train_scans: pd.DataFrame,
    test_scans: pd.DataFrame,
    ap_coords: pd.DataFrame,
    location_coords: pd.DataFrame,
    **params,
) -> pd.DataFrame:
    if name not in REGISTRY:
        raise ValueError(
            f"unknown method: {name!r}; available: {available_methods()}"
        )
    # Why not trust callers to pre-filter: test location の座標を fit に渡さない
    #   ことを構造で保証する。train_scans に現れる location_p のみへ絞り込めば、
    #   リークは呼び出し側の注意力に依存せず起こり得なくなる。
    train_location_coords = location_coords[
        location_coords["location_p"].isin(train_scans["location_p"].unique())
    ]
    # Why not accept partial coverage silently: 座標が欠けた train 地点は fit 側で
    #   暗黙に脱落して DB が静かに痩せ、重複行は index 整合を壊す。どちらも
    #   契約違反としてここで弾く。
    dup_mask = train_location_coords["location_p"].duplicated()
    if dup_mask.any():
        dups = sorted(train_location_coords.loc[dup_mask, "location_p"].unique())
        raise ValueError(f"duplicate location_p in location_coords: {dups}")
    missing = sorted(
        set(train_scans["location_p"].unique())
        - set(train_location_coords["location_p"])
    )
    if missing:
        raise ValueError(f"location_coords missing train locations: {missing}")
    method = REGISTRY[name](**params).fit(train_scans, ap_coords, train_location_coords)
    return method.predict(test_scans)


# Why-not (明示 import ではなく自動探索):
#   将来のメソッドスライスが並列にこの共有ファイルを編集して衝突するのを避ける。
#   自動探索なら各メソッドは自身のモジュールに完全に閉じる。register を束縛した
#   後に走らせる必要があるため、この探索は末尾に置く。
for _module in pkgutil.iter_modules(__path__):
    _mod = importlib.import_module(f"{__name__}.{_module.name}")
    # Why not import 副作用（@register）だけに頼らない: importlib.reload で本
    #   パッケージを再実行すると REGISTRY は空に戻るが、子モジュールは sys.modules
    #   にキャッシュ済みなので import_module が decorator を再実行しない。空のまま
    #   にならぬよう、キャッシュ済み名前空間から具象 Method を拾って setdefault で
    #   REGISTRY を再構築する。
    for _obj in vars(_mod).values():
        if (
            isinstance(_obj, type)
            and issubclass(_obj, Method)
            and not inspect.isabstract(_obj)
            and getattr(_obj, "name", None) is not None
        ):
            REGISTRY.setdefault(_obj.name, _obj)


__all__ = ["REGISTRY", "Method", "available_methods", "register", "run_method"]
