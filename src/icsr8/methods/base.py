"""推定メソッドの抽象基底。fit/predict の統一エントリを定義する。

`uses_geometry`
    True  … AP 座標を幾何的に消費するメソッド。座標既知の 3F 13-AP 表を
            与えなければならない。
    False … 指紋のみで動くメソッド。座標未知 AP も含め全 AP を利用してよい。
"""

from __future__ import annotations

import abc
from typing import ClassVar

import pandas as pd


class Method(abc.ABC):
    name: ClassVar[str]
    uses_geometry: ClassVar[bool]

    @abc.abstractmethod
    def fit(
        self,
        train_scans: pd.DataFrame,
        ap_coords: pd.DataFrame,
        location_coords: pd.DataFrame,
    ) -> "Method":
        """指紋 DB を学習する。

        `location_coords`
            列 [location_p, x, y] = 学習に用いる TRAINING location のみの
            実測座標（指紋 DB の参照点位置）。WKNN/GP など座標を要する手法は
            これを唯一の学習用グラウンドトゥルースとして扱うこと。run_method が
            train_scans の location_p で事前フィルタ済みなので、test location の
            座標は構造的に混入しない。
        """
        ...

    @abc.abstractmethod
    def predict(self, test_scans: pd.DataFrame) -> pd.DataFrame:
        ...
