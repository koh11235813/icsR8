"""改善手法（Tier 1–3）実装が共有する定数。

推定器・CV 分割・bootstrap の再現性を単一ソースで担保する。
"""

from __future__ import annotations

RANDOM_SEED = 0

# 周波数帯の分類境界 (MHz, 両端含む)。
# Why not チャネル列挙: データセットの帯域判定には粗い 3 区分で十分で、
# 閉区間なら将来チャネルが増えても表の更新が不要。
BAND_BOUNDARIES_MHZ: dict[str, tuple[int, int]] = {
    "2.4G": (2400, 2500),
    "5G": (5150, 5895),
    "6G": (5925, 7125),
}

# Student-t / 分散重みの数値ガード（doc/improvement_methods_note.txt 手法2, 11）。
SIGMA_MIN_DB = 1.0
MIN_COUNT = 3

# fingerprint 特徴ベクトルの非検出埋め値 (dBm)。手法1 (WKNN) 仕様。
NON_DETECT_DBM = -100.0

# L 字廊下の区分線形経路（doc/improvement_methods_note.txt 手法7）。
# 59 測定点は全てこの折れ線上に正確に載ることを実データで確認済み
# （最大直交距離 0.0 m）。
CORRIDOR_SEGMENTS: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = (
    ((32.0, 0.0), (0.0, 0.0)),  # C 棟   (east-west)
    ((0.0, 0.0), (0.0, 56.0)),  # C2 棟  (north-south)
    ((0.0, 56.0), (28.0, 56.0)),  # C3 棟  (east-west)
)

# 公表ベースラインが暗黙に用いる棟フィルタと blacklist 対象
# （手法10: 廊下から離れた部屋内設置 AP）。
BLACKLIST_APS: frozenset[str] = frozenset({"AP-C0-3F-04"})
