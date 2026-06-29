from typing import Literal, NamedTuple

Direction = Literal["forward", "backward"]


class Candidate(NamedTuple):
    ap_name: str
    ssid: str
    frequency: int
    rssi_median: float
    x: float
    y: float


class Estimate(NamedTuple):
    location_p: int
    x: float
    y: float
