from dataclasses import dataclass
from typing import Optional



def _require(cond, msg):
    if cond is None:
        raise ValueError(msg)
    return cond


# -------- Context object all trajectories receive --------
@dataclass(frozen=True)
class TrajContext:
    """
    Bag of knobs/options shared by all trajectory adapters so the call
    signature is uniform: f(t: float, ctx: TrajContext) -> NDArray[np.float64]
    """
    sim: bool

    # Optional knobs used by specific trajectories
    hover_mode: Optional[int] = None               # hover: which preset (1..9)

    spin: Optional[bool] = None
    double_speed: Optional[bool] = None            
    short: Optional[bool] = None              # fig8_vertical: which figure-8
    # flight_time: Optional[float] = None         # sawtooth/triangle
    # state: Optional[FlatOutput] = None          # triangle: current state
    # made_it: Optional[bool] = None              # triangle: first-point latch
