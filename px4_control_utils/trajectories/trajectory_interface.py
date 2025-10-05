import jax
import numpy as np
from enum import StrEnum
from typing import Callable, Dict, Tuple, Union
from px4_control_utils.trajectories.trajectory_context import TrajContext

# from .nondifferentiable_trajectories import(
#     hover
# )

from .differentiable_trajectories import(
    hover,
    circle_horizontal,
    circle_vertical,
    fig8_horizontal,
    fig8_vertical
)


TrajArray = jax.Array | np.ndarray
TrajReturn = Union[TrajArray, Tuple[TrajArray, TrajArray]]

# using Callable avoids parameter-name nitpicks
TrajectoryFunc = Callable[[float, TrajContext], TrajReturn]


class TrajectoryType(StrEnum):
    HOVER = "hover"
    CIRCLE_HORIZONTAL = "circle_horz"
    CIRCLE_VERTICAL = "circle_vert"
    FIG8_HORIZONTAL = "fig8_horz"
    FIG8_VERTICAL = "fig8_vert"
    # HELIX = "helix"
    # HELIX_SPIN = "helix_spin"
    # YAW_ONLY = "yaw_only"
    # SAWTOOTH = "sawtooth"
    # TRIANGLE = "triangle"


TRAJ_FUNC_REGISTRY: Dict[str, TrajectoryFunc] = {
    TrajectoryType.HOVER: hover,
    TrajectoryType.CIRCLE_HORIZONTAL: circle_horizontal,
    TrajectoryType.CIRCLE_VERTICAL: circle_vertical,
    TrajectoryType.FIG8_HORIZONTAL: fig8_horizontal,
    TrajectoryType.FIG8_VERTICAL: fig8_vertical
    }

