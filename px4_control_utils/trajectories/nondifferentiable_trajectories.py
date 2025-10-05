# import numpy as np
# from typing import Dict
# from px4_control_utils.trajectories.trajectory_context import TrajContext

# def hover(t_traj: float, ctx: TrajContext):
#     """Returns constant hover reference trajectories at a few positions."""
#     mode = ctx.mode
#     sim = ctx.sim

#     hover_dict: Dict[int, np.ndarray] = {
#         1: np.array([0.0, 0.0, -0.9, 0.0]),
#         2: np.array([0.0, 0.8, -0.8, 0.0]),
#         3: np.array([0.8, 0.0, -0.8, 0.0]),
#         4: np.array([0.8, 0.8, -0.8, 0.0]),
#         5: np.array([0.0, 0.0, -10.0, 0.0]),
#         6: np.array([1.0, 1.0, -4.0, 0.0]),
#         7: np.array([3.0, 4.0, -5.0, 0.0]),
#         8: np.array([1.0, 1.0, -3.0, 0.0]),
#         9: np.array([0.0, 0.0, -(0.8*np.sin((2*np.pi/5)*t_traj)+1.0), 0.0]),
#     }

#     if mode not in hover_dict:
#         raise ValueError(f"hover_dict #{mode} not found")

#     if not sim and mode > 4:
#         raise RuntimeError("hover modes 5+ not available for hardware")

#     print(f"hover_dict #{mode}")
#     return hover_dict[mode], np.zeros(4)