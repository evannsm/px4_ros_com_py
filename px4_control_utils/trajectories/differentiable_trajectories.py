import jax.numpy as jnp
from jax import jacfwd
from typing import Dict
from px4_control_utils.jax_utils import jit
from px4_control_utils.trajectories.trajectory_context import TrajContext


@jit(static_argnames=("ctx",))
def hover(t_traj: float, ctx: TrajContext):
    """Returns constant hover reference trajectories at a few positions."""
    mode = ctx.hover_mode
    sim = ctx.sim

    hover_dict: Dict[int, jnp.ndarray] = {
        1: jnp.array([0.0, 0.0, -0.9, 0.0]),
        2: jnp.array([0.0, 0.8, -0.8, 0.0]),
        3: jnp.array([0.8, 0.0, -0.8, 0.0]),
        4: jnp.array([0.8, 0.8, -0.8, 0.0]),
        5: jnp.array([0.0, 0.0, -10.0, 0.0]),
        6: jnp.array([1.0, 1.0, -4.0, 0.0]),
        7: jnp.array([3.0, 4.0, -5.0, 0.0]),
        8: jnp.array([1.0, 1.0, -3.0, 0.0]),
        9: jnp.array([0.0, 0.0, -(0.8*jnp.sin((2*jnp.pi/5)*t_traj)+1.0), 0.0]),
    }

    if mode not in hover_dict:
        raise ValueError(f"hover_dict #{mode} not found")

    if not sim and mode > 4:
        raise RuntimeError("hover modes 5+ not available for hardware")

    return hover_dict[mode], jnp.zeros(4)


@jit(static_argnames=("ctx",))
def circle_horizontal(t_traj: float, ctx: TrajContext) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns a horizontal circle trajectory."""
    radius = 0.6
    period_pos = 13.0
    height = 0.7 if not ctx.sim else 5.0
    period_spin = 20.0 

    omega_spin = 2 * jnp.pi / period_spin if ctx.spin else 0.0
    if ctx.double_speed:
        period_pos /= 2.0
    omega_pos = 2 * jnp.pi / period_pos

    def get_trajectory(t: float) -> jnp.ndarray:
        x = radius*jnp.cos(omega_pos*t)
        y = radius*jnp.sin(omega_pos*t)
        z = -height
        yaw = omega_spin*t
        return jnp.array([x, y, z, yaw], dtype=jnp.float64)

    pos = get_trajectory(t_traj)
    vel = jacfwd(get_trajectory)(t_traj)

    return pos, vel


@jit
def circle_vertical(t_traj: float, ctx: TrajContext) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns a vertical circle trajectory."""
    radius = 0.35
    period_pos = 13.0
    height = 0.75 if not ctx.sim else 5.0
    period_spin = 20.0 

    omega_spin = 2 * jnp.pi / period_spin if ctx.spin else 0.0
    if ctx.double_speed:
        period_pos /= 2.0
    omega_pos = 2 * jnp.pi / period_pos

    def get_trajectory(t: float) -> jnp.ndarray:
        x = radius*jnp.cos(omega_pos*t)
        y = 0.0
        z = -radius*jnp.sin(omega_pos*t) - height
        yaw = omega_spin*t
        return jnp.array([x, y, z, yaw], dtype=jnp.float64)

    pos = get_trajectory(t_traj)
    vel = jacfwd(get_trajectory)(t_traj)

    return pos, vel

@jit
def fig8_horizontal(t_traj: float, ctx: TrajContext) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns a horizontal figure-8 trajectory."""
    radius = 0.35
    period_pos = 13.0
    height = 0.8 if not ctx.sim else 5.0
    period_spin = 20.0

    omega_spin = 2 * jnp.pi / period_spin if ctx.spin else 0.0
    if ctx.double_speed:
        period_pos /= 2.0
    omega_pos = 2 * jnp.pi / period_pos


    def get_trajectory(t: float) -> jnp.ndarray:
        x = radius * jnp.sin(2 * omega_pos * t)
        y = radius * jnp.sin(omega_pos * t)
        z = -height
        yaw = omega_spin*t
        return jnp.array([x, y, z, yaw], dtype=jnp.float64)

    pos = get_trajectory(t_traj)
    vel = jacfwd(get_trajectory)(t_traj)

    return pos, vel



@jit
def fig8_vertical(t_traj: float, ctx: TrajContext) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns a vertical figure-8 trajectory."""
    radius = 0.35
    period_pos = 13.0
    height = 0.8 if not ctx.sim else 5.0
    period_spin = 20.0

    omega_spin = 2 * jnp.pi / period_spin if ctx.spin else 0.0
    if ctx.double_speed:
        period_pos /= 2.0
    omega_pos = 2 * jnp.pi / period_pos


    def get_trajectory(t: float) -> jnp.ndarray:
        x = 0.0

        yz1 = radius * jnp.sin(omega_pos * t)
        yz2 = radius * jnp.sin(2 * omega_pos * t)
        y = yz1 if ctx.short else yz2
        z = -yz2 - height if ctx.short else -yz1 - height

        yaw = omega_spin*t
        return jnp.array([x, y, z, yaw], dtype=jnp.float64)

    pos = get_trajectory(t_traj)
    vel = jacfwd(get_trajectory)(t_traj)

    return pos, vel



#TODO: do helixes, sawtooth, and triangle