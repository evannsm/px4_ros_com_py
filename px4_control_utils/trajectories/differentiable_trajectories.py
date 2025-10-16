import jax.numpy as jnp
from jax import jacfwd
from typing import Dict
from px4_control_utils.jax_utils import jit
from px4_control_utils.trajectories.trajectory_context import TrajContext

# Default height settings for trajectories
SIM_HEIGHT = 5.0
HARDWARE_HEIGHT = 0.85


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

@jit
def yawing_only(t_traj: float, ctx: TrajContext) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns stationary yawing reference trajectory."""
    height = HARDWARE_HEIGHT if not ctx.sim else SIM_HEIGHT
    spin_period = 20.0

    if ctx.double_speed:
        spin_period /= 2.0

    def get_trajectory(t: float) -> jnp.ndarray:
        x = 0.0
        y = 0.0
        z = -height
        yaw = t / (spin_period / (2 * jnp.pi))
        return jnp.array([x, y, z, yaw], dtype=jnp.float64)

    pos = get_trajectory(t_traj)
    vel = jacfwd(get_trajectory)(t_traj)

    return pos, vel



@jit(static_argnames=("ctx",))
def circle_horizontal(t_traj: float, ctx: TrajContext) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns a horizontal circle trajectory."""
    radius = 0.6
    period_pos = 13.0
    height = HARDWARE_HEIGHT if not ctx.sim else SIM_HEIGHT
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
    height = HARDWARE_HEIGHT if not ctx.sim else SIM_HEIGHT
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
    height = HARDWARE_HEIGHT if not ctx.sim else SIM_HEIGHT
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
    height = HARDWARE_HEIGHT if not ctx.sim else SIM_HEIGHT
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



@jit
def helix(t_traj: float, ctx: TrajContext) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns a helix trajectory that spirals up and down."""
    z0 = HARDWARE_HEIGHT if not ctx.sim else 2.0
    z_max = 2.6 if not ctx.sim else SIM_HEIGHT
    radius = 0.6
    num_turns = 3
    cycle_time = 30.0
    period_spin = 20.0

    omega_spin = 2 * jnp.pi / period_spin if ctx.spin else 0.0

    if ctx.double_speed:
        cycle_time /= 2.0

    def get_trajectory(t: float) -> jnp.ndarray:
        t_cycle = t % cycle_time
        T_half = cycle_time / 2.0

        # Use jnp.where for differentiability instead of if/else
        # Going up branch
        z_up = z0 + (z_max - z0) * (t_cycle / T_half)
        progress_up = (z_up - z0) / (z_max - z0)

        # Going down branch
        t_down = t_cycle - T_half
        z_down = z_max - (z_max - z0) * (t_down / T_half)
        progress_down = (z_down - z0) / (z_max - z0)

        # Select based on condition using jnp.where for differentiability
        z = jnp.where(t_cycle <= T_half, z_up, z_down)
        progress = jnp.where(t_cycle <= T_half, progress_up, progress_down)

        # Angle is based on progress through the cycle
        theta = 2 * jnp.pi * num_turns * progress
        x = radius * jnp.cos(theta)
        y = radius * jnp.sin(theta)
        yaw = omega_spin * t

        return jnp.array([x, y, -z, yaw], dtype=jnp.float64)

    pos = get_trajectory(t_traj)
    vel = jacfwd(get_trajectory)(t_traj)

    return pos, vel


@jit
def sawtooth(t_traj: float, ctx: TrajContext) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns a sawtooth pattern reference trajectory."""
    height = HARDWARE_HEIGHT if not ctx.sim else SIM_HEIGHT
    flight_time = 60.0
    num_repeats = 2 if ctx.double_speed else 1
    period_spin = 20.0

    omega_spin = 2 * jnp.pi / period_spin if ctx.spin else 0.0

    # Define the waypoints for the sawtooth trajectory
    points = jnp.array([
        [0.0, 0.0], [0.0, 0.4], [0.4, -0.4], [0.4, 0.4], [0.4, -0.4],
        [0.0, 0.4], [0.0, -0.4], [-0.4, 0.4], [-0.4, -0.4],
        [-0.4, 0.4], [0.0, -0.4], [0.0, 0.0]
    ], dtype=jnp.float64)

    def get_trajectory(t: float) -> jnp.ndarray:
        # Adjust flight time based on number of repetitions
        adjusted_flight_time = flight_time / num_repeats
        num_segments = len(points) - 1
        T_seg = adjusted_flight_time / num_segments

        # Calculate time within current cycle
        cycle_time = t % (num_segments * T_seg)

        # Determine segment index (continuous)
        segment_idx = jnp.floor(cycle_time / T_seg)
        segment_idx = jnp.clip(segment_idx, 0, num_segments - 1).astype(int)

        # Time within the current segment
        local_time = cycle_time - segment_idx * T_seg

        # Linear interpolation
        start_point = points[segment_idx]
        end_point = points[(segment_idx + 1) % len(points)]

        alpha = local_time / T_seg
        x = start_point[0] + (end_point[0] - start_point[0]) * alpha
        y = start_point[1] + (end_point[1] - start_point[1]) * alpha
        z = -height
        yaw = omega_spin * t

        return jnp.array([x, y, z, yaw], dtype=jnp.float64)

    pos = get_trajectory(t_traj)
    vel = jacfwd(get_trajectory)(t_traj)

    return pos, vel


@jit
def triangle(t_traj: float, ctx: TrajContext) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns an equilateral triangle reference trajectory."""
    height = HARDWARE_HEIGHT if not ctx.sim else SIM_HEIGHT
    side_length = 0.6
    flight_time = 60.0
    num_repeats = 2 if ctx.double_speed else 1
    period_spin = 20.0

    omega_spin = 2 * jnp.pi / period_spin if ctx.spin else 0.0

    # Define triangle vertices
    h = jnp.sqrt(side_length**2 - (side_length/2)**2)
    points = jnp.array([
        [0.0, h/2],
        [side_length/2, -h/2],
        [-side_length/2, -h/2]
    ], dtype=jnp.float64)

    def get_trajectory(t: float) -> jnp.ndarray:
        # Calculate segment time
        T_seg = flight_time / (3 * num_repeats)

        # Calculate time within current cycle
        cycle_time = t % (3 * T_seg)

        # Determine segment index
        segment_idx = jnp.floor(cycle_time / T_seg)
        segment_idx = jnp.clip(segment_idx, 0, 2).astype(int)

        # Time within the current segment
        local_time = cycle_time - segment_idx * T_seg

        # Linear interpolation
        start_point = points[segment_idx]
        end_point = points[(segment_idx + 1) % 3]

        alpha = local_time / T_seg
        x = start_point[0] + (end_point[0] - start_point[0]) * alpha
        y = start_point[1] + (end_point[1] - start_point[1]) * alpha
        z = -height
        yaw = omega_spin * t

        return jnp.array([x, y, z, yaw], dtype=jnp.float64)

    pos = get_trajectory(t_traj)
    vel = jacfwd(get_trajectory)(t_traj)

    return pos, vel