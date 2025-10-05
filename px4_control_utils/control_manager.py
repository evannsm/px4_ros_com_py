"""
Control Manager Module
======================

This module orchestrates the control loop by coordinating controllers, trajectories,
and platform configurations. It implements the Facade Pattern to provide a simplified
interface to complex control computations.

Design Philosophy:
-----------------
1. **Facade Pattern**: Hide complexity behind simple interface
2. **Composition**: Combine controller + trajectory + platform
3. **State Management**: Maintain control state across iterations
4. **Registry-Based Selection**: Use registries for runtime polymorphism

Why a Control Manager?
---------------------
Without it, the main node would need to:
- Manually manage controller state (last inputs, alpha, rng)
- Coordinate between trajectory and controller
- Handle platform-specific conversions
- Know internal details of each controller

With it, the main node just calls:
- compute_reference() to get trajectory
- compute_control() to get control input

Benefits:
---------
- Simplified main control loop
- Centralized control state management
- Easy to swap controllers/trajectories
- Platform-agnostic control computation
- Testable in isolation from ROS

Example Usage:
-------------
    # Setup
    platform = PLATFORM_REGISTRY[PlatformType.SIM]()
    manager = ControlManager(
        controller_type=ControllerType.NR_STANDARD,
        trajectory_type=TrajectoryType.CIRCLE_HORIZONTAL,
        platform=platform,
        traj_context=TrajContext(sim=True)
    )

    # Control loop
    ref, ref_dot = manager.compute_reference(lookahead_time)
    output = manager.compute_control(state, ref, ref_dot)
    throttle = platform.get_throttle_from_force(output.control_input[0])
"""

from dataclasses import dataclass
from typing import Optional, Any
import numpy as np
import jax.numpy as jnp
from jax.random import PRNGKey

from px4_control_utils.controllers.control_interface import (
    ControllerInterface,
    ControllerType,
    ControlOutput,
    CTRL_REGISTRY
)
from px4_control_utils.trajectories.trajectory_interface import (
    TrajectoryFunc,
    TrajectoryType,
    TRAJ_FUNC_REGISTRY
)
from px4_control_utils.trajectories.trajectory_context import TrajContext
from px4_control_utils.vehicles.platform_interface import PlatformConfig


@dataclass
class ControlParams:
    """Control computation parameters.

    Encapsulates all tuning parameters for control algorithms.
    Using a dataclass makes it easy to pass around configuration
    and modify parameters without changing function signatures.

    Design Rationale:
    ----------------
    - Separate from ControlManager: Configuration vs. logic
    - Dataclass: Immutable configuration with defaults
    - Explicit fields: Type checking and documentation

    Attributes:
        t_lookahead: Prediction horizon for MPC/NR controllers (seconds)
                    - Longer = smoother but more conservative
                    - Typical: 0.5-2.0 seconds

        lookahead_step: Discretization step for prediction (seconds)
                       - Smaller = more accurate but slower
                       - Typical: 0.02-0.1 seconds

        integration_step: Time step for dynamics integration (seconds)
                         - Must be <= lookahead_step
                         - Smaller = more accurate
                         - Typical: 0.01-0.05 seconds

        gravity: Gravitational acceleration (m/s²)
                - Used in dynamics model
                - Default: 9.806 m/s² (Earth standard)
    """
    t_lookahead: float = 1.2
    lookahead_step: float = 0.05
    integration_step: float = 0.01
    gravity: float = 9.806


class ControlManager:
    """Manages control computation using controllers and trajectories.

    This class implements the Facade Pattern, providing a simplified interface
    to the complex subsystem of controllers, trajectories, and platform configs.
    It also manages stateful control information across iterations.

    Responsibilities:
    ----------------
    1. Select controller and trajectory from registries
    2. Maintain control state (last inputs, optimization parameters)
    3. Coordinate trajectory generation and control computation
    4. Abstract platform details from control algorithms

    Design Patterns Used:
    --------------------
    - **Facade**: Simplify complex subsystem (controller + trajectory + platform)
    - **Strategy**: Controllers and trajectories selected at runtime
    - **Dependency Injection**: Platform, types injected via constructor

    State Management:
    ----------------
    Control algorithms often need state from previous iterations:
    - last_input: For warm-starting and control smoothness
    - last_alpha: For optimization convergence
    - rng: For reproducible stochastic methods

    The manager maintains this state, freeing the main loop from that burden.

    Why use registries?
    ------------------
    Instead of:
        if controller_type == "nr_standard":
            controller = StandardNRController()

    We do:
        controller = CTRL_REGISTRY[controller_type]

    This follows Open/Closed Principle: add new controllers without modifying this code.
    """

    def __init__(
        self,
        controller_type: ControllerType,
        trajectory_type: TrajectoryType,
        platform: PlatformConfig,
        traj_context: TrajContext,
        params: ControlParams = None
    ):
        """Initialize the control manager.

        Sets up the control system by:
        1. Selecting controller and trajectory from registries
        2. Initializing control state for warm-starting
        3. Configuring control parameters

        Args:
            controller_type: Type of controller to use (e.g., NR_STANDARD)
                           - Used to lookup controller in CTRL_REGISTRY
                           - Enables runtime controller selection

            trajectory_type: Type of trajectory to track (e.g., CIRCLE_HORIZONTAL)
                           - Used to lookup trajectory in TRAJ_FUNC_REGISTRY
                           - Enables runtime trajectory selection

            platform: Platform configuration instance
                     - Provides mass, thrust conversion, etc.
                     - Platform-specific details abstracted away

            traj_context: Trajectory context with configuration
                         - Contains trajectory-specific settings (spin, speed, etc.)
                         - Passed to trajectory functions

            params: Control parameters (lookahead, step sizes, gravity)
                   - If None, uses ControlParams() defaults
                   - Allows easy parameter tuning

        Design Note:
        -----------
        We use dependency injection (passing in platform, types) rather than
        creating instances internally. This makes the code:
        - More testable (can inject mocks)
        - More flexible (caller controls configuration)
        - More explicit (dependencies are visible)
        """
        self.controller_type = controller_type
        self.trajectory_type = trajectory_type
        self.platform = platform
        self.traj_context = traj_context
        self.params = params or ControlParams()

        # Get controller and trajectory from registries (Strategy Pattern)
        self.controller: ControllerInterface = CTRL_REGISTRY[controller_type]
        self.trajectory: TrajectoryFunc = TRAJ_FUNC_REGISTRY[trajectory_type]

        # Initialize control state for warm-starting
        # Controllers often converge faster when starting from a good guess
        first_thrust = platform.mass * self.params.gravity  # Hover thrust
        self._last_input = jnp.array([first_thrust, 0.1, 0.2, 0.3])
        self._last_alpha: Optional[jnp.ndarray] = jnp.array([20.0, 30.0, 30.0, 30.0])
        self._rng = PRNGKey(0)  # Reproducible randomness

        # Trajectory tracking state
        self._start_time: float = 0.0
        self._trajectory_started: bool = False

    def start_trajectory(self, time: float) -> None:
        """Mark the start of trajectory tracking.

        Called when entering the trajectory phase to establish a time reference.

        Args:
            time: Absolute time when trajectory starts (e.g., time.time())

        Note: This is separate from FlightPhaseManager because the control
              manager needs its own time reference for trajectory evaluation.
        """
        self._start_time = time
        self._trajectory_started = True

    @property
    def trajectory_time(self) -> float:
        """Get elapsed time since trajectory start.

        Returns:
            float: Seconds since start_trajectory() was called

        Note: Used for trajectory evaluation and logging, not phase management.
        """
        return self._start_time

    def compute_reference(self, lookahead_time: float) -> tuple:
        """Compute reference trajectory at given time.

        Calls the selected trajectory function with appropriate context
        to generate reference position/yaw and derivatives.

        Args:
            lookahead_time: Time to evaluate trajectory at (seconds)
                          - For MPC/NR: current_time + t_lookahead
                          - Enables predictive control

        Returns:
            tuple: (reference, reference_derivative)
            - reference: [x, y, z, yaw] in meters and radians
            - reference_derivative: [vx, vy, vz, yaw_rate] in m/s and rad/s
              (may be None for non-differentiable trajectories)

        Example:
            >>> # At t=10s, look ahead 1.2s
            >>> ref, ref_dot = manager.compute_reference(10 + 1.2)
            >>> target_pos = ref[:3]  # [x, y, z]
            >>> target_yaw = ref[3]
        """
        return self.trajectory(lookahead_time, self.traj_context)

    def compute_control(
        self,
        state: np.ndarray,
        ref: np.ndarray,
        ref_dot: Optional[np.ndarray] = None
    ) -> ControlOutput:
        """Compute control input using the selected controller.

        This is the main control computation method. It:
        1. Calls the controller with all necessary inputs
        2. Extracts and updates internal state for next iteration
        3. Returns standardized control output

        Args:
            state: Current full state [x, y, z, vx, vy, vz, roll, pitch, yaw]
                  - Position, velocity, and attitude
                  - From vehicle state estimation

            ref: Reference state to track [x, y, z, yaw]
                - Target position and yaw
                - From compute_reference()

            ref_dot: Reference derivative [vx, vy, vz, yaw_rate] (optional)
                    - Target velocities
                    - Used by feedforward controllers
                    - Can be None for controllers that don't need it

        Returns:
            ControlOutput: Standardized control output containing:
                - control_input: [thrust, roll_rate, pitch_rate, yaw_rate]
                - metadata: Controller-specific diagnostic data

        State Management:
        ----------------
        The manager automatically updates internal state from controller metadata:
        - _last_input: For next iteration warm-start
        - _last_alpha: For optimization convergence
        - _rng: For stochastic reproducibility

        This hidden state management is key to why we need a manager class.

        Example:
            >>> state = np.array([0, 0, -5, 0, 0, 0, 0, 0, 0])
            >>> ref = np.array([2, 2, -5, 0])
            >>> output = manager.compute_control(state, ref)
            >>> thrust = output.control_input[0]
            >>> roll_rate = output.control_input[1]
        """
        # Delegate to controller (Strategy Pattern)
        output = self.controller.compute_control(
            state=state,
            last_input=self._last_input,
            ref=ref,
            ref_dot=ref_dot,
            t_lookahead=self.params.t_lookahead,
            lookahead_step=self.params.lookahead_step,
            integration_step=self.params.integration_step,
            mass=self.platform.mass,
            last_alpha=self._last_alpha,
            rng=self._rng
        )

        # Update internal state from controller metadata
        # This enables warm-starting and stateful optimization
        self._last_input = output.control_input
        if 'best_alpha' in output.metadata:
            self._last_alpha = output.metadata['best_alpha']
        if 'rng' in output.metadata:
            self._rng = output.metadata['rng']

        return output
