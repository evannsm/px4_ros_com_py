"""
Controller Interface Module
============================

This module defines the controller abstraction layer using the Protocol Pattern (duck typing)
and Adapter Pattern to enable seamless switching between different control algorithms without
modifying core flight logic.

Design Philosophy:
-----------------
1. **Protocol over Inheritance**: Use structural typing (Protocol) for flexibility
2. **Adapter Pattern**: Wrap existing controllers to conform to standard interface
3. **Normalized Output**: All controllers return standardized ControlOutput
4. **Metadata Encapsulation**: Controller-specific data lives in metadata dict

Why Protocol instead of ABC?
----------------------------
- More flexible: Any class with the right signature works
- No forced inheritance: Existing controllers can be wrapped
- Duck typing: "If it quacks like a controller, it's a controller"

Benefits:
---------
- Swap controllers without changing control loop code
- Each controller can have unique internal state
- Metadata allows controller-specific diagnostics
- Easy to add new controllers by wrapping existing implementations

Example Usage:
-------------
    # Get controller from registry
    controller = CTRL_REGISTRY[ControllerType.NR_STANDARD]

    # All controllers have same interface
    output = controller.compute_control(
        state=state, ref=ref, ...
    )

    # Access control input (always same structure)
    throttle = output.control_input[0]

    # Access controller-specific metadata
    if 'best_alpha' in output.metadata:
        alpha = output.metadata['best_alpha']
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Dict, Callable, Protocol, Any, Optional
import jax.numpy as jnp
import numpy as np

from .dev_newton_raphson import (
    newton_raphson_standard,
    newton_raphson_enhanced
)


@dataclass
class ControlOutput:
    """Standardized output structure from any controller.

    This dataclass enforces a consistent return format across all controllers,
    enabling the main control loop to handle any controller uniformly.

    Design Rationale:
    ----------------
    Different controllers may compute different auxiliary information:
    - NR Standard: alpha parameters, cost, random state
    - NR Enhanced: CBF terms, error terms, Jacobians
    - Future PID: integral terms, derivative estimates

    Instead of having different return signatures (which would break polymorphism),
    we use a common structure with a flexible metadata dictionary.

    Attributes:
        control_input: The computed control [thrust, roll_rate, pitch_rate, yaw_rate]
                      - Always a 4-element array in this order
                      - Thrust in Newtons
                      - Rates in rad/s
        metadata: Dictionary containing controller-specific diagnostic information
                 - Keys and values depend on controller implementation
                 - Examples: 'best_alpha', 'cost', 'cbf_term', 'rng'
                 - Optional data that doesn't affect the control input

    Example:
        >>> output = ControlOutput(
        ...     control_input=jnp.array([20.0, 0.1, 0.2, 0.0]),
        ...     metadata={'cost': 0.05, 'alpha': jnp.array([...])}
        ... )
        >>> thrust = output.control_input[0]
        >>> if 'cost' in output.metadata:
        ...     print(f"Optimization cost: {output.metadata['cost']}")
    """
    control_input: jnp.ndarray  # [thrust (N), roll_rate (rad/s), pitch_rate (rad/s), yaw_rate (rad/s)]
    metadata: Dict[str, Any]  # Controller-specific diagnostic data


class ControllerInterface(Protocol):
    """Protocol defining the interface all controllers must implement.

    This is a Protocol (PEP 544 - Structural Subtyping), not an abstract base class.
    Any class that has a method matching this signature is automatically considered
    a ControllerInterface, without explicit inheritance.

    Design Rationale:
    ----------------
    Using Protocol instead of ABC allows:
    1. **No Inheritance Required**: Wrap existing controller functions without modifying them
    2. **Structural Typing**: "If it has compute_control(), it's a controller"
    3. **Flexibility**: Controllers can inherit from other bases if needed
    4. **Simplicity**: Just match the signature, no registration needed

    The compute_control signature is comprehensive to support various controller types:
    - Model Predictive Controllers (MPC): need lookahead, integration steps
    - Optimization-based: need last_alpha for warm-starting
    - Stochastic: need rng for reproducibility
    - Feedforward: need ref_dot for derivative terms
    - All: need state, reference, and platform mass

    Why so many parameters?
    ----------------------
    We could use a single config dict, but explicit parameters provide:
    - Type checking at compile time
    - Clear documentation of what each controller needs
    - IDE autocomplete support
    - Easier to spot missing/wrong arguments
    """

    def compute_control(
        self,
        state: np.ndarray,
        last_input: jnp.ndarray,
        ref: np.ndarray,
        ref_dot: Optional[np.ndarray],
        t_lookahead: float,
        lookahead_step: float,
        integration_step: float,
        mass: float,
        last_alpha: Optional[jnp.ndarray] = None,
        rng: Optional[Any] = None
    ) -> ControlOutput:
        """Compute control input given current state and reference.

        This method is called at every control loop iteration to compute the
        control input (thrust and body rates) needed to track the reference trajectory.

        Args:
            state: Current full state vector [x, y, z, vx, vy, vz, roll, pitch, yaw]
                  - Position in meters (NED frame)
                  - Velocity in m/s (body or inertial, controller-dependent)
                  - Euler angles in radians

            last_input: Previous control input [thrust, p, q, r]
                       - Used for warm-starting optimization
                       - Provides smoothness in control
                       - thrust in Newtons, rates in rad/s

            ref: Reference state [x, y, z, yaw]
                - Target position in meters
                - Target yaw in radians
                - What the controller should track

            ref_dot: Reference derivative [vx, vy, vz, yaw_rate] (optional)
                    - Some controllers (enhanced NR) use this for feedforward
                    - None if controller doesn't need it

            t_lookahead: Prediction horizon in seconds
                        - How far ahead to predict (MPC, NR)
                        - Typically 0.5-2.0 seconds

            lookahead_step: Time step for prediction in seconds
                          - Discretization of prediction horizon
                          - Affects prediction accuracy vs. computation

            integration_step: Integration time step in seconds
                            - For numerical integration in dynamics
                            - Smaller = more accurate but slower

            mass: Platform mass in kg
                 - Needed for dynamics model
                 - Different platforms have different masses

            last_alpha: Previous optimization parameters (optional)
                       - Used by optimization-based controllers (NR)
                       - Warm-start speeds up convergence
                       - Shape depends on controller

            rng: Random number generator state (optional)
                - For stochastic controllers (sampling-based NR)
                - Ensures reproducibility
                - JAX PRNGKey or similar

        Returns:
            ControlOutput containing:
            - control_input: [thrust, roll_rate, pitch_rate, yaw_rate]
            - metadata: Controller-specific diagnostic data

        Example:
            >>> controller = StandardNRController()
            >>> output = controller.compute_control(
            ...     state=np.array([0, 0, -5, 0, 0, 0, 0, 0, 0]),
            ...     last_input=jnp.array([19.6, 0, 0, 0]),
            ...     ref=np.array([2, 2, -5, 0]),
            ...     ref_dot=None,
            ...     t_lookahead=1.2,
            ...     lookahead_step=0.05,
            ...     integration_step=0.01,
            ...     mass=2.0
            ... )
            >>> print(output.control_input)  # [thrust, p, q, r]
        """
        ...


class ControllerType(StrEnum):
    """Enumeration of supported controller types.

    Add new controller types here as you implement them.
    The string values are used in configuration files and logs.
    """
    NR_STANDARD = "nr"
    NR_ENHANCED = "nr_enhanced"
    # Future controllers:
    # PID = "pid"
    # MPC = "mpc"
    # LQR = "lqr"


# ============================================================================
# Controller Registry - Dependency Injection
# ============================================================================

# Import controller adapters/wrappers
from .controller_wrappers import StandardNRController, EnhancedNRController


# Controller registry maps types to instantiated controllers
CTRL_REGISTRY: Dict[str, ControllerInterface] = {
    ControllerType.NR_STANDARD: StandardNRController(),
    ControllerType.NR_ENHANCED: EnhancedNRController()
}
"""
Controller Registry - Dependency Injection Container
===================================================

This registry enables runtime controller selection and follows the Strategy Pattern.

Design Decisions:
----------------
1. **Instantiated controllers**: Registry contains instances, not classes
   - Controllers maintain internal state (JIT compiled functions, caches)
   - Instantiating once at startup amortizes JIT compilation cost
   - If controllers were stateless, we'd store classes instead

2. **String keys**: Using ControllerType enum as keys
   - Type-safe selection
   - Easy to extend
   - Works with configuration files

3. **Protocol typing**: Values are ControllerInterface Protocol
   - No forced inheritance
   - Any object with compute_control() works

Adding a New Controller:
-----------------------
1. Create controller class/wrapper with compute_control() method
2. Ensure it returns ControlOutput
3. Add entry to CTRL_REGISTRY
4. Add type to ControllerType enum

Example:
    class PIDController:
        def compute_control(...) -> ControlOutput:
            # PID logic here
            return ControlOutput(...)

    CTRL_REGISTRY[ControllerType.PID] = PIDController()
"""

# Legacy registry (deprecated, kept for backwards compatibility)
CTRL_FUNC_REGISTRY: Dict[str, Callable] = {
    ControllerType.NR_STANDARD: newton_raphson_standard,
    ControllerType.NR_ENHANCED: newton_raphson_enhanced
}
# Note: CTRL_FUNC_REGISTRY will be removed in a future version
# It maps to raw functions instead of wrapped controllers
# Use CTRL_REGISTRY instead

