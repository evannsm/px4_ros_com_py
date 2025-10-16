"""Wrapper classes for controllers to implement ControllerInterface."""

from typing import Optional, Any, Union
import numpy as np
import jax.numpy as jnp

from .control_interface import ControllerInterface, ControlOutput
from .dev_newton_raphson import (
    newton_raphson_standard,
    newton_raphson_enhanced
)
Array = Union[np.ndarray, jnp.ndarray]  # what the protocol accepts


class StandardNRController:
    """Wrapper for standard Newton-Raphson controller."""

    def compute_control(
        self,
        state: Array,
        last_input: Array,
        ref: Array,
        ref_dot: Array | None,
        t_lookahead: float,
        lookahead_step: float,
        integration_step: float,
        mass: float,
        last_alpha: Array | None = None,
        rng: Optional[Any] = None
    ) -> ControlOutput:
        """Compute control using standard Newton-Raphson method."""

        # Call the underlying function
        u, v, best_alpha, best_cost, new_rng = newton_raphson_standard(
            state,
            last_input,
            ref,
            t_lookahead,
            lookahead_step,
            integration_step,
            mass,
            last_alpha,
            rng
        )

        # Normalize output
        return ControlOutput(
            control_input=u,
            metadata={
                'v': v,
                'best_alpha': best_alpha,
                'best_cost': best_cost,
                'rng': new_rng
            }
        )


class EnhancedNRController:
    """Wrapper for enhanced Newton-Raphson controller."""

    def compute_control(
        self,
        state: Array,
        last_input: Array,
        ref: Array,
        ref_dot: Array | None,
        t_lookahead: float,
        lookahead_step: float,
        integration_step: float,
        mass: float,
        last_alpha: Array | None = None,
        rng: Optional[Any] = None
    ) -> ControlOutput:
        """Compute control using enhanced Newton-Raphson method."""

        # Call the underlying function (enhanced version uses ref_dot)
        u, v, best_alpha, best_cost, new_rng = newton_raphson_enhanced(
            state,
            last_input,
            ref,
            t_lookahead,
            lookahead_step,
            integration_step,
            mass,
            ref_dot,
            last_alpha,
            rng
        )

        # Normalize output
        return ControlOutput(
            control_input=u,
            metadata={
                'v': v,
                'best_alpha': best_alpha,
                'best_cost': best_cost,
                'rng': new_rng
            }
        )