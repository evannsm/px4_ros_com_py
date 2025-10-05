"""Wrapper classes for controllers to implement ControllerInterface."""

from typing import Optional, Any
import numpy as np
import jax.numpy as jnp

from .control_interface import ControllerInterface, ControlOutput
from .dev_newton_raphson import (
    newton_raphson_standard,
    newton_raphson_enhanced
)


class StandardNRController:
    """Wrapper for standard Newton-Raphson controller."""

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
        """Compute control using enhanced Newton-Raphson method."""

        # Call the underlying function (enhanced version uses ref_dot)
        u, cbf_term, enhanced_error_term, dgdu_inv, NR = newton_raphson_enhanced(
            state,
            last_input,
            ref,
            t_lookahead,
            lookahead_step,
            integration_step,
            mass,
            ref_dot
        )

        # Normalize output
        return ControlOutput(
            control_input=u,
            metadata={
                'cbf_term': cbf_term,
                'enhanced_error_term': enhanced_error_term,
                'dgdu_inv': dgdu_inv,
                'NR': NR
            }
        )
