import jax
from jax import lax
import jax.numpy as jnp

from px4_control_utils.jax_utils import jit
from px4_control_utils.controllers.dev_newton_raphson.nr_utils import predict_output, get_tracking_error, get_inv_jac_pred_u, get_integral_cbf
from px4_control_utils.controllers.dev_newton_raphson.nr_utils import do_sampling

ALPHA = jnp.array([20.0, 30.0, 30.0, 30.0])
USE_CBF: bool = True
USE_SAMPLING: bool = False

# @jit
# def newton_raphson_standard(state, last_input, reference, lookahead_horizon_s, lookahead_stage_dt, integration_dt, mass):
#     """Standard Newton-Raphson method to track the reference trajectory with forward euler integration of dynamics for prediction."""
#     y_pred = predict_output(state, last_input, lookahead_horizon_s, lookahead_stage_dt, mass)
#     error = get_tracking_error(reference, y_pred) # calculates tracking error
#     dgdu_inv = get_inv_jac_pred_u(state, last_input, lookahead_horizon_s, lookahead_stage_dt, mass)

#     NR = dgdu_inv @ error
#     v = get_integral_cbf(last_input, NR)
#     udot = NR + v if USE_CBF else NR + jnp.zeros_like(NR)
#     change_u = udot * integration_dt
#     u = last_input + ALPHA * change_u
#     return u, v, dgdu_inv, NR



@jit
def newton_raphson_standard(state,
                            last_input,
                            reference,
                            lookahead_horizon_s,
                            lookahead_stage_dt,
                            integration_dt,
                            mass,
                            last_alpha,
                            rng):
    # Baseline NR pieces
    y_pred = predict_output(state, last_input, lookahead_horizon_s, lookahead_stage_dt, mass)
    error = get_tracking_error(reference, y_pred)
    dgdu_inv = get_inv_jac_pred_u(state, last_input, lookahead_horizon_s, lookahead_stage_dt, mass)
    NR = dgdu_inv @ error
    v = get_integral_cbf(last_input, NR) if USE_CBF else jnp.zeros_like(NR)
    udot = NR + v
    change_u = udot * integration_dt  # (4,)

    # Use fixed ALPHA (no sampling)
    u = last_input + ALPHA * change_u
    # Return dummy values for consistency with sampling version
    # dummy_cost = jnp.array(0.0)
    return u, v#, ALPHA, dummy_cost, rng
