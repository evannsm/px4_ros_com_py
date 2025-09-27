import jax.numpy as jnp
from px4_control_utils.jax_utils import jit
from px4_control_utils.controllers.dev_newton_raphson.nr_utils import(
    predict_output,
    get_tracking_error,
    get_jac_pred_x_uinv,
    get_enhanced_error,
    get_inv_jac_pred_u,
    get_integral_cbf,
    dynamics
)

ALPHA = jnp.array([20.0, 30.0, 30.0, 30.0])
USE_CBF: bool = True


@jit
def newton_raphson_enhanced(state, last_input, reference, lookahead_horizon_s, lookahead_stage_dt, integration_dt, mass, reference_rate):
    """Enhanced NR tracker with Jacobian terms and CBF; flat (n,) shapes throughout."""
    y_pred = predict_output(state, last_input, lookahead_horizon_s, lookahead_stage_dt, mass)
    regular_error = get_tracking_error(reference, y_pred) # calculates tracking error
    jacobianX, dgdu_inv = get_jac_pred_x_uinv(state, last_input, lookahead_horizon_s, lookahead_stage_dt, mass)

    regular_term = ALPHA * regular_error
    enhanced_error_term = get_enhanced_error(jacobianX, reference_rate, state, last_input, mass)


    NR = dgdu_inv @ (regular_term + enhanced_error_term)
    v = get_integral_cbf(last_input, NR)
    udot = NR + v if USE_CBF else NR + jnp.zeros_like(NR)
    change_u = udot * integration_dt
    u = last_input + change_u
    return u, v, enhanced_error_term, dgdu_inv, NR
