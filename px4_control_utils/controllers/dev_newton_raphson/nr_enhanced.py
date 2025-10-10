import jax.numpy as jnp
from px4_control_utils.jax_utils import jit
from px4_control_utils.controllers.dev_newton_raphson.nr_utils import do_sampling
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
USE_SAMPLING: bool = False


@jit
def newton_raphson_enhanced(state,
                            last_input,
                            reference,
                            lookahead_horizon_s,
                            lookahead_stage_dt,
                            integration_dt, mass,
                            reference_rate,
                            last_alpha,
                            rng):
    """Enhanced NR tracker with Jacobian terms and CBF; flat (n,) shapes throughout."""
    y_pred = predict_output(state, last_input, lookahead_horizon_s, lookahead_stage_dt, mass)
    regular_error = get_tracking_error(reference, y_pred) # calculates tracking error
    jacobianX, dgdu_inv = get_jac_pred_x_uinv(state, last_input, lookahead_horizon_s, lookahead_stage_dt, mass)

    if USE_SAMPLING:
        # Cost wrt alpha
        def cost_fn(alpha):
            regular_term = alpha * regular_error
            enhanced_error_term = get_enhanced_error(jacobianX, reference_rate, state, last_input, mass)
            NR = dgdu_inv @ (regular_term + enhanced_error_term)
            v = get_integral_cbf(last_input, NR)
            udot = NR + v if USE_CBF else NR + jnp.zeros_like(NR)
            change_u = udot * integration_dt
            u_alpha = last_input + change_u
            y_pred_alpha = predict_output(state, u_alpha, lookahead_horizon_s, lookahead_stage_dt, mass)
            err_alpha = get_tracking_error(reference, y_pred_alpha)
            return 0.5 * jnp.dot(err_alpha, err_alpha)  # scalar

        # RNG-threaded sampling step
        alpha_new, rng_out = do_sampling(cost_fn, last_alpha, rng,
                                         num_samples=16, noise_scale=5.0, lr=1e-2)
        # Optional clipping
        alpha_new = jnp.clip(alpha_new, 10.0, 1_000.0)

        # Apply
        regular_term = alpha_new * regular_error
        enhanced_error_term = get_enhanced_error(jacobianX, reference_rate, state, last_input, mass)
        NR = dgdu_inv @ (regular_term + enhanced_error_term)
        v = get_integral_cbf(last_input, NR)
        udot = NR + v if USE_CBF else NR + jnp.zeros_like(NR)
        change_u = udot * integration_dt
        u = last_input + change_u

        cost_new = cost_fn(alpha_new)
        return u, v, alpha_new, cost_new, rng_out
    else:
        regular_term = ALPHA * regular_error
        enhanced_error_term = get_enhanced_error(jacobianX, reference_rate, state, last_input, mass)
        NR = dgdu_inv @ (regular_term + enhanced_error_term)
        v = get_integral_cbf(last_input, NR)
        udot = NR + v if USE_CBF else NR + jnp.zeros_like(NR)
        change_u = udot * integration_dt
        u = last_input + change_u
        dummy_cost = jnp.array(0.0)
        return u, v, ALPHA, dummy_cost, rng
