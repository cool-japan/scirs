//! Adaptive ODE solver methods
//!
//! This module implements adaptive methods for solving ODEs,
//! including Dormand-Prince (RK45), Bogacki-Shampine (RK23),
//! and Dormand-Prince 8th order (DOP853) methods.

use crate::common::IntegrateFloat;
use crate::error::IntegrateResult;
use crate::ode::types::{ODEMethod, ODEOptions, ODEResult};
use scirs2_core::ndarray::{Array1, ArrayView1};

/// Solve ODE using the Dormand-Prince method (RK45)
///
/// This is an adaptive step size method based on embedded Runge-Kutta formulas.
/// It uses a 5th-order method with a 4th-order error estimate.
///
/// # Arguments
///
/// * `f` - ODE function dy/dt = f(t, y)
/// * `t_span` - Time span [t_start, t_end]
/// * `y0` - Initial condition
/// * `opts` - Solver options
///
/// # Returns
///
/// The solution as an ODEResult or an error
#[allow(dead_code)]
pub fn rk45_method<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Initialize
    let [t_start, t_end] = t_span;
    let n_dim = y0.len();

    // Determine initial step size if not provided
    let h0 = opts.h0.unwrap_or_else(|| {
        // Simple heuristic for initial step size
        let _span = t_end - t_start;
        _span / F::from_usize(100).expect("Operation failed")
    });

    // Determine minimum and maximum step sizes
    let min_step = opts.min_step.unwrap_or_else(|| {
        let _span = t_end - t_start;
        _span * F::from_f64(1e-8).expect("Operation failed") // Minimal step size
    });

    let max_step = opts.max_step.unwrap_or_else(|| {
        t_end - t_start // Maximum step can be the whole interval
    });

    // Current state
    let mut t = t_start;
    let mut y = y0.clone();
    let mut h = h0;

    // Storage for results
    let mut t_values = vec![t_start];
    let mut y_values = vec![y0.clone()];

    // Statistics
    let mut func_evals = 0;
    let mut step_count = 0;
    let mut accepted_steps = 0;
    let mut rejected_steps = 0;

    // Dormand-Prince coefficients
    // Time steps
    let c2 = F::from_f64(1.0 / 5.0).expect("Operation failed");
    let c3 = F::from_f64(3.0 / 10.0).expect("Operation failed");
    let c4 = F::from_f64(4.0 / 5.0).expect("Operation failed");
    let c5 = F::from_f64(8.0 / 9.0).expect("Operation failed");
    let c6 = F::one();

    // Main integration loop
    while t < t_end && step_count < opts.max_steps {
        // Adjust step size for the last step if needed
        if t + h > t_end {
            h = t_end - t;
        }

        // Limit step size to bounds
        h = h.min(max_step).max(min_step);

        // Runge-Kutta stages
        let k1 = f(t, y.view());

        // Manually compute the stages to avoid type mismatches
        let mut y_stage = y.clone();
        for i in 0..n_dim {
            y_stage[i] = y[i] + h * F::from_f64(1.0 / 5.0).expect("Operation failed") * k1[i];
        }
        let k2 = f(t + c2 * h, y_stage.view());

        let mut y_stage = y.clone();
        for i in 0..n_dim {
            y_stage[i] = y[i]
                + h * (F::from_f64(3.0 / 40.0).expect("Operation failed") * k1[i]
                    + F::from_f64(9.0 / 40.0).expect("Operation failed") * k2[i]);
        }
        let k3 = f(t + c3 * h, y_stage.view());

        let mut y_stage = y.clone();
        for i in 0..n_dim {
            y_stage[i] = y[i]
                + h * (F::from_f64(44.0 / 45.0).expect("Operation failed") * k1[i]
                    + F::from_f64(-56.0 / 15.0).expect("Operation failed") * k2[i]
                    + F::from_f64(32.0 / 9.0).expect("Operation failed") * k3[i]);
        }
        let k4 = f(t + c4 * h, y_stage.view());

        let mut y_stage = y.clone();
        for i in 0..n_dim {
            y_stage[i] = y[i]
                + h * (F::from_f64(19372.0 / 6561.0).expect("Operation failed") * k1[i]
                    + F::from_f64(-25360.0 / 2187.0).expect("Operation failed") * k2[i]
                    + F::from_f64(64448.0 / 6561.0).expect("Operation failed") * k3[i]
                    + F::from_f64(-212.0 / 729.0).expect("Operation failed") * k4[i]);
        }
        let k5 = f(t + c5 * h, y_stage.view());

        let mut y_stage = y.clone();
        for i in 0..n_dim {
            y_stage[i] = y[i]
                + h * (F::from_f64(9017.0 / 3168.0).expect("Operation failed") * k1[i]
                    + F::from_f64(-355.0 / 33.0).expect("Operation failed") * k2[i]
                    + F::from_f64(46732.0 / 5247.0).expect("Operation failed") * k3[i]
                    + F::from_f64(49.0 / 176.0).expect("Operation failed") * k4[i]
                    + F::from_f64(-5103.0 / 18656.0).expect("Operation failed") * k5[i]);
        }
        let k6 = f(t + c6 * h, y_stage.view());

        let mut y_stage = y.clone();
        for i in 0..n_dim {
            y_stage[i] = y[i]
                + h * (F::from_f64(35.0 / 384.0).expect("Operation failed") * k1[i]
                    + F::zero() * k2[i]
                    + F::from_f64(500.0 / 1113.0).expect("Operation failed") * k3[i]
                    + F::from_f64(125.0 / 192.0).expect("Operation failed") * k4[i]
                    + F::from_f64(-2187.0 / 6784.0).expect("Operation failed") * k5[i]
                    + F::from_f64(11.0 / 84.0).expect("Operation failed") * k6[i]);
        }
        let k7 = f(t + h, y_stage.view());

        func_evals += 7;

        // 5th order solution
        let mut y5 = y.clone();
        for i in 0..n_dim {
            y5[i] = y[i]
                + h * (F::from_f64(35.0 / 384.0).expect("Operation failed") * k1[i]
                    + F::zero() * k2[i]
                    + F::from_f64(500.0 / 1113.0).expect("Operation failed") * k3[i]
                    + F::from_f64(125.0 / 192.0).expect("Operation failed") * k4[i]
                    + F::from_f64(-2187.0 / 6784.0).expect("Operation failed") * k5[i]
                    + F::from_f64(11.0 / 84.0).expect("Operation failed") * k6[i]
                    + F::zero() * k7[i]);
        }

        // 4th order solution
        let mut y4 = y.clone();
        for i in 0..n_dim {
            y4[i] = y[i]
                + h * (F::from_f64(5179.0 / 57600.0).expect("Operation failed") * k1[i]
                    + F::zero() * k2[i]
                    + F::from_f64(7571.0 / 16695.0).expect("Operation failed") * k3[i]
                    + F::from_f64(393.0 / 640.0).expect("Operation failed") * k4[i]
                    + F::from_f64(-92097.0 / 339200.0).expect("Operation failed") * k5[i]
                    + F::from_f64(187.0 / 2100.0).expect("Operation failed") * k6[i]
                    + F::from_f64(1.0 / 40.0).expect("Operation failed") * k7[i]);
        }

        // Error estimation
        let mut err_norm = F::zero();
        for i in 0..n_dim {
            let sc = opts.atol + opts.rtol * y5[i].abs();
            let err = (y5[i] - y4[i]).abs() / sc;
            err_norm = err_norm.max(err);
        }

        // Step size control
        let order = F::from_f64(5.0).expect("Operation failed"); // 5th order method
        let exponent = F::one() / (order + F::one());
        let safety = F::from_f64(0.9).expect("Operation failed");
        let factor = safety * (F::one() / err_norm).powf(exponent);
        let factor_min = F::from_f64(0.2).expect("Operation failed");
        let factor_max = F::from_f64(5.0).expect("Operation failed");
        let factor = factor.min(factor_max).max(factor_min);

        if err_norm <= F::one() {
            // Step accepted
            t += h;
            y = y5; // Use higher order solution

            // Store results
            t_values.push(t);
            y_values.push(y.clone());

            // Increase step size for next step
            if err_norm <= F::from_f64(0.1).expect("Operation failed") {
                // For very accurate steps, try a larger increase
                h *= factor.max(F::from_f64(2.0).expect("Operation failed"));
            } else {
                h *= factor;
            }

            step_count += 1;
            accepted_steps += 1;
        } else {
            // Step rejected
            h *= factor.min(F::one());
            rejected_steps += 1;

            // If step size is too small, return error
            if h < min_step {
                return Err(crate::error::IntegrateError::StepSizeTooSmall(format!(
                    "Step size {h} too small at t {t}"
                )));
            }
        }
    }

    // Check if integration was successful
    let success = t >= t_end;
    let message = if !success {
        Some(format!(
            "Maximum number of steps ({}) reached",
            opts.max_steps
        ))
    } else {
        None
    };

    // Return the solution
    Ok(ODEResult {
        t: t_values,
        y: y_values,
        success,
        message,
        n_eval: func_evals,
        n_steps: step_count,
        n_accepted: accepted_steps,
        n_rejected: rejected_steps,
        n_lu: 0,  // No LU decompositions in explicit methods
        n_jac: 0, // No Jacobian evaluations in explicit methods
        method: ODEMethod::RK45,
    })
}

/// Shared PI (proportional-integral) step-size controller in the style of
/// Hairer, Nørsett & Wanner's "Lund stabilization" (see e.g. the reference
/// `dopri5.f`/`dop853.f` codes and Gustafsson (1991), "Control theoretic
/// techniques for stepsize selection in explicit Runge-Kutta methods",
/// ACM TOMS 17(4)).
///
/// The controller combines the current step's tolerance-normalized local
/// error norm (`err_norm`, where `<= 1` means "accept") with the *previous
/// accepted* step's error norm (`err_prev`) to damp the step-size
/// oscillations that a purely elementary (integral-only) controller can
/// exhibit. `beta_gain = 0` degenerates it to a plain elementary
/// controller (used for DOP853, matching the reference implementation's
/// own default of no Lund stabilization).
///
/// Returns `(step_size_factor, updated_err_prev)`, where `step_size_factor`
/// is the multiplier to apply to `h`. `err_prev` is only advanced when
/// `accepted` is true, mirroring the reference behavior of only updating
/// the controller's memory on accepted steps.
#[allow(clippy::too_many_arguments)]
fn pi_step_factor<F: IntegrateFloat>(
    err_norm: F,
    err_prev: F,
    accepted: bool,
    just_recovered_from_rejection: bool,
    alpha: F,
    beta_gain: F,
    safety: F,
    growth_min: F,
    growth_max: F,
) -> (F, F) {
    let tiny = F::from_f64(1e-300).expect("Operation failed");
    let err_eff = err_norm.max(tiny);
    let fac_elementary = err_eff.powf(alpha);

    if accepted {
        let history = err_prev.max(tiny).powf(beta_gain);
        let mut factor = safety / (fac_elementary * history);
        factor = factor.min(growth_max).max(growth_min);
        if just_recovered_from_rejection {
            // Never grow immediately after recovering from a rejection.
            factor = factor.min(F::one());
        }
        let updated_err_prev = err_norm.max(F::from_f64(1e-10).expect("Operation failed"));
        (factor, updated_err_prev)
    } else {
        // On rejection, ignore step-size history entirely (matches the
        // reference codes) and guarantee an actual shrink.
        let factor = (safety / fac_elementary).min(F::one()).max(growth_min);
        (factor, err_prev)
    }
}

/// Solve ODE using the Bogacki-Shampine method (RK23)
///
/// This is an adaptive step size method based on embedded Runge-Kutta
/// formulas. It uses the Bogacki & Shampine (1989) 3(2) pair: a 3rd-order
/// solution is advanced at every step, with a 2nd-order embedded solution
/// used purely for error estimation (local extrapolation). The method has
/// the "first same as last" (FSAL) property: the 4th stage is evaluated at
/// the accepted 3rd-order solution, so the embedded error estimate comes at
/// the cost of only one extra function evaluation per step. Step sizes are
/// chosen with a PI (proportional-integral) controller.
///
/// # Arguments
///
/// * `f` - ODE function dy/dt = f(t, y)
/// * `t_span` - Time span [t_start, t_end]
/// * `y0` - Initial condition
/// * `opts` - Solver options
///
/// # Returns
///
/// The solution as an ODEResult or an error
///
/// # References
///
/// P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
/// Appl. Math. Lett. Vol. 2, No. 4, pp. 321-325, 1989.
#[allow(dead_code)]
pub fn rk23_method<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Initialize
    let [t_start, t_end] = t_span;
    let n_dim = y0.len();

    // Determine initial step size if not provided
    let h0 = opts.h0.unwrap_or_else(|| {
        // Simple heuristic for initial step size
        let _span = t_end - t_start;
        _span / F::from_usize(100).expect("Operation failed")
    });

    // Determine minimum and maximum step sizes
    let min_step = opts.min_step.unwrap_or_else(|| {
        let _span = t_end - t_start;
        _span * F::from_f64(1e-8).expect("Operation failed") // Minimal step size
    });

    let max_step = opts.max_step.unwrap_or_else(|| {
        t_end - t_start // Maximum step can be the whole interval
    });

    // Current state
    let mut t = t_start;
    let mut y = y0.clone();
    let mut h = h0;

    // Storage for results
    let mut t_values = vec![t_start];
    let mut y_values = vec![y0.clone()];

    // Statistics
    let mut func_evals = 0;
    let mut step_count = 0;
    let mut accepted_steps = 0;
    let mut rejected_steps = 0;

    // Bogacki-Shampine (1989) coefficients.
    let c2 = F::from_f64(0.5).expect("Operation failed");
    let c3 = F::from_f64(0.75).expect("Operation failed");
    let a21 = F::from_f64(0.5).expect("Operation failed");
    let a32 = F::from_f64(0.75).expect("Operation failed");
    let b1 = F::from_f64(2.0 / 9.0).expect("Operation failed");
    let b2 = F::from_f64(1.0 / 3.0).expect("Operation failed");
    let b3 = F::from_f64(4.0 / 9.0).expect("Operation failed");
    // Embedded error-estimator weights: err = h * (e1*k1+e2*k2+e3*k3+e4*k4)
    let e1 = F::from_f64(5.0 / 72.0).expect("Operation failed");
    let e2 = F::from_f64(-1.0 / 12.0).expect("Operation failed");
    let e3 = F::from_f64(-1.0 / 9.0).expect("Operation failed");
    let e4 = F::from_f64(0.125).expect("Operation failed");

    // PI step-size controller parameters. `alpha` is the elementary
    // exponent 1/(error_order+1) with error_order=2 (the embedded
    // 2nd-order solution); `beta_gain` is the PI (history feedback) gain.
    let alpha = F::one() / F::from_f64(3.0).expect("Operation failed");
    let beta_gain = F::from_f64(0.08).expect("Operation failed");
    let safety = F::from_f64(0.9).expect("Operation failed");
    let growth_min = F::from_f64(0.2).expect("Operation failed");
    let growth_max = F::from_f64(10.0).expect("Operation failed");
    let mut err_prev = F::one();
    let mut just_rejected = false;

    // Main integration loop
    while t < t_end && step_count < opts.max_steps {
        // Adjust step size for the last step if needed
        if t + h > t_end {
            h = t_end - t;
        }

        // Limit step size to bounds
        h = h.min(max_step).max(min_step);

        // Stage 1
        let k1 = f(t, y.view());

        // Stage 2
        let mut y_stage = y.clone();
        for i in 0..n_dim {
            y_stage[i] = y[i] + h * a21 * k1[i];
        }
        let k2 = f(t + c2 * h, y_stage.view());

        // Stage 3
        let mut y_stage = y.clone();
        for i in 0..n_dim {
            y_stage[i] = y[i] + h * a32 * k2[i];
        }
        let k3 = f(t + c3 * h, y_stage.view());

        // 3rd order solution (the point at which the FSAL stage is
        // evaluated).
        let mut y3 = y.clone();
        for i in 0..n_dim {
            y3[i] = y[i] + h * (b1 * k1[i] + b2 * k2[i] + b3 * k3[i]);
        }

        // FSAL stage: derivative at the accepted 3rd-order point.
        let k4 = f(t + h, y3.view());
        func_evals += 4;

        // Embedded error estimate (2nd order vs. 3rd order), WRMS-style.
        let mut err_norm = F::zero();
        for i in 0..n_dim {
            let sc = opts.atol + opts.rtol * y3[i].abs().max(y[i].abs());
            let err_i = e1 * k1[i] + e2 * k2[i] + e3 * k3[i] + e4 * k4[i];
            err_norm = err_norm.max((h * err_i / sc).abs());
        }

        let accepted = err_norm <= F::one();
        let (factor, new_err_prev) = pi_step_factor(
            err_norm,
            err_prev,
            accepted,
            just_rejected,
            alpha,
            beta_gain,
            safety,
            growth_min,
            growth_max,
        );

        if accepted {
            // Step accepted
            t += h;
            y = y3;

            // Store results
            t_values.push(t);
            y_values.push(y.clone());

            err_prev = new_err_prev;
            just_rejected = false;
            h *= factor;

            step_count += 1;
            accepted_steps += 1;
        } else {
            // Step rejected
            h *= factor;
            just_rejected = true;
            rejected_steps += 1;

            // If step size is too small, return error
            if h < min_step {
                return Err(crate::error::IntegrateError::StepSizeTooSmall(format!(
                    "Step size {h} too small at t {t}"
                )));
            }
        }
    }

    // Check if integration was successful
    let success = t >= t_end;
    let message = if !success {
        Some(format!(
            "Maximum number of steps ({}) reached",
            opts.max_steps
        ))
    } else {
        None
    };

    // Return the solution
    Ok(ODEResult {
        t: t_values,
        y: y_values,
        success,
        message,
        n_eval: func_evals,
        n_steps: step_count,
        n_accepted: accepted_steps,
        n_rejected: rejected_steps,
        n_lu: 0,  // No LU decompositions in explicit methods
        n_jac: 0, // No Jacobian evaluations in explicit methods
        method: ODEMethod::RK23,
    })
}

/// Number of stages used to advance the DOP853 solution.
///
/// The reference `dop853.f`/`dop853_coefficients.py` tableau has 16 rows:
/// 12 stages for the main 8th/5th/3rd order solution and error estimate,
/// a 13th "first same as last" (FSAL) stage, and 3 more stages used only to
/// build a dense-output (continuous extension) interpolant. Neither error
/// estimator weight vector (`DOP853_E5`, the `E3` derived from
/// `DOP853_B`) has a nonzero coefficient on the FSAL stage, and this
/// crate's ODE solver contract reconstructs dense/continuous output
/// externally (see `ode::utils::dense_output::DenseSolution`, built via
/// cubic Hermite interpolation over the returned samples) rather than
/// through a per-step interpolant -- exactly the same contract already
/// used by `rk45_method` in this file. The FSAL and dense-output-only
/// stages are therefore omitted here as genuinely unneeded evaluations.
const DOP853_STAGES: usize = 12;

/// c_i: stage time fractions for the DOP853 tableau (Hairer, Nørsett &
/// Wanner, "Solving Ordinary Differential Equations I", 2nd ed.).
const DOP853_C: [f64; 12] = [
    0.0,
    0.05260015195876773,
    0.0789002279381516,
    0.1183503419072274,
    0.2816496580927726,
    0.3333333333333333,
    0.25,
    0.3076923076923077,
    0.6512820512820513,
    0.6,
    0.8571428571428571,
    1.0,
];

/// a_{i,j} strictly-lower-triangular stage coefficients. Row `i` (0-indexed)
/// holds the weights on stages `1..=i` used to build stage `i+1`'s
/// argument; row 0 (stage 1, always `f(t, y)`) needs no combination.
const DOP853_A: [[f64; 12]; 12] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [
        0.05260015195876773,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.0197250569845379,
        0.0591751709536137,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.02958758547680685,
        0.0,
        0.08876275643042054,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.2413651341592667,
        0.0,
        -0.8845494793282861,
        0.924834003261792,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.037037037037037035,
        0.0,
        0.0,
        0.17082860872947386,
        0.12546768756682242,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.037109375,
        0.0,
        0.0,
        0.17025221101954405,
        0.06021653898045596,
        -0.017578125,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.03709200011850479,
        0.0,
        0.0,
        0.17038392571223998,
        0.10726203044637328,
        -0.015319437748624402,
        0.008273789163814023,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.6241109587160757,
        0.0,
        0.0,
        -3.3608926294469414,
        -0.868219346841726,
        27.59209969944671,
        20.154067550477894,
        -43.48988418106996,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.47766253643826434,
        0.0,
        0.0,
        -2.4881146199716677,
        -0.590290826836843,
        21.230051448181193,
        15.279233632882423,
        -33.28821096898486,
        -0.020331201708508627,
        0.0,
        0.0,
        0.0,
    ],
    [
        -0.9371424300859873,
        0.0,
        0.0,
        5.186372428844064,
        1.0914373489967295,
        -8.149787010746927,
        -18.52006565999696,
        22.739487099350505,
        2.4936055526796523,
        -3.0467644718982196,
        0.0,
        0.0,
    ],
    [
        2.273310147516538,
        0.0,
        0.0,
        -10.53449546673725,
        -2.0008720582248625,
        -17.9589318631188,
        27.94888452941996,
        -2.8589982771350235,
        -8.87285693353063,
        12.360567175794303,
        0.6433927460157636,
        0.0,
    ],
];

/// b_i: 8th-order solution weights.
const DOP853_B: [f64; 12] = [
    0.054293734116568765,
    0.0,
    0.0,
    0.0,
    0.0,
    4.450312892752409,
    1.8915178993145003,
    -5.801203960010585,
    0.3111643669578199,
    -0.1521609496625161,
    0.20136540080403034,
    0.04471061572777259,
];

/// 5th-order error-estimator weights (`err5 = dot(K, E5)`).
const DOP853_E5: [f64; 12] = [
    0.01312004499419488,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.2251564463762044,
    -0.4957589496572502,
    1.6643771824549864,
    -0.35032884874997366,
    0.3341791187130175,
    0.08192320648511571,
    -0.022355307863886294,
];

/// Tabulated adjustments turning the 8th-order weight vector `B` into the
/// secondary 3rd-order error-check weight vector `E3`: only 3 of the 12
/// stage weights differ between the two (Hairer, Nørsett & Wanner).
const DOP853_E3_ADJUST: [(usize, f64); 3] = [
    (0, 0.2440944881889764),
    (8, 0.7338466882816118),
    (11, 0.022058823529411766),
];

/// Solve ODE using the Dormand-Prince 8th order method (DOP853)
///
/// This is a high-accuracy adaptive step size method based on embedded
/// Runge-Kutta formulas: the full Hairer-Nørsett-Wanner 12-stage 8th-order
/// method, with a blended 5th/3rd-order embedded error estimate. The
/// blend combines a primary 5th-order error estimator with a secondary
/// 3rd-order check so that the estimate stays reliable even in the rare
/// case where the 5th-order estimator alone would (spuriously) nearly
/// vanish. Step sizes are chosen with the same PI-capable controller used
/// by [`rk23_method`], with the PI (history) term disabled by default to
/// match the reference implementation's own default (no Lund
/// stabilization for DOP853 unless explicitly requested).
///
/// # Arguments
///
/// * `f` - ODE function dy/dt = f(t, y)
/// * `t_span` - Time span [t_start, t_end]
/// * `y0` - Initial condition
/// * `opts` - Solver options
///
/// # Returns
///
/// The solution as an ODEResult or an error
///
/// # References
///
/// E. Hairer, S.P. Nørsett, G. Wanner, "Solving Ordinary Differential
/// Equations I: Nonstiff Problems", 2nd ed., Springer, 1993, Section II.10.
#[allow(dead_code)]
pub fn dop853_method<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Initialize
    let [t_start, t_end] = t_span;
    let n_dim = y0.len();

    // Determine initial step size if not provided
    let h0 = opts.h0.unwrap_or_else(|| {
        // Simple heuristic for initial step size
        let _span = t_end - t_start;
        _span / F::from_usize(100).expect("Operation failed")
    });

    // Determine minimum and maximum step sizes
    let min_step = opts.min_step.unwrap_or_else(|| {
        let _span = t_end - t_start;
        _span * F::from_f64(1e-8).expect("Operation failed") // Minimal step size
    });

    let max_step = opts.max_step.unwrap_or_else(|| {
        t_end - t_start // Maximum step can be the whole interval
    });

    // Current state
    let mut t = t_start;
    let mut y = y0.clone();
    let mut h = h0;

    // Storage for results
    let mut t_values = vec![t_start];
    let mut y_values = vec![y0.clone()];

    // Statistics
    let mut func_evals = 0;
    let mut step_count = 0;
    let mut accepted_steps = 0;
    let mut rejected_steps = 0;

    // Secondary 3rd-order check weight vector: E3 = B with 3 tabulated
    // adjustments (Hairer, Nørsett & Wanner).
    let mut e3 = DOP853_B;
    for &(idx, adjust) in DOP853_E3_ADJUST.iter() {
        e3[idx] -= adjust;
    }

    // Step-size controller parameters. `alpha` is the reference
    // implementation's `expo1` with its default `beta = 0` (i.e. no
    // Lund/PI stabilization for DOP853 by default).
    let alpha = F::from_f64(0.125).expect("Operation failed");
    let beta_gain = F::zero();
    let safety = F::from_f64(0.9).expect("Operation failed");
    let growth_min = F::from_f64(1.0 / 3.0).expect("Operation failed");
    let growth_max = F::from_f64(6.0).expect("Operation failed");
    let mut err_prev = F::one();
    let mut just_rejected = false;

    let n_dim_f = F::from_usize(n_dim).expect("Operation failed");
    let err3_weight = F::from_f64(0.01).expect("Operation failed");
    let non_finite_penalty = F::from_f64(1e30).expect("Operation failed");

    // Main integration loop
    while t < t_end && step_count < opts.max_steps {
        // Adjust step size for the last step if needed
        if t + h > t_end {
            h = t_end - t;
        }

        // Limit step size to bounds
        h = h.min(max_step).max(min_step);

        // Compute the 12 stages of the DOP853 tableau.
        let mut k_stages: Vec<Array1<F>> = Vec::with_capacity(DOP853_STAGES);
        k_stages.push(f(t, y.view()));
        for s in 1..DOP853_STAGES {
            let mut y_stage = y.clone();
            for (j, k_j) in k_stages.iter().enumerate().take(s) {
                let a_sj = DOP853_A[s][j];
                if a_sj != 0.0 {
                    let a_sj = F::from_f64(a_sj).expect("Operation failed");
                    for d in 0..n_dim {
                        y_stage[d] += h * a_sj * k_j[d];
                    }
                }
            }
            let c_s = F::from_f64(DOP853_C[s]).expect("Operation failed");
            k_stages.push(f(t + c_s * h, y_stage.view()));
        }
        func_evals += DOP853_STAGES;

        // 8th-order solution.
        let mut y8 = y.clone();
        for (j, k_j) in k_stages.iter().enumerate() {
            let b_j = DOP853_B[j];
            if b_j != 0.0 {
                let b_j = F::from_f64(b_j).expect("Operation failed");
                for d in 0..n_dim {
                    y8[d] += h * b_j * k_j[d];
                }
            }
        }

        // Blended 5th/3rd-order error estimate.
        let mut err5_sq = F::zero();
        let mut err3_sq = F::zero();
        for d in 0..n_dim {
            let scale = opts.atol + opts.rtol * y[d].abs().max(y8[d].abs());
            let mut e5_d = F::zero();
            let mut e3_d = F::zero();
            for (j, k_j) in k_stages.iter().enumerate() {
                let kjd = k_j[d];
                let e5j = DOP853_E5[j];
                if e5j != 0.0 {
                    e5_d += F::from_f64(e5j).expect("Operation failed") * kjd;
                }
                let e3j = e3[j];
                if e3j != 0.0 {
                    e3_d += F::from_f64(e3j).expect("Operation failed") * kjd;
                }
            }
            let e5s = e5_d / scale;
            let e3s = e3_d / scale;
            err5_sq += e5s * e5s;
            err3_sq += e3s * e3s;
        }
        let denom = err5_sq + err3_weight * err3_sq;
        let err_norm = if denom <= F::zero() {
            F::zero()
        } else {
            let raw = h.abs() * err5_sq / (denom * n_dim_f).sqrt();
            if raw.is_finite() {
                raw
            } else {
                // A diverging/blown-up derivative produced a non-finite
                // error estimate; treat it as maximally bad so the
                // controller shrinks the step aggressively instead of
                // stalling.
                non_finite_penalty
            }
        };

        let accepted = err_norm <= F::one();
        let (factor, new_err_prev) = pi_step_factor(
            err_norm,
            err_prev,
            accepted,
            just_rejected,
            alpha,
            beta_gain,
            safety,
            growth_min,
            growth_max,
        );

        if accepted {
            // Step accepted
            t += h;
            y = y8;

            // Store results
            t_values.push(t);
            y_values.push(y.clone());

            err_prev = new_err_prev;
            just_rejected = false;
            h *= factor;

            step_count += 1;
            accepted_steps += 1;
        } else {
            // Step rejected
            h *= factor;
            just_rejected = true;
            rejected_steps += 1;

            // If step size is too small, return error
            if h < min_step {
                return Err(crate::error::IntegrateError::StepSizeTooSmall(format!(
                    "Step size {h} too small at t {t}"
                )));
            }
        }
    }

    // Check if integration was successful
    let success = t >= t_end;
    let message = if !success {
        Some(format!(
            "Maximum number of steps ({}) reached",
            opts.max_steps
        ))
    } else {
        None
    };

    // Return the solution
    Ok(ODEResult {
        t: t_values,
        y: y_values,
        success,
        message,
        n_eval: func_evals,
        n_steps: step_count,
        n_accepted: accepted_steps,
        n_rejected: rejected_steps,
        n_lu: 0,  // No LU decompositions in explicit methods
        n_jac: 0, // No Jacobian evaluations in explicit methods
        method: ODEMethod::DOP853,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Classical fixed-step convergence-order check: setting
    /// `min_step == max_step == h` pins every attempted step to exactly
    /// `h`, since the `h.min(max_step).max(min_step)` clamp at the top of
    /// the main loop wins regardless of what the adaptive controller's
    /// accept/reject decision would otherwise have chosen. A very loose
    /// tolerance guarantees every fixed-size step is accepted, so this
    /// isolates the raw Butcher tableau's truncation-error order: halving
    /// `h` should shrink the global error by ~2^order.
    #[test]
    fn rk23_empirical_convergence_order_is_three() {
        let k = 3.0_f64;
        let f =
            move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![-k * y[0]]) };
        let t_end = 1.0_f64;
        let y0 = 2.0_f64;
        let y_exact = y0 * (-k * t_end).exp();

        let mut prev_err: Option<f64> = None;
        let mut min_ratio = f64::INFINITY;
        let mut h = t_end / 8.0;
        for _ in 0..3 {
            let opts = ODEOptions {
                method: ODEMethod::RK23,
                rtol: 1.0,
                atol: 1.0,
                h0: Some(h),
                min_step: Some(h),
                max_step: Some(h),
                max_steps: 1_000_000,
                ..Default::default()
            };
            let result = rk23_method(f, [0.0, t_end], Array1::from_vec(vec![y0]), opts)
                .expect("RK23 fixed-step solve failed");
            assert!(
                result.success,
                "RK23 fixed-step integration did not complete"
            );
            let y_final = result.y.last().expect("empty result")[0];
            let err = (y_final - y_exact).abs();

            if let Some(prev) = prev_err {
                min_ratio = min_ratio.min(prev / err);
            }
            prev_err = Some(err);
            h /= 2.0;
        }

        // A genuine 3rd-order method shrinks the error by ~2^3=8x each
        // time h halves. Use a generous lower bound (5x) that a
        // mislabeled 1st-order Euler stand-in (ratio ~2x) cannot meet.
        assert!(
            min_ratio > 5.0,
            "RK23 empirical convergence order too low: min ratio={min_ratio} (expected ~8)"
        );
    }

    #[test]
    fn dop853_empirical_convergence_order_is_eight() {
        let k = 3.0_f64;
        let f =
            move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![-k * y[0]]) };
        let t_end = 1.0_f64;
        let y0 = 2.0_f64;
        let y_exact = y0 * (-k * t_end).exp();

        let mut prev_err: Option<f64> = None;
        let mut min_ratio = f64::INFINITY;
        let mut h = t_end / 2.0;
        for _ in 0..3 {
            let opts = ODEOptions {
                method: ODEMethod::DOP853,
                rtol: 1.0,
                atol: 1.0,
                h0: Some(h),
                min_step: Some(h),
                max_step: Some(h),
                max_steps: 1_000_000,
                ..Default::default()
            };
            let result = dop853_method(f, [0.0, t_end], Array1::from_vec(vec![y0]), opts)
                .expect("DOP853 fixed-step solve failed");
            assert!(
                result.success,
                "DOP853 fixed-step integration did not complete"
            );
            let y_final = result.y.last().expect("empty result")[0];
            let err = (y_final - y_exact).abs();

            if let Some(prev) = prev_err {
                min_ratio = min_ratio.min(prev / err);
            }
            prev_err = Some(err);
            h /= 2.0;
        }

        // A genuine 8th-order method shrinks the error by ~2^8=256x each
        // time h halves. Use a generous lower bound (50x).
        assert!(
            min_ratio > 50.0,
            "DOP853 empirical convergence order too low: min ratio={min_ratio} (expected ~256)"
        );
    }

    /// Cross-method agreement + sanity check directly at the method-function
    /// level (bypassing the public `solve_ivp` dispatcher): RK23 and DOP853
    /// must agree closely with the crate's existing RK45 on a nonlinear,
    /// non-constant-trajectory problem (Van der Pol oscillator).
    #[test]
    fn rk23_dop853_agree_with_rk45_van_der_pol() {
        let mu = 1.0_f64;
        let f = move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> {
            Array1::from_vec(vec![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]])
        };
        let y0 = Array1::from_vec(vec![0.5_f64, 0.0]);
        let t_span = [0.0_f64, 3.0];

        let opts_of = |method: ODEMethod| ODEOptions {
            method,
            rtol: 1e-10,
            atol: 1e-12,
            max_steps: 500_000,
            ..Default::default()
        };

        let rk45 = rk45_method(f, t_span, y0.clone(), opts_of(ODEMethod::RK45))
            .expect("RK45 Van der Pol solve failed");
        let rk23 = rk23_method(f, t_span, y0.clone(), opts_of(ODEMethod::RK23))
            .expect("RK23 Van der Pol solve failed");
        let dop853 = dop853_method(f, t_span, y0, opts_of(ODEMethod::DOP853))
            .expect("DOP853 Van der Pol solve failed");

        assert!(rk45.success && rk23.success && dop853.success);

        let y_rk45 = rk45.y.last().expect("empty result");
        let y_rk23 = rk23.y.last().expect("empty result");
        let y_dop853 = dop853.y.last().expect("empty result");

        for i in 0..2 {
            assert!(
                (y_rk23[i] - y_rk45[i]).abs() < 1e-5,
                "RK23 vs RK45 disagreement at index {i}: {} vs {}",
                y_rk23[i],
                y_rk45[i]
            );
            assert!(
                (y_dop853[i] - y_rk45[i]).abs() < 1e-5,
                "DOP853 vs RK45 disagreement at index {i}: {} vs {}",
                y_dop853[i],
                y_rk45[i]
            );
        }
    }
}
