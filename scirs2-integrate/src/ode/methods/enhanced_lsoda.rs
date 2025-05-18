//! Enhanced LSODA method for ODE solving
//!
//! This module implements an enhanced version of LSODA (Livermore Solver for Ordinary
//! Differential Equations with Automatic method switching) for solving ODE systems.
//! It features improved stiffness detection, more robust method switching, and
//! better Jacobian handling.

use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::types::{ODEMethod, ODEOptions, ODEResult};
use crate::ode::utils::common::{
    calculate_error_weights, estimate_initial_step, extrapolate, finite_difference_jacobian,
    scaled_norm, solve_linear_system,
};
use crate::ode::utils::stiffness::integration::{AdaptiveMethodState, AdaptiveMethodType};
use crate::ode::utils::stiffness::StiffnessDetectionConfig;
use crate::IntegrateFloat;
use ndarray::{Array1, Array2, ArrayView1};

/// Enhanced LSODA method state information
struct EnhancedLsodaState<F: IntegrateFloat> {
    /// Current time
    t: F,
    /// Current solution
    y: Array1<F>,
    /// Current derivative
    dy: Array1<F>,
    /// Current integration step size
    h: F,
    /// History of time points
    t_history: Vec<F>,
    /// History of solution values
    y_history: Vec<Array1<F>>,
    /// History of derivatives
    dy_history: Vec<Array1<F>>,
    /// Adaptive method state for method switching
    adaptive_state: AdaptiveMethodState<F>,
    /// Jacobian matrix
    jacobian: Option<Array2<F>>,
    /// Time since last Jacobian update
    jacobian_age: usize,
    /// Function evaluations
    func_evals: usize,
    /// LU decompositions performed
    n_lu: usize,
    /// Jacobian evaluations performed
    n_jac: usize,
    /// Steps taken
    steps: usize,
    /// Accepted steps
    accepted_steps: usize,
    /// Rejected steps
    rejected_steps: usize,
    /// Tolerance scaling for error control
    tol_scale: Array1<F>,
}

impl<F: IntegrateFloat> EnhancedLsodaState<F> {
    /// Create a new LSODA state
    fn new(t: F, y: Array1<F>, dy: Array1<F>, h: F, rtol: F, atol: F) -> Self {
        let _n_dim = y.len();

        // Calculate tolerance scaling for error control
        let tol_scale = calculate_error_weights(&y, atol, rtol);

        // Create stiffness detection configuration
        let stiffness_config = StiffnessDetectionConfig::default();

        EnhancedLsodaState {
            t,
            y: y.clone(),
            dy: dy.clone(),
            h,
            t_history: vec![t],
            y_history: vec![y],
            dy_history: vec![dy],
            adaptive_state: AdaptiveMethodState::with_config(stiffness_config),
            jacobian: None,
            jacobian_age: 0,
            func_evals: 0,
            n_lu: 0,
            n_jac: 0,
            steps: 0,
            accepted_steps: 0,
            rejected_steps: 0,
            tol_scale,
        }
    }

    /// Update tolerance scaling factors
    fn update_tol_scale(&mut self, rtol: F, atol: F) {
        self.tol_scale = calculate_error_weights(&self.y, atol, rtol);
    }

    /// Add current state to history
    fn add_to_history(&mut self) {
        self.t_history.push(self.t);
        self.y_history.push(self.y.clone());
        self.dy_history.push(self.dy.clone());

        // Keep history limited to what's needed
        let max_history = match self.adaptive_state.method_type {
            AdaptiveMethodType::Explicit => 12, // Adams can use up to order 12
            AdaptiveMethodType::Implicit => 5,  // BDF can use up to order 5
            AdaptiveMethodType::Adams => 12,    // Adams can use up to order 12
            AdaptiveMethodType::BDF => 5,       // BDF can use up to order 5
            AdaptiveMethodType::RungeKutta => 4, // RK methods typically don't need much history
        };

        if self.t_history.len() > max_history {
            self.t_history.remove(0);
            self.y_history.remove(0);
            self.dy_history.remove(0);
        }
    }

    /// Switch method type (between Adams and BDF)
    fn switch_method(&mut self, new_method: AdaptiveMethodType) -> IntegrateResult<()> {
        // Let the adaptive state handle the switching logic
        self.adaptive_state.switch_method(new_method, self.steps)?;

        // Additional state adjustments
        match new_method {
            AdaptiveMethodType::Implicit | AdaptiveMethodType::BDF => {
                // When switching to BDF, reset Jacobian
                self.jacobian = None;
                self.jacobian_age = 0;
            }
            AdaptiveMethodType::Explicit | AdaptiveMethodType::Adams => {
                // When switching to Adams, be more conservative with step size
                if self.rejected_steps > 2 {
                    self.h *= F::from_f64(0.5).unwrap();
                }
            }
            AdaptiveMethodType::RungeKutta => {
                // RK methods - reset step size to be conservative
                self.h *= F::from_f64(0.8).unwrap();
            }
        }

        Ok(())
    }
}

/// Solve ODE using enhanced LSODA method with improved stiffness detection
///
/// This enhanced LSODA method features:
/// - More sophisticated stiffness detection algorithms
/// - Improved method switching logic
/// - Better Jacobian handling and reuse
/// - More efficient linear system solving
/// - Comprehensive diagnostics and statistics
///
/// The method automatically switches between Adams methods (explicit, non-stiff)
/// and BDF methods (implicit, stiff) based on detected stiffness characteristics.
pub fn enhanced_lsoda_method<F, Func>(
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
    let _n_dim = y0.len();

    // Initial evaluation
    let dy0 = f(t_start, y0.view());
    let mut func_evals = 1;

    // Estimate initial step size if not provided
    let h0 = opts.h0.unwrap_or_else(|| {
        // Use more sophisticated step size estimation
        let tol = opts.atol + opts.rtol;
        estimate_initial_step(&f, t_start, &y0, &dy0, tol, t_end)
    });

    // Determine minimum and maximum step sizes
    let min_step = opts.min_step.unwrap_or_else(|| {
        let span = t_end - t_start;
        span * F::from_f64(1e-10).unwrap() // Minimal step size
    });

    let max_step = opts.max_step.unwrap_or_else(|| {
        t_end - t_start // Maximum step can be the whole interval
    });

    // Initialize LSODA state
    let mut state = EnhancedLsodaState::new(t_start, y0.clone(), dy0, h0, opts.rtol, opts.atol);

    // Result storage
    let mut t_values = vec![t_start];
    let mut y_values = vec![y0.clone()];

    // Main integration loop
    while state.t < t_end && state.steps < opts.max_steps {
        // Adjust step size for the last step if needed
        if state.t + state.h > t_end {
            state.h = t_end - state.t;
        }

        // Limit step size to bounds
        state.h = state.h.min(max_step).max(min_step);

        // Step with the current method
        let step_result = match state.adaptive_state.method_type {
            AdaptiveMethodType::Explicit | AdaptiveMethodType::Adams => {
                enhanced_adams_step(&mut state, &f, &opts, &mut func_evals)
            }
            AdaptiveMethodType::Implicit | AdaptiveMethodType::BDF => {
                enhanced_bdf_step(&mut state, &f, &opts, &mut func_evals)
            }
            AdaptiveMethodType::RungeKutta => {
                // For RK methods, use Adams for now
                enhanced_adams_step(&mut state, &f, &opts, &mut func_evals)
            }
        };

        state.steps += 1;
        state.adaptive_state.steps_since_switch += 1;

        match step_result {
            Ok(accepted) => {
                if accepted {
                    // Step accepted

                    // Add to history and results
                    state.add_to_history();
                    t_values.push(state.t);
                    y_values.push(state.y.clone());

                    state.accepted_steps += 1;

                    // Record step data for stiffness analysis
                    let error = F::zero(); // We don't have direct error estimate for reporting
                    let _newton_iterations = 0; // Would need to be passed from step methods
                    state.adaptive_state.record_step(error);

                    // Check for method switching
                    if let Some(_new_method) = state.adaptive_state.check_method_switch() {
                        // Method switching already happened in the check_method_switch call
                    }

                    // Update tolerance scaling for next step
                    state.update_tol_scale(opts.rtol, opts.atol);

                    // Increment Jacobian age if we're using BDF
                    if state.adaptive_state.method_type == AdaptiveMethodType::Implicit
                        && state.jacobian.is_some()
                    {
                        state.jacobian_age += 1;
                    }
                } else {
                    // Step rejected
                    state.rejected_steps += 1;

                    // Record step data for stiffness analysis
                    let error = F::one(); // Placeholder for rejected step
                    let _newton_iterations = 0; // Would need to be passed from step methods
                    state.adaptive_state.record_step(error);
                }
            }
            Err(e) => {
                // Handle specific errors that might indicate stiffness changes
                match &e {
                    IntegrateError::ConvergenceError(msg) if msg.contains("stiff") => {
                        if state.adaptive_state.method_type == AdaptiveMethodType::Explicit {
                            // Problem appears to be stiff - switch to BDF
                            state.switch_method(AdaptiveMethodType::Implicit)?;

                            // Reduce step size
                            state.h *= F::from_f64(0.5).unwrap();
                            if state.h < min_step {
                                return Err(IntegrateError::ConvergenceError(
                                    "Step size too small after method switch".to_string(),
                                ));
                            }
                        } else {
                            // Already using BDF and still failing
                            return Err(e);
                        }
                    }
                    IntegrateError::ConvergenceError(msg) if msg.contains("non-stiff") => {
                        if state.adaptive_state.method_type == AdaptiveMethodType::Implicit {
                            // Problem appears to be non-stiff - switch to Adams
                            state.switch_method(AdaptiveMethodType::Explicit)?;

                            // Reduce step size for stability
                            state.h *= F::from_f64(0.5).unwrap();
                            if state.h < min_step {
                                return Err(IntegrateError::ConvergenceError(
                                    "Step size too small after method switch".to_string(),
                                ));
                            }
                        } else {
                            // Already using Adams and still failing
                            return Err(e);
                        }
                    }
                    _ => return Err(e), // Other errors are passed through
                }
            }
        }
    }

    let success = state.t >= t_end;
    let message = if !success {
        Some(format!(
            "Maximum number of steps ({}) reached",
            opts.max_steps
        ))
    } else {
        // Include method switching diagnostic information
        Some(state.adaptive_state.generate_diagnostic_message())
    };

    // Return the solution
    Ok(ODEResult {
        t: t_values,
        y: y_values,
        success,
        message,
        n_eval: func_evals,
        n_steps: state.steps,
        n_accepted: state.accepted_steps,
        n_rejected: state.rejected_steps,
        n_lu: state.n_lu,
        n_jac: state.n_jac,
        method: ODEMethod::LSODA,
    })
}

/// Enhanced Adams method (predictor-corrector) for non-stiff regions
fn enhanced_adams_step<F, Func>(
    state: &mut EnhancedLsodaState<F>,
    f: &Func,
    opts: &ODEOptions<F>,
    func_evals: &mut usize,
) -> IntegrateResult<bool>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Coefficients for Adams-Bashforth (predictor)
    // These are the coefficients for different orders (1-12)
    let ab_coeffs: [Vec<F>; 12] = [
        // Order 1 (Euler)
        vec![F::one()],
        // Order 2
        vec![
            F::from_f64(3.0 / 2.0).unwrap(),
            F::from_f64(-1.0 / 2.0).unwrap(),
        ],
        // Order 3
        vec![
            F::from_f64(23.0 / 12.0).unwrap(),
            F::from_f64(-16.0 / 12.0).unwrap(),
            F::from_f64(5.0 / 12.0).unwrap(),
        ],
        // Order 4
        vec![
            F::from_f64(55.0 / 24.0).unwrap(),
            F::from_f64(-59.0 / 24.0).unwrap(),
            F::from_f64(37.0 / 24.0).unwrap(),
            F::from_f64(-9.0 / 24.0).unwrap(),
        ],
        // Order 5
        vec![
            F::from_f64(1901.0 / 720.0).unwrap(),
            F::from_f64(-2774.0 / 720.0).unwrap(),
            F::from_f64(2616.0 / 720.0).unwrap(),
            F::from_f64(-1274.0 / 720.0).unwrap(),
            F::from_f64(251.0 / 720.0).unwrap(),
        ],
        // Order 6
        vec![
            F::from_f64(4277.0 / 1440.0).unwrap(),
            F::from_f64(-7923.0 / 1440.0).unwrap(),
            F::from_f64(9982.0 / 1440.0).unwrap(),
            F::from_f64(-7298.0 / 1440.0).unwrap(),
            F::from_f64(2877.0 / 1440.0).unwrap(),
            F::from_f64(-475.0 / 1440.0).unwrap(),
        ],
        // Order 7
        vec![
            F::from_f64(198721.0 / 60480.0).unwrap(),
            F::from_f64(-447288.0 / 60480.0).unwrap(),
            F::from_f64(705549.0 / 60480.0).unwrap(),
            F::from_f64(-688256.0 / 60480.0).unwrap(),
            F::from_f64(407139.0 / 60480.0).unwrap(),
            F::from_f64(-134472.0 / 60480.0).unwrap(),
            F::from_f64(19087.0 / 60480.0).unwrap(),
        ],
        // Order 8+
        vec![
            F::from_f64(434241.0 / 120960.0).unwrap(),
            F::from_f64(-1152169.0 / 120960.0).unwrap(),
            F::from_f64(2183877.0 / 120960.0).unwrap(),
            F::from_f64(-2664477.0 / 120960.0).unwrap(),
            F::from_f64(2102243.0 / 120960.0).unwrap(),
            F::from_f64(-1041723.0 / 120960.0).unwrap(),
            F::from_f64(295767.0 / 120960.0).unwrap(),
            F::from_f64(-36799.0 / 120960.0).unwrap(),
        ],
        // Order 9
        vec![
            F::from_f64(14097247.0 / 3628800.0).unwrap(),
            F::from_f64(-43125206.0 / 3628800.0).unwrap(),
            F::from_f64(95476786.0 / 3628800.0).unwrap(),
            F::from_f64(-139855262.0 / 3628800.0).unwrap(),
            F::from_f64(137968480.0 / 3628800.0).unwrap(),
            F::from_f64(-91172642.0 / 3628800.0).unwrap(),
            F::from_f64(38833486.0 / 3628800.0).unwrap(),
            F::from_f64(-9664106.0 / 3628800.0).unwrap(),
            F::from_f64(1070017.0 / 3628800.0).unwrap(),
        ],
        // Order 10
        vec![
            F::from_f64(30277247.0 / 7257600.0).unwrap(),
            F::from_f64(-104995189.0 / 7257600.0).unwrap(),
            F::from_f64(265932680.0 / 7257600.0).unwrap(),
            F::from_f64(-454661776.0 / 7257600.0).unwrap(),
            F::from_f64(538363838.0 / 7257600.0).unwrap(),
            F::from_f64(-444772162.0 / 7257600.0).unwrap(),
            F::from_f64(252618224.0 / 7257600.0).unwrap(),
            F::from_f64(-94307320.0 / 7257600.0).unwrap(),
            F::from_f64(20884811.0 / 7257600.0).unwrap(),
            F::from_f64(-2082753.0 / 7257600.0).unwrap(),
        ],
        // Order 11
        vec![
            F::from_f64(35256204767.0 / 7983360000.0).unwrap(),
            F::from_f64(-134336876800.0 / 7983360000.0).unwrap(),
            F::from_f64(385146025457.0 / 7983360000.0).unwrap(),
            F::from_f64(-754734083733.0 / 7983360000.0).unwrap(),
            F::from_f64(1045594573504.0 / 7983360000.0).unwrap(),
            F::from_f64(-1029725952608.0 / 7983360000.0).unwrap(),
            F::from_f64(717313887930.0 / 7983360000.0).unwrap(),
            F::from_f64(-344156361067.0 / 7983360000.0).unwrap(),
            F::from_f64(109301088672.0 / 7983360000.0).unwrap(),
            F::from_f64(-21157613775.0 / 7983360000.0).unwrap(),
            F::from_f64(1832380165.0 / 7983360000.0).unwrap(),
        ],
        // Order 12
        vec![
            F::from_f64(77737505967.0 / 16876492800.0).unwrap(),
            F::from_f64(-328202700680.0 / 16876492800.0).unwrap(),
            F::from_f64(1074851727475.0 / 16876492800.0).unwrap(),
            F::from_f64(-2459572352768.0 / 16876492800.0).unwrap(),
            F::from_f64(4013465151807.0 / 16876492800.0).unwrap(),
            F::from_f64(-4774671405984.0 / 16876492800.0).unwrap(),
            F::from_f64(4127030565077.0 / 16876492800.0).unwrap(),
            F::from_f64(-2538584431976.0 / 16876492800.0).unwrap(),
            F::from_f64(1077984741336.0 / 16876492800.0).unwrap(),
            F::from_f64(-295501032385.0 / 16876492800.0).unwrap(),
            F::from_f64(48902348238.0 / 16876492800.0).unwrap(),
            F::from_f64(-3525779602.0 / 16876492800.0).unwrap(),
        ],
    ];

    // Coefficients for Adams-Moulton (corrector)
    // These are the coefficients for different orders (1-12)
    let am_coeffs: [Vec<F>; 12] = [
        // Order 1 (Backward Euler)
        vec![F::one()],
        // Order 2 (Trapezoidal)
        vec![
            F::from_f64(1.0 / 2.0).unwrap(),
            F::from_f64(1.0 / 2.0).unwrap(),
        ],
        // Order 3
        vec![
            F::from_f64(5.0 / 12.0).unwrap(),
            F::from_f64(8.0 / 12.0).unwrap(),
            F::from_f64(-1.0 / 12.0).unwrap(),
        ],
        // Order 4
        vec![
            F::from_f64(9.0 / 24.0).unwrap(),
            F::from_f64(19.0 / 24.0).unwrap(),
            F::from_f64(-5.0 / 24.0).unwrap(),
            F::from_f64(1.0 / 24.0).unwrap(),
        ],
        // Orders 5-12 (truncated for brevity - would include full coefficients)
        // First few orders are the most commonly used
        vec![F::zero()],
        vec![F::zero()],
        vec![F::zero()],
        vec![F::zero()],
        vec![F::zero()],
        vec![F::zero()],
        vec![F::zero()],
        vec![F::zero()],
    ];

    // Get the current order from the adaptive state
    let order = state
        .adaptive_state
        .order
        .min(state.dy_history.len() + 1)
        .min(12);

    // If we don't have enough history, use lower order
    if order == 1 || state.dy_history.is_empty() {
        // Explicit Euler method (1st order Adams-Bashforth)
        let next_t = state.t + state.h;
        let next_y = &state.y + &(state.dy.clone() * state.h);

        // Evaluate at the new point
        let next_dy = f(next_t, next_y.view());
        *func_evals += 1;
        state.func_evals += 1;

        // Update state
        state.t = next_t;
        state.y = next_y;
        state.dy = next_dy;

        // Order can now be increased next step
        if state.adaptive_state.order < 2 {
            state.adaptive_state.order += 1;
        }

        return Ok(true);
    }

    // Adams-Bashforth predictor (explicit step)
    let next_t = state.t + state.h;
    let ab_coefs = &ab_coeffs[order - 1];

    // Apply Adams-Bashforth formula to predict next value
    // y_{n+1} = y_n + h * sum(b_i * f_{n-i+1})
    let mut ab_sum = state.dy.clone() * ab_coefs[0];

    for (i, &coeff) in ab_coefs.iter().enumerate().take(order).skip(1) {
        if i <= state.dy_history.len() {
            let idx = state.dy_history.len() - i;
            ab_sum += &(state.dy_history[idx].clone() * coeff);
        }
    }

    let y_pred = &state.y + &(ab_sum * state.h);

    // Evaluate function at the predicted point
    let dy_pred = f(next_t, y_pred.view());
    *func_evals += 1;
    state.func_evals += 1;

    // Adams-Moulton corrector (implicit step)
    // For simplicity, we'll use lower order corrector
    let am_order = order.min(4); // Only using up to 4th order corrector for simplicity
    let am_coefs = &am_coeffs[am_order - 1];

    // Apply Adams-Moulton formula to correct the prediction
    // y_{n+1} = y_n + h * (b_0 * f_{n+1} + sum(b_i * f_{n-i+1}))
    let mut am_sum = dy_pred.clone() * am_coefs[0]; // f_{n+1} term

    for (i, &coeff) in am_coefs.iter().enumerate().take(am_order).skip(1) {
        if i == 1 {
            // Current derivative (f_n)
            am_sum += &(state.dy.clone() * coeff);
        } else if i - 1 < state.dy_history.len() {
            // Historical derivatives (f_{n-1}, f_{n-2}, ...)
            let idx = state.dy_history.len() - (i - 1);
            am_sum += &(state.dy_history[idx].clone() * coeff);
        }
    }

    let y_corr = &state.y + &(am_sum * state.h);

    // Evaluate function at the corrected point
    let dy_corr = f(next_t, y_corr.view());
    *func_evals += 1;
    state.func_evals += 1;

    // Error estimation based on predictor-corrector difference
    let error = scaled_norm(&(&y_corr - &y_pred), &state.tol_scale);

    // Step size adjustment factor based on error
    let err_order = F::from_usize(order + 1).unwrap(); // Error order is one higher than method order
    let err_factor = if error > F::zero() {
        F::from_f64(0.9).unwrap() * (F::one() / error).powf(F::one() / err_order)
    } else {
        F::from_f64(5.0).unwrap() // Max increase if error is zero
    };

    // Safety factor and limits for step size adjustment
    let safety = F::from_f64(0.9).unwrap();
    let factor_max = F::from_f64(5.0).unwrap();
    let factor_min = F::from_f64(0.2).unwrap();
    let factor = safety * err_factor.min(factor_max).max(factor_min);

    // Check if step is acceptable
    if error <= F::one() {
        // Step accepted

        // Update state
        state.t = next_t;
        state.y = y_corr;
        state.dy = dy_corr;

        // Update step size for next step
        state.h *= factor;

        // Order adaptation
        if order < 12 && error < opts.rtol && state.dy_history.len() >= order {
            state.adaptive_state.order = (state.adaptive_state.order + 1).min(12);
        } else if order > 1 && error > F::from_f64(0.5).unwrap() {
            state.adaptive_state.order = (state.adaptive_state.order - 1).max(1);
        }

        // Trigger stiffness detector to record this step
        state.adaptive_state.record_step(error);

        Ok(true)
    } else {
        // Step rejected

        // Adjust step size for retry
        state.h *= factor;

        // Trigger stiffness detector to record this rejected step
        state.adaptive_state.record_step(error);

        // If error is very large, this might indicate stiffness
        if error > F::from_f64(10.0).unwrap() {
            return Err(IntegrateError::ConvergenceError(
                "Problem appears stiff - consider using BDF method".to_string(),
            ));
        }

        Ok(false)
    }
}

/// Enhanced BDF method for stiff regions
fn enhanced_bdf_step<F, Func>(
    state: &mut EnhancedLsodaState<F>,
    f: &Func,
    opts: &ODEOptions<F>,
    func_evals: &mut usize,
) -> IntegrateResult<bool>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Coefficients for BDF methods of different orders
    let bdf_coefs: [Vec<F>; 5] = [
        // BDF1 (Implicit Euler): y_{n+1} - y_n = h * f(t_{n+1}, y_{n+1})
        vec![F::one(), F::from_f64(-1.0).unwrap()],
        // BDF2: 3/2 * y_{n+1} - 2 * y_n + 1/2 * y_{n-1} = h * f(t_{n+1}, y_{n+1})
        vec![
            F::from_f64(3.0 / 2.0).unwrap(),
            F::from_f64(-2.0).unwrap(),
            F::from_f64(1.0 / 2.0).unwrap(),
        ],
        // BDF3
        vec![
            F::from_f64(11.0 / 6.0).unwrap(),
            F::from_f64(-3.0).unwrap(),
            F::from_f64(3.0 / 2.0).unwrap(),
            F::from_f64(-1.0 / 3.0).unwrap(),
        ],
        // BDF4
        vec![
            F::from_f64(25.0 / 12.0).unwrap(),
            F::from_f64(-4.0).unwrap(),
            F::from_f64(3.0).unwrap(),
            F::from_f64(-4.0 / 3.0).unwrap(),
            F::from_f64(1.0 / 4.0).unwrap(),
        ],
        // BDF5
        vec![
            F::from_f64(137.0 / 60.0).unwrap(),
            F::from_f64(-5.0).unwrap(),
            F::from_f64(5.0).unwrap(),
            F::from_f64(-10.0 / 3.0).unwrap(),
            F::from_f64(5.0 / 4.0).unwrap(),
            F::from_f64(-1.0 / 5.0).unwrap(),
        ],
    ];

    // Use the appropriate order based on history availability
    let order = state.adaptive_state.order.min(state.y_history.len()).min(5);

    // If we don't have enough history for the requested order, use lower order
    if order == 1 || state.y_history.is_empty() {
        // Implicit Euler method (1st order BDF)
        let next_t = state.t + state.h;

        // Predict the next value (simple extrapolation)
        let y_pred = state.y.clone();

        // Newton's method for solving the implicit equation
        let max_newton_iters = 10;
        let newton_tol = F::from_f64(1e-8).unwrap();
        let mut y_next = y_pred.clone();
        let mut converged = false;
        let mut iter_count = 0;

        // Store initial function eval for potential Jacobian computation
        let mut f_eval = f(next_t, y_next.view());
        *func_evals += 1;
        state.func_evals += 1;

        while iter_count < max_newton_iters {
            // Compute residual for BDF1: y_{n+1} - y_n - h * f(t_{n+1}, y_{n+1}) = 0
            let residual = &y_next - &state.y - &(f_eval.clone() * state.h);

            // Check convergence
            let error = scaled_norm(&residual, &state.tol_scale);

            if error <= newton_tol {
                converged = true;
                break;
            }

            // Compute or reuse Jacobian
            let eps = F::from_f64(1e-8).unwrap();
            let n_dim = y_next.len();

            // Create approximate Jacobian using finite differences if needed
            let compute_new_jacobian =
                state.jacobian.is_none() || state.jacobian_age > 20 || iter_count == 0;
            let jacobian = if compute_new_jacobian {
                state.n_jac += 1;

                // Create finite difference Jacobian
                let new_jacobian = finite_difference_jacobian(f, next_t, &y_next, &f_eval, eps);

                // Modify for solving BDF: I - h*J
                let mut jac = Array2::<F>::eye(n_dim);
                for i in 0..n_dim {
                    for j in 0..n_dim {
                        jac[[i, j]] = if i == j { F::one() } else { F::zero() };
                        jac[[i, j]] -= state.h * new_jacobian[[i, j]];
                    }
                }

                // Store the Jacobian for potential reuse
                state.jacobian = Some(jac.clone());
                state.jacobian_age = 0;
                jac
            } else {
                // Reuse previous Jacobian
                state.jacobian.clone().unwrap()
            };

            // Solve the linear system J*delta_y = residual
            state.n_lu += 1;

            // Use our more robust linear solver
            let delta_y = match solve_linear_system(&jacobian, &residual) {
                Ok(delta) => delta,
                Err(_) => {
                    // Nearly singular, reduce step size and try again
                    state.h *= F::from_f64(0.5).unwrap();
                    return Ok(false);
                }
            };

            // Update solution
            y_next = &y_next - &delta_y;

            // Evaluate function at new point
            f_eval = f(next_t, y_next.view());
            *func_evals += 1;
            state.func_evals += 1;

            iter_count += 1;

            // Record Newton iteration count for stiffness detection
            state.adaptive_state.record_step(error);
        }

        if !converged {
            // Newton iteration failed, reduce step size
            state.h *= F::from_f64(0.5).unwrap();

            // If we've reduced step size too much, the problem might be non-stiff
            if state.h < opts.min_step.unwrap_or(F::from_f64(1e-10).unwrap()) {
                return Err(IntegrateError::ConvergenceError(
                    "BDF1 failed to converge - problem might be non-stiff".to_string(),
                ));
            }

            return Ok(false);
        }

        // Step accepted

        // Update state
        state.t = next_t;
        state.y = y_next;
        state.dy = f_eval;

        // Order can now be increased next step
        if state.adaptive_state.order < 2 {
            state.adaptive_state.order += 1;
        }

        return Ok(true);
    }

    // Higher-order BDF methods (2-5)

    // Get BDF coefficients for the current order
    let coeffs = &bdf_coefs[order - 1];

    // Next time and step size
    let next_t = state.t + state.h;

    // Predict initial value using extrapolation from previous points
    let mut y_pred = state.y.clone();

    // For higher orders, use previous points for prediction
    if order > 1 && !state.y_history.is_empty() {
        let _y_prev = &state.y_history[state.y_history.len() - 1];

        // Use more sophisticated extrapolation
        y_pred = extrapolate(&state.t_history[..], &state.y_history[..], next_t);
    }

    // Newton's method for solving the BDF equation
    let max_newton_iters = 10;
    let newton_tol = F::from_f64(1e-8).unwrap();
    let mut y_next = y_pred.clone();
    let mut converged = false;
    let mut iter_count = 0;

    // Initial function evaluation
    let mut f_eval = f(next_t, y_next.view());
    *func_evals += 1;
    state.func_evals += 1;

    while iter_count < max_newton_iters {
        // Compute residual for BDF: c_0 * y_{n+1} - sum(c_j * y_{n+1-j}) - h * f(t_{n+1}, y_{n+1}) = 0
        let mut residual = y_next.clone() * coeffs[0];

        // Subtract previous terms
        residual -= &(state.y.clone() * coeffs[1]);

        for (j, &coeff) in coeffs.iter().enumerate().skip(2) {
            if j - 1 < state.y_history.len() {
                let idx = state.y_history.len() - (j - 1);
                residual -= &(state.y_history[idx].clone() * coeff);
            }
        }

        // Subtract h * f term
        residual -= &(f_eval.clone() * state.h);

        // Check convergence
        let error = scaled_norm(&residual, &state.tol_scale);

        if error <= newton_tol {
            converged = true;
            break;
        }

        // Compute or reuse Jacobian
        let eps = F::from_f64(1e-8).unwrap();
        let n_dim = y_next.len();

        // Create approximate Jacobian using finite differences if needed
        let compute_new_jacobian =
            state.jacobian.is_none() || state.jacobian_age > 20 || iter_count == 0;
        let jacobian = if compute_new_jacobian {
            state.n_jac += 1;

            // Create finite difference Jacobian
            let new_jacobian = finite_difference_jacobian(f, next_t, &y_next, &f_eval, eps);

            // Modify for solving BDF: c_0*I - h*J
            let mut jac = Array2::<F>::zeros((n_dim, n_dim));
            for i in 0..n_dim {
                for j in 0..n_dim {
                    jac[[i, j]] = if i == j { coeffs[0] } else { F::zero() };
                    jac[[i, j]] -= state.h * new_jacobian[[i, j]];
                }
            }

            // Store the Jacobian for potential reuse
            state.jacobian = Some(jac.clone());
            state.jacobian_age = 0;
            jac
        } else {
            // Reuse previous Jacobian
            state.jacobian.clone().unwrap()
        };

        // Solve the linear system J*delta_y = residual
        state.n_lu += 1;

        // Use our more robust linear solver
        let delta_y = match solve_linear_system(&jacobian, &residual) {
            Ok(delta) => delta,
            Err(_) => {
                // Nearly singular, reduce step size and try again
                state.h *= F::from_f64(0.5).unwrap();
                return Ok(false);
            }
        };

        // Update solution
        y_next = &y_next - &delta_y;

        // Evaluate function at new point
        f_eval = f(next_t, y_next.view());
        *func_evals += 1;
        state.func_evals += 1;

        iter_count += 1;

        // Record Newton iteration count for stiffness detection
        state.adaptive_state.record_step(error);
    }

    if !converged {
        // Newton iteration failed, reduce step size
        state.h *= F::from_f64(0.5).unwrap();

        // If the problem is consistently difficult to solve, it might not be stiff
        if iter_count >= max_newton_iters - 1 {
            // Record as potential non-stiffness indicator
            state.adaptive_state.record_step(
                F::one(), // placeholder for error
            );
        }

        // If we've reduced step size too much, the problem might not be stiff
        if state.h < opts.min_step.unwrap_or(F::from_f64(1e-10).unwrap()) {
            return Err(IntegrateError::ConvergenceError(
                "BDF failed to converge - problem might be non-stiff".to_string(),
            ));
        }

        return Ok(false);
    }

    // Step accepted

    // Update state
    state.t = next_t;
    state.y = y_next;
    state.dy = f_eval;

    // Error estimation is based on Newton convergence for now
    // A more sophisticated error estimator could be implemented later

    // Step size and order adaptation based on convergence rate
    if iter_count <= 2 {
        // Converged quickly - can increase step size
        state.h *= F::from_f64(1.1).unwrap();

        // Maybe increase order if convergence is very good
        if state.adaptive_state.order < 5 && state.y_history.len() >= state.adaptive_state.order {
            state.adaptive_state.order += 1;
        }
    } else if iter_count >= 8 {
        // Converged slowly - reduce step size
        state.h *= F::from_f64(0.8).unwrap();

        // Decrease order if we're struggling
        if state.adaptive_state.order > 1 {
            state.adaptive_state.order -= 1;
        }
    }

    // Increment Jacobian age
    state.jacobian_age += 1;

    Ok(true)
}
