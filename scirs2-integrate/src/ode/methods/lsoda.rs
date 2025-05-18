//! LSODA method for ODE solving
//!
//! This module implements the LSODA (Livermore Solver for Ordinary Differential
//! Equations with Automatic method switching) method for solving ODE systems.
//! LSODA automatically switches between Adams methods (non-stiff) and Bdf methods (stiff)
//! based on the detected stiffness of the problem during integration.

use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::types::{ODEMethod, ODEOptions, ODEResult};
use crate::IntegrateFloat;
use ndarray::{Array1, Array2, ArrayView1};
use std::fmt::Debug;

/// Method type for LSODA
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LsodaMethodType {
    /// Adams method (explicit, non-stiff)
    Adams,
    /// Bdf method (implicit, stiff)
    Bdf,
}

/// State information for the LSODA integrator
struct LsodaState<F: IntegrateFloat> {
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
    /// Current method type
    method_type: LsodaMethodType,
    /// Current order of the method
    order: usize,
    /// Jacobian matrix
    jacobian: Option<Array2<F>>,
    /// Time since last Jacobian update
    jacobian_age: usize,
    /// Method switching statistics
    stiff_to_nonstiff_switches: usize,
    nonstiff_to_stiff_switches: usize,
    /// Steps since last method switch
    steps_since_switch: usize,
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
    /// Stiffness detection counters
    stiffness_detected_count: usize,
    non_stiffness_detected_count: usize,
    /// Method has recently switched
    recently_switched: bool,
    /// Tolerance scaling for error control
    tol_scale: Array1<F>,
}

impl<F: IntegrateFloat> LsodaState<F> {
    /// Create a new LSODA state
    fn new(t: F, y: Array1<F>, dy: Array1<F>, h: F, rtol: F, atol: F) -> Self {
        let n_dim = y.len();

        // Calculate tolerance scaling for error control
        let mut tol_scale = Array1::<F>::zeros(n_dim);
        for i in 0..n_dim {
            tol_scale[i] = atol + rtol * y[i].abs();
        }

        LsodaState {
            t,
            y: y.clone(),
            dy: dy.clone(),
            h,
            t_history: vec![t],
            y_history: vec![y],
            dy_history: vec![dy],
            method_type: LsodaMethodType::Adams, // Start with non-stiff method
            order: 1,                            // Start with first-order
            jacobian: None,
            jacobian_age: 0,
            stiff_to_nonstiff_switches: 0,
            nonstiff_to_stiff_switches: 0,
            steps_since_switch: 0,
            func_evals: 0,
            n_lu: 0,
            n_jac: 0,
            steps: 0,
            accepted_steps: 0,
            rejected_steps: 0,
            stiffness_detected_count: 0,
            non_stiffness_detected_count: 0,
            recently_switched: false,
            tol_scale,
        }
    }

    /// Update tolerance scaling factors
    fn update_tol_scale(&mut self, rtol: F, atol: F) {
        for i in 0..self.y.len() {
            self.tol_scale[i] = atol + rtol * self.y[i].abs();
        }
    }

    /// Add current state to history
    fn add_to_history(&mut self) {
        self.t_history.push(self.t);
        self.y_history.push(self.y.clone());
        self.dy_history.push(self.dy.clone());

        // Keep history limited to what's needed
        let max_history = match self.method_type {
            LsodaMethodType::Adams => 12, // Adams can use up to order 12
            LsodaMethodType::Bdf => 5,    // Bdf can use up to order 5
        };

        if self.t_history.len() > max_history {
            self.t_history.remove(0);
            self.y_history.remove(0);
            self.dy_history.remove(0);
        }
    }

    /// Switch method type
    fn switch_method(&mut self, new_method: LsodaMethodType) {
        // Track switches between methods
        if self.method_type == LsodaMethodType::Adams && new_method == LsodaMethodType::Bdf {
            self.nonstiff_to_stiff_switches += 1;

            // When switching to Bdf, reset order and jacobian
            self.order = 1;
            self.jacobian = None;
            self.jacobian_age = 0;
        } else if self.method_type == LsodaMethodType::Bdf && new_method == LsodaMethodType::Adams {
            self.stiff_to_nonstiff_switches += 1;

            // When switching to Adams, be more conservative
            self.order = 1;

            // Optionally reduce step size when switching to non-stiff method
            if self.rejected_steps > 2 {
                self.h *= F::from_f64(0.5).unwrap();
            }
        }

        // Reset tracking variables
        self.steps_since_switch = 0;
        self.recently_switched = true;

        // Update method type
        self.method_type = new_method;
    }
}

/// Stiffness detector for LSODA
struct StiffnessDetector<F: IntegrateFloat> {
    // Minimum number of steps before considering method switch
    min_steps_before_switch: usize,
    // How many stiffness indicators needed to detect stiffness
    stiffness_threshold: usize,
    // How many non-stiffness indicators needed to go back to Adams
    non_stiffness_threshold: usize,
    // Scale factors for detection
    #[allow(dead_code)]
    step_size_ratio_threshold: F,
}

impl<F: IntegrateFloat> StiffnessDetector<F> {
    /// Create a new stiffness detector
    fn new() -> Self {
        StiffnessDetector {
            min_steps_before_switch: 5,
            stiffness_threshold: 3,
            non_stiffness_threshold: 5,
            step_size_ratio_threshold: F::from_f64(0.1).unwrap(),
        }
    }

    /// Check if the problem is stiff based on multiple indicators
    fn is_stiff(&self, state: &LsodaState<F>) -> bool {
        // Don't switch methods too frequently
        if state.steps_since_switch < self.min_steps_before_switch {
            return false;
        }

        // If already using Bdf, require more evidence to switch back to Adams
        if state.method_type == LsodaMethodType::Bdf {
            return state.non_stiffness_detected_count < self.non_stiffness_threshold;
        }

        // If using Adams, check if we should switch to Bdf
        state.stiffness_detected_count >= self.stiffness_threshold
    }
}

/// Solve ODE using LSODA method (Livermore Solver for Ordinary Differential Equations with Automatic method switching)
///
/// LSODA automatically switches between Adams methods (non-stiff) and Bdf methods (stiff) based on
/// the detected stiffness of the problem during integration. This makes it especially suitable for
/// problems that change character during the integration process.
///
/// ## Features
///
/// This implementation includes:
/// - Automatic stiffness detection and method switching
/// - Variable-order Adams methods (1-12) for non-stiff regions
/// - Variable-order Bdf methods (1-5) for stiff regions
/// - Adaptive step size control based on error estimation
/// - Jacobian approximation via finite differences
///
/// ## Method Details
///
/// - For non-stiff regions: Uses Adams-Moulton predictor-corrector methods (orders 1-12)
/// - For stiff regions: Uses Backward Differentiation Formula (Bdf) methods (orders 1-5)
/// - Automatic switching based on step size efficiency and convergence behavior
/// - Comprehensive error control with relative and absolute tolerance
///
/// ## Usage Tips
///
/// - Increasing `rtol` and `atol` can improve performance for less demanding accuracy
/// - For problems known to be stiff, consider specifying a larger initial step size
/// - The solver automatically detects when to switch methods, but benefits from good initial settings
pub fn lsoda_method<F, Func>(
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

    // Determine initial step size if not provided
    let h0 = opts.h0.unwrap_or_else(|| {
        // Simple heuristic for initial step size
        let span = t_end - t_start;
        span / F::from_usize(100).unwrap() * F::from_f64(0.1).unwrap() // 0.1% of interval
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
    let mut state = LsodaState::new(t_start, y0.clone(), dy0, h0, opts.rtol, opts.atol);
    let stiffness_detector = StiffnessDetector::new();

    // Result storage
    let mut t_values = vec![t_start];
    let mut y_values = vec![y0.clone()];

    // Main integration loop
    while state.t < t_end && state.steps < opts.max_steps {
        // Reset recently switched flag if we've taken enough steps
        if state.recently_switched
            && state.steps_since_switch >= stiffness_detector.min_steps_before_switch
        {
            state.recently_switched = false;
        }

        // Adjust step size for the last step if needed
        if state.t + state.h > t_end {
            state.h = t_end - state.t;
        }

        // Limit step size to bounds
        state.h = state.h.min(max_step).max(min_step);

        // Step with the current method
        let step_result = match state.method_type {
            LsodaMethodType::Adams => adams_step(&mut state, &f, &opts, &mut func_evals),
            LsodaMethodType::Bdf => bdf_step(&mut state, &f, &opts, &mut func_evals),
        };

        state.steps += 1;
        state.steps_since_switch += 1;

        match step_result {
            Ok(accepted) => {
                if accepted {
                    // Step accepted

                    // Add to history and results
                    state.add_to_history();
                    t_values.push(state.t);
                    y_values.push(state.y.clone());

                    state.accepted_steps += 1;

                    // Check for method switching if not recently switched
                    if !state.recently_switched {
                        let is_stiff = stiffness_detector.is_stiff(&state);

                        if state.method_type == LsodaMethodType::Adams && is_stiff {
                            // Switch from Adams to Bdf
                            state.switch_method(LsodaMethodType::Bdf);
                        } else if state.method_type == LsodaMethodType::Bdf && !is_stiff {
                            // Switch from Bdf to Adams
                            state.switch_method(LsodaMethodType::Adams);
                        }
                    }

                    // Update tolerance scaling for next step
                    state.update_tol_scale(opts.rtol, opts.atol);

                    // Increment Jacobian age if we're using Bdf
                    if state.method_type == LsodaMethodType::Bdf && state.jacobian.is_some() {
                        state.jacobian_age += 1;
                    }
                } else {
                    // Step rejected
                    state.rejected_steps += 1;
                }
            }
            Err(e) => {
                // Handle specific errors that might indicate stiffness changes
                match &e {
                    IntegrateError::ConvergenceError(msg) if msg.contains("stiff") => {
                        if state.method_type == LsodaMethodType::Adams {
                            // Problem appears to be stiff - switch to Bdf
                            state.stiffness_detected_count += 1;
                            state.switch_method(LsodaMethodType::Bdf);

                            // Reduce step size
                            state.h *= F::from_f64(0.5).unwrap();
                            if state.h < min_step {
                                return Err(IntegrateError::ConvergenceError(
                                    "Step size too small after method switch".to_string(),
                                ));
                            }
                        } else {
                            // Already using Bdf and still failing
                            return Err(e);
                        }
                    }
                    IntegrateError::ConvergenceError(msg) if msg.contains("non-stiff") => {
                        if state.method_type == LsodaMethodType::Bdf {
                            // Problem appears to be non-stiff - switch to Adams
                            state.non_stiffness_detected_count += 1;
                            state.switch_method(LsodaMethodType::Adams);

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
        None
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

/// Take a step using Adams method (predictor-corrector) for non-stiff regions
fn adams_step<F, Func>(
    state: &mut LsodaState<F>,
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
        // Order 8
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
        // Order 5
        vec![
            F::from_f64(251.0 / 720.0).unwrap(),
            F::from_f64(646.0 / 720.0).unwrap(),
            F::from_f64(-264.0 / 720.0).unwrap(),
            F::from_f64(106.0 / 720.0).unwrap(),
            F::from_f64(-19.0 / 720.0).unwrap(),
        ],
        // Rest of coefficients for orders 6-12...
        // Order 6
        vec![
            F::from_f64(475.0 / 1440.0).unwrap(),
            F::from_f64(1427.0 / 1440.0).unwrap(),
            F::from_f64(-798.0 / 1440.0).unwrap(),
            F::from_f64(482.0 / 1440.0).unwrap(),
            F::from_f64(-173.0 / 1440.0).unwrap(),
            F::from_f64(27.0 / 1440.0).unwrap(),
        ],
        // Order 7
        vec![
            F::from_f64(19087.0 / 60480.0).unwrap(),
            F::from_f64(65112.0 / 60480.0).unwrap(),
            F::from_f64(-46461.0 / 60480.0).unwrap(),
            F::from_f64(37504.0 / 60480.0).unwrap(),
            F::from_f64(-20211.0 / 60480.0).unwrap(),
            F::from_f64(6312.0 / 60480.0).unwrap(),
            F::from_f64(-863.0 / 60480.0).unwrap(),
        ],
        // Order 8
        vec![
            F::from_f64(36799.0 / 120960.0).unwrap(),
            F::from_f64(139849.0 / 120960.0).unwrap(),
            F::from_f64(-121797.0 / 120960.0).unwrap(),
            F::from_f64(123133.0 / 120960.0).unwrap(),
            F::from_f64(-88547.0 / 120960.0).unwrap(),
            F::from_f64(41499.0 / 120960.0).unwrap(),
            F::from_f64(-11351.0 / 120960.0).unwrap(),
            F::from_f64(1375.0 / 120960.0).unwrap(),
        ],
        // Order 9
        vec![
            F::from_f64(1070017.0 / 3628800.0).unwrap(),
            F::from_f64(4467094.0 / 3628800.0).unwrap(),
            F::from_f64(-4604594.0 / 3628800.0).unwrap(),
            F::from_f64(5595358.0 / 3628800.0).unwrap(),
            F::from_f64(-5033120.0 / 3628800.0).unwrap(),
            F::from_f64(3146338.0 / 3628800.0).unwrap(),
            F::from_f64(-1291214.0 / 3628800.0).unwrap(),
            F::from_f64(312874.0 / 3628800.0).unwrap(),
            F::from_f64(-33953.0 / 3628800.0).unwrap(),
        ],
        // Order 10
        vec![
            F::from_f64(2082753.0 / 7257600.0).unwrap(),
            F::from_f64(9449717.0 / 7257600.0).unwrap(),
            F::from_f64(-11271304.0 / 7257600.0).unwrap(),
            F::from_f64(16002320.0 / 7257600.0).unwrap(),
            F::from_f64(-17283646.0 / 7257600.0).unwrap(),
            F::from_f64(13510082.0 / 7257600.0).unwrap(),
            F::from_f64(-7394032.0 / 7257600.0).unwrap(),
            F::from_f64(2687864.0 / 7257600.0).unwrap(),
            F::from_f64(-583435.0 / 7257600.0).unwrap(),
            F::from_f64(57281.0 / 7257600.0).unwrap(),
        ],
        // Order 11
        vec![
            F::from_f64(1832380165.0 / 7983360000.0).unwrap(),
            F::from_f64(8862145928.0 / 7983360000.0).unwrap(),
            F::from_f64(-11901858253.0 / 7983360000.0).unwrap(),
            F::from_f64(19151811844.0 / 7983360000.0).unwrap(),
            F::from_f64(-23709112128.0 / 7983360000.0).unwrap(),
            F::from_f64(22186204517.0 / 7983360000.0).unwrap(),
            F::from_f64(-15364126130.0 / 7983360000.0).unwrap(),
            F::from_f64(7503814963.0 / 7983360000.0).unwrap(),
            F::from_f64(-2395311906.0 / 7983360000.0).unwrap(),
            F::from_f64(467772723.0 / 7983360000.0).unwrap(),
            F::from_f64(-41469557.0 / 7983360000.0).unwrap(),
        ],
        // Order 12
        vec![
            F::from_f64(3525779602.0 / 16876492800.0).unwrap(),
            F::from_f64(17870808964.0 / 16876492800.0).unwrap(),
            F::from_f64(-26564533485.0 / 16876492800.0).unwrap(),
            F::from_f64(47566383032.0 / 16876492800.0).unwrap(),
            F::from_f64(-66692205045.0 / 16876492800.0).unwrap(),
            F::from_f64(72077402760.0 / 16876492800.0).unwrap(),
            F::from_f64(-59658274307.0 / 16876492800.0).unwrap(),
            F::from_f64(36174330240.0 / 16876492800.0).unwrap(),
            F::from_f64(-15568150189.0 / 16876492800.0).unwrap(),
            F::from_f64(4443502217.0 / 16876492800.0).unwrap(),
            F::from_f64(-772653805.0 / 16876492800.0).unwrap(),
            F::from_f64(62628216.0 / 16876492800.0).unwrap(),
        ],
    ];

    // We need at least order history points to use the desired order
    let order = state.order.min(state.dy_history.len() + 1).min(12);

    // If we don't have enough history, use lower order
    if order == 1 || state.dy_history.is_empty() {
        // Explicit Euler method (1st order Adams-Bashforth)
        let next_t = state.t + state.h;
        let next_y = state.y.clone() + state.dy.clone() * state.h;

        // Evaluate at the new point
        let next_dy = f(next_t, next_y.view());
        *func_evals += 1;
        state.func_evals += 1;

        // Update state
        state.t = next_t;
        state.y = next_y;
        state.dy = next_dy;

        // Order can now be increased next step
        if state.order < 2 {
            state.order += 1;
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
            ab_sum = ab_sum + state.dy_history[idx].clone() * coeff;
        }
    }

    let y_pred = state.y.clone() + ab_sum * state.h;

    // Evaluate function at the predicted point
    let dy_pred = f(next_t, y_pred.view());
    *func_evals += 1;
    state.func_evals += 1;

    // Adams-Moulton corrector (implicit step)
    let am_coefs = &am_coeffs[order - 1];

    // Apply Adams-Moulton formula to correct the prediction
    // y_{n+1} = y_n + h * (b_0 * f_{n+1} + sum(b_i * f_{n-i+1}))
    let mut am_sum = dy_pred.clone() * am_coefs[0]; // f_{n+1} term

    for (i, &coeff) in am_coefs.iter().enumerate().take(order).skip(1) {
        if i == 1 {
            // Current derivative (f_n)
            am_sum = am_sum + state.dy.clone() * coeff;
        } else if i - 1 < state.dy_history.len() {
            // Historical derivatives (f_{n-1}, f_{n-2}, ...)
            let idx = state.dy_history.len() - (i - 1);
            am_sum = am_sum + state.dy_history[idx].clone() * coeff;
        }
    }

    let y_corr = state.y.clone() + am_sum * state.h;

    // Evaluate function at the corrected point
    let dy_corr = f(next_t, y_corr.view());
    *func_evals += 1;
    state.func_evals += 1;

    // Error estimation based on predictor-corrector difference
    let mut max_err = F::zero();
    for i in 0..state.y.len() {
        let err = (y_corr[i] - y_pred[i]).abs();
        let scale = state.tol_scale[i];
        max_err = max_err.max(err / scale);
    }

    // Step size adjustment factor based on error
    let err_order = order + 1; // Error order is one higher than method order
    let err_factor = if max_err > F::zero() {
        F::from_f64(0.9).unwrap()
            * (F::one() / max_err).powf(F::one() / F::from_usize(err_order).unwrap())
    } else {
        F::from_f64(5.0).unwrap() // Max increase if error is zero
    };

    // Safety factor and limits for step size adjustment
    let safety = F::from_f64(0.9).unwrap();
    let factor_max = F::from_f64(5.0).unwrap();
    let factor_min = F::from_f64(0.2).unwrap();
    let factor = safety * err_factor.min(factor_max).max(factor_min);

    // Check if step is acceptable
    if max_err <= F::one() {
        // Step accepted

        // Update state
        state.t = next_t;
        state.y = y_corr;
        state.dy = dy_corr;

        // Update step size for next step
        state.h *= factor;

        // Order adaptation (simplified)
        if order < 12 && max_err < opts.rtol && state.dy_history.len() >= order {
            state.order = (state.order + 1).min(12);
        } else if order > 1 && max_err > F::from_f64(0.5).unwrap() {
            state.order = (state.order - 1).max(1);
        }

        // If solution appears very smooth, this indicates non-stiffness
        if max_err < opts.rtol * F::from_f64(0.01).unwrap() {
            state.non_stiffness_detected_count += 1;
        }

        Ok(true)
    } else {
        // Step rejected

        // Adjust step size for retry
        state.h *= factor;

        // If error is very large, this might indicate stiffness
        if max_err > F::from_f64(10.0).unwrap() {
            state.stiffness_detected_count += 1;

            // If stiffness is consistently detected, suggest switching
            if state.stiffness_detected_count > 2 {
                return Err(IntegrateError::ConvergenceError(
                    "Problem appears stiff - consider using Bdf method".to_string(),
                ));
            }
        }

        Ok(false)
    }
}

/// Take a step using Bdf method for stiff regions
fn bdf_step<F, Func>(
    state: &mut LsodaState<F>,
    f: &Func,
    opts: &ODEOptions<F>,
    func_evals: &mut usize,
) -> IntegrateResult<bool>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Coefficients for Bdf methods of different orders
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
    let order = state.order.min(state.y_history.len()).min(5);

    // If we don't have enough history for the requested order, use lower order
    if order == 1 || state.y_history.is_empty() {
        // Implicit Euler method (1st order Bdf)
        let next_t = state.t + state.h;

        // Predict the next value (simple extrapolation)
        let y_pred = state.y.clone();

        // Newton's method for solving the implicit equation
        let max_newton_iters = 10;
        let newton_tol = F::from_f64(1e-8).unwrap();
        let mut y_next = y_pred.clone();
        let mut converged = false;
        let mut iter_count = 0;

        while iter_count < max_newton_iters {
            // Evaluate function at current iterate
            let f_eval = f(next_t, y_next.view());
            *func_evals += 1;
            state.func_evals += 1;

            // Compute residual for BDF1: y_{n+1} - y_n - h * f(t_{n+1}, y_{n+1}) = 0
            let residual = y_next.clone() - state.y.clone() - f_eval.clone() * state.h;

            // Check convergence
            let mut max_res = F::zero();
            for i in 0..y_next.len() {
                let scale = state.tol_scale[i];
                max_res = max_res.max(residual[i].abs() / scale);
            }

            if max_res <= newton_tol {
                converged = true;
                break;
            }

            // Compute or reuse Jacobian
            let eps = F::from_f64(1e-8).unwrap();
            let n_dim = y_next.len();
            let mut jacobian = Array2::<F>::zeros((n_dim, n_dim));

            // Create approximate Jacobian using finite differences if needed
            let compute_new_jacobian =
                state.jacobian.is_none() || state.jacobian_age > 20 || iter_count == 0;

            if compute_new_jacobian {
                state.n_jac += 1;

                for i in 0..n_dim {
                    let mut y_perturbed = y_next.clone();
                    y_perturbed[i] += eps;

                    let f_perturbed = f(next_t, y_perturbed.view());
                    *func_evals += 1;
                    state.func_evals += 1;

                    for j in 0..n_dim {
                        // Finite difference approximation of df_j/dy_i
                        let df_dy = (f_perturbed[j] - f_eval[j]) / eps;

                        // J_{ji} = δ_{ji} - h * df_j/dy_i
                        jacobian[[j, i]] = if i == j {
                            F::one() - state.h * df_dy
                        } else {
                            -state.h * df_dy
                        };
                    }
                }

                // Store the Jacobian for potential reuse
                state.jacobian = Some(jacobian.clone());
                state.jacobian_age = 0;
            } else {
                // Reuse previous Jacobian
                jacobian = state.jacobian.clone().unwrap();
            }

            // Solve the linear system J*delta_y = residual using Gaussian elimination
            let mut aug = Array2::<F>::zeros((n_dim, n_dim + 1));
            for i in 0..n_dim {
                for j in 0..n_dim {
                    aug[[i, j]] = jacobian[[i, j]];
                }
                aug[[i, n_dim]] = residual[i];
            }

            state.n_lu += 1;

            // Gaussian elimination with partial pivoting
            for i in 0..n_dim {
                // Find pivot
                let mut max_idx = i;
                let mut max_val = aug[[i, i]].abs();

                for j in i + 1..n_dim {
                    if aug[[j, i]].abs() > max_val {
                        max_idx = j;
                        max_val = aug[[j, i]].abs();
                    }
                }

                // Check if matrix is singular
                if max_val < F::from_f64(1e-10).unwrap() {
                    // Nearly singular, reduce step size and try again
                    state.h *= F::from_f64(0.5).unwrap();
                    return Ok(false);
                }

                // Swap rows if necessary
                if max_idx != i {
                    for j in 0..n_dim + 1 {
                        let temp = aug[[i, j]];
                        aug[[i, j]] = aug[[max_idx, j]];
                        aug[[max_idx, j]] = temp;
                    }
                }

                // Eliminate below
                for j in i + 1..n_dim {
                    let factor = aug[[j, i]] / aug[[i, i]];
                    for k in i..n_dim + 1 {
                        aug[[j, k]] = aug[[j, k]] - factor * aug[[i, k]];
                    }
                }
            }

            // Back substitution
            let mut delta_y = Array1::<F>::zeros(n_dim);
            for i in (0..n_dim).rev() {
                let mut sum = aug[[i, n_dim]];
                for j in i + 1..n_dim {
                    sum -= aug[[i, j]] * delta_y[j];
                }
                delta_y[i] = sum / aug[[i, i]];
            }

            // Update solution
            for i in 0..n_dim {
                y_next[i] -= delta_y[i];
            }

            iter_count += 1;
        }

        if !converged {
            // Newton iteration failed, reduce step size
            state.h *= F::from_f64(0.5).unwrap();

            // If we've reduced step size too much, the problem might be non-stiff
            // or our initial guess might be poor
            if state.h < opts.min_step.unwrap_or(F::from_f64(1e-10).unwrap()) {
                // Track non-stiffness indicators
                state.non_stiffness_detected_count += 1;

                return Err(IntegrateError::ConvergenceError(
                    "BDF1 failed to converge - problem might be non-stiff".to_string(),
                ));
            }

            return Ok(false);
        }

        // Step accepted

        // Evaluate derivative at the new point (for potential reuse in next step)
        let next_dy = f(next_t, y_next.view());
        *func_evals += 1;
        state.func_evals += 1;

        // Update state
        state.t = next_t;
        state.y = y_next;
        state.dy = next_dy;

        // Order can now be increased next step
        if state.order < 2 {
            state.order += 1;
        }

        return Ok(true);
    }

    // Higher-order Bdf methods (2-5)

    // Get Bdf coefficients for the current order
    let coeffs = &bdf_coefs[order - 1];

    // Next time and step size
    let next_t = state.t + state.h;

    // Predict initial value using extrapolation from previous points
    let mut y_pred = state.y.clone();

    // For higher orders, use previous points for prediction
    if order > 1 && !state.y_history.is_empty() {
        let y_prev = &state.y_history[state.y_history.len() - 1];

        // Basic extrapolation
        let dt_ratio = state.h / (state.t - state.t_history[state.t_history.len() - 1]);
        y_pred = state.y.clone() + (state.y.clone() - y_prev) * dt_ratio;
    }

    // Newton's method for solving the Bdf equation
    let max_newton_iters = 10;
    let newton_tol = F::from_f64(1e-8).unwrap();
    let mut y_next = y_pred.clone();
    let mut converged = false;
    let mut iter_count = 0;

    while iter_count < max_newton_iters {
        // Evaluate function at current iterate
        let f_eval = f(next_t, y_next.view());
        *func_evals += 1;
        state.func_evals += 1;

        // Compute residual for Bdf: c_0 * y_{n+1} - sum(c_j * y_{n+1-j}) - h * f(t_{n+1}, y_{n+1}) = 0
        let mut residual = y_next.clone() * coeffs[0];

        // Subtract previous terms
        residual = residual - state.y.clone() * coeffs[1];

        for (j, &coeff) in coeffs.iter().enumerate().skip(2) {
            if j - 1 < state.y_history.len() {
                let idx = state.y_history.len() - (j - 1);
                residual = residual - state.y_history[idx].clone() * coeff;
            }
        }

        // Subtract h * f term
        residual = residual - f_eval.clone() * state.h;

        // Check convergence
        let mut max_res = F::zero();
        for i in 0..y_next.len() {
            let scale = state.tol_scale[i];
            max_res = max_res.max(residual[i].abs() / scale);
        }

        if max_res <= newton_tol {
            converged = true;
            break;
        }

        // Compute or reuse Jacobian
        let eps = F::from_f64(1e-8).unwrap();
        let n_dim = y_next.len();
        let mut jacobian = Array2::<F>::zeros((n_dim, n_dim));

        // Create approximate Jacobian using finite differences if needed
        let compute_new_jacobian =
            state.jacobian.is_none() || state.jacobian_age > 20 || iter_count == 0;

        if compute_new_jacobian {
            state.n_jac += 1;

            for i in 0..n_dim {
                let mut y_perturbed = y_next.clone();
                y_perturbed[i] += eps;

                let f_perturbed = f(next_t, y_perturbed.view());
                *func_evals += 1;
                state.func_evals += 1;

                for j in 0..n_dim {
                    // Finite difference approximation of df_j/dy_i
                    let df_dy = (f_perturbed[j] - f_eval[j]) / eps;

                    // J_{ji} = c_0 * δ_{ji} - h * df_j/dy_i
                    jacobian[[j, i]] = if i == j {
                        coeffs[0] - state.h * df_dy
                    } else {
                        -state.h * df_dy
                    };
                }
            }

            // Store the Jacobian for potential reuse
            state.jacobian = Some(jacobian.clone());
            state.jacobian_age = 0;
        } else {
            // Reuse previous Jacobian
            jacobian = state.jacobian.clone().unwrap();
        }

        // Solve the linear system J*delta_y = residual using Gaussian elimination
        state.n_lu += 1;

        let mut aug = Array2::<F>::zeros((n_dim, n_dim + 1));
        for i in 0..n_dim {
            for j in 0..n_dim {
                aug[[i, j]] = jacobian[[i, j]];
            }
            aug[[i, n_dim]] = residual[i];
        }

        // Gaussian elimination with partial pivoting
        for i in 0..n_dim {
            // Find pivot
            let mut max_idx = i;
            let mut max_val = aug[[i, i]].abs();

            for j in i + 1..n_dim {
                if aug[[j, i]].abs() > max_val {
                    max_idx = j;
                    max_val = aug[[j, i]].abs();
                }
            }

            // Check if matrix is singular
            if max_val < F::from_f64(1e-10).unwrap() {
                // Nearly singular, reduce step size and try again
                state.h *= F::from_f64(0.5).unwrap();
                return Ok(false);
            }

            // Swap rows if necessary
            if max_idx != i {
                for j in 0..n_dim + 1 {
                    let temp = aug[[i, j]];
                    aug[[i, j]] = aug[[max_idx, j]];
                    aug[[max_idx, j]] = temp;
                }
            }

            // Eliminate below
            for j in i + 1..n_dim {
                let factor = aug[[j, i]] / aug[[i, i]];
                for k in i..n_dim + 1 {
                    aug[[j, k]] = aug[[j, k]] - factor * aug[[i, k]];
                }
            }
        }

        // Back substitution
        let mut delta_y = Array1::<F>::zeros(n_dim);
        for i in (0..n_dim).rev() {
            let mut sum = aug[[i, n_dim]];
            for j in i + 1..n_dim {
                sum -= aug[[i, j]] * delta_y[j];
            }
            delta_y[i] = sum / aug[[i, i]];
        }

        // Update solution
        for i in 0..n_dim {
            y_next[i] -= delta_y[i];
        }

        iter_count += 1;
    }

    if !converged {
        // Newton iteration failed, reduce step size
        state.h *= F::from_f64(0.5).unwrap();

        // If the problem is consistently difficult to solve, it might not be stiff
        if iter_count >= max_newton_iters - 1 {
            state.non_stiffness_detected_count += 1;
        }

        // If we've reduced step size too much, the problem might not be stiff
        if state.h < opts.min_step.unwrap_or(F::from_f64(1e-10).unwrap()) {
            return Err(IntegrateError::ConvergenceError(
                "Bdf failed to converge - problem might be non-stiff".to_string(),
            ));
        }

        return Ok(false);
    }

    // Step accepted

    // Evaluate derivative at the new point (for reuse in future steps)
    let next_dy = f(next_t, y_next.view());
    *func_evals += 1;
    state.func_evals += 1;

    // Error estimation using comparison with lower order method
    // This is a simplified implementation

    // Compute error estimate by comparing with lower order solution
    let lower_order = (order - 1).max(1);
    let lower_coeffs = &bdf_coefs[lower_order - 1];

    // Approximate lower order solution
    let mut y_lower = Array1::<F>::zeros(y_next.len());

    // Compute the values for the lower order solution
    let mut rhs = Array1::<F>::zeros(y_next.len());
    for (j, &coeff) in lower_coeffs.iter().enumerate().skip(1).take(lower_order) {
        if j == 1 {
            rhs = rhs + state.y.clone() * coeff;
        } else if j - 1 < state.y_history.len() {
            let idx = state.y_history.len() - (j - 1);
            rhs = rhs + state.y_history[idx].clone() * coeff;
        }
    }

    // Add h * f term
    rhs = rhs + next_dy.clone() * state.h;

    // Solve for y_(n+1) with lower order formula
    for i in 0..y_next.len() {
        y_lower[i] = rhs[i] / lower_coeffs[0];
    }

    // Error estimate is the difference between solutions of different orders
    let mut max_err = F::zero();
    for i in 0..y_next.len() {
        let local_err = (y_next[i] - y_lower[i]).abs();
        let scale = state.tol_scale[i];
        max_err = max_err.max(local_err / scale);
    }

    // Step size adjustment factor based on error
    let err_order = order + 1; // Error order is one higher than method order
    let err_factor = if max_err > F::zero() {
        F::from_f64(0.9).unwrap()
            * (F::one() / max_err).powf(F::one() / F::from_usize(err_order).unwrap())
    } else {
        F::from_f64(5.0).unwrap() // Max increase if error is zero
    };

    // Safety factor and limits for step size adjustment
    let safety = F::from_f64(0.9).unwrap();
    let factor_max = F::from_f64(5.0).unwrap();
    let factor_min = F::from_f64(0.2).unwrap();
    let factor = safety * err_factor.min(factor_max).max(factor_min);

    // Update state
    state.t = next_t;
    state.y = y_next;
    state.dy = next_dy;

    // Update step size for next step
    state.h *= factor;

    // Order adaptation (simplified)
    // Increase order if we're converging well and have enough history
    if order < 5 && max_err < opts.rtol && state.y_history.len() >= order {
        state.order = (state.order + 1).min(5);
    }
    // Decrease order if we're struggling to converge
    else if order > 1 && (max_err > F::from_f64(0.5).unwrap() || iter_count > 2) {
        state.order = (state.order - 1).max(1);
    }

    // Successful step with minimum Newton iterations might indicate non-stiffness
    if iter_count <= 2 {
        state.non_stiffness_detected_count += 1;
    }

    // Increment Jacobian age
    state.jacobian_age += 1;

    Ok(true)
}
