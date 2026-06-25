//! Delay Differential Equation (DDE) solvers
//!
//! This module provides numerical methods for solving delay differential equations
//! of the form:
//!
//!   y'(t) = f(t, y(t), y(t - tau_1), y(t - tau_2), ...)
//!
//! where tau_i are the delays. The history function phi(t) for t <= t0 must be provided.
//!
//! # Features
//!
//! - **Method of steps**: Integrates interval-by-interval, using the known
//!   solution on previous intervals to evaluate delayed terms.
//! - **Dense output interpolation**: Hermite cubic interpolation for evaluating
//!   the solution at arbitrary points in the history.
//! - **Multiple delays**: Supports any number of constant delays.
//! - **State-dependent delays**: Delays that depend on the current state y(t).
//! - **Discontinuity tracking**: Automatically tracks and resolves discontinuities
//!   propagated by the delays.
//!
//! # References
//!
//! - Bellen & Zennaro: "Numerical Methods for Delay Differential Equations" (2003)
//! - Shampine & Thompson: "Solving DDEs in Matlab" (2001)
//! - Baker, Paul & Willé: "Issues in the numerical solution of DDEs" (1995)

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{array, Array1, ArrayView1};
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Specification of delays in a DDE system.
#[derive(Debug, Clone)]
pub enum DelayType<F: IntegrateFloat> {
    /// Constant delays: y(t - tau_i) for fixed tau_i > 0
    Constant(Vec<F>),
    /// State-dependent delays: tau_i = tau_i(t, y(t))
    /// The function returns the vector of delay values.
    StateDependent,
}

/// Options for the DDE solver.
#[derive(Debug, Clone)]
pub struct DDEOptions<F: IntegrateFloat> {
    /// Relative tolerance
    pub rtol: F,
    /// Absolute tolerance
    pub atol: F,
    /// Initial step size (None for automatic)
    pub h0: Option<F>,
    /// Maximum step size
    pub max_step: Option<F>,
    /// Minimum step size
    pub min_step: Option<F>,
    /// Maximum number of steps
    pub max_steps: usize,
    /// Whether to track discontinuities
    pub track_discontinuities: bool,
    /// Maximum discontinuity order to track (0 = jump, 1 = corner, etc.)
    pub max_discontinuity_order: usize,
}

impl<F: IntegrateFloat> Default for DDEOptions<F> {
    fn default() -> Self {
        DDEOptions {
            rtol: F::from_f64(1e-6).unwrap_or_else(|| F::epsilon()),
            atol: F::from_f64(1e-9).unwrap_or_else(|| F::epsilon()),
            h0: None,
            max_step: None,
            min_step: None,
            max_steps: 100_000,
            track_discontinuities: true,
            max_discontinuity_order: 5,
        }
    }
}

/// Result of DDE integration.
#[derive(Debug, Clone)]
pub struct DDEResult<F: IntegrateFloat> {
    /// Time points
    pub t: Vec<F>,
    /// Solution values at each time point
    pub y: Vec<Array1<F>>,
    /// Whether integration completed successfully
    pub success: bool,
    /// Status message
    pub message: Option<String>,
    /// Number of function evaluations
    pub n_eval: usize,
    /// Number of steps taken
    pub n_steps: usize,
    /// Number of accepted steps
    pub n_accepted: usize,
    /// Number of rejected steps
    pub n_rejected: usize,
    /// Detected discontinuity times
    pub discontinuities: Vec<F>,
}

// ---------------------------------------------------------------------------
// Dense output (Hermite cubic interpolation)
// ---------------------------------------------------------------------------

/// A segment of dense output for Hermite interpolation on [t_left, t_right].
#[derive(Debug, Clone)]
struct DenseSegment<F: IntegrateFloat> {
    t_left: F,
    t_right: F,
    y_left: Array1<F>,
    y_right: Array1<F>,
    yp_left: Array1<F>,
    yp_right: Array1<F>,
}

impl<F: IntegrateFloat> DenseSegment<F> {
    /// Evaluate the Hermite interpolant at time t in [t_left, t_right].
    fn evaluate(&self, t: F) -> Array1<F> {
        let h = self.t_right - self.t_left;
        if h.abs() < F::from_f64(1e-30).unwrap_or_else(|| F::epsilon()) {
            return self.y_left.clone();
        }

        let s = (t - self.t_left) / h;
        let s2 = s * s;
        let s3 = s2 * s;

        // Hermite basis functions: h00(s) = 2s^3 - 3s^2 + 1,  h10(s) = s^3 - 2s^2 + s
        //                          h01(s) = -2s^3 + 3s^2,     h11(s) = s^3 - s^2
        let two = F::one() + F::one();
        let three = two + F::one();
        let h00 = two * s3 - three * s2 + F::one();
        let h10 = s3 - two * s2 + s;
        let h01 = three * s2 - two * s3;
        let h11 = s3 - s2;

        &self.y_left * h00
            + &(&self.yp_left * (h * h10))
            + &(&self.y_right * h01)
            + &(&self.yp_right * (h * h11))
    }
}

/// History storage and interpolation for delayed values.
#[derive(Debug, Clone)]
struct HistoryBuffer<F: IntegrateFloat> {
    /// Dense output segments, ordered by time
    segments: Vec<DenseSegment<F>>,
    /// History function for t <= t0
    /// Stored as a vector of (t, y) pairs for the pre-initial-time history
    pre_history: Vec<(F, Array1<F>)>,
    /// The earliest time in the solution buffer
    t_start: F,
}

impl<F: IntegrateFloat> HistoryBuffer<F> {
    fn new(t0: F) -> Self {
        HistoryBuffer {
            segments: Vec::new(),
            pre_history: Vec::new(),
            t_start: t0,
        }
    }

    /// Add pre-history samples (for t <= t0)
    fn add_pre_history(&mut self, t: F, y: Array1<F>) {
        self.pre_history.push((t, y));
        // Keep sorted
        self.pre_history
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Add a new dense output segment
    fn add_segment(&mut self, seg: DenseSegment<F>) {
        self.segments.push(seg);
    }

    /// Evaluate the history/solution at time t
    fn evaluate(&self, t: F) -> IntegrateResult<Array1<F>> {
        // Check if t falls in the pre-history
        if t < self.t_start || (t <= self.t_start && self.segments.is_empty()) {
            return self.evaluate_pre_history(t);
        }

        // Find the segment containing t
        for seg in &self.segments {
            if t >= seg.t_left && t <= seg.t_right {
                return Ok(seg.evaluate(t));
            }
        }

        // If t is at the very end, use the last segment
        if let Some(last) = self.segments.last() {
            if (t - last.t_right).abs() < F::from_f64(1e-12).unwrap_or_else(|| F::epsilon()) {
                return Ok(last.y_right.clone());
            }
        }

        Err(IntegrateError::ValueError(format!(
            "Time {t} is outside the computed solution range"
        )))
    }

    /// Evaluate pre-history via linear interpolation of stored samples
    fn evaluate_pre_history(&self, t: F) -> IntegrateResult<Array1<F>> {
        if self.pre_history.is_empty() {
            return Err(IntegrateError::ValueError(
                "No pre-history available for the requested time".into(),
            ));
        }

        // If only one point, return it
        if self.pre_history.len() == 1 {
            return Ok(self.pre_history[0].1.clone());
        }

        // Clamp to the range of pre-history
        let first_t = self.pre_history[0].0;
        let last_t = self.pre_history[self.pre_history.len() - 1].0;

        if t <= first_t {
            return Ok(self.pre_history[0].1.clone());
        }
        if t >= last_t {
            return Ok(self.pre_history[self.pre_history.len() - 1].1.clone());
        }

        // Find bracketing interval
        for i in 0..self.pre_history.len() - 1 {
            let (t_i, ref y_i) = self.pre_history[i];
            let (t_ip1, ref y_ip1) = self.pre_history[i + 1];

            if t >= t_i && t <= t_ip1 {
                let dt = t_ip1 - t_i;
                if dt.abs() < F::from_f64(1e-30).unwrap_or_else(|| F::epsilon()) {
                    return Ok(y_i.clone());
                }
                let s = (t - t_i) / dt;
                return Ok(y_i * (F::one() - s) + y_ip1 * s);
            }
        }

        Ok(self.pre_history[self.pre_history.len() - 1].1.clone())
    }
}

// ---------------------------------------------------------------------------
// Discontinuity tracker
// ---------------------------------------------------------------------------

/// Tracks discontinuities propagated by delays.
///
/// If the history function has a discontinuity at t0, the DDE solution will
/// have a derivative discontinuity at t0 + tau, and a higher-order
/// discontinuity at t0 + 2*tau, etc. The solver must step exactly to
/// these points for accuracy.
#[derive(Debug, Clone)]
struct DiscontinuityTracker<F: IntegrateFloat> {
    /// Queue of discontinuity times, sorted in ascending order
    queue: Vec<F>,
    /// Maximum order of discontinuity to track
    max_order: usize,
}

impl<F: IntegrateFloat> DiscontinuityTracker<F> {
    fn new(max_order: usize) -> Self {
        DiscontinuityTracker {
            queue: Vec::new(),
            max_order,
        }
    }

    /// Seed the tracker with initial discontinuity at t0 and constant delays.
    fn seed(&mut self, t0: F, tf: F, delays: &[F]) {
        // The initial time t0 is a discontinuity in the derivative
        // It propagates through the delays: t0 + k*tau for each delay tau
        for order in 0..=self.max_order {
            for tau in delays {
                let disc_t = t0 + F::from_usize(order + 1).unwrap_or_else(|| F::one()) * (*tau);
                if disc_t > t0 && disc_t <= tf {
                    self.queue.push(disc_t);
                }
            }
        }

        // Sort and deduplicate
        self.queue
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.queue
            .dedup_by(|a, b| (*a - *b).abs() < F::from_f64(1e-12).unwrap_or_else(|| F::epsilon()));
    }

    /// Get the next discontinuity time after `t_current`.
    fn next_after(&self, t_current: F) -> Option<F> {
        let eps = F::from_f64(1e-12).unwrap_or_else(|| F::epsilon());
        self.queue.iter().find(|&&td| td > t_current + eps).copied()
    }

    /// Get all tracked discontinuity times.
    fn all_times(&self) -> &[F] {
        &self.queue
    }
}

// ---------------------------------------------------------------------------
// DDE System trait
// ---------------------------------------------------------------------------

/// Trait for a DDE right-hand side function.
///
/// The user implements this to define f(t, y(t), y(t-tau_1), ..., y(t-tau_k)).
pub trait DDESystem<F: IntegrateFloat> {
    /// Dimension of the system
    fn ndim(&self) -> usize;

    /// Evaluate the right-hand side.
    ///
    /// * `t` - current time
    /// * `y` - current state y(t)
    /// * `y_delayed` - delayed states [y(t-tau_1), y(t-tau_2), ...]
    fn rhs(&self, t: F, y: ArrayView1<F>, y_delayed: &[Array1<F>]) -> IntegrateResult<Array1<F>>;

    /// Constant delays (if DelayType::Constant)
    fn delays(&self) -> Vec<F>;

    /// State-dependent delays: given (t, y), return the delay values.
    /// Default: calls `self.delays()` ignoring state.
    fn state_dependent_delays(&self, _t: F, _y: ArrayView1<F>) -> Vec<F> {
        self.delays()
    }

    /// History function phi(t) for t <= t0.
    fn history(&self, t: F) -> Array1<F>;
}

// ---------------------------------------------------------------------------
// Method of Steps solver (RK45 based)
// ---------------------------------------------------------------------------

/// Solve a DDE using the method of steps with an embedded RK45 integrator.
///
/// # Arguments
/// * `sys` - the DDE system
/// * `t_span` - [t0, tf]
/// * `options` - solver options
///
/// # Returns
/// A `DDEResult` with the solution trajectory
pub fn solve_dde<F: IntegrateFloat>(
    sys: &dyn DDESystem<F>,
    t_span: [F; 2],
    options: &DDEOptions<F>,
) -> IntegrateResult<DDEResult<F>> {
    let t0 = t_span[0];
    let tf = t_span[1];
    let n = sys.ndim();

    if tf <= t0 {
        return Err(IntegrateError::ValueError("tf must be > t0".into()));
    }

    let delays = sys.delays();
    if delays.is_empty() {
        return Err(IntegrateError::ValueError(
            "DDE system must have at least one delay".into(),
        ));
    }

    // Validate delays
    for (i, &tau) in delays.iter().enumerate() {
        if tau <= F::zero() {
            return Err(IntegrateError::ValueError(format!(
                "Delay {} must be positive, got {tau}",
                i
            )));
        }
    }

    // Initialize history buffer
    let mut history = HistoryBuffer::new(t0);

    // Sample the history function on a grid before t0
    let max_delay = delays.iter().fold(F::zero(), |a, &b| a.max(b));
    let n_pre_samples = 100;
    let pre_dt = max_delay / F::from_usize(n_pre_samples).unwrap_or_else(|| F::one());
    for i in 0..=n_pre_samples {
        let t_pre = t0 - max_delay + F::from_usize(i).unwrap_or_else(|| F::zero()) * pre_dt;
        let y_pre = sys.history(t_pre);
        history.add_pre_history(t_pre, y_pre);
    }

    // Initialize discontinuity tracker
    let mut disc_tracker = DiscontinuityTracker::new(if options.track_discontinuities {
        options.max_discontinuity_order
    } else {
        0
    });
    if options.track_discontinuities {
        disc_tracker.seed(t0, tf, &delays);
    }

    // Initial state from history
    let y0 = sys.history(t0);

    // Step size
    let span = tf - t0;
    let mut h = match options.h0 {
        Some(h0) => h0,
        None => {
            let h_init = span * F::from_f64(0.001).unwrap_or_else(|| F::epsilon());
            if let Some(max_h) = options.max_step {
                h_init.min(max_h)
            } else {
                h_init
            }
        }
    };

    let min_step = options
        .min_step
        .unwrap_or_else(|| span * F::from_f64(1e-14).unwrap_or_else(|| F::epsilon()));

    // Result storage
    let mut t_out = vec![t0];
    let mut y_out = vec![y0.clone()];
    let mut disc_times: Vec<F> = Vec::new();

    let mut t = t0;
    let mut y = y0;
    let mut n_eval: usize = 0;
    let mut n_steps: usize = 0;
    let mut n_accepted: usize = 0;
    let mut n_rejected: usize = 0;

    let safety = F::from_f64(0.9).unwrap_or_else(|| F::one());
    let fac_min = F::from_f64(0.2).unwrap_or_else(|| F::one());
    let fac_max = F::from_f64(5.0).unwrap_or_else(|| F::one());

    while t < tf && n_steps < options.max_steps {
        // Clamp step to not overshoot tf
        if t + h > tf {
            h = tf - t;
        }

        // Clamp step to next discontinuity
        if let Some(t_disc) = disc_tracker.next_after(t) {
            if t + h > t_disc {
                h = t_disc - t;
                // Record this discontinuity
                let eps = F::from_f64(1e-12).unwrap_or_else(|| F::epsilon());
                if (t + h - t_disc).abs() < eps {
                    disc_times.push(t_disc);
                }
            }
        }

        if h < min_step {
            break;
        }

        // RK45 step with delay evaluation
        let step_result = rk45_dde_step(sys, &history, t, h, &y, n, options.rtol, options.atol)?;
        n_eval += 6;

        let err_norm = step_result.error_norm;

        if err_norm <= F::one() {
            // Accept step
            let t_new = t + h;

            // Compute derivative at old and new points for dense output
            let yp_old = evaluate_rhs(sys, &history, t, &y)?;
            let yp_new = evaluate_rhs(sys, &history, t_new, &step_result.y_new)?;

            // Store dense output segment
            history.add_segment(DenseSegment {
                t_left: t,
                t_right: t_new,
                y_left: y.clone(),
                y_right: step_result.y_new.clone(),
                yp_left: yp_old,
                yp_right: yp_new,
            });

            t = t_new;
            y = step_result.y_new;
            n_accepted += 1;

            t_out.push(t);
            y_out.push(y.clone());
        } else {
            n_rejected += 1;
        }

        // Adjust step size
        let factor = if err_norm > F::zero() {
            safety
                * (F::one() / err_norm)
                    .powf(F::one() / F::from_f64(5.0).unwrap_or_else(|| F::one()))
        } else {
            fac_max
        };
        let factor = factor.max(fac_min).min(fac_max);
        h *= factor;

        if let Some(max_h) = options.max_step {
            h = h.min(max_h);
        }

        n_steps += 1;
    }

    Ok(DDEResult {
        t: t_out,
        y: y_out,
        success: t >= tf - min_step,
        message: if t >= tf - min_step {
            Some("DDE integration completed successfully".into())
        } else {
            Some(format!("DDE integration stopped at t = {t}"))
        },
        n_eval,
        n_steps,
        n_accepted,
        n_rejected,
        discontinuities: disc_times,
    })
}

// ---------------------------------------------------------------------------
// RK45 step for DDE
// ---------------------------------------------------------------------------

struct RK45StepResult<F: IntegrateFloat> {
    y_new: Array1<F>,
    error_norm: F,
}

/// Perform one RK45 (Dormand-Prince) step with delay evaluation.
fn rk45_dde_step<F: IntegrateFloat>(
    sys: &dyn DDESystem<F>,
    history: &HistoryBuffer<F>,
    t: F,
    h: F,
    y: &Array1<F>,
    n: usize,
    rtol: F,
    atol: F,
) -> IntegrateResult<RK45StepResult<F>> {
    // Dormand-Prince coefficients
    let a21 = F::from_f64(1.0 / 5.0).unwrap_or_else(|| F::zero());
    let a31 = F::from_f64(3.0 / 40.0).unwrap_or_else(|| F::zero());
    let a32 = F::from_f64(9.0 / 40.0).unwrap_or_else(|| F::zero());
    let a41 = F::from_f64(44.0 / 45.0).unwrap_or_else(|| F::zero());
    let a42 = F::from_f64(-56.0 / 15.0).unwrap_or_else(|| F::zero());
    let a43 = F::from_f64(32.0 / 9.0).unwrap_or_else(|| F::zero());
    let a51 = F::from_f64(19372.0 / 6561.0).unwrap_or_else(|| F::zero());
    let a52 = F::from_f64(-25360.0 / 2187.0).unwrap_or_else(|| F::zero());
    let a53 = F::from_f64(64448.0 / 6561.0).unwrap_or_else(|| F::zero());
    let a54 = F::from_f64(-212.0 / 729.0).unwrap_or_else(|| F::zero());
    let a61 = F::from_f64(9017.0 / 3168.0).unwrap_or_else(|| F::zero());
    let a62 = F::from_f64(-355.0 / 33.0).unwrap_or_else(|| F::zero());
    let a63 = F::from_f64(46732.0 / 5247.0).unwrap_or_else(|| F::zero());
    let a64 = F::from_f64(49.0 / 176.0).unwrap_or_else(|| F::zero());
    let a65 = F::from_f64(-5103.0 / 18656.0).unwrap_or_else(|| F::zero());

    // 5th order weights
    let b1 = F::from_f64(35.0 / 384.0).unwrap_or_else(|| F::zero());
    let b3 = F::from_f64(500.0 / 1113.0).unwrap_or_else(|| F::zero());
    let b4 = F::from_f64(125.0 / 192.0).unwrap_or_else(|| F::zero());
    let b5 = F::from_f64(-2187.0 / 6784.0).unwrap_or_else(|| F::zero());
    let b6 = F::from_f64(11.0 / 84.0).unwrap_or_else(|| F::zero());

    // 4th order weights (for error estimate)
    let e1 = F::from_f64(71.0 / 57600.0).unwrap_or_else(|| F::zero());
    let e3 = F::from_f64(-71.0 / 16695.0).unwrap_or_else(|| F::zero());
    let e4 = F::from_f64(71.0 / 1920.0).unwrap_or_else(|| F::zero());
    let e5 = F::from_f64(-17253.0 / 339200.0).unwrap_or_else(|| F::zero());
    let e6 = F::from_f64(22.0 / 525.0).unwrap_or_else(|| F::zero());
    let e7 = F::from_f64(-1.0 / 40.0).unwrap_or_else(|| F::zero());

    // Node points
    let c2 = F::from_f64(1.0 / 5.0).unwrap_or_else(|| F::zero());
    let c3 = F::from_f64(3.0 / 10.0).unwrap_or_else(|| F::zero());
    let c4 = F::from_f64(4.0 / 5.0).unwrap_or_else(|| F::zero());
    let c5 = F::from_f64(8.0 / 9.0).unwrap_or_else(|| F::zero());

    // Stage 1
    let k1 = evaluate_rhs(sys, history, t, y)?;

    // Stage 2
    let y2 = y + &(&k1 * (h * a21));
    let k2 = evaluate_rhs(sys, history, t + c2 * h, &y2)?;

    // Stage 3
    let y3 = y + &(&k1 * (h * a31) + &k2 * (h * a32));
    let k3 = evaluate_rhs(sys, history, t + c3 * h, &y3)?;

    // Stage 4
    let y4 = y + &(&k1 * (h * a41) + &k2 * (h * a42) + &k3 * (h * a43));
    let k4 = evaluate_rhs(sys, history, t + c4 * h, &y4)?;

    // Stage 5
    let y5 = y + &(&k1 * (h * a51) + &k2 * (h * a52) + &k3 * (h * a53) + &k4 * (h * a54));
    let k5 = evaluate_rhs(sys, history, t + c5 * h, &y5)?;

    // Stage 6
    let y6 = y + &(&k1 * (h * a61)
        + &k2 * (h * a62)
        + &k3 * (h * a63)
        + &k4 * (h * a64)
        + &k5 * (h * a65));
    let k6 = evaluate_rhs(sys, history, t + h, &y6)?;

    // 5th order solution
    let y_new =
        y + &(&k1 * (h * b1) + &k3 * (h * b3) + &k4 * (h * b4) + &k5 * (h * b5) + &k6 * (h * b6));

    // Stage 7 (for error estimate, FSAL property)
    let k7 = evaluate_rhs(sys, history, t + h, &y_new)?;

    // Error estimate
    let err = &k1 * (h * e1)
        + &k3 * (h * e3)
        + &k4 * (h * e4)
        + &k5 * (h * e5)
        + &k6 * (h * e6)
        + &k7 * (h * e7);

    // Compute error norm
    let mut sum = F::zero();
    for i in 0..n {
        let scale = atol + rtol * y[i].abs().max(y_new[i].abs());
        let ratio = err[i] / scale;
        sum += ratio * ratio;
    }
    let err_norm = (sum / F::from_usize(n).unwrap_or_else(|| F::one())).sqrt();

    Ok(RK45StepResult {
        y_new,
        error_norm: err_norm,
    })
}

/// Evaluate the DDE right-hand side at (t, y), looking up delayed values from history.
fn evaluate_rhs<F: IntegrateFloat>(
    sys: &dyn DDESystem<F>,
    history: &HistoryBuffer<F>,
    t: F,
    y: &Array1<F>,
) -> IntegrateResult<Array1<F>> {
    // Get delays (possibly state-dependent)
    let delays = sys.state_dependent_delays(t, y.view());

    // Look up delayed values
    let mut y_delayed = Vec::with_capacity(delays.len());
    for tau in &delays {
        let t_delayed = t - *tau;
        let y_del = history.evaluate(t_delayed)?;
        y_delayed.push(y_del);
    }

    sys.rhs(t, y.view(), &y_delayed)
}

// ---------------------------------------------------------------------------
// Convenience: Simple DDE (single constant delay)
// ---------------------------------------------------------------------------

/// A simple DDE system with a single constant delay.
///
/// y'(t) = f(t, y(t), y(t - tau))
pub struct SimpleConstantDDE<F: IntegrateFloat> {
    ndim: usize,
    delay: F,
    rhs_fn: Box<dyn Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F> + Send + Sync>,
    history_fn: Box<dyn Fn(F) -> Array1<F> + Send + Sync>,
}

impl<F: IntegrateFloat> SimpleConstantDDE<F> {
    /// Create a simple DDE with a single constant delay.
    ///
    /// * `ndim` - system dimension
    /// * `delay` - the constant delay tau > 0
    /// * `rhs_fn` - f(t, y, y_delayed)
    /// * `history_fn` - phi(t) for t <= t0
    pub fn new<R, H>(ndim: usize, delay: F, rhs_fn: R, history_fn: H) -> Self
    where
        R: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F> + Send + Sync + 'static,
        H: Fn(F) -> Array1<F> + Send + Sync + 'static,
    {
        SimpleConstantDDE {
            ndim,
            delay,
            rhs_fn: Box::new(rhs_fn),
            history_fn: Box::new(history_fn),
        }
    }
}

impl<F: IntegrateFloat> DDESystem<F> for SimpleConstantDDE<F> {
    fn ndim(&self) -> usize {
        self.ndim
    }

    fn rhs(&self, t: F, y: ArrayView1<F>, y_delayed: &[Array1<F>]) -> IntegrateResult<Array1<F>> {
        if y_delayed.is_empty() {
            return Err(IntegrateError::ValueError(
                "Expected at least one delayed value".into(),
            ));
        }
        Ok((self.rhs_fn)(t, y, y_delayed[0].view()))
    }

    fn delays(&self) -> Vec<F> {
        vec![self.delay]
    }

    fn history(&self, t: F) -> Array1<F> {
        (self.history_fn)(t)
    }
}

/// A DDE system with multiple constant delays.
pub struct MultiDelayDDE<F: IntegrateFloat> {
    ndim: usize,
    delays_vec: Vec<F>,
    rhs_fn: Box<dyn Fn(F, ArrayView1<F>, &[Array1<F>]) -> Array1<F> + Send + Sync>,
    history_fn: Box<dyn Fn(F) -> Array1<F> + Send + Sync>,
}

impl<F: IntegrateFloat> MultiDelayDDE<F> {
    /// Create a DDE with multiple constant delays.
    pub fn new<R, H>(ndim: usize, delays_vec: Vec<F>, rhs_fn: R, history_fn: H) -> Self
    where
        R: Fn(F, ArrayView1<F>, &[Array1<F>]) -> Array1<F> + Send + Sync + 'static,
        H: Fn(F) -> Array1<F> + Send + Sync + 'static,
    {
        MultiDelayDDE {
            ndim,
            delays_vec,
            rhs_fn: Box::new(rhs_fn),
            history_fn: Box::new(history_fn),
        }
    }
}

impl<F: IntegrateFloat> DDESystem<F> for MultiDelayDDE<F> {
    fn ndim(&self) -> usize {
        self.ndim
    }

    fn rhs(&self, t: F, y: ArrayView1<F>, y_delayed: &[Array1<F>]) -> IntegrateResult<Array1<F>> {
        Ok((self.rhs_fn)(t, y, y_delayed))
    }

    fn delays(&self) -> Vec<F> {
        self.delays_vec.clone()
    }

    fn history(&self, t: F) -> Array1<F> {
        (self.history_fn)(t)
    }
}

/// A DDE system with state-dependent delays.
pub struct StateDependentDDE<F: IntegrateFloat> {
    ndim: usize,
    n_delays: usize,
    rhs_fn: Box<dyn Fn(F, ArrayView1<F>, &[Array1<F>]) -> Array1<F> + Send + Sync>,
    delay_fn: Box<dyn Fn(F, ArrayView1<F>) -> Vec<F> + Send + Sync>,
    history_fn: Box<dyn Fn(F) -> Array1<F> + Send + Sync>,
}

impl<F: IntegrateFloat> StateDependentDDE<F> {
    /// Create a DDE with state-dependent delays.
    ///
    /// * `delay_fn` - function (t, y) -> vec of delay values
    pub fn new<R, D, H>(ndim: usize, n_delays: usize, rhs_fn: R, delay_fn: D, history_fn: H) -> Self
    where
        R: Fn(F, ArrayView1<F>, &[Array1<F>]) -> Array1<F> + Send + Sync + 'static,
        D: Fn(F, ArrayView1<F>) -> Vec<F> + Send + Sync + 'static,
        H: Fn(F) -> Array1<F> + Send + Sync + 'static,
    {
        StateDependentDDE {
            ndim,
            n_delays,
            rhs_fn: Box::new(rhs_fn),
            delay_fn: Box::new(delay_fn),
            history_fn: Box::new(history_fn),
        }
    }
}

impl<F: IntegrateFloat> DDESystem<F> for StateDependentDDE<F> {
    fn ndim(&self) -> usize {
        self.ndim
    }

    fn rhs(&self, t: F, y: ArrayView1<F>, y_delayed: &[Array1<F>]) -> IntegrateResult<Array1<F>> {
        Ok((self.rhs_fn)(t, y, y_delayed))
    }

    fn delays(&self) -> Vec<F> {
        // The authoritative, time-varying delays are produced by
        // `state_dependent_delays`.  `delays()` is still consulted by the
        // integrator to size the pre-history sampling window and to seed the
        // discontinuity tracker, so it must return genuine delay magnitudes
        // rather than a fabricated constant.  We evaluate the real delay
        // function at the system's initial state (history at the natural
        // reference time `0`) to obtain a representative estimate.
        let y0 = (self.history_fn)(F::zero());
        let estimated = (self.delay_fn)(F::zero(), y0.view());
        if estimated.len() == self.n_delays {
            estimated
        } else {
            // The delay function reported a different number of delays than
            // declared; fall back to the declared count using the values we
            // did obtain (padding with the largest observed delay so the
            // history window is not under-sized).
            let fill = estimated
                .iter()
                .cloned()
                .fold(F::zero(), |a, b| if b > a { b } else { a });
            let mut out = estimated;
            out.resize(self.n_delays, fill);
            out
        }
    }

    fn state_dependent_delays(&self, t: F, y: ArrayView1<F>) -> Vec<F> {
        (self.delay_fn)(t, y)
    }

    fn history(&self, t: F) -> Array1<F> {
        (self.history_fn)(t)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_simple_constant_delay() {
        // y'(t) = -y(t - 1), y(t) = 1 for t <= 0
        // On [0, 1]: y'(t) = -1, so y(t) = 1 - t
        // At t=1: y(1) = 0
        let sys = SimpleConstantDDE::new(
            1,
            1.0,
            |_t, _y: ArrayView1<f64>, y_del: ArrayView1<f64>| -> Array1<f64> { array![-y_del[0]] },
            |_t| array![1.0],
        );

        let opts = DDEOptions {
            h0: Some(0.01),
            rtol: 1e-6,
            atol: 1e-9,
            ..Default::default()
        };

        let result = solve_dde(&sys, [0.0, 1.0], &opts).expect("DDE solve should succeed");

        assert!(result.success, "DDE integration should succeed");

        // Check y(1) ~ 0
        let y_final = result.y.last().expect("should have final state");
        assert!(
            (y_final[0] - 0.0).abs() < 0.05,
            "y(1) = {} should be near 0",
            y_final[0]
        );
    }

    #[test]
    fn test_dde_first_interval_exact() {
        // y'(t) = -y(t-1), phi(t) = 1
        // On [0,1], y' = -phi(t-1) = -1
        // => y(t) = 1 - t on [0,1]
        let sys = SimpleConstantDDE::new(
            1,
            1.0,
            |_t, _y: ArrayView1<f64>, y_del: ArrayView1<f64>| -> Array1<f64> { array![-y_del[0]] },
            |_t| array![1.0],
        );

        let opts = DDEOptions {
            h0: Some(0.005),
            rtol: 1e-8,
            atol: 1e-12,
            ..Default::default()
        };

        let result = solve_dde(&sys, [0.0, 0.5], &opts).expect("DDE solve should succeed");

        // y(0.5) = 1 - 0.5 = 0.5
        let y_final = result.y.last().expect("should have final state");
        assert!(
            (y_final[0] - 0.5).abs() < 0.01,
            "y(0.5) = {} should be near 0.5",
            y_final[0]
        );
    }

    #[test]
    fn test_multi_delay_dde() {
        // y'(t) = -y(t-0.5) - y(t-1.0), phi(t) = 1
        // On [0, 0.5], y' = -1 - 1 = -2, so y(t) = 1 - 2t
        let sys = MultiDelayDDE::new(
            1,
            vec![0.5, 1.0],
            |_t, _y: ArrayView1<f64>, y_del: &[Array1<f64>]| -> Array1<f64> {
                array![-y_del[0][0] - y_del[1][0]]
            },
            |_t| array![1.0],
        );

        let opts = DDEOptions {
            h0: Some(0.005),
            rtol: 1e-8,
            atol: 1e-12,
            ..Default::default()
        };

        let result = solve_dde(&sys, [0.0, 0.5], &opts).expect("multi-delay DDE should succeed");

        // y(0.5) = 1 - 2*0.5 = 0
        let y_final = result.y.last().expect("should have final state");
        assert!(
            (y_final[0] - 0.0).abs() < 0.05,
            "y(0.5) = {} should be near 0",
            y_final[0]
        );
    }

    #[test]
    fn test_discontinuity_tracking() {
        let sys = SimpleConstantDDE::new(
            1,
            0.5,
            |_t, _y: ArrayView1<f64>, y_del: ArrayView1<f64>| -> Array1<f64> { array![-y_del[0]] },
            |_t| array![1.0],
        );

        let opts = DDEOptions {
            h0: Some(0.01),
            track_discontinuities: true,
            max_discontinuity_order: 3,
            ..Default::default()
        };

        let result = solve_dde(&sys, [0.0, 2.0], &opts).expect("DDE with disc tracking");
        assert!(result.success);

        // Discontinuity tracker should have seeded times at 0.5, 1.0, 1.5, 2.0
        // (though some may not be hit exactly)
    }

    #[test]
    fn test_state_dependent_delay() {
        // y'(t) = -y(t - |y(t)|), phi(t) = 1
        // When y near 1, delay is 1.
        let sys = StateDependentDDE::new(
            1,
            1,
            |_t, _y: ArrayView1<f64>, y_del: &[Array1<f64>]| -> Array1<f64> {
                array![-y_del[0][0]]
            },
            |_t, y: ArrayView1<f64>| -> Vec<f64> {
                vec![y[0].abs().max(0.1)] // clamp delay to at least 0.1
            },
            |_t| array![1.0],
        );

        let opts = DDEOptions {
            h0: Some(0.005),
            rtol: 1e-5,
            atol: 1e-8,
            ..Default::default()
        };

        let result = solve_dde(&sys, [0.0, 0.5], &opts).expect("state-dep DDE should succeed");
        assert!(result.success);
        // Just verify it doesn't crash and produces reasonable output
        assert!(result.y.len() > 2);
    }

    #[test]
    fn test_dde_invalid_inputs() {
        let sys = SimpleConstantDDE::new(
            1,
            1.0,
            |_t, _y: ArrayView1<f64>, y_del: ArrayView1<f64>| -> Array1<f64> { array![-y_del[0]] },
            |_t| array![1.0],
        );

        let opts = DDEOptions::default();

        // tf <= t0
        let result = solve_dde(&sys, [1.0, 0.0], &opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_hermite_interpolation() {
        // Test the dense output segment interpolation
        let seg = DenseSegment {
            t_left: 0.0,
            t_right: 1.0,
            y_left: array![1.0],
            y_right: array![0.0],
            yp_left: array![-1.0],  // derivative at left
            yp_right: array![-1.0], // derivative at right
        };

        // For a linear function y = 1 - t, the Hermite interpolant should be exact
        let y_mid = seg.evaluate(0.5_f64);
        assert!(
            (y_mid[0] - 0.5_f64).abs() < 1e-10,
            "Hermite at 0.5: {}",
            y_mid[0]
        );

        // Check endpoints
        let y0 = seg.evaluate(0.0_f64);
        assert!((y0[0] - 1.0_f64).abs() < 1e-10);

        let y1 = seg.evaluate(1.0_f64);
        assert!((y1[0] - 0.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_history_buffer() {
        let mut buf = HistoryBuffer::new(0.0);
        buf.add_pre_history(-1.0, array![2.0]);
        buf.add_pre_history(-0.5, array![1.5]);
        buf.add_pre_history(0.0, array![1.0]);

        // Interpolate in pre-history
        let y = buf.evaluate(-0.75).expect("pre-history eval");
        assert!((y[0] - 1.75_f64).abs() < 1e-10, "y(-0.75) = {}", y[0]);

        // Add a segment
        buf.add_segment(DenseSegment {
            t_left: 0.0,
            t_right: 0.5,
            y_left: array![1.0],
            y_right: array![0.5],
            yp_left: array![-1.0],
            yp_right: array![-1.0],
        });

        let y = buf.evaluate(0.25).expect("segment eval");
        assert!((y[0] - 0.75_f64).abs() < 0.1, "y(0.25) = {}", y[0]);
    }
}
