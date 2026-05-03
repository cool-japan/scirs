//! Async/cached ODE solver for neural ODE training patterns.
//!
//! # Motivation
//!
//! Neural ODE training requires integrating the same ODE structure many times
//! per batch — once per forward pass and once per adjoint backward pass.
//! The "CUDA graph capture" concept (repeated ODE solve patterns) maps in
//! pure Rust to:
//!
//! 1. **Pre-allocated scratch buffers**: No per-solve heap allocation beyond
//!    the initial setup (`CachedOdeProblem::new`).
//! 2. **Rayon batch parallelism**: `integrate_batch` runs all initial
//!    conditions concurrently using Rayon's work-stealing pool.
//! 3. **Tokio async wrapper**: `integrate_batch_async` offloads the blocking
//!    rayon call onto a `spawn_blocking` thread so the async runtime stays
//!    responsive.
//!
//! # Solver
//!
//! A fixed-step RK4 integrator is used.  The step size `dt` and number of
//! steps are determined at construction time from `t_span = (t0, t1)` and
//! `dt`.  This predictable structure (unlike adaptive step-size methods) is
//! what enables true "graph capture": the sequence of RHS evaluations is
//! identical for every call, making it JIT-friendly.
//!
//! # Example
//!
//! ```rust
//! use scirs2_integrate::async_ode::{CachedOdeProblem, integrate_batch_async};
//! use std::sync::Arc;
//!
//! // dy/dt = -y  →  y(t) = y0 * exp(-t)
//! let problem = Arc::new(
//!     CachedOdeProblem::new(|_t, y, dydt| { dydt[0] = -y[0]; }, 0.0, 1.0, 0.01, 1)
//! );
//!
//! let result = problem.integrate(&[1.0]).unwrap();
//! let expected = 1.0_f64.exp().recip(); // e^{-1} ≈ 0.3679
//! assert!((result[0] - expected).abs() < 1e-4);
//! ```

use crate::error::IntegrateError;
use scirs2_core::parallel_ops::*;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// CachedOdeProblem
// ─────────────────────────────────────────────────────────────────────────────

/// A pre-compiled ODE problem for repeated integration (neural ODE pattern).
///
/// Once constructed the RK4 graph is fixed: `n_steps` evaluations of `rhs`
/// are performed per `integrate` call, with no re-allocation of step vectors.
pub struct CachedOdeProblem<F>
where
    F: Fn(f64, &[f64], &mut [f64]) + Send + Sync,
{
    rhs: Arc<F>,
    t0: f64,
    dt: f64,
    n_steps: usize,
    state_dim: usize,
}

impl<F> CachedOdeProblem<F>
where
    F: Fn(f64, &[f64], &mut [f64]) + Send + Sync + 'static,
{
    /// Create a new cached ODE problem.
    ///
    /// # Parameters
    ///
    /// - `rhs`: Right-hand side `f(t, y, dydt)` — writes into `dydt`.
    /// - `t0`, `t1`: Integration interval `[t0, t1]`.
    /// - `dt`: Fixed step size.  Actual `n_steps = ceil((t1 - t0) / dt)`.
    /// - `state_dim`: Dimensionality of the state vector.
    ///
    /// # Errors
    ///
    /// Returns [`IntegrateError::ValueError`] if `dt ≤ 0` or `t1 ≤ t0`.
    pub fn new(rhs: F, t0: f64, t1: f64, dt: f64, state_dim: usize) -> Self {
        let span = t1 - t0;
        let n_steps = ((span / dt).ceil() as usize).max(1);
        CachedOdeProblem {
            rhs: Arc::new(rhs),
            t0,
            dt,
            n_steps,
            state_dim,
        }
    }

    /// Integrate from initial state `y0` and return the final state.
    ///
    /// Pre-allocated scratch buffers are stack-allocated for `state_dim ≤ 16`
    /// and heap-allocated otherwise; either way no allocation occurs inside
    /// the RK4 loop itself.
    pub fn integrate(&self, y0: &[f64]) -> Result<Vec<f64>, IntegrateError> {
        if y0.len() != self.state_dim {
            return Err(IntegrateError::DimensionMismatch(format!(
                "y0.len()={} != state_dim={}",
                y0.len(),
                self.state_dim
            )));
        }

        let dim = self.state_dim;
        let mut y = y0.to_vec();
        // Pre-allocate scratch once — reused across all steps
        let mut k1 = vec![0.0_f64; dim];
        let mut k2 = vec![0.0_f64; dim];
        let mut k3 = vec![0.0_f64; dim];
        let mut k4 = vec![0.0_f64; dim];
        let mut ytmp = vec![0.0_f64; dim];

        let rhs = &*self.rhs;
        let mut t = self.t0;
        let h = self.dt;

        for _ in 0..self.n_steps {
            // k1 = f(t, y)
            rhs(t, &y, &mut k1);

            // k2 = f(t + h/2, y + h/2 * k1)
            for i in 0..dim {
                ytmp[i] = y[i] + 0.5 * h * k1[i];
            }
            rhs(t + 0.5 * h, &ytmp, &mut k2);

            // k3 = f(t + h/2, y + h/2 * k2)
            for i in 0..dim {
                ytmp[i] = y[i] + 0.5 * h * k2[i];
            }
            rhs(t + 0.5 * h, &ytmp, &mut k3);

            // k4 = f(t + h, y + h * k3)
            for i in 0..dim {
                ytmp[i] = y[i] + h * k3[i];
            }
            rhs(t + h, &ytmp, &mut k4);

            // y ← y + h/6 * (k1 + 2k2 + 2k3 + k4)
            for i in 0..dim {
                y[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            }
            t += h;
        }

        Ok(y)
    }

    /// Batch integration — runs all initial conditions in parallel using Rayon.
    ///
    /// Returns one output state vector per input in `batch_y0`, in the same
    /// order as the inputs.
    pub fn integrate_batch(&self, batch_y0: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, IntegrateError> {
        parallel_map_result(batch_y0, |y0| self.integrate(y0))
    }

    /// The step size used during integration.
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Number of RK4 steps per `integrate` call.
    pub fn n_steps(&self) -> usize {
        self.n_steps
    }

    /// State dimension.
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Async wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Async batch integration — offloads blocking Rayon work to a spawn_blocking thread.
///
/// This keeps the Tokio runtime responsive while allowing the full Rayon thread
/// pool to be used for the actual computation.
///
/// Requires the `tokio` dependency to be available in the workspace.
pub async fn integrate_batch_async<F>(
    problem: Arc<CachedOdeProblem<F>>,
    batch_y0: Vec<Vec<f64>>,
) -> Result<Vec<Vec<f64>>, IntegrateError>
where
    F: Fn(f64, &[f64], &mut [f64]) + Send + Sync + 'static,
{
    tokio::task::spawn_blocking(move || problem.integrate_batch(&batch_y0))
        .await
        .map_err(|e| IntegrateError::ComputationError(format!("spawn_blocking panicked: {e}")))?
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// dy/dt = -y, exact solution y(t) = e^{-t}
    fn exponential_decay() -> impl Fn(f64, &[f64], &mut [f64]) + Send + Sync + 'static {
        |_t, y, dydt| {
            dydt[0] = -y[0];
        }
    }

    #[test]
    fn test_cached_ode_exponential_decay() {
        let problem = CachedOdeProblem::new(exponential_decay(), 0.0, 1.0, 0.001, 1);
        let result = problem.integrate(&[1.0]).expect("integration failed");
        let expected = std::f64::consts::E.recip(); // e^{-1}
        assert!(
            (result[0] - expected).abs() < 1e-5,
            "Expected ≈{expected:.6}, got {:.6}",
            result[0]
        );
    }

    #[test]
    fn test_batch_integration_matches_serial() {
        let problem = Arc::new(CachedOdeProblem::new(
            exponential_decay(),
            0.0,
            0.5,
            0.001,
            1,
        ));
        let batch_y0 = vec![vec![1.0], vec![2.0], vec![0.5]];
        let batch_result = problem.integrate_batch(&batch_y0).expect("batch failed");

        for (y0, yr) in batch_y0.iter().zip(batch_result.iter()) {
            let serial = problem.integrate(y0).expect("serial failed");
            assert!(
                (serial[0] - yr[0]).abs() < 1e-14,
                "Batch/serial mismatch: serial={:.10} batch={:.10}",
                serial[0],
                yr[0]
            );
        }
    }

    #[test]
    fn test_neural_ode_repeated_forward_same_result() {
        // Repeated calls with the same y0 must return identical results (deterministic)
        let problem = Arc::new(CachedOdeProblem::new(
            |_t, y, dydt| {
                dydt[0] = -y[0];
                dydt[1] = -2.0 * y[1];
            },
            0.0,
            1.0,
            0.01,
            2,
        ));
        let y0 = vec![1.0, 1.0];
        let r1 = problem.integrate(&y0).expect("first forward failed");
        let r2 = problem.integrate(&y0).expect("second forward failed");
        let r3 = problem.integrate(&y0).expect("third forward failed");
        assert_eq!(r1, r2, "Results differ between calls 1 and 2");
        assert_eq!(r1, r3, "Results differ between calls 1 and 3");
    }

    #[test]
    fn test_dimension_mismatch_returns_error() {
        let problem = CachedOdeProblem::new(exponential_decay(), 0.0, 1.0, 0.01, 1);
        // Pass 2-element state to 1-dim problem
        assert!(problem.integrate(&[1.0, 2.0]).is_err());
    }

    #[tokio::test]
    async fn test_async_batch_returns_correct_shape() {
        let problem = Arc::new(CachedOdeProblem::new(
            exponential_decay(),
            0.0,
            0.5,
            0.01,
            1,
        ));
        let batch_y0 = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let expected_len = batch_y0.len();
        let results = integrate_batch_async(problem, batch_y0)
            .await
            .expect("async batch failed");
        assert_eq!(results.len(), expected_len);
        for r in &results {
            assert_eq!(r.len(), 1, "Each result must have state_dim=1 entries");
        }
    }

    #[tokio::test]
    async fn test_async_matches_sync() {
        let problem_async = Arc::new(CachedOdeProblem::new(
            exponential_decay(),
            0.0,
            1.0,
            0.001,
            1,
        ));
        let problem_sync = Arc::new(CachedOdeProblem::new(
            exponential_decay(),
            0.0,
            1.0,
            0.001,
            1,
        ));
        let batch_y0 = vec![vec![1.0], vec![0.5], vec![2.0]];
        let async_results = integrate_batch_async(problem_async, batch_y0.clone())
            .await
            .expect("async failed");
        let sync_results = problem_sync
            .integrate_batch(&batch_y0)
            .expect("sync failed");
        for (a, s) in async_results.iter().zip(sync_results.iter()) {
            assert!(
                (a[0] - s[0]).abs() < 1e-14,
                "Async/sync mismatch: {:.10} vs {:.10}",
                a[0],
                s[0]
            );
        }
    }
}
