//! Stochastic Differential Equation (SDE) solvers
//!
//! This module provides numerical methods for solving SDEs of the form:
//!
//! ```text
//! dx = f(t, x) dt + g(t, x) dW
//! ```
//!
//! where:
//! - `f(t, x)` is the drift coefficient (deterministic part)
//! - `g(t, x)` is the diffusion matrix (stochastic part)
//! - `dW` is an increment of a Wiener process (Brownian motion)
//!
//! ## Methods
//!
//! | Method | Module | Strong Order | Weak Order |
//! |--------|--------|-------------|------------|
//! | Euler-Maruyama | `euler_maruyama` | 0.5 | 1.0 |
//! | Milstein | `milstein` | 1.0 | 1.0 |
//! | Stochastic Runge-Kutta | `runge_kutta_sde` | 1.5 | 2.0 |
//! | Platen explicit | `runge_kutta_sde` | 1.5 | 2.0 |
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_integrate::sde::{SdeProblem, SdeOptions};
//! use scirs2_integrate::sde::euler_maruyama::euler_maruyama;
//! use scirs2_core::ndarray::{array, Array1, Array2};
//! use scirs2_core::random::prelude::*;
//!
//! // Geometric Brownian Motion: dX = μ X dt + σ X dW
//! let mu = 0.05_f64;
//! let sigma = 0.2_f64;
//! let x0 = array![100.0_f64];
//!
//! let drift = move |_t: f64, x: &Array1<f64>| -> Array1<f64> {
//!     array![mu * x[0]]
//! };
//! let diffusion = move |_t: f64, x: &Array1<f64>| -> Array2<f64> {
//!     let mut g = Array2::zeros((1, 1));
//!     g[[0, 0]] = sigma * x[0];
//!     g
//! };
//!
//! let prob = SdeProblem::new(x0, [0.0, 1.0], 1, drift, diffusion);
//! let mut rng = seeded_rng(42);
//! let sol = euler_maruyama(&prob, 0.01, &mut rng).unwrap();
//! assert!(!sol.t.is_empty());
//! ```

pub mod euler_maruyama;
pub mod examples;
pub mod fractional_brownian;
pub mod jump_diffusion;
pub mod milstein;
pub mod particle_filter;
pub mod processes;
pub mod rough_sde;
pub mod runge_kutta_sde;
pub mod srk;
pub mod streaming_particle_filter;
pub mod weak_order2;
pub mod weak_schemes;

pub use streaming_particle_filter::{
    FilterEstimate, SimpleRng, StreamingParticleFilter, StreamingParticleFilterBuilder,
};

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};

/// Defines a Stochastic Differential Equation problem of the form:
///
/// ```text
/// dx = f(t, x) dt + g(t, x) dW,   x(t0) = x0
/// ```
///
/// where `f` is the drift coefficient, `g` is the diffusion matrix,
/// and `dW` is an m-dimensional Wiener process increment.
///
/// # Type Parameters
///
/// * `F` - drift function type: `Fn(f64, &Array1<f64>) -> Array1<f64>`
/// * `G` - diffusion function type: `Fn(f64, &Array1<f64>) -> Array2<f64>`
pub struct SdeProblem<F, G>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    /// Initial state vector x(t0), dimension n
    pub x0: Array1<f64>,
    /// Time span [t0, t1]
    pub t_span: [f64; 2],
    /// Number of independent Brownian motions (Wiener processes), m
    pub n_brownian: usize,
    /// Drift coefficient f(t, x): R × R^n → R^n
    pub f_drift: F,
    /// Diffusion matrix g(t, x): R × R^n → R^{n×m}
    pub g_diffusion: G,
}

impl<F, G> SdeProblem<F, G>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    /// Create a new SDE problem.
    ///
    /// # Arguments
    ///
    /// * `x0` - Initial state vector (length n)
    /// * `t_span` - Time interval [t0, t1]
    /// * `n_brownian` - Number of independent Brownian motions
    /// * `f_drift` - Drift function f(t, x) → R^n
    /// * `g_diffusion` - Diffusion function g(t, x) → R^{n×m}
    ///
    /// # Errors
    ///
    /// Returns an error if the time span is invalid (t0 >= t1).
    pub fn new(
        x0: Array1<f64>,
        t_span: [f64; 2],
        n_brownian: usize,
        f_drift: F,
        g_diffusion: G,
    ) -> Self {
        Self {
            x0,
            t_span,
            n_brownian,
            f_drift,
            g_diffusion,
        }
    }

    /// Dimension of the state space (n)
    pub fn dim(&self) -> usize {
        self.x0.len()
    }

    /// Validate the problem parameters
    pub fn validate(&self) -> IntegrateResult<()> {
        if self.t_span[0] >= self.t_span[1] {
            return Err(IntegrateError::InvalidInput(format!(
                "t_span must satisfy t0 < t1, got [{}, {}]",
                self.t_span[0], self.t_span[1]
            )));
        }
        if self.n_brownian == 0 {
            return Err(IntegrateError::InvalidInput(
                "n_brownian must be at least 1".to_string(),
            ));
        }
        if self.x0.is_empty() {
            return Err(IntegrateError::InvalidInput(
                "Initial state x0 must be non-empty".to_string(),
            ));
        }
        Ok(())
    }
}

/// Solution to an SDE, containing the time points and state trajectories.
#[derive(Debug, Clone)]
pub struct SdeSolution {
    /// Time points t_0, t_1, ..., t_N
    pub t: Vec<f64>,
    /// State trajectory x(t_0), x(t_1), ..., x(t_N)
    pub x: Vec<Array1<f64>>,
}

impl SdeSolution {
    /// Create a new empty solution with pre-allocated capacity.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            t: Vec::with_capacity(n),
            x: Vec::with_capacity(n),
        }
    }

    /// Push a new time-state pair.
    pub fn push(&mut self, t: f64, x: Array1<f64>) {
        self.t.push(t);
        self.x.push(x);
    }

    /// Number of time points in the solution.
    pub fn len(&self) -> usize {
        self.t.len()
    }

    /// Returns true if the solution is empty.
    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }

    /// Final time value.
    pub fn t_final(&self) -> Option<f64> {
        self.t.last().copied()
    }

    /// Final state.
    pub fn x_final(&self) -> Option<&Array1<f64>> {
        self.x.last()
    }

    /// Compute the mean trajectory across an ensemble of solutions.
    ///
    /// All solutions must have the same time points and state dimensions.
    pub fn ensemble_mean(solutions: &[SdeSolution]) -> IntegrateResult<SdeSolution> {
        if solutions.is_empty() {
            return Err(IntegrateError::InvalidInput(
                "Cannot compute mean of empty ensemble".to_string(),
            ));
        }
        let n_steps = solutions[0].len();
        let n_ensemble = solutions.len();
        let mut result = SdeSolution::with_capacity(n_steps);

        for step in 0..n_steps {
            let t = solutions[0].t[step];
            let dim = solutions[0].x[step].len();
            let mut mean_x = Array1::zeros(dim);
            for sol in solutions {
                if sol.len() != n_steps {
                    return Err(IntegrateError::DimensionMismatch(
                        "All solutions in ensemble must have the same number of steps".to_string(),
                    ));
                }
                mean_x += &sol.x[step];
            }
            mean_x /= n_ensemble as f64;
            result.push(t, mean_x);
        }
        Ok(result)
    }

    /// Compute the variance trajectory across an ensemble of solutions.
    pub fn ensemble_variance(solutions: &[SdeSolution]) -> IntegrateResult<SdeSolution> {
        if solutions.is_empty() {
            return Err(IntegrateError::InvalidInput(
                "Cannot compute variance of empty ensemble".to_string(),
            ));
        }
        let n_steps = solutions[0].len();
        let n_ensemble = solutions.len();
        if n_ensemble < 2 {
            return Err(IntegrateError::InvalidInput(
                "Need at least 2 solutions to compute variance".to_string(),
            ));
        }
        let mean_sol = Self::ensemble_mean(solutions)?;
        let mut result = SdeSolution::with_capacity(n_steps);

        for step in 0..n_steps {
            let t = solutions[0].t[step];
            let dim = solutions[0].x[step].len();
            let mut var_x = Array1::zeros(dim);
            for sol in solutions {
                let diff = &sol.x[step] - &mean_sol.x[step];
                var_x += &diff.mapv(|v| v * v);
            }
            var_x /= (n_ensemble - 1) as f64;
            result.push(t, var_x);
        }
        Ok(result)
    }
}

/// Options for SDE solvers.
#[derive(Debug, Clone)]
pub struct SdeOptions {
    /// Whether to save the solution at every step (true) or only at final time (false)
    pub save_all_steps: bool,
    /// Maximum number of steps (safety limit)
    pub max_steps: usize,
}

impl Default for SdeOptions {
    fn default() -> Self {
        Self {
            save_all_steps: true,
            max_steps: 10_000_000,
        }
    }
}

/// Compute the number of steps needed for a given time span and step size,
/// clamping to `max_steps` as a safety limit.
pub(crate) fn compute_n_steps(
    t0: f64,
    t1: f64,
    dt: f64,
    max_steps: usize,
) -> IntegrateResult<usize> {
    if dt <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "Step size dt must be positive, got {}",
            dt
        )));
    }
    let n = ((t1 - t0) / dt).ceil() as usize;
    if n > max_steps {
        return Err(IntegrateError::InvalidInput(format!(
            "Required steps {} exceeds maximum {}",
            n, max_steps
        )));
    }
    Ok(n.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_sde_problem_creation() {
        let x0 = array![1.0_f64];
        let prob = SdeProblem::new(
            x0,
            [0.0, 1.0],
            1,
            |_t, x| x.clone(),
            |_t, _x| Array2::eye(1),
        );
        assert_eq!(prob.dim(), 1);
        assert_eq!(prob.n_brownian, 1);
        prob.validate().expect("Validation should pass");
    }

    #[test]
    fn test_sde_problem_invalid_tspan() {
        let x0 = array![1.0_f64];
        let prob = SdeProblem::new(
            x0,
            [1.0, 0.0], // t0 > t1 is invalid
            1,
            |_t, x| x.clone(),
            |_t, _x| Array2::eye(1),
        );
        assert!(prob.validate().is_err());
    }

    #[test]
    fn test_sde_solution_push_and_query() {
        let mut sol = SdeSolution::with_capacity(3);
        sol.push(0.0, array![1.0_f64]);
        sol.push(0.5, array![1.1_f64]);
        sol.push(1.0, array![1.2_f64]);
        assert_eq!(sol.len(), 3);
        assert!(!sol.is_empty());
        assert!((sol.t_final().expect("solution has time steps") - 1.0).abs() < 1e-12);
        assert!((sol.x_final().expect("solution has state")[0] - 1.2).abs() < 1e-12);
    }

    #[test]
    fn test_ensemble_mean() {
        let mut sol1 = SdeSolution::with_capacity(2);
        sol1.push(0.0, array![1.0_f64]);
        sol1.push(1.0, array![2.0_f64]);

        let mut sol2 = SdeSolution::with_capacity(2);
        sol2.push(0.0, array![1.0_f64]);
        sol2.push(1.0, array![4.0_f64]);

        let mean = SdeSolution::ensemble_mean(&[sol1, sol2]).expect("ensemble_mean should succeed");
        assert!((mean.x[1][0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_n_steps() {
        let n = compute_n_steps(0.0, 1.0, 0.1, 1000).expect("compute_n_steps should succeed");
        assert_eq!(n, 10);
    }

    #[test]
    fn test_compute_n_steps_invalid_dt() {
        assert!(compute_n_steps(0.0, 1.0, -0.1, 1000).is_err());
        assert!(compute_n_steps(0.0, 1.0, 0.0, 1000).is_err());
    }
}
