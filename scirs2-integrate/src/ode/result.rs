//! Result types for ODE solvers.

use scirs2_core::ndarray::{Array1, Array2};

/// Result of an ODE solver.
#[derive(Debug, Clone)]
pub struct ODEResult {
    /// Time points for the solution.
    pub t: Array1<f64>,
    /// Solution values at each time point.
    pub y: Array2<f64>,
    /// Number of function evaluations.
    pub nfev: usize,
    /// Number of jacobian evaluations.
    pub njev: usize,
    /// Number of steps taken by the solver.
    pub nsteps: usize,
    /// Status of the solver (success or error).
    pub status: ODEStatus,
    /// Optional message about the solver's execution.
    pub message: String,
    /// Time points where solver failed (for diagnostics).
    pub t_failed: Vec<f64>,
}

/// Status of the ODE solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ODEStatus {
    /// The solver successfully reached the end of the integration interval.
    Success,
    /// The solver failed at some point during integration.
    Failed,
}

impl ODEResult {
    /// Returns whether the solver completed successfully.
    pub fn is_success(&self) -> bool {
        self.status == ODEStatus::Success
    }

    /// Returns whether the solver failed.
    pub fn is_failed(&self) -> bool {
        self.status == ODEStatus::Failed
    }
}
