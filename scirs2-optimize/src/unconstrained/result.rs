//! Result structure for unconstrained optimization

use ndarray::{Array1, Array2};

/// Result structure for optimization algorithms
#[derive(Debug, Clone)]
pub struct OptimizeResult<T> {
    /// Solution vector
    pub x: Array1<f64>,
    /// Objective function value at solution
    pub fun: T,
    /// Number of iterations performed
    pub iterations: usize,
    /// Number of function evaluations
    pub func_evals: usize,
    /// Whether optimization was successful
    pub success: bool,
    /// Status message
    pub message: String,
    /// Gradient at solution (optional)
    pub jacobian: Option<Array1<f64>>,
    /// Hessian at solution (optional)
    pub hessian: Option<Array2<f64>>,
}
