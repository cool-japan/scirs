//! Result structure for unconstrained optimization

use scirs2_core::ndarray::{Array1, Array2};

/// Result structure for optimization algorithms
#[derive(Debug, Clone)]
pub struct OptimizeResult<T> {
    /// Solution vector
    pub x: Array1<f64>,
    /// Objective function value at solution
    pub fun: T,
    /// Number of iterations performed
    pub nit: usize,
    /// Number of function evaluations
    pub func_evals: usize,
    /// Number of function evaluations (alternative name)
    pub nfev: usize,
    /// Whether optimization was successful
    pub success: bool,
    /// Status message
    pub message: String,
    /// Gradient at solution (optional)
    pub jacobian: Option<Array1<f64>>,
    /// Hessian at solution (optional)
    pub hessian: Option<Array2<f64>>,
}

impl<T: Default> Default for OptimizeResult<T> {
    fn default() -> Self {
        Self {
            x: Array1::zeros(0),
            fun: T::default(),
            nit: 0,
            func_evals: 0,
            nfev: 0,
            success: false,
            message: String::new(),
            jacobian: None,
            hessian: None,
        }
    }
}
