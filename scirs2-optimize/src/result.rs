//! Optimization result structures
//!
//! This module defines the common result structures for optimization algorithms.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Zero;
use std::fmt;

/// A structure that contains the results of an optimization.
///
/// This structure is modeled after SciPy's `OptimizeResult` structure.
#[derive(Clone, Debug)]
pub struct OptimizeResults<T> {
    /// The solution array
    pub x: Array1<T>,

    /// Value of objective function at the solution
    pub fun: T,

    /// Value of the gradient/jacobian at the solution
    pub jac: Option<Vec<T>>,

    /// Value of the Hessian at the solution
    pub hess: Option<Vec<T>>,

    /// Value of constraint functions at the solution (if applicable)
    pub constr: Option<Array1<T>>,

    /// Number of iterations performed
    pub nit: usize,

    /// Number of evaluations of the objective function
    pub nfev: usize,

    /// Number of evaluations of the gradient
    pub njev: usize,

    /// Number of evaluations of the Hessian
    pub nhev: usize,

    /// Maximum number of iterations exceeded flag
    pub maxcv: usize,

    /// Termination message
    pub message: String,

    /// Whether or not the optimizer exited successfully
    pub success: bool,

    /// Termination status code
    pub status: i32,
}

impl<T: fmt::Display + fmt::Debug> fmt::Display for OptimizeResults<T>
where
    T: Copy,
    Array1<T>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Optimization Results:")?;
        writeln!(f, "  success: {}", self.success)?;
        writeln!(f, "  status: {}", self.status)?;
        writeln!(f, "  message: {}", self.message)?;
        writeln!(f, "  nfev: {}", self.nfev)?;
        writeln!(f, "  njev: {}", self.njev)?;
        writeln!(f, "  nhev: {}", self.nhev)?;
        writeln!(f, "  nit: {}", self.nit)?;
        writeln!(f, "  final value: {}", self.fun)?;
        writeln!(f, "  solution: {:?}", self.x)?;
        if let Some(ref jac) = self.jac {
            writeln!(f, "  jacobian: {:?} (vector)", jac)?;
        }
        Ok(())
    }
}

impl<T> Default for OptimizeResults<T>
where
    T: Default + Clone + Zero,
{
    fn default() -> Self {
        OptimizeResults {
            x: Array1::<T>::zeros(0),
            fun: T::default(),
            jac: None,
            hess: None,
            constr: None,
            nit: 0,
            nfev: 0,
            njev: 0,
            nhev: 0,
            maxcv: 0,
            message: String::new(),
            success: false,
            status: 0,
        }
    }
}
