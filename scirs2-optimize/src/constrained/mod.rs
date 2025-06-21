//! Constrained optimization algorithms
//!
//! This module provides methods for constrained optimization of scalar
//! functions of one or more variables.
//!
//! ## Example
//!
//! ```ignore
//! # FIXME: This doctest requires LAPACK libraries to be linked
//! use ndarray::{array, Array1};
//! use scirs2_optimize::constrained::{minimize_constrained, Method, Constraint};
//!
//! // Define a simple function to minimize
//! fn objective(x: &[f64]) -> f64 {
//!     (x[0] - 1.0).powi(2) + (x[1] - 2.5).powi(2)
//! }
//!
//! // Define a constraint: x[0] + x[1] <= 3
//! fn constraint(x: &[f64]) -> f64 {
//!     3.0 - x[0] - x[1]  // Should be >= 0
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Minimize the function starting at [0.0, 0.0]
//! let initial_point = array![0.0, 0.0];
//! let constraints = vec![Constraint::new(constraint, Constraint::INEQUALITY)];
//!
//! let result = minimize_constrained(
//!     objective,
//!     &initial_point,
//!     &constraints,
//!     Method::SLSQP,
//!     None
//! )?;
//!
//! // The constrained minimum should be at [0.5, 2.5]
//! # Ok(())
//! # }
//! ```

use crate::error::OptimizeResult;
use crate::result::OptimizeResults;
use ndarray::{Array1, ArrayBase, Data, Ix1};
use std::fmt;

// Re-export optimization methods
pub mod augmented_lagrangian;
pub mod cobyla;
pub mod interior_point;
pub mod slsqp;
pub mod trust_constr;

// Re-export main functions
pub use augmented_lagrangian::{
    minimize_augmented_lagrangian, minimize_equality_constrained, minimize_inequality_constrained,
    AugmentedLagrangianOptions, AugmentedLagrangianResult,
};
pub use cobyla::minimize_cobyla;
pub use interior_point::{
    minimize_interior_point, minimize_interior_point_constrained, InteriorPointOptions,
    InteriorPointResult,
};
pub use slsqp::minimize_slsqp;
pub use trust_constr::minimize_trust_constr;

#[cfg(test)]
mod tests;

/// Type alias for constraint functions that take a slice of f64 and return f64
pub type ConstraintFn = fn(&[f64]) -> f64;

/// Optimization methods for constrained minimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Sequential Least SQuares Programming
    SLSQP,

    /// Trust-region constrained algorithm
    TrustConstr,

    /// Linear programming using the simplex algorithm
    COBYLA,

    /// Interior point method
    InteriorPoint,

    /// Augmented Lagrangian method
    AugmentedLagrangian,
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Method::SLSQP => write!(f, "SLSQP"),
            Method::TrustConstr => write!(f, "trust-constr"),
            Method::COBYLA => write!(f, "COBYLA"),
            Method::InteriorPoint => write!(f, "interior-point"),
            Method::AugmentedLagrangian => write!(f, "augmented-lagrangian"),
        }
    }
}

/// Options for the constrained optimizer.
#[derive(Debug, Clone)]
pub struct Options {
    /// Maximum number of iterations to perform
    pub maxiter: Option<usize>,

    /// Precision goal for the value in the stopping criterion
    pub ftol: Option<f64>,

    /// Precision goal for the gradient in the stopping criterion (relative)
    pub gtol: Option<f64>,

    /// Precision goal for constraint violation
    pub ctol: Option<f64>,

    /// Step size used for numerical approximation of the jacobian
    pub eps: Option<f64>,

    /// Whether to print convergence messages
    pub disp: bool,

    /// Return the optimization result after each iteration
    pub return_all: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            maxiter: None,
            ftol: Some(1e-8),
            gtol: Some(1e-8),
            ctol: Some(1e-8),
            eps: Some(1e-8),
            disp: false,
            return_all: false,
        }
    }
}

/// Constraint type for constrained optimization
pub struct Constraint<F> {
    /// The constraint function
    pub fun: F,

    /// The type of constraint (equality or inequality)
    pub kind: ConstraintKind,

    /// Lower bound for a box constraint
    pub lb: Option<f64>,

    /// Upper bound for a box constraint
    pub ub: Option<f64>,
}

/// The kind of constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    /// Equality constraint: fun(x) = 0
    Equality,

    /// Inequality constraint: fun(x) >= 0
    Inequality,
}

impl Constraint<fn(&[f64]) -> f64> {
    /// Constant for equality constraint
    pub const EQUALITY: ConstraintKind = ConstraintKind::Equality;

    /// Constant for inequality constraint
    pub const INEQUALITY: ConstraintKind = ConstraintKind::Inequality;

    /// Create a new constraint
    pub fn new(fun: fn(&[f64]) -> f64, kind: ConstraintKind) -> Self {
        Constraint {
            fun,
            kind,
            lb: None,
            ub: None,
        }
    }

    /// Create a new box constraint
    pub fn new_bounds(lb: Option<f64>, ub: Option<f64>) -> Self {
        Constraint {
            fun: |_| 0.0, // Dummy function for box constraints
            kind: ConstraintKind::Inequality,
            lb,
            ub,
        }
    }
}

impl<F> Constraint<F> {
    /// Check if this is a box constraint
    pub fn is_bounds(&self) -> bool {
        self.lb.is_some() || self.ub.is_some()
    }
}

/// Minimizes a scalar function of one or more variables with constraints.
///
/// # Arguments
///
/// * `func` - A function that takes a slice of values and returns a scalar
/// * `x0` - The initial guess
/// * `constraints` - Vector of constraints
/// * `method` - The optimization method to use
/// * `options` - Options for the optimizer
///
/// # Returns
///
/// * `OptimizeResults` containing the optimization results
///
/// # Example
///
/// ```ignore
/// # FIXME: This doctest requires LAPACK libraries to be linked
/// use ndarray::array;
/// use scirs2_optimize::constrained::{minimize_constrained, Method, Constraint};
///
/// // Function to minimize
/// fn objective(x: &[f64]) -> f64 {
///     (x[0] - 1.0).powi(2) + (x[1] - 2.5).powi(2)
/// }
///
/// // Constraint: x[0] + x[1] <= 3
/// fn constraint(x: &[f64]) -> f64 {
///     3.0 - x[0] - x[1]  // Should be >= 0
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let initial_point = array![0.0, 0.0];
/// let constraints = vec![Constraint::new(constraint, Constraint::INEQUALITY)];
///
/// let result = minimize_constrained(
///     objective,
///     &initial_point,
///     &constraints,
///     Method::SLSQP,
///     None
/// )?;
/// # Ok(())
/// # }
/// ```
pub fn minimize_constrained<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    constraints: &[Constraint<ConstraintFn>],
    method: Method,
    options: Option<Options>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64 + Clone,
    S: Data<Elem = f64>,
{
    let options = options.unwrap_or_default();

    // Implementation of various methods will go here
    match method {
        Method::SLSQP => minimize_slsqp(func, x0, constraints, &options),
        Method::TrustConstr => minimize_trust_constr(func, x0, constraints, &options),
        Method::COBYLA => minimize_cobyla(func, x0, constraints, &options),
        Method::InteriorPoint => {
            // Convert constraints to interior point format
            let x0_arr = Array1::from_vec(x0.to_vec());

            // Create interior point options from general options
            let ip_options = InteriorPointOptions {
                max_iter: options.maxiter.unwrap_or(100),
                tol: options.gtol.unwrap_or(1e-8),
                feas_tol: options.ctol.unwrap_or(1e-8),
                ..Default::default()
            };

            // Convert to OptimizeResults format
            match minimize_interior_point_constrained(func, x0_arr, constraints, Some(ip_options)) {
                Ok(result) => {
                    let opt_result = OptimizeResults {
                        x: result.x,
                        fun: result.fun,
                        nit: result.nit,
                        nfev: result.nfev,
                        success: result.success,
                        message: result.message,
                        jac: None,
                        hess: None,
                        constr: None,
                        njev: 0,  // Not tracked by interior point method
                        nhev: 0,  // Not tracked by interior point method
                        maxcv: 0, // Not applicable for interior point
                        status: if result.success { 0 } else { 1 },
                    };
                    Ok(opt_result)
                }
                Err(e) => Err(e),
            }
        }
        Method::AugmentedLagrangian => {
            // Convert to augmented Lagrangian method format (simplified for now)
            Err(crate::error::OptimizeError::NotImplementedError(
                "Augmented Lagrangian method integration with minimize_constrained not yet implemented"
                    .to_string(),
            ))
        }
    }
}
