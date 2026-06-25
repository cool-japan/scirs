//! Constrained optimization algorithms
//!
//! This module provides methods for constrained optimization of scalar
//! functions of one or more variables.
//!
//! ## Example
//!
//! ```no_run
//! use scirs2_core::ndarray::{array, Array1};
//! use scirs2_optimize::constrained::{minimize_constrained, Method, Constraint};
//!
//! // Define a simple function to minimize: f(x) = (x[0] - 1)² + (x[1] - 2.5)²
//! // Unconstrained minimum is at (1.0, 2.5), but we add a constraint.
//! fn objective(x: &[f64]) -> f64 {
//!     (x[0] - 1.0).powi(2) + (x[1] - 2.5).powi(2)
//! }
//!
//! // Define a constraint: x[0] + x[1] <= 3
//! // Written as g(x) >= 0, so: g(x) = 3 - x[0] - x[1]
//! fn constraint(x: &[f64]) -> f64 {
//!     3.0 - x[0] - x[1]  // Should be >= 0
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Minimize the function starting at [1.0, 1.0]
//! // Note: Initial point should be feasible (satisfy constraints) for best convergence
//! let initial_point = array![1.0, 1.0];
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
//! // The constrained minimum is at [0.75, 2.25] with f(x) = 0.125
//! // This is where the gradient of f is parallel to the constraint boundary,
//! // solved via Lagrange multipliers on x[0] + x[1] = 3.
//! # Ok(())
//! # }
//! ```
//!
//! Note: This function requires LAPACK libraries to be linked for certain optimization methods.

use crate::error::OptimizeResult;
use crate::result::OptimizeResults;
use scirs2_core::ndarray::{Array1, ArrayBase, Data, Ix1};
use std::fmt;

// Re-export optimization methods
pub mod augmented_lagrangian;
pub mod cobyla;
pub mod enhanced_sqp;
pub mod epsilon_constraint;
pub mod feasibility_rules;
pub mod interior_point;
pub mod lp_qp_interior;
pub mod penalty;
pub mod slsqp;
pub mod sqp;
pub mod sqp_advanced;
pub mod trust_constr;
pub mod trust_constr_advanced;

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
pub use sqp::{minimize_sqp, SqpOptions, SqpResult};
pub use trust_constr::{
    minimize_trust_constr, minimize_trust_constr_with_derivatives, GradientFn, HessianFn,
    HessianUpdate,
};

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

    /// Sequential Quadratic Programming
    SQP,
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Method::SLSQP => write!(f, "SLSQP"),
            Method::TrustConstr => write!(f, "trust-constr"),
            Method::COBYLA => write!(f, "COBYLA"),
            Method::InteriorPoint => write!(f, "interior-point"),
            Method::AugmentedLagrangian => write!(f, "augmented-lagrangian"),
            Method::SQP => write!(f, "SQP"),
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

/// Constraint type for constrained optimization.
///
/// The constraint callable is stored as a boxed trait object so that a
/// `Vec<Constraint>` can hold heterogeneous closures (issue #126). Closures
/// that capture outer variables (for example a `threshold`) are accepted via
/// [`Constraint::new`]. An optional analytical Jacobian (gradient of the
/// constraint with respect to each variable) can be attached via
/// [`Constraint::with_jacobian`] (issue #127); when present the solvers use it
/// instead of finite differences.
pub struct Constraint {
    /// The constraint function
    pub fun: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,

    /// Optional analytical Jacobian (gradient) of the constraint, returning a
    /// length-`n` vector of partial derivatives. `None` selects finite
    /// differences.
    pub jac: Option<Box<dyn Fn(&[f64]) -> Array1<f64> + Send + Sync>>,

    /// The type of constraint (equality or inequality)
    pub kind: ConstraintKind,

    /// Lower bound for a box constraint
    pub lb: Option<f64>,

    /// Upper bound for a box constraint
    pub ub: Option<f64>,
}

impl fmt::Debug for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The boxed closures are not `Debug`; elide them.
        f.debug_struct("Constraint")
            .field("kind", &self.kind)
            .field("lb", &self.lb)
            .field("ub", &self.ub)
            .field("has_jac", &self.jac.is_some())
            .finish_non_exhaustive()
    }
}

/// The kind of constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    /// Equality constraint: fun(x) = 0
    Equality,

    /// Inequality constraint: fun(x) >= 0
    Inequality,
}

impl Constraint {
    /// Constant for equality constraint
    pub const EQUALITY: ConstraintKind = ConstraintKind::Equality;

    /// Constant for inequality constraint
    pub const INEQUALITY: ConstraintKind = ConstraintKind::Inequality;

    /// Create a new constraint.
    ///
    /// Accepts any callable implementing `Fn(&[f64]) -> f64`, including
    /// closures that capture outer variables and plain `fn` pointers.
    pub fn new<F>(fun: F, kind: ConstraintKind) -> Self
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        Constraint {
            fun: Box::new(fun),
            jac: None,
            kind,
            lb: None,
            ub: None,
        }
    }

    /// Create a new box constraint
    pub fn new_bounds(lb: Option<f64>, ub: Option<f64>) -> Self {
        Constraint {
            fun: Box::new(|_| 0.0), // Dummy function for box constraints
            jac: None,
            kind: ConstraintKind::Inequality,
            lb,
            ub,
        }
    }

    /// Attach an analytical Jacobian (gradient) for this constraint (issue #127).
    ///
    /// The supplied callable returns the length-`n` gradient of the constraint
    /// with respect to each variable, which fills one row of the constraint
    /// Jacobian matrix. When present, the solvers use it instead of finite
    /// differences.
    pub fn with_jacobian<J>(mut self, jac: J) -> Self
    where
        J: Fn(&[f64]) -> Array1<f64> + Send + Sync + 'static,
    {
        self.jac = Some(Box::new(jac));
        self
    }

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
/// ```no_run
/// use scirs2_core::ndarray::array;
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
#[allow(dead_code)]
pub fn minimize_constrained<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    constraints: &[Constraint],
    method: Method,
    options: Option<Options>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64 + Clone,
    S: Data<Elem = f64>,
{
    // Delegate to the gradient-aware variant with no analytical objective gradient
    // (finite differences are used for the objective; per-constraint analytical
    // Jacobians, if attached via `Constraint::with_jacobian`, are still honoured).
    minimize_constrained_with_jac(
        func,
        None::<fn(&[f64]) -> Array1<f64>>,
        x0,
        constraints,
        method,
        options,
    )
}

/// Minimizes a scalar function of one or more variables with constraints,
/// optionally using an analytical objective gradient (issue #127).
///
/// This is the gradient-aware counterpart of [`minimize_constrained`]. When
/// `jac` is `Some`, the supplied objective gradient is used in place of finite
/// differences by the gradient-based methods (SLSQP and Trust-Constr).
/// Per-constraint analytical Jacobians attached via
/// [`Constraint::with_jacobian`] are honoured regardless of `jac`.
///
/// # Arguments
///
/// * `func` - The objective function to minimize
/// * `jac` - Optional analytical gradient of the objective, returning a
///   length-`n` vector. `None` selects finite differences.
/// * `x0` - The initial guess
/// * `constraints` - Vector of constraints
/// * `method` - The optimization method to use
/// * `options` - Options for the optimizer
///
/// # Example
///
/// ```no_run
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_optimize::constrained::{minimize_constrained_with_jac, Method, Constraint};
///
/// fn objective(x: &[f64]) -> f64 {
///     (x[0] - 1.0).powi(2) + (x[1] - 2.5).powi(2)
/// }
///
/// fn objective_grad(x: &[f64]) -> Array1<f64> {
///     array![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.5)]
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let x0 = array![0.0, 0.0];
/// let constraints = vec![Constraint::new(|x: &[f64]| 3.0 - x[0] - x[1], Constraint::INEQUALITY)];
///
/// let result = minimize_constrained_with_jac(
///     objective,
///     Some(objective_grad),
///     &x0,
///     &constraints,
///     Method::SLSQP,
///     None,
/// )?;
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub fn minimize_constrained_with_jac<F, G, S>(
    func: F,
    jac: Option<G>,
    x0: &ArrayBase<S, Ix1>,
    constraints: &[Constraint],
    method: Method,
    options: Option<Options>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64 + Clone,
    G: Fn(&[f64]) -> Array1<f64> + Clone,
    S: Data<Elem = f64>,
{
    let options = options.unwrap_or_default();

    // Box the optional objective gradient so it can be threaded into the
    // internal solvers as a trait object (`&dyn Fn(&[f64]) -> Array1<f64>`).
    let obj_jac: Option<Box<dyn Fn(&[f64]) -> Array1<f64>>> =
        jac.map(|g| Box::new(g) as Box<dyn Fn(&[f64]) -> Array1<f64>>);
    let obj_jac_ref: Option<&dyn Fn(&[f64]) -> Array1<f64>> = obj_jac.as_ref().map(|b| b.as_ref());

    // Implementation of various methods will go here
    match method {
        Method::SLSQP => minimize_slsqp(func, x0, constraints, obj_jac_ref, &options),
        Method::TrustConstr => minimize_trust_constr(func, x0, constraints, obj_jac_ref, &options),
        // COBYLA is derivative-free by design; the analytical objective gradient
        // is intentionally not used here.
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
                    let opt_result = OptimizeResults::<f64> {
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
            use scirs2_core::ndarray::{Array1, ArrayView1};
            use std::sync::Arc;

            let x0_arr = Array1::from_vec(x0.to_vec());

            // Partition constraints into equality and inequality index groups.
            // The boxed closures cannot be copied out of the `Constraint` slice,
            // so instead we capture the indices (cheaply `Clone`-able via `Arc`)
            // and a shared reference to the original `constraints` slice, then
            // evaluate the constraints in place. `Send + Sync` on the boxed
            // closures preserves the threaded augmented-Lagrangian behaviour.
            let eq_idx: Arc<Vec<usize>> = Arc::new(
                constraints
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.kind == ConstraintKind::Equality)
                    .map(|(i, _)| i)
                    .collect(),
            );

            let ineq_idx: Arc<Vec<usize>> = Arc::new(
                constraints
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.kind == ConstraintKind::Inequality)
                    .map(|(i, _)| i)
                    .collect(),
            );

            let has_eq = !eq_idx.is_empty();
            let has_ineq = !ineq_idx.is_empty();

            // Wrap the objective to accept an ArrayView
            let func_clone = func.clone();
            let al_fun = move |x: &ArrayView1<f64>| func_clone(x.as_slice().unwrap_or(&[]));

            // Build combined equality constraint closure (Clone via Arc)
            let al_options = AugmentedLagrangianOptions {
                max_iter: options.maxiter.unwrap_or(100),
                constraint_tol: options.ctol.unwrap_or(1e-8),
                optimality_tol: options.gtol.unwrap_or(1e-8),
                ..Default::default()
            };

            // Helper: emit a slice-backed value for a contiguous ArrayView1
            #[inline]
            fn view_to_slice(x: &ArrayView1<f64>) -> Vec<f64> {
                x.iter().copied().collect()
            }

            let result = if has_eq && has_ineq {
                let eq_arc = Arc::clone(&eq_idx);
                let eq_closure = move |x: &ArrayView1<f64>| {
                    let xs = view_to_slice(x);
                    Array1::from_vec(eq_arc.iter().map(|&i| (constraints[i].fun)(&xs)).collect())
                };
                let ineq_arc = Arc::clone(&ineq_idx);
                let ineq_closure = move |x: &ArrayView1<f64>| {
                    let xs = view_to_slice(x);
                    Array1::from_vec(
                        ineq_arc
                            .iter()
                            .map(|&i| (constraints[i].fun)(&xs))
                            .collect(),
                    )
                };
                minimize_augmented_lagrangian(
                    al_fun,
                    x0_arr,
                    Some(eq_closure),
                    Some(ineq_closure),
                    Some(al_options),
                )?
            } else if has_eq {
                let eq_arc = Arc::clone(&eq_idx);
                let eq_closure = move |x: &ArrayView1<f64>| {
                    let xs = view_to_slice(x);
                    Array1::from_vec(eq_arc.iter().map(|&i| (constraints[i].fun)(&xs)).collect())
                };
                minimize_augmented_lagrangian(
                    al_fun,
                    x0_arr,
                    Some(eq_closure),
                    None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
                    Some(al_options),
                )?
            } else {
                let ineq_arc = Arc::clone(&ineq_idx);
                let ineq_closure = move |x: &ArrayView1<f64>| {
                    let xs = view_to_slice(x);
                    Array1::from_vec(
                        ineq_arc
                            .iter()
                            .map(|&i| (constraints[i].fun)(&xs))
                            .collect(),
                    )
                };
                minimize_augmented_lagrangian(
                    al_fun,
                    x0_arr,
                    None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
                    Some(ineq_closure),
                    Some(al_options),
                )?
            };

            Ok(OptimizeResults::<f64> {
                x: result.x,
                fun: result.fun,
                nit: result.nit,
                nfev: result.nfev,
                success: result.success,
                message: result.message,
                jac: None,
                hess: None,
                constr: None,
                njev: 0,
                nhev: 0,
                maxcv: 0,
                status: if result.success { 0 } else { 1 },
            })
        }
        Method::SQP => sqp::minimize_sqp_compat(func, x0, constraints, &options),
    }
}
