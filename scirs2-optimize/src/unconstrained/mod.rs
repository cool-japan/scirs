//! Unconstrained optimization algorithms
//!
//! This module provides various algorithms for unconstrained minimization problems.

use crate::error::OptimizeError;
use ndarray::{Array1, ArrayView1};
use std::fmt;

// Sub-modules
pub mod bfgs;
pub mod conjugate_gradient;
pub mod lbfgs;
pub mod line_search;
pub mod nelder_mead;
pub mod newton;
pub mod powell;
pub mod result;
pub mod trust_region;
pub mod utils;

// Import result type
pub use result::OptimizeResult;

// Re-export commonly used items
pub use bfgs::minimize_bfgs;
pub use conjugate_gradient::minimize_conjugate_gradient;
pub use lbfgs::{minimize_lbfgs, minimize_lbfgsb};
pub use nelder_mead::minimize_nelder_mead;
pub use newton::minimize_newton_cg;
pub use powell::minimize_powell;
pub use trust_region::{minimize_trust_exact, minimize_trust_krylov, minimize_trust_ncg};

/// Optimization methods for unconstrained minimization.
#[derive(Debug, Clone, Copy)]
pub enum Method {
    /// Nelder-Mead simplex method
    NelderMead,
    /// Powell's method
    Powell,
    /// Conjugate gradient method
    CG,
    /// BFGS quasi-Newton method
    BFGS,
    /// Limited-memory BFGS method
    LBFGS,
    /// Limited-memory BFGS method with bounds support
    LBFGSB,
    /// Newton's method with conjugate gradient solver
    NewtonCG,
    /// Trust-region Newton method with conjugate gradient solver
    TrustNCG,
    /// Trust-region method with Krylov subproblem solver
    TrustKrylov,
    /// Trust-region method with exact subproblem solver
    TrustExact,
}

/// Bounds for optimization variables
#[derive(Debug, Clone)]
pub struct Bounds {
    /// Lower bounds
    pub lower: Vec<Option<f64>>,
    /// Upper bounds
    pub upper: Vec<Option<f64>>,
}

/// Options for optimization algorithms
#[derive(Debug, Clone)]
pub struct Options {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Maximum number of function evaluations
    pub max_fev: Option<usize>,
    /// Function tolerance for convergence
    pub ftol: f64,
    /// Change tolerance for convergence
    pub xtol: f64,
    /// Gradient tolerance for convergence
    pub gtol: f64,
    /// Initial step size
    pub initial_step: Option<f64>,
    /// Maximum step size for line search
    pub maxstep: Option<f64>,
    /// Whether to use finite differences for gradient
    pub finite_diff: bool,
    /// Finite difference step size
    pub eps: f64,
    /// Initial trust-region radius for trust-region methods
    pub trust_radius: Option<f64>,
    /// Maximum trust-region radius for trust-region methods
    pub max_trust_radius: Option<f64>,
    /// Minimum trust-region radius for trust-region methods
    pub min_trust_radius: Option<f64>,
    /// Tolerance for the trust-region subproblem
    pub trust_tol: Option<f64>,
    /// Maximum iterations for trust-region subproblem
    pub trust_max_iter: Option<usize>,
    /// Threshold for accepting a step in the trust-region method
    pub trust_eta: Option<f64>,
    /// Bounds constraints for variables
    pub bounds: Option<Bounds>,
}

// Implement Display for Method
impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Method::NelderMead => write!(f, "Nelder-Mead"),
            Method::Powell => write!(f, "Powell"),
            Method::CG => write!(f, "Conjugate Gradient"),
            Method::BFGS => write!(f, "BFGS"),
            Method::LBFGS => write!(f, "L-BFGS"),
            Method::LBFGSB => write!(f, "L-BFGS-B"),
            Method::NewtonCG => write!(f, "Newton-CG"),
            Method::TrustNCG => write!(f, "Trust-NCG"),
            Method::TrustKrylov => write!(f, "Trust-Krylov"),
            Method::TrustExact => write!(f, "Trust-Exact"),
        }
    }
}

// Implement Default for Options
impl Default for Options {
    fn default() -> Self {
        Options {
            max_iter: 1000,
            max_fev: None,
            ftol: 1e-8,
            xtol: 1e-8,
            gtol: 1e-5,
            initial_step: None,
            maxstep: None,
            finite_diff: false,
            eps: 1.4901161193847656e-8,
            trust_radius: Some(1.0),
            max_trust_radius: Some(100.0),
            min_trust_radius: Some(1e-10),
            trust_tol: Some(1e-8),
            trust_max_iter: Some(100),
            trust_eta: Some(0.1),
            bounds: None,
        }
    }
}

// Implement Bounds methods
impl Bounds {
    /// Create new bounds from arrays
    pub fn new(bounds: &[(Option<f64>, Option<f64>)]) -> Self {
        let (lower, upper): (Vec<_>, Vec<_>) = bounds.iter().cloned().unzip();
        Self { lower, upper }
    }

    /// Create bounds from vectors
    pub fn from_vecs(lb: Vec<Option<f64>>, ub: Vec<Option<f64>>) -> Result<Self, OptimizeError> {
        if lb.len() != ub.len() {
            return Err(OptimizeError::ValueError(
                "Lower and upper bounds must have the same length".to_string(),
            ));
        }

        for (l, u) in lb.iter().zip(ub.iter()) {
            if let (Some(l_val), Some(u_val)) = (l, u) {
                if l_val > u_val {
                    return Err(OptimizeError::ValueError(
                        "Lower bound must be less than or equal to upper bound".to_string(),
                    ));
                }
            }
        }

        Ok(Self {
            lower: lb,
            upper: ub,
        })
    }

    /// Check if point is feasible
    pub fn is_feasible(&self, x: &[f64]) -> bool {
        if x.len() != self.lower.len() {
            return false;
        }

        for (&xi, (&lb, &ub)) in x.iter().zip(self.lower.iter().zip(self.upper.iter())) {
            if let Some(l) = lb {
                if xi < l {
                    return false;
                }
            }
            if let Some(u) = ub {
                if xi > u {
                    return false;
                }
            }
        }
        true
    }

    /// Project point onto feasible region
    pub fn project(&self, x: &mut [f64]) {
        for (xi, (&lb, &ub)) in x.iter_mut().zip(self.lower.iter().zip(self.upper.iter())) {
            if let Some(l) = lb {
                if *xi < l {
                    *xi = l;
                }
            }
            if let Some(u) = ub {
                if *xi > u {
                    *xi = u;
                }
            }
        }
    }

    /// Check if bounds are active
    pub fn has_bounds(&self) -> bool {
        self.lower.iter().any(|b| b.is_some()) || self.upper.iter().any(|b| b.is_some())
    }
}

/// Main minimize function for unconstrained optimization
pub fn minimize<F, S>(
    fun: F,
    x0: &[f64],
    method: Method,
    options: Option<Options>,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S + Clone,
    S: Into<f64> + Clone,
{
    let options = &options.unwrap_or_default();
    let x0 = Array1::from_vec(x0.to_vec());

    // Check initial point feasibility if bounds are provided
    if let Some(ref bounds) = options.bounds {
        if !bounds.is_feasible(x0.as_slice().unwrap()) {
            return Err(OptimizeError::ValueError(
                "Initial point is not feasible".to_string(),
            ));
        }
    }

    match method {
        Method::NelderMead => nelder_mead::minimize_nelder_mead(fun, x0, options),
        Method::Powell => powell::minimize_powell(fun, x0, options),
        Method::CG => conjugate_gradient::minimize_conjugate_gradient(fun, x0, options),
        Method::BFGS => bfgs::minimize_bfgs(fun, x0, options),
        Method::LBFGS => lbfgs::minimize_lbfgs(fun, x0, options),
        Method::LBFGSB => lbfgs::minimize_lbfgsb(fun, x0, options),
        Method::NewtonCG => newton::minimize_newton_cg(fun, x0, options),
        Method::TrustNCG => trust_region::minimize_trust_ncg(fun, x0, options),
        Method::TrustKrylov => trust_region::minimize_trust_krylov(fun, x0, options),
        Method::TrustExact => trust_region::minimize_trust_exact(fun, x0, options),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_simple_quadratic() {
        let quadratic = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };

        let x0 = vec![1.0, 1.0];
        let result = minimize(quadratic, &x0, Method::BFGS, None);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-6);
    }
}
