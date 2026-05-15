//! Derivative-Free Optimization Methods
//!
//! This module provides optimization algorithms that do not require gradient information.
//! These methods are suitable for black-box functions, noisy objectives, or functions
//! where gradients are unavailable or too expensive to compute.
//!
//! # Algorithms
//!
//! - [`NelderMeadSolver`]: Nelder-Mead simplex method
//! - [`BOBYQASolver`]: Bound Optimization BY Quadratic Approximation
//! - [`PatternSearchSolver`]: Generalized Pattern Search (GPS)
//!
//! # References
//!
//! - Nelder, J.A. & Mead, R. (1965). "A simplex method for function minimization"
//! - Powell, M.J.D. (1994). "A direct search optimization method that models the objective by quadratic interpolation"
//! - Torczon, V. (1997). "On the convergence of pattern search algorithms"

use crate::error::OptimizeResult;
use crate::result::OptimizeResults;
use scirs2_core::ndarray::{s, Array1, Array2};

pub mod bobyqa;
pub mod nelder_mead;
pub mod pattern_search;

pub use bobyqa::{BOBYQAOptions, BOBYQASolver};
pub use nelder_mead::{NelderMeadOptions, NelderMeadSolver};
pub use pattern_search::{PatternSearchOptions, PatternSearchSolver};

/// Derivative-free optimization result
#[derive(Debug, Clone)]
pub struct DfOptResult {
    /// Optimal solution
    pub x: Array1<f64>,
    /// Optimal function value
    pub fun: f64,
    /// Number of function evaluations
    pub nfev: usize,
    /// Number of iterations
    pub nit: usize,
    /// Whether convergence was achieved
    pub success: bool,
    /// Status message
    pub message: String,
}

impl From<DfOptResult> for OptimizeResults<f64> {
    fn from(r: DfOptResult) -> Self {
        OptimizeResults {
            x: r.x,
            fun: r.fun,
            jac: None,
            hess: None,
            constr: None,
            nit: r.nit,
            nfev: r.nfev,
            njev: 0,
            nhev: 0,
            maxcv: 0,
            message: r.message,
            success: r.success,
            status: if r.success { 0 } else { 1 },
        }
    }
}

/// Trait for derivative-free optimizers
pub trait DerivativeFreeOptimizer {
    /// Minimize a function starting from x0.
    fn minimize<F>(&self, func: F, x0: &[f64]) -> OptimizeResult<DfOptResult>
    where
        F: Fn(&[f64]) -> f64;
}

/// Compute the centroid of the best n vertices in the simplex (excluding the worst).
pub(crate) fn centroid(simplex: &Array2<f64>, n: usize) -> Array1<f64> {
    let mut c = Array1::zeros(n);
    for i in 0..n {
        c += &simplex.slice(s![i, ..]);
    }
    c / n as f64
}

/// L2 norm of a slice
pub(crate) fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Finite difference gradient (central differences)
#[allow(dead_code)]
pub(crate) fn finite_diff_grad<F: Fn(&[f64]) -> f64>(f: &F, x: &[f64], h: f64) -> Vec<f64> {
    let n = x.len();
    let mut g = vec![0.0; n];
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    for i in 0..n {
        xp[i] += h;
        xm[i] -= h;
        g[i] = (f(&xp) - f(&xm)) / (2.0 * h);
        xp[i] = x[i];
        xm[i] = x[i];
    }
    g
}

/// Clip a value to bounds
#[inline]
pub(crate) fn clip(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_centroid_2d() {
        let mut s = Array2::zeros((3, 2));
        s[[0, 0]] = 0.0;
        s[[0, 1]] = 0.0;
        s[[1, 0]] = 1.0;
        s[[1, 1]] = 0.0;
        s[[2, 0]] = 0.0;
        s[[2, 1]] = 1.0;
        // centroid of first 2 rows
        let c = centroid(&s, 2);
        assert!((c[0] - 0.5).abs() < 1e-12);
        assert!((c[1] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_norm() {
        let v = vec![3.0_f64, 4.0];
        assert!((norm(&v) - 5.0).abs() < 1e-12);
    }
}
