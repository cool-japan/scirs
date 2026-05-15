//! `cas::observed_fisher` — observed Fisher information matrix via `-Hessian(log-likelihood)`.
//!
//! The observed Fisher information matrix is defined as:
//!
//! ```text
//! I(θ)_ij = -∂²/∂θᵢ∂θⱼ  ℓ(θ)
//! ```
//!
//! where `ℓ(θ)` is the log-likelihood. This module computes it symbolically
//! from a `LoweredOp` representation of the log-likelihood.
//!
//! # Usage
//!
//! ```rust
//! use scirs2_symbolic::eml::LoweredOp;
//! use scirs2_symbolic::cas::observed_fisher_matrix;
//!
//! // log-lik = -0.5 * x₀²  => second derivative = -1 => Fisher = 1
//! let log_lik = LoweredOp::Mul(
//!     Box::new(LoweredOp::Const(-0.5)),
//!     Box::new(LoweredOp::Pow(
//!         Box::new(LoweredOp::Var(0)),
//!         Box::new(LoweredOp::Const(2.0)),
//!     )),
//! );
//! let fim = observed_fisher_matrix(&log_lik, &[0]);
//! assert_eq!(fim.len(), 1);
//! ```

use crate::cas::canonicalize::canonicalize;
use crate::eml::grad::hessian;
use crate::eml::op::LoweredOp;

// -------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------

/// Compute the observed Fisher information matrix symbolically.
///
/// Given a log-likelihood `log_lik` expressed over variables `Var(0)..Var(K-1)`,
/// and a slice `param_indices` identifying the parameter variables of interest,
/// returns the `|param_indices| × |param_indices|` matrix:
///
/// ```text
/// FIM[i][j] = -∂²ℓ/∂θ_{param_indices[i]} ∂θ_{param_indices[j]}
/// ```
///
/// Each entry is canonicalised before return.
///
/// # Returns
/// - Empty `Vec` when `param_indices.is_empty()`.
/// - Otherwise a `Vec<Vec<LoweredOp>>` of shape `m × m` where `m = param_indices.len()`.
pub fn observed_fisher_matrix(log_lik: &LoweredOp, param_indices: &[usize]) -> Vec<Vec<LoweredOp>> {
    if param_indices.is_empty() {
        return Vec::new();
    }

    // Determine the number of variables needed for the Hessian.
    // We compute partials for at least (max_param_idx + 1) variables.
    let max_param_idx = param_indices.iter().copied().max().unwrap_or(0);
    let n_vars = max_param_idx + 1;

    // Compute the full Hessian of log_lik.
    let hess = hessian(log_lik, n_vars);

    // Extract sub-matrix at param_indices and negate (FIM = -H).
    let m = param_indices.len();
    let mut fim = Vec::with_capacity(m);

    for &row_idx in param_indices {
        let mut row = Vec::with_capacity(m);
        for &col_idx in param_indices {
            let h_entry = if row_idx < hess.len() && col_idx < hess[row_idx].len() {
                hess[row_idx][col_idx].clone()
            } else {
                // Out-of-range — log_lik is constant w.r.t. this variable pair.
                LoweredOp::Const(0.0)
            };

            // FIM[i][j] = -H[row][col]
            let neg_entry = LoweredOp::Neg(Box::new(h_entry));
            let canon_entry = canonicalize(&neg_entry).into_op();
            row.push(canon_entry);
        }
        fim.push(row);
    }

    fim
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::eval::{eval_real, EvalCtx};

    fn eval_op(op: &LoweredOp, vals: &[f64]) -> f64 {
        let ctx = EvalCtx::new(vals);
        eval_real(op, &ctx).expect("eval_real failed in test")
    }

    #[test]
    fn single_param_quadratic_log_lik() {
        // log_lik = -0.5 * x₀²
        // ∂²ℓ/∂x₀² = -1  =>  FIM = 1
        let log_lik = LoweredOp::Mul(
            Box::new(LoweredOp::Const(-0.5)),
            Box::new(LoweredOp::Pow(
                Box::new(LoweredOp::Var(0)),
                Box::new(LoweredOp::Const(2.0)),
            )),
        );
        let fim = observed_fisher_matrix(&log_lik, &[0]);
        assert_eq!(fim.len(), 1);
        assert_eq!(fim[0].len(), 1);
        let val = eval_op(&fim[0][0], &[1.0]);
        assert!((val - 1.0).abs() < 1e-12, "FIM[0][0] = {val}");
    }

    #[test]
    fn normal_log_lik_one_data_point() {
        // log_lik = -(Var(0) - Var(2))² / 2.0 - 0.5 * ln(1.0)
        // Var(0) = data point x, Var(1) = unused, Var(2) = μ  (σ²=1 hardcoded)
        // ∂²ℓ/∂μ² = -1  =>  FIM[0][0] = 1
        let data_minus_mu =
            LoweredOp::Sub(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(2)));
        let sq = LoweredOp::Pow(Box::new(data_minus_mu), Box::new(LoweredOp::Const(2.0)));
        let term1 = LoweredOp::Div(
            Box::new(LoweredOp::Neg(Box::new(sq))),
            Box::new(LoweredOp::Const(2.0)),
        );
        // -0.5 * ln(1.0) = 0; include for representational fidelity
        let term2 = LoweredOp::Mul(
            Box::new(LoweredOp::Const(-0.5)),
            Box::new(LoweredOp::Ln(Box::new(LoweredOp::Const(1.0)))),
        );
        let log_lik = LoweredOp::Add(Box::new(term1), Box::new(term2));

        let fim = observed_fisher_matrix(&log_lik, &[2]);
        assert_eq!(fim.len(), 1);
        assert_eq!(fim[0].len(), 1);
        // eval at [data=0.5, unused=0.0, mu=0.5] — data doesn't affect ∂²ℓ/∂μ²
        let val = eval_op(&fim[0][0], &[0.5, 0.0, 0.5]);
        assert!((val - 1.0).abs() < 1e-12, "FIM[0][0] = {val}");
    }

    #[test]
    fn separable_log_lik_cross_derivative_zero() {
        // log_lik = -Var(0)² - Var(1)²
        // Cross-partial ∂²ℓ/∂x₀∂x₁ = 0  =>  FIM[0][1] = 0
        let term0 = LoweredOp::Neg(Box::new(LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(2.0)),
        )));
        let term1 = LoweredOp::Neg(Box::new(LoweredOp::Pow(
            Box::new(LoweredOp::Var(1)),
            Box::new(LoweredOp::Const(2.0)),
        )));
        let log_lik = LoweredOp::Add(Box::new(term0), Box::new(term1));

        let fim = observed_fisher_matrix(&log_lik, &[0, 1]);
        assert_eq!(fim.len(), 2);
        assert_eq!(fim[0].len(), 2);
        // Off-diagonal entry at (0,1) — cross derivative
        let cross = eval_op(&fim[0][1], &[1.0, 1.0]);
        assert!((cross - 0.0).abs() < 1e-12, "FIM[0][1] = {cross}");
    }

    #[test]
    fn empty_param_indices_returns_empty() {
        let log_lik = LoweredOp::Var(0);
        let fim = observed_fisher_matrix(&log_lik, &[]);
        assert!(fim.is_empty(), "Expected empty FIM for empty param_indices");
    }
}
