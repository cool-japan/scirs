//! Gauss-Legendre quadrature for symbolic (`LoweredOp`) integrands.
//!
//! # Variable convention
//!
//! `Var(0)` represents the integration variable `x`. The integrand must
//! be expressed purely in terms of `Var(0)` (and constants). Higher-index
//! variables are permitted but will be treated as constants during
//! evaluation — they must appear in the `EvalCtx` bindings if used, so
//! for simplicity avoid them in integrands passed to this function.

use std::sync::Arc;

use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};
use scirs2_symbolic::eml::LoweredOp;

use crate::quadrature::gaussian::gauss_legendre as gauss_legendre_nodes;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Errors from [`quad_gauss_legendre_symbolic`].
#[derive(Debug)]
pub enum SymbolicQuadError {
    /// The symbolic evaluator returned an error (domain violation, unbound
    /// variable, division by zero, etc.).
    EvalError(String),
    /// The integration interval is empty or reversed: `a >= b`.
    InvalidInterval(f64, f64),
    /// The requested number of quadrature nodes is zero.
    InvalidNodeCount(usize),
    /// The internal Gauss-Legendre node/weight computation failed.
    NodeComputationError(String),
}

impl std::fmt::Display for SymbolicQuadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolicQuadError::EvalError(msg) => write!(f, "symbolic eval error: {msg}"),
            SymbolicQuadError::InvalidInterval(a, b) => {
                write!(f, "invalid interval [{a}, {b}]: must have a < b")
            }
            SymbolicQuadError::InvalidNodeCount(n) => {
                write!(f, "invalid node count {n}: must be >= 1")
            }
            SymbolicQuadError::NodeComputationError(msg) => {
                write!(f, "Gauss-Legendre node computation failed: {msg}")
            }
        }
    }
}

impl std::error::Error for SymbolicQuadError {}

// ---------------------------------------------------------------------------
// Main function
// ---------------------------------------------------------------------------

/// Gauss-Legendre quadrature for a symbolic integrand.
///
/// Approximates `∫_a^b integrand(x) dx` using an `n`-point Gauss-Legendre
/// rule. The integrand must be expressed as a [`LoweredOp`] with `Var(0)`
/// representing the integration variable `x`.
///
/// An `n`-point rule integrates polynomials of degree up to `2n − 1`
/// exactly.
///
/// # Arguments
///
/// - `integrand` — symbolic expression with `Var(0) = x`
/// - `a`, `b` — integration limits (`a < b` required)
/// - `n` — number of quadrature nodes (≥ 1)
///
/// # Errors
///
/// - [`SymbolicQuadError::InvalidInterval`] when `a >= b`
/// - [`SymbolicQuadError::InvalidNodeCount`] when `n == 0`
/// - [`SymbolicQuadError::EvalError`] when symbolic evaluation fails at a
///   quadrature node
/// - [`SymbolicQuadError::NodeComputationError`] when the internal
///   Golub-Welsch eigensolver fails
pub fn quad_gauss_legendre_symbolic(
    integrand: &Arc<LoweredOp>,
    a: f64,
    b: f64,
    n: usize,
) -> Result<f64, SymbolicQuadError> {
    if a >= b {
        return Err(SymbolicQuadError::InvalidInterval(a, b));
    }
    if n == 0 {
        return Err(SymbolicQuadError::InvalidNodeCount(n));
    }

    // Obtain Gauss-Legendre nodes and weights on [-1, 1]
    let (xi_vec, wi_vec) = gauss_legendre_nodes(n)
        .map_err(|e| SymbolicQuadError::NodeComputationError(e.to_string()))?;

    // Linear transformation from [-1, 1] to [a, b]:
    //   x = 0.5 * (b - a) * ξ + 0.5 * (a + b)
    //   w̃ = 0.5 * (b - a) * w
    let half_len = 0.5 * (b - a);
    let mid = 0.5 * (a + b);

    let mut integral = 0.0_f64;

    // Single-element binding buffer — Var(0) = x
    let mut binding = [0.0_f64; 1];

    for (xi, wi) in xi_vec.iter().zip(wi_vec.iter()) {
        let x_i = half_len * xi + mid;
        let w_i = half_len * wi;

        binding[0] = x_i;
        let ctx = EvalCtx::new(&binding);

        let val = eval_real(integrand.as_ref(), &ctx)
            .map_err(|e| SymbolicQuadError::EvalError(e.to_string()))?;

        integral += w_i * val;
    }

    Ok(integral)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn const_op(v: f64) -> Arc<LoweredOp> {
        Arc::new(LoweredOp::Const(v))
    }
    fn var0() -> Arc<LoweredOp> {
        Arc::new(LoweredOp::Var(0))
    }

    #[test]
    fn quad_constant() {
        // ∫_0^1 3 dx = 3
        let result = quad_gauss_legendre_symbolic(&const_op(3.0), 0.0, 1.0, 5).unwrap();
        assert!((result - 3.0).abs() < 1e-12, "result = {result}");
    }

    #[test]
    fn quad_linear_exact() {
        // ∫_0^1 x dx = 0.5  (exact with n ≥ 1)
        let result = quad_gauss_legendre_symbolic(&var0(), 0.0, 1.0, 1).unwrap();
        assert!((result - 0.5).abs() < 1e-12, "result = {result}");
    }

    #[test]
    fn quad_invalid_interval() {
        let err = quad_gauss_legendre_symbolic(&var0(), 1.0, 0.0, 5);
        assert!(matches!(
            err,
            Err(SymbolicQuadError::InvalidInterval(1.0, 0.0))
        ));
    }

    #[test]
    fn quad_invalid_node_count() {
        let err = quad_gauss_legendre_symbolic(&var0(), 0.0, 1.0, 0);
        assert!(matches!(err, Err(SymbolicQuadError::InvalidNodeCount(0))));
    }
}
