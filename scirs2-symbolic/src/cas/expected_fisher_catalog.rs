//! `cas::expected_fisher_catalog` — symbolic expected Fisher information matrices.
//!
//! Returns the per-sample Fisher information matrix (times n) for each supported
//! distribution family as a 2-D `Vec<Vec<LoweredOp>>` of canonicalized expressions.
//!
//! # Variable convention
//! - `Var(0)` = first parameter (μ for Normal, λ for Exp, p for Bernoulli/Geometric)
//! - `Var(1)` = second parameter (σ for Normal)
//!
//! # Uniform
//!
//! The Uniform distribution has a boundary-determined support, which violates
//! the standard Fisher regularity conditions (interchange of differentiation
//! and integration). It therefore returns
//! [`ExpectedFisherError::UnsupportedFamily`].

use crate::cas::canonicalize::canonicalize;
use crate::cas::mle_catalog::DistFamily;
use crate::eml::op::LoweredOp;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error returned by [`expected_fisher_catalog`].
#[derive(Debug, Clone, PartialEq)]
pub enum ExpectedFisherError {
    /// The family is not supported (e.g. Uniform violates regularity conditions).
    UnsupportedFamily(DistFamily),
    /// `n_samples` was zero.
    ZeroSamples,
}

impl std::fmt::Display for ExpectedFisherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpectedFisherError::UnsupportedFamily(fam) => {
                write!(
                    f,
                    "expected Fisher catalog does not support family {fam:?} \
                     (boundary-determined support violates regularity)"
                )
            }
            ExpectedFisherError::ZeroSamples => {
                write!(f, "n_samples must be ≥ 1")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Builder helpers (no unwrap)
// ---------------------------------------------------------------------------

fn con(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}
fn var(i: usize) -> LoweredOp {
    LoweredOp::Var(i)
}
fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Box::new(a), Box::new(b))
}
fn div(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Div(Box::new(a), Box::new(b))
}
fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Sub(Box::new(a), Box::new(b))
}
fn canon(op: LoweredOp) -> LoweredOp {
    canonicalize(&op).into_op()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Return the expected Fisher information matrix for `n_samples` i.i.d.
/// observations from the given distribution family.
///
/// The returned matrix has entries as canonicalized [`LoweredOp`] expressions
/// using the variable conventions of [`crate::cas::moments_catalog`].
///
/// # Errors
/// - [`ExpectedFisherError::ZeroSamples`] if `n_samples == 0`.
/// - [`ExpectedFisherError::UnsupportedFamily`] for `Uniform` (non-standard
///   regularity due to boundary-determined support).
pub fn expected_fisher_catalog(
    family: DistFamily,
    n_samples: usize,
) -> Result<Vec<Vec<LoweredOp>>, ExpectedFisherError> {
    if n_samples == 0 {
        return Err(ExpectedFisherError::ZeroSamples);
    }

    match family {
        DistFamily::Normal => {
            // Parameters: μ = Var(0), σ = Var(1)
            // Per-sample I(μ) = 1/σ², I(σ) = 2/σ² (for σ parameterization)
            // Total (n samples): n * [[1/σ², 0], [0, 2/σ²]]
            let n = n_samples as f64;
            let sigma_sq = mul(var(1), var(1));

            let i_00 = canon(mul(con(n), div(con(1.0), sigma_sq.clone())));
            let i_01 = canon(con(0.0));
            let i_10 = canon(con(0.0));
            let i_11 = canon(mul(con(2.0 * n), div(con(1.0), sigma_sq)));

            Ok(vec![vec![i_00, i_01], vec![i_10, i_11]])
        }

        DistFamily::Exponential => {
            // Parameter: λ = Var(0)
            // Per-sample I(λ) = 1/λ²
            // Total: n / λ²
            let n = n_samples as f64;
            let lambda_sq = mul(var(0), var(0));
            let i_00 = canon(mul(con(n), div(con(1.0), lambda_sq)));

            Ok(vec![vec![i_00]])
        }

        DistFamily::Bernoulli => {
            // Parameter: p = Var(0)
            // Per-sample I(p) = 1 / (p*(1-p))
            // Total: n / (p*(1-p))
            let n = n_samples as f64;
            let p_times_one_minus_p = mul(var(0), sub(con(1.0), var(0)));
            let i_00 = canon(mul(con(n), div(con(1.0), p_times_one_minus_p)));

            Ok(vec![vec![i_00]])
        }

        DistFamily::Geometric => {
            // Convention: P(X=k) = p*(1-p)^k
            // Parameter: p = Var(0)
            // Per-sample I(p) = 1 / (p²*(1-p))
            // Total: n / (p²*(1-p))
            let n = n_samples as f64;
            let p_sq = mul(var(0), var(0));
            let one_minus_p = sub(con(1.0), var(0));
            let denom = mul(p_sq, one_minus_p);
            let i_00 = canon(mul(con(n), div(con(1.0), denom)));

            Ok(vec![vec![i_00]])
        }

        DistFamily::Uniform => Err(ExpectedFisherError::UnsupportedFamily(DistFamily::Uniform)),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cas::mle_catalog::DistFamily;
    use crate::eml::eval::{eval_real, EvalCtx};

    fn eval(op: &LoweredOp, vars: &[f64]) -> f64 {
        let ctx = EvalCtx::new(vars);
        eval_real(op, &ctx).expect("eval_real failed in test")
    }

    #[test]
    fn test_normal_fisher_n100_sigma1() {
        // At σ=1, n=100: I[0][0] = 100/1 = 100.0, I[0][1]=0, I[1][0]=0, I[1][1]=200
        let fisher = expected_fisher_catalog(DistFamily::Normal, 100).expect("catalog failed");
        // vars: Var(0)=μ=0 (irrelevant), Var(1)=σ=1
        let vars = [0.0_f64, 1.0];
        let i00 = eval(&fisher[0][0], &vars);
        let i01 = eval(&fisher[0][1], &vars);
        let i10 = eval(&fisher[1][0], &vars);
        let i11 = eval(&fisher[1][1], &vars);
        assert!(
            (i00 - 100.0).abs() < 1e-10,
            "I[0][0] should be 100.0, got {i00}"
        );
        assert!(i01.abs() < 1e-10, "I[0][1] should be 0.0, got {i01}");
        assert!(i10.abs() < 1e-10, "I[1][0] should be 0.0, got {i10}");
        assert!(
            (i11 - 200.0).abs() < 1e-10,
            "I[1][1] should be 200.0, got {i11}"
        );
    }

    #[test]
    fn test_exponential_fisher_n10_lambda2() {
        // At λ=2, n=10: I[0][0] = 10/4 = 2.5
        let fisher = expected_fisher_catalog(DistFamily::Exponential, 10).expect("catalog failed");
        let vars = [2.0_f64];
        let i00 = eval(&fisher[0][0], &vars);
        assert!(
            (i00 - 2.5).abs() < 1e-10,
            "Exponential I[0][0] at λ=2, n=10 should be 2.5, got {i00}"
        );
    }

    #[test]
    fn test_bernoulli_fisher_n10_p_half() {
        // At p=0.5, n=10: I[0][0] = 10 / (0.5*0.5) = 40.0
        let fisher = expected_fisher_catalog(DistFamily::Bernoulli, 10).expect("catalog failed");
        let vars = [0.5_f64];
        let i00 = eval(&fisher[0][0], &vars);
        assert!(
            (i00 - 40.0).abs() < 1e-10,
            "Bernoulli I[0][0] at p=0.5, n=10 should be 40.0, got {i00}"
        );
    }

    #[test]
    fn test_uniform_unsupported() {
        let result = expected_fisher_catalog(DistFamily::Uniform, 10);
        assert!(
            matches!(
                result,
                Err(ExpectedFisherError::UnsupportedFamily(DistFamily::Uniform))
            ),
            "Uniform should return UnsupportedFamily, got {result:?}"
        );
    }
}
