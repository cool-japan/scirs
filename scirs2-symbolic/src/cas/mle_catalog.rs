//! `cas::mle_catalog` — symbolic MLE estimators for a catalog of distributions.
//!
//! For each distribution family, data samples are represented as `Var(0)..Var(n-1)`
//! in `LoweredOp` expressions, and the MLE estimators are returned as closed-form
//! symbolic expressions over those variables.
//!
//! # Supported families
//! - [`DistFamily::Normal`] — `μ̂ = mean(xᵢ)`, `σ̂² = (1/n) Σ(xᵢ - μ̂)²`
//! - [`DistFamily::Exponential`] — `λ̂ = n / Σxᵢ`
//! - [`DistFamily::Bernoulli`] — `p̂ = (1/n) Σxᵢ`
//! - [`DistFamily::Geometric`] — `p̂ = n / (n + Σxᵢ)`
//! - [`DistFamily::Uniform`] — closed form requires `min`/`max`, not available in EML

use crate::cas::canonicalize::canonicalize;
use crate::eml::op::LoweredOp;

// -------------------------------------------------------------------------
// Public types
// -------------------------------------------------------------------------

/// A distribution family supported (or noted as not closed-form) in the catalog.
#[derive(Debug, Clone, PartialEq)]
pub enum DistFamily {
    /// Normal (Gaussian) distribution parameterised by `(μ, σ²)`.
    Normal,
    /// Exponential distribution parameterised by rate `λ`.
    Exponential,
    /// Bernoulli distribution parameterised by success probability `p`.
    Bernoulli,
    /// Geometric distribution (number of failures before first success) parameterised by `p`.
    Geometric,
    /// Uniform distribution — MLE requires `min`/`max`, not available in EML.
    Uniform,
}

/// Error type for [`symbolic_mle_catalog`].
#[derive(Debug)]
pub enum MleError {
    /// Fewer samples than the minimum required (n ≥ 1 for all families).
    TooFewSamples,
    /// No closed-form MLE expression in EML IR for this family.
    NotInClosedFormInEml {
        /// Name of the distribution family.
        family: &'static str,
    },
}

impl std::fmt::Display for MleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MleError::TooFewSamples => write!(f, "n_samples must be ≥ 1 — got 0"),
            MleError::NotInClosedFormInEml { family } => write!(
                f,
                "no closed-form EML MLE for distribution family '{family}'"
            ),
        }
    }
}

impl std::error::Error for MleError {}

/// A symbolic MLE estimator for a given distribution family with `n_samples` data points.
///
/// `estimators[i] = (param_name, LoweredOp)` where `Var(k)` for `k < n_samples`
/// represent the data samples.
pub struct MleEstimator {
    /// The distribution family this estimator targets.
    pub family: DistFamily,
    /// The number of data-sample variables (`Var(0)..Var(n_samples-1)`).
    pub n_samples: usize,
    /// Named parameter estimators as canonicalised `LoweredOp` expressions.
    ///
    /// Each element is `(parameter_name, expression_over_Var(0..n_samples-1))`.
    pub estimators: Vec<(&'static str, LoweredOp)>,
}

// -------------------------------------------------------------------------
// Public entry point
// -------------------------------------------------------------------------

/// Build symbolic MLE estimators for `family` given `n_samples` observations.
///
/// Observations are represented as `Var(0)`, `Var(1)`, … `Var(n_samples-1)`.
///
/// # Errors
/// - [`MleError::TooFewSamples`] when `n_samples == 0`.
/// - [`MleError::NotInClosedFormInEml`] when the family does not admit a
///   closed-form EML expression (currently `Uniform`).
pub fn symbolic_mle_catalog(
    family: DistFamily,
    n_samples: usize,
) -> Result<MleEstimator, MleError> {
    if n_samples == 0 {
        return Err(MleError::TooFewSamples);
    }
    let n_f64 = n_samples as f64;
    let vars: Vec<LoweredOp> = (0..n_samples).map(LoweredOp::Var).collect();

    match &family {
        DistFamily::Normal => {
            // μ̂ = (1/n) Σ xᵢ
            let sum = balanced_sum(vars.clone());
            let mu_hat = LoweredOp::Div(Box::new(sum), Box::new(LoweredOp::Const(n_f64)));
            let mu_hat = canonicalize(&mu_hat).into_op();

            // σ̂² = (1/n) Σ (xᵢ - μ̂)²
            let sq_diffs: Vec<LoweredOp> = vars
                .iter()
                .map(|xi| {
                    let diff = LoweredOp::Sub(Box::new(xi.clone()), Box::new(mu_hat.clone()));
                    LoweredOp::Pow(Box::new(diff), Box::new(LoweredOp::Const(2.0)))
                })
                .collect();
            let sum_sq = balanced_sum(sq_diffs);
            let sigma2_hat = LoweredOp::Div(Box::new(sum_sq), Box::new(LoweredOp::Const(n_f64)));
            let sigma2_hat = canonicalize(&sigma2_hat).into_op();

            Ok(MleEstimator {
                family,
                n_samples,
                estimators: vec![("mu", mu_hat), ("sigma2", sigma2_hat)],
            })
        }

        DistFamily::Exponential => {
            // λ̂ = n / Σxᵢ
            let sum = balanced_sum(vars);
            let lambda_hat = LoweredOp::Div(Box::new(LoweredOp::Const(n_f64)), Box::new(sum));
            let lambda_hat = canonicalize(&lambda_hat).into_op();
            Ok(MleEstimator {
                family,
                n_samples,
                estimators: vec![("lambda", lambda_hat)],
            })
        }

        DistFamily::Bernoulli => {
            // p̂ = (1/n) Σxᵢ
            let sum = balanced_sum(vars);
            let p_hat = LoweredOp::Div(Box::new(sum), Box::new(LoweredOp::Const(n_f64)));
            let p_hat = canonicalize(&p_hat).into_op();
            Ok(MleEstimator {
                family,
                n_samples,
                estimators: vec![("p", p_hat)],
            })
        }

        DistFamily::Geometric => {
            // p̂ = n / (n + Σxᵢ)
            let sum = balanced_sum(vars);
            let denom = LoweredOp::Add(Box::new(LoweredOp::Const(n_f64)), Box::new(sum));
            let p_hat = LoweredOp::Div(Box::new(LoweredOp::Const(n_f64)), Box::new(denom));
            let p_hat = canonicalize(&p_hat).into_op();
            Ok(MleEstimator {
                family,
                n_samples,
                estimators: vec![("p", p_hat)],
            })
        }

        DistFamily::Uniform => Err(MleError::NotInClosedFormInEml { family: "Uniform" }),
    }
}

// -------------------------------------------------------------------------
// Private helpers
// -------------------------------------------------------------------------

/// Combine a non-empty list of `LoweredOp` values into a balanced binary `Add` tree.
///
/// The pairwise reduction approach keeps the tree height O(log n), which matters
/// for gradient performance on large `n`.
///
/// # Panics
/// Panics if `ops` is empty — callers must ensure non-empty input.
fn balanced_sum(ops: Vec<LoweredOp>) -> LoweredOp {
    debug_assert!(
        !ops.is_empty(),
        "balanced_sum requires at least one element"
    );
    let mut work: Vec<LoweredOp> = ops;
    while work.len() > 1 {
        let mut next = Vec::with_capacity(work.len().div_ceil(2));
        let mut i = 0;
        while i < work.len() {
            if i + 1 < work.len() {
                next.push(LoweredOp::Add(
                    Box::new(work[i].clone()),
                    Box::new(work[i + 1].clone()),
                ));
                i += 2;
            } else {
                next.push(work[i].clone());
                i += 1;
            }
        }
        work = next;
    }
    // Safety: `ops` was non-empty and the loop only terminates when `work.len() == 1`.
    work.into_iter()
        .next()
        .expect("balanced_sum: non-empty work queue invariant violated")
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::eval::{eval_real, EvalCtx};

    fn eval(op: &LoweredOp, vals: &[f64]) -> f64 {
        let ctx = EvalCtx::new(vals);
        eval_real(op, &ctx).expect("eval_real failed in test")
    }

    #[test]
    fn normal_mle_mean_n3() {
        // μ̂([1,2,3]) should be 2.0
        let est = symbolic_mle_catalog(DistFamily::Normal, 3).expect("Normal MLE");
        let mu = &est.estimators[0].1;
        assert_eq!(est.estimators[0].0, "mu");
        let result = eval(mu, &[1.0, 2.0, 3.0]);
        assert!((result - 2.0).abs() < 1e-12, "μ̂ = {result}");
    }

    #[test]
    fn normal_mle_variance_n3() {
        // σ̂²([1,2,3]) = (1/3)[(1-2)² + (2-2)² + (3-2)²] = 2/3
        let est = symbolic_mle_catalog(DistFamily::Normal, 3).expect("Normal MLE");
        let sigma2 = &est.estimators[1].1;
        assert_eq!(est.estimators[1].0, "sigma2");
        let result = eval(sigma2, &[1.0, 2.0, 3.0]);
        let expected = 2.0_f64 / 3.0;
        assert!(
            (result - expected).abs() < 1e-12,
            "σ̂² = {result}, expected {expected}"
        );
    }

    #[test]
    fn exponential_mle_n2() {
        // λ̂([2,3]) = 2/(2+3) = 0.4
        let est = symbolic_mle_catalog(DistFamily::Exponential, 2).expect("Exponential MLE");
        let lambda = &est.estimators[0].1;
        assert_eq!(est.estimators[0].0, "lambda");
        let result = eval(lambda, &[2.0, 3.0]);
        assert!((result - 0.4).abs() < 1e-12, "λ̂ = {result}");
    }

    #[test]
    fn bernoulli_mle_n4() {
        // p̂([1,0,1,1]) = 3/4 = 0.75
        let est = symbolic_mle_catalog(DistFamily::Bernoulli, 4).expect("Bernoulli MLE");
        let p = &est.estimators[0].1;
        assert_eq!(est.estimators[0].0, "p");
        let result = eval(p, &[1.0, 0.0, 1.0, 1.0]);
        assert!((result - 0.75).abs() < 1e-12, "p̂ = {result}");
    }

    #[test]
    fn uniform_mle_not_in_closed_form() {
        let result = symbolic_mle_catalog(DistFamily::Uniform, 5);
        assert!(
            matches!(
                result,
                Err(MleError::NotInClosedFormInEml { family: "Uniform" })
            ),
            "Expected NotInClosedFormInEml for Uniform"
        );
    }
}
