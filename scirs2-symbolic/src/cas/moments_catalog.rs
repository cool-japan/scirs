//! `cas::moments_catalog` — closed-form symbolic moments for a catalog of distributions.
//!
//! For each supported family, returns the mean, variance, and MGF as
//! [`LoweredOp`] expressions. Variable conventions:
//! - `Var(0)` = first parameter (μ for Normal, λ for Exp, p for Bernoulli/Geometric, a for Uniform)
//! - `Var(1)` = second parameter (σ for Normal, b for Uniform)
//! - `Var(2)` = `t` (MGF parameter, M(t) = E[exp(tX)])
//!
//! # Geometric convention
//!
//! The Geometric distribution here follows the convention `P(X=k) = p·(1-p)^k`
//! for `k ∈ {0,1,2,...}` (number of failures before first success). Under this
//! convention `E[X] = (1-p)/p` and `Var(X) = (1-p)/p²`.
//!
//! # Uniform MGF
//!
//! The Uniform distribution MGF `(exp(tb) - exp(ta)) / (t*(b-a))` requires
//! a case split at `t=0` which cannot be expressed in real EML IR. The `mgf`
//! field is therefore `None` for `Uniform`.

use crate::cas::canonicalize::canonicalize;
use crate::cas::mle_catalog::DistFamily;
use crate::eml::op::LoweredOp;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Symbolic moments for a distribution family.
pub struct MomentsCatalog {
    /// The distribution family.
    pub family: DistFamily,
    /// Symbolic mean `E[X]`.
    pub mean: LoweredOp,
    /// Symbolic variance Var(X).
    pub variance: LoweredOp,
    /// Symbolic moment-generating function M(t) = E[exp(tX)].
    ///
    /// `None` if the MGF cannot be expressed in real EML IR (e.g. Uniform,
    /// where the formula requires a case split at t=0).
    pub mgf: Option<LoweredOp>,
}

/// Error returned by [`symbolic_moments_catalog`].
#[derive(Debug, Clone, PartialEq)]
pub enum MomentsError {
    /// The family is not yet supported by the moments catalog.
    UnsupportedFamily(DistFamily),
    /// The MGF does not exist or cannot be expressed in EML IR.
    MgfDoesNotExist,
}

impl std::fmt::Display for MomentsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MomentsError::UnsupportedFamily(fam) => {
                write!(f, "moments catalog does not support family {fam:?}")
            }
            MomentsError::MgfDoesNotExist => {
                write!(f, "MGF does not exist or cannot be expressed in EML IR")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Builder helpers (no unwrap)
// ---------------------------------------------------------------------------

/// Convenience constructors so the builder code stays readable.
fn con(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}
fn var(i: usize) -> LoweredOp {
    LoweredOp::Var(i)
}
fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Box::new(a), Box::new(b))
}
fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Sub(Box::new(a), Box::new(b))
}
fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Box::new(a), Box::new(b))
}
fn div(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Div(Box::new(a), Box::new(b))
}
fn pow(base: LoweredOp, exp: LoweredOp) -> LoweredOp {
    LoweredOp::Pow(Box::new(base), Box::new(exp))
}
fn exp(inner: LoweredOp) -> LoweredOp {
    LoweredOp::Exp(Box::new(inner))
}
fn canon(op: LoweredOp) -> LoweredOp {
    canonicalize(&op).into_op()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Return symbolic moments for the given distribution family.
///
/// # Variable convention
/// - `Var(0)` = first parameter (μ, λ, p, or a)
/// - `Var(1)` = second parameter (σ for Normal, b for Uniform)
/// - `Var(2)` = `t` (MGF argument)
///
/// All returned `LoweredOp` values are canonicalized.
pub fn symbolic_moments_catalog(family: DistFamily) -> Result<MomentsCatalog, MomentsError> {
    match family {
        DistFamily::Normal => {
            // Parameters: μ = Var(0), σ = Var(1)
            // Mean: μ
            let mean = canon(var(0));

            // Variance: σ²
            let variance = canon(mul(var(1), var(1)));

            // MGF: M(t) = exp(μ*t + σ²*t²/2)
            // = exp(Var(0)*Var(2) + Var(1)^2 * Var(2)^2 / 2)
            let mu_t = mul(var(0), var(2));
            let sigma_sq = mul(var(1), var(1));
            let t_sq = mul(var(2), var(2));
            let sigma_sq_t_sq_half = div(mul(sigma_sq, t_sq), con(2.0));
            let exponent = add(mu_t, sigma_sq_t_sq_half);
            let mgf = canon(exp(exponent));

            Ok(MomentsCatalog {
                family: DistFamily::Normal,
                mean,
                variance,
                mgf: Some(mgf),
            })
        }

        DistFamily::Exponential => {
            // Parameter: λ = Var(0)
            // Mean: 1/λ
            let mean = canon(div(con(1.0), var(0)));

            // Variance: 1/λ²
            let variance = canon(div(con(1.0), mul(var(0), var(0))));

            // MGF: λ/(λ-t)  (symbolically; valid for t < λ)
            // Var(2) = t
            let mgf = canon(div(var(0), sub(var(0), var(2))));

            Ok(MomentsCatalog {
                family: DistFamily::Exponential,
                mean,
                variance,
                mgf: Some(mgf),
            })
        }

        DistFamily::Bernoulli => {
            // Parameter: p = Var(0)
            // Mean: p
            let mean = canon(var(0));

            // Variance: p*(1-p)
            let variance = canon(mul(var(0), sub(con(1.0), var(0))));

            // MGF: (1-p) + p*exp(t)
            // Var(2) = t
            let mgf = canon(add(sub(con(1.0), var(0)), mul(var(0), exp(var(2)))));

            Ok(MomentsCatalog {
                family: DistFamily::Bernoulli,
                mean,
                variance,
                mgf: Some(mgf),
            })
        }

        DistFamily::Geometric => {
            // Convention: P(X=k) = p*(1-p)^k for k in {0,1,2,...}
            // Parameter: p = Var(0)
            // Mean: (1-p)/p
            let mean = canon(div(sub(con(1.0), var(0)), var(0)));

            // Variance: (1-p)/p²
            let variance = canon(div(sub(con(1.0), var(0)), mul(var(0), var(0))));

            // MGF: p*exp(t) / (1-(1-p)*exp(t))
            // Var(2) = t
            let p_exp_t = mul(var(0), exp(var(2)));
            let one_minus_p = sub(con(1.0), var(0));
            let denom = sub(con(1.0), mul(one_minus_p, exp(var(2))));
            let mgf = canon(div(p_exp_t, denom));

            Ok(MomentsCatalog {
                family: DistFamily::Geometric,
                mean,
                variance,
                mgf: Some(mgf),
            })
        }

        DistFamily::Uniform => {
            // Parameters: a = Var(0), b = Var(1)
            // Mean: (a+b)/2
            let mean = canon(div(add(var(0), var(1)), con(2.0)));

            // Variance: (b-a)²/12
            let b_minus_a = sub(var(1), var(0));
            let variance = canon(div(pow(b_minus_a, con(2.0)), con(12.0)));

            // MGF: not expressible in real EML (case split at t=0)
            Ok(MomentsCatalog {
                family: DistFamily::Uniform,
                mean,
                variance,
                mgf: None,
            })
        }
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
    use crate::eml::grad::grad;

    fn eval(op: &LoweredOp, vars: &[f64]) -> f64 {
        let ctx = EvalCtx::new(vars);
        eval_real(op, &ctx).expect("eval_real failed in test")
    }

    #[test]
    fn test_normal_mean_is_var0() {
        let m = symbolic_moments_catalog(DistFamily::Normal).expect("catalog failed");
        let expected_mean = canonicalize(&LoweredOp::Var(0)).into_op();
        assert_eq!(m.mean, expected_mean, "Normal mean should be Var(0)");
    }

    #[test]
    fn test_normal_variance_at_sigma2() {
        let m = symbolic_moments_catalog(DistFamily::Normal).expect("catalog failed");
        // Var(0)=μ=0, Var(1)=σ=2 → variance = σ² = 4
        let val = eval(&m.variance, &[0.0, 2.0]);
        assert!(
            (val - 4.0).abs() < 1e-10,
            "Normal variance at σ=2 should be 4, got {val}"
        );
    }

    #[test]
    fn test_normal_mgf_at_t0_is_1() {
        let m = symbolic_moments_catalog(DistFamily::Normal).expect("catalog failed");
        let mgf = m.mgf.expect("Normal should have MGF");
        // M(0) = exp(0) = 1 for any μ, σ
        let val = eval(&mgf, &[3.0, 1.0, 0.0]);
        assert!(
            (val - 1.0).abs() < 1e-10,
            "Normal MGF at t=0 should be 1, got {val}"
        );
    }

    #[test]
    fn test_normal_mgf_grad_at_t0_equals_mu() {
        let m = symbolic_moments_catalog(DistFamily::Normal).expect("catalog failed");
        let mgf = m.mgf.expect("Normal should have MGF");
        // d/dt M(t)|_{t=0} = E[X] = μ
        let dm_dt = grad(&mgf, 2);
        let mu = 3.0_f64;
        let val = eval(&dm_dt, &[mu, 1.0, 0.0]);
        assert!(
            (val - mu).abs() < 1e-8,
            "Normal MGF gradient at t=0 should be μ={mu}, got {val}"
        );
    }

    #[test]
    fn test_exponential_mean_at_lambda2() {
        let m = symbolic_moments_catalog(DistFamily::Exponential).expect("catalog failed");
        // Mean = 1/λ; at λ=2, mean = 0.5
        let val = eval(&m.mean, &[2.0]);
        assert!(
            (val - 0.5).abs() < 1e-10,
            "Exponential mean at λ=2 should be 0.5, got {val}"
        );
    }

    #[test]
    fn test_exponential_mgf_at_t0_is_1() {
        let m = symbolic_moments_catalog(DistFamily::Exponential).expect("catalog failed");
        let mgf = m.mgf.expect("Exponential should have MGF");
        // MGF = Var(0)/(Var(0)-Var(2)) where Var(0)=λ, Var(2)=t
        // At t=0, M(0) = λ/λ = 1. Need 3 variable slots: [λ, unused, t]
        let val = eval(&mgf, &[2.0, 0.0, 0.0]);
        assert!(
            (val - 1.0).abs() < 1e-10,
            "Exponential MGF at t=0 should be 1, got {val}"
        );
    }

    #[test]
    fn test_bernoulli_mean_and_variance() {
        let m = symbolic_moments_catalog(DistFamily::Bernoulli).expect("catalog failed");
        // At p=0.3: mean=0.3, variance=0.3*(1-0.3)=0.21
        let mean_val = eval(&m.mean, &[0.3]);
        let var_val = eval(&m.variance, &[0.3]);
        assert!(
            (mean_val - 0.3).abs() < 1e-10,
            "Bernoulli mean at p=0.3 should be 0.3, got {mean_val}"
        );
        assert!(
            (var_val - 0.21).abs() < 1e-10,
            "Bernoulli variance at p=0.3 should be 0.21, got {var_val}"
        );
    }

    #[test]
    fn test_geometric_mean_at_p_half() {
        let m = symbolic_moments_catalog(DistFamily::Geometric).expect("catalog failed");
        // Mean = (1-p)/p; at p=0.5, mean = 0.5/0.5 = 1.0
        let val = eval(&m.mean, &[0.5]);
        assert!(
            (val - 1.0).abs() < 1e-10,
            "Geometric mean at p=0.5 should be 1.0, got {val}"
        );
    }
}
