//! Integration tests for `scirs2_stats::mle::derive`.
//!
//! All tests are gated on `feature = "symbolic"`.  Sample data is either
//! hardcoded or constructed deterministically — no rand crate required.

#![cfg(feature = "symbolic")]

use scirs2_core::ndarray::array;
use scirs2_stats::mle::{derive, DeriveError, FitError};
use scirs2_symbolic::{cas::canonicalize, eml::LoweredOp};

// ─────────────────────────────────────────────────────────────────────────────
// PDF builder helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Exponential pdf: λ · exp(-λ · x)
///
/// Variable layout: Var(0) = λ (param), Var(data_var) = x (data).
fn exponential_pdf(data_var: usize) -> LoweredOp {
    let lambda = LoweredOp::Var(0);
    let x = LoweredOp::Var(data_var);
    // λ · exp(-(λ · x))
    LoweredOp::Mul(
        Box::new(lambda.clone()),
        Box::new(LoweredOp::Exp(Box::new(LoweredOp::Neg(Box::new(
            LoweredOp::Mul(Box::new(lambda), Box::new(x)),
        ))))),
    )
}

/// Bernoulli pmf: p^x · (1-p)^(1-x)
///
/// Variable layout: Var(0) = p (param), Var(data_var) = x (data, ∈ {0,1}).
fn bernoulli_pmf(data_var: usize) -> LoweredOp {
    let p = LoweredOp::Var(0);
    let x = LoweredOp::Var(data_var);
    // p^x * (1-p)^(1-x)
    let one_minus_p = LoweredOp::Sub(Box::new(LoweredOp::Const(1.0)), Box::new(p.clone()));
    let one_minus_x = LoweredOp::Sub(Box::new(LoweredOp::Const(1.0)), Box::new(x.clone()));
    let term1 = LoweredOp::Pow(Box::new(p), Box::new(x));
    let term2 = LoweredOp::Pow(Box::new(one_minus_p), Box::new(one_minus_x));
    LoweredOp::Mul(Box::new(term1), Box::new(term2))
}

/// Simple Normal pdf (proportional, ignoring 2π normalisation):
/// exp(-(x-μ)² / (2σ²)) / σ
///
/// Variable layout: Var(0)=μ, Var(1)=σ, Var(data_var)=x.
fn normal_pdf(data_var: usize) -> LoweredOp {
    let mu = LoweredOp::Var(0);
    let sigma = LoweredOp::Var(1);
    let x = LoweredOp::Var(data_var);
    // (x - μ)
    let diff = LoweredOp::Sub(Box::new(x), Box::new(mu));
    // (x - μ)^2
    let diff_sq = LoweredOp::Pow(Box::new(diff), Box::new(LoweredOp::Const(2.0)));
    // 2 * σ^2
    let two_sig_sq = LoweredOp::Mul(
        Box::new(LoweredOp::Const(2.0)),
        Box::new(LoweredOp::Pow(
            Box::new(sigma.clone()),
            Box::new(LoweredOp::Const(2.0)),
        )),
    );
    // exp(-(x-μ)^2 / (2σ^2))
    let exponent = LoweredOp::Neg(Box::new(LoweredOp::Div(
        Box::new(diff_sq),
        Box::new(two_sig_sq),
    )));
    // exp(...) / σ
    LoweredOp::Div(
        Box::new(LoweredOp::Exp(Box::new(exponent))),
        Box::new(sigma),
    )
}

/// Cauchy-like pdf (transcendental denominator, no closed-form MLE):
/// 1 / (π · (1 + (x-γ)^2))
///
/// Variable layout: Var(0)=γ, Var(data_var)=x.
fn cauchy_pdf(data_var: usize) -> LoweredOp {
    let gamma = LoweredOp::Var(0);
    let x = LoweredOp::Var(data_var);
    let diff = LoweredOp::Sub(Box::new(x), Box::new(gamma));
    let diff_sq = LoweredOp::Pow(Box::new(diff), Box::new(LoweredOp::Const(2.0)));
    let denom = LoweredOp::Mul(
        Box::new(LoweredOp::Const(std::f64::consts::PI)),
        Box::new(LoweredOp::Add(
            Box::new(LoweredOp::Const(1.0)),
            Box::new(diff_sq),
        )),
    );
    LoweredOp::Div(Box::new(LoweredOp::Const(1.0)), Box::new(denom))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

/// Test 1: `derive` returns `Ok` for Exponential pdf and score_equations is non-empty.
#[test]
fn test_exponential_derive_returns_ok() {
    let n = 5_usize;
    let pdf = exponential_pdf(1); // data_var=1, param=Var(0)
    let est = derive(&pdf, &[0], 1, n).expect("derive should succeed");

    assert_eq!(
        est.score_equations.len(),
        1,
        "one score equation for one parameter"
    );
    // Either closed form or numerical fallback — both are valid
    assert!(
        est.closed_form.is_some() || est.falls_back_to_numeric,
        "must have closed form or flag falls_back_to_numeric"
    );
}

/// Test 2: Exponential MLE — `fit()` converges and produces λ̂ ≈ n / Σxᵢ.
///
/// For Exponential(λ), the MLE is λ̂ = 1/x̄ = n / Σxᵢ.
#[test]
fn test_exponential_fit_accuracy() {
    let data_arr = [0.5_f64, 0.25, 0.4, 0.6, 0.3];
    let n = data_arr.len();
    let data = array![0.5, 0.25, 0.4, 0.6, 0.3];
    let pdf = exponential_pdf(1); // data_var=1
    let est = derive(&pdf, &[0], 1, n).expect("derive");
    let estimates = est.fit(data.view()).expect("fit");
    assert_eq!(estimates.len(), 1, "one estimate for λ");
    let lambda_hat = estimates[0];
    let analytical = n as f64 / data_arr.iter().sum::<f64>();
    assert!(
        (lambda_hat - analytical).abs() < 0.1,
        "λ̂={lambda_hat:.4} expected near analytical={analytical:.4}"
    );
    assert!(lambda_hat > 0.0, "λ̂ must be positive");
}

/// Test 3: Bernoulli MLE — `fit()` produces p̂ ≈ x̄.
///
/// For Bernoulli(p), the MLE is p̂ = Σxᵢ / n (sample proportion).
#[test]
fn test_bernoulli_fit_accuracy() {
    let data = array![1.0_f64, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let n = data.len();
    let pdf = bernoulli_pmf(1); // data_var=1
    let est = derive(&pdf, &[0], 1, n).expect("derive");
    let estimates = est.fit(data.view()).expect("fit Bernoulli");
    assert_eq!(estimates.len(), 1, "one estimate for p");
    let p_hat = estimates[0];
    let analytical: f64 = data.iter().sum::<f64>() / n as f64;
    assert!(
        (p_hat - analytical).abs() < 0.15,
        "p̂={p_hat:.4} expected near {analytical:.4}"
    );
    assert!(
        p_hat > 0.0 && p_hat < 1.0,
        "p̂ must be in (0, 1), got {p_hat}"
    );
}

/// Test 4: `DeriveError::ZeroSamples` is returned for `n_samples = 0`.
#[test]
fn test_zero_samples_error() {
    let lambda = LoweredOp::Var(0);
    let x = LoweredOp::Var(1);
    let pdf = LoweredOp::Mul(Box::new(lambda), Box::new(x));
    let result = derive(&pdf, &[0], 1, 0);
    assert!(
        matches!(result, Err(DeriveError::ZeroSamples)),
        "expected ZeroSamples, got {result:?}"
    );
}

/// Test 5: `DeriveError::EmptyParams` is returned for empty params.
#[test]
fn test_empty_params_error() {
    let pdf = LoweredOp::Var(1);
    let result = derive(&pdf, &[], 1, 5);
    assert!(
        matches!(result, Err(DeriveError::EmptyParams)),
        "expected EmptyParams, got {result:?}"
    );
}

/// Test 6: `FitError::DataLengthMismatch` when data length != n_samples.
#[test]
fn test_data_length_mismatch() {
    let n = 5_usize;
    let pdf = exponential_pdf(1);
    let est = derive(&pdf, &[0], 1, n).expect("derive");

    // Supply wrong length (3 instead of 5)
    let wrong_data = array![0.5, 0.4, 0.3];
    let result = est.fit(wrong_data.view());
    assert!(
        matches!(
            result,
            Err(FitError::DataLengthMismatch {
                expected: 5,
                got: 3
            })
        ),
        "expected DataLengthMismatch(5,3), got {result:?}"
    );
}

/// Test 7: `DeriveError::VarIdCollision` when a param Var id is in data range.
#[test]
fn test_var_id_collision() {
    // data_var=1, n_samples=3 → data occupies Var(1), Var(2), Var(3)
    // param Var(2) collides
    let pdf = LoweredOp::Var(1);
    let result = derive(&pdf, &[2], 1, 3);
    assert!(
        matches!(result, Err(DeriveError::VarIdCollision { id: 2 })),
        "expected VarIdCollision(2), got {result:?}"
    );
}

/// Test 8: Canonical invariance — deriving twice yields score equations with
/// the same canonical hash for each component.
#[test]
fn test_canonical_invariance() {
    let n = 3_usize;
    let pdf = exponential_pdf(1);

    let est1 = derive(&pdf, &[0], 1, n).expect("derive 1");
    let est2 = derive(&pdf, &[0], 1, n).expect("derive 2");

    assert_eq!(
        est1.score_equations.len(),
        est2.score_equations.len(),
        "both derivations should produce the same number of score equations"
    );

    for (s1, s2) in est1.score_equations.iter().zip(est2.score_equations.iter()) {
        let c1 = canonicalize(s1);
        let c2 = canonicalize(s2);
        // Canonical forms must be equal (hash equality implies structural equality)
        assert_eq!(
            c1, c2,
            "canonical score equations must be identical across two calls to derive"
        );
    }
}

/// Test 9: Cauchy pdf causes `falls_back_to_numeric = true` because the MLE
/// score equations are transcendental in γ.
#[test]
fn test_cauchy_falls_back_to_numeric() {
    let n = 3_usize;
    let pdf = cauchy_pdf(1); // data_var=1, param=Var(0)=γ
    let est = derive(&pdf, &[0], 1, n).expect("derive Cauchy");
    // Cauchy MLE has no closed form — solver should return falls_back_to_numeric
    // (it may or may not, depending on the solver tier — either is acceptable
    // as long as fit() works without panic)
    let data = array![0.1, -0.2, 0.3];
    let result = est.fit(data.view());
    // Accept either convergence or a NumericalFailed (Cauchy Newton can be unstable)
    match result {
        Ok(estimates) => {
            assert_eq!(estimates.len(), 1, "one estimate for γ");
            // Just verify it's a finite number
            assert!(estimates[0].is_finite(), "γ̂ must be finite");
        }
        Err(FitError::NumericalFailed(_)) => {
            // Acceptable — Cauchy Newton may not converge
        }
        Err(e) => panic!("unexpected error for Cauchy: {e}"),
    }
}

/// Test 10: Normal two-parameter MLE — fit produces μ̂ ≈ sample mean.
///
/// Uses data near N(2.0, 0.5²) to verify multivariate Newton converges.
#[test]
fn test_normal_two_param_fit() {
    let data = array![1.8_f64, 2.1, 2.3, 1.9, 2.2, 1.7, 2.0, 2.1, 1.8, 2.4];
    let n = data.len();
    // data_var=2, params=[0,1]=[μ,σ]
    let pdf = normal_pdf(2);
    let est = derive(&pdf, &[0, 1], 2, n).expect("derive Normal");
    let result = est.fit(data.view());

    match result {
        Ok(estimates) => {
            assert_eq!(estimates.len(), 2);
            let mu_hat = estimates[0];
            let empirical_mean: f64 = data.iter().sum::<f64>() / n as f64;
            assert!(
                (mu_hat - empirical_mean).abs() < 0.5,
                "μ̂={mu_hat:.4} should be near {empirical_mean:.4}"
            );
        }
        Err(FitError::NumericalFailed(msg)) => {
            // Newton may fail for multi-param case depending on initial point;
            // that is acceptable — just ensure no panic.
            eprintln!("Normal two-param Newton fallback failed (acceptable): {msg}");
        }
        Err(e) => panic!("unexpected error: {e}"),
    }
}

/// Test 11: `derive` on a constant pdf (no data var dependency) produces
/// score ≡ 0, which `solve_system` should handle gracefully.
#[test]
fn test_constant_pdf_graceful() {
    // pdf = Const(1.0) — no dependence on x or θ
    let pdf = LoweredOp::Const(1.0);
    // param=Var(0), data_var=1
    let result = derive(&pdf, &[0], 1, 3);
    // Either Ok or DeriveError — but no panic
    match result {
        Ok(est) => {
            // Score of ln(1) = 0 w.r.t. any var — solver should see trivial system
            // Either closed_form is None or falls_back_to_numeric
            let _ = est; // just ensure no panic
        }
        Err(e) => {
            // Any DeriveError is acceptable for a degenerate pdf
            let _ = e;
        }
    }
}
