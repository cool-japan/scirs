//! Integration tests for `scirs2_stats::mle_symbolic::fit_mle_symbolic`.
//!
//! All tests are gated on `feature = "symbolic"` and require no random crate —
//! sample data is either hardcoded or constructed deterministically.

#![cfg(feature = "symbolic")]

use scirs2_core::ndarray::{array, Array1};
use scirs2_stats::{fit_mle_symbolic, MleSymbolicError};
use scirs2_symbolic::eml::LoweredOp;
use std::sync::Arc;

// ─── NLL builder helpers ─────────────────────────────────────────────────────

/// Build the Gaussian NLL (summed over data):
///
/// `NLL(μ, σ) = Σ_i (x_i - μ)² / (2σ²) + n · ln(σ)`
///
/// Variable layout:
/// - `Var(0..n_data)` = observations x_i
/// - `Var(n_data)`   = μ
/// - `Var(n_data+1)` = σ  (must be > 0)
fn build_gaussian_nll(n_data: usize) -> Arc<LoweredOp> {
    let mu = LoweredOp::Var(n_data);
    let sigma = LoweredOp::Var(n_data + 1);

    let mut sum = LoweredOp::Const(0.0);

    for i in 0..n_data {
        let xi = LoweredOp::Var(i);

        // (x_i - μ)
        let diff = LoweredOp::Sub(Box::new(xi), Box::new(mu.clone()));

        // (x_i - μ)²
        let diff_sq = LoweredOp::Pow(Box::new(diff), Box::new(LoweredOp::Const(2.0)));

        // σ²
        let sigma_sq = LoweredOp::Pow(Box::new(sigma.clone()), Box::new(LoweredOp::Const(2.0)));

        // 2 · σ²
        let two_sigma_sq = LoweredOp::Mul(Box::new(LoweredOp::Const(2.0)), Box::new(sigma_sq));

        // (x_i - μ)² / (2σ²)
        let term = LoweredOp::Div(Box::new(diff_sq), Box::new(two_sigma_sq));

        sum = LoweredOp::Add(Box::new(sum), Box::new(term));
    }

    // n · ln(σ)
    let n_ln_sigma = LoweredOp::Mul(
        Box::new(LoweredOp::Const(n_data as f64)),
        Box::new(LoweredOp::Ln(Box::new(sigma.clone()))),
    );

    Arc::new(LoweredOp::Add(Box::new(sum), Box::new(n_ln_sigma)))
}

/// Build the Bernoulli NLL (summed over data):
///
/// `NLL(p) = -Σ_i [ x_i · ln(p) + (1 - x_i) · ln(1 - p) ]`
///
/// Variable layout:
/// - `Var(0..n_data)` = binary observations x_i ∈ {0, 1}
/// - `Var(n_data)`    = p  (must be in (0, 1))
fn build_bernoulli_nll(n_data: usize) -> Arc<LoweredOp> {
    let p = LoweredOp::Var(n_data);

    // ln(p)
    let ln_p = LoweredOp::Ln(Box::new(p.clone()));

    // ln(1 - p)
    let one_minus_p = LoweredOp::Sub(Box::new(LoweredOp::Const(1.0)), Box::new(p.clone()));
    let ln_1mp = LoweredOp::Ln(Box::new(one_minus_p));

    let mut sum = LoweredOp::Const(0.0);

    for i in 0..n_data {
        let xi = LoweredOp::Var(i);

        // x_i · ln(p)
        let pos_term = LoweredOp::Mul(Box::new(xi.clone()), Box::new(ln_p.clone()));

        // (1 - x_i) · ln(1 - p)
        let one_minus_xi = LoweredOp::Sub(Box::new(LoweredOp::Const(1.0)), Box::new(xi.clone()));
        let neg_term = LoweredOp::Mul(Box::new(one_minus_xi), Box::new(ln_1mp.clone()));

        // x_i · ln(p) + (1 - x_i) · ln(1 - p)
        let log_lik_i = LoweredOp::Add(Box::new(pos_term), Box::new(neg_term));

        // NLL accumulates the negative
        let neg_log_lik_i = LoweredOp::Neg(Box::new(log_lik_i));

        sum = LoweredOp::Add(Box::new(sum), Box::new(neg_log_lik_i));
    }

    Arc::new(sum)
}

// ─── Fixed sample data ───────────────────────────────────────────────────────

/// 20 observations drawn near N(3.0, 0.2) (mean ≈ 3.025, std ≈ 0.19).
fn gaussian_samples() -> Array1<f64> {
    array![
        2.8, 3.1, 3.3, 2.9, 3.2, 2.7, 3.0, 3.1, 2.8, 3.4, 3.1, 2.9, 3.0, 3.2, 2.6, 3.3, 3.1, 2.9,
        3.0, 3.2
    ]
}

/// 50 binary observations with empirical success rate ≈ 0.40.
fn bernoulli_samples() -> Array1<f64> {
    array![
        1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0
    ]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// Test 1: MLE recovers mean μ from Gaussian samples.
///
/// Initial guess (μ=3.0, σ=1.0) is near the MLE to test gradient convergence.
/// Gradient descent with symbolic gradient converges to the sample mean and
/// MLE std dev.
#[test]
fn test_gaussian_mle_recovers_mu() {
    let data = gaussian_samples();
    let n = data.len();
    let nll = build_gaussian_nll(n);

    // Initial guess: μ = 3.0, σ = 1.0 (within the basin of attraction)
    let init = array![3.0_f64, 1.0_f64];
    let result = fit_mle_symbolic(&nll, data.view(), init.view(), 1000, 1e-6, 0.1)
        .expect("MLE should converge for Gaussian data");

    let mu_hat = result.params[0];
    let empirical_mean: f64 = data.iter().sum::<f64>() / n as f64;
    // MLE for Gaussian is sample mean — check we're close
    assert!(
        (mu_hat - empirical_mean).abs() < 0.3,
        "mu_hat={} expected near empirical_mean={}",
        mu_hat,
        empirical_mean
    );
}

/// Test 2: MLE recovers sigma σ from Gaussian samples.
///
/// Initial guess (μ=3.0, σ=1.0) is near the MLE. MLE sigma = biased std dev ≈ 0.19.
#[test]
fn test_gaussian_mle_recovers_sigma() {
    let data = gaussian_samples();
    let n = data.len();
    let nll = build_gaussian_nll(n);

    let init = array![3.0_f64, 1.0_f64];
    let result = fit_mle_symbolic(&nll, data.view(), init.view(), 1000, 1e-6, 0.1)
        .expect("MLE should converge for Gaussian data");

    let sigma_hat = result.params[1];
    assert!(
        sigma_hat > 0.0,
        "sigma_hat must be positive, got {}",
        sigma_hat
    );
    // MLE sigma should be in a reasonable range for our data (empirical std ≈ 0.19)
    assert!(
        (sigma_hat - 0.2_f64).abs() < 0.3,
        "sigma_hat={} expected near 0.2",
        sigma_hat
    );
}

/// Test 3: Gaussian MLE converges within max_iter.
#[test]
fn test_gaussian_mle_converges() {
    let data = gaussian_samples();
    let n = data.len();
    let nll = build_gaussian_nll(n);

    let max_iter = 1000_usize;
    let init = array![3.0_f64, 1.0_f64];
    let result = fit_mle_symbolic(&nll, data.view(), init.view(), max_iter, 1e-6, 0.1)
        .expect("MLE should not return error");

    assert!(
        result.converged,
        "expected converged=true, got iters={}",
        result.iters
    );
    assert!(result.iters > 0, "iters must be positive");
    assert!(
        result.iters <= max_iter,
        "iters={} exceeds max_iter={}",
        result.iters,
        max_iter
    );
}

/// Test 4: max_iter=0 returns init_params unchanged with converged=false.
#[test]
fn test_max_iter_zero_returns_init() {
    let data = array![1.0_f64, 2.0, 3.0];
    let n = data.len();
    let nll = build_gaussian_nll(n);

    let init = array![7.5_f64, 2.3_f64];
    let result = fit_mle_symbolic(&nll, data.view(), init.view(), 0, 1e-6, 0.1)
        .expect("max_iter=0 should return Ok");

    assert!(!result.converged, "converged must be false when max_iter=0");
    assert_eq!(result.iters, 0, "iters must be 0");
    assert!(
        (result.params[0] - 7.5).abs() < 1e-12,
        "params[0] must equal init"
    );
    assert!(
        (result.params[1] - 2.3).abs() < 1e-12,
        "params[1] must equal init"
    );
    assert!(
        result.nll_final.is_nan(),
        "nll_final must be NaN when max_iter=0"
    );
}

/// Test 5: NLL evaluated at sigma=0 returns Err (domain violation from ln(0)), not Ok(NaN).
#[test]
fn test_nll_eval_nan_propagates_err() {
    let data = array![1.0_f64, 2.0, 3.0];
    let n = data.len();
    let nll = build_gaussian_nll(n);

    // sigma=0 forces ln(0) → EvalDomain error inside fit_mle_symbolic
    let init = array![2.0_f64, 0.0_f64];
    let result = fit_mle_symbolic(&nll, data.view(), init.view(), 10, 1e-6, 0.1);

    assert!(
        result.is_err(),
        "expected Err when sigma=0 causes ln(0), got Ok"
    );
    match result {
        Err(MleSymbolicError::EvalError(_)) => { /* expected */ }
        other => panic!("expected EvalError, got {:?}", other),
    }
}

/// Test 6: Bernoulli MLE recovers p ≈ 0.40.
#[test]
fn test_bernoulli_mle_recovers_p() {
    let data = bernoulli_samples();
    let n = data.len();
    let nll = build_bernoulli_nll(n);

    // Initial guess p = 0.5 (interior of (0,1))
    let init = array![0.5_f64];
    let result = fit_mle_symbolic(&nll, data.view(), init.view(), 1000, 1e-6, 0.5)
        .expect("Bernoulli MLE should converge");

    let p_hat = result.params[0];
    // Empirical success rate in our data
    let empirical_p: f64 = data.iter().sum::<f64>() / n as f64;
    assert!(
        (p_hat - empirical_p).abs() < 0.12,
        "p_hat={} expected within 0.12 of empirical_p={}",
        p_hat,
        empirical_p
    );
    assert!(
        p_hat > 0.0 && p_hat < 1.0,
        "p_hat={} must be in (0,1)",
        p_hat
    );
}

/// Test 7: Trivial one-parameter quadratic NLL = (θ - 5)² converges to θ ≈ 5.
///
/// No data observations; the single variable `Var(0)` is the parameter θ.
#[test]
fn test_trivial_one_param_quadratic() {
    // NLL = (Var(0) - 5)²
    let nll = Arc::new(LoweredOp::Pow(
        Box::new(LoweredOp::Sub(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(5.0)),
        )),
        Box::new(LoweredOp::Const(2.0)),
    ));

    // Empty data slice (n_data = 0), one parameter at Var(0)
    let data = Array1::<f64>::zeros(0);
    let init = array![0.0_f64];
    let result = fit_mle_symbolic(&nll, data.view(), init.view(), 500, 1e-4, 0.5)
        .expect("quadratic MLE should converge");

    assert!(result.converged, "must converge; iters={}", result.iters);
    assert!(
        (result.params[0] - 5.0).abs() < 1e-3,
        "theta_hat={} expected near 5.0",
        result.params[0]
    );
}

/// Test 8: Running with very few iterations on a hard NLL returns NotConverged or converged=false.
///
/// Uses NLL = (θ - 100)² with max_iter=3. Either the backtracking line search
/// fails (returning Err(NotConverged)) or we exhaust iterations (converged=false).
/// What must NOT happen is a panic.
#[test]
fn test_not_converged_returns_err_not_panic() {
    // NLL = (Var(0) - 100)²  starting from θ=0  (far from minimum)
    let nll = Arc::new(LoweredOp::Pow(
        Box::new(LoweredOp::Sub(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(100.0)),
        )),
        Box::new(LoweredOp::Const(2.0)),
    ));

    let data = Array1::<f64>::zeros(0);
    let init = array![0.0_f64];
    // Very small learning_rate and few iters so we don't reach the minimum
    let outcome = fit_mle_symbolic(&nll, data.view(), init.view(), 3, 1e-12, 1e-8);

    // Accept either: Err(NotConverged), or Ok with converged=false
    match outcome {
        Err(MleSymbolicError::NotConverged) => { /* line search failed — acceptable */ }
        Ok(r) if !r.converged => { /* ran out of iterations — acceptable */ }
        Ok(r) if r.converged => {
            // If it somehow converged to the minimum in 3 steps, still valid
            // (gradient must be < 1e-12 at theta ≈ 100)
        }
        Err(e) => panic!("unexpected error variant: {}", e),
        Ok(_) => {}
    }
    // The critical property: no panic reached this point
}
