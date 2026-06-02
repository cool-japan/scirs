//! Integration tests for `scirs2_symbolic::neural_priors`.
//!
//! Tests verify:
//! 1. Discovery on a sinusoidal series finds a high-R² formula.
//! 2. Discovery on a linear series finds a high-R² formula.
//! 3. Evaluation of a hand-crafted AR(1) prior returns the correct value.
//! 4. `series_prior_regularization` is zero at perfect match and `lambda` at unit deviation.
//! 5. Too-short series returns a `SymbolicError::DomainError`.

use scirs2_symbolic::neural_priors::{
    discover_series_prior, eval_series_prior, series_prior_regularization, SeriesPrior,
};
use scirs2_symbolic::{LoweredOp, SymbolicError};

// ─────────────────────────────────────────────────────────────────────────────
// Test 1 — sinusoidal series: top formula achieves R² > 0.9
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn discover_sin_series_prior() {
    // y[t] = sin(t * 0.1) — a smooth, slowly-varying sinusoid.
    // With lookback=5, the identity Var(4) (= y[t-1]) should achieve R² ≈ 0.995
    // because consecutive samples are nearly identical at this frequency.
    let series: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1_f64).sin()).collect();

    let config = scirs2_symbolic::regression::SrConfig::default()
        .with_max_iter(20)
        .with_top_n(5)
        .with_beam_width(20);

    let prior =
        discover_series_prior(&series, 5, 1, 5, Some(config)).expect("discovery should succeed");

    assert!(
        !prior.formulas.is_empty(),
        "should return at least one formula"
    );
    assert_eq!(prior.lookback, 5);
    assert_eq!(prior.variable_names.len(), 5);

    // Best formula (sorted by R² descending) must have R² > 0.9.
    let (_, best_r2) = &prior.formulas[0];
    assert!(
        *best_r2 > 0.9,
        "expected R² > 0.9 for sinusoidal series, got {best_r2}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2 — linear series: top formula has high R²
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn discover_linear_series_prior() {
    // y[t] = 0.5 * t + small_noise
    // With lookback=3, Var(2)=y[t-1] is a near-perfect linear predictor
    // (consecutive values differ by 0.5, so R² ≈ 0.99+ for n=100).
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    let series: Vec<f64> = (0..100_usize)
        .map(|i| {
            i.hash(&mut hasher);
            let noise = (hasher.finish() as f64 / u64::MAX as f64 - 0.5) * 0.02;
            0.5 * i as f64 + noise
        })
        .collect();

    let config = scirs2_symbolic::regression::SrConfig::default()
        .with_max_iter(20)
        .with_top_n(3)
        .with_beam_width(16);

    let prior =
        discover_series_prior(&series, 3, 1, 3, Some(config)).expect("discovery should succeed");

    assert!(!prior.formulas.is_empty());
    let (_, best_r2) = &prior.formulas[0];
    // The identity formula Var(2) = y[t-1] is linear in the feature; R² ≈ 1.0.
    assert!(
        *best_r2 > 0.9,
        "expected R² > 0.9 for near-linear series, got {best_r2}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3 — evaluate hand-crafted AR(1) prior
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn eval_series_prior_at_window() {
    // Construct the prior directly: f(window) = window[lookback-1] = Var(2)
    // for lookback=3.  This is the AR(1) formula "predict y[t] as y[t-1]".
    let prior = SeriesPrior {
        formulas: vec![(LoweredOp::Var(2), 1.0)],
        variable_names: vec!["y[t-3]".into(), "y[t-2]".into(), "y[t-1]".into()],
        lookback: 3,
    };

    let window = [10.0_f64, 20.0, 42.0]; // Var(2) = 42.0
    let preds = eval_series_prior(&prior, &window).expect("eval should succeed");

    assert_eq!(preds.len(), 1);
    assert!(
        (preds[0] - 42.0).abs() < 1e-12,
        "AR(1) should predict last window value, got {}",
        preds[0]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4 — regularization: zero when matching, lambda when deviating by 1
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn series_prior_regularization_agrees_with_prior() {
    // Exact match → regularization = 0.
    let reg_zero = series_prior_regularization(3.5, &[3.5, 10.0], 2.0);
    assert!(
        reg_zero < 1e-20,
        "expected zero regularization on exact match, got {reg_zero}"
    );

    // Deviation of exactly 1.0 from the only prior, lambda=1.0 → penalty = 1.0.
    let reg_one = series_prior_regularization(3.0, &[2.0], 1.0);
    assert!(
        (reg_one - 1.0).abs() < 1e-14,
        "expected regularization = 1.0 for unit deviation, got {reg_one}"
    );

    // lambda=0.0 → always zero regardless of deviation.
    let reg_lambda0 = series_prior_regularization(99.0, &[0.0, 1.0], 0.0);
    assert_eq!(reg_lambda0, 0.0, "lambda=0 should give zero penalty");

    // Min-over-formulas: neural_pred=5.0, prior_preds=[5.0, 100.0] → min sq = 0.
    let reg_min = series_prior_regularization(5.0, &[5.0, 100.0], 3.0);
    assert!(
        reg_min < 1e-20,
        "min-over-formulas should give zero when one formula matches exactly, got {reg_min}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5 — short series returns error, not panic
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn discover_prior_small_series() {
    // lookback=5, target_lag=1 requires at least 5+1+1=7 samples; give only 6.
    let short: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = discover_series_prior(&short, 5, 1, 3, None);

    assert!(
        result.is_err(),
        "should return error for series shorter than 2*lookback+1"
    );
    match result {
        Err(SymbolicError::DomainError(msg)) => {
            assert!(
                msg.contains("too short"),
                "error message should mention 'too short', got: {msg}"
            );
        }
        Err(e) => panic!("expected DomainError, got {e:?}"),
        Ok(_) => panic!("expected error for short series"),
    }
}
