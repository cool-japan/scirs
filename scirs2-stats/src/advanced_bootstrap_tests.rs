// Tests for `advanced_bootstrap.rs`, split into a separate file (via
// `#[path = ...]`) to keep the main implementation file under the
// workspace's 2000-line guideline; this file's top-level content *is* the
// `tests` module body (same pattern used by `regularized_tests.rs` /
// `robust_tests.rs` / `enhanced_sequences_tests.rs`).

use super::*;
use scirs2_core::ndarray::array;

#[test]
fn test_basicbootstrap() {
    let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let mean_fn = |x: &ArrayView1<f64>| -> StatsResult<f64> { Ok(x.sum() / x.len() as f64) };

    let config = AdvancedBootstrapConfig {
        n_bootstrap: 100,
        seed: Some(42),
        ..Default::default()
    };

    let mut processor = AdvancedBootstrapProcessor::new(config);
    let result = processor
        .bootstrap(&data.view(), mean_fn)
        .expect("Operation failed");

    assert_eq!(result.n_successful, 100);
    assert!(result.bootstrap_samples.len() == 100);
    assert!((result.original_statistic - 3.0).abs() < 1e-10);
}

#[test]
fn test_stratifiedbootstrap() {
    let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let strata = vec![0, 0, 1, 1, 2, 2]; // Three strata
    let mean_fn = |x: &ArrayView1<f64>| -> StatsResult<f64> { Ok(x.sum() / x.len() as f64) };

    let result = stratified_bootstrap(
        &data.view(),
        &strata,
        mean_fn,
        Some(AdvancedBootstrapConfig {
            n_bootstrap: 50,
            seed: Some(123),
            ..Default::default()
        }),
    )
    .expect("Operation failed");

    assert_eq!(result.n_successful, 50);
    assert!(matches!(result.method, BootstrapType::Stratified { .. }));
}

#[test]
fn test_moving_blockbootstrap() {
    let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mean_fn = |x: &ArrayView1<f64>| -> StatsResult<f64> { Ok(x.sum() / x.len() as f64) };

    let result = moving_block_bootstrap(
        &data.view(),
        mean_fn,
        Some(3),  // block length
        Some(50), // n_bootstrap
    )
    .expect("Operation failed");

    assert_eq!(result.n_successful, 50);
    assert!(result.effective_samplesize.is_some());
}

#[test]
fn test_circular_blockbootstrap() {
    let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mean_fn = |x: &ArrayView1<f64>| -> StatsResult<f64> { Ok(x.sum() / x.len() as f64) };

    let result = circular_block_bootstrap(&data.view(), mean_fn, Some(2), Some(30))
        .expect("Operation failed");

    assert_eq!(result.n_successful, 30);
}

#[test]
fn test_confidence_intervals() {
    let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let mean_fn = |x: &ArrayView1<f64>| -> StatsResult<f64> { Ok(x.sum() / x.len() as f64) };

    let config = AdvancedBootstrapConfig {
        n_bootstrap: 200,
        confidence_level: 0.95,
        seed: Some(42),
        ..Default::default()
    };

    let mut processor = AdvancedBootstrapProcessor::new(config);
    let result = processor
        .bootstrap(&data.view(), mean_fn)
        .expect("Operation failed");

    let ci = &result.confidence_intervals;
    assert!(ci.percentile.0 <= ci.percentile.1);
    assert!(ci.basic.0 <= ci.basic.1);
}

#[test]
fn test_bootstrap_diagnostics() {
    let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let var_fn = |x: &ArrayView1<f64>| -> StatsResult<f64> {
        let mean = x.sum() / x.len() as f64;
        let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (x.len() - 1) as f64;
        Ok(var)
    };

    let config = AdvancedBootstrapConfig {
        n_bootstrap: 100,
        seed: Some(456),
        ..Default::default()
    };

    let mut processor = AdvancedBootstrapProcessor::new(config);
    let result = processor
        .bootstrap(&data.view(), var_fn)
        .expect("Operation failed");

    assert!(result.diagnostics.convergence_info.converged);
    assert!(
        result.diagnostics.distribution_stats.min_value
            <= result.diagnostics.distribution_stats.max_value
    );
}

// ========================================================================
// `generate_beta_sample` / `generate_gamma_sample` fix tests.
//
// Wave-1 finding: `generate_beta_sample` combined two independent
// `Beta(a, 1)`-style power transforms of uniforms
// (`x = U1^(1/alpha)`, `y = U2^(1/beta)`, `value = x/(x+y)`) rather than
// the ratio of true `Gamma(alpha, 1)`/`Gamma(beta, 1)` variates the old
// comment claimed. Fixed by drawing real Gamma variates via
// `sample_standard_gamma` (Marsaglia & Tsang's method) and forming
// `X / (X + Y)`, which is an exact `Beta(alpha, beta)` sampler.
//
// Reference moments (`mean = alpha/(alpha+beta)`,
// `var = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))`) are the
// textbook Beta-distribution moments, NOT derived from this crate. The
// old formula's mean under these same (asymmetric) parameters was
// independently verified via a 1,000,000-draw Python/`random` Monte
// Carlo simulation of the exact old expression, NOT derived from this
// crate either.
// ========================================================================

#[test]
fn test_generate_beta_sample_matches_theoretical_mean_not_old_formula() {
    let config = AdvancedBootstrapConfig {
        seed: Some(2026_07_29),
        ..Default::default()
    };
    let mut processor: AdvancedBootstrapProcessor<f64> = AdvancedBootstrapProcessor::new(config);

    // Asymmetric shape parameters: NON-CONSTANT data with a strongly
    // skewed target distribution, so the old ratio-of-powered-uniforms
    // formula and the true Beta(alpha, beta) distribution disagree
    // sharply on the mean (0.2 vs. an old-formula Monte Carlo mean of
    // ~0.415 for alpha=2, beta=8; verified independently in Python).
    let n = 100_000;
    let alpha = 2.0;
    let beta = 8.0;
    let sample = processor
        .generate_beta_sample(n, alpha, beta)
        .expect("beta sample generation should succeed");
    assert_eq!(sample.len(), n);

    let mean: f64 = sample.iter().sum::<f64>() / n as f64;
    let true_mean = alpha / (alpha + beta);
    assert!(
        (mean - true_mean).abs() < 0.01,
        "expected sample mean ~= {true_mean} (true Beta({alpha},{beta}) mean), got {mean}"
    );
    // This is the assertion that would have FAILED under the old
    // `x/(x+y)` power-transform formula: its Monte Carlo mean for these
    // parameters is ~0.415, far outside a 0.01 tolerance of the true
    // 0.2 mean.
    assert!(
        (mean - 0.415).abs() > 0.05,
        "sample mean {mean} looks suspiciously close to the old buggy formula's ~0.415"
    );

    // Second, independent asymmetric case (alpha=5, beta=1): true mean
    // 0.8333 vs. the old formula's Monte Carlo mean of ~0.653.
    let sample2 = processor
        .generate_beta_sample(n, 5.0, 1.0)
        .expect("beta sample generation should succeed");
    let mean2: f64 = sample2.iter().sum::<f64>() / n as f64;
    assert!(
        (mean2 - 5.0 / 6.0).abs() < 0.01,
        "expected sample mean ~= {} (true Beta(5,1) mean), got {mean2}",
        5.0 / 6.0
    );
    assert!(
        (mean2 - 0.653).abs() > 0.05,
        "sample mean {mean2} looks suspiciously close to the old buggy formula's ~0.653"
    );
}

#[test]
fn test_generate_beta_sample_values_in_unit_interval() {
    let config = AdvancedBootstrapConfig {
        seed: Some(7),
        ..Default::default()
    };
    let mut processor: AdvancedBootstrapProcessor<f64> = AdvancedBootstrapProcessor::new(config);
    let sample = processor
        .generate_beta_sample(10_000, 0.5, 3.0)
        .expect("beta sample generation should succeed");
    for &v in sample.iter() {
        assert!((0.0..=1.0).contains(&v), "Beta sample out of range: {v}");
    }
}

#[test]
fn test_generate_beta_sample_rejects_invalid_parameters() {
    let config = AdvancedBootstrapConfig {
        seed: Some(1),
        ..Default::default()
    };
    let mut processor: AdvancedBootstrapProcessor<f64> = AdvancedBootstrapProcessor::new(config);
    assert!(processor.generate_beta_sample(10, 0.0, 1.0).is_err());
    assert!(processor.generate_beta_sample(10, 1.0, -1.0).is_err());
    assert!(processor.generate_beta_sample(10, f64::NAN, 1.0).is_err());
}

#[test]
fn test_generate_gamma_sample_matches_theoretical_moments() {
    let config = AdvancedBootstrapConfig {
        seed: Some(2026),
        ..Default::default()
    };
    let mut processor: AdvancedBootstrapProcessor<f64> = AdvancedBootstrapProcessor::new(config);

    // shape >= 1: exercises the direct Marsaglia & Tsang squeeze method.
    let n = 100_000;
    let shape = 3.0;
    let scale = 2.0;
    let sample = processor
        .generate_gamma_sample(n, shape, scale)
        .expect("gamma sample generation should succeed");
    assert_eq!(sample.len(), n);

    let mean: f64 = sample.iter().sum::<f64>() / n as f64;
    let true_mean = shape * scale;
    assert!(
        (mean - true_mean).abs() < 0.1,
        "expected sample mean ~= {true_mean}, got {mean}"
    );

    let var: f64 = sample.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let true_var = shape * scale * scale;
    assert!(
        (var - true_var).abs() < 1.0,
        "expected sample var ~= {true_var}, got {var}"
    );

    // 0 < shape < 1: exercises the boosting-transform recursive branch.
    let shape_small = 0.4;
    let sample_small = processor
        .generate_gamma_sample(n, shape_small, 1.0)
        .expect("gamma sample generation should succeed");
    let mean_small: f64 = sample_small.iter().sum::<f64>() / n as f64;
    assert!(
        (mean_small - shape_small).abs() < 0.01,
        "expected sample mean ~= {shape_small}, got {mean_small}"
    );
    for &v in sample_small.iter() {
        assert!(v >= 0.0, "Gamma sample must be non-negative, got {v}");
    }
}

// ============================================================================
// `jarque_bera` / `jarque_bera_p_value` / `anderson_darling` fix tests.
//
// Wave-1 finding: `compute_distribution_stats` hardcoded
// `jarque_bera = F::zero()` and `anderson_darling = F::zero()`
// ("Simplified") regardless of the bootstrap distribution's actual shape,
// silently reporting perfectly-normal-looking diagnostics for every
// bootstrap run. Fixed by computing the real Jarque-Bera statistic (plus,
// new, a real chi-square(2) p-value for it) and the real (classical)
// Anderson-Darling statistic.
//
// Reference values below were computed independently in Python via numpy
// (skewness/kurtosis/JB, using the SAME sample-standard-deviation --
// `ddof=1` -- convention as this crate's `compute_std`) and
// `scipy.stats.chi2.sf`. The from-scratch Anderson-Darling reference
// formula was cross-checked against `scipy.stats.anderson(...,
// dist='norm').statistic` and matches to >= 9 decimal places for every
// fixture below -- NOT derived from this crate:
//
// ```python
// import numpy as np
// from scipy import stats
// mean = x.mean(); std = np.std(x, ddof=1); z = (x - mean) / std
// skew = np.mean(z**3); kurt = np.mean(z**4) - 3.0
// jb = (len(x) / 6.0) * (skew**2 + kurt**2 / 4.0)
// p = stats.chi2.sf(jb, df=2)
// ad_reference = stats.anderson(x, dist='norm').statistic
// ```
// ============================================================================
mod distribution_stats_fix_tests {
    use super::*;
    use approx::assert_relative_eq;

    fn processor() -> AdvancedBootstrapProcessor<f64> {
        AdvancedBootstrapProcessor::new(AdvancedBootstrapConfig {
            seed: Some(2026_07_29),
            ..Default::default()
        })
    }

    /// Roughly symmetric, non-constant data (a small discretized
    /// approximately-normal histogram): skewness == 0, modest excess
    /// kurtosis, so both the Jarque-Bera and Anderson-Darling statistics
    /// should be small and the JB p-value should be large (no evidence
    /// against normality).
    #[test]
    fn test_jarque_bera_and_anderson_darling_normal_like_sample_matches_reference() {
        let data = array![
            -3.0, -2.0, -2.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0
        ];
        let stats = processor()
            .compute_distribution_stats(&data)
            .expect("compute_distribution_stats should succeed");

        assert_relative_eq!(stats.skewness, 0.0, epsilon = 1e-9);
        assert_relative_eq!(stats.kurtosis, -0.9609375, epsilon = 1e-9);
        assert_relative_eq!(stats.jarque_bera, 0.6156005859375, epsilon = 1e-9);
        assert_relative_eq!(
            stats.jarque_bera_p_value,
            0.7350621004217005,
            max_relative = 1e-6
        );
        // `max_relative` here (rather than a tighter `epsilon`, as used
        // for the other reference values above) accommodates the ~1.5e-7
        // absolute-error Abramowitz-Stegun `erf` approximation this
        // crate's `distributions::normal::Normal::cdf` uses internally
        // (see `Normal`'s own `erf` helper) -- shared, pre-existing
        // infrastructure this fix reuses rather than reimplements.
        assert_relative_eq!(
            stats.anderson_darling,
            0.25752393814464725,
            max_relative = 1e-4
        );

        // The bug under test: `jarque_bera`/`anderson_darling` were
        // previously ALWAYS exactly 0.0 regardless of data, and no
        // p-value was computed at all.
        assert!(
            stats.jarque_bera_p_value > 0.5,
            "normal-like data should look highly consistent with normality, got p={}",
            stats.jarque_bera_p_value
        );
    }

    /// Heavily right-skewed data (one large outlier among otherwise
    /// tightly clustered values): both the Jarque-Bera and
    /// Anderson-Darling statistics should be substantially larger than the
    /// normal-like case above, and the JB p-value should be small (reject
    /// normality at the usual 5% level) -- exactly the
    /// "heavily skewed samples low p" case named in this fix's scope.
    #[test]
    fn test_jarque_bera_and_anderson_darling_skewed_sample_matches_reference() {
        let data = array![1.0, 1.1, 1.0, 0.9, 1.05, 0.95, 1.0, 1.1, 50.0];
        let stats = processor()
            .compute_distribution_stats(&data)
            .expect("compute_distribution_stats should succeed");

        assert_relative_eq!(stats.skewness, 2.074010781025194, max_relative = 1e-6);
        assert_relative_eq!(stats.kurtosis, 2.6294608777797803, max_relative = 1e-6);
        assert_relative_eq!(stats.jarque_bera, 9.04505527012851, max_relative = 1e-6);
        assert_relative_eq!(
            stats.jarque_bera_p_value,
            0.010861534945879272,
            max_relative = 1e-4
        );
        // See the normal-like test above for why `max_relative` is
        // 1e-4 (the shared `Normal::cdf` erf approximation's precision
        // floor), not tighter.
        assert_relative_eq!(
            stats.anderson_darling,
            2.780359416768915,
            max_relative = 1e-4
        );

        // This is the assertion that would have FAILED under the old
        // `jarque_bera = F::zero()` code: a hardcoded 0.0 statistic (and
        // no p-value at all) looks "perfectly normal" regardless of this
        // obviously non-normal, heavily-skewed data.
        assert!(
            stats.jarque_bera_p_value < 0.05,
            "heavily skewed data should look inconsistent with normality, got p={}",
            stats.jarque_bera_p_value
        );
        assert!(
            stats.anderson_darling > 1.0,
            "heavily skewed data should have a clearly elevated Anderson-Darling statistic, \
             got {}",
            stats.anderson_darling
        );
    }

    /// Extremely skewed data (19 identical values + 1 huge outlier): the
    /// true Jarque-Bera p-value (~2.8e-43) underflows to exactly 0.0 in
    /// this crate's `f64` `1 - cdf` computation (the same documented
    /// underflow-to-zero behavior as
    /// `regression::stat_tests::f_test_p_value` for very small
    /// probabilities), so this is asserted as a bound rather than an
    /// exact match; the statistics themselves remain exact-matchable.
    #[test]
    fn test_jarque_bera_extremely_skewed_sample_p_value_near_zero() {
        let mut data_vec = vec![2.0; 19];
        data_vec.push(100.0);
        let data = Array1::from(data_vec);
        let stats = processor()
            .compute_distribution_stats(&data)
            .expect("compute_distribution_stats should succeed");

        assert_relative_eq!(stats.jarque_bera, 195.97713020833336, max_relative = 1e-6);
        assert!(stats.jarque_bera_p_value < 1e-9);
        assert!((0.0..=1.0).contains(&stats.jarque_bera_p_value));
        // See the normal-like test above for why `max_relative` is
        // 1e-4 (the shared `Normal::cdf` erf approximation's precision
        // floor), not tighter.
        assert_relative_eq!(
            stats.anderson_darling,
            7.176182639932591,
            max_relative = 1e-4
        );
    }

    /// Degenerate (zero-variance) samples have no shape to test: the
    /// statistics must be the honest "no evidence against normality"
    /// values (0 / 1), not NaN/garbage from a division by a zero standard
    /// deviation.
    #[test]
    fn test_jarque_bera_constant_sample_is_well_defined() {
        let data = array![5.0, 5.0, 5.0, 5.0];
        let stats = processor()
            .compute_distribution_stats(&data)
            .expect("compute_distribution_stats should succeed");
        assert_eq!(stats.jarque_bera, 0.0);
        assert_eq!(stats.jarque_bera_p_value, 1.0);
        assert_eq!(stats.anderson_darling, 0.0);
    }

    /// `jarque_bera_p_value` must always be a valid probability.
    #[test]
    fn test_jarque_bera_p_value_always_in_unit_interval() {
        let cases: [Vec<f64>; 3] = [
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 1.0, 1.0, 1.0, 100.0],
            vec![-5.0, -1.0, 0.0, 1.0, 5.0, 5.0, 5.0],
        ];
        for data in cases {
            let arr = Array1::from(data.clone());
            let stats = processor()
                .compute_distribution_stats(&arr)
                .expect("compute_distribution_stats should succeed");
            assert!(
                (0.0..=1.0).contains(&stats.jarque_bera_p_value),
                "p-value out of range for {data:?}: {}",
                stats.jarque_bera_p_value
            );
            assert!(stats.jarque_bera >= 0.0);
            assert!(stats.anderson_darling >= 0.0);
        }
    }

    /// End-to-end: the real diagnostics must also be reachable through the
    /// public `bootstrap` entry point (not just the private
    /// `compute_distribution_stats` helper tested directly above).
    #[test]
    fn test_bootstrap_diagnostics_exposes_real_jarque_bera_and_anderson_darling() {
        // A statistic function returning a value that depends on which
        // resampled indices were drawn (here, the max of the resample)
        // tends to produce a skewed bootstrap distribution -- unlike the
        // mean, whose bootstrap distribution is close to symmetric even
        // for skewed underlying data.
        let data = array![1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 100.0];
        let max_fn = |x: &ArrayView1<f64>| -> StatsResult<f64> {
            Ok(x.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        };
        let config = AdvancedBootstrapConfig {
            n_bootstrap: 300,
            seed: Some(9),
            ..Default::default()
        };
        let mut processor = AdvancedBootstrapProcessor::new(config);
        let result = processor
            .bootstrap(&data.view(), max_fn)
            .expect("bootstrap should succeed");

        let stats = &result.diagnostics.distribution_stats;
        assert!((0.0..=1.0).contains(&stats.jarque_bera_p_value));
        assert!(stats.jarque_bera >= 0.0);
        assert!(stats.anderson_darling >= 0.0);
        assert!(stats.jarque_bera.is_finite());
        assert!(stats.anderson_darling.is_finite());
    }
}
