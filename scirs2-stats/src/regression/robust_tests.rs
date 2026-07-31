// Tests for `robust.rs`, split into a separate file (via `#[path = ...]`) to
// keep the main implementation file under the workspace's 2000-line
// guideline. This file's top-level content *is* the `tests` module body
// (same pattern used by `cross_platform_optimized_tests.rs`).

use super::*;
use scirs2_core::ndarray::{array, Array1, Array2};

// -------------------------------------------------------------------
// LTS regression tests (pre-existing, moved here verbatim)
// -------------------------------------------------------------------

#[test]
fn test_lts_basic() {
    let x = Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("shape");
    let y = array![2.1, 4.0, 5.9, 8.1, 10.0, 12.0, 14.0, 16.1, 18.0, 20.1];
    let result = lts_regression(&x.view(), &y.view(), None, None, None, Some(42))
        .expect("LTS should succeed");
    // Slope near 2.0
    assert!((result.coefficients[1] - 2.0).abs() < 0.3);
}

#[test]
fn test_lts_with_outlier() {
    let x = Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("shape");
    let y = array![2.1, 4.0, 5.9, 8.1, 10.0, 12.0, 14.0, 16.1, 18.0, 50.0];
    let result = lts_regression(&x.view(), &y.view(), None, None, None, Some(42))
        .expect("LTS should succeed");
    // Slope should be close to 2 despite the outlier
    assert!((result.coefficients[1] - 2.0).abs() < 0.5);
    // Outlier should be excluded
    assert!(!result.inlier_mask[9]);
}

#[test]
fn test_lts_multiple_outliers() {
    let x = Array2::from_shape_vec(
        (12, 1),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .expect("shape");
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 50.0, 60.0, 22.0, 24.0];
    let result = lts_regression(&x.view(), &y.view(), Some(0.2), Some(200), None, Some(42))
        .expect("LTS should succeed");
    assert!((result.coefficients[1] - 2.0).abs() < 0.5);
}

#[test]
fn test_lts_r_squared() {
    let x = Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("shape");
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
    let result = lts_regression(&x.view(), &y.view(), None, None, None, Some(42))
        .expect("LTS should succeed");
    let r2: f64 = scirs2_core::numeric::NumCast::from(result.r_squared).expect("cast");
    assert!(r2 > 0.95);
}

#[test]
fn test_lts_dimension_mismatch() {
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("shape");
    let y = array![1.0, 2.0, 3.0];
    assert!(lts_regression(&x.view(), &y.view(), None, None, None, None).is_err());
}

#[test]
fn test_lts_too_few_observations() {
    let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).expect("shape");
    let y = array![1.0, 2.0];
    assert!(lts_regression(&x.view(), &y.view(), None, None, None, None).is_err());
}

// -------------------------------------------------------------------
// Bisquare regression tests (pre-existing, moved here verbatim)
// -------------------------------------------------------------------

#[test]
fn test_bisquare_basic() {
    let x = Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("shape");
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
    let result = bisquare_regression(&x.view(), &y.view(), None, None, None, None)
        .expect("bisquare should succeed");
    assert!((result.coefficients[1] - 2.0).abs() < 0.3);
}

#[test]
fn test_bisquare_with_outlier() {
    let x = Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("shape");
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 50.0];
    let result = bisquare_regression(&x.view(), &y.view(), None, None, None, None)
        .expect("bisquare should succeed");
    // Slope should be closer to 2.0 than OLS would give
    assert!((result.coefficients[1] - 2.0).abs() < 1.0);
}

#[test]
fn test_bisquare_r_squared() {
    let x = Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("shape");
    let y = array![2.1, 4.0, 5.9, 8.1, 10.0, 12.0, 14.1, 16.0, 18.0, 20.1];
    let result = bisquare_regression(&x.view(), &y.view(), None, None, None, None)
        .expect("bisquare should succeed");
    let r2: f64 = scirs2_core::numeric::NumCast::from(result.r_squared).expect("cast");
    assert!(r2 > 0.95);
}

#[test]
fn test_bisquare_custom_c() {
    let x = Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("shape");
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 50.0];
    // Very small c means more aggressive down-weighting
    let result = bisquare_regression(&x.view(), &y.view(), Some(2.0), None, None, None)
        .expect("bisquare should succeed");
    assert!((result.coefficients[1] - 2.0).abs() < 1.0);
}

#[test]
fn test_bisquare_dimension_mismatch() {
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("shape");
    let y = array![1.0, 2.0, 3.0];
    assert!(bisquare_regression(&x.view(), &y.view(), None, None, None, None).is_err());
}

#[test]
fn test_bisquare_too_few() {
    let x = Array2::from_shape_vec((1, 1), vec![1.0]).expect("shape");
    let y = array![1.0];
    assert!(bisquare_regression(&x.view(), &y.view(), None, None, None, None).is_err());
}

// ============================================================================
// `f_p_value` fix tests.
//
// Wave-1 finding: `f_p_value` was hardcoded to `F::zero()` in
// `simple_linear_regression` (reachable via `ransac`), `huber_regression`
// and `bisquare_regression`, unconditionally signalling maximal statistical
// significance regardless of the actual fit. Fixed by computing the true
// `F(df_model, df_residuals)` survival function via
// `stat_tests::f_test_p_value`.
//
// Fixture reference values computed independently in Python via
// `numpy.linalg.lstsq` + `scipy.stats.f.sf` (see comments below), NOT
// derived from this crate.
// ============================================================================
mod f_p_value_fix_tests {
    use super::*;
    use approx::assert_relative_eq;

    // Single predictor + intercept, 16 observations, NON-CONSTANT data:
    //   - y_strong: an almost-exact linear function of x (tiny deterministic
    //     perturbation) -> should be highly significant.
    //   - y_noise: values with essentially no linear relationship to x
    //     -> should NOT look significant. This is the case the old
    //     hardcoded-`F::zero()` code got completely wrong.
    //
    // Reference (numpy.linalg.lstsq + scipy.stats.f.sf(F, 1, 14)):
    //   strong: coef=[5.04475, 1.99664706] f_stat=61302.055... p_val=6.78e-27
    //   noise:  coef=[6.0625, 0.02205882]  f_stat=0.019909616660871855 p_val=0.8897999949258737
    fn fixture_x_col() -> Vec<f64> {
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]
    }

    fn fixture_y_strong() -> Array1<f64> {
        array![
            7.2, 8.9, 11.15, 12.8, 15.1, 17.05, 18.85, 21.2, 22.9, 25.12, 26.95, 29.18, 30.88,
            33.08, 34.8, 37.1
        ]
    }

    fn fixture_y_noise() -> Array1<f64> {
        array![4.0, 9.0, 3.0, 10.0, 5.0, 8.0, 2.0, 7.0, 6.0, 10.0, 3.5, 8.5, 4.5, 9.5, 2.5, 7.5]
    }

    /// `simple_linear_regression` (the private helper containing the
    /// originally-flagged line, reached in production via `ransac`) is a
    /// direct `lstsq` fit with no regularization, so it must match the
    /// scipy/numpy reference to high precision.
    #[test]
    fn test_simple_linear_regression_f_p_value_matches_scipy() {
        let x1 = fixture_x_col();
        let n = x1.len();
        // simple_linear_regression takes a design matrix that ALREADY
        // includes the intercept column (it does not add one itself).
        let mut x = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            x[[i, 1]] = x1[i];
        }

        let strong = simple_linear_regression(&x.view(), &fixture_y_strong().view())
            .expect("regression should succeed");
        assert_relative_eq!(strong.f_statistic, 61302.05537972725, max_relative = 1e-4);
        assert!(
            strong.f_p_value < 1e-12,
            "expected ~0 (near-perfect fit), got {}",
            strong.f_p_value
        );

        let noise = simple_linear_regression(&x.view(), &fixture_y_noise().view())
            .expect("regression should succeed");
        assert_relative_eq!(noise.f_statistic, 0.019909616660871855, max_relative = 1e-4);
        assert_relative_eq!(
            noise.f_p_value,
            0.8897999949258737,
            max_relative = 1e-3,
            epsilon = 1e-6
        );
        // This assertion would have FAILED under the old
        // `f_p_value = F::zero()` code: the true p-value here is ~0.89
        // (no significant relationship), but the old code always reported
        // 0.0 (maximal significance) regardless of data.
        assert!(
            noise.f_p_value > 0.5,
            "expected a large, non-significant p-value, got {}",
            noise.f_p_value
        );
    }

    #[test]
    fn test_huber_regression_f_p_value_distinguishes_signal_from_noise() {
        let x1 = fixture_x_col();
        let n = x1.len();
        let x = Array2::from_shape_vec((n, 1), x1).expect("shape ok");

        let strong = huber_regression(
            &x.view(),
            &fixture_y_strong().view(),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("huber regression should succeed");
        let noise = huber_regression(
            &x.view(),
            &fixture_y_noise().view(),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("huber regression should succeed");

        assert!((0.0..=1.0).contains(&strong.f_p_value));
        assert!((0.0..=1.0).contains(&noise.f_p_value));
        assert!(
            strong.f_p_value < 0.01,
            "strong-signal fit should be highly significant, got {}",
            strong.f_p_value
        );
        // Bug under test: `f_p_value` was previously ALWAYS exactly 0.0
        // regardless of data.
        assert!(
            noise.f_p_value > 0.3,
            "weak-signal fit should not look significant, got {}",
            noise.f_p_value
        );
    }

    #[test]
    fn test_bisquare_regression_f_p_value_distinguishes_signal_from_noise() {
        let x1 = fixture_x_col();
        let n = x1.len();
        let x = Array2::from_shape_vec((n, 1), x1).expect("shape ok");

        let strong = bisquare_regression(
            &x.view(),
            &fixture_y_strong().view(),
            None,
            None,
            None,
            None,
        )
        .expect("bisquare regression should succeed");
        let noise =
            bisquare_regression(&x.view(), &fixture_y_noise().view(), None, None, None, None)
                .expect("bisquare regression should succeed");

        assert!((0.0..=1.0).contains(&strong.f_p_value));
        assert!((0.0..=1.0).contains(&noise.f_p_value));
        assert!(
            strong.f_p_value < 0.01,
            "strong-signal fit should be highly significant, got {}",
            strong.f_p_value
        );
        assert!(
            noise.f_p_value > 0.3,
            "weak-signal fit should not look significant, got {}",
            noise.f_p_value
        );
    }

    #[test]
    fn test_ransac_f_p_value_distinguishes_signal_from_noise() {
        let x1 = fixture_x_col();
        let n = x1.len();
        let x = Array2::from_shape_vec((n, 1), x1).expect("shape ok");

        // Fixed random_seed for reproducibility; generous residual
        // threshold since neither fixture has genuine outliers to reject.
        let strong = ransac(
            &x.view(),
            &fixture_y_strong().view(),
            None,
            Some(5.0),
            None,
            None,
            Some(42),
        )
        .expect("ransac should succeed");
        let noise = ransac(
            &x.view(),
            &fixture_y_noise().view(),
            None,
            Some(5.0),
            None,
            None,
            Some(42),
        )
        .expect("ransac should succeed");

        assert!((0.0..=1.0).contains(&strong.f_p_value));
        assert!((0.0..=1.0).contains(&noise.f_p_value));
        assert!(
            strong.f_p_value < 0.05,
            "strong-signal fit should be significant, got {}",
            strong.f_p_value
        );
        assert!(
            noise.f_p_value > 0.3,
            "weak-signal fit should not look significant, got {}",
            noise.f_p_value
        );
    }
}

// ============================================================================
// `t_test_p_value` fix tests (per-coefficient p-values).
//
// Follow-up Wave-1 finding (discovered while fixing `f_p_value` in this
// same file): `simple_linear_regression`, `huber_regression` and
// `bisquare_regression` all computed per-coefficient `p_values` via
// `2 * (1 - |t| / sqrt(df + t^2))`, a formula that is not a valid p-value
// (it commonly exceeds 1.0). Fixed to use the real Student's t-distribution
// survival function via `stat_tests::t_test_p_value`. See
// `regularized_tests.rs::t_p_value_fix_tests / stat_tests.rs::tests` for direct scipy-verified
// reference values for the helper itself; these tests instead check the
// public regression entry points end-to-end.
// ============================================================================
mod t_p_value_fix_tests {
    use super::*;

    fn fixture_x_col() -> Vec<f64> {
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]
    }

    fn fixture_y_strong() -> Array1<f64> {
        array![
            7.2, 8.9, 11.15, 12.8, 15.1, 17.05, 18.85, 21.2, 22.9, 25.12, 26.95, 29.18, 30.88,
            33.08, 34.8, 37.1
        ]
    }

    fn fixture_y_noise() -> Array1<f64> {
        array![4.0, 9.0, 3.0, 10.0, 5.0, 8.0, 2.0, 7.0, 6.0, 10.0, 3.5, 8.5, 4.5, 9.5, 2.5, 7.5]
    }

    /// This is the assertion that would have FAILED under the old formula:
    /// per-coefficient p-values must always lie in the valid `[0, 1]`
    /// probability range (the old formula routinely exceeded 1.0).
    #[test]
    fn test_simple_linear_regression_p_values_in_range() {
        let x1 = fixture_x_col();
        let n = x1.len();
        let mut x = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            x[[i, 1]] = x1[i];
        }
        for y in [fixture_y_strong(), fixture_y_noise()] {
            let result = simple_linear_regression(&x.view(), &y.view()).expect("regression ok");
            for &p in result.p_values.iter() {
                assert!((0.0..=1.0).contains(&p), "p-value out of range: {p}");
            }
        }
    }

    #[test]
    fn test_huber_regression_p_values_distinguish_signal_from_noise() {
        let x1 = fixture_x_col();
        let n = x1.len();
        let x = Array2::from_shape_vec((n, 1), x1).expect("shape ok");

        let strong = huber_regression(
            &x.view(),
            &fixture_y_strong().view(),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("huber regression should succeed");
        let noise = huber_regression(
            &x.view(),
            &fixture_y_noise().view(),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("huber regression should succeed");

        for &p in strong.p_values.iter() {
            assert!((0.0..=1.0).contains(&p), "p-value out of range: {p}");
        }
        for &p in noise.p_values.iter() {
            assert!((0.0..=1.0).contains(&p), "p-value out of range: {p}");
        }
        // Slope coefficient (index 1): strongly significant for the
        // near-perfect linear fixture, not for the noise-only fixture.
        assert!(
            strong.p_values[1] < 0.01,
            "expected the slope to look significant, got {}",
            strong.p_values[1]
        );
        assert!(
            noise.p_values[1] > 0.3,
            "expected the slope to NOT look significant for noise, got {}",
            noise.p_values[1]
        );
    }

    #[test]
    fn test_bisquare_regression_p_values_distinguish_signal_from_noise() {
        let x1 = fixture_x_col();
        let n = x1.len();
        let x = Array2::from_shape_vec((n, 1), x1).expect("shape ok");

        let strong = bisquare_regression(
            &x.view(),
            &fixture_y_strong().view(),
            None,
            None,
            None,
            None,
        )
        .expect("bisquare regression should succeed");
        let noise =
            bisquare_regression(&x.view(), &fixture_y_noise().view(), None, None, None, None)
                .expect("bisquare regression should succeed");

        for &p in strong.p_values.iter() {
            assert!((0.0..=1.0).contains(&p), "p-value out of range: {p}");
        }
        for &p in noise.p_values.iter() {
            assert!((0.0..=1.0).contains(&p), "p-value out of range: {p}");
        }
        assert!(
            strong.p_values[1] < 0.01,
            "expected the slope to look significant, got {}",
            strong.p_values[1]
        );
        assert!(
            noise.p_values[1] > 0.3,
            "expected the slope to NOT look significant for noise, got {}",
            noise.p_values[1]
        );
    }
}
