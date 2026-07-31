// Tests for `regularized.rs`, split into a separate file (via `#[path = ...]`)
// to keep the main implementation file under the workspace's 2000-line
// guideline. This file's top-level content *is* the `tests` module body
// (same pattern used by `cross_platform_optimized_tests.rs`).

use super::*;
use scirs2_core::ndarray::{array, Array1, Array2};

// ============================================================================
// Pre-existing `RidgeRegression` struct tests (moved here verbatim).
// ============================================================================
mod ridge_regression_struct_tests {
    use super::*;

    fn make_dataset() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
        )
        .expect("shape ok");
        let y = array![1.0_f64, 3.0, 5.0, 7.0, 9.0, 11.0];
        (x, y)
    }

    /// RidgeRegression is publicly accessible (compile test).
    #[test]
    fn test_ridge_regression_is_pub() {
        let _ = RidgeRegression::new(1.0);
    }

    /// RidgeRegression::fit returns a result without error.
    #[test]
    fn test_ridge_regression_fit() {
        let (x, y) = make_dataset();
        let mut model = RidgeRegression::new(0.5);
        let result = model.fit(&x.view(), &y.view());
        assert!(
            result.is_ok(),
            "Ridge fit should succeed: {:?}",
            result.err()
        );
    }

    /// Predictions from FittedRidgeRegression have the correct shape.
    #[test]
    fn test_ridge_regression_predict_shape() {
        let (x, y) = make_dataset();
        let mut model = RidgeRegression::new(1.0);
        let fitted = model.fit(&x.view(), &y.view()).expect("fit ok");
        let preds = fitted.predict(&x.view()).expect("predict ok");
        assert_eq!(preds.len(), x.nrows());
    }

    /// Ridge with a very small alpha should still yield reasonable predictions.
    #[test]
    fn test_ridge_regression_low_alpha() {
        let (x, y) = make_dataset();
        let mut model = RidgeRegression::new(1e-8);
        let fitted = model.fit(&x.view(), &y.view()).expect("fit ok");
        let preds = fitted.predict(&x.view()).expect("predict ok");
        for (p, t) in preds.iter().zip(y.iter()) {
            assert!((p - t).abs() < 0.5, "pred={p} target={t}");
        }
    }
}

// ============================================================================
// `f_p_value` fix tests.
//
// Wave-1 finding: `f_p_value` (the overall-model F-test p-value) was
// hardcoded to `F::zero()` in `ridge_regression`, `lasso_regression`,
// `elastic_net` and `group_lasso`, unconditionally signalling maximal
// statistical significance regardless of the actual fit quality. Fixed by
// computing the true `F(df_model, df_residuals)` survival function via
// `f_test_p_value` (now in `regression::stat_tests`, moved out of
// `regularized.rs` to keep it under the workspace's 2000-line guideline --
// see `stat_tests.rs` for direct unit tests of the helper itself, scipy
// reference values included) / `distributions::f::F`.
//
// The fixture reference values below (`f_statistic`, `f_p_value`) were
// computed independently in Python via `numpy.linalg.lstsq` +
// `scipy.stats.f.sf`, NOT derived from this crate:
//
// ```python
// import numpy as np
// from scipy import stats
// X = np.column_stack([np.ones(20), x1, x2])
// coef, *_ = np.linalg.lstsq(X, y, rcond=None)
// resid = y - X @ coef
// ss_total = np.sum((y - y.mean())**2)
// ss_resid = np.sum(resid**2)
// f_stat = ((ss_total - ss_resid)/2) / (ss_resid/17)
// p_val = stats.f.sf(f_stat, 2, 17)
// ```
// ============================================================================
mod f_p_value_fix_tests {
    use super::*;
    use approx::assert_relative_eq;

    // ------------------------------------------------------------------
    // Shared fixture: 20 observations, 2 predictors + intercept.
    //
    // `x1` = 1..=20 (monotonic); `x2` is a fixed non-monotonic pattern.
    // Two response vectors, both NON-CONSTANT:
    //   - `y_strong`: an almost-exact linear function of x1, x2 (small
    //     deterministic perturbation) -> should be highly significant.
    //   - `y_noise`: values with no real linear relationship to x1, x2
    //     -> should NOT look significant. This is the case the pre-fix
    //     hardcoded-`F::zero()` code got completely wrong (it always
    //     reported "0.0", i.e. maximal significance, even here).
    // ------------------------------------------------------------------

    fn fixture_x() -> Array2<f64> {
        let x1 = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
        ];
        let x2 = [
            5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0, 1.0, 6.0, 10.0, 2.0, 8.0, 3.0, 9.0, 1.0, 7.0, 4.0,
            10.0, 5.0, 6.0,
        ];
        let n = x1.len();
        let mut x = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            x[[i, 0]] = x1[i];
            x[[i, 1]] = x2[i];
        }
        x
    }

    fn fixture_y_strong() -> Array1<f64> {
        array![
            -2.2, 3.3, -0.9, 10.6, 3.7, 14.05, 12.35, 24.75, 19.9, 17.12, 31.95, 26.18, 36.28,
            30.58, 45.2, 39.64, 46.92, 41.22, 51.32, 53.1
        ]
    }

    fn fixture_y_noise() -> Array1<f64> {
        array![
            3.0, 7.0, 2.0, 9.0, 4.0, 8.0, 1.0, 6.0, 5.0, 10.0, 2.5, 7.5, 3.5, 9.5, 1.5, 6.5, 4.5,
            10.5, 5.5, 8.5
        ]
    }

    #[test]
    fn test_ridge_regression_f_p_value_strong_signal_matches_scipy() {
        let x = fixture_x();
        let y = fixture_y_strong();
        // alpha = 0.0 makes ridge mathematically identical to plain OLS:
        // `solve_ridge_system` always dispatches to a direct `lstsq` solve,
        // and appending all-zero rows (sqrt(0) * I) to both X and y does
        // not change the least-squares solution.
        let result = ridge_regression(
            &x.view(),
            &y.view(),
            Some(0.0),
            None,
            None,
            None,
            None,
            None,
        )
        .expect("ridge regression should succeed");
        assert_relative_eq!(result.f_statistic, 97408.17758838173, max_relative = 1e-4);
        // True p-value (~3.14e-35) underflows to exactly 0.0 once computed
        // as `1 - cdf` in f64 (cdf rounds to 1.0) -- still correctly
        // signals "essentially certain" rather than the old fabricated 0.0
        // (which claimed the *same* certainty regardless of data).
        assert!(
            result.f_p_value < 1e-12,
            "expected ~0 (perfect fit), got {}",
            result.f_p_value
        );
        assert!(result.f_p_value >= 0.0);
    }

    #[test]
    fn test_ridge_regression_f_p_value_weak_signal_matches_scipy() {
        let x = fixture_x();
        let y = fixture_y_noise();
        let result = ridge_regression(
            &x.view(),
            &y.view(),
            Some(0.0),
            None,
            None,
            None,
            None,
            None,
        )
        .expect("ridge regression should succeed");
        assert_relative_eq!(result.f_statistic, 1.6747197597833423, max_relative = 1e-4);
        assert_relative_eq!(
            result.f_p_value,
            0.21683030932143513,
            max_relative = 1e-3,
            epsilon = 1e-6
        );
        // This is the assertion that would have FAILED under the old
        // `f_p_value = F::zero()` code: the true p-value here is large
        // (~0.22), meaning this noise-only fit is NOT statistically
        // significant, but the old code always reported 0.0 (maximal
        // significance) regardless.
        assert!(
            result.f_p_value > 0.05,
            "expected a large, non-significant p-value, got {}",
            result.f_p_value
        );
    }

    #[test]
    fn test_lasso_regression_f_p_value_distinguishes_signal_from_noise() {
        let x = fixture_x();
        let strong = lasso_regression(
            &x.view(),
            &fixture_y_strong().view(),
            Some(0.01),
            None,
            None,
            None,
            None,
            None,
        )
        .expect("lasso regression should succeed");
        let noise = lasso_regression(
            &x.view(),
            &fixture_y_noise().view(),
            Some(0.01),
            None,
            None,
            None,
            None,
            None,
        )
        .expect("lasso regression should succeed");

        assert!((0.0..=1.0).contains(&strong.f_p_value));
        assert!((0.0..=1.0).contains(&noise.f_p_value));
        assert!(
            strong.f_p_value < 0.01,
            "strong-signal fit should be highly significant, got {}",
            strong.f_p_value
        );
        // The bug under test: `f_p_value` was previously ALWAYS exactly
        // 0.0 regardless of data, so this assertion on the noise fit
        // would have failed before the fix.
        assert!(
            noise.f_p_value > 0.05,
            "weak-signal fit should not look significant, got {}",
            noise.f_p_value
        );
    }

    #[test]
    fn test_elastic_net_f_p_value_distinguishes_signal_from_noise() {
        let x = fixture_x();
        let strong = elastic_net(
            &x.view(),
            &fixture_y_strong().view(),
            Some(0.01),
            Some(0.5),
            None,
            None,
            None,
            None,
            None,
        )
        .expect("elastic net should succeed");
        let noise = elastic_net(
            &x.view(),
            &fixture_y_noise().view(),
            Some(0.01),
            Some(0.5),
            None,
            None,
            None,
            None,
            None,
        )
        .expect("elastic net should succeed");

        assert!((0.0..=1.0).contains(&strong.f_p_value));
        assert!((0.0..=1.0).contains(&noise.f_p_value));
        assert!(
            strong.f_p_value < 0.01,
            "strong-signal fit should be highly significant, got {}",
            strong.f_p_value
        );
        assert!(
            noise.f_p_value > 0.05,
            "weak-signal fit should not look significant, got {}",
            noise.f_p_value
        );
    }

    #[test]
    fn test_group_lasso_f_p_value_distinguishes_signal_from_noise() {
        let x = fixture_x();
        let groups = [0usize, 1usize];
        let strong = group_lasso(
            &x.view(),
            &fixture_y_strong().view(),
            &groups,
            Some(0.01),
            None,
            None,
            None,
            None,
            None,
        )
        .expect("group lasso should succeed");
        let noise = group_lasso(
            &x.view(),
            &fixture_y_noise().view(),
            &groups,
            Some(0.01),
            None,
            None,
            None,
            None,
            None,
        )
        .expect("group lasso should succeed");

        assert!((0.0..=1.0).contains(&strong.f_p_value));
        assert!((0.0..=1.0).contains(&noise.f_p_value));
        assert!(
            strong.f_p_value < 0.01,
            "strong-signal fit should be highly significant, got {}",
            strong.f_p_value
        );
        assert!(
            noise.f_p_value > 0.05,
            "weak-signal fit should not look significant, got {}",
            noise.f_p_value
        );
    }
}

// ============================================================================
// `t_test_p_value` fix tests.
//
// Wave-1 follow-up finding (discovered while fixing `f_p_value`): the
// per-coefficient `p_values` in every regression variant in this module (and
// in `robust.rs`, `stepwise.rs`, `linear.rs`) were computed via
// `2 * (1 - |t| / sqrt(df + t^2))` -- a formula that is NOT a valid p-value:
// it commonly evaluates to well over 1.0 (see
// `stat_tests::tests::test_old_formula_produced_out_of_range_values`), the
// exact kind of "looks like a real computation" silent fabrication this
// audit targets. Fixed by computing the true two-sided Student's
// t-distribution survival probability via `t_test_p_value` (now in
// `regression::stat_tests`, moved out of `regularized.rs` to keep it under
// the workspace's 2000-line guideline -- see `stat_tests.rs` for direct
// unit tests of the helper itself, scipy reference values included) /
// `distributions::student_t::StudentT`.
// ============================================================================
mod t_p_value_fix_tests {
    use super::*;

    // ------------------------------------------------------------------
    // End-to-end: the public regression entry points must report
    // per-coefficient p-values that are (a) always in [0, 1] -- the old
    // formula routinely was not -- and (b) actually reflect which
    // predictor carries real signal, using the same NON-CONSTANT fixtures
    // as the `f_p_value` tests above.
    // ------------------------------------------------------------------

    // Local copy of the `f_p_value_fix_tests` fixtures (that module's
    // helpers are private to it, so a sibling test module cannot reuse them
    // directly): 20 observations, 2 NON-CONSTANT predictors, and two
    // response vectors (near-perfect linear signal vs. no real
    // relationship).
    fn fixture_x() -> Array2<f64> {
        let x1 = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
        ];
        let x2 = [
            5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0, 1.0, 6.0, 10.0, 2.0, 8.0, 3.0, 9.0, 1.0, 7.0, 4.0,
            10.0, 5.0, 6.0,
        ];
        let n = x1.len();
        let mut x = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            x[[i, 0]] = x1[i];
            x[[i, 1]] = x2[i];
        }
        x
    }

    fn fixture_y_strong() -> Array1<f64> {
        array![
            -2.2, 3.3, -0.9, 10.6, 3.7, 14.05, 12.35, 24.75, 19.9, 17.12, 31.95, 26.18, 36.28,
            30.58, 45.2, 39.64, 46.92, 41.22, 51.32, 53.1
        ]
    }

    #[test]
    fn test_ridge_regression_p_values_in_range_and_reflect_signal() {
        let x = fixture_x();
        let strong = ridge_regression(
            &x.view(),
            &fixture_y_strong().view(),
            Some(0.0),
            None,
            None,
            None,
            None,
            None,
        )
        .expect("ridge regression should succeed");
        let noise = ridge_regression(
            &x.view(),
            &array![
                3.0, 7.0, 2.0, 9.0, 4.0, 8.0, 1.0, 6.0, 5.0, 10.0, 2.5, 7.5, 3.5, 9.5, 1.5, 6.5,
                4.5, 10.5, 5.5, 8.5
            ]
            .view(),
            Some(0.0),
            None,
            None,
            None,
            None,
            None,
        )
        .expect("ridge regression should succeed");

        for &p in strong.p_values.iter() {
            assert!(
                (0.0..=1.0).contains(&p),
                "strong-signal p-value out of range: {p}"
            );
        }
        for &p in noise.p_values.iter() {
            assert!(
                (0.0..=1.0).contains(&p),
                "noise-only p-value out of range: {p}"
            );
        }
        // The near-perfect-fit strong-signal model should have highly
        // significant (near-zero) p-values for its (non-intercept)
        // predictors -- x1 and x2 are coefficients 1 and 2 (index 0 is the
        // intercept column added internally is not present here since this
        // fixture has no explicit intercept column; both columns 0 and 1
        // are real predictors).
        assert!(
            strong.p_values.iter().all(|&p| p < 0.01),
            "expected all strong-signal coefficients to look significant, got {:?}",
            strong.p_values
        );
    }
}
