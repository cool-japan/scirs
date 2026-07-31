//! Shared hypothesis-test p-value helpers for the regression module.
//!
//! Split out of `regularized.rs` to keep that file under the workspace's
//! 2000-line guideline; these two helpers are used across
//! `regularized.rs`, `robust.rs`, `stepwise.rs` and `linear.rs`.

use scirs2_core::numeric::Float;

/// Compute the p-value for the overall-model F-statistic.
///
/// This is the right-tail (survival) probability `P(X >= f_statistic)` for
/// `X ~ F(df_model, df_residuals)`, i.e. `1 - CDF(f_statistic)`. It answers
/// the standard regression question "could a model with no real explanatory
/// power (all non-intercept coefficients zero) plausibly have produced an
/// F-statistic this large by chance?" -- a small p-value rejects that null
/// hypothesis.
///
/// Degenerate/edge cases are handled explicitly because the general
/// `distributions::f::F::cdf` routine does not special-case `x = +inf`
/// (it would otherwise compute `inf / inf = NaN` internally):
/// * `df_model == 0` or `df_residuals == 0`: no F-test is defined, return 1.
/// * `f_statistic` is NaN: propagate NaN.
/// * `f_statistic <= 0`: cannot occur for a genuine least-squares fit, but a
///   handful of callers here use IRLS/robust refits that do not strictly
///   minimize the residual sum of squares, so treat it as "no evidence
///   against the null" (p = 1).
/// * `f_statistic` is `+inf` (a perfect fit drives residual SS to 0): the
///   tail probability is exactly 0 in the limit.
///
/// Shared by every regression flavor in `regularized.rs` as well as
/// `robust.rs`, `stepwise.rs` and `linear.rs`, which all populate the same
/// `RegressionResults::f_p_value` field.
pub(crate) fn f_test_p_value<F>(f_statistic: F, df_model: usize, df_residuals: usize) -> F
where
    F: Float,
{
    if df_model == 0 || df_residuals == 0 {
        return F::one();
    }
    if f_statistic.is_nan() {
        return F::nan();
    }
    if f_statistic.is_infinite() {
        return if f_statistic > F::zero() {
            F::zero()
        } else {
            F::one()
        };
    }
    if f_statistic <= F::zero() {
        return F::one();
    }

    let dfn = F::from(df_model).expect("Failed to convert degrees of freedom to float");
    let dfd = F::from(df_residuals).expect("Failed to convert degrees of freedom to float");

    match crate::distributions::f::F::new(dfn, dfd, F::zero(), F::one()) {
        Ok(dist) => {
            let p = F::one() - dist.cdf(f_statistic);
            if p < F::zero() {
                F::zero()
            } else if p > F::one() {
                F::one()
            } else {
                p
            }
        }
        Err(_) => F::one(),
    }
}

/// Compute the two-sided p-value for a single coefficient's t-statistic.
///
/// This is `2 * P(T >= |t_statistic|)` for `T ~ StudentT(df)`, the standard
/// two-sided test of the null hypothesis that a regression coefficient is
/// zero, computed via the crate's real Student's t-distribution CDF
/// (`distributions::student_t::StudentT`, itself backed by the regularized
/// incomplete beta function).
///
/// Every caller (`regularized.rs`, `robust.rs`, `stepwise.rs`, `linear.rs`)
/// previously used a closed-form-looking but mathematically invalid
/// substitute: `2 * (1 - |t| / sqrt(df + t^2))`. That expression is not a
/// valid p-value at all -- for typical (df, t) pairs it evaluates to well
/// over 1.0 (e.g. `t=1.0, df=17` gives ~1.53, not a probability), wildly
/// overstating how "non-significant" a fit looks. See `tests` below for
/// scipy-verified reference values and a direct demonstration that the old
/// formula produced out-of-range results.
pub(crate) fn t_test_p_value<F>(t_statistic: F, df: usize) -> F
where
    F: Float + Send + Sync + 'static + std::fmt::Display,
{
    if df == 0 {
        return F::one();
    }
    if t_statistic.is_nan() {
        return F::nan();
    }
    let t_abs = if t_statistic < F::zero() {
        -t_statistic
    } else {
        t_statistic
    };
    if t_abs.is_infinite() {
        return F::zero();
    }

    let dff = F::from(df).expect("Failed to convert degrees of freedom to float");

    match crate::distributions::student_t::StudentT::new(dff, F::zero(), F::one()) {
        Ok(dist) => {
            let p = (F::one() - dist.cdf(t_abs))
                * F::from(2.0).expect("Failed to convert constant to float");
            if p < F::zero() {
                F::zero()
            } else if p > F::one() {
                F::one()
            } else {
                p
            }
        }
        Err(_) => F::one(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ========================================================================
    // `f_test_p_value` tests.
    //
    // Reference values computed independently via `scipy.stats.f.sf`, NOT
    // derived from this crate.
    // ========================================================================

    #[test]
    fn test_f_test_p_value_matches_scipy_stats_f_sf() {
        let cases: &[(f64, usize, usize, f64)] = &[
            (1.0, 1, 10, 0.34089313230206),
            (5.0, 2, 20, 0.01734152991583262),
            (10.0, 3, 16, 0.0005942866595794999),
            (2.5, 4, 30, 0.0634764397982508),
            (0.5, 5, 25, 0.7732635745548724),
            (4.35, 1, 18, 0.05151514478735162),
        ];
        for &(f_stat, df1, df2, expected_p) in cases {
            let p = f_test_p_value(f_stat, df1, df2);
            assert_relative_eq!(p, expected_p, epsilon = 1e-6, max_relative = 1e-4);
        }
    }

    /// This is the assertion that would have FAILED against the historical
    /// bug (`f_p_value` hardcoded to exactly `F::zero()` regardless of the
    /// F-statistic): a middling F=1.0 on (1,10) degrees of freedom has a
    /// true p-value of ~0.34, nowhere near 0.
    #[test]
    fn test_f_test_p_value_not_hardcoded_zero() {
        let p = f_test_p_value(1.0_f64, 1, 10);
        assert!(p > 0.3, "expected p ~= 0.34, got {p}");
    }

    #[test]
    fn test_f_test_p_value_edge_cases() {
        assert_eq!(f_test_p_value(5.0_f64, 0, 10), 1.0);
        assert_eq!(f_test_p_value(5.0_f64, 3, 0), 1.0);
        assert_eq!(f_test_p_value(f64::INFINITY, 2, 10), 0.0);
        assert!(f_test_p_value(f64::NAN, 2, 10).is_nan());
        assert_eq!(f_test_p_value(0.0_f64, 2, 10), 1.0);
        assert_eq!(f_test_p_value(-1.0_f64, 2, 10), 1.0);
    }

    #[test]
    fn test_f_test_p_value_monotonically_decreasing_in_f() {
        let fs = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0];
        let mut prev = f_test_p_value(fs[0], 3, 20);
        for &f in &fs[1..] {
            let p = f_test_p_value(f, 3, 20);
            assert!(p < prev, "p({f}) = {p} should be < p of smaller F = {prev}");
            prev = p;
        }
    }

    // ========================================================================
    // `t_test_p_value` tests.
    //
    // Reference values computed independently via `scipy.stats.t.cdf`, NOT
    // derived from this crate:
    //   2 * (1 - scipy.stats.t.cdf(abs(t), df))
    // ========================================================================

    #[test]
    fn test_t_test_p_value_matches_scipy_stats_t_sf() {
        let cases: &[(f64, usize, f64)] = &[
            (1.0, 10, 0.34089313230205986),
            (2.11, 17, 0.04998212471210528),
            (3.5, 5, 0.017284431785293375),
            (0.5, 25, 0.6214477851902287),
            (4.35, 18, 0.0003858868406823035),
            (10.0, 3, 0.002128399058414221),
        ];
        for &(t, df, expected_p) in cases {
            let p = t_test_p_value(t, df);
            assert_relative_eq!(p, expected_p, epsilon = 1e-6, max_relative = 1e-4);
            // Symmetry: the two-sided p-value only depends on |t|.
            let p_neg = t_test_p_value(-t, df);
            assert_relative_eq!(p, p_neg, epsilon = 1e-12);
        }
    }

    /// This is the demonstration that the OLD formula
    /// (`2 * (1 - |t| / sqrt(df + t^2))`, reproduced here verbatim and
    /// standalone -- NOT calling into the crate -- purely to document why
    /// the fix was necessary) is not a valid p-value at all: for these
    /// ordinary (t, df) pairs it exceeds 1.0, which is impossible for any
    /// real probability.
    #[test]
    fn test_old_formula_produced_out_of_range_values() {
        fn old_formula(t: f64, df: f64) -> f64 {
            let t_abs = t.abs();
            2.0 * (1.0 - t_abs / (df + t_abs * t_abs).sqrt())
        }
        let cases: &[(f64, f64)] = &[(1.0, 10.0), (2.0, 17.0), (0.5, 30.0), (3.0, 100.0)];
        for &(t, df) in cases {
            let old_p = old_formula(t, df);
            assert!(
                old_p > 1.0,
                "expected the old formula to exceed 1.0 (demonstrating it was invalid) for t={t}, df={df}, got {old_p}"
            );
        }
    }

    /// This is the assertion that would have FAILED under the old formula:
    /// `t_test_p_value` must always return a value in the valid probability
    /// range, unlike its predecessor.
    #[test]
    fn test_t_test_p_value_always_in_valid_range() {
        for &df in &[1usize, 2, 5, 10, 17, 30, 100] {
            for &t in &[0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0] {
                let p = t_test_p_value(t, df);
                assert!(
                    (0.0..=1.0).contains(&p),
                    "t_test_p_value({t}, {df}) = {p} is out of the valid [0, 1] range"
                );
            }
        }
    }

    #[test]
    fn test_t_test_p_value_edge_cases() {
        assert_eq!(t_test_p_value(5.0_f64, 0), 1.0);
        assert_eq!(t_test_p_value(f64::INFINITY, 10), 0.0);
        assert_eq!(t_test_p_value(f64::NEG_INFINITY, 10), 0.0);
        assert!(t_test_p_value(f64::NAN, 10).is_nan());
        assert_relative_eq!(t_test_p_value(0.0_f64, 10), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_t_test_p_value_monotonically_decreasing_in_t() {
        let ts = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0];
        let mut prev = t_test_p_value(ts[0], 15);
        for &t in &ts[1..] {
            let p = t_test_p_value(t, 15);
            assert!(p < prev, "p({t}) = {p} should be < p of smaller t = {prev}");
            prev = p;
        }
    }
}
