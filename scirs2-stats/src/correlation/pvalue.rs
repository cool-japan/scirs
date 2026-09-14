//! Numerically stable p-values for the correlation coefficients.
//!
//! Every correlation coefficient in the parent module is tested against the
//! same null distribution: under H0 (no correlation) the statistic
//! `t = r * sqrt(df / (1 - r^2))` follows a Student's t-distribution with `df`
//! degrees of freedom, so the two-sided p-value is
//!
//! ```text
//! p = P(|T| >= |t|) = I_x(df/2, 1/2),   x = df / (df + t^2) = 1 - r^2
//! ```
//!
//! where `I_x(a, b)` is the regularized incomplete beta function. Evaluating
//! the p-value through `x = (1 - |r|) * (1 + |r|)` rather than through `t`
//! keeps the whole computation in the well-behaved variable: `t` blows up as
//! `|r| -> 1` while `x` simply approaches 0, `1 - |r|` is exact for `|r|` near
//! 1 (Sterbenz), and `I_x` underflows gracefully to 0.0 in the far tail --
//! the same limit `scipy.stats.pearsonr` reports.
//!
//! The previous implementation instead built `B(df/2, 1/2)` from a Lanczos
//! *gamma* approximation and returned `2 * (1 - CDF(t))`. Both steps break
//! down on large samples: that approximation overflows to `inf` for arguments
//! above ~142, so `Gamma(a) * Gamma(b) / Gamma(a + b)` degenerated into
//! `inf / inf = NaN` and *every* p-value for `n >= 286` came back `NaN`
//! (cool-japan/scirs#131), while the `1 - CDF` cancellation flushed every
//! p-value below ~1e-16 to exactly 0.
//!
//! `I_x` itself is delegated to `statrs::function::beta::beta_reg`, the same
//! routine that backs `distributions::student_t::StudentT::cdf`; it uses a
//! log-gamma prefactor with a modified Lentz continued fraction and the
//! standard symmetry switch, so it stays finite for arbitrarily large `df`.

use scirs2_core::numeric::{Float, NumCast};
use statrs::function::beta::beta_reg;

/// One-sided tail probability `P(T >= |t|)` of the correlation t-statistic.
///
/// This is `0.5 * I_x(df/2, 1/2)` with `x = 1 - r^2`, i.e. the probability of
/// observing a correlation at least as extreme as `r` in the direction of
/// `r`'s sign. The result is always in `[0, 0.5]`; it is 0 for a perfect
/// correlation (`|r| = 1`) and 0.5 for `r = 0`.
///
/// `NaN` inputs (which reach here when the input data itself contains `NaN`)
/// propagate as `NaN`; every other input yields a finite probability.
pub(super) fn tail_probability<F: Float + NumCast>(r: F, df: F) -> F {
    let r_f64 = match <f64 as NumCast>::from(r) {
        Some(value) => value,
        None => return F::nan(),
    };
    let df_f64 = match <f64 as NumCast>::from(df) {
        Some(value) => value,
        None => return F::nan(),
    };

    F::from(tail_probability_f64(r_f64, df_f64)).unwrap_or_else(F::nan)
}

/// Two-sided p-value `P(|T| >= |t|) = I_x(df/2, 1/2)` for a correlation `r`.
///
/// Equal to twice [`tail_probability`], clamped to `[0, 1]`.
pub(super) fn two_sided<F: Float + NumCast>(r: F, df: F) -> F {
    let two = F::from(2.0).unwrap_or_else(|| F::one() + F::one());
    let p = two * tail_probability(r, df);
    if p > F::one() {
        F::one()
    } else {
        p
    }
}

/// `f64` kernel of [`tail_probability`].
///
/// The guards are not decoration: `beta_reg` panics on arguments outside its
/// domain (`a > 0`, `b > 0`, `0 <= x <= 1`), so `NaN` and degenerate degrees
/// of freedom are filtered out before the call.
fn tail_probability_f64(r: f64, df: f64) -> f64 {
    if r.is_nan() || df.is_nan() {
        return f64::NAN;
    }

    // No degrees of freedom left: the correlation carries no information
    // about H0, so the two-sided p-value is 1.
    if df <= 0.0 {
        return 0.5;
    }

    let r_abs = r.abs();

    // A perfect correlation drives t to infinity; the tail probability is
    // exactly 0 in the limit (scipy reports 0.0 here as well).
    if r_abs >= 1.0 {
        return 0.0;
    }

    // x = 1 - r^2, factored to avoid cancellation when |r| is close to 1.
    let x = ((1.0 - r_abs) * (1.0 + r_abs)).clamp(0.0, 1.0);

    let regularized = beta_reg(df / 2.0, 0.5, x).clamp(0.0, 1.0);
    0.5 * regularized
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative-error assertion: `approx`'s absolute epsilon would make the
    /// deep-tail comparisons below vacuous (every value under 1e-16 is within
    /// `f64::EPSILON` of every other), so compare relative error directly.
    fn assert_close(got: f64, want: f64, max_relative: f64) {
        let relative = ((got - want) / want).abs();
        assert!(
            relative <= max_relative,
            "got {got:e}, want {want:e}, relative error {relative:e} > {max_relative:e}"
        );
    }

    /// Two-sided p-values for (r, df) pairs. Reference values computed
    /// independently with mpmath at 60 digits as `I_{1-r^2}(df/2, 1/2)`, NOT
    /// derived from this crate. The moderate ones agree with
    /// `scipy.stats.pearsonr` to all printed digits.
    #[test]
    fn test_two_sided_matches_high_precision_references() {
        let cases: &[(f64, f64, f64)] = &[
            (0.3, 48.0, 0.034286180032929973),
            (0.5, 18.0, 0.024769558804109693),
            (0.9, 3.0, 0.037386073468498633),
            (0.02, 9998.0, 0.045505662655076112),
            (0.2, 343.0, 0.00018460673316322536),
            (0.3, 498.0, 7.3945863308112839e-12),
            (0.3, 998.0, 3.0374833803511221e-22),
            (0.5, 284.0, 1.6989683475327041e-19),
            (0.99, 48.0, 1.7207592265010956e-42),
            (0.7, 343.0, 4.323775024028807e-52),
            (0.1, 9998.0, 1.1970504236520487e-23),
            (0.05, 99998.0, 2.2261912329063711e-56),
            (0.9, 598.0, 8.0597665526349417e-218),
        ];

        for &(r, df, expected) in cases {
            assert_close(two_sided(r, df), expected, 1e-9);
            // The two-sided p-value depends on |r| only.
            assert_close(two_sided(-r, df), expected, 1e-9);
            // ... and the one-sided tail is exactly half of it.
            assert_close(tail_probability(r, df), expected / 2.0, 1e-9);
        }
    }

    /// The regression this module exists for: the p-value must stay finite
    /// and inside [0, 1] for every sample size and every correlation, in
    /// particular past the `n >= 286` cliff where the old Lanczos-gamma beta
    /// function produced `inf` and `NaN`.
    #[test]
    fn test_pvalue_is_always_a_probability() {
        let dfs = [
            1.0_f64, 2.0, 3.0, 8.0, 48.0, 283.0, 284.0, 343.0, 598.0, 9998.0, 999_998.0,
        ];
        for &df in &dfs {
            for step in -2000..=2000 {
                let r = step as f64 / 2000.0;
                let p = two_sided(r, df);
                assert!(
                    p.is_finite() && (0.0..=1.0).contains(&p),
                    "two_sided({r}, {df}) = {p} is not a probability"
                );
                let tail = tail_probability(r, df);
                assert!(
                    tail.is_finite() && (0.0..=0.5).contains(&tail),
                    "tail_probability({r}, {df}) = {tail} is out of range"
                );
            }
        }
    }

    #[test]
    fn test_boundary_values() {
        for &df in &[1.0_f64, 48.0, 598.0, 99998.0] {
            // No correlation: the observation is the least extreme possible.
            assert_close(two_sided(0.0, df), 1.0, 1e-12);
            // Perfect correlation: zero probability of anything more extreme.
            assert_eq!(two_sided(1.0, df), 0.0);
            assert_eq!(two_sided(-1.0, df), 0.0);
            assert_eq!(tail_probability(1.0, df), 0.0);
        }
        // Degenerate degrees of freedom must not reach `beta_reg`.
        assert_eq!(two_sided(0.9, 0.0), 1.0);
        // NaN propagates instead of panicking inside `beta_reg`.
        assert!(two_sided(f64::NAN, 48.0).is_nan());
        assert!(tail_probability(0.5, f64::NAN).is_nan());
    }

    #[test]
    fn test_monotonically_decreasing_in_correlation() {
        for &df in &[3.0_f64, 48.0, 598.0, 9998.0] {
            let mut previous = two_sided(0.0, df);
            for step in 1..=100 {
                let r = step as f64 / 100.0;
                let p = two_sided(r, df);
                assert!(
                    p <= previous,
                    "p({r}, {df}) = {p} should not exceed p of a weaker correlation ({previous})"
                );
                previous = p;
            }
        }
    }

    /// Deep tails must underflow to 0.0 rather than to `NaN`: this is the
    /// exact configuration reported in cool-japan/scirs#131.
    #[test]
    fn test_deep_tail_underflows_to_zero() {
        let p = two_sided(0.9999998334444583_f64, 598.0);
        assert!(p.is_finite(), "expected a finite p-value, got {p}");
        assert_eq!(p, 0.0);
    }

    /// `f32` inputs go through the same `f64` kernel.
    #[test]
    fn test_f32_correlations() {
        let p = two_sided(0.3_f32, 48.0_f32);
        assert!((p - 0.034_286_18_f32).abs() < 1e-6, "got {p}");
        assert!(two_sided(0.9_f32, 598.0_f32).is_finite());
    }
}
