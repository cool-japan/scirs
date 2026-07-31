//! Polynomial regression implementations

use crate::error::{StatsError, StatsResult};
use crate::regression::stat_tests::{f_test_p_value, t_test_p_value};
use crate::regression::utils::*;
use crate::regression::RegressionResults;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Float;
use scirs2_linalg::lstsq;

/// Fit a polynomial of specified degree to data.
///
/// This function fits a polynomial of the form:
/// `p(x) = c[0] + c[1] * x + c[2] * x^2 + ... + c[deg] * x^deg`
///
/// # Arguments
///
/// * `x` - Independent variable data (1-dimensional)
/// * `y` - Dependent variable data (must be same length as x)
/// * `deg` - Degree of the polynomial to fit
///
/// # Returns
///
/// A RegressionResults struct with the polynomial coefficients and fit statistics.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::polyfit;
///
/// let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
/// let y = array![1.0, 3.0, 9.0, 19.0, 33.0];  // y = 1 + 2x + x^2
///
/// let result = polyfit(&x.view(), &y.view(), 2).expect("Operation failed");
///
/// // Check we get the correct number of coefficients (intercept, x, x^2)
/// assert_eq!(result.coefficients.len(), 3);
///
/// // Check that the fit is good (high R^2 value)
/// assert!(result.r_squared > 0.95);
/// ```
#[allow(dead_code)]
pub fn polyfit<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    deg: usize,
) -> StatsResult<RegressionResults<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + std::fmt::Debug
        + std::fmt::Display
        + 'static
        + scirs2_core::numeric::NumAssign
        + scirs2_core::numeric::One
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync,
{
    // Check input dimensions
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "Input x has length {} but y has length {}",
            x.len(),
            y.len()
        )));
    }

    let n = x.len();
    let p = deg + 1; // Number of parameters (coefficients)

    // We need more observations than parameters for inference
    if n <= deg {
        return Err(StatsError::InvalidArgument(format!(
            "Number of data points ({}) must be greater than polynomial degree ({})",
            n, deg
        )));
    }

    // Create the Vandermonde matrix
    let mut vandermonde = Array2::<F>::zeros((n, p));

    // Fill the design matrix
    for i in 0..n {
        vandermonde[[i, 0]] = F::one(); // Constant term

        for j in 1..=deg {
            vandermonde[[i, j]] = scirs2_core::numeric::Float::powi(x[i], j as i32);
        }
    }

    // Solve the least squares problem
    let coefficients = match lstsq(&vandermonde.view(), y, None) {
        Ok(result) => result.x,
        Err(e) => {
            return Err(StatsError::ComputationError(format!(
                "Least squares computation failed: {:?}",
                e
            )));
        }
    };

    // Calculate predicted values
    let fitted_values = vandermonde.dot(&coefficients);

    // Calculate residuals
    let residuals = y.to_owned() - &fitted_values;

    // Calculate degrees of freedom
    let df_model = p - 1;
    let df_residuals = n - p;

    // Calculate sum of squares
    let (_y_mean, ss_total, ss_residual, ss_explained) =
        calculate_sum_of_squares(y, &residuals.view());

    // Calculate R-squared and adjusted R-squared
    let r_squared = ss_explained / ss_total;
    let adj_r_squared = F::one()
        - (F::one() - r_squared) * F::from(n - 1).expect("Operation failed")
            / F::from(df_residuals).expect("Operation failed");

    // Calculate mean squared error and residual standard error
    let mse = ss_residual / F::from(df_residuals).expect("Operation failed");
    let residual_std_error = scirs2_core::numeric::Float::sqrt(mse);

    // Calculate standard errors for coefficients
    let std_errors =
        match calculate_std_errors(&vandermonde.view(), &residuals.view(), df_residuals) {
            Ok(se) => se,
            Err(_) => Array1::<F>::zeros(p),
        };

    // Calculate t-values
    let t_values = calculate_t_values(&coefficients, &std_errors);

    // Calculate real two-sided per-coefficient p-values from the Student's
    // t-distribution (see `stat_tests::t_test_p_value`), matching the fix
    // already applied to `regularized.rs`/`robust.rs`/`stepwise.rs`.
    let p_values = t_values.mapv(|t| t_test_p_value(t, df_residuals));

    // Calculate confidence intervals using a normal-quantile margin (95%),
    // matching the convention used by the sibling regression fits in
    // `robust.rs`/`stepwise.rs` rather than a bare +/- one-std-error band
    // (which corresponds to roughly a 68% interval, not a 95% CI).
    let z = norm_ppf(F::from(0.975).expect("Operation failed"));
    let mut conf_intervals = Array2::<F>::zeros((p, 2));
    for i in 0..p {
        let margin = std_errors[i] * z;
        conf_intervals[[i, 0]] = coefficients[i] - margin;
        conf_intervals[[i, 1]] = coefficients[i] + margin;
    }

    // Calculate F-statistic
    let f_statistic = if df_model > 0 && df_residuals > 0 {
        (ss_explained / F::from(df_model).expect("Operation failed"))
            / (ss_residual / F::from(df_residuals).expect("Operation failed"))
    } else {
        F::infinity()
    };

    // Calculate the real p-value for the overall-model F-statistic using the
    // F(df_model, df_residuals) survival function (see
    // `stat_tests::f_test_p_value`), matching the fix already applied to
    // `regularized.rs`/`robust.rs`/`stepwise.rs`.
    let f_p_value = f_test_p_value(f_statistic, df_model, df_residuals);

    // Create and return the results structure
    Ok(RegressionResults {
        coefficients,
        std_errors,
        t_values,
        p_values,
        conf_intervals,
        r_squared,
        adj_r_squared,
        f_statistic,
        f_p_value,
        residual_std_error,
        df_residuals,
        residuals,
        fitted_values,
        inlier_mask: vec![true; n], // All points are inliers in polynomial regression
    })
}

// ============================================================================
// `f_p_value` / `p_values` fix tests.
//
// Wave-1 finding: `polyfit` hardcoded `f_p_value = F::zero()` and
// `p_values = Array1::zeros(p)` unconditionally, regardless of the actual
// fit quality -- the same silent-fabrication family already fixed in
// `regularized.rs`/`robust.rs`/`stepwise.rs`. Fixed here by reusing the
// shared `stat_tests::f_test_p_value` / `stat_tests::t_test_p_value`
// helpers (real F- and Student's t-distribution survival functions). The
// confidence-interval computation (previously a bare
// `coefficient +/- std_error`, i.e. roughly a 68% interval mislabeled as a
// CI) was fixed at the same time to use a 95% normal-quantile margin,
// matching the convention in `robust.rs`/`stepwise.rs`.
//
// Reference values below were computed independently in Python via
// `numpy.linalg.lstsq` + `scipy.stats.{f,t}.sf`, NOT derived from this
// crate:
//
// ```python
// import numpy as np
// from scipy import stats
// V = np.vander(x, N=deg + 1, increasing=True)
// coeffs, *_ = np.linalg.lstsq(V, y, rcond=None)
// resid = y - V @ coeffs
// ss_total = np.sum((y - y.mean())**2)
// ss_resid = np.sum(resid**2)
// mse = ss_resid / df_residuals
// std_errors = np.sqrt(np.diag(np.linalg.inv(V.T @ V)) * mse)
// t_values = coeffs / std_errors
// p_values = 2 * stats.t.sf(np.abs(t_values), df_residuals)
// f_stat = ((ss_total - ss_resid) / df_model) / mse
// f_p_value = stats.f.sf(f_stat, df_model, df_residuals)
// ```
//
// Some reference p-values below are astronomically small (e.g. ~1e-17 or
// ~1e-23). Because `f_test_p_value`/`t_test_p_value` compute
// `1 - cdf(...)` in `f64`, once `cdf(...)` itself rounds to exactly `1.0`
// (unavoidable once the true tail mass is smaller than `f64`'s ULP near
// 1.0, ~1.1e-16) the subtraction underflows to exactly `0.0` -- the same
// documented behavior already relied on by
// `regularized_tests.rs::f_p_value_fix_tests` and
// `stepwise.rs::f_p_value_fix_tests`. Those cases are asserted as bounds
// (`< 1e-9`) rather than exact scipy matches.
// ============================================================================
#[cfg(test)]
mod f_p_value_and_p_values_fix_tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_polyfit_degree1_strong_signal_f_p_value_and_p_values_match_scipy() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let y = array![2.1, 3.9, 5.8, 8.2, 9.7, 12.3, 13.5, 16.4, 17.9, 20.5];
        let result = polyfit(&x.view(), &y.view(), 1).expect("polyfit should succeed");

        assert_relative_eq!(result.f_statistic, 3157.3741037536925, max_relative = 1e-4);
        assert_relative_eq!(
            result.f_p_value,
            1.1167535971635081e-11,
            max_relative = 1e-3,
            epsilon = 1e-14
        );
        assert!((0.0..=1.0).contains(&result.f_p_value));

        assert_relative_eq!(
            result.p_values[0],
            9.868959309924845e-06,
            max_relative = 1e-3
        );
        assert_relative_eq!(
            result.p_values[1],
            1.1167535971635009e-11,
            max_relative = 1e-3,
            epsilon = 1e-14
        );
        for &p in result.p_values.iter() {
            assert!((0.0..=1.0).contains(&p), "p-value out of range: {p}");
        }
    }

    #[test]
    fn test_polyfit_degree1_flat_data_distinguishes_signal_from_noise() {
        // Effectively flat data (small jitter around a constant ~5.0): the
        // intercept carries real signal (the mean is far from zero) but
        // the slope carries none -- there is no real linear trend.
        let x = array![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0
        ];
        let y =
            array![5.1, 4.8, 5.3, 4.9, 5.2, 5.0, 4.7, 5.4, 5.1, 4.9, 5.0, 5.2, 4.8, 5.3, 5.0, 4.9];
        let result = polyfit(&x.view(), &y.view(), 1).expect("polyfit should succeed");

        assert_relative_eq!(
            result.f_statistic,
            0.016690510252749525,
            max_relative = 1e-3
        );
        assert_relative_eq!(
            result.f_p_value,
            0.8990436185686632,
            max_relative = 1e-3,
            epsilon = 1e-6
        );
        // The bug under test: `f_p_value` was previously ALWAYS exactly
        // 0.0 regardless of data. The true p-value here is large (~0.90:
        // no real linear trend), but the old code always reported 0.0
        // (maximal significance) regardless.
        assert!(
            result.f_p_value > 0.05,
            "expected a large, non-significant p-value, got {}",
            result.f_p_value
        );

        // Reference intercept p-value (~3.14e-17) underflows to exactly
        // 0.0 in this crate's `f64` `1 - cdf` computation (see module
        // doc); assert a bound rather than the exact scipy value.
        assert!(result.p_values[0] < 1e-9);
        assert_relative_eq!(
            result.p_values[1],
            0.8990436185686801,
            max_relative = 1e-3,
            epsilon = 1e-6
        );
        // Old code hardcoded EVERY per-coefficient p-value to exactly 0.0
        // (`Array1::zeros(p)`), i.e. it claimed the slope was maximally
        // significant even though it carries no real signal at all.
        assert!(
            result.p_values[1] > 0.05,
            "expected a non-significant p-value for the (noise-only) slope, got {}",
            result.p_values[1]
        );
        for &p in result.p_values.iter() {
            assert!((0.0..=1.0).contains(&p), "p-value out of range: {p}");
        }
    }

    #[test]
    fn test_polyfit_degree2_f_p_value_and_p_values_match_scipy() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let y = array![1.2, 2.8, 6.5, 11.7, 18.9, 27.6, 38.4, 50.8, 65.3, 80.9, 99.1, 119.0];
        let result = polyfit(&x.view(), &y.view(), 2).expect("polyfit should succeed");

        assert_relative_eq!(result.f_statistic, 362654.7065548911, max_relative = 1e-3);
        // Reference f_p_value (~8.35e-23) underflows to exactly 0.0 (see
        // module doc).
        assert!(result.f_p_value < 1e-9);
        assert!((0.0..=1.0).contains(&result.f_p_value));

        assert_relative_eq!(
            result.p_values[0],
            2.094418202299678e-06,
            max_relative = 1e-3
        );
        assert_relative_eq!(
            result.p_values[1],
            5.6845212254180934e-08,
            max_relative = 1e-3
        );
        // Reference p_values[2] (~5.83e-18) underflows to exactly 0.0.
        assert!(result.p_values[2] < 1e-9);
        for &p in result.p_values.iter() {
            assert!((0.0..=1.0).contains(&p), "p-value out of range: {p}");
        }
    }

    #[test]
    fn test_polyfit_conf_intervals_use_95pct_normal_margin_not_bare_std_error() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let y = array![2.1, 3.9, 5.8, 8.2, 9.7, 12.3, 13.5, 16.4, 17.9, 20.5];
        let result = polyfit(&x.view(), &y.view(), 1).expect("polyfit should succeed");

        // Reference values use the *same* Abramowitz-Stegun normal-quantile
        // approximation this crate's `norm_ppf` implements
        // (z = norm_ppf(0.975) ~= 1.9603949169253396, vs. the exact
        // 1.959963984540054), applied to coefficients/std_errors computed
        // independently via numpy: ci = coefficient +/- z * std_error.
        assert_relative_eq!(
            result.conf_intervals[[0, 0]],
            1.5126464415195235,
            max_relative = 1e-6
        );
        assert_relative_eq!(
            result.conf_intervals[[0, 1]],
            2.2691717402986518,
            max_relative = 1e-6
        );
        assert_relative_eq!(
            result.conf_intervals[[1, 0]],
            1.9600540048151793,
            max_relative = 1e-6
        );
        assert_relative_eq!(
            result.conf_intervals[[1, 1]],
            2.1017641770030058,
            max_relative = 1e-6
        );

        // The bug under test: previously this was `coefficient +/-
        // std_error` (no z-factor at all, i.e. ~68% coverage, not a 95%
        // CI). Verify the half-width is ~1.96x the raw std error, not 1x.
        let half_width_1 = result.conf_intervals[[1, 1]] - result.coefficients[1];
        assert_relative_eq!(
            half_width_1 / result.std_errors[1],
            1.9603949169253396,
            max_relative = 1e-6
        );
    }
}
