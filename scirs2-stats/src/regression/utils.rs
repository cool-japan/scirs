//! Utilities for regression analysis

use crate::error::StatsResult;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;
use scirs2_linalg::inv;

/// Helper functions for working with Float trait to avoid method ambiguity
/// Returns the absolute value of a Float
#[inline]
pub(crate) fn float_abs<F>(x: F) -> F
where
    F: Float + 'static + std::fmt::Display,
{
    scirs2_core::numeric::Float::abs(x)
}

/// Returns the maximum of two Float values
#[inline]
pub(crate) fn _float_max<F>(a: F, b: F) -> F
where
    F: Float + 'static + std::fmt::Display,
{
    scirs2_core::numeric::Float::max(a, b)
}

/// Returns the minimum of two Float values
#[inline]
pub(crate) fn _float_min<F>(a: F, b: F) -> F
where
    F: Float + 'static + std::fmt::Display,
{
    scirs2_core::numeric::Float::min(a, b)
}

/// Returns the natural logarithm of a Float
#[inline]
pub(crate) fn float_ln<F>(x: F) -> F
where
    F: Float + 'static + std::fmt::Display,
{
    scirs2_core::numeric::Float::ln(x)
}

/// Raises a Float to an integer power
#[inline]
pub(crate) fn float_powi<F>(x: F, n: i32) -> F
where
    F: Float + 'static + std::fmt::Display,
{
    scirs2_core::numeric::Float::powi(x, n)
}

/// Returns the square root of a Float
#[inline]
pub(crate) fn float_sqrt<F>(x: F) -> F
where
    F: Float + 'static + std::fmt::Display,
{
    scirs2_core::numeric::Float::sqrt(x)
}

/// Calculate the standard errors for regression coefficients
pub(crate) fn calculate_std_errors<F>(
    x: &ArrayView2<F>,
    residuals: &ArrayView1<F>,
    df: usize,
) -> StatsResult<Array1<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + std::fmt::Debug
        + 'static
        + scirs2_core::numeric::NumAssign
        + scirs2_core::numeric::One
        + scirs2_core::ndarray::ScalarOperand
        + std::fmt::Display
        + Send
        + Sync,
{
    // Calculate the mean squared error of the residuals
    let mse = residuals
        .iter()
        .map(|&r| scirs2_core::numeric::Float::powi(r, 2))
        .sum::<F>()
        / F::from(df).unwrap();

    // Calculate X'X
    let xtx = x.t().dot(x);

    // Invert X'X to get (X'X)^-1
    let xtx_inv = match inv(&xtx.view(), None) {
        Ok(inv_result) => inv_result,
        Err(_) => {
            // If inversion fails, return zeros for standard errors
            return Ok(Array1::<F>::zeros(x.ncols()));
        }
    };

    // The diagonal elements of (X'X)^-1 * MSE are the variances of the coefficients
    let std_errors = xtx_inv
        .diag()
        .mapv(|v| scirs2_core::numeric::Float::sqrt(v * mse));

    Ok(std_errors)
}

/// Calculate t-values for regression coefficients
pub(crate) fn calculate_t_values<F>(_coefficients: &Array1<F>, stderrors: &Array1<F>) -> Array1<F>
where
    F: Float + 'static + std::fmt::Display,
{
    // Calculate t-values for each coefficient
    _coefficients
        .iter()
        .zip(stderrors.iter())
        .map(|(&coef, &se)| {
            if se < F::epsilon() {
                F::from(1e10).unwrap() // Large t-value for small standard error
            } else {
                coef / se
            }
        })
        .collect::<Array1<F>>()
}

/// Find repeated elements in an array and return their indices
pub(crate) fn find_repeats<F>(x: &ArrayView1<F>) -> Vec<Vec<usize>>
where
    F: Float + 'static + std::fmt::Display,
{
    let n = x.len();
    let mut sorted_indices: Vec<usize> = (0..n).collect();

    // Sort indices based on values in x
    sorted_indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap_or(std::cmp::Ordering::Equal));

    // Find repeats
    let mut result = Vec::new();
    let mut i = 0;

    while i < n {
        let mut j = i + 1;
        // Find end of repeated sequence
        while j < n && (x[sorted_indices[j]] - x[sorted_indices[i]]).abs() < F::epsilon() {
            j += 1;
        }

        // If we found repeats
        if j - i > 1 {
            let mut indices = Vec::new();
            for &idx in sorted_indices.iter().skip(i).take(j - i) {
                indices.push(idx);
            }
            result.push(indices);
        }

        i = j;
    }

    result
}

/// Compute the median of slopes between all pairs of points
pub(crate) fn compute_median_slope<F>(x: &ArrayView1<F>, y: &ArrayView1<F>) -> F
where
    F: Float + std::iter::Sum<F> + 'static + std::fmt::Display,
{
    let n = x.len();
    let mut slopes = Vec::with_capacity(n * (n - 1) / 2);

    // Calculate slopes between all pairs of points
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[j] - x[i];
            if dx.abs() > F::epsilon() {
                let dy = y[j] - y[i];
                slopes.push(dy / dx);
            }
        }
    }

    // Sort slopes to find median
    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Get median slope
    let mid = slopes.len() / 2;
    if slopes.len() % 2 == 0 && !slopes.is_empty() {
        (slopes[mid - 1] + slopes[mid]) / F::from(2.0).unwrap()
    } else if !slopes.is_empty() {
        slopes[mid]
    } else {
        F::zero() // No valid slopes, return zero
    }
}

/// Compute the normal distribution percent point function (inverse CDF)
pub(crate) fn norm_ppf<F>(p: F) -> F
where
    F: Float + 'static + std::fmt::Display,
{
    // This is a simplified approximation of the normal inverse CDF
    // For more accurate calculations, use the special functions in the stats module

    // Handle edge cases
    let p = p
        .min(F::from(0.9999).unwrap())
        .max(F::from(0.0001).unwrap());

    // Constants for Abramowitz and Stegun formula
    let a = [
        F::from(2.515517).unwrap(),
        F::from(0.802853).unwrap(),
        F::from(0.010328).unwrap(),
    ];

    let b = [
        F::from(1.432788).unwrap(),
        F::from(0.189269).unwrap(),
        F::from(0.001308).unwrap(),
    ];

    // Calculate intermediate values
    let p_adj = if p <= F::from(0.5).unwrap() {
        p
    } else {
        F::one() - p
    };
    let t = scirs2_core::numeric::Float::sqrt(
        -F::from(2.0).unwrap() * scirs2_core::numeric::Float::ln(p_adj),
    );

    // Apply Abramowitz and Stegun approximation
    let v = t
        - (a[0] + a[1] * t + a[2] * scirs2_core::numeric::Float::powi(t, 2))
            / (F::one()
                + b[0] * t
                + b[1] * scirs2_core::numeric::Float::powi(t, 2)
                + b[2] * scirs2_core::numeric::Float::powi(t, 3));

    // Adjust sign for p > 0.5
    if p <= F::from(0.5).unwrap() {
        -v
    } else {
        v
    }
}

/// Compute the median absolute deviation from zero for a set of data
pub(crate) fn median_abs_deviation_from_zero<F>(x: &ArrayView1<F>) -> F
where
    F: Float + 'static + std::fmt::Display,
{
    let abs_x: Vec<F> = x.iter().map(|&val| float_abs(val)).collect();

    // Sort the absolute values
    let mut sorted_abs_x = abs_x.clone();
    sorted_abs_x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute median
    let n = sorted_abs_x.len();
    if n == 0 {
        return F::zero();
    }

    let mid = n / 2;
    if n.is_multiple_of(2) {
        (sorted_abs_x[mid - 1] + sorted_abs_x[mid]) / F::from(2.0).unwrap()
    } else {
        sorted_abs_x[mid]
    }
}

/// Add a column of ones to a matrix for an intercept term
pub(crate) fn add_intercept<F>(x: &ArrayView2<F>) -> Array2<F>
where
    F: Float + 'static + std::fmt::Display,
{
    let n = x.nrows();
    let p = x.ncols();

    let mut x_with_intercept = Array2::zeros((n, p + 1));

    // Fill the first column with 1.0 for the intercept
    for i in 0..n {
        x_with_intercept[[i, 0]] = F::one();
    }

    // Copy the original data to the remaining columns
    for i in 0..n {
        for j in 0..p {
            x_with_intercept[[i, j + 1]] = x[[i, j]];
        }
    }

    x_with_intercept
}

/// Calculate residuals from actual y and predicted y
pub(crate) fn _calculate_residuals<F>(y: &ArrayView1<F>, ypred: &Array1<F>) -> Array1<F>
where
    F: Float + 'static + std::fmt::Display,
{
    y.to_owned() - ypred
}

/// Calculate mean and sum of squares
pub(crate) fn calculate_sum_of_squares<F>(
    y: &ArrayView1<F>,
    residuals: &ArrayView1<F>,
) -> (F, F, F, F)
where
    F: Float + std::iter::Sum<F> + 'static + std::fmt::Display,
{
    let n = y.len();
    let y_mean = y.iter().cloned().sum::<F>() / F::from(n).unwrap();

    // Total sum of squares
    let ss_total = y
        .iter()
        .map(|&yi| scirs2_core::numeric::Float::powi(yi - y_mean, 2))
        .sum::<F>();

    // Residual sum of squares
    let ss_residual = residuals
        .iter()
        .map(|&ri| scirs2_core::numeric::Float::powi(ri, 2))
        .sum::<F>();

    // Explained sum of squares
    let ss_explained = ss_total - ss_residual;

    (y_mean, ss_total, ss_residual, ss_explained)
}
