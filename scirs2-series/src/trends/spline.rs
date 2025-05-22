//! Spline-based trend estimation methods
//!
//! This module provides functions for estimating trends using various types of splines,
//! including cubic splines, natural cubic splines, B-splines, and P-splines.

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::{
    ConfidenceIntervalOptions, KnotPlacementStrategy, SplineTrendOptions, SplineType,
    TrendWithConfidenceInterval,
};
use crate::error::{Result, TimeSeriesError};

/// Estimates a trend using spline methods
///
/// This function fits a spline to time series data to estimate the underlying trend.
/// Various spline types are supported, including cubic splines, natural cubic splines,
/// B-splines, and P-splines (penalized B-splines).
///
/// # Arguments
///
/// * `ts` - The input time series data
/// * `options` - Options controlling the spline trend estimation
///
/// # Returns
///
/// The estimated trend as a time series with the same length as the input
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_series::trends::{estimate_spline_trend, SplineTrendOptions, SplineType, KnotPlacementStrategy};
///
/// // Create a sample time series with a trend and noise
/// let n = 100;
/// let mut ts = Array1::from_vec((0..n).map(|t| (t as f64 / 10.0).sin() + t as f64 / 50.0).collect());
///
/// // Configure spline options
/// let options = SplineTrendOptions {
///     spline_type: SplineType::Cubic,
///     num_knots: 10,
///     knot_placement: KnotPlacementStrategy::Uniform,
///     ..Default::default()
/// };
///
/// // Estimate trend
/// let trend = estimate_spline_trend(&ts, &options).unwrap();
///
/// // The trend should have the same length as the input
/// assert_eq!(trend.len(), ts.len());
/// ```
pub fn estimate_spline_trend<F>(ts: &Array1<F>, options: &SplineTrendOptions) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 4 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Time series too short for spline trend estimation".to_string(),
            required: 4,
            actual: ts.len(),
        });
    }

    // Generate knot positions based on the selected strategy
    let knots = match options.knot_placement {
        KnotPlacementStrategy::Uniform => generate_uniform_knots(ts.len(), options.num_knots),
        KnotPlacementStrategy::Quantile => generate_quantile_knots(ts.len(), options.num_knots),
        KnotPlacementStrategy::Custom => {
            if let Some(positions) = &options.knot_positions {
                // Validate custom knot positions
                for &pos in positions {
                    if pos >= ts.len() {
                        return Err(TimeSeriesError::InvalidInput(format!(
                            "Custom knot position {} is out of bounds (time series length: {})",
                            pos,
                            ts.len()
                        )));
                    }
                }
                positions.clone()
            } else {
                return Err(TimeSeriesError::InvalidInput(
                    "Custom knot placement selected but no knot positions provided".to_string(),
                ));
            }
        }
    };

    // Check that we have enough knots
    if knots.len() < 2 {
        return Err(TimeSeriesError::InvalidInput(
            "At least 2 knots are required for spline trend estimation".to_string(),
        ));
    }

    // Fit the appropriate spline type
    match options.spline_type {
        SplineType::Cubic => fit_cubic_spline(ts, &knots, options.extrapolate),
        SplineType::NaturalCubic => fit_natural_cubic_spline(ts, &knots, options.extrapolate),
        SplineType::BSpline => fit_bspline(ts, &knots, options.degree, options.extrapolate),
        SplineType::PSpline => fit_pspline(
            ts,
            &knots,
            options.degree,
            F::from_f64(options.lambda).unwrap(),
            options.extrapolate,
        ),
    }
}

/// Generates uniformly spaced knot positions
fn generate_uniform_knots(n: usize, num_knots: usize) -> Vec<usize> {
    let mut knots = Vec::with_capacity(num_knots);

    // Ensure first and last points are included
    knots.push(0);

    if num_knots > 2 {
        let step = (n - 1) as f64 / (num_knots - 1) as f64;
        for i in 1..(num_knots - 1) {
            let pos = (i as f64 * step).round() as usize;
            knots.push(pos);
        }
    }

    knots.push(n - 1);
    knots
}

/// Generates knots at quantile positions of the data
fn generate_quantile_knots(n: usize, num_knots: usize) -> Vec<usize> {
    let mut knots = Vec::with_capacity(num_knots);

    // Ensure first and last points are included
    knots.push(0);

    if num_knots > 2 {
        for i in 1..(num_knots - 1) {
            let quantile = i as f64 / (num_knots - 1) as f64;
            let pos = (quantile * (n - 1) as f64).round() as usize;
            knots.push(pos);
        }
    }

    knots.push(n - 1);
    knots
}

/// Fits a cubic spline to the time series data
fn fit_cubic_spline<F>(ts: &Array1<F>, knots: &[usize], extrapolate: bool) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if knots.len() < 2 {
        return Err(TimeSeriesError::InvalidInput(
            "At least 2 knots are required for cubic spline".to_string(),
        ));
    }

    let n = ts.len();
    let mut result = Array1::<F>::zeros(n);

    // For cubic spline, we solve a tridiagonal system to find the second derivatives
    let num_knots = knots.len();
    let mut a = Array2::<F>::zeros((num_knots, num_knots));
    let mut b = Vec::with_capacity(num_knots);

    // Set up the tridiagonal system
    for i in 1..(num_knots - 1) {
        let h_i_minus_1 = F::from_usize(knots[i] - knots[i - 1]).unwrap();
        let h_i = F::from_usize(knots[i + 1] - knots[i]).unwrap();

        // Diagonal elements
        a[[i, i - 1]] = h_i_minus_1 / F::from_f64(6.0).unwrap();
        a[[i, i]] = (h_i_minus_1 + h_i) / F::from_f64(3.0).unwrap();
        a[[i, i + 1]] = h_i / F::from_f64(6.0).unwrap();

        // Right-hand side
        let f_i_minus_1 = ts[knots[i - 1]];
        let f_i = ts[knots[i]];
        let f_i_plus_1 = ts[knots[i + 1]];

        let rhs = (f_i_plus_1 - f_i) / h_i - (f_i - f_i_minus_1) / h_i_minus_1;
        b.push(rhs);
    }

    // Add boundary conditions for natural spline
    a[[0, 0]] = F::one();
    b.insert(0, F::zero());

    a[[num_knots - 1, num_knots - 1]] = F::one();
    b.push(F::zero());

    // Solve the system to get second derivatives at knots
    let second_derivs = solve_linear_system(a, b)?;

    // Interpolate between knots
    for i in 0..(num_knots - 1) {
        let x_left = knots[i];
        let x_right = knots[i + 1];
        let h = F::from_usize(x_right - x_left).unwrap();

        let y_left = ts[x_left];
        let y_right = ts[x_right];

        let d2_left = second_derivs[i];
        let d2_right = second_derivs[i + 1];

        // Apply cubic spline formula for each point in this interval
        for x in x_left..=x_right {
            let t = F::from_usize(x - x_left).unwrap() / h;
            let one_minus_t = F::one() - t;

            result[x] = one_minus_t * y_left
                + t * y_right
                + ((one_minus_t.powi(3) - one_minus_t) * d2_left + (t.powi(3) - t) * d2_right)
                    * h
                    * h
                    / F::from_f64(6.0).unwrap();
        }
    }

    // Extrapolate if requested
    if extrapolate {
        // Linear extrapolation using the slope at the endpoints
        // Left endpoint
        let x_0 = knots[0];
        let x_1 = knots[1];
        let h = F::from_usize(x_1 - x_0).unwrap();
        let _slope_left = (ts[x_1] - ts[x_0]) / h
            - h * (F::from_f64(2.0).unwrap() * second_derivs[0] + second_derivs[1])
                / F::from_f64(6.0).unwrap();

        // Right endpoint
        let x_n_minus_1 = knots[num_knots - 2];
        let x_n = knots[num_knots - 1];
        let h = F::from_usize(x_n - x_n_minus_1).unwrap();
        let _slope_right = (ts[x_n] - ts[x_n_minus_1]) / h
            + h * (second_derivs[num_knots - 2]
                + F::from_f64(2.0).unwrap() * second_derivs[num_knots - 1])
                / F::from_f64(6.0).unwrap();

        // No need to extrapolate in this implementation as we already include all data points
    }

    Ok(result)
}

/// Fits a natural cubic spline to the time series data
fn fit_natural_cubic_spline<F>(
    ts: &Array1<F>,
    knots: &[usize],
    _extrapolate: bool,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    // Natural cubic spline is the same as cubic spline but with second derivatives
    // at the endpoints set to zero, which is already handled in fit_cubic_spline
    fit_cubic_spline(ts, knots, _extrapolate)
}

/// Fits a B-spline to the time series data
fn fit_bspline<F>(
    ts: &Array1<F>,
    knots: &[usize],
    degree: usize,
    _extrapolate: bool,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();

    // Convert knot indices to float values
    let x_values: Vec<F> = (0..n).map(|i| F::from_usize(i).unwrap()).collect();

    // Create B-spline basis
    let basis = create_bspline_basis(x_values, knots, degree)?;

    // Solve for coefficients
    let coeffs = solve_spline_system(&basis, ts)?;

    // Reconstruct the trend
    let mut trend = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut val = F::zero();
        for j in 0..coeffs.len() {
            if j < basis.shape()[1] {
                val = val + coeffs[j] * basis[[i, j]];
            }
        }
        trend[i] = val;
    }

    Ok(trend)
}

/// Fits a P-spline (penalized B-spline) to the time series data
fn fit_pspline<F>(
    ts: &Array1<F>,
    knots: &[usize],
    degree: usize,
    lambda: F,
    _extrapolate: bool,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();

    // Convert knot indices to float values
    let x_values: Vec<F> = (0..n).map(|i| F::from_usize(i).unwrap()).collect();

    // Create B-spline basis
    let basis = create_bspline_basis(x_values, knots, degree)?;

    // Solve for coefficients with regularization
    let coeffs = solve_regularized_system(basis.clone(), ts, lambda)?;

    // Reconstruct the trend
    let mut trend = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut val = F::zero();
        for j in 0..coeffs.len() {
            if j < basis.shape()[1] {
                val = val + coeffs[j] * basis[[i, j]];
            }
        }
        trend[i] = val;
    }

    Ok(trend)
}

/// Creates a B-spline basis matrix
fn create_bspline_basis<F>(x_values: Vec<F>, knots: &[usize], degree: usize) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = x_values.len();

    // For a B-spline of degree p with n knots, we have n+p-1 basis functions
    let num_basis = knots.len() + degree - 1;
    let mut basis = Array2::<F>::zeros((n, num_basis));

    // Convert knot indices to float x values
    let mut augmented_knots = Vec::with_capacity(knots.len() + 2 * degree);

    // Add p knots at the beginning and end for proper boundary behavior
    for _ in 0..degree {
        augmented_knots.push(F::from_usize(knots[0]).unwrap());
    }

    for &k in knots {
        augmented_knots.push(F::from_usize(k).unwrap());
    }

    for _ in 0..degree {
        augmented_knots.push(F::from_usize(knots[knots.len() - 1]).unwrap());
    }

    // Implement the Cox-de Boor recursion formula
    // Start with degree 0 (piecewise constant)
    for i in 0..n {
        let x = x_values[i];

        for j in 0..(augmented_knots.len() - 1) {
            if j < num_basis {
                if x >= augmented_knots[j] && x < augmented_knots[j + 1]
                    || (x == augmented_knots[augmented_knots.len() - 1]
                        && j == augmented_knots.len() - 2)
                {
                    basis[[i, j]] = F::one();
                } else {
                    basis[[i, j]] = F::zero();
                }
            }
        }
    }

    // Recursively compute basis functions of higher degrees
    let mut temp_basis = basis.clone();

    for d in 1..=degree {
        for i in 0..n {
            let x = x_values[i];

            for j in 0..(num_basis - d) {
                let mut sum = F::zero();

                // First term
                let denom1 = augmented_knots[j + d] - augmented_knots[j];
                if denom1 > F::zero() {
                    sum = sum + (x - augmented_knots[j]) / denom1 * temp_basis[[i, j]];
                }

                // Second term
                let denom2 = augmented_knots[j + d + 1] - augmented_knots[j + 1];
                if denom2 > F::zero() {
                    sum = sum + (augmented_knots[j + d + 1] - x) / denom2 * temp_basis[[i, j + 1]];
                }

                basis[[i, j]] = sum;
            }
        }

        temp_basis = basis.clone();
    }

    Ok(basis)
}

/// Solves a linear system for spline coefficients
fn solve_spline_system<F>(basis: &Array2<F>, ts: &Array1<F>) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let p = basis.shape()[1];

    // Compute B'B
    let mut btb = Array2::<F>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            let mut sum = F::zero();
            for k in 0..n {
                sum = sum + basis[[k, i]] * basis[[k, j]];
            }
            btb[[i, j]] = sum;
        }
    }

    // Compute B'y
    let mut bty = Vec::with_capacity(p);
    for i in 0..p {
        let mut sum = F::zero();
        for k in 0..n {
            sum = sum + basis[[k, i]] * ts[k];
        }
        bty.push(sum);
    }

    // Solve the system
    solve_linear_system(btb, bty)
}

/// Solves a regularized linear system for penalized spline coefficients
fn solve_regularized_system<F>(basis: Array2<F>, ts: &Array1<F>, lambda: F) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let p = basis.shape()[1];

    // Compute B'B
    let mut btb = Array2::<F>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            let mut sum = F::zero();
            for k in 0..n {
                sum = sum + basis[[k, i]] * basis[[k, j]];
            }
            btb[[i, j]] = sum;
        }
    }

    // Create penalty matrix (second-order differences)
    let mut penalty = Array2::<F>::zeros((p, p));
    for i in 0..(p - 2) {
        penalty[[i, i]] = F::one();
        penalty[[i, i + 1]] = F::from_f64(-2.0).unwrap();
        penalty[[i, i + 2]] = F::one();
    }

    // Compute D'D
    let mut dtd = Array2::<F>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            let mut sum = F::zero();
            for k in 0..(p - 2) {
                sum = sum + penalty[[k, i]] * penalty[[k, j]];
            }
            dtd[[i, j]] = sum;
        }
    }

    // Add penalty term to normal equations: (B'B + λD'D)β = B'y
    for i in 0..p {
        for j in 0..p {
            btb[[i, j]] = btb[[i, j]] + lambda * dtd[[i, j]];
        }
    }

    // Compute B'y
    let mut bty = Vec::with_capacity(p);
    for i in 0..p {
        let mut sum = F::zero();
        for k in 0..n {
            sum = sum + basis[[k, i]] * ts[k];
        }
        bty.push(sum);
    }

    // Solve the system
    solve_linear_system(btb, bty)
}

/// Solves a linear system using Cholesky decomposition
fn solve_linear_system<F>(a: Array2<F>, b: Vec<F>) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = a.shape()[0];
    if n != b.len() {
        return Err(TimeSeriesError::InvalidInput(format!(
            "Matrix and vector dimensions do not match: A is {}x{}, b is {}",
            n,
            a.shape()[1],
            b.len()
        )));
    }

    // Cholesky decomposition: A = LL^T
    let mut l = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = F::zero();

            if i == j {
                for k in 0..j {
                    sum = sum + l[[j, k]] * l[[j, k]];
                }
                let val = (a[[j, j]] - sum).sqrt();
                if !val.is_finite() {
                    return Err(TimeSeriesError::NumericalInstability(
                        "Cholesky decomposition failed due to non-positive definite matrix"
                            .to_string(),
                    ));
                }
                l[[j, j]] = val;
            } else {
                for k in 0..j {
                    sum = sum + l[[i, k]] * l[[j, k]];
                }
                if l[[j, j]] > F::zero() {
                    l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
                } else {
                    return Err(TimeSeriesError::NumericalInstability(
                        "Cholesky decomposition failed due to zero pivot".to_string(),
                    ));
                }
            }
        }
    }

    // Forward substitution to solve Ly = b
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let mut sum = F::zero();
        for j in 0..i {
            sum = sum + l[[i, j]] * y[j];
        }
        y.push((b[i] - sum) / l[[i, i]]);
    }

    // Backward substitution to solve L^Tx = y
    let mut x = vec![F::zero(); n];
    for i in (0..n).rev() {
        let mut sum = F::zero();
        for j in (i + 1)..n {
            sum = sum + l[[j, i]] * x[j];
        }
        x[i] = (y[i] - sum) / l[[i, i]];
    }

    Ok(x)
}

/// Estimates a trend using spline methods with confidence intervals
///
/// This function is a wrapper around `estimate_spline_trend` that also computes
/// confidence intervals for the estimated trend.
///
/// # Arguments
///
/// * `ts` - The input time series data
/// * `options` - Options controlling the spline trend estimation
/// * `ci_options` - Options controlling the confidence interval calculation
///
/// # Returns
///
/// A `TrendWithConfidenceInterval` struct containing the estimated trend and confidence bounds
pub fn estimate_spline_trend_with_ci<F>(
    ts: &Array1<F>,
    options: &SplineTrendOptions,
    ci_options: &ConfidenceIntervalOptions,
) -> Result<TrendWithConfidenceInterval<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    // First, compute the main trend estimate
    let trend = estimate_spline_trend(ts, options)?;

    // Then compute confidence intervals
    let (lower, upper) =
        super::confidence::compute_trend_confidence_interval(ts, &trend, ci_options, |data| {
            estimate_spline_trend(data, options)
        })?;

    Ok(TrendWithConfidenceInterval {
        trend,
        lower,
        upper,
    })
}
