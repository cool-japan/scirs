//! Robust trend filtering methods
//!
//! This module provides robust methods for filtering trends in time series data,
//! including Hodrick-Prescott filter, L1 trend filter, and Whittaker smoother.
//! These methods are designed to be robust to outliers and can handle non-stationary data.

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::{
    ConfidenceIntervalOptions, RobustFilterMethod, RobustFilterOptions,
    TrendWithConfidenceInterval, WeightFunction,
};
use crate::error::{Result, TimeSeriesError};

/// Applies robust trend filtering to a time series
///
/// This function estimates the underlying trend in a time series using robust filtering methods
/// that are less sensitive to outliers and can handle non-stationary data.
///
/// # Arguments
///
/// * `ts` - The input time series data
/// * `options` - Options controlling the robust filtering
///
/// # Returns
///
/// The estimated trend as a time series with the same length as the input
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_series::trends::{robust_trend_filter, RobustFilterOptions, RobustFilterMethod, WeightFunction};
///
/// // Create a sample time series with a trend, noise, and outliers
/// let n = 100;
/// let mut ts = Array1::from_vec((0..n).map(|t| (t as f64 / 10.0).sin() + t as f64 / 50.0).collect());
///
/// // Add some outliers
/// ts[20] += 5.0;
/// ts[60] -= 5.0;
///
/// // Configure filter options
/// let options = RobustFilterOptions {
///     method: RobustFilterMethod::HodrickPrescott,
///     lambda: 1600.0,
///     weight_function: WeightFunction::Bisquare,
///     ..Default::default()
/// };
///
/// // Apply robust trend filter
/// let trend = robust_trend_filter(&ts, &options).unwrap();
///
/// // The trend should have the same length as the input
/// assert_eq!(trend.len(), ts.len());
/// ```
pub fn robust_trend_filter<F>(ts: &Array1<F>, options: &RobustFilterOptions) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    if ts.len() < 3 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Time series too short for robust trend filtering (minimum length: 3)"
                .to_string(),
            required: 3,
            actual: ts.len(),
        });
    }

    // Validate options
    if options.lambda <= 0.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Lambda must be positive for robust trend filtering".to_string(),
        ));
    }

    if options.order < 1 {
        return Err(TimeSeriesError::InvalidInput(
            "Difference order must be at least 1 for robust trend filtering".to_string(),
        ));
    }

    // Apply the selected filtering method
    match options.method {
        RobustFilterMethod::HodrickPrescott => robust_hodrick_prescott(ts, options),
        RobustFilterMethod::L1Filter => l1_trend_filter(ts, options),
        RobustFilterMethod::Whittaker => robust_whittaker_smoother(ts, options),
    }
}

/// Calculates robust weights based on residuals
fn calculate_robust_weights<F>(
    residuals: &[F],
    weight_function: WeightFunction,
    tuning_parameter: F,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug,
{
    // Calculate the median absolute deviation (MAD)
    let mut abs_residuals: Vec<F> = residuals.iter().map(|&r| r.abs()).collect();
    abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = abs_residuals.len();
    let median_idx = n / 2;
    let mad = if n % 2 == 0 {
        (abs_residuals[median_idx - 1] + abs_residuals[median_idx]) / F::from_f64(2.0).unwrap()
    } else {
        abs_residuals[median_idx]
    };

    // Use MAD to scale residuals (with consistency factor for normal distribution)
    let mad_scale = mad / F::from_f64(0.6745).unwrap();

    // If MAD is zero, use a small positive value to avoid division by zero
    let scale = if mad_scale > F::from_f64(1e-10).unwrap() {
        mad_scale
    } else {
        F::from_f64(1e-10).unwrap()
    };

    let scaled_residuals: Vec<F> = residuals.iter().map(|&r| r.abs() / scale).collect();

    // Apply the selected weight function
    let weights: Vec<F> = scaled_residuals
        .iter()
        .map(|&u| {
            let scaled_u = u / tuning_parameter;

            match weight_function {
                WeightFunction::Huber => {
                    if scaled_u <= F::one() {
                        F::one()
                    } else {
                        F::one() / scaled_u
                    }
                }
                WeightFunction::Bisquare => {
                    if scaled_u < F::one() {
                        let temp = F::one() - scaled_u * scaled_u;
                        temp * temp
                    } else {
                        F::zero()
                    }
                }
                WeightFunction::Andrews => {
                    if scaled_u < F::from_f64(std::f64::consts::PI).unwrap() {
                        let sin_val = (scaled_u * F::from_f64(std::f64::consts::PI).unwrap()).sin();
                        sin_val / (scaled_u * F::from_f64(std::f64::consts::PI).unwrap())
                    } else {
                        F::zero()
                    }
                }
                WeightFunction::Cauchy => F::one() / (F::one() + scaled_u * scaled_u),
            }
        })
        .collect();

    Ok(weights)
}

/// Implements the robust Hodrick-Prescott filter
fn robust_hodrick_prescott<F>(ts: &Array1<F>, options: &RobustFilterOptions) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = ts.len();
    let lambda = F::from_f64(options.lambda).unwrap();
    let max_iter = options.max_iter;
    let tol = F::from_f64(options.tol).unwrap();
    let tuning_parameter = F::from_f64(options.tuning_parameter).unwrap();

    // Create second-difference matrix (for Hodrick-Prescott)
    let d = super::create_difference_matrix::<F>(n, 2);

    // Initialize weights for the robust estimation
    let mut weights = vec![F::one(); n];

    // Initialize trend estimate with standard HP filter
    let mut trend = Array1::<F>::zeros(n);
    let mut prev_trend = Array1::<F>::zeros(n);

    // Iteratively reweighted least squares
    for _iter in 0..max_iter {
        // Create diagonal weight matrix
        let mut w = Array2::<F>::zeros((n, n));
        for i in 0..n {
            w[[i, i]] = weights[i];
        }

        // Solve the normal equations: (W + λD'D)β = Wy
        let mut lhs = w.clone();
        let dtd = d.t().dot(&d);

        for i in 0..n {
            for j in 0..n {
                if i < dtd.shape()[0] && j < dtd.shape()[1] {
                    lhs[[i, j]] = lhs[[i, j]] + lambda * dtd[[i, j]];
                }
            }
        }

        // Compute right-hand side
        let mut rhs = Array1::<F>::zeros(n);
        for i in 0..n {
            rhs[i] = w[[i, i]] * ts[i];
        }

        // Solve the system using Cholesky decomposition
        let mut l = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let mut sum = F::zero();

                if i == j {
                    for k in 0..j {
                        sum = sum + l[[j, k]] * l[[j, k]];
                    }
                    l[[j, j]] = (lhs[[j, j]] - sum).sqrt();
                } else {
                    for k in 0..j {
                        sum = sum + l[[i, k]] * l[[j, k]];
                    }
                    l[[i, j]] = (lhs[[i, j]] - sum) / l[[j, j]];
                }
            }
        }

        // Forward substitution
        let mut y = Array1::<F>::zeros(n);
        for i in 0..n {
            let mut sum = F::zero();
            for j in 0..i {
                sum = sum + l[[i, j]] * y[j];
            }
            y[i] = (rhs[i] - sum) / l[[i, i]];
        }

        // Backward substitution
        prev_trend.assign(&trend);
        for i in (0..n).rev() {
            let mut sum = F::zero();
            for j in (i + 1)..n {
                sum = sum + l[[j, i]] * trend[j];
            }
            trend[i] = (y[i] - sum) / l[[i, i]];
        }

        // Check convergence
        let mut max_change = F::zero();
        for i in 0..n {
            let change = (trend[i] - prev_trend[i]).abs();
            if change > max_change {
                max_change = change;
            }
        }

        if max_change < tol {
            break;
        }

        // Update weights based on residuals
        let residuals: Vec<F> = (0..n).map(|i| ts[i] - trend[i]).collect();
        weights = calculate_robust_weights(&residuals, options.weight_function, tuning_parameter)?;
    }

    Ok(trend)
}

/// Implements the L1 trend filter
fn l1_trend_filter<F>(ts: &Array1<F>, options: &RobustFilterOptions) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = ts.len();
    let lambda = F::from_f64(options.lambda).unwrap();
    let max_iter = options.max_iter;
    let tol = F::from_f64(options.tol).unwrap();
    let order = options.order;

    // Create difference matrix of specified order
    let d = super::create_difference_matrix::<F>(n, order);
    let m = d.shape()[0];

    // Initialize trend estimate as the original time series
    let mut trend = ts.clone();
    let mut prev_trend = Array1::<F>::zeros(n);

    // Auxiliary variables for ADMM optimization
    let mut z = Array1::<F>::zeros(m);
    let mut u = Array1::<F>::zeros(m);

    // Precompute factorization for the linear system
    let identity = Array2::<F>::eye(n);
    let dtd = d.t().dot(&d);

    let mut lhs = identity.clone();
    for i in 0..n {
        for j in 0..n {
            if i < dtd.shape()[0] && j < dtd.shape()[1] {
                lhs[[i, j]] = lhs[[i, j]] + dtd[[i, j]];
            }
        }
    }

    // Cholesky decomposition of LHS
    let mut l = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = F::zero();

            if i == j {
                for k in 0..j {
                    sum = sum + l[[j, k]] * l[[j, k]];
                }
                l[[j, j]] = (lhs[[j, j]] - sum).sqrt();
            } else {
                for k in 0..j {
                    sum = sum + l[[i, k]] * l[[j, k]];
                }
                l[[i, j]] = (lhs[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    // ADMM optimization
    let rho = F::one(); // Penalty parameter
    for _iter in 0..max_iter {
        // Store previous trend for convergence check
        prev_trend.assign(&trend);

        // x-update: solve (I + D'D)x = y + D'(z - u)
        let dt_zu = d.t().dot(&(z.clone() - u.clone()));
        let rhs = ts + &dt_zu;

        // Forward substitution
        let mut y_temp = Array1::<F>::zeros(n);
        for i in 0..n {
            let mut sum = F::zero();
            for j in 0..i {
                sum = sum + l[[i, j]] * y_temp[j];
            }
            y_temp[i] = (rhs[i] - sum) / l[[i, i]];
        }

        // Backward substitution
        for i in (0..n).rev() {
            let mut sum = F::zero();
            for j in (i + 1)..n {
                sum = sum + l[[j, i]] * trend[j];
            }
            trend[i] = (y_temp[i] - sum) / l[[i, i]];
        }

        // z-update: soft thresholding
        let dx = d.dot(&trend);
        let z_new = dx.clone() + u.clone();

        for i in 0..m {
            let val = z_new[i];
            let abs_val = val.abs();
            let threshold = lambda / rho;

            if abs_val <= threshold {
                z[i] = F::zero();
            } else {
                let sign = if val > F::zero() { F::one() } else { -F::one() };
                z[i] = sign * (abs_val - threshold);
            }
        }

        // u-update: dual update
        u = u + dx - z.clone();

        // Check convergence
        let mut max_change = F::zero();
        for i in 0..n {
            let change = (trend[i] - prev_trend[i]).abs();
            if change > max_change {
                max_change = change;
            }
        }

        if max_change < tol {
            break;
        }
    }

    Ok(trend)
}

/// Implements the robust Whittaker smoother
fn robust_whittaker_smoother<F>(ts: &Array1<F>, options: &RobustFilterOptions) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = ts.len();
    let lambda = F::from_f64(options.lambda).unwrap();
    let max_iter = options.max_iter;
    let tol = F::from_f64(options.tol).unwrap();
    let order = options.order;
    let tuning_parameter = F::from_f64(options.tuning_parameter).unwrap();

    // Create difference matrix of specified order
    let d = super::create_difference_matrix::<F>(n, order);

    // Initialize weights for the robust estimation
    let mut weights = vec![F::one(); n];

    // Initialize trend estimate
    let mut trend = ts.clone();
    let mut prev_trend = Array1::<F>::zeros(n);

    // Iteratively reweighted least squares
    for _iter in 0..max_iter {
        // Create diagonal weight matrix
        let mut w = Array2::<F>::zeros((n, n));
        for i in 0..n {
            w[[i, i]] = weights[i];
        }

        // Solve the normal equations: (W + λD'D)β = Wy
        let dtd = d.t().dot(&d);
        let mut lhs = w.clone();

        for i in 0..n {
            for j in 0..n {
                if i < dtd.shape()[0] && j < dtd.shape()[1] {
                    lhs[[i, j]] = lhs[[i, j]] + lambda * dtd[[i, j]];
                }
            }
        }

        // Compute right-hand side
        let mut rhs = Array1::<F>::zeros(n);
        for i in 0..n {
            rhs[i] = w[[i, i]] * ts[i];
        }

        // Solve the system using Cholesky decomposition
        let mut l = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let mut sum = F::zero();

                if i == j {
                    for k in 0..j {
                        sum = sum + l[[j, k]] * l[[j, k]];
                    }
                    l[[j, j]] = (lhs[[j, j]] - sum).sqrt();
                } else {
                    for k in 0..j {
                        sum = sum + l[[i, k]] * l[[j, k]];
                    }
                    l[[i, j]] = (lhs[[i, j]] - sum) / l[[j, j]];
                }
            }
        }

        // Forward substitution
        let mut y = Array1::<F>::zeros(n);
        for i in 0..n {
            let mut sum = F::zero();
            for j in 0..i {
                sum = sum + l[[i, j]] * y[j];
            }
            y[i] = (rhs[i] - sum) / l[[i, i]];
        }

        // Backward substitution
        prev_trend.assign(&trend);
        for i in (0..n).rev() {
            let mut sum = F::zero();
            for j in (i + 1)..n {
                sum = sum + l[[j, i]] * trend[j];
            }
            trend[i] = (y[i] - sum) / l[[i, i]];
        }

        // Check convergence
        let mut max_change = F::zero();
        for i in 0..n {
            let change = (trend[i] - prev_trend[i]).abs();
            if change > max_change {
                max_change = change;
            }
        }

        if max_change < tol {
            break;
        }

        // Update weights based on residuals
        let residuals: Vec<F> = (0..n).map(|i| ts[i] - trend[i]).collect();
        weights = calculate_robust_weights(&residuals, options.weight_function, tuning_parameter)?;
    }

    Ok(trend)
}

/// Applies robust trend filtering with confidence intervals
///
/// This function is a wrapper around `robust_trend_filter` that also computes
/// confidence intervals for the estimated trend.
///
/// # Arguments
///
/// * `ts` - The input time series data
/// * `options` - Options controlling the robust filtering
/// * `ci_options` - Options controlling the confidence interval calculation
///
/// # Returns
///
/// A `TrendWithConfidenceInterval` struct containing the estimated trend and confidence bounds
pub fn robust_trend_filter_with_ci<F>(
    ts: &Array1<F>,
    options: &RobustFilterOptions,
    ci_options: &ConfidenceIntervalOptions,
) -> Result<TrendWithConfidenceInterval<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    // First, compute the main trend estimate
    let trend = robust_trend_filter(ts, options)?;

    // Then compute confidence intervals
    let (lower, upper) =
        super::confidence::compute_trend_confidence_interval(ts, &trend, ci_options, |data| {
            robust_trend_filter(data, options)
        })?;

    Ok(TrendWithConfidenceInterval {
        trend,
        lower,
        upper,
    })
}
