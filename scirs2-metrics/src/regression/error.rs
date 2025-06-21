//! Error metrics for regression models
//!
//! This module provides functions for calculating error metrics between
//! predicted values and true values in regression models.

use ndarray::{ArrayBase, Data, Dimension};
use num_traits::{Float, FromPrimitive, NumCast};
use std::cmp::Ordering;

use super::check_same_shape;
use crate::error::{MetricsError, Result};

/// Calculates the mean squared error (MSE)
///
/// # Mathematical Formulation
///
/// Mean Squared Error is defined as:
///
/// ```text
/// MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
/// ```
///
/// Where:
/// - n = number of samples
/// - yᵢ = true value for sample i
/// - ŷᵢ = predicted value for sample i
/// - Σ = sum over all samples
///
/// # Properties
///
/// - MSE is always non-negative (≥ 0)
/// - MSE = 0 indicates perfect predictions
/// - MSE penalizes larger errors more heavily due to squaring
/// - Units: squared units of the target variable
/// - Differentiable everywhere (useful for optimization)
///
/// # Interpretation
///
/// MSE measures the average squared difference between predicted and actual values:
/// - Lower MSE indicates better model performance
/// - Sensitive to outliers due to squaring of errors
/// - Large errors contribute disproportionately to the total error
///
/// # Relationship to Other Metrics
///
/// - RMSE = √MSE (same units as target variable)
/// - MAE typically ≤ RMSE, with equality when all errors are equal
/// - MSE is the expected value of squared error in probabilistic terms
///
/// # Use Cases
///
/// MSE is widely used because:
/// - It's differentiable (good for gradient-based optimization)
/// - It heavily penalizes large errors
/// - It's the basis for ordinary least squares regression
/// - It corresponds to Gaussian likelihood in probabilistic models
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The mean squared error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::mean_squared_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let mse = mean_squared_error(&y_true, &y_pred).unwrap();
/// // Expecting: ((3.0-2.5)² + (-0.5-0.0)² + (2.0-2.0)² + (7.0-8.0)²) / 4
/// assert!(mse < 0.38 && mse > 0.37);
/// ```
pub fn mean_squared_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    let n_samples = y_true.len();

    let mut squared_error_sum = F::zero();
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let error = *yt - *yp;
        squared_error_sum = squared_error_sum + error * error;
    }

    Ok(squared_error_sum / NumCast::from(n_samples).unwrap())
}

/// Calculates the root mean squared error (RMSE)
///
/// Root mean squared error is the square root of the mean squared error.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The root mean squared error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::root_mean_squared_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let rmse = root_mean_squared_error(&y_true, &y_pred).unwrap();
/// // RMSE is the square root of MSE
/// assert!(rmse < 0.62 && rmse > 0.61);
/// ```
pub fn root_mean_squared_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    let mse = mean_squared_error(y_true, y_pred)?;
    Ok(mse.sqrt())
}

/// Calculates the mean absolute error (MAE)
///
/// Mean absolute error measures the average absolute difference between
/// the estimated values and the actual value.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The mean absolute error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::mean_absolute_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
/// // Expecting: (|3.0-2.5| + |-0.5-0.0| + |2.0-2.0| + |7.0-8.0|) / 4 = 0.5
/// assert!(mae > 0.499 && mae < 0.501);
/// ```
pub fn mean_absolute_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    let n_samples = y_true.len();

    let mut abs_error_sum = F::zero();
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let error = (*yt - *yp).abs();
        abs_error_sum = abs_error_sum + error;
    }

    Ok(abs_error_sum / NumCast::from(n_samples).unwrap())
}

/// Calculates the mean absolute percentage error (MAPE)
///
/// Mean absolute percentage error expresses the difference between true
/// and predicted values as a percentage of the true values.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The mean absolute percentage error
///
/// # Notes
///
/// MAPE is undefined when true values are zero. This implementation
/// excludes those samples from the calculation.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::mean_absolute_percentage_error;
///
/// let y_true = array![3.0, 0.5, 2.0, 7.0];
/// let y_pred = array![2.7, 0.4, 1.8, 7.7];
///
/// let mape = mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
/// // Example calculation: (|3.0-2.7|/3.0 + |0.5-0.4|/0.5 + |2.0-1.8|/2.0 + |7.0-7.7|/7.0) / 4 * 100
/// assert!(mape < 13.0 && mape > 9.0);
/// ```
pub fn mean_absolute_percentage_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    let mut percentage_error_sum = F::zero();
    let mut valid_samples = 0;

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        if yt.abs() > F::epsilon() {
            let percentage_error = ((*yt - *yp) / *yt).abs();
            percentage_error_sum = percentage_error_sum + percentage_error;
            valid_samples += 1;
        }
    }

    if valid_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "All y_true values are zero. MAPE is undefined.".to_string(),
        ));
    }

    // Multiply by 100 to get percentage
    Ok(percentage_error_sum / NumCast::from(valid_samples).unwrap() * NumCast::from(100).unwrap())
}

/// Calculates the symmetric mean absolute percentage error (SMAPE)
///
/// SMAPE is an alternative to MAPE that handles zero or near-zero values better.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The symmetric mean absolute percentage error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::symmetric_mean_absolute_percentage_error;
///
/// let y_true = array![3.0, 0.01, 2.0, 7.0];
/// let y_pred = array![2.7, 0.0, 1.8, 7.7];
///
/// let smape = symmetric_mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
/// assert!(smape > 0.0);
/// ```
pub fn symmetric_mean_absolute_percentage_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    let mut percentage_error_sum = F::zero();
    let mut valid_samples = 0;

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        // Skip samples where both y_true and y_pred are zero to avoid undefined values
        if yt.abs() > F::epsilon() || yp.abs() > F::epsilon() {
            let percentage_error = ((*yt - *yp).abs()) / (yt.abs() + yp.abs());
            percentage_error_sum = percentage_error_sum + percentage_error;
            valid_samples += 1;
        }
    }

    if valid_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "All values are zero. SMAPE is undefined.".to_string(),
        ));
    }

    // Multiply by 200 to get percentage (SMAPE is typically defined with factor of 2)
    Ok(percentage_error_sum / NumCast::from(valid_samples).unwrap() * NumCast::from(200).unwrap())
}

/// Calculates the maximum error
///
/// Maximum error is the maximum absolute difference between the true and predicted values.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The maximum error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::max_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let me = max_error(&y_true, &y_pred).unwrap();
/// // Maximum of [|3.0-2.5|, |-0.5-0.0|, |2.0-2.0|, |7.0-8.0|]
/// assert_eq!(me, 1.0);
/// ```
pub fn max_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    let mut max_err = F::zero();
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let error = (*yt - *yp).abs();
        if error > max_err {
            max_err = error;
        }
    }

    Ok(max_err)
}

/// Calculates the median absolute error
///
/// Median absolute error is the median of all absolute differences between
/// the true and predicted values. It is robust to outliers.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The median absolute error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::median_absolute_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let medae = median_absolute_error(&y_true, &y_pred).unwrap();
/// // Median of [|3.0-2.5|, |-0.5-0.0|, |2.0-2.0|, |7.0-8.0|] = Median of [0.5, 0.5, 0.0, 1.0]
/// assert_eq!(medae, 0.5);
/// ```
pub fn median_absolute_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    let n_samples = y_true.len();

    // Calculate absolute errors
    let mut abs_errors = Vec::with_capacity(n_samples);
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        abs_errors.push((*yt - *yp).abs());
    }

    // Sort and get median
    abs_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    if n_samples % 2 == 1 {
        // Odd number of samples
        Ok(abs_errors[n_samples / 2])
    } else {
        // Even number of samples
        let mid = n_samples / 2;
        Ok((abs_errors[mid - 1] + abs_errors[mid]) / NumCast::from(2).unwrap())
    }
}

/// Calculates the mean squared logarithmic error (MSLE)
///
/// Mean squared logarithmic error measures the average squared difference
/// between the logarithm of the predicted and true values. This metric penalizes
/// underestimates more than overestimates.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The mean squared logarithmic error
///
/// # Notes
///
/// * This metric cannot be used with negative values
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::mean_squared_log_error;
///
/// let y_true = array![3.0, 5.0, 2.5, 7.0];
/// let y_pred = array![2.5, 5.0, 3.0, 8.0];
///
/// let msle = mean_squared_log_error(&y_true, &y_pred).unwrap();
/// assert!(msle > 0.0);
/// ```
pub fn mean_squared_log_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    let n_samples = y_true.len();

    // Check that all values are non-negative
    for &val in y_true.iter() {
        if val < F::zero() {
            return Err(MetricsError::InvalidInput(
                "y_true contains negative values".to_string(),
            ));
        }
    }

    for &val in y_pred.iter() {
        if val < F::zero() {
            return Err(MetricsError::InvalidInput(
                "y_pred contains negative values".to_string(),
            ));
        }
    }

    let mut squared_log_diff_sum = F::zero();
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        // Add 1 to avoid taking log of 0
        let log_yt = (*yt + F::one()).ln();
        let log_yp = (*yp + F::one()).ln();
        let log_diff = log_yt - log_yp;
        squared_log_diff_sum = squared_log_diff_sum + log_diff * log_diff;
    }

    Ok(squared_log_diff_sum / NumCast::from(n_samples).unwrap())
}

/// Calculates the Huber loss
///
/// Huber loss is less sensitive to outliers than squared error loss.
/// For small errors, it behaves like squared error, and for large errors,
/// it behaves like absolute error.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `delta` - Threshold where the loss changes from squared to linear
///
/// # Returns
///
/// * The Huber loss
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::huber_loss;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
/// let delta = 0.5;
///
/// let loss = huber_loss(&y_true, &y_pred, delta).unwrap();
/// assert!(loss > 0.0);
/// ```
pub fn huber_loss<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    delta: F,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    if delta <= F::zero() {
        return Err(MetricsError::InvalidInput(
            "delta must be positive".to_string(),
        ));
    }

    let n_samples = y_true.len();
    let mut loss_sum = F::zero();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let error = (*yt - *yp).abs();
        if error <= delta {
            // Quadratic part
            loss_sum = loss_sum + F::from(0.5).unwrap() * error * error;
        } else {
            // Linear part
            loss_sum = loss_sum + delta * (error - F::from(0.5).unwrap() * delta);
        }
    }

    Ok(loss_sum / NumCast::from(n_samples).unwrap())
}

/// Calculates the normalized root mean squared error (NRMSE)
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `normalization` - Method used for normalization:
///   * "mean" - RMSE / mean(y_true)
///   * "range" - RMSE / (max(y_true) - min(y_true))
///   * "iqr" - RMSE / interquartile range of y_true
///
/// # Returns
///
/// * The normalized root mean squared error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::normalized_root_mean_squared_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let nrmse_mean = normalized_root_mean_squared_error(&y_true, &y_pred, "mean").unwrap();
/// let nrmse_range = normalized_root_mean_squared_error(&y_true, &y_pred, "range").unwrap();
/// assert!(nrmse_mean > 0.0);
/// assert!(nrmse_range > 0.0);
/// ```
pub fn normalized_root_mean_squared_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    normalization: &str,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    let rmse = root_mean_squared_error(y_true, y_pred)?;

    match normalization {
        "mean" => {
            // RMSE / mean(y_true)
            let mean = y_true.iter().fold(F::zero(), |acc, &y| acc + y)
                / NumCast::from(y_true.len()).unwrap();
            if mean.abs() < F::epsilon() {
                return Err(MetricsError::InvalidInput(
                    "Mean of y_true is zero, cannot normalize by mean".to_string(),
                ));
            }
            Ok(rmse / mean.abs())
        }
        "range" => {
            // RMSE / (max(y_true) - min(y_true))
            let max = y_true
                .iter()
                .fold(F::neg_infinity(), |acc, &y| if y > acc { y } else { acc });
            let min = y_true
                .iter()
                .fold(F::infinity(), |acc, &y| if y < acc { y } else { acc });
            let range = max - min;
            if range < F::epsilon() {
                return Err(MetricsError::InvalidInput(
                    "Range of y_true is zero, cannot normalize by range".to_string(),
                ));
            }
            Ok(rmse / range)
        }
        "iqr" => {
            // RMSE / interquartile range of y_true
            let mut values: Vec<F> = y_true.iter().cloned().collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            let n = values.len();
            let q1_idx = n / 4;
            let q3_idx = 3 * n / 4;

            let q1 = if n % 4 == 0 {
                (values[q1_idx - 1] + values[q1_idx]) / NumCast::from(2).unwrap()
            } else {
                values[q1_idx]
            };

            let q3 = if n % 4 == 0 {
                (values[q3_idx - 1] + values[q3_idx]) / NumCast::from(2).unwrap()
            } else {
                values[q3_idx]
            };

            let iqr = q3 - q1;
            if iqr < F::epsilon() {
                return Err(MetricsError::InvalidInput(
                    "Interquartile range of y_true is zero, cannot normalize by IQR".to_string(),
                ));
            }
            Ok(rmse / iqr)
        }
        _ => Err(MetricsError::InvalidInput(format!(
            "Unknown normalization method: {}. Valid options are 'mean', 'range', 'iqr'.",
            normalization
        ))),
    }
}

/// Calculates the relative absolute error (RAE)
///
/// RAE is the ratio of the sum of absolute errors to the sum of absolute
/// deviations from the mean of the true values.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The relative absolute error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::relative_absolute_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let rae = relative_absolute_error(&y_true, &y_pred).unwrap();
/// assert!(rae > 0.0 && rae < 1.0);
/// ```
pub fn relative_absolute_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    // Calculate mean of y_true
    let y_true_mean =
        y_true.iter().fold(F::zero(), |acc, &y| acc + y) / NumCast::from(y_true.len()).unwrap();

    let mut abs_error_sum = F::zero();
    let mut abs_mean_diff_sum = F::zero();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        abs_error_sum = abs_error_sum + (*yt - *yp).abs();
        abs_mean_diff_sum = abs_mean_diff_sum + (*yt - y_true_mean).abs();
    }

    if abs_mean_diff_sum < F::epsilon() {
        return Err(MetricsError::InvalidInput(
            "Sum of absolute deviations from mean is zero".to_string(),
        ));
    }

    Ok(abs_error_sum / abs_mean_diff_sum)
}

/// Calculates the relative squared error (RSE)
///
/// RSE is the ratio of the sum of squared errors to the sum of squared
/// deviations from the mean of the true values.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The relative squared error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::relative_squared_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let rse = relative_squared_error(&y_true, &y_pred).unwrap();
/// assert!(rse > 0.0 && rse < 1.0);
/// ```
pub fn relative_squared_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    // Calculate mean of y_true
    let y_true_mean =
        y_true.iter().fold(F::zero(), |acc, &y| acc + y) / NumCast::from(y_true.len()).unwrap();

    let mut squared_error_sum = F::zero();
    let mut squared_mean_diff_sum = F::zero();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let error = *yt - *yp;
        squared_error_sum = squared_error_sum + error * error;

        let mean_diff = *yt - y_true_mean;
        squared_mean_diff_sum = squared_mean_diff_sum + mean_diff * mean_diff;
    }

    if squared_mean_diff_sum < F::epsilon() {
        return Err(MetricsError::InvalidInput(
            "Sum of squared deviations from mean is zero".to_string(),
        ));
    }

    Ok(squared_error_sum / squared_mean_diff_sum)
}
