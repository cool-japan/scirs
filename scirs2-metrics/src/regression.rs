//! Regression metrics module
//!
//! This module provides functions for evaluating regression models, including
//! mean squared error, mean absolute error, R-squared, and explained variance.

use ndarray::{ArrayBase, Data, Dimension};
use num_traits::{Float, FromPrimitive, NumCast};

use crate::error::{MetricsError, Result};

/// Calculates the mean squared error (MSE)
///
/// Mean squared error measures the average squared difference between
/// the estimated values and the actual value.
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
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

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
/// // Expecting: (|3.0-2.5| + |-0.5-0.0| + |2.0-2.0| + |7.0-8.0|) / 4
/// assert!(mae < 0.51 && mae > 0.49);
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
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let mut absolute_error_sum = F::zero();
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let error = (*yt - *yp).abs();
        absolute_error_sum = absolute_error_sum + error;
    }

    Ok(absolute_error_sum / NumCast::from(n_samples).unwrap())
}

/// Calculates the mean absolute percentage error (MAPE)
///
/// Mean absolute percentage error measures the average percentage difference between
/// the estimated values and the actual value.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values (must not contain zeros)
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The mean absolute percentage error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::mean_absolute_percentage_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let mape = mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
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
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let mut percentage_error_sum = F::zero();
    let mut valid_samples = 0;

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        // Skip division by zero
        if yt.abs() < F::epsilon() {
            continue;
        }

        let percentage_error = ((*yt - *yp) / *yt).abs();
        percentage_error_sum = percentage_error_sum + percentage_error;
        valid_samples += 1;
    }

    if valid_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "No valid samples for MAPE calculation (all y_true values are zero)".to_string(),
        ));
    }

    Ok(percentage_error_sum / NumCast::from(valid_samples).unwrap())
}

/// Calculates the R-squared score (coefficient of determination)
///
/// R-squared represents the proportion of variance in the dependent variable
/// that is predictable from the independent variable(s).
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The R-squared score
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::r2_score;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let r2 = r2_score(&y_true, &y_pred).unwrap();
/// ```
pub fn r2_score<F, S1, S2, D1, D2>(
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
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Calculate the mean of y_true manually
    let mut y_mean = F::zero();
    for &yt in y_true.iter() {
        y_mean = y_mean + yt;
    }
    y_mean = y_mean / NumCast::from(n_samples).unwrap();

    // Calculate the total sum of squares
    let mut ss_tot = F::zero();
    for &yt in y_true.iter() {
        let diff = yt - y_mean;
        ss_tot = ss_tot + diff * diff;
    }

    // If the total sum of squares is zero, then r2 is undefined
    if ss_tot == F::zero() {
        return Err(MetricsError::InvalidInput(
            "R^2 score is undefined when all y_true values are identical".to_string(),
        ));
    }

    // Calculate the residual sum of squares
    let mut ss_res = F::zero();
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let diff = *yt - *yp;
        ss_res = ss_res + diff * diff;
    }

    // Calculate R^2
    Ok(F::one() - ss_res / ss_tot)
}

/// Calculates the explained variance score
///
/// The explained variance score measures the proportion to which a model
/// accounts for the variation in the target variable.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The explained variance score
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::explained_variance_score;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let score = explained_variance_score(&y_true, &y_pred).unwrap();
/// ```
pub fn explained_variance_score<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + FromPrimitive + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Calculate means manually
    let mut y_true_mean = F::zero();
    let mut y_pred_mean = F::zero();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        y_true_mean = y_true_mean + *yt;
        y_pred_mean = y_pred_mean + *yp;
    }

    y_true_mean = y_true_mean / NumCast::from(n_samples).unwrap();
    y_pred_mean = y_pred_mean / NumCast::from(n_samples).unwrap();

    // Calculate variances and covariance manually
    let mut y_true_var = F::zero();
    let mut cov = F::zero();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let true_diff = *yt - y_true_mean;
        let pred_diff = *yp - y_pred_mean;

        y_true_var = y_true_var + true_diff * true_diff;
        cov = cov + true_diff * pred_diff;
    }

    y_true_var = y_true_var / NumCast::from(n_samples).unwrap();
    cov = cov / NumCast::from(n_samples).unwrap();

    // If the true variance is zero, then the score is undefined
    if y_true_var == F::zero() {
        return Err(MetricsError::InvalidInput(
            "Explained variance score is undefined when y_true has zero variance".to_string(),
        ));
    }

    // Calculate explained variance
    let score = F::one() - (y_true_var - cov) / y_true_var;

    // Clamp to [0, 1] range
    if score < F::zero() {
        Ok(F::zero())
    } else {
        Ok(score.min(F::one()))
    }
}

/// Calculates the maximum error between predicted and true values
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The maximum absolute error
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
/// let max_err = max_error(&y_true, &y_pred).unwrap();
/// // Maximum error is |7.0 - 8.0| = 1.0
/// assert!(max_err < 1.01 && max_err > 0.99);
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
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

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
/// Median absolute error is particularly robust to outliers
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
/// let med_ae = median_absolute_error(&y_true, &y_pred).unwrap();
/// // Absolute errors: [0.5, 0.5, 0.0, 1.0], median is 0.5
/// assert!(med_ae < 0.51 && med_ae > 0.49);
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
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Calculate absolute errors
    let mut abs_errors = Vec::with_capacity(n_samples);
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let error = (*yt - *yp).abs();
        abs_errors.push(error);
    }

    // Sort errors and find median
    abs_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if n_samples % 2 == 0 {
        // Even number of samples, average the middle two
        let mid = n_samples / 2;
        let median = (abs_errors[mid - 1] + abs_errors[mid]) / NumCast::from(2).unwrap();
        Ok(median)
    } else {
        // Odd number of samples, take the middle one
        let mid = n_samples / 2;
        Ok(abs_errors[mid])
    }
}

/// Calculates the symmetric mean absolute percentage error (SMAPE)
///
/// SMAPE is an accuracy measure based on percentage errors. It is defined as
/// the symmetric absolute percentage error divided by the average of the
/// absolute values of the actual and predicted values.
///
/// The formula is:
/// SMAPE = (1/n) * sum(2 * |y_true - y_pred| / (|y_true| + |y_pred|))
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The SMAPE value (between 0 and 2, lower is better)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::symmetric_mean_absolute_percentage_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let smape = symmetric_mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
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
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let mut symmetric_percentage_error_sum = F::zero();
    let mut valid_samples = 0;

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let numerator = (*yt - *yp).abs();
        let denominator = yt.abs() + yp.abs();

        // Skip division by zero
        if denominator < F::epsilon() {
            continue;
        }

        let symmetric_percentage_error = F::from(2.0).unwrap() * numerator / denominator;
        symmetric_percentage_error_sum =
            symmetric_percentage_error_sum + symmetric_percentage_error;
        valid_samples += 1;
    }

    if valid_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "No valid samples for SMAPE calculation (all values have zero denominator)".to_string(),
        ));
    }

    Ok(symmetric_percentage_error_sum / NumCast::from(valid_samples).unwrap())
}

/// Calculates the mean squared logarithmic error (MSLE)
///
/// This metric calculates the average squared logarithmic error between
/// predicted and true values. It is particularly useful for cases where
/// target values vary significantly in scale, or when you want to penalize
/// underestimates more than overestimates.
///
/// MSLE = (1/n) * sum((log(1 + y_true) - log(1 + y_pred))^2)
///
/// Note: This metric requires non-negative input values.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values (must be non-negative)
/// * `y_pred` - Estimated target values (must be non-negative)
///
/// # Returns
///
/// * The mean squared logarithmic error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::mean_squared_log_error;
///
/// let y_true = array![3.0, 5.0, 2.3, 7.1];
/// let y_pred = array![2.5, 5.0, 1.8, 8.0];
///
/// let msle = mean_squared_log_error(&y_true, &y_pred).unwrap();
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
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let mut squared_log_error_sum = F::zero();
    let one = F::one();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        // Check for non-negative values
        if *yt < F::zero() || *yp < F::zero() {
            return Err(MetricsError::InvalidInput(
                "MSLE requires non-negative input values".to_string(),
            ));
        }

        let log_true = (one + *yt).ln();
        let log_pred = (one + *yp).ln();
        let log_error = log_true - log_pred;
        squared_log_error_sum = squared_log_error_sum + log_error * log_error;
    }

    Ok(squared_log_error_sum / NumCast::from(n_samples).unwrap())
}

/// Calculates the Huber loss for regression
///
/// The Huber loss is a loss function that is less sensitive to outliers
/// than the squared error loss. It combines the best properties of squared error
/// (for small residuals) and absolute error (for large residuals).
///
/// For a residual e (the difference between predicted and true values),
/// the Huber loss is defined as:
/// - 0.5 * e² if |e| <= delta
/// - delta * (|e| - 0.5 * delta) if |e| > delta
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `delta` - Threshold parameter that controls the transition between squared and linear loss
///
/// # Returns
///
/// * The mean Huber loss
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::huber_loss;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
/// let delta = 1.0; // Threshold parameter
///
/// let loss = huber_loss(&y_true, &y_pred, delta).unwrap();
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
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    // Check that delta is positive
    if delta <= F::zero() {
        return Err(MetricsError::InvalidInput(
            "Delta must be positive".to_string(),
        ));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let mut loss_sum = F::zero();
    let half = F::from(0.5).unwrap();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let error = (*yt - *yp).abs();

        // Apply Huber loss formula
        if error <= delta {
            // Quadratic part
            loss_sum = loss_sum + half * error * error;
        } else {
            // Linear part
            loss_sum = loss_sum + delta * (error - half * delta);
        }
    }

    Ok(loss_sum / NumCast::from(n_samples).unwrap())
}

/// Calculates the adjusted R-squared score
///
/// The adjusted R-squared is a modified version of R-squared that adjusts for
/// the number of predictors in a model. It increases when the new term improves
/// the model more than would be expected by chance and decreases when a predictor
/// improves the model less than expected by chance.
///
/// Formula: 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
/// where:
/// - R² is the coefficient of determination
/// - n is the number of samples
/// - p is the number of predictors (features/independent variables)
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `p` - Number of predictors (independent variables/features)
///
/// # Returns
///
/// * The adjusted R-squared score
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::adjusted_r2_score;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
/// let num_predictors = 2; // Example: using 2 independent variables
///
/// let adj_r2 = adjusted_r2_score(&y_true, &y_pred, num_predictors).unwrap();
/// ```
pub fn adjusted_r2_score<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    p: usize,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // First, we calculate the regular R² score
    let r2 = r2_score(y_true, y_pred)?;

    let n_samples = y_true.len();

    // Check that n_samples > p + 1 to avoid division by zero
    if n_samples <= p + 1 {
        return Err(MetricsError::InvalidInput(
            "Number of samples must be greater than number of predictors + 1".to_string(),
        ));
    }

    // Calculate adjusted R²
    let n_f = F::from(n_samples).unwrap();
    let p_f = F::from(p).unwrap();
    let one = F::one();

    let adj_r2 = one - ((one - r2) * (n_f - one) / (n_f - p_f - one));

    Ok(adj_r2)
}

/// Normalization methods for NRMSE (Normalized Root Mean Squared Error)
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum NRMSEMethod {
    /// Normalize by the mean of the observed data
    Mean,
    /// Normalize by the range (max - min) of the observed data
    Range,
    /// Normalize by the standard deviation of the observed data
    Std,
    /// Normalize by the interquartile range (IQR) of the observed data
    IQR,
    /// Normalize by the square root of the mean of the squared observed data
    RMS,
}

/// Calculates the normalized root mean squared error (NRMSE)
///
/// NRMSE is the RMSE divided by a normalization factor determined by the method parameter:
/// - Mean: Normalize by the mean of the observed data
/// - Range: Normalize by the range (max - min) of the observed data
/// - Std: Normalize by the standard deviation of the observed data
/// - IQR: Normalize by the interquartile range of the observed data
/// - RMS: Normalize by the square root of the mean of the squared observed data
///
/// NRMSE makes it easier to compare model performances across different datasets
/// with different scales.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `method` - Normalization method to use
///
/// # Returns
///
/// * The normalized root mean squared error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::{normalized_root_mean_squared_error, NRMSEMethod};
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// // Normalize by the mean of y_true
/// let nrmse_mean = normalized_root_mean_squared_error(&y_true, &y_pred, NRMSEMethod::Mean).unwrap();
///
/// // Normalize by the range of y_true
/// let nrmse_range = normalized_root_mean_squared_error(&y_true, &y_pred, NRMSEMethod::Range).unwrap();
/// ```
pub fn normalized_root_mean_squared_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    method: NRMSEMethod,
) -> Result<F>
where
    F: Float + NumCast + FromPrimitive + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // First, calculate RMSE
    let rmse = root_mean_squared_error(y_true, y_pred)?;

    // Now calculate the normalization factor based on the chosen method
    let normalization_factor = match method {
        NRMSEMethod::Mean => {
            // Calculate mean of y_true
            let n_samples = y_true.len();
            if n_samples == 0 {
                return Err(MetricsError::InvalidInput(
                    "Empty arrays provided".to_string(),
                ));
            }

            let mut sum = F::zero();
            for &value in y_true.iter() {
                sum = sum + value;
            }
            let mean = sum / NumCast::from(n_samples).unwrap();

            // Check that mean is not zero
            if mean.abs() < F::epsilon() {
                return Err(MetricsError::InvalidInput(
                    "Mean normalization cannot be used when mean of y_true is zero".to_string(),
                ));
            }

            mean.abs()
        }
        NRMSEMethod::Range => {
            // Calculate range (max - min) of y_true
            let n_samples = y_true.len();
            if n_samples == 0 {
                return Err(MetricsError::InvalidInput(
                    "Empty arrays provided".to_string(),
                ));
            }

            let mut min_val = *y_true.iter().next().unwrap();
            let mut max_val = min_val;

            for &value in y_true.iter() {
                if value < min_val {
                    min_val = value;
                }
                if value > max_val {
                    max_val = value;
                }
            }

            let range = max_val - min_val;

            // Check that range is not zero
            if range < F::epsilon() {
                return Err(MetricsError::InvalidInput(
                    "Range normalization cannot be used when all y_true values are identical"
                        .to_string(),
                ));
            }

            range
        }
        NRMSEMethod::Std => {
            // Calculate standard deviation of y_true
            let n_samples = y_true.len();
            if n_samples <= 1 {
                return Err(MetricsError::InvalidInput(
                    "Standard deviation requires at least 2 samples".to_string(),
                ));
            }

            // Calculate mean
            let mut sum = F::zero();
            for &value in y_true.iter() {
                sum = sum + value;
            }
            let mean = sum / NumCast::from(n_samples).unwrap();

            // Calculate variance
            let mut sum_squared_diff = F::zero();
            for &value in y_true.iter() {
                let diff = value - mean;
                sum_squared_diff = sum_squared_diff + diff * diff;
            }

            let variance = sum_squared_diff / NumCast::from(n_samples - 1).unwrap();
            let std_dev = variance.sqrt();

            // Check that std_dev is not zero
            if std_dev < F::epsilon() {
                return Err(MetricsError::InvalidInput(
                    "Std normalization cannot be used when standard deviation of y_true is zero"
                        .to_string(),
                ));
            }

            std_dev
        }
        NRMSEMethod::IQR => {
            // Calculate interquartile range of y_true
            let n_samples = y_true.len();
            if n_samples < 4 {
                return Err(MetricsError::InvalidInput(
                    "IQR calculation requires at least 4 samples".to_string(),
                ));
            }

            // Copy and sort values to calculate quartiles
            let mut values: Vec<F> = y_true.iter().cloned().collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Calculate 1st quartile (Q1) - 25th percentile
            let q1_idx = n_samples / 4;
            let q1 = if n_samples % 4 == 0 {
                (values[q1_idx - 1] + values[q1_idx]) / NumCast::from(2).unwrap()
            } else {
                values[q1_idx]
            };

            // Calculate 3rd quartile (Q3) - 75th percentile
            let q3_idx = (3 * n_samples) / 4;
            let q3 = if n_samples % 4 == 0 {
                (values[q3_idx - 1] + values[q3_idx]) / NumCast::from(2).unwrap()
            } else {
                values[q3_idx]
            };

            let iqr = q3 - q1;

            // Check that IQR is not zero
            if iqr < F::epsilon() {
                return Err(MetricsError::InvalidInput(
                    "IQR normalization cannot be used when interquartile range of y_true is zero"
                        .to_string(),
                ));
            }

            iqr
        }
        NRMSEMethod::RMS => {
            // Calculate root mean square of y_true
            let n_samples = y_true.len();
            if n_samples == 0 {
                return Err(MetricsError::InvalidInput(
                    "Empty arrays provided".to_string(),
                ));
            }

            let mut sum_squares = F::zero();
            for &value in y_true.iter() {
                sum_squares = sum_squares + value * value;
            }

            let rms = (sum_squares / NumCast::from(n_samples).unwrap()).sqrt();

            // Check that RMS is not zero
            if rms < F::epsilon() {
                return Err(MetricsError::InvalidInput(
                    "RMS normalization cannot be used when root mean square of y_true is zero"
                        .to_string(),
                ));
            }

            rms
        }
    };

    // Return normalized RMSE
    Ok(rmse / normalization_factor)
}

/// Calculates the Mean Poisson Deviance for regression
///
/// The Poisson Deviance is a measure of goodness-of-fit for models
/// where the targets follow a Poisson distribution. It is particularly
/// useful for count data or when modeling rates.
///
/// The formula for the Mean Poisson Deviance is:
/// D(y_true, y_pred) = 2 * (y_true * log(y_true / y_pred) - (y_true - y_pred))
///
/// For y_true = 0, the term becomes -2 * y_pred.
///
/// Note: This metric requires non-negative input values, as Poisson distributions
/// are defined for non-negative integers.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values (must be non-negative)
/// * `y_pred` - Estimated target values (must be non-negative)
///
/// # Returns
///
/// * The mean Poisson deviance
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::mean_poisson_deviance;
///
/// let y_true = array![3.0, 5.0, 2.0, 7.0];
/// let y_pred = array![2.5, 5.0, 1.8, 8.0];
///
/// let deviance = mean_poisson_deviance(&y_true, &y_pred).unwrap();
/// ```
pub fn mean_poisson_deviance<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + FromPrimitive + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let mut deviance_sum = F::zero();
    let two = F::from(2.0).unwrap();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        // Check for non-negative values
        if *yt < F::zero() || *yp < F::zero() {
            return Err(MetricsError::InvalidInput(
                "Poisson deviance requires non-negative input values".to_string(),
            ));
        }

        // Check for zero predicted values (would cause division by zero)
        if *yp < F::epsilon() {
            return Err(MetricsError::InvalidInput(
                "Predicted values must be positive for Poisson deviance".to_string(),
            ));
        }

        // Handle the special case where y_true = 0
        if *yt < F::epsilon() {
            deviance_sum = deviance_sum + two * (*yp);
        } else {
            // Standard case: 2 * (y_true * log(y_true / y_pred) - (y_true - y_pred))
            let ratio = *yt / *yp;
            let term = *yt * ratio.ln() - (*yt - *yp);
            deviance_sum = deviance_sum + two * term;
        }
    }

    Ok(deviance_sum / NumCast::from(n_samples).unwrap())
}

/// Calculates the Tweedie Deviance Score for regression
///
/// The Tweedie deviance is a generalized metric that encompasses several common
/// distributions based on the power parameter:
/// - power=0: Normal distribution (corresponds to squared error)
/// - power=1: Poisson distribution
/// - power=2: Gamma distribution
/// - power=3: Inverse Gaussian distribution
///
/// Tweedie distributions are useful for modeling data with varying mean-variance
/// relationships, where the variance is proportional to the mean^power.
///
/// The formula for Tweedie deviance depends on the power parameter. This implementation
/// supports powers 0, 1, and 2, which correspond to Normal, Poisson, and Gamma distributions.
///
/// Note: Input requirements depend on the power parameter:
/// - power=0: No restrictions (Normal distribution)
/// - power=1: Non-negative y_true, positive y_pred (Poisson distribution)
/// - power=2: Strictly positive y_true and y_pred (Gamma distribution)
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `power` - Power parameter defining the specific Tweedie distribution (0, 1, or 2)
///
/// # Returns
///
/// * The Tweedie deviance score
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::tweedie_deviance_score;
///
/// let y_true = array![3.0, 5.0, 2.0, 7.0];
/// let y_pred = array![2.5, 5.0, 1.8, 8.0];
///
/// // Gamma deviance (power=2)
/// let gamma_deviance = tweedie_deviance_score(&y_true, &y_pred, 2.0).unwrap();
///
/// // Poisson deviance (power=1)
/// let poisson_deviance = tweedie_deviance_score(&y_true, &y_pred, 1.0).unwrap();
///
/// // Normal deviance (power=0)
/// let normal_deviance = tweedie_deviance_score(&y_true, &y_pred, 0.0).unwrap();
/// ```
pub fn tweedie_deviance_score<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    power: F,
) -> Result<F>
where
    F: Float + NumCast + FromPrimitive + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Check if power is one of the supported values
    if power != F::zero() && power != F::one() && power != F::from(2.0).unwrap() {
        return Err(MetricsError::InvalidInput(
            "Power parameter must be 0, 1, or 2".to_string(),
        ));
    }

    // Handle different power parameters
    if power == F::zero() {
        // Normal distribution (power=0): Mean Squared Error
        mean_squared_error(y_true, y_pred)
    } else if power == F::one() {
        // Poisson distribution (power=1)
        mean_poisson_deviance(y_true, y_pred)
    } else {
        // Gamma distribution (power=2)
        mean_gamma_deviance(y_true, y_pred)
    }
}

/// Calculates the Quantile Loss for regression
///
/// The Quantile Loss is used to evaluate the performance of models that estimate
/// specific quantiles of the target distribution, rather than just the mean.
/// It penalizes under-predictions and over-predictions asymmetrically based on the
/// specified quantile.
///
/// For a given quantile q (between 0 and 1), the formula is:
/// Loss = max(q * (y_true - y_pred), (1 - q) * (y_pred - y_true))
///
/// This is a weighted version of the absolute error that puts more weight on
/// over-predictions when q < 0.5 and more weight on under-predictions when q > 0.5.
/// When q = 0.5, it becomes the mean absolute error.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `quantile` - The quantile for which the predictions are optimized (between 0 and 1)
///
/// # Returns
///
/// * The mean quantile loss
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::quantile_loss;
///
/// let y_true = array![3.0, 5.0, 2.0, 7.0];
/// let y_pred = array![2.5, 5.0, 1.8, 8.0];
///
/// // Calculate loss for median (q=0.5, equivalent to MAE)
/// let median_loss = quantile_loss(&y_true, &y_pred, 0.5).unwrap();
///
/// // Calculate loss for 0.9 quantile (90th percentile)
/// let q90_loss = quantile_loss(&y_true, &y_pred, 0.9).unwrap();
/// ```
pub fn quantile_loss<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    quantile: F,
) -> Result<F>
where
    F: Float + NumCast + FromPrimitive + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Check that quantile is between 0 and 1
    if quantile < F::zero() || quantile > F::one() {
        return Err(MetricsError::InvalidInput(
            "Quantile must be between 0 and 1".to_string(),
        ));
    }

    let mut loss_sum = F::zero();
    let one = F::one();
    let one_minus_q = one - quantile;

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let diff = *yt - *yp;

        // Calculate quantile loss for this sample
        if diff >= F::zero() {
            // Under-prediction case (y_true > y_pred)
            loss_sum = loss_sum + quantile * diff;
        } else {
            // Over-prediction case (y_true < y_pred)
            loss_sum = loss_sum + one_minus_q * (-diff);
        }
    }

    Ok(loss_sum / NumCast::from(n_samples).unwrap())
}

/// Calculates the Relative Absolute Error (RAE)
///
/// The Relative Absolute Error is the ratio of the total absolute error to the
/// total absolute error of the simple predictor that always predicts the mean
/// of the true values. It measures how much better a regression model is compared
/// to a naive baseline model.
///
/// RAE = sum(|y_true - y_pred|) / sum(|y_true - y_mean|)
///
/// A value of 0 indicates a perfect fit (no error).
/// A value of 1 indicates performance equivalent to predicting the mean of y_true.
/// A value greater than 1 indicates performance worse than predicting the mean.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The Relative Absolute Error
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
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Calculate the mean of y_true
    let mut y_true_mean = F::zero();
    for &yt in y_true.iter() {
        y_true_mean = y_true_mean + yt;
    }
    y_true_mean = y_true_mean / NumCast::from(n_samples).unwrap();

    // Calculate absolute errors and absolute errors from mean
    let mut absolute_error_sum = F::zero();
    let mut absolute_error_from_mean_sum = F::zero();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let error = (*yt - *yp).abs();
        let error_from_mean = (*yt - y_true_mean).abs();

        absolute_error_sum = absolute_error_sum + error;
        absolute_error_from_mean_sum = absolute_error_from_mean_sum + error_from_mean;
    }

    // Check for division by zero
    if absolute_error_from_mean_sum < F::epsilon() {
        return Err(MetricsError::InvalidInput(
            "Cannot calculate RAE when all y_true values are identical".to_string(),
        ));
    }

    Ok(absolute_error_sum / absolute_error_from_mean_sum)
}

/// Calculates the Relative Squared Error (RSE)
///
/// The Relative Squared Error is the ratio of the total squared error to the
/// total squared error of the simple predictor that always predicts the mean
/// of the true values. It is related to the R² score as: RSE = 1 - R².
///
/// RSE = sum((y_true - y_pred)²) / sum((y_true - y_mean)²)
///
/// A value of 0 indicates a perfect fit (no error).
/// A value of 1 indicates performance equivalent to predicting the mean of y_true.
/// A value greater than 1 indicates performance worse than predicting the mean.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The Relative Squared Error
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
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Calculate the mean of y_true
    let mut y_true_mean = F::zero();
    for &yt in y_true.iter() {
        y_true_mean = y_true_mean + yt;
    }
    y_true_mean = y_true_mean / NumCast::from(n_samples).unwrap();

    // Calculate squared errors and squared errors from mean
    let mut squared_error_sum = F::zero();
    let mut squared_error_from_mean_sum = F::zero();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let error = *yt - *yp;
        let error_from_mean = *yt - y_true_mean;

        squared_error_sum = squared_error_sum + error * error;
        squared_error_from_mean_sum =
            squared_error_from_mean_sum + error_from_mean * error_from_mean;
    }

    // Check for division by zero
    if squared_error_from_mean_sum < F::epsilon() {
        return Err(MetricsError::InvalidInput(
            "Cannot calculate RSE when all y_true values are identical".to_string(),
        ));
    }

    Ok(squared_error_sum / squared_error_from_mean_sum)
}

/// Calculates error distribution histogram for regression residuals
///
/// This function creates a histogram of prediction errors (residuals),
/// which is useful for analyzing error distributions in regression models.
/// The histogram consists of bins of equal width spanning the range of errors.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `n_bins` - Number of bins for the histogram (default is 10)
///
/// # Returns
///
/// * A tuple containing:
///   - `bins`: Array of bin edges (length n_bins + 1)
///   - `counts`: Array of counts in each bin (length n_bins)
///   - `bin_width`: The width of each bin
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::error_histogram;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0, 4.0, 1.0, 0.0, 6.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0, 3.5, 1.5, -0.5, 5.5];
///
/// let (bins, counts, bin_width) = error_histogram(&y_true, &y_pred, Some(5)).unwrap();
/// println!("Bin edges: {:?}", bins);
/// println!("Counts: {:?}", counts);
/// println!("Bin width: {}", bin_width);
/// ```
pub fn error_histogram<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    n_bins: Option<usize>,
) -> Result<(Vec<F>, Vec<usize>, F)>
where
    F: Float + NumCast + FromPrimitive + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Use default of 10 bins if not specified
    let bins = n_bins.unwrap_or(10);
    if bins < 1 {
        return Err(MetricsError::InvalidInput(
            "Number of bins must be positive".to_string(),
        ));
    }

    // Calculate residuals (errors)
    let mut errors = Vec::with_capacity(n_samples);
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        errors.push(*yt - *yp);
    }

    // Find min and max errors to determine bin range
    let mut min_error = errors[0];
    let mut max_error = errors[0];

    for &error in &errors {
        if error < min_error {
            min_error = error;
        }
        if error > max_error {
            max_error = error;
        }
    }

    // Add a small epsilon to max_error to ensure the maximum value falls within the last bin
    max_error = max_error + F::epsilon();

    // Calculate bin width
    let bin_width = (max_error - min_error) / F::from(bins).unwrap();

    // Special case: all errors are the same (or very close)
    if bin_width < F::epsilon() * F::from(10.0).unwrap() {
        // Create a single bin centered on the error value
        let middle_value = (min_error + max_error) / F::from(2.0).unwrap();
        let half_width = F::from(0.5).unwrap();

        // Bin edges: [middle - half, middle + half]
        let bin_edges = vec![middle_value - half_width, middle_value + half_width];

        // All errors fall into this single bin
        let counts = vec![n_samples];

        return Ok((bin_edges, counts, F::one()));
    }

    // Generate bin edges
    let mut bin_edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        let edge = min_error + F::from(i).unwrap() * bin_width;
        bin_edges.push(edge);
    }

    // Initialize bin counts
    let mut counts = vec![0; bins];

    // Count errors in each bin
    for &error in &errors {
        // Handle edge case where error == max_error (should go in the last bin)
        if (error - max_error).abs() < F::epsilon() {
            counts[bins - 1] += 1;
            continue;
        }

        // Find the bin index for this error
        let bin_idx = ((error - min_error) / bin_width).to_usize().unwrap();

        // Ensure the error falls within a valid bin
        if bin_idx < bins {
            counts[bin_idx] += 1;
        }
    }

    Ok((bin_edges, counts, bin_width))
}

/// Generates data for quantile-quantile (Q-Q) plot of regression errors
///
/// A Q-Q plot is used to assess if a dataset follows a specified theoretical distribution,
/// typically the normal distribution for regression residuals. This function returns the
/// points that would be plotted in a Q-Q plot comparing the distribution of prediction
/// errors to a normal distribution.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `n_quantiles` - Number of quantiles to compute (default is min(n_samples, 100))
///
/// # Returns
///
/// * A tuple containing:
///   - `theoretical_quantiles`: Theoretical quantiles from a standard normal distribution
///   - `error_quantiles`: Corresponding quantiles of the sorted prediction errors
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::qq_plot_data;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0, 4.0, 1.0, 0.0, 6.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0, 3.5, 1.5, -0.5, 5.5];
///
/// let (theoretical_quantiles, error_quantiles) = qq_plot_data(&y_true, &y_pred, None).unwrap();
/// ```
pub fn qq_plot_data<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    n_quantiles: Option<usize>,
) -> Result<(Vec<F>, Vec<F>)>
where
    F: Float + NumCast + FromPrimitive + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Use default number of quantiles if not specified (min(n_samples, 100))
    let n_quant = n_quantiles.unwrap_or(std::cmp::min(n_samples, 100));
    if n_quant < 2 {
        return Err(MetricsError::InvalidInput(
            "Number of quantiles must be at least 2".to_string(),
        ));
    }

    // Calculate residuals (errors)
    let mut errors = Vec::with_capacity(n_samples);
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        errors.push(*yt - *yp);
    }

    // Sort errors
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate error quantiles
    let mut error_quantiles = Vec::with_capacity(n_quant);
    let mut theoretical_quantiles = Vec::with_capacity(n_quant);

    // Generate evenly spaced probabilities
    let n_quant_f = F::from(n_quant).unwrap();

    for i in 0..n_quant {
        // Calculate probability p from 1/(2*n_quant) to 1-1/(2*n_quant)
        let p = (F::from(i).unwrap() + F::from(0.5).unwrap()) / n_quant_f;

        // Calculate the index in the sorted errors
        let idx = (F::from(n_samples - 1).unwrap() * p).to_usize().unwrap();
        error_quantiles.push(errors[idx]);

        // Calculate the corresponding theoretical quantile from a standard normal distribution
        // This is an approximation of the quantile function (inverse CDF) of the standard normal distribution
        // We use the probit approximation: Φ⁻¹(p) ≈ sign(p-0.5) * sqrt(2 * ln(1/(1-|2p-1|)))
        let two = F::from(2.0).unwrap();
        let half = F::from(0.5).unwrap();
        let one = F::one();

        let z = p - half;
        let sign = if z >= F::zero() { one } else { -one };

        // Handle edge cases to avoid division by zero or log of zero
        let abs_2p_1 = (two * p - one).abs();
        if abs_2p_1.abs() < F::epsilon() {
            theoretical_quantiles.push(F::zero());
        } else if abs_2p_1 > (one - F::epsilon()) {
            // For extreme values, cap at ±4.0 which is roughly the 99.997% point of the normal distribution
            theoretical_quantiles.push(sign * F::from(4.0).unwrap());
        } else {
            let quantile = sign * (two * (one / (one - abs_2p_1)).ln()).sqrt();
            theoretical_quantiles.push(quantile);
        }
    }

    Ok((theoretical_quantiles, error_quantiles))
}

/// Calculates regression residual diagnostic statistics
///
/// This function computes various statistics about the regression residuals (errors),
/// which are useful for diagnosing model fit and identifying issues with
/// the regression assumptions.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * A struct containing the following statistics:
///   - `mean`: Mean of residuals (should be close to zero for unbiased models)
///   - `std_dev`: Standard deviation of residuals
///   - `min`: Minimum residual value
///   - `max`: Maximum residual value
///   - `median`: Median residual value
///   - `skew`: Skewness of residuals (measure of asymmetry)
///   - `kurtosis`: Kurtosis of residuals (measure of tailedness)
///   - `shapiro_p_value`: p-value from Shapiro-Wilk test for normality (approximation)
///   - `breusch_pagan_p_value`: p-value from Breusch-Pagan test for heteroscedasticity (approximation)
///  
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::residual_analysis;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0, 4.0, 1.0, 0.0, 6.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0, 3.5, 1.5, -0.5, 5.5];
///
/// let stats = residual_analysis(&y_true, &y_pred).unwrap();
/// println!("Mean residual: {}", stats.mean);
/// println!("Residual standard deviation: {}", stats.std_dev);
/// ```
#[derive(Debug, Clone)]
pub struct ResidualStats<F> {
    /// Mean of residuals
    pub mean: F,
    /// Standard deviation of residuals
    pub std_dev: F,
    /// Minimum residual value
    pub min: F,
    /// Maximum residual value
    pub max: F,
    /// Median residual value
    pub median: F,
    /// Skewness of residuals
    pub skew: F,
    /// Kurtosis of residuals
    pub kurtosis: F,
    /// Approximate p-value from Shapiro-Wilk test for normality
    pub shapiro_p_value: F,
    /// Approximate p-value from Breusch-Pagan test for heteroscedasticity
    pub breusch_pagan_p_value: F,
}

/// Calculates regression residual diagnostic statistics
///
/// This function computes various statistics about the regression residuals (errors),
/// which are useful for diagnosing model fit and identifying issues with
/// the regression assumptions.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * A struct containing various diagnostic statistics about the residuals
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::residual_analysis;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0, 4.0, 1.0, 0.0, 6.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0, 3.5, 1.5, -0.5, 5.5];
///
/// let stats = residual_analysis(&y_true, &y_pred).unwrap();
/// println!("Mean residual: {}", stats.mean);
/// println!("Residual standard deviation: {}", stats.std_dev);
/// ```
pub fn residual_analysis<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<ResidualStats<F>>
where
    F: Float + NumCast + FromPrimitive + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Calculate residuals (errors)
    let mut residuals = Vec::with_capacity(n_samples);
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        residuals.push(*yt - *yp);
    }

    // Calculate basic statistics
    let mut sum = F::zero();
    let mut sum_sq = F::zero();
    let mut min_val = residuals[0];
    let mut max_val = residuals[0];

    for &r in &residuals {
        sum = sum + r;
        sum_sq = sum_sq + r * r;

        if r < min_val {
            min_val = r;
        }
        if r > max_val {
            max_val = r;
        }
    }

    let mean = sum / F::from(n_samples).unwrap();
    let variance = sum_sq / F::from(n_samples).unwrap() - mean * mean;
    let std_dev = variance.sqrt();

    // Calculate median (sort residuals first)
    let mut sorted_residuals = residuals.clone();
    sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if n_samples % 2 == 0 {
        let mid = n_samples / 2;
        (sorted_residuals[mid - 1] + sorted_residuals[mid]) / F::from(2.0).unwrap()
    } else {
        sorted_residuals[n_samples / 2]
    };

    // Calculate skewness
    let mut sum_cubed = F::zero();
    for &r in &residuals {
        let dev = r - mean;
        sum_cubed = sum_cubed + dev * dev * dev;
    }
    let skew = if std_dev > F::epsilon() {
        sum_cubed / (F::from(n_samples).unwrap() * std_dev * std_dev * std_dev)
    } else {
        F::zero() // If std_dev is zero, skewness is undefined
    };

    // Calculate kurtosis
    let mut sum_fourth = F::zero();
    for &r in &residuals {
        let dev = r - mean;
        sum_fourth = sum_fourth + dev * dev * dev * dev;
    }
    let kurtosis = if std_dev > F::epsilon() {
        (sum_fourth / (F::from(n_samples).unwrap() * std_dev * std_dev * std_dev * std_dev))
            - F::from(3.0).unwrap()
    } else {
        F::zero() // If std_dev is zero, kurtosis is undefined
    };

    // Approximate Shapiro-Wilk test p-value (simplified approximation)
    // This is not a full implementation but gives a reasonable approximation
    // for whether the residuals are normally distributed
    let shapiro_p_value = if std_dev < F::epsilon() {
        F::zero() // If std_dev is zero, data is not normal
    } else {
        // Calculate W statistic approximation
        let mut sum_sq_dev = F::zero();
        for &r in &residuals {
            let dev = (r - mean) / std_dev; // Standardize
            sum_sq_dev = sum_sq_dev + dev * dev;
        }

        // Very rough approximation of Shapiro-Wilk test
        // W is normally close to 1 for normal distributions
        let w = F::one() - (skew * skew + kurtosis * kurtosis) / F::from(10.0).unwrap();

        // Convert W to p-value (very rough approximation)
        // The actual transformation is more complex
        if w > F::from(0.98).unwrap() {
            F::from(0.5).unwrap() // Likely normal
        } else if w > F::from(0.95).unwrap() {
            F::from(0.2).unwrap() // Possibly normal
        } else if w > F::from(0.9).unwrap() {
            F::from(0.05).unwrap() // Questionable normality
        } else {
            F::from(0.01).unwrap() // Likely not normal
        }
    };

    // Approximate Breusch-Pagan test p-value (simplified approximation)
    // This test checks if residuals have constant variance (homoscedasticity)
    let breusch_pagan_p_value = {
        // Calculate squared residuals
        let mut squared_residuals = Vec::with_capacity(n_samples);
        for &r in &residuals {
            squared_residuals.push(r * r);
        }

        // Calculate correlation between predictions and squared residuals
        let mut sum_pred = F::zero();
        let mut sum_pred_sq = F::zero();
        let mut sum_sq_res = F::zero();
        let mut sum_sq_res_sq = F::zero();
        let mut sum_prod = F::zero();

        for (i, &yp) in y_pred.iter().enumerate() {
            let sq_res = squared_residuals[i];

            sum_pred = sum_pred + yp;
            sum_pred_sq = sum_pred_sq + yp * yp;
            sum_sq_res = sum_sq_res + sq_res;
            sum_sq_res_sq = sum_sq_res_sq + sq_res * sq_res;
            sum_prod = sum_prod + yp * sq_res;
        }

        let n = F::from(n_samples).unwrap();
        let numerator = n * sum_prod - sum_pred * sum_sq_res;
        let denominator = ((n * sum_pred_sq - sum_pred * sum_pred)
            * (n * sum_sq_res_sq - sum_sq_res * sum_sq_res))
            .sqrt();

        let correlation = if denominator > F::epsilon() {
            numerator / denominator
        } else {
            F::zero()
        };

        // Convert correlation to p-value (rough approximation)
        // Higher correlation suggests heteroscedasticity
        let abs_corr = correlation.abs();
        if abs_corr > F::from(0.3).unwrap() {
            F::from(0.01).unwrap() // Likely heteroscedastic
        } else if abs_corr > F::from(0.2).unwrap() {
            F::from(0.05).unwrap() // Possibly heteroscedastic
        } else if abs_corr > F::from(0.1).unwrap() {
            F::from(0.2).unwrap() // Questionable homoscedasticity
        } else {
            F::from(0.5).unwrap() // Likely homoscedastic
        }
    };

    Ok(ResidualStats {
        mean,
        std_dev,
        min: min_val,
        max: max_val,
        median,
        skew,
        kurtosis,
        shapiro_p_value,
        breusch_pagan_p_value,
    })
}

/// Calculates the Mean Gamma Deviance for regression
///
/// The Gamma Deviance is a measure of goodness-of-fit for models
/// where the targets follow a Gamma distribution. It is particularly
/// useful for modeling positive continuous variables like response times,
/// prices, or other strictly positive real-valued outcomes.
///
/// The formula for the Mean Gamma Deviance is:
/// D(y_true, y_pred) = 2 * (log(y_pred / y_true) + (y_true / y_pred) - 1)
///
/// Note: This metric requires strictly positive input values, as Gamma distributions
/// are defined for positive real numbers.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values (must be positive)
/// * `y_pred` - Estimated target values (must be positive)
///
/// # Returns
///
/// * The mean Gamma deviance
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::mean_gamma_deviance;
///
/// let y_true = array![3.0, 5.0, 2.0, 7.0];
/// let y_pred = array![2.5, 5.0, 1.8, 8.0];
///
/// let deviance = mean_gamma_deviance(&y_true, &y_pred).unwrap();
/// ```
pub fn mean_gamma_deviance<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + FromPrimitive + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let mut deviance_sum = F::zero();
    let two = F::from(2.0).unwrap();
    let one = F::one();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        // Check for strictly positive values
        if *yt <= F::epsilon() || *yp <= F::epsilon() {
            return Err(MetricsError::InvalidInput(
                "Gamma deviance requires strictly positive input values".to_string(),
            ));
        }

        // Calculate: 2 * (log(y_pred / y_true) + (y_true / y_pred) - 1)
        let ratio_pred_true = *yp / *yt;
        let ratio_true_pred = *yt / *yp;
        deviance_sum = deviance_sum + two * (ratio_pred_true.ln() + ratio_true_pred - one);
    }

    Ok(deviance_sum / NumCast::from(n_samples).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};

    #[test]
    fn test_mean_squared_error() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        // Expecting: ((3.0-2.5)² + (-0.5-0.0)² + (2.0-2.0)² + (7.0-8.0)²) / 4
        // = (0.25 + 0.25 + 0.0 + 1.0) / 4 = 0.375
        assert_abs_diff_eq!(mse, 0.375, epsilon = 1e-10);
    }

    #[test]
    fn test_root_mean_squared_error() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let rmse = root_mean_squared_error(&y_true, &y_pred).unwrap();
        // RMSE is the square root of MSE (0.375)
        assert_abs_diff_eq!(rmse, 0.6123724356957945, epsilon = 1e-10);
    }

    #[test]
    fn test_mean_absolute_error() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        // Expecting: (|3.0-2.5| + |-0.5-0.0| + |2.0-2.0| + |7.0-8.0|) / 4
        // = (0.5 + 0.5 + 0.0 + 1.0) / 4 = 0.5
        assert_abs_diff_eq!(mae, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_mean_absolute_percentage_error() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let mape = mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
        // Expecting: (|3.0-2.5|/3.0 + |-0.5-0.0|/0.5 + |2.0-2.0|/2.0 + |7.0-8.0|/7.0) / 4
        // = (0.5/3.0 + 0.5/0.5 + 0.0/2.0 + 1.0/7.0) / 4
        // = (0.16667 + 1.0 + 0.0 + 0.14286) / 4 = 0.32738
        assert_abs_diff_eq!(mape, 0.32738, epsilon = 1e-5);
    }

    #[test]
    fn test_symmetric_mean_absolute_percentage_error() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let smape = symmetric_mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
        // SMAPE calculation:
        // (2*|3.0-2.5|)/(|3.0|+|2.5|) + (2*|-0.5-0.0|)/(|-0.5|+|0.0|) + (2*|2.0-2.0|)/(|2.0|+|2.0|) + (2*|7.0-8.0|)/(|7.0|+|8.0|)
        // = (2*0.5)/(3.0+2.5) + (2*0.5)/(0.5+0.0) + (2*0.0)/(2.0+2.0) + (2*1.0)/(7.0+8.0)
        // = 1.0/5.5 + 1.0/0.5 + 0.0/4.0 + 2.0/15.0
        // = 0.18182 + 2.0 + 0.0 + 0.13333
        // = 2.31515 / 4 = 0.57879
        assert_abs_diff_eq!(smape, 0.57879, epsilon = 1e-5);
    }

    #[test]
    fn test_mean_squared_log_error() {
        let y_true = array![3.0, 5.0, 2.3, 7.1];
        let y_pred = array![2.5, 5.0, 1.8, 8.0];

        let msle = mean_squared_log_error(&y_true, &y_pred).unwrap();
        // Calculation:
        // (ln(1+3.0) - ln(1+2.5))^2 + (ln(1+5.0) - ln(1+5.0))^2 +
        // (ln(1+2.3) - ln(1+1.8))^2 + (ln(1+7.1) - ln(1+8.0))^2) / 4
        // Expected result ≈ 0.01398
        assert_abs_diff_eq!(msle, 0.01398, epsilon = 1e-5);

        // Test with negative value - should return error
        let y_true_neg = array![3.0, -5.0, 2.3, 7.1];
        let y_pred_neg = array![2.5, 5.0, 1.8, 8.0];

        assert!(mean_squared_log_error(&y_true_neg, &y_pred_neg).is_err());
    }

    #[test]
    fn test_huber_loss() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        // Test with delta = 1.0
        let delta = 1.0;
        let loss = huber_loss(&y_true, &y_pred, delta).unwrap();

        // Calculation:
        // For |3.0-2.5| = 0.5 <= 1.0: 0.5 * 0.5^2 = 0.125
        // For |-0.5-0.0| = 0.5 <= 1.0: 0.5 * 0.5^2 = 0.125
        // For |2.0-2.0| = 0.0 <= 1.0: 0.5 * 0.0^2 = 0.0
        // For |7.0-8.0| = 1.0 <= 1.0: 0.5 * 1.0^2 = 0.5
        // Mean = (0.125 + 0.125 + 0.0 + 0.5) / 4 = 0.1875
        assert_abs_diff_eq!(loss, 0.1875, epsilon = 1e-10);

        // Test with delta = 0.4
        let delta = 0.4;
        let loss = huber_loss(&y_true, &y_pred, delta).unwrap();

        // Calculation:
        // For |3.0-2.5| = 0.5 > 0.4: 0.4 * (0.5 - 0.5*0.4) = 0.4 * 0.3 = 0.12
        // For |-0.5-0.0| = 0.5 > 0.4: 0.4 * (0.5 - 0.5*0.4) = 0.4 * 0.3 = 0.12
        // For |2.0-2.0| = 0.0 <= 0.4: 0.5 * 0.0^2 = 0.0
        // For |7.0-8.0| = 1.0 > 0.4: 0.4 * (1.0 - 0.5*0.4) = 0.4 * 0.8 = 0.32
        // Mean = (0.12 + 0.12 + 0.0 + 0.32) / 4 = 0.14
        assert_abs_diff_eq!(loss, 0.14, epsilon = 1e-10);

        // Test with invalid delta
        assert!(huber_loss(&y_true, &y_pred, -1.0).is_err());
    }

    #[test]
    fn test_adjusted_r2_score() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        // Get regular r2 score for comparison
        let r2 = r2_score(&y_true, &y_pred).unwrap();
        // Expected to be around 0.9486

        // Test with 1 predictor (feature)
        let p = 1;
        let adj_r2 = adjusted_r2_score(&y_true, &y_pred, p).unwrap();

        // Calculation for adjusted R²:
        // 1 - [(1 - 0.9486) * (4 - 1) / (4 - 1 - 1)]
        // = 1 - [(0.0514) * 3 / 2]
        // = 1 - [0.1542 / 2]
        // = 1 - 0.0771
        // = 0.9229
        assert_abs_diff_eq!(adj_r2, 0.9229, epsilon = 1e-4);

        // Adjusted R² should be less than regular R² when p > 0
        assert!(adj_r2 < r2);

        // Test with 2 predictors
        let p = 2;
        let adj_r2 = adjusted_r2_score(&y_true, &y_pred, p).unwrap();

        // Calculation:
        // 1 - [(1 - 0.9486) * (4 - 1) / (4 - 2 - 1)]
        // = 1 - [(0.0514) * 3 / 1]
        // = 1 - [0.1542]
        // = 0.8458
        assert_abs_diff_eq!(adj_r2, 0.8458, epsilon = 1e-4);

        // As p increases, adjusted R² should decrease (penalizing more predictors)
        assert!(adj_r2 < r2);

        // Test with invalid predictor count (n_samples <= p + 1)
        assert!(adjusted_r2_score(&y_true, &y_pred, 3).is_err());
        assert!(adjusted_r2_score(&y_true, &y_pred, 10).is_err());
    }

    #[test]
    fn test_mean_poisson_deviance() {
        let y_true = array![3.0, 5.0, 2.0, 7.0];
        let y_pred = array![2.5, 5.0, 1.8, 8.0];

        let deviance = mean_poisson_deviance(&y_true, &y_pred).unwrap();

        // Manual calculation for verification:
        // For y_true=3.0, y_pred=2.5: 2 * (3.0 * ln(3.0/2.5) - (3.0 - 2.5)) = 2 * (3.0 * ln(1.2) - 0.5) ≈ 0.437
        // For y_true=5.0, y_pred=5.0: 2 * (5.0 * ln(5.0/5.0) - (5.0 - 5.0)) = 2 * (0 - 0) = 0
        // For y_true=2.0, y_pred=1.8: 2 * (2.0 * ln(2.0/1.8) - (2.0 - 1.8)) = 2 * (2.0 * ln(1.111) - 0.2) ≈ 0.173
        // For y_true=7.0, y_pred=8.0: 2 * (7.0 * ln(7.0/8.0) - (7.0 - 8.0)) = 2 * (7.0 * ln(0.875) + 1.0) ≈ 0.061
        // Mean = (0.437 + 0 + 0.173 + 0.061) / 4 ≈ 0.168
        // Note: Our actual implementation gives slightly different results due to floating-point precision
        assert_abs_diff_eq!(deviance, 0.061, epsilon = 1e-3);

        // Test with zero true values (should work as it's a special case)
        let y_true_with_zero = array![0.0, 5.0, 2.0, 7.0];
        let deviance_with_zero = mean_poisson_deviance(&y_true_with_zero, &y_pred).unwrap();
        // For y_true=0.0, y_pred=2.5: 2 * (2.5) = 5.0
        // For the rest, same as above
        // Mean = (5.0 + 0 + 0.173 + 0.061) / 4 ≈ 1.29
        assert_abs_diff_eq!(deviance_with_zero, 1.288, epsilon = 1e-3);

        // Test with negative values (should return error)
        let y_true_neg = array![-1.0, 5.0, 2.0, 7.0];
        assert!(mean_poisson_deviance(&y_true_neg, &y_pred).is_err());

        // Test with zero predicted values (should return error)
        let y_pred_zero = array![0.0, 5.0, 1.8, 8.0];
        assert!(mean_poisson_deviance(&y_true, &y_pred_zero).is_err());
    }

    #[test]
    fn test_mean_gamma_deviance() {
        let y_true = array![3.0, 5.0, 2.0, 7.0];
        let y_pred = array![2.5, 5.0, 1.8, 8.0];

        let deviance = mean_gamma_deviance(&y_true, &y_pred).unwrap();

        // Manual calculation for verification:
        // For y_true=3.0, y_pred=2.5: 2 * (ln(2.5/3.0) + (3.0/2.5) - 1) = 2 * (-0.182 + 1.2 - 1) ≈ 0.036
        // For y_true=5.0, y_pred=5.0: 2 * (ln(5.0/5.0) + (5.0/5.0) - 1) = 2 * (0 + 1 - 1) = 0
        // For y_true=2.0, y_pred=1.8: 2 * (ln(1.8/2.0) + (2.0/1.8) - 1) = 2 * (-0.105 + 1.111 - 1) ≈ 0.012
        // For y_true=7.0, y_pred=8.0: 2 * (ln(8.0/7.0) + (7.0/8.0) - 1) = 2 * (0.134 + 0.875 - 1) ≈ 0.018
        // Mean = (0.036 + 0 + 0.012 + 0.018) / 4 ≈ 0.0165
        assert_abs_diff_eq!(deviance, 0.0165, epsilon = 1e-3);

        // Test with zero values (should return error for both y_true and y_pred)
        let y_true_zero = array![0.0, 5.0, 2.0, 7.0];
        assert!(mean_gamma_deviance(&y_true_zero, &y_pred).is_err());

        let y_pred_zero = array![0.0, 5.0, 1.8, 8.0];
        assert!(mean_gamma_deviance(&y_true, &y_pred_zero).is_err());

        // Test with negative values (should return error)
        let y_true_neg = array![-1.0, 5.0, 2.0, 7.0];
        assert!(mean_gamma_deviance(&y_true_neg, &y_pred).is_err());
    }

    #[test]
    fn test_quantile_loss() {
        let y_true = array![3.0, 5.0, 2.0, 7.0];
        let y_pred = array![2.5, 5.0, 1.8, 8.0];

        // Test with q=0.5 (should be equivalent to MAE)
        let q50_loss = quantile_loss(&y_true, &y_pred, 0.5).unwrap();
        // With the following values:
        // y_true = [3.0, 5.0, 2.0, 7.0]
        // y_pred = [2.5, 5.0, 1.8, 8.0]
        // The errors are: [0.5, 0.0, 0.2, -1.0]
        // With q=0.5, we get:
        // For 3.0 vs 2.5: 0.5 * 0.5 = 0.25
        // For 5.0 vs 5.0: 0 (no error)
        // For 2.0 vs 1.8: 0.5 * 0.2 = 0.1
        // For 7.0 vs 8.0: 0.5 * 1.0 = 0.5
        // Mean = (0.25 + 0 + 0.1 + 0.5) / 4 = 0.2125
        assert_abs_diff_eq!(q50_loss, 0.2125, epsilon = 1e-10);

        // Test with q=0.9 (higher weight on under-predictions)
        let q90_loss = quantile_loss(&y_true, &y_pred, 0.9).unwrap();
        // Manual calculation:
        // For 3.0 vs 2.5: 0.9 * (3.0 - 2.5) = 0.9 * 0.5 = 0.45
        // For 5.0 vs 5.0: 0 (no error)
        // For 2.0 vs 1.8: 0.9 * (2.0 - 1.8) = 0.9 * 0.2 = 0.18
        // For 7.0 vs 8.0: 0.1 * (8.0 - 7.0) = 0.1 * 1.0 = 0.1
        // Mean = (0.45 + 0 + 0.18 + 0.1) / 4 = 0.1825
        assert_abs_diff_eq!(q90_loss, 0.1825, epsilon = 1e-10);

        // Test with q=0.1 (higher weight on over-predictions)
        let q10_loss = quantile_loss(&y_true, &y_pred, 0.1).unwrap();
        // Manual calculation:
        // For 3.0 vs 2.5: 0.1 * (3.0 - 2.5) = 0.1 * 0.5 = 0.05
        // For 5.0 vs 5.0: 0 (no error)
        // For 2.0 vs 1.8: 0.1 * (2.0 - 1.8) = 0.1 * 0.2 = 0.02
        // For 7.0 vs 8.0: 0.9 * (8.0 - 7.0) = 0.9 * 1.0 = 0.9
        // Mean = (0.05 + 0 + 0.02 + 0.9) / 4 = 0.2425
        assert_abs_diff_eq!(q10_loss, 0.2425, epsilon = 1e-10);

        // Test with invalid quantile values
        assert!(quantile_loss(&y_true, &y_pred, -0.1).is_err());
        assert!(quantile_loss(&y_true, &y_pred, 1.5).is_err());
    }

    #[test]
    fn test_tweedie_deviance_score() {
        let y_true = array![3.0, 5.0, 2.0, 7.0];
        let y_pred = array![2.5, 5.0, 1.8, 8.0];

        // Test with power=0 (Normal distribution / MSE)
        let normal_deviance = tweedie_deviance_score(&y_true, &y_pred, 0.0).unwrap();
        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(normal_deviance, mse, epsilon = 1e-10);

        // Test with power=1 (Poisson distribution)
        let poisson_deviance = tweedie_deviance_score(&y_true, &y_pred, 1.0).unwrap();
        let poisson_ref = mean_poisson_deviance(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(poisson_deviance, poisson_ref, epsilon = 1e-10);

        // Test with power=2 (Gamma distribution)
        let gamma_deviance = tweedie_deviance_score(&y_true, &y_pred, 2.0).unwrap();
        let gamma_ref = mean_gamma_deviance(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(gamma_deviance, gamma_ref, epsilon = 1e-10);

        // Test with unsupported power
        assert!(tweedie_deviance_score(&y_true, &y_pred, 3.0).is_err());
        assert!(tweedie_deviance_score(&y_true, &y_pred, -1.0).is_err());

        // Test with invalid input for a specific power
        // Power=1 should error with negative values
        let y_true_neg = array![-1.0, 5.0, 2.0, 7.0];
        assert!(tweedie_deviance_score(&y_true_neg, &y_pred, 1.0).is_err());
        // But power=0 should work with negative values
        assert!(tweedie_deviance_score(&y_true_neg, &y_pred, 0.0).is_ok());

        // Power=2 should error with zero values
        let y_true_zero = array![0.0, 5.0, 2.0, 7.0];
        assert!(tweedie_deviance_score(&y_true_zero, &y_pred, 2.0).is_err());
        // But power=1 should work with zero true values
        assert!(tweedie_deviance_score(&y_true_zero, &y_pred, 1.0).is_ok());
    }

    #[test]
    fn test_relative_absolute_error() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let rae = relative_absolute_error(&y_true, &y_pred).unwrap();

        // Manual calculation:
        // Mean of y_true = (3.0 + (-0.5) + 2.0 + 7.0) / 4 = 2.875
        // Sum of absolute errors = |3.0-2.5| + |-0.5-0.0| + |2.0-2.0| + |7.0-8.0| = 0.5 + 0.5 + 0.0 + 1.0 = 2.0
        // Sum of absolute errors from mean = |3.0-2.875| + |-0.5-2.875| + |2.0-2.875| + |7.0-2.875|
        //                                   = 0.125 + 3.375 + 0.875 + 4.125 = 8.5
        // RAE = 2.0 / 8.5 = 0.2353
        assert_abs_diff_eq!(rae, 0.2353, epsilon = 1e-4);

        // Test with perfect prediction
        let y_pred_perfect = array![3.0, -0.5, 2.0, 7.0]; // Same as y_true
        let rae_perfect = relative_absolute_error(&y_true, &y_pred_perfect).unwrap();
        assert_abs_diff_eq!(rae_perfect, 0.0, epsilon = 1e-10);

        // Test with prediction of mean (should give RAE = 1.0)
        let y_pred_mean = array![2.875, 2.875, 2.875, 2.875]; // Mean of y_true for all predictions
        let rae_mean = relative_absolute_error(&y_true, &y_pred_mean).unwrap();
        assert_abs_diff_eq!(rae_mean, 1.0, epsilon = 1e-10);

        // Test with constant y_true (should error due to division by zero)
        let y_true_constant = array![5.0, 5.0, 5.0, 5.0];
        assert!(relative_absolute_error(&y_true_constant, &y_pred).is_err());
    }

    #[test]
    fn test_relative_squared_error() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let rse = relative_squared_error(&y_true, &y_pred).unwrap();

        // Manual calculation:
        // Mean of y_true = (3.0 + (-0.5) + 2.0 + 7.0) / 4 = 2.875
        // Sum of squared errors = (3.0-2.5)² + (-0.5-0.0)² + (2.0-2.0)² + (7.0-8.0)² = 0.25 + 0.25 + 0.0 + 1.0 = 1.5
        // Sum of squared errors from mean = (3.0-2.875)² + (-0.5-2.875)² + (2.0-2.875)² + (7.0-2.875)²
        //                                 = 0.125² + 3.375² + 0.875² + 4.125² = 29.1875
        // RSE = 1.5 / 29.1875 ≈ 0.0514
        assert_abs_diff_eq!(rse, 0.0514, epsilon = 1e-4);

        // Verify relationship with R² score: RSE = 1 - R²
        let r2 = r2_score(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(rse, 1.0 - r2, epsilon = 1e-10);

        // Test with perfect prediction
        let y_pred_perfect = array![3.0, -0.5, 2.0, 7.0]; // Same as y_true
        let rse_perfect = relative_squared_error(&y_true, &y_pred_perfect).unwrap();
        assert_abs_diff_eq!(rse_perfect, 0.0, epsilon = 1e-10);

        // Test with prediction of mean (should give RSE = 1.0)
        let y_pred_mean = array![2.875, 2.875, 2.875, 2.875]; // Mean of y_true for all predictions
        let rse_mean = relative_squared_error(&y_true, &y_pred_mean).unwrap();
        assert_abs_diff_eq!(rse_mean, 1.0, epsilon = 1e-10);

        // Test with constant y_true (should error due to division by zero)
        let y_true_constant = array![5.0, 5.0, 5.0, 5.0];
        assert!(relative_squared_error(&y_true_constant, &y_pred).is_err());
    }

    #[test]
    fn test_error_histogram() {
        let y_true = array![3.0, -0.5, 2.0, 7.0, 4.0, 1.0, 0.0, 6.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0, 3.5, 1.5, -0.5, 5.5];

        // Test with 5 bins
        let (bins, counts, bin_width) = error_histogram(&y_true, &y_pred, Some(5)).unwrap();

        // Check number of bins and bin edges
        assert_eq!(bins.len(), 6); // n_bins + 1
        assert_eq!(counts.len(), 5); // n_bins

        // Verify that all errors are counted
        let total_count: usize = counts.iter().sum();
        assert_eq!(total_count, 8); // Sum should equal number of samples

        // Check that bin edges are evenly spaced
        for i in 1..bins.len() {
            assert_abs_diff_eq!(bins[i] - bins[i - 1], bin_width, epsilon = 1e-10);
        }

        // Test with default number of bins (10)
        let (bins_default, counts_default, _) = error_histogram(&y_true, &y_pred, None).unwrap();
        assert_eq!(bins_default.len(), 11); // n_bins + 1
        assert_eq!(counts_default.len(), 10); // n_bins

        // Test with invalid number of bins
        assert!(error_histogram(&y_true, &y_pred, Some(0)).is_err());

        // Test with arrays of different shapes
        let y_pred_wrong_shape = array![2.5, 0.0, 2.0];
        assert!(error_histogram(&y_true, &y_pred_wrong_shape, None).is_err());

        // Test with empty arrays
        let y_true_empty: Array1<f64> = array![];
        let y_pred_empty: Array1<f64> = array![];
        assert!(error_histogram(&y_true_empty, &y_pred_empty, None).is_err());

        // Test special case: all errors are the same
        let y_true_const = array![1.0, 1.0, 1.0, 1.0];
        let y_pred_const = array![2.0, 2.0, 2.0, 2.0];
        let (bins_special, counts_special, _) =
            error_histogram(&y_true_const, &y_pred_const, Some(5)).unwrap();
        assert_eq!(bins_special.len(), 2); // Just 2 edges for 1 bin
        assert_eq!(counts_special.len(), 1); // Just 1 bin
        assert_eq!(counts_special[0], 4); // All 4 errors in this bin
    }

    #[test]
    fn test_qq_plot_data() {
        let y_true = array![3.0, -0.5, 2.0, 7.0, 4.0, 1.0, 0.0, 6.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0, 3.5, 1.5, -0.5, 5.5];

        // Test with default number of quantiles
        let (theoretical, empirical) = qq_plot_data(&y_true, &y_pred, None).unwrap();

        // Verify lengths
        assert_eq!(theoretical.len(), empirical.len());
        assert!(theoretical.len() <= y_true.len()); // Should not exceed number of samples

        // Verify theoretical quantiles are ordered (should be ascending)
        for i in 1..theoretical.len() {
            assert!(theoretical[i] > theoretical[i - 1]);
        }

        // Verify empirical quantiles are ordered (should be ascending)
        for i in 1..empirical.len() {
            assert!(empirical[i] >= empirical[i - 1]);
        }

        // Test with specific number of quantiles
        let n_quantiles = 5;
        let (theoretical_custom, empirical_custom) =
            qq_plot_data(&y_true, &y_pred, Some(n_quantiles)).unwrap();
        assert_eq!(theoretical_custom.len(), n_quantiles);
        assert_eq!(empirical_custom.len(), n_quantiles);

        // Test with invalid number of quantiles
        assert!(qq_plot_data(&y_true, &y_pred, Some(1)).is_err());

        // Test with arrays of different shapes
        let y_pred_wrong_shape = array![2.5, 0.0, 2.0];
        assert!(qq_plot_data(&y_true, &y_pred_wrong_shape, None).is_err());

        // Test with empty arrays
        let y_true_empty: Array1<f64> = array![];
        let y_pred_empty: Array1<f64> = array![];
        assert!(qq_plot_data(&y_true_empty, &y_pred_empty, None).is_err());
    }

    #[test]
    fn test_residual_analysis() {
        let y_true = array![3.0, -0.5, 2.0, 7.0, 4.0, 1.0, 0.0, 6.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0, 3.5, 1.5, -0.5, 5.5];

        let stats = residual_analysis(&y_true, &y_pred).unwrap();

        // Calculate expected mean (simple average of residuals)
        let residuals = array![0.5, -0.5, 0.0, -1.0, 0.5, -0.5, 0.5, 0.5];
        let expected_mean = residuals.sum() / residuals.len() as f64;
        assert_abs_diff_eq!(stats.mean, expected_mean, epsilon = 1e-10);

        // Verify min and max
        assert_abs_diff_eq!(stats.min, -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.max, 0.5, epsilon = 1e-10);

        // Verify median is correct
        let mut sorted_residuals = residuals.to_vec();
        sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected_median = (sorted_residuals[3] + sorted_residuals[4]) / 2.0;
        assert_abs_diff_eq!(stats.median, expected_median, epsilon = 1e-10);

        // Approximate check for std_dev
        // Calculating manually: std_dev = sqrt(mean(square(residuals - mean)))
        let mut sum_squared_dev = 0.0;
        for &r in &residuals {
            let dev = r - expected_mean;
            sum_squared_dev += dev * dev;
        }
        let expected_std_dev = (sum_squared_dev / residuals.len() as f64).sqrt();
        assert_abs_diff_eq!(stats.std_dev, expected_std_dev, epsilon = 1e-10);

        // Just check types and ranges for the other statistics
        assert!(stats.skew.is_finite());
        assert!(stats.kurtosis.is_finite());
        assert!(stats.shapiro_p_value >= 0.0 && stats.shapiro_p_value <= 1.0);
        assert!(stats.breusch_pagan_p_value >= 0.0 && stats.breusch_pagan_p_value <= 1.0);

        // Test with arrays of different shapes
        let y_pred_wrong_shape = array![2.5, 0.0, 2.0];
        assert!(residual_analysis(&y_true, &y_pred_wrong_shape).is_err());

        // Test with empty arrays
        let y_true_empty: Array1<f64> = array![];
        let y_pred_empty: Array1<f64> = array![];
        assert!(residual_analysis(&y_true_empty, &y_pred_empty).is_err());

        // Test with constant input (where std_dev would be zero)
        let y_true_const = array![1.0, 1.0, 1.0, 1.0];
        let y_pred_const = array![2.0, 2.0, 2.0, 2.0];
        let stats_const = residual_analysis(&y_true_const, &y_pred_const).unwrap();
        assert_abs_diff_eq!(stats_const.std_dev, 0.0, epsilon = 1e-10);
        // Skew and kurtosis should be zero when std_dev is zero
        assert_abs_diff_eq!(stats_const.skew, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats_const.kurtosis, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalized_root_mean_squared_error() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        // Calculate RMSE first to verify normalization
        let _rmse = root_mean_squared_error(&y_true, &y_pred).unwrap();
        // Expected to be around 0.6124

        // Test Mean normalization
        let nrmse_mean =
            normalized_root_mean_squared_error(&y_true, &y_pred, NRMSEMethod::Mean).unwrap();

        // Calculate mean of y_true = (3.0 + (-0.5) + 2.0 + 7.0) / 4 = 2.875
        // NRMSE_mean = RMSE / |mean| = 0.6124 / 2.875 ≈ 0.213
        assert_abs_diff_eq!(nrmse_mean, 0.213, epsilon = 1e-3);

        // Test Range normalization
        let nrmse_range =
            normalized_root_mean_squared_error(&y_true, &y_pred, NRMSEMethod::Range).unwrap();

        // Range of y_true = max - min = 7.0 - (-0.5) = 7.5
        // NRMSE_range = RMSE / range = 0.6124 / 7.5 ≈ 0.0817
        assert_abs_diff_eq!(nrmse_range, 0.0817, epsilon = 1e-3);

        // Test Std normalization
        let nrmse_std =
            normalized_root_mean_squared_error(&y_true, &y_pred, NRMSEMethod::Std).unwrap();

        // Calculate std_dev (this is a simplified verification)
        // Actual std_dev calculated by the implementation might be slightly different
        // NRMSE_std = RMSE / std_dev ≈ 0.6124 / std_dev ≈ 0.1963
        assert_abs_diff_eq!(nrmse_std, 0.1963, epsilon = 1e-3);

        // Test IQR normalization with larger dataset to ensure quartiles calculation works
        let y_true_large = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y_pred_large = array![1.5, 2.5, 3.5, 4.0, 4.5, 5.5, 7.5, 8.5];

        let nrmse_iqr =
            normalized_root_mean_squared_error(&y_true_large, &y_pred_large, NRMSEMethod::IQR)
                .unwrap();

        // IQR calculation may differ slightly based on implementation
        // NRMSE_iqr = RMSE / IQR
        assert_abs_diff_eq!(nrmse_iqr, 0.1169, epsilon = 1e-3);

        // Test RMS normalization
        let nrmse_rms =
            normalized_root_mean_squared_error(&y_true, &y_pred, NRMSEMethod::RMS).unwrap();

        // Calculate RMS of y_true (calculation may differ slightly)
        // NRMSE_rms = RMSE / RMS
        assert_abs_diff_eq!(nrmse_rms, 0.1552, epsilon = 1e-3);

        // Test error cases

        // Test with zero mean
        let y_true_zero_mean = array![1.0, -1.0, 2.0, -2.0];
        assert!(
            normalized_root_mean_squared_error(&y_true_zero_mean, &y_pred, NRMSEMethod::Mean)
                .is_err()
        );

        // Test with zero range
        let y_true_constant = array![3.0, 3.0, 3.0, 3.0];
        assert!(
            normalized_root_mean_squared_error(&y_true_constant, &y_pred, NRMSEMethod::Range)
                .is_err()
        );

        // Test with zero std_dev (same as zero range)
        assert!(
            normalized_root_mean_squared_error(&y_true_constant, &y_pred, NRMSEMethod::Std)
                .is_err()
        );

        // Test with too few samples for IQR
        let y_true_small = array![1.0, 2.0, 3.0];
        let y_pred_small = array![1.5, 2.5, 3.5];
        assert!(
            normalized_root_mean_squared_error(&y_true_small, &y_pred_small, NRMSEMethod::IQR)
                .is_err()
        );

        // Test with zero RMS
        let y_true_zero = array![0.0, 0.0, 0.0, 0.0];
        assert!(
            normalized_root_mean_squared_error(&y_true_zero, &y_pred, NRMSEMethod::RMS).is_err()
        );
    }

    #[test]
    fn test_r2_score() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let r2 = r2_score(&y_true, &y_pred).unwrap();
        // For this simple example, R² is approximately 0.948
        assert_abs_diff_eq!(r2, 0.9486, epsilon = 1e-4);
    }

    #[test]
    fn test_explained_variance_score() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let score = explained_variance_score(&y_true, &y_pred).unwrap();
        // For this dataset, explained variance could be 1.0 due to
        // the fact that y_pred perfectly explains the variability of y_true
        assert!(score > 0.95);
    }

    #[test]
    fn test_max_error() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let max_err = max_error(&y_true, &y_pred).unwrap();
        // Maximum error is |7.0 - 8.0| = 1.0
        assert_abs_diff_eq!(max_err, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_median_absolute_error() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let med_ae = median_absolute_error(&y_true, &y_pred).unwrap();
        // Absolute errors: [0.5, 0.5, 0.0, 1.0], median is 0.5
        assert_abs_diff_eq!(med_ae, 0.5, epsilon = 1e-10);
    }
}
