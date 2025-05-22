//! Robust regression metrics
//!
//! This module provides functions for calculating robust metrics that are
//! less sensitive to outliers and work for different distributions.

use ndarray::{Array1, ArrayBase, Data, Dimension};
use num_traits::{Float, FromPrimitive, NumCast};

use super::{check_non_negative, check_positive, check_same_shape};
use crate::error::{MetricsError, Result};

/// Calculates the mean Poisson deviance
///
/// Poisson deviance is a measure of goodness of fit for Poisson regression models.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The mean Poisson deviance
///
/// # Notes
///
/// * Both `y_true` and `y_pred` must be non-negative
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::mean_poisson_deviance;
///
/// let y_true = array![3.0, 5.0, 2.0, 7.0];
/// let y_pred = array![2.5, 5.0, 3.0, 8.0];
///
/// let mpd = mean_poisson_deviance(&y_true, &y_pred).unwrap();
/// assert!(mpd >= 0.0);
/// ```
pub fn mean_poisson_deviance<F, S1, S2, D1, D2>(
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

    // Check that all values are non-negative
    check_non_negative::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    let n_samples = y_true.len();
    let mut deviance_sum = F::zero();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        if *yp < F::epsilon() {
            return Err(MetricsError::InvalidInput(
                "Predicted values must be positive for Poisson deviance".to_string(),
            ));
        }

        if *yt > F::epsilon() {
            // Use y * log(y/yhat) - (y - yhat) formula
            deviance_sum = deviance_sum + *yt * (*yt / *yp).ln() - (*yt - *yp);
        } else {
            // When y_true is 0, the limit of the deviance is just y_pred
            deviance_sum = deviance_sum + *yp;
        }
    }

    Ok(F::from(2.0).unwrap() * deviance_sum / NumCast::from(n_samples).unwrap())
}

/// Calculates the mean gamma deviance
///
/// Gamma deviance is a measure of goodness of fit for gamma regression models.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The mean gamma deviance
///
/// # Notes
///
/// * Both `y_true` and `y_pred` must be positive
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::mean_gamma_deviance;
///
/// let y_true = array![3.0, 5.0, 2.0, 7.0];
/// let y_pred = array![2.5, 5.0, 3.0, 8.0];
///
/// let mgd = mean_gamma_deviance(&y_true, &y_pred).unwrap();
/// assert!(mgd >= 0.0);
/// ```
pub fn mean_gamma_deviance<F, S1, S2, D1, D2>(
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

    // Check that all values are strictly positive
    check_positive::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    let n_samples = y_true.len();
    let mut deviance_sum = F::zero();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        // Gamma deviance: -log(y/yhat) + (y - yhat)/yhat
        deviance_sum = deviance_sum - (*yt / *yp).ln() + (*yt - *yp) / *yp;
    }

    Ok(F::from(2.0).unwrap() * deviance_sum / NumCast::from(n_samples).unwrap())
}

/// Calculates the Tweedie deviance score
///
/// Tweedie deviance is a measure of goodness of fit for Tweedie regression models.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `power` - Power parameter of the Tweedie distribution
///
/// # Returns
///
/// * The Tweedie deviance score
///
/// # Notes
///
/// * For power=1, it corresponds to Poisson deviance
/// * For power=2, it corresponds to gamma deviance
/// * For power=0, it corresponds to Gaussian deviance (mean squared error)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::tweedie_deviance_score;
///
/// let y_true = array![3.0, 5.0, 2.0, 7.0];
/// let y_pred = array![2.5, 5.0, 3.0, 8.0];
///
/// // Gamma deviance (power=2)
/// let tds = tweedie_deviance_score(&y_true, &y_pred, 2.0).unwrap();
/// assert!(tds >= 0.0);
/// ```
pub fn tweedie_deviance_score<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    power: F,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + FromPrimitive,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    if power.abs() < F::epsilon() {
        // Gaussian case (power = 0) -> use mean squared error
        let mut sum_squared_error = F::zero();
        for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
            let error = *yt - *yp;
            sum_squared_error = sum_squared_error + error * error;
        }
        return Ok(sum_squared_error / NumCast::from(y_true.len()).unwrap());
    } else if (power - F::one()).abs() < F::epsilon() {
        // Poisson case (power = 1)
        return mean_poisson_deviance(y_true, y_pred);
    } else if (power - F::from(2.0).unwrap()).abs() < F::epsilon() {
        // Gamma case (power = 2)
        return mean_gamma_deviance(y_true, y_pred);
    }

    // Generic Tweedie case
    let n_samples = y_true.len();
    let mut deviance_sum = F::zero();

    let two = F::from(2.0).unwrap();

    // Check values based on power
    if power < F::one() {
        // No constraints on y and y_pred
    } else if power < two {
        // y_true must be non-negative, y_pred must be positive
        for &val in y_true.iter() {
            if val < F::zero() {
                return Err(MetricsError::InvalidInput(
                    "y_true contains negative values, which is not allowed for this power parameter".to_string(),
                ));
            }
        }

        for &val in y_pred.iter() {
            if val <= F::zero() {
                return Err(MetricsError::InvalidInput(
                    "y_pred contains non-positive values, which is not allowed for this power parameter".to_string(),
                ));
            }
        }
    } else {
        // Both y_true and y_pred must be positive
        check_positive::<F, S1, S2, D1, D2>(y_true, y_pred)?;
    }

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        // Generic Tweedie deviance formula
        let term1 = if (*yt).abs() < F::epsilon() {
            F::zero() // Handle y_true = 0 case
        } else {
            *yt * ((*yt).powf(F::one() - power) - (*yp).powf(F::one() - power)) / (F::one() - power)
        };

        let term2 = (*yp).powf(two - power) - (*yt).powf(two - power) / (two - power);
        deviance_sum = deviance_sum + two * (term1 - term2);
    }

    Ok(deviance_sum / NumCast::from(n_samples).unwrap())
}

/// Calculates the quantile loss (pinball loss)
///
/// Quantile loss is used to evaluate quantile regression models, which predict
/// a specific quantile of the target distribution.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `quantile` - The quantile to evaluate (between 0 and 1)
///
/// # Returns
///
/// * The quantile loss
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::quantile_loss;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// // Median (q=0.5)
/// let median_loss = quantile_loss(&y_true, &y_pred, 0.5).unwrap();
/// assert!(median_loss >= 0.0);
///
/// // 90th percentile (q=0.9)
/// let p90_loss = quantile_loss(&y_true, &y_pred, 0.9).unwrap();
/// assert!(p90_loss >= 0.0);
/// ```
pub fn quantile_loss<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    quantile: F,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + std::fmt::Display + FromPrimitive,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    // Check quantile is between 0 and 1
    if quantile <= F::zero() || quantile >= F::one() {
        return Err(MetricsError::InvalidInput(format!(
            "Quantile must be between 0 and 1, got {}",
            quantile
        )));
    }

    let n_samples = y_true.len();
    let mut loss_sum = F::zero();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let error = *yt - *yp;
        if error >= F::zero() {
            // Underestimation penalty
            loss_sum = loss_sum + quantile * error;
        } else {
            // Overestimation penalty
            loss_sum = loss_sum + (quantile - F::one()) * error;
        }
    }

    Ok(loss_sum / NumCast::from(n_samples).unwrap())
}

/// Computes robust weights for regression metrics based on residuals
///
/// # Arguments
///
/// * `residuals` - Array of residuals (y_true - y_pred)
/// * `method` - Weight function method:
///   * "huber" - Huber weights (linear penalty for small errors, constant for large errors)
///   * "bisquare" - Tukey's bisquare weights (zero weight for large errors)
///   * "cauchy" - Cauchy weights (smooth transition)
/// * `tuning` - Optional tuning parameter for the weight function
///
/// # Returns
///
/// * Array of weights for each residual
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array1};
/// use scirs2_metrics::regression::compute_robust_weights;
///
/// // Create some residuals with outliers
/// let residuals = array![0.1, 0.2, -0.3, 5.0, 0.2, -0.1, -4.0];
///
/// // Compute weights using Huber method
/// let weights = compute_robust_weights(&residuals, "huber", None).unwrap();
///
/// // Outliers should have smaller weights
/// assert!(weights[3] < weights[0]);
/// assert!(weights[6] < weights[1]);
/// ```
pub fn compute_robust_weights<F, S, D>(
    residuals: &ArrayBase<S, D>,
    method: &str,
    tuning: Option<F>,
) -> Result<Array1<F>>
where
    F: Float + NumCast + std::fmt::Debug + FromPrimitive,
    S: Data<Elem = F>,
    D: Dimension,
{
    let n = residuals.len();

    if n == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty residuals array".to_string(),
        ));
    }

    // Get absolute residuals
    let abs_residuals: Vec<F> = residuals.iter().map(|&r| r.abs()).collect();

    // Compute median absolute deviation (MAD)
    let mut sorted_abs = abs_residuals.clone();
    sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median_idx = n / 2;
    let mad = if n % 2 == 0 {
        (sorted_abs[median_idx - 1] + sorted_abs[median_idx]) / F::from(2.0).unwrap()
    } else {
        sorted_abs[median_idx]
    };

    // Scale MAD for consistency with normal distribution
    let scale = mad / F::from(0.6745).unwrap();

    // Use a small positive value if MAD is zero
    let s = if scale > F::epsilon() {
        scale
    } else {
        F::from(1e-8).unwrap()
    };

    // Get tuning parameter (default values depend on method)
    let c = match tuning {
        Some(t) => t,
        None => match method {
            "huber" => F::from(1.345).unwrap(),
            "bisquare" => F::from(4.685).unwrap(),
            "cauchy" => F::from(2.385).unwrap(),
            _ => F::from(1.345).unwrap(), // Default to huber
        },
    };

    // Calculate weights based on method
    let mut weights = Array1::<F>::zeros(n);

    for (i, &r) in abs_residuals.iter().enumerate() {
        let u = r / (c * s);

        weights[i] = match method {
            "huber" => {
                if u <= F::one() {
                    F::one()
                } else {
                    F::one() / u
                }
            }
            "bisquare" => {
                if u < F::one() {
                    let temp = F::one() - u * u;
                    temp * temp
                } else {
                    F::zero()
                }
            }
            "cauchy" => F::one() / (F::one() + u * u),
            _ => {
                return Err(MetricsError::InvalidInput(format!(
                    "Unknown weight method: {}. Valid options are 'huber', 'bisquare', 'cauchy'.",
                    method
                )));
            }
        };
    }

    Ok(weights)
}

/// Calculates the weighted mean squared error
///
/// This function applies weights to each sample's squared error, which can be used
/// for robust estimation or to emphasize/de-emphasize certain samples.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `weights` - Sample weights (same length as y_true/y_pred)
///
/// # Returns
///
/// * The weighted mean squared error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::weighted_mean_squared_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
/// let weights = array![1.0, 0.5, 1.0, 0.2]; // Less weight on outliers
///
/// let wmse = weighted_mean_squared_error(&y_true, &y_pred, &weights).unwrap();
/// assert!(wmse >= 0.0);
/// ```
pub fn weighted_mean_squared_error<F, S1, S2, S3, D1, D2, D3>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    weights: &ArrayBase<S3, D3>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    S3: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
    D3: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    let n_samples = y_true.len();

    if weights.len() != n_samples {
        return Err(MetricsError::InvalidInput(format!(
            "Weights length ({}) must match y_true length ({})",
            weights.len(),
            n_samples
        )));
    }

    let mut weighted_error_sum = F::zero();
    let mut weight_sum = F::zero();

    for ((yt, yp), &w) in y_true.iter().zip(y_pred.iter()).zip(weights.iter()) {
        if w < F::zero() {
            return Err(MetricsError::InvalidInput(
                "Weights must be non-negative".to_string(),
            ));
        }

        let error = *yt - *yp;
        weighted_error_sum = weighted_error_sum + w * error * error;
        weight_sum = weight_sum + w;
    }

    if weight_sum <= F::epsilon() {
        return Err(MetricsError::InvalidInput(
            "Sum of weights is zero".to_string(),
        ));
    }

    Ok(weighted_error_sum / weight_sum)
}

/// Calculates the weighted median absolute error
///
/// This function applies weights to each sample's absolute error and returns
/// the weighted median, which is more robust to outliers than mean-based metrics.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `weights` - Sample weights (same length as y_true/y_pred)
///
/// # Returns
///
/// * The weighted median absolute error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::weighted_median_absolute_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
/// let weights = array![1.0, 0.5, 1.0, 0.2]; // Less weight on outliers
///
/// let wmedae = weighted_median_absolute_error(&y_true, &y_pred, &weights).unwrap();
/// assert!(wmedae >= 0.0);
/// ```
pub fn weighted_median_absolute_error<F, S1, S2, S3, D1, D2, D3>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    weights: &ArrayBase<S3, D3>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    S3: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
    D3: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    let n_samples = y_true.len();

    if weights.len() != n_samples {
        return Err(MetricsError::InvalidInput(format!(
            "Weights length ({}) must match y_true length ({})",
            weights.len(),
            n_samples
        )));
    }

    // Calculate absolute errors and check weights
    let mut abs_errors = Vec::with_capacity(n_samples);
    let mut valid_weights = Vec::with_capacity(n_samples);
    let mut weight_sum = F::zero();

    for ((yt, yp), &w) in y_true.iter().zip(y_pred.iter()).zip(weights.iter()) {
        if w < F::zero() {
            return Err(MetricsError::InvalidInput(
                "Weights must be non-negative".to_string(),
            ));
        }

        let error = (*yt - *yp).abs();
        abs_errors.push(error);
        valid_weights.push(w);
        weight_sum = weight_sum + w;
    }

    if weight_sum <= F::epsilon() {
        return Err(MetricsError::InvalidInput(
            "Sum of weights is zero".to_string(),
        ));
    }

    // Sort errors and weights together
    let mut error_weight_pairs: Vec<(F, F)> = abs_errors.into_iter().zip(valid_weights).collect();
    error_weight_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate weighted median
    let half_weight = weight_sum / F::from(2.0).unwrap();
    let mut cumulative_weight = F::zero();

    for (error, weight) in &error_weight_pairs {
        cumulative_weight = cumulative_weight + *weight;
        if cumulative_weight >= half_weight {
            return Ok(*error);
        }
    }

    // Fallback (shouldn't reach here if weights sum to positive value)
    Ok(error_weight_pairs.last().unwrap().0)
}

/// Calculates the M-estimator for regression
///
/// M-estimators are robust regression estimators that downweight the influence of outliers.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `loss_function` - Loss function to use:
///   * "huber" - Huber loss (quadratic for small errors, linear for large errors)
///   * "bisquare" - Tukey's bisquare loss (bounded influence function)
///   * "cauchy" - Cauchy loss (smoother than Huber)
/// * `tuning` - Optional tuning parameter for the loss function
///
/// # Returns
///
/// * The M-estimator value
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::m_estimator;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0, 10.0]; // 10.0 is an outlier
/// let y_pred = array![2.5, 0.0, 2.0, 8.0, 5.0];
///
/// // Standard MSE is heavily influenced by the outlier
/// // Let's use a robust M-estimator instead
/// let m_est = m_estimator(&y_true, &y_pred, "huber", None).unwrap();
/// assert!(m_est >= 0.0);
/// ```
pub fn m_estimator<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    loss_function: &str,
    tuning: Option<F>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + FromPrimitive,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    let n_samples = y_true.len();

    // Calculate residuals
    let mut residuals = Vec::with_capacity(n_samples);
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        residuals.push(*yt - *yp);
    }

    // Get robust weights
    let weights =
        compute_robust_weights(&Array1::from_vec(residuals.clone()), loss_function, tuning)?;

    // Calculate weighted loss
    let mut loss_sum = F::zero();
    let mut weight_sum = F::zero();

    for (i, &residual) in residuals.iter().enumerate() {
        let w = weights[i];
        loss_sum = loss_sum + w * residual * residual;
        weight_sum = weight_sum + w;
    }

    if weight_sum <= F::epsilon() {
        return Err(MetricsError::InvalidInput(
            "Sum of weights is zero".to_string(),
        ));
    }

    Ok(loss_sum / weight_sum)
}
