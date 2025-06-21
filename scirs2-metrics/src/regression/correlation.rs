//! Correlation metrics for regression models
//!
//! This module provides functions for calculating correlation-based metrics
//! between predicted values and true values in regression models.

use ndarray::{ArrayBase, Data, Dimension};
use num_traits::{Float, FromPrimitive, NumCast};

use super::{check_same_shape, mean};
use crate::error::{MetricsError, Result};

/// Calculates the R² score (coefficient of determination)
///
/// # Mathematical Formulation
///
/// The R² score is defined as:
///
/// ```text
/// R² = 1 - (SS_res / SS_tot)
/// ```
///
/// Where:
/// - SS_res = Σ(yᵢ - ŷᵢ)² (sum of squares of residuals)
/// - SS_tot = Σ(yᵢ - ȳ)² (total sum of squares)
/// - ȳ = mean of true values
/// - n = number of samples
///
/// Alternatively, R² can be expressed as:
///
/// ```text
/// R² = 1 - (Var(y_true - y_pred) / Var(y_true))
/// ```
///
/// This shows R² as the proportion of variance explained by the model.
///
/// # Interpretation
///
/// R² represents the proportion of variance in the dependent variable
/// that is predictable from the independent variable(s):
///
/// - R² = 1.0: Perfect predictions (all variance explained)
/// - R² = 0.0: Model performs as well as predicting the mean
/// - R² < 0.0: Model performs worse than predicting the mean
/// - R² = 0.5: Model explains 50% of the variance
///
/// # Range and Properties
///
/// - Maximum value: 1.0 (perfect fit)
/// - No minimum value (can be arbitrarily negative)
/// - Scale-invariant (unitless metric)
/// - Not necessarily increasing with more features (unlike adjusted R²)
///
/// # Relationship to Correlation
///
/// For simple linear regression:
/// ```text
/// R² = r²
/// ```
/// Where r is the Pearson correlation coefficient between y_true and y_pred.
///
/// # Use Cases
///
/// R² is widely used because:
/// - It provides an intuitive interpretation (% variance explained)
/// - It's scale-invariant and unitless
/// - It's standard in statistical modeling
/// - It allows comparison between different models
///
/// # Limitations
///
/// - Can be misleading with non-linear relationships
/// - Always increases with additional features (use adjusted R² instead)
/// - May not reflect prediction quality for extreme values
/// - Assumes linear relationship between features and target
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The R² score
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
/// assert!(r2 > 0.9);
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
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    let _n_samples = y_true.len();

    // Calculate mean of y_true
    let y_mean = mean(y_true);

    let mut ss_total = F::zero();
    let mut ss_residual = F::zero();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        // Sum of squares of residuals
        let residual = *yt - *yp;
        ss_residual = ss_residual + residual * residual;

        // Total sum of squares
        let deviation = *yt - y_mean;
        ss_total = ss_total + deviation * deviation;
    }

    if ss_total < F::epsilon() {
        return Err(MetricsError::InvalidInput(
            "Total sum of squares is zero. R^2 score is not defined when all true values are identical.".to_string(),
        ));
    }

    Ok(F::one() - ss_residual / ss_total)
}

/// Calculates the adjusted R^2 score
///
/// Adjusted R^2 adjusts the R^2 score for the number of predictors in the model.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `n_features` - Number of features/predictors used in the model
///
/// # Returns
///
/// * The adjusted R^2 score
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::adjusted_r2_score;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
/// let n_features = 2; // Assume 2 features were used to make predictions
///
/// let adj_r2 = adjusted_r2_score(&y_true, &y_pred, n_features).unwrap();
/// assert!(adj_r2 < 1.0);
/// ```
pub fn adjusted_r2_score<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    n_features: usize,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + FromPrimitive,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    let n_samples = y_true.len();

    if n_samples <= n_features + 1 {
        return Err(MetricsError::InvalidInput(
            "Number of samples must be greater than number of features + 1".to_string(),
        ));
    }

    let r2 = r2_score(y_true, y_pred)?;

    let n: F = NumCast::from(n_samples).unwrap();
    let p = NumCast::from(n_features).unwrap();
    let one = F::one();

    // Adjusted R^2 formula: 1 - (1 - R^2) * (n - 1) / (n - p - 1)
    let numerator = (one - r2) * (n - one);
    let denominator = n - p - one;

    Ok(one - numerator / denominator)
}

/// Calculates the explained variance score
///
/// Explained variance measures the proportion to which a model accounts
/// for the variation in the target variable.
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
/// assert!(score > 0.9);
/// ```
pub fn explained_variance_score<F, S1, S2, D1, D2>(
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

    // Calculate means
    let y_true_mean = mean(y_true);
    let y_pred_mean = mean(y_pred);

    // Calculate variances and covariance
    let mut y_true_var = F::zero();
    let mut y_pred_var = F::zero();
    let mut covar = F::zero();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let true_dev = *yt - y_true_mean;
        let pred_dev = *yp - y_pred_mean;

        y_true_var = y_true_var + true_dev * true_dev;
        y_pred_var = y_pred_var + pred_dev * pred_dev;
        covar = covar + true_dev * pred_dev;
    }

    y_true_var = y_true_var / NumCast::from(n_samples).unwrap();
    covar = covar / NumCast::from(n_samples).unwrap();

    if y_true_var < F::epsilon() {
        return Err(MetricsError::InvalidInput(
            "Variance of y_true is zero. Explained variance score is not defined when all true values are identical.".to_string(),
        ));
    }

    // Calculate explained variance
    let explained_var = F::one() - (y_true_var - covar) / y_true_var;

    // Clamp to [0, 1] range in case of numerical issues
    Ok(explained_var.max(F::zero()).min(F::one()))
}

/// Calculates the Pearson correlation coefficient between true and predicted values
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The Pearson correlation coefficient
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::pearson_correlation;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let corr = pearson_correlation(&y_true, &y_pred).unwrap();
/// assert!(corr > 0.9);
/// ```
pub fn pearson_correlation<F, S1, S2, D1, D2>(
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

    // Calculate means
    let y_true_mean = mean(y_true);
    let y_pred_mean = mean(y_pred);

    // Calculate numerator and denominators
    let mut numerator = F::zero();
    let mut denom_true = F::zero();
    let mut denom_pred = F::zero();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let true_dev = *yt - y_true_mean;
        let pred_dev = *yp - y_pred_mean;

        numerator = numerator + true_dev * pred_dev;
        denom_true = denom_true + true_dev * true_dev;
        denom_pred = denom_pred + pred_dev * pred_dev;
    }

    if denom_true < F::epsilon() || denom_pred < F::epsilon() {
        return Err(MetricsError::InvalidInput(
            "Pearson correlation is not defined when variance of either variable is zero."
                .to_string(),
        ));
    }

    let correlation = numerator / (denom_true.sqrt() * denom_pred.sqrt());

    // Clamp to [-1, 1] range in case of numerical issues
    Ok(correlation.max(-F::one()).min(F::one()))
}

/// Calculates the Spearman rank correlation coefficient between true and predicted values
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The Spearman rank correlation coefficient
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::spearman_correlation;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let corr = spearman_correlation(&y_true, &y_pred).unwrap();
/// assert!(corr > 0.9);
/// ```
pub fn spearman_correlation<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
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

    let _n_samples = y_true.len();

    // Convert to ranks
    let y_true_ranks = compute_ranks(y_true)?;
    let y_pred_ranks = compute_ranks(y_pred)?;

    // Use Pearson correlation on the ranks
    pearson_correlation(&y_true_ranks, &y_pred_ranks)
}

/// Helper function to compute ranks of an array
fn compute_ranks<F, S, D>(x: &ArrayBase<S, D>) -> Result<ndarray::Array1<F>>
where
    F: Float + NumCast + std::fmt::Debug + FromPrimitive,
    S: Data<Elem = F>,
    D: Dimension,
{
    let n = x.len();

    // Create array of (value, original index) pairs
    let mut pairs: Vec<(F, usize)> = x.iter().cloned().zip(0..n).collect();

    // Sort by value
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks (handle ties by averaging)
    let mut ranks = vec![F::zero(); n];
    let mut i = 0;
    while i < n {
        let val = pairs[i].0;
        let mut j = i + 1;
        while j < n && pairs[j].0 == val {
            j += 1;
        }

        // Found a group of equal values from i to j-1
        let rank_val = F::from(i + j - 1).unwrap() / NumCast::from(2).unwrap() + F::one();
        for k in i..j {
            ranks[pairs[k].1] = rank_val;
        }

        i = j;
    }

    Ok(ndarray::Array1::from_vec(ranks))
}

/// Calculates the concordance correlation coefficient (CCC)
///
/// The concordance correlation coefficient measures the agreement between
/// two variables and is a measure of both precision and accuracy.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The concordance correlation coefficient
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::concordance_correlation;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let ccc = concordance_correlation(&y_true, &y_pred).unwrap();
/// assert!(ccc > 0.9);
/// ```
pub fn concordance_correlation<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
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

    // Calculate means
    let y_true_mean = mean(y_true);
    let y_pred_mean = mean(y_pred);

    // Calculate variances and covariance
    let mut y_true_var = F::zero();
    let mut y_pred_var = F::zero();
    let mut covar = F::zero();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let true_dev = *yt - y_true_mean;
        let pred_dev = *yp - y_pred_mean;

        y_true_var = y_true_var + true_dev * true_dev;
        y_pred_var = y_pred_var + pred_dev * pred_dev;
        covar = covar + true_dev * pred_dev;
    }

    let n = NumCast::from(y_true.len()).unwrap();
    y_true_var = y_true_var / n;
    y_pred_var = y_pred_var / n;
    covar = covar / n;

    if y_true_var < F::epsilon() || y_pred_var < F::epsilon() {
        return Err(MetricsError::InvalidInput(
            "Concordance correlation is not defined when variance of either variable is zero."
                .to_string(),
        ));
    }

    let sd_true = y_true_var.sqrt();
    let sd_pred = y_pred_var.sqrt();

    // Pearson correlation
    let corr = covar / (sd_true * sd_pred);

    // Scale shift
    let scale_shift = (F::from(2.0).unwrap() * covar)
        / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean).powi(2));

    Ok(corr * scale_shift)
}
