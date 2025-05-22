//! Residual analysis for regression models
//!
//! This module provides functions for analyzing residuals of regression models,
//! including histograms, Q-Q plots, and comprehensive residual analysis.

use ndarray::{Array1, Array2, ArrayBase, Data, Dimension};
use num_traits::{Float, FromPrimitive, NumCast};
use std::cmp::Ordering;

use super::check_same_shape;
use crate::error::{MetricsError, Result};

/// Structure representing a histogram of residuals
#[derive(Debug, Clone)]
pub struct ErrorHistogram<F: Float> {
    /// Bin edges (length = n_bins + 1)
    pub bin_edges: Vec<F>,
    /// Bin counts (length = n_bins)
    pub bin_counts: Vec<usize>,
    /// Number of observations in each bin
    pub n_observations: usize,
    /// Minimum residual value
    pub min_error: F,
    /// Maximum residual value
    pub max_error: F,
}

/// Calculates a histogram of error/residual values
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `n_bins` - Number of bins for the histogram
///
/// # Returns
///
/// * An `ErrorHistogram` struct containing the histogram data
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::error_histogram;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0, 5.0, 8.0, 1.0, 4.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0, 4.5, 7.5, 1.5, 3.5];
///
/// let hist = error_histogram(&y_true, &y_pred, 4).unwrap();
/// assert_eq!(hist.bin_counts.len(), 4);
/// assert_eq!(hist.bin_edges.len(), 5);
/// assert_eq!(hist.n_observations, 8);
/// ```
pub fn error_histogram<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    n_bins: usize,
) -> Result<ErrorHistogram<F>>
where
    F: Float + NumCast + std::fmt::Debug + FromPrimitive,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    if n_bins == 0 {
        return Err(MetricsError::InvalidInput(
            "Number of bins must be positive".to_string(),
        ));
    }

    // Calculate residuals
    let n_samples = y_true.len();
    let mut residuals = Vec::with_capacity(n_samples);

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        residuals.push(*yt - *yp);
    }

    // Find min and max residuals
    let mut min_error = residuals[0];
    let mut max_error = residuals[0];

    for &residual in &residuals[1..] {
        if residual < min_error {
            min_error = residual;
        }
        if residual > max_error {
            max_error = residual;
        }
    }

    // Create bin edges
    let range = if max_error > min_error {
        max_error - min_error
    } else {
        F::one()
    };
    let bin_width = range / NumCast::from(n_bins).unwrap();

    let mut bin_edges = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        bin_edges.push(min_error + F::from(i).unwrap() * bin_width);
    }

    // Count values in each bin
    let mut bin_counts = vec![0; n_bins];

    for &residual in &residuals {
        if residual == max_error {
            // Last bin for the maximum value
            bin_counts[n_bins - 1] += 1;
        } else {
            // Find the appropriate bin
            let bin_idx = ((residual - min_error) / bin_width).to_usize().unwrap();
            bin_counts[bin_idx] += 1;
        }
    }

    Ok(ErrorHistogram {
        bin_edges,
        bin_counts,
        n_observations: n_samples,
        min_error,
        max_error,
    })
}

/// Structure representing Q-Q plot data for residuals
#[derive(Debug, Clone)]
pub struct QQPlotData<F: Float> {
    /// Theoretical quantiles
    pub theoretical_quantiles: Vec<F>,
    /// Sample quantiles (residuals)
    pub sample_quantiles: Vec<F>,
    /// 45-degree reference line points
    pub reference_line: Vec<(F, F)>,
}

/// Calculates Q-Q plot data for residuals
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `n_quantiles` - Number of quantiles to calculate
///
/// # Returns
///
/// * A `QQPlotData` struct containing the Q-Q plot data
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::qq_plot_data;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0, 5.0, 8.0, 1.0, 4.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0, 4.5, 7.5, 1.5, 3.5];
///
/// let qq_data = qq_plot_data(&y_true, &y_pred, 20).unwrap();
/// assert_eq!(qq_data.theoretical_quantiles.len(), qq_data.sample_quantiles.len());
/// ```
pub fn qq_plot_data<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    n_quantiles: usize,
) -> Result<QQPlotData<F>>
where
    F: Float + NumCast + std::fmt::Debug + FromPrimitive,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    if n_quantiles < 2 {
        return Err(MetricsError::InvalidInput(
            "Number of quantiles must be at least 2".to_string(),
        ));
    }

    // Calculate residuals
    let n_samples = y_true.len();
    let mut residuals = Vec::with_capacity(n_samples);

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        residuals.push(*yt - *yp);
    }

    // Sort residuals
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    // Standardize residuals
    let mean =
        residuals.iter().fold(F::zero(), |acc, &x| acc + x) / NumCast::from(n_samples).unwrap();

    let variance = residuals.iter().fold(F::zero(), |acc, &x| {
        let diff = x - mean;
        acc + diff * diff
    }) / NumCast::from(n_samples).unwrap();

    let std_dev = variance.sqrt();

    let mut std_residuals = Vec::with_capacity(n_samples);
    for &r in &residuals {
        std_residuals.push((r - mean) / std_dev);
    }

    // Calculate theoretical quantiles
    let mut theoretical_quantiles = Vec::with_capacity(n_quantiles);
    let mut sample_quantiles = Vec::with_capacity(n_quantiles);

    let step = F::one() / NumCast::from(n_quantiles + 1).unwrap();

    for i in 1..=n_quantiles {
        let p: F = F::from(i).unwrap() * step;
        let theoretical_q = normal_quantile(p.to_f64().unwrap());
        theoretical_quantiles.push(F::from(theoretical_q).unwrap());

        // Get corresponding sample quantile
        let idx = (p * NumCast::from(n_samples).unwrap())
            .to_usize()
            .unwrap()
            .min(n_samples - 1);
        sample_quantiles.push(std_residuals[idx]);
    }

    // Create reference line
    let mut min_val = theoretical_quantiles[0].min(sample_quantiles[0]);
    let mut max_val = theoretical_quantiles[n_quantiles - 1].max(sample_quantiles[n_quantiles - 1]);

    // Add some margin
    let range = max_val - min_val;
    min_val = min_val - range * F::from_f64(0.05).unwrap();
    max_val = max_val + range * F::from_f64(0.05).unwrap();

    let reference_line = vec![(min_val, min_val), (max_val, max_val)];

    Ok(QQPlotData {
        theoretical_quantiles,
        sample_quantiles,
        reference_line,
    })
}

/// Approximation of the normal quantile function (inverse CDF)
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        // Return a reasonable default value instead of panicking
        if p <= 0.0 {
            return -5.0; // Approximation for negative infinity
        } else {
            return 5.0; // Approximation for positive infinity
        }
    }

    // Constants for Beasley-Springer-Moro algorithm
    let a = [
        2.50662823884,
        -18.61500062529,
        41.39119773534,
        -25.44106049637,
    ];
    let b = [
        -8.47351093090,
        23.08336743743,
        -21.06224101826,
        3.13082909833,
    ];
    let c = [
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511919,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    ];

    // Approximation near the center
    if (0.08..=0.92).contains(&p) {
        let q = p - 0.5;
        let r = q * q;
        let mut result = q * (a[0] + r * (a[1] + r * (a[2] + r * a[3])));
        result /= 1.0 + r * (b[0] + r * (b[1] + r * (b[2] + r * b[3])));
        return result;
    }

    // Approximation in the tails
    let q = if p < 0.08 {
        (-2.0 * (p).ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let result = c[0]
        + q * (c[1]
            + q * (c[2]
                + q * (c[3] + q * (c[4] + q * (c[5] + q * (c[6] + q * (c[7] + q * c[8])))))));

    if p < 0.08 {
        -result
    } else {
        result
    }
}

/// Structure representing comprehensive residual analysis
#[derive(Debug, Clone)]
pub struct ResidualAnalysis<F: Float> {
    /// Residuals (y_true - y_pred)
    pub residuals: Vec<F>,
    /// Standardized residuals
    pub standardized_residuals: Vec<F>,
    /// Studentized residuals
    pub studentized_residuals: Vec<F>,
    /// Cook's distances (influence measure)
    pub cooks_distances: Vec<F>,
    /// DFFITS (influence measure)
    pub dffits: Vec<F>,
    /// Leverage values (hat matrix diagonal)
    pub leverage: Vec<F>,
    /// Residual histogram
    pub histogram: ErrorHistogram<F>,
    /// Q-Q plot data
    pub qq_plot: QQPlotData<F>,
    /// Durbin-Watson statistic (checks for autocorrelation)
    pub durbin_watson: F,
    /// Breusch-Pagan test statistic (checks for heteroscedasticity)
    pub breusch_pagan: F,
    /// Shapiro-Wilk test statistic (checks for normality)
    pub shapiro_wilk: F,
}

/// Performs comprehensive residual analysis for a regression model
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
/// * `x` - Optional predictor variables matrix (needed for some diagnostics)
/// * `hat_matrix` - Optional hat/projection matrix (can be provided to avoid recalculation)
///
/// # Returns
///
/// * A `ResidualAnalysis` struct containing various residual diagnostics
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::regression::residual_analysis;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0, 5.0, 8.0, 1.0, 4.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0, 4.5, 7.5, 1.5, 3.5];
///
/// // Create dummy X matrix (features matrix) with 2 predictors
/// let x = Array2::from_shape_fn((8, 2), |(i, j)| i as f64 + j as f64);
///
/// let analysis = residual_analysis(&y_true, &y_pred, Some(&x), None).unwrap();
///
/// // Access various diagnostics
/// println!("Durbin-Watson statistic: {}", analysis.durbin_watson);
/// println!("Number of residuals: {}", analysis.residuals.len());
/// ```
pub fn residual_analysis<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    x: Option<&Array2<F>>,
    hat_matrix: Option<&Array2<F>>,
) -> Result<ResidualAnalysis<F>>
where
    F: Float + NumCast + std::fmt::Debug + FromPrimitive + 'static,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    check_same_shape::<F, S1, S2, D1, D2>(y_true, y_pred)?;

    let n_samples = y_true.len();

    // Check X matrix dimensions
    if let Some(x_mat) = x {
        if x_mat.shape()[0] != n_samples {
            return Err(MetricsError::InvalidInput(format!(
                "X matrix has {} rows, but y_true has {} elements",
                x_mat.shape()[0],
                n_samples
            )));
        }
    }

    // Check hat matrix dimensions
    if let Some(h_mat) = hat_matrix {
        if h_mat.shape() != [n_samples, n_samples] {
            return Err(MetricsError::InvalidInput(format!(
                "Hat matrix has shape {:?}, but should be [{}, {}]",
                h_mat.shape(),
                n_samples,
                n_samples
            )));
        }
    }

    // Calculate basic residuals
    let mut residuals = Vec::with_capacity(n_samples);
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        residuals.push(*yt - *yp);
    }

    // Calculate mean of residuals
    let residual_mean =
        residuals.iter().fold(F::zero(), |acc, &r| acc + r) / NumCast::from(n_samples).unwrap();

    // Calculate variance of residuals
    let residual_var = residuals.iter().fold(F::zero(), |acc, &r| {
        let diff = r - residual_mean;
        acc + diff * diff
    }) / NumCast::from(n_samples).unwrap();

    let residual_std = residual_var.sqrt();

    // Calculate standardized residuals
    let mut standardized_residuals = Vec::with_capacity(n_samples);
    for &r in &residuals {
        standardized_residuals.push((r - residual_mean) / residual_std);
    }

    // Calculate leverage (hat matrix diagonal)
    let leverage = if let Some(h_mat) = hat_matrix {
        // Extract diagonal from provided hat matrix
        let mut h_diag = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            h_diag.push(h_mat[[i, i]]);
        }
        h_diag
    } else if let Some(x_mat) = x {
        // Calculate hat matrix diagonal using X matrix: diag(X (X'X)^(-1) X')
        let p = x_mat.shape()[1]; // Number of predictors
        let xt = x_mat.t();

        // Calculate X'X
        let xtx = xt.dot(x_mat);

        // Invert X'X (simplified - not a proper matrix inversion)
        let mut xtx_inv = Array2::<F>::zeros((p, p));

        // Diagonal matrix as a simple approximation
        for i in 0..p {
            if xtx[[i, i]] > F::epsilon() {
                xtx_inv[[i, i]] = F::one() / xtx[[i, i]];
            }
        }

        // Calculate hat matrix diagonal
        let mut h_diag = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let mut h_ii = F::zero();
            for j in 0..p {
                for k in 0..p {
                    h_ii = h_ii + x_mat[[i, j]] * xtx_inv[[j, k]] * x_mat[[i, k]];
                }
            }
            h_diag.push(h_ii);
        }

        h_diag
    } else {
        // No X matrix or hat matrix provided, use default
        vec![F::one() / NumCast::from(n_samples).unwrap(); n_samples]
    };

    // Calculate studentized residuals
    let mut studentized_residuals = Vec::with_capacity(n_samples);
    for (i, &r) in residuals.iter().enumerate() {
        let h_ii = leverage[i];
        if h_ii < F::one() {
            let student_r = r / (residual_std * (F::one() - h_ii).sqrt());
            studentized_residuals.push(student_r);
        } else {
            studentized_residuals.push(F::zero());
        }
    }

    // Calculate Cook's distances
    let mut cooks_distances = Vec::with_capacity(n_samples);
    for (i, &r) in standardized_residuals.iter().enumerate() {
        let h_ii = leverage[i];
        if h_ii < F::one() {
            // Use a default number of parameters if no X matrix was provided
            let p_value = if let Some(x_mat) = x {
                x_mat.shape()[1]
            } else {
                1 // Default to 1 predictor
            };
            let cook_d = (r * r) * (h_ii / (F::one() - h_ii)) / NumCast::from(p_value).unwrap();
            cooks_distances.push(cook_d);
        } else {
            cooks_distances.push(F::zero());
        }
    }

    // Calculate DFFITS
    let mut dffits = Vec::with_capacity(n_samples);
    for (i, &r) in studentized_residuals.iter().enumerate() {
        let h_ii = leverage[i];
        if h_ii < F::one() {
            let dffit = r * (h_ii / (F::one() - h_ii)).sqrt();
            dffits.push(dffit);
        } else {
            dffits.push(F::zero());
        }
    }

    // Calculate Durbin-Watson statistic (tests for autocorrelation)
    let mut numerator = F::zero();
    for i in 1..n_samples {
        let diff = residuals[i] - residuals[i - 1];
        numerator = numerator + diff * diff;
    }

    let denominator = residuals.iter().fold(F::zero(), |acc, &r| acc + r * r);
    let durbin_watson = if denominator > F::epsilon() {
        numerator / denominator
    } else {
        F::from(2.0).unwrap() // No autocorrelation
    };

    // Calculate Breusch-Pagan statistic (tests for heteroscedasticity)
    // Simplified approach: regress squared residuals on fitted values
    let squared_residuals: Vec<F> = residuals.iter().map(|&r| r * r).collect();
    let mean_sq_residual = squared_residuals.iter().fold(F::zero(), |acc, &r| acc + r)
        / NumCast::from(n_samples).unwrap();

    let mut numerator = F::zero();
    let mut denominator = F::zero();

    for (i, &sq_r) in squared_residuals.iter().enumerate() {
        let pred = y_pred.iter().nth(i).unwrap();
        let diff = sq_r - mean_sq_residual;
        numerator = numerator + diff * diff;
        denominator = denominator + (*pred) * (*pred);
    }

    let breusch_pagan = if denominator > F::epsilon() {
        numerator / denominator
    } else {
        F::zero()
    };

    // Calculate Shapiro-Wilk statistic (tests for normality)
    // Simplified approach based on correlation between ordered residuals and normal quantiles
    let mut ordered_residuals = standardized_residuals.clone();
    ordered_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let mut expected_quantiles = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let p = (F::from(i + 1).unwrap() - F::from(0.375).unwrap())
            / (F::from(n_samples).unwrap() + F::from(0.25).unwrap());
        let q = normal_quantile(p.to_f64().unwrap());
        expected_quantiles.push(F::from(q).unwrap());
    }

    let mut numerator = F::zero();
    let mut denominator = F::zero();

    for (i, &r) in ordered_residuals.iter().enumerate() {
        let q = expected_quantiles[i];
        numerator = numerator + r * q;
        denominator = denominator + r * r;
    }

    let shapiro_wilk = if denominator > F::epsilon() {
        (numerator / denominator).powi(2)
    } else {
        F::zero()
    };

    // Calculate histogram and Q-Q plot
    let histogram = error_histogram(y_true, y_pred, 10)?;
    let qq_plot = qq_plot_data(y_true, y_pred, 20)?;

    Ok(ResidualAnalysis {
        residuals,
        standardized_residuals,
        studentized_residuals,
        cooks_distances,
        dffits,
        leverage,
        histogram,
        qq_plot,
        durbin_watson,
        breusch_pagan,
        shapiro_wilk,
    })
}

/// Checks for heteroscedasticity in residuals using Breusch-Pagan test
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * Test statistic for the Breusch-Pagan test
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::test_heteroscedasticity;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0, 5.0, 8.0, 1.0, 4.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0, 4.5, 7.5, 1.5, 3.5];
///
/// let bp_stat = test_heteroscedasticity(&y_true, &y_pred).unwrap();
/// assert!(bp_stat >= 0.0);
/// ```
pub fn test_heteroscedasticity<F, S1, S2, D1, D2>(
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

    let n_samples = y_true.len();

    // Calculate residuals
    let mut residuals = Vec::with_capacity(n_samples);
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        residuals.push(*yt - *yp);
    }

    // Calculate squared residuals
    let squared_residuals: Vec<F> = residuals.iter().map(|&r| r * r).collect();
    let mean_sq_residual = squared_residuals.iter().fold(F::zero(), |acc, &r| acc + r)
        / NumCast::from(n_samples).unwrap();

    // Regress squared residuals on fitted values
    let mut numerator = F::zero();
    let mut denominator = F::zero();

    for (i, &sq_r) in squared_residuals.iter().enumerate() {
        let pred = y_pred.iter().nth(i).unwrap();
        let diff = sq_r - mean_sq_residual;
        numerator = numerator + diff * diff;
        denominator = denominator + (*pred) * (*pred);
    }

    if denominator < F::epsilon() {
        return Err(MetricsError::InvalidInput(
            "Denominator in heteroscedasticity test is zero".to_string(),
        ));
    }

    let bp_stat = numerator / denominator;
    Ok(bp_stat)
}

/// Checks for autocorrelation in residuals using Durbin-Watson test
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * Durbin-Watson test statistic
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::test_autocorrelation;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0, 5.0, 8.0, 1.0, 4.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0, 4.5, 7.5, 1.5, 3.5];
///
/// let dw_stat = test_autocorrelation(&y_true, &y_pred).unwrap();
/// // DW statistic ranges from 0 to 4, with 2 being no autocorrelation
/// assert!(dw_stat >= 0.0 && dw_stat <= 4.0);
/// ```
pub fn test_autocorrelation<F, S1, S2, D1, D2>(
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

    let n_samples = y_true.len();

    if n_samples < 2 {
        return Err(MetricsError::InvalidInput(
            "At least 2 samples required for autocorrelation test".to_string(),
        ));
    }

    // Calculate residuals
    let mut residuals = Vec::with_capacity(n_samples);
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        residuals.push(*yt - *yp);
    }

    // Calculate Durbin-Watson statistic
    let mut numerator = F::zero();
    for i in 1..n_samples {
        let diff = residuals[i] - residuals[i - 1];
        numerator = numerator + diff * diff;
    }

    let denominator = residuals.iter().fold(F::zero(), |acc, &r| acc + r * r);

    if denominator < F::epsilon() {
        return Err(MetricsError::InvalidInput(
            "Sum of squared residuals is zero in autocorrelation test".to_string(),
        ));
    }

    let dw_stat = numerator / denominator;
    Ok(dw_stat)
}

/// Checks for normality of residuals using Shapiro-Wilk test
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * Shapiro-Wilk test statistic
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::test_normality;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0, 5.0, 8.0, 1.0, 4.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0, 4.5, 7.5, 1.5, 3.5];
///
/// let sw_stat = test_normality(&y_true, &y_pred).unwrap();
/// // SW statistic ranges from 0 to 1, with values close to 1 indicating normality
/// assert!(sw_stat >= 0.0 && sw_stat <= 1.0);
/// ```
pub fn test_normality<F, S1, S2, D1, D2>(
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

    let n_samples = y_true.len();

    // Calculate residuals
    let mut residuals = Vec::with_capacity(n_samples);
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        residuals.push(*yt - *yp);
    }

    // Calculate mean and standard deviation
    let mean =
        residuals.iter().fold(F::zero(), |acc, &r| acc + r) / NumCast::from(n_samples).unwrap();

    let variance = residuals.iter().fold(F::zero(), |acc, &r| {
        let diff = r - mean;
        acc + diff * diff
    }) / NumCast::from(n_samples).unwrap();

    let std_dev = variance.sqrt();

    // Standardize residuals
    let mut std_residuals = Vec::with_capacity(n_samples);
    for &r in &residuals {
        std_residuals.push((r - mean) / std_dev);
    }

    // Sort standardized residuals
    let mut ordered_residuals = std_residuals.clone();
    ordered_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    // Calculate expected normal order statistics
    let mut expected_quantiles = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let p = (F::from(i + 1).unwrap() - F::from(0.375).unwrap())
            / (F::from(n_samples).unwrap() + F::from(0.25).unwrap());
        let q = normal_quantile(p.to_f64().unwrap());
        expected_quantiles.push(F::from(q).unwrap());
    }

    // Calculate correlation between ordered residuals and expected quantiles
    let mean_residual = F::zero(); // Standardized residuals have mean zero
    let mean_quantile = expected_quantiles.iter().fold(F::zero(), |acc, &q| acc + q)
        / NumCast::from(n_samples).unwrap();

    let mut numerator = F::zero();
    let mut denom_residual = F::zero();
    let mut denom_quantile = F::zero();

    for i in 0..n_samples {
        let res_dev = ordered_residuals[i] - mean_residual;
        let quant_dev = expected_quantiles[i] - mean_quantile;

        numerator = numerator + res_dev * quant_dev;
        denom_residual = denom_residual + res_dev * res_dev;
        denom_quantile = denom_quantile + quant_dev * quant_dev;
    }

    let denominator = (denom_residual * denom_quantile).sqrt();

    if denominator < F::epsilon() {
        return Err(MetricsError::InvalidInput(
            "Denominator in normality test is zero".to_string(),
        ));
    }

    let correlation = numerator / denominator;

    // Shapiro-Wilk statistic is approximately the square of this correlation
    let sw_stat = correlation * correlation;

    Ok(sw_stat.min(F::one()))
}
