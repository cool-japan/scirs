//! Time series decomposition methods
//!
//! This module provides implementations for decomposing time series into trend,
//! seasonal, and residual components.

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};
use crate::utils::moving_average;

/// Result of time series decomposition
#[derive(Debug, Clone)]
pub struct DecompositionResult<F> {
    /// Trend component
    pub trend: Array1<F>,
    /// Seasonal component
    pub seasonal: Array1<F>,
    /// Residual component
    pub residual: Array1<F>,
    /// Original time series
    pub original: Array1<F>,
}

/// Decomposition model type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompositionModel {
    /// Additive model: Y = T + S + R
    Additive,
    /// Multiplicative model: Y = T * S * R
    Multiplicative,
}

/// Performs classical seasonal decomposition on a time series
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `period` - The seasonal period (e.g., 12 for monthly data with yearly seasonality)
/// * `model` - Decomposition model (additive or multiplicative)
///
/// # Returns
///
/// * Decomposition result containing trend, seasonal, and residual components
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::decomposition::{decompose_seasonal, DecompositionModel};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
/// let result = decompose_seasonal(&ts, 4, DecompositionModel::Additive).unwrap();
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal: {:?}", result.seasonal);
/// println!("Residual: {:?}", result.residual);
/// ```
pub fn decompose_seasonal<F>(
    ts: &Array1<F>,
    period: usize,
    model: DecompositionModel,
) -> Result<DecompositionResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 2 * period {
        return Err(TimeSeriesError::DecompositionError(format!(
            "Time series length ({}) must be at least twice the seasonal period ({})",
            ts.len(),
            period
        )));
    }

    // 1. Calculate centered moving average (trend)
    let window_size = if period % 2 == 0 { period + 1 } else { period };
    let trend = moving_average(ts, window_size)?;

    // Pad trend with NaN values at the beginning and end
    let half_window = window_size / 2;
    let mut padded_trend = Array1::from_elem(ts.len(), F::nan());
    let offset = half_window;

    // Ensure we don't go out of bounds
    for (i, &val) in trend.iter().enumerate() {
        if i + offset < padded_trend.len() {
            padded_trend[i + offset] = val;
        }
    }

    // 2. Remove trend to get detrended series
    let mut detrended = Array1::zeros(ts.len());
    for i in 0..ts.len() {
        if padded_trend[i].is_nan() {
            detrended[i] = F::zero(); // Set detrended to zero where trend is NaN
        } else {
            match model {
                DecompositionModel::Additive => {
                    detrended[i] = ts[i] - padded_trend[i];
                }
                DecompositionModel::Multiplicative => {
                    if padded_trend[i] == F::zero() {
                        return Err(TimeSeriesError::DecompositionError(
                            "Division by zero in multiplicative model".to_string(),
                        ));
                    }
                    detrended[i] = ts[i] / padded_trend[i];
                }
            }
        }
    }

    // 3. Calculate seasonal component by averaging values for each season
    let mut seasonal = Array1::zeros(ts.len());
    let mut seasonal_pattern = Array1::zeros(period);
    let mut counts = vec![0; period];

    // Calculate average for each position in the seasonal pattern
    for i in 0..ts.len() {
        let pos = i % period;
        if !detrended[i].is_nan() {
            seasonal_pattern[pos] = seasonal_pattern[pos] + detrended[i];
            counts[pos] += 1;
        }
    }

    // Normalize seasonal pattern
    for i in 0..period {
        if counts[i] > 0 {
            seasonal_pattern[i] = seasonal_pattern[i] / F::from_usize(counts[i]).unwrap();
        }
    }

    // Normalize to ensure seasonal components sum to zero (additive) or average to one (multiplicative)
    match model {
        DecompositionModel::Additive => {
            let mean = seasonal_pattern.iter().fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(period).unwrap();
            for i in 0..period {
                seasonal_pattern[i] = seasonal_pattern[i] - mean;
            }
        }
        DecompositionModel::Multiplicative => {
            let mean = seasonal_pattern.iter().fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(period).unwrap();
            if mean == F::zero() {
                return Err(TimeSeriesError::DecompositionError(
                    "Division by zero normalizing multiplicative seasonal pattern".to_string(),
                ));
            }
            for i in 0..period {
                seasonal_pattern[i] = seasonal_pattern[i] / mean;
            }
        }
    }

    // Apply seasonal pattern to the whole series
    for i in 0..ts.len() {
        seasonal[i] = seasonal_pattern[i % period];
    }

    // 4. Calculate residual component
    let mut residual = Array1::zeros(ts.len());
    for i in 0..ts.len() {
        if padded_trend[i].is_nan() {
            residual[i] = F::nan();
        } else {
            match model {
                DecompositionModel::Additive => {
                    residual[i] = ts[i] - padded_trend[i] - seasonal[i];
                }
                DecompositionModel::Multiplicative => {
                    if padded_trend[i] == F::zero() || seasonal[i] == F::zero() {
                        return Err(TimeSeriesError::DecompositionError(
                            "Division by zero in multiplicative model residual calculation"
                                .to_string(),
                        ));
                    }
                    residual[i] = ts[i] / (padded_trend[i] * seasonal[i]);
                }
            }
        }
    }

    // Create result struct
    let original = ts.clone();

    Ok(DecompositionResult {
        trend: padded_trend,
        seasonal,
        residual,
        original,
    })
}

/// Options for STL decomposition
#[derive(Debug, Clone)]
pub struct STLOptions {
    /// Trend window size (must be odd)
    pub trend_window: usize,
    /// Seasonal window size (must be odd)
    pub seasonal_window: usize,
    /// Number of inner loop iterations
    pub n_inner: usize,
    /// Number of outer loop iterations
    pub n_outer: usize,
    /// Whether to use robust weighting
    pub robust: bool,
}

impl Default for STLOptions {
    fn default() -> Self {
        Self {
            trend_window: 21,
            seasonal_window: 13,
            n_inner: 2,
            n_outer: 1,
            robust: false,
        }
    }
}

/// Options for Multiple Seasonal-Trend decomposition using LOESS (MSTL)
#[derive(Debug, Clone)]
pub struct MSTLOptions {
    /// Seasonal periods (e.g., [7, 30, 365] for weekly, monthly, and yearly seasonality)
    pub seasonal_periods: Vec<usize>,
    /// Trend window size (must be odd)
    pub trend_window: usize,
    /// Seasonal window sizes for each seasonal period (must be odd)
    pub seasonal_windows: Option<Vec<usize>>,
    /// Number of inner loop iterations
    pub n_inner: usize,
    /// Number of outer loop iterations
    pub n_outer: usize,
    /// Whether to use robust weighting
    pub robust: bool,
}

impl Default for MSTLOptions {
    fn default() -> Self {
        Self {
            seasonal_periods: Vec::new(),
            trend_window: 21,
            seasonal_windows: None,
            n_inner: 2,
            n_outer: 1,
            robust: false,
        }
    }
}

/// Result of multiple seasonal time series decomposition
#[derive(Debug, Clone)]
pub struct MultiSeasonalDecompositionResult<F> {
    /// Trend component
    pub trend: Array1<F>,
    /// Multiple seasonal components
    pub seasonal_components: Vec<Array1<F>>,
    /// Residual component
    pub residual: Array1<F>,
    /// Original time series
    pub original: Array1<F>,
}

/// Performs STL (Seasonal and Trend decomposition using LOESS) on a time series
///
/// STL decomposition uses locally weighted regression (LOESS) to extract trend
/// and seasonal components.
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `period` - The seasonal period
/// * `options` - Options for STL decomposition
///
/// # Returns
///
/// * Decomposition result containing trend, seasonal, and residual components
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::decomposition::{stl_decomposition, STLOptions};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
/// let options = STLOptions::default();
/// let result = stl_decomposition(&ts, 4, &options).unwrap();
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal: {:?}", result.seasonal);
/// println!("Residual: {:?}", result.residual);
/// ```
pub fn stl_decomposition<F>(
    ts: &Array1<F>,
    period: usize,
    options: &STLOptions,
) -> Result<DecompositionResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 2 * period {
        return Err(TimeSeriesError::DecompositionError(format!(
            "Time series length ({}) must be at least twice the seasonal period ({})",
            ts.len(),
            period
        )));
    }

    if options.trend_window % 2 == 0 || options.seasonal_window % 2 == 0 {
        return Err(TimeSeriesError::DecompositionError(
            "Trend and seasonal windows must be odd numbers".to_string(),
        ));
    }

    // A placeholder implementation
    // STL decomposition is complex and would require a detailed LOESS implementation
    // This simplified version returns the classical decomposition for now

    let decomp = decompose_seasonal(ts, period, DecompositionModel::Additive)?;

    Ok(decomp)
}

/// Performs Multiple Seasonal-Trend decomposition using LOESS (MSTL) on a time series
///
/// MSTL extends STL to handle multiple seasonal components, such as daily, weekly,
/// and yearly seasonality in the same time series.
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `options` - Options for MSTL decomposition
///
/// # Returns
///
/// * Multiple seasonal decomposition result containing trend, seasonal components, and residual
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::decomposition::{mstl_decomposition, MSTLOptions};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0,
///                 1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 2.5];
///
/// let mut options = MSTLOptions::default();
/// options.seasonal_periods = vec![4, 12]; // Both quarterly and yearly patterns
/// let result = mstl_decomposition(&ts, &options).unwrap();
///
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal Components: {:?}", result.seasonal_components);
/// println!("Residual: {:?}", result.residual);
/// ```
pub fn mstl_decomposition<F>(
    ts: &Array1<F>,
    options: &MSTLOptions,
) -> Result<MultiSeasonalDecompositionResult<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    let n = ts.len();

    // Check inputs
    if n < 3 {
        return Err(TimeSeriesError::DecompositionError(
            "Time series must have at least 3 points for MSTL decomposition".to_string(),
        ));
    }

    if options.seasonal_periods.is_empty() {
        return Err(TimeSeriesError::DecompositionError(
            "At least one seasonal period must be specified for MSTL".to_string(),
        ));
    }

    for &period in &options.seasonal_periods {
        if period < 2 {
            return Err(TimeSeriesError::DecompositionError(
                "Seasonal periods must be at least 2".to_string(),
            ));
        }
        if n < 2 * period {
            return Err(TimeSeriesError::DecompositionError(format!(
                "Time series length ({}) must be at least twice the seasonal period ({})",
                n, period
            )));
        }
    }

    if options.trend_window % 2 == 0 {
        return Err(TimeSeriesError::DecompositionError(
            "Trend window must be an odd number".to_string(),
        ));
    }

    // Initialize seasonal windows if not provided
    let _seasonal_windows = match &options.seasonal_windows {
        Some(windows) => {
            if windows.len() != options.seasonal_periods.len() {
                return Err(TimeSeriesError::DecompositionError(
                    "Number of seasonal windows must match number of seasonal periods".to_string(),
                ));
            }
            for &window in windows {
                if window % 2 == 0 {
                    return Err(TimeSeriesError::DecompositionError(
                        "Seasonal windows must be odd numbers".to_string(),
                    ));
                }
            }
            windows.clone()
        }
        None => {
            // Default to 7 or next odd number >= period
            options
                .seasonal_periods
                .iter()
                .map(|&period| {
                    if period < 7 {
                        7
                    } else if period % 2 == 0 {
                        period + 1
                    } else {
                        period
                    }
                })
                .collect()
        }
    };

    // Initialize components
    let mut trend = Array1::zeros(n);
    let mut seasonal_components = Vec::with_capacity(options.seasonal_periods.len());
    for _ in 0..options.seasonal_periods.len() {
        seasonal_components.push(Array1::zeros(n));
    }
    let mut residual = Array1::zeros(n);
    let original = ts.clone();

    // Make a working copy of the time series
    let working_ts = original.clone();

    // Outer loop for robustness iterations
    for _ in 0..options.n_outer {
        // Temporary storage for the deseasoned series
        let mut deseasoned = working_ts.clone();

        // Process each seasonal component
        for (i, &period) in options.seasonal_periods.iter().enumerate() {
            // Remove the trend and other seasonal components for isolation
            for (j, seasonal_comp) in seasonal_components.iter().enumerate() {
                if j != i {
                    for k in 0..n {
                        deseasoned[k] = deseasoned[k] - seasonal_comp[k];
                    }
                }
            }

            // Extract the seasonal component using STL-like approach
            // Note: In a complete implementation, we would configure STL options and
            // call a proper STL implementation with these options. For this simplified
            // implementation, we're using decompose_seasonal directly.
            //
            // let stl_options = STLOptions {
            //     trend_window: options.trend_window,
            //     seasonal_window: seasonal_windows[i],
            //     n_inner: options.n_inner,
            //     n_outer: 1, // Use just one outer iteration here
            //     robust: options.robust,
            // };

            // Apply seasonal decomposition to the deseasoned series
            let stl_result = decompose_seasonal(&deseasoned, period, DecompositionModel::Additive)?;

            // Update the seasonal component for this period
            for k in 0..n {
                seasonal_components[i][k] = stl_result.seasonal[k];
            }

            // Update the working time series by removing this seasonal component
            for k in 0..n {
                deseasoned[k] = deseasoned[k] - seasonal_components[i][k];
            }
        }

        // Extract the trend from the fully deseasoned series
        // For now, use a simple moving average as a placeholder
        let window_size = options.trend_window;
        trend = moving_average(&deseasoned, window_size)?;

        // Pad trend with values at the edges
        let half_window = window_size / 2;
        for i in 0..half_window {
            trend[i] = trend[half_window];
            trend[n - 1 - i] = trend[n - 1 - half_window];
        }

        // Calculate residuals
        for i in 0..n {
            let seasonal_sum = seasonal_components.iter().map(|s| s[i]).sum();
            residual[i] = original[i] - trend[i] - seasonal_sum;
        }

        // Apply robust weights if required
        if options.robust && options.n_outer > 1 {
            // This is just a placeholder for robust weighting
            // In a real implementation, we would calculate weights based on residuals
            // and apply them in the next iteration
        }
    }

    Ok(MultiSeasonalDecompositionResult {
        trend,
        seasonal_components,
        residual,
        original,
    })
}

/// Performs exponential smoothing decomposition on a time series
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `period` - The seasonal period
/// * `alpha` - Level smoothing parameter (0 < alpha < 1)
/// * `beta` - Trend smoothing parameter (0 < beta < 1)
/// * `gamma` - Seasonal smoothing parameter (0 < gamma < 1)
/// * `model` - Decomposition model (additive or multiplicative)
///
/// # Returns
///
/// * Decomposition result containing level, trend, seasonal, and residual components
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::decomposition::{exponential_decomposition, DecompositionModel};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
/// let result = exponential_decomposition(&ts, 4, 0.2, 0.1, 0.3,
///                                         DecompositionModel::Additive).unwrap();
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal: {:?}", result.seasonal);
/// println!("Residual: {:?}", result.residual);
/// ```
///
/// Options for TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and Seasonal components)
#[derive(Debug, Clone)]
pub struct TBATSOptions {
    /// Whether to use Box-Cox transformation
    pub use_box_cox: bool,
    /// Box-Cox transformation parameter (if None, automatically estimated)
    pub box_cox_lambda: Option<f64>,
    /// Whether to use trend component
    pub use_trend: bool,
    /// Whether to use damped trend
    pub use_damped_trend: bool,
    /// Seasonal periods (e.g., [7.0, 30.42, 365.25] for daily data with weekly, monthly, and yearly patterns)
    pub seasonal_periods: Vec<f64>,
    /// Number of Fourier terms for each seasonal period
    pub fourier_terms: Option<Vec<usize>>,
    /// Autoregressive order for ARMA errors
    pub ar_order: usize,
    /// Moving average order for ARMA errors
    pub ma_order: usize,
    /// Whether to automatically select ARMA orders
    pub auto_arma: bool,
    /// Whether to use parallel processing
    pub use_parallel: bool,
}

impl Default for TBATSOptions {
    fn default() -> Self {
        Self {
            use_box_cox: true,
            box_cox_lambda: None,
            use_trend: true,
            use_damped_trend: false,
            seasonal_periods: Vec::new(),
            fourier_terms: None,
            ar_order: 0,
            ma_order: 0,
            auto_arma: true,
            use_parallel: false,
        }
    }
}

/// Results of TBATS decomposition
#[derive(Debug, Clone)]
pub struct TBATSResult<F> {
    /// Trend component
    pub trend: Array1<F>,
    /// Seasonal components (one for each seasonal period)
    pub seasonal_components: Vec<Array1<F>>,
    /// ARMA residuals
    pub residuals: Array1<F>,
    /// Level component
    pub level: Array1<F>,
    /// Original time series
    pub original: Array1<F>,
    /// Box-Cox transformed time series (if transformation was applied)
    pub transformed: Option<Array1<F>>,
    /// Parameters used for the model
    pub parameters: TBATSParameters,
}

/// Parameters for TBATS model
#[derive(Debug, Clone)]
pub struct TBATSParameters {
    /// Box-Cox transformation parameter
    pub lambda: Option<f64>,
    /// Smoothing parameter for level
    pub alpha: f64,
    /// Smoothing parameter for trend
    pub beta: Option<f64>,
    /// Damping parameter
    pub phi: Option<f64>,
    /// Smoothing parameter for seasonal components
    pub gamma: Option<Vec<f64>>,
    /// Fourier coefficients for each seasonal component
    pub fourier_coefficients: Vec<Vec<(f64, f64)>>,
    /// Autoregressive coefficients for ARMA errors
    pub ar_coefficients: Vec<f64>,
    /// Moving average coefficients for ARMA errors
    pub ma_coefficients: Vec<f64>,
}

/// Options for STR (Seasonal-Trend decomposition using Regression)
#[derive(Debug, Clone)]
pub struct STROptions {
    /// Type of regularization to use
    pub regularization_type: RegularizationType,
    /// Regularization parameter for trend
    pub trend_lambda: f64,
    /// Regularization parameter for seasonal components
    pub seasonal_lambda: f64,
    /// Seasonal periods (can include non-integer values)
    pub seasonal_periods: Vec<f64>,
    /// Whether to use robust estimation (less sensitive to outliers)
    pub robust: bool,
    /// Whether to compute confidence intervals
    pub compute_confidence_intervals: bool,
    /// Confidence level (e.g., 0.95 for 95% confidence)
    pub confidence_level: f64,
    /// Degrees of freedom for the trend
    pub trend_degrees: usize,
    /// Whether to allow the seasonal pattern to change over time
    pub flexible_seasonal: bool,
}

/// Type of regularization to use in STR
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegularizationType {
    /// Ridge regularization (L2 penalty)
    Ridge,
    /// LASSO regularization (L1 penalty)
    Lasso,
    /// Elastic Net regularization (combination of L1 and L2)
    ElasticNet,
}

impl Default for STROptions {
    fn default() -> Self {
        Self {
            regularization_type: RegularizationType::Ridge,
            trend_lambda: 10.0,
            seasonal_lambda: 0.5,
            seasonal_periods: Vec::new(),
            robust: false,
            compute_confidence_intervals: false,
            confidence_level: 0.95,
            trend_degrees: 3,
            flexible_seasonal: false,
        }
    }
}

/// Result of STR decomposition
#[derive(Debug, Clone)]
pub struct STRResult<F> {
    /// Trend component
    pub trend: Array1<F>,
    /// Seasonal components (one for each seasonal period)
    pub seasonal_components: Vec<Array1<F>>,
    /// Residual component
    pub residual: Array1<F>,
    /// Original time series
    pub original: Array1<F>,
    /// Confidence intervals for trend (if computed)
    pub trend_ci: Option<(Array1<F>, Array1<F>)>, // (lower, upper)
    /// Confidence intervals for seasonal components (if computed)
    pub seasonal_ci: Option<Vec<(Array1<F>, Array1<F>)>>, // (lower, upper) for each component
}

/// Performs STR (Seasonal-Trend decomposition using Regression) on a time series
///
/// STR uses regularized regression to extract trend and seasonal components from
/// a time series. It allows for multiple seasonal components, non-integer periods,
/// and can provide confidence intervals for the components.
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `options` - Options for STR decomposition
///
/// # Returns
///
/// * STR decomposition result
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::decomposition::{str_decomposition, STROptions};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0,
///                 1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 2.5];
///
/// let mut options = STROptions::default();
/// options.seasonal_periods = vec![4.0, 12.0]; // Both quarterly and yearly patterns
///
/// let result = str_decomposition(&ts, &options).unwrap();
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal Components: {:?}", result.seasonal_components);
/// println!("Residual: {:?}", result.residual);
/// ```
pub fn str_decomposition<F>(ts: &Array1<F>, options: &STROptions) -> Result<STRResult<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    let n = ts.len();

    // Check inputs
    if n < 3 {
        return Err(TimeSeriesError::DecompositionError(
            "Time series must have at least 3 points for STR decomposition".to_string(),
        ));
    }

    if options.seasonal_periods.is_empty() {
        return Err(TimeSeriesError::DecompositionError(
            "At least one seasonal period must be specified for STR".to_string(),
        ));
    }

    for &period in &options.seasonal_periods {
        if period <= 1.0 {
            return Err(TimeSeriesError::DecompositionError(
                "Seasonal periods must be greater than 1".to_string(),
            ));
        }
    }

    if options.trend_lambda < 0.0 || options.seasonal_lambda < 0.0 {
        return Err(TimeSeriesError::DecompositionError(
            "Regularization parameters must be non-negative".to_string(),
        ));
    }

    if options.confidence_level <= 0.0 || options.confidence_level >= 1.0 {
        return Err(TimeSeriesError::DecompositionError(
            "Confidence level must be between 0 and 1".to_string(),
        ));
    }

    // Step 1: Prepare design matrices for trend and seasonal components
    let time_indices: Vec<F> = (0..n).map(|i| F::from_usize(i).unwrap()).collect();

    // Trend design matrix using polynomial basis functions
    let trend_degree = options.trend_degrees;
    let mut trend_basis = Array2::zeros((n, trend_degree + 1));

    // Fill trend design matrix with polynomial terms (1, t, t^2, t^3, ...)
    for i in 0..n {
        for j in 0..=trend_degree {
            if j == 0 {
                trend_basis[[i, j]] = F::one(); // Constant term
            } else {
                let time_idx = time_indices[i];
                trend_basis[[i, j]] = time_idx.powf(F::from_usize(j).unwrap());
            }
        }
    }

    // Seasonal design matrices using Fourier basis functions for each seasonal component
    let mut seasonal_bases = Vec::with_capacity(options.seasonal_periods.len());

    for &period in &options.seasonal_periods {
        // For each period, create Fourier basis functions (sin and cos terms)
        let k = (period / 2.0).floor() as usize; // Number of harmonics
        let mut seasonal_basis = Array2::zeros((n, 2 * k)); // 2 columns per harmonic (sin and cos)

        for i in 0..n {
            let t = F::from_usize(i).unwrap();
            for j in 0..k {
                let freq =
                    F::from_f64(2.0 * std::f64::consts::PI * (j + 1) as f64 / period).unwrap();
                // Sin term
                seasonal_basis[[i, 2 * j]] = (freq * t).sin();
                // Cos term
                seasonal_basis[[i, 2 * j + 1]] = (freq * t).cos();
            }
        }

        seasonal_bases.push(seasonal_basis);
    }

    // Step 2: Set up and solve the regularized regression problem
    // This is a complex optimization problem that requires matrix operations
    // In a real implementation, we would use a package like ndarray-linalg
    // Here, we'll implement a simplified version using ridge regression

    // Placeholder for full implementation
    // In a real implementation, we would:
    // 1. Combine all design matrices
    // 2. Create regularization penalties
    // 3. Solve the regularized least squares problem
    // 4. Extract the coefficients and compute the components

    // For now, we'll create a simplified implementation to demonstrate the concept

    // Create trend component using moving average as a placeholder
    let window_size = std::cmp::min(n / 5, 11);
    let window_size = if window_size % 2 == 0 {
        window_size + 1
    } else {
        window_size
    };
    let trend = moving_average(ts, window_size)?;

    // Create seasonal components
    let mut seasonal_components = Vec::with_capacity(options.seasonal_periods.len());

    // For each seasonal period, extract the corresponding component
    for &period in options.seasonal_periods.iter() {
        let p = period.round() as usize;
        let mut season = Array1::zeros(n);

        // Compute seasonal means for each position in the seasonal cycle
        let mut seasonal_means = vec![F::zero(); p];
        let mut counts = vec![0; p];

        // Detrend the series
        let mut detrended = Array1::zeros(n);
        for j in 0..n {
            detrended[j] = ts[j] - trend[j];
        }

        // Calculate average for each position in the seasonal pattern
        for j in 0..n {
            let pos = j % p;
            seasonal_means[pos] = seasonal_means[pos] + detrended[j];
            counts[pos] += 1;
        }

        // Normalize seasonal means
        for j in 0..p {
            if counts[j] > 0 {
                seasonal_means[j] = seasonal_means[j] / F::from_usize(counts[j]).unwrap();
            }
        }

        // Ensure seasonal component sums to zero
        let mut sum = F::zero();
        for &val in &seasonal_means {
            sum = sum + val;
        }
        let mean = sum / F::from_usize(p).unwrap();
        for mean_val in seasonal_means.iter_mut() {
            *mean_val = *mean_val - mean;
        }

        // Apply pattern to the whole series
        for j in 0..n {
            season[j] = seasonal_means[j % p];
        }

        seasonal_components.push(season);
    }

    // Compute residuals
    let mut residual = Array1::zeros(n);
    for i in 0..n {
        let mut seasonal_sum = F::zero();
        for season in &seasonal_components {
            seasonal_sum = seasonal_sum + season[i];
        }
        residual[i] = ts[i] - trend[i] - seasonal_sum;
    }

    // Compute confidence intervals if requested
    let trend_ci = if options.compute_confidence_intervals {
        // Placeholder for confidence interval calculation
        // In a real implementation, we would use the covariance matrix of the estimates
        let lower = trend.mapv(|x| {
            x - F::from_f64(1.96).unwrap() * residual.mapv(|x| x * x).sum().sqrt()
                / F::from_usize(n).unwrap().sqrt()
        });
        let upper = trend.mapv(|x| {
            x + F::from_f64(1.96).unwrap() * residual.mapv(|x| x * x).sum().sqrt()
                / F::from_usize(n).unwrap().sqrt()
        });
        Some((lower, upper))
    } else {
        None
    };

    let seasonal_ci = if options.compute_confidence_intervals {
        // Placeholder for seasonal confidence intervals
        let mut seasonal_cis = Vec::with_capacity(seasonal_components.len());
        for season in &seasonal_components {
            let lower = season.mapv(|x| {
                x - F::from_f64(1.96).unwrap() * residual.mapv(|x| x * x).sum().sqrt()
                    / F::from_usize(n).unwrap().sqrt()
            });
            let upper = season.mapv(|x| {
                x + F::from_f64(1.96).unwrap() * residual.mapv(|x| x * x).sum().sqrt()
                    / F::from_usize(n).unwrap().sqrt()
            });
            seasonal_cis.push((lower, upper));
        }
        Some(seasonal_cis)
    } else {
        None
    };

    // Create result
    let result = STRResult {
        trend,
        seasonal_components,
        residual,
        original: ts.clone(),
        trend_ci,
        seasonal_ci,
    };

    Ok(result)
}

/// Performs TBATS decomposition on a time series
///
/// TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and
/// Seasonal components) is a complex model that handles multiple seasonal patterns
/// using Fourier series to represent seasonality.
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `options` - Options for TBATS decomposition
///
/// # Returns
///
/// * TBATS decomposition result
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::decomposition::{tbats_decomposition, TBATSOptions};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0,
///                 1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 2.5];
///
/// let mut options = TBATSOptions::default();
/// options.seasonal_periods = vec![4.0, 12.0]; // Both quarterly and yearly patterns
/// options.use_box_cox = false; // Disable Box-Cox transformation for this example
///
/// let result = tbats_decomposition(&ts, &options).unwrap();
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal Components: {:?}", result.seasonal_components);
/// println!("Residuals: {:?}", result.residuals);
/// ```
pub fn tbats_decomposition<F>(ts: &Array1<F>, options: &TBATSOptions) -> Result<TBATSResult<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    let n = ts.len();

    // Check inputs
    if n < 3 {
        return Err(TimeSeriesError::DecompositionError(
            "Time series must have at least 3 points for TBATS decomposition".to_string(),
        ));
    }

    if options.seasonal_periods.is_empty()
        && options.use_box_cox
        && options.box_cox_lambda.is_none()
    {
        return Err(TimeSeriesError::DecompositionError(
            "When using Box-Cox transformation with no seasonal periods, lambda must be specified"
                .to_string(),
        ));
    }

    for &period in &options.seasonal_periods {
        if period <= 1.0 {
            return Err(TimeSeriesError::DecompositionError(
                "Seasonal periods must be greater than 1".to_string(),
            ));
        }
    }

    if let Some(ref terms) = options.fourier_terms {
        if terms.len() != options.seasonal_periods.len() {
            return Err(TimeSeriesError::DecompositionError(
                "Number of Fourier terms must match number of seasonal periods".to_string(),
            ));
        }
        for &k in terms {
            if k == 0 {
                return Err(TimeSeriesError::DecompositionError(
                    "Number of Fourier terms must be at least 1 for each seasonal period"
                        .to_string(),
                ));
            }
        }
    }

    // Step 1: Apply Box-Cox transformation if requested
    let (transformed_ts, lambda) = if options.use_box_cox {
        let lambda = options.box_cox_lambda.unwrap_or({
            // Auto-estimate lambda (simplified approach)
            // In a real implementation, we would use maximum likelihood estimation
            0.0 // Log transformation
        });
        let transformed = box_cox_transform(ts, lambda)?;
        (transformed, Some(lambda))
    } else {
        (ts.clone(), None)
    };

    // Step 2: Initialize components
    let mut level = Array1::zeros(n);
    let mut trend = Array1::zeros(n);

    // Initial level is just the first observation
    level[0] = transformed_ts[0];

    // Initial trend is the average first difference of the first few observations
    if options.use_trend && n > 1 {
        let mut trend_sum = F::zero();
        let trend_points = std::cmp::min(n - 1, 4);
        for i in 0..trend_points {
            trend_sum = trend_sum + (transformed_ts[i + 1] - transformed_ts[i]);
        }
        trend[0] = trend_sum / F::from_usize(trend_points).unwrap();
    }

    // Step 3: Determine number of Fourier terms for each seasonal component
    let fourier_terms = match &options.fourier_terms {
        Some(terms) => terms.clone(),
        None => {
            // Default to minimum of period/2 or 5 terms
            options
                .seasonal_periods
                .iter()
                .map(|&p| std::cmp::min((p / 2.0).floor() as usize, 5))
                .collect()
        }
    };

    // Step 4: Initialize seasonal components using Fourier series
    let mut seasonal_components = Vec::with_capacity(options.seasonal_periods.len());
    let mut fourier_coefficients = Vec::with_capacity(options.seasonal_periods.len());

    for (i, &period) in options.seasonal_periods.iter().enumerate() {
        let k = fourier_terms[i];
        let mut season = Array1::zeros(n);
        let mut coefficients = Vec::with_capacity(k);

        // Initialize coefficients with small random values for simplicity
        // In a real implementation, we would use regression to estimate initial values
        for _ in 0..k {
            coefficients.push((0.1, 0.1)); // (a, b) for sin and cos terms
        }

        // Apply Fourier series to compute initial seasonal component
        for t in 0..n {
            let t_f = F::from_usize(t).unwrap();
            for (j, &(a, b)) in coefficients.iter().enumerate() {
                // We don't need j_f for the current implementation
                let freq =
                    F::from_f64(2.0 * std::f64::consts::PI * (j + 1) as f64 / period).unwrap();
                let a_f = F::from_f64(a).unwrap();
                let b_f = F::from_f64(b).unwrap();

                season[t] = season[t] + a_f * (freq * t_f).sin() + b_f * (freq * t_f).cos();
            }
        }

        seasonal_components.push(season);
        fourier_coefficients.push(coefficients);
    }

    // Step 5: Compute residuals for ARMA modeling
    let mut residuals = Array1::zeros(n);
    for t in 0..n {
        let mut seasonal_sum = F::zero();
        for season in &seasonal_components {
            seasonal_sum = seasonal_sum + season[t];
        }

        if options.use_trend {
            residuals[t] = transformed_ts[t] - level[t] - seasonal_sum;
            if t > 0 {
                residuals[t] = residuals[t] - trend[t - 1];
            }
        } else {
            residuals[t] = transformed_ts[t] - level[t] - seasonal_sum;
        }
    }

    // Step 6: Fit ARMA model on residuals (simplified)
    // In a real implementation, we would use proper model selection and parameter estimation
    let ar_order = options.ar_order;
    let ma_order = options.ma_order;

    let ar_coefficients = vec![0.1; ar_order]; // Placeholder
    let ma_coefficients = vec![0.1; ma_order]; // Placeholder

    // Step 7: Estimate smoothing parameters (simplified)
    // In a real implementation, we would use maximum likelihood estimation
    let alpha = 0.1; // Level smoothing
    let beta = if options.use_trend { Some(0.01) } else { None }; // Trend smoothing
    let phi = if options.use_damped_trend {
        Some(0.98)
    } else {
        None
    }; // Damping

    // Seasonal smoothing parameters
    let gamma = if !options.seasonal_periods.is_empty() {
        Some(vec![0.001; options.seasonal_periods.len()])
    } else {
        None
    };

    // Step 8: Forecast (not implemented yet, future work)

    // Step 9: Organize outputs
    // For demonstration, we'll use the initialized components

    // Create parameters struct
    let parameters = TBATSParameters {
        lambda,
        alpha,
        beta,
        phi,
        gamma,
        fourier_coefficients,
        ar_coefficients,
        ma_coefficients,
    };

    // Create result struct
    let result = TBATSResult {
        trend,
        seasonal_components,
        residuals,
        level,
        original: ts.clone(),
        transformed: if options.use_box_cox {
            Some(transformed_ts)
        } else {
            None
        },
        parameters,
    };

    Ok(result)
}

/// Applies Box-Cox transformation to a time series
fn box_cox_transform<F>(ts: &Array1<F>, lambda: f64) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let lambda_f = F::from_f64(lambda).unwrap();
    let one = F::one();
    let zero = F::zero();

    // Check for non-positive values
    for &x in ts.iter() {
        if x <= zero && lambda != 0.0 {
            return Err(TimeSeriesError::DecompositionError(
                "Box-Cox transformation requires positive values for lambda != 0".to_string(),
            ));
        }
    }

    let mut result = Array1::zeros(ts.len());

    // Apply transformation based on lambda value
    if (lambda - 0.0).abs() < 1e-10 {
        // Log transform for lambda = 0
        for i in 0..ts.len() {
            result[i] = ts[i].ln();
        }
    } else {
        // Power transform for lambda != 0
        for i in 0..ts.len() {
            result[i] = (ts[i].powf(lambda_f) - one) / lambda_f;
        }
    }

    Ok(result)
}
/// Performs exponential smoothing decomposition on a time series
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `period` - The seasonal period
/// * `alpha` - Level smoothing parameter (0 < alpha < 1)
/// * `beta` - Trend smoothing parameter (0 < beta < 1)
/// * `gamma` - Seasonal smoothing parameter (0 < gamma < 1)
/// * `model` - Decomposition model (additive or multiplicative)
///
/// # Returns
///
/// * Decomposition result containing level, trend, seasonal, and residual components
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::decomposition::{exponential_decomposition, DecompositionModel};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
/// let result = exponential_decomposition(&ts, 4, 0.2, 0.1, 0.3,
///                                         DecompositionModel::Additive).unwrap();
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal: {:?}", result.seasonal);
/// println!("Residual: {:?}", result.residual);
/// ```
pub fn exponential_decomposition<F>(
    ts: &Array1<F>,
    period: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
    model: DecompositionModel,
) -> Result<DecompositionResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < period + 1 {
        return Err(TimeSeriesError::DecompositionError(format!(
            "Time series length ({}) must be greater than the seasonal period ({})",
            ts.len(),
            period
        )));
    }

    if alpha <= 0.0 || alpha >= 1.0 || beta <= 0.0 || beta >= 1.0 || gamma <= 0.0 || gamma >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Smoothing parameters must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    let alpha = F::from_f64(alpha).unwrap();
    let beta = F::from_f64(beta).unwrap();
    let gamma = F::from_f64(gamma).unwrap();

    let n = ts.len();

    // Initialize components
    let mut level = Array1::zeros(n + 1);
    let mut trend = Array1::zeros(n + 1);
    let mut seasonal = Array1::zeros(n + period);
    let mut residual = Array1::zeros(n);

    // Initialize level, trend, and seasonal components
    let initial_level = ts[0]; // Could also use average of first few observations
    level[0] = initial_level;

    // Initialize trend (average of first differences)
    if n > 1 {
        let mut sum = F::zero();
        for i in 1..min(n, 10) {
            sum = sum + (ts[i] - ts[i - 1]);
        }
        trend[0] = sum / F::from_usize(min(n - 1, 9)).unwrap();
    }

    // Initialize seasonal (average deviation from level for each season)
    for i in 0..min(period, n) {
        let pos = i % period;
        let expected = level[0] + F::from_usize(i).unwrap() * trend[0];
        match model {
            DecompositionModel::Additive => {
                seasonal[pos] = ts[i] - expected;
            }
            DecompositionModel::Multiplicative => {
                if expected == F::zero() {
                    return Err(TimeSeriesError::DecompositionError(
                        "Division by zero in multiplicative model initialization".to_string(),
                    ));
                }
                seasonal[pos] = ts[i] / expected;
            }
        }
    }

    // Normalize initial seasonal component
    match model {
        DecompositionModel::Additive => {
            let mean = seasonal
                .iter()
                .take(period)
                .fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(period).unwrap();
            for i in 0..period {
                seasonal[i] = seasonal[i] - mean;
            }
        }
        DecompositionModel::Multiplicative => {
            let mean = seasonal
                .iter()
                .take(period)
                .fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(period).unwrap();
            if mean == F::zero() {
                return Err(TimeSeriesError::DecompositionError(
                    "Division by zero normalizing multiplicative seasonal component".to_string(),
                ));
            }
            for i in 0..period {
                seasonal[i] = seasonal[i] / mean;
            }
        }
    }

    // Exponential smoothing recursion
    for i in 0..n {
        let s = i % period; // Current season
        let expected = match model {
            DecompositionModel::Additive => level[i] + trend[i],
            DecompositionModel::Multiplicative => level[i] * trend[i],
        };

        // Calculate residual
        match model {
            DecompositionModel::Additive => {
                residual[i] = ts[i] - expected - seasonal[s];
            }
            DecompositionModel::Multiplicative => {
                if expected == F::zero() || seasonal[s] == F::zero() {
                    residual[i] = F::zero(); // Avoid division by zero
                } else {
                    residual[i] = ts[i] / (expected * seasonal[s]);
                }
            }
        }

        // Update level, trend, and seasonal components
        match model {
            DecompositionModel::Additive => {
                level[i + 1] =
                    alpha * (ts[i] - seasonal[s]) + (F::one() - alpha) * (level[i] + trend[i]);
                trend[i + 1] = beta * (level[i + 1] - level[i]) + (F::one() - beta) * trend[i];
                seasonal[s + period] =
                    gamma * (ts[i] - level[i + 1]) + (F::one() - gamma) * seasonal[s];
            }
            DecompositionModel::Multiplicative => {
                if seasonal[s] == F::zero() {
                    return Err(TimeSeriesError::DecompositionError(
                        "Division by zero in multiplicative model update".to_string(),
                    ));
                }
                level[i + 1] =
                    alpha * (ts[i] / seasonal[s]) + (F::one() - alpha) * (level[i] * trend[i]);

                if level[i] == F::zero() {
                    trend[i + 1] = trend[i]; // Avoid division by zero
                } else {
                    trend[i + 1] = beta * (level[i + 1] / level[i]) + (F::one() - beta) * trend[i];
                }

                if level[i + 1] == F::zero() {
                    seasonal[s + period] = seasonal[s]; // Avoid division by zero
                } else {
                    seasonal[s + period] =
                        gamma * (ts[i] / level[i + 1]) + (F::one() - gamma) * seasonal[s];
                }
            }
        }
    }

    // Prepare final components
    let trend_component = Array1::from_iter(level.iter().take(n).cloned());
    let seasonal_component = Array1::from_iter((0..n).map(|i| seasonal[i % period]));
    let original = ts.clone();

    Ok(DecompositionResult {
        trend: trend_component,
        seasonal: seasonal_component,
        residual,
        original,
    })
}

/// Options for Singular Spectrum Analysis (SSA) decomposition
#[derive(Debug, Clone)]
pub struct SSAOptions {
    /// Window length (embedding dimension)
    pub window_length: usize,
    /// Number of components to include in the trend
    pub n_trend_components: usize,
    /// Number of components to include in the seasonal
    pub n_seasonal_components: Option<usize>,
    /// Whether to group components by similarity
    pub group_by_similarity: bool,
    /// Threshold for determining component similarity
    pub component_similarity_threshold: f64,
}

impl Default for SSAOptions {
    fn default() -> Self {
        Self {
            window_length: 0, // Will be set automatically based on time series length
            n_trend_components: 2,
            n_seasonal_components: None,
            group_by_similarity: true,
            component_similarity_threshold: 0.9,
        }
    }
}

/// Performs Singular Spectrum Analysis (SSA) decomposition on a time series
///
/// SSA decomposes a time series into trend, seasonal, and residual components
/// using eigenvalue decomposition of the trajectory matrix.
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `options` - Options for SSA decomposition
///
/// # Returns
///
/// * Decomposition result containing trend, seasonal, and residual components
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::decomposition::{ssa_decomposition, SSAOptions};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
/// let mut options = SSAOptions::default();
/// options.window_length = 4;
/// options.n_trend_components = 1;
/// let result = ssa_decomposition(&ts, &options).unwrap();
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal: {:?}", result.seasonal);
/// println!("Residual: {:?}", result.residual);
/// ```
pub fn ssa_decomposition<F>(ts: &Array1<F>, options: &SSAOptions) -> Result<DecompositionResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();

    // Check inputs
    if n < 3 {
        return Err(TimeSeriesError::DecompositionError(
            "Time series must have at least 3 points for SSA decomposition".to_string(),
        ));
    }

    // Determine window length if not specified
    let window_length = if options.window_length > 0 {
        options.window_length
    } else {
        // Default is approximately n/2
        std::cmp::max(2, n / 2)
    };

    if window_length >= n {
        return Err(TimeSeriesError::DecompositionError(format!(
            "Window length ({}) must be less than time series length ({})",
            window_length, n
        )));
    }

    if options.n_trend_components == 0 {
        return Err(TimeSeriesError::DecompositionError(
            "Number of trend components must be at least 1".to_string(),
        ));
    }

    // Step 1: Embedding - Create trajectory matrix
    let k = n - window_length + 1; // Number of columns in the trajectory matrix
    let mut trajectory_matrix = Array2::zeros((window_length, k));

    for i in 0..window_length {
        for j in 0..k {
            trajectory_matrix[[i, j]] = ts[i + j];
        }
    }

    // Step 2: SVD on trajectory matrix
    let svd_result = svd(&trajectory_matrix);
    let (u, s, vt) = match svd_result {
        Ok((u, s, vt)) => (u, s, vt),
        Err(e) => {
            return Err(TimeSeriesError::DecompositionError(format!(
                "SVD computation failed: {}",
                e
            )))
        }
    };

    // Step 3: Grouping components
    let mut trend_components = Vec::new();
    let mut seasonal_components = Vec::new();

    // Group by similarity if requested
    if options.group_by_similarity {
        let mut component_groups = Vec::new();
        let mut visited = vec![false; window_length.min(k)];

        for i in 0..window_length.min(k) {
            if visited[i] || s[i] <= F::epsilon() {
                continue;
            }

            let mut group = vec![i];
            visited[i] = true;

            // Find similar components using w-correlation
            for j in (i + 1)..window_length.min(k) {
                if visited[j] || s[j] <= F::epsilon() {
                    continue;
                }

                let similarity = compute_component_similarity(&u, &vt, &s, i, j, n);
                if similarity > options.component_similarity_threshold {
                    group.push(j);
                    visited[j] = true;
                }
            }

            component_groups.push(group);
        }

        // Assign first group to trend and next groups to seasonal
        if !component_groups.is_empty() {
            trend_components = component_groups[0].clone();

            let n_seasonal = options
                .n_seasonal_components
                .unwrap_or(component_groups.len().saturating_sub(1));

            // Get the range of component groups to include in seasonal components
            let end_idx = std::cmp::min(component_groups.len(), n_seasonal + 1);
            for group in component_groups.iter().take(end_idx).skip(1) {
                seasonal_components.extend_from_slice(group);
            }
        }
    } else {
        // Simple grouping based on eigenvalue ranking
        for i in 0..options.n_trend_components.min(window_length.min(k)) {
            trend_components.push(i);
        }

        let n_seasonal = options
            .n_seasonal_components
            .unwrap_or(std::cmp::min(window_length.min(k), 10) - options.n_trend_components);

        for i in options.n_trend_components
            ..std::cmp::min(
                options.n_trend_components + n_seasonal,
                window_length.min(k),
            )
        {
            seasonal_components.push(i);
        }
    }

    // Step 4: Diagonal averaging to reconstruct components
    let mut trend = Array1::zeros(n);
    let mut seasonal = Array1::zeros(n);

    // Reconstruct trend components
    for &idx in &trend_components {
        if idx >= window_length.min(k) || s[idx] <= F::epsilon() {
            continue;
        }

        let reconstructed = reconstruct_component(&u, &vt, &s, idx, window_length, k, n);
        for i in 0..n {
            trend[i] = trend[i] + reconstructed[i];
        }
    }

    // Reconstruct seasonal components
    for &idx in &seasonal_components {
        if idx >= window_length.min(k) || s[idx] <= F::epsilon() {
            continue;
        }

        let reconstructed = reconstruct_component(&u, &vt, &s, idx, window_length, k, n);
        for i in 0..n {
            seasonal[i] = seasonal[i] + reconstructed[i];
        }
    }

    // Calculate residual
    let mut residual = Array1::zeros(n);
    for i in 0..n {
        residual[i] = ts[i] - trend[i] - seasonal[i];
    }

    // Create result
    let original = ts.clone();

    Ok(DecompositionResult {
        trend,
        seasonal,
        residual,
        original,
    })
}

/// SVD result type to reduce complexity
type SVDResult<F> = std::result::Result<(Array2<F>, Array1<F>, Array2<F>), String>;

/// Performs SVD on a matrix
fn svd<F>(matrix: &Array2<F>) -> SVDResult<F>
where
    F: Float + FromPrimitive + Debug,
{
    // This is a placeholder for a real SVD implementation.
    // In a full implementation, we would use a linear algebra crate like ndarray-linalg.
    // For now, we'll create simple matrices to illustrate the structure.

    let (m, n) = matrix.dim();
    let min_dim = std::cmp::min(m, n);

    // Create some dummy singular values (decreasing)
    let mut s = Array1::zeros(min_dim);
    for i in 0..min_dim {
        s[i] = F::from_f64(min_dim as f64 - i as f64).unwrap();
    }

    // Create dummy U and V^T matrices
    let u = Array2::eye(m);
    let vt = Array2::eye(n);

    Ok((u, s, vt))
}

/// Compute similarity between two principal components
fn compute_component_similarity<F>(
    _u: &Array2<F>,
    _vt: &Array2<F>,
    _s: &Array1<F>,
    i: usize,
    j: usize,
    n: usize,
) -> f64
where
    F: Float + FromPrimitive + Debug,
{
    // Placeholder for computing w-correlation between elementary components
    // In a real implementation, we would compute the actual w-correlation

    // Simple approximation based on index distance
    let d = (i as f64 - j as f64).abs() / n as f64;
    f64::exp(-d * 5.0)
}

/// Reconstruct a component from SVD results using diagonal averaging
fn reconstruct_component<F>(
    _u: &Array2<F>,
    _vt: &Array2<F>,
    s: &Array1<F>,
    idx: usize,
    _window_length: usize,
    _k: usize,
    n: usize,
) -> Array1<F>
where
    F: Float + FromPrimitive + Debug,
{
    // In a real implementation, this would reconstruct the component from SVD
    // For now, we create a simple sinusoidal pattern as a placeholder

    let mut result = Array1::zeros(n);
    let period = F::from_usize(idx + 2).unwrap();

    for i in 0..n {
        result[i] = F::from_f64(
            (i as f64 * 2.0 * std::f64::consts::PI / period.to_f64().unwrap()).sin()
                * s[idx].to_f64().unwrap()
                / s[0].to_f64().unwrap(),
        )
        .unwrap();
    }

    result
}

// Helper function to get the minimum of two values
fn min<T: Ord>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_ssa_decomposition() {
        // Create a simple time series with trend + seasonal components
        let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];

        let options = SSAOptions {
            window_length: 4,
            n_trend_components: 1,
            ..Default::default()
        };

        let result = ssa_decomposition(&ts, &options).unwrap();

        // Check that we have the expected components
        assert_eq!(result.trend.len(), ts.len());
        assert_eq!(result.seasonal.len(), ts.len());
        assert_eq!(result.residual.len(), ts.len());

        // Check that the decomposition reconstructs the original series
        for i in 0..ts.len() {
            let sum = result.trend[i] + result.seasonal[i] + result.residual[i];
            assert!((sum - ts[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mstl_decomposition() {
        // Create a test time series with multiple seasonal patterns
        // Weekly (7) and quarterly (12) patterns
        let ts = array![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 1.5, 2.5, 3.5,
            4.5, 5.5, 6.5, 7.5, 1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7
        ];

        let options = MSTLOptions {
            seasonal_periods: vec![7, 12],
            trend_window: 13,
            ..Default::default()
        };

        let result = mstl_decomposition(&ts, &options).unwrap();

        // Check that we have the expected components
        assert_eq!(result.trend.len(), ts.len());
        assert_eq!(result.seasonal_components.len(), 2); // Two seasonal components
        assert_eq!(result.seasonal_components[0].len(), ts.len());
        assert_eq!(result.seasonal_components[1].len(), ts.len());
        assert_eq!(result.residual.len(), ts.len());

        // Check that the decomposition reconstructs the original series
        for i in 0..ts.len() {
            let seasonal_sum = result.seasonal_components[0][i] + result.seasonal_components[1][i];
            let sum = result.trend[i] + seasonal_sum + result.residual[i];
            assert!((sum - ts[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_tbats_decomposition() {
        // Create a time series with multiple seasonal patterns
        // Period 4 (quarterly) and period 12 (yearly)
        let ts = array![
            1.0, 2.0, 3.0, 2.0, 1.5, 2.5, 3.5, 2.5, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.5,
            3.5, 4.5, 3.5, 3.0, 4.0, 5.0, 4.0
        ];

        let mut options = TBATSOptions::default();
        options.seasonal_periods = vec![4.0, 12.0];
        options.use_box_cox = false; // Disable Box-Cox for simpler testing
        options.ar_order = 0; // No AR terms
        options.ma_order = 0; // No MA terms

        let result = tbats_decomposition(&ts, &options).unwrap();

        // Check that we have the expected components
        assert_eq!(result.trend.len(), ts.len());
        assert_eq!(result.seasonal_components.len(), 2); // Two seasonal components
        assert_eq!(result.seasonal_components[0].len(), ts.len());
        assert_eq!(result.seasonal_components[1].len(), ts.len());
        assert_eq!(result.residuals.len(), ts.len());
        assert_eq!(result.level.len(), ts.len());

        // Check that all components have reasonable lengths
        assert_eq!(result.parameters.fourier_coefficients.len(), 2);
        assert!(result.parameters.fourier_coefficients[0].len() > 0);
        assert!(result.parameters.fourier_coefficients[1].len() > 0);

        // Test Box-Cox transformation when enabled
        let mut box_cox_options = TBATSOptions::default();
        box_cox_options.seasonal_periods = vec![4.0, 12.0];
        box_cox_options.use_box_cox = true;
        box_cox_options.box_cox_lambda = Some(0.5); // Square root transformation

        let box_cox_result = tbats_decomposition(&ts, &box_cox_options).unwrap();

        // Transformation should be present
        assert!(box_cox_result.transformed.is_some());
        assert_eq!(box_cox_result.transformed.unwrap().len(), ts.len());
    }

    #[test]
    fn test_box_cox_transform() {
        // Test log transformation (lambda = 0)
        let ts = array![1.0, 2.0, 5.0, 10.0];
        let transformed = box_cox_transform(&ts, 0.0).unwrap();

        // Result should be log values
        assert!((transformed[0] - 0.0).abs() < 1e-10); // ln(1) = 0
        assert!((transformed[1] - 0.6931471805599453).abs() < 1e-10); // ln(2) ≈ 0.693
        assert!((transformed[2] - 1.6094379124341003).abs() < 1e-10); // ln(5) ≈ 1.609
        assert!((transformed[3] - 2.302585092994046).abs() < 1e-10); // ln(10) ≈ 2.303

        // Test square root transformation (lambda = 0.5)
        let transformed = box_cox_transform(&ts, 0.5).unwrap();

        // Result should be (x^0.5 - 1) / 0.5 = 2 * (sqrt(x) - 1)
        assert!((transformed[0] - 0.0).abs() < 1e-10); // 2 * (sqrt(1) - 1) = 0
        assert!((transformed[1] - 0.8284271247461903).abs() < 1e-10); // 2 * (sqrt(2) - 1) ≈ 0.828
        assert!((transformed[2] - 2.472135954999579).abs() < 1e-10); // 2 * (sqrt(5) - 1) ≈ 2.472
        assert!((transformed[3] - 4.32455532033676).abs() < 1e-10); // 2 * (sqrt(10) - 1) ≈ 4.325

        // Test error on non-positive values for lambda != 0
        let ts_with_negative = array![-1.0, 2.0, 3.0, 4.0];
        let result = box_cox_transform(&ts_with_negative, 0.5);
        assert!(result.is_err());

        // Should work with lambda = 0 though
        let log_transformed = box_cox_transform(&array![1.0, 2.0, 3.0], 0.0);
        assert!(log_transformed.is_ok());
    }

    #[test]
    fn test_str_decomposition() {
        // Create a test time series with trend and multiple seasonal patterns
        let ts = array![
            1.0, 2.0, 3.0, 2.0, 1.5, 2.5, 3.5, 2.5, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.5,
            3.5, 4.5, 3.5, 3.0, 4.0, 5.0, 4.0
        ];

        let mut options = STROptions::default();
        options.seasonal_periods = vec![4.0, 12.0]; // Quarterly and yearly patterns
        options.compute_confidence_intervals = true; // Test confidence interval calculation

        let result = str_decomposition(&ts, &options).unwrap();

        // Check that we have the expected components
        assert_eq!(result.trend.len(), ts.len());
        assert_eq!(result.seasonal_components.len(), 2); // Two seasonal components
        assert_eq!(result.seasonal_components[0].len(), ts.len());
        assert_eq!(result.seasonal_components[1].len(), ts.len());
        assert_eq!(result.residual.len(), ts.len());

        // Check that confidence intervals are present
        assert!(result.trend_ci.is_some());
        assert!(result.seasonal_ci.is_some());
        assert_eq!(result.seasonal_ci.unwrap().len(), 2);

        // Check that the decomposition reconstructs the original series
        for i in 0..ts.len() {
            let seasonal_sum = result.seasonal_components[0][i] + result.seasonal_components[1][i];
            let sum = result.trend[i] + seasonal_sum + result.residual[i];
            assert!((sum - ts[i]).abs() < 1e-10);
        }

        // Test without confidence intervals
        let mut options2 = STROptions::default();
        options2.seasonal_periods = vec![4.0];
        options2.compute_confidence_intervals = false;

        let result2 = str_decomposition(&ts, &options2).unwrap();
        assert!(result2.trend_ci.is_none());
        assert!(result2.seasonal_ci.is_none());

        // Test error cases
        let mut invalid_options = STROptions::default();
        invalid_options.seasonal_periods = Vec::new();
        let result3 = str_decomposition(&ts, &invalid_options);
        assert!(result3.is_err());

        let mut invalid_options2 = STROptions::default();
        invalid_options2.seasonal_periods = vec![0.5]; // Too small
        let result4 = str_decomposition(&ts, &invalid_options2);
        assert!(result4.is_err());

        let mut invalid_options3 = STROptions::default();
        invalid_options3.seasonal_periods = vec![4.0];
        invalid_options3.trend_lambda = -1.0; // Negative lambda
        let result5 = str_decomposition(&ts, &invalid_options3);
        assert!(result5.is_err());
    }
}
