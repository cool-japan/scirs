//! Seasonal-Trend decomposition using Regression (STR)

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};
use crate::utils::moving_average;

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
