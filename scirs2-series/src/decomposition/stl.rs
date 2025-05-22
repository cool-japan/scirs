//! STL (Seasonal-Trend decomposition using LOESS) and MSTL implementation

use ndarray::Array1;
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::common::DecompositionResult;
use crate::error::{Result, TimeSeriesError};

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

    // Validate options
    if options.trend_window % 2 == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "trend_window".to_string(),
            message: "Trend window size must be odd".to_string(),
        });
    }
    if options.seasonal_window % 2 == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "seasonal_window".to_string(),
            message: "Seasonal window size must be odd".to_string(),
        });
    }

    // Working arrays
    let n = ts.len();
    let mut seasonal = Array1::zeros(n);
    let mut trend = Array1::zeros(n);
    let mut weights = Array1::from_elem(n, F::one());
    let original = ts.clone();

    // STL Outer Loop
    for _ in 0..options.n_outer {
        // STL Inner Loop
        for _ in 0..options.n_inner {
            // 1. Detrend
            let detrended = if trend.iter().all(|&x| x == F::zero()) {
                // First iteration, trend is zero
                original.clone()
            } else {
                // Subsequent iterations, remove trend
                original.clone() - &trend
            };

            // 2. Cycle-subseries Smoothing for Seasonal Component
            let mut cycle_subseries = vec![Vec::new(); period];
            let mut smoothed_seasonal = Array1::zeros(n);

            // Group by seasonal position
            for i in 0..n {
                cycle_subseries[i % period].push((i, detrended[i]));
            }

            // Process each subseries
            for subseries in cycle_subseries.iter() {
                if subseries.is_empty() {
                    continue;
                }

                // Extract subseries values
                let mut indices = Vec::with_capacity(subseries.len());
                let mut values = Vec::with_capacity(subseries.len());
                let mut subseries_weights = Vec::with_capacity(subseries.len());

                for &(idx, val) in subseries {
                    indices.push(idx);
                    values.push(val);
                    subseries_weights.push(weights[idx]);
                }

                // Convert to ndarray
                let indices_array = Array1::from_vec(indices);
                let values_array = Array1::from_vec(values);
                let weights_array = Array1::from_vec(subseries_weights);

                // Calculate locally weighted smoothed value for each index
                // (In production code, replace with actual LOESS implementation)
                // Simplified for this example:
                let mut smoothed_values = Array1::zeros(indices_array.len());
                // ... smoothing algorithm would go here ...

                // For now, just use a simple moving average as a placeholder
                for i in 0..indices_array.len() {
                    let mut count = 0;
                    let mut sum = F::zero();
                    let window = options.seasonal_window / 2;

                    for j in 0..indices_array.len() {
                        if i >= window
                            && i < indices_array.len() - window
                            && j >= i - window
                            && j <= i + window
                        {
                            sum = sum + values_array[j] * weights_array[j];
                            count += 1;
                        }
                    }

                    if count > 0 {
                        smoothed_values[i] = sum / F::from_usize(count).unwrap();
                    } else {
                        smoothed_values[i] = values_array[i];
                    }
                }

                // Assign smoothed values back to the seasonal component
                for (idx, val) in indices_array.iter().zip(smoothed_values.iter()) {
                    smoothed_seasonal[*idx] = *val;
                }
            }

            // 3. Low-Pass Filter of Seasonal
            // (Actual implementation would use a proper low-pass filter)
            // This is a simplified placeholder
            let filtered_seasonal = smoothed_seasonal.clone();
            // ... filtering would go here ...

            // 4. Deseasonalize
            let deseasonalized = original.clone() - &filtered_seasonal;

            // 5. Trend Smoothing
            // (In production code, replace with actual LOESS implementation)
            // Simplified for this example:
            let mut new_trend = Array1::zeros(n);
            // ... smoothing algorithm would go here ...

            // For now, use simple moving average as placeholder
            for i in 0..n {
                let mut count = 0;
                let mut sum = F::zero();
                let window = options.trend_window / 2;

                for j in 0..n {
                    if i >= window && i < n - window && j >= i - window && j <= i + window {
                        sum = sum + deseasonalized[j] * weights[j];
                        count += 1;
                    }
                }

                if count > 0 {
                    new_trend[i] = sum / F::from_usize(count).unwrap();
                } else {
                    new_trend[i] = deseasonalized[i];
                }
            }

            // 6. Update components
            trend = new_trend;
            seasonal = filtered_seasonal;
        }

        // Update robustness weights (if using robust method)
        if options.robust {
            // Calculate residuals
            let residual = original.clone() - &trend - &seasonal;

            // Calculate robust weights
            // (In production code, implement proper bisquare weights)
            // Simplified for this example:
            let abs_residuals = residual.mapv(|x| x.abs());
            let max_residual = abs_residuals.fold(F::zero(), |a, &b| if a > b { a } else { b });

            if max_residual > F::zero() {
                for i in 0..n {
                    let r = abs_residuals[i] / max_residual;
                    if r < F::from_f64(0.5).unwrap() {
                        weights[i] = F::one();
                    } else if r < F::one() {
                        let tmp = F::one() - r * r;
                        weights[i] = tmp * tmp;
                    } else {
                        weights[i] = F::zero();
                    }
                }
            }
        }
    }

    // Calculate final residual
    let residual = original.clone() - &trend - &seasonal;

    // Return result
    Ok(DecompositionResult {
        trend,
        seasonal,
        residual,
        original,
    })
}

/// Performs Multiple STL decomposition on a time series with multiple seasonal components
///
/// MSTL (Multiple Seasonal-Trend decomposition using LOESS) extends STL to handle
/// multiple seasonal periods.
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `options` - Options for MSTL decomposition
///
/// # Returns
///
/// * MultiSeasonalDecompositionResult containing trend, multiple seasonal components, and residual
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
/// options.seasonal_periods = vec![4, 12]; // Weekly and monthly patterns
///
/// let result = mstl_decomposition(&ts, &options).unwrap();
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal Components: {}", result.seasonal_components.len());
/// println!("Residual: {:?}", result.residual);
/// ```
pub fn mstl_decomposition<F>(
    ts: &Array1<F>,
    options: &MSTLOptions,
) -> Result<MultiSeasonalDecompositionResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    // Validate options
    if options.seasonal_periods.is_empty() {
        return Err(TimeSeriesError::InvalidParameter {
            name: "seasonal_periods".to_string(),
            message: "At least one seasonal period must be specified".to_string(),
        });
    }

    let n_seasons = options.seasonal_periods.len();
    if let Some(ref windows) = options.seasonal_windows {
        if windows.len() != n_seasons {
            return Err(TimeSeriesError::InvalidParameter {
                name: "seasonal_windows".to_string(),
                message: format!(
                    "Number of seasonal windows ({}) must match number of seasonal periods ({})",
                    windows.len(),
                    n_seasons
                ),
            });
        }
    }

    // For each seasonal period, check that the time series is long enough
    for &period in &options.seasonal_periods {
        if ts.len() < 2 * period {
            return Err(TimeSeriesError::DecompositionError(format!(
                "Time series length ({}) must be at least twice the seasonal period ({})",
                ts.len(),
                period
            )));
        }
    }

    let n = ts.len();
    let original = ts.clone();
    let mut seasonal_components = Vec::with_capacity(n_seasons);
    let _weights = Array1::from_elem(n, F::one());
    let mut deseasonal = original.clone();

    // For each seasonal component
    for (i, &period) in options.seasonal_periods.iter().enumerate() {
        let seasonal_window = if let Some(ref windows) = options.seasonal_windows {
            windows[i]
        } else {
            // Default to 7 or (period / 2), whichever is larger
            std::cmp::max(7, period / 2) | 1 // Ensure odd
        };

        let stl_options = STLOptions {
            trend_window: options.trend_window,
            seasonal_window,
            n_inner: options.n_inner,
            n_outer: options.n_outer,
            robust: options.robust,
        };

        // Apply STL with current data
        let result = stl_decomposition(&deseasonal, period, &stl_options)?;

        // Save this seasonal component
        seasonal_components.push(result.seasonal);

        // Remove this seasonal component
        deseasonal = deseasonal - &seasonal_components[i];
    }

    // The remaining deseasonal series is the trend
    let trend = deseasonal.clone();

    // Calculate final residual
    let mut residual = original.clone();
    residual = residual - &trend;
    for seasonal in &seasonal_components {
        residual = residual - seasonal;
    }

    Ok(MultiSeasonalDecompositionResult {
        trend,
        seasonal_components,
        residual,
        original,
    })
}
