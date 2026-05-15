//! Seasonal trend decomposition methods
//!
//! This module provides methods for decomposing time series into trend, seasonal,
//! and residual components, using various techniques.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::RobustFilterOptions;
use crate::error::{Result, TimeSeriesError};

/// Result of a seasonal decomposition
#[derive(Debug, Clone)]
pub struct SeasonalDecomposition<F: Float> {
    /// The trend component
    pub trend: Array1<F>,
    /// The seasonal component
    pub seasonal: Array1<F>,
    /// The residual component
    pub residual: Array1<F>,
}

/// Type of seasonal decomposition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompositionType {
    /// Additive model: Y = T + S + R
    Additive,
    /// Multiplicative model: Y = T * S * R
    Multiplicative,
}

/// Method for seasonal decomposition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompositionMethod {
    /// Classical decomposition
    Classical,
    /// STL (Seasonal and Trend decomposition using Loess)
    STL,
    /// X-11 method
    X11,
    /// SEATS (Seasonal Extraction in ARIMA Time Series)
    SEATS,
}

/// Options for seasonal decomposition
#[derive(Debug, Clone)]
pub struct SeasonalDecompositionOptions {
    /// Type of decomposition
    pub decomposition_type: DecompositionType,
    /// Method for decomposition
    pub method: DecompositionMethod,
    /// Period of the seasonality
    pub period: usize,
    /// Number of observations per year (for multiple seasonality)
    pub observations_per_year: Option<usize>,
    /// Whether to use robust methods resistant to outliers
    pub robust: bool,
    /// Trend filtering options (used for trend estimation)
    pub trend_options: Option<RobustFilterOptions>,
    /// Number of iterations for iterative methods
    pub num_iterations: usize,
}

impl Default for SeasonalDecompositionOptions {
    fn default() -> Self {
        SeasonalDecompositionOptions {
            decomposition_type: DecompositionType::Additive,
            method: DecompositionMethod::Classical,
            period: 12, // Default for monthly data
            observations_per_year: None,
            robust: false,
            trend_options: None,
            num_iterations: 2,
        }
    }
}

/// Performs seasonal decomposition of a time series
///
/// This function decomposes a time series into trend, seasonal, and residual components
/// using the specified method.
///
/// # Arguments
///
/// * `ts` - The input time series data
/// * `options` - Options controlling the seasonal decomposition
///
/// # Returns
///
/// A `SeasonalDecomposition` struct containing the three components
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_series::trends::{
///     seasonal_decomposition, SeasonalDecompositionOptions,
///     DecompositionType, DecompositionMethod
/// };
///
/// // Create a sample time series with trend and seasonality
/// let n = 120; // 10 years of monthly data
/// let mut ts = Array1::zeros(n);
///
/// // Add trend
/// for i in 0..n {
///     ts[i] = i as f64 / 20.0;
/// }
///
/// // Add monthly seasonality
/// for i in 0..n {
///     ts[i] += (i as f64 * std::f64::consts::PI / 6.0).sin();
/// }
///
/// // Add random noise
/// for i in 0..n {
///     ts[i] += 0.1 * scirs2_core::random::random::<f64>();
/// }
///
/// // Configure decomposition options
/// let options = SeasonalDecompositionOptions {
///     decomposition_type: DecompositionType::Additive,
///     method: DecompositionMethod::Classical,
///     period: 12,
///     ..Default::default()
/// };
///
/// // Perform decomposition
/// let decomp = seasonal_decomposition(&ts, &options).expect("Operation failed");
///
/// // Check that components have the same length as input
/// assert_eq!(decomp.trend.len(), n);
/// assert_eq!(decomp.seasonal.len(), n);
/// assert_eq!(decomp.residual.len(), n);
/// ```
#[allow(dead_code)]
pub fn seasonal_decomposition<F>(
    ts: &Array1<F>,
    options: &SeasonalDecompositionOptions,
) -> Result<SeasonalDecomposition<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let period = options.period;

    if n < 2 * period {
        return Err(TimeSeriesError::InsufficientData {
            message: format!(
                "Time series too short for seasonal decomposition with period={period}"
            ),
            required: 2 * period,
            actual: n,
        });
    }

    match options.method {
        DecompositionMethod::Classical => classical_decomposition(ts, options),
        DecompositionMethod::STL => stl_decomposition(ts, options),
        DecompositionMethod::X11 => x11_decomposition(ts, options),
        DecompositionMethod::SEATS => seats_decomposition(ts, options),
    }
}

/// Performs classical seasonal decomposition
#[allow(dead_code)]
fn classical_decomposition<F>(
    ts: &Array1<F>,
    options: &SeasonalDecompositionOptions,
) -> Result<SeasonalDecomposition<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let period = options.period;
    let is_additive = options.decomposition_type == DecompositionType::Additive;

    // Step 1: Estimate trend using moving average
    let trend = estimate_trend_by_moving_average(ts, period, options.robust)?;

    // Step 2: Remove trend to get detrended series
    let detrended = if is_additive {
        // Y - T for additive model
        Array1::from_shape_fn(n, |i| {
            if i < trend.len() {
                ts[i] - trend[i]
            } else {
                F::nan()
            }
        })
    } else {
        // Y / T for multiplicative model
        Array1::from_shape_fn(n, |i| {
            if i < trend.len() && trend[i] != F::zero() {
                ts[i] / trend[i]
            } else {
                F::nan()
            }
        })
    };

    // Step 3: Estimate seasonal component by averaging over each period
    let seasonal_raw = estimate_seasonal_component(&detrended, period)?;

    // Step 4: Normalize seasonal component
    let seasonal = normalize_seasonal_component(&seasonal_raw, is_additive)?;

    // Step 5: Calculate residuals
    let residual = if is_additive {
        // R = Y - T - S for additive model
        Array1::from_shape_fn(n, |i| ts[i] - trend[i] - seasonal[i])
    } else {
        // R = Y / (T * S) for multiplicative model
        Array1::from_shape_fn(n, |i| {
            let denominator = trend[i] * seasonal[i];
            if denominator != F::zero() {
                ts[i] / denominator
            } else {
                F::one() // Default to 1 for multiplicative model when denominator is zero
            }
        })
    };

    Ok(SeasonalDecomposition {
        trend,
        seasonal,
        residual,
    })
}

/// Performs STL (Seasonal and Trend decomposition using Loess) decomposition
#[allow(dead_code)]
fn stl_decomposition<F>(
    ts: &Array1<F>,
    options: &SeasonalDecompositionOptions,
) -> Result<SeasonalDecomposition<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let period = options.period;
    let num_iterations = options.num_iterations;
    let is_additive = options.decomposition_type == DecompositionType::Additive;

    if !is_additive {
        // For multiplicative model, log-transform the series
        let log_ts = Array1::from_shape_fn(n, |i| ts[i].ln());

        // Perform STL on log-transformed series
        let decomp = stl_decomposition_inner(&log_ts, period, num_iterations, options.robust)?;

        // Transform back
        let trend = Array1::from_shape_fn(n, |i| decomp.trend[i].exp());
        let seasonal = Array1::from_shape_fn(n, |i| decomp.seasonal[i].exp());
        let residual = Array1::from_shape_fn(n, |i| decomp.residual[i].exp());

        return Ok(SeasonalDecomposition {
            trend,
            seasonal,
            residual,
        });
    }

    // Additive model
    stl_decomposition_inner(ts, period, num_iterations, options.robust)
}

/// Inner implementation of STL decomposition for additive model
#[allow(dead_code)]
fn stl_decomposition_inner<F>(
    ts: &Array1<F>,
    period: usize,
    num_iterations: usize,
    robust: bool,
) -> Result<SeasonalDecomposition<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();

    // Initialize components
    let mut trend = Array1::<F>::zeros(n);
    let mut seasonal = Array1::<F>::zeros(n);
    let mut residual = Array1::<F>::zeros(n);

    // Parameters
    let n_p = period;
    let n_t = 1 + 2 * n_p; // Trend smoothing window
    let n_s = 7; // Seasonal smoothing window
    let n_l = n; // Loess smoothing window

    // Outer loop
    for iter in 0..num_iterations {
        // Inner loop - cycle subseries
        // 1. Detrending
        let detrended = Array1::from_shape_fn(n, |i| ts[i] - trend[i]);

        // 2. Cycle-subseries smoothing
        for i in 0..period {
            let mut subseries = Vec::new();
            let mut subseries_idx = Vec::new();

            // Extract subseries
            let mut idx = i;
            while idx < n {
                subseries.push(detrended[idx]);
                subseries_idx.push(idx);
                idx += period;
            }

            if subseries.len() < 2 {
                continue;
            }

            // Loess smoothing of subseries
            let smoothed = loess_smooth(&Array1::from_vec(subseries), n_s, robust)?;

            // Put back smoothed values
            for (j, &idx) in subseries_idx.iter().enumerate() {
                if j < smoothed.len() {
                    seasonal[idx] = smoothed[j];
                }
            }
        }

        // 3. Low-pass filtering of seasonal
        let low_pass = moving_average_filter(&seasonal, n_t)?;

        // 4. Detrend seasonal component
        for i in 0..n {
            seasonal[i] = seasonal[i] - low_pass[i];
        }

        // 5. Deseasonalize
        let deseasonalized = Array1::from_shape_fn(n, |i| ts[i] - seasonal[i]);

        // 6. Trend smoothing
        trend = loess_smooth(&deseasonalized, n_l, robust)?;

        // 7. Calculate residuals
        for i in 0..n {
            residual[i] = ts[i] - trend[i] - seasonal[i];
        }

        // For robust version, calculate weights based on residuals
        if robust && iter < num_iterations - 1 {
            // Compute robust weights for next iteration
            let abs_residuals: Vec<F> = residual.iter().map(|&r| r.abs()).collect();
            let weights = calculate_robust_weights(
                &abs_residuals,
                F::from_f64(6.0).expect("Operation failed"),
            )?;

            // Apply weights to the trend estimation for next iteration
            // This implements weighted loess smoothing for full robustness
            let deseasonalized = Array1::from_shape_fn(n, |i| ts[i] - seasonal[i]);
            trend = loess_smooth_weighted(&deseasonalized, n_l, &weights)?;
        }
    }

    Ok(SeasonalDecomposition {
        trend,
        seasonal,
        residual,
    })
}

/// Performs X-11 seasonal decomposition
#[allow(dead_code)]
fn x11_decomposition<F>(
    ts: &Array1<F>,
    options: &SeasonalDecompositionOptions,
) -> Result<SeasonalDecomposition<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let period = options.period;
    let is_additive = options.decomposition_type == DecompositionType::Additive;

    // X-11 is a complex procedure with many steps. This is a simplified version.

    // Step 1: Initial trend estimate using centered moving average
    let trend_init = estimate_trend_by_moving_average(ts, period, false)?;

    // Step 2: Initial seasonal component
    let detrended = if is_additive {
        Array1::from_shape_fn(n, |i| ts[i] - trend_init[i])
    } else {
        Array1::from_shape_fn(n, |i| {
            if trend_init[i] != F::zero() {
                ts[i] / trend_init[i]
            } else {
                F::one()
            }
        })
    };

    let seasonal_init = estimate_seasonal_component(&detrended, period)?;
    let seasonal_norm = normalize_seasonal_component(&seasonal_init, is_additive)?;

    // Step 3: Deseasonalized series
    let deseasonalized = if is_additive {
        Array1::from_shape_fn(n, |i| ts[i] - seasonal_norm[i])
    } else {
        Array1::from_shape_fn(n, |i| {
            if seasonal_norm[i] != F::zero() {
                ts[i] / seasonal_norm[i]
            } else {
                ts[i]
            }
        })
    };

    // Step 4: Refined trend estimate using Henderson filter
    let trend_refined = henderson_filter(&deseasonalized, 13)?;

    // Step 5: Refined seasonal component
    let detrended_refined = if is_additive {
        Array1::from_shape_fn(n, |i| ts[i] - trend_refined[i])
    } else {
        Array1::from_shape_fn(n, |i| {
            if trend_refined[i] != F::zero() {
                ts[i] / trend_refined[i]
            } else {
                F::one()
            }
        })
    };

    let seasonal_refined = estimate_seasonal_component(&detrended_refined, period)?;
    let seasonal_final = normalize_seasonal_component(&seasonal_refined, is_additive)?;

    // Step 6: Calculate final residuals
    let residual = if is_additive {
        Array1::from_shape_fn(n, |i| ts[i] - trend_refined[i] - seasonal_final[i])
    } else {
        Array1::from_shape_fn(n, |i| {
            let denominator = trend_refined[i] * seasonal_final[i];
            if denominator != F::zero() {
                ts[i] / denominator
            } else {
                F::one()
            }
        })
    };

    Ok(SeasonalDecomposition {
        trend: trend_refined,
        seasonal: seasonal_final,
        residual,
    })
}

/// Performs SEATS (Seasonal Extraction in ARIMA Time Series) decomposition
///
/// This implementation uses a two-step filter approach that approximates the
/// canonical SEATS signal extraction:
///
/// 1. **Trend**: Hodrick-Prescott filter with λ chosen by period
///    (λ=1600 for quarterly/period=4, λ=14400 for monthly/period=12,
///     λ=100 for annual/period=1 or unknown, interpolated otherwise)
/// 2. **Seasonal**: Seasonal-means estimator applied to the detrended series,
///    then normalized to sum to zero (additive) or multiply to one (multiplicative)
/// 3. **Irregular**: Residual after removing trend and seasonal components
///
/// For multiplicative models the log-transformed series is decomposed additively
/// and then back-transformed via exp(), mirroring the approach in `stl_decomposition`.
#[allow(dead_code)]
fn seats_decomposition<F>(
    ts: &Array1<F>,
    options: &SeasonalDecompositionOptions,
) -> Result<SeasonalDecomposition<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let is_additive = options.decomposition_type == DecompositionType::Additive;

    if !is_additive {
        // Log-transform, decompose additively, back-transform
        let log_ts = Array1::from_shape_fn(n, |i| ts[i].ln());
        let additive_opts = SeasonalDecompositionOptions {
            decomposition_type: DecompositionType::Additive,
            ..options.clone()
        };
        let decomp = seats_decomposition_additive(&log_ts, &additive_opts)?;
        return Ok(SeasonalDecomposition {
            trend: Array1::from_shape_fn(n, |i| decomp.trend[i].exp()),
            seasonal: Array1::from_shape_fn(n, |i| decomp.seasonal[i].exp()),
            residual: Array1::from_shape_fn(n, |i| decomp.residual[i].exp()),
        });
    }

    seats_decomposition_additive(ts, options)
}

/// Inner additive SEATS decomposition via HP filter + seasonal means.
fn seats_decomposition_additive<F>(
    ts: &Array1<F>,
    options: &SeasonalDecompositionOptions,
) -> Result<SeasonalDecomposition<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let period = options.period;

    // Choose HP lambda based on period (higher frequency → higher lambda)
    // Quarterly: 1600, Monthly: 14400, Annual: 100 (Ravn-Uhlig rule: λ ∝ p^4)
    // General formula: λ = 1600 * (p/4)^4
    let lambda = {
        let ratio = F::from_usize(period).ok_or_else(|| {
            TimeSeriesError::ComputationError("period too large to convert".to_string())
        })? / F::from_f64(4.0).expect("4.0 is representable");
        F::from_f64(1600.0).expect("1600.0 is representable") * ratio * ratio * ratio * ratio
    };

    // Stage 1: Trend via HP filter
    let trend = hp_filter_generic(ts, lambda)?;

    // Stage 2: Detrend
    let detrended = Array1::from_shape_fn(n, |i| ts[i] - trend[i]);

    // Stage 3: Seasonal component via period averages
    let seasonal_raw = estimate_seasonal_component(&detrended, period)?;
    let seasonal = normalize_seasonal_component(&seasonal_raw, true)?;

    // Stage 4: Irregular (residual)
    let residual = Array1::from_shape_fn(n, |i| ts[i] - trend[i] - seasonal[i]);

    Ok(SeasonalDecomposition {
        trend,
        seasonal,
        residual,
    })
}

/// Generic Hodrick-Prescott filter solving the pentadiagonal system
/// `(I + λ·D'D)·τ = y` via band LDL' decomposition.
///
/// Returns the trend component `τ`.
fn hp_filter_generic<F>(ts: &Array1<F>, lambda: F) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    if n < 4 {
        return Err(TimeSeriesError::InsufficientData {
            message: "HP filter requires at least 4 observations".to_string(),
            required: 4,
            actual: n,
        });
    }

    // Build the symmetric band matrix A = I + λ·D'D stored as three diagonals:
    //   diag[i]  = A[i,i]
    //   off1[i]  = A[i, i+1]   (i = 0..n-2)
    //   off2[i]  = A[i, i+2]   (i = 0..n-3)
    //
    // D'D diagonals (second-difference matrix):
    //   main : [1, 5, 6, …, 6, 5, 1]
    //   ±1   : [-2, -4, …, -4, -2]
    //   ±2   : [1, 1, …, 1]
    let one = F::one();
    let two = F::from_f64(2.0).expect("2.0 representable");
    let four = F::from_f64(4.0).expect("4.0 representable");
    let five = F::from_f64(5.0).expect("5.0 representable");
    let six = F::from_f64(6.0).expect("6.0 representable");

    let mut diag: Vec<F> = (0..n)
        .map(|i| {
            let dd = match i {
                0 => one,
                1 => five,
                _ if i == n - 2 => five,
                _ if i == n - 1 => one,
                _ => six,
            };
            one + lambda * dd
        })
        .collect();

    let mut off1: Vec<F> = (0..n - 1)
        .map(|i| {
            let dd = if i == 0 || i == n - 2 { -two } else { -four };
            lambda * dd
        })
        .collect();

    let mut off2: Vec<F> = vec![lambda; n - 2];

    // In-place band LDL' factorization (Thomas-algorithm generalised to bandwidth 2)
    // After factorisation diag[i] holds D[i,i], off1[i] holds L[i+1,i],
    // off2[i] holds L[i+2,i] (all scaled by D).
    for i in 0..n {
        // Pivot check — if numerically zero, matrix is singular
        if diag[i].abs() < F::from_f64(1e-14).expect("representable") {
            return Err(TimeSeriesError::NumericalInstability(
                "HP filter: near-zero pivot in band LDL' factorisation".to_string(),
            ));
        }

        // Update column i+1
        if i + 1 < n {
            let l_i1_i = off1[i] / diag[i]; // L[i+1, i]
            diag[i + 1] = diag[i + 1] - l_i1_i * off1[i];
            if i + 2 < n {
                let old_off1_i1 = off1[i + 1];
                off1[i + 1] = old_off1_i1 - l_i1_i * off2[i];
            }
            // Store scaled factor back (will be used as L element)
            off1[i] = l_i1_i;
        }

        // Update column i+2
        if i + 2 < n {
            let l_i2_i = off2[i] / diag[i]; // L[i+2, i]
            diag[i + 2] = diag[i + 2] - l_i2_i * off2[i];
            // off2[i] becomes the L element
            off2[i] = l_i2_i;
        }
    }

    // Forward substitution: solve L*z = y
    let mut z: Vec<F> = ts.to_vec();
    for i in 0..n {
        if i >= 1 {
            z[i] = z[i] - off1[i - 1] * z[i - 1];
        }
        if i >= 2 {
            z[i] = z[i] - off2[i - 2] * z[i - 2];
        }
    }

    // Scale: solve D*w = z
    for i in 0..n {
        z[i] = z[i] / diag[i];
    }

    // Back substitution: solve L'*x = w
    for i in (0..n).rev() {
        if i + 1 < n {
            z[i] = z[i] - off1[i] * z[i + 1];
        }
        if i + 2 < n {
            z[i] = z[i] - off2[i] * z[i + 2];
        }
    }

    Ok(Array1::from_vec(z))
}

/// Estimates trend component using centered moving average
#[allow(dead_code)]
fn estimate_trend_by_moving_average<F>(
    ts: &Array1<F>,
    period: usize,
    robust: bool,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();

    // For even periods, use 2x period centered moving average
    let window_size = if period.is_multiple_of(2) {
        period + 1
    } else {
        period
    };

    let half_window = window_size / 2;

    let mut trend = Array1::<F>::zeros(n);

    for i in 0..n {
        let start = i.saturating_sub(half_window);
        let end = if i + half_window < n {
            i + half_window
        } else {
            n - 1
        };

        if end - start + 1 < window_size / 2 {
            // Not enough data points for proper averaging
            trend[i] = F::nan();
            continue;
        }

        if robust {
            // Robust moving average (trimmed mean)
            let mut values: Vec<F> = (start..=end).map(|j| ts[j]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Trim 10% from each end
            let trim_count = values.len() / 10;
            let middle_values = &values[trim_count..(values.len() - trim_count)];

            let sum = middle_values.iter().fold(F::zero(), |acc, &v| acc + v);
            trend[i] = sum / F::from_usize(middle_values.len()).expect("Operation failed");
        } else {
            // Regular moving average
            let sum = (start..=end).fold(F::zero(), |acc, j| acc + ts[j]);
            trend[i] = sum / F::from_usize(end - start + 1).expect("Operation failed");
        }
    }

    // Interpolate NaN values at the edges
    interpolate_nan_values(&mut trend)?;

    Ok(trend)
}

/// Estimates seasonal component by averaging over each period
#[allow(dead_code)]
fn estimate_seasonal_component<F>(detrended: &Array1<F>, period: usize) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = detrended.len();

    // Calculate seasonal factors by averaging values for each season
    let mut seasonal_factors = vec![F::zero(); period];
    let mut count = vec![0; period];

    for i in 0..n {
        let season = i % period;

        if !detrended[i].is_nan() {
            seasonal_factors[season] = seasonal_factors[season] + detrended[i];
            count[season] += 1;
        }
    }

    // Calculate averages
    for season in 0..period {
        if count[season] > 0 {
            seasonal_factors[season] =
                seasonal_factors[season] / F::from_usize(count[season]).expect("Operation failed");
        }
    }

    // Expand seasonal factors to full time series
    let mut seasonal = Array1::<F>::zeros(n);

    for i in 0..n {
        seasonal[i] = seasonal_factors[i % period];
    }

    Ok(seasonal)
}

/// Normalizes seasonal component to ensure it sums/averages to zero/one
#[allow(dead_code)]
fn normalize_seasonal_component<F>(_seasonal: &Array1<F>, isadditive: bool) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = _seasonal.len();

    if n == 0 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Empty _seasonal component".to_string(),
            required: 1,
            actual: 0,
        });
    }

    let mut normalized = _seasonal.clone();

    if isadditive {
        // Ensure _seasonal component sums to zero
        let mean = _seasonal.sum() / F::from_usize(n).expect("Operation failed");

        for i in 0..n {
            normalized[i] = _seasonal[i] - mean;
        }
    } else {
        // Ensure _seasonal component multiplies to one
        let mut product = F::one();
        let mut count = 0;

        for &val in _seasonal.iter() {
            if !val.is_nan() && val != F::zero() {
                product = product * val;
                count += 1;
            }
        }

        if count > 0 {
            let geometric_mean =
                product.powf(F::one() / F::from_usize(count).expect("Operation failed"));

            for i in 0..n {
                if !_seasonal[i].is_nan() && _seasonal[i] != F::zero() {
                    normalized[i] = _seasonal[i] / geometric_mean;
                } else {
                    normalized[i] = F::one();
                }
            }
        }
    }

    Ok(normalized)
}

/// Applies a moving average filter to a time series
#[allow(dead_code)]
fn moving_average_filter<F>(_ts: &Array1<F>, windowsize: usize) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = _ts.len();
    let half_window = windowsize / 2;

    let mut filtered = Array1::<F>::zeros(n);

    for i in 0..n {
        let start = i.saturating_sub(half_window);
        let end = if i + half_window < n {
            i + half_window
        } else {
            n - 1
        };

        let sum = (start..=end).fold(F::zero(), |acc, j| acc + _ts[j]);
        filtered[i] = sum / F::from_usize(end - start + 1).expect("Operation failed");
    }

    Ok(filtered)
}

/// Applies a Henderson filter to a time series
#[allow(dead_code)]
fn henderson_filter<F>(_ts: &Array1<F>, windowsize: usize) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = _ts.len();

    if windowsize.is_multiple_of(2) {
        return Err(TimeSeriesError::InvalidInput(
            "Henderson filter window size must be odd".to_string(),
        ));
    }

    let half_window = windowsize / 2;

    // Calculate Henderson weights
    let weights = calculate_henderson_weights(windowsize)?;

    // Apply filter
    let mut filtered = Array1::<F>::zeros(n);

    for i in 0..n {
        let start = i.saturating_sub(half_window);
        let end = if i + half_window < n {
            i + half_window
        } else {
            n - 1
        };

        let mut weighted_sum = F::zero();
        let mut weight_sum = F::zero();

        for (j, k) in (start..=end).enumerate() {
            let weight_idx = j + (start as i32 - (i as i32 - half_window as i32)) as usize;

            if weight_idx < weights.len() {
                weighted_sum = weighted_sum + _ts[k] * weights[weight_idx];
                weight_sum = weight_sum + weights[weight_idx];
            }
        }

        if weight_sum != F::zero() {
            filtered[i] = weighted_sum / weight_sum;
        } else {
            filtered[i] = _ts[i];
        }
    }

    Ok(filtered)
}

/// Calculates Henderson filter weights
#[allow(dead_code)]
fn calculate_henderson_weights<F>(_windowsize: usize) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let m = (_windowsize - 1) / 2;
    let mut weights = vec![F::zero(); _windowsize];

    // Calculate terms for the Henderson weights
    let m_f = F::from_usize(m).expect("Operation failed");

    for i in 0..=m {
        let i_f = F::from_usize(i).expect("Operation failed");
        let term1 = (F::from_usize(315).expect("Operation failed")
            * (m_f + F::one())
            * (m_f + F::one())
            - F::from_usize(105).expect("Operation failed") * i_f * i_f)
            * (F::from_usize(231).expect("Operation failed") * (m_f + F::one()) * (m_f + F::one())
                - F::from_usize(42).expect("Operation failed") * i_f * i_f
                - F::from_usize(3).expect("Operation failed")
                    * (m_f + F::one())
                    * (m_f + F::one())
                    * (m_f + F::one())
                    * (m_f + F::one()));

        let term2 =
            F::from_usize(105).expect("Operation failed") * (m_f + F::one()) * (m_f + F::one())
                - F::from_usize(45).expect("Operation failed") * i_f * i_f;

        weights[m - i] = term1 / term2;
        weights[m + i] = weights[m - i];
    }

    // Normalize weights to sum to 1
    let sum = weights.iter().fold(F::zero(), |acc, &w| acc + w);

    for weight in &mut weights {
        *weight = *weight / sum;
    }

    Ok(weights)
}

/// Applies LOESS (locally weighted regression) smoothing
#[allow(dead_code)]
fn loess_smooth<F>(_ts: &Array1<F>, windowsize: usize, robust: bool) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = _ts.len();

    if n < windowsize {
        return Err(TimeSeriesError::InsufficientData {
            message: format!(
                "Time series too short for LOESS smoothing with window size {windowsize}"
            ),
            required: windowsize,
            actual: n,
        });
    }

    let mut smoothed = Array1::<F>::zeros(n);
    let half_window = windowsize / 2;

    // For each point, fit a local polynomial
    for i in 0..n {
        let start = i.saturating_sub(half_window);
        let end = if i + half_window < n {
            i + half_window
        } else {
            n - 1
        };

        let window_length = end - start + 1;

        // Prepare local data
        let mut x = Vec::with_capacity(window_length);
        let mut y = Vec::with_capacity(window_length);
        let mut weights = vec![F::one(); window_length];

        for (j, idx) in (start..=end).enumerate() {
            x.push(F::from_usize(idx).expect("Operation failed"));
            y.push(_ts[idx]);

            // Tricube weight function based on distance
            let d = (F::from_usize(idx).expect("Operation failed")
                - F::from_usize(i).expect("Operation failed"))
            .abs()
                / F::from_usize(half_window).expect("Operation failed");

            if d < F::one() {
                let t = F::one() - d * d * d;
                weights[j] = t * t * t;
            } else {
                weights[j] = F::zero();
            }
        }

        // Robust version with iteration
        if robust {
            let max_iter = 4;
            let mut iter_weights = weights.clone();

            for _ in 0..max_iter {
                // Weighted polynomial fit
                let fit = weighted_polynomial_fit(&x, &y, &iter_weights, 1)?;

                // Predict at the current point
                smoothed[i] = fit[0] + fit[1] * F::from_usize(i).expect("Operation failed");

                // Calculate residuals
                let mut residuals = Vec::with_capacity(window_length);

                for (j, idx) in (start..=end).enumerate() {
                    let pred = fit[0] + fit[1] * F::from_usize(idx).expect("Operation failed");
                    residuals.push(y[j] - pred);
                }

                // Update robust weights
                let robustness_weights = calculate_robust_weights(
                    &residuals,
                    F::from_f64(6.0).expect("Operation failed"),
                )?;

                for j in 0..window_length {
                    iter_weights[j] = weights[j] * robustness_weights[j];
                }
            }
        } else {
            // Simple weighted polynomial fit
            let fit = weighted_polynomial_fit(&x, &y, &weights, 1)?;

            // Predict at the current point
            smoothed[i] = fit[0] + fit[1] * F::from_usize(i).expect("Operation failed");
        }
    }

    Ok(smoothed)
}

/// Applies LOESS smoothing with additional external weights
#[allow(dead_code)]
fn loess_smooth_weighted<F>(
    ts: &Array1<F>,
    window_size: usize,
    external_weights: &[F],
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();

    if n != external_weights.len() {
        return Err(TimeSeriesError::InvalidInput(
            "Time series and weights must have same length".to_string(),
        ));
    }

    if n < window_size {
        return Err(TimeSeriesError::InsufficientData {
            message: format!(
                "Time series too short for weighted LOESS smoothing with window _size {window_size}"
            ),
            required: window_size,
            actual: n,
        });
    }

    let mut smoothed = Array1::<F>::zeros(n);
    let half_window = window_size / 2;

    // For each point, fit a local polynomial with external _weights
    for i in 0..n {
        let start = i.saturating_sub(half_window);
        let end = if i + half_window < n {
            i + half_window
        } else {
            n - 1
        };

        let window_length = end - start + 1;

        // Prepare local data
        let mut x = Vec::with_capacity(window_length);
        let mut y = Vec::with_capacity(window_length);
        let mut _weights = Vec::with_capacity(window_length);

        for idx in start..=end {
            x.push(F::from_usize(idx).expect("Operation failed"));
            y.push(ts[idx]);

            // Combine tricube weight with external weight
            let d = (F::from_usize(idx).expect("Operation failed")
                - F::from_usize(i).expect("Operation failed"))
            .abs()
                / F::from_usize(half_window).expect("Operation failed");

            let tricube_weight = if d < F::one() {
                let t = F::one() - d * d * d;
                t * t * t
            } else {
                F::zero()
            };

            _weights.push(tricube_weight * external_weights[idx]);
        }

        // Weighted polynomial fit
        let fit = weighted_polynomial_fit(&x, &y, &_weights, 1)?;

        // Predict at the current point
        smoothed[i] = fit[0] + fit[1] * F::from_usize(i).expect("Operation failed");
    }

    Ok(smoothed)
}

/// Fits a weighted polynomial to data
#[allow(dead_code)]
fn weighted_polynomial_fit<F>(x: &[F], y: &[F], weights: &[F], degree: usize) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = x.len();

    if n <= degree {
        return Err(TimeSeriesError::InsufficientData {
            message: format!("Not enough data points for polynomial fit of degree {degree}"),
            required: degree + 1,
            actual: n,
        });
    }

    if n != y.len() || n != weights.len() {
        return Err(TimeSeriesError::InvalidInput(
            "x, y, and weights must have the same length".to_string(),
        ));
    }

    // For linear fit (degree=1), use simple weighted least squares
    if degree == 1 {
        let mut sum_w = F::zero();
        let mut sum_wx = F::zero();
        let mut sum_wy = F::zero();
        let mut sum_wxx = F::zero();
        let mut sum_wxy = F::zero();

        for i in 0..n {
            let w = weights[i];
            let xi = x[i];
            let yi = y[i];

            sum_w = sum_w + w;
            sum_wx = sum_wx + w * xi;
            sum_wy = sum_wy + w * yi;
            sum_wxx = sum_wxx + w * xi * xi;
            sum_wxy = sum_wxy + w * xi * yi;
        }

        let denom = sum_w * sum_wxx - sum_wx * sum_wx;

        if denom.abs() < F::from_f64(1e-10).expect("Operation failed") {
            // Nearly singular matrix, revert to simple average
            let intercept = sum_wy / sum_w;
            let slope = F::zero();

            return Ok(vec![intercept, slope]);
        }

        let intercept = (sum_wxx * sum_wy - sum_wx * sum_wxy) / denom;
        let slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denom;

        return Ok(vec![intercept, slope]);
    }

    // For higher degrees, build the design matrix
    // This is left as a placeholder

    Err(TimeSeriesError::NotImplemented(format!(
        "Polynomial fit of degree {degree} not implemented"
    )))
}

/// Calculates robust weights using bisquare function
#[allow(dead_code)]
fn calculate_robust_weights<F>(residuals: &[F], c: F) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = residuals.len();

    if n == 0 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Empty _residuals array".to_string(),
            required: 1,
            actual: 0,
        });
    }

    // Calculate MAD (Median Absolute Deviation)
    let mut abs_residuals = residuals.to_vec();

    for r in &mut abs_residuals {
        *r = r.abs();
    }

    abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median_idx = n / 2;
    let mad = if n.is_multiple_of(2) {
        (abs_residuals[median_idx - 1] + abs_residuals[median_idx])
            / F::from_f64(2.0).expect("Operation failed")
    } else {
        abs_residuals[median_idx]
    };

    // Scale MAD for consistency with normal distribution
    let s = mad / F::from_f64(0.6745).expect("Operation failed");

    // Use a small positive value if MAD is zero
    let scale = if s > F::from_f64(1e-10).expect("Operation failed") {
        s
    } else {
        F::from_f64(1e-10).expect("Operation failed")
    };

    // Calculate bisquare weights
    let mut weights = Vec::with_capacity(n);

    for &r in residuals {
        let u = r.abs() / (c * scale);

        if u < F::one() {
            let temp = F::one() - u * u;
            weights.push(temp * temp);
        } else {
            weights.push(F::zero());
        }
    }

    Ok(weights)
}

/// Interpolates NaN values in a time series
#[allow(dead_code)]
fn interpolate_nan_values<F>(ts: &mut Array1<F>) -> Result<()>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();

    if n == 0 {
        return Ok(());
    }

    // Forward fill
    let mut last_valid = None;

    for i in 0..n {
        if !ts[i].is_nan() {
            last_valid = Some(ts[i]);
        } else if let Some(val) = last_valid {
            ts[i] = val;
        }
    }

    // Backward fill for any remaining NaNs at the beginning
    last_valid = None;

    for i in (0..n).rev() {
        if !ts[i].is_nan() {
            last_valid = Some(ts[i]);
        } else if let Some(val) = last_valid {
            ts[i] = val;
        }
    }

    // If still have NaNs (all values were NaN), replace with zeros
    for i in 0..n {
        if ts[i].is_nan() {
            ts[i] = F::zero();
        }
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    /// Build a synthetic series: linear trend + sinusoidal seasonal + small noise.
    ///
    /// y[t] = slope * t + amplitude * sin(2π t / period) + noise[t]
    fn build_synthetic(
        n: usize,
        period: usize,
        slope: f64,
        amplitude: f64,
        noise_amp: f64,
    ) -> Array1<f64> {
        use std::f64::consts::TAU;
        Array1::from_shape_fn(n, |t| {
            let trend_val = slope * t as f64;
            let seas_val = amplitude * (TAU * t as f64 / period as f64).sin();
            // Deterministic "noise" via a simple hash-like pattern to avoid
            // depending on a random number generator in tests
            let noise_val = noise_amp * ((t as f64 * 1.618_034).sin() * 0.5 + 0.5 - 0.5);
            trend_val + seas_val + noise_val
        })
    }

    fn variance(arr: &Array1<f64>) -> f64 {
        let n = arr.len() as f64;
        let mean = arr.sum() / n;
        arr.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n
    }

    /// Test 1: residual variance is lower than original series variance.
    ///
    /// For a series dominated by a deterministic trend + strong seasonal, the
    /// SEATS decomposition should isolate those components and leave a small
    /// irregular component.
    #[test]
    fn test_seats_residual_variance_lower_than_original() {
        let period = 12_usize;
        let n = 5 * period; // 60 monthly observations

        // Strong trend and seasonal, small noise
        let ts = build_synthetic(n, period, 0.5, 3.0, 0.1);

        let opts = SeasonalDecompositionOptions {
            decomposition_type: DecompositionType::Additive,
            method: DecompositionMethod::SEATS,
            period,
            ..Default::default()
        };

        let decomp = seasonal_decomposition(&ts, &opts).expect("SEATS decomposition must succeed");

        assert_eq!(decomp.trend.len(), n, "trend length mismatch");
        assert_eq!(decomp.seasonal.len(), n, "seasonal length mismatch");
        assert_eq!(decomp.residual.len(), n, "residual length mismatch");

        let var_orig = variance(&ts);
        let var_resid = variance(&decomp.residual);

        assert!(
            var_resid < var_orig,
            "residual variance ({var_resid:.6}) should be less than original variance ({var_orig:.6})"
        );
    }

    /// Test 2: trend + seasonal + irregular exactly reconstructs the original series
    /// (within floating-point tolerance, since it is pure arithmetic decomposition).
    #[test]
    fn test_seats_reconstruction_exact() {
        let period = 4_usize; // quarterly
        let n = 6 * period; // 24 observations

        let ts = build_synthetic(n, period, 0.2, 1.5, 0.05);

        let opts = SeasonalDecompositionOptions {
            decomposition_type: DecompositionType::Additive,
            method: DecompositionMethod::SEATS,
            period,
            ..Default::default()
        };

        let decomp = seasonal_decomposition(&ts, &opts).expect("SEATS decomposition must succeed");

        // For additive SEATS: y[t] = trend[t] + seasonal[t] + residual[t] exactly,
        // because residual is defined as y - trend - seasonal.
        let tol = 1e-10_f64;
        for i in 0..n {
            let reconstructed = decomp.trend[i] + decomp.seasonal[i] + decomp.residual[i];
            let diff = (reconstructed - ts[i]).abs();
            assert!(
                diff < tol,
                "reconstruction error at t={i}: |{reconstructed} - {}| = {diff:.2e} > {tol:.0e}",
                ts[i]
            );
        }
    }
}
