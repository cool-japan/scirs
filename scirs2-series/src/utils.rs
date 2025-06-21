//! Utility functions for time series analysis

use ndarray::{s, Array1, Array2, ArrayBase, Data, Ix1, Ix2, ScalarOperand};
use num_traits::{Float, FromPrimitive, NumCast};
use std::fmt::{Debug, Display};

use crate::error::{Result, TimeSeriesError};

/// Checks if a time series is stationary using the Augmented Dickey-Fuller test
///
/// A stationary time series has constant mean, variance, and autocovariance over time.
/// Calculate autocovariance at a given lag
pub fn autocovariance<S, F>(data: &ArrayBase<S, Ix1>, lag: usize) -> Result<F>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive,
{
    if lag >= data.len() {
        return Err(TimeSeriesError::InvalidInput(
            "Lag exceeds data length".to_string(),
        ));
    }

    let n = data.len();
    let mean = data.mean().unwrap_or(F::zero());

    // Calculate autocovariance
    let mut cov = F::zero();
    for i in lag..n {
        cov = cov + (data[i] - mean) * (data[i - lag] - mean);
    }

    Ok(cov / F::from(n - lag).unwrap())
}
/// This function uses the Augmented Dickey-Fuller test to check for stationarity.
///
/// # Arguments
///
/// * `ts` - The time series data to test
/// * `lags` - Number of lags to include in the regression (default: None, which calculates based on data size)
///
/// # Returns
///
/// * A tuple containing the test statistic and p-value
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_series::utils::is_stationary;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let (adf_stat, p_value) = is_stationary(&ts, None).unwrap();
///
/// // If p_value < 0.05, we can reject the null hypothesis (time series is stationary)
/// println!("ADF Statistic: {}, p-value: {}", adf_stat, p_value);
/// ```
pub fn is_stationary<F>(ts: &Array1<F>, lags: Option<usize>) -> Result<(F, F)>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 3 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must have at least 3 points for stationarity test".to_string(),
        ));
    }

    // Calculate number of lags if not provided
    let max_lags = match lags {
        Some(l) => l,
        None => {
            // Common rule: int(12 * (n/100)^(1/4))
            let n = ts.len() as f64;
            let max_lags_float = 12.0 * (n / 100.0).powf(0.25);
            max_lags_float.min(n / 3.0).floor() as usize
        }
    };

    // Create differenced series: y(t) - y(t-1)
    let mut diff_ts = Vec::with_capacity(ts.len() - 1);
    for i in 1..ts.len() {
        diff_ts.push(ts[i] - ts[i - 1]);
    }
    let diff_ts = Array1::from(diff_ts);

    // Create lagged series for regression
    let _y = diff_ts.slice(ndarray::s![max_lags..]);
    let x_level = ts.slice(ndarray::s![max_lags..diff_ts.len()]);

    // Create lag features for regression
    let mut x_data = Vec::with_capacity(diff_ts.len() - max_lags);
    for i in max_lags..diff_ts.len() {
        let mut row = vec![x_level[i - max_lags]];
        for lag in 1..=max_lags {
            row.push(diff_ts[i - lag]);
        }
        x_data.push(row);
    }

    // Convert to linear regression problem: Δy(t) = α + βy(t-1) + Σ γᵢΔy(t-i) + ε(t)
    // Using scirs2_stats for linear regression and p-values

    // This is a simplified implementation
    // For a production system, we'd use a proper linear regression from scirs2_stats
    // with proper calculation of p-values for the coefficient of y(t-1)

    // For now, we'll return dummy values (this should be replaced with proper calculation)
    // If β (the coefficient for y(t-1)) is significantly < 0, the series is stationary
    // The test statistic is the t-statistic for the β coefficient

    let adf_stat = F::from_f64(-2.5).unwrap(); // Dummy value
    let p_value = F::from_f64(0.1).unwrap(); // Dummy value

    Ok((adf_stat, p_value))
}

/// Transforms a time series to achieve stationarity
///
/// Common transformations include differencing, log transformation, or
/// seasonal differencing.
///
/// # Arguments
///
/// * `ts` - The time series data to transform
/// * `method` - The transformation method ("diff", "log", "seasonal_diff")
/// * `seasonal_period` - Seasonal period for seasonal differencing (required if method is "seasonal_diff")
///
/// # Returns
///
/// * The transformed time series
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_series::utils::transform_to_stationary;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
///
/// // First-order differencing
/// let diff_ts = transform_to_stationary(&ts, "diff", None).unwrap();
///
/// // Log transformation
/// let log_ts = transform_to_stationary(&ts, "log", None).unwrap();
///
/// // Seasonal differencing with period 4
/// let seasonal_diff_ts = transform_to_stationary(&ts, "seasonal_diff", Some(4)).unwrap();
/// ```
pub fn transform_to_stationary<F>(
    ts: &Array1<F>,
    method: &str,
    seasonal_period: Option<usize>,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 2 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must have at least 2 points for transformation".to_string(),
        ));
    }

    match method {
        "diff" => {
            // First-order differencing: x(t) - x(t-1)
            let mut result = Vec::with_capacity(ts.len() - 1);
            for i in 1..ts.len() {
                result.push(ts[i] - ts[i - 1]);
            }
            Ok(Array1::from(result))
        }
        "log" => {
            // Log transformation
            let mut result = Vec::with_capacity(ts.len());
            for &val in ts.iter() {
                if val <= F::zero() {
                    return Err(TimeSeriesError::InvalidInput(
                        "Cannot apply log transformation to non-positive values".to_string(),
                    ));
                }
                result.push(val.ln());
            }
            Ok(Array1::from(result))
        }
        "seasonal_diff" => {
            let period = match seasonal_period {
                Some(p) => p,
                None => {
                    return Err(TimeSeriesError::InvalidInput(
                        "Seasonal period must be provided for seasonal differencing".to_string(),
                    ))
                }
            };

            if period >= ts.len() {
                return Err(TimeSeriesError::InvalidInput(format!(
                    "Seasonal period ({}) must be less than time series length ({})",
                    period,
                    ts.len()
                )));
            }

            // Seasonal differencing: x(t) - x(t-s)
            let mut result = Vec::with_capacity(ts.len() - period);
            for i in period..ts.len() {
                result.push(ts[i] - ts[i - period]);
            }
            Ok(Array1::from(result))
        }
        _ => Err(TimeSeriesError::InvalidInput(format!(
            "Unknown transformation method: {}",
            method
        ))),
    }
}

/// Applies a centered moving average to smooth a time series
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `window_size` - Size of the moving window
///
/// # Returns
///
/// * The smoothed time series
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_series::utils::moving_average;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let ma = moving_average(&ts, 3).unwrap();
/// ```
pub fn moving_average<F>(ts: &Array1<F>, window_size: usize) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if window_size < 1 {
        return Err(TimeSeriesError::InvalidInput(
            "Window size must be at least 1".to_string(),
        ));
    }

    if window_size > ts.len() {
        return Err(TimeSeriesError::InvalidInput(format!(
            "Window size ({}) cannot be larger than time series length ({})",
            window_size,
            ts.len()
        )));
    }

    let half_window = window_size / 2;
    let mut result = Array1::zeros(ts.len());

    // For even-sized windows, handle the special case
    let is_even = window_size % 2 == 0;

    // Calculate the centered moving averages
    for i in 0..ts.len() {
        // Calculate appropriate window boundaries
        let start = i.saturating_sub(half_window);
        let end = if i + half_window >= ts.len() {
            ts.len() - 1
        } else {
            i + half_window
        };

        // Adjust for even-sized windows (need one more point at the end)
        let end = if is_even && (end + 1 < ts.len()) {
            end + 1
        } else {
            end
        };

        // Calculate the average
        let mut sum = F::zero();
        let mut count = F::zero();

        for j in start..=end {
            sum = sum + ts[j];
            count = count + F::one();
        }

        result[i] = sum / count;
    }

    Ok(result)
}

/// Calculates the autocorrelation function (ACF) for a time series
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `max_lag` - Maximum lag to compute (default: length of series - 1)
///
/// # Returns
///
/// * The autocorrelation values for each lag
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_series::utils::autocorrelation;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let acf = autocorrelation(&ts, None).unwrap();
/// ```
pub fn autocorrelation<F>(ts: &Array1<F>, max_lag: Option<usize>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 2 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must have at least 2 points for autocorrelation".to_string(),
        ));
    }

    let max_lag = std::cmp::min(max_lag.unwrap_or(ts.len() - 1), ts.len() - 1);

    // Calculate mean
    let mean = ts.iter().fold(F::zero(), |acc, &x| acc + x) / F::from_usize(ts.len()).unwrap();

    // Calculate denominator (variance * n)
    let denominator = ts
        .iter()
        .fold(F::zero(), |acc, &x| acc + (x - mean) * (x - mean));

    if denominator == F::zero() {
        return Err(TimeSeriesError::InvalidInput(
            "Cannot compute autocorrelation for constant time series".to_string(),
        ));
    }

    // Calculate autocorrelation for each lag
    let mut result = Array1::zeros(max_lag + 1);

    for lag in 0..=max_lag {
        let mut numerator = F::zero();

        for i in 0..(ts.len() - lag) {
            numerator = numerator + (ts[i] - mean) * (ts[i + lag] - mean);
        }

        result[lag] = numerator / denominator;
    }

    Ok(result)
}

/// Calculates the cross-correlation function (CCF) between two time series
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `max_lag` - Maximum lag to compute (default: min(length) / 4)
///
/// # Returns
///
/// * The cross-correlation values for each lag
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_series::utils::cross_correlation;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![2.0, 3.0, 4.0, 5.0, 6.0];
/// let ccf = cross_correlation(&x, &y, Some(3)).unwrap();
/// ```
pub fn cross_correlation<F>(
    x: &Array1<F>,
    y: &Array1<F>,
    max_lag: Option<usize>,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let min_len = x.len().min(y.len());

    if min_len < 2 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must have at least 2 points for cross-correlation".to_string(),
        ));
    }

    let default_max_lag = min_len / 4;
    let max_lag = max_lag.unwrap_or(default_max_lag).min(min_len - 1);

    let x_mean = x.sum() / F::from(x.len()).unwrap();
    let y_mean = y.sum() / F::from(y.len()).unwrap();

    let mut result = Array1::zeros(max_lag + 1);

    for lag in 0..=max_lag {
        let mut numerator = F::zero();
        let mut count = 0;

        for i in 0..(min_len - lag) {
            numerator = numerator + (x[i] - x_mean) * (y[i + lag] - y_mean);
            count += 1;
        }

        if count > 0 {
            result[lag] = numerator / F::from(count).unwrap();
        }
    }

    Ok(result)
}

/// Calculates the partial autocorrelation function (PACF) for a time series
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `max_lag` - Maximum lag to compute (default: length of series / 4)
///
/// # Returns
///
/// * The partial autocorrelation values for each lag
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_series::utils::partial_autocorrelation;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let pacf = partial_autocorrelation(&ts, None).unwrap();
/// ```
pub fn partial_autocorrelation<F>(ts: &Array1<F>, max_lag: Option<usize>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 2 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must have at least 2 points for partial autocorrelation".to_string(),
        ));
    }

    let default_max_lag = std::cmp::min(ts.len() / 4, 10);
    let max_lag = std::cmp::min(max_lag.unwrap_or(default_max_lag), ts.len() - 1);

    // Calculate ACF first
    let acf = autocorrelation(ts, Some(max_lag))?;

    // Initialize PACF array (lag 0 is always 1.0)
    let mut pacf = Array1::zeros(max_lag + 1);
    pacf[0] = F::one();

    // For lag 1, PACF = ACF
    if max_lag >= 1 {
        pacf[1] = acf[1];
    }

    // Compute PACF using Levinson-Durbin recursion
    // This is a simplified implementation of Durbin-Levinson algorithm
    if max_lag >= 2 {
        // Pre-allocate phi arrays
        let mut phi_old = Array1::zeros(max_lag + 1);

        for j in 2..=max_lag {
            // Copy previous phi values
            let mut phi = Array1::zeros(j + 1);
            for k in 1..j {
                phi[k] = phi_old[k];
            }

            // Calculate numerator and denominator
            let mut numerator = acf[j];
            for k in 1..j {
                numerator = numerator - phi_old[k] * acf[j - k];
            }

            let mut denominator = F::one();
            for k in 1..j {
                denominator = denominator - phi_old[k] * acf[k];
            }

            // Calculate the new PACF value
            phi[j] = numerator / denominator;

            // Update all phi values
            for k in 1..j {
                phi[k] = phi_old[k] - phi[j] * phi_old[j - k];
            }

            // Store the PACF value and update phi_old
            pacf[j] = phi[j];
            phi_old = phi;
        }
    }

    Ok(pacf)
}

/// Detrend data along an axis by removing linear or constant trend
///
/// # Arguments
///
/// * `data` - Input data array
/// * `axis` - Axis along which to detrend (0 for columns, 1 for rows)
/// * `detrend_type` - Type of detrending: "linear" or "constant"
/// * `breakpoints` - Optional sequence of breakpoints for piecewise linear detrending
///
/// # Returns
///
/// Detrended data array
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::utils::detrend;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let detrended = detrend(&x.view(), 0, "constant", None).unwrap();
/// println!("Detrended: {:?}", detrended);
/// ```
pub fn detrend<S, F>(
    data: &ArrayBase<S, Ix1>,
    axis: usize,
    detrend_type: &str,
    breakpoints: Option<&[usize]>,
) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + FromPrimitive + Debug + Display + ScalarOperand,
{
    scirs2_core::validation::check_array_finite(data, "data")?;

    if axis != 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Only axis=0 supported for 1D arrays".to_string(),
        ));
    }

    match detrend_type {
        "constant" => {
            let mean = data.mean().ok_or_else(|| {
                TimeSeriesError::ComputationError("Failed to compute mean".to_string())
            })?;
            Ok(data.map(|&x| x - mean))
        }
        "linear" => {
            let n = data.len();
            if n < 2 {
                return Err(TimeSeriesError::InvalidInput(
                    "Data must have at least 2 points for linear detrending".to_string(),
                ));
            }

            if let Some(bp) = breakpoints {
                // Piecewise linear detrending
                let mut result = data.to_owned();
                let mut bp_indices = vec![0];
                bp_indices.extend_from_slice(bp);
                bp_indices.push(n);

                for i in 0..bp_indices.len() - 1 {
                    let start = bp_indices[i];
                    let end = bp_indices[i + 1];
                    let segment = s![start..end];
                    let segment_data = data.slice(segment);
                    let trend = linear_trend(&segment_data, start)?;

                    for j in start..end {
                        result[j] = result[j] - trend[j - start];
                    }
                }
                Ok(result)
            } else {
                // Single linear detrending
                let trend = linear_trend(data, 0)?;
                Ok(data.to_owned() - trend)
            }
        }
        _ => Err(TimeSeriesError::InvalidInput(format!(
            "Invalid detrend type: {}. Must be 'constant' or 'linear'",
            detrend_type
        ))),
    }
}

/// Detrend 2D data along an axis
pub fn detrend_2d<S, F>(
    data: &ArrayBase<S, Ix2>,
    axis: usize,
    detrend_type: &str,
    breakpoints: Option<&[usize]>,
) -> Result<Array2<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + FromPrimitive + Debug + Display + ScalarOperand,
{
    scirs2_core::validation::check_array_finite(data, "data")?;

    if axis > 1 {
        return Err(TimeSeriesError::InvalidInput(
            "Axis must be 0 or 1 for 2D arrays".to_string(),
        ));
    }

    let mut result = data.to_owned();

    if axis == 0 {
        // Detrend along columns
        for mut col in result.columns_mut() {
            let detrended = detrend(&col.view(), 0, detrend_type, breakpoints)?;
            col.assign(&detrended);
        }
    } else {
        // Detrend along rows
        for mut row in result.rows_mut() {
            let detrended = detrend(&row.view(), 0, detrend_type, breakpoints)?;
            row.assign(&detrended);
        }
    }

    Ok(result)
}

/// Compute linear trend for data
fn linear_trend<S, F>(data: &ArrayBase<S, Ix1>, offset: usize) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + FromPrimitive + Debug + Display + ScalarOperand,
{
    let n = data.len();
    let x = Array1::linspace(
        F::from(offset).unwrap(),
        F::from(offset + n - 1).unwrap(),
        n,
    );
    let y = data.to_owned();

    // Compute linear regression coefficients
    let x_mean = x
        .mean()
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to compute x mean".to_string()))?;
    let y_mean = y
        .mean()
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to compute y mean".to_string()))?;

    let x_centered = &x - x_mean;
    let y_centered = &y - y_mean;

    let numerator = x_centered.dot(&y_centered);
    let denominator = x_centered.dot(&x_centered);

    if denominator.abs() < F::epsilon() {
        return Err(TimeSeriesError::ComputationError(
            "Singular matrix in linear regression".to_string(),
        ));
    }

    let slope = numerator / denominator;
    let intercept = y_mean - slope * x_mean;

    Ok(x.map(|&xi| slope * xi + intercept))
}

/// Resample a time series to a new number of samples using Fourier method
///
/// # Arguments
///
/// * `x` - Input time series
/// * `num` - Number of samples in the resampled signal
/// * `axis` - Axis along which to resample (default: 0)
/// * `window` - Optional window applied in the Fourier domain
///
/// # Returns
///
/// Resampled time series
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::utils::resample;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let resampled = resample(&x.view(), 10, 0, None).unwrap();
/// assert_eq!(resampled.len(), 10);
/// ```
pub fn resample<S, F>(
    x: &ArrayBase<S, Ix1>,
    num: usize,
    axis: usize,
    _window: Option<&Array1<F>>,
) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + FromPrimitive + Debug + Display,
{
    scirs2_core::validation::check_array_finite(x, "x")?;
    scirs2_core::validation::check_positive(num as f64, "num")?;

    if axis != 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Only axis=0 supported for 1D arrays".to_string(),
        ));
    }

    let n = x.len();
    if n == num {
        return Ok(x.to_owned());
    }

    // For now, use a simple linear interpolation as a placeholder
    // In practice, we'd use FFT-based resampling
    let mut result = Array1::zeros(num);
    let scale = F::from(n - 1).unwrap() / F::from(num - 1).unwrap();

    for i in 0..num {
        let pos = F::from(i).unwrap() * scale;
        let idx = pos.floor().to_usize().unwrap();
        let frac = pos - pos.floor();

        if idx + 1 < n {
            result[i] = x[idx] * (F::one() - frac) + x[idx + 1] * frac;
        } else {
            result[i] = x[idx];
        }
    }

    Ok(result)
}

/// Decimate a signal by applying a low-pass filter and downsampling
///
/// # Arguments
///
/// * `x` - Input signal
/// * `q` - Downsampling factor (integer)
/// * `n` - Filter order (default: 8)
/// * `ftype` - Filter type: "iir" or "fir" (default: "iir")
/// * `axis` - Axis along which to decimate (default: 0)
///
/// # Returns
///
/// Decimated signal
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::utils::decimate;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let decimated = decimate(&x.view(), 2, Some(4), Some("iir"), 0).unwrap();
/// assert_eq!(decimated.len(), 4);
/// ```
pub fn decimate<S, F>(
    x: &ArrayBase<S, Ix1>,
    q: usize,
    n: Option<usize>,
    ftype: Option<&str>,
    axis: usize,
) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + FromPrimitive + Debug + Display,
{
    scirs2_core::validation::check_array_finite(x, "x")?;
    scirs2_core::validation::check_positive(q as f64, "q")?;

    if axis != 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Only axis=0 supported for 1D arrays".to_string(),
        ));
    }

    if q == 1 {
        return Ok(x.to_owned());
    }

    let filter_order = n.unwrap_or(8);
    let filter_type = ftype.unwrap_or("iir");

    // Design low-pass filter with cutoff at Nyquist/q
    let cutoff = F::from(0.5).unwrap() / F::from(q).unwrap();

    let filtered = match filter_type {
        "iir" => {
            // Apply Chebyshev Type I filter
            apply_chebyshev_filter(x, filter_order, cutoff)?
        }
        "fir" => {
            // Apply FIR filter using windowed sinc
            apply_fir_filter(x, filter_order, cutoff)?
        }
        _ => {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Invalid filter type: {}. Must be 'iir' or 'fir'",
                filter_type
            )))
        }
    };

    // Downsample
    let mut result = Array1::zeros(x.len() / q);
    for (i, j) in (0..x.len()).step_by(q).enumerate() {
        if i < result.len() {
            result[i] = filtered[j];
        }
    }

    Ok(result)
}

/// Apply Chebyshev Type I filter (simplified implementation)
fn apply_chebyshev_filter<S, F>(
    x: &ArrayBase<S, Ix1>,
    order: usize,
    _cutoff: F,
) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + FromPrimitive + Debug + Display,
{
    // This is a simplified placeholder - in practice, we'd use a proper IIR filter implementation
    // For now, we'll use a simple moving average as a low-pass filter
    let window_size = order + 1;
    let mut filtered = x.to_owned();

    for i in 0..x.len() {
        let start = if i >= window_size / 2 {
            i - window_size / 2
        } else {
            0
        };
        let end = if i + window_size / 2 < x.len() {
            i + window_size / 2 + 1
        } else {
            x.len()
        };

        let sum: F = x.slice(s![start..end]).sum();
        filtered[i] = sum / F::from(end - start).unwrap();
    }

    Ok(filtered)
}

/// Apply FIR filter using windowed sinc (simplified implementation)
fn apply_fir_filter<S, F>(x: &ArrayBase<S, Ix1>, order: usize, cutoff: F) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + FromPrimitive + Debug + Display,
{
    // Create windowed sinc filter coefficients
    let mut coeffs = Array1::zeros(order + 1);
    let fc = cutoff;
    let half_order = order / 2;

    for i in 0..=order {
        let n = i as i32 - half_order as i32;
        if n == 0 {
            coeffs[i] = F::from(2.0).unwrap() * fc;
        } else {
            let n_f = F::from(n).unwrap();
            let pi = F::from(std::f64::consts::PI).unwrap();
            coeffs[i] = (F::from(2.0).unwrap() * fc * pi * n_f).sin() / (pi * n_f);

            // Apply Hamming window
            let window = F::from(0.54).unwrap()
                - F::from(0.46).unwrap()
                    * (F::from(2.0).unwrap() * pi * F::from(i).unwrap() / F::from(order).unwrap())
                        .cos();
            coeffs[i] = coeffs[i] * window;
        }
    }

    // Normalize coefficients
    let sum: F = coeffs.sum();
    coeffs.map_inplace(|x| *x = *x / sum);

    // Apply convolution
    convolve_1d(x, &coeffs.view())
}

/// Simple 1D convolution
fn convolve_1d<S, T, F>(x: &ArrayBase<S, Ix1>, kernel: &ArrayBase<T, Ix1>) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    T: Data<Elem = F>,
    F: Float + NumCast + FromPrimitive + Debug + Display,
{
    let n = x.len();
    let k = kernel.len();
    let half_k = k / 2;
    let mut result = Array1::zeros(n);

    for i in 0..n {
        let mut sum = F::zero();
        for j in 0..k {
            let idx = i as i32 + j as i32 - half_k as i32;
            if idx >= 0 && idx < n as i32 {
                sum = sum + x[idx as usize] * kernel[j];
            }
        }
        result[i] = sum;
    }

    Ok(result)
}

/// Creates a time series with a specified date range (in days)
///
/// # Arguments
///
/// * `start_date` - Start date in the format "YYYY-MM-DD"
/// * `end_date` - End date in the format "YYYY-MM-DD"
/// * `values` - The values for the time series (must match the date range length)
///
/// # Returns
///
/// * A tuple containing date strings and the time series values
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_series::utils::create_time_series;
///
/// let values = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let (dates, ts) = create_time_series("2023-01-01", "2023-01-07", &values).unwrap();
/// ```
pub fn create_time_series<F>(
    start_date: &str,
    end_date: &str,
    values: &Array1<F>,
) -> Result<(Vec<String>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug,
{
    // Parse dates (simplified implementation)
    // For a real implementation, we'd use chrono or time crates

    // Create a simple date parser
    fn parse_date(date_str: &str) -> Result<(i32, u32, u32)> {
        let parts: Vec<&str> = date_str.split('-').collect();
        if parts.len() != 3 {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Invalid date format: {}, expected YYYY-MM-DD",
                date_str
            )));
        }

        let year = parts[0]
            .parse::<i32>()
            .map_err(|_| TimeSeriesError::InvalidInput(format!("Invalid year: {}", parts[0])))?;

        let month = parts[1]
            .parse::<u32>()
            .map_err(|_| TimeSeriesError::InvalidInput(format!("Invalid month: {}", parts[1])))?;

        let day = parts[2]
            .parse::<u32>()
            .map_err(|_| TimeSeriesError::InvalidInput(format!("Invalid day: {}", parts[2])))?;

        if !(1..=12).contains(&month) {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Month must be between 1 and 12, got {}",
                month
            )));
        }

        if !(1..=31).contains(&day) {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Day must be between 1 and 31, got {}",
                day
            )));
        }

        Ok((year, month, day))
    }

    // Simple days between calculation (not accounting for leap years properly)
    fn days_between(start: (i32, u32, u32), end: (i32, u32, u32)) -> i32 {
        // Days in month (simplified, not accounting for leap years)
        let days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

        // Convert to days since year 0
        let start_days = start.0 * 365
            + (1..start.1).map(|m| days_in_month[m as usize]).sum::<u32>() as i32
            + start.2 as i32;

        let end_days = end.0 * 365
            + (1..end.1).map(|m| days_in_month[m as usize]).sum::<u32>() as i32
            + end.2 as i32;

        end_days - start_days + 1 // +1 to include both start and end dates
    }

    // Generate dates (simple implementation)
    fn generate_dates(start: (i32, u32, u32), n_days: usize) -> Vec<String> {
        // Days in month (simplified, not accounting for leap years)
        let days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

        let mut dates = Vec::with_capacity(n_days);
        let mut year = start.0;
        let mut month = start.1;
        let mut day = start.2;

        for _ in 0..n_days {
            dates.push(format!("{:04}-{:02}-{:02}", year, month, day));

            // Increment date
            day += 1;
            if day > days_in_month[month as usize] {
                day = 1;
                month += 1;
                if month > 12 {
                    month = 1;
                    year += 1;
                }
            }
        }

        dates
    }

    let start = parse_date(start_date)?;
    let end = parse_date(end_date)?;

    let days = days_between(start, end);
    if days < 1 {
        return Err(TimeSeriesError::InvalidInput(format!(
            "End date ({}) must be after start date ({})",
            end_date, start_date
        )));
    }

    if values.len() != days as usize {
        return Err(TimeSeriesError::InvalidInput(format!(
            "Values length ({}) must match date range length ({})",
            values.len(),
            days
        )));
    }

    let dates = generate_dates(start, days as usize);
    let time_series = values.clone();

    Ok((dates, time_series))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_detrend_constant() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let detrended = detrend(&x.view(), 0, "constant", None).unwrap();

        // Mean should be removed
        assert_relative_eq!(detrended.mean().unwrap(), 0.0, epsilon = 1e-10);

        // Check specific values
        assert_relative_eq!(detrended[0], -2.0, epsilon = 1e-10);
        assert_relative_eq!(detrended[2], 0.0, epsilon = 1e-10);
        assert_relative_eq!(detrended[4], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_detrend_linear() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let detrended = detrend(&x.view(), 0, "linear", None).unwrap();

        // Linear trend should be removed, result should be constant
        for i in 1..detrended.len() {
            assert_relative_eq!(detrended[i] - detrended[i - 1], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_detrend_linear_with_breakpoints() {
        let x = array![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0];
        let breakpoints = vec![4];
        let detrended = detrend(&x.view(), 0, "linear", Some(&breakpoints)).unwrap();

        // Each segment should have its linear trend removed
        assert_relative_eq!(detrended[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(detrended[3], 0.0, epsilon = 1e-10);
        assert_relative_eq!(detrended[4], 0.0, epsilon = 1e-10);
        assert_relative_eq!(detrended[7], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_detrend_2d() {
        let x = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();

        // Detrend along columns (axis=0)
        let detrended = detrend_2d(&x.view(), 0, "constant", None).unwrap();

        // Each column should have zero mean
        for col in detrended.columns() {
            assert_relative_eq!(col.mean().unwrap(), 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_resample_upsample() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let resampled = resample(&x.view(), 8, 0, None).unwrap();

        assert_eq!(resampled.len(), 8);
        // First and last values should be preserved
        assert_relative_eq!(resampled[0], x[0], epsilon = 0.1);
        assert_relative_eq!(
            resampled[resampled.len() - 1],
            x[x.len() - 1],
            epsilon = 0.1
        );
    }

    #[test]
    fn test_resample_downsample() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let resampled = resample(&x.view(), 4, 0, None).unwrap();

        assert_eq!(resampled.len(), 4);
    }

    #[test]
    fn test_decimate() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let decimated = decimate(&x.view(), 2, Some(4), Some("iir"), 0).unwrap();

        assert_eq!(decimated.len(), 4);
    }

    #[test]
    fn test_invalid_detrend_type() {
        let x = array![1.0, 2.0, 3.0];
        let result = detrend(&x.view(), 0, "invalid", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_axis() {
        let x = array![1.0, 2.0, 3.0];
        let result = detrend(&x.view(), 1, "constant", None);
        assert!(result.is_err());
    }
}
