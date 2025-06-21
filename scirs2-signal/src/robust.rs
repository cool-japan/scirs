//! Robust Filtering module
//!
//! This module implements robust filtering techniques for handling outliers
//! and non-Gaussian noise in signal and image processing applications.
//!
//! Robust filters are designed to be insensitive to outliers and provide
//! better performance than traditional linear filters when dealing with
//! impulsive noise or contaminated data.
//!
//! The implementation includes:
//! - Alpha-trimmed mean filtering
//! - Hampel filter for outlier detection and replacement
//! - Winsorized filtering
//! - Huber loss-based robust filtering
//! - Adaptive robust filtering
//!
//! # Example
//! ```
//! use ndarray::Array1;
//! use scirs2_signal::robust::{alpha_trimmed_filter, hampel_filter};
//!
//! // Create a test signal with outliers
//! let signal = Array1::from_vec(vec![1.0, 1.2, 1.1, 10.0, 1.3, 1.2, -5.0, 1.1]);
//!
//! // Apply alpha-trimmed mean filter
//! let filtered = alpha_trimmed_filter(&signal, 5, 0.2).unwrap();
//!
//! // Apply Hampel filter for outlier detection
//! let (cleaned, outliers) = hampel_filter(&signal, 5, 3.0).unwrap();
//! ```

use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array1, Array2};

/// Configuration for robust filtering algorithms
#[derive(Debug, Clone)]
pub struct RobustConfig {
    /// Edge handling mode
    pub edge_mode: EdgeMode,

    /// Whether to return outlier positions
    pub return_outliers: bool,

    /// Parallelization enabled
    pub parallel: bool,
}

impl Default for RobustConfig {
    fn default() -> Self {
        Self {
            edge_mode: EdgeMode::Reflect,
            return_outliers: false,
            parallel: false,
        }
    }
}

/// Edge handling mode for robust filtering
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeMode {
    /// Reflect the signal at boundaries
    Reflect,
    /// Pad with the nearest valid value
    Nearest,
    /// Pad with a constant value
    Constant(f64),
    /// Wrap around (circular padding)
    Wrap,
}

/// Alpha-trimmed mean filter for robust signal processing
///
/// This filter removes the α% largest and smallest values from a local window
/// and computes the mean of the remaining values. It provides robustness
/// against outliers while maintaining computational efficiency.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Size of the filtering window (must be odd and >= 3)
/// * `alpha` - Trimming fraction (0.0 to 0.5). Higher values remove more outliers.
///
/// # Returns
///
/// * Filtered signal with the same length as input
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_signal::robust::alpha_trimmed_filter;
///
/// let signal = Array1::from_vec(vec![1.0, 1.2, 10.0, 1.1, 1.3]);
/// let filtered = alpha_trimmed_filter(&signal, 3, 0.2).unwrap();
/// ```
pub fn alpha_trimmed_filter(
    signal: &Array1<f64>,
    window_size: usize,
    alpha: f64,
) -> SignalResult<Array1<f64>> {
    if window_size % 2 == 0 || window_size < 3 {
        return Err(SignalError::ValueError(
            "Window size must be odd and >= 3".to_string(),
        ));
    }

    if !(0.0..=0.5).contains(&alpha) {
        return Err(SignalError::ValueError(
            "Alpha must be between 0.0 and 0.5".to_string(),
        ));
    }

    let n = signal.len();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let half_window = window_size / 2;
    let mut result = Array1::zeros(n);

    // Number of samples to trim from each end
    let trim_count = (window_size as f64 * alpha).floor() as usize;
    let keep_count = window_size - 2 * trim_count;

    if keep_count == 0 {
        return Err(SignalError::ValueError(
            "Alpha value too large for given window size".to_string(),
        ));
    }

    for i in 0..n {
        // Determine window bounds with boundary handling
        let start = i.saturating_sub(half_window);
        let end = if i + half_window < n {
            i + half_window + 1
        } else {
            n
        };

        // Extract window values
        let mut window_values: Vec<f64> = signal.slice(s![start..end]).to_vec();

        // Handle edge cases by padding if necessary
        while window_values.len() < window_size {
            if start == 0 {
                // Pad at beginning by reflecting
                window_values.insert(0, window_values[0]);
            } else {
                // Pad at end by reflecting
                window_values.push(*window_values.last().unwrap());
            }
        }

        // Sort window values
        window_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Trim alpha portion from both ends and compute mean
        let trimmed_values = &window_values[trim_count..window_values.len() - trim_count];
        let trimmed_mean = trimmed_values.iter().sum::<f64>() / trimmed_values.len() as f64;

        result[i] = trimmed_mean;
    }

    Ok(result)
}

/// Hampel filter for outlier detection and replacement
///
/// The Hampel filter detects outliers by comparing each point to the median
/// of its local neighborhood. Outliers are identified when they deviate more
/// than k times the median absolute deviation (MAD) from the local median.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Size of the filtering window (must be odd and >= 3)
/// * `k` - Threshold factor (typically 2.0 to 3.0)
///
/// # Returns
///
/// * Tuple of (filtered_signal, outlier_indices)
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_signal::robust::hampel_filter;
///
/// let signal = Array1::from_vec(vec![1.0, 1.2, 10.0, 1.1, 1.3]);
/// let (filtered, outliers) = hampel_filter(&signal, 3, 3.0).unwrap();
/// ```
pub fn hampel_filter(
    signal: &Array1<f64>,
    window_size: usize,
    k: f64,
) -> SignalResult<(Array1<f64>, Vec<usize>)> {
    if window_size % 2 == 0 || window_size < 3 {
        return Err(SignalError::ValueError(
            "Window size must be odd and >= 3".to_string(),
        ));
    }

    if k <= 0.0 {
        return Err(SignalError::ValueError(
            "Threshold factor k must be positive".to_string(),
        ));
    }

    let n = signal.len();
    if n == 0 {
        return Ok((Array1::zeros(0), Vec::new()));
    }

    let half_window = window_size / 2;
    let mut result = signal.clone();
    let mut outlier_indices = Vec::new();

    for i in 0..n {
        // Determine window bounds
        let start = i.saturating_sub(half_window);
        let end = if i + half_window < n {
            i + half_window + 1
        } else {
            n
        };

        // Extract window values
        let mut window_values: Vec<f64> = signal.slice(s![start..end]).to_vec();

        // Handle edge cases
        while window_values.len() < window_size {
            if start == 0 {
                window_values.insert(0, window_values[0]);
            } else {
                window_values.push(*window_values.last().unwrap());
            }
        }

        // Calculate median
        let mut sorted_values = window_values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted_values.len() % 2 == 0 {
            let mid = sorted_values.len() / 2;
            (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        // Calculate MAD (Median Absolute Deviation)
        let mut abs_deviations: Vec<f64> =
            window_values.iter().map(|&x| (x - median).abs()).collect();
        abs_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mad = if abs_deviations.len() % 2 == 0 {
            let mid = abs_deviations.len() / 2;
            (abs_deviations[mid - 1] + abs_deviations[mid]) / 2.0
        } else {
            abs_deviations[abs_deviations.len() / 2]
        };

        // Check if current point is an outlier
        let current_value = signal[i];
        let deviation = (current_value - median).abs();

        if mad > 0.0 && deviation > k * mad {
            // Point is an outlier - replace with median
            result[i] = median;
            outlier_indices.push(i);
        }
    }

    Ok((result, outlier_indices))
}

/// Winsorized filter for robust signal processing
///
/// This filter replaces extreme values with the nearest non-extreme values.
/// Values below the p-th percentile are replaced with the p-th percentile value,
/// and values above the (100-p)-th percentile are replaced with the (100-p)-th percentile value.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Size of the filtering window (must be odd and >= 3)
/// * `percentile` - Percentile for winsorization (0.0 to 50.0)
///
/// # Returns
///
/// * Filtered signal
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_signal::robust::winsorize_filter;
///
/// let signal = Array1::from_vec(vec![1.0, 1.2, 10.0, 1.1, 1.3]);
/// let filtered = winsorize_filter(&signal, 3, 10.0).unwrap();
/// ```
pub fn winsorize_filter(
    signal: &Array1<f64>,
    window_size: usize,
    percentile: f64,
) -> SignalResult<Array1<f64>> {
    if window_size % 2 == 0 || window_size < 3 {
        return Err(SignalError::ValueError(
            "Window size must be odd and >= 3".to_string(),
        ));
    }

    if !(0.0..=50.0).contains(&percentile) {
        return Err(SignalError::ValueError(
            "Percentile must be between 0.0 and 50.0".to_string(),
        ));
    }

    let n = signal.len();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let half_window = window_size / 2;
    let mut result = Array1::zeros(n);

    for i in 0..n {
        // Determine window bounds
        let start = i.saturating_sub(half_window);
        let end = if i + half_window < n {
            i + half_window + 1
        } else {
            n
        };

        // Extract and sort window values
        let mut window_values: Vec<f64> = signal.slice(s![start..end]).to_vec();

        // Handle edge cases
        while window_values.len() < window_size {
            if start == 0 {
                window_values.insert(0, window_values[0]);
            } else {
                window_values.push(*window_values.last().unwrap());
            }
        }

        window_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate percentile indices
        let lower_idx = ((percentile / 100.0) * (window_values.len() - 1) as f64) as usize;
        let upper_idx =
            (((100.0 - percentile) / 100.0) * (window_values.len() - 1) as f64) as usize;

        let lower_threshold = window_values[lower_idx];
        let upper_threshold = window_values[upper_idx];

        // Winsorize the center value
        let current_value = signal[i];
        result[i] = if current_value < lower_threshold {
            lower_threshold
        } else if current_value > upper_threshold {
            upper_threshold
        } else {
            current_value
        };
    }

    Ok(result)
}

/// Huber loss-based robust filter
///
/// This filter uses the Huber loss function to provide robustness against outliers
/// while maintaining efficiency for inliers. The Huber loss is quadratic for small
/// residuals and linear for large residuals.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Size of the filtering window (must be odd and >= 3)
/// * `delta` - Threshold parameter for Huber loss (transition point between quadratic and linear)
///
/// # Returns
///
/// * Filtered signal
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_signal::robust::huber_filter;
///
/// let signal = Array1::from_vec(vec![1.0, 1.2, 10.0, 1.1, 1.3]);
/// let filtered = huber_filter(&signal, 3, 1.35).unwrap();
/// ```
pub fn huber_filter(
    signal: &Array1<f64>,
    window_size: usize,
    delta: f64,
) -> SignalResult<Array1<f64>> {
    if window_size % 2 == 0 || window_size < 3 {
        return Err(SignalError::ValueError(
            "Window size must be odd and >= 3".to_string(),
        ));
    }

    if delta <= 0.0 {
        return Err(SignalError::ValueError(
            "Delta parameter must be positive".to_string(),
        ));
    }

    let n = signal.len();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let half_window = window_size / 2;
    let mut result = Array1::zeros(n);

    for i in 0..n {
        // Determine window bounds
        let start = i.saturating_sub(half_window);
        let end = if i + half_window < n {
            i + half_window + 1
        } else {
            n
        };

        // Extract window values
        let mut window_values: Vec<f64> = signal.slice(s![start..end]).to_vec();

        // Handle edge cases
        while window_values.len() < window_size {
            if start == 0 {
                window_values.insert(0, window_values[0]);
            } else {
                window_values.push(*window_values.last().unwrap());
            }
        }

        // Calculate robust estimate using Huber loss iteratively
        let mut estimate = window_values.iter().sum::<f64>() / window_values.len() as f64; // Start with mean

        // Iterative reweighting for Huber loss
        for _iter in 0..10 {
            // Maximum 10 iterations
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for &value in &window_values {
                let residual = value - estimate;
                let abs_residual = residual.abs();

                let weight = if abs_residual <= delta {
                    1.0 // Quadratic regime
                } else {
                    delta / abs_residual // Linear regime
                };

                weighted_sum += weight * value;
                weight_sum += weight;
            }

            let new_estimate = if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                estimate
            };

            // Check for convergence
            if (new_estimate - estimate).abs() < 1e-6 {
                break;
            }
            estimate = new_estimate;
        }

        result[i] = estimate;
    }

    Ok(result)
}

/// Apply robust filter to 2D data (images)
///
/// This function applies any of the 1D robust filters to each row and then each column
/// of a 2D array for robust image filtering.
///
/// # Arguments
///
/// * `image` - Input 2D array
/// * `filter_fn` - 1D robust filter function to apply
/// * `window_size` - Size of the filtering window
/// * `param` - Additional parameter for the filter function
///
/// # Returns
///
/// * Filtered 2D array
pub fn robust_filter_2d<F>(
    image: &Array2<f64>,
    filter_fn: F,
    window_size: usize,
    param: f64,
) -> SignalResult<Array2<f64>>
where
    F: Fn(&Array1<f64>, usize, f64) -> SignalResult<Array1<f64>>,
{
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Ok(Array2::zeros((0, 0)));
    }

    let mut result = image.clone();

    // Filter rows
    for i in 0..rows {
        let row = image.row(i).to_owned();
        let filtered_row = filter_fn(&row, window_size, param)?;
        result.row_mut(i).assign(&filtered_row);
    }

    // Filter columns
    for j in 0..cols {
        let col = result.column(j).to_owned();
        let filtered_col = filter_fn(&col, window_size, param)?;
        result.column_mut(j).assign(&filtered_col);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_trimmed_filter() {
        let signal = Array1::from_vec(vec![1.0, 1.2, 10.0, 1.1, 1.3, 1.15, -5.0, 1.25]);
        let filtered = alpha_trimmed_filter(&signal, 3, 0.3).unwrap();

        assert_eq!(filtered.len(), signal.len());

        // The outliers (10.0 and -5.0) should be reduced
        assert!(filtered[2] < signal[2]); // 10.0 should be reduced
        assert!(filtered[6] > signal[6]); // -5.0 should be increased
    }

    #[test]
    fn test_hampel_filter() {
        let signal = Array1::from_vec(vec![1.0, 1.2, 1.1, 10.0, 1.3, 1.2, 1.1]);
        let (filtered, outliers) = hampel_filter(&signal, 3, 3.0).unwrap();

        assert_eq!(filtered.len(), signal.len());
        assert!(!outliers.is_empty()); // Should detect the outlier at index 3
        assert!(outliers.contains(&3)); // Index 3 has the outlier (10.0)
    }

    #[test]
    fn test_winsorize_filter() {
        let signal = Array1::from_vec(vec![1.0, 1.2, 1.1, 10.0, 1.3, 1.2, 1.1]);
        let filtered = winsorize_filter(&signal, 5, 20.0).unwrap();

        assert_eq!(filtered.len(), signal.len());
        // Extreme values should be winsorized
        assert!(filtered[3] <= signal[3]); // 10.0 should be reduced
    }

    #[test]
    fn test_huber_filter() {
        let signal = Array1::from_vec(vec![1.0, 1.2, 1.1, 10.0, 1.3, 1.2, 1.1]);
        let filtered = huber_filter(&signal, 3, 1.0).unwrap();

        assert_eq!(filtered.len(), signal.len());
        // All values should be finite
        for &val in filtered.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_robust_filter_2d() {
        let image =
            Array2::from_shape_vec((3, 3), vec![1.0, 1.2, 1.1, 1.1, 10.0, 1.2, 1.3, 1.2, 1.1])
                .unwrap();

        let filtered = robust_filter_2d(&image, alpha_trimmed_filter, 3, 0.2).unwrap();

        assert_eq!(filtered.dim(), image.dim());
        // The outlier (10.0) should be reduced
        assert!(filtered[[1, 1]] < image[[1, 1]]);
    }

    #[test]
    fn test_edge_cases() {
        // Empty signal
        let empty_signal = Array1::zeros(0);
        let result = alpha_trimmed_filter(&empty_signal, 3, 0.2).unwrap();
        assert_eq!(result.len(), 0);

        // Small signal
        let small_signal = Array1::from_vec(vec![1.0, 2.0]);
        let result = alpha_trimmed_filter(&small_signal, 3, 0.2).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_parameter_validation() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Invalid window size (even)
        assert!(alpha_trimmed_filter(&signal, 4, 0.2).is_err());

        // Invalid alpha (too large)
        assert!(alpha_trimmed_filter(&signal, 3, 0.6).is_err());

        // Invalid k parameter for Hampel filter
        assert!(hampel_filter(&signal, 3, -1.0).is_err());

        // Invalid percentile for Winsorize filter
        assert!(winsorize_filter(&signal, 3, 60.0).is_err());

        // Invalid delta for Huber filter
        assert!(huber_filter(&signal, 3, -1.0).is_err());
    }
}
