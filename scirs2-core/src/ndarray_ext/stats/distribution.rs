//! Distribution-related functions for ndarray arrays
//!
//! This module provides functions for working with data distributions,
//! including histograms, binning, and quantile calculations.

use ndarray::{Array, ArrayView, Ix1, Ix2};
use num_traits::{Float, FromPrimitive};

/// Result type for histogram function
pub type HistogramResult<T> = Result<(Array<usize, Ix1>, Array<T, Ix1>), &'static str>;

/// Result type for histogram2d function
pub type Histogram2dResult<T> =
    Result<(Array<usize, Ix2>, Array<T, Ix1>, Array<T, Ix1>), &'static str>;

/// Calculate a histogram of data
///
/// # Arguments
///
/// * `array` - The input 1D array
/// * `bins` - The number of bins
/// * `range` - Optional tuple (min, max) to use. If None, the range is based on data
/// * `weights` - Optional array of weights for each data point
///
/// # Returns
///
/// A tuple containing (histogram, bin_edges)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::histogram;
///
/// let data = array![0.1, 0.5, 1.1, 1.5, 2.2, 2.9, 3.1, 3.8, 4.1, 4.9];
/// let (hist, bin_edges) = histogram(data.view(), 5, None, None).unwrap();
///
/// assert_eq!(hist.len(), 5);
/// assert_eq!(bin_edges.len(), 6);
/// ```
pub fn histogram<T>(
    array: ArrayView<T, Ix1>,
    bins: usize,
    range: Option<(T, T)>,
    weights: Option<ArrayView<T, Ix1>>,
) -> HistogramResult<T>
where
    T: Clone + Float + FromPrimitive,
{
    if array.is_empty() {
        return Err("Cannot compute histogram of an empty array");
    }

    if bins == 0 {
        return Err("Number of bins must be positive");
    }

    // Get range (min, max) of the data
    let (min_val, max_val) = match range {
        Some(r) => r,
        None => {
            let mut min_val = T::infinity();
            let mut max_val = T::neg_infinity();

            for &val in array.iter() {
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }
            (min_val, max_val)
        }
    };

    if min_val >= max_val {
        return Err("Range must be (min, max) with min < max");
    }

    // Create bin edges
    let mut bin_edges = Array::<T, Ix1>::zeros(bins + 1);
    let bin_width = (max_val - min_val) / T::from_usize(bins).unwrap();

    for i in 0..=bins {
        bin_edges[i] = min_val + bin_width * T::from_usize(i).unwrap();
    }

    // Ensure the last bin edge is exactly max_val
    bin_edges[bins] = max_val;

    // Initialize histogram array
    let mut hist = Array::<usize, Ix1>::zeros(bins);

    // Fill histogram
    match weights {
        Some(w) => {
            if w.len() != array.len() {
                return Err("Weights array must have the same length as the data array");
            }

            for (&val, &weight) in array.iter().zip(w.iter()) {
                // Skip values outside the range
                if val < min_val || val > max_val {
                    continue;
                }

                // Handle edge case where val == max_val (include in the last bin)
                if val == max_val {
                    hist[bins - 1] += 1;
                    continue;
                }

                // Find bin index
                let scaled_val = (val - min_val) / bin_width;
                let bin_idx = scaled_val.to_usize().unwrap_or(0);
                let bin_idx = bin_idx.min(bins - 1); // Ensure index is in bounds

                // Add to histogram (with weight)
                let weight_int = weight.to_usize().unwrap_or(1);
                hist[bin_idx] += weight_int;
            }
        }
        None => {
            for &val in array.iter() {
                // Skip values outside the range
                if val < min_val || val > max_val {
                    continue;
                }

                // Handle edge case where val == max_val (include in the last bin)
                if val == max_val {
                    hist[bins - 1] += 1;
                    continue;
                }

                // Find bin index
                let scaled_val = (val - min_val) / bin_width;
                let bin_idx = scaled_val.to_usize().unwrap_or(0);
                let bin_idx = bin_idx.min(bins - 1); // Ensure index is in bounds

                // Add to histogram
                hist[bin_idx] += 1;
            }
        }
    }

    Ok((hist, bin_edges))
}

/// Calculate a 2D histogram of data
///
/// # Arguments
///
/// * `x` - The x coordinates of the data points
/// * `y` - The y coordinates of the data points
/// * `bins` - Either a tuple (x_bins, y_bins) for the number of bins, or None for 10 bins in each direction
/// * `range` - Optional tuple ((x_min, x_max), (y_min, y_max)) to use. If None, the range is based on data
/// * `weights` - Optional array of weights for each data point
///
/// # Returns
///
/// A tuple containing (histogram, x_edges, y_edges)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::histogram2d;
///
/// let x = array![0.1, 0.5, 1.3, 2.5, 3.1, 3.8, 4.2, 4.9];
/// let y = array![0.2, 0.8, 1.5, 2.0, 3.0, 3.2, 3.5, 4.5];
/// let (hist, x_edges, y_edges) = histogram2d(x.view(), y.view(), Some((4, 4)), None, None).unwrap();
///
/// assert_eq!(hist.shape(), &[4, 4]);
/// assert_eq!(x_edges.len(), 5);
/// assert_eq!(y_edges.len(), 5);
/// ```
pub fn histogram2d<T>(
    x: ArrayView<T, Ix1>,
    y: ArrayView<T, Ix1>,
    bins: Option<(usize, usize)>,
    range: Option<((T, T), (T, T))>,
    weights: Option<ArrayView<T, Ix1>>,
) -> Histogram2dResult<T>
where
    T: Clone + Float + FromPrimitive,
{
    if x.is_empty() || y.is_empty() {
        return Err("Cannot compute histogram of empty arrays");
    }

    if x.len() != y.len() {
        return Err("x and y arrays must have the same length");
    }

    // Default to 10 bins in each direction if not specified
    let (x_bins, y_bins) = bins.unwrap_or((10, 10));

    if x_bins == 0 || y_bins == 0 {
        return Err("Number of bins must be positive");
    }

    // Get range for x and y
    let ((x_min, x_max), (y_min, y_max)) = match range {
        Some(r) => r,
        None => {
            let mut x_min = T::infinity();
            let mut x_max = T::neg_infinity();
            let mut y_min = T::infinity();
            let mut y_max = T::neg_infinity();

            for (&x_val, &y_val) in x.iter().zip(y.iter()) {
                if x_val < x_min {
                    x_min = x_val;
                }
                if x_val > x_max {
                    x_max = x_val;
                }
                if y_val < y_min {
                    y_min = y_val;
                }
                if y_val > y_max {
                    y_max = y_val;
                }
            }
            ((x_min, x_max), (y_min, y_max))
        }
    };

    if x_min >= x_max || y_min >= y_max {
        return Err("Range must be (min, max) with min < max");
    }

    // Create bin edges
    let mut x_edges = Array::<T, Ix1>::zeros(x_bins + 1);
    let mut y_edges = Array::<T, Ix1>::zeros(y_bins + 1);

    let x_bin_width = (x_max - x_min) / T::from_usize(x_bins).unwrap();
    let y_bin_width = (y_max - y_min) / T::from_usize(y_bins).unwrap();

    for i in 0..=x_bins {
        x_edges[i] = x_min + x_bin_width * T::from_usize(i).unwrap();
    }

    for i in 0..=y_bins {
        y_edges[i] = y_min + y_bin_width * T::from_usize(i).unwrap();
    }

    // Ensure the last bin edges are exactly max values
    x_edges[x_bins] = x_max;
    y_edges[y_bins] = y_max;

    // Initialize histogram array
    let mut hist = Array::<usize, Ix2>::zeros((y_bins, x_bins));

    // Fill histogram
    match weights {
        Some(w) => {
            if w.len() != x.len() {
                return Err("Weights array must have the same length as the data arrays");
            }

            for ((&x_val, &y_val), &weight) in x.iter().zip(y.iter()).zip(w.iter()) {
                // Skip values outside the range
                if x_val < x_min || x_val > x_max || y_val < y_min || y_val > y_max {
                    continue;
                }

                // Find bin indices
                let x_scaled = (x_val - x_min) / x_bin_width;
                let y_scaled = (y_val - y_min) / y_bin_width;

                let mut x_idx = x_scaled.to_usize().unwrap_or(0);
                let mut y_idx = y_scaled.to_usize().unwrap_or(0);

                // Handle edge cases where val == max_val
                if x_val == x_max {
                    x_idx = x_bins - 1;
                } else {
                    x_idx = x_idx.min(x_bins - 1);
                }

                if y_val == y_max {
                    y_idx = y_bins - 1;
                } else {
                    y_idx = y_idx.min(y_bins - 1);
                }

                // Add to histogram (with weight)
                let weight_int = weight.to_usize().unwrap_or(1);
                hist[[y_idx, x_idx]] += weight_int;
            }
        }
        None => {
            for (&x_val, &y_val) in x.iter().zip(y.iter()) {
                // Skip values outside the range
                if x_val < x_min || x_val > x_max || y_val < y_min || y_val > y_max {
                    continue;
                }

                // Find bin indices
                let x_scaled = (x_val - x_min) / x_bin_width;
                let y_scaled = (y_val - y_min) / y_bin_width;

                let mut x_idx = x_scaled.to_usize().unwrap_or(0);
                let mut y_idx = y_scaled.to_usize().unwrap_or(0);

                // Handle edge cases where val == max_val
                if x_val == x_max {
                    x_idx = x_bins - 1;
                } else {
                    x_idx = x_idx.min(x_bins - 1);
                }

                if y_val == y_max {
                    y_idx = y_bins - 1;
                } else {
                    y_idx = y_idx.min(y_bins - 1);
                }

                // Add to histogram
                hist[[y_idx, x_idx]] += 1;
            }
        }
    }

    Ok((hist, x_edges, y_edges))
}

/// Calculate the quantile values from a 1D array
///
/// # Arguments
///
/// * `array` - The input 1D array
/// * `q` - The quantile or array of quantiles to compute (between 0 and 1)
/// * `method` - The interpolation method to use: "linear" (default), "lower", "higher", "midpoint", or "nearest"
///
/// # Returns
///
/// An array containing the quantile values
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::quantile;
///
/// let data = array![1.0, 3.0, 5.0, 7.0, 9.0];
///
/// // Median (50th percentile)
/// let median = quantile(data.view(), array![0.5].view(), Some("linear")).unwrap();
/// assert_eq!(median[0], 5.0);
///
/// // Multiple quantiles
/// let quartiles = quantile(data.view(), array![0.25, 0.5, 0.75].view(), None).unwrap();
/// assert_eq!(quartiles[0], 3.0);  // 25th percentile
/// assert_eq!(quartiles[1], 5.0);  // 50th percentile
/// assert_eq!(quartiles[2], 7.0);  // 75th percentile
/// ```
///
/// This function is equivalent to NumPy's `np.quantile` function.
pub fn quantile<T>(
    array: ArrayView<T, Ix1>,
    q: ArrayView<T, Ix1>,
    method: Option<&str>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    if array.is_empty() {
        return Err("Cannot compute quantile of an empty array");
    }

    // Validate q values
    for &val in q.iter() {
        if val < T::from_f64(0.0).unwrap() || val > T::from_f64(1.0).unwrap() {
            return Err("Quantile values must be between 0 and 1");
        }
    }

    // Clone and sort the array
    let mut sorted: Vec<T> = array.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let mut result = Array::<T, Ix1>::zeros(q.len());

    // The interpolation method to use
    let method = method.unwrap_or("linear");

    for (i, &q_val) in q.iter().enumerate() {
        if q_val == T::from_f64(0.0).unwrap() {
            result[i] = sorted[0];
            continue;
        }

        if q_val == T::from_f64(1.0).unwrap() {
            result[i] = sorted[n - 1];
            continue;
        }

        // Calculate the position in the sorted array
        let h = T::from_usize(n - 1).unwrap() * q_val;
        let h_floor = h.floor();
        let idx_low = h_floor.to_usize().unwrap_or(0).min(n - 1);
        let idx_high = (idx_low + 1).min(n - 1);

        match method {
            "linear" => {
                let weight = h - h_floor;
                result[i] = sorted[idx_low] * (T::from_f64(1.0).unwrap() - weight) + sorted[idx_high] * weight;
            }
            "lower" => {
                result[i] = sorted[idx_low];
            }
            "higher" => {
                result[i] = sorted[idx_high];
            }
            "midpoint" => {
                result[i] = (sorted[idx_low] + sorted[idx_high]) / T::from_f64(2.0).unwrap();
            }
            "nearest" => {
                let weight = h - h_floor;
                if weight < T::from_f64(0.5).unwrap() {
                    result[i] = sorted[idx_low];
                } else {
                    result[i] = sorted[idx_high];
                }
            }
            _ => return Err("Invalid interpolation method. Use 'linear', 'lower', 'higher', 'midpoint', or 'nearest'"),
        }
    }

    Ok(result)
}

/// Count number of occurrences of each value in array of non-negative ints.
///
/// # Arguments
///
/// * `array` - Input array of non-negative integers
/// * `minlength` - Minimum number of bins for the output array. If None, the output array length is determined by the maximum value in `array`.
/// * `weights` - Optional weights array. If specified, must have same shape as `array`.
///
/// # Returns
///
/// An array of counts
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::bincount;
///
/// let data = array![1, 2, 3, 1, 2, 1, 0, 1, 3, 2];
/// let counts = bincount(data.view(), None, None).unwrap();
/// assert_eq!(counts.len(), 4);
/// assert_eq!(counts[0], 1.0); // '0' occurs once
/// assert_eq!(counts[1], 4.0); // '1' occurs four times
/// assert_eq!(counts[2], 3.0); // '2' occurs three times
/// assert_eq!(counts[3], 2.0); // '3' occurs twice
/// ```
///
/// This function is equivalent to NumPy's `np.bincount` function.
pub fn bincount(
    array: ArrayView<usize, Ix1>,
    minlength: Option<usize>,
    weights: Option<ArrayView<f64, Ix1>>,
) -> Result<Array<f64, Ix1>, &'static str> {
    if array.is_empty() {
        return Err("Cannot compute bincount of an empty array");
    }

    // Find maximum value to determine number of bins
    let mut max_val = 0;
    for &val in array.iter() {
        if val > max_val {
            max_val = val;
        }
    }

    // Determine length of output array
    let length = if let Some(min_len) = minlength {
        max_val.max(min_len - 1) + 1
    } else {
        max_val + 1
    };

    let mut result = Array::<f64, Ix1>::zeros(length);

    match weights {
        Some(w) => {
            if w.len() != array.len() {
                return Err("Weights array must have same length as input array");
            }
            for (&idx, &weight) in array.iter().zip(w.iter()) {
                result[idx] += weight;
            }
        }
        None => {
            for &idx in array.iter() {
                result[idx] += 1.0;
            }
        }
    }

    Ok(result)
}

/// Return the indices of the bins to which each value in input array belongs.
///
/// # Arguments
///
/// * `array` - Input array
/// * `bins` - Array of bin edges
/// * `right` - Indicates whether the intervals include the right or left bin edge
/// * `result_type` - Whether to return the indices ('indices') or the bin values ('values')
///
/// # Returns
///
/// Array of indices or values depending on result_type
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::digitize;
///
/// let data = array![1.2, 3.5, 5.1, 0.8, 2.9];
/// let bins = array![1.0, 3.0, 5.0];
/// let indices = digitize(data.view(), bins.view(), false, "indices").unwrap();
///
/// assert_eq!(indices[0], 1); // 1.2 is in the first bin (1.0 <= x < 3.0)
/// assert_eq!(indices[1], 2); // 3.5 is in the second bin (3.0 <= x < 5.0)
/// assert_eq!(indices[2], 3); // 5.1 is after the last bin (>= 5.0)
/// assert_eq!(indices[3], 0); // 0.8 is before the first bin (< 1.0)
/// assert_eq!(indices[4], 1); // 2.9 is in the first bin (1.0 <= x < 3.0)
/// ```
///
/// This function is equivalent to NumPy's `np.digitize` function.
pub fn digitize<T>(
    array: ArrayView<T, Ix1>,
    bins: ArrayView<T, Ix1>,
    right: bool,
    result_type: &str,
) -> Result<Array<usize, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    if array.is_empty() {
        return Err("Cannot digitize an empty array");
    }

    if bins.is_empty() {
        return Err("Bins array cannot be empty");
    }

    // Check that bins are monotonically increasing
    for i in 1..bins.len() {
        if bins[i] <= bins[i - 1] {
            return Err("Bins must be monotonically increasing");
        }
    }

    let mut result = Array::<usize, Ix1>::zeros(array.len());

    for (i, &val) in array.iter().enumerate() {
        let mut bin_idx = 0;

        if right {
            // Right inclusive: val <= edge
            for j in 0..bins.len() {
                if val <= bins[j] {
                    bin_idx = j;
                    break;
                }
                bin_idx = bins.len();
            }
        } else {
            // Left inclusive: val < edge
            for j in 0..bins.len() {
                if val < bins[j] {
                    bin_idx = j;
                    break;
                }
                bin_idx = bins.len();
            }
        }

        result[i] = bin_idx;
    }

    if result_type == "indices" {
        Ok(result)
    } else if result_type == "values" {
        Err("'values' result_type is not yet implemented")
    } else {
        Err("result_type must be 'indices' or 'values'")
    }
}
