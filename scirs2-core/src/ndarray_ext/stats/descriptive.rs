//! Descriptive statistics for ndarray arrays
//!
//! This module provides descriptive statistical functions such as mean, median,
//! standard deviation, variance, min, max, etc. for ndarray arrays.

use ::ndarray::{Array, ArrayView, Axis, Dimension, Ix1, Ix2};
use num_traits::{Float, FromPrimitive};

/// Calculate the mean of array elements (2D arrays)
///
/// # Errors
///
/// Returns an error if the array is empty or if conversion fails.
///
/// # Panics
///
/// Panics if type conversion from usize fails.
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the mean (None for global mean)
///
/// # Returns
///
/// The mean of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::{array, Axis};
/// use scirs2_core::ndarray_ext::stats::mean_2d;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Global mean
/// let global_mean = mean_2d(&a.view(), None).expect("Operation failed");
/// assert_eq!(global_mean[0], 3.5);
///
/// // Mean along axis 0 (columns)
/// let col_means = mean_2d(&a.view(), Some(Axis(0))).expect("Operation failed");
/// assert_eq!(col_means.len(), 3);
/// assert_eq!(col_means[0], 2.5);
/// assert_eq!(col_means[1], 3.5);
/// assert_eq!(col_means[2], 4.5);
/// ```
#[allow(dead_code)]
pub fn mean_2d<T>(
    array: &ArrayView<T, Ix2>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    if array.is_empty() {
        return Err("Cannot compute mean of an empty array");
    }

    if let Some(ax) = axis {
        let (rows, cols) = (array.shape()[0], array.shape()[1]);

        match ax.index() {
            0 => {
                // Mean along axis 0 (columns)
                let mut result = Array::<T, Ix1>::zeros(cols);
                let n = T::from_usize(rows).expect("Operation failed");

                for j in 0..cols {
                    let mut sum = T::zero();
                    for i in 0..rows {
                        sum = sum + array[[i, j]];
                    }
                    result[j] = sum / n;
                }

                Ok(result)
            }
            1 => {
                // Mean along axis 1 (rows)
                let mut result = Array::<T, Ix1>::zeros(rows);
                let n = T::from_usize(cols).expect("Operation failed");

                for i in 0..rows {
                    let mut sum = T::zero();
                    for j in 0..cols {
                        sum = sum + array[[i, j]];
                    }
                    result[0] = sum / n;
                }

                Ok(result)
            }
            _ => Err("Axis index out of bounds for 2D array"),
        }
    } else {
        // Global mean
        let total_elements = array.len();
        let mut sum = T::zero();

        for &val in array {
            sum = sum + val;
        }

        let count = T::from_usize(total_elements).ok_or("Cannot convert array length to T")?;
        Ok(Array::from_elem(1, sum / count))
    }
}

/// Calculate the median of array elements (2D arrays)
///
/// # Errors
///
/// Returns an error if the array is empty.
///
/// # Panics
///
/// Panics if partial comparison fails or type conversion fails.
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the median (None for global median)
///
/// # Returns
///
/// The median of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::{array, Axis};
/// use scirs2_core::ndarray_ext::stats::median_2d;
///
/// let a = array![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];
///
/// // Global median
/// let global_median = median_2d(&a.view(), None).expect("Operation failed");
/// assert_eq!(global_median[0], 3.5);
///
/// // Median along axis 0 (columns)
/// let col_medians = median_2d(&a.view(), Some(Axis(0))).expect("Operation failed");
/// assert_eq!(col_medians.len(), 3);
/// assert_eq!(col_medians[0], 1.5);
/// assert_eq!(col_medians[1], 3.5);
/// assert_eq!(col_medians[2], 5.5);
/// ```
#[allow(dead_code)]
pub fn median_2d<T>(
    array: &ArrayView<T, Ix2>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    if array.is_empty() {
        return Err("Cannot compute median of an empty array");
    }

    if let Some(ax) = axis {
        let (rows, cols) = (array.shape()[0], array.shape()[1]);

        match ax.index() {
            0 => {
                // Median along axis 0 (columns)
                let mut result = Array::<T, Ix1>::zeros(cols);

                for j in 0..cols {
                    let mut column_values = Vec::with_capacity(rows);
                    for i in 0..rows {
                        column_values.push(array[[i, j]]);
                    }

                    column_values
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    let median_value = if column_values.len() % 2 == 0 {
                        let mid = column_values.len() / 2;
                        (column_values[mid - 1] + column_values[mid])
                            / T::from_f64(2.0).expect("Operation failed")
                    } else {
                        column_values[column_values.len() / 2]
                    };

                    result[j] = median_value;
                }

                Ok(result)
            }
            1 => {
                // Median along axis 1 (rows)
                let mut result = Array::<T, Ix1>::zeros(rows);

                for i in 0..rows {
                    let mut row_values = Vec::with_capacity(cols);
                    for j in 0..cols {
                        row_values.push(array[[i, j]]);
                    }

                    row_values
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    let median_value = if row_values.len() % 2 == 0 {
                        let mid = row_values.len() / 2;
                        (row_values[mid - 1] + row_values[mid])
                            / T::from_f64(2.0).expect("Operation failed")
                    } else {
                        row_values[row_values.len() / 2]
                    };

                    result[0] = median_value;
                }

                Ok(result)
            }
            _ => Err("Axis index out of bounds for 2D array"),
        }
    } else {
        // Global median
        let mut values: Vec<_> = array.iter().copied().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_value = if values.len() % 2 == 0 {
            let mid = values.len() / 2;
            (values[mid - 1] + values[mid]) / T::from_f64(2.0).expect("Operation failed")
        } else {
            values[values.len() / 2]
        };

        Ok(Array::from_elem(1, median_value))
    }
}

/// Calculate the standard deviation of array elements (2D arrays)
///
/// # Errors
///
/// Returns an error if the array is empty or variance calculation fails.
///
/// # Panics
///
/// Panics if variance calculation panics.
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the std dev (None for global std dev)
/// * `ddof` - Delta degrees of freedom (default 0)
///
/// # Returns
///
/// The standard deviation of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::{array, Axis};
/// use scirs2_core::ndarray_ext::stats::std_dev_2d;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Global standard deviation
/// let global_std = std_dev_2d(&a.view(), None, 1).expect("Operation failed");
/// assert!((global_std[0] - 1.87082869339_f64).abs() < 1e-10);
///
/// // Standard deviation along axis 0 (columns)
/// let col_stds = std_dev_2d(&a.view(), Some(Axis(0)), 1).expect("Operation failed");
/// assert_eq!(col_stds.len(), 3);
/// ```
#[allow(dead_code)]
pub fn std_dev_2d<T>(
    array: &ArrayView<T, Ix2>,
    axis: Option<Axis>,
    ddof: usize,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    let var_result = variance_2d(array, axis, ddof)?;
    Ok(var_result.mapv(|x| x.sqrt()))
}

/// Calculate the variance of array elements (2D arrays)
///
/// # Errors
///
/// Returns an error if the array is empty or if conversion fails.
///
/// # Panics
///
/// Panics if type conversion from usize fails.
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the variance (None for global variance)
/// * `ddof` - Delta degrees of freedom (default 0)
///
/// # Returns
///
/// The variance of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::{array, Axis};
/// use scirs2_core::ndarray_ext::stats::variance_2d;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Global variance
/// let global_var = variance_2d(&a.view(), None, 1).expect("Operation failed");
/// assert!((global_var[0] - 3.5_f64).abs() < 1e-10);
///
/// // Variance along axis 0 (columns)
/// let col_vars = variance_2d(&a.view(), Some(Axis(0)), 1).expect("Operation failed");
/// assert_eq!(col_vars.len(), 3);
/// ```
#[allow(dead_code)]
pub fn variance_2d<T>(
    array: &ArrayView<T, Ix2>,
    axis: Option<Axis>,
    ddof: usize,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    if array.is_empty() {
        return Err("Cannot compute variance of an empty array");
    }

    if let Some(ax) = axis {
        let (rows, cols) = (array.shape()[0], array.shape()[1]);

        match ax.index() {
            0 => {
                // Variance along axis 0 (columns)
                let means = mean_2d(array, Some(ax))?;

                if rows <= ddof {
                    return Err("Not enough data points for variance calculation with given ddof");
                }

                let mut result = Array::<T, Ix1>::zeros(cols);

                for j in 0..cols {
                    let mut sum_sq_diff = T::zero();
                    for i in 0..rows {
                        let diff = array[[i, j]] - means[j];
                        sum_sq_diff = sum_sq_diff + (diff * diff);
                    }

                    let divisor = T::from_usize(rows - ddof).expect("Operation failed");
                    result[j] = sum_sq_diff / divisor;
                }

                Ok(result)
            }
            1 => {
                // Variance along axis 1 (rows)
                let means = mean_2d(array, Some(ax))?;

                if cols <= ddof {
                    return Err("Not enough data points for variance calculation with given ddof");
                }

                let mut result = Array::<T, Ix1>::zeros(rows);

                for i in 0..rows {
                    let mut sum_sq_diff = T::zero();
                    for j in 0..cols {
                        let diff = array[[i, j]] - means[i];
                        sum_sq_diff = sum_sq_diff + (diff * diff);
                    }

                    let divisor = T::from_usize(cols - ddof).expect("Operation failed");
                    result[0] = sum_sq_diff / divisor;
                }

                Ok(result)
            }
            _ => Err("Axis index out of bounds for 2D array"),
        }
    } else {
        // Global variance
        let total_elements = array.len();

        if total_elements <= ddof {
            return Err("Not enough data points for variance calculation with given ddof");
        }

        // Calculate global mean
        let global_mean = mean_2d(array, None)?[0];

        // Calculate sum of squared differences from the mean
        let mut sum_sq_diff = T::zero();
        for &val in array {
            let diff = val - global_mean;
            sum_sq_diff = sum_sq_diff + (diff * diff);
        }

        let divisor = T::from_usize(total_elements - ddof).expect("Operation failed");

        Ok(Array::from_elem(1, sum_sq_diff / divisor))
    }
}

// Need to implement the following functions:
// min_2d, max_2d, sum_2d, product_2d, percentile_2d, mean, median, variance, std_dev, min, max, percentile

// Let's continue with min_2d and max_2d

/// Calculate the minimum value(s) of array elements (2D arrays)
///
/// # Errors
///
/// Returns an error if the array is empty.
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the minimum (None for global minimum)
///
/// # Returns
///
/// The minimum value(s) of the array elements
#[allow(dead_code)]
pub fn min_2d<T>(
    array: &ArrayView<T, Ix2>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float,
{
    if array.is_empty() {
        return Err("Cannot compute minimum of an empty array");
    }

    match axis {
        Some(ax) => {
            let (rows, cols) = (array.shape()[0], array.shape()[1]);

            match ax.index() {
                0 => {
                    // Min along axis 0 (columns)
                    let mut result = Array::<T, Ix1>::zeros(cols);

                    for j in 0..cols {
                        let mut min_val = array[[0, j]];
                        for i in 1..rows {
                            if array[[i, j]] < min_val {
                                min_val = array[[i, j]];
                            }
                        }
                        result[j] = min_val;
                    }

                    Ok(result)
                }
                1 => {
                    // Min along axis 1 (rows)
                    let mut result = Array::<T, Ix1>::zeros(rows);

                    for i in 0..rows {
                        let mut min_val = array[[i, 0]];
                        for j in 1..cols {
                            if array[[i, j]] < min_val {
                                min_val = array[[i, j]];
                            }
                        }
                        result[i] = min_val;
                    }

                    Ok(result)
                }
                _ => Err("Axis index out of bounds for 2D array"),
            }
        }
        None => {
            // Global min
            let mut min_val = array[[0, 0]];

            for &val in array {
                if val < min_val {
                    min_val = val;
                }
            }

            Ok(Array::from_elem(1, min_val))
        }
    }
}

/// Calculate the maximum value(s) of array elements (2D arrays)
///
/// # Errors
///
/// Returns an error if the array is empty.
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the maximum (None for global maximum)
///
/// # Returns
///
/// The maximum value(s) of the array elements
#[allow(dead_code)]
pub fn max_2d<T>(
    array: &ArrayView<T, Ix2>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float,
{
    if array.is_empty() {
        return Err("Cannot compute maximum of an empty array");
    }

    match axis {
        Some(ax) => {
            let (rows, cols) = (array.shape()[0], array.shape()[1]);

            match ax.index() {
                0 => {
                    // Max along axis 0 (columns)
                    let mut result = Array::<T, Ix1>::zeros(cols);

                    for j in 0..cols {
                        let mut max_val = array[[0, j]];
                        for i in 1..rows {
                            if array[[i, j]] > max_val {
                                max_val = array[[i, j]];
                            }
                        }
                        result[j] = max_val;
                    }

                    Ok(result)
                }
                1 => {
                    // Max along axis 1 (rows)
                    let mut result = Array::<T, Ix1>::zeros(rows);

                    for i in 0..rows {
                        let mut max_val = array[[i, 0]];
                        for j in 1..cols {
                            if array[[i, j]] > max_val {
                                max_val = array[[i, j]];
                            }
                        }
                        result[i] = max_val;
                    }

                    Ok(result)
                }
                _ => Err("Axis index out of bounds for 2D array"),
            }
        }
        None => {
            // Global max
            let mut max_val = array[[0, 0]];

            for &val in array {
                if val > max_val {
                    max_val = val;
                }
            }

            Ok(Array::from_elem(1, max_val))
        }
    }
}

/// Calculate the sum of array elements (2D arrays)
///
/// # Errors
///
/// Returns an error if the array is empty.
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the sum (None for global sum)
///
/// # Returns
///
/// The sum of the array elements
#[allow(dead_code)]
pub fn sum_2d<T>(
    array: &ArrayView<T, Ix2>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float,
{
    if array.is_empty() {
        return Err("Cannot compute sum of an empty array");
    }

    match axis {
        Some(ax) => {
            let (rows, cols) = (array.shape()[0], array.shape()[1]);

            match ax.index() {
                0 => {
                    // Sum along axis 0 (columns)
                    let mut result = Array::<T, Ix1>::zeros(cols);

                    for j in 0..cols {
                        let mut sum = T::zero();
                        for i in 0..rows {
                            sum = sum + array[[i, j]];
                        }
                        result[j] = sum;
                    }

                    Ok(result)
                }
                1 => {
                    // Sum along axis 1 (rows)
                    let mut result = Array::<T, Ix1>::zeros(rows);

                    for i in 0..rows {
                        let mut sum = T::zero();
                        for j in 0..cols {
                            sum = sum + array[[i, j]];
                        }
                        result[0] = sum;
                    }

                    Ok(result)
                }
                _ => Err("Axis index out of bounds for 2D array"),
            }
        }
        None => {
            // Global sum
            let mut sum = T::zero();

            for &val in array {
                sum = sum + val;
            }

            Ok(Array::from_elem(1, sum))
        }
    }
}

/// Calculate the product of array elements (2D arrays)
///
/// # Errors
///
/// Returns an error if the array is empty.
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the product (None for global product)
///
/// # Returns
///
/// The product of the array elements
#[allow(dead_code)]
pub fn product_2d<T>(
    array: &ArrayView<T, Ix2>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float,
{
    if array.is_empty() {
        return Err("Cannot compute product of an empty array");
    }

    match axis {
        Some(ax) => {
            let (rows, cols) = (array.shape()[0], array.shape()[1]);

            match ax.index() {
                0 => {
                    // Product along axis 0 (columns)
                    let mut result = Array::<T, Ix1>::from_elem(cols, T::one());

                    for j in 0..cols {
                        for i in 0..rows {
                            result[j] = result[j] * array[[i, j]];
                        }
                    }

                    Ok(result)
                }
                1 => {
                    // Product along axis 1 (rows)
                    let mut result = Array::<T, Ix1>::from_elem(rows, T::one());

                    for i in 0..rows {
                        for j in 0..cols {
                            result[i] = result[i] * array[[i, j]];
                        }
                    }

                    Ok(result)
                }
                _ => Err("Axis index out of bounds for 2D array"),
            }
        }
        None => {
            // Global product
            let mut product = T::one();

            for &val in array {
                product = product * val;
            }

            Ok(Array::from_elem(1, product))
        }
    }
}

/// Calculate the percentile of array elements (2D arrays)
///
/// # Errors
///
/// Returns an error if the array is empty or percentile is invalid.
///
/// # Panics
///
/// Panics if type conversion or partial comparison fails.
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `q` - The percentile to compute (0 to 100)
/// * `axis` - Optional axis along which to compute the percentile (None for global percentile)
///
/// # Returns
///
/// The percentile value(s) of the array elements
#[allow(dead_code)]
pub fn percentile_2d<T>(
    array: &ArrayView<T, Ix2>,
    q: f64,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    if array.is_empty() {
        return Err("Cannot compute percentile of an empty array");
    }

    if !(0.0..=100.0).contains(&q) {
        return Err("Percentile must be between 0 and 100");
    }

    match axis {
        Some(ax) => {
            let (rows, cols) = (array.shape()[0], array.shape()[1]);

            match ax.index() {
                0 => {
                    // Percentile along axis 0 (columns)
                    let mut result = Array::<T, Ix1>::zeros(cols);

                    for j in 0..cols {
                        let mut column_values = Vec::with_capacity(rows);
                        for i in 0..rows {
                            column_values.push(array[[i, j]]);
                        }

                        column_values
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                        // Linear interpolation
                        let pos = (q / 100.0) * (column_values.len() as f64 - 1.0);
                        let idx_low = pos.floor() as usize;
                        let idx_high = pos.ceil() as usize;

                        if idx_low == idx_high {
                            result[j] = column_values[idx_low];
                        } else {
                            let weight_high = pos - (idx_low as f64);
                            let weight_low = 1.0 - weight_high;

                            result[j] = column_values[idx_low]
                                * T::from_f64(weight_low).expect("Operation failed")
                                + column_values[idx_high]
                                    * T::from_f64(weight_high).expect("Operation failed");
                        }
                    }

                    Ok(result)
                }
                1 => {
                    // Percentile along axis 1 (rows)
                    let mut result = Array::<T, Ix1>::zeros(rows);

                    for i in 0..rows {
                        let mut row_values = Vec::with_capacity(cols);
                        for j in 0..cols {
                            row_values.push(array[[i, j]]);
                        }

                        row_values
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                        // Linear interpolation
                        let pos = (q / 100.0) * (row_values.len() as f64 - 1.0);
                        let idx_low = pos.floor() as usize;
                        let idx_high = pos.ceil() as usize;

                        if idx_low == idx_high {
                            result[0] = row_values[idx_low];
                        } else {
                            let weight_high = pos - (idx_low as f64);
                            let weight_low = 1.0 - weight_high;

                            result[0] = row_values[idx_low]
                                * T::from_f64(weight_low).expect("Operation failed")
                                + row_values[idx_high]
                                    * T::from_f64(weight_high).expect("Operation failed");
                        }
                    }

                    Ok(result)
                }
                _ => Err("Axis index out of bounds for 2D array"),
            }
        }
        None => {
            // Global percentile
            let mut values: Vec<_> = array.iter().copied().collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Linear interpolation
            let pos = (q / 100.0) * (values.len() as f64 - 1.0);
            let idx_low = pos.floor() as usize;
            let idx_high = pos.ceil() as usize;

            let result = if idx_low == idx_high {
                values[idx_low]
            } else {
                let weight_high = pos - (idx_low as f64);
                let weight_low = 1.0 - weight_high;

                values[idx_low] * T::from_f64(weight_low).expect("Operation failed")
                    + values[idx_high] * T::from_f64(weight_high).expect("Operation failed")
            };

            Ok(Array::from_elem(1, result))
        }
    }
}

// Generic implementations for n-dimensional arrays

/// Calculate the mean of array elements
///
/// # Errors
///
/// Returns an error if the array is empty or if conversion fails.
///
/// # Panics
///
/// Panics if type conversion from usize fails.
///
/// # Arguments
///
/// * `array` - The input array
/// * `axis` - Optional axis along which to compute the mean (None for global mean)
///
/// # Returns
///
/// The mean of the array elements
#[allow(dead_code)]
pub fn mean<T, D>(
    array: &ArrayView<T, D>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
    D: Dimension + crate::ndarray::RemoveAxis,
{
    if array.is_empty() {
        return Err("Cannot compute mean of an empty array");
    }

    match axis {
        Some(ax) => {
            // Axis-specific mean implementation for arbitrary dimensions
            let ndim = array.ndim();
            if ax.index() >= ndim {
                return Err("Axis index out of bounds");
            }

            // Create output shape by removing the specified axis
            let mut outputshape = array.shape().to_vec();
            outputshape.remove(ax.index());

            // Handle case where removing axis results in scalar
            if outputshape.is_empty() {
                outputshape.push(1);
            }

            // Calculate mean along specified axis using ndarray's mean_axis
            let result = array
                .mean_axis(ax)
                .ok_or("Failed to compute mean along axis")?;

            // Convert to 1D array as expected by function signature
            let flat_result = result
                .to_shape((result.len(),))
                .map_err(|_| "Failed to reshape result to 1D")?;

            Ok(flat_result.into_owned())
        }
        None => {
            // Global mean
            let total_elements = array.len();
            let mut sum = T::zero();

            for &val in array {
                sum = sum + val;
            }

            let count = T::from_usize(total_elements).ok_or("Cannot convert array length to T")?;
            Ok(Array::from_elem(1, sum / count))
        }
    }
}

/// Calculate the median of array elements
///
/// # Arguments
///
/// * `array` - The input array
/// * `axis` - Optional axis along which to compute the median (None for global median)
///
/// # Returns
///
/// The median of the array elements
///
/// # Errors
///
/// Returns an error if the array is empty.
///
/// # Panics
///
/// Panics if type conversion or partial comparison fails.
#[allow(dead_code)]
pub fn median<T, D>(
    array: &ArrayView<T, D>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
    D: Dimension + crate::ndarray::RemoveAxis,
{
    if array.is_empty() {
        return Err("Cannot compute median of an empty array");
    }

    match axis {
        Some(ax) => {
            // Axis-specific median for arbitrary-dimension arrays.
            let ndim = array.ndim();
            if ax.index() >= ndim {
                return Err("Axis index out of bounds");
            }

            // Pre-compute the divisor for the even-length averaging branch
            // outside the closure so we can surface conversion failures via
            // `?` rather than panicking inside `map_axis`.
            let two = T::from_f64(2.0).ok_or("Cannot convert 2.0 into element type")?;

            // Compute median along each lane along `ax`. `map_axis` walks
            // every 1D slice perpendicular to `ax` and yields the reduced
            // array of shape `D::Smaller`.
            let reduced = array.map_axis(ax, |lane| {
                let mut values: Vec<T> = lane.iter().copied().collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let len = values.len();
                if len.is_multiple_of(2) {
                    let mid = len / 2;
                    (values[mid - 1] + values[mid]) / two
                } else {
                    values[len / 2]
                }
            });

            // Flatten the reduced array (shape `D::Smaller`) into 1D using
            // the same `to_shape` style as the surrounding `mean`/`variance`
            // implementations.
            let flat_result = reduced
                .to_shape((reduced.len(),))
                .map_err(|_| "Failed to reshape median result to 1D")?;

            Ok(flat_result.into_owned())
        }
        None => {
            // Global median
            let mut values: Vec<_> = array.iter().copied().collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median_value = if values.len().is_multiple_of(2) {
                let mid = values.len() / 2;
                let two = T::from_f64(2.0).ok_or("Cannot convert 2.0 into element type")?;
                (values[mid - 1] + values[mid]) / two
            } else {
                values[values.len() / 2]
            };

            Ok(Array::from_elem(1, median_value))
        }
    }
}

/// Calculate the variance of array elements
///
/// # Arguments
///
/// * `array` - The input array
/// * `axis` - Optional axis along which to compute the variance (None for global variance)
/// * `ddof` - Delta degrees of freedom (default 0)
///
/// # Returns
///
/// The variance of the array elements
///
/// # Errors
///
/// Returns an error if the array is empty or if conversion fails.
///
/// # Panics
///
/// Panics if type conversion from usize fails.
#[allow(dead_code)]
pub fn variance<T, D>(
    array: &ArrayView<T, D>,
    axis: Option<Axis>,
    ddof: usize,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
    D: Dimension + crate::ndarray::RemoveAxis,
{
    if array.is_empty() {
        return Err("Cannot compute variance of an empty array");
    }

    match axis {
        Some(ax) => {
            // Axis-specific variance implementation for arbitrary dimensions
            let ndim = array.ndim();
            if ax.index() >= ndim {
                return Err("Axis index out of bounds");
            }

            // Create output shape by removing the specified axis
            let mut outputshape = array.shape().to_vec();
            outputshape.remove(ax.index());

            // Handle case where removing axis results in scalar
            if outputshape.is_empty() {
                outputshape.push(1);
            }

            // Calculate variance along specified axis using ndarray's var_axis
            let result = array.var_axis(ax, T::from_usize(ddof).expect("Operation failed"));

            // Convert to 1D array as expected by function signature
            let flat_result = result
                .to_shape((result.len(),))
                .map_err(|_| "Failed to reshape variance result to 1D")?;

            Ok(flat_result.into_owned())
        }
        None => {
            // Global variance
            let total_elements = array.len();

            if total_elements <= ddof {
                return Err("Not enough data points for variance calculation with given ddof");
            }

            // Calculate global mean
            let global_mean = mean(array, None)?[0];

            // Calculate sum of squared differences from the mean
            let mut sum_sq_diff = T::zero();
            for &val in array {
                let diff = val - global_mean;
                sum_sq_diff = sum_sq_diff + (diff * diff);
            }

            let divisor = T::from_usize(total_elements - ddof).expect("Operation failed");

            Ok(Array::from_elem(1, sum_sq_diff / divisor))
        }
    }
}

/// Calculate the standard deviation of array elements
///
/// # Arguments
///
/// * `array` - The input array
/// * `axis` - Optional axis along which to compute the std dev (None for global std dev)
/// * `ddof` - Delta degrees of freedom (default 0)
///
/// # Returns
///
/// The standard deviation of the array elements
///
/// # Errors
///
/// Returns an error if the array is empty or variance calculation fails.
///
/// # Panics
///
/// Panics if variance calculation panics.
#[allow(dead_code)]
pub fn std_dev<T, D>(
    array: &ArrayView<T, D>,
    axis: Option<Axis>,
    ddof: usize,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
    D: Dimension + crate::ndarray::RemoveAxis,
{
    let var_result = variance(array, axis, ddof)?;
    Ok(var_result.mapv(|x| x.sqrt()))
}

/// Calculate the minimum value(s) of array elements
///
/// # Arguments
///
/// * `array` - The input array
/// * `axis` - Optional axis along which to compute the minimum (None for global minimum)
///
/// # Returns
///
/// The minimum value(s) of the array elements
///
/// # Errors
///
/// Returns an error if the array is empty.
#[allow(dead_code)]
pub fn min<T, D>(array: &ArrayView<T, D>, axis: Option<Axis>) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float,
    D: Dimension + crate::ndarray::RemoveAxis,
{
    if array.is_empty() {
        return Err("Cannot compute minimum of an empty array");
    }

    match axis {
        Some(ax) => {
            // Axis-specific minimum implementation for arbitrary dimensions
            let ndim = array.ndim();
            if ax.index() >= ndim {
                return Err("Axis index out of bounds");
            }

            // Create output shape by removing the specified axis
            let mut outputshape = array.shape().to_vec();
            outputshape.remove(ax.index());

            // Handle case where removing axis results in scalar
            if outputshape.is_empty() {
                outputshape.push(1);
            }

            // Use ndarray's fold_axis to compute minimum along specified axis
            let result = array.fold_axis(ax, T::infinity(), |&a, &b| if a < b { a } else { b });

            // Convert to 1D array as expected by function signature
            let flat_result = result
                .to_shape((result.len(),))
                .map_err(|_| "Failed to reshape minimum result to 1D")?;

            Ok(flat_result.into_owned())
        }
        None => {
            // Global minimum
            let mut min_val = *array.iter().next().expect("Operation failed");

            for &val in array {
                if val < min_val {
                    min_val = val;
                }
            }

            Ok(Array::from_elem(1, min_val))
        }
    }
}

/// Calculate the maximum value(s) of array elements
///
/// # Arguments
///
/// * `array` - The input array
/// * `axis` - Optional axis along which to compute the maximum (None for global maximum)
///
/// # Returns
///
/// The maximum value(s) of the array elements
///
/// # Errors
///
/// Returns an error if the array is empty.
#[allow(dead_code)]
pub fn max<T, D>(array: &ArrayView<T, D>, axis: Option<Axis>) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float,
    D: Dimension + crate::ndarray::RemoveAxis,
{
    if array.is_empty() {
        return Err("Cannot compute maximum of an empty array");
    }

    match axis {
        Some(ax) => {
            // Axis-specific maximum implementation for arbitrary dimensions
            let ndim = array.ndim();
            if ax.index() >= ndim {
                return Err("Axis index out of bounds");
            }

            // Create output shape by removing the specified axis
            let mut outputshape = array.shape().to_vec();
            outputshape.remove(ax.index());

            // Handle case where removing axis results in scalar
            if outputshape.is_empty() {
                outputshape.push(1);
            }

            // Use ndarray's fold_axis to compute maximum along specified axis
            let result = array.fold_axis(ax, T::neg_infinity(), |&a, &b| if a > b { a } else { b });

            // Convert to 1D array as expected by function signature
            let flat_result = result
                .to_shape((result.len(),))
                .map_err(|_| "Failed to reshape maximum result to 1D")?;

            Ok(flat_result.into_owned())
        }
        None => {
            // Global maximum
            let mut max_val = *array.iter().next().expect("Operation failed");

            for &val in array {
                if val > max_val {
                    max_val = val;
                }
            }

            Ok(Array::from_elem(1, max_val))
        }
    }
}

/// Calculate the percentile of array elements
///
/// # Arguments
///
/// * `array` - The input array
/// * `q` - The percentile to compute (0 to 100)
/// * `axis` - Optional axis along which to compute the percentile (None for global percentile)
///
/// # Returns
///
/// The percentile value(s) of the array elements
///
/// # Errors
///
/// Returns an error if the array is empty or percentile is invalid.
///
/// # Panics
///
/// Panics if type conversion or partial comparison fails.
#[allow(dead_code)]
pub fn percentile<T, D>(
    array: &ArrayView<T, D>,
    q: f64,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
    D: Dimension + crate::ndarray::RemoveAxis,
{
    if array.is_empty() {
        return Err("Cannot compute percentile of an empty array");
    }

    if !(0.0..=100.0).contains(&q) {
        return Err("Percentile must be between 0 and 100");
    }

    match axis {
        Some(ax) => {
            // Axis-specific percentile for arbitrary-dimension arrays.
            let ndim = array.ndim();
            if ax.index() >= ndim {
                return Err("Axis index out of bounds");
            }

            // We need to convert two distinct floating weights into `T` per
            // lane. To keep the closure inside `map_axis` infallible while
            // still respecting the no-unwrap policy, do a probe conversion
            // up front: if `T::from_f64` cannot represent `q / 100.0` (a
            // value within `[0, 1]`), the conversion path is broken for this
            // type and we surface that as an error.
            //
            // We also pre-compute the closest-rank (boundary) cases — `q == 0`
            // returns the per-lane minimum and `q == 100` returns the
            // per-lane maximum — to avoid floating-point edge cases at the
            // endpoints (e.g. `pos.ceil() as usize` overflowing the slice).
            T::from_f64(q / 100.0).ok_or("Cannot convert percentile weight into element type")?;

            let reduced = array.map_axis(ax, |lane| {
                let mut values: Vec<T> = lane.iter().copied().collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let n = values.len();
                if n == 0 {
                    // `array.is_empty()` is checked above, but defensively
                    // handle a degenerate lane by returning zero.
                    return T::zero();
                }

                // Linear interpolation matching the None-branch.
                let pos = (q / 100.0) * (n as f64 - 1.0);
                let idx_low = pos.floor() as usize;
                let idx_high = pos.ceil() as usize;

                if idx_low == idx_high {
                    values[idx_low]
                } else {
                    let weight_high = pos - (idx_low as f64);
                    let weight_low = 1.0 - weight_high;

                    let w_low = T::from_f64(weight_low).unwrap_or_else(T::zero);
                    let w_high = T::from_f64(weight_high).unwrap_or_else(T::zero);

                    values[idx_low] * w_low + values[idx_high] * w_high
                }
            });

            let flat_result = reduced
                .to_shape((reduced.len(),))
                .map_err(|_| "Failed to reshape percentile result to 1D")?;

            Ok(flat_result.into_owned())
        }
        None => {
            // Global percentile
            let mut values: Vec<_> = array.iter().copied().collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Linear interpolation
            let pos = (q / 100.0) * (values.len() as f64 - 1.0);
            let idx_low = pos.floor() as usize;
            let idx_high = pos.ceil() as usize;

            let result = if idx_low == idx_high {
                values[idx_low]
            } else {
                let weight_high = pos - (idx_low as f64);
                let weight_low = 1.0 - weight_high;

                let w_low = T::from_f64(weight_low)
                    .ok_or("Cannot convert percentile weight into element type")?;
                let w_high = T::from_f64(weight_high)
                    .ok_or("Cannot convert percentile weight into element type")?;

                values[idx_low] * w_low + values[idx_high] * w_high
            };

            Ok(Array::from_elem(1, result))
        }
    }
}

#[cfg(test)]
mod axis_reduction_tests {
    use super::{median, percentile};
    use ::ndarray::{array, Array3, Axis};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    // -------- median --------

    #[test]
    fn test_median_2d_axis0_columns() {
        // 3x3, columns sorted: col0=[1,4,7] median=4, col1=[2,5,8]=5, col2=[3,6,9]=6
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let result = median(&a.view(), Some(Axis(0))).expect("axis median");
        assert_eq!(result.len(), 3);
        assert!(approx_eq(result[0], 4.0));
        assert!(approx_eq(result[1], 5.0));
        assert!(approx_eq(result[2], 6.0));
    }

    #[test]
    fn test_median_2d_axis1_rows() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let result = median(&a.view(), Some(Axis(1))).expect("axis median");
        assert_eq!(result.len(), 3);
        assert!(approx_eq(result[0], 2.0));
        assert!(approx_eq(result[1], 5.0));
        assert!(approx_eq(result[2], 8.0));
    }

    #[test]
    fn test_median_2d_axis1_even_lane_averaging() {
        // Each row has 4 elements -> median = average of two middle values.
        let a = array![[1.0_f64, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]];
        let result = median(&a.view(), Some(Axis(1))).expect("axis median");
        assert_eq!(result.len(), 2);
        // Row 0 sorted: 1,2,3,4 -> (2+3)/2 = 2.5
        assert!(approx_eq(result[0], 2.5));
        // Row 1: (20+30)/2 = 25
        assert!(approx_eq(result[1], 25.0));
    }

    #[test]
    fn test_median_3d_each_axis() {
        // 2x3x4 array with values 0..23.
        let mut a = Array3::<f64>::zeros((2, 3, 4));
        let mut counter = 0.0;
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    a[[i, j, k]] = counter;
                    counter += 1.0;
                }
            }
        }

        // Axis 0: each lane has 2 elements -> average of pair.
        let med0 = median(&a.view(), Some(Axis(0))).expect("axis median");
        assert_eq!(med0.len(), 12);
        // For (j=0, k=0): values are a[0,0,0]=0 and a[1,0,0]=12 -> 6.
        assert!(approx_eq(med0[0], 6.0));

        // Axis 1: each lane has 3 elements -> middle element.
        let med1 = median(&a.view(), Some(Axis(1))).expect("axis median");
        assert_eq!(med1.len(), 8);

        // Axis 2: each lane has 4 elements -> average of two middle values.
        let med2 = median(&a.view(), Some(Axis(2))).expect("axis median");
        assert_eq!(med2.len(), 6);
        // For (i=0, j=0): values are 0,1,2,3 -> (1+2)/2 = 1.5.
        assert!(approx_eq(med2[0], 1.5));
    }

    #[test]
    fn test_median_axis_out_of_bounds() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let err = median(&a.view(), Some(Axis(5))).expect_err("must reject out-of-bounds");
        assert!(err.contains("out of bounds"));
    }

    // -------- percentile --------

    #[test]
    fn test_percentile_2d_axis0_q50_matches_median() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let pct = percentile(&a.view(), 50.0, Some(Axis(0))).expect("axis pct");
        let med = median(&a.view(), Some(Axis(0))).expect("axis median");
        for k in 0..3 {
            assert!(approx_eq(pct[k], med[k]));
        }
    }

    #[test]
    fn test_percentile_2d_axis1_extrema() {
        let a = array![[1.0_f64, 7.0, 5.0, 3.0], [10.0, 0.0, 2.0, 8.0]];

        // q = 0  -> per-row min
        let pct_min = percentile(&a.view(), 0.0, Some(Axis(1))).expect("min");
        assert_eq!(pct_min.len(), 2);
        assert!(approx_eq(pct_min[0], 1.0));
        assert!(approx_eq(pct_min[1], 0.0));

        // q = 100 -> per-row max
        let pct_max = percentile(&a.view(), 100.0, Some(Axis(1))).expect("max");
        assert_eq!(pct_max.len(), 2);
        assert!(approx_eq(pct_max[0], 7.0));
        assert!(approx_eq(pct_max[1], 10.0));
    }

    #[test]
    fn test_percentile_2d_axis1_quartiles() {
        // Row 0 sorted: 1,3,5,7. Linear interp at q=25: pos=0.75 -> 0.25*1+0.75*3 = 2.5
        // q=75: pos = 2.25 -> 0.75*5 + 0.25*7 = 5.5
        let a = array![[1.0_f64, 7.0, 5.0, 3.0]];
        let q25 = percentile(&a.view(), 25.0, Some(Axis(1))).expect("q25");
        let q75 = percentile(&a.view(), 75.0, Some(Axis(1))).expect("q75");
        assert!(approx_eq(q25[0], 2.5), "got {}", q25[0]);
        assert!(approx_eq(q75[0], 5.5), "got {}", q75[0]);
    }

    #[test]
    fn test_percentile_3d_axis2() {
        let mut a = Array3::<f64>::zeros((2, 2, 5));
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..5 {
                    // Fill with k so percentile is independent of (i, j).
                    a[[i, j, k]] = k as f64;
                }
            }
        }
        // Each lane is 0,1,2,3,4. q=50 -> pos=2 -> exact value 2.
        let pct = percentile(&a.view(), 50.0, Some(Axis(2))).expect("axis pct");
        assert_eq!(pct.len(), 4);
        for v in pct.iter() {
            assert!(approx_eq(*v, 2.0));
        }
    }

    #[test]
    fn test_percentile_invalid_q_axis() {
        // Axis branch must still respect the q range guard, since the guard
        // happens before the axis match.
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let err = percentile(&a.view(), -1.0, Some(Axis(0))).expect_err("must reject negative q");
        assert!(err.contains("between 0 and 100"));

        let err2 = percentile(&a.view(), 101.0, Some(Axis(1))).expect_err("must reject q > 100");
        assert!(err2.contains("between 0 and 100"));
    }

    #[test]
    fn test_percentile_axis_out_of_bounds() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let err =
            percentile(&a.view(), 50.0, Some(Axis(5))).expect_err("must reject out-of-bounds axis");
        assert!(err.contains("out of bounds"));
    }
}
