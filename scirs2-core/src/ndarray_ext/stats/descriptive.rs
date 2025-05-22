//! Descriptive statistics for ndarray arrays
//!
//! This module provides descriptive statistical functions such as mean, median,
//! standard deviation, variance, min, max, etc. for ndarray arrays.

use ndarray::{Array, ArrayView, Axis, Dimension, Ix1, Ix2};
use num_traits::{Float, FromPrimitive};

/// Calculate the mean of array elements (2D arrays)
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
/// let global_mean = mean_2d(&a.view(), None).unwrap();
/// assert_eq!(global_mean[0], 3.5);
///
/// // Mean along axis 0 (columns)
/// let col_means = mean_2d(&a.view(), Some(Axis(0))).unwrap();
/// assert_eq!(col_means.len(), 3);
/// assert_eq!(col_means[0], 2.5);
/// assert_eq!(col_means[1], 3.5);
/// assert_eq!(col_means[2], 4.5);
/// ```
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

    match axis {
        Some(ax) => {
            let (rows, cols) = (array.shape()[0], array.shape()[1]);

            match ax.index() {
                0 => {
                    // Mean along axis 0 (columns)
                    let mut result = Array::<T, Ix1>::zeros(cols);
                    let n = T::from_usize(rows).unwrap();

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
                    let n = T::from_usize(cols).unwrap();

                    for i in 0..rows {
                        let mut sum = T::zero();
                        for j in 0..cols {
                            sum = sum + array[[i, j]];
                        }
                        result[i] = sum / n;
                    }

                    Ok(result)
                }
                _ => Err("Axis index out of bounds for 2D array"),
            }
        }
        None => {
            // Global mean
            let total_elements = array.len();
            let mut sum = T::zero();

            for &val in array.iter() {
                sum = sum + val;
            }

            let count = T::from_usize(total_elements).ok_or("Cannot convert array length to T")?;
            Ok(Array::from_elem(1, sum / count))
        }
    }
}

/// Calculate the median of array elements (2D arrays)
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
/// let global_median = median_2d(&a.view(), None).unwrap();
/// assert_eq!(global_median[0], 3.5);
///
/// // Median along axis 0 (columns)
/// let col_medians = median_2d(&a.view(), Some(Axis(0))).unwrap();
/// assert_eq!(col_medians.len(), 3);
/// assert_eq!(col_medians[0], 1.5);
/// assert_eq!(col_medians[1], 3.5);
/// assert_eq!(col_medians[2], 5.5);
/// ```
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

    match axis {
        Some(ax) => {
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
                                / T::from_f64(2.0).unwrap()
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
                            (row_values[mid - 1] + row_values[mid]) / T::from_f64(2.0).unwrap()
                        } else {
                            row_values[row_values.len() / 2]
                        };

                        result[i] = median_value;
                    }

                    Ok(result)
                }
                _ => Err("Axis index out of bounds for 2D array"),
            }
        }
        None => {
            // Global median
            let mut values: Vec<_> = array.iter().copied().collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median_value = if values.len() % 2 == 0 {
                let mid = values.len() / 2;
                (values[mid - 1] + values[mid]) / T::from_f64(2.0).unwrap()
            } else {
                values[values.len() / 2]
            };

            Ok(Array::from_elem(1, median_value))
        }
    }
}

/// Calculate the standard deviation of array elements (2D arrays)
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
/// let global_std = std_dev_2d(&a.view(), None, 1).unwrap();
/// assert!((global_std[0] - 1.87082869339_f64).abs() < 1e-10);
///
/// // Standard deviation along axis 0 (columns)
/// let col_stds = std_dev_2d(&a.view(), Some(Axis(0)), 1).unwrap();
/// assert_eq!(col_stds.len(), 3);
/// ```
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
/// let global_var = variance_2d(&a.view(), None, 1).unwrap();
/// assert!((global_var[0] - 3.5_f64).abs() < 1e-10);
///
/// // Variance along axis 0 (columns)
/// let col_vars = variance_2d(&a.view(), Some(Axis(0)), 1).unwrap();
/// assert_eq!(col_vars.len(), 3);
/// ```
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

    match axis {
        Some(ax) => {
            let (rows, cols) = (array.shape()[0], array.shape()[1]);

            match ax.index() {
                0 => {
                    // Variance along axis 0 (columns)
                    let means = mean_2d(array, Some(ax))?;

                    if rows <= ddof {
                        return Err(
                            "Not enough data points for variance calculation with given ddof",
                        );
                    }

                    let mut result = Array::<T, Ix1>::zeros(cols);

                    for j in 0..cols {
                        let mut sum_sq_diff = T::zero();
                        for i in 0..rows {
                            let diff = array[[i, j]] - means[j];
                            sum_sq_diff = sum_sq_diff + (diff * diff);
                        }

                        let divisor = T::from_usize(rows - ddof).unwrap();
                        result[j] = sum_sq_diff / divisor;
                    }

                    Ok(result)
                }
                1 => {
                    // Variance along axis 1 (rows)
                    let means = mean_2d(array, Some(ax))?;

                    if cols <= ddof {
                        return Err(
                            "Not enough data points for variance calculation with given ddof",
                        );
                    }

                    let mut result = Array::<T, Ix1>::zeros(rows);

                    for i in 0..rows {
                        let mut sum_sq_diff = T::zero();
                        for j in 0..cols {
                            let diff = array[[i, j]] - means[i];
                            sum_sq_diff = sum_sq_diff + (diff * diff);
                        }

                        let divisor = T::from_usize(cols - ddof).unwrap();
                        result[i] = sum_sq_diff / divisor;
                    }

                    Ok(result)
                }
                _ => Err("Axis index out of bounds for 2D array"),
            }
        }
        None => {
            // Global variance
            let total_elements = array.len();

            if total_elements <= ddof {
                return Err("Not enough data points for variance calculation with given ddof");
            }

            // Calculate global mean
            let global_mean = mean_2d(array, None)?[0];

            // Calculate sum of squared differences from the mean
            let mut sum_sq_diff = T::zero();
            for &val in array.iter() {
                let diff = val - global_mean;
                sum_sq_diff = sum_sq_diff + (diff * diff);
            }

            let divisor = T::from_usize(total_elements - ddof).unwrap();

            Ok(Array::from_elem(1, sum_sq_diff / divisor))
        }
    }
}

// Need to implement the following functions:
// min_2d, max_2d, sum_2d, product_2d, percentile_2d, mean, median, variance, std_dev, min, max, percentile

// Let's continue with min_2d and max_2d

/// Calculate the minimum value(s) of array elements (2D arrays)
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the minimum (None for global minimum)
///
/// # Returns
///
/// The minimum value(s) of the array elements
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

            for &val in array.iter() {
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
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the maximum (None for global maximum)
///
/// # Returns
///
/// The maximum value(s) of the array elements
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

            for &val in array.iter() {
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
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the sum (None for global sum)
///
/// # Returns
///
/// The sum of the array elements
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
                        result[i] = sum;
                    }

                    Ok(result)
                }
                _ => Err("Axis index out of bounds for 2D array"),
            }
        }
        None => {
            // Global sum
            let mut sum = T::zero();

            for &val in array.iter() {
                sum = sum + val;
            }

            Ok(Array::from_elem(1, sum))
        }
    }
}

/// Calculate the product of array elements (2D arrays)
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the product (None for global product)
///
/// # Returns
///
/// The product of the array elements
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

            for &val in array.iter() {
                product = product * val;
            }

            Ok(Array::from_elem(1, product))
        }
    }
}

/// Calculate the percentile of array elements (2D arrays)
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

                            result[j] = column_values[idx_low] * T::from_f64(weight_low).unwrap()
                                + column_values[idx_high] * T::from_f64(weight_high).unwrap();
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
                            result[i] = row_values[idx_low];
                        } else {
                            let weight_high = pos - (idx_low as f64);
                            let weight_low = 1.0 - weight_high;

                            result[i] = row_values[idx_low] * T::from_f64(weight_low).unwrap()
                                + row_values[idx_high] * T::from_f64(weight_high).unwrap();
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

                values[idx_low] * T::from_f64(weight_low).unwrap()
                    + values[idx_high] * T::from_f64(weight_high).unwrap()
            };

            Ok(Array::from_elem(1, result))
        }
    }
}

// Generic implementations for n-dimensional arrays

/// Calculate the mean of array elements
///
/// # Arguments
///
/// * `array` - The input array
/// * `axis` - Optional axis along which to compute the mean (None for global mean)
///
/// # Returns
///
/// The mean of the array elements
pub fn mean<T, D>(
    array: &ArrayView<T, D>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
    D: Dimension,
{
    if array.is_empty() {
        return Err("Cannot compute mean of an empty array");
    }

    match axis {
        Some(_) => {
            // For higher dimensional arrays, we need to implement axis-specific logic
            // This is a placeholder for now
            Err("Axis-specific mean for arbitrary dimension arrays not yet implemented")
        }
        None => {
            // Global mean
            let total_elements = array.len();
            let mut sum = T::zero();

            for &val in array.iter() {
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
pub fn median<T, D>(
    array: &ArrayView<T, D>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
    D: Dimension,
{
    if array.is_empty() {
        return Err("Cannot compute median of an empty array");
    }

    match axis {
        Some(_) => {
            // For higher dimensional arrays, we need to implement axis-specific logic
            // This is a placeholder for now
            Err("Axis-specific median for arbitrary dimension arrays not yet implemented")
        }
        None => {
            // Global median
            let mut values: Vec<_> = array.iter().copied().collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median_value = if values.len() % 2 == 0 {
                let mid = values.len() / 2;
                (values[mid - 1] + values[mid]) / T::from_f64(2.0).unwrap()
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
pub fn variance<T, D>(
    array: &ArrayView<T, D>,
    axis: Option<Axis>,
    ddof: usize,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
    D: Dimension,
{
    if array.is_empty() {
        return Err("Cannot compute variance of an empty array");
    }

    match axis {
        Some(_) => {
            // For higher dimensional arrays, we need to implement axis-specific logic
            // This is a placeholder for now
            Err("Axis-specific variance for arbitrary dimension arrays not yet implemented")
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
            for &val in array.iter() {
                let diff = val - global_mean;
                sum_sq_diff = sum_sq_diff + (diff * diff);
            }

            let divisor = T::from_usize(total_elements - ddof).unwrap();

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
pub fn std_dev<T, D>(
    array: &ArrayView<T, D>,
    axis: Option<Axis>,
    ddof: usize,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
    D: Dimension,
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
pub fn min<T, D>(array: &ArrayView<T, D>, axis: Option<Axis>) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float,
    D: Dimension,
{
    if array.is_empty() {
        return Err("Cannot compute minimum of an empty array");
    }

    match axis {
        Some(_) => {
            // For higher dimensional arrays, we need to implement axis-specific logic
            // This is a placeholder for now
            Err("Axis-specific minimum for arbitrary dimension arrays not yet implemented")
        }
        None => {
            // Global minimum
            let mut min_val = *array.iter().next().unwrap();

            for &val in array.iter() {
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
pub fn max<T, D>(array: &ArrayView<T, D>, axis: Option<Axis>) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float,
    D: Dimension,
{
    if array.is_empty() {
        return Err("Cannot compute maximum of an empty array");
    }

    match axis {
        Some(_) => {
            // For higher dimensional arrays, we need to implement axis-specific logic
            // This is a placeholder for now
            Err("Axis-specific maximum for arbitrary dimension arrays not yet implemented")
        }
        None => {
            // Global maximum
            let mut max_val = *array.iter().next().unwrap();

            for &val in array.iter() {
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
pub fn percentile<T, D>(
    array: &ArrayView<T, D>,
    q: f64,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
    D: Dimension,
{
    if array.is_empty() {
        return Err("Cannot compute percentile of an empty array");
    }

    if !(0.0..=100.0).contains(&q) {
        return Err("Percentile must be between 0 and 100");
    }

    match axis {
        Some(_) => {
            // For higher dimensional arrays, we need to implement axis-specific logic
            // This is a placeholder for now
            Err("Axis-specific percentile for arbitrary dimension arrays not yet implemented")
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

                values[idx_low] * T::from_f64(weight_low).unwrap()
                    + values[idx_high] * T::from_f64(weight_high).unwrap()
            };

            Ok(Array::from_elem(1, result))
        }
    }
}
