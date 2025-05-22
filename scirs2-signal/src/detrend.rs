//! Detrending functionality
//!
//! This module provides functions for removing linear trends and constant offsets
//! from signal data, which is often useful as a preprocessing step before
//! spectral analysis or other signal processing operations.

use crate::error::{SignalError, SignalResult};
use ndarray::{Array1, Array2};
use num_traits::{Float, NumCast};
use std::fmt::Debug;

/// Detrend a signal by removing a linear trend or constant offset.
///
/// This function removes a linear trend or constant offset from the input data.
/// Detrending is commonly used before certain signal processing operations like
/// FFT or spectral analysis to remove biases that might affect the results.
///
/// # Arguments
///
/// * `x` - The input signal
/// * `detrend_type` - The type of detrending to apply:
///   * "linear" - Remove a linear trend (default)
///   * "constant" - Remove the mean (DC offset)
///   * "none" - No detrending
///
/// # Returns
///
/// * The detrended signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::detrend;
///
/// // Create a signal with a linear trend
/// let mut x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
/// for i in 0..x.len() {
///     x[i] += 0.5 * i as f64; // Add a linear trend
/// }
///
/// // Remove the linear trend
/// let detrended = detrend(&x, Some("linear")).unwrap();
///
/// // The result should be close to the original signal without the trend
/// for val in &detrended {
///     assert!(val.abs() < 1e-12);
/// }
/// ```
///
/// Remove just the mean value:
///
/// ```
/// use scirs2_signal::detrend;
///
/// // Create a signal with a constant offset
/// let x = vec![5.0, 6.0, 7.0, 8.0, 9.0];
///
/// // Remove the constant offset
/// let detrended = detrend(&x, Some("constant")).unwrap();
///
/// // The result should have zero mean
/// let mean: f64 = detrended.iter().sum::<f64>() / detrended.len() as f64;
/// assert!(mean.abs() < 1e-12);
/// ```
pub fn detrend<T>(x: &[T], detrend_type: Option<&str>) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    // Check input
    if x.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    // Default to linear detrending
    let detrend_str = detrend_type.unwrap_or("linear");

    // Convert input to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val)
                .ok_or_else(|| SignalError::ValueError(format!("Could not convert {val:?} to f64")))
        })
        .collect::<SignalResult<Vec<_>>>()?;

    match detrend_str {
        "none" => Ok(x_f64),
        "constant" => {
            // Remove the mean
            let mean = x_f64.iter().sum::<f64>() / x_f64.len() as f64;
            Ok(x_f64.iter().map(|&x| x - mean).collect())
        }
        "linear" => {
            // Remove a linear trend
            let n = x_f64.len();
            let x_indices: Vec<f64> = (0..n).map(|i| i as f64).collect();

            // Calculate the linear regression
            let mean_x = x_indices.iter().sum::<f64>() / n as f64;
            let mean_y = x_f64.iter().sum::<f64>() / n as f64;

            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for i in 0..n {
                let x_diff = x_indices[i] - mean_x;
                let y_diff = x_f64[i] - mean_y;
                numerator += x_diff * y_diff;
                denominator += x_diff * x_diff;
            }

            let slope = if denominator.abs() < f64::EPSILON {
                0.0 // Handle the case where all x values are the same
            } else {
                numerator / denominator
            };

            let intercept = mean_y - slope * mean_x;

            // Remove the trend
            Ok(x_f64
                .iter()
                .zip(x_indices.iter())
                .map(|(&y, &x)| y - (intercept + slope * x))
                .collect())
        }
        _ => Err(SignalError::ValueError(format!(
            "Unknown detrend type: {detrend_str}. Must be 'linear', 'constant', or 'none'."
        ))),
    }
}

/// Detrend data along a specified axis.
///
/// This function applies detrending along a specified axis of a 2D array.
/// It's useful for processing multiple signals or time series at once.
///
/// # Arguments
///
/// * `x` - The input 2D array
/// * `detrend_type` - The type of detrending to apply ("linear", "constant", or "none")
/// * `axis` - The axis along which to detrend (0 for rows, 1 for columns)
///
/// # Returns
///
/// * The detrended 2D array
///
/// # Examples
///
/// ```
/// use scirs2_signal::detrend_axis;
/// use ndarray::array;
///
/// // Create a 2D array with a trend along columns
/// let mut x = array![
///     [1.0, 2.0, 3.0, 4.0],
///     [2.0, 3.0, 4.0, 5.0],
///     [3.0, 4.0, 5.0, 6.0]
/// ];
///
/// // Detrend along axis 0 (columns)
/// let detrended = detrend_axis(&x, Some("linear"), 0).unwrap();
///
/// // Each column should now have zero trend
/// for col in 0..x.shape()[1] {
///     let col_view = detrended.column(col);
///     
///     // Check if the sum of products with position is close to zero (indicating no trend)
///     let mut trend_measure = 0.0;
///     for (i, &val) in col_view.iter().enumerate() {
///         trend_measure += val * i as f64;
///     }
///     
///     assert!(trend_measure.abs() < 1e-10);
/// }
/// ```
pub fn detrend_axis(
    x: &Array2<f64>,
    detrend_type: Option<&str>,
    axis: usize,
) -> SignalResult<Array2<f64>> {
    // Check input
    if x.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    if axis > 1 {
        return Err(SignalError::ValueError(
            "Axis must be 0 or 1 for a 2D array".to_string(),
        ));
    }

    // Default to linear detrending
    let detrend_str = detrend_type.unwrap_or("linear");

    // Get dimensions
    let shape = x.shape();
    let (n_rows, n_cols) = (shape[0], shape[1]);

    // Create output array with the same shape
    let mut result = Array2::zeros((n_rows, n_cols));

    match detrend_str {
        "none" => {
            // No detrending, just copy the input
            result.assign(x);
        }
        "constant" => {
            // Remove the mean along the specified axis
            for i in 0..n_rows {
                for j in 0..n_cols {
                    if axis == 0 {
                        // Detrend columns
                        let col = x.column(j);
                        let mean = col.sum() / col.len() as f64;
                        result[[i, j]] = x[[i, j]] - mean;
                    } else {
                        // Detrend rows
                        let row = x.row(i);
                        let mean = row.sum() / row.len() as f64;
                        result[[i, j]] = x[[i, j]] - mean;
                    }
                }
            }
        }
        "linear" => {
            // Remove a linear trend along the specified axis
            if axis == 0 {
                // Detrend columns
                for j in 0..n_cols {
                    let col = x.column(j).to_vec();
                    let detrended = detrend(&col, Some("linear"))?;
                    for (i, &val) in detrended.iter().enumerate() {
                        result[[i, j]] = val;
                    }
                }
            } else {
                // Detrend rows
                for i in 0..n_rows {
                    let row = x.row(i).to_vec();
                    let detrended = detrend(&row, Some("linear"))?;
                    for (j, &val) in detrended.iter().enumerate() {
                        result[[i, j]] = val;
                    }
                }
            }
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown detrend type: {detrend_str}. Must be 'linear', 'constant', or 'none'."
            )));
        }
    }

    Ok(result)
}

/// Detrend a signal by removing a polynomial trend of specified order.
///
/// This function removes a polynomial trend of specified order from the input data.
/// It's a generalization of linear detrending that can handle higher-order trends.
///
/// # Arguments
///
/// * `x` - The input signal
/// * `order` - The order of the polynomial to fit and remove (1 = linear, 2 = quadratic, etc.)
///
/// # Returns
///
/// * The detrended signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::detrend_poly;
///
/// // Create a signal with a quadratic trend: y = 0.1*x^2 + 0.5*x + 1
/// let n = 10;
/// let mut x = vec![0.0; n];
/// for i in 0..n {
///     let t = i as f64;
///     x[i] = 0.1*t*t + 0.5*t + 1.0;
/// }
///
/// // Remove a quadratic trend (polynomial of order 2)
/// let detrended = detrend_poly(&x, 2).unwrap();
///
/// // The result should be close to zero (the quadratic trend removed)
/// for val in &detrended {
///     assert!(val.abs() < 1e-10);
/// }
///
/// // With a lower order, there will still be residual trend
/// let detrended_linear = detrend_poly(&x, 1).unwrap();
/// for i in 1..detrended_linear.len() - 1 {
///     // Non-zero residuals expected when removing only linear component
///     assert!(detrended_linear[i].abs() > 1e-10);
/// }
/// ```
pub fn detrend_poly<T>(x: &[T], order: usize) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    // Check input
    if x.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    if order == 0 {
        // Special case: order 0 is just removing the mean (constant detrending)
        return detrend(x, Some("constant"));
    }

    // Convert input to f64
    let y: Vec<f64> = x
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val)
                .ok_or_else(|| SignalError::ValueError(format!("Could not convert {val:?} to f64")))
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Generate x values and powers
    let n = y.len();
    let x_indices: Vec<f64> = (0..n).map(|i| i as f64).collect();

    // Create Vandermonde matrix for polynomial fitting
    let mut vandermonde = Array2::zeros((n, order + 1));
    for i in 0..n {
        for j in 0..=order {
            vandermonde[[i, j]] = x_indices[i].powi(j as i32);
        }
    }

    // Convert y to ndarray for matrix operations
    let y_array = Array1::from_vec(y.clone());

    // Solve for polynomial coefficients using least squares
    // (V^T V) c = V^T y
    let vt_v = vandermonde.t().dot(&vandermonde);
    let vt_y = vandermonde.t().dot(&y_array);

    // Use simple matrix solve for this small system
    let coefficients = match solve_linear_system(&vt_v, &vt_y) {
        Ok(c) => c,
        Err(e) => {
            return Err(SignalError::ComputationError(format!(
                "Failed to fit polynomial: {e}"
            )))
        }
    };

    // Evaluate the polynomial at each x value
    let mut trend = vec![0.0; n];
    for i in 0..n {
        for (j, &coeff) in coefficients.iter().enumerate().take(order + 1) {
            trend[i] += coeff * x_indices[i].powi(j as i32);
        }
    }

    // Subtract the trend from the original signal
    Ok(y.iter().zip(trend.iter()).map(|(&y, &t)| y - t).collect())
}

/// Helper function to solve a small linear system Ax = b
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> SignalResult<Vec<f64>> {
    // Check dimensions
    let (n, m) = a.dim();
    if n != m {
        return Err(SignalError::ValueError(
            "Matrix A must be square".to_string(),
        ));
    }
    if n != b.len() {
        return Err(SignalError::ValueError(
            "Dimensions of A and b must match".to_string(),
        ));
    }

    // A simple direct solver - not the most efficient, but works for small systems
    // In practice, we'd use a more robust solver from scirs2-linalg
    let mut a_copy = a.clone();
    let mut b_copy = b.clone();

    // Gaussian elimination with partial pivoting
    for i in 0..n - 1 {
        // Find pivot
        let mut max_idx = i;
        let mut max_val = a_copy[[i, i]].abs();

        for j in i + 1..n {
            let val = a_copy[[j, i]].abs();
            if val > max_val {
                max_idx = j;
                max_val = val;
            }
        }

        // Check for singularity
        if max_val < 1e-10 {
            return Err(SignalError::ComputationError(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows if needed
        if max_idx != i {
            for j in i..n {
                let temp = a_copy[[i, j]];
                a_copy[[i, j]] = a_copy[[max_idx, j]];
                a_copy[[max_idx, j]] = temp;
            }
            let temp = b_copy[i];
            b_copy[i] = b_copy[max_idx];
            b_copy[max_idx] = temp;
        }

        // Eliminate
        for j in i + 1..n {
            let factor = a_copy[[j, i]] / a_copy[[i, i]];
            b_copy[j] -= factor * b_copy[i];

            for k in i..n {
                a_copy[[j, k]] -= factor * a_copy[[i, k]];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in i + 1..n {
            sum += a_copy[[i, j]] * x[j];
        }
        x[i] = (b_copy[i] - sum) / a_copy[[i, i]];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    #[test]
    fn test_detrend_constant() {
        // Test signal with constant offset
        let signal = vec![5.0, 6.0, 7.0, 8.0, 9.0];
        let detrended = detrend(&signal, Some("constant")).unwrap();

        // Check mean is zero
        let mean = detrended.iter().sum::<f64>() / detrended.len() as f64;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-12);

        // Check shape is preserved
        assert_eq!(detrended.len(), signal.len());
    }

    #[test]
    fn test_detrend_linear() {
        // Test signal with linear trend: y = 2x + 1
        let mut signal = vec![0.0; 10];
        signal.iter_mut().enumerate().for_each(|(i, val)| {
            *val = 2.0 * i as f64 + 1.0;
        });

        let detrended = detrend(&signal, Some("linear")).unwrap();

        // Check all values are near zero (trend removed)
        for val in &detrended {
            assert_relative_eq!(*val, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_detrend_none() {
        // Test with no detrending
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let detrended = detrend(&signal, Some("none")).unwrap();

        // Check signal is unchanged
        for (a, b) in signal.iter().zip(detrended.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_detrend_axis() {
        // Create a 2D array with trends along both axes
        let mut data = Array2::zeros((3, 4));
        for i in 0..3 {
            for j in 0..4 {
                data[[i, j]] = 1.0 + 2.0 * i as f64 + 3.0 * j as f64;
            }
        }

        // Detrend along axis 0 (columns)
        let detrended_cols = detrend_axis(&data, Some("linear"), 0).unwrap();

        // Check each column has zero trend
        for j in 0..4 {
            let col = detrended_cols.column(j).to_vec();

            // Linear regression should give nearly zero slope
            let n = col.len();
            let x: Vec<f64> = (0..n).map(|i| i as f64).collect();

            let mean_x = x.iter().sum::<f64>() / n as f64;
            let mean_y = col.iter().sum::<f64>() / n as f64;

            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for i in 0..n {
                let x_diff = x[i] - mean_x;
                let y_diff = col[i] - mean_y;
                numerator += x_diff * y_diff;
                denominator += x_diff * x_diff;
            }

            let slope = numerator / denominator;
            assert_relative_eq!(slope, 0.0, epsilon = 1e-12);
        }

        // Detrend along axis 1 (rows)
        let detrended_rows = detrend_axis(&data, Some("linear"), 1).unwrap();

        // Check each row has zero trend
        for i in 0..3 {
            let row = detrended_rows.row(i).to_vec();

            // Linear regression should give nearly zero slope
            let n = row.len();
            let x: Vec<f64> = (0..n).map(|i| i as f64).collect();

            let mean_x = x.iter().sum::<f64>() / n as f64;
            let mean_y = row.iter().sum::<f64>() / n as f64;

            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for i in 0..n {
                let x_diff = x[i] - mean_x;
                let y_diff = row[i] - mean_y;
                numerator += x_diff * y_diff;
                denominator += x_diff * x_diff;
            }

            let slope = numerator / denominator;
            assert_relative_eq!(slope, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_detrend_poly() {
        // Create a signal with a cubic trend: y = 0.1*x^3 - 0.5*x^2 + 2*x - 3
        let n = 20;
        let mut signal = vec![0.0; n];
        signal.iter_mut().enumerate().for_each(|(i, val)| {
            let x = i as f64;
            *val = 0.1 * x.powi(3) - 0.5 * x.powi(2) + 2.0 * x - 3.0;
        });

        // Remove cubic trend (polynomial of order 3)
        let detrended = detrend_poly(&signal, 3).unwrap();

        // Should be nearly zero
        for val in &detrended {
            assert_relative_eq!(*val, 0.0, epsilon = 1e-8);
        }

        // With lower order, there will still be trend
        let detrended_quadratic = detrend_poly(&signal, 2).unwrap();

        // Should have significant non-zero values
        let sum_sq = detrended_quadratic.iter().map(|&x| x * x).sum::<f64>();
        assert!(sum_sq > 1.0);
    }
}
