//! Basic interpolation methods for signal processing
//!
//! This module provides simple, fast interpolation algorithms including
//! linear interpolation and nearest neighbor interpolation.

use super::core::find_nearest_valid_index;
use crate::error::{SignalError, SignalResult};
use ndarray::Array1;

/// Applies linear interpolation to fill missing values in a signal
///
/// Linear interpolation connects missing points with straight lines between
/// neighboring valid points. This is one of the simplest and fastest
/// interpolation methods.
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
///
/// # Returns
///
/// * Interpolated signal where missing values are filled using linear interpolation
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_signal::interpolate::basic::linear_interpolate;
///
/// let mut signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
/// let result = linear_interpolate(&signal).unwrap();
/// // Result will be approximately [1.0, 2.0, 3.0, 4.0, 5.0]
/// ```
pub fn linear_interpolate(signal: &Array1<f64>) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    let mut result = signal.clone();

    // Find all non-missing points
    let mut valid_indices = Vec::new();
    let mut valid_values = Vec::new();

    for i in 0..n {
        if !signal[i].is_nan() {
            valid_indices.push(i);
            valid_values.push(signal[i]);
        }
    }

    if valid_indices.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    if valid_indices.len() == 1 {
        // If only one valid point, fill with that value
        let value = valid_values[0];
        for i in 0..n {
            result[i] = value;
        }
        return Ok(result);
    }

    // Interpolate missing values
    for i in 0..n {
        if signal[i].is_nan() {
            // Find the valid points surrounding the missing value
            let mut left_idx = None;
            let mut right_idx = None;

            for (j, &valid_idx) in valid_indices.iter().enumerate() {
                match valid_idx.cmp(&i) {
                    std::cmp::Ordering::Less => left_idx = Some(j),
                    std::cmp::Ordering::Greater => {
                        right_idx = Some(j);
                        break;
                    }
                    std::cmp::Ordering::Equal => {} // No action needed for exact match
                }
            }

            match (left_idx, right_idx) {
                (Some(left), Some(right)) => {
                    // Interpolate between left and right valid points
                    let x1 = valid_indices[left] as f64;
                    let x2 = valid_indices[right] as f64;
                    let y1 = valid_values[left];
                    let y2 = valid_values[right];
                    let x = i as f64;

                    // Linear interpolation formula
                    result[i] = y1 + (y2 - y1) * (x - x1) / (x2 - x1);
                }
                (Some(left), None) => {
                    // Extrapolate using the last valid point
                    result[i] = valid_values[left];
                }
                (None, Some(right)) => {
                    // Extrapolate using the first valid point
                    result[i] = valid_values[right];
                }
                (None, None) => {
                    // This shouldn't happen if we have at least one valid point
                    result[i] = 0.0;
                }
            }
        }
    }

    Ok(result)
}

/// Applies nearest neighbor interpolation to fill missing values in a signal
///
/// Nearest neighbor interpolation fills each missing value with the value
/// of the nearest valid (non-missing) sample. This method preserves the
/// original values exactly and doesn't introduce new values.
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
///
/// # Returns
///
/// * Interpolated signal where missing values are filled with nearest valid values
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_signal::interpolate::basic::nearest_neighbor_interpolate;
///
/// let mut signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
/// let result = nearest_neighbor_interpolate(&signal).unwrap();
/// // Result will be [1.0, 1.0, 3.0, 3.0, 5.0] or [1.0, 3.0, 3.0, 5.0, 5.0]
/// // depending on which valid point is closer
/// ```
pub fn nearest_neighbor_interpolate(signal: &Array1<f64>) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    // Find indices of non-missing points
    let mut valid_indices = Vec::new();
    let mut valid_values = Vec::new();

    for i in 0..n {
        if !signal[i].is_nan() {
            valid_indices.push(i);
            valid_values.push(signal[i]);
        }
    }

    if valid_indices.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    // Create result by filling missing values with nearest valid value
    let mut result = signal.clone();

    for i in 0..n {
        if signal[i].is_nan() {
            let nearest_idx = find_nearest_valid_index(i, &valid_indices);
            result[i] = valid_values[nearest_idx];
        }
    }

    Ok(result)
}

/// Unit tests for basic interpolation methods
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_linear_interpolate_no_missing() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = linear_interpolate(&signal).unwrap();
        assert_eq!(result, signal);
    }

    #[test]
    fn test_linear_interpolate_simple() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);
        let result = linear_interpolate(&signal).unwrap();
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
    }

    #[test]
    fn test_linear_interpolate_multiple_missing() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, f64::NAN, 4.0]);
        let result = linear_interpolate(&signal).unwrap();
        assert_eq!(result[0], 1.0);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 3.0).abs() < 1e-10);
        assert_eq!(result[3], 4.0);
    }

    #[test]
    fn test_linear_interpolate_boundary_missing() {
        let signal = Array1::from_vec(vec![f64::NAN, 2.0, 3.0, f64::NAN]);
        let result = linear_interpolate(&signal).unwrap();
        assert_eq!(result[0], 2.0); // Extrapolated from first valid point
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
        assert_eq!(result[3], 3.0); // Extrapolated from last valid point
    }

    #[test]
    fn test_linear_interpolate_single_valid() {
        let signal = Array1::from_vec(vec![f64::NAN, 2.0, f64::NAN]);
        let result = linear_interpolate(&signal).unwrap();
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 2.0);
    }

    #[test]
    fn test_linear_interpolate_all_missing() {
        let signal = Array1::from_vec(vec![f64::NAN, f64::NAN, f64::NAN]);
        let result = linear_interpolate(&signal);
        assert!(result.is_err());
    }

    #[test]
    fn test_nearest_neighbor_interpolate_no_missing() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = nearest_neighbor_interpolate(&signal).unwrap();
        assert_eq!(result, signal);
    }

    #[test]
    fn test_nearest_neighbor_interpolate_simple() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, 5.0]);
        let result = nearest_neighbor_interpolate(&signal).unwrap();
        assert_eq!(result[0], 1.0);
        // Middle point should be filled with nearest neighbor (either 1.0 or 5.0)
        assert!(result[1] == 1.0 || result[1] == 5.0);
        assert_eq!(result[2], 5.0);
    }

    #[test]
    fn test_nearest_neighbor_interpolate_multiple_missing() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, f64::NAN, f64::NAN, 10.0]);
        let result = nearest_neighbor_interpolate(&signal).unwrap();
        assert_eq!(result[0], 1.0);
        // Missing values should be filled with nearest neighbors
        assert!(result.iter().all(|&x| !x.is_nan()));
        assert_eq!(result[4], 10.0);
    }

    #[test]
    fn test_nearest_neighbor_interpolate_boundary_missing() {
        let signal = Array1::from_vec(vec![f64::NAN, 2.0, 3.0, f64::NAN]);
        let result = nearest_neighbor_interpolate(&signal).unwrap();
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
        // Boundary missing values should be filled with nearest valid values
        assert!(result[0] == 2.0 || result[0] == 3.0);
        assert!(result[3] == 2.0 || result[3] == 3.0);
    }

    #[test]
    fn test_nearest_neighbor_interpolate_all_missing() {
        let signal = Array1::from_vec(vec![f64::NAN, f64::NAN, f64::NAN]);
        let result = nearest_neighbor_interpolate(&signal);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_signal() {
        let signal = Array1::from_vec(vec![]);
        let result = linear_interpolate(&signal).unwrap();
        assert_eq!(result.len(), 0);

        let result = nearest_neighbor_interpolate(&signal).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_single_element_signal() {
        let signal = Array1::from_vec(vec![5.0]);
        let result = linear_interpolate(&signal).unwrap();
        assert_eq!(result[0], 5.0);

        let result = nearest_neighbor_interpolate(&signal).unwrap();
        assert_eq!(result[0], 5.0);
    }

    #[test]
    fn test_single_missing_element() {
        let signal = Array1::from_vec(vec![f64::NAN]);
        let result = linear_interpolate(&signal);
        assert!(result.is_err());

        let result = nearest_neighbor_interpolate(&signal);
        assert!(result.is_err());
    }
}
