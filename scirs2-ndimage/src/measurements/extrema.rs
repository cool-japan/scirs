//! Functions for finding extrema in arrays

use ndarray::{Array, Dimension};
use num_traits::{Float, FromPrimitive, NumAssign};
use std::fmt::Debug;

use crate::error::{NdimageError, Result};

/// Find the extrema (min, max, min_loc, max_loc) of an array
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// * `Result<(T, T, Vec<usize>, Vec<usize>)>` - (min, max, min_loc, max_loc)
pub fn extrema<T, D>(input: &Array<T, D>) -> Result<(T, T, Vec<usize>, Vec<usize>)>
where
    T: Float + FromPrimitive + Debug + NumAssign + PartialOrd,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if input.is_empty() {
        return Err(NdimageError::InvalidInput("Input array is empty".into()));
    }

    // Convert to dynamic array for easier indexing
    let input_dyn = input.clone().into_dyn();

    let mut min_val = None;
    let mut max_val = None;
    let mut min_loc = vec![0; input.ndim()];
    let mut max_loc = vec![0; input.ndim()];

    // Find min and max values and their locations
    for (idx, &value) in input_dyn.indexed_iter() {
        let idx_vec: Vec<usize> = idx.as_array_view().to_vec();

        match min_val {
            None => {
                min_val = Some(value);
                max_val = Some(value);
                min_loc = idx_vec.clone();
                max_loc = idx_vec;
            }
            Some(current_min) => {
                if value < current_min {
                    min_val = Some(value);
                    min_loc = idx_vec.clone();
                }
                if let Some(current_max) = max_val {
                    if value > current_max {
                        max_val = Some(value);
                        max_loc = idx_vec;
                    }
                }
            }
        }
    }

    match (min_val, max_val) {
        (Some(min), Some(max)) => Ok((min, max, min_loc, max_loc)),
        _ => {
            // This should not happen since we check for empty array above
            let origin = vec![0; input.ndim()];
            Ok((T::zero(), T::zero(), origin.clone(), origin))
        }
    }
}

/// Find the local extrema of an array
///
/// # Arguments
///
/// * `input` - Input array
/// * `size` - Size of neighborhood (default: 3 for each dimension)
/// * `mode` - Mode for comparison: "min", "max", or "both" (default: "both")
///
/// # Returns
///
/// * `Result<(Array<bool, D>, Array<bool, D>)>` - Arrays indicating locations of minima and maxima
pub fn local_extrema<T, D>(
    input: &Array<T, D>,
    size: Option<&[usize]>,
    mode: Option<&str>,
) -> Result<(Array<bool, D>, Array<bool, D>)>
where
    T: Float + FromPrimitive + Debug + NumAssign + PartialOrd,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if input.is_empty() {
        return Err(NdimageError::InvalidInput("Input array is empty".into()));
    }

    if let Some(s) = size {
        if s.len() != input.ndim() {
            return Err(NdimageError::DimensionError(format!(
                "Size must have same length as input dimensions (got {} expected {})",
                s.len(),
                input.ndim()
            )));
        }

        for &val in s {
            if val == 0 || val % 2 == 0 {
                return Err(NdimageError::InvalidInput(
                    "Size values must be positive odd integers".into(),
                ));
            }
        }
    }

    let m = mode.unwrap_or("both");
    if m != "min" && m != "max" && m != "both" {
        return Err(NdimageError::InvalidInput(format!(
            "Mode must be 'min', 'max', or 'both', got '{}'",
            m
        )));
    }

    // Placeholder implementation
    let minima = Array::<bool, _>::from_elem(input.raw_dim(), false);
    let maxima = Array::<bool, _>::from_elem(input.raw_dim(), false);

    Ok((minima, maxima))
}

/// Find peak prominence in a 1D array
///
/// # Arguments
///
/// * `input` - Input 1D array
/// * `peaks` - Indices of peaks
/// * `wlen` - Window length for calculating prominence (default: None)
///
/// # Returns
///
/// * `Result<Vec<T>>` - Prominence values for each peak
pub fn peak_prominences<T>(
    input: &Array<T, ndarray::Ix1>,
    peaks: &[usize],
    _wlen: Option<usize>,
) -> Result<Vec<T>>
where
    T: Float + FromPrimitive + Debug + NumAssign,
{
    // Validate inputs
    if input.is_empty() {
        return Err(NdimageError::InvalidInput("Input array is empty".into()));
    }

    if peaks.is_empty() {
        return Ok(Vec::new());
    }

    for &p in peaks {
        if p >= input.len() {
            return Err(NdimageError::InvalidInput(format!(
                "Peak index {} is out of bounds for array of length {}",
                p,
                input.len()
            )));
        }
    }

    // Placeholder implementation
    Ok(vec![T::one(); peaks.len()])
}

/// Find peak widths in a 1D array
///
/// # Arguments
///
/// * `input` - Input 1D array
/// * `peaks` - Indices of peaks
/// * `rel_height` - Relative height at which to measure peak width (default: 0.5)
///
/// # Returns
///
/// Type alias for width measurement results
pub type PeakWidthsResult<T> = (Vec<T>, Vec<T>, Vec<T>, Vec<T>);

/// * `Result<PeakWidthsResult<T>>` - (widths, width_heights, left_ips, right_ips)
pub fn peak_widths<T>(
    input: &Array<T, ndarray::Ix1>,
    peaks: &[usize],
    rel_height: Option<T>,
) -> Result<PeakWidthsResult<T>>
where
    T: Float + FromPrimitive + Debug + NumAssign,
{
    // Validate inputs
    if input.is_empty() {
        return Err(NdimageError::InvalidInput("Input array is empty".into()));
    }

    if peaks.is_empty() {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }

    for &p in peaks {
        if p >= input.len() {
            return Err(NdimageError::InvalidInput(format!(
                "Peak index {} is out of bounds for array of length {}",
                p,
                input.len()
            )));
        }
    }

    let height = rel_height.unwrap_or_else(|| T::from_f64(0.5).unwrap());
    if height <= T::zero() || height >= T::one() {
        return Err(NdimageError::InvalidInput(format!(
            "rel_height must be between 0 and 1, got {:?}",
            height
        )));
    }

    // Placeholder implementation
    let widths = vec![T::one(); peaks.len()];
    let heights = vec![T::zero(); peaks.len()];
    let left_ips = vec![T::zero(); peaks.len()];
    let right_ips = vec![T::from_usize(input.len() - 1).unwrap(); peaks.len()];

    Ok((widths, heights, left_ips, right_ips))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_extrema() {
        let input: Array2<f64> = Array2::eye(3);
        let (min, max, min_loc, max_loc) = extrema(&input).unwrap();
        assert!(max >= min);
        assert_eq!(min_loc.len(), input.ndim());
        assert_eq!(max_loc.len(), input.ndim());
    }

    #[test]
    fn test_local_extrema() {
        let input: Array2<f64> = Array2::eye(3);
        let (minima, maxima) = local_extrema(&input, None, None).unwrap();
        assert_eq!(minima.shape(), input.shape());
        assert_eq!(maxima.shape(), input.shape());
    }
}
