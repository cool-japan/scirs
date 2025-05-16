//! Convolution functions for n-dimensional arrays

use ndarray::{Array, Dimension};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::BorderMode;
use crate::error::{NdimageError, Result};

/// Apply a uniform filter (box filter or moving average) to an n-dimensional array
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `size` - Size of the filter kernel in each dimension
/// * `mode` - Border handling mode (defaults to Reflect)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
pub fn uniform_filter<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: Option<BorderMode>,
) -> Result<Array<T, D>>
where
    T: Float + FromPrimitive + Debug,
    D: Dimension,
{
    let _border_mode = mode.unwrap_or(BorderMode::Reflect);

    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if size.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Size must have same length as input dimensions (got {} expected {})",
            size.len(),
            input.ndim()
        )));
    }

    for &s in size {
        if s == 0 {
            return Err(NdimageError::InvalidInput(
                "Kernel size cannot be zero".into(),
            ));
        }
    }

    // Placeholder implementation returning a copy of the input
    // This will be implemented with proper uniform filtering
    Ok(input.to_owned())
}

/// Convolve an n-dimensional array with a filter kernel
///
/// # Arguments
///
/// * `input` - Input array to convolve
/// * `weights` - Convolution kernel
/// * `mode` - Border handling mode (defaults to Reflect)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Convolved array
pub fn convolve<T, D, E>(
    input: &Array<T, D>,
    weights: &Array<T, E>,
    mode: Option<BorderMode>,
) -> Result<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + Clone,
    D: Dimension,
    E: Dimension,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if weights.ndim() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Weights must have same rank as input (got {} expected {})",
            weights.ndim(),
            input.ndim()
        )));
    }

    // For 2D arrays, use specialized implementation
    if input.ndim() == 2 && weights.ndim() == 2 {
        match (input.ndim(), weights.ndim()) {
            (2, 2) => {
                // Convert to 2D arrays
                let input_2d = input
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| {
                        NdimageError::DimensionError("Failed to convert to 2D array".into())
                    })?;

                let weights_2d = weights
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| {
                        NdimageError::DimensionError("Failed to convert weights to 2D array".into())
                    })?;

                // Call 2D implementation
                let result = convolve_2d(&input_2d, &weights_2d, &border_mode)?;

                // Convert back to original dimensionality
                result.into_dimensionality::<D>().map_err(|_| {
                    NdimageError::DimensionError(
                        "Failed to convert result back to original dimensions".into(),
                    )
                })
            }
            _ => unreachable!(), // Already checked dimensions above
        }
    } else {
        // For now, return a not implemented error for other dimensionalities
        Err(NdimageError::NotImplementedError(
            "Convolution for arrays with dimensions other than 2 is not yet implemented".into(),
        ))
    }
}

/// Perform 2D convolution with a kernel
fn convolve_2d<T>(
    input: &Array<T, ndarray::Ix2>,
    weights: &Array<T, ndarray::Ix2>,
    mode: &BorderMode,
) -> Result<Array<T, ndarray::Ix2>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + Clone,
{
    // Get dimensions
    let (input_rows, input_cols) = input.dim();
    let (weights_rows, weights_cols) = weights.dim();

    // Calculate padding required
    let pad_rows_before = weights_rows / 2;
    let pad_rows_after = weights_rows - pad_rows_before - 1;
    let pad_cols_before = weights_cols / 2;
    let pad_cols_after = weights_cols - pad_cols_before - 1;

    // Create output array
    let mut output = Array::<T, ndarray::Ix2>::zeros((input_rows, input_cols));

    // Create padding configuration
    let pad_width = vec![
        (pad_rows_before, pad_rows_after),
        (pad_cols_before, pad_cols_after),
    ];

    // Import pad_array from parent module
    use super::pad_array;

    // Pad the input array
    let padded = pad_array(input, &pad_width, mode, None)?;

    // Apply convolution
    for i in 0..input_rows {
        for j in 0..input_cols {
            let mut sum = T::zero();

            // Apply the kernel at each position
            for ki in 0..weights_rows {
                for kj in 0..weights_cols {
                    let input_val = padded[[i + ki, j + kj]];
                    let weight = weights[[weights_rows - ki - 1, weights_cols - kj - 1]]; // Flip kernel for convolution
                    sum += input_val * weight;
                }
            }

            output[[i, j]] = sum;
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_uniform_filter() {
        // Create a simple test image
        let image: Array2<f64> = Array2::eye(5);

        // Apply filter
        let result = uniform_filter(&image, &[3, 3], None).unwrap();

        // Check that result has the same shape
        assert_eq!(result.shape(), image.shape());
    }

    #[test]
    fn test_convolve() {
        // Create a simple test image and kernel
        let image: Array2<f64> = Array2::eye(5);
        let kernel: Array2<f64> = Array2::from_elem((3, 3), 1.0 / 9.0);

        // Apply convolution
        let result = convolve(&image, &kernel, None).unwrap();

        // Check that result has the same shape
        assert_eq!(result.shape(), image.shape());
    }
}
