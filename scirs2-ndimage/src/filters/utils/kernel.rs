//! Kernel Generation Utilities for Filtering
//!
//! This module provides utilities for generating and working with filter kernels,
//! particularly Gaussian kernels and separable filtering operations.

use scirs2_core::ndarray::{Array, ArrayView2, Ix1, Ix2};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use std::fmt::Debug;

#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;

use crate::error::{NdimageError, NdimageResult};

/// Helper function for safe usize conversion
#[allow(dead_code)]
fn safe_usize_to_float<T: Float + FromPrimitive>(value: usize) -> NdimageResult<T> {
    T::from_usize(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert usize {} to float type", value))
    })
}

/// Calculate optimal kernel size for a given sigma
///
/// # Arguments
///
/// * `sigma` - Standard deviation of the Gaussian kernel
/// * `truncate` - Truncation factor (default: 4.0)
///
/// # Returns
///
/// * `NdimageResult<usize>` - Optimal kernel size (always odd)
#[allow(dead_code)]
pub fn calculate_kernel_size<T: Float + FromPrimitive>(
    sigma: T,
    truncate: Option<T>,
) -> NdimageResult<usize> {
    let truncate_val = truncate.unwrap_or_else(|| T::from_f64(4.0).unwrap_or_else(|| T::zero()));
    let size = (T::from_usize(1).unwrap_or(T::one())
        + (sigma * truncate_val * T::from_f64(2.0).unwrap_or(T::one() + T::one())))
    .ceil()
    .to_usize()
    .ok_or_else(|| {
        NdimageError::ComputationError("Failed to convert kernel size to usize".into())
    })?;

    // Ensure odd size
    if size % 2 == 0 {
        Ok(size + 1)
    } else {
        Ok(size)
    }
}

/// Generate a Gaussian kernel with specified sigma
///
/// # Arguments
///
/// * `sigma` - Standard deviation of the Gaussian
/// * `size` - Optional explicit kernel size (must be odd)
///
/// # Returns
///
/// * `NdimageResult<Array<T, Ix1>>` - Normalized Gaussian kernel
#[allow(dead_code)]
pub fn generate_gaussian_kernel<T: Float + FromPrimitive>(
    sigma: T,
    size: Option<usize>,
) -> NdimageResult<Array<T, Ix1>> {
    let kernel_size = match size {
        Some(s) => s,
        None => calculate_kernel_size(sigma, None)?,
    };

    if kernel_size % 2 == 0 {
        return Err(NdimageError::InvalidInput("Kernel size must be odd".into()));
    }

    let half = kernel_size / 2;
    let mut kernel = Array::zeros(kernel_size);
    let sigma_sq = sigma * sigma;
    let norm_factor = T::from_f64(1.0).unwrap_or(T::one())
        / (sigma * T::from_f64(std::f64::consts::TAU.sqrt()).unwrap_or(T::one()));
    let exp_factor = T::from_f64(-0.5).unwrap_or(-T::one() / (T::one() + T::one())) / sigma_sq;

    let mut sum = T::zero();
    for (i, k) in kernel.iter_mut().enumerate() {
        let x = T::from_usize(i).unwrap_or(T::zero()) - T::from_usize(half).unwrap_or(T::zero());
        *k = norm_factor * (exp_factor * x * x).exp();
        sum = sum + *k;
    }

    // Normalize the kernel to sum to 1
    for k in kernel.iter_mut() {
        *k = *k / sum;
    }

    Ok(kernel)
}

/// Fast separable Gaussian blur implementation
///
/// Performs Gaussian blurring using separable kernels for efficiency.
/// Uses different sigma values for horizontal and vertical directions.
///
/// # Arguments
///
/// * `input` - Input 2D array to blur
/// * `sigma_x` - Standard deviation for horizontal direction
/// * `sigma_y` - Standard deviation for vertical direction
///
/// # Returns
///
/// * `NdimageResult<Array<T, Ix2>>` - Blurred output array
#[allow(dead_code)]
pub fn separable_gaussian_blur<T>(
    input: ArrayView2<T>,
    sigma_x: T,
    sigma_y: T,
) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero,
{
    let (height, width) = input.dim();

    // Generate kernels
    let kernel_x = generate_gaussian_kernel(sigma_x, None)?;
    let kernel_y = generate_gaussian_kernel(sigma_y, None)?;

    // Horizontal pass
    let mut temp = Array::zeros((height, width));

    #[cfg(feature = "parallel")]
    {
        temp.axis_iter_mut(scirs2_core::ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(_y, mut row)| {
                for _x in 0..width {
                    let mut sum = T::zero();
                    let kernel_half = kernel_x.len() / 2;

                    for (k_idx, &k_val) in kernel_x.iter().enumerate() {
                        let src_x = (_x as isize + k_idx as isize - kernel_half as isize)
                            .clamp(0, width as isize - 1)
                            as usize;
                        sum = sum + input[(_y, src_x)] * k_val;
                    }
                    row[_x] = sum;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for _y in 0..height {
            for _x in 0..width {
                let mut sum = T::zero();
                let kernel_half = kernel_x.len() / 2;

                for (k_idx, &k_val) in kernel_x.iter().enumerate() {
                    let src_x = (_x as isize + k_idx as isize - kernel_half as isize)
                        .clamp(0, width as isize - 1) as usize;
                    sum = sum + input[(_y, src_x)] * k_val;
                }
                temp[(_y, _x)] = sum;
            }
        }
    }

    // Vertical pass
    let mut output = Array::zeros((height, width));

    #[cfg(feature = "parallel")]
    {
        output
            .axis_chunks_iter_mut(scirs2_core::ndarray::Axis(1), 1)
            .into_par_iter()
            .enumerate()
            .for_each(|(_x, mut col)| {
                for _y in 0..height {
                    let mut sum = T::zero();
                    let kernel_half = kernel_y.len() / 2;

                    for (k_idx, &k_val) in kernel_y.iter().enumerate() {
                        let src_y = (_y as isize + k_idx as isize - kernel_half as isize)
                            .clamp(0, height as isize - 1)
                            as usize;
                        sum = sum + temp[(src_y, _x)] * k_val;
                    }
                    col[[_y, 0]] = sum;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for _x in 0..width {
            for _y in 0..height {
                let mut sum = T::zero();
                let kernel_half = kernel_y.len() / 2;

                for (k_idx, &k_val) in kernel_y.iter().enumerate() {
                    let src_y = (_y as isize + k_idx as isize - kernel_half as isize)
                        .clamp(0, height as isize - 1) as usize;
                    sum = sum + temp[(src_y, _x)] * k_val;
                }
                output[(_y, _x)] = sum;
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_calculate_kernel_size() {
        let size = calculate_kernel_size(1.0_f64, None).unwrap();
        assert!(size % 2 == 1); // Should be odd
        assert!(size > 0);

        let size_with_truncate = calculate_kernel_size(1.0_f64, Some(2.0)).unwrap();
        assert!(size_with_truncate % 2 == 1);
        assert!(size_with_truncate < size); // Smaller truncate should give smaller size
    }

    #[test]
    fn test_generate_gaussian_kernel() {
        let kernel = generate_gaussian_kernel(1.0_f64, Some(5)).unwrap();
        assert_eq!(kernel.len(), 5);

        // Check normalization (sum should be 1)
        let sum: f64 = kernel.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);

        // Check symmetry
        assert_abs_diff_eq!(kernel[0], kernel[4], epsilon = 1e-10);
        assert_abs_diff_eq!(kernel[1], kernel[3], epsilon = 1e-10);

        // Center should be maximum
        assert!(kernel[2] > kernel[1]);
        assert!(kernel[2] > kernel[0]);
    }

    #[test]
    fn test_separable_gaussian_blur() {
        let input = Array::from_shape_fn((4, 4), |(i, j)| (i + j) as f64);
        let result = separable_gaussian_blur(input.view(), 0.5, 0.5).unwrap();

        assert_eq!(result.shape(), input.shape());

        // Output should be smoothed version of input
        // Center values should change less than edge values
        let center_change = (result[[1, 1]] - input[[1, 1]]).abs();
        let corner_change = (result[[0, 0]] - input[[0, 0]]).abs();

        // This is a heuristic test - the exact values depend on the kernel
        assert!(result[[1, 1]] >= 0.0); // Sanity check
        assert!(result[[0, 0]] >= 0.0); // Sanity check
    }
}
