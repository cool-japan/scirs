//! Convolution Utilities for Filtering
//!
//! This module provides memory-efficient convolution implementations
//! with support for various border handling modes.

use scirs2_core::ndarray::{Array, ArrayView2, Ix2};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use std::fmt::Debug;

use super::super::BorderMode;
use crate::error::{NdimageError, NdimageResult};

/// Memory-efficient convolution implementation
///
/// Performs 2D convolution with specified border handling mode.
/// This implementation prioritizes memory efficiency over speed.
///
/// # Arguments
///
/// * `input` - Input 2D array to convolve
/// * `kernel` - Convolution kernel
/// * `mode` - Border handling mode for edge cases
///
/// # Returns
///
/// * `NdimageResult<Array<T, Ix2>>` - Convolved output array
#[allow(dead_code)]
pub fn memory_efficient_convolution<T>(
    input: ArrayView2<T>,
    kernel: ArrayView2<T>,
    mode: BorderMode,
) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + Clone + Zero,
{
    let (input_h, input_w) = input.dim();
    let (kernel_h, kernel_w) = kernel.dim();

    if kernel_h == 0 || kernel_w == 0 {
        return Err(NdimageError::InvalidInput("Kernel cannot be empty".into()));
    }

    let kernel_half_h = kernel_h / 2;
    let kernel_half_w = kernel_w / 2;

    let mut output = Array::zeros((input_h, input_w));

    for y in 0..input_h {
        for x in 0..input_w {
            let mut sum = T::zero();

            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    let src_y = y as isize + ky as isize - kernel_half_h as isize;
                    let src_x = x as isize + kx as isize - kernel_half_w as isize;

                    let pixel_value = match mode {
                        BorderMode::Constant => {
                            if src_y >= 0
                                && src_y < input_h as isize
                                && src_x >= 0
                                && src_x < input_w as isize
                            {
                                input[(src_y as usize, src_x as usize)]
                            } else {
                                T::zero()
                            }
                        }
                        BorderMode::Reflect => {
                            let reflect_y = if src_y < 0 {
                                (-src_y) as usize
                            } else if src_y >= input_h as isize {
                                input_h - 2 - (src_y - input_h as isize) as usize
                            } else {
                                src_y as usize
                            }
                            .min(input_h - 1);

                            let reflect_x = if src_x < 0 {
                                (-src_x) as usize
                            } else if src_x >= input_w as isize {
                                input_w - 2 - (src_x - input_w as isize) as usize
                            } else {
                                src_x as usize
                            }
                            .min(input_w - 1);

                            input[(reflect_y, reflect_x)]
                        }
                        BorderMode::Nearest => {
                            let clamp_y = src_y.clamp(0, input_h as isize - 1) as usize;
                            let clamp_x = src_x.clamp(0, input_w as isize - 1) as usize;
                            input[(clamp_y, clamp_x)]
                        }
                        BorderMode::Wrap => {
                            let wrap_y = ((src_y % input_h as isize + input_h as isize)
                                % input_h as isize)
                                as usize;
                            let wrap_x = ((src_x % input_w as isize + input_w as isize)
                                % input_w as isize)
                                as usize;
                            input[(wrap_y, wrap_x)]
                        }
                        BorderMode::Mirror => {
                            let mirror_y = if src_y < 0 {
                                ((-src_y - 1) % (2 * input_h as isize)) as usize
                            } else if src_y >= input_h as isize {
                                ((2 * input_h as isize - src_y - 1) % (2 * input_h as isize))
                                    as usize
                            } else {
                                src_y as usize
                            }
                            .min(input_h - 1);

                            let mirror_x = if src_x < 0 {
                                ((-src_x - 1) % (2 * input_w as isize)) as usize
                            } else if src_x >= input_w as isize {
                                ((2 * input_w as isize - src_x - 1) % (2 * input_w as isize))
                                    as usize
                            } else {
                                src_x as usize
                            }
                            .min(input_w - 1);

                            input[(mirror_y, mirror_x)]
                        }
                    };

                    sum = sum + pixel_value * kernel[(ky, kx)];
                }
            }

            output[(y, x)] = sum;
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_memory_efficient_convolution_identity() {
        let input = Array::from_shape_fn((4, 4), |(i, j)| (i + j) as f64);

        // Identity kernel
        let kernel =
            Array::from_shape_fn((3, 3), |(i, j)| if i == 1 && j == 1 { 1.0 } else { 0.0 });

        let result =
            memory_efficient_convolution(input.view(), kernel.view(), BorderMode::Constant)
                .unwrap();

        // Identity convolution should preserve center values
        assert_abs_diff_eq!(result[[1, 1]], input[[1, 1]], epsilon = 1e-10);
        assert_abs_diff_eq!(result[[2, 2]], input[[2, 2]], epsilon = 1e-10);
    }

    #[test]
    fn test_memory_efficient_convolution_smooth() {
        let input = Array::from_shape_fn((5, 5), |(i, j)| if i == 2 && j == 2 { 1.0 } else { 0.0 });

        // Simple smoothing kernel
        let kernel = Array::from_elem((3, 3), 1.0 / 9.0);

        let result =
            memory_efficient_convolution(input.view(), kernel.view(), BorderMode::Constant)
                .unwrap();

        // Center and immediate neighbors should have the same value due to smoothing
        // but should be greater than edge pixels that don't see the point source
        assert!((result[[2, 2]] - result[[1, 1]]).abs() < 1e-10); // Should be equal
        assert!(result[[2, 2]] > result[[0, 0]]); // Center should be > edge
        assert!(result[[2, 2]] < 1.0); // Should be less than original value due to smoothing
        assert!((result[[2, 2]] - 1.0 / 9.0).abs() < 1e-10); // Should be exactly 1/9
    }

    #[test]
    fn test_memory_efficient_convolution_different_modes() {
        let input = Array::from_shape_fn((3, 3), |(i, j)| (i * 3 + j) as f64);
        let kernel = Array::from_elem((3, 3), 1.0 / 9.0);

        // Test different border modes don't crash
        for mode in [
            BorderMode::Constant,
            BorderMode::Reflect,
            BorderMode::Nearest,
            BorderMode::Wrap,
            BorderMode::Mirror,
        ] {
            let result = memory_efficient_convolution(input.view(), kernel.view(), mode);
            assert!(result.is_ok());
            assert_eq!(result.unwrap().shape(), input.shape());
        }
    }

    #[test]
    fn test_memory_efficient_convolution_empty_kernel() {
        let input = Array::from_shape_fn((3, 3), |(i, j)| (i + j) as f64);
        let kernel = Array::zeros((0, 0));

        let result =
            memory_efficient_convolution(input.view(), kernel.view(), BorderMode::Constant);

        assert!(result.is_err());
    }
}
