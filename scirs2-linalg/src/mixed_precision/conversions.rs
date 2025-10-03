//! Type conversion utilities for mixed-precision computations
//!
//! This module provides fundamental conversion functions between different
//! numerical precisions, enabling seamless transitions between f32, f64,
//! and other numeric types in mixed-precision linear algebra operations.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{NumCast, ToPrimitive, Zero};
use std::fmt::Debug;

/// Convert an array to a different numeric type
///
/// # Arguments
/// * `arr` - Input array view
///
/// # Returns
/// * Array of the new type B
///
/// # Type Parameters
/// * `A` - Input numeric type
/// * `B` - Output numeric type
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::mixed_precision::conversions::convert;
///
/// let arr_f64 = array![1.0, 2.0, 3.0];
/// let arr_f32 = convert::<f64, f32>(&arr_f64.view());
///
/// assert_eq!(arr_f32[0], 1.0f32);
/// ```
#[allow(dead_code)]
pub fn convert<A, B>(arr: &ArrayView1<A>) -> Array1<B>
where
    A: Clone + Debug + ToPrimitive + Copy,
    B: Clone + Zero + NumCast + Debug,
{
    let mut result = Array1::<B>::zeros(arr.len());

    for (i, &val) in arr.iter().enumerate() {
        result[i] = B::from(val).unwrap_or_else(|| {
            // This should never happen with normal floating point conversions
            // between f32 and f64, but we need to handle the case
            B::zero()
        });
    }

    result
}

/// Convert a 2D array to a different numeric type
///
/// # Arguments
/// * `arr` - Input 2D array view
///
/// # Returns
/// * 2D Array of the new type B
///
/// # Type Parameters
/// * `A` - Input numeric type
/// * `B` - Output numeric type
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::mixed_precision::conversions::convert_2d;
///
/// let arr_f64 = array![[1.0, 2.0], [3.0, 4.0]];
/// let arr_f32 = convert_2d::<f64, f32>(&arr_f64.view());
///
/// assert_eq!(arr_f32[[0, 0]], 1.0f32);
/// ```
#[allow(dead_code)]
pub fn convert_2d<A, B>(arr: &ArrayView2<A>) -> Array2<B>
where
    A: Clone + Debug + ToPrimitive + Copy,
    B: Clone + Zero + NumCast + Debug,
{
    let shape = arr.shape();
    let mut result = Array2::<B>::zeros((shape[0], shape[1]));

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            result[[i, j]] = B::from(arr[[i, j]]).unwrap_or_else(|| B::zero());
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_convert_f64_to_f32() {
        let arr_f64 = array![1.0_f64, 2.5, 3.75, -1.25];
        let arr_f32 = convert::<f64, f32>(&arr_f64.view());

        assert_eq!(arr_f32.len(), 4);
        assert_relative_eq!(arr_f32[0], 1.0_f32, epsilon = 1e-6);
        assert_relative_eq!(arr_f32[1], 2.5_f32, epsilon = 1e-6);
        assert_relative_eq!(arr_f32[2], 3.75_f32, epsilon = 1e-6);
        assert_relative_eq!(arr_f32[3], -1.25_f32, epsilon = 1e-6);
    }

    #[test]
    fn test_convert_f32_to_f64() {
        let arr_f32 = array![1.0_f32, 2.5, 3.75, -1.25];
        let arr_f64 = convert::<f32, f64>(&arr_f32.view());

        assert_eq!(arr_f64.len(), 4);
        assert_relative_eq!(arr_f64[0], 1.0_f64, epsilon = 1e-15);
        assert_relative_eq!(arr_f64[1], 2.5_f64, epsilon = 1e-15);
        assert_relative_eq!(arr_f64[2], 3.75_f64, epsilon = 1e-15);
        assert_relative_eq!(arr_f64[3], -1.25_f64, epsilon = 1e-15);
    }

    #[test]
    fn test_convert_2d_f64_to_f32() {
        let arr_f64 = array![[1.0_f64, 2.0], [3.0, 4.0], [-1.0, -2.0]];
        let arr_f32 = convert_2d::<f64, f32>(&arr_f64.view());

        assert_eq!(arr_f32.shape(), &[3, 2]);
        assert_relative_eq!(arr_f32[[0, 0]], 1.0_f32, epsilon = 1e-6);
        assert_relative_eq!(arr_f32[[0, 1]], 2.0_f32, epsilon = 1e-6);
        assert_relative_eq!(arr_f32[[1, 0]], 3.0_f32, epsilon = 1e-6);
        assert_relative_eq!(arr_f32[[1, 1]], 4.0_f32, epsilon = 1e-6);
        assert_relative_eq!(arr_f32[[2, 0]], -1.0_f32, epsilon = 1e-6);
        assert_relative_eq!(arr_f32[[2, 1]], -2.0_f32, epsilon = 1e-6);
    }

    #[test]
    fn test_convert_2d_f32_to_f64() {
        let arr_f32 = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let arr_f64 = convert_2d::<f32, f64>(&arr_f32.view());

        assert_eq!(arr_f64.shape(), &[2, 2]);
        assert_relative_eq!(arr_f64[[0, 0]], 1.0_f64, epsilon = 1e-15);
        assert_relative_eq!(arr_f64[[0, 1]], 2.0_f64, epsilon = 1e-15);
        assert_relative_eq!(arr_f64[[1, 0]], 3.0_f64, epsilon = 1e-15);
        assert_relative_eq!(arr_f64[[1, 1]], 4.0_f64, epsilon = 1e-15);
    }

    #[test]
    fn test_convert_emptyarrays() {
        let arr_f64 = Array1::<f64>::zeros(0);
        let arr_f32 = convert::<f64, f32>(&arr_f64.view());
        assert_eq!(arr_f32.len(), 0);

        let arr_2d_f64 = Array2::<f64>::zeros((0, 0));
        let arr_2d_f32 = convert_2d::<f64, f32>(&arr_2d_f64.view());
        assert_eq!(arr_2d_f32.shape(), &[0, 0]);
    }

    #[test]
    fn test_convert_edge_values() {
        // Test very small and very large values
        let arr_f64 = array![1e-10_f64, 1e10_f64, std::f64::consts::PI];
        let arr_f32 = convert::<f64, f32>(&arr_f64.view());

        assert_relative_eq!(arr_f32[0], 1e-10_f32, epsilon = 1e-16);
        assert_relative_eq!(arr_f32[1], 1e10_f32, epsilon = 1e4); // Less precision for large numbers
        assert_relative_eq!(arr_f32[2], std::f32::consts::PI, epsilon = 1e-6);
    }
}
