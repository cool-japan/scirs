//! Data preprocessing operations with SIMD acceleration
//!
//! This module provides optimized implementations of common data preprocessing
//! operations used in machine learning and scientific computing.

use ndarray::{Array1, ArrayView1};

// Import from sibling modules
use super::arithmetic::{simd_scalar_mul_f32, simd_scalar_mul_f64};
use super::norms::{simd_norm_l2_f32, simd_norm_l2_f64};
use super::reductions::{simd_mean_f32, simd_mean_f64, simd_std_f32, simd_std_f64};

/// SIMD-accelerated L2 normalization for f32 arrays
///
/// Normalizes the input array to unit L2 norm (Euclidean length = 1).
/// If the norm is zero or NaN, returns a zero array.
///
/// # Arguments
///
/// * `input` - Input array to normalize
///
/// # Returns
///
/// Normalized array where ||output||_2 = 1 (or zero array if input norm is zero)
///
/// # Performance
///
/// - f32 (100K elements): 0.44x-0.60x vs NumPy (limited by two-pass algorithm)
/// - f64 (100K elements): 0.39x vs NumPy
///
/// Note: Performance is limited by the two-pass nature of the algorithm
/// (calculate norm, then divide). NumPy uses highly optimized BLAS routines.
#[allow(dead_code)]
pub fn simd_normalize_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    let norm = simd_norm_l2_f32(input);
    if norm == 0.0 || norm.is_nan() {
        return Array1::zeros(input.len());
    }

    let inv_norm = 1.0 / norm;
    simd_scalar_mul_f32(input, inv_norm)
}

/// SIMD-accelerated L2 normalization for f64 arrays
///
/// Normalizes the input array to unit L2 norm (Euclidean length = 1).
/// If the norm is zero or NaN, returns a zero array.
///
/// # Arguments
///
/// * `input` - Input array to normalize
///
/// # Returns
///
/// Normalized array where ||output||_2 = 1 (or zero array if input norm is zero)
#[allow(dead_code)]
pub fn simd_normalize_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    let norm = simd_norm_l2_f64(input);
    if norm == 0.0 || norm.is_nan() {
        return Array1::zeros(input.len());
    }

    let inv_norm = 1.0 / norm;
    simd_scalar_mul_f64(input, inv_norm)
}

/// SIMD-accelerated standardization for f32 arrays
///
/// Returns (x - mean) / std for each element (z-score normalization).
/// If std is zero or NaN, returns a zero array.
///
/// # Arguments
///
/// * `input` - Input array to standardize
///
/// # Returns
///
/// Standardized array with mean=0 and std=1 (or zero array if input std is zero)
///
/// # Performance
///
/// - f32 (100K elements): **1.22x faster than NumPy** ✓
/// - f64 (100K elements): 0.88x-0.94x vs NumPy
///
/// This implementation uses direct SIMD pointer writes for optimal performance.
#[allow(dead_code)]
pub fn simd_standardize_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    let mean = simd_mean_f32(input);
    let std = simd_std_f32(input);

    if std == 0.0 || std.is_nan() {
        return Array1::zeros(input.len());
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        let inv_std = 1.0 / std;
        return input.mapv(|x| (x - mean) * inv_std);
    }

    let len = input.len();
    let mut result = Array1::zeros(len);
    let input_slice = input.as_slice().expect("contiguity checked above");
    let result_slice: &mut [f32] = result.as_slice_mut().expect("Operation failed");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let mean_vec = _mm256_set1_ps(mean);
                let inv_std = 1.0 / std;
                let inv_std_vec = _mm256_set1_ps(inv_std);
                let mut i = 0;

                while i + 8 <= len {
                    let input_vec = _mm256_loadu_ps(input_slice.as_ptr().add(i));

                    let centered = _mm256_sub_ps(input_vec, mean_vec);
                    let scaled = _mm256_mul_ps(centered, inv_std_vec);

                    _mm256_storeu_ps(result_slice.as_mut_ptr().add(i), scaled);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = (input_slice[j] - mean) * inv_std;
                }

                return result;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;

                let mean_vec = vdupq_n_f32(mean);
                let inv_std = 1.0 / std;
                let inv_std_vec = vdupq_n_f32(inv_std);
                let mut i = 0;

                while i + 4 <= len {
                    let input_vec = vld1q_f32(input_slice.as_ptr().add(i));

                    let centered = vsubq_f32(input_vec, mean_vec);
                    let scaled = vmulq_f32(centered, inv_std_vec);

                    vst1q_f32(result_slice.as_mut_ptr().add(i), scaled);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = (input_slice[j] - mean) * inv_std;
                }

                return result;
            }
        }
    }

    // Scalar fallback
    let inv_std = 1.0 / std;
    for (i, val) in input.iter().enumerate() {
        result_slice[i] = (val - mean) * inv_std;
    }
    result
}

/// SIMD-accelerated standardization for f64 arrays
///
/// Returns (x - mean) / std for each element (z-score normalization).
/// If std is zero or NaN, returns a zero array.
///
/// # Arguments
///
/// * `input` - Input array to standardize
///
/// # Returns
///
/// Standardized array with mean=0 and std=1 (or zero array if input std is zero)
#[allow(dead_code)]
pub fn simd_standardize_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    let mean = simd_mean_f64(input);
    let std = simd_std_f64(input);

    if std == 0.0 || std.is_nan() {
        return Array1::zeros(input.len());
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        let inv_std = 1.0 / std;
        return input.mapv(|x| (x - mean) * inv_std);
    }

    let len = input.len();
    let mut result = Array1::zeros(len);
    let input_slice = input.as_slice().expect("contiguity checked above");
    let result_slice: &mut [f64] = result.as_slice_mut().expect("Operation failed");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let mean_vec = _mm256_set1_pd(mean);
                let inv_std = 1.0 / std;
                let inv_std_vec = _mm256_set1_pd(inv_std);
                let mut i = 0;

                while i + 4 <= len {
                    let input_vec = _mm256_loadu_pd(input_slice.as_ptr().add(i));

                    let centered = _mm256_sub_pd(input_vec, mean_vec);
                    let scaled = _mm256_mul_pd(centered, inv_std_vec);

                    _mm256_storeu_pd(result_slice.as_mut_ptr().add(i), scaled);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = (input_slice[j] - mean) * inv_std;
                }

                return result;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;

                let mean_vec = vdupq_n_f64(mean);
                let inv_std = 1.0 / std;
                let inv_std_vec = vdupq_n_f64(inv_std);
                let mut i = 0;

                while i + 2 <= len {
                    let input_vec = vld1q_f64(input_slice.as_ptr().add(i));

                    let centered = vsubq_f64(input_vec, mean_vec);
                    let scaled = vmulq_f64(centered, inv_std_vec);

                    vst1q_f64(result_slice.as_mut_ptr().add(i), scaled);
                    i += 2;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = (input_slice[j] - mean) * inv_std;
                }

                return result;
            }
        }
    }

    // Scalar fallback
    let inv_std = 1.0 / std;
    for (i, val) in input.iter().enumerate() {
        result_slice[i] = (val - mean) * inv_std;
    }
    result
}
