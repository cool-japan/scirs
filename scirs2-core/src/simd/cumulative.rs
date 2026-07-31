//! Cumulative operations with SIMD acceleration
//!
//! This module provides optimized implementations of cumulative operations
//! including cumulative sum, cumulative product, and differences.

use ndarray::{Array1, ArrayView1};

/// SIMD-accelerated cumulative sum for f32 arrays
///
/// Computes the running sum where each element is the sum of all preceding elements
/// (including itself).
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// * Array of cumulative sums
#[allow(dead_code)]
pub fn simd_cumsum_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    let len = input.len();
    let mut result = Vec::with_capacity(len);

    // Cumulative operations are inherently sequential
    // SIMD can help with prefix sums within small blocks, but the overall
    // algorithm remains largely sequential
    let mut cumsum = 0.0f32;
    for &val in input.iter() {
        cumsum += val;
        result.push(cumsum);
    }

    Array1::from_vec(result)
}

/// SIMD-accelerated cumulative sum for f64 arrays
#[allow(dead_code)]
pub fn simd_cumsum_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    let len = input.len();
    let mut result = Vec::with_capacity(len);

    let mut cumsum = 0.0f64;
    for &val in input.iter() {
        cumsum += val;
        result.push(cumsum);
    }

    Array1::from_vec(result)
}

/// SIMD-accelerated cumulative product for f32 arrays
///
/// Computes the running product where each element is the product of all preceding elements
/// (including itself).
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// Array of cumulative products
#[allow(dead_code)]
pub fn simd_cumprod_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    let len = input.len();
    let mut result = Vec::with_capacity(len);

    let mut cumprod = 1.0f32;
    for &val in input.iter() {
        cumprod *= val;
        result.push(cumprod);
    }

    Array1::from_vec(result)
}

/// SIMD-accelerated cumulative product for f64 arrays
#[allow(dead_code)]
pub fn simd_cumprod_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    let len = input.len();
    let mut result = Vec::with_capacity(len);

    let mut cumprod = 1.0f64;
    for &val in input.iter() {
        cumprod *= val;
        result.push(cumprod);
    }

    Array1::from_vec(result)
}

/// SIMD-accelerated first-order difference for f32 arrays
///
/// Computes `diff[i]` = `input[i+1]` - `input[i]` for i in 0..len-1.
/// Returns an array of length len-1.
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// Array of first-order differences
#[allow(dead_code)]
pub fn simd_diff_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    if input.len() <= 1 {
        return Array1::zeros(0);
    }

    let len = input.len() - 1;

    // Fall back to scalar for non-contiguous arrays (e.g. column slices).
    // Correctness over speed here: a plain sequential scan, not vectorized.
    if input.as_slice().is_none() {
        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            result.push(input[i + 1] - input[i]);
        }
        return Array1::from_vec(result);
    }

    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let mut i = 0;
                while i + 8 <= len {
                    let curr_slice = &input.as_slice().expect("contiguity checked above")[i..i + 8];
                    let next_slice =
                        &input.as_slice().expect("contiguity checked above")[i + 1..i + 9];

                    let curr_vec = _mm256_loadu_ps(curr_slice.as_ptr());
                    let next_vec = _mm256_loadu_ps(next_slice.as_ptr());
                    let diff_vec = _mm256_sub_ps(next_vec, curr_vec);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), diff_vec);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(input[j + 1] - input[j]);
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;

                let mut i = 0;
                while i + 4 <= len {
                    let curr_slice = &input.as_slice().expect("contiguity checked above")[i..i + 4];
                    let next_slice =
                        &input.as_slice().expect("contiguity checked above")[i + 1..i + 5];

                    let curr_vec = vld1q_f32(curr_slice.as_ptr());
                    let next_vec = vld1q_f32(next_slice.as_ptr());
                    let diff_vec = vsubq_f32(next_vec, curr_vec);

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), diff_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    result.push(input[j + 1] - input[j]);
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for i in 0..len {
        result.push(input[i + 1] - input[i]);
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated first-order difference for f64 arrays
#[allow(dead_code)]
pub fn simd_diff_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    if input.len() <= 1 {
        return Array1::zeros(0);
    }

    let len = input.len() - 1;

    // Fall back to scalar for non-contiguous arrays (e.g. column slices).
    // Correctness over speed here: a plain sequential scan, not vectorized.
    if input.as_slice().is_none() {
        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            result.push(input[i + 1] - input[i]);
        }
        return Array1::from_vec(result);
    }

    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let mut i = 0;
                while i + 4 <= len {
                    let curr_slice = &input.as_slice().expect("contiguity checked above")[i..i + 4];
                    let next_slice =
                        &input.as_slice().expect("contiguity checked above")[i + 1..i + 5];

                    let curr_vec = _mm256_loadu_pd(curr_slice.as_ptr());
                    let next_vec = _mm256_loadu_pd(next_slice.as_ptr());
                    let diff_vec = _mm256_sub_pd(next_vec, curr_vec);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), diff_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    result.push(input[j + 1] - input[j]);
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;

                let mut i = 0;
                while i + 2 <= len {
                    let curr_slice = &input.as_slice().expect("contiguity checked above")[i..i + 2];
                    let next_slice =
                        &input.as_slice().expect("contiguity checked above")[i + 1..i + 3];

                    let curr_vec = vld1q_f64(curr_slice.as_ptr());
                    let next_vec = vld1q_f64(next_slice.as_ptr());
                    let diff_vec = vsubq_f64(next_vec, curr_vec);

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), diff_vec);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                for j in i..len {
                    result.push(input[j + 1] - input[j]);
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for i in 0..len {
        result.push(input[i + 1] - input[i]);
    }
    Array1::from_vec(result)
}
