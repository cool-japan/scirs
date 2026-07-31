//! Arithmetic operations with SIMD acceleration
//!
//! This module provides optimized implementations of arithmetic operations.
//! Currently includes scalar multiplication, with more operations to be added.

use ndarray::{Array1, ArrayView1};

pub fn simd_scalar_mul_f32(a: &ArrayView1<f32>, scalar: f32) -> Array1<f32> {
    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() {
        return a.mapv(|x| x * scalar);
    }

    let len = a.len();
    let mut result = Array1::zeros(len);
    let a_slice = a.as_slice().expect("contiguity checked above");
    let result_slice: &mut [f32] = result.as_slice_mut().expect("Operation failed");

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let scalar_vec = _mm256_set1_ps(scalar);
                let mut i = 0;

                // Process 8 f32s at a time with AVX2 - direct pointer writes
                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a_slice.as_ptr().add(i));
                    let result_vec = _mm256_mul_ps(a_vec, scalar_vec);
                    _mm256_storeu_ps(result_slice.as_mut_ptr().add(i), result_vec);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = a_slice[j] * scalar;
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let scalar_vec = _mm_set1_ps(scalar);
                let mut i = 0;

                // Process 4 f32s at a time with SSE
                while i + 4 <= len {
                    let a_vec = _mm_loadu_ps(a_slice.as_ptr().add(i));
                    let result_vec = _mm_mul_ps(a_vec, scalar_vec);
                    _mm_storeu_ps(result_slice.as_mut_ptr().add(i), result_vec);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = a_slice[j] * scalar;
                }

                return result;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let scalar_vec = vdupq_n_f32(scalar);
                let mut i = 0;

                // Process 4 f32s at a time with NEON
                while i + 4 <= len {
                    let a_vec = vld1q_f32(a_slice.as_ptr().add(i));
                    let result_vec = vmulq_f32(a_vec, scalar_vec);
                    vst1q_f32(result_slice.as_mut_ptr().add(i), result_vec);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = a_slice[j] * scalar;
                }

                return result;
            }
        }
    }

    // Fallback to scalar implementation
    for i in 0..len {
        result_slice[i] = a_slice[i] * scalar;
    }

    result
}

/// Apply scalar multiplication to an f64 array using unified SIMD operations
#[allow(dead_code)]
pub fn simd_scalar_mul_f64(a: &ArrayView1<f64>, scalar: f64) -> Array1<f64> {
    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() {
        return a.mapv(|x| x * scalar);
    }

    let len = a.len();
    let mut result = Array1::zeros(len);
    let a_slice = a.as_slice().expect("contiguity checked above");
    let result_slice: &mut [f64] = result.as_slice_mut().expect("Operation failed");

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let scalar_vec = _mm256_set1_pd(scalar);
                let mut i = 0;

                // Process 4 f64s at a time with AVX2 - direct pointer writes
                while i + 4 <= len {
                    let a_vec = _mm256_loadu_pd(a_slice.as_ptr().add(i));
                    let result_vec = _mm256_mul_pd(a_vec, scalar_vec);
                    _mm256_storeu_pd(result_slice.as_mut_ptr().add(i), result_vec);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = a_slice[j] * scalar;
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let scalar_vec = _mm_set1_pd(scalar);
                let mut i = 0;

                // Process 2 f64s at a time with SSE2
                while i + 2 <= len {
                    let a_vec = _mm_loadu_pd(a_slice.as_ptr().add(i));
                    let result_vec = _mm_mul_pd(a_vec, scalar_vec);
                    _mm_storeu_pd(result_slice.as_mut_ptr().add(i), result_vec);
                    i += 2;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = a_slice[j] * scalar;
                }

                return result;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let scalar_vec = vdupq_n_f64(scalar);
                let mut i = 0;

                // Process 2 f64s at a time with NEON
                while i + 2 <= len {
                    let a_vec = vld1q_f64(a_slice.as_ptr().add(i));
                    let result_vec = vmulq_f64(a_vec, scalar_vec);
                    vst1q_f64(result_slice.as_mut_ptr().add(i), result_vec);
                    i += 2;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = a_slice[j] * scalar;
                }

                return result;
            }
        }
    }

    // Fallback to scalar implementation
    for i in 0..len {
        result_slice[i] = a_slice[i] * scalar;
    }

    result
}

/// SIMD accelerated linspace function for f32 values
///
/// Creates a linearly spaced array between start and end (inclusive)
/// using SIMD instructions for better performance.
///
/// # Arguments
///
/// * `start` - Start value
/// * `end` - End value (inclusive)
/// * `num` - Number of points
///
/// # Returns
///
/// * Array of linearly spaced values
#[allow(dead_code)]
pub fn linspace_f32(startval: f32, end: f32, num: usize) -> Array1<f32> {
    if num < 2 {
        return Array1::from_vec(vec![startval]);
    }

    let mut result = Array1::zeros(num);
    let step = (end - startval) / (num as f32 - 1.0);

    // Use scalar implementation for now - could be optimized with SIMD
    for (i, elem) in result.iter_mut().enumerate() {
        *elem = startval + step * i as f32;
    }

    // Make sure the last value is exactly end to avoid floating point precision issues
    if let Some(last) = result.last_mut() {
        *last = end;
    }

    result
}

/// SIMD accelerated linspace function for f64 values
#[allow(dead_code)]
pub fn linspace_f64(startval: f64, end: f64, num: usize) -> Array1<f64> {
    if num < 2 {
        return Array1::from_vec(vec![startval]);
    }

    let mut result = Array1::zeros(num);
    let step = (end - startval) / (num as f64 - 1.0);

    // Use scalar implementation for now - could be optimized with SIMD
    for (i, elem) in result.iter_mut().enumerate() {
        *elem = startval + step * i as f64;
    }

    // Make sure the last value is exactly end to avoid floating point precision issues
    if let Some(last) = result.last_mut() {
        *last = end;
    }

    result
}
