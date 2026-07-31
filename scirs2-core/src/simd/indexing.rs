//! Indexing operations with SIMD acceleration
//!
//! This module provides optimized implementations of indexing operations
//! including argmin (index of minimum), argmax (index of maximum), and clip.

use ndarray::{Array1, ArrayView1};

/// SIMD-accelerated argmin for f32 arrays
///
/// Returns the index of the minimum value in the array.
/// Returns None for empty arrays.
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// * Index of the minimum value, or None if array is empty
#[allow(dead_code)]
pub fn simd_argmin_f32(input: &ArrayView1<f32>) -> Option<usize> {
    let len = input.len();
    if len == 0 {
        return None;
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        let mut min_val = input[0];
        let mut min_idx = 0;
        for (i, &val) in input.iter().enumerate().skip(1) {
            if val < min_val {
                min_val = val;
                min_idx = i;
            }
        }
        return Some(min_idx);
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") && len >= 8 {
            unsafe {
                let mut min_val = f32::INFINITY;
                let mut min_idx = 0usize;
                let mut i = 0;

                // Process 8 f32s at a time
                while i + 8 <= len {
                    let slice = &input.as_slice().expect("contiguity checked above")[i..i + 8];
                    let vec = _mm256_loadu_ps(slice.as_ptr());

                    // Extract and compare
                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), vec);

                    for (j, &val) in temp.iter().enumerate() {
                        if val < min_val {
                            min_val = val;
                            min_idx = i + j;
                        }
                    }
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    if input[j] < min_val {
                        min_val = input[j];
                        min_idx = j;
                    }
                }

                return Some(min_idx);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") && len >= 4 {
            unsafe {
                let mut min_val = f32::INFINITY;
                let mut min_idx = 0usize;
                let mut i = 0;

                while i + 4 <= len {
                    let slice = &input.as_slice().expect("contiguity checked above")[i..i + 4];
                    let vec = vld1q_f32(slice.as_ptr());

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), vec);

                    for (j, &val) in temp.iter().enumerate() {
                        if val < min_val {
                            min_val = val;
                            min_idx = i + j;
                        }
                    }
                    i += 4;
                }

                for j in i..len {
                    if input[j] < min_val {
                        min_val = input[j];
                        min_idx = j;
                    }
                }

                return Some(min_idx);
            }
        }
    }

    // Scalar fallback
    let mut min_val = input[0];
    let mut min_idx = 0;
    for (i, &val) in input.iter().enumerate().skip(1) {
        if val < min_val {
            min_val = val;
            min_idx = i;
        }
    }
    Some(min_idx)
}

/// SIMD-accelerated argmin for f64 arrays
#[allow(dead_code)]
pub fn simd_argmin_f64(input: &ArrayView1<f64>) -> Option<usize> {
    let len = input.len();
    if len == 0 {
        return None;
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        let mut min_val = input[0];
        let mut min_idx = 0;
        for (i, &val) in input.iter().enumerate().skip(1) {
            if val < min_val {
                min_val = val;
                min_idx = i;
            }
        }
        return Some(min_idx);
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") && len >= 4 {
            unsafe {
                let mut min_val = f64::INFINITY;
                let mut min_idx = 0usize;
                let mut i = 0;

                while i + 4 <= len {
                    let slice = &input.as_slice().expect("contiguity checked above")[i..i + 4];
                    let vec = _mm256_loadu_pd(slice.as_ptr());

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), vec);

                    for (j, &val) in temp.iter().enumerate() {
                        if val < min_val {
                            min_val = val;
                            min_idx = i + j;
                        }
                    }
                    i += 4;
                }

                for j in i..len {
                    if input[j] < min_val {
                        min_val = input[j];
                        min_idx = j;
                    }
                }

                return Some(min_idx);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") && len >= 2 {
            unsafe {
                let mut min_val = f64::INFINITY;
                let mut min_idx = 0usize;
                let mut i = 0;

                while i + 2 <= len {
                    let slice = &input.as_slice().expect("contiguity checked above")[i..i + 2];
                    let vec = vld1q_f64(slice.as_ptr());

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), vec);

                    for (j, &val) in temp.iter().enumerate() {
                        if val < min_val {
                            min_val = val;
                            min_idx = i + j;
                        }
                    }
                    i += 2;
                }

                for j in i..len {
                    if input[j] < min_val {
                        min_val = input[j];
                        min_idx = j;
                    }
                }

                return Some(min_idx);
            }
        }
    }

    // Scalar fallback
    let mut min_val = input[0];
    let mut min_idx = 0;
    for (i, &val) in input.iter().enumerate().skip(1) {
        if val < min_val {
            min_val = val;
            min_idx = i;
        }
    }
    Some(min_idx)
}

/// SIMD-accelerated argmax for f32 arrays
#[allow(dead_code)]
pub fn simd_argmax_f32(input: &ArrayView1<f32>) -> Option<usize> {
    let len = input.len();
    if len == 0 {
        return None;
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        let mut max_val = input[0];
        let mut max_idx = 0;
        for (i, &val) in input.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        return Some(max_idx);
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") && len >= 8 {
            unsafe {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_idx = 0usize;
                let mut i = 0;

                while i + 8 <= len {
                    let slice = &input.as_slice().expect("contiguity checked above")[i..i + 8];
                    let vec = _mm256_loadu_ps(slice.as_ptr());

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), vec);

                    for (j, &val) in temp.iter().enumerate() {
                        if val > max_val {
                            max_val = val;
                            max_idx = i + j;
                        }
                    }
                    i += 8;
                }

                for j in i..len {
                    if input[j] > max_val {
                        max_val = input[j];
                        max_idx = j;
                    }
                }

                return Some(max_idx);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") && len >= 4 {
            unsafe {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_idx = 0usize;
                let mut i = 0;

                while i + 4 <= len {
                    let slice = &input.as_slice().expect("contiguity checked above")[i..i + 4];
                    let vec = vld1q_f32(slice.as_ptr());

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), vec);

                    for (j, &val) in temp.iter().enumerate() {
                        if val > max_val {
                            max_val = val;
                            max_idx = i + j;
                        }
                    }
                    i += 4;
                }

                for j in i..len {
                    if input[j] > max_val {
                        max_val = input[j];
                        max_idx = j;
                    }
                }

                return Some(max_idx);
            }
        }
    }

    // Scalar fallback
    let mut max_val = input[0];
    let mut max_idx = 0;
    for (i, &val) in input.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    Some(max_idx)
}

/// SIMD-accelerated argmax for f64 arrays
#[allow(dead_code)]
pub fn simd_argmax_f64(input: &ArrayView1<f64>) -> Option<usize> {
    let len = input.len();
    if len == 0 {
        return None;
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        let mut max_val = input[0];
        let mut max_idx = 0;
        for (i, &val) in input.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        return Some(max_idx);
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") && len >= 4 {
            unsafe {
                let mut max_val = f64::NEG_INFINITY;
                let mut max_idx = 0usize;
                let mut i = 0;

                while i + 4 <= len {
                    let slice = &input.as_slice().expect("contiguity checked above")[i..i + 4];
                    let vec = _mm256_loadu_pd(slice.as_ptr());

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), vec);

                    for (j, &val) in temp.iter().enumerate() {
                        if val > max_val {
                            max_val = val;
                            max_idx = i + j;
                        }
                    }
                    i += 4;
                }

                for j in i..len {
                    if input[j] > max_val {
                        max_val = input[j];
                        max_idx = j;
                    }
                }

                return Some(max_idx);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") && len >= 2 {
            unsafe {
                let mut max_val = f64::NEG_INFINITY;
                let mut max_idx = 0usize;
                let mut i = 0;

                while i + 2 <= len {
                    let slice = &input.as_slice().expect("contiguity checked above")[i..i + 2];
                    let vec = vld1q_f64(slice.as_ptr());

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), vec);

                    for (j, &val) in temp.iter().enumerate() {
                        if val > max_val {
                            max_val = val;
                            max_idx = i + j;
                        }
                    }
                    i += 2;
                }

                for j in i..len {
                    if input[j] > max_val {
                        max_val = input[j];
                        max_idx = j;
                    }
                }

                return Some(max_idx);
            }
        }
    }

    // Scalar fallback
    let mut max_val = input[0];
    let mut max_idx = 0;
    for (i, &val) in input.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    Some(max_idx)
}

// ==================== Clip/Clamp Operations ====================

/// SIMD-accelerated clip for f32 arrays
///
/// Clips all values to be within [min_val, max_val]
#[allow(dead_code)]
pub fn simd_clip_f32(input: &ArrayView1<f32>, min_val: f32, max_val: f32) -> Array1<f32> {
    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x.max(min_val).min(max_val));
    }

    let len = input.len();
    let mut result = Array1::zeros(len);
    let input_slice = input.as_slice().expect("contiguity checked above");
    let result_slice: &mut [f32] = result.as_slice_mut().expect("Operation failed");

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let min_vec = _mm256_set1_ps(min_val);
                let max_vec = _mm256_set1_ps(max_val);
                let mut i = 0;

                // Process 8 f32s at a time with AVX2 - direct pointer writes
                while i + 8 <= len {
                    let vec = _mm256_loadu_ps(input_slice.as_ptr().add(i));
                    let clipped = _mm256_max_ps(min_vec, _mm256_min_ps(max_vec, vec));
                    _mm256_storeu_ps(result_slice.as_mut_ptr().add(i), clipped);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = input_slice[j].max(min_val).min(max_val);
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
                let min_vec = vdupq_n_f32(min_val);
                let max_vec = vdupq_n_f32(max_val);
                let mut i = 0;

                // Process 4 f32s at a time with NEON - direct pointer writes
                while i + 4 <= len {
                    let vec = vld1q_f32(input_slice.as_ptr().add(i));
                    let clipped = vmaxq_f32(min_vec, vminq_f32(max_vec, vec));
                    vst1q_f32(result_slice.as_mut_ptr().add(i), clipped);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = input_slice[j].max(min_val).min(max_val);
                }

                return result;
            }
        }
    }

    // Scalar fallback
    for i in 0..len {
        result_slice[i] = input_slice[i].max(min_val).min(max_val);
    }
    result
}

/// SIMD-accelerated clip for f64 arrays
#[allow(dead_code)]
pub fn simd_clip_f64(input: &ArrayView1<f64>, min_val: f64, max_val: f64) -> Array1<f64> {
    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x.max(min_val).min(max_val));
    }

    let len = input.len();
    let mut result = Array1::zeros(len);
    let input_slice = input.as_slice().expect("contiguity checked above");
    let result_slice: &mut [f64] = result.as_slice_mut().expect("Operation failed");

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let min_vec = _mm256_set1_pd(min_val);
                let max_vec = _mm256_set1_pd(max_val);
                let mut i = 0;

                // Process 4 f64s at a time with AVX2 - direct pointer writes
                while i + 4 <= len {
                    let vec = _mm256_loadu_pd(input_slice.as_ptr().add(i));
                    let clipped = _mm256_max_pd(min_vec, _mm256_min_pd(max_vec, vec));
                    _mm256_storeu_pd(result_slice.as_mut_ptr().add(i), clipped);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = input_slice[j].max(min_val).min(max_val);
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
                let min_vec = vdupq_n_f64(min_val);
                let max_vec = vdupq_n_f64(max_val);
                let mut i = 0;

                // Process 2 f64s at a time with NEON - direct pointer writes
                while i + 2 <= len {
                    let vec = vld1q_f64(input_slice.as_ptr().add(i));
                    let clipped = vmaxq_f64(min_vec, vminq_f64(max_vec, vec));
                    vst1q_f64(result_slice.as_mut_ptr().add(i), clipped);
                    i += 2;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = input_slice[j].max(min_val).min(max_val);
                }

                return result;
            }
        }
    }

    // Scalar fallback
    for i in 0..len {
        result_slice[i] = input_slice[i].max(min_val).min(max_val);
    }
    result
}

// ==================== Log-Sum-Exp (Numerically Stable) ====================
