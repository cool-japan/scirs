//! Unary operations with SIMD acceleration
//!
//! This module provides optimized implementations of unary (element-wise) operations
//! including absolute value, square root, and sign function.

use ndarray::{Array1, ArrayView1};

/// SIMD-accelerated absolute value for f32 arrays
///
/// Computes |x| for each element.
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// * Array of absolute values
#[allow(dead_code)]
pub fn simd_abs_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x.abs());
    }

    let len = input.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                // Mask for clearing sign bit (all bits except sign bit)
                let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
                let mut i = 0;

                while i + 8 <= len {
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 8];
                    let input_vec = _mm256_loadu_ps(input_slice.as_ptr());
                    let abs_vec = _mm256_and_ps(input_vec, sign_mask);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), abs_vec);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                for j in i..len {
                    result.push(input[j].abs());
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
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 4];
                    let input_vec = vld1q_f32(input_slice.as_ptr());
                    let abs_vec = vabsq_f32(input_vec);

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), abs_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    result.push(input[j].abs());
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for val in input.iter() {
        result.push(val.abs());
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated absolute value for f64 arrays
#[allow(dead_code)]
pub fn simd_abs_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x.abs());
    }

    let len = input.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                // Mask for clearing sign bit
                let sign_mask = _mm256_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));
                let mut i = 0;

                while i + 4 <= len {
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 4];
                    let input_vec = _mm256_loadu_pd(input_slice.as_ptr());
                    let abs_vec = _mm256_and_pd(input_vec, sign_mask);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), abs_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    result.push(input[j].abs());
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
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 2];
                    let input_vec = vld1q_f64(input_slice.as_ptr());
                    let abs_vec = vabsq_f64(input_vec);

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), abs_vec);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                for j in i..len {
                    result.push(input[j].abs());
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for val in input.iter() {
        result.push(val.abs());
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated square root for f32 arrays
#[allow(dead_code)]
pub fn simd_sqrt_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x.sqrt());
    }

    let len = input.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let mut i = 0;
                while i + 8 <= len {
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 8];
                    let input_vec = _mm256_loadu_ps(input_slice.as_ptr());
                    let sqrt_vec = _mm256_sqrt_ps(input_vec);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), sqrt_vec);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                for j in i..len {
                    result.push(input[j].sqrt());
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
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 4];
                    let input_vec = vld1q_f32(input_slice.as_ptr());
                    let sqrt_vec = vsqrtq_f32(input_vec);

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), sqrt_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    result.push(input[j].sqrt());
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for val in input.iter() {
        result.push(val.sqrt());
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated square root for f64 arrays
#[allow(dead_code)]
pub fn simd_sqrt_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x.sqrt());
    }

    let len = input.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let mut i = 0;
                while i + 4 <= len {
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 4];
                    let input_vec = _mm256_loadu_pd(input_slice.as_ptr());
                    let sqrt_vec = _mm256_sqrt_pd(input_vec);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), sqrt_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    result.push(input[j].sqrt());
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
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 2];
                    let input_vec = vld1q_f64(input_slice.as_ptr());
                    let sqrt_vec = vsqrtq_f64(input_vec);

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), sqrt_vec);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                for j in i..len {
                    result.push(input[j].sqrt());
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for val in input.iter() {
        result.push(val.sqrt());
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated sign function for f32 arrays
///
/// Returns -1.0 for negative values, 0.0 for zero, 1.0 for positive values.
#[allow(dead_code)]
pub fn simd_sign_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        });
    }

    let len = input.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let zero = _mm256_setzero_ps();
                let one = _mm256_set1_ps(1.0);
                let neg_one = _mm256_set1_ps(-1.0);
                let mut i = 0;

                while i + 8 <= len {
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 8];
                    let input_vec = _mm256_loadu_ps(input_slice.as_ptr());

                    // Compare: input > 0
                    let pos_mask = _mm256_cmp_ps(input_vec, zero, _CMP_GT_OQ);
                    // Compare: input < 0
                    let neg_mask = _mm256_cmp_ps(input_vec, zero, _CMP_LT_OQ);

                    // sign = pos_mask ? 1.0 : (neg_mask ? -1.0 : 0.0)
                    let pos_part = _mm256_and_ps(pos_mask, one);
                    let neg_part = _mm256_and_ps(neg_mask, neg_one);
                    let sign_vec = _mm256_or_ps(pos_part, neg_part);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), sign_vec);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                for j in i..len {
                    let v = input[j];
                    result.push(if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    });
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

                let zero = vdupq_n_f32(0.0);
                let one = vdupq_n_f32(1.0);
                let neg_one = vdupq_n_f32(-1.0);
                let mut i = 0;

                while i + 4 <= len {
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 4];
                    let input_vec = vld1q_f32(input_slice.as_ptr());

                    // Compare masks
                    let pos_mask = vcgtq_f32(input_vec, zero);
                    let neg_mask = vcltq_f32(input_vec, zero);

                    // Select values based on masks
                    let pos_part = vbslq_f32(pos_mask, one, zero);
                    let sign_vec = vbslq_f32(neg_mask, neg_one, pos_part);

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), sign_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    let v = input[j];
                    result.push(if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    });
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for val in input.iter() {
        result.push(if *val > 0.0 {
            1.0
        } else if *val < 0.0 {
            -1.0
        } else {
            0.0
        });
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated sign function for f64 arrays
#[allow(dead_code)]
pub fn simd_sign_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        });
    }

    let len = input.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let zero = _mm256_setzero_pd();
                let one = _mm256_set1_pd(1.0);
                let neg_one = _mm256_set1_pd(-1.0);
                let mut i = 0;

                while i + 4 <= len {
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 4];
                    let input_vec = _mm256_loadu_pd(input_slice.as_ptr());

                    let pos_mask = _mm256_cmp_pd(input_vec, zero, _CMP_GT_OQ);
                    let neg_mask = _mm256_cmp_pd(input_vec, zero, _CMP_LT_OQ);

                    let pos_part = _mm256_and_pd(pos_mask, one);
                    let neg_part = _mm256_and_pd(neg_mask, neg_one);
                    let sign_vec = _mm256_or_pd(pos_part, neg_part);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), sign_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    let v = input[j];
                    result.push(if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    });
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

                let zero = vdupq_n_f64(0.0);
                let one = vdupq_n_f64(1.0);
                let neg_one = vdupq_n_f64(-1.0);
                let mut i = 0;

                while i + 2 <= len {
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 2];
                    let input_vec = vld1q_f64(input_slice.as_ptr());

                    let pos_mask = vcgtq_f64(input_vec, zero);
                    let neg_mask = vcltq_f64(input_vec, zero);

                    let pos_part = vbslq_f64(pos_mask, one, zero);
                    let sign_vec = vbslq_f64(neg_mask, neg_one, pos_part);

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), sign_vec);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                for j in i..len {
                    let v = input[j];
                    result.push(if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    });
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for val in input.iter() {
        result.push(if *val > 0.0 {
            1.0
        } else if *val < 0.0 {
            -1.0
        } else {
            0.0
        });
    }
    Array1::from_vec(result)
}
