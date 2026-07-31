//! Activation functions with SIMD acceleration
//!
//! This module provides optimized implementations of common neural network
//! activation functions including ReLU, leaky ReLU, softmax, and log-sum-exp.
//!
//! Phase 75 optimization: These functions now use SIMD-accelerated exp
//! approximation from the transcendental module for significant speedup.

use ndarray::{Array1, ArrayView1};
// Import simd_max functions from reductions module
use super::reductions::{simd_max_f32, simd_max_f64};
// Import SIMD exp functions (Phase 75)
use super::transcendental::{simd_exp_f32 as simd_exp_vec_f32, simd_exp_f64 as simd_exp_vec_f64};

/// SIMD-accelerated log-sum-exp for f32 arrays
///
/// Computes log(sum(exp(x_i))) with numerical stability.
/// Uses the max-value trick: log(sum(exp(x - max))) + max
///
/// Phase 75: Now uses SIMD-accelerated exp for significant speedup.
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// * Log-sum-exp value
#[allow(dead_code)]
pub fn simd_log_sum_exp_f32(input: &ArrayView1<f32>) -> f32 {
    let len = input.len();
    if len == 0 {
        return f32::NEG_INFINITY;
    }

    // Find max for numerical stability
    let max_val = simd_max_f32(input);
    if max_val.is_infinite() {
        return max_val;
    }

    // Phase 75: Use SIMD-accelerated exp
    // Create shifted array: (x - max)
    let shifted: Array1<f32> = input.mapv(|x| x - max_val);

    // Apply SIMD exp to all elements
    let exp_vals = simd_exp_vec_f32(&shifted.view());

    // Sum using SIMD
    let sum: f32 = super::reductions::simd_sum_f32(&exp_vals.view());

    max_val + sum.ln()
}

/// SIMD-accelerated log-sum-exp for f64 arrays
///
/// Phase 75: Now uses SIMD-accelerated exp for significant speedup.
#[allow(dead_code)]
pub fn simd_log_sum_exp_f64(input: &ArrayView1<f64>) -> f64 {
    let len = input.len();
    if len == 0 {
        return f64::NEG_INFINITY;
    }

    // Find max for numerical stability
    let max_val = simd_max_f64(input);
    if max_val.is_infinite() {
        return max_val;
    }

    // Phase 75: Use SIMD-accelerated exp
    // Create shifted array: (x - max)
    let shifted: Array1<f64> = input.mapv(|x| x - max_val);

    // Apply SIMD exp to all elements
    let exp_vals = simd_exp_vec_f64(&shifted.view());

    // Sum using SIMD
    let sum: f64 = super::reductions::simd_sum_f64(&exp_vals.view());

    max_val + sum.ln()
}

/// SIMD-accelerated softmax for f32 arrays
///
/// Computes softmax(x) = exp(x - log_sum_exp(x)) for numerical stability.
/// The output probabilities sum to 1.
///
/// Phase 75: Now uses SIMD-accelerated exp for significant speedup.
///
/// # Arguments
/// * `input` - Input array of logits
///
/// # Returns
/// Array of softmax probabilities
#[allow(dead_code)]
pub fn simd_softmax_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    // Compute log-sum-exp for numerical stability (already uses SIMD exp)
    let lse = simd_log_sum_exp_f32(input);

    // Phase 75: Use SIMD-accelerated exp
    // Create shifted array: (x - lse)
    let shifted: Array1<f32> = input.mapv(|x| x - lse);

    // Apply SIMD exp to all elements
    simd_exp_vec_f32(&shifted.view())
}

/// SIMD-accelerated softmax for f64 arrays
///
/// Computes softmax(x) = exp(x - log_sum_exp(x)) for numerical stability.
/// The output probabilities sum to 1.
///
/// Phase 75: Now uses SIMD-accelerated exp for significant speedup.
///
/// # Arguments
/// * `input` - Input array of logits
///
/// # Returns
/// Array of softmax probabilities
#[allow(dead_code)]
pub fn simd_softmax_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    // Compute log-sum-exp for numerical stability (already uses SIMD exp)
    let lse = simd_log_sum_exp_f64(input);

    // Phase 75: Use SIMD-accelerated exp
    // Create shifted array: (x - lse)
    let shifted: Array1<f64> = input.mapv(|x| x - lse);

    // Apply SIMD exp to all elements
    simd_exp_vec_f64(&shifted.view())
}

/// SIMD-accelerated cumulative sum for f32 arrays
///
/// Computes the running sum where each element is the sum of all preceding elements
/// (including itself).
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// Array of cumulative sums
#[allow(dead_code)]
pub fn simd_relu_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x.max(0.0));
    }

    let len = input.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let zero = _mm256_setzero_ps();
                let mut i = 0;

                while i + 8 <= len {
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 8];
                    let input_vec = _mm256_loadu_ps(input_slice.as_ptr());
                    let relu_vec = _mm256_max_ps(input_vec, zero);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), relu_vec);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                for j in i..len {
                    result.push(input[j].max(0.0));
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
                let mut i = 0;

                while i + 4 <= len {
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 4];
                    let input_vec = vld1q_f32(input_slice.as_ptr());
                    let relu_vec = vmaxq_f32(input_vec, zero);

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), relu_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    result.push(input[j].max(0.0));
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for val in input.iter() {
        result.push(val.max(0.0));
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated ReLU activation for f64 arrays
#[allow(dead_code)]
pub fn simd_relu_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x.max(0.0));
    }

    let len = input.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let zero = _mm256_setzero_pd();
                let mut i = 0;

                while i + 4 <= len {
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 4];
                    let input_vec = _mm256_loadu_pd(input_slice.as_ptr());
                    let relu_vec = _mm256_max_pd(input_vec, zero);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), relu_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    result.push(input[j].max(0.0));
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
                let mut i = 0;

                while i + 2 <= len {
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 2];
                    let input_vec = vld1q_f64(input_slice.as_ptr());
                    let relu_vec = vmaxq_f64(input_vec, zero);

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), relu_vec);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                for j in i..len {
                    result.push(input[j].max(0.0));
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for val in input.iter() {
        result.push(val.max(0.0));
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated Leaky ReLU activation for f32 arrays
///
/// Computes max(alpha*x, x) for each element.
#[allow(dead_code)]
pub fn simd_leaky_relu_f32(input: &ArrayView1<f32>, alpha: f32) -> Array1<f32> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| if x > 0.0 { x } else { alpha * x });
    }

    let len = input.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let zero = _mm256_setzero_ps();
                let alpha_vec = _mm256_set1_ps(alpha);
                let mut i = 0;

                while i + 8 <= len {
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 8];
                    let input_vec = _mm256_loadu_ps(input_slice.as_ptr());

                    // negative_part = alpha * x (for when x < 0)
                    let scaled = _mm256_mul_ps(input_vec, alpha_vec);
                    // leaky_relu = max(scaled, input)
                    let leaky_relu = _mm256_max_ps(scaled, input_vec);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), leaky_relu);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                for j in i..len {
                    let x = input[j];
                    result.push(if x > 0.0 { x } else { alpha * x });
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
                let alpha_vec = vdupq_n_f32(alpha);
                let mut i = 0;

                while i + 4 <= len {
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 4];
                    let input_vec = vld1q_f32(input_slice.as_ptr());

                    let scaled = vmulq_f32(input_vec, alpha_vec);
                    let mask = vcgtq_f32(input_vec, zero);
                    let leaky_relu = vbslq_f32(mask, input_vec, scaled);

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), leaky_relu);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    let x = input[j];
                    result.push(if x > 0.0 { x } else { alpha * x });
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for val in input.iter() {
        result.push(if *val > 0.0 { *val } else { alpha * val });
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated Leaky ReLU activation for f64 arrays
#[allow(dead_code)]
pub fn simd_leaky_relu_f64(input: &ArrayView1<f64>, alpha: f64) -> Array1<f64> {
    if input.is_empty() {
        return Array1::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| if x > 0.0 { x } else { alpha * x });
    }

    let len = input.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let zero = _mm256_setzero_pd();
                let alpha_vec = _mm256_set1_pd(alpha);
                let mut i = 0;

                while i + 4 <= len {
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 4];
                    let input_vec = _mm256_loadu_pd(input_slice.as_ptr());

                    let scaled = _mm256_mul_pd(input_vec, alpha_vec);
                    let leaky_relu = _mm256_max_pd(scaled, input_vec);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), leaky_relu);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    let x = input[j];
                    result.push(if x > 0.0 { x } else { alpha * x });
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
                let alpha_vec = vdupq_n_f64(alpha);
                let mut i = 0;

                while i + 2 <= len {
                    let input_slice =
                        &input.as_slice().expect("contiguity checked above")[i..i + 2];
                    let input_vec = vld1q_f64(input_slice.as_ptr());

                    let scaled = vmulq_f64(input_vec, alpha_vec);
                    let mask = vcgtq_f64(input_vec, zero);
                    let leaky_relu = vbslq_f64(mask, input_vec, scaled);

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), leaky_relu);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                for j in i..len {
                    let x = input[j];
                    result.push(if x > 0.0 { x } else { alpha * x });
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for val in input.iter() {
        result.push(if *val > 0.0 { *val } else { alpha * val });
    }
    Array1::from_vec(result)
}
