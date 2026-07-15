//! Weighted operations with SIMD acceleration
//!
//! This module provides optimized implementations of weighted statistical operations
//! including weighted sum and weighted mean.

use ndarray::ArrayView1;

/// SIMD-accelerated weighted sum for f32 arrays
///
/// Computes `sum(values[i] * weights[i])` for all i
///
/// # Arguments
///
/// * `values` - Input values array
/// * `weights` - Weights array
///
/// # Returns
///
/// * Weighted sum of the values
///
/// # Panics
///
/// Panics if arrays have different lengths
///
/// # Stack-safety note
///
/// Deliberately `#[inline(never)]` (same precautionary reasoning as the squared-Euclidean
/// distance leaves in `simd/distances.rs`): callers of this kernel may sit deep inside
/// recursive numeric code, so its wide-register locals are kept out of ancestor frames.
#[allow(dead_code)]
#[inline(never)]
pub fn simd_weighted_sum_f32(values: &ArrayView1<f32>, weights: &ArrayView1<f32>) -> f32 {
    assert_eq!(
        values.len(),
        weights.len(),
        "Values and weights must have the same length"
    );
    let len = values.len();
    if len == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut sum_vec = _mm256_setzero_ps();
                let mut i = 0;

                while i + 8 <= len {
                    let v_slice = &values.as_slice().expect("Operation failed")[i..i + 8];
                    let w_slice = &weights.as_slice().expect("Operation failed")[i..i + 8];
                    let v_vec = _mm256_loadu_ps(v_slice.as_ptr());
                    let w_vec = _mm256_loadu_ps(w_slice.as_ptr());
                    let prod = _mm256_mul_ps(v_vec, w_vec);
                    sum_vec = _mm256_add_ps(sum_vec, prod);
                    i += 8;
                }

                // Horizontal sum
                let low = _mm256_castps256_ps128(sum_vec);
                let high = _mm256_extractf128_ps(sum_vec, 1);
                let sum128 = _mm_add_ps(low, high);
                let shuffled = _mm_movehl_ps(sum128, sum128);
                let sum64 = _mm_add_ps(sum128, shuffled);
                let shuffled2 = _mm_shuffle_ps(sum64, sum64, 0x55);
                let sum_scalar = _mm_add_ss(sum64, shuffled2);
                let mut result = _mm_cvtss_f32(sum_scalar);

                for j in i..len {
                    result += values[j] * weights[j];
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let mut sum_vec = _mm_setzero_ps();
                let mut i = 0;

                while i + 4 <= len {
                    let v_slice = &values.as_slice().expect("Operation failed")[i..i + 4];
                    let w_slice = &weights.as_slice().expect("Operation failed")[i..i + 4];
                    let v_vec = _mm_loadu_ps(v_slice.as_ptr());
                    let w_vec = _mm_loadu_ps(w_slice.as_ptr());
                    sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(v_vec, w_vec));
                    i += 4;
                }

                let shuffled = _mm_movehl_ps(sum_vec, sum_vec);
                let sum64 = _mm_add_ps(sum_vec, shuffled);
                let shuffled2 = _mm_shuffle_ps(sum64, sum64, 0x55);
                let sum_scalar = _mm_add_ss(sum64, shuffled2);
                let mut result = _mm_cvtss_f32(sum_scalar);

                for j in i..len {
                    result += values[j] * weights[j];
                }

                return result;
            }
        } else {
            let mut result = 0.0f32;
            for i in 0..len {
                result += values[i] * weights[i];
            }
            return result;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut sum_vec = vdupq_n_f32(0.0);
            let mut i = 0;

            while i + 4 <= len {
                let v_slice = &values.as_slice().expect("Operation failed")[i..i + 4];
                let w_slice = &weights.as_slice().expect("Operation failed")[i..i + 4];
                let v_vec = vld1q_f32(v_slice.as_ptr());
                let w_vec = vld1q_f32(w_slice.as_ptr());
                sum_vec = vaddq_f32(sum_vec, vmulq_f32(v_vec, w_vec));
                i += 4;
            }

            let sum64 = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
            let sum_pair = vpadd_f32(sum64, sum64);
            let mut result = vget_lane_f32(sum_pair, 0);

            for j in i..len {
                result += values[j] * weights[j];
            }

            return result;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut result = 0.0f32;
        for i in 0..len {
            result += values[i] * weights[i];
        }
        result
    }
}

/// SIMD-accelerated weighted sum for f64 arrays
///
/// Computes Σ(w_i × x_i)
///
/// # Stack-safety note
///
/// Deliberately `#[inline(never)]` — see [`simd_weighted_sum_f32`] for rationale.
#[allow(dead_code)]
#[inline(never)]
pub fn simd_weighted_sum_f64(values: &ArrayView1<f64>, weights: &ArrayView1<f64>) -> f64 {
    assert_eq!(
        values.len(),
        weights.len(),
        "Values and weights must have the same length"
    );
    let len = values.len();
    if len == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut sum_vec = _mm256_setzero_pd();
                let mut i = 0;

                while i + 4 <= len {
                    let v_slice = &values.as_slice().expect("Operation failed")[i..i + 4];
                    let w_slice = &weights.as_slice().expect("Operation failed")[i..i + 4];
                    let v_vec = _mm256_loadu_pd(v_slice.as_ptr());
                    let w_vec = _mm256_loadu_pd(w_slice.as_ptr());
                    let prod = _mm256_mul_pd(v_vec, w_vec);
                    sum_vec = _mm256_add_pd(sum_vec, prod);
                    i += 4;
                }

                // Horizontal sum
                let low = _mm256_castpd256_pd128(sum_vec);
                let high = _mm256_extractf128_pd(sum_vec, 1);
                let sum128 = _mm_add_pd(low, high);
                let high_lane = _mm_unpackhi_pd(sum128, sum128);
                let sum_scalar = _mm_add_sd(sum128, high_lane);
                let mut result = _mm_cvtsd_f64(sum_scalar);

                for j in i..len {
                    result += values[j] * weights[j];
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let mut sum_vec = _mm_setzero_pd();
                let mut i = 0;

                while i + 2 <= len {
                    let v_slice = &values.as_slice().expect("Operation failed")[i..i + 2];
                    let w_slice = &weights.as_slice().expect("Operation failed")[i..i + 2];
                    let v_vec = _mm_loadu_pd(v_slice.as_ptr());
                    let w_vec = _mm_loadu_pd(w_slice.as_ptr());
                    sum_vec = _mm_add_pd(sum_vec, _mm_mul_pd(v_vec, w_vec));
                    i += 2;
                }

                let high_lane = _mm_unpackhi_pd(sum_vec, sum_vec);
                let sum_scalar = _mm_add_sd(sum_vec, high_lane);
                let mut result = _mm_cvtsd_f64(sum_scalar);

                for j in i..len {
                    result += values[j] * weights[j];
                }

                return result;
            }
        } else {
            let mut result = 0.0f64;
            for i in 0..len {
                result += values[i] * weights[i];
            }
            return result;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut sum_vec = vdupq_n_f64(0.0);
            let mut i = 0;

            while i + 2 <= len {
                let v_slice = &values.as_slice().expect("Operation failed")[i..i + 2];
                let w_slice = &weights.as_slice().expect("Operation failed")[i..i + 2];
                let v_vec = vld1q_f64(v_slice.as_ptr());
                let w_vec = vld1q_f64(w_slice.as_ptr());
                sum_vec = vaddq_f64(sum_vec, vmulq_f64(v_vec, w_vec));
                i += 2;
            }

            let low = vgetq_lane_f64(sum_vec, 0);
            let high = vgetq_lane_f64(sum_vec, 1);
            let mut result = low + high;

            for j in i..len {
                result += values[j] * weights[j];
            }

            return result;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut result = 0.0f64;
        for i in 0..len {
            result += values[i] * weights[i];
        }
        result
    }
}

/// SIMD-accelerated weighted mean for f32 arrays
///
/// Computes Σ(w_i × x_i) / Σ(w_i)
#[allow(dead_code)]
pub fn simd_weighted_mean_f32(values: &ArrayView1<f32>, weights: &ArrayView1<f32>) -> f32 {
    assert_eq!(
        values.len(),
        weights.len(),
        "Values and weights must have the same length"
    );
    let len = values.len();
    if len == 0 {
        return f32::NAN;
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut sum_vec = _mm256_setzero_ps();
                let mut weight_sum_vec = _mm256_setzero_ps();
                let mut i = 0;

                while i + 8 <= len {
                    let v_slice = &values.as_slice().expect("Operation failed")[i..i + 8];
                    let w_slice = &weights.as_slice().expect("Operation failed")[i..i + 8];
                    let v_vec = _mm256_loadu_ps(v_slice.as_ptr());
                    let w_vec = _mm256_loadu_ps(w_slice.as_ptr());
                    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(v_vec, w_vec));
                    weight_sum_vec = _mm256_add_ps(weight_sum_vec, w_vec);
                    i += 8;
                }

                let hsum = |v: __m256| -> f32 {
                    let low = _mm256_castps256_ps128(v);
                    let high = _mm256_extractf128_ps(v, 1);
                    let sum128 = _mm_add_ps(low, high);
                    let shuffled = _mm_movehl_ps(sum128, sum128);
                    let sum64 = _mm_add_ps(sum128, shuffled);
                    let shuffled2 = _mm_shuffle_ps(sum64, sum64, 0x55);
                    let sum_scalar = _mm_add_ss(sum64, shuffled2);
                    _mm_cvtss_f32(sum_scalar)
                };

                let mut sum = hsum(sum_vec);
                let mut weight_sum = hsum(weight_sum_vec);

                for j in i..len {
                    sum += values[j] * weights[j];
                    weight_sum += weights[j];
                }

                if weight_sum == 0.0 {
                    return f32::NAN;
                }
                return sum / weight_sum;
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let mut sum_vec = _mm_setzero_ps();
                let mut weight_sum_vec = _mm_setzero_ps();
                let mut i = 0;

                while i + 4 <= len {
                    let v_slice = &values.as_slice().expect("Operation failed")[i..i + 4];
                    let w_slice = &weights.as_slice().expect("Operation failed")[i..i + 4];
                    let v_vec = _mm_loadu_ps(v_slice.as_ptr());
                    let w_vec = _mm_loadu_ps(w_slice.as_ptr());
                    sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(v_vec, w_vec));
                    weight_sum_vec = _mm_add_ps(weight_sum_vec, w_vec);
                    i += 4;
                }

                let hsum_sse = |v: __m128| -> f32 {
                    let shuffled = _mm_movehl_ps(v, v);
                    let sum64 = _mm_add_ps(v, shuffled);
                    let shuffled2 = _mm_shuffle_ps(sum64, sum64, 0x55);
                    let sum_scalar = _mm_add_ss(sum64, shuffled2);
                    _mm_cvtss_f32(sum_scalar)
                };

                let mut sum = hsum_sse(sum_vec);
                let mut weight_sum = hsum_sse(weight_sum_vec);

                for j in i..len {
                    sum += values[j] * weights[j];
                    weight_sum += weights[j];
                }

                if weight_sum == 0.0 {
                    return f32::NAN;
                }
                return sum / weight_sum;
            }
        } else {
            let mut sum = 0.0f32;
            let mut weight_sum = 0.0f32;
            for i in 0..len {
                sum += values[i] * weights[i];
                weight_sum += weights[i];
            }
            if weight_sum == 0.0 {
                return f32::NAN;
            }
            return sum / weight_sum;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut sum_vec = vdupq_n_f32(0.0);
            let mut weight_sum_vec = vdupq_n_f32(0.0);
            let mut i = 0;

            while i + 4 <= len {
                let v_slice = &values.as_slice().expect("Operation failed")[i..i + 4];
                let w_slice = &weights.as_slice().expect("Operation failed")[i..i + 4];
                let v_vec = vld1q_f32(v_slice.as_ptr());
                let w_vec = vld1q_f32(w_slice.as_ptr());
                sum_vec = vaddq_f32(sum_vec, vmulq_f32(v_vec, w_vec));
                weight_sum_vec = vaddq_f32(weight_sum_vec, w_vec);
                i += 4;
            }

            let hsum_neon = |v: float32x4_t| -> f32 {
                let sum64 = vpadd_f32(vget_low_f32(v), vget_high_f32(v));
                let sum_pair = vpadd_f32(sum64, sum64);
                vget_lane_f32(sum_pair, 0)
            };

            let mut sum = hsum_neon(sum_vec);
            let mut weight_sum = hsum_neon(weight_sum_vec);

            for j in i..len {
                sum += values[j] * weights[j];
                weight_sum += weights[j];
            }

            if weight_sum == 0.0 {
                return f32::NAN;
            }
            return sum / weight_sum;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;
        for i in 0..len {
            sum += values[i] * weights[i];
            weight_sum += weights[i];
        }
        if weight_sum == 0.0 {
            f32::NAN
        } else {
            sum / weight_sum
        }
    }
}

/// SIMD-accelerated weighted mean for f64 arrays
///
/// Computes Σ(w_i × x_i) / Σ(w_i)
#[allow(dead_code)]
pub fn simd_weighted_mean_f64(values: &ArrayView1<f64>, weights: &ArrayView1<f64>) -> f64 {
    assert_eq!(
        values.len(),
        weights.len(),
        "Values and weights must have the same length"
    );
    let len = values.len();
    if len == 0 {
        return f64::NAN;
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut sum_vec = _mm256_setzero_pd();
                let mut weight_sum_vec = _mm256_setzero_pd();
                let mut i = 0;

                while i + 4 <= len {
                    let v_slice = &values.as_slice().expect("Operation failed")[i..i + 4];
                    let w_slice = &weights.as_slice().expect("Operation failed")[i..i + 4];
                    let v_vec = _mm256_loadu_pd(v_slice.as_ptr());
                    let w_vec = _mm256_loadu_pd(w_slice.as_ptr());
                    sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(v_vec, w_vec));
                    weight_sum_vec = _mm256_add_pd(weight_sum_vec, w_vec);
                    i += 4;
                }

                let hsum = |v: __m256d| -> f64 {
                    let low = _mm256_castpd256_pd128(v);
                    let high = _mm256_extractf128_pd(v, 1);
                    let sum128 = _mm_add_pd(low, high);
                    let high_lane = _mm_unpackhi_pd(sum128, sum128);
                    let sum_scalar = _mm_add_sd(sum128, high_lane);
                    _mm_cvtsd_f64(sum_scalar)
                };

                let mut sum = hsum(sum_vec);
                let mut weight_sum = hsum(weight_sum_vec);

                for j in i..len {
                    sum += values[j] * weights[j];
                    weight_sum += weights[j];
                }

                if weight_sum == 0.0 {
                    return f64::NAN;
                }
                return sum / weight_sum;
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let mut sum_vec = _mm_setzero_pd();
                let mut weight_sum_vec = _mm_setzero_pd();
                let mut i = 0;

                while i + 2 <= len {
                    let v_slice = &values.as_slice().expect("Operation failed")[i..i + 2];
                    let w_slice = &weights.as_slice().expect("Operation failed")[i..i + 2];
                    let v_vec = _mm_loadu_pd(v_slice.as_ptr());
                    let w_vec = _mm_loadu_pd(w_slice.as_ptr());
                    sum_vec = _mm_add_pd(sum_vec, _mm_mul_pd(v_vec, w_vec));
                    weight_sum_vec = _mm_add_pd(weight_sum_vec, w_vec);
                    i += 2;
                }

                let hsum_sse = |v: __m128d| -> f64 {
                    let high = _mm_unpackhi_pd(v, v);
                    let sum = _mm_add_sd(v, high);
                    _mm_cvtsd_f64(sum)
                };

                let mut sum = hsum_sse(sum_vec);
                let mut weight_sum = hsum_sse(weight_sum_vec);

                for j in i..len {
                    sum += values[j] * weights[j];
                    weight_sum += weights[j];
                }

                if weight_sum == 0.0 {
                    return f64::NAN;
                }
                return sum / weight_sum;
            }
        } else {
            let mut sum = 0.0f64;
            let mut weight_sum = 0.0f64;
            for i in 0..len {
                sum += values[i] * weights[i];
                weight_sum += weights[i];
            }
            if weight_sum == 0.0 {
                return f64::NAN;
            }
            return sum / weight_sum;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut sum_vec = vdupq_n_f64(0.0);
            let mut weight_sum_vec = vdupq_n_f64(0.0);
            let mut i = 0;

            while i + 2 <= len {
                let v_slice = &values.as_slice().expect("Operation failed")[i..i + 2];
                let w_slice = &weights.as_slice().expect("Operation failed")[i..i + 2];
                let v_vec = vld1q_f64(v_slice.as_ptr());
                let w_vec = vld1q_f64(w_slice.as_ptr());
                sum_vec = vaddq_f64(sum_vec, vmulq_f64(v_vec, w_vec));
                weight_sum_vec = vaddq_f64(weight_sum_vec, w_vec);
                i += 2;
            }

            let low_sum = vgetq_lane_f64(sum_vec, 0);
            let high_sum = vgetq_lane_f64(sum_vec, 1);
            let low_w = vgetq_lane_f64(weight_sum_vec, 0);
            let high_w = vgetq_lane_f64(weight_sum_vec, 1);

            let mut sum = low_sum + high_sum;
            let mut weight_sum = low_w + high_w;

            for j in i..len {
                sum += values[j] * weights[j];
                weight_sum += weights[j];
            }

            if weight_sum == 0.0 {
                return f64::NAN;
            }
            return sum / weight_sum;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut sum = 0.0f64;
        let mut weight_sum = 0.0f64;
        for i in 0..len {
            sum += values[i] * weights[i];
            weight_sum += weights[i];
        }
        if weight_sum == 0.0 {
            f64::NAN
        } else {
            sum / weight_sum
        }
    }
}
