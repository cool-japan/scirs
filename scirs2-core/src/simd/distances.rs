//! Distance metric operations with SIMD acceleration
//!
//! This module provides optimized implementations of common distance metrics
//! including Euclidean, squared Euclidean, Manhattan, and Chebyshev distances.

use ndarray::ArrayView1;

/// SIMD-accelerated Euclidean distance (L2 distance) between two f32 arrays
///
/// Computes d(a, b) = sqrt((a_1 - b_1)^2 + (a_2 - b_2)^2 + ... + (a_n - b_n)^2)
///
/// # Arguments
///
/// * `a` - First input array
/// * `b` - Second input array
///
/// # Returns
///
/// * Euclidean distance between the two arrays
///
/// # Panics
///
/// Panics if arrays have different lengths
#[allow(dead_code)]
pub fn simd_distance_euclidean_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");
    let len = a.len();
    if len == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut sum_sq_vec = _mm256_setzero_ps();
                let mut i = 0;

                // Process 8 f32s at a time
                while i + 8 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 8];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 8];
                    let a_vec = _mm256_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_ps(b_slice.as_ptr());
                    let diff = _mm256_sub_ps(a_vec, b_vec);
                    let sq = _mm256_mul_ps(diff, diff);
                    sum_sq_vec = _mm256_add_ps(sum_sq_vec, sq);
                    i += 8;
                }

                // Horizontal sum: reduce 8 lanes to 1
                let low = _mm256_castps256_ps128(sum_sq_vec);
                let high = _mm256_extractf128_ps(sum_sq_vec, 1);
                let sum128 = _mm_add_ps(low, high);

                let shuffled = _mm_movehl_ps(sum128, sum128);
                let sum64 = _mm_add_ps(sum128, shuffled);
                let shuffled2 = _mm_shuffle_ps(sum64, sum64, 0x55);
                let sum_scalar = _mm_add_ss(sum64, shuffled2);
                let mut sum_sq = _mm_cvtss_f32(sum_scalar);

                // Handle remaining elements
                for j in i..len {
                    let diff = a[j] - b[j];
                    sum_sq += diff * diff;
                }

                return sum_sq.sqrt();
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let mut sum_sq_vec = _mm_setzero_ps();
                let mut i = 0;

                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];
                    let a_vec = _mm_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm_loadu_ps(b_slice.as_ptr());
                    let diff = _mm_sub_ps(a_vec, b_vec);
                    let sq = _mm_mul_ps(diff, diff);
                    sum_sq_vec = _mm_add_ps(sum_sq_vec, sq);
                    i += 4;
                }

                let shuffled = _mm_movehl_ps(sum_sq_vec, sum_sq_vec);
                let sum64 = _mm_add_ps(sum_sq_vec, shuffled);
                let shuffled2 = _mm_shuffle_ps(sum64, sum64, 0x55);
                let sum_scalar = _mm_add_ss(sum64, shuffled2);
                let mut sum_sq = _mm_cvtss_f32(sum_scalar);

                for j in i..len {
                    let diff = a[j] - b[j];
                    sum_sq += diff * diff;
                }

                return sum_sq.sqrt();
            }
        } else {
            let mut sum_sq = 0.0f32;
            for i in 0..len {
                let diff = a[i] - b[i];
                sum_sq += diff * diff;
            }
            return sum_sq.sqrt();
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut sum_sq_vec = vdupq_n_f32(0.0);
            let mut i = 0;

            while i + 4 <= len {
                let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];
                let a_vec = vld1q_f32(a_slice.as_ptr());
                let b_vec = vld1q_f32(b_slice.as_ptr());
                let diff = vsubq_f32(a_vec, b_vec);
                let sq = vmulq_f32(diff, diff);
                sum_sq_vec = vaddq_f32(sum_sq_vec, sq);
                i += 4;
            }

            let sum64 = vpadd_f32(vget_low_f32(sum_sq_vec), vget_high_f32(sum_sq_vec));
            let sum_pair = vpadd_f32(sum64, sum64);
            let mut sum_sq = vget_lane_f32(sum_pair, 0);

            for j in i..len {
                let diff = a[j] - b[j];
                sum_sq += diff * diff;
            }

            return sum_sq.sqrt();
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut sum_sq = 0.0f32;
        for i in 0..len {
            let diff = a[i] - b[i];
            sum_sq += diff * diff;
        }
        sum_sq.sqrt()
    }
}

/// SIMD-accelerated Euclidean distance (L2 distance) between two f64 arrays
///
/// Computes d(a, b) = sqrt((a_1 - b_1)^2 + (a_2 - b_2)^2 + ... + (a_n - b_n)^2)
#[allow(dead_code)]
pub fn simd_distance_euclidean_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");
    let len = a.len();
    if len == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut sum_sq_vec = _mm256_setzero_pd();
                let mut i = 0;

                // Process 4 f64s at a time
                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];
                    let a_vec = _mm256_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_pd(b_slice.as_ptr());
                    let diff = _mm256_sub_pd(a_vec, b_vec);
                    let sq = _mm256_mul_pd(diff, diff);
                    sum_sq_vec = _mm256_add_pd(sum_sq_vec, sq);
                    i += 4;
                }

                // Horizontal sum
                let low = _mm256_castpd256_pd128(sum_sq_vec);
                let high = _mm256_extractf128_pd(sum_sq_vec, 1);
                let sum128 = _mm_add_pd(low, high);
                let high_lane = _mm_unpackhi_pd(sum128, sum128);
                let sum_scalar = _mm_add_sd(sum128, high_lane);
                let mut sum_sq = _mm_cvtsd_f64(sum_scalar);

                // Handle remaining elements
                for j in i..len {
                    let diff = a[j] - b[j];
                    sum_sq += diff * diff;
                }

                return sum_sq.sqrt();
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let mut sum_sq_vec = _mm_setzero_pd();
                let mut i = 0;

                while i + 2 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 2];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 2];
                    let a_vec = _mm_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm_loadu_pd(b_slice.as_ptr());
                    let diff = _mm_sub_pd(a_vec, b_vec);
                    let sq = _mm_mul_pd(diff, diff);
                    sum_sq_vec = _mm_add_pd(sum_sq_vec, sq);
                    i += 2;
                }

                let high_lane = _mm_unpackhi_pd(sum_sq_vec, sum_sq_vec);
                let sum_scalar = _mm_add_sd(sum_sq_vec, high_lane);
                let mut sum_sq = _mm_cvtsd_f64(sum_scalar);

                for j in i..len {
                    let diff = a[j] - b[j];
                    sum_sq += diff * diff;
                }

                return sum_sq.sqrt();
            }
        } else {
            let mut sum_sq = 0.0f64;
            for i in 0..len {
                let diff = a[i] - b[i];
                sum_sq += diff * diff;
            }
            return sum_sq.sqrt();
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut sum_sq_vec = vdupq_n_f64(0.0);
            let mut i = 0;

            while i + 2 <= len {
                let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 2];
                let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 2];
                let a_vec = vld1q_f64(a_slice.as_ptr());
                let b_vec = vld1q_f64(b_slice.as_ptr());
                let diff = vsubq_f64(a_vec, b_vec);
                let sq = vmulq_f64(diff, diff);
                sum_sq_vec = vaddq_f64(sum_sq_vec, sq);
                i += 2;
            }

            let low = vgetq_lane_f64(sum_sq_vec, 0);
            let high = vgetq_lane_f64(sum_sq_vec, 1);
            let mut sum_sq = low + high;

            for j in i..len {
                let diff = a[j] - b[j];
                sum_sq += diff * diff;
            }

            return sum_sq.sqrt();
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut sum_sq = 0.0f64;
        for i in 0..len {
            let diff = a[i] - b[i];
            sum_sq += diff * diff;
        }
        sum_sq.sqrt()
    }
}

/// SIMD-accelerated squared Euclidean distance between two f32 arrays
///
/// Computes d^2(a, b) = (a_1 - b_1)^2 + (a_2 - b_2)^2 + ... + (a_n - b_n)^2
/// This is faster than Euclidean distance when you don't need the sqrt (e.g., k-NN)
///
/// # Stack-safety note
///
/// Deliberately `#[inline(never)]`: this leaf holds several wide `__m256`/`__m128`
/// (or NEON `float32x4_t`) locals. Callers of this function may sit deep inside
/// recursive spatial-tree descents (KdTree/BallTree); keeping the wide-register
/// frame confined to a single, un-inlined leaf call prevents it from being
/// duplicated into every level of such a recursive call chain.
#[allow(dead_code)]
#[inline(never)]
pub fn simd_distance_squared_euclidean_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");
    let len = a.len();
    if len == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut sum_sq_vec = _mm256_setzero_ps();
                let mut i = 0;

                while i + 8 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 8];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 8];
                    let a_vec = _mm256_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_ps(b_slice.as_ptr());
                    let diff = _mm256_sub_ps(a_vec, b_vec);
                    let sq = _mm256_mul_ps(diff, diff);
                    sum_sq_vec = _mm256_add_ps(sum_sq_vec, sq);
                    i += 8;
                }

                let low = _mm256_castps256_ps128(sum_sq_vec);
                let high = _mm256_extractf128_ps(sum_sq_vec, 1);
                let sum128 = _mm_add_ps(low, high);
                let shuffled = _mm_movehl_ps(sum128, sum128);
                let sum64 = _mm_add_ps(sum128, shuffled);
                let shuffled2 = _mm_shuffle_ps(sum64, sum64, 0x55);
                let sum_scalar = _mm_add_ss(sum64, shuffled2);
                let mut sum_sq = _mm_cvtss_f32(sum_scalar);

                for j in i..len {
                    let diff = a[j] - b[j];
                    sum_sq += diff * diff;
                }

                return sum_sq;
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let mut sum_sq_vec = _mm_setzero_ps();
                let mut i = 0;

                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];
                    let a_vec = _mm_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm_loadu_ps(b_slice.as_ptr());
                    let diff = _mm_sub_ps(a_vec, b_vec);
                    let sq = _mm_mul_ps(diff, diff);
                    sum_sq_vec = _mm_add_ps(sum_sq_vec, sq);
                    i += 4;
                }

                let shuffled = _mm_movehl_ps(sum_sq_vec, sum_sq_vec);
                let sum64 = _mm_add_ps(sum_sq_vec, shuffled);
                let shuffled2 = _mm_shuffle_ps(sum64, sum64, 0x55);
                let sum_scalar = _mm_add_ss(sum64, shuffled2);
                let mut sum_sq = _mm_cvtss_f32(sum_scalar);

                for j in i..len {
                    let diff = a[j] - b[j];
                    sum_sq += diff * diff;
                }

                return sum_sq;
            }
        } else {
            let mut sum_sq = 0.0f32;
            for i in 0..len {
                let diff = a[i] - b[i];
                sum_sq += diff * diff;
            }
            return sum_sq;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut sum_sq_vec = vdupq_n_f32(0.0);
            let mut i = 0;

            while i + 4 <= len {
                let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];
                let a_vec = vld1q_f32(a_slice.as_ptr());
                let b_vec = vld1q_f32(b_slice.as_ptr());
                let diff = vsubq_f32(a_vec, b_vec);
                let sq = vmulq_f32(diff, diff);
                sum_sq_vec = vaddq_f32(sum_sq_vec, sq);
                i += 4;
            }

            let sum64 = vpadd_f32(vget_low_f32(sum_sq_vec), vget_high_f32(sum_sq_vec));
            let sum_pair = vpadd_f32(sum64, sum64);
            let mut sum_sq = vget_lane_f32(sum_pair, 0);

            for j in i..len {
                let diff = a[j] - b[j];
                sum_sq += diff * diff;
            }

            return sum_sq;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut sum_sq = 0.0f32;
        for i in 0..len {
            let diff = a[i] - b[i];
            sum_sq += diff * diff;
        }
        sum_sq
    }
}

/// SIMD-accelerated squared Euclidean distance between two f64 arrays
///
/// Computes d^2(a, b) = (a_1 - b_1)^2 + (a_2 - b_2)^2 + ... + (a_n - b_n)^2
/// This is faster than Euclidean distance when you don't need the sqrt (e.g., k-NN)
///
/// # Stack-safety note
///
/// Deliberately `#[inline(never)]`: this leaf holds several wide `__m256d`/`__m128d`
/// (or NEON `float64x2_t`) locals. Callers of this function may sit deep inside
/// recursive spatial-tree descents (KdTree/BallTree); keeping the wide-register
/// frame confined to a single, un-inlined leaf call prevents it from being
/// duplicated into every level of such a recursive call chain.
#[allow(dead_code)]
#[inline(never)]
pub fn simd_distance_squared_euclidean_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");
    let len = a.len();
    if len == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut sum_sq_vec = _mm256_setzero_pd();
                let mut i = 0;

                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];
                    let a_vec = _mm256_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_pd(b_slice.as_ptr());
                    let diff = _mm256_sub_pd(a_vec, b_vec);
                    let sq = _mm256_mul_pd(diff, diff);
                    sum_sq_vec = _mm256_add_pd(sum_sq_vec, sq);
                    i += 4;
                }

                let low = _mm256_castpd256_pd128(sum_sq_vec);
                let high = _mm256_extractf128_pd(sum_sq_vec, 1);
                let sum128 = _mm_add_pd(low, high);
                let high_lane = _mm_unpackhi_pd(sum128, sum128);
                let sum_scalar = _mm_add_sd(sum128, high_lane);
                let mut sum_sq = _mm_cvtsd_f64(sum_scalar);

                for j in i..len {
                    let diff = a[j] - b[j];
                    sum_sq += diff * diff;
                }

                return sum_sq;
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let mut sum_sq_vec = _mm_setzero_pd();
                let mut i = 0;

                while i + 2 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 2];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 2];
                    let a_vec = _mm_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm_loadu_pd(b_slice.as_ptr());
                    let diff = _mm_sub_pd(a_vec, b_vec);
                    let sq = _mm_mul_pd(diff, diff);
                    sum_sq_vec = _mm_add_pd(sum_sq_vec, sq);
                    i += 2;
                }

                let high_lane = _mm_unpackhi_pd(sum_sq_vec, sum_sq_vec);
                let sum_scalar = _mm_add_sd(sum_sq_vec, high_lane);
                let mut sum_sq = _mm_cvtsd_f64(sum_scalar);

                for j in i..len {
                    let diff = a[j] - b[j];
                    sum_sq += diff * diff;
                }

                return sum_sq;
            }
        } else {
            let mut sum_sq = 0.0f64;
            for i in 0..len {
                let diff = a[i] - b[i];
                sum_sq += diff * diff;
            }
            return sum_sq;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut sum_sq_vec = vdupq_n_f64(0.0);
            let mut i = 0;

            while i + 2 <= len {
                let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 2];
                let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 2];
                let a_vec = vld1q_f64(a_slice.as_ptr());
                let b_vec = vld1q_f64(b_slice.as_ptr());
                let diff = vsubq_f64(a_vec, b_vec);
                let sq = vmulq_f64(diff, diff);
                sum_sq_vec = vaddq_f64(sum_sq_vec, sq);
                i += 2;
            }

            let low = vgetq_lane_f64(sum_sq_vec, 0);
            let high = vgetq_lane_f64(sum_sq_vec, 1);
            let mut sum_sq = low + high;

            for j in i..len {
                let diff = a[j] - b[j];
                sum_sq += diff * diff;
            }

            return sum_sq;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut sum_sq = 0.0f64;
        for i in 0..len {
            let diff = a[i] - b[i];
            sum_sq += diff * diff;
        }
        sum_sq
    }
}

/// SIMD-accelerated Manhattan distance (L1 distance) between two f32 arrays
///
/// Computes d(a, b) = |a_1 - b_1| + |a_2 - b_2| + ... + |a_n - b_n|
#[allow(dead_code)]
pub fn simd_distance_manhattan_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");
    let len = a.len();
    if len == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
                let mut sum_vec = _mm256_setzero_ps();
                let mut i = 0;

                while i + 8 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 8];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 8];
                    let a_vec = _mm256_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_ps(b_slice.as_ptr());
                    let diff = _mm256_sub_ps(a_vec, b_vec);
                    let abs_diff = _mm256_and_ps(diff, abs_mask);
                    sum_vec = _mm256_add_ps(sum_vec, abs_diff);
                    i += 8;
                }

                let low = _mm256_castps256_ps128(sum_vec);
                let high = _mm256_extractf128_ps(sum_vec, 1);
                let sum128 = _mm_add_ps(low, high);
                let shuffled = _mm_movehl_ps(sum128, sum128);
                let sum64 = _mm_add_ps(sum128, shuffled);
                let shuffled2 = _mm_shuffle_ps(sum64, sum64, 0x55);
                let sum_scalar = _mm_add_ss(sum64, shuffled2);
                let mut result = _mm_cvtss_f32(sum_scalar);

                for j in i..len {
                    result += (a[j] - b[j]).abs();
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let abs_mask = _mm_set1_ps(f32::from_bits(0x7FFF_FFFF));
                let mut sum_vec = _mm_setzero_ps();
                let mut i = 0;

                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];
                    let a_vec = _mm_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm_loadu_ps(b_slice.as_ptr());
                    let diff = _mm_sub_ps(a_vec, b_vec);
                    let abs_diff = _mm_and_ps(diff, abs_mask);
                    sum_vec = _mm_add_ps(sum_vec, abs_diff);
                    i += 4;
                }

                let shuffled = _mm_movehl_ps(sum_vec, sum_vec);
                let sum64 = _mm_add_ps(sum_vec, shuffled);
                let shuffled2 = _mm_shuffle_ps(sum64, sum64, 0x55);
                let sum_scalar = _mm_add_ss(sum64, shuffled2);
                let mut result = _mm_cvtss_f32(sum_scalar);

                for j in i..len {
                    result += (a[j] - b[j]).abs();
                }

                return result;
            }
        } else {
            let mut result = 0.0f32;
            for i in 0..len {
                result += (a[i] - b[i]).abs();
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
                let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];
                let a_vec = vld1q_f32(a_slice.as_ptr());
                let b_vec = vld1q_f32(b_slice.as_ptr());
                let diff = vsubq_f32(a_vec, b_vec);
                let abs_diff = vabsq_f32(diff);
                sum_vec = vaddq_f32(sum_vec, abs_diff);
                i += 4;
            }

            let sum64 = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
            let sum_pair = vpadd_f32(sum64, sum64);
            let mut result = vget_lane_f32(sum_pair, 0);

            for j in i..len {
                result += (a[j] - b[j]).abs();
            }

            return result;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut result = 0.0f32;
        for i in 0..len {
            result += (a[i] - b[i]).abs();
        }
        result
    }
}

/// SIMD-accelerated Manhattan distance (L1 distance) between two f64 arrays
///
/// Computes d(a, b) = |a_1 - b_1| + |a_2 - b_2| + ... + |a_n - b_n|
#[allow(dead_code)]
pub fn simd_distance_manhattan_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");
    let len = a.len();
    if len == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let abs_mask = _mm256_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));
                let mut sum_vec = _mm256_setzero_pd();
                let mut i = 0;

                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];
                    let a_vec = _mm256_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_pd(b_slice.as_ptr());
                    let diff = _mm256_sub_pd(a_vec, b_vec);
                    let abs_diff = _mm256_and_pd(diff, abs_mask);
                    sum_vec = _mm256_add_pd(sum_vec, abs_diff);
                    i += 4;
                }

                let low = _mm256_castpd256_pd128(sum_vec);
                let high = _mm256_extractf128_pd(sum_vec, 1);
                let sum128 = _mm_add_pd(low, high);
                let high_lane = _mm_unpackhi_pd(sum128, sum128);
                let sum_scalar = _mm_add_sd(sum128, high_lane);
                let mut result = _mm_cvtsd_f64(sum_scalar);

                for j in i..len {
                    result += (a[j] - b[j]).abs();
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let abs_mask = _mm_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));
                let mut sum_vec = _mm_setzero_pd();
                let mut i = 0;

                while i + 2 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 2];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 2];
                    let a_vec = _mm_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm_loadu_pd(b_slice.as_ptr());
                    let diff = _mm_sub_pd(a_vec, b_vec);
                    let abs_diff = _mm_and_pd(diff, abs_mask);
                    sum_vec = _mm_add_pd(sum_vec, abs_diff);
                    i += 2;
                }

                let high_lane = _mm_unpackhi_pd(sum_vec, sum_vec);
                let sum_scalar = _mm_add_sd(sum_vec, high_lane);
                let mut result = _mm_cvtsd_f64(sum_scalar);

                for j in i..len {
                    result += (a[j] - b[j]).abs();
                }

                return result;
            }
        } else {
            let mut result = 0.0f64;
            for i in 0..len {
                result += (a[i] - b[i]).abs();
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
                let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 2];
                let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 2];
                let a_vec = vld1q_f64(a_slice.as_ptr());
                let b_vec = vld1q_f64(b_slice.as_ptr());
                let diff = vsubq_f64(a_vec, b_vec);
                let abs_diff = vabsq_f64(diff);
                sum_vec = vaddq_f64(sum_vec, abs_diff);
                i += 2;
            }

            let low = vgetq_lane_f64(sum_vec, 0);
            let high = vgetq_lane_f64(sum_vec, 1);
            let mut result = low + high;

            for j in i..len {
                result += (a[j] - b[j]).abs();
            }

            return result;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut result = 0.0f64;
        for i in 0..len {
            result += (a[i] - b[i]).abs();
        }
        result
    }
}

/// SIMD-accelerated Chebyshev distance (L-infinity distance) between two f32 arrays
///
/// Computes d(a, b) = max(|a_1 - b_1|, |a_2 - b_2|, ..., |a_n - b_n|)
#[allow(dead_code)]
pub fn simd_distance_chebyshev_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");
    let len = a.len();
    if len == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
                let mut max_vec = _mm256_setzero_ps();
                let mut i = 0;

                while i + 8 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 8];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 8];
                    let a_vec = _mm256_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_ps(b_slice.as_ptr());
                    let diff = _mm256_sub_ps(a_vec, b_vec);
                    let abs_diff = _mm256_and_ps(diff, abs_mask);
                    max_vec = _mm256_max_ps(max_vec, abs_diff);
                    i += 8;
                }

                let low = _mm256_castps256_ps128(max_vec);
                let high = _mm256_extractf128_ps(max_vec, 1);
                let max128 = _mm_max_ps(low, high);
                let shuffled = _mm_movehl_ps(max128, max128);
                let max64 = _mm_max_ps(max128, shuffled);
                let shuffled2 = _mm_shuffle_ps(max64, max64, 0x55);
                let max_scalar = _mm_max_ss(max64, shuffled2);
                let mut result = _mm_cvtss_f32(max_scalar);

                for j in i..len {
                    let abs_diff = (a[j] - b[j]).abs();
                    if abs_diff > result {
                        result = abs_diff;
                    }
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let abs_mask = _mm_set1_ps(f32::from_bits(0x7FFF_FFFF));
                let mut max_vec = _mm_setzero_ps();
                let mut i = 0;

                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];
                    let a_vec = _mm_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm_loadu_ps(b_slice.as_ptr());
                    let diff = _mm_sub_ps(a_vec, b_vec);
                    let abs_diff = _mm_and_ps(diff, abs_mask);
                    max_vec = _mm_max_ps(max_vec, abs_diff);
                    i += 4;
                }

                let shuffled = _mm_movehl_ps(max_vec, max_vec);
                let max64 = _mm_max_ps(max_vec, shuffled);
                let shuffled2 = _mm_shuffle_ps(max64, max64, 0x55);
                let max_scalar = _mm_max_ss(max64, shuffled2);
                let mut result = _mm_cvtss_f32(max_scalar);

                for j in i..len {
                    let abs_diff = (a[j] - b[j]).abs();
                    if abs_diff > result {
                        result = abs_diff;
                    }
                }

                return result;
            }
        } else {
            let mut result = 0.0f32;
            for i in 0..len {
                let abs_diff = (a[i] - b[i]).abs();
                if abs_diff > result {
                    result = abs_diff;
                }
            }
            return result;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut max_vec = vdupq_n_f32(0.0);
            let mut i = 0;

            while i + 4 <= len {
                let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];
                let a_vec = vld1q_f32(a_slice.as_ptr());
                let b_vec = vld1q_f32(b_slice.as_ptr());
                let diff = vsubq_f32(a_vec, b_vec);
                let abs_diff = vabsq_f32(diff);
                max_vec = vmaxq_f32(max_vec, abs_diff);
                i += 4;
            }

            let max64 = vpmax_f32(vget_low_f32(max_vec), vget_high_f32(max_vec));
            let max_pair = vpmax_f32(max64, max64);
            let mut result = vget_lane_f32(max_pair, 0);

            for j in i..len {
                let abs_diff = (a[j] - b[j]).abs();
                if abs_diff > result {
                    result = abs_diff;
                }
            }

            return result;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut result = 0.0f32;
        for i in 0..len {
            let abs_diff = (a[i] - b[i]).abs();
            if abs_diff > result {
                result = abs_diff;
            }
        }
        result
    }
}

/// SIMD-accelerated Chebyshev distance (L-infinity distance) between two f64 arrays
///
/// Computes d(a, b) = max(|a_1 - b_1|, |a_2 - b_2|, ..., |a_n - b_n|)
#[allow(dead_code)]
pub fn simd_distance_chebyshev_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");
    let len = a.len();
    if len == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let abs_mask = _mm256_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));
                let mut max_vec = _mm256_setzero_pd();
                let mut i = 0;

                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];
                    let a_vec = _mm256_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_pd(b_slice.as_ptr());
                    let diff = _mm256_sub_pd(a_vec, b_vec);
                    let abs_diff = _mm256_and_pd(diff, abs_mask);
                    max_vec = _mm256_max_pd(max_vec, abs_diff);
                    i += 4;
                }

                let low = _mm256_castpd256_pd128(max_vec);
                let high = _mm256_extractf128_pd(max_vec, 1);
                let max128 = _mm_max_pd(low, high);
                let high_lane = _mm_unpackhi_pd(max128, max128);
                let max_scalar = _mm_max_sd(max128, high_lane);
                let mut result = _mm_cvtsd_f64(max_scalar);

                for j in i..len {
                    let abs_diff = (a[j] - b[j]).abs();
                    if abs_diff > result {
                        result = abs_diff;
                    }
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let abs_mask = _mm_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));
                let mut max_vec = _mm_setzero_pd();
                let mut i = 0;

                while i + 2 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 2];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 2];
                    let a_vec = _mm_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm_loadu_pd(b_slice.as_ptr());
                    let diff = _mm_sub_pd(a_vec, b_vec);
                    let abs_diff = _mm_and_pd(diff, abs_mask);
                    max_vec = _mm_max_pd(max_vec, abs_diff);
                    i += 2;
                }

                let high_lane = _mm_unpackhi_pd(max_vec, max_vec);
                let max_scalar = _mm_max_sd(max_vec, high_lane);
                let mut result = _mm_cvtsd_f64(max_scalar);

                for j in i..len {
                    let abs_diff = (a[j] - b[j]).abs();
                    if abs_diff > result {
                        result = abs_diff;
                    }
                }

                return result;
            }
        } else {
            let mut result = 0.0f64;
            for i in 0..len {
                let abs_diff = (a[i] - b[i]).abs();
                if abs_diff > result {
                    result = abs_diff;
                }
            }
            return result;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut max_vec = vdupq_n_f64(0.0);
            let mut i = 0;

            while i + 2 <= len {
                let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 2];
                let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 2];
                let a_vec = vld1q_f64(a_slice.as_ptr());
                let b_vec = vld1q_f64(b_slice.as_ptr());
                let diff = vsubq_f64(a_vec, b_vec);
                let abs_diff = vabsq_f64(diff);
                max_vec = vmaxq_f64(max_vec, abs_diff);
                i += 2;
            }

            let low = vgetq_lane_f64(max_vec, 0);
            let high = vgetq_lane_f64(max_vec, 1);
            let mut result = if low > high { low } else { high };

            for j in i..len {
                let abs_diff = (a[j] - b[j]).abs();
                if abs_diff > result {
                    result = abs_diff;
                }
            }

            return result;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut result = 0.0f64;
        for i in 0..len {
            let abs_diff = (a[i] - b[i]).abs();
            if abs_diff > result {
                result = abs_diff;
            }
        }
        result
    }
}
