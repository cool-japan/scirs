//! Similarity metric operations with SIMD acceleration
//!
//! This module provides optimized implementations of similarity metrics
//! including cosine similarity and cosine distance.

use ndarray::ArrayView1;

/// SIMD-accelerated cosine similarity between two f32 arrays
///
/// Computes sim(a, b) = (a · b) / (||a|| * ||b||)
///
/// Returns NaN if either array has zero norm.
///
/// # Arguments
///
/// * `a` - First input array
/// * `b` - Second input array
///
/// # Returns
///
/// * Cosine similarity between the two arrays (range: [-1, 1])
///
/// # Panics
///
/// Panics if arrays have different lengths
#[allow(dead_code)]
pub fn simd_cosine_similarity_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");
    let len = a.len();
    if len == 0 {
        return f32::NAN;
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() || b.as_slice().is_none() {
        let mut dot = 0.0f32;
        let mut norm_a_sq = 0.0f32;
        let mut norm_b_sq = 0.0f32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a_sq += x * x;
            norm_b_sq += y * y;
        }
        let denom = (norm_a_sq * norm_b_sq).sqrt();
        return if denom == 0.0 { f32::NAN } else { dot / denom };
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut dot_vec = _mm256_setzero_ps();
                let mut norm_a_vec = _mm256_setzero_ps();
                let mut norm_b_vec = _mm256_setzero_ps();
                let mut i = 0;

                // Process 8 f32s at a time, computing all three sums together
                while i + 8 <= len {
                    let a_slice = &a.as_slice().expect("contiguity checked above")[i..i + 8];
                    let b_slice = &b.as_slice().expect("contiguity checked above")[i..i + 8];
                    let a_vec = _mm256_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_ps(b_slice.as_ptr());

                    // dot product: a * b
                    let prod = _mm256_mul_ps(a_vec, b_vec);
                    dot_vec = _mm256_add_ps(dot_vec, prod);

                    // norm_a: a * a
                    let sq_a = _mm256_mul_ps(a_vec, a_vec);
                    norm_a_vec = _mm256_add_ps(norm_a_vec, sq_a);

                    // norm_b: b * b
                    let sq_b = _mm256_mul_ps(b_vec, b_vec);
                    norm_b_vec = _mm256_add_ps(norm_b_vec, sq_b);

                    i += 8;
                }

                // Horizontal sum for all three vectors
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

                let mut dot = hsum(dot_vec);
                let mut norm_a_sq = hsum(norm_a_vec);
                let mut norm_b_sq = hsum(norm_b_vec);

                // Handle remaining elements
                for j in i..len {
                    dot += a[j] * b[j];
                    norm_a_sq += a[j] * a[j];
                    norm_b_sq += b[j] * b[j];
                }

                let denom = (norm_a_sq * norm_b_sq).sqrt();
                if denom == 0.0 {
                    return f32::NAN;
                }
                return dot / denom;
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let mut dot_vec = _mm_setzero_ps();
                let mut norm_a_vec = _mm_setzero_ps();
                let mut norm_b_vec = _mm_setzero_ps();
                let mut i = 0;

                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("contiguity checked above")[i..i + 4];
                    let b_slice = &b.as_slice().expect("contiguity checked above")[i..i + 4];
                    let a_vec = _mm_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm_loadu_ps(b_slice.as_ptr());

                    dot_vec = _mm_add_ps(dot_vec, _mm_mul_ps(a_vec, b_vec));
                    norm_a_vec = _mm_add_ps(norm_a_vec, _mm_mul_ps(a_vec, a_vec));
                    norm_b_vec = _mm_add_ps(norm_b_vec, _mm_mul_ps(b_vec, b_vec));
                    i += 4;
                }

                let hsum_sse = |v: __m128| -> f32 {
                    let shuffled = _mm_movehl_ps(v, v);
                    let sum64 = _mm_add_ps(v, shuffled);
                    let shuffled2 = _mm_shuffle_ps(sum64, sum64, 0x55);
                    let sum_scalar = _mm_add_ss(sum64, shuffled2);
                    _mm_cvtss_f32(sum_scalar)
                };

                let mut dot = hsum_sse(dot_vec);
                let mut norm_a_sq = hsum_sse(norm_a_vec);
                let mut norm_b_sq = hsum_sse(norm_b_vec);

                for j in i..len {
                    dot += a[j] * b[j];
                    norm_a_sq += a[j] * a[j];
                    norm_b_sq += b[j] * b[j];
                }

                let denom = (norm_a_sq * norm_b_sq).sqrt();
                if denom == 0.0 {
                    return f32::NAN;
                }
                return dot / denom;
            }
        } else {
            let mut dot = 0.0f32;
            let mut norm_a_sq = 0.0f32;
            let mut norm_b_sq = 0.0f32;
            for i in 0..len {
                dot += a[i] * b[i];
                norm_a_sq += a[i] * a[i];
                norm_b_sq += b[i] * b[i];
            }
            let denom = (norm_a_sq * norm_b_sq).sqrt();
            if denom == 0.0 {
                return f32::NAN;
            }
            return dot / denom;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut dot_vec = vdupq_n_f32(0.0);
            let mut norm_a_vec = vdupq_n_f32(0.0);
            let mut norm_b_vec = vdupq_n_f32(0.0);
            let mut i = 0;

            while i + 4 <= len {
                let a_slice = &a.as_slice().expect("contiguity checked above")[i..i + 4];
                let b_slice = &b.as_slice().expect("contiguity checked above")[i..i + 4];
                let a_vec = vld1q_f32(a_slice.as_ptr());
                let b_vec = vld1q_f32(b_slice.as_ptr());

                dot_vec = vaddq_f32(dot_vec, vmulq_f32(a_vec, b_vec));
                norm_a_vec = vaddq_f32(norm_a_vec, vmulq_f32(a_vec, a_vec));
                norm_b_vec = vaddq_f32(norm_b_vec, vmulq_f32(b_vec, b_vec));
                i += 4;
            }

            let hsum_neon = |v: float32x4_t| -> f32 {
                let sum64 = vpadd_f32(vget_low_f32(v), vget_high_f32(v));
                let sum_pair = vpadd_f32(sum64, sum64);
                vget_lane_f32(sum_pair, 0)
            };

            let mut dot = hsum_neon(dot_vec);
            let mut norm_a_sq = hsum_neon(norm_a_vec);
            let mut norm_b_sq = hsum_neon(norm_b_vec);

            for j in i..len {
                dot += a[j] * b[j];
                norm_a_sq += a[j] * a[j];
                norm_b_sq += b[j] * b[j];
            }

            let denom = (norm_a_sq * norm_b_sq).sqrt();
            if denom == 0.0 {
                return f32::NAN;
            }
            return dot / denom;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut dot = 0.0f32;
        let mut norm_a_sq = 0.0f32;
        let mut norm_b_sq = 0.0f32;
        for i in 0..len {
            dot += a[i] * b[i];
            norm_a_sq += a[i] * a[i];
            norm_b_sq += b[i] * b[i];
        }
        let denom = (norm_a_sq * norm_b_sq).sqrt();
        if denom == 0.0 {
            f32::NAN
        } else {
            dot / denom
        }
    }
}

/// SIMD-accelerated cosine similarity between two f64 arrays
///
/// Computes cos(θ) = (a · b) / (||a|| × ||b||)
/// Returns NaN if either vector has zero norm.
#[allow(dead_code)]
pub fn simd_cosine_similarity_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");
    let len = a.len();
    if len == 0 {
        return f64::NAN;
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() || b.as_slice().is_none() {
        let mut dot = 0.0f64;
        let mut norm_a_sq = 0.0f64;
        let mut norm_b_sq = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a_sq += x * x;
            norm_b_sq += y * y;
        }
        let denom = (norm_a_sq * norm_b_sq).sqrt();
        return if denom == 0.0 { f64::NAN } else { dot / denom };
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut dot_vec = _mm256_setzero_pd();
                let mut norm_a_vec = _mm256_setzero_pd();
                let mut norm_b_vec = _mm256_setzero_pd();
                let mut i = 0;

                // Process 4 f64s at a time
                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("contiguity checked above")[i..i + 4];
                    let b_slice = &b.as_slice().expect("contiguity checked above")[i..i + 4];
                    let a_vec = _mm256_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_pd(b_slice.as_ptr());

                    dot_vec = _mm256_add_pd(dot_vec, _mm256_mul_pd(a_vec, b_vec));
                    norm_a_vec = _mm256_add_pd(norm_a_vec, _mm256_mul_pd(a_vec, a_vec));
                    norm_b_vec = _mm256_add_pd(norm_b_vec, _mm256_mul_pd(b_vec, b_vec));
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

                let mut dot = hsum(dot_vec);
                let mut norm_a_sq = hsum(norm_a_vec);
                let mut norm_b_sq = hsum(norm_b_vec);

                for j in i..len {
                    dot += a[j] * b[j];
                    norm_a_sq += a[j] * a[j];
                    norm_b_sq += b[j] * b[j];
                }

                let denom = (norm_a_sq * norm_b_sq).sqrt();
                if denom == 0.0 {
                    return f64::NAN;
                }
                return dot / denom;
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let mut dot_vec = _mm_setzero_pd();
                let mut norm_a_vec = _mm_setzero_pd();
                let mut norm_b_vec = _mm_setzero_pd();
                let mut i = 0;

                while i + 2 <= len {
                    let a_slice = &a.as_slice().expect("contiguity checked above")[i..i + 2];
                    let b_slice = &b.as_slice().expect("contiguity checked above")[i..i + 2];
                    let a_vec = _mm_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm_loadu_pd(b_slice.as_ptr());

                    dot_vec = _mm_add_pd(dot_vec, _mm_mul_pd(a_vec, b_vec));
                    norm_a_vec = _mm_add_pd(norm_a_vec, _mm_mul_pd(a_vec, a_vec));
                    norm_b_vec = _mm_add_pd(norm_b_vec, _mm_mul_pd(b_vec, b_vec));
                    i += 2;
                }

                let hsum_sse = |v: __m128d| -> f64 {
                    let high = _mm_unpackhi_pd(v, v);
                    let sum = _mm_add_sd(v, high);
                    _mm_cvtsd_f64(sum)
                };

                let mut dot = hsum_sse(dot_vec);
                let mut norm_a_sq = hsum_sse(norm_a_vec);
                let mut norm_b_sq = hsum_sse(norm_b_vec);

                for j in i..len {
                    dot += a[j] * b[j];
                    norm_a_sq += a[j] * a[j];
                    norm_b_sq += b[j] * b[j];
                }

                let denom = (norm_a_sq * norm_b_sq).sqrt();
                if denom == 0.0 {
                    return f64::NAN;
                }
                return dot / denom;
            }
        } else {
            let mut dot = 0.0f64;
            let mut norm_a_sq = 0.0f64;
            let mut norm_b_sq = 0.0f64;
            for i in 0..len {
                dot += a[i] * b[i];
                norm_a_sq += a[i] * a[i];
                norm_b_sq += b[i] * b[i];
            }
            let denom = (norm_a_sq * norm_b_sq).sqrt();
            if denom == 0.0 {
                return f64::NAN;
            }
            return dot / denom;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut dot_vec = vdupq_n_f64(0.0);
            let mut norm_a_vec = vdupq_n_f64(0.0);
            let mut norm_b_vec = vdupq_n_f64(0.0);
            let mut i = 0;

            while i + 2 <= len {
                let a_slice = &a.as_slice().expect("contiguity checked above")[i..i + 2];
                let b_slice = &b.as_slice().expect("contiguity checked above")[i..i + 2];
                let a_vec = vld1q_f64(a_slice.as_ptr());
                let b_vec = vld1q_f64(b_slice.as_ptr());

                dot_vec = vaddq_f64(dot_vec, vmulq_f64(a_vec, b_vec));
                norm_a_vec = vaddq_f64(norm_a_vec, vmulq_f64(a_vec, a_vec));
                norm_b_vec = vaddq_f64(norm_b_vec, vmulq_f64(b_vec, b_vec));
                i += 2;
            }

            let low_dot = vgetq_lane_f64(dot_vec, 0);
            let high_dot = vgetq_lane_f64(dot_vec, 1);
            let low_a = vgetq_lane_f64(norm_a_vec, 0);
            let high_a = vgetq_lane_f64(norm_a_vec, 1);
            let low_b = vgetq_lane_f64(norm_b_vec, 0);
            let high_b = vgetq_lane_f64(norm_b_vec, 1);

            let mut dot = low_dot + high_dot;
            let mut norm_a_sq = low_a + high_a;
            let mut norm_b_sq = low_b + high_b;

            for j in i..len {
                dot += a[j] * b[j];
                norm_a_sq += a[j] * a[j];
                norm_b_sq += b[j] * b[j];
            }

            let denom = (norm_a_sq * norm_b_sq).sqrt();
            if denom == 0.0 {
                return f64::NAN;
            }
            return dot / denom;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut dot = 0.0f64;
        let mut norm_a_sq = 0.0f64;
        let mut norm_b_sq = 0.0f64;
        for i in 0..len {
            dot += a[i] * b[i];
            norm_a_sq += a[i] * a[i];
            norm_b_sq += b[i] * b[i];
        }
        let denom = (norm_a_sq * norm_b_sq).sqrt();
        if denom == 0.0 {
            f64::NAN
        } else {
            dot / denom
        }
    }
}

/// SIMD-accelerated cosine distance between two f32 arrays
///
/// Computes 1 - cos(θ), where cos(θ) is the cosine similarity.
/// Range: [0, 2], where 0 = identical, 2 = opposite
#[allow(dead_code)]
pub fn simd_distance_cosine_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    1.0 - simd_cosine_similarity_f32(a, b)
}

/// SIMD-accelerated cosine distance between two f64 arrays
///
/// Computes 1 - cos(θ), where cos(θ) is the cosine similarity.
/// Range: [0, 2], where 0 = identical, 2 = opposite
#[allow(dead_code)]
pub fn simd_distance_cosine_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    1.0 - simd_cosine_similarity_f64(a, b)
}
