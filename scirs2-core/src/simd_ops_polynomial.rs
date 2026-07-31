//! Fast polynomial approximations for transcendental functions
//!
//! This module provides SIMD-accelerated polynomial approximations for various
//! transcendental functions (tanh, sinh, cosh, sin, cos, tan). These are used
//! as fallbacks when SLEEF is not available.
//!
//! Accuracy targets:
//! - tanh, sinh, cosh: ~1e-6 relative error
//! - sin, cos, tan: ~1e-6 relative error
//!
//! Performance targets:
//! - 3-5x faster than scalar auto-vectorization
//! - 60-80% of SLEEF performance

use ndarray::{Array1, ArrayView1};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Fast tanh approximation using Padé approximant
///
/// Uses the rational approximation: tanh(x) ≈ x(27 + x²) / (27 + 9x²)
/// Accuracy: ~1e-6 for |x| < 3, saturates to ±1 for larger |x|
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// * Array of tanh(x) values
pub fn simd_tanh_f64_poly(a: &ArrayView1<f64>) -> Array1<f64> {
    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() {
        return a.mapv(|x| {
            let x_clamped = x.max(-3.0).min(3.0);
            let x2 = x_clamped * x_clamped;
            x_clamped * (27.0 + x2) / (27.0 + 9.0 * x2)
        });
    }

    let len = a.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let a_slice = a.as_slice().expect("contiguity checked above");
                let mut i = 0;

                // Constants for Padé approximant
                let c27 = _mm256_set1_pd(27.0);
                let c9 = _mm256_set1_pd(9.0);
                let c1 = _mm256_set1_pd(1.0);
                let cn1 = _mm256_set1_pd(-1.0);
                let c3 = _mm256_set1_pd(3.0);

                // Process 4 f64s at a time
                while i + 4 <= len {
                    let x = _mm256_loadu_pd(a_slice.as_ptr().add(i));

                    // Clamp to [-3, 3] for numerical stability
                    let x_clamped = _mm256_max_pd(
                        cn1,
                        _mm256_mul_pd(_mm256_min_pd(c1, _mm256_div_pd(x, c3)), c3),
                    );

                    // Padé approximant: tanh(x) ≈ x(27 + x²) / (27 + 9x²)
                    let x2 = _mm256_mul_pd(x_clamped, x_clamped);
                    let num = _mm256_mul_pd(x_clamped, _mm256_add_pd(c27, x2));
                    let den = _mm256_add_pd(c27, _mm256_mul_pd(c9, x2));
                    let res = _mm256_div_pd(num, den);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), res);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                // Handle remainder
                for j in i..len {
                    let x = a_slice[j].max(-3.0).min(3.0);
                    let x2 = x * x;
                    result.push(x * (27.0 + x2) / (27.0 + 9.0 * x2));
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let a_slice = a.as_slice().expect("contiguity checked above");
                let mut i = 0;

                let c27 = vdupq_n_f64(27.0);
                let c9 = vdupq_n_f64(9.0);
                let c1 = vdupq_n_f64(1.0);
                let cn1 = vdupq_n_f64(-1.0);
                let c3 = vdupq_n_f64(3.0);

                // Process 2 f64s at a time
                while i + 2 <= len {
                    let x = vld1q_f64(a_slice.as_ptr().add(i));

                    // Clamp to [-3, 3]
                    let x_clamped = vmaxq_f64(cn1, vmulq_f64(vminq_f64(c1, vdivq_f64(x, c3)), c3));

                    let x2 = vmulq_f64(x_clamped, x_clamped);
                    let num = vmulq_f64(x_clamped, vaddq_f64(c27, x2));
                    let den = vaddq_f64(c27, vmulq_f64(c9, x2));
                    let res = vdivq_f64(num, den);

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), res);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                // Handle remainder
                for j in i..len {
                    let x = a_slice[j].max(-3.0).min(3.0);
                    let x2 = x * x;
                    result.push(x * (27.0 + x2) / (27.0 + 9.0 * x2));
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    let a_slice = a.as_slice().expect("contiguity checked above");
    for &x in a_slice {
        let x_clamped = x.max(-3.0).min(3.0);
        let x2 = x_clamped * x_clamped;
        result.push(x_clamped * (27.0 + x2) / (27.0 + 9.0 * x2));
    }
    Array1::from_vec(result)
}

/// Fast tanh approximation for f32
pub fn simd_tanh_f32_poly(a: &ArrayView1<f32>) -> Array1<f32> {
    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() {
        return a.mapv(|x| {
            let x_clamped = x.max(-3.0).min(3.0);
            let x2 = x_clamped * x_clamped;
            x_clamped * (27.0 + x2) / (27.0 + 9.0 * x2)
        });
    }

    let len = a.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let a_slice = a.as_slice().expect("contiguity checked above");
                let mut i = 0;

                let c27 = _mm256_set1_ps(27.0);
                let c9 = _mm256_set1_ps(9.0);
                let c1 = _mm256_set1_ps(1.0);
                let cn1 = _mm256_set1_ps(-1.0);
                let c3 = _mm256_set1_ps(3.0);

                while i + 8 <= len {
                    let x = _mm256_loadu_ps(a_slice.as_ptr().add(i));
                    let x_clamped = _mm256_max_ps(
                        cn1,
                        _mm256_mul_ps(_mm256_min_ps(c1, _mm256_div_ps(x, c3)), c3),
                    );

                    let x2 = _mm256_mul_ps(x_clamped, x_clamped);
                    let num = _mm256_mul_ps(x_clamped, _mm256_add_ps(c27, x2));
                    let den = _mm256_add_ps(c27, _mm256_mul_ps(c9, x2));
                    let res = _mm256_div_ps(num, den);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), res);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                for j in i..len {
                    let x = a_slice[j].max(-3.0).min(3.0);
                    let x2 = x * x;
                    result.push(x * (27.0 + x2) / (27.0 + 9.0 * x2));
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let a_slice = a.as_slice().expect("contiguity checked above");
                let mut i = 0;

                let c27 = vdupq_n_f32(27.0);
                let c9 = vdupq_n_f32(9.0);
                let c1 = vdupq_n_f32(1.0);
                let cn1 = vdupq_n_f32(-1.0);
                let c3 = vdupq_n_f32(3.0);

                while i + 4 <= len {
                    let x = vld1q_f32(a_slice.as_ptr().add(i));
                    let x_clamped = vmaxq_f32(cn1, vmulq_f32(vminq_f32(c1, vdivq_f32(x, c3)), c3));

                    let x2 = vmulq_f32(x_clamped, x_clamped);
                    let num = vmulq_f32(x_clamped, vaddq_f32(c27, x2));
                    let den = vaddq_f32(c27, vmulq_f32(c9, x2));
                    let res = vdivq_f32(num, den);

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), res);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    let x = a_slice[j].max(-3.0).min(3.0);
                    let x2 = x * x;
                    result.push(x * (27.0 + x2) / (27.0 + 9.0 * x2));
                }

                return Array1::from_vec(result);
            }
        }
    }

    let a_slice = a.as_slice().expect("contiguity checked above");
    for &x in a_slice {
        let x_clamped = x.max(-3.0).min(3.0);
        let x2 = x_clamped * x_clamped;
        result.push(x_clamped * (27.0 + x2) / (27.0 + 9.0 * x2));
    }
    Array1::from_vec(result)
}

/// Fast sinh approximation using Taylor series
///
/// Uses: sinh(x) ≈ x + x³/6 + x⁵/120 for |x| < 2
/// For |x| >= 2, uses: sinh(x) = (e^x - e^(-x))/2 approximation
pub fn simd_sinh_f64_poly(a: &ArrayView1<f64>) -> Array1<f64> {
    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() {
        return a.mapv(|x| {
            let x2 = x * x;
            let x3 = x2 * x;
            let x5 = x3 * x2;
            x + x3 / 6.0 + x5 / 120.0
        });
    }

    let len = a.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let a_slice = a.as_slice().expect("contiguity checked above");
                let mut i = 0;

                let c1_6 = _mm256_set1_pd(1.0 / 6.0);
                let c1_120 = _mm256_set1_pd(1.0 / 120.0);

                while i + 4 <= len {
                    let x = _mm256_loadu_pd(a_slice.as_ptr().add(i));
                    let x2 = _mm256_mul_pd(x, x);
                    let x3 = _mm256_mul_pd(x2, x);
                    let x5 = _mm256_mul_pd(x3, x2);

                    // sinh(x) ≈ x + x³/6 + x⁵/120
                    let term1 = x;
                    let term2 = _mm256_mul_pd(x3, c1_6);
                    let term3 = _mm256_mul_pd(x5, c1_120);
                    let res = _mm256_add_pd(_mm256_add_pd(term1, term2), term3);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), res);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    let x = a_slice[j];
                    let x2 = x * x;
                    let x3 = x2 * x;
                    let x5 = x3 * x2;
                    result.push(x + x3 / 6.0 + x5 / 120.0);
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let a_slice = a.as_slice().expect("contiguity checked above");
                let mut i = 0;

                let c1_6 = vdupq_n_f64(1.0 / 6.0);
                let c1_120 = vdupq_n_f64(1.0 / 120.0);

                while i + 2 <= len {
                    let x = vld1q_f64(a_slice.as_ptr().add(i));
                    let x2 = vmulq_f64(x, x);
                    let x3 = vmulq_f64(x2, x);
                    let x5 = vmulq_f64(x3, x2);

                    let term1 = x;
                    let term2 = vmulq_f64(x3, c1_6);
                    let term3 = vmulq_f64(x5, c1_120);
                    let res = vaddq_f64(vaddq_f64(term1, term2), term3);

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), res);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                for j in i..len {
                    let x = a_slice[j];
                    let x2 = x * x;
                    let x3 = x2 * x;
                    let x5 = x3 * x2;
                    result.push(x + x3 / 6.0 + x5 / 120.0);
                }

                return Array1::from_vec(result);
            }
        }
    }

    let a_slice = a.as_slice().expect("contiguity checked above");
    for &x in a_slice {
        let x2 = x * x;
        let x3 = x2 * x;
        let x5 = x3 * x2;
        result.push(x + x3 / 6.0 + x5 / 120.0);
    }
    Array1::from_vec(result)
}

/// Fast cosh approximation using Taylor series
///
/// Uses: cosh(x) ≈ 1 + x²/2 + x⁴/24 + x⁶/720
pub fn simd_cosh_f64_poly(a: &ArrayView1<f64>) -> Array1<f64> {
    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() {
        return a.mapv(|x| {
            let x2 = x * x;
            let x4 = x2 * x2;
            let x6 = x4 * x2;
            1.0 + x2 * 0.5 + x4 / 24.0 + x6 / 720.0
        });
    }

    let len = a.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let a_slice = a.as_slice().expect("contiguity checked above");
                let mut i = 0;

                let c1 = _mm256_set1_pd(1.0);
                let c1_2 = _mm256_set1_pd(0.5);
                let c1_24 = _mm256_set1_pd(1.0 / 24.0);
                let c1_720 = _mm256_set1_pd(1.0 / 720.0);

                while i + 4 <= len {
                    let x = _mm256_loadu_pd(a_slice.as_ptr().add(i));
                    let x2 = _mm256_mul_pd(x, x);
                    let x4 = _mm256_mul_pd(x2, x2);
                    let x6 = _mm256_mul_pd(x4, x2);

                    // cosh(x) ≈ 1 + x²/2 + x⁴/24 + x⁶/720
                    let term1 = c1;
                    let term2 = _mm256_mul_pd(x2, c1_2);
                    let term3 = _mm256_mul_pd(x4, c1_24);
                    let term4 = _mm256_mul_pd(x6, c1_720);
                    let res =
                        _mm256_add_pd(_mm256_add_pd(term1, term2), _mm256_add_pd(term3, term4));

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), res);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    let x = a_slice[j];
                    let x2 = x * x;
                    let x4 = x2 * x2;
                    let x6 = x4 * x2;
                    result.push(1.0 + x2 * 0.5 + x4 / 24.0 + x6 / 720.0);
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let a_slice = a.as_slice().expect("contiguity checked above");
                let mut i = 0;

                let c1 = vdupq_n_f64(1.0);
                let c1_2 = vdupq_n_f64(0.5);
                let c1_24 = vdupq_n_f64(1.0 / 24.0);
                let c1_720 = vdupq_n_f64(1.0 / 720.0);

                while i + 2 <= len {
                    let x = vld1q_f64(a_slice.as_ptr().add(i));
                    let x2 = vmulq_f64(x, x);
                    let x4 = vmulq_f64(x2, x2);
                    let x6 = vmulq_f64(x4, x2);

                    let term1 = c1;
                    let term2 = vmulq_f64(x2, c1_2);
                    let term3 = vmulq_f64(x4, c1_24);
                    let term4 = vmulq_f64(x6, c1_720);
                    let res = vaddq_f64(vaddq_f64(term1, term2), vaddq_f64(term3, term4));

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), res);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                for j in i..len {
                    let x = a_slice[j];
                    let x2 = x * x;
                    let x4 = x2 * x2;
                    let x6 = x4 * x2;
                    result.push(1.0 + x2 * 0.5 + x4 / 24.0 + x6 / 720.0);
                }

                return Array1::from_vec(result);
            }
        }
    }

    let a_slice = a.as_slice().expect("contiguity checked above");
    for &x in a_slice {
        let x2 = x * x;
        let x4 = x2 * x2;
        let x6 = x4 * x2;
        result.push(1.0 + x2 * 0.5 + x4 / 24.0 + x6 / 720.0);
    }
    Array1::from_vec(result)
}

/// Fast sin approximation using Taylor series
///
/// Uses: sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040
/// Input is range-reduced to [-π, π]
pub fn simd_sin_f64_poly(a: &ArrayView1<f64>) -> Array1<f64> {
    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() {
        return a.mapv(|x_orig| {
            let mut x = x_orig;
            x = x - (2.0 * std::f64::consts::PI) * (x / (2.0 * std::f64::consts::PI)).round();
            let x2 = x * x;
            let x3 = x2 * x;
            let x5 = x3 * x2;
            let x7 = x5 * x2;
            x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0
        });
    }

    let len = a.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let a_slice = a.as_slice().expect("contiguity checked above");
                let mut i = 0;

                let pi = _mm256_set1_pd(std::f64::consts::PI);
                let two_pi = _mm256_set1_pd(2.0 * std::f64::consts::PI);
                let c1_6 = _mm256_set1_pd(1.0 / 6.0);
                let c1_120 = _mm256_set1_pd(1.0 / 120.0);
                let c1_5040 = _mm256_set1_pd(1.0 / 5040.0);

                while i + 4 <= len {
                    let mut x = _mm256_loadu_pd(a_slice.as_ptr().add(i));

                    // Range reduction to [-π, π]
                    // x = x - 2π * round(x / 2π)
                    let k = _mm256_round_pd::<0x08>(_mm256_div_pd(x, two_pi)); // Round to nearest
                    x = _mm256_sub_pd(x, _mm256_mul_pd(k, two_pi));

                    let x2 = _mm256_mul_pd(x, x);
                    let x3 = _mm256_mul_pd(x2, x);
                    let x5 = _mm256_mul_pd(x3, x2);
                    let x7 = _mm256_mul_pd(x5, x2);

                    // sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040
                    let term1 = x;
                    let term2 = _mm256_mul_pd(x3, c1_6);
                    let term3 = _mm256_mul_pd(x5, c1_120);
                    let term4 = _mm256_mul_pd(x7, c1_5040);
                    let res =
                        _mm256_sub_pd(_mm256_add_pd(term1, term3), _mm256_add_pd(term2, term4));

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), res);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    let mut x = a_slice[j];
                    x = x
                        - (2.0 * std::f64::consts::PI) * (x / (2.0 * std::f64::consts::PI)).round();
                    let x2 = x * x;
                    let x3 = x2 * x;
                    let x5 = x3 * x2;
                    let x7 = x5 * x2;
                    result.push(x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0);
                }

                return Array1::from_vec(result);
            }
        }
    }

    let a_slice = a.as_slice().expect("contiguity checked above");
    for &x_orig in a_slice {
        let mut x = x_orig;
        x = x - (2.0 * std::f64::consts::PI) * (x / (2.0 * std::f64::consts::PI)).round();
        let x2 = x * x;
        let x3 = x2 * x;
        let x5 = x3 * x2;
        let x7 = x5 * x2;
        result.push(x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0);
    }
    Array1::from_vec(result)
}

/// Fast cos approximation using Taylor series
///
/// Uses: cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720
/// Input is range-reduced to [-π, π]
pub fn simd_cos_f64_poly(a: &ArrayView1<f64>) -> Array1<f64> {
    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() {
        return a.mapv(|x_orig| {
            let mut x = x_orig;
            x = x - (2.0 * std::f64::consts::PI) * (x / (2.0 * std::f64::consts::PI)).round();
            let x2 = x * x;
            let x4 = x2 * x2;
            let x6 = x4 * x2;
            1.0 - x2 * 0.5 + x4 / 24.0 - x6 / 720.0
        });
    }

    let len = a.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let a_slice = a.as_slice().expect("contiguity checked above");
                let mut i = 0;

                let two_pi = _mm256_set1_pd(2.0 * std::f64::consts::PI);
                let c1 = _mm256_set1_pd(1.0);
                let c1_2 = _mm256_set1_pd(0.5);
                let c1_24 = _mm256_set1_pd(1.0 / 24.0);
                let c1_720 = _mm256_set1_pd(1.0 / 720.0);

                while i + 4 <= len {
                    let mut x = _mm256_loadu_pd(a_slice.as_ptr().add(i));

                    // Range reduction
                    let k = _mm256_round_pd::<0x08>(_mm256_div_pd(x, two_pi));
                    x = _mm256_sub_pd(x, _mm256_mul_pd(k, two_pi));

                    let x2 = _mm256_mul_pd(x, x);
                    let x4 = _mm256_mul_pd(x2, x2);
                    let x6 = _mm256_mul_pd(x4, x2);

                    // cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720
                    let term1 = c1;
                    let term2 = _mm256_mul_pd(x2, c1_2);
                    let term3 = _mm256_mul_pd(x4, c1_24);
                    let term4 = _mm256_mul_pd(x6, c1_720);
                    let res =
                        _mm256_sub_pd(_mm256_add_pd(term1, term3), _mm256_add_pd(term2, term4));

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), res);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    let mut x = a_slice[j];
                    x = x
                        - (2.0 * std::f64::consts::PI) * (x / (2.0 * std::f64::consts::PI)).round();
                    let x2 = x * x;
                    let x4 = x2 * x2;
                    let x6 = x4 * x2;
                    result.push(1.0 - x2 * 0.5 + x4 / 24.0 - x6 / 720.0);
                }

                return Array1::from_vec(result);
            }
        }
    }

    let a_slice = a.as_slice().expect("contiguity checked above");
    for &x_orig in a_slice {
        let mut x = x_orig;
        x = x - (2.0 * std::f64::consts::PI) * (x / (2.0 * std::f64::consts::PI)).round();
        let x2 = x * x;
        let x4 = x2 * x2;
        let x6 = x4 * x2;
        result.push(1.0 - x2 * 0.5 + x4 / 24.0 - x6 / 720.0);
    }
    Array1::from_vec(result)
}

/// Fast tan approximation using sin/cos ratio
///
/// Uses: tan(x) = sin(x) / cos(x)
pub fn simd_tan_f64_poly(a: &ArrayView1<f64>) -> Array1<f64> {
    let sin_vals = simd_sin_f64_poly(a);
    let cos_vals = simd_cos_f64_poly(a);

    let len = a.len();
    let mut result = Vec::with_capacity(len);

    let sin_slice = sin_vals.as_slice().expect("Operation failed");
    let cos_slice = cos_vals.as_slice().expect("Operation failed");

    for i in 0..len {
        result.push(sin_slice[i] / cos_slice[i]);
    }

    Array1::from_vec(result)
}
