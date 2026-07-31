//! Integer exponentiation with SIMD acceleration (Phase 25)
//!
//! Implements fast integer exponentiation using the exponentiation by squaring algorithm
//! with SIMD acceleration for multiplication operations.

use ndarray::{Array1, ArrayView1};

/// SIMD-accelerated integer exponentiation for f32 arrays
///
/// Computes base^n for each element using exponentiation by squaring algorithm.
///
/// # Arguments
///
/// * `base` - Input array of base values
/// * `n` - Integer exponent
///
/// # Returns
///
/// * Array of power values (`base[i]^n` for each element)
///
/// # Performance
///
/// - Uses exponentiation by squaring: O(log n) multiplications per element
/// - SIMD acceleration for multiplication operations
/// - Speedup: 2-4x over scalar implementation for large arrays
///
/// # Algorithm
///
/// Exponentiation by squaring reduces the number of multiplications from O(n) to O(log n):
/// ```text
/// base^10 = ((base^2)^2) * base^2  # Only 4 multiplications instead of 10
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::simd::unary_powi::simd_powi_f32;
///
/// let base = array![2.0, 3.0, 4.0, 5.0];
/// let result = simd_powi_f32(&base.view(), 3);
/// // Result: [8.0, 27.0, 64.0, 125.0]
/// ```
#[allow(dead_code)]
pub fn simd_powi_f32(base: &ArrayView1<f32>, n: i32) -> Array1<f32> {
    if base.is_empty() {
        return Array1::zeros(0);
    }

    let len = base.len();

    // Special cases for common exponents
    if n == 0 {
        return Array1::from_elem(len, 1.0);
    }
    if n == 1 {
        return base.to_owned();
    }
    if n == 2 {
        // x^2 can be optimized as x*x
        return simd_square_f32(base);
    }

    // Handle negative exponent: base^(-n) = (1/base)^n
    let (actual_base, actual_n) = if n < 0 {
        let reciprocal = base.mapv(|x| 1.0 / x);
        (reciprocal, n.wrapping_neg() as u32)
    } else {
        (base.to_owned(), n as u32)
    };

    // Pre-allocate result array (optimization from SIMD report)
    let mut result = Array1::from_elem(len, 1.0);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let result_slice = result.as_slice_mut().expect("Operation failed");

                // Exponentiation by squaring with SIMD
                let mut current_power = actual_base.clone();
                let mut current_power_slice =
                    current_power.as_slice_mut().expect("Operation failed");
                let mut exp = actual_n;

                while exp > 0 {
                    // If exp is odd, multiply result by current_power
                    if exp & 1 == 1 {
                        let mut i = 0;
                        while i + 8 <= len {
                            let result_vec = _mm256_loadu_ps(result_slice.as_ptr().add(i));
                            let power_vec = _mm256_loadu_ps(current_power_slice.as_ptr().add(i));
                            let mul_vec = _mm256_mul_ps(result_vec, power_vec);
                            _mm256_storeu_ps(result_slice.as_mut_ptr().add(i), mul_vec);
                            i += 8;
                        }
                        // Scalar remainder
                        for j in i..len {
                            result_slice[j] *= current_power_slice[j];
                        }
                    }

                    // Square current_power for next iteration
                    exp >>= 1;
                    if exp > 0 {
                        let mut i = 0;
                        while i + 8 <= len {
                            let power_vec = _mm256_loadu_ps(current_power_slice.as_ptr().add(i));
                            let squared = _mm256_mul_ps(power_vec, power_vec);
                            _mm256_storeu_ps(current_power_slice.as_mut_ptr().add(i), squared);
                            i += 8;
                        }
                        // Scalar remainder
                        for j in i..len {
                            current_power_slice[j] *= current_power_slice[j];
                        }
                    }
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

                let result_slice = result.as_slice_mut().expect("Operation failed");

                let mut current_power = actual_base.clone();
                let mut current_power_slice =
                    current_power.as_slice_mut().expect("Operation failed");
                let mut exp = actual_n;

                while exp > 0 {
                    if exp & 1 == 1 {
                        let mut i = 0;
                        while i + 4 <= len {
                            let result_vec = vld1q_f32(result_slice.as_ptr().add(i));
                            let power_vec = vld1q_f32(current_power_slice.as_ptr().add(i));
                            let mul_vec = vmulq_f32(result_vec, power_vec);
                            vst1q_f32(result_slice.as_mut_ptr().add(i), mul_vec);
                            i += 4;
                        }
                        for j in i..len {
                            result_slice[j] *= current_power_slice[j];
                        }
                    }

                    exp >>= 1;
                    if exp > 0 {
                        let mut i = 0;
                        while i + 4 <= len {
                            let power_vec = vld1q_f32(current_power_slice.as_ptr().add(i));
                            let squared = vmulq_f32(power_vec, power_vec);
                            vst1q_f32(current_power_slice.as_mut_ptr().add(i), squared);
                            i += 4;
                        }
                        for j in i..len {
                            current_power_slice[j] *= current_power_slice[j];
                        }
                    }
                }

                return result;
            }
        }
    }

    // Scalar fallback using exponentiation by squaring
    let mut current_power = actual_base;
    let mut exp = actual_n;

    while exp > 0 {
        if exp & 1 == 1 {
            for i in 0..len {
                result[i] *= current_power[i];
            }
        }
        exp >>= 1;
        if exp > 0 {
            for i in 0..len {
                current_power[i] *= current_power[i];
            }
        }
    }

    result
}

/// SIMD-accelerated integer exponentiation for f64 arrays
///
/// Computes base^n for each element using exponentiation by squaring algorithm.
///
/// # Arguments
///
/// * `base` - Input array of base values
/// * `n` - Integer exponent
///
/// # Returns
///
/// * Array of power values (`base[i]^n` for each element)
#[allow(dead_code)]
pub fn simd_powi_f64(base: &ArrayView1<f64>, n: i32) -> Array1<f64> {
    if base.is_empty() {
        return Array1::zeros(0);
    }

    let len = base.len();

    // Special cases
    if n == 0 {
        return Array1::from_elem(len, 1.0);
    }
    if n == 1 {
        return base.to_owned();
    }
    if n == 2 {
        return simd_square_f64(base);
    }

    // Handle negative exponent
    let (actual_base, actual_n) = if n < 0 {
        let reciprocal = base.mapv(|x| 1.0 / x);
        (reciprocal, n.wrapping_neg() as u32)
    } else {
        (base.to_owned(), n as u32)
    };

    let mut result = Array1::from_elem(len, 1.0);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let result_slice = result.as_slice_mut().expect("Operation failed");

                let mut current_power = actual_base.clone();
                let mut current_power_slice =
                    current_power.as_slice_mut().expect("Operation failed");
                let mut exp = actual_n;

                while exp > 0 {
                    if exp & 1 == 1 {
                        let mut i = 0;
                        while i + 4 <= len {
                            let result_vec = _mm256_loadu_pd(result_slice.as_ptr().add(i));
                            let power_vec = _mm256_loadu_pd(current_power_slice.as_ptr().add(i));
                            let mul_vec = _mm256_mul_pd(result_vec, power_vec);
                            _mm256_storeu_pd(result_slice.as_mut_ptr().add(i), mul_vec);
                            i += 4;
                        }
                        for j in i..len {
                            result_slice[j] *= current_power_slice[j];
                        }
                    }

                    exp >>= 1;
                    if exp > 0 {
                        let mut i = 0;
                        while i + 4 <= len {
                            let power_vec = _mm256_loadu_pd(current_power_slice.as_ptr().add(i));
                            let squared = _mm256_mul_pd(power_vec, power_vec);
                            _mm256_storeu_pd(current_power_slice.as_mut_ptr().add(i), squared);
                            i += 4;
                        }
                        for j in i..len {
                            current_power_slice[j] *= current_power_slice[j];
                        }
                    }
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

                let result_slice = result.as_slice_mut().expect("Operation failed");

                let mut current_power = actual_base.clone();
                let mut current_power_slice =
                    current_power.as_slice_mut().expect("Operation failed");
                let mut exp = actual_n;

                while exp > 0 {
                    if exp & 1 == 1 {
                        let mut i = 0;
                        while i + 2 <= len {
                            let result_vec = vld1q_f64(result_slice.as_ptr().add(i));
                            let power_vec = vld1q_f64(current_power_slice.as_ptr().add(i));
                            let mul_vec = vmulq_f64(result_vec, power_vec);
                            vst1q_f64(result_slice.as_mut_ptr().add(i), mul_vec);
                            i += 2;
                        }
                        for j in i..len {
                            result_slice[j] *= current_power_slice[j];
                        }
                    }

                    exp >>= 1;
                    if exp > 0 {
                        let mut i = 0;
                        while i + 2 <= len {
                            let power_vec = vld1q_f64(current_power_slice.as_ptr().add(i));
                            let squared = vmulq_f64(power_vec, power_vec);
                            vst1q_f64(current_power_slice.as_mut_ptr().add(i), squared);
                            i += 2;
                        }
                        for j in i..len {
                            current_power_slice[j] *= current_power_slice[j];
                        }
                    }
                }

                return result;
            }
        }
    }

    // Scalar fallback
    let mut current_power = actual_base;
    let mut exp = actual_n;

    while exp > 0 {
        if exp & 1 == 1 {
            for i in 0..len {
                result[i] *= current_power[i];
            }
        }
        exp >>= 1;
        if exp > 0 {
            for i in 0..len {
                current_power[i] *= current_power[i];
            }
        }
    }

    result
}

/// Helper: SIMD-accelerated square for f32 arrays (x^2 = x*x optimization)
#[inline]
fn simd_square_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x * x);
    }

    let len = input.len();
    let mut result = Array1::zeros(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice: &mut [f32] = result.as_slice_mut().expect("Operation failed");

                let mut i = 0;
                while i + 8 <= len {
                    let vec = _mm256_loadu_ps(input_slice.as_ptr().add(i));
                    let squared = _mm256_mul_ps(vec, vec);
                    _mm256_storeu_ps(result_slice.as_mut_ptr().add(i), squared);
                    i += 8;
                }
                for j in i..len {
                    result_slice[j] = input_slice[j] * input_slice[j];
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
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice: &mut [f32] = result.as_slice_mut().expect("Operation failed");

                let mut i = 0;
                while i + 4 <= len {
                    let vec = vld1q_f32(input_slice.as_ptr().add(i));
                    let squared = vmulq_f32(vec, vec);
                    vst1q_f32(result_slice.as_mut_ptr().add(i), squared);
                    i += 4;
                }
                for j in i..len {
                    result_slice[j] = input_slice[j] * input_slice[j];
                }
                return result;
            }
        }
    }

    // Scalar fallback
    result.zip_mut_with(input, |r, &x| *r = x * x);
    result
}

/// Helper: SIMD-accelerated square for f64 arrays
#[inline]
fn simd_square_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x * x);
    }

    let len = input.len();
    let mut result = Array1::zeros(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice: &mut [f64] = result.as_slice_mut().expect("Operation failed");

                let mut i = 0;
                while i + 4 <= len {
                    let vec = _mm256_loadu_pd(input_slice.as_ptr().add(i));
                    let squared = _mm256_mul_pd(vec, vec);
                    _mm256_storeu_pd(result_slice.as_mut_ptr().add(i), squared);
                    i += 4;
                }
                for j in i..len {
                    result_slice[j] = input_slice[j] * input_slice[j];
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
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice: &mut [f64] = result.as_slice_mut().expect("Operation failed");

                let mut i = 0;
                while i + 2 <= len {
                    let vec = vld1q_f64(input_slice.as_ptr().add(i));
                    let squared = vmulq_f64(vec, vec);
                    vst1q_f64(result_slice.as_mut_ptr().add(i), squared);
                    i += 2;
                }
                for j in i..len {
                    result_slice[j] = input_slice[j] * input_slice[j];
                }
                return result;
            }
        }
    }

    // Scalar fallback
    result.zip_mut_with(input, |r, &x| *r = x * x);
    result
}
