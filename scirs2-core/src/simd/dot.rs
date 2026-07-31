//! Dot product and FMA operations with SIMD acceleration
//!
//! This module provides optimized implementations of dot product (inner product)
//! and fused multiply-add (FMA) operations using SIMD instructions.

use super::detect::get_cpu_features;
use ndarray::{Array1, ArrayView1};

/// Adaptive SIMD dot product that selects implementation based on array size
///
/// For very large arrays (>= 500,000 elements), uses ultra-optimized implementation.
/// For smaller arrays, uses standard implementation for better cache utilization.
///
/// # Arguments
///
/// * `a` - First input array
/// * `b` - Second input array
///
/// # Returns
///
/// * Dot product of the two arrays
#[allow(dead_code)]
pub fn simd_dot_f32_adaptive(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    let len = a.len();

    if len >= 500_000 {
        // Very large arrays: Use ultra-optimized dot product
        simd_dot_f32_ultra(a, b)
    } else {
        // Smaller arrays: Use standard implementation (known to work)
        simd_dot_f32(a, b)
    }
}

/// PHASE 3: High-performance SIMD dot product with horizontal reduction optimization
#[allow(dead_code)]
pub fn simd_dot_f32_ultra(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() || b.as_slice().is_none() {
        return a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    }

    let len = a.len();
    let a_ptr = a.as_slice().expect("contiguity checked above").as_ptr();
    let b_ptr = b.as_slice().expect("contiguity checked above").as_ptr();

    let features = get_cpu_features();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if features.has_avx2 {
            unsafe {
                let mut sums = [_mm256_setzero_ps(); 4]; // 4 accumulator registers
                let mut i = 0;

                // Process in chunks with multiple accumulators to avoid dependency chains
                while i + 32 <= len {
                    // Prefetch ahead
                    if i + 256 < len {
                        _mm_prefetch(a_ptr.add(i + 256) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(b_ptr.add(i + 256) as *const i8, _MM_HINT_T0);
                    }

                    // Load and multiply 4 pairs simultaneously
                    let a_vec1 = _mm256_loadu_ps(a_ptr.add(i));
                    let a_vec2 = _mm256_loadu_ps(a_ptr.add(i + 8));
                    let a_vec3 = _mm256_loadu_ps(a_ptr.add(i + 16));
                    let a_vec4 = _mm256_loadu_ps(a_ptr.add(i + 24));

                    let b_vec1 = _mm256_loadu_ps(b_ptr.add(i));
                    let b_vec2 = _mm256_loadu_ps(b_ptr.add(i + 8));
                    let b_vec3 = _mm256_loadu_ps(b_ptr.add(i + 16));
                    let b_vec4 = _mm256_loadu_ps(b_ptr.add(i + 24));

                    // Use FMA if available, otherwise separate mul+add
                    if features.has_fma {
                        sums[0] = _mm256_fmadd_ps(a_vec1, b_vec1, sums[0]);
                        sums[1] = _mm256_fmadd_ps(a_vec2, b_vec2, sums[1]);
                        sums[2] = _mm256_fmadd_ps(a_vec3, b_vec3, sums[2]);
                        sums[3] = _mm256_fmadd_ps(a_vec4, b_vec4, sums[3]);
                    } else {
                        let products1 = _mm256_mul_ps(a_vec1, b_vec1);
                        let products2 = _mm256_mul_ps(a_vec2, b_vec2);
                        let products3 = _mm256_mul_ps(a_vec3, b_vec3);
                        let products4 = _mm256_mul_ps(a_vec4, b_vec4);

                        sums[0] = _mm256_add_ps(sums[0], products1);
                        sums[1] = _mm256_add_ps(sums[1], products2);
                        sums[2] = _mm256_add_ps(sums[2], products3);
                        sums[3] = _mm256_add_ps(sums[3], products4);
                    }

                    i += 32;
                }

                // Combine all accumulators
                let combined1 = _mm256_add_ps(sums[0], sums[1]);
                let combined2 = _mm256_add_ps(sums[2], sums[3]);
                let final_sum = _mm256_add_ps(combined1, combined2);

                // Horizontal reduction
                let high = _mm256_extractf128_ps(final_sum, 1);
                let low = _mm256_castps256_ps128(final_sum);
                let sum128 = _mm_add_ps(low, high);

                let shuf = _mm_shuffle_ps(sum128, sum128, 0b1110);
                let sum_partial = _mm_add_ps(sum128, shuf);
                let shuf2 = _mm_shuffle_ps(sum_partial, sum_partial, 0b0001);
                let final_result = _mm_add_ps(sum_partial, shuf2);

                let mut result = _mm_cvtss_f32(final_result);

                // Handle remaining elements
                while i < len {
                    result += *a_ptr.add(i) * *b_ptr.add(i);
                    i += 1;
                }

                return result;
            }
        }
    }

    // Fallback to existing implementation
    #[cfg(not(target_arch = "x86_64"))]
    {
        simd_dot_f32(a, b)
    }

    #[cfg(target_arch = "x86_64")]
    {
        simd_dot_f32(a, b)
    }
}

/// PHASE 3: High-performance FMA (Fused Multiply-Add) operation
#[allow(dead_code)]
pub fn simd_fma_f32_ultra(
    a: &ArrayView1<f32>,
    b: &ArrayView1<f32>,
    c: &ArrayView1<f32>,
) -> Array1<f32> {
    assert_eq!(a.len(), b.len(), "Arrays a and b must have the same length");
    assert_eq!(a.len(), c.len(), "Arrays a and c must have the same length");

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() || b.as_slice().is_none() || c.as_slice().is_none() {
        return Array1::from_iter(
            a.iter()
                .zip(b.iter())
                .zip(c.iter())
                .map(|((&x, &y), &z)| x * y + z),
        );
    }

    let len = a.len();
    let mut result = vec![0.0f32; len];

    let a_ptr = a.as_slice().expect("contiguity checked above").as_ptr();
    let b_ptr = b.as_slice().expect("contiguity checked above").as_ptr();
    let c_ptr = c.as_slice().expect("contiguity checked above").as_ptr();
    let result_ptr = result.as_mut_ptr();

    let features = get_cpu_features();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if features.has_fma && features.has_avx2 {
            unsafe {
                let mut i = 0;

                // Optimized FMA path with prefetching
                while i + 32 <= len {
                    if i + 256 < len {
                        _mm_prefetch(a_ptr.add(i + 256) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(b_ptr.add(i + 256) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(c_ptr.add(i + 256) as *const i8, _MM_HINT_T0);
                    }

                    // 4x unrolled FMA operations
                    for j in (0..32).step_by(8) {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i + j));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i + j));
                        let c_vec = _mm256_loadu_ps(c_ptr.add(i + j));

                        // Single instruction: result = a * b + c
                        let result_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                        _mm256_storeu_ps(result_ptr.add(i + j), result_vec);
                    }

                    i += 32;
                }

                // Handle remaining elements
                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                    let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                    let c_vec = _mm256_loadu_ps(c_ptr.add(i));
                    let result_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                    _mm256_storeu_ps(result_ptr.add(i), result_vec);
                    i += 8;
                }

                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i) + *c_ptr.add(i);
                    i += 1;
                }
            }
        } else {
            // Fallback: separate multiply and add
            unsafe {
                let mut i = 0;
                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                    let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                    let c_vec = _mm256_loadu_ps(c_ptr.add(i));

                    let product_vec = _mm256_mul_ps(a_vec, b_vec);
                    let result_vec = _mm256_add_ps(product_vec, c_vec);
                    _mm256_storeu_ps(result_ptr.add(i), result_vec);
                    i += 8;
                }

                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i) + *c_ptr.add(i);
                    i += 1;
                }
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Scalar fallback for other architectures
        unsafe {
            for i in 0..len {
                *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i) + *c_ptr.add(i);
            }
        }
    }

    Array1::from_vec(result)
}

/// Fast multiplication implementation with optimized memory access
#[allow(dead_code)]
pub fn simd_mul_f32_fast(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() || b.as_slice().is_none() {
        return (a * b).to_owned();
    }

    let len = a.len();
    let mut result = vec![0.0f32; len];

    let a_ptr = a.as_slice().expect("contiguity checked above").as_ptr();
    let b_ptr = b.as_slice().expect("contiguity checked above").as_ptr();
    let result_ptr = result.as_mut_ptr();

    let features = get_cpu_features();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if features.has_avx2 {
            unsafe {
                let mut i = 0;

                // Phase 2: Loop unrolling for multiplication
                while i + 32 <= len {
                    let a_vec1 = _mm256_loadu_ps(a_ptr.add(i));
                    let a_vec2 = _mm256_loadu_ps(a_ptr.add(i + 8));
                    let a_vec3 = _mm256_loadu_ps(a_ptr.add(i + 16));
                    let a_vec4 = _mm256_loadu_ps(a_ptr.add(i + 24));

                    let b_vec1 = _mm256_loadu_ps(b_ptr.add(i));
                    let b_vec2 = _mm256_loadu_ps(b_ptr.add(i + 8));
                    let b_vec3 = _mm256_loadu_ps(b_ptr.add(i + 16));
                    let b_vec4 = _mm256_loadu_ps(b_ptr.add(i + 24));

                    let result_vec1 = _mm256_mul_ps(a_vec1, b_vec1);
                    let result_vec2 = _mm256_mul_ps(a_vec2, b_vec2);
                    let result_vec3 = _mm256_mul_ps(a_vec3, b_vec3);
                    let result_vec4 = _mm256_mul_ps(a_vec4, b_vec4);

                    _mm256_storeu_ps(result_ptr.add(i), result_vec1);
                    _mm256_storeu_ps(result_ptr.add(i + 8), result_vec2);
                    _mm256_storeu_ps(result_ptr.add(i + 16), result_vec3);
                    _mm256_storeu_ps(result_ptr.add(i + 24), result_vec4);

                    i += 32;
                }

                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                    let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                    let result_vec = _mm256_mul_ps(a_vec, b_vec);
                    _mm256_storeu_ps(result_ptr.add(i), result_vec);
                    i += 8;
                }

                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                    i += 1;
                }
            }
        } else {
            unsafe {
                for i in 0..len {
                    *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                }
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        unsafe {
            for i in 0..len {
                *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
            }
        }
    }

    Array1::from_vec(result)
}

/// Compute element-wise subtraction of two f32 arrays using unified SIMD operations
#[allow(dead_code)]
pub fn simd_sub_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    // Direct implementation to avoid circular dependency
    (a - b).to_owned()
}

/// Compute element-wise subtraction of two f64 arrays using SIMD operations
#[allow(dead_code)]
pub fn simd_sub_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    let len = a.len();

    // Fall back to scalar subtraction if arrays are non-contiguous (e.g. column slices)
    if a.as_slice().is_none() || b.as_slice().is_none() {
        return (a - b).to_owned();
    }

    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;
                // Process 4 f64s at a time with AVX2
                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];

                    let a_vec = _mm256_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_pd(b_slice.as_ptr());
                    let result_vec = _mm256_sub_pd(a_vec, b_vec);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] - b[j]);
                }
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let mut i = 0;
                // Process 2 f64s at a time with SSE2
                while i + 2 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 2];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 2];

                    let a_vec = _mm_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm_loadu_pd(b_slice.as_ptr());
                    let result_vec = _mm_sub_pd(a_vec, b_vec);

                    let mut temp = [0.0f64; 2];
                    _mm_storeu_pd(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] - b[j]);
                }
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..len {
                result.push(a[i] - b[i]);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let mut i = 0;
                // Process 2 f64s at a time with NEON
                while i + 2 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 2];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 2];

                    let a_vec = vld1q_f64(a_slice.as_ptr());
                    let b_vec = vld1q_f64(b_slice.as_ptr());
                    let result_vec = vsubq_f64(a_vec, b_vec);

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] - b[j]);
                }
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..len {
                result.push(a[i] - b[i]);
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Fallback to scalar implementation for other architectures
        for i in 0..len {
            result.push(a[i] - b[i]);
        }
    }

    Array1::from_vec(result)
}

/// Compute element-wise multiplication of two f32 arrays using unified SIMD operations
#[allow(dead_code)]
pub fn simd_mul_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() || b.as_slice().is_none() {
        return (a * b).to_owned();
    }

    let len = a.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;
                // Process 8 f32s at a time with AVX2
                while i + 8 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 8];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 8];

                    let a_vec = _mm256_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_ps(b_slice.as_ptr());
                    let result_vec = _mm256_mul_ps(a_vec, b_vec);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] * b[j]);
                }
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let mut i = 0;
                // Process 4 f32s at a time with SSE
                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];

                    let a_vec = _mm_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm_loadu_ps(b_slice.as_ptr());
                    let result_vec = _mm_mul_ps(a_vec, b_vec);

                    let mut temp = [0.0f32; 4];
                    _mm_storeu_ps(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] * b[j]);
                }
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..len {
                result.push(a[i] * b[i]);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let mut i = 0;
                // Process 4 f32s at a time with NEON
                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];

                    let a_vec = vld1q_f32(a_slice.as_ptr());
                    let b_vec = vld1q_f32(b_slice.as_ptr());
                    let result_vec = vmulq_f32(a_vec, b_vec);

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] * b[j]);
                }
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..len {
                result.push(a[i] * b[i]);
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Fallback to scalar implementation for other architectures
        for i in 0..len {
            result.push(a[i] * b[i]);
        }
    }

    Array1::from_vec(result)
}

/// Compute element-wise multiplication of two f64 arrays using unified SIMD operations
#[allow(dead_code)]
pub fn simd_mul_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() || b.as_slice().is_none() {
        return (a * b).to_owned();
    }

    let len = a.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;
                // Process 4 f64s at a time with AVX2
                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];

                    let a_vec = _mm256_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_pd(b_slice.as_ptr());
                    let result_vec = _mm256_mul_pd(a_vec, b_vec);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] * b[j]);
                }
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let mut i = 0;
                // Process 2 f64s at a time with SSE2
                while i + 2 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 2];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 2];

                    let a_vec = _mm_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm_loadu_pd(b_slice.as_ptr());
                    let result_vec = _mm_mul_pd(a_vec, b_vec);

                    let mut temp = [0.0f64; 2];
                    _mm_storeu_pd(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] * b[j]);
                }
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..len {
                result.push(a[i] * b[i]);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let mut i = 0;
                // Process 2 f64s at a time with NEON
                while i + 2 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 2];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 2];

                    let a_vec = vld1q_f64(a_slice.as_ptr());
                    let b_vec = vld1q_f64(b_slice.as_ptr());
                    let result_vec = vmulq_f64(a_vec, b_vec);

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] * b[j]);
                }
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..len {
                result.push(a[i] * b[i]);
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Fallback to scalar implementation for other architectures
        for i in 0..len {
            result.push(a[i] * b[i]);
        }
    }

    Array1::from_vec(result)
}

/// Compute element-wise division of two f32 arrays using SIMD operations
#[allow(dead_code)]
pub fn simd_div_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() || b.as_slice().is_none() {
        return (a / b).to_owned();
    }

    let len = a.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;
                // Process 8 f32s at a time with AVX2
                while i + 8 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 8];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 8];

                    let a_vec = _mm256_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_ps(b_slice.as_ptr());
                    let result_vec = _mm256_div_ps(a_vec, b_vec);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] / b[j]);
                }
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let mut i = 0;
                // Process 4 f32s at a time with SSE
                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];

                    let a_vec = _mm_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm_loadu_ps(b_slice.as_ptr());
                    let result_vec = _mm_div_ps(a_vec, b_vec);

                    let mut temp = [0.0f32; 4];
                    _mm_storeu_ps(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] / b[j]);
                }
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..len {
                result.push(a[i] / b[i]);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let mut i = 0;
                // Process 4 f32s at a time with NEON
                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];

                    let a_vec = vld1q_f32(a_slice.as_ptr());
                    let b_vec = vld1q_f32(b_slice.as_ptr());
                    let result_vec = vdivq_f32(a_vec, b_vec);

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] / b[j]);
                }
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..len {
                result.push(a[i] / b[i]);
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Fallback to scalar implementation for other architectures
        for i in 0..len {
            result.push(a[i] / b[i]);
        }
    }

    Array1::from_vec(result)
}

/// Compute element-wise division of two f64 arrays using SIMD operations
#[allow(dead_code)]
pub fn simd_div_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() || b.as_slice().is_none() {
        return (a / b).to_owned();
    }

    let len = a.len();
    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;
                // Process 4 f64s at a time with AVX2
                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];

                    let a_vec = _mm256_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_pd(b_slice.as_ptr());
                    let result_vec = _mm256_div_pd(a_vec, b_vec);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] / b[j]);
                }
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let mut i = 0;
                // Process 2 f64s at a time with SSE2
                while i + 2 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 2];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 2];

                    let a_vec = _mm_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm_loadu_pd(b_slice.as_ptr());
                    let result_vec = _mm_div_pd(a_vec, b_vec);

                    let mut temp = [0.0f64; 2];
                    _mm_storeu_pd(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] / b[j]);
                }
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..len {
                result.push(a[i] / b[i]);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let mut i = 0;
                // Process 2 f64s at a time with NEON
                while i + 2 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 2];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 2];

                    let a_vec = vld1q_f64(a_slice.as_ptr());
                    let b_vec = vld1q_f64(b_slice.as_ptr());
                    let result_vec = vdivq_f64(a_vec, b_vec);

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] / b[j]);
                }
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..len {
                result.push(a[i] / b[i]);
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Fallback to scalar implementation for other architectures
        for i in 0..len {
            result.push(a[i] / b[i]);
        }
    }

    Array1::from_vec(result)
}

/// Compute dot product of two f32 arrays using unified SIMD operations
#[allow(dead_code)]
pub fn simd_dot_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() || b.as_slice().is_none() {
        return a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    }

    let len = a.len();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut sums = _mm256_setzero_ps();
                let mut i = 0;

                // Process 8 f32s at a time with AVX2
                while i + 8 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 8];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 8];

                    let a_vec = _mm256_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_ps(b_slice.as_ptr());
                    let product = _mm256_mul_ps(a_vec, b_vec);
                    sums = _mm256_add_ps(sums, product);
                    i += 8;
                }

                // Horizontal sum of the AVX2 register
                let high = _mm256_extractf128_ps(sums, 1);
                let low = _mm256_castps256_ps128(sums);
                let sum128 = _mm_add_ps(low, high);

                // Sum the 4 elements in the 128-bit register
                let shuf = _mm_shuffle_ps(sum128, sum128, 0b1110);
                let sum_partial = _mm_add_ps(sum128, shuf);
                let shuf2 = _mm_shuffle_ps(sum_partial, sum_partial, 0b0001);
                let final_sum = _mm_add_ps(sum_partial, shuf2);

                let mut result = _mm_cvtss_f32(final_sum);

                // Handle remaining elements
                for j in i..len {
                    result += a[j] * b[j];
                }

                result
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let mut sums = _mm_setzero_ps();
                let mut i = 0;

                // Process 4 f32s at a time with SSE
                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];

                    let a_vec = _mm_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm_loadu_ps(b_slice.as_ptr());
                    let product = _mm_mul_ps(a_vec, b_vec);
                    sums = _mm_add_ps(sums, product);
                    i += 4;
                }

                // Horizontal sum of the SSE register
                let shuf = _mm_shuffle_ps(sums, sums, 0b1110);
                let sum_partial = _mm_add_ps(sums, shuf);
                let shuf2 = _mm_shuffle_ps(sum_partial, sum_partial, 0b0001);
                let final_sum = _mm_add_ps(sum_partial, shuf2);

                let mut result = _mm_cvtss_f32(final_sum);

                // Handle remaining elements
                for j in i..len {
                    result += a[j] * b[j];
                }

                result
            }
        } else {
            // Fallback to scalar implementation
            let mut sum = 0.0f32;
            for i in 0..len {
                sum += a[i] * b[i];
            }
            sum
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let mut sum_vec = vdupq_n_f32(0.0);
                let mut i = 0;

                // Process 4 f32s at a time with NEON
                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("Test operation failed")[i..i + 4];
                    let b_slice = &b.as_slice().expect("Test operation failed")[i..i + 4];

                    let a_vec = vld1q_f32(a_slice.as_ptr());
                    let b_vec = vld1q_f32(b_slice.as_ptr());
                    sum_vec = vfmaq_f32(sum_vec, a_vec, b_vec);
                    i += 4;
                }

                // Horizontal sum of NEON register
                let sum_pair = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
                let final_sum = vpadd_f32(sum_pair, sum_pair);
                let mut result = vget_lane_f32(final_sum, 0);

                // Handle remaining elements
                for j in i..len {
                    result += a[j] * b[j];
                }

                result
            }
        } else {
            // Fallback to scalar implementation
            let mut sum = 0.0f32;
            for i in 0..len {
                sum += a[i] * b[i];
            }
            sum
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Fallback to scalar implementation for other architectures
        let mut sum = 0.0f32;
        for i in 0..len {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// Compute dot product of two f64 arrays using SIMD operations
///
/// This function uses hardware SIMD instructions (AVX2/SSE on x86_64, NEON on ARM)
/// to accelerate dot product computation for f64 arrays.
///
/// # Arguments
///
/// * `a` - First input array
/// * `b` - Second input array
///
/// # Returns
///
/// * Dot product of the two arrays
///
/// # Panics
///
/// Panics if arrays have different lengths.
#[allow(dead_code)]
pub fn simd_dot_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    let len = a.len();

    // Ensure contiguous memory for SIMD operations.
    // Column views from 2D arrays are strided and as_slice() returns None.
    let a_buf: Vec<f64>;
    let a_slice = match a.as_slice() {
        Some(s) => s,
        None => {
            a_buf = a.iter().copied().collect();
            &a_buf
        }
    };
    let b_buf: Vec<f64>;
    let b_slice = match b.as_slice() {
        Some(s) => s,
        None => {
            b_buf = b.iter().copied().collect();
            &b_buf
        }
    };

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut sums = _mm256_setzero_pd();
                let mut i = 0;

                while i + 4 <= len {
                    let a_vec = _mm256_loadu_pd(a_slice.as_ptr().add(i));
                    let b_vec = _mm256_loadu_pd(b_slice.as_ptr().add(i));

                    #[cfg(target_feature = "fma")]
                    {
                        sums = _mm256_fmadd_pd(a_vec, b_vec, sums);
                    }
                    #[cfg(not(target_feature = "fma"))]
                    {
                        let product = _mm256_mul_pd(a_vec, b_vec);
                        sums = _mm256_add_pd(sums, product);
                    }
                    i += 4;
                }

                let high = _mm256_extractf128_pd(sums, 1);
                let low = _mm256_castpd256_pd128(sums);
                let sum128 = _mm_add_pd(low, high);
                let high64 = _mm_unpackhi_pd(sum128, sum128);
                let final_sum = _mm_add_sd(sum128, high64);
                let mut result = _mm_cvtsd_f64(final_sum);

                for j in i..len {
                    result += a_slice[j] * b_slice[j];
                }

                result
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let mut sums = _mm_setzero_pd();
                let mut i = 0;

                while i + 2 <= len {
                    let a_vec = _mm_loadu_pd(a_slice.as_ptr().add(i));
                    let b_vec = _mm_loadu_pd(b_slice.as_ptr().add(i));
                    let product = _mm_mul_pd(a_vec, b_vec);
                    sums = _mm_add_pd(sums, product);
                    i += 2;
                }

                let high64 = _mm_unpackhi_pd(sums, sums);
                let final_sum = _mm_add_sd(sums, high64);
                let mut result = _mm_cvtsd_f64(final_sum);

                for j in i..len {
                    result += a_slice[j] * b_slice[j];
                }

                result
            }
        } else {
            let mut sum0 = 0.0f64;
            let mut sum1 = 0.0f64;
            let mut sum2 = 0.0f64;
            let mut sum3 = 0.0f64;
            let mut i = 0;

            while i + 4 <= len {
                sum0 += a_slice[i] * b_slice[i];
                sum1 += a_slice[i + 1] * b_slice[i + 1];
                sum2 += a_slice[i + 2] * b_slice[i + 2];
                sum3 += a_slice[i + 3] * b_slice[i + 3];
                i += 4;
            }

            let mut result = sum0 + sum1 + sum2 + sum3;
            for j in i..len {
                result += a_slice[j] * b_slice[j];
            }
            result
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let mut sum_vec = vdupq_n_f64(0.0);
                let mut i = 0;

                while i + 2 <= len {
                    let a_vec = vld1q_f64(a_slice.as_ptr().add(i));
                    let b_vec = vld1q_f64(b_slice.as_ptr().add(i));
                    sum_vec = vfmaq_f64(sum_vec, a_vec, b_vec);
                    i += 2;
                }

                let low = vgetq_lane_f64(sum_vec, 0);
                let high = vgetq_lane_f64(sum_vec, 1);
                let mut result = low + high;

                for j in i..len {
                    result += a_slice[j] * b_slice[j];
                }

                result
            }
        } else {
            let mut sum0 = 0.0f64;
            let mut sum1 = 0.0f64;
            let mut i = 0;

            while i + 2 <= len {
                sum0 += a_slice[i] * b_slice[i];
                sum1 += a_slice[i + 1] * b_slice[i + 1];
                i += 2;
            }

            let mut result = sum0 + sum1;
            for j in i..len {
                result += a_slice[j] * b_slice[j];
            }
            result
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut sum0 = 0.0f64;
        let mut sum1 = 0.0f64;
        let mut sum2 = 0.0f64;
        let mut sum3 = 0.0f64;
        let mut i = 0;

        while i + 4 <= len {
            sum0 += a_slice[i] * b_slice[i];
            sum1 += a_slice[i + 1] * b_slice[i + 1];
            sum2 += a_slice[i + 2] * b_slice[i + 2];
            sum3 += a_slice[i + 3] * b_slice[i + 3];
            i += 4;
        }

        let mut result = sum0 + sum1 + sum2 + sum3;
        for j in i..len {
            result += a_slice[j] * b_slice[j];
        }
        result
    }
}
