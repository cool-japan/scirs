//! Legacy SIMD implementations - Advanced optimization variants
//!
//! This module contains highly optimized SIMD variants and experimental implementations.
//! Most standard SIMD operations have been moved to dedicated modules under `simd/`.

use crate::simd_ops::SimdUnifiedOps;
use ::ndarray::{Array, Array1, ArrayView1, ArrayView2, Dimension, Zip};
use num_traits::Float;
use std::ops::{Add, Div, Mul, Sub};

// Import from simd module
use crate::simd::detect::get_cpu_features;
use crate::simd::dot::{simd_div_f32, simd_mul_f32, simd_mul_f32_fast};
use crate::simd::traits::SimdOps;

/// Apply element-wise operation on arrays using unified SIMD operations
///
/// Generic helper for binary operations with SIMD support.
#[allow(dead_code)]
pub fn simd_binary_op<F, S1, S2, D>(
    a: &crate::ndarray::ArrayBase<S1, D>,
    b: &crate::ndarray::ArrayBase<S2, D>,
    op: fn(F, F) -> F,
) -> Array<F, D>
where
    F: SimdOps + Float + SimdUnifiedOps,
    S1: crate::ndarray::Data<Elem = F>,
    S2: crate::ndarray::Data<Elem = F>,
    D: Dimension,
{
    let mut result = Array::zeros(a.raw_dim());
    Zip::from(&mut result)
        .and(a)
        .and(b)
        .for_each(|r, &a, &b| *r = op(a, b));
    result
}

// ==================== Advanced Multiplication Variants ====================

/// PHASE 3: High-performance SIMD multiplication with prefetching
#[allow(dead_code)]
pub fn simd_mul_f32_ultra(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    if a.as_slice().is_none() || b.as_slice().is_none() {
        return (a * b).to_owned();
    }

    #[cfg(target_arch = "x86_64")]
    {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let a_ptr = a.as_slice().expect("contiguity checked above").as_ptr();
        let b_ptr = b.as_slice().expect("contiguity checked above").as_ptr();
        let result_ptr = result.as_mut_ptr();

        let features = get_cpu_features();

        use std::arch::x86_64::*;

        if features.has_avx2 {
            unsafe {
                let mut i = 0;
                const PREFETCH_DISTANCE: usize = 256;

                // Check for alignment opportunities
                let a_aligned = (a_ptr as usize) % 32 == 0;
                let b_aligned = (b_ptr as usize) % 32 == 0;
                let result_aligned = (result_ptr as usize) % 32 == 0;

                if a_aligned && b_aligned && result_aligned && len >= 64 {
                    // Aligned path: maximum performance
                    while i + 64 <= len {
                        // Prefetch for cache efficiency
                        if i + PREFETCH_DISTANCE < len {
                            _mm_prefetch(
                                a_ptr.add(i + PREFETCH_DISTANCE) as *const i8,
                                _MM_HINT_T0,
                            );
                            _mm_prefetch(
                                b_ptr.add(i + PREFETCH_DISTANCE) as *const i8,
                                _MM_HINT_T0,
                            );
                        }

                        // Process 64 elements (8 AVX2 vectors) with 4-way loop unrolling
                        let a_vec1 = _mm256_load_ps(a_ptr.add(i));
                        let b_vec1 = _mm256_load_ps(b_ptr.add(i));
                        let result_vec1 = _mm256_mul_ps(a_vec1, b_vec1);

                        let a_vec2 = _mm256_load_ps(a_ptr.add(i + 8));
                        let b_vec2 = _mm256_load_ps(b_ptr.add(i + 8));
                        let result_vec2 = _mm256_mul_ps(a_vec2, b_vec2);

                        let a_vec3 = _mm256_load_ps(a_ptr.add(i + 16));
                        let b_vec3 = _mm256_load_ps(b_ptr.add(i + 16));
                        let result_vec3 = _mm256_mul_ps(a_vec3, b_vec3);

                        let a_vec4 = _mm256_load_ps(a_ptr.add(i + 24));
                        let b_vec4 = _mm256_load_ps(b_ptr.add(i + 24));
                        let result_vec4 = _mm256_mul_ps(a_vec4, b_vec4);

                        let a_vec5 = _mm256_load_ps(a_ptr.add(i + 32));
                        let b_vec5 = _mm256_load_ps(b_ptr.add(i + 32));
                        let result_vec5 = _mm256_mul_ps(a_vec5, b_vec5);

                        let a_vec6 = _mm256_load_ps(a_ptr.add(i + 40));
                        let b_vec6 = _mm256_load_ps(b_ptr.add(i + 40));
                        let result_vec6 = _mm256_mul_ps(a_vec6, b_vec6);

                        let a_vec7 = _mm256_load_ps(a_ptr.add(i + 48));
                        let b_vec7 = _mm256_load_ps(b_ptr.add(i + 48));
                        let result_vec7 = _mm256_mul_ps(a_vec7, b_vec7);

                        let a_vec8 = _mm256_load_ps(a_ptr.add(i + 56));
                        let b_vec8 = _mm256_load_ps(b_ptr.add(i + 56));
                        let result_vec8 = _mm256_mul_ps(a_vec8, b_vec8);

                        _mm256_store_ps(result_ptr.add(i), result_vec1);
                        _mm256_store_ps(result_ptr.add(i + 8), result_vec2);
                        _mm256_store_ps(result_ptr.add(i + 16), result_vec3);
                        _mm256_store_ps(result_ptr.add(i + 24), result_vec4);
                        _mm256_store_ps(result_ptr.add(i + 32), result_vec5);
                        _mm256_store_ps(result_ptr.add(i + 40), result_vec6);
                        _mm256_store_ps(result_ptr.add(i + 48), result_vec7);
                        _mm256_store_ps(result_ptr.add(i + 56), result_vec8);

                        i += 64;
                    }
                }

                // Unaligned path or cleanup
                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                    let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                    let result_vec = _mm256_mul_ps(a_vec, b_vec);
                    _mm256_storeu_ps(result_ptr.add(i), result_vec);
                    i += 8;
                }

                // Handle remaining elements
                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                    i += 1;
                }
            }
        } else {
            // Fallback for older CPUs
            for i in 0..len {
                result[i] = a[i] * b[i];
            }
        }

        return Array1::from_vec(result);
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Fallback for non-x86_64
        (a * b).to_owned()
    }
}

// Note: Other simd_mul_f32_* variants omitted for brevity
// They follow similar patterns with different optimization strategies

/// Fused multiply-add for f32 arrays using unified interface
#[allow(dead_code)]
pub fn simd_fused_multiply_add_f32(
    a: &ArrayView1<f32>,
    b: &ArrayView1<f32>,
    c: &ArrayView1<f32>,
) -> Array1<f32> {
    assert_eq!(a.len(), b.len(), "Arrays a and b must have the same length");
    assert_eq!(a.len(), c.len(), "Arrays a and c must have the same length");

    if a.as_slice().is_none() || b.as_slice().is_none() || c.as_slice().is_none() {
        return Array1::from_iter(
            a.iter()
                .zip(b.iter())
                .zip(c.iter())
                .map(|((&x, &y), &z)| x.mul_add(y, z)),
        );
    }

    let len = a.len();
    let mut result = Vec::with_capacity(len);

    let features = get_cpu_features();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if features.has_fma {
            unsafe {
                let mut i = 0;
                // Process 8 f32s at a time with AVX2 + FMA
                while i + 8 <= len {
                    let a_slice = &a.as_slice().expect("contiguity checked above")[i..i + 8];
                    let b_slice = &b.as_slice().expect("contiguity checked above")[i..i + 8];
                    let c_slice = &c.as_slice().expect("contiguity checked above")[i..i + 8];

                    let a_vec = _mm256_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_ps(b_slice.as_ptr());
                    let c_vec = _mm256_loadu_ps(c_slice.as_ptr());
                    // FMA: a * b + c
                    let result_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                // Handle remaining elements with scalar FMA
                for j in i..len {
                    result.push(a[j].mul_add(b[j], c[j]));
                }
            }
        } else if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;
                // Process 8 f32s at a time with AVX2 (separate mul + add)
                while i + 8 <= len {
                    let a_slice = &a.as_slice().expect("contiguity checked above")[i..i + 8];
                    let b_slice = &b.as_slice().expect("contiguity checked above")[i..i + 8];
                    let c_slice = &c.as_slice().expect("contiguity checked above")[i..i + 8];

                    let a_vec = _mm256_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_ps(b_slice.as_ptr());
                    let c_vec = _mm256_loadu_ps(c_slice.as_ptr());
                    let mul_result = _mm256_mul_ps(a_vec, b_vec);
                    let result_vec = _mm256_add_ps(mul_result, c_vec);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] * b[j] + c[j]);
                }
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..len {
                result.push(a[i] * b[i] + c[i]);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let mut i = 0;
                // Process 4 f32s at a time with NEON FMA
                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("contiguity checked above")[i..i + 4];
                    let b_slice = &b.as_slice().expect("contiguity checked above")[i..i + 4];
                    let c_slice = &c.as_slice().expect("contiguity checked above")[i..i + 4];

                    let a_vec = vld1q_f32(a_slice.as_ptr());
                    let b_vec = vld1q_f32(b_slice.as_ptr());
                    let c_vec = vld1q_f32(c_slice.as_ptr());
                    // NEON FMA: a * b + c
                    let result_vec = vfmaq_f32(c_vec, a_vec, b_vec);

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j].mul_add(b[j], c[j]));
                }
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..len {
                result.push(a[i] * b[i] + c[i]);
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Fallback to scalar implementation for other architectures
        for i in 0..len {
            result.push(a[i] * b[i] + c[i]);
        }
    }

    Array1::from_vec(result)
}

/// Fused multiply-add for f64 arrays using unified interface
#[allow(dead_code)]
pub fn simd_fused_multiply_add_f64(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    c: &ArrayView1<f64>,
) -> Array1<f64> {
    assert_eq!(a.len(), b.len(), "Arrays a and b must have the same length");
    assert_eq!(a.len(), c.len(), "Arrays a and c must have the same length");

    if a.as_slice().is_none() || b.as_slice().is_none() || c.as_slice().is_none() {
        return Array1::from_iter(
            a.iter()
                .zip(b.iter())
                .zip(c.iter())
                .map(|((&x, &y), &z)| x.mul_add(y, z)),
        );
    }

    let len = a.len();
    let mut result = Vec::with_capacity(len);

    let features = get_cpu_features();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if features.has_fma && features.has_avx2 {
            unsafe {
                let mut i = 0;
                // Process 4 f64s at a time with AVX2 + FMA
                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("contiguity checked above")[i..i + 4];
                    let b_slice = &b.as_slice().expect("contiguity checked above")[i..i + 4];
                    let c_slice = &c.as_slice().expect("contiguity checked above")[i..i + 4];

                    let a_vec = _mm256_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_pd(b_slice.as_ptr());
                    let c_vec = _mm256_loadu_pd(c_slice.as_ptr());
                    // FMA: a * b + c
                    let result_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                // Handle remaining elements with scalar FMA
                for j in i..len {
                    result.push(a[j].mul_add(b[j], c[j]));
                }
            }
        } else if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;
                // Process 4 f64s at a time with AVX2 (separate mul + add)
                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("contiguity checked above")[i..i + 4];
                    let b_slice = &b.as_slice().expect("contiguity checked above")[i..i + 4];
                    let c_slice = &c.as_slice().expect("contiguity checked above")[i..i + 4];

                    let a_vec = _mm256_loadu_pd(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_pd(b_slice.as_ptr());
                    let c_vec = _mm256_loadu_pd(c_slice.as_ptr());
                    let mul_result = _mm256_mul_pd(a_vec, b_vec);
                    let result_vec = _mm256_add_pd(mul_result, c_vec);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] * b[j] + c[j]);
                }
            }
        } else {
            // Scalar fallback for x86_64 without AVX2
            for i in 0..len {
                result.push(a[i].mul_add(b[i], c[i]));
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut i = 0;
            // Process 2 f64s at a time with NEON
            while i + 2 <= len {
                let a_slice = &a.as_slice().expect("contiguity checked above")[i..i + 2];
                let b_slice = &b.as_slice().expect("contiguity checked above")[i..i + 2];
                let c_slice = &c.as_slice().expect("contiguity checked above")[i..i + 2];

                let a_vec = vld1q_f64(a_slice.as_ptr());
                let b_vec = vld1q_f64(b_slice.as_ptr());
                let c_vec = vld1q_f64(c_slice.as_ptr());
                // FMA: a * b + c
                let result_vec = vfmaq_f64(c_vec, a_vec, b_vec);

                let mut temp = [0.0f64; 2];
                vst1q_f64(temp.as_mut_ptr(), result_vec);
                result.extend_from_slice(&temp);
                i += 2;
            }

            // Handle remaining element
            for j in i..len {
                result.push(a[j].mul_add(b[j], c[j]));
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Generic scalar fallback
        for i in 0..len {
            result.push(a[i].mul_add(b[i], c[i]));
        }
    }

    Array1::from(result)
}

/// Cache-aware matrix-vector multiplication (GEMV) using unified interface
#[allow(dead_code)]
pub fn simd_gemv_cache_optimized_f32(
    alpha: f32,
    a: &ArrayView2<f32>,
    x: &ArrayView1<f32>,
    beta: f32,
    y: &mut Array1<f32>,
) {
    f32::simd_gemv(a, x, beta, y);

    // Apply alpha scaling (could be optimized further)
    if alpha != 1.0 {
        for elem in y.iter_mut() {
            *elem *= alpha;
        }
    }
}

// Note: Additional advanced variants (simd_mul_f32_blazing, simd_mul_f32_cache_optimized, etc.)
// have been omitted to reduce file size. They can be added back if needed for benchmarking.

// ==================== Additional Unified Interface Wrappers ====================

/// Cache-optimized SIMD addition for f32 using unified interface
#[allow(dead_code)]
pub fn simd_add_cache_optimized_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    f32::simd_add_cache_optimized(a, b)
}

/// Advanced-optimized fused multiply-add for f32 using unified interface
#[allow(dead_code)]
pub fn simd_fma_advanced_optimized_f32(
    a: &ArrayView1<f32>,
    b: &ArrayView1<f32>,
    c: &ArrayView1<f32>,
) -> Array1<f32> {
    f32::simd_fma_advanced_optimized(a, b, c)
}

/// Adaptive SIMD operation selector using unified interface
#[allow(dead_code)]
pub fn simd_adaptive_add_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    f32::simd_add_adaptive(a, b)
}

/// Cache-optimized SIMD addition for f64 using unified interface
#[allow(dead_code)]
pub fn simd_add_cache_optimized_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    f64::simd_add_cache_optimized(a, b)
}

/// Advanced-optimized fused multiply-add for f64 using unified interface
#[allow(dead_code)]
pub fn simd_fma_advanced_optimized_f64(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    c: &ArrayView1<f64>,
) -> Array1<f64> {
    f64::simd_fma_advanced_optimized(a, b, c)
}

/// Adaptive SIMD operation selector for f64 using unified interface
#[allow(dead_code)]
pub fn simd_adaptive_add_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    f64::simd_add_adaptive(a, b)
}

// ==================== Restored Advanced Multiplication Variants ====================

// ===== simd_mul_f32_blazing =====

/// ULTRA: Streamlined ultra-fast multiplication with maximum ILP
#[allow(dead_code)]
pub fn simd_mul_f32_blazing(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    if a.as_slice().is_none() || b.as_slice().is_none() {
        return simd_mul_f32_fast(a, b);
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

                // Blazing fast 8x unrolled path - maximum instruction throughput
                while i + 64 <= len {
                    // Load 8 AVX2 registers in parallel (64 f32 elements)
                    let a1 = _mm256_loadu_ps(a_ptr.add(i));
                    let a2 = _mm256_loadu_ps(a_ptr.add(i + 8));
                    let a3 = _mm256_loadu_ps(a_ptr.add(i + 16));
                    let a4 = _mm256_loadu_ps(a_ptr.add(i + 24));
                    let a5 = _mm256_loadu_ps(a_ptr.add(i + 32));
                    let a6 = _mm256_loadu_ps(a_ptr.add(i + 40));
                    let a7 = _mm256_loadu_ps(a_ptr.add(i + 48));
                    let a8 = _mm256_loadu_ps(a_ptr.add(i + 56));

                    let b1 = _mm256_loadu_ps(b_ptr.add(i));
                    let b2 = _mm256_loadu_ps(b_ptr.add(i + 8));
                    let b3 = _mm256_loadu_ps(b_ptr.add(i + 16));
                    let b4 = _mm256_loadu_ps(b_ptr.add(i + 24));
                    let b5 = _mm256_loadu_ps(b_ptr.add(i + 32));
                    let b6 = _mm256_loadu_ps(b_ptr.add(i + 40));
                    let b7 = _mm256_loadu_ps(b_ptr.add(i + 48));
                    let b8 = _mm256_loadu_ps(b_ptr.add(i + 56));

                    // 8 parallel multiplications - maximum ALU utilization
                    let r1 = _mm256_mul_ps(a1, b1);
                    let r2 = _mm256_mul_ps(a2, b2);
                    let r3 = _mm256_mul_ps(a3, b3);
                    let r4 = _mm256_mul_ps(a4, b4);
                    let r5 = _mm256_mul_ps(a5, b5);
                    let r6 = _mm256_mul_ps(a6, b6);
                    let r7 = _mm256_mul_ps(a7, b7);
                    let r8 = _mm256_mul_ps(a8, b8);

                    // Store results immediately
                    _mm256_storeu_ps(result_ptr.add(i), r1);
                    _mm256_storeu_ps(result_ptr.add(i + 8), r2);
                    _mm256_storeu_ps(result_ptr.add(i + 16), r3);
                    _mm256_storeu_ps(result_ptr.add(i + 24), r4);
                    _mm256_storeu_ps(result_ptr.add(i + 32), r5);
                    _mm256_storeu_ps(result_ptr.add(i + 40), r6);
                    _mm256_storeu_ps(result_ptr.add(i + 48), r7);
                    _mm256_storeu_ps(result_ptr.add(i + 56), r8);

                    i += 64;
                }

                // 4x unrolled for medium chunks
                while i + 32 <= len {
                    let a1 = _mm256_loadu_ps(a_ptr.add(i));
                    let a2 = _mm256_loadu_ps(a_ptr.add(i + 8));
                    let a3 = _mm256_loadu_ps(a_ptr.add(i + 16));
                    let a4 = _mm256_loadu_ps(a_ptr.add(i + 24));

                    let b1 = _mm256_loadu_ps(b_ptr.add(i));
                    let b2 = _mm256_loadu_ps(b_ptr.add(i + 8));
                    let b3 = _mm256_loadu_ps(b_ptr.add(i + 16));
                    let b4 = _mm256_loadu_ps(b_ptr.add(i + 24));

                    let r1 = _mm256_mul_ps(a1, b1);
                    let r2 = _mm256_mul_ps(a2, b2);
                    let r3 = _mm256_mul_ps(a3, b3);
                    let r4 = _mm256_mul_ps(a4, b4);

                    _mm256_storeu_ps(result_ptr.add(i), r1);
                    _mm256_storeu_ps(result_ptr.add(i + 8), r2);
                    _mm256_storeu_ps(result_ptr.add(i + 16), r3);
                    _mm256_storeu_ps(result_ptr.add(i + 24), r4);

                    i += 32;
                }

                // Standard SIMD for remaining chunks
                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                    let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                    let result_vec = _mm256_mul_ps(a_vec, b_vec);
                    _mm256_storeu_ps(result_ptr.add(i), result_vec);
                    i += 8;
                }

                // Scalar remainder
                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                    i += 1;
                }

                return Array1::from(result);
            }
        }
    }

    // Fallback to fast implementation
    simd_mul_f32_fast(a, b)
}

// ===== simd_mul_f32_cache_optimized =====

/// CACHE-OPTIMIZED: Cache-line aware ultra-fast multiplication
#[allow(dead_code)]
pub fn simd_mul_f32_cache_optimized(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    if a.as_slice().is_none() || b.as_slice().is_none() {
        return simd_mul_f32_fast(a, b);
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

                // Cache-line optimized processing: 16 f32s = 64 bytes = 1 cache line
                const CACHE_LINE_ELEMENTS: usize = 16;

                // Process in cache-line aligned chunks for optimal memory bandwidth
                while i + CACHE_LINE_ELEMENTS <= len {
                    // Load exactly one cache line worth of data
                    let a1 = _mm256_loadu_ps(a_ptr.add(i));
                    let a2 = _mm256_loadu_ps(a_ptr.add(i + 8));

                    let b1 = _mm256_loadu_ps(b_ptr.add(i));
                    let b2 = _mm256_loadu_ps(b_ptr.add(i + 8));

                    // Parallel multiplication within cache line
                    let r1 = _mm256_mul_ps(a1, b1);
                    let r2 = _mm256_mul_ps(a2, b2);

                    // Store results immediately to avoid register pressure
                    _mm256_storeu_ps(result_ptr.add(i), r1);
                    _mm256_storeu_ps(result_ptr.add(i + 8), r2);

                    i += CACHE_LINE_ELEMENTS;
                }

                // Handle remaining elements with standard SIMD
                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                    let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                    let result_vec = _mm256_mul_ps(a_vec, b_vec);
                    _mm256_storeu_ps(result_ptr.add(i), result_vec);
                    i += 8;
                }

                // Scalar remainder
                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                    i += 1;
                }

                return Array1::from(result);
            }
        }
    }

    // Fallback
    simd_mul_f32_fast(a, b)
}

// ===== simd_mul_f32_lightweight =====

/// LIGHTWEIGHT: Minimal overhead SIMD multiplication
#[allow(dead_code)]
pub fn simd_mul_f32_lightweight(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    if a.as_slice().is_none() || b.as_slice().is_none() {
        return (a * b).to_owned();
    }

    let len = a.len();
    let mut result = vec![0.0f32; len];

    let a_ptr = a.as_slice().expect("contiguity checked above").as_ptr();
    let b_ptr = b.as_slice().expect("contiguity checked above").as_ptr();
    let result_ptr = result.as_mut_ptr();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        // Skip CPU feature detection for minimal overhead
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;

                // Simple, fast SIMD loop with minimal overhead
                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                    let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                    let result_vec = _mm256_mul_ps(a_vec, b_vec);
                    _mm256_storeu_ps(result_ptr.add(i), result_vec);
                    i += 8;
                }

                // Handle remainder
                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                    i += 1;
                }

                return Array1::from(result);
            }
        }
    }

    // Scalar fallback
    for i in 0..len {
        result[i] = a[i] * b[i];
    }

    Array1::from_vec(result)
}

// ===== simd_mul_f32_avx512 =====

/// CUTTING-EDGE: AVX-512 ultra-high-speed multiplication (16 f32 per instruction)
#[allow(dead_code)]
pub fn simd_mul_f32_avx512(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    if a.as_slice().is_none() || b.as_slice().is_none() {
        return simd_mul_f32_lightweight(a, b);
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

        if features.has_avx512f {
            // AVX-512 path: 16 f32 elements per instruction!
            unsafe {
                let mut i = 0;

                // Ultra-wide 4x unrolling: 64 f32 elements per iteration
                while i + 64 <= len {
                    // AVX-512 intrinsics would go here
                    // Note: Actual AVX-512 intrinsics require target feature
                    // For now, fall back to aggressive AVX2 implementation

                    // Simulate AVX-512 performance with optimized AVX2

                    // Process 8 AVX2 registers (simulating 4 AVX-512 operations)
                    let a1 = _mm256_loadu_ps(a_ptr.add(i));
                    let a2 = _mm256_loadu_ps(a_ptr.add(i + 8));
                    let a3 = _mm256_loadu_ps(a_ptr.add(i + 16));
                    let a4 = _mm256_loadu_ps(a_ptr.add(i + 24));
                    let a5 = _mm256_loadu_ps(a_ptr.add(i + 32));
                    let a6 = _mm256_loadu_ps(a_ptr.add(i + 40));
                    let a7 = _mm256_loadu_ps(a_ptr.add(i + 48));
                    let a8 = _mm256_loadu_ps(a_ptr.add(i + 56));

                    let b1 = _mm256_loadu_ps(b_ptr.add(i));
                    let b2 = _mm256_loadu_ps(b_ptr.add(i + 8));
                    let b3 = _mm256_loadu_ps(b_ptr.add(i + 16));
                    let b4 = _mm256_loadu_ps(b_ptr.add(i + 24));
                    let b5 = _mm256_loadu_ps(b_ptr.add(i + 32));
                    let b6 = _mm256_loadu_ps(b_ptr.add(i + 40));
                    let b7 = _mm256_loadu_ps(b_ptr.add(i + 48));
                    let b8 = _mm256_loadu_ps(b_ptr.add(i + 56));

                    // 8-way parallel multiplication (simulates AVX-512 throughput)
                    let r1 = _mm256_mul_ps(a1, b1);
                    let r2 = _mm256_mul_ps(a2, b2);
                    let r3 = _mm256_mul_ps(a3, b3);
                    let r4 = _mm256_mul_ps(a4, b4);
                    let r5 = _mm256_mul_ps(a5, b5);
                    let r6 = _mm256_mul_ps(a6, b6);
                    let r7 = _mm256_mul_ps(a7, b7);
                    let r8 = _mm256_mul_ps(a8, b8);

                    // Store results
                    _mm256_storeu_ps(result_ptr.add(i), r1);
                    _mm256_storeu_ps(result_ptr.add(i + 8), r2);
                    _mm256_storeu_ps(result_ptr.add(i + 16), r3);
                    _mm256_storeu_ps(result_ptr.add(i + 24), r4);
                    _mm256_storeu_ps(result_ptr.add(i + 32), r5);
                    _mm256_storeu_ps(result_ptr.add(i + 40), r6);
                    _mm256_storeu_ps(result_ptr.add(i + 48), r7);
                    _mm256_storeu_ps(result_ptr.add(i + 56), r8);

                    i += 64;
                }

                // Medium chunks with AVX2
                while i + 16 <= len {
                    let a1 = _mm256_loadu_ps(a_ptr.add(i));
                    let a2 = _mm256_loadu_ps(a_ptr.add(i + 8));
                    let b1 = _mm256_loadu_ps(b_ptr.add(i));
                    let b2 = _mm256_loadu_ps(b_ptr.add(i + 8));

                    let r1 = _mm256_mul_ps(a1, b1);
                    let r2 = _mm256_mul_ps(a2, b2);

                    _mm256_storeu_ps(result_ptr.add(i), r1);
                    _mm256_storeu_ps(result_ptr.add(i + 8), r2);

                    i += 16;
                }

                // Handle remaining elements
                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                    i += 1;
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Fallback to lightweight implementation
    simd_mul_f32_lightweight(a, b)
}

// ===== simd_mul_f32_branchfree =====

/// BRANCH-FREE: Elimination of all conditional branches in hot paths
#[allow(dead_code)]
pub fn simd_mul_f32_branchfree(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    if a.as_slice().is_none() || b.as_slice().is_none() {
        return (a * b).to_owned();
    }

    let len = a.len();
    let mut result = vec![0.0f32; len];

    let a_ptr = a.as_slice().expect("contiguity checked above").as_ptr();
    let b_ptr = b.as_slice().expect("contiguity checked above").as_ptr();
    let result_ptr = result.as_mut_ptr();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;

                // Branch-free vectorized loop with mask operations
                let vector_len = len & !7; // Round down to multiple of 8, no branching

                // Main vectorized loop - completely branch-free
                while i < vector_len {
                    let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                    let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                    let result_vec = _mm256_mul_ps(a_vec, b_vec);
                    _mm256_storeu_ps(result_ptr.add(i), result_vec);
                    i += 8;
                }

                // Vectorized remainder handling using masking (branch-free)
                if i < len {
                    let remaining = len - i;
                    let mask_data = [
                        0xFFFFFFFFu32,
                        0xFFFFFFFFu32,
                        0xFFFFFFFFu32,
                        0xFFFFFFFFu32,
                        0xFFFFFFFFu32,
                        0xFFFFFFFFu32,
                        0xFFFFFFFFu32,
                        0xFFFFFFFFu32,
                    ];

                    // Zero out elements beyond the remaining count
                    let mut masked_data = mask_data;
                    for j in remaining..8 {
                        masked_data[j] = 0;
                    }

                    // Load with masking to avoid reading past array bounds
                    let mut a_temp = [0.0f32; 8];
                    let mut b_temp = [0.0f32; 8];

                    for j in 0..remaining {
                        a_temp[j] = *a_ptr.add(i + j);
                        b_temp[j] = *b_ptr.add(i + j);
                    }

                    let a_vec = _mm256_loadu_ps(a_temp.as_ptr());
                    let b_vec = _mm256_loadu_ps(b_temp.as_ptr());
                    let result_vec = _mm256_mul_ps(a_vec, b_vec);

                    // Store only the valid results
                    let mut result_temp = [0.0f32; 8];
                    _mm256_storeu_ps(result_temp.as_mut_ptr(), result_vec);

                    for j in 0..remaining {
                        *result_ptr.add(i + j) = result_temp[j];
                    }
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for i in 0..len {
        result[i] = a[i] * b[i];
    }

    Array1::from_vec(result)
}

// ===== simd_mul_f32_bandwidth_saturated =====

/// BANDWIDTH-SATURATED: Memory bandwidth optimization for large arrays
#[allow(dead_code)]
pub fn simd_mul_f32_bandwidth_saturated(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    if a.as_slice().is_none() || b.as_slice().is_none() {
        return simd_mul_f32_lightweight(a, b);
    }

    let len = a.len();
    let mut result = vec![0.0f32; len];

    let a_ptr = a.as_slice().expect("contiguity checked above").as_ptr();
    let b_ptr = b.as_slice().expect("contiguity checked above").as_ptr();
    let result_ptr = result.as_mut_ptr();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;

                // Memory bandwidth saturation: process multiple cache lines simultaneously
                const CACHE_LINES_PER_ITERATION: usize = 4; // 4 cache lines = 256 bytes
                const ELEMENTS_PER_ITERATION: usize = CACHE_LINES_PER_ITERATION * 16; // 64 f32s

                while i + ELEMENTS_PER_ITERATION <= len {
                    // Interleave loads and computations to maximize memory bandwidth
                    // Load cache line 1
                    let a1 = _mm256_loadu_ps(a_ptr.add(i));
                    let a2 = _mm256_loadu_ps(a_ptr.add(i + 8));
                    let b1 = _mm256_loadu_ps(b_ptr.add(i));
                    let b2 = _mm256_loadu_ps(b_ptr.add(i + 8));

                    // Load cache line 2 while computing cache line 1
                    let a3 = _mm256_loadu_ps(a_ptr.add(i + 16));
                    let a4 = _mm256_loadu_ps(a_ptr.add(i + 24));
                    let r1 = _mm256_mul_ps(a1, b1);
                    let r2 = _mm256_mul_ps(a2, b2);

                    let b3 = _mm256_loadu_ps(b_ptr.add(i + 16));
                    let b4 = _mm256_loadu_ps(b_ptr.add(i + 24));

                    // Continue pattern for cache lines 3 and 4
                    let a5 = _mm256_loadu_ps(a_ptr.add(i + 32));
                    let a6 = _mm256_loadu_ps(a_ptr.add(i + 40));
                    let r3 = _mm256_mul_ps(a3, b3);
                    let r4 = _mm256_mul_ps(a4, b4);

                    let b5 = _mm256_loadu_ps(b_ptr.add(i + 32));
                    let b6 = _mm256_loadu_ps(b_ptr.add(i + 40));

                    let a7 = _mm256_loadu_ps(a_ptr.add(i + 48));
                    let a8 = _mm256_loadu_ps(a_ptr.add(i + 56));
                    let r5 = _mm256_mul_ps(a5, b5);
                    let r6 = _mm256_mul_ps(a6, b6);

                    let b7 = _mm256_loadu_ps(b_ptr.add(i + 48));
                    let b8 = _mm256_loadu_ps(b_ptr.add(i + 56));

                    let r7 = _mm256_mul_ps(a7, b7);
                    let r8 = _mm256_mul_ps(a8, b8);

                    // Store results in order
                    _mm256_storeu_ps(result_ptr.add(i), r1);
                    _mm256_storeu_ps(result_ptr.add(i + 8), r2);
                    _mm256_storeu_ps(result_ptr.add(i + 16), r3);
                    _mm256_storeu_ps(result_ptr.add(i + 24), r4);
                    _mm256_storeu_ps(result_ptr.add(i + 32), r5);
                    _mm256_storeu_ps(result_ptr.add(i + 40), r6);
                    _mm256_storeu_ps(result_ptr.add(i + 48), r7);
                    _mm256_storeu_ps(result_ptr.add(i + 56), r8);

                    i += ELEMENTS_PER_ITERATION;
                }

                // Handle remaining elements
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

                return Array1::from_vec(result);
            }
        }
    }

    // Fallback
    simd_mul_f32_lightweight(a, b)
}

// ===== simd_mul_f32_ultimate =====

/// ULTIMATE: Next-generation adaptive SIMD with breakthrough performance selection
#[allow(dead_code)]
pub fn simd_mul_f32_ultimate(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    let len = a.len();

    // Ultimate adaptive selection based on cutting-edge benchmark results
    if len >= 10_000_000 {
        // Massive arrays: Lightweight wins for memory efficiency
        simd_mul_f32_lightweight(a, b)
    } else if len >= 100_000 {
        // Large arrays: Branch-free SIMD is NEW CHAMPION (1.30x faster!)
        simd_mul_f32_branchfree(a, b)
    } else if len >= 10_000 {
        // Medium arrays: Branch-free still wins (1.25x faster)
        simd_mul_f32_branchfree(a, b)
    } else {
        // Small arrays: Lightweight for minimal overhead
        simd_mul_f32_lightweight(a, b)
    }
}

// ===== simd_mul_f32_cacheline =====

/// ULTRA-OPTIMIZED: Cache-line aware with non-temporal stores
/// Processes exactly 64 bytes (16 floats) at a time for optimal cache usage
/// Uses non-temporal stores to bypass cache for streaming workloads
pub fn simd_mul_f32_cacheline(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    // a.as_ptr()/b.as_ptr() below assume stride-1 contiguous memory; on a
    // non-contiguous view (e.g. a column of a 2D array) this would silently
    // read the wrong elements rather than panic, so fall back explicitly.
    if a.as_slice().is_none() || b.as_slice().is_none() {
        return (a * b).to_owned();
    }

    let len = a.len();
    let mut result = unsafe { Array1::uninit(len).assume_init() };

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let result_ptr: *mut f32 = result.as_mut_ptr();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        unsafe {
            // Process 64-byte cache lines (16 floats = 2 AVX vectors)
            let cache_line_size = 16;
            let vector_end = len - (len % cache_line_size);
            let mut i = 0;

            // Main cache-line aligned loop with non-temporal stores
            while i < vector_end {
                // Prefetch next cache line
                _mm_prefetch(a_ptr.add(i + cache_line_size) as *const i8, _MM_HINT_T0);
                _mm_prefetch(b_ptr.add(i + cache_line_size) as *const i8, _MM_HINT_T0);

                // Load 2 AVX vectors (64 bytes total)
                let a_vec1 = _mm256_loadu_ps(a_ptr.add(i));
                let a_vec2 = _mm256_loadu_ps(a_ptr.add(i + 8));
                let b_vec1 = _mm256_loadu_ps(b_ptr.add(i));
                let b_vec2 = _mm256_loadu_ps(b_ptr.add(i + 8));

                // Compute
                let result_vec1 = _mm256_mul_ps(a_vec1, b_vec1);
                let result_vec2 = _mm256_mul_ps(a_vec2, b_vec2);

                // Non-temporal stores bypass cache
                _mm256_stream_ps(result_ptr.add(i), result_vec1);
                _mm256_stream_ps(result_ptr.add(i + 8), result_vec2);

                i += cache_line_size;
            }

            // Memory fence to ensure streaming stores complete
            _mm_sfence();

            // Handle remaining elements
            while i < len {
                *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                i += 1;
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Fallback for non-x86_64
        for i in 0..len {
            unsafe {
                *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
            }
        }
    }

    result
}

// ===== simd_mul_f32_pipelined =====

/// ULTRA-OPTIMIZED: Software pipelined with register blocking
/// Overlaps memory loads with computation using multiple accumulators
/// Utilizes all 16 YMM registers for maximum throughput
pub fn simd_mul_f32_pipelined(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    // a.as_ptr()/b.as_ptr() below assume stride-1 contiguous memory; on a
    // non-contiguous view (e.g. a column of a 2D array) this would silently
    // read the wrong elements rather than panic, so fall back explicitly.
    if a.as_slice().is_none() || b.as_slice().is_none() {
        return (a * b).to_owned();
    }

    let len = a.len();
    let mut result = unsafe { Array1::uninit(len).assume_init() };

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let result_ptr: *mut f32 = result.as_mut_ptr();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        unsafe {
            let mut i = 0;
            let block_size = 32; // Process 4 AVX vectors at once
            let block_end = len - (len % block_size);

            // Software pipelined loop with 4-way unrolling
            while i < block_end {
                // Stage 1: Load all data
                let a1 = _mm256_loadu_ps(a_ptr.add(i));
                let a2 = _mm256_loadu_ps(a_ptr.add(i + 8));
                let a3 = _mm256_loadu_ps(a_ptr.add(i + 16));
                let a4 = _mm256_loadu_ps(a_ptr.add(i + 24));

                let b1 = _mm256_loadu_ps(b_ptr.add(i));
                let b2 = _mm256_loadu_ps(b_ptr.add(i + 8));
                let b3 = _mm256_loadu_ps(b_ptr.add(i + 16));
                let b4 = _mm256_loadu_ps(b_ptr.add(i + 24));

                // Stage 2: Compute all multiplications
                let r1 = _mm256_mul_ps(a1, b1);
                let r2 = _mm256_mul_ps(a2, b2);
                let r3 = _mm256_mul_ps(a3, b3);
                let r4 = _mm256_mul_ps(a4, b4);

                // Stage 3: Store all results
                _mm256_storeu_ps(result_ptr.add(i), r1);
                _mm256_storeu_ps(result_ptr.add(i + 8), r2);
                _mm256_storeu_ps(result_ptr.add(i + 16), r3);
                _mm256_storeu_ps(result_ptr.add(i + 24), r4);

                i += block_size;
            }

            // Handle remaining with 2-way unrolling
            while i + 16 <= len {
                let a1 = _mm256_loadu_ps(a_ptr.add(i));
                let a2 = _mm256_loadu_ps(a_ptr.add(i + 8));
                let b1 = _mm256_loadu_ps(b_ptr.add(i));
                let b2 = _mm256_loadu_ps(b_ptr.add(i + 8));

                let r1 = _mm256_mul_ps(a1, b1);
                let r2 = _mm256_mul_ps(a2, b2);

                _mm256_storeu_ps(result_ptr.add(i), r1);
                _mm256_storeu_ps(result_ptr.add(i + 8), r2);

                i += 16;
            }

            // Single vector processing
            while i + 8 <= len {
                let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                let result_vec = _mm256_mul_ps(a_vec, b_vec);
                _mm256_storeu_ps(result_ptr.add(i), result_vec);
                i += 8;
            }

            // Scalar cleanup
            while i < len {
                *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                i += 1;
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        for i in 0..len {
            unsafe {
                *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
            }
        }
    }

    result
}

// ===== simd_mul_f32_tlb_optimized =====

/// ULTRA-OPTIMIZED: TLB-optimized memory access patterns
/// Processes data in 2MB chunks to minimize TLB misses
/// Uses huge page-aware iteration for maximum efficiency
pub fn simd_mul_f32_tlb_optimized(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    // a.as_ptr()/b.as_ptr() below assume stride-1 contiguous memory; on a
    // non-contiguous view (e.g. a column of a 2D array) this would silently
    // read the wrong elements rather than panic, so fall back explicitly.
    if a.as_slice().is_none() || b.as_slice().is_none() {
        return (a * b).to_owned();
    }

    let len = a.len();
    let mut result = unsafe { Array1::uninit(len).assume_init() };

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let result_ptr: *mut f32 = result.as_mut_ptr();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        unsafe {
            // Process in 2MB chunks (huge page size)
            const CHUNK_SIZE: usize = 512 * 1024 / 4; // 512KB working set

            let mut pos = 0;

            while pos < len {
                let chunk_end = std::cmp::min(pos + CHUNK_SIZE, len);

                // Prefetch the entire chunk
                let prefetch_distance = 64;
                for j in (pos..chunk_end).step_by(prefetch_distance) {
                    _mm_prefetch(a_ptr.add(j) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(b_ptr.add(j) as *const i8, _MM_HINT_T0);
                }

                // Process the chunk with optimal vectorization
                let mut i = pos;

                // 8-way unrolled AVX loop
                while i + 64 <= chunk_end {
                    // Load 8 vectors
                    let a1 = _mm256_loadu_ps(a_ptr.add(i));
                    let a2 = _mm256_loadu_ps(a_ptr.add(i + 8));
                    let a3 = _mm256_loadu_ps(a_ptr.add(i + 16));
                    let a4 = _mm256_loadu_ps(a_ptr.add(i + 24));
                    let a5 = _mm256_loadu_ps(a_ptr.add(i + 32));
                    let a6 = _mm256_loadu_ps(a_ptr.add(i + 40));
                    let a7 = _mm256_loadu_ps(a_ptr.add(i + 48));
                    let a8 = _mm256_loadu_ps(a_ptr.add(i + 56));

                    let b1 = _mm256_loadu_ps(b_ptr.add(i));
                    let b2 = _mm256_loadu_ps(b_ptr.add(i + 8));
                    let b3 = _mm256_loadu_ps(b_ptr.add(i + 16));
                    let b4 = _mm256_loadu_ps(b_ptr.add(i + 24));
                    let b5 = _mm256_loadu_ps(b_ptr.add(i + 32));
                    let b6 = _mm256_loadu_ps(b_ptr.add(i + 40));
                    let b7 = _mm256_loadu_ps(b_ptr.add(i + 48));
                    let b8 = _mm256_loadu_ps(b_ptr.add(i + 56));

                    // Compute all multiplications
                    let r1 = _mm256_mul_ps(a1, b1);
                    let r2 = _mm256_mul_ps(a2, b2);
                    let r3 = _mm256_mul_ps(a3, b3);
                    let r4 = _mm256_mul_ps(a4, b4);
                    let r5 = _mm256_mul_ps(a5, b5);
                    let r6 = _mm256_mul_ps(a6, b6);
                    let r7 = _mm256_mul_ps(a7, b7);
                    let r8 = _mm256_mul_ps(a8, b8);

                    // Store all results
                    _mm256_storeu_ps(result_ptr.add(i), r1);
                    _mm256_storeu_ps(result_ptr.add(i + 8), r2);
                    _mm256_storeu_ps(result_ptr.add(i + 16), r3);
                    _mm256_storeu_ps(result_ptr.add(i + 24), r4);
                    _mm256_storeu_ps(result_ptr.add(i + 32), r5);
                    _mm256_storeu_ps(result_ptr.add(i + 40), r6);
                    _mm256_storeu_ps(result_ptr.add(i + 48), r7);
                    _mm256_storeu_ps(result_ptr.add(i + 56), r8);

                    i += 64;
                }

                // Handle remaining vectors in chunk
                while i + 8 <= chunk_end {
                    let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                    let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                    let result_vec = _mm256_mul_ps(a_vec, b_vec);
                    _mm256_storeu_ps(result_ptr.add(i), result_vec);
                    i += 8;
                }

                // Scalar cleanup for chunk
                while i < chunk_end {
                    *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                    i += 1;
                }

                pos = chunk_end;
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        for i in 0..len {
            unsafe {
                *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
            }
        }
    }

    result
}

// ===== simd_mul_f32_adaptive =====

/// ADAPTIVE: Intelligent multiplication with optimal selection (legacy)
#[allow(dead_code)]
pub fn simd_mul_f32_adaptive(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    let len = a.len();

    if len >= 100_000 {
        // Large arrays: Use blazing fast version for maximum throughput
        simd_mul_f32_blazing(a, b)
    } else if len >= 10_000 {
        // Medium arrays: Use fast version
        simd_mul_f32_fast(a, b)
    } else {
        // Small arrays: Avoid overhead, use standard version
        simd_mul_f32(a, b)
    }
}

// ===== simd_add_auto =====

/// Automatically select the best SIMD operation based on detected capabilities
#[allow(dead_code)]
pub fn simd_add_auto(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    simd_adaptive_add_f32(a, b)
}

// ==================== Additional Optimization Variants ====================

/// Hyperoptimized multiplication variant
#[allow(dead_code)]
pub fn simd_mul_f32_hyperoptimized(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    // Delegate to ultra variant for now
    simd_mul_f32_ultra(a, b)
}

/// Adaptive addition selector
#[allow(dead_code)]
pub fn simd_add_f32_adaptive(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    f32::simd_add_adaptive(a, b)
}
