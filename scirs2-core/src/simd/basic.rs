//! Basic arithmetic operations with SIMD acceleration
//!
//! This module provides optimized implementations of fundamental array operations
//! including element-wise maximum, minimum, and addition with various optimization levels.

use ndarray::{Array1, ArrayView1};

/// Compute element-wise maximum of two f32 arrays using unified SIMD operations
///
/// This function uses the unified SIMD interface for better performance when
/// processing large arrays of f32 values.
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise maximum array
#[allow(dead_code)]
pub fn simd_maximum_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    // Direct implementation to avoid circular dependency
    let mut result = Array1::zeros(a.len());
    for i in 0..a.len() {
        result[i] = a[i].max(b[i]);
    }
    result
}

/// Compute element-wise maximum of two f64 arrays using unified SIMD operations
///
/// This function uses the unified SIMD interface for better performance when
/// processing large arrays of f64 values.
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise maximum array
#[allow(dead_code)]
pub fn simd_maximum_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    // Direct implementation to avoid circular dependency
    let mut result = Array1::zeros(a.len());
    for i in 0..a.len() {
        result[i] = a[i].max(b[i]);
    }
    result
}

/// Compute element-wise minimum of two f32 arrays using unified SIMD operations
///
/// This function uses the unified SIMD interface for better performance when
/// processing large arrays of f32 values.
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise minimum array
#[allow(dead_code)]
pub fn simd_minimum_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    // Direct implementation to avoid circular dependency
    let mut result = Array1::zeros(a.len());
    for i in 0..a.len() {
        result[i] = a[i].min(b[i]);
    }
    result
}

/// Compute element-wise minimum of two f64 arrays using unified SIMD operations
///
/// This function uses the unified SIMD interface for better performance when
/// processing large arrays of f64 values.
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise minimum array
#[allow(dead_code)]
pub fn simd_minimum_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    // Direct implementation to avoid circular dependency
    let mut result = Array1::zeros(a.len());
    for i in 0..a.len() {
        result[i] = a[i].min(b[i]);
    }
    result
}

/// Compute element-wise addition of two f32 arrays using unified SIMD operations
///
/// This function uses the unified SIMD interface for better performance when
/// processing large arrays of f32 values.
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise sum array
#[allow(dead_code)]
pub fn simd_add_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() || b.as_slice().is_none() {
        return (a + b).to_owned();
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
                    let a_slice = &a.as_slice().expect("contiguity checked above")[i..i + 8];
                    let b_slice = &b.as_slice().expect("contiguity checked above")[i..i + 8];

                    let a_vec = _mm256_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm256_loadu_ps(b_slice.as_ptr());
                    let result_vec = _mm256_add_ps(a_vec, b_vec);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] + b[j]);
                }
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let mut i = 0;
                // Process 4 f32s at a time with SSE
                while i + 4 <= len {
                    let a_slice = &a.as_slice().expect("contiguity checked above")[i..i + 4];
                    let b_slice = &b.as_slice().expect("contiguity checked above")[i..i + 4];

                    let a_vec = _mm_loadu_ps(a_slice.as_ptr());
                    let b_vec = _mm_loadu_ps(b_slice.as_ptr());
                    let result_vec = _mm_add_ps(a_vec, b_vec);

                    let mut temp = [0.0f32; 4];
                    _mm_storeu_ps(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] + b[j]);
                }
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..len {
                result.push(a[i] + b[i]);
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
                    let a_slice = &a.as_slice().expect("contiguity checked above")[i..i + 4];
                    let b_slice = &b.as_slice().expect("contiguity checked above")[i..i + 4];

                    let a_vec = vld1q_f32(a_slice.as_ptr());
                    let b_vec = vld1q_f32(b_slice.as_ptr());
                    let result_vec = vaddq_f32(a_vec, b_vec);

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), result_vec);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] + b[j]);
                }
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..len {
                result.push(a[i] + b[i]);
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Fallback to scalar implementation for other architectures
        for i in 0..len {
            result.push(a[i] + b[i]);
        }
    }

    Array1::from_vec(result)
}

/// OPTIMIZED: Compute element-wise addition of two f32 arrays using SIMD operations
/// This is the Phase 1 optimization version with direct memory access patterns
#[allow(dead_code)]
pub fn simd_add_f32_optimized(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() || b.as_slice().is_none() {
        return (a + b).to_owned();
    }

    let len = a.len();

    // Phase 1 Optimization: Pre-allocate exact size, avoid dynamic growth
    let mut result = vec![0.0f32; len];

    // Get contiguous data pointers for direct access
    let a_ptr = a.as_slice().expect("contiguity checked above").as_ptr();
    let b_ptr = b.as_slice().expect("contiguity checked above").as_ptr();
    let result_ptr = result.as_mut_ptr();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;
                // Process 8 f32s at a time with AVX2 - direct memory access
                while i + 8 <= len {
                    // Direct pointer access - no bounds checking, no slicing
                    let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                    let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                    let result_vec = _mm256_add_ps(a_vec, b_vec);

                    // Direct store - no temporary buffer, no copying
                    _mm256_storeu_ps(result_ptr.add(i), result_vec);
                    i += 8;
                }

                // Handle remaining elements efficiently
                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                    i += 1;
                }
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let mut i = 0;
                // Process 4 f32s at a time with SSE - direct memory access
                while i + 4 <= len {
                    let a_vec = _mm_loadu_ps(a_ptr.add(i));
                    let b_vec = _mm_loadu_ps(b_ptr.add(i));
                    let result_vec = _mm_add_ps(a_vec, b_vec);

                    // Direct store - no temporary buffer
                    _mm_storeu_ps(result_ptr.add(i), result_vec);
                    i += 4;
                }

                // Handle remaining elements
                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                    i += 1;
                }
            }
        } else {
            // Fallback to scalar implementation
            unsafe {
                for i in 0..len {
                    *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let mut i = 0;
                // Process 4 f32s at a time with NEON - direct memory access
                while i + 4 <= len {
                    let a_vec = vld1q_f32(a_ptr.add(i));
                    let b_vec = vld1q_f32(b_ptr.add(i));
                    let result_vec = vaddq_f32(a_vec, b_vec);

                    // Direct store - no temporary buffer
                    vst1q_f32(result_ptr.add(i), result_vec);
                    i += 4;
                }

                // Handle remaining elements
                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                    i += 1;
                }
            }
        } else {
            // Fallback to scalar implementation
            unsafe {
                for i in 0..len {
                    *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                }
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Fallback to scalar implementation for other architectures
        unsafe {
            for i in 0..len {
                *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
            }
        }
    }

    Array1::from_vec(result)
}

/// PHASE 2: CPU feature caching for optimal performance
use std::sync::OnceLock;

struct CpuFeatures {
    has_avx512f: bool,
    has_avx2: bool,
    has_sse: bool,
    has_fma: bool,
    has_neon: bool,
}

static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

fn get_cpu_features() -> &'static CpuFeatures {
    CPU_FEATURES.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            CpuFeatures {
                has_avx512f: std::arch::is_x86_feature_detected!("avx512f"),
                has_avx2: std::arch::is_x86_feature_detected!("avx2"),
                has_sse: std::arch::is_x86_feature_detected!("sse"),
                has_fma: std::arch::is_x86_feature_detected!("fma"),
                has_neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            CpuFeatures {
                has_avx512f: false,
                has_avx2: false,
                has_sse: false,
                has_fma: false, // ARM uses different FMA instructions
                has_neon: std::arch::is_aarch64_feature_detected!("neon"),
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            CpuFeatures {
                has_avx512f: false,
                has_avx2: false,
                has_sse: false,
                has_fma: false,
                has_neon: false,
            }
        }
    })
}

/// PHASE 2: Advanced SIMD addition with CPU feature caching and loop unrolling
#[allow(dead_code)]
pub fn simd_add_f32_fast(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if a.as_slice().is_none() || b.as_slice().is_none() {
        return (a + b).to_owned();
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

                // Phase 2: Loop unrolling - process 32 elements (4x AVX2) at once
                while i + 32 <= len {
                    // Load 4 AVX2 registers worth of data
                    let a_vec1 = _mm256_loadu_ps(a_ptr.add(i));
                    let a_vec2 = _mm256_loadu_ps(a_ptr.add(i + 8));
                    let a_vec3 = _mm256_loadu_ps(a_ptr.add(i + 16));
                    let a_vec4 = _mm256_loadu_ps(a_ptr.add(i + 24));

                    let b_vec1 = _mm256_loadu_ps(b_ptr.add(i));
                    let b_vec2 = _mm256_loadu_ps(b_ptr.add(i + 8));
                    let b_vec3 = _mm256_loadu_ps(b_ptr.add(i + 16));
                    let b_vec4 = _mm256_loadu_ps(b_ptr.add(i + 24));

                    // Parallel execution of 4 SIMD operations
                    let result_vec1 = _mm256_add_ps(a_vec1, b_vec1);
                    let result_vec2 = _mm256_add_ps(a_vec2, b_vec2);
                    let result_vec3 = _mm256_add_ps(a_vec3, b_vec3);
                    let result_vec4 = _mm256_add_ps(a_vec4, b_vec4);

                    // Store results
                    _mm256_storeu_ps(result_ptr.add(i), result_vec1);
                    _mm256_storeu_ps(result_ptr.add(i + 8), result_vec2);
                    _mm256_storeu_ps(result_ptr.add(i + 16), result_vec3);
                    _mm256_storeu_ps(result_ptr.add(i + 24), result_vec4);

                    i += 32;
                }

                // Handle remaining 8-element chunks
                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                    let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                    let result_vec = _mm256_add_ps(a_vec, b_vec);
                    _mm256_storeu_ps(result_ptr.add(i), result_vec);
                    i += 8;
                }

                // Handle remaining elements
                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                    i += 1;
                }
            }
        } else if features.has_sse {
            unsafe {
                let mut i = 0;

                // SSE version with 2x unrolling
                while i + 8 <= len {
                    let a_vec1 = _mm_loadu_ps(a_ptr.add(i));
                    let a_vec2 = _mm_loadu_ps(a_ptr.add(i + 4));
                    let b_vec1 = _mm_loadu_ps(b_ptr.add(i));
                    let b_vec2 = _mm_loadu_ps(b_ptr.add(i + 4));

                    let result_vec1 = _mm_add_ps(a_vec1, b_vec1);
                    let result_vec2 = _mm_add_ps(a_vec2, b_vec2);

                    _mm_storeu_ps(result_ptr.add(i), result_vec1);
                    _mm_storeu_ps(result_ptr.add(i + 4), result_vec2);
                    i += 8;
                }

                while i + 4 <= len {
                    let a_vec = _mm_loadu_ps(a_ptr.add(i));
                    let b_vec = _mm_loadu_ps(b_ptr.add(i));
                    let result_vec = _mm_add_ps(a_vec, b_vec);
                    _mm_storeu_ps(result_ptr.add(i), result_vec);
                    i += 4;
                }

                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                    i += 1;
                }
            }
        } else {
            unsafe {
                for i in 0..len {
                    *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if features.has_neon {
            unsafe {
                let mut i = 0;

                // NEON version with 2x unrolling
                while i + 8 <= len {
                    let a_vec1 = vld1q_f32(a_ptr.add(i));
                    let a_vec2 = vld1q_f32(a_ptr.add(i + 4));
                    let b_vec1 = vld1q_f32(b_ptr.add(i));
                    let b_vec2 = vld1q_f32(b_ptr.add(i + 4));

                    let result_vec1 = vaddq_f32(a_vec1, b_vec1);
                    let result_vec2 = vaddq_f32(a_vec2, b_vec2);

                    vst1q_f32(result_ptr.add(i), result_vec1);
                    vst1q_f32(result_ptr.add(i + 4), result_vec2);
                    i += 8;
                }

                while i + 4 <= len {
                    let a_vec = vld1q_f32(a_ptr.add(i));
                    let b_vec = vld1q_f32(b_ptr.add(i));
                    let result_vec = vaddq_f32(a_vec, b_vec);
                    vst1q_f32(result_ptr.add(i), result_vec);
                    i += 4;
                }

                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                    i += 1;
                }
            }
        } else {
            unsafe {
                for i in 0..len {
                    *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                }
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        unsafe {
            for i in 0..len {
                *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
            }
        }
    }

    Array1::from_vec(result)
}

/// PHASE 3: Memory-aligned SIMD with prefetching for maximum performance
#[allow(dead_code)]
pub fn simd_add_f32_ultra(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    let len = a.len();
    let mut result = vec![0.0f32; len];

    let a_ptr = a.as_slice().expect("Operation failed").as_ptr();
    let b_ptr = b.as_slice().expect("Operation failed").as_ptr();
    let result_ptr = result.as_mut_ptr();

    let features = get_cpu_features();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if features.has_avx2 {
            unsafe {
                let mut i = 0;

                // Phase 3: Advanced optimizations with prefetching
                const PREFETCH_DISTANCE: usize = 256; // 64 cache lines ahead

                // Check if we can use aligned loads (rare but optimal)
                let a_aligned = (a_ptr as usize) % 32 == 0;
                let b_aligned = (b_ptr as usize) % 32 == 0;
                let result_aligned = (result_ptr as usize) % 32 == 0;

                if a_aligned && b_aligned && result_aligned && len >= 64 {
                    // Optimal path: All data is aligned - use fastest instructions
                    while i + 64 <= len {
                        // Prefetch future data
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

                        // Process 64 elements (8x AVX2 registers) with aligned loads
                        let a_vec1 = _mm256_load_ps(a_ptr.add(i));
                        let a_vec2 = _mm256_load_ps(a_ptr.add(i + 8));
                        let a_vec3 = _mm256_load_ps(a_ptr.add(i + 16));
                        let a_vec4 = _mm256_load_ps(a_ptr.add(i + 24));
                        let a_vec5 = _mm256_load_ps(a_ptr.add(i + 32));
                        let a_vec6 = _mm256_load_ps(a_ptr.add(i + 40));
                        let a_vec7 = _mm256_load_ps(a_ptr.add(i + 48));
                        let a_vec8 = _mm256_load_ps(a_ptr.add(i + 56));

                        let b_vec1 = _mm256_load_ps(b_ptr.add(i));
                        let b_vec2 = _mm256_load_ps(b_ptr.add(i + 8));
                        let b_vec3 = _mm256_load_ps(b_ptr.add(i + 16));
                        let b_vec4 = _mm256_load_ps(b_ptr.add(i + 24));
                        let b_vec5 = _mm256_load_ps(b_ptr.add(i + 32));
                        let b_vec6 = _mm256_load_ps(b_ptr.add(i + 40));
                        let b_vec7 = _mm256_load_ps(b_ptr.add(i + 48));
                        let b_vec8 = _mm256_load_ps(b_ptr.add(i + 56));

                        // Parallel execution of 8 SIMD operations
                        let result_vec1 = _mm256_add_ps(a_vec1, b_vec1);
                        let result_vec2 = _mm256_add_ps(a_vec2, b_vec2);
                        let result_vec3 = _mm256_add_ps(a_vec3, b_vec3);
                        let result_vec4 = _mm256_add_ps(a_vec4, b_vec4);
                        let result_vec5 = _mm256_add_ps(a_vec5, b_vec5);
                        let result_vec6 = _mm256_add_ps(a_vec6, b_vec6);
                        let result_vec7 = _mm256_add_ps(a_vec7, b_vec7);
                        let result_vec8 = _mm256_add_ps(a_vec8, b_vec8);

                        // Aligned stores
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
                } else {
                    // Standard path: Unaligned data with prefetching
                    while i + 32 <= len {
                        // Prefetch future data for better cache performance
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

                        // 4x unrolled loop with unaligned loads
                        let a_vec1 = _mm256_loadu_ps(a_ptr.add(i));
                        let a_vec2 = _mm256_loadu_ps(a_ptr.add(i + 8));
                        let a_vec3 = _mm256_loadu_ps(a_ptr.add(i + 16));
                        let a_vec4 = _mm256_loadu_ps(a_ptr.add(i + 24));

                        let b_vec1 = _mm256_loadu_ps(b_ptr.add(i));
                        let b_vec2 = _mm256_loadu_ps(b_ptr.add(i + 8));
                        let b_vec3 = _mm256_loadu_ps(b_ptr.add(i + 16));
                        let b_vec4 = _mm256_loadu_ps(b_ptr.add(i + 24));

                        let result_vec1 = _mm256_add_ps(a_vec1, b_vec1);
                        let result_vec2 = _mm256_add_ps(a_vec2, b_vec2);
                        let result_vec3 = _mm256_add_ps(a_vec3, b_vec3);
                        let result_vec4 = _mm256_add_ps(a_vec4, b_vec4);

                        _mm256_storeu_ps(result_ptr.add(i), result_vec1);
                        _mm256_storeu_ps(result_ptr.add(i + 8), result_vec2);
                        _mm256_storeu_ps(result_ptr.add(i + 16), result_vec3);
                        _mm256_storeu_ps(result_ptr.add(i + 24), result_vec4);

                        i += 32;
                    }
                }

                // Handle remaining 8-element chunks
                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                    let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                    let result_vec = _mm256_add_ps(a_vec, b_vec);
                    _mm256_storeu_ps(result_ptr.add(i), result_vec);
                    i += 8;
                }

                // Handle remaining elements
                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                    i += 1;
                }
            }
        } else if features.has_sse {
            unsafe {
                let mut i = 0;

                // SSE version with prefetching
                while i + 16 <= len {
                    if i + 128 < len {
                        _mm_prefetch(a_ptr.add(i + 128) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(b_ptr.add(i + 128) as *const i8, _MM_HINT_T0);
                    }

                    // 4x unrolled SSE
                    let a_vec1 = _mm_loadu_ps(a_ptr.add(i));
                    let a_vec2 = _mm_loadu_ps(a_ptr.add(i + 4));
                    let a_vec3 = _mm_loadu_ps(a_ptr.add(i + 8));
                    let a_vec4 = _mm_loadu_ps(a_ptr.add(i + 12));

                    let b_vec1 = _mm_loadu_ps(b_ptr.add(i));
                    let b_vec2 = _mm_loadu_ps(b_ptr.add(i + 4));
                    let b_vec3 = _mm_loadu_ps(b_ptr.add(i + 8));
                    let b_vec4 = _mm_loadu_ps(b_ptr.add(i + 12));

                    let result_vec1 = _mm_add_ps(a_vec1, b_vec1);
                    let result_vec2 = _mm_add_ps(a_vec2, b_vec2);
                    let result_vec3 = _mm_add_ps(a_vec3, b_vec3);
                    let result_vec4 = _mm_add_ps(a_vec4, b_vec4);

                    _mm_storeu_ps(result_ptr.add(i), result_vec1);
                    _mm_storeu_ps(result_ptr.add(i + 4), result_vec2);
                    _mm_storeu_ps(result_ptr.add(i + 8), result_vec3);
                    _mm_storeu_ps(result_ptr.add(i + 12), result_vec4);

                    i += 16;
                }

                while i + 4 <= len {
                    let a_vec = _mm_loadu_ps(a_ptr.add(i));
                    let b_vec = _mm_loadu_ps(b_ptr.add(i));
                    let result_vec = _mm_add_ps(a_vec, b_vec);
                    _mm_storeu_ps(result_ptr.add(i), result_vec);
                    i += 4;
                }

                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                    i += 1;
                }
            }
        } else {
            // Scalar with prefetching
            unsafe {
                let mut i = 0;
                while i + 64 <= len {
                    if i + 256 < len {
                        _mm_prefetch(a_ptr.add(i + 256) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(b_ptr.add(i + 256) as *const i8, _MM_HINT_T0);
                    }

                    // Unrolled scalar loop
                    for j in 0..64 {
                        *result_ptr.add(i + j) = *a_ptr.add(i + j) + *b_ptr.add(i + j);
                    }
                    i += 64;
                }

                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                    i += 1;
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        Array1::from_vec(result)
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Use Phase 2 implementation for other architectures
        simd_add_f32_fast(a, b)
    }
}

/// PHASE 3: AlignedVec integration for maximum SIMD performance
use crate::simd_aligned::AlignedVec;

#[allow(dead_code)]
pub fn simd_add_aligned_ultra(
    a: &AlignedVec<f32>,
    b: &AlignedVec<f32>,
) -> Result<AlignedVec<f32>, &'static str> {
    if a.len() != b.len() {
        return Err("Arrays must have the same length");
    }

    let len = a.len();
    let mut result =
        AlignedVec::with_capacity(len).map_err(|_| "Failed to allocate aligned memory")?;

    unsafe {
        let a_ptr = a.as_slice().as_ptr();
        let b_ptr = b.as_slice().as_ptr();
        let result_ptr = result.as_slice().as_ptr() as *mut f32;

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;

            if is_x86_feature_detected!("avx2") {
                let mut i = 0;

                // Guaranteed aligned access - use fastest instructions
                while i + 64 <= len {
                    // Prefetch ahead
                    if i + 256 < len {
                        _mm_prefetch(a_ptr.add(i + 256) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(b_ptr.add(i + 256) as *const i8, _MM_HINT_T0);
                    }

                    // 8x AVX2 operations with aligned loads/stores
                    for j in (0..64).step_by(8) {
                        let a_vec = _mm256_load_ps(a_ptr.add(i + j)); // Aligned load
                        let b_vec = _mm256_load_ps(b_ptr.add(i + j)); // Aligned load
                        let result_vec = _mm256_add_ps(a_vec, b_vec);
                        _mm256_store_ps(result_ptr.add(i + j), result_vec); // Aligned store
                    }

                    i += 64;
                }

                // Handle remaining elements
                while i + 8 <= len {
                    let a_vec = _mm256_load_ps(a_ptr.add(i));
                    let b_vec = _mm256_load_ps(b_ptr.add(i));
                    let result_vec = _mm256_add_ps(a_vec, b_vec);
                    _mm256_store_ps(result_ptr.add(i), result_vec);
                    i += 8;
                }

                while i < len {
                    *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                    i += 1;
                }
            } else {
                // Scalar fallback: this x86_64 CPU does not report AVX2 at
                // runtime (e.g. pre-Haswell, or a VM/CI runner with AVX2
                // masked out of CPUID). Previously this branch was entirely
                // absent, so `result.set_len(len)` below was skipped too --
                // the function silently returned a *correctly-typed but
                // empty* `AlignedVec` (len 0, capacity `len`) instead of the
                // element-wise sum, a silent-data-loss bug for any non-AVX2
                // x86_64 caller. Every element must be written here so the
                // unconditional `set_len(len)` after this if/else is sound.
                for i in 0..len {
                    *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                }
            }

            result.set_len(len);
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback for other architectures
            for i in 0..len {
                *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
            }
            result.set_len(len);
        }
    }

    Ok(result)
}

/// Compute element-wise addition of two f64 arrays using unified SIMD operations
#[allow(dead_code)]
pub fn simd_add_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    // Direct implementation to avoid circular dependency
    (a + b).to_owned()
}
