//! SIMD-accelerated rounding operations
//!
//! This module provides SIMD implementations for floor, ceil, round, and trunc
//! using AVX2/SSE4.1/NEON intrinsics.

use ndarray::{Array1, ArrayView1};

/// SIMD-accelerated floor for f32 arrays
#[allow(dead_code)]
pub fn simd_floor_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::<f32>::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x.floor());
    }

    let mut result = Array1::<f32>::zeros(len);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx") {
            unsafe {
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice: &mut [f32] =
                    result.as_slice_mut().expect("Test operation failed");
                let mut i = 0;

                // Process 8 f32s at a time with AVX
                while i + 8 <= len {
                    let vec = _mm256_loadu_ps(input_slice.as_ptr().add(i));
                    let floored = _mm256_floor_ps(vec);
                    _mm256_storeu_ps(result_slice.as_mut_ptr().add(i), floored);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = input_slice[j].floor();
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe {
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice = result.as_slice_mut().expect("Test operation failed");
                let mut i = 0;

                // Process 4 f32s at a time with SSE4.1
                while i + 4 <= len {
                    let vec = _mm_loadu_ps(input_slice.as_ptr().add(i));
                    let floored = _mm_floor_ps(vec);
                    _mm_storeu_ps(result_slice.as_mut_ptr().add(i), floored);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = input_slice[j].floor();
                }

                return result;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let input_slice = input.as_slice().expect("contiguity checked above");
            let result_slice: &mut [f32] = result.as_slice_mut().expect("Test operation failed");
            let mut i = 0;

            // Process 4 f32s at a time with NEON
            while i + 4 <= len {
                let vec = vld1q_f32(input_slice.as_ptr().add(i));
                let floored = vrndmq_f32(vec);
                vst1q_f32(result_slice.as_mut_ptr().add(i), floored);
                i += 4;
            }

            // Handle remaining elements
            for j in i..len {
                result_slice[j] = input_slice[j].floor();
            }

            return result;
        }
    }

    // Scalar fallback
    #[allow(unreachable_code)]
    input.mapv(|x| x.floor())
}

/// SIMD-accelerated floor for f64 arrays
#[allow(dead_code)]
pub fn simd_floor_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::<f64>::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x.floor());
    }

    let mut result = Array1::<f64>::zeros(len);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx") {
            unsafe {
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice: &mut [f64] =
                    result.as_slice_mut().expect("Test operation failed");
                let mut i = 0;

                // Process 4 f64s at a time with AVX
                while i + 4 <= len {
                    let vec = _mm256_loadu_pd(input_slice.as_ptr().add(i));
                    let floored = _mm256_floor_pd(vec);
                    _mm256_storeu_pd(result_slice.as_mut_ptr().add(i), floored);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = input_slice[j].floor();
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe {
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice = result.as_slice_mut().expect("Test operation failed");
                let mut i = 0;

                // Process 2 f64s at a time with SSE4.1
                while i + 2 <= len {
                    let vec = _mm_loadu_pd(input_slice.as_ptr().add(i));
                    let floored = _mm_floor_pd(vec);
                    _mm_storeu_pd(result_slice.as_mut_ptr().add(i), floored);
                    i += 2;
                }

                // Handle remaining elements
                for j in i..len {
                    result_slice[j] = input_slice[j].floor();
                }

                return result;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let input_slice = input.as_slice().expect("contiguity checked above");
            let result_slice: &mut [f64] = result.as_slice_mut().expect("Test operation failed");
            let mut i = 0;

            // Process 2 f64s at a time with NEON
            while i + 2 <= len {
                let vec = vld1q_f64(input_slice.as_ptr().add(i));
                let floored = vrndmq_f64(vec);
                vst1q_f64(result_slice.as_mut_ptr().add(i), floored);
                i += 2;
            }

            // Handle remaining elements
            for j in i..len {
                result_slice[j] = input_slice[j].floor();
            }

            return result;
        }
    }

    // Scalar fallback
    #[allow(unreachable_code)]
    input.mapv(|x| x.floor())
}

/// SIMD-accelerated ceil for f32 arrays
#[allow(dead_code)]
pub fn simd_ceil_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::<f32>::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x.ceil());
    }

    let mut result = Array1::<f32>::zeros(len);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx") {
            unsafe {
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice: &mut [f32] =
                    result.as_slice_mut().expect("Test operation failed");
                let mut i = 0;

                while i + 8 <= len {
                    let vec = _mm256_loadu_ps(input_slice.as_ptr().add(i));
                    let ceiled = _mm256_ceil_ps(vec);
                    _mm256_storeu_ps(result_slice.as_mut_ptr().add(i), ceiled);
                    i += 8;
                }

                for j in i..len {
                    result_slice[j] = input_slice[j].ceil();
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe {
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice = result.as_slice_mut().expect("Test operation failed");
                let mut i = 0;

                while i + 4 <= len {
                    let vec = _mm_loadu_ps(input_slice.as_ptr().add(i));
                    let ceiled = _mm_ceil_ps(vec);
                    _mm_storeu_ps(result_slice.as_mut_ptr().add(i), ceiled);
                    i += 4;
                }

                for j in i..len {
                    result_slice[j] = input_slice[j].ceil();
                }

                return result;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let input_slice = input.as_slice().expect("contiguity checked above");
            let result_slice: &mut [f32] = result.as_slice_mut().expect("Test operation failed");
            let mut i = 0;

            while i + 4 <= len {
                let vec = vld1q_f32(input_slice.as_ptr().add(i));
                let ceiled = vrndpq_f32(vec);
                vst1q_f32(result_slice.as_mut_ptr().add(i), ceiled);
                i += 4;
            }

            for j in i..len {
                result_slice[j] = input_slice[j].ceil();
            }

            return result;
        }
    }

    // Scalar fallback
    #[allow(unreachable_code)]
    input.mapv(|x| x.ceil())
}

/// SIMD-accelerated ceil for f64 arrays
#[allow(dead_code)]
pub fn simd_ceil_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::<f64>::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x.ceil());
    }

    let mut result = Array1::<f64>::zeros(len);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx") {
            unsafe {
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice: &mut [f64] =
                    result.as_slice_mut().expect("Test operation failed");
                let mut i = 0;

                while i + 4 <= len {
                    let vec = _mm256_loadu_pd(input_slice.as_ptr().add(i));
                    let ceiled = _mm256_ceil_pd(vec);
                    _mm256_storeu_pd(result_slice.as_mut_ptr().add(i), ceiled);
                    i += 4;
                }

                for j in i..len {
                    result_slice[j] = input_slice[j].ceil();
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe {
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice = result.as_slice_mut().expect("Test operation failed");
                let mut i = 0;

                while i + 2 <= len {
                    let vec = _mm_loadu_pd(input_slice.as_ptr().add(i));
                    let ceiled = _mm_ceil_pd(vec);
                    _mm_storeu_pd(result_slice.as_mut_ptr().add(i), ceiled);
                    i += 2;
                }

                for j in i..len {
                    result_slice[j] = input_slice[j].ceil();
                }

                return result;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let input_slice = input.as_slice().expect("contiguity checked above");
            let result_slice: &mut [f64] = result.as_slice_mut().expect("Test operation failed");
            let mut i = 0;

            while i + 2 <= len {
                let vec = vld1q_f64(input_slice.as_ptr().add(i));
                let ceiled = vrndpq_f64(vec);
                vst1q_f64(result_slice.as_mut_ptr().add(i), ceiled);
                i += 2;
            }

            for j in i..len {
                result_slice[j] = input_slice[j].ceil();
            }

            return result;
        }
    }

    // Scalar fallback
    #[allow(unreachable_code)]
    input.mapv(|x| x.ceil())
}

/// SIMD-accelerated round for f32 arrays
#[allow(dead_code)]
pub fn simd_round_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::<f32>::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x.round());
    }

    let mut result = Array1::<f32>::zeros(len);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx") {
            unsafe {
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice: &mut [f32] =
                    result.as_slice_mut().expect("Test operation failed");
                let mut i = 0;

                while i + 8 <= len {
                    let vec = _mm256_loadu_ps(input_slice.as_ptr().add(i));
                    // Round to nearest even (banker's rounding)
                    let rounded =
                        _mm256_round_ps(vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    _mm256_storeu_ps(result_slice.as_mut_ptr().add(i), rounded);
                    i += 8;
                }

                for j in i..len {
                    result_slice[j] = input_slice[j].round();
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe {
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice = result.as_slice_mut().expect("Test operation failed");
                let mut i = 0;

                while i + 4 <= len {
                    let vec = _mm_loadu_ps(input_slice.as_ptr().add(i));
                    let rounded = _mm_round_ps(vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    _mm_storeu_ps(result_slice.as_mut_ptr().add(i), rounded);
                    i += 4;
                }

                for j in i..len {
                    result_slice[j] = input_slice[j].round();
                }

                return result;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let input_slice = input.as_slice().expect("contiguity checked above");
            let result_slice: &mut [f32] = result.as_slice_mut().expect("Test operation failed");
            let mut i = 0;

            while i + 4 <= len {
                let vec = vld1q_f32(input_slice.as_ptr().add(i));
                let rounded = vrndnq_f32(vec);
                vst1q_f32(result_slice.as_mut_ptr().add(i), rounded);
                i += 4;
            }

            for j in i..len {
                result_slice[j] = input_slice[j].round();
            }

            return result;
        }
    }

    // Scalar fallback
    #[allow(unreachable_code)]
    input.mapv(|x| x.round())
}

/// SIMD-accelerated round for f64 arrays
#[allow(dead_code)]
pub fn simd_round_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::<f64>::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x.round());
    }

    let mut result = Array1::<f64>::zeros(len);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx") {
            unsafe {
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice: &mut [f64] =
                    result.as_slice_mut().expect("Test operation failed");
                let mut i = 0;

                while i + 4 <= len {
                    let vec = _mm256_loadu_pd(input_slice.as_ptr().add(i));
                    let rounded =
                        _mm256_round_pd(vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    _mm256_storeu_pd(result_slice.as_mut_ptr().add(i), rounded);
                    i += 4;
                }

                for j in i..len {
                    result_slice[j] = input_slice[j].round();
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe {
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice = result.as_slice_mut().expect("Test operation failed");
                let mut i = 0;

                while i + 2 <= len {
                    let vec = _mm_loadu_pd(input_slice.as_ptr().add(i));
                    let rounded = _mm_round_pd(vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    _mm_storeu_pd(result_slice.as_mut_ptr().add(i), rounded);
                    i += 2;
                }

                for j in i..len {
                    result_slice[j] = input_slice[j].round();
                }

                return result;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let input_slice = input.as_slice().expect("contiguity checked above");
            let result_slice: &mut [f64] = result.as_slice_mut().expect("Test operation failed");
            let mut i = 0;

            while i + 2 <= len {
                let vec = vld1q_f64(input_slice.as_ptr().add(i));
                let rounded = vrndnq_f64(vec);
                vst1q_f64(result_slice.as_mut_ptr().add(i), rounded);
                i += 2;
            }

            for j in i..len {
                result_slice[j] = input_slice[j].round();
            }

            return result;
        }
    }

    // Scalar fallback
    #[allow(unreachable_code)]
    input.mapv(|x| x.round())
}

/// SIMD-accelerated trunc for f32 arrays (round toward zero)
#[allow(dead_code)]
pub fn simd_trunc_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::<f32>::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x.trunc());
    }

    let mut result = Array1::<f32>::zeros(len);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx") {
            unsafe {
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice: &mut [f32] =
                    result.as_slice_mut().expect("Test operation failed");
                let mut i = 0;

                while i + 8 <= len {
                    let vec = _mm256_loadu_ps(input_slice.as_ptr().add(i));
                    // Round toward zero (truncate)
                    let trunced = _mm256_round_ps(vec, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                    _mm256_storeu_ps(result_slice.as_mut_ptr().add(i), trunced);
                    i += 8;
                }

                for j in i..len {
                    result_slice[j] = input_slice[j].trunc();
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe {
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice = result.as_slice_mut().expect("Test operation failed");
                let mut i = 0;

                while i + 4 <= len {
                    let vec = _mm_loadu_ps(input_slice.as_ptr().add(i));
                    let trunced = _mm_round_ps(vec, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                    _mm_storeu_ps(result_slice.as_mut_ptr().add(i), trunced);
                    i += 4;
                }

                for j in i..len {
                    result_slice[j] = input_slice[j].trunc();
                }

                return result;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let input_slice = input.as_slice().expect("contiguity checked above");
            let result_slice: &mut [f32] = result.as_slice_mut().expect("Test operation failed");
            let mut i = 0;

            while i + 4 <= len {
                let vec = vld1q_f32(input_slice.as_ptr().add(i));
                let trunced = vrndq_f32(vec);
                vst1q_f32(result_slice.as_mut_ptr().add(i), trunced);
                i += 4;
            }

            for j in i..len {
                result_slice[j] = input_slice[j].trunc();
            }

            return result;
        }
    }

    // Scalar fallback
    #[allow(unreachable_code)]
    input.mapv(|x| x.trunc())
}

/// SIMD-accelerated trunc for f64 arrays (round toward zero)
#[allow(dead_code)]
pub fn simd_trunc_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::<f64>::zeros(0);
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.mapv(|x| x.trunc());
    }

    let mut result = Array1::<f64>::zeros(len);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx") {
            unsafe {
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice: &mut [f64] =
                    result.as_slice_mut().expect("Test operation failed");
                let mut i = 0;

                while i + 4 <= len {
                    let vec = _mm256_loadu_pd(input_slice.as_ptr().add(i));
                    let trunced = _mm256_round_pd(vec, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                    _mm256_storeu_pd(result_slice.as_mut_ptr().add(i), trunced);
                    i += 4;
                }

                for j in i..len {
                    result_slice[j] = input_slice[j].trunc();
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe {
                let input_slice = input.as_slice().expect("contiguity checked above");
                let result_slice = result.as_slice_mut().expect("Test operation failed");
                let mut i = 0;

                while i + 2 <= len {
                    let vec = _mm_loadu_pd(input_slice.as_ptr().add(i));
                    let trunced = _mm_round_pd(vec, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                    _mm_storeu_pd(result_slice.as_mut_ptr().add(i), trunced);
                    i += 2;
                }

                for j in i..len {
                    result_slice[j] = input_slice[j].trunc();
                }

                return result;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let input_slice = input.as_slice().expect("contiguity checked above");
            let result_slice: &mut [f64] = result.as_slice_mut().expect("Test operation failed");
            let mut i = 0;

            while i + 2 <= len {
                let vec = vld1q_f64(input_slice.as_ptr().add(i));
                let trunced = vrndq_f64(vec);
                vst1q_f64(result_slice.as_mut_ptr().add(i), trunced);
                i += 2;
            }

            for j in i..len {
                result_slice[j] = input_slice[j].trunc();
            }

            return result;
        }
    }

    // Scalar fallback
    #[allow(unreachable_code)]
    input.mapv(|x| x.trunc())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_floor_f32() {
        let x = array![1.5f32, -1.5, 2.0, -2.0, 0.9, -0.9];
        let result = simd_floor_f32(&x.view());
        let expected = array![1.0f32, -2.0, 2.0, -2.0, 0.0, -1.0];
        for i in 0..x.len() {
            assert_eq!(result[i], expected[i], "floor mismatch at index {}", i);
        }
    }

    #[test]
    fn test_ceil_f32() {
        let x = array![1.5f32, -1.5, 2.0, -2.0, 0.9, -0.9];
        let result = simd_ceil_f32(&x.view());
        let expected = array![2.0f32, -1.0, 2.0, -2.0, 1.0, 0.0];
        for i in 0..x.len() {
            assert_eq!(result[i], expected[i], "ceil mismatch at index {}", i);
        }
    }

    #[test]
    fn test_trunc_f32() {
        let x = array![1.5f32, -1.5, 2.0, -2.0, 0.9, -0.9];
        let result = simd_trunc_f32(&x.view());
        let expected = array![1.0f32, -1.0, 2.0, -2.0, 0.0, 0.0];
        for i in 0..x.len() {
            assert_eq!(result[i], expected[i], "trunc mismatch at index {}", i);
        }
    }

    #[test]
    fn test_trunc_large_array() {
        let size = 1000;
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push(((i as f32) - 500.0) * 0.137);
        }
        let x = Array1::from_vec(data);
        let result = simd_trunc_f32(&x.view());

        for i in 0..size {
            let expected = x[i].trunc();
            assert_eq!(result[i], expected, "trunc mismatch at index {}", i);
        }
    }
}
