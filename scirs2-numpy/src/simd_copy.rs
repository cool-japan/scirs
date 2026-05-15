//! SIMD-accelerated copy for non-contiguous-to-contiguous coercion.
//!
//! When Python passes a non-contiguous (strided) NumPy array, it must be gathered
//! into a contiguous buffer before it can be used as an `ndarray::ArrayView`.
//! These routines provide a fast path for that gather operation.
//!
//! ## Dispatch strategy
//!
//! | Platform      | Condition                        | Implementation            |
//! |---------------|----------------------------------|---------------------------|
//! | x86_64        | `avx2` detected at runtime       | AVX2 256-bit gather        |
//! | x86_64        | no avx2 or fallback required     | scalar loop               |
//! | all others    | always                           | scalar loop               |
//!
//! When `stride == 1` the memory is already contiguous; `ptr::copy_nonoverlapping`
//! is used for the fastest possible copy.
//!
//! ## Safety contract
//!
//! Both public functions are `unsafe` because they operate on raw pointers.  The
//! caller must guarantee:
//!
//! - `src` points to a valid, aligned allocation of at least
//!   `n_elements * stride * size_of::<T>()` bytes.
//! - `dst.len() >= n_elements`.
//! - The source and destination ranges do not overlap.
//! - `stride * (n_elements.saturating_sub(1))` fits in `isize` (i.e., no pointer
//!   overflow on the source side).
//! - All `n_elements` source elements are properly initialised.

use std::ptr;

// ── f32 ──────────────────────────────────────────────────────────────────────

/// Copy `n_elements` strided `f32` values from `src` into the contiguous slice
/// `dst`.
///
/// `stride` is the gap, **in elements** (not bytes), between successive source
/// elements. A stride of 1 means the data is already contiguous.
///
/// # Safety
///
/// See the [module-level safety contract](self).
///
/// # Examples
///
/// ```
/// use scirs2_numpy::simd_copy::copy_strided_to_contiguous_f32;
///
/// // Source: every second element → [1.0, 3.0, 5.0]
/// let src = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let mut dst = vec![0.0_f32; 3];
/// unsafe {
///     copy_strided_to_contiguous_f32(src.as_ptr(), &mut dst, 3, 2);
/// }
/// assert_eq!(dst, [1.0, 3.0, 5.0]);
/// ```
pub unsafe fn copy_strided_to_contiguous_f32(
    src: *const f32,
    dst: &mut [f32],
    n_elements: usize,
    stride: usize,
) {
    debug_assert!(
        dst.len() >= n_elements,
        "dst must have at least n_elements slots"
    );

    if stride == 1 {
        // Already contiguous — single memcpy.
        // SAFETY: caller guarantees non-overlap and src validity.
        unsafe {
            ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), n_elements);
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Guard: the index vector holds 8 lanes with offsets [0..7*stride].
        // `7 * stride` must fit in i32 for _mm256_i32gather_ps; divide to
        // avoid multiplication overflow before the comparison.
        const AVX2_LANES: usize = 8;
        if is_x86_feature_detected!("avx2") && stride <= (i32::MAX as usize) / (AVX2_LANES - 1) {
            // SAFETY: we just checked the feature flag.  The stride bound ensures
            // that (AVX2_LANES - 1) * stride <= i32::MAX, so all lane offsets
            // in the gather index vector fit in i32 without overflow.
            unsafe {
                gather_f32_avx2(src, dst, n_elements, stride);
            }
            return;
        }
    }

    // Scalar fallback.
    unsafe {
        scalar_gather_f32(src, dst, n_elements, stride);
    }
}

// ── f64 ──────────────────────────────────────────────────────────────────────

/// Copy `n_elements` strided `f64` values from `src` into the contiguous slice
/// `dst`.
///
/// `stride` is the gap, **in elements** (not bytes), between successive source
/// elements.
///
/// # Safety
///
/// See the [module-level safety contract](self).
///
/// # Examples
///
/// ```
/// use scirs2_numpy::simd_copy::copy_strided_to_contiguous_f64;
///
/// // Source: every third element → [0.0, 3.0, 6.0]
/// let src = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let mut dst = vec![0.0_f64; 3];
/// unsafe {
///     copy_strided_to_contiguous_f64(src.as_ptr(), &mut dst, 3, 3);
/// }
/// assert_eq!(dst, [0.0, 3.0, 6.0]);
/// ```
pub unsafe fn copy_strided_to_contiguous_f64(
    src: *const f64,
    dst: &mut [f64],
    n_elements: usize,
    stride: usize,
) {
    debug_assert!(
        dst.len() >= n_elements,
        "dst must have at least n_elements slots"
    );

    if stride == 1 {
        // Already contiguous — single memcpy.
        unsafe {
            ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), n_elements);
        }
        return;
    }

    // AVX2 gather for f64 (256-bit = 4 × f64) is available but the index vector
    // for _mm256_i64gather_pd requires 64-bit offsets.  Since stride is already
    // usize the conversion is safe when stride fits in i64, which covers all
    // practical cases.
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                gather_f64_avx2(src, dst, n_elements, stride);
            }
            return;
        }
    }

    unsafe {
        scalar_gather_f64(src, dst, n_elements, stride);
    }
}

// ── scalar helpers ────────────────────────────────────────────────────────────

/// Scalar gather for f32.
///
/// # Safety
/// Caller must uphold the module-level contract.
#[inline]
unsafe fn scalar_gather_f32(src: *const f32, dst: &mut [f32], n_elements: usize, stride: usize) {
    for i in 0..n_elements {
        // SAFETY: caller guarantees src validity for stride*n_elements elements.
        *dst.get_unchecked_mut(i) = *src.add(i * stride);
    }
}

/// Scalar gather for f64.
///
/// # Safety
/// Caller must uphold the module-level contract.
#[inline]
unsafe fn scalar_gather_f64(src: *const f64, dst: &mut [f64], n_elements: usize, stride: usize) {
    for i in 0..n_elements {
        *dst.get_unchecked_mut(i) = *src.add(i * stride);
    }
}

// ── AVX2 paths ────────────────────────────────────────────────────────────────

/// AVX2 gather for f32 using `_mm256_i32gather_ps`.
///
/// Processes 8 elements per iteration (256 bits / 32 bits = 8 lanes).
/// A tail scalar loop handles the remainder.
///
/// # Safety
/// - AVX2 must be available (checked by caller via `is_x86_feature_detected!`).
/// - `stride <= i32::MAX` must hold (checked by caller).
/// - All module-level pointer-validity constraints apply.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gather_f32_avx2(src: *const f32, dst: &mut [f32], n_elements: usize, stride: usize) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let stride_i32 = stride as i32;

    // Build a constant index vector [0, stride, 2*stride, …, 7*stride].
    // These are element-wise offsets; the intrinsic interprets them in bytes
    // when scale=4 (4 bytes per f32).
    let vindex = _mm256_set_epi32(
        7 * stride_i32,
        6 * stride_i32,
        5 * stride_i32,
        4 * stride_i32,
        3 * stride_i32,
        2 * stride_i32,
        stride_i32,
        0,
    );

    let chunks = n_elements / 8;
    let remainder = n_elements % 8;

    let mut dst_ptr = dst.as_mut_ptr();

    for chunk in 0..chunks {
        let chunk_src = src.add(chunk * 8 * stride);
        // _mm256_i32gather_ps: gather 8 f32s at base + vindex[i] * scale.
        // scale=4 because we provide element offsets and each f32 is 4 bytes.
        let gathered = _mm256_i32gather_ps(chunk_src, vindex, 4);
        // Store unaligned — dst_ptr may not be 32-byte aligned.
        _mm256_storeu_ps(dst_ptr, gathered);
        dst_ptr = dst_ptr.add(8);
    }

    // Scalar tail.
    let tail_src_base = src.add(chunks * 8 * stride);
    for i in 0..remainder {
        *dst_ptr.add(i) = *tail_src_base.add(i * stride);
    }
}

/// AVX2 gather for f64 using `_mm256_i64gather_pd`.
///
/// Processes 4 elements per iteration (256 bits / 64 bits = 4 lanes).
///
/// # Safety
/// Same as [`gather_f32_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gather_f64_avx2(src: *const f64, dst: &mut [f64], n_elements: usize, stride: usize) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // stride is usize; safe cast because any stride that could OOM the machine
    // would have already caused the allocation to fail.
    let stride_i64 = stride as i64;

    let vindex = _mm256_set_epi64x(3 * stride_i64, 2 * stride_i64, stride_i64, 0);

    let chunks = n_elements / 4;
    let remainder = n_elements % 4;
    let mut dst_ptr = dst.as_mut_ptr();

    for chunk in 0..chunks {
        let chunk_src = src.add(chunk * 4 * stride);
        // scale=8: each f64 occupies 8 bytes; vindex gives element offsets.
        let gathered = _mm256_i64gather_pd(chunk_src, vindex, 8);
        _mm256_storeu_pd(dst_ptr, gathered);
        dst_ptr = dst_ptr.add(4);
    }

    let tail_src_base = src.add(chunks * 4 * stride);
    for i in 0..remainder {
        *dst_ptr.add(i) = *tail_src_base.add(i * stride);
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_stride1_is_memcpy() {
        let src: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let mut dst = vec![0.0_f32; 16];
        unsafe {
            copy_strided_to_contiguous_f32(src.as_ptr(), &mut dst, 16, 1);
        }
        assert_eq!(dst, src);
    }

    #[test]
    fn test_f32_stride2() {
        // Every other element: [0, 2, 4, 6, 8, 10, 12, 14, 16]
        let src: Vec<f32> = (0..18).map(|x| x as f32).collect();
        let expected: Vec<f32> = (0..9).map(|x| (x * 2) as f32).collect();
        let mut dst = vec![0.0_f32; 9];
        unsafe {
            copy_strided_to_contiguous_f32(src.as_ptr(), &mut dst, 9, 2);
        }
        assert_eq!(dst, expected);
    }

    #[test]
    fn test_f32_stride3() {
        let src: Vec<f32> = (0..21).map(|x| x as f32).collect();
        let expected: Vec<f32> = (0..7).map(|x| (x * 3) as f32).collect();
        let mut dst = vec![0.0_f32; 7];
        unsafe {
            copy_strided_to_contiguous_f32(src.as_ptr(), &mut dst, 7, 3);
        }
        assert_eq!(dst, expected);
    }

    #[test]
    fn test_f64_stride1_is_memcpy() {
        let src: Vec<f64> = (0..16).map(|x| x as f64).collect();
        let mut dst = vec![0.0_f64; 16];
        unsafe {
            copy_strided_to_contiguous_f64(src.as_ptr(), &mut dst, 16, 1);
        }
        assert_eq!(dst, src);
    }

    #[test]
    fn test_f64_stride2() {
        let src: Vec<f64> = (0..18).map(|x| x as f64).collect();
        let expected: Vec<f64> = (0..9).map(|x| (x * 2) as f64).collect();
        let mut dst = vec![0.0_f64; 9];
        unsafe {
            copy_strided_to_contiguous_f64(src.as_ptr(), &mut dst, 9, 2);
        }
        assert_eq!(dst, expected);
    }

    #[test]
    fn test_f64_stride4() {
        // 1M-element benchmark documents copy overhead.
        // We use a smaller size here to keep test fast.
        let n = 10_000_usize;
        let stride = 4_usize;
        let src: Vec<f64> = (0..(n * stride)).map(|x| x as f64).collect();
        let expected: Vec<f64> = (0..n).map(|x| (x * stride) as f64).collect();
        let mut dst = vec![0.0_f64; n];
        unsafe {
            copy_strided_to_contiguous_f64(src.as_ptr(), &mut dst, n, stride);
        }
        assert_eq!(dst, expected);
    }

    /// Document the overhead of copying a 1M-element strided f64 array.
    /// This test is not a performance gate — it simply ensures the operation
    /// completes without error and produces the correct first/last values.
    #[test]
    fn benchmark_copy_overhead_documentation() {
        let n = 1_000_000_usize;
        let stride = 3_usize;
        let src: Vec<f64> = (0..(n * stride)).map(|x| x as f64).collect();
        let mut dst = vec![0.0_f64; n];

        let start = std::time::Instant::now();
        unsafe {
            copy_strided_to_contiguous_f64(src.as_ptr(), &mut dst, n, stride);
        }
        let elapsed = start.elapsed();

        // Correctness check: first and last element.
        assert_eq!(dst[0], 0.0);
        assert_eq!(dst[n - 1], ((n - 1) * stride) as f64);

        // Overhead documentation (never fails, only informs).
        eprintln!(
            "copy_strided_to_contiguous_f64: {} elements, stride={}, elapsed={:.2?}",
            n, stride, elapsed
        );
    }
}
