//! Hardware-specific SIMD optimizations
//!
//! This module provides optimized implementations for specific CPU architectures
//! including Intel AVX/AVX2/AVX-512 and ARM Neon instruction sets.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign, One, Zero};
use scirs2_core::parallel_ops::*;

/// Hardware capabilities detection
#[derive(Debug, Clone, Copy)]
pub struct HardwareCapabilities {
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_fma: bool,
    pub has_neon: bool,
}

impl HardwareCapabilities {
    /// Detect available hardware features
    pub fn detect() -> Self {
        HardwareCapabilities {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            has_avx: is_x86_feature_detected!("avx"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            has_avx: false,

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            has_avx2: is_x86_feature_detected!("avx2"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            has_avx2: false,

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            has_avx512: is_x86_feature_detected!("avx512f"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            has_avx512: false,

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            has_fma: is_x86_feature_detected!("fma"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            has_fma: false,

            has_neon: cfg!(target_arch = "aarch64"),
        }
    }

    /// Get the optimal vector width for the current hardware
    pub fn optimal_vector_width(&self) -> usize {
        if self.has_avx512 {
            64 // AVX-512 uses 512-bit vectors (64 bytes)
        } else if self.has_avx2 || self.has_avx {
            32 // AVX/AVX2 uses 256-bit vectors (32 bytes)
        } else {
            16 // ARM Neon and SSE both use 128-bit vectors (16 bytes)
        }
    }
}

/// Hardware-optimized dot product computation
#[allow(dead_code)]
pub fn hardware_optimized_dot<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    capabilities: &HardwareCapabilities,
) -> LinalgResult<F>
where
    F: Float + NumAssign + Zero + Send + Sync + 'static,
{
    if x.len() != y.len() {
        return Err(LinalgError::ShapeError(format!(
            "Vector lengths must match: {} vs {}",
            x.len(),
            y.len()
        )));
    }

    let n = x.len();
    let mut result = F::zero();

    // Choose optimization based on available hardware and type
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f64>() {
        if capabilities.has_avx2 && n >= 16 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let raw_result =
                    avx2_dot_f64(x.as_ptr() as *const f64, y.as_ptr() as *const f64, n)?;
                result = F::from(raw_result).unwrap_or(F::zero());
            }
        } else if capabilities.has_neon && cfg!(target_arch = "aarch64") && n >= 8 {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                let raw_result =
                    neon_dot_f64(x.as_ptr() as *const f64, y.as_ptr() as *const f64, n)?;
                result = F::from(raw_result).unwrap_or(F::zero());
            }
        } else {
            // Fallback to standard implementation
            for i in 0..n {
                result += x[i] * y[i];
            }
        }
    } else if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f32>() {
        if capabilities.has_avx2 && n >= 32 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let raw_result =
                    avx2_dot_f32(x.as_ptr() as *const f32, y.as_ptr() as *const f32, n)?;
                result = F::from(raw_result).unwrap_or(F::zero());
            }
        } else if capabilities.has_neon && cfg!(target_arch = "aarch64") && n >= 16 {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                let raw_result =
                    neon_dot_f32(x.as_ptr() as *const f32, y.as_ptr() as *const f32, n)?;
                result = F::from(raw_result).unwrap_or(F::zero());
            }
        } else {
            // Fallback to standard implementation
            for i in 0..n {
                result += x[i] * y[i];
            }
        }
    } else {
        // Fallback for other types
        for i in 0..n {
            result += x[i] * y[i];
        }
    }

    Ok(result)
}

/// Hardware-optimized matrix-vector multiplication
#[allow(dead_code)]
pub fn hardware_optimized_matvec<F>(
    a: &ArrayView2<F>,
    x: &ArrayView1<F>,
    capabilities: &HardwareCapabilities,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Zero + Send + Sync + 'static,
{
    let (m, n) = a.dim();

    if x.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "Matrix columns ({}) must match vector length ({})",
            n,
            x.len()
        )));
    }

    let mut result = Array1::zeros(m);

    // Choose optimization based on available hardware
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f64>() {
        if capabilities.has_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                avx2_matvec_f64(
                    a.as_ptr() as *const f64,
                    x.as_ptr() as *const f64,
                    result.as_mut_ptr() as *mut f64,
                    m,
                    n,
                    a.strides()[0],
                )?;
            }
        } else if capabilities.has_neon && cfg!(target_arch = "aarch64") {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                neon_matvec_f64(
                    a.as_ptr() as *const f64,
                    x.as_ptr() as *const f64,
                    result.as_mut_ptr() as *mut f64,
                    m,
                    n,
                    a.strides()[0],
                )?;
            }
        } else {
            // Fallback to standard implementation
            return standard_matvec(a, x);
        }
    } else if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f32>() {
        if capabilities.has_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                avx2_matvec_f32(
                    a.as_ptr() as *const f32,
                    x.as_ptr() as *const f32,
                    result.as_mut_ptr() as *mut f32,
                    m,
                    n,
                    a.strides()[0],
                )?;
            }
        } else if capabilities.has_neon && cfg!(target_arch = "aarch64") {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                neon_matvec_f32(
                    a.as_ptr() as *const f32,
                    x.as_ptr() as *const f32,
                    result.as_mut_ptr() as *mut f32,
                    m,
                    n,
                    a.strides()[0],
                )?;
            }
        } else {
            // Fallback to standard implementation
            return standard_matvec(a, x);
        }
    } else {
        // Fallback for other types
        return standard_matvec(a, x);
    }

    Ok(result)
}

/// Standard fallback matrix-vector multiplication
#[allow(dead_code)]
fn standard_matvec<F>(a: &ArrayView2<F>, x: &ArrayView1<F>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Zero,
{
    let (m, n) = a.dim();
    let mut result = Array1::zeros(m);

    for i in 0..m {
        let mut sum = F::zero();
        for j in 0..n {
            sum += a[[i, j]] * x[j];
        }
        result[i] = sum;
    }

    Ok(result)
}

/// AVX2-optimized matrix-vector multiplication for f64
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn avx2_matvec_f64(
    a_ptr: *const f64,
    x_ptr: *const f64,
    y_ptr: *mut f64,
    m: usize,
    n: usize,
    a_stride: isize,
) -> LinalgResult<()> {
    use std::arch::x86_64::*;

    const BLOCK_SIZE: usize = 4; // AVX2 can process 4 f64 values at once

    for i in 0..m {
        let row_ptr = a_ptr.offset(i as isize * a_stride);
        let mut sum = _mm256_setzero_pd();

        // Process 4 elements at a time
        let mut j = 0;
        while j + BLOCK_SIZE <= n {
            let a_vec = _mm256_loadu_pd(row_ptr.add(j));
            let x_vec = _mm256_loadu_pd(x_ptr.add(j));
            sum = _mm256_fmadd_pd(a_vec, x_vec, sum);
            j += BLOCK_SIZE;
        }

        // Horizontal sum of the 4 elements in sum
        let sum_high = _mm256_extractf128_pd(sum, 1);
        let sum_low = _mm256_castpd256_pd128(sum);
        let sum_quad = _mm_add_pd(sum_low, sum_high);
        let sum_dual = _mm_hadd_pd(sum_quad, sum_quad);
        let mut result = _mm_cvtsd_f64(sum_dual);

        // Handle remaining elements
        while j < n {
            result += *row_ptr.add(j) * *x_ptr.add(j);
            j += 1;
        }

        *y_ptr.add(i) = result;
    }

    Ok(())
}

/// AVX2-optimized matrix-vector multiplication for f32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn avx2_matvec_f32(
    a_ptr: *const f32,
    x_ptr: *const f32,
    y_ptr: *mut f32,
    m: usize,
    n: usize,
    a_stride: isize,
) -> LinalgResult<()> {
    use std::arch::x86_64::*;

    const BLOCK_SIZE: usize = 8; // AVX2 can process 8 f32 values at once

    for i in 0..m {
        let row_ptr = a_ptr.offset(i as isize * a_stride);
        let mut sum = _mm256_setzero_ps();

        // Process 8 elements at a time
        let mut j = 0;
        while j + BLOCK_SIZE <= n {
            let a_vec = _mm256_loadu_ps(row_ptr.add(j));
            let x_vec = _mm256_loadu_ps(x_ptr.add(j));
            sum = _mm256_fmadd_ps(a_vec, x_vec, sum);
            j += BLOCK_SIZE;
        }

        // Horizontal sum of the 8 elements in sum
        let sum_high = _mm256_extractf128_ps(sum, 1);
        let sum_low = _mm256_castps256_ps128(sum);
        let sum_quad = _mm_add_ps(sum_low, sum_high);
        let sum_dual = _mm_hadd_ps(sum_quad, sum_quad);
        let sum_final = _mm_hadd_ps(sum_dual, sum_dual);
        let mut result = _mm_cvtss_f32(sum_final);

        // Handle remaining elements
        while j < n {
            result += *row_ptr.add(j) * *x_ptr.add(j);
            j += 1;
        }

        *y_ptr.add(i) = result;
    }

    Ok(())
}

/// ARM Neon-optimized matrix-vector multiplication for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_matvec_f64(
    a_ptr: *const f64,
    x_ptr: *const f64,
    y_ptr: *mut f64,
    m: usize,
    n: usize,
    a_stride: isize,
) -> LinalgResult<()> {
    use std::arch::aarch64::*;

    const BLOCK_SIZE: usize = 2; // Neon can process 2 f64 values at once

    for i in 0..m {
        let row_ptr = a_ptr.offset(i as isize * a_stride);
        let mut sum = vdupq_n_f64(0.0);

        // Process 2 elements at a time
        let mut j = 0;
        while j + BLOCK_SIZE <= n {
            let a_vec = vld1q_f64(row_ptr.add(j));
            let x_vec = vld1q_f64(x_ptr.add(j));
            sum = vfmaq_f64(sum, a_vec, x_vec);
            j += BLOCK_SIZE;
        }

        // Horizontal sum
        let mut result = vaddvq_f64(sum);

        // Handle remaining elements
        while j < n {
            result += *row_ptr.add(j) * *x_ptr.add(j);
            j += 1;
        }

        *y_ptr.add(i) = result;
    }

    Ok(())
}

/// ARM Neon-optimized matrix-vector multiplication for f32
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_matvec_f32(
    a_ptr: *const f32,
    x_ptr: *const f32,
    y_ptr: *mut f32,
    m: usize,
    n: usize,
    a_stride: isize,
) -> LinalgResult<()> {
    use std::arch::aarch64::*;

    const BLOCK_SIZE: usize = 4; // Neon can process 4 f32 values at once

    for i in 0..m {
        let row_ptr = a_ptr.offset(i as isize * a_stride);
        let mut sum = vdupq_n_f32(0.0);

        // Process 4 elements at a time
        let mut j = 0;
        while j + BLOCK_SIZE <= n {
            let a_vec = vld1q_f32(row_ptr.add(j));
            let x_vec = vld1q_f32(x_ptr.add(j));
            sum = vfmaq_f32(sum, a_vec, x_vec);
            j += BLOCK_SIZE;
        }

        // Horizontal sum
        let mut result = vaddvq_f32(sum);

        // Handle remaining elements
        while j < n {
            result += *row_ptr.add(j) * *x_ptr.add(j);
            j += 1;
        }

        *y_ptr.add(i) = result;
    }

    Ok(())
}

/// Standard fallback dot product
#[allow(dead_code)]
fn standard_dot<F>(x: &ArrayView1<F>, y: &ArrayView1<F>) -> F
where
    F: Float + NumAssign + Zero,
{
    let mut result = F::zero();
    for (a, b) in x.iter().zip(y.iter()) {
        result += *a * *b;
    }
    result
}

/// AVX2-optimized dot product for f64
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn avx2_dot_f64(_x_ptr: *const f64, yptr: *const f64, n: usize) -> LinalgResult<f64> {
    use std::arch::x86_64::*;

    const BLOCK_SIZE: usize = 4;
    let mut sum = _mm256_setzero_pd();

    // Process 4 elements at a time
    let mut i = 0;
    while i + BLOCK_SIZE <= n {
        let x_vec = _mm256_loadu_pd(_x_ptr.add(i));
        let y_vec = _mm256_loadu_pd(yptr.add(i));
        sum = _mm256_fmadd_pd(x_vec, y_vec, sum);
        i += BLOCK_SIZE;
    }

    // Horizontal sum
    let sum_high = _mm256_extractf128_pd(sum, 1);
    let sum_low = _mm256_castpd256_pd128(sum);
    let sum_quad = _mm_add_pd(sum_low, sum_high);
    let sum_dual = _mm_hadd_pd(sum_quad, sum_quad);
    let mut result = _mm_cvtsd_f64(sum_dual);

    // Handle remaining elements
    while i < n {
        result += *_x_ptr.add(i) * *yptr.add(i);
        i += 1;
    }

    Ok(result)
}

/// AVX2-optimized dot product for f32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn avx2_dot_f32(_x_ptr: *const f32, yptr: *const f32, n: usize) -> LinalgResult<f32> {
    use std::arch::x86_64::*;

    const BLOCK_SIZE: usize = 8;
    let mut sum = _mm256_setzero_ps();

    // Process 8 elements at a time
    let mut i = 0;
    while i + BLOCK_SIZE <= n {
        let x_vec = _mm256_loadu_ps(_x_ptr.add(i));
        let y_vec = _mm256_loadu_ps(yptr.add(i));
        sum = _mm256_fmadd_ps(x_vec, y_vec, sum);
        i += BLOCK_SIZE;
    }

    // Horizontal sum
    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let sum_quad = _mm_add_ps(sum_low, sum_high);
    let sum_dual = _mm_hadd_ps(sum_quad, sum_quad);
    let sum_final = _mm_hadd_ps(sum_dual, sum_dual);
    let mut result = _mm_cvtss_f32(sum_final);

    // Handle remaining elements
    while i < n {
        result += *_x_ptr.add(i) * *yptr.add(i);
        i += 1;
    }

    Ok(result)
}

/// ARM Neon-optimized dot product for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot_f64(_x_ptr: *const f64, yptr: *const f64, n: usize) -> LinalgResult<f64> {
    use std::arch::aarch64::*;

    const BLOCK_SIZE: usize = 2;
    let mut sum = vdupq_n_f64(0.0);

    // Process 2 elements at a time
    let mut i = 0;
    while i + BLOCK_SIZE <= n {
        let x_vec = vld1q_f64(_x_ptr.add(i));
        let y_vec = vld1q_f64(yptr.add(i));
        sum = vfmaq_f64(sum, x_vec, y_vec);
        i += BLOCK_SIZE;
    }

    // Horizontal sum
    let mut result = vaddvq_f64(sum);

    // Handle remaining elements
    while i < n {
        result += *_x_ptr.add(i) * *yptr.add(i);
        i += 1;
    }

    Ok(result)
}

/// ARM Neon-optimized dot product for f32
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot_f32(_x_ptr: *const f32, yptr: *const f32, n: usize) -> LinalgResult<f32> {
    use std::arch::aarch64::*;

    const BLOCK_SIZE: usize = 4;
    let mut sum = vdupq_n_f32(0.0);

    // Process 4 elements at a time
    let mut i = 0;
    while i + BLOCK_SIZE <= n {
        let x_vec = vld1q_f32(_x_ptr.add(i));
        let y_vec = vld1q_f32(yptr.add(i));
        sum = vfmaq_f32(sum, x_vec, y_vec);
        i += BLOCK_SIZE;
    }

    // Horizontal sum
    let mut result = vaddvq_f32(sum);

    // Handle remaining elements
    while i < n {
        result += *_x_ptr.add(i) * *yptr.add(i);
        i += 1;
    }

    Ok(result)
}

/// SIMD-Parallel hybrid matrix multiplication for large matrices
///
/// This function combines SIMD vectorization with parallel processing
/// to achieve optimal performance for large matrix operations.
#[allow(dead_code)]
pub fn simd_parallel_gemm<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    workers: usize,
    capabilities: &HardwareCapabilities,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Zero + One + Send + Sync + 'static,
{
    let (m, k) = a.dim();
    let (k2, n) = b.dim();

    if k != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions incompatible: {m}x{k} * {k2}x{n}"
        )));
    }

    // For small matrices, use sequential SIMD
    if m * n < 10000 || workers == 1 {
        return hardware_optimized_gemm(a, b, capabilities);
    }

    let mut result = Array2::zeros((m, n));

    // Determine optimal block size based on hardware capabilities
    let simd_width = capabilities.optimal_vector_width();
    let cache_blocksize = if capabilities.has_avx2 {
        256 // Medium blocks for AVX2
    } else {
        128 // Smaller blocks for SSE/Neon
    };

    // Parallel block processing using scirs2_core parallel operations
    let chunks: Vec<(usize, usize)> = (0..workers)
        .map(|worker_id| {
            let rows_per_worker = m.div_ceil(workers);
            let start_row = worker_id * rows_per_worker;
            let end_row = ((worker_id + 1) * rows_per_worker).min(m);
            (start_row, end_row)
        })
        .filter(|(start, end)| start < end)
        .collect();

    let results: Vec<Array2<F>> = parallel_map(&chunks, |(start_row, end_row)| {
        let a_block = a.slice(s![*start_row..*end_row, ..]);
        hardware_optimized_gemm_block(&a_block, b, capabilities, cache_blocksize, simd_width)
            .unwrap()
    });

    // Assemble results
    for (i, (start_row, _)) in chunks.iter().enumerate() {
        let block_result = &results[i];
        for (row_idx, row) in block_result
            .axis_iter(scirs2_core::ndarray::Axis(0))
            .enumerate()
        {
            for (col_idx, &val) in row.iter().enumerate() {
                result[[start_row + row_idx, col_idx]] = val;
            }
        }
    }

    Ok(result)
}

/// Hardware-optimized GEMM for sequential execution
#[allow(dead_code)]
fn hardware_optimized_gemm<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    capabilities: &HardwareCapabilities,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Zero + Send + Sync + 'static,
{
    let (m, k) = a.dim();
    let (_, n) = b.dim();

    let mut result = Array2::zeros((m, n));
    let cache_blocksize = capabilities.optimal_vector_width() * 4;

    // Blocked GEMM with SIMD optimization
    for ii in (0..m).step_by(cache_blocksize) {
        for jj in (0..n).step_by(cache_blocksize) {
            for kk in (0..k).step_by(cache_blocksize) {
                let i_end = (ii + cache_blocksize).min(m);
                let j_end = (jj + cache_blocksize).min(n);
                let k_end = (kk + cache_blocksize).min(k);

                // Process each block using SIMD operations
                for i in ii..i_end {
                    for j in jj..j_end {
                        let a_row = a.slice(s![i, kk..k_end]);
                        let b_col = b.slice(s![kk..k_end, j]);

                        // Use hardware-optimized dot product
                        let dot_result = hardware_optimized_dot(&a_row, &b_col, capabilities)?;
                        result[[i, j]] += dot_result;
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Hardware-optimized block GEMM for worker threads
#[allow(dead_code)]
fn hardware_optimized_gemm_block<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    capabilities: &HardwareCapabilities,
    cache_blocksize: usize,
    _simd_width: usize,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Zero + Send + Sync + 'static,
{
    let (m, k) = a.dim();
    let (_, n) = b.dim();

    let mut result = Array2::zeros((m, n));

    // Optimized block multiplication
    for ii in (0..m).step_by(cache_blocksize) {
        for jj in (0..n).step_by(cache_blocksize) {
            for kk in (0..k).step_by(cache_blocksize) {
                let i_end = (ii + cache_blocksize).min(m);
                let j_end = (jj + cache_blocksize).min(n);
                let k_end = (kk + cache_blocksize).min(k);

                for i in ii..i_end {
                    for j in jj..j_end {
                        let a_row = a.slice(s![i, kk..k_end]);
                        let b_col = b.slice(s![kk..k_end, j]);

                        let dot_result = hardware_optimized_dot(&a_row, &b_col, capabilities)?;
                        result[[i, j]] += dot_result;
                    }
                }
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_hardware_capabilities_detection() {
        let caps = HardwareCapabilities::detect();
        // Just ensure it doesn't panic and returns reasonable values
        assert!(caps.optimal_vector_width() >= 16);
        assert!(caps.optimal_vector_width() <= 64);
    }

    #[test]
    fn test_hardware_optimized_dot_product() {
        let caps = HardwareCapabilities::detect();
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = array![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let result = hardware_optimized_dot(&x.view(), &y.view(), &caps).unwrap();
        let expected = 120.0; // 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1

        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_hardware_optimized_matvec() {
        let caps = HardwareCapabilities::detect();
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x = array![1.0, 2.0, 3.0];

        let result = hardware_optimized_matvec(&a.view(), &x.view(), &caps).unwrap();
        let expected = array![14.0, 32.0]; // [1*1+2*2+3*3, 4*1+5*2+6*3]

        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-10);
        }
    }
}
