//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{Dwt2dConfig, Dwt2dResult, MemoryPool, ThresholdMethod};
use crate::dwt::{self, Wavelet};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::PlatformCapabilities;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::fmt::Debug;

use super::functions_3::apply_threshold;

#[allow(unused_imports)]
/// Helper function for ceiling division (divide and round up)
/// This replaces the unstable div_ceil method
#[inline]
#[allow(dead_code)]
pub(super) fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}
/// SIMD-optimized threshold function for wavelet coefficients
/// Applies thresholding to a slice of coefficients using SIMD operations when available
#[inline]
#[allow(dead_code)]
pub fn simd_threshold_coefficients(
    coeffs: &mut [f64],
    threshold: f64,
    method: ThresholdMethod,
) {
    let caps = PlatformCapabilities::detect();
    let simd_threshold = 64;
    if coeffs.len() >= simd_threshold && caps.simd_available {
        simd_threshold_avx2(coeffs, threshold, method);
    } else {
        for coeff in coeffs.iter_mut() {
            *coeff = apply_threshold(*coeff, threshold, method);
        }
    }
}
/// AVX2-optimized thresholding implementation
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(dead_code)]
fn simd_threshold_avx2(coeffs: &mut [f64], threshold: f64, method: ThresholdMethod) {
    let len = coeffs.len();
    let simd_len = len - (len % 4);
    unsafe {
        let threshold_vec = _mm256_set1_pd(threshold);
        let zero_vec = _mm256_setzero_pd();
        let one_vec = _mm256_set1_pd(1.0);
        for i in (0..simd_len).step_by(4) {
            let data = _mm256_loadu_pd(coeffs.as_ptr().add(i));
            let result = match method {
                ThresholdMethod::Hard => {
                    let abs_data = _mm256_andnot_pd(_mm256_set1_pd(-0.0), data);
                    let mask = _mm256_cmp_pd(abs_data, threshold_vec, _CMP_GT_OQ);
                    _mm256_and_pd(data, mask)
                }
                ThresholdMethod::Soft => {
                    let abs_data = _mm256_andnot_pd(_mm256_set1_pd(-0.0), data);
                    let mask = _mm256_cmp_pd(abs_data, threshold_vec, _CMP_GT_OQ);
                    let sign_mask = _mm256_cmp_pd(data, zero_vec, _CMP_GE_OQ);
                    let sign = _mm256_blendv_pd(
                        _mm256_set1_pd(-1.0),
                        one_vec,
                        sign_mask,
                    );
                    let shrunk = _mm256_mul_pd(
                        sign,
                        _mm256_sub_pd(abs_data, threshold_vec),
                    );
                    _mm256_and_pd(shrunk, mask)
                }
                ThresholdMethod::Garrote => {
                    let abs_data = _mm256_andnot_pd(_mm256_set1_pd(-0.0), data);
                    let mask = _mm256_cmp_pd(abs_data, threshold_vec, _CMP_GT_OQ);
                    let threshold_sq = _mm256_mul_pd(threshold_vec, threshold_vec);
                    let data_sq = _mm256_mul_pd(data, data);
                    let ratio = _mm256_div_pd(threshold_sq, data_sq);
                    let factor = _mm256_sub_pd(one_vec, ratio);
                    let result = _mm256_mul_pd(data, factor);
                    _mm256_and_pd(result, mask)
                }
            };
            _mm256_storeu_pd(coeffs.as_mut_ptr().add(i), result);
        }
    }
    for coeff in &mut coeffs[simd_len..] {
        *coeff = apply_threshold(*coeff, threshold, method);
    }
}
/// Fallback scalar thresholding for non-x86_64 architectures
#[cfg(not(target_arch = "x86_64"))]
#[inline]
#[allow(dead_code)]
fn simd_threshold_avx2(coeffs: &mut [f64], threshold: f64, method: ThresholdMethod) {
    for coeff in coeffs.iter_mut() {
        *coeff = apply_threshold(*coeff, threshold, method);
    }
}
/// SIMD-optimized energy calculation for large arrays
#[inline]
#[allow(dead_code)]
pub(super) fn simd_calculate_energy(data: &[f64]) -> f64 {
    let caps = PlatformCapabilities::detect();
    let simd_threshold = 64;
    if data.len() >= simd_threshold && caps.simd_available {
        simd_energy_avx2(data)
    } else {
        data.iter().map(|&x| x * x).sum()
    }
}
/// AVX2-optimized energy calculation
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(dead_code)]
fn simd_energy_avx2(data: &[f64]) -> f64 {
    let len = data.len();
    let simd_len = len - (len % 4);
    let mut sum = 0.0;
    unsafe {
        let mut sum_vec = _mm256_setzero_pd();
        for i in (0..simd_len).step_by(4) {
            let data_vec = _mm256_loadu_pd(data.as_ptr().add(i));
            let squared = _mm256_mul_pd(data_vec, data_vec);
            sum_vec = _mm256_add_pd(sum_vec, squared);
        }
        let sum_array: [f64; 4] = std::mem::transmute(sum_vec);
        sum = sum_array.iter().sum();
    }
    sum += data[simd_len..].iter().map(|&x| x * x).sum::<f64>();
    sum
}
/// Fallback scalar energy calculation for non-x86_64 architectures
#[cfg(not(target_arch = "x86_64"))]
#[inline]
#[allow(dead_code)]
fn simd_energy_avx2(data: &[f64]) -> f64 {
    data.iter().map(|&x| x * x).sum()
}
thread_local! {
    static MEMORY_POOL : std::cell::RefCell < MemoryPool > =
    std::cell::RefCell::new(MemoryPool::new());
}
/// Get a temporary buffer from the thread-local memory pool
#[allow(dead_code)]
pub(super) fn get_temp_buffer(size: usize) -> Vec<f64> {
    MEMORY_POOL.with(|pool| pool.borrow_mut().get_buffer(size))
}
/// Return a temporary buffer to the thread-local memory pool
#[allow(dead_code)]
pub(super) fn return_temp_buffer(buffer: Vec<f64>) {
    MEMORY_POOL.with(|pool| pool.borrow_mut().return_buffer(buffer));
}
#[cfg(feature = "parallel")]
/// Type alias for column processing results to reduce complexity
#[allow(dead_code)]
type ColumnResult = (usize, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
/// Performs a single-level 2D discrete wavelet transform with enhanced validation.
///
/// The 2D DWT is computed by applying the 1D DWT first along the rows and then
/// along the columns of the data. This results in four subbands: approximation (LL),
/// horizontal detail (LH), vertical detail (HL), and diagonal detail (HH).
///
/// This function is the 2D equivalent of the 1D `dwt_decompose` function and is useful
/// for image processing applications such as compression, denoising, and feature extraction.
///
/// # Algorithm
///
/// The 2D DWT is computed using separable filtering:
/// 1. Apply 1D DWT to each row of the input data, producing low-pass and high-pass outputs
/// 2. Organize these outputs side by side, maintaining spatial correspondence
/// 3. Apply 1D DWT to each column of both the low-pass and high-pass results
/// 4. This creates the four subbands: LL (approx), LH (detail_h), HL (detail_v), and HH (detail_d)
///
/// # Enhanced Features
///
/// - Comprehensive input validation including NaN/Infinity detection
/// - Numerical stability checks for extreme values
/// - Memory-efficient processing with optional parallel computation
/// - Robust boundary condition handling
/// - Detailed error reporting for debugging
///
/// # Arguments
///
/// * `data` - The input 2D array (image) of any floating-point type
/// * `wavelet` - The wavelet to use for the transform (e.g., Haar, DB1-20, Sym2-20, Coif1-5)
/// * `mode` - The signal extension mode for handling boundaries:
///   - "symmetric" (default): Reflects the signal at boundaries
///   - "periodic": Treats the signal as periodic
///   - "zero": Pads with zeros
///   - "constant": Pads with edge values
///   - "reflect": Similar to symmetric but without repeating edge values
///
/// # Returns
///
/// * A `Dwt2dResult` containing the four subbands of the decomposition
/// * Each subband is approximately half the size of the original in each dimension
///
/// # Errors
///
/// Returns an error if:
/// * The input array is empty
/// * There are issues with the wavelet filters
/// * Numerical conversion problems occur
///
/// # Examples
///
/// Basic usage with a simple 4×4 image:
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt2d::dwt2d_decompose;
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a simple 4x4 "image"
/// let data = Array2::from_shape_vec((4, 4), vec![
///     1.0, 2.0, 3.0, 4.0,
///     5.0, 6.0, 7.0, 8.0,
///     9.0, 10.0, 11.0, 12.0,
///     13.0, 14.0, 15.0, 16.0
/// ]).expect("operation should succeed");
///
/// // Perform 2D DWT using the Haar wavelet
/// let result = dwt2d_decompose(&data, Wavelet::Haar, None).expect("operation should succeed");
///
/// // Check the shape of the result (should be half the original size in each dimension)
/// assert_eq!(result.approx.shape(), &[2, 2]);
/// assert_eq!(result.detail_h.shape(), &[2, 2]);
/// assert_eq!(result.detail_v.shape(), &[2, 2]);
/// assert_eq!(result.detail_d.shape(), &[2, 2]);
/// ```
///
/// Using a different wavelet and boundary extension mode:
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt2d::dwt2d_decompose;
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a simple 16x16 "image" (larger to avoid overflow issues)
/// let mut data = Array2::zeros((16, 16));
/// for i in 0..16 {
///     for j in 0..16 {
///         data[[i, j]] = (i * j)  as f64;
///     }
/// }
///
/// // Perform 2D DWT using the Daubechies 4 wavelet with periodic boundary extension
/// let result = dwt2d_decompose(&data, Wavelet::DB(4), Some("periodic")).expect("operation should succeed");
///
/// // The output size depends on the input size and the wavelet length
/// assert_eq!(result.approx.shape(), &[8, 8]);
/// ```
#[allow(dead_code)]
pub fn dwt2d_decompose<T>(
    data: &Array2<T>,
    wavelet: Wavelet,
    mode: Option<&str>,
) -> SignalResult<Dwt2dResult>
where
    T: Float + NumCast + Debug,
{
    if data.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }
    let (rows, cols) = data.dim();
    if rows < 2 || cols < 2 {
        return Err(
            SignalError::ValueError(
                format!(
                    "Input array dimensions too small: {}x{}. Minimum size is 2x2", rows,
                    cols
                ),
            ),
        );
    }
    const MAX_DIMENSION: usize = 65536;
    if rows > MAX_DIMENSION || cols > MAX_DIMENSION {
        return Err(
            SignalError::ValueError(
                format!(
                    "Input array dimensions too large: {}x{}. Maximum supported size is {}x{}",
                    rows, cols, MAX_DIMENSION, MAX_DIMENSION
                ),
            ),
        );
    }
    let mut data_f64 = Array2::zeros(data.dim());
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut extreme_count = 0;
    for ((i, j), &val) in data.indexed_iter() {
        match NumCast::from(val) {
            Some(converted) => {
                if converted.is_nan() {
                    nan_count += 1;
                    if nan_count <= 5 {
                        eprintln!("Warning: NaN detected at position ({}, {})", i, j);
                    }
                    data_f64[[i, j]] = 0.0;
                } else if converted.is_infinite() {
                    inf_count += 1;
                    if inf_count <= 5 {
                        eprintln!(
                            "Warning: Infinity detected at position ({}, {})", i, j
                        );
                    }
                    data_f64[[i, j]] = if converted.is_sign_positive() {
                        1e10
                    } else {
                        -1e10
                    };
                } else if converted.abs() > 1e12 {
                    extreme_count += 1;
                    if extreme_count <= 5 {
                        eprintln!(
                            "Warning: Extreme value {} detected at position ({}, {})",
                            converted, i, j
                        );
                    }
                    data_f64[[i, j]] = converted;
                } else {
                    data_f64[[i, j]] = converted;
                }
            }
            None => {
                return Err(
                    SignalError::ValueError(
                        format!(
                            "Failed to convert input data to f64 at position ({}, {})",
                            i, j
                        ),
                    ),
                );
            }
        }
    }
    if nan_count > 0 {
        eprintln!("Processed {} NaN values (replaced with 0.0)", nan_count);
    }
    if inf_count > 0 {
        eprintln!("Processed {} infinite values (clamped to ±1e10)", inf_count);
    }
    if extreme_count > 0 {
        eprintln!("Detected {} extreme values (>1e12)", extreme_count);
    }
    let filter_length = match wavelet.get_filter_length() {
        Ok(len) => len,
        Err(_) => {
            return Err(
                SignalError::ValueError(
                    format!(
                        "Invalid wavelet: {:?}. Cannot determine filter length.", wavelet
                    ),
                ),
            );
        }
    };
    let min_size = filter_length.max(4);
    if rows < min_size || cols < min_size {
        return Err(
            SignalError::ValueError(
                format!(
                    "Input dimensions {}x{} too small for wavelet {:?} (requires minimum {}x{})",
                    rows, cols, wavelet, min_size, min_size
                ),
            ),
        );
    }
    let output_rows = div_ceil(rows, 2);
    let output_cols = div_ceil(cols, 2);
    let mut ll = Array2::zeros((output_rows, output_cols));
    let mut lh = Array2::zeros((output_rows, output_cols));
    let mut hl = Array2::zeros((output_rows, output_cols));
    let mut hh = Array2::zeros((output_rows, output_cols));
    let mut rows_lo = Array2::zeros((rows, output_cols));
    let mut rows_hi = Array2::zeros((rows, output_cols));
    #[cfg(feature = "parallel")]
    {
        #[allow(unused_mut)]
        let row_results: Result<Vec<(usize, Vec<f64>, Vec<f64>)>, SignalError> = (0..rows)
            .into_par_iter()
            .map(|i| {
                let row = data_f64.slice(scirs2_core::ndarray::s![i, ..]).to_vec();
                let (approx, detail) = dwt::dwt_decompose(&row, wavelet, mode)
                    .map_err(|e| {
                        SignalError::ComputationError(
                            format!("Row transform failed: {}", e),
                        )
                    })?;
                Ok((i, approx, detail))
            })
            .collect();
        let row_results = row_results?;
        for (i, approx, detail) in row_results {
            for j in 0..approx.len() {
                if j < output_cols {
                    rows_lo[[i, j]] = approx[j];
                    rows_hi[[i, j]] = detail[j];
                }
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..rows {
            let row = data_f64.slice(scirs2_core::ndarray::s![i, ..]).to_vec();
            let (approx, detail) = dwt::dwt_decompose(&row, wavelet, mode)?;
            for j in 0..approx.len() {
                if j < output_cols {
                    rows_lo[[i, j]] = approx[j];
                    rows_hi[[i, j]] = detail[j];
                }
            }
        }
    }
    #[cfg(feature = "parallel")]
    {
        let column_results: Result<Vec<ColumnResult>, SignalError> = (0..output_cols)
            .into_par_iter()
            .map(|j| {
                let col_lo = rows_lo.slice(scirs2_core::ndarray::s![.., j]).to_vec();
                let (approx_lo, detail_lo) = dwt::dwt_decompose(&col_lo, wavelet, mode)
                    .map_err(|e| {
                        SignalError::ComputationError(
                            format!("Column transform failed (low-pass): {}", e),
                        )
                    })?;
                let col_hi = rows_hi.slice(scirs2_core::ndarray::s![.., j]).to_vec();
                let (approx_hi, detail_hi) = dwt::dwt_decompose(&col_hi, wavelet, mode)
                    .map_err(|e| {
                        SignalError::ComputationError(
                            format!("Column transform failed (high-pass): {}", e),
                        )
                    })?;
                Ok((j, approx_lo, detail_lo, approx_hi, detail_hi))
            })
            .collect();
        let column_results = column_results?;
        for (j, approx_lo, detail_lo, approx_hi, detail_hi) in column_results {
            for i in 0..approx_lo.len() {
                if i < output_rows {
                    ll[[i, j]] = approx_lo[i];
                    hl[[i, j]] = detail_lo[i];
                }
            }
            for i in 0..approx_hi.len() {
                if i < output_rows {
                    lh[[i, j]] = approx_hi[i];
                    hh[[i, j]] = detail_hi[i];
                }
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        for j in 0..output_cols {
            let col_lo = rows_lo.slice(scirs2_core::ndarray::s![.., j]).to_vec();
            let (approx, detail) = dwt::dwt_decompose(&col_lo, wavelet, mode)?;
            for i in 0..approx.len() {
                if i < output_rows {
                    ll[[i, j]] = approx[i];
                    hl[[i, j]] = detail[i];
                }
            }
            let col_hi = rows_hi.slice(scirs2_core::ndarray::s![.., j]).to_vec();
            let (approx, detail) = dwt::dwt_decompose(&col_hi, wavelet, mode)?;
            for i in 0..approx.len() {
                if i < output_rows {
                    lh[[i, j]] = approx[i];
                    hh[[i, j]] = detail[i];
                }
            }
        }
    }
    Ok(Dwt2dResult {
        approx: ll,
        detail_h: lh,
        detail_v: hl,
        detail_d: hh,
    })
}
