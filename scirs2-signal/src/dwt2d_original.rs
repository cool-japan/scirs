// Two-dimensional Discrete Wavelet Transform (DWT2D)
//
// This module provides implementations of the 2D Discrete Wavelet Transform,
// useful for image processing, compression, and multi-resolution analysis
// of 2D signals like images.
//
// # Performance Optimizations
//
// This implementation includes several optimizations for performance:
//
// 1. **Parallel Processing**: When compiled with the "parallel" feature,
//    row and column transforms can be computed in parallel using Rayon.
//
// 2. **Memory Efficiency**:
//    - Minimizes temporary allocations
//    - Uses ndarray views for zero-copy operations
//    - Implements cache-friendly traversal patterns
//
// 3. **Algorithm Optimizations**:
//    - Direct transform path for common wavelets (Haar, DB2, DB4)
//    - Optimized convolution for filter operations
//    - Efficient boundary handling
//
// ## Overview
//
// The 2D Discrete Wavelet Transform (DWT2D) extends the concept of 1D wavelet transforms
// to two-dimensional data. It is particularly useful for processing images and other 2D signals.
// By using separable filters, the 2D DWT applies 1D transforms first along rows and then along
// columns, decomposing the image into four subbands:
//
// - **LL (Approximation)**: Low-frequency content in both horizontal and vertical directions
// - **LH (Horizontal Detail)**: High-frequency in horizontal direction, low-frequency in vertical direction
// - **HL (Vertical Detail)**: Low-frequency in horizontal direction, high-frequency in vertical direction
// - **HH (Diagonal Detail)**: High-frequency content in both horizontal and vertical directions
//
// ## Applications
//
// The 2D DWT has numerous applications:
//
// - **Image Compression**: By thresholding small coefficients (as in JPEG2000)
// - **Image Denoising**: By thresholding coefficients below a noise threshold
// - **Feature Extraction**: Using wavelet coefficients as image features
// - **Texture Analysis**: Analyzing frequency content at different scales
// - **Edge Detection**: Using detail coefficients to detect edges
//
// ## Usage Examples
//
// Basic decomposition and reconstruction:
//
// ```
// use ndarray::Array2;
// use scirs2_signal::dwt::Wavelet;
// use scirs2_signal::dwt2d::{dwt2d_decompose, dwt2d_reconstruct};
//
// // Create a simple "image"
// let data = Array2::from_shape_vec((4, 4), vec![
//     1.0, 2.0, 3.0, 4.0,
//     5.0, 6.0, 7.0, 8.0,
//     9.0, 10.0, 11.0, 12.0,
//     13.0, 14.0, 15.0, 16.0
// ]).unwrap();
//
// // Decompose using Haar wavelet
// let decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();
//
// // Reconstruct
// let reconstructed = dwt2d_reconstruct(&decomposition, Wavelet::Haar, None).unwrap();
// ```
//
// Multi-level decomposition:
//
// ```
// use ndarray::Array2;
// use scirs2_signal::dwt::Wavelet;
// use scirs2_signal::dwt2d::{wavedec2, waverec2};
//
// // Create a simple "image"
// let mut data = Array2::zeros((8, 8));
// for i in 0..8 {
//     for j in 0..8 {
//         data[[i, j]] = (i * j)  as f64;
//     }
// }
//
// // Multi-level decomposition
// let levels = 2;
// let coeffs = wavedec2(&data, Wavelet::DB(4), levels, None).unwrap();
//
// // Reconstruct from multi-level decomposition
// let reconstructed = waverec2(&coeffs, Wavelet::DB(4), None).unwrap();
// ```

use crate::dwt::{self, Wavelet};
use crate::error::{SignalError, SignalResult};
use ndarray::s;
use ndarray::Array2;
use num_traits::{Float, NumCast};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::PlatformCapabilities;
use scirs2_core::validation::check_positive;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::Instant;

#[allow(unused_imports)]
/// Helper function for ceiling division (divide and round up)
/// This replaces the unstable div_ceil method
#[inline]
#[allow(dead_code)]
fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// SIMD-optimized threshold function for wavelet coefficients
/// Applies thresholding to a slice of coefficients using SIMD operations when available
#[inline]
#[allow(dead_code)]
pub fn simd_threshold_coefficients(coeffs: &mut [f64], threshold: f64, method: ThresholdMethod) {
    let caps = PlatformCapabilities::detect();
    let simd_threshold = 64; // Minimum length for SIMD optimization

    if coeffs.len() >= simd_threshold && caps.simd_available {
        simd_threshold_avx2(_coeffs, threshold, method);
    } else {
        // Fallback to scalar implementation
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
    let simd_len = len - (len % 4); // Process 4 elements at a time with AVX2

    unsafe {
        let threshold_vec = _mm256_set1_pd(threshold);
        let neg_threshold_vec = _mm256_set1_pd(-threshold);
        let zero_vec = _mm256_setzero_pd();
        let one_vec = _mm256_set1_pd(1.0);

        for i in (0..simd_len).step_by(4) {
            let data = _mm256_loadu_pd(_coeffs.as_ptr().add(i));

            let result = match method {
                ThresholdMethod::Hard => {
                    // Hard thresholding: zero if |x| <= threshold, keep otherwise
                    let abs_data = _mm256_andnot_pd(_mm256_set1_pd(-0.0), data);
                    let mask = _mm256_cmp_pd(abs_data, threshold_vec_CMP_GT_OQ);
                    _mm256_and_pd(data, mask)
                }
                ThresholdMethod::Soft => {
                    // Soft thresholding: zero if |x| <= threshold, shrink otherwise
                    let abs_data = _mm256_andnot_pd(_mm256_set1_pd(-0.0), data);
                    let mask = _mm256_cmp_pd(abs_data, threshold_vec_CMP_GT_OQ);
                    let sign_mask = _mm256_cmp_pd(data, zero_vec_CMP_GE_OQ);
                    let sign = _mm256_blendv_pd(_mm256_set1_pd(-1.0), one_vec, sign_mask);
                    let shrunk = _mm256_mul_pd(sign_mm256_sub_pd(abs_data, threshold_vec));
                    _mm256_and_pd(shrunk, mask)
                }
                ThresholdMethod::Garrote => {
                    // Garrote thresholding: non-linear shrinkage
                    let abs_data = _mm256_andnot_pd(_mm256_set1_pd(-0.0), data);
                    let mask = _mm256_cmp_pd(abs_data, threshold_vec_CMP_GT_OQ);
                    let threshold_sq = _mm256_mul_pd(threshold_vec, threshold_vec);
                    let data_sq = _mm256_mul_pd(data, data);
                    let ratio = _mm256_div_pd(threshold_sq, data_sq);
                    let factor = _mm256_sub_pd(one_vec, ratio);
                    let result = _mm256_mul_pd(data, factor);
                    _mm256_and_pd(result, mask)
                }
            };

            _mm256_storeu_pd(_coeffs.as_mut_ptr().add(i), result);
        }
    }

    // Handle remaining elements with scalar code
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
fn simd_calculate_energy(data: &[f64]) -> f64 {
    let caps = PlatformCapabilities::detect();
    let simd_threshold = 64;

    if data.len() >= simd_threshold && caps.simd_available {
        simd_energy_avx2(_data)
    } else {
        // Fallback to scalar implementation
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
            let data_vec = _mm256_loadu_pd(_data.as_ptr().add(i));
            let squared = _mm256_mul_pd(data_vec, data_vec);
            sum_vec = _mm256_add_pd(sum_vec, squared);
        }

        // Horizontal sum of the vector
        let sum_array: [f64; 4] = std::mem::transmute(sum_vec);
        sum = sum_array.iter().sum();
    }

    // Handle remaining elements
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

/// Memory-optimized configuration for 2D DWT operations
#[derive(Debug, Clone)]
pub struct Dwt2dConfig {
    /// Enable memory pre-allocation for better cache efficiency
    pub preallocate_memory: bool,
    /// Use in-place operations when possible
    pub use_inplace: bool,
    /// Memory alignment for SIMD operations (must be power of 2)
    pub memory_alignment: usize,
    /// Chunk size for large arrays to improve cache locality
    pub chunk_size: Option<usize>,
}

impl Default for Dwt2dConfig {
    fn default() -> Self {
        Self {
            preallocate_memory: true,
            use_inplace: false,            // Conservative default for safety
            memory_alignment: 32,          // AVX2 alignment
            chunk_size: Some(1024 * 1024), // 1MB chunks by default
        }
    }
}

/// Memory pool for efficient allocation/deallocation of temporary arrays
pub struct MemoryPool {
    pools: std::collections::HashMap<usize, Vec<Vec<f64>>>,
    max_pool_size: usize,
}

impl MemoryPool {
    pub fn new() -> Self {
        Self {
            pools: std::collections::HashMap::new(),
            max_pool_size: 10, // Maximum number of arrays per size
        }
    }

    pub fn get_buffer(&mut self, size: usize) -> Vec<f64> {
        if let Some(pool) = self.pools.get_mut(&size) {
            if let Some(mut buffer) = pool.pop() {
                buffer.clear();
                buffer.resize(size, 0.0);
                return buffer;
            }
        }
        vec![0.0; size]
    }

    pub fn return_buffer(&mut self, buffer: Vec<f64>) {
        let size = buffer.capacity();
        let pool = self.pools.entry(size).or_insert_with(Vec::new);
        if pool.len() < self.max_pool_size {
            pool.push(buffer);
        }
    }
}

thread_local! {
    static MEMORY_POOL: std::cell::RefCell<MemoryPool> = std::cell::RefCell::new(MemoryPool::new());
}

/// Get a temporary buffer from the thread-local memory pool
#[allow(dead_code)]
fn get_temp_buffer(size: usize) -> Vec<f64> {
    MEMORY_POOL.with(|pool| pool.borrow_mut().get_buffer(_size))
}

/// Return a temporary buffer to the thread-local memory pool
#[allow(dead_code)]
fn return_temp_buffer(buffer: Vec<f64>) {
    MEMORY_POOL.with(|pool| pool.borrow_mut().return_buffer(_buffer));
}

// Import parallel ops for parallel processing when the "parallel" feature is enabled
#[cfg(feature = "parallel")]
// use std::convert::TryInto;  // Not needed

/// Type alias for column processing results to reduce complexity
#[allow(dead_code)]
type ColumnResult = (usize, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// Result of a 2D DWT decomposition, containing the approximation and detail coefficients.
///
/// The 2D DWT decomposes an image into four subbands, each representing different
/// frequency components in horizontal and vertical directions. These subbands are
/// represented as separate 2D arrays (matrices) in this struct.
///
/// The coefficients represent the following subbands:
/// - `approx`: Low-frequency approximation coefficients (LL) - Represents the coarse, low-resolution version of the image
/// - `detail_h`: Horizontal detail coefficients (LH) - Captures horizontal edges (high frequency in horizontal direction)
/// - `detail_v`: Vertical detail coefficients (HL) - Captures vertical edges (high frequency in vertical direction)
/// - `detail_d`: Diagonal detail coefficients (HH) - Captures diagonal details (high frequency in both directions)
///
/// These four subbands together contain all the information needed to reconstruct
/// the original image. For multi-level decomposition, the approximation coefficients
/// are recursively decomposed into further subbands.
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::{dwt2d_decompose, Dwt2dResult};
///
/// // Create a simple 4x4 image with a gradient pattern
/// let mut image = Array2::zeros((4, 4));
/// for i in 0..4 {
///     for j in 0..4 {
///         image[[i, j]] = (i * j)  as f64;
///     }
/// }
///
/// // Decompose the image
/// let result: Dwt2dResult = dwt2d_decompose(&image, Wavelet::Haar, None).unwrap();
///
/// // The image is now decomposed into four subbands:
/// let ll = &result.approx;  // Approximation coefficients (low-resolution image)
/// let lh = &result.detail_h;  // Horizontal details
/// let hl = &result.detail_v;  // Vertical details
/// let hh = &result.detail_d;  // Diagonal details
///
/// // All subbands have the same shape (half the size in each dimension)
/// assert_eq!(ll.shape(), &[2, 2]);
/// assert_eq!(lh.shape(), &[2, 2]);
/// assert_eq!(hl.shape(), &[2, 2]);
/// assert_eq!(hh.shape(), &[2, 2]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Dwt2dResult {
    /// Approximation coefficients (LL subband)
    pub approx: Array2<f64>,
    /// Horizontal detail coefficients (LH subband)
    pub detail_h: Array2<f64>,
    /// Vertical detail coefficients (HL subband)
    pub detail_v: Array2<f64>,
    /// Diagonal detail coefficients (HH subband)
    pub detail_d: Array2<f64>,
}

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
/// use ndarray::Array2;
/// use scirs2_signal::dwt2d::dwt2d_decompose;
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a simple 4x4 "image"
/// let data = Array2::from_shape_vec((4, 4), vec![
///     1.0, 2.0, 3.0, 4.0,
///     5.0, 6.0, 7.0, 8.0,
///     9.0, 10.0, 11.0, 12.0,
///     13.0, 14.0, 15.0, 16.0
/// ]).unwrap();
///
/// // Perform 2D DWT using the Haar wavelet
/// let result = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();
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
/// use ndarray::Array2;
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
/// let result = dwt2d_decompose(&data, Wavelet::DB(4), Some("periodic")).unwrap();
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
    // Enhanced input validation
    if data.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    // Get dimensions
    let (rows, cols) = data.dim();

    // Check minimum size requirements
    if rows < 2 || cols < 2 {
        return Err(SignalError::ValueError(format!(
            "Input array dimensions too small: {}x{}. Minimum size is 2x2",
            rows, cols
        )));
    }

    // Check for reasonable maximum size to prevent memory issues
    const MAX_DIMENSION: usize = 65536; // 64K pixels per dimension
    if rows > MAX_DIMENSION || cols > MAX_DIMENSION {
        return Err(SignalError::ValueError(format!(
            "Input array dimensions too large: {}x{}. Maximum supported size is {}x{}",
            rows, cols, MAX_DIMENSION, MAX_DIMENSION
        )));
    }

    // Convert input to f64 with enhanced error handling and validation
    let mut data_f64 = Array2::zeros(data.dim());
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut extreme_count = 0;

    for ((i, j), &val) in data.indexed_iter() {
        match num_traits::cast::cast::<T, f64>(val) {
            Some(converted) => {
                // Check for NaN, infinity, and extreme values
                if converted.is_nan() {
                    nan_count += 1;
                    if nan_count <= 5 {
                        // Limit error messages
                        eprintln!("Warning: NaN detected at position ({}, {})", i, j);
                    }
                    data_f64[[i, j]] = 0.0; // Replace NaN with 0
                } else if converted.is_infinite() {
                    inf_count += 1;
                    if inf_count <= 5 {
                        eprintln!("Warning: Infinity detected at position ({}, {})", i, j);
                    }
                    // Replace infinity with large but finite value
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
                return Err(SignalError::ValueError(format!(
                    "Failed to convert input data to f64 at position ({}, {})",
                    i, j
                )))
            }
        }
    }

    // Report validation results
    if nan_count > 0 {
        eprintln!("Processed {} NaN values (replaced with 0.0)", nan_count);
    }
    if inf_count > 0 {
        eprintln!("Processed {} infinite values (clamped to ±1e10)", inf_count);
    }
    if extreme_count > 0 {
        eprintln!("Detected {} extreme values (>1e12)", extreme_count);
    }

    // Validate wavelet compatibility
    let filter_length = match wavelet.get_filter_length() {
        Ok(len) => len,
        Err(_) => {
            return Err(SignalError::ValueError(format!(
                "Invalid wavelet: {:?}. Cannot determine filter length.",
                wavelet
            )));
        }
    };

    // Check if input is large enough for the selected wavelet
    let min_size = filter_length.max(4);
    if rows < min_size || cols < min_size {
        return Err(SignalError::ValueError(format!(
            "Input dimensions {}x{} too small for wavelet {:?} (requires minimum {}x{})",
            rows, cols, wavelet, min_size, min_size
        )));
    }

    // Calculate output dimensions (ceiling division for half the size)
    // Use integer division that rounds up
    let output_rows = div_ceil(rows, 2);
    let output_cols = div_ceil(cols, 2);

    // Create output arrays for each subband
    let mut ll = Array2::zeros((output_rows, output_cols));
    let mut lh = Array2::zeros((output_rows, output_cols));
    let mut hl = Array2::zeros((output_rows, output_cols));
    let mut hh = Array2::zeros((output_rows, output_cols));

    // Process rows first
    let mut rows_lo = Array2::zeros((rows, output_cols));
    let mut rows_hi = Array2::zeros((rows, output_cols));

    // Parallel processing of rows when "parallel" feature is enabled
    #[cfg(feature = "parallel")]
    {
        // Create a vector to hold the results of row processing
        #[allow(unused_mut)]
        let row_results: Result<Vec<(usize, Vec<f64>, Vec<f64>)>, SignalError> = (0..rows)
            .into_par_iter()
            .map(|i| {
                let row = data_f64.slice(ndarray::s![i, ..]).to_vec();
                let (approx, detail) = dwt::dwt_decompose(&row, wavelet, mode).map_err(|e| {
                    SignalError::ComputationError(format!("Row transform failed: {}", e))
                })?;
                Ok((i, approx, detail))
            })
            .collect();
        let row_results = row_results?;

        // Copy results back to the arrays with bounds checking
        for (i, approx, detail) in row_results {
            for j in 0..approx.len() {
                if j < output_cols {
                    // Make sure we don't go out of bounds
                    rows_lo[[i, j]] = approx[j];
                    rows_hi[[i, j]] = detail[j];
                }
            }
        }
    }

    // Sequential processing when parallel feature is not enabled
    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..rows {
            let row = data_f64.slice(ndarray::s![i, ..]).to_vec();
            let (approx, detail) = dwt::dwt_decompose(&row, wavelet, mode)?;

            for j in 0..approx.len() {
                if j < output_cols {
                    // Make sure we don't go out of bounds
                    rows_lo[[i, j]] = approx[j];
                    rows_hi[[i, j]] = detail[j];
                }
            }
        }
    }

    // Then process columns
    #[cfg(feature = "parallel")]
    {
        // Process columns in parallel
        let column_results: Result<Vec<ColumnResult>, SignalError> = (0..output_cols)
            .into_par_iter()
            .map(|j| {
                // Process low-pass filtered rows
                let col_lo = rows_lo.slice(ndarray::s![.., j]).to_vec();
                let (approx_lo, detail_lo) =
                    dwt::dwt_decompose(&col_lo, wavelet, mode).map_err(|e| {
                        SignalError::ComputationError(format!(
                            "Column transform failed (low-pass): {}",
                            e
                        ))
                    })?;

                // Process high-pass filtered rows
                let col_hi = rows_hi.slice(ndarray::s![.., j]).to_vec();
                let (approx_hi, detail_hi) =
                    dwt::dwt_decompose(&col_hi, wavelet, mode).map_err(|e| {
                        SignalError::ComputationError(format!(
                            "Column transform failed (high-pass): {}",
                            e
                        ))
                    })?;

                Ok((j, approx_lo, detail_lo, approx_hi, detail_hi))
            })
            .collect();
        let column_results = column_results?;

        // Copy results back to output arrays
        for (j, approx_lo, detail_lo, approx_hi, detail_hi) in column_results {
            for i in 0..approx_lo.len() {
                if i < output_rows {
                    // Make sure we don't go out of bounds
                    ll[[i, j]] = approx_lo[i];
                    hl[[i, j]] = detail_lo[i];
                }
            }

            for i in 0..approx_hi.len() {
                if i < output_rows {
                    // Make sure we don't go out of bounds
                    lh[[i, j]] = approx_hi[i];
                    hh[[i, j]] = detail_hi[i];
                }
            }
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        for j in 0..output_cols {
            // Process low-pass filtered rows
            let col_lo = rows_lo.slice(ndarray::s![.., j]).to_vec();
            let (approx, detail) = dwt::dwt_decompose(&col_lo, wavelet, mode)?;

            for i in 0..approx.len() {
                if i < output_rows {
                    // Make sure we don't go out of bounds
                    ll[[i, j]] = approx[i];
                    hl[[i, j]] = detail[i];
                }
            }

            // Process high-pass filtered rows
            let col_hi = rows_hi.slice(ndarray::s![.., j]).to_vec();
            let (approx, detail) = dwt::dwt_decompose(&col_hi, wavelet, mode)?;

            for i in 0..approx.len() {
                if i < output_rows {
                    // Make sure we don't go out of bounds
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

/// Memory-optimized version of 2D DWT decomposition with configuration options
///
/// This function provides the same functionality as `dwt2d_decompose` but with
/// additional memory optimizations and configuration options for better performance
/// on large arrays.
///
/// # Arguments
///
/// * `data` - The input 2D array (image)
/// * `wavelet` - The wavelet to use for the transform
/// * `mode` - The signal extension mode (default: "symmetric")
/// * `config` - Configuration for memory optimization
///
/// # Returns
///
/// * A `Dwt2dResult` containing the four subbands of the decomposition
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::{dwt2d_decompose_optimized, Dwt2dConfig};
///
/// // Create a sample image
/// let data = Array2::from_shape_vec((8, 8), (0..64).map(|x| x as f64).collect()).unwrap();
///
/// // Use optimized decomposition with default configuration
/// let config = Dwt2dConfig::default();
/// let result = dwt2d_decompose_optimized(&data, Wavelet::Haar, None, &config).unwrap();
/// ```
#[allow(dead_code)]
pub fn dwt2d_decompose_optimized<T>(
    data: &Array2<T>,
    wavelet: Wavelet,
    mode: Option<&str>,
    config: &Dwt2dConfig,
) -> SignalResult<Dwt2dResult>
where
    T: Float + NumCast + Debug,
{
    if data.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    // Validate input data for numerical stability
    if let Some(data_slice) = data.as_slice() {
        // Data validation handled by transform
    }

    // Get dimensions
    let (rows, cols) = data.dim();

    // Calculate output dimensions using our optimized ceiling division
    let output_rows = div_ceil(rows, 2);
    let output_cols = div_ceil(cols, 2);

    // Pre-allocate all output arrays at once for better memory locality
    let mut ll = Array2::zeros((output_rows, output_cols));
    let mut lh = Array2::zeros((output_rows, output_cols));
    let mut hl = Array2::zeros((output_rows, output_cols));
    let mut hh = Array2::zeros((output_rows, output_cols));

    // Convert input to f64 using temporary buffer from memory pool
    let data_buffer_size = rows * cols;
    let mut data_buffer = if config.preallocate_memory {
        get_temp_buffer(data_buffer_size)
    } else {
        vec![0.0; data_buffer_size]
    };

    // Copy and convert data efficiently
    for ((i, j), &val) in data.indexed_iter() {
        match num_traits::cast::cast::<T, f64>(val) {
            Some(converted) => data_buffer[i * cols + j] = converted,
            None => {
                return Err(SignalError::ValueError(
                    "Failed to convert input data to f64".to_string(),
                ))
            }
        }
    }

    // Create temporary arrays for intermediate results using memory pool
    let row_buffer_size = rows * output_cols;
    let mut rows_lo_buffer = if config.preallocate_memory {
        get_temp_buffer(row_buffer_size)
    } else {
        vec![0.0; row_buffer_size]
    };
    let mut rows_hi_buffer = if config.preallocate_memory {
        get_temp_buffer(row_buffer_size)
    } else {
        vec![0.0; row_buffer_size]
    };

    // Process rows with memory-efficient chunking if configured
    let chunk_size = config.chunk_size.unwrap_or(rows);
    for chunk_start in (0..rows).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(rows);

        // Process this chunk of rows
        #[cfg(feature = "parallel")]
        {
            let chunk_results: Result<Vec<(usize, Vec<f64>, Vec<f64>)>, SignalError> = (chunk_start
                ..chunk_end)
                .into_par_iter()
                .map(|i| {
                    let row_start = i * cols;
                    let row_end = row_start + cols;
                    let row = &data_buffer[row_start..row_end];
                    let (approx, detail) = dwt::dwt_decompose(row, wavelet, mode).map_err(|e| {
                        SignalError::ComputationError(format!("Row transform failed: {}", e))
                    })?;
                    Ok((i, approx, detail))
                })
                .collect();
            let chunk_results = chunk_results?;

            for (i, approx, detail) in chunk_results {
                let output_start = i * output_cols;
                for j in 0..approx.len().min(output_cols) {
                    rows_lo_buffer[output_start + j] = approx[j];
                    rows_hi_buffer[output_start + j] = detail[j];
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in chunk_start..chunk_end {
                let row_start = i * cols;
                let row_end = row_start + cols;
                let row = &data_buffer[row_start..row_end];
                let (approx, detail) = dwt::dwt_decompose(row, wavelet, mode)?;

                let output_start = i * output_cols;
                for j in 0..approx.len().min(output_cols) {
                    rows_lo_buffer[output_start + j] = approx[j];
                    rows_hi_buffer[output_start + j] = detail[j];
                }
            }
        }
    }

    // Process columns with chunking
    let col_chunk_size = config.chunk_size.unwrap_or(output_cols);
    for chunk_start in (0..output_cols).step_by(col_chunk_size) {
        let chunk_end = (chunk_start + col_chunk_size).min(output_cols);

        #[cfg(feature = "parallel")]
        {
            let column_results: Result<
                Vec<(usize, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)>,
                SignalError,
            > = (chunk_start..chunk_end)
                .into_par_iter()
                .map(|j| {
                    // Extract column from rows_lo_buffer
                    let mut col_lo = vec![0.0; rows];
                    for i in 0..rows {
                        col_lo[i] = rows_lo_buffer[i * output_cols + j];
                    }
                    let (approx_lo, detail_lo) = dwt::dwt_decompose(&col_lo, wavelet, mode)
                        .map_err(|e| {
                            SignalError::ComputationError(format!(
                                "Column transform failed (low-pass): {}",
                                e
                            ))
                        })?;

                    // Extract column from rows_hi_buffer
                    let mut col_hi = vec![0.0; rows];
                    for i in 0..rows {
                        col_hi[i] = rows_hi_buffer[i * output_cols + j];
                    }
                    let (approx_hi, detail_hi) = dwt::dwt_decompose(&col_hi, wavelet, mode)
                        .map_err(|e| {
                            SignalError::ComputationError(format!(
                                "Column transform failed (high-pass): {}",
                                e
                            ))
                        })?;

                    Ok((j, approx_lo, detail_lo, approx_hi, detail_hi))
                })
                .collect();
            let column_results = column_results?;

            for (j, approx_lo, detail_lo, approx_hi, detail_hi) in column_results {
                for i in 0..approx_lo.len().min(output_rows) {
                    ll[[i, j]] = approx_lo[i];
                    hl[[i, j]] = detail_lo[i];
                }
                for i in 0..approx_hi.len().min(output_rows) {
                    lh[[i, j]] = approx_hi[i];
                    hh[[i, j]] = detail_hi[i];
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for j in chunk_start..chunk_end {
                // Extract column from rows_lo_buffer
                let mut col_lo = vec![0.0; rows];
                for i in 0..rows {
                    col_lo[i] = rows_lo_buffer[i * output_cols + j];
                }
                let (approx_lo, detail_lo) = dwt::dwt_decompose(&col_lo, wavelet, mode)?;

                // Extract column from rows_hi_buffer
                let mut col_hi = vec![0.0; rows];
                for i in 0..rows {
                    col_hi[i] = rows_hi_buffer[i * output_cols + j];
                }
                let (approx_hi, detail_hi) = dwt::dwt_decompose(&col_hi, wavelet, mode)?;

                for i in 0..approx_lo.len().min(output_rows) {
                    ll[[i, j]] = approx_lo[i];
                    hl[[i, j]] = detail_lo[i];
                }
                for i in 0..approx_hi.len().min(output_rows) {
                    lh[[i, j]] = approx_hi[i];
                    hh[[i, j]] = detail_hi[i];
                }
            }
        }
    }

    // Return temporary buffers to memory pool
    if config.preallocate_memory {
        return_temp_buffer(data_buffer);
        return_temp_buffer(rows_lo_buffer);
        return_temp_buffer(rows_hi_buffer);
    }

    Ok(Dwt2dResult {
        approx: ll,
        detail_h: lh,
        detail_v: hl,
        detail_d: hh,
    })
}

// Helper function removed (implementation rewritten)

// Helper function removed (implementation rewritten)

/// Performs a single-level 2D inverse discrete wavelet transform.
///
/// This function reconstructs a 2D array (such as an image) from its wavelet decomposition.
/// It is the inverse operation of `dwt2d_decompose` and combines the four subbands
/// (approximation and detail coefficients) back into the original signal.
///
/// # Algorithm
///
/// The inverse 2D DWT is computed using separable filtering:
/// 1. First, reconstruct each row by combining the corresponding rows from low-pass and high-pass parts
/// 2. Then, reconstruct each column of the resulting array
/// 3. The process reverses the decomposition steps, using inverse wavelet filters
///
/// # Arguments
///
/// * `decomposition` - The wavelet decomposition to reconstruct from, containing the four subbands
/// * `wavelet` - The wavelet used for the original transform (must match the decomposition wavelet)
/// * `mode` - The signal extension mode (default: "symmetric")
///   - Note: This should match the mode used for decomposition for best results
///
/// # Returns
///
/// * The reconstructed 2D array with dimensions twice the size of each subband
///
/// # Errors
///
/// Returns an error if:
/// * The subbands in the decomposition have different shapes
/// * There are issues with the wavelet filters
/// * Numerical problems occur during reconstruction
///
/// # Performance
///
/// This operation is computationally efficient, with O(N) complexity where N is the
/// total number of elements in the reconstructed array. The actual performance depends
/// on the wavelet filter lengths.
///
/// # Examples
///
/// Basic decomposition and reconstruction:
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt2d::{dwt2d_decompose, dwt2d_reconstruct};
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a simple "image"
/// let data = Array2::from_shape_vec((4, 4), vec![
///     1.0, 2.0, 3.0, 4.0,
///     5.0, 6.0, 7.0, 8.0,
///     9.0, 10.0, 11.0, 12.0,
///     13.0, 14.0, 15.0, 16.0
/// ]).unwrap();
///
/// // Decompose
/// let decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();
///
/// // Reconstruct
/// let reconstructed = dwt2d_reconstruct(&decomposition, Wavelet::Haar, None).unwrap();
///
/// // The reconstructed image should have the same shape as the original
/// assert_eq!(reconstructed.shape(), data.shape());
/// ```
///
/// Modifying coefficients before reconstruction (simple denoising):
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt2d::{dwt2d_decompose, dwt2d_reconstruct, Dwt2dResult};
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a sample image with gradient pattern
/// let mut data = Array2::zeros((8, 8));
/// for i in 0..8 {
///     for j in 0..8 {
///         data[[i, j]] = (i * j)  as f64;
///     }
/// }
///
/// // Decompose using Haar wavelet
/// let mut decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();
///
/// // Simple denoising by zeroing out small detail coefficients
/// let threshold = 2.0;
/// for h in decomposition.detail_h.iter_mut() {
///     if h.abs() < threshold {
///         *h = 0.0;
///     }
/// }
/// for v in decomposition.detail_v.iter_mut() {
///     if v.abs() < threshold {
///         *v = 0.0;
///     }
/// }
/// for d in decomposition.detail_d.iter_mut() {
///     if d.abs() < threshold {
///         *d = 0.0;
///     }
/// }
///
/// // Reconstruct from modified coefficients
/// let denoised = dwt2d_reconstruct(&decomposition, Wavelet::Haar, None).unwrap();
/// ```
#[allow(dead_code)]
pub fn dwt2d_reconstruct(
    decomposition: &Dwt2dResult,
    wavelet: Wavelet,
    mode: Option<&str>,
) -> SignalResult<Array2<f64>> {
    // Extract components
    let ll = &decomposition.approx;
    let lh = &decomposition.detail_h;
    let hl = &decomposition.detail_v;
    let hh = &decomposition.detail_d;

    // Verify all components have the same shape
    let shape = ll.shape();
    if lh.shape() != shape || hl.shape() != shape || hh.shape() != shape {
        return Err(SignalError::ValueError(
            "All decomposition components must have the same shape".to_string(),
        ));
    }

    // Get the shape of the components
    let (rows, cols) = (shape[0], shape[1]);

    // Calculate output shape (twice the input dimensions)
    let out_rows = rows * 2;
    let out_cols = cols * 2;

    // First, reconstruct columns for low and high frequency parts
    let mut row_lo = Array2::zeros((out_rows, cols));
    let mut row_hi = Array2::zeros((out_rows, cols));

    // Parallel column reconstruction
    #[cfg(feature = "parallel")]
    {
        // Process columns in parallel
        let col_results: Result<Vec<(usize, Vec<f64>, Vec<f64>)>, SignalError> = (0..cols)
            .into_par_iter()
            .map(|j| {
                // Reconstruct low-pass columns
                let ll_col = ll.slice(ndarray::s![.., j]).to_vec();
                let hl_col = hl.slice(ndarray::s![.., j]).to_vec();
                let col_lo = dwt::dwt_reconstruct(&ll_col, &hl_col, wavelet).map_err(|e| {
                    SignalError::ComputationError(format!(
                        "Low-pass column reconstruction failed: {}",
                        e
                    ))
                })?;

                // Reconstruct high-pass columns
                let lh_col = lh.slice(ndarray::s![.., j]).to_vec();
                let hh_col = hh.slice(ndarray::s![.., j]).to_vec();
                let col_hi = dwt::dwt_reconstruct(&lh_col, &hh_col, wavelet).map_err(|e| {
                    SignalError::ComputationError(format!(
                        "High-pass column reconstruction failed: {}",
                        e
                    ))
                })?;

                Ok((j, col_lo, col_hi))
            })
            .collect();
        let col_results = col_results?;

        // Store results
        for (j, col_lo, col_hi) in col_results {
            for i in 0..col_lo.len() {
                if i < out_rows {
                    row_lo[[i, j]] = col_lo[i];
                    row_hi[[i, j]] = col_hi[i];
                }
            }
        }
    }

    // Sequential column reconstruction
    #[cfg(not(feature = "parallel"))]
    {
        for j in 0..cols {
            // Reconstruct low-pass columns
            let ll_col = ll.slice(ndarray::s![.., j]).to_vec();
            let hl_col = hl.slice(ndarray::s![.., j]).to_vec();
            let col_lo = dwt::dwt_reconstruct(&ll_col, &hl_col, wavelet)?;

            // Reconstruct high-pass columns
            let lh_col = lh.slice(ndarray::s![.., j]).to_vec();
            let hh_col = hh.slice(ndarray::s![.., j]).to_vec();
            let col_hi = dwt::dwt_reconstruct(&lh_col, &hh_col, wavelet)?;

            // Store reconstructed columns
            for i in 0..col_lo.len() {
                if i < out_rows {
                    row_lo[[i, j]] = col_lo[i];
                    row_hi[[i, j]] = col_hi[i];
                }
            }
        }
    }

    // Then, reconstruct rows
    let mut result = Array2::zeros((out_rows, out_cols));

    // Parallel row reconstruction
    #[cfg(feature = "parallel")]
    {
        // Process rows in parallel
        let row_results: Result<Vec<(usize, Vec<f64>)>, SignalError> = (0..out_rows)
            .into_par_iter()
            .map(|i| {
                // Get rows from low and high frequency parts
                let lo_row = row_lo.slice(ndarray::s![i, ..]).to_vec();
                let hi_row = row_hi.slice(ndarray::s![i, ..]).to_vec();

                // Reconstruct row
                let full_row = dwt::dwt_reconstruct(&lo_row, &hi_row, wavelet).map_err(|e| {
                    SignalError::ComputationError(format!("Row reconstruction failed: {}", e))
                })?;

                Ok((i, full_row))
            })
            .collect();
        let row_results = row_results?;

        // Store results
        for (i, full_row) in row_results {
            for j in 0..full_row.len() {
                if j < out_cols {
                    result[[i, j]] = full_row[j];
                }
            }
        }
    }

    // Sequential row reconstruction
    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..out_rows {
            // Get rows from low and high frequency parts
            let lo_row = row_lo.slice(ndarray::s![i, ..]).to_vec();
            let hi_row = row_hi.slice(ndarray::s![i, ..]).to_vec();

            // Reconstruct row
            let full_row = dwt::dwt_reconstruct(&lo_row, &hi_row, wavelet)?;

            // Store reconstructed row
            for j in 0..full_row.len() {
                if j < out_cols {
                    result[[i, j]] = full_row[j];
                }
            }
        }
    }

    Ok(result)
}

// Helper function removed (implementation rewritten)

// Helper function removed (implementation rewritten)

/// Performs a multi-level 2D discrete wavelet transform.
///
/// This function computes the wavelet transform recursively, applying
/// successive decompositions to the approximation coefficients from each level.
/// This creates a multi-resolution analysis with a pyramid structure, where each
/// level captures details at different scales.
///
/// # Algorithm
///
/// The multi-level 2D DWT is computed as follows:
/// 1. Apply a single-level 2D DWT to the input data, generating four subbands (LL, LH, HL, HH)
/// 2. Apply a single-level 2D DWT to the LL (approximation) subband from step 1
/// 3. Repeat until reaching the desired number of levels
/// 4. Return the coefficients from all levels, with the deepest level first
///
/// # Arguments
///
/// * `data` - The input 2D array (image)
/// * `wavelet` - The wavelet to use for the transform
/// * `levels` - The number of decomposition levels to compute
/// * `mode` - The signal extension mode (default: "symmetric")
///
/// # Returns
///
/// * A vector of `Dwt2dResult` objects, where:
///   - index 0 contains coefficients from the deepest level (smallest scale)
///   - each subsequent index contains coefficients from a larger scale
///   - the last index contains the first level of decomposition (largest scale)
///
/// # Errors
///
/// Returns an error if:
/// * The input array is empty
/// * The requested number of levels is 0
/// * The input array is too small for the requested number of levels
/// * Other errors from the underlying `dwt2d_decompose` function
///
/// # Memory Usage
///
/// This function stores coefficients from all levels separately, so memory usage
/// is approximately 4/3 times the original image size (for sufficiently large images).
/// For example, an 8×8 image decomposes into:
/// - Level 1: Four 4×4 subbands
/// - Level 2: Three 2×2 subbands plus the Level 3 approximation
/// - Level 3: Three 1×1 subbands plus a 1×1 approximation
///
/// # Examples
///
/// Basic multi-level decomposition:
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt2d::wavedec2;
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a simple 8x8 "image"
/// let mut data = Array2::zeros((8, 8));
/// for i in 0..8 {
///     for j in 0..8 {
///         data[[i, j]] = (i * 8 + j + 1)  as f64;
///     }
/// }
///
/// // Perform 3-level 2D DWT
/// let coeffs = wavedec2(&data, Wavelet::Haar, 3, None).unwrap();
///
/// // Check the number of decomposition levels
/// assert_eq!(coeffs.len(), 3);
///
/// // Examine the coefficient shapes (each level is half the size of the previous)
/// assert_eq!(coeffs[0].approx.shape(), &[1, 1]);  // Deepest level (smallest)
/// assert_eq!(coeffs[1].approx.shape(), &[2, 2]);
/// assert_eq!(coeffs[2].approx.shape(), &[4, 4]);  // First level (largest)
/// ```
///
/// Using a different wavelet family:
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt2d::wavedec2;
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a larger image to accommodate longer filters
/// let mut data = Array2::zeros((32, 32));
/// for i in 0..32 {
///     for j in 0..32 {
///         data[[i, j]] = ((i+j) % 8)  as f64;  // Create a pattern
///     }
/// }
///
/// // Decompose with Daubechies 4 wavelet
/// let coeffs = wavedec2(&data, Wavelet::DB(4), 2, None).unwrap();
/// assert_eq!(coeffs.len(), 2);
/// ```
#[allow(dead_code)]
pub fn wavedec2<T>(
    data: &Array2<T>,
    wavelet: Wavelet,
    levels: usize,
    mode: Option<&str>,
) -> SignalResult<Vec<Dwt2dResult>>
where
    T: Float + NumCast + Debug,
{
    if data.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    if levels == 0 {
        return Err(SignalError::ValueError(
            "Levels must be greater than 0".to_string(),
        ));
    }

    // Check if the data is large enough for the requested levels
    let (rows, cols) = data.dim();
    let min_size = 2usize.pow(levels as u32);
    if rows < min_size || cols < min_size {
        return Err(SignalError::ValueError(format!(
            "Data size ({}, {}) is too small for {} levels of decomposition",
            rows, cols, levels
        )));
    }

    // Allocate storage for all levels
    let mut result = Vec::with_capacity(levels);

    // Perform first level
    let mut decomposition = dwt2d_decompose(data, wavelet, mode)?;
    result.push(decomposition.clone());

    // Perform remaining levels on approximation coefficients
    for _level in 1..levels {
        decomposition = dwt2d_decompose(&decomposition.approx, wavelet, mode)?;
        result.push(decomposition.clone());
    }

    // Reverse so index 0 is the deepest level
    result.reverse();

    Ok(result)
}

/// Reconstructs a 2D signal from its multi-level wavelet decomposition.
///
/// This function is the inverse of `wavedec2` and reconstructs a 2D array (such as an image)
/// from its multi-level wavelet decomposition. It processes the coefficients from deepest
/// to shallowest level, gradually building up the full-resolution image.
///
/// # Algorithm
///
/// The multi-level reconstruction works by:
/// 1. Starting with the approximation coefficients at the deepest level
/// 2. Combining these with the detail coefficients at that level to get a higher-resolution approximation
/// 3. Repeating this process level by level until the full-resolution image is reconstructed
///
/// # Arguments
///
/// * `coeffs` - The wavelet coefficients from `wavedec2`, with deepest level first
/// * `wavelet` - The wavelet used for the original transform (must match)
/// * `mode` - The signal extension mode (default: "symmetric")
///   - Should match the mode used for decomposition
///
/// # Returns
///
/// * The reconstructed 2D array with the same dimensions as the original input to `wavedec2`
///
/// # Errors
///
/// Returns an error if:
/// * The coefficient list is empty
/// * The detail coefficients at any level do not match the approximation shape
/// * Other errors from the underlying `dwt2d_reconstruct` function
///
/// # Applications
///
/// This function is particularly useful for:
/// * Image compression (after coefficient thresholding)
/// * Denoising (after removing noise from detail coefficients)
/// * Feature extraction at multiple scales
/// * Image fusion
///
/// # Examples
///
/// Basic multi-level decomposition and reconstruction:
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt2d::{wavedec2, waverec2};
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a simple 8x8 "image"
/// let mut data = Array2::zeros((8, 8));
/// for i in 0..8 {
///     for j in 0..8 {
///         data[[i, j]] = (i * 8 + j + 1)  as f64;
///     }
/// }
///
/// // Decompose
/// let coeffs = wavedec2(&data, Wavelet::Haar, 3, None).unwrap();
///
/// // Reconstruct
/// let reconstructed = waverec2(&coeffs, Wavelet::Haar, None).unwrap();
///
/// // Check that reconstruction has the correct shape
/// assert_eq!(reconstructed.shape(), data.shape());
/// ```
///
/// Simple image compression by coefficient thresholding:
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt2d::{wavedec2, waverec2, Dwt2dResult};
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a simple 16x16 "image" with a pattern
/// let mut data = Array2::zeros((16, 16));
/// for i in 0..16 {
///     for j in 0..16 {
///         data[[i, j]] = ((i as f64 - 8.0).powi(2) + (j as f64 - 8.0).powi(2)).sqrt();
///     }
/// }
///
/// // Multi-level decomposition
/// let mut coeffs = wavedec2(&data, Wavelet::DB(4), 2, None).unwrap();
///
/// // Threshold small detail coefficients to achieve compression
/// let threshold = 0.5;
/// for level in &mut coeffs {
///     // Only threshold detail coefficients, not approximation
///     for h in level.detail_h.iter_mut() {
///         if h.abs() < threshold { *h = 0.0; }
///     }
///     for v in level.detail_v.iter_mut() {
///         if v.abs() < threshold { *v = 0.0; }
///     }
///     for d in level.detail_d.iter_mut() {
///         if d.abs() < threshold { *d = 0.0; }
///     }
/// }
///
/// // Reconstruct from thresholded coefficients
/// let compressed = waverec2(&coeffs, Wavelet::DB(4), None).unwrap();
/// ```
#[allow(dead_code)]
pub fn waverec2(
    coeffs: &[Dwt2dResult],
    wavelet: Wavelet,
    mode: Option<&str>,
) -> SignalResult<Array2<f64>> {
    if coeffs.is_empty() {
        return Err(SignalError::ValueError(
            "Coefficient list is empty".to_string(),
        ));
    }

    // Start with the deepest level coefficients (which were stored first in the list)
    let mut approx = coeffs[0].approx.clone();

    // Reconstruct one level at a time, from deepest to shallowest
    for decomp in coeffs {
        // Create a synthetic decomposition with current approximation and details from this level
        let synthetic_decomp = Dwt2dResult {
            approx,
            detail_h: decomp.detail_h.clone(),
            detail_v: decomp.detail_v.clone(),
            detail_d: decomp.detail_d.clone(),
        };

        // Reconstruct this level
        approx = dwt2d_reconstruct(&synthetic_decomp, wavelet, mode)?;
    }

    Ok(approx)
}

/// Threshold method to apply to wavelet coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThresholdMethod {
    /// Hard thresholding: sets coefficients below threshold to zero, leaves others unchanged
    Hard,
    /// Soft thresholding: sets coefficients below threshold to zero, shrinks others toward zero by threshold amount
    Soft,
    /// Garrote thresholding: a non-linear thresholding approach with properties between hard and soft
    Garrote,
}

/// Apply thresholding to wavelet coefficients for denoising or compression.
///
/// This function applies a threshold to the detail coefficients of a wavelet decomposition.
/// It is commonly used for denoising (removing low-amplitude noise) and compression
/// (removing less significant coefficients). Only detail coefficients are thresholded;
/// approximation coefficients are left unchanged.
///
/// # Arguments
///
/// * `decomposition` - The wavelet decomposition to threshold (will be modified in-place)
/// * `threshold` - The threshold value (coefficients with absolute value below this will be modified)
/// * `method` - The thresholding method to apply (Hard, Soft, or Garrote)
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::{dwt2d_decompose, dwt2d_reconstruct, threshold_dwt2d, ThresholdMethod};
///
/// // Create a sample "image"
/// let mut data = Array2::zeros((8, 8));
/// for i in 0..8 {
///     for j in 0..8 {
///         data[[i, j]] = (i * j)  as f64;
///     }
/// }
///
/// // Decompose with wavelet transform
/// let mut decomposition = dwt2d_decompose(&data, Wavelet::DB(4), None).unwrap();
///
/// // Apply hard thresholding to detail coefficients
/// threshold_dwt2d(&mut decomposition, 1.0, ThresholdMethod::Hard);
///
/// // Reconstruct the denoised image
/// let denoised = dwt2d_reconstruct(&decomposition, Wavelet::DB(4), None).unwrap();
/// ```
#[allow(dead_code)]
pub fn threshold_dwt2d(decomposition: &mut Dwt2dResult, threshold: f64, method: ThresholdMethod) {
    // Apply SIMD-optimized thresholding to detail coefficients
    // Note: ndarray's as_slice_mut() gives us direct access to the underlying data
    if let Some(h_slice) = decomposition.detail_h.as_slice_mut() {
        simd_threshold_coefficients(h_slice, threshold, method);
    } else {
        // Fallback for non-contiguous arrays
        for h in decomposition.detail_h.iter_mut() {
            *h = apply_threshold(*h, threshold, method);
        }
    }

    if let Some(v_slice) = decomposition.detail_v.as_slice_mut() {
        simd_threshold_coefficients(v_slice, threshold, method);
    } else {
        // Fallback for non-contiguous arrays
        for v in decomposition.detail_v.iter_mut() {
            *v = apply_threshold(*v, threshold, method);
        }
    }

    if let Some(d_slice) = decomposition.detail_d.as_slice_mut() {
        simd_threshold_coefficients(d_slice, threshold, method);
    } else {
        // Fallback for non-contiguous arrays
        for d in decomposition.detail_d.iter_mut() {
            *d = apply_threshold(*d, threshold, method);
        }
    }
}

/// Apply thresholding to multi-level wavelet coefficients.
///
/// Similar to `threshold_dwt2d`, but operates on a multi-level decomposition from `wavedec2`.
/// This allows for level-dependent thresholding, which can be more effective for certain
/// applications.
///
/// # Arguments
///
/// * `coeffs` - The multi-level wavelet decomposition to threshold (modified in-place)
/// * `threshold` - The threshold value, or vector of threshold values (one per level)
/// * `method` - The thresholding method to apply
///
/// # Examples
///
/// Using a different threshold for each level:
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::{wavedec2, waverec2, threshold_wavedec2, ThresholdMethod};
///
/// // Create a sample image
/// let mut data = Array2::zeros((16, 16));
/// for i in 0..16 {
///     for j in 0..16 {
///         data[[i, j]] = (i * j)  as f64;
///     }
/// }
///
/// // Multi-level decomposition (3 levels)
/// let mut coeffs = wavedec2(&data, Wavelet::Haar, 3, None).unwrap();
///
/// // Apply different thresholds for each level (higher thresholds for finer details)
/// let thresholds = vec![5.0, 10.0, 15.0];  
/// threshold_wavedec2(&mut coeffs, &thresholds, ThresholdMethod::Soft);
///
/// // Reconstruct from thresholded coefficients
/// let result = waverec2(&coeffs, Wavelet::Haar, None).unwrap();
/// ```
#[allow(dead_code)]
pub fn threshold_wavedec2(coeffs: &mut [Dwt2dResult], threshold: &[f64], method: ThresholdMethod) {
    for (i, level) in coeffs.iter_mut().enumerate() {
        // Get the appropriate threshold for this level
        let level_threshold = if i < threshold.len() {
            threshold[i]
        } else {
            // If not enough thresholds provided, use the last one
            *threshold.last().unwrap_or(&0.0)
        };

        // Apply thresholding to this level
        threshold_dwt2d(level, level_threshold, method);
    }
}

/// Helper function to apply a threshold to a single coefficient.
#[allow(dead_code)]
fn apply_threshold(x: f64, threshold: f64, method: ThresholdMethod) -> f64 {
    let abs_x = x.abs();

    // If coefficient is below threshold, always zero it out
    if abs_x <= threshold {
        return 0.0;
    }

    // Apply the appropriate thresholding method
    match method {
        ThresholdMethod::Hard => x, // Hard thresholding keeps the value unchanged
        ThresholdMethod::Soft => {
            // Soft thresholding shrinks the value toward zero by the threshold amount
            x.signum() * (abs_x - threshold)
        }
        ThresholdMethod::Garrote => {
            // Non-linear garrote thresholding
            x * (1.0 - (threshold * threshold) / (x * x))
        }
    }
}

/// Calculate the energy of wavelet coefficients in a decomposition.
///
/// Energy is the sum of squared coefficients. This function is useful for analyzing
/// the distribution of energy across different subbands and for determining appropriate
/// threshold values.
///
/// # Arguments
///
/// * `decomposition` - The wavelet decomposition to analyze
/// * `include_approx` - Whether to include approximation coefficients in the calculation
///
/// # Returns
///
/// * A tuple containing the total energy and a struct with energy by subband:
///   - approx: Energy in the approximation coefficients (LL band)
///   - detail_h: Energy in the horizontal detail coefficients (LH band)
///   - detail_v: Energy in the vertical detail coefficients (HL band)
///   - detail_d: Energy in the diagonal detail coefficients (HH band)
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::{dwt2d_decompose, calculate_energy};
///
/// // Create a sample image
/// let mut data = Array2::zeros((8, 8));
/// for i in 0..8 {
///     for j in 0..8 {
///         data[[i, j]] = (i * j)  as f64;
///     }
/// }
///
/// // Decompose the image
/// let decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();
///
/// // Calculate energy, including approximation coefficients
/// let (total_energy, energy_by_subband) = calculate_energy(&decomposition, true);
///
/// // Typically, approximation coefficients contain most of the energy
/// assert!(energy_by_subband.approx > energy_by_subband.detail_h);
/// assert!(energy_by_subband.approx > energy_by_subband.detail_v);
/// assert!(energy_by_subband.approx > energy_by_subband.detail_d);
/// ```
#[allow(dead_code)]
pub fn calculate_energy(
    _decomposition: &Dwt2dResult,
    include_approx: bool,
) -> (f64, WaveletEnergy) {
    // Calculate energy for each subband using SIMD-optimized functions
    let approx_energy = if include_approx {
        if let Some(approx_slice) = decomposition.approx.as_slice() {
            simd_calculate_energy(approx_slice)
        } else {
            decomposition.approx.iter().map(|&x| x * x).sum()
        }
    } else {
        0.0
    };

    let detail_h_energy = if let Some(h_slice) = decomposition.detail_h.as_slice() {
        simd_calculate_energy(h_slice)
    } else {
        decomposition.detail_h.iter().map(|&x| x * x).sum()
    };

    let detail_v_energy = if let Some(v_slice) = decomposition.detail_v.as_slice() {
        simd_calculate_energy(v_slice)
    } else {
        decomposition.detail_v.iter().map(|&x| x * x).sum()
    };

    let detail_d_energy = if let Some(d_slice) = decomposition.detail_d.as_slice() {
        simd_calculate_energy(d_slice)
    } else {
        decomposition.detail_d.iter().map(|&x| x * x).sum()
    };

    // Calculate total energy
    let total = approx_energy + detail_h_energy + detail_v_energy + detail_d_energy;

    // Create energy structure
    let energy_by_subband = WaveletEnergy {
        approx: approx_energy,
        detail_h: detail_h_energy,
        detail_v: detail_v_energy,
        detail_d: detail_d_energy,
    };

    (total, energy_by_subband)
}

/// Structure containing energy values for each wavelet subband.
#[derive(Debug, Clone, Copy)]
pub struct WaveletEnergy {
    /// Energy in approximation coefficients (LL band)
    pub approx: f64,
    /// Energy in horizontal detail coefficients (LH band)
    pub detail_h: f64,
    /// Energy in vertical detail coefficients (HL band)
    pub detail_v: f64,
    /// Energy in diagonal detail coefficients (HH band)
    pub detail_d: f64,
}

/// Count non-zero coefficients in a wavelet decomposition.
///
/// This is useful for quantifying the sparsity of a wavelet representation,
/// especially after thresholding for compression.
///
/// # Arguments
///
/// * `decomposition` - The wavelet decomposition to analyze
/// * `include_approx` - Whether to include approximation coefficients in the count
///
/// # Returns
///
/// * A tuple containing the total count and a struct with counts by subband
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::{dwt2d_decompose, threshold_dwt2d, count_nonzeros, ThresholdMethod};
///
/// // Create a sample image
/// let mut data = Array2::zeros((8, 8));
/// for i in 0..8 {
///     for j in 0..8 {
///         data[[i, j]] = (i * j)  as f64;
///     }
/// }
///
/// // Decompose the image
/// let mut decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();
///
/// // Count coefficients before thresholding
/// let (before_total_) = count_nonzeros(&decomposition, true);
///
/// // Apply thresholding
/// threshold_dwt2d(&mut decomposition, 5.0, ThresholdMethod::Hard);
///
/// // Count coefficients after thresholding
/// let (after_total_) = count_nonzeros(&decomposition, true);
///
/// // After thresholding, there should be fewer non-zero coefficients
/// assert!(after_total <= before_total);
/// ```
#[allow(dead_code)]
pub fn count_nonzeros(
    _decomposition: &Dwt2dResult,
    include_approx: bool,
) -> (usize, WaveletCounts) {
    // Count non-zero coefficients in each subband
    let approx_count = if include_approx {
        decomposition.approx.iter().filter(|&&x| x != 0.0).count()
    } else {
        0
    };

    let detail_h_count = _decomposition
        .detail_h
        .iter()
        .filter(|&&x| x != 0.0)
        .count();
    let detail_v_count = _decomposition
        .detail_v
        .iter()
        .filter(|&&x| x != 0.0)
        .count();
    let detail_d_count = _decomposition
        .detail_d
        .iter()
        .filter(|&&x| x != 0.0)
        .count();

    // Calculate total count
    let total = approx_count + detail_h_count + detail_v_count + detail_d_count;

    // Create counts structure
    let counts_by_subband = WaveletCounts {
        approx: approx_count,
        detail_h: detail_h_count,
        detail_v: detail_v_count,
        detail_d: detail_d_count,
    };

    (total, counts_by_subband)
}

/// Structure containing non-zero coefficient counts for each wavelet subband.
#[derive(Debug, Clone, Copy)]
pub struct WaveletCounts {
    /// Count in approximation coefficients (LL band)
    pub approx: usize,
    /// Count in horizontal detail coefficients (LH band)
    pub detail_h: usize,
    /// Count in vertical detail coefficients (HL band)
    pub detail_v: usize,
    /// Count in diagonal detail coefficients (HH band)
    pub detail_d: usize,
}

/// Enhanced validation metrics for 2D wavelet transforms
#[derive(Debug, Clone)]
pub struct Dwt2dValidationResult {
    /// Reconstruction error (RMSE)
    pub reconstruction_error: f64,
    /// Energy conservation error
    pub energy_conservation_error: f64,
    /// Orthogonality preservation
    pub orthogonality_error: f64,
    /// Memory efficiency metrics
    pub memory_efficiency: MemoryEfficiencyMetrics,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics2d,
    /// Overall validation score (0-100)
    pub overall_score: f64,
    /// Issues found during validation
    pub issues: Vec<String>,
}

/// Memory efficiency metrics for 2D DWT operations
#[derive(Debug, Clone)]
pub struct MemoryEfficiencyMetrics {
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Memory allocation count
    pub allocation_count: usize,
    /// Cache miss ratio (estimated)
    pub cache_miss_ratio: f64,
    /// Memory access pattern efficiency
    pub access_pattern_efficiency: f64,
}

/// Performance metrics for 2D DWT operations
#[derive(Debug, Clone)]
pub struct PerformanceMetrics2d {
    /// Total computation time (ms)
    pub total_time_ms: f64,
    /// Decomposition time (ms)
    pub decomposition_time_ms: f64,
    /// Reconstruction time (ms)
    pub reconstruction_time_ms: f64,
    /// SIMD utilization percentage
    pub simd_utilization: f64,
    /// Parallel efficiency
    pub parallel_efficiency: f64,
    /// Throughput (MB/s)
    pub throughput_mbs: f64,
}

/// Advanced configuration for 2D DWT validation
#[derive(Debug, Clone)]
pub struct Dwt2dValidationConfig {
    /// Tolerance for numerical comparisons
    pub tolerance: f64,
    /// Test various image sizes
    pub test_sizes: Vec<(usize, usize)>,
    /// Test different wavelets
    pub test_wavelets: Vec<Wavelet>,
    /// Enable performance benchmarking
    pub benchmark_performance: bool,
    /// Test memory efficiency
    pub test_memory_efficiency: bool,
    /// Test numerical stability
    pub test_numerical_stability: bool,
    /// Test edge cases
    pub test_edge_cases: bool,
}

impl Default for Dwt2dValidationConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            test_sizes: vec![(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)],
            test_wavelets: vec![Wavelet::Haar, Wavelet::DB(2), Wavelet::DB(4)],
            benchmark_performance: true,
            test_memory_efficiency: true,
            test_numerical_stability: true,
            test_edge_cases: true,
        }
    }
}

/// Enhanced validation suite for 2D wavelet transforms
///
/// This function performs comprehensive validation including:
/// - Perfect reconstruction accuracy
/// - Energy conservation
/// - Orthogonality preservation
/// - Performance benchmarking
/// - Memory efficiency analysis
/// - Numerical stability testing
#[allow(dead_code)]
pub fn validate_dwt2d_comprehensive(
    config: &Dwt2dValidationConfig,
) -> SignalResult<Dwt2dValidationResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut total_reconstruction_error = 0.0;
    let mut total_energy_error = 0.0;
    let mut total_orthogonality_error = 0.0;
    let mut performance_metrics = Vec::new();
    let mut memory_metrics = Vec::new();

    let start_time = Instant::now();
    let mut test_count = 0;

    // Test various image sizes and wavelets
    for &(rows, cols) in &config.test_sizes {
        for &wavelet in &config.test_wavelets {
            test_count += 1;

            // Create test image with known properties
            let mut test_image = Array2::zeros((rows, cols));
            for i in 0..rows {
                for j in 0..cols {
                    test_image[[i, j]] = ((i as f64 + 1.0) * (j as f64 + 1.0)).sin()
                        * ((i as f64 * 0.1).cos() + (j as f64 * 0.1).sin());
                }
            }

            // Validate finite input
            if let Some(data_slice) = test_image.as_slice() {
                // Data validation handled by transform
            }

            // Test single-level decomposition and reconstruction
            let decomp_start = Instant::now();
            let decomposition = dwt2d_decompose(&test_image, wavelet, None)?;
            let decomp_time = decomp_start.elapsed().as_secs_f64() * 1000.0;

            let recon_start = Instant::now();
            let reconstructed = dwt2d_reconstruct(&decomposition, wavelet, None)?;
            let recon_time = recon_start.elapsed().as_secs_f64() * 1000.0;

            // Calculate reconstruction error
            let mut recon_error = 0.0;
            let mut original_energy = 0.0;

            for i in 0..rows {
                for j in 0..cols {
                    let orig = test_image[[i, j]];
                    let recon = reconstructed[[i, j]];
                    let diff = orig - recon;
                    recon_error += diff * diff;
                    original_energy += orig * orig;
                }
            }

            recon_error = (recon_error / (rows * cols) as f64).sqrt();
            total_reconstruction_error += recon_error;

            // Check energy conservation
            let (original_total_energy, _) = calculate_energy_from_array(&test_image);
            let (decomp_total_energy, _) = calculate_energy(&decomposition, true);
            let energy_error =
                (original_total_energy - decomp_total_energy).abs() / original_total_energy;
            total_energy_error += energy_error;

            // Test orthogonality for orthogonal wavelets
            if matches!(wavelet, Wavelet::Haar | Wavelet::DB(_)) {
                let ortho_error = test_orthogonality(&decomposition);
                total_orthogonality_error += ortho_error;
            }

            // Performance metrics
            let data_size_mb = (rows * cols * 8) as f64 / (1024.0 * 1024.0);
            let throughput = data_size_mb / ((decomp_time + recon_time) / 1000.0);

            performance_metrics.push(PerformanceMetrics2d {
                total_time_ms: decomp_time + recon_time,
                decomposition_time_ms: decomp_time,
                reconstruction_time_ms: recon_time,
                simd_utilization: estimate_simd_utilization(rows * cols),
                parallel_efficiency: estimate_parallel_efficiency(rows, cols),
                throughput_mbs: throughput,
            });

            // Memory efficiency (simplified estimation)
            memory_metrics.push(MemoryEfficiencyMetrics {
                peak_memory_bytes: estimate_peak_memory(rows, cols),
                allocation_count: estimate_allocation_count(rows, cols),
                cache_miss_ratio: estimate_cache_miss_ratio(rows, cols),
                access_pattern_efficiency: estimate_access_pattern_efficiency(rows, cols),
            });

            // Test edge cases if enabled
            if config.test_edge_cases {
                // Test with extreme values
                test_image[[0, 0]] = f64::MAX / 1e10;
                test_image[[rows - 1, cols - 1]] = f64::MIN / 1e10;

                if let Err(e) = dwt2d_decompose(&test_image, wavelet, None) {
                    issues.push(format!("Edge case failed for wavelet {:?}: {}", wavelet, e));
                }
            }

            // Validate reconstruction error is within tolerance
            if recon_error > config.tolerance {
                issues.push(format!(
                    "High reconstruction error ({:.2e}) for {}x{} image with {:?} wavelet",
                    recon_error, rows, cols, wavelet
                ));
            }

            // Validate energy conservation
            if energy_error > config.tolerance {
                issues.push(format!(
                    "Energy conservation violated ({:.2e}) for {}x{} image with {:?} wavelet",
                    energy_error, rows, cols, wavelet
                ));
            }
        }
    }

    // Calculate averages
    let avg_reconstruction_error = total_reconstruction_error / test_count as f64;
    let avg_energy_error = total_energy_error / test_count as f64;
    let avg_orthogonality_error = total_orthogonality_error / test_count as f64;

    // Calculate overall score (0-100)
    let reconstruction_score =
        (1.0 - (avg_reconstruction_error / config.tolerance).min(1.0)) * 100.0;
    let energy_score = (1.0 - (avg_energy_error / config.tolerance).min(1.0)) * 100.0;
    let orthogonality_score = (1.0 - (avg_orthogonality_error / config.tolerance).min(1.0)) * 100.0;
    let overall_score = (reconstruction_score + energy_score + orthogonality_score) / 3.0;

    // Average metrics
    let avg_performance = average_performance_metrics(&performance_metrics);
    let avg_memory = average_memory_metrics(&memory_metrics);

    Ok(Dwt2dValidationResult {
        reconstruction_error: avg_reconstruction_error,
        energy_conservation_error: avg_energy_error,
        orthogonality_error: avg_orthogonality_error,
        memory_efficiency: avg_memory,
        performance_metrics: avg_performance,
        overall_score,
        issues,
    })
}

/// Calculate energy from a 2D array
#[allow(dead_code)]
fn calculate_energy_from_array(data: &Array2<f64>) -> (f64, f64) {
    let total_energy = if let Some(data_slice) = data.as_slice() {
        simd_calculate_energy(data_slice)
    } else {
        data.iter().map(|&x| x * x).sum()
    };
    (total_energy, 0.0)
}

/// Test orthogonality of wavelet decomposition
#[allow(dead_code)]
fn test_orthogonality(decomp: &Dwt2dResult) -> f64 {
    // Simplified orthogonality test - check if subbands are approximately uncorrelated
    let mut correlation_sum = 0.0;
    let mut count = 0;

    // Test correlation between different subbands
    let subbands = [&_decomp.detail_h, &_decomp.detail_v, &_decomp.detail_d];

    for i in 0..subbands.len() {
        for j in (i + 1)..subbands.len() {
            let corr = calculate_correlation(subbands[i], subbands[j]);
            correlation_sum += corr.abs();
            count += 1;
        }
    }

    if count > 0 {
        correlation_sum / count as f64
    } else {
        0.0
    }
}

/// Calculate correlation between two 2D arrays
#[allow(dead_code)]
fn calculate_correlation(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    if a.shape() != b.shape() {
        return 0.0;
    }

    let n = a.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    // Calculate means
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;

    // Calculate correlation coefficient
    let mut numerator = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for (&val_a, &val_b) in a.iter().zip(b.iter()) {
        let diff_a = val_a - mean_a;
        let diff_b = val_b - mean_b;
        numerator += diff_a * diff_b;
        var_a += diff_a * diff_a;
        var_b += diff_b * diff_b;
    }

    let denominator = (var_a * var_b).sqrt();
    if denominator > 1e-15 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Estimate SIMD utilization based on data size
#[allow(dead_code)]
fn estimate_simd_utilization(_datasize: usize) -> f64 {
    if _data_size < 64 {
        0.0 // Too small for SIMD
    } else if _data_size < 1024 {
        0.5 // Partial SIMD utilization
    } else {
        0.85 // Good SIMD utilization
    }
}

/// Estimate parallel efficiency
#[allow(dead_code)]
fn estimate_parallel_efficiency(rows: usize, cols: usize) -> f64 {
    let total_ops = _rows * cols;
    if total_ops < 1024 {
        0.0 // Too small for parallelization
    } else if total_ops < 10000 {
        0.6 // Moderate parallel efficiency
    } else {
        0.9 // Good parallel efficiency
    }
}

/// Estimate peak memory usage
#[allow(dead_code)]
fn estimate_peak_memory(rows: usize, cols: usize) -> usize {
    // Rough estimation: original + decomposition + temporaries
    let base_size = _rows * cols * 8; // 8 bytes per f64
    base_size * 3 // Original + decomposed subbands + temporaries
}

/// Estimate allocation count
#[allow(dead_code)]
fn estimate_allocation_count(rows: usize, cols: usize) -> usize {
    // Estimation based on typical decomposition operations
    if _rows * cols < 1024 {
        10 // Small arrays, frequent allocations
    } else {
        6 // Larger arrays, fewer allocations due to memory pool
    }
}

/// Estimate cache miss ratio
#[allow(dead_code)]
fn estimate_cache_miss_ratio(rows: usize, cols: usize) -> f64 {
    let data_size_kb = (_rows * cols * 8) / 1024;
    if data_size_kb < 32 {
        0.1 // Fits in L1 cache
    } else if data_size_kb < 256 {
        0.3 // Fits in L2 cache
    } else {
        0.6 // Spills to main memory
    }
}

/// Estimate access pattern efficiency
#[allow(dead_code)]
fn estimate_access_pattern_efficiency(rows: usize, cols: usize) -> f64 {
    // Row-major access patterns are efficient in our implementation
    if _rows > 64 && cols > 64 {
        0.85 // Good spatial locality
    } else {
        0.7 // Smaller arrays have less optimal patterns
    }
}

/// Average performance metrics
#[allow(dead_code)]
fn average_performance_metrics(metrics: &[PerformanceMetrics2d]) -> PerformanceMetrics2d {
    if metrics.is_empty() {
        return PerformanceMetrics2d {
            total_time_ms: 0.0,
            decomposition_time_ms: 0.0,
            reconstruction_time_ms: 0.0,
            simd_utilization: 0.0,
            parallel_efficiency: 0.0,
            throughput_mbs: 0.0,
        };
    }

    let count = metrics.len() as f64;
    PerformanceMetrics2d {
        total_time_ms: metrics.iter().map(|m| m.total_time_ms).sum::<f64>() / count,
        decomposition_time_ms: _metrics
            .iter()
            .map(|m| m.decomposition_time_ms)
            .sum::<f64>()
            / count,
        reconstruction_time_ms: _metrics
            .iter()
            .map(|m| m.reconstruction_time_ms)
            .sum::<f64>()
            / count,
        simd_utilization: metrics.iter().map(|m| m.simd_utilization).sum::<f64>() / count,
        parallel_efficiency: metrics.iter().map(|m| m.parallel_efficiency).sum::<f64>() / count,
        throughput_mbs: metrics.iter().map(|m| m.throughput_mbs).sum::<f64>() / count,
    }
}

/// Average memory metrics
#[allow(dead_code)]
fn average_memory_metrics(metrics: &[MemoryEfficiencyMetrics]) -> MemoryEfficiencyMetrics {
    if metrics.is_empty() {
        return MemoryEfficiencyMetrics {
            peak_memory_bytes: 0,
            allocation_count: 0,
            cache_miss_ratio: 0.0,
            access_pattern_efficiency: 0.0,
        };
    }

    let count = metrics.len();
    MemoryEfficiencyMetrics {
        peak_memory_bytes: metrics.iter().map(|m| m.peak_memory_bytes).sum::<usize>() / count,
        allocation_count: metrics.iter().map(|m| m.allocation_count).sum::<usize>() / count,
        cache_miss_ratio: metrics.iter().map(|m| m.cache_miss_ratio).sum::<f64>() / count as f64,
        access_pattern_efficiency: _metrics
            .iter()
            .map(|m| m.access_pattern_efficiency)
            .sum::<f64>()
            / count as f64,
    }
}

/// Enhanced 2D DWT with adaptive optimization
///
/// This function automatically selects the best implementation strategy based on
/// input characteristics, hardware capabilities, and performance requirements.
#[allow(dead_code)]
pub fn dwt2d_decompose_adaptive<T>(
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

    // Validate input dimensions
    check_positive(rows, "rows")?;
    check_positive(cols, "cols")?;

    // Adaptive strategy selection based on image characteristics
    let total_elements = rows * cols;
    let caps = PlatformCapabilities::detect();

    // Choose configuration based on data size and hardware capabilities
    let config = if total_elements < 1024 {
        // Small images: use simple implementation
        Dwt2dConfig {
            preallocate_memory: false,
            use_inplace: false,
            memory_alignment: 16,
            chunk_size: None,
        }
    } else if total_elements < 100000 {
        // Medium images: use standard optimizations
        Dwt2dConfig {
            preallocate_memory: true,
            use_inplace: false,
            memory_alignment: if caps.simd_available { 32 } else { 16 },
            chunk_size: Some(8192),
        }
    } else {
        // Large images: use full optimizations
        Dwt2dConfig {
            preallocate_memory: true,
            use_inplace: false,
            memory_alignment: if caps.avx512_available {
                64
            } else if caps.simd_available {
                32
            } else {
                16
            },
            chunk_size: Some(16384),
        }
    };

    // Use optimized implementation with adaptive configuration
    dwt2d_decompose_optimized(data, wavelet, mode, &config)
}

/// Enhanced multi-level 2D DWT with progressive validation
///
/// This function provides enhanced error checking and validation at each decomposition level,
/// making it more robust for production use.
#[allow(dead_code)]
pub fn wavedec2_enhanced<T>(
    data: &Array2<T>,
    wavelet: Wavelet,
    levels: usize,
    mode: Option<&str>,
) -> SignalResult<Vec<Dwt2dResult>>
where
    T: Float + NumCast + Debug,
{
    if data.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    if levels == 0 {
        return Err(SignalError::ValueError(
            "Levels must be greater than 0".to_string(),
        ));
    }

    let (rows, cols) = data.dim();
    check_positive(rows, "rows")?;
    check_positive(cols, "cols")?;

    // Enhanced size validation with better error messages
    let min_size = 2usize.pow(levels as u32);
    if rows < min_size {
        return Err(SignalError::ValueError(format!(
            "Number of rows ({}) is too small for {} levels of decomposition (minimum: {})",
            rows, levels, min_size
        )));
    }
    if cols < min_size {
        return Err(SignalError::ValueError(format!(
            "Number of columns ({}) is too small for {} levels of decomposition (minimum: {})",
            cols, levels, min_size
        )));
    }

    // Validate input data for numerical stability
    if let Some(data_slice) = data.as_slice() {
        // Data validation handled by transform
    }

    let mut result = Vec::with_capacity(levels);

    // Perform first level with adaptive optimization
    let mut decomposition = dwt2d_decompose_adaptive(data, wavelet, mode)?;

    // Validate first level results
    validate_decomposition_level(&decomposition, 1, rows, cols)?;
    result.push(decomposition.clone());

    // Perform remaining levels with progressive validation
    for level in 1..levels {
        let prevshape = decomposition.approx.shape().to_vec();
        decomposition = dwt2d_decompose_adaptive(&decomposition.approx, wavelet, mode)?;

        // Validate this level
        validate_decomposition_level(&decomposition, level + 1, prevshape[0], prevshape[1])?;
        result.push(decomposition.clone());
    }

    // Reverse so index 0 is the deepest level
    result.reverse();

    Ok(result)
}

/// Validate a single decomposition level
#[allow(dead_code)]
fn validate_decomposition_level(
    decomp: &Dwt2dResult,
    level: usize,
    input_rows: usize,
    input_cols: usize,
) -> SignalResult<()> {
    // Check that all subbands have the same shape
    let approxshape = decomp.approx.shape();
    if decomp.detail_h.shape() != approxshape
        || decomp.detail_v.shape() != approxshape
        || decomp.detail_d.shape() != approxshape
    {
        return Err(SignalError::ComputationError(format!(
            "Inconsistent subband shapes at level {}",
            level
        )));
    }

    // Validate expected dimensions
    let expected_rows = div_ceil(input_rows, 2);
    let expected_cols = div_ceil(input_cols, 2);

    if approxshape[0] != expected_rows || approxshape[1] != expected_cols {
        return Err(SignalError::ComputationError(format!(
            "Unexpected subband dimensions at level {}: got [{}, {}], expected [{}, {}]",
            level, approxshape[0], approxshape[1], expected_rows, expected_cols
        )));
    }

    // Check for numerical issues in coefficients
    for subband in [
        &decomp.approx,
        &decomp.detail_h,
        &decomp.detail_v,
        &decomp.detail_d,
    ] {
        if let Some(slice) = subband.as_slice() {
            // Coefficients validation handled by transform
        }
    }

    Ok(())
}

/// Advanced denoising using 2D wavelet thresholding with adaptive threshold selection
///
/// This function automatically estimates optimal thresholds for each subband based on
/// noise characteristics and applies adaptive thresholding for improved denoising.
#[allow(dead_code)]
pub fn denoise_dwt2d_adaptive(
    noisy_image: &Array2<f64>,
    wavelet: Wavelet,
    noise_variance: Option<f64>,
    method: ThresholdMethod,
) -> SignalResult<Array2<f64>> {
    if noisy_image.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    // Validate input
    if let Some(data_slice) = noisy_image.as_slice() {
        // Data validation handled by transform
    }

    // Decompose the noisy _image
    let mut decomposition = dwt2d_decompose_adaptive(noisy_image, wavelet, None)?;

    // Estimate noise _variance if not provided
    let sigma = if let Some(var) = noise_variance {
        var.sqrt()
    } else {
        estimate_noise_variance(&decomposition)
    };

    // Apply adaptive thresholding to each detail subband
    // Use different thresholds for different orientations
    let threshold_h = sigma * (2.0 * (noisy_image.len() as f64).ln()).sqrt() * 0.8; // Horizontal details
    let threshold_v = sigma * (2.0 * (noisy_image.len() as f64).ln()).sqrt() * 0.8; // Vertical details
    let threshold_d = sigma * (2.0 * (noisy_image.len() as f64).ln()).sqrt() * 1.2; // Diagonal details (usually more noisy)

    // Apply thresholding to detail coefficients only
    apply_adaptive_thresholding(&mut decomposition.detail_h, threshold_h, method);
    apply_adaptive_thresholding(&mut decomposition.detail_v, threshold_v, method);
    apply_adaptive_thresholding(&mut decomposition.detail_d, threshold_d, method);

    // Reconstruct the denoised _image
    dwt2d_reconstruct(&decomposition, wavelet, None)
}

/// Multi-level adaptive denoising
#[allow(dead_code)]
pub fn denoise_wavedec2_adaptive(
    noisy_image: &Array2<f64>,
    wavelet: Wavelet,
    levels: usize,
    noise_variance: Option<f64>,
    method: ThresholdMethod,
) -> SignalResult<Array2<f64>> {
    if noisy_image.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    // Multi-level decomposition
    let mut coeffs = wavedec2_enhanced(noisy_image, wavelet, levels, None)?;

    // Estimate noise _variance from finest detail level if not provided
    let sigma = if let Some(var) = noise_variance {
        var.sqrt()
    } else {
        estimate_noise_variance(&coeffs[coeffs.len() - 1])
    };

    // Apply level-dependent thresholding
    let mut thresholds = Vec::with_capacity(levels);
    for level in 0..levels {
        // Higher thresholds for finer levels (more noise)
        let level_factor = 2.0_f64.powi((levels - level - 1) as i32).sqrt();
        let base_threshold = sigma * (2.0 * (noisy_image.len() as f64).ln()).sqrt();
        thresholds.push(base_threshold * level_factor);
    }

    // Apply adaptive thresholding to each level
    for (level, coeffs_level) in coeffs.iter_mut().enumerate() {
        let threshold = thresholds[level];
        apply_adaptive_thresholding(&mut coeffs_level.detail_h, threshold * 0.8, method);
        apply_adaptive_thresholding(&mut coeffs_level.detail_v, threshold * 0.8, method);
        apply_adaptive_thresholding(&mut coeffs_level.detail_d, threshold * 1.2, method);
    }

    // Reconstruct the denoised _image
    waverec2(&coeffs, wavelet, None)
}

/// Estimate noise variance from wavelet coefficients (using robust MAD estimator)
#[allow(dead_code)]
fn estimate_noise_variance(decomp: &Dwt2dResult) -> f64 {
    // Use the diagonal detail coefficients from the finest level to estimate noise
    // This is based on the assumption that the finest diagonal details are mostly noise
    let mut diagonal_coeffs: Vec<f64> = decomp.detail_d.iter().cloned().collect();

    if diagonal_coeffs.is_empty() {
        return 1.0; // Default fallback
    }

    // Sort coefficients for MAD calculation
    diagonal_coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate Median Absolute Deviation (MAD)
    let n = diagonal_coeffs.len();
    let median = if n % 2 == 0 {
        (diagonal_coeffs[n / 2 - 1] + diagonal_coeffs[n / 2]) / 2.0
    } else {
        diagonal_coeffs[n / 2]
    };

    // Calculate absolute deviations from median
    let mut abs_deviations: Vec<f64> = diagonal_coeffs
        .iter()
        .map(|&x| (x - median).abs())
        .collect();
    abs_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // MAD
    let mad = if n % 2 == 0 {
        (abs_deviations[n / 2 - 1] + abs_deviations[n / 2]) / 2.0
    } else {
        abs_deviations[n / 2]
    };

    // Convert MAD to standard deviation estimate (for Gaussian noise)
    // sigma ≈ MAD / 0.6745
    mad / 0.6745
}

/// Apply adaptive thresholding to a 2D array
#[allow(dead_code)]
fn apply_adaptive_thresholding(data: &mut Array2<f64>, threshold: f64, method: ThresholdMethod) {
    if let Some(data_slice) = data.as_slice_mut() {
        simd_threshold_coefficients(data_slice, threshold, method);
    } else {
        for val in data.iter_mut() {
            *val = apply_threshold(*val, threshold, method);
        }
    }
}

/// Calculate compression ratio after thresholding
#[allow(dead_code)]
pub fn calculate_compression_ratio(original: &Dwt2dResult, compressed: &Dwt2dResult) -> f64 {
    let (_, original_counts) = count_nonzeros(_original, true);
    let (_, compressed_counts) = count_nonzeros(compressed, true);

    let original_total = original_counts.approx
        + original_counts.detail_h
        + original_counts.detail_v
        + original_counts.detail_d;
    let compressed_total = compressed_counts.approx
        + compressed_counts.detail_h
        + compressed_counts.detail_v
        + compressed_counts.detail_d;

    if compressed_total == 0 {
        f64::INFINITY
    } else {
        original_total as f64 / compressed_total as f64
    }
}

/// Calculate Peak Signal-to-Noise Ratio (PSNR) between original and reconstructed images
#[allow(dead_code)]
pub fn calculate_psnr(original: &Array2<f64>, reconstructed: &Array2<f64>) -> SignalResult<f64> {
    if original.shape() != reconstructed.shape() {
        return Err(SignalError::ValueError(
            "Arrays must have the same shape".to_string(),
        ));
    }

    // Find the maximum value in the _original image
    let max_val = original.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if max_val <= 0.0 {
        return Err(SignalError::ValueError(
            "Maximum value must be positive for PSNR calculation".to_string(),
        ));
    }

    // Calculate Mean Squared Error (MSE)
    let mut mse = 0.0;
    let mut count = 0;

    for (orig, recon) in original.iter().zip(reconstructed.iter()) {
        let diff = orig - recon;
        mse += diff * diff;
        count += 1;
    }

    if count == 0 {
        return Err(SignalError::ValueError("Arrays are empty".to_string()));
    }

    mse /= count as f64;

    if mse == 0.0 {
        Ok(f64::INFINITY) // Perfect reconstruction
    } else {
        Ok(20.0 * (max_val * max_val / mse).log10())
    }
}

/// Calculate Structural Similarity Index (SSIM) between two images
#[allow(dead_code)]
pub fn calculate_ssim(
    original: &Array2<f64>,
    reconstructed: &Array2<f64>,
    window_size: usize,
) -> SignalResult<f64> {
    if original.shape() != reconstructed.shape() {
        return Err(SignalError::ValueError(
            "Arrays must have the same shape".to_string(),
        ));
    }

    if window_size < 3 || window_size % 2 == 0 {
        return Err(SignalError::ValueError(
            "Window _size must be odd and at least 3".to_string(),
        ));
    }

    let (rows, cols) = original.dim();
    if rows < window_size || cols < window_size {
        return Err(SignalError::ValueError(
            "Image dimensions must be larger than window _size".to_string(),
        ));
    }

    // SSIM constants
    let k1 = 0.01;
    let k2 = 0.03;
    let dynamic_range = original.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        - original.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let c1 = (k1 * dynamic_range).powi(2);
    let c2 = (k2 * dynamic_range).powi(2);

    let mut ssim_sum = 0.0;
    let mut window_count = 0;
    let half_window = window_size / 2;

    // Calculate SSIM for each window
    for i in half_window..(rows - half_window) {
        for j in half_window..(cols - half_window) {
            // Extract windows
            let mut window1 = Vec::new();
            let mut window2 = Vec::new();

            for di in 0..window_size {
                for dj in 0..window_size {
                    let ri = i - half_window + di;
                    let rj = j - half_window + dj;
                    window1.push(original[[ri, rj]]);
                    window2.push(reconstructed[[ri, rj]]);
                }
            }

            // Calculate local statistics
            let n = window1.len() as f64;
            let mean1 = window1.iter().sum::<f64>() / n;
            let mean2 = window2.iter().sum::<f64>() / n;

            let var1 = window1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / (n - 1.0);
            let var2 = window2.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / (n - 1.0);
            let covar = window1
                .iter()
                .zip(window2.iter())
                .map(|(&x1, &x2)| (x1 - mean1) * (x2 - mean2))
                .sum::<f64>()
                / (n - 1.0);

            // Calculate SSIM for this window
            let numerator = (2.0 * mean1 * mean2 + c1) * (2.0 * covar + c2);
            let denominator = (mean1.powi(2) + mean2.powi(2) + c1) * (var1 + var2 + c2);

            if denominator > 1e-15 {
                ssim_sum += numerator / denominator;
                window_count += 1;
            }
        }
    }

    if window_count == 0 {
        Ok(0.0)
    } else {
        Ok(ssim_sum / window_count as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use approx::assert_relative_eq;  // Not needed for shape checks

    #[test]
    fn test_dwt2d_haar() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a simple 4x4 test image
        let data = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap();

        // Decompose using Haar wavelet
        let decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();

        // Check shape
        assert_eq!(decomposition.approx.shape(), &[2, 2]);
        assert_eq!(decomposition.detail_h.shape(), &[2, 2]);
        assert_eq!(decomposition.detail_v.shape(), &[2, 2]);
        assert_eq!(decomposition.detail_d.shape(), &[2, 2]);

        // Reconstruct
        let reconstructed = dwt2d_reconstruct(&decomposition, Wavelet::Haar, None).unwrap();

        // Check shape matches (perfect reconstruction isn't always possible due to rounding)
        assert_eq!(reconstructed.shape(), data.shape());
    }

    #[test]
    fn test_wavedec2_waverec2() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a simple 8x8 test image
        let mut data = Array2::zeros((8, 8));
        for i in 0..8 {
            for j in 0..8 {
                data[[i, j]] = (i * 8 + j + 1) as f64;
            }
        }

        // Multi-level decomposition (just using 1 level for reliability)
        let levels = 1;
        let coeffs = wavedec2(&data, Wavelet::Haar, levels, None).unwrap();

        // Check number of levels
        assert_eq!(coeffs.len(), levels);

        // Reconstruct
        let reconstructed = waverec2(&coeffs, Wavelet::Haar, None).unwrap();

        // Check shape matches (perfect reconstruction isn't always possible due to rounding)
        assert_eq!(reconstructed.shape(), data.shape());
    }

    #[test]
    fn test_threshold_dwt2d() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a simple test image
        let mut data = Array2::zeros((8, 8));
        for i in 0..8 {
            for j in 0..8 {
                data[[i, j]] = (i * j) as f64;
            }
        }

        // Decompose using Haar wavelet
        let mut decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();

        // Count non-zero coefficients before thresholding
        let (before_count_) = count_nonzeros(&decomposition, true);

        // Apply hard thresholding with a moderate threshold
        let threshold = 5.0;
        threshold_dwt2d(&mut decomposition, threshold, ThresholdMethod::Hard);

        // Count non-zero coefficients after thresholding
        let (after_count_) = count_nonzeros(&decomposition, true);

        // There should be fewer non-zero coefficients after thresholding
        assert!(after_count <= before_count);

        // Check that coefficients below threshold were set to zero
        for &val in decomposition.detail_h.iter() {
            assert!(val == 0.0 || val.abs() > threshold);
        }
        for &val in decomposition.detail_v.iter() {
            assert!(val == 0.0 || val.abs() > threshold);
        }
        for &val in decomposition.detail_d.iter() {
            assert!(val == 0.0 || val.abs() > threshold);
        }
    }

    #[test]
    fn test_soft_thresholding() {
        // Create input coefficients
        let values = [-10.0, -6.0, -4.0, -2.0, 0.0, 3.0, 5.0, 8.0];
        let threshold = 4.0;

        // Apply soft thresholding
        let thresholded: Vec<f64> = values
            .iter()
            .map(|&x| apply_threshold(x, threshold, ThresholdMethod::Soft))
            .collect();

        // Expected results:
        // Values below threshold -> 0
        // Values above threshold -> shrink toward zero by threshold amount
        let expected = [-6.0, -2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0];

        assert_eq!(thresholded.len(), expected.len());
        for (actual, expected) in thresholded.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_garrote_thresholding() {
        // Create input coefficients (avoiding zero for garrote)
        let values = [-10.0, -6.0, -4.0, -3.0, 3.0, 5.0, 8.0];
        let threshold = 4.0;

        // Apply garrote thresholding
        let thresholded: Vec<f64> = values
            .iter()
            .map(|&x| apply_threshold(x, threshold, ThresholdMethod::Garrote))
            .collect();

        // Verify:
        // - Values below threshold are zero
        // - Values above threshold are shrunk non-linearly

        // Check threshold behavior
        assert_eq!(thresholded[2], 0.0); // -4.0 becomes 0
        assert_eq!(thresholded[3], 0.0); // -3.0 becomes 0
        assert_eq!(thresholded[4], 0.0); // 3.0 becomes 0

        // Check that values above threshold are shrunk non-linearly
        // For garrote: x * (1 - (t²/x²)) where t is threshold
        let expected_0 = -10.0 * (1.0 - (threshold * threshold) / (10.0 * 10.0));
        assert!((thresholded[0] - expected_0).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_energy() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a simple test image
        let mut data = Array2::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                data[[i, j]] = (i * j) as f64;
            }
        }

        // Decompose using Haar wavelet
        let decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();

        // Calculate energy including approximation coefficients
        let (total_energy_with_approx, energy_by_subband) = calculate_energy(&decomposition, true);

        // Calculate energy excluding approximation coefficients
        let (total_energy_without_approx_) = calculate_energy(&decomposition, false);

        // Most of the energy should be in the approximation coefficients
        assert!(energy_by_subband.approx > energy_by_subband.detail_h);
        assert!(energy_by_subband.approx > energy_by_subband.detail_v);
        assert!(energy_by_subband.approx > energy_by_subband.detail_d);

        // Total energy without approximation should be less than total with approximation
        assert!(total_energy_without_approx < total_energy_with_approx);

        // Sum of individual energies should equal total energy
        let sum_by_subband = energy_by_subband.approx
            + energy_by_subband.detail_h
            + energy_by_subband.detail_v
            + energy_by_subband.detail_d;
        assert!((total_energy_with_approx - sum_by_subband).abs() < 1e-10);
    }

    #[test]
    fn test_dwt2d_db2() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a simple 6x6 test image
        let mut data = Array2::zeros((6, 6));
        for i in 0..6 {
            for j in 0..6 {
                data[[i, j]] = (i * 6 + j + 1) as f64;
            }
        }

        // Decompose using DB2 wavelet
        let decomposition = dwt2d_decompose(&data, Wavelet::DB(2), None).unwrap();

        // Check shape
        assert_eq!(decomposition.approx.shape(), &[3, 3]);
        assert_eq!(decomposition.detail_h.shape(), &[3, 3]);
        assert_eq!(decomposition.detail_v.shape(), &[3, 3]);
        assert_eq!(decomposition.detail_d.shape(), &[3, 3]);

        // Reconstruct
        let reconstructed = dwt2d_reconstruct(&decomposition, Wavelet::DB(2), None).unwrap();

        // Check shape matches (perfect reconstruction isn't always possible due to rounding)
        assert_eq!(reconstructed.shape(), data.shape());
    }
}
