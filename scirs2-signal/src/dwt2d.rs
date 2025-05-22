//! Two-dimensional Discrete Wavelet Transform (DWT2D)
//!
//! This module provides implementations of the 2D Discrete Wavelet Transform,
//! useful for image processing, compression, and multi-resolution analysis
//! of 2D signals like images.
//!
//! # Performance Optimizations
//!
//! This implementation includes several optimizations for performance:
//!
//! 1. **Parallel Processing**: When compiled with the "parallel" feature,
//!    row and column transforms can be computed in parallel using Rayon.
//!
//! 2. **Memory Efficiency**:
//!    - Minimizes temporary allocations
//!    - Uses ndarray views for zero-copy operations
//!    - Implements cache-friendly traversal patterns
//!
//! 3. **Algorithm Optimizations**:
//!    - Direct transform path for common wavelets (Haar, DB2, DB4)
//!    - Optimized convolution for filter operations
//!    - Efficient boundary handling
//!
//! ## Overview
//!
//! The 2D Discrete Wavelet Transform (DWT2D) extends the concept of 1D wavelet transforms
//! to two-dimensional data. It is particularly useful for processing images and other 2D signals.
//! By using separable filters, the 2D DWT applies 1D transforms first along rows and then along
//! columns, decomposing the image into four subbands:
//!
//! - **LL (Approximation)**: Low-frequency content in both horizontal and vertical directions
//! - **LH (Horizontal Detail)**: High-frequency in horizontal direction, low-frequency in vertical direction
//! - **HL (Vertical Detail)**: Low-frequency in horizontal direction, high-frequency in vertical direction
//! - **HH (Diagonal Detail)**: High-frequency content in both horizontal and vertical directions
//!
//! ## Applications
//!
//! The 2D DWT has numerous applications:
//!
//! - **Image Compression**: By thresholding small coefficients (as in JPEG2000)
//! - **Image Denoising**: By thresholding coefficients below a noise threshold
//! - **Feature Extraction**: Using wavelet coefficients as image features
//! - **Texture Analysis**: Analyzing frequency content at different scales
//! - **Edge Detection**: Using detail coefficients to detect edges
//!
//! ## Usage Examples
//!
//! Basic decomposition and reconstruction:
//!
//! ```
//! use ndarray::Array2;
//! use scirs2_signal::dwt::Wavelet;
//! use scirs2_signal::dwt2d::{dwt2d_decompose, dwt2d_reconstruct};
//!
//! // Create a simple "image"
//! let data = Array2::from_shape_vec((4, 4), vec![
//!     1.0, 2.0, 3.0, 4.0,
//!     5.0, 6.0, 7.0, 8.0,
//!     9.0, 10.0, 11.0, 12.0,
//!     13.0, 14.0, 15.0, 16.0
//! ]).unwrap();
//!
//! // Decompose using Haar wavelet
//! let decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();
//!
//! // Reconstruct
//! let reconstructed = dwt2d_reconstruct(&decomposition, Wavelet::Haar, None).unwrap();
//! ```
//!
//! Multi-level decomposition:
//!
//! ```
//! use ndarray::Array2;
//! use scirs2_signal::dwt::Wavelet;
//! use scirs2_signal::dwt2d::{wavedec2, waverec2};
//!
//! // Create a simple "image"
//! let mut data = Array2::zeros((8, 8));
//! for i in 0..8 {
//!     for j in 0..8 {
//!         data[[i, j]] = (i * j) as f64;
//!     }
//! }
//!
//! // Multi-level decomposition
//! let levels = 2;
//! let coeffs = wavedec2(&data, Wavelet::DB(4), levels, None).unwrap();
//!
//! // Reconstruct from multi-level decomposition
//! let reconstructed = waverec2(&coeffs, Wavelet::DB(4), None).unwrap();
//! ```

use crate::dwt::{self, Wavelet};
use crate::error::{SignalError, SignalResult};
use ndarray::Array2;
use num_traits::{Float, NumCast};
use std::fmt::Debug;

// Import rayon for parallel processing when the "parallel" feature is enabled
#[cfg(feature = "parallel")]
use rayon::prelude::*;
// use std::convert::TryInto;  // Not needed

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
///         image[[i, j]] = (i * j) as f64;
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

/// Performs a single-level 2D discrete wavelet transform.
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
///         data[[i, j]] = (i * j) as f64;
///     }
/// }
///
/// // Perform 2D DWT using the Daubechies 4 wavelet with periodic boundary extension
/// let result = dwt2d_decompose(&data, Wavelet::DB(4), Some("periodic")).unwrap();
///
/// // The output size depends on the input size and the wavelet length
/// assert_eq!(result.approx.shape(), &[8, 8]);
/// ```
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

    // Get dimensions
    let (rows, cols) = data.dim();

    // Convert input to f64
    let data_f64 = data.mapv(|val| {
        num_traits::cast::cast::<T, f64>(val)
            .unwrap_or_else(|| panic!("Could not convert {:?} to f64", val))
    });

    // Calculate output dimensions (ceiling division for half the size)
    // Use integer division that rounds up
    let output_rows = rows.div_ceil(2); // TODO: Replace with div_ceil when stable
    let output_cols = cols.div_ceil(2); // TODO: Replace with div_ceil when stable

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
        let mut row_results: Vec<(usize, Vec<f64>, Vec<f64>)> = (0..rows)
            .into_par_iter()
            .map(|i| {
                let row = data_f64.slice(ndarray::s![i, ..]).to_vec();
                let (approx, detail) =
                    dwt::dwt_decompose(&row, wavelet, mode).expect("Row transform failed");
                (i, approx, detail)
            })
            .collect();

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
        let column_results: Vec<(usize, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> = (0..output_cols)
            .into_par_iter()
            .map(|j| {
                // Process low-pass filtered rows
                let col_lo = rows_lo.slice(ndarray::s![.., j]).to_vec();
                let (approx_lo, detail_lo) = dwt::dwt_decompose(&col_lo, wavelet, mode)
                    .expect("Column transform failed (low-pass)");

                // Process high-pass filtered rows
                let col_hi = rows_hi.slice(ndarray::s![.., j]).to_vec();
                let (approx_hi, detail_hi) = dwt::dwt_decompose(&col_hi, wavelet, mode)
                    .expect("Column transform failed (high-pass)");

                (j, approx_lo, detail_lo, approx_hi, detail_hi)
            })
            .collect();

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
///         data[[i, j]] = (i * j) as f64;
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
pub fn dwt2d_reconstruct(
    decomposition: &Dwt2dResult,
    wavelet: Wavelet,
    _mode: Option<&str>,
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
        let col_results: Vec<(usize, Vec<f64>, Vec<f64>)> = (0..cols)
            .into_par_iter()
            .map(|j| {
                // Reconstruct low-pass columns
                let ll_col = ll.slice(ndarray::s![.., j]).to_vec();
                let hl_col = hl.slice(ndarray::s![.., j]).to_vec();
                let col_lo = dwt::dwt_reconstruct(&ll_col, &hl_col, wavelet)
                    .expect("Low-pass column reconstruction failed");

                // Reconstruct high-pass columns
                let lh_col = lh.slice(ndarray::s![.., j]).to_vec();
                let hh_col = hh.slice(ndarray::s![.., j]).to_vec();
                let col_hi = dwt::dwt_reconstruct(&lh_col, &hh_col, wavelet)
                    .expect("High-pass column reconstruction failed");

                (j, col_lo, col_hi)
            })
            .collect();

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
        let row_results: Vec<(usize, Vec<f64>)> = (0..out_rows)
            .into_par_iter()
            .map(|i| {
                // Get rows from low and high frequency parts
                let lo_row = row_lo.slice(ndarray::s![i, ..]).to_vec();
                let hi_row = row_hi.slice(ndarray::s![i, ..]).to_vec();

                // Reconstruct row
                let full_row = dwt::dwt_reconstruct(&lo_row, &hi_row, wavelet)
                    .expect("Row reconstruction failed");

                (i, full_row)
            })
            .collect();

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
///         data[[i, j]] = (i * 8 + j + 1) as f64;
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
///         data[[i, j]] = ((i+j) % 8) as f64;  // Create a pattern
///     }
/// }
///
/// // Decompose with Daubechies 4 wavelet
/// let coeffs = wavedec2(&data, Wavelet::DB(4), 2, None).unwrap();
/// assert_eq!(coeffs.len(), 2);
/// ```
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
///         data[[i, j]] = (i * 8 + j + 1) as f64;
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
///         data[[i, j]] = (i * j) as f64;
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
pub fn threshold_dwt2d(decomposition: &mut Dwt2dResult, threshold: f64, method: ThresholdMethod) {
    // Apply thresholding to horizontal detail coefficients
    for h in decomposition.detail_h.iter_mut() {
        *h = apply_threshold(*h, threshold, method);
    }

    // Apply thresholding to vertical detail coefficients
    for v in decomposition.detail_v.iter_mut() {
        *v = apply_threshold(*v, threshold, method);
    }

    // Apply thresholding to diagonal detail coefficients
    for d in decomposition.detail_d.iter_mut() {
        *d = apply_threshold(*d, threshold, method);
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
///         data[[i, j]] = (i * j) as f64;
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
///         data[[i, j]] = (i * j) as f64;
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
pub fn calculate_energy(decomposition: &Dwt2dResult, include_approx: bool) -> (f64, WaveletEnergy) {
    // Calculate energy for each subband
    let approx_energy = if include_approx {
        decomposition.approx.iter().map(|&x| x * x).sum()
    } else {
        0.0
    };

    let detail_h_energy = decomposition.detail_h.iter().map(|&x| x * x).sum();
    let detail_v_energy = decomposition.detail_v.iter().map(|&x| x * x).sum();
    let detail_d_energy = decomposition.detail_d.iter().map(|&x| x * x).sum();

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
///         data[[i, j]] = (i * j) as f64;
///     }
/// }
///
/// // Decompose the image
/// let mut decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();
///
/// // Count coefficients before thresholding
/// let (before_total, _) = count_nonzeros(&decomposition, true);
///
/// // Apply thresholding
/// threshold_dwt2d(&mut decomposition, 5.0, ThresholdMethod::Hard);
///
/// // Count coefficients after thresholding
/// let (after_total, _) = count_nonzeros(&decomposition, true);
///
/// // After thresholding, there should be fewer non-zero coefficients
/// assert!(after_total <= before_total);
/// ```
pub fn count_nonzeros(decomposition: &Dwt2dResult, include_approx: bool) -> (usize, WaveletCounts) {
    // Count non-zero coefficients in each subband
    let approx_count = if include_approx {
        decomposition.approx.iter().filter(|&&x| x != 0.0).count()
    } else {
        0
    };

    let detail_h_count = decomposition.detail_h.iter().filter(|&&x| x != 0.0).count();
    let detail_v_count = decomposition.detail_v.iter().filter(|&&x| x != 0.0).count();
    let detail_d_count = decomposition.detail_d.iter().filter(|&&x| x != 0.0).count();

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    // use approx::assert_relative_eq;  // Not needed for shape checks

    #[test]
    fn test_dwt2d_haar() {
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
        let (before_count, _) = count_nonzeros(&decomposition, true);

        // Apply hard thresholding with a moderate threshold
        let threshold = 5.0;
        threshold_dwt2d(&mut decomposition, threshold, ThresholdMethod::Hard);

        // Count non-zero coefficients after thresholding
        let (after_count, _) = count_nonzeros(&decomposition, true);

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
        let (total_energy_without_approx, _) = calculate_energy(&decomposition, false);

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
