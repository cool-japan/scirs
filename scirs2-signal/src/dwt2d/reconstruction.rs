//! 2D DWT Reconstruction Functions
//!
//! This module provides reconstruction functions for 2D Discrete Wavelet Transform (DWT),
//! including single-level and multi-level reconstruction capabilities with optimized
//! implementations for various wavelet types.

use crate::dwt::Wavelet;
use crate::error::{SignalError, SignalResult};
use super::types::Dwt2dResult;
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::{Float, NumCast};
use std::fmt::Debug;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Performs a single-level 2D discrete wavelet reconstruction.
///
/// This function reconstructs a 2D signal from its wavelet decomposition,
/// which consists of approximation coefficients (LL), horizontal details (LH),
/// vertical details (HL), and diagonal details (HH).
///
/// # Algorithm
///
/// The 2D reconstruction algorithm operates in two stages:
/// 1. **Column reconstruction**: For each column, reconstruct from LL+HL (low-pass) and LH+HH (high-pass) pairs
/// 2. **Row reconstruction**: For each row, reconstruct from the intermediate low-pass and high-pass results
///
/// This separable approach allows efficient implementation and is mathematically equivalent
/// to a full 2D wavelet reconstruction.
///
/// # Arguments
///
/// * `decomposition` - The wavelet decomposition containing four subbands
///   - `approx`: Low-frequency approximation coefficients (LL)
///   - `detail_h`: Horizontal edge details (LH)
///   - `detail_v`: Vertical edge details (HL)
///   - `detail_d`: Diagonal edge details (HH)
/// * `wavelet` - The wavelet family to use for reconstruction (must match original decomposition)
/// * `mode` - Signal extension mode (currently not fully implemented, defaults to "symmetric")
///
/// # Returns
///
/// * A 2D array with dimensions twice the size of the input subbands
/// * For input subbands of size `(M, N)`, output will be `(2M, 2N)`
///
/// # Errors
///
/// Returns an error if:
/// * Any subband is empty
/// * Subbands have inconsistent shapes
/// * The underlying 1D DWT reconstruction fails
/// * Memory allocation fails
///
/// # Parallel Processing
///
/// When compiled with the "parallel" feature, this function uses Rayon to parallelize:
/// * Column reconstruction across different columns
/// * Row reconstruction across different rows
///
/// This can provide significant speedups for larger images on multi-core systems.
///
/// # Examples
///
/// Basic 2D reconstruction:
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::{dwt2d_decompose, dwt2d_reconstruct};
///
/// // Create a simple 4x4 test image
/// let mut data = Array2::zeros((4, 4));
/// for i in 0..4 {
///     for j in 0..4 {
///         data[[i, j]] = (i * 4 + j + 1) as f64;
///     }
/// }
///
/// // Decompose
/// let decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();
///
/// // Reconstruct
/// let reconstructed = dwt2d_reconstruct(&decomposition, Wavelet::Haar, None).unwrap();
///
/// // Check dimensions
/// assert_eq!(reconstructed.shape(), data.shape());
/// ```
///
/// Reconstruction after coefficient modification:
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::{dwt2d_decompose, dwt2d_reconstruct, Dwt2dResult};
///
/// let mut data = Array2::ones((8, 8));
/// let mut decomposition = dwt2d_decompose(&data, Wavelet::DB(4), None).unwrap();
///
/// // Zero out high-frequency details for lossy compression
/// decomposition.detail_h.fill(0.0);
/// decomposition.detail_v.fill(0.0);
/// decomposition.detail_d.fill(0.0);
///
/// // Reconstruct from modified coefficients
/// let denoised = dwt2d_reconstruct(&decomposition, Wavelet::DB(4), None).unwrap();
/// ```
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
                let ll_col = ll.slice(scirs2_core::ndarray::s![.., j]).to_vec();
                let hl_col = hl.slice(scirs2_core::ndarray::s![.., j]).to_vec();
                let col_lo = crate::dwt::dwt_reconstruct(&ll_col, &hl_col, wavelet).map_err(|e| {
                    SignalError::ComputationError(format!(
                        "Low-pass column reconstruction failed: {}",
                        e
                    ))
                })?;

                // Reconstruct high-pass columns
                let lh_col = lh.slice(scirs2_core::ndarray::s![.., j]).to_vec();
                let hh_col = hh.slice(scirs2_core::ndarray::s![.., j]).to_vec();
                let col_hi = crate::dwt::dwt_reconstruct(&lh_col, &hh_col, wavelet).map_err(|e| {
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
            let ll_col = ll.slice(scirs2_core::ndarray::s![.., j]).to_vec();
            let hl_col = hl.slice(scirs2_core::ndarray::s![.., j]).to_vec();
            let col_lo = crate::dwt::dwt_reconstruct(&ll_col, &hl_col, wavelet)?;

            // Reconstruct high-pass columns
            let lh_col = lh.slice(scirs2_core::ndarray::s![.., j]).to_vec();
            let hh_col = hh.slice(scirs2_core::ndarray::s![.., j]).to_vec();
            let col_hi = crate::dwt::dwt_reconstruct(&lh_col, &hh_col, wavelet)?;

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
                let lo_row = row_lo.slice(scirs2_core::ndarray::s![i, ..]).to_vec();
                let hi_row = row_hi.slice(scirs2_core::ndarray::s![i, ..]).to_vec();

                // Reconstruct row
                let full_row = crate::dwt::dwt_reconstruct(&lo_row, &hi_row, wavelet).map_err(|e| {
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
            let lo_row = row_lo.slice(scirs2_core::ndarray::s![i, ..]).to_vec();
            let hi_row = row_hi.slice(scirs2_core::ndarray::s![i, ..]).to_vec();

            // Reconstruct row
            let full_row = crate::dwt::dwt_reconstruct(&lo_row, &hi_row, wavelet)?;

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

/// Performs a multi-level 2D discrete wavelet reconstruction.
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
/// use scirs2_core::ndarray::Array2;
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
/// use scirs2_core::ndarray::Array2;
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
/// use scirs2_core::ndarray::Array2;
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
/// use scirs2_core::ndarray::Array2;
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
    let mut decomposition = crate::dwt2d::dwt2d_decompose(data, wavelet, mode)?;
    result.push(decomposition.clone());

    // Perform remaining levels on approximation coefficients
    for _level in 1..levels {
        decomposition = crate::dwt2d::dwt2d_decompose(&decomposition.approx, wavelet, mode)?;
        result.push(decomposition.clone());
    }

    // Reverse so index 0 is the deepest level
    result.reverse();

    Ok(result)
}

/// Helper function for integer ceiling division
fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}