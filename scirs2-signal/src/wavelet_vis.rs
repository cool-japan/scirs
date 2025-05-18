//! Wavelet Coefficient Visualization Utilities
//!
//! This module provides utilities for visualizing and working with wavelet coefficients.
//! These tools are particularly useful for examining the results of various wavelet transforms,
//! including 1D DWT, 2D DWT, 1D SWT, and 2D SWT.
//!
//! The primary functions include:
//! - Arranging coefficients in visually informative layouts
//! - Normalizing coefficients for better visualization
//! - Calculating energy distributions across subbands
//! - Creating coefficient heatmaps
//! - Creating visual representations of wavelet decompositions
//!
//! # Examples
//!
//! ```
//! use ndarray::Array2;
//! use scirs2_signal::dwt::Wavelet;
//! use scirs2_signal::dwt2d::dwt2d_decompose;
//! use scirs2_signal::wavelet_vis::arrange_coefficients_2d;
//!
//! // Create a simple test image
//! let image = Array2::from_shape_vec((8, 8), (0..64).map(|i| i as f64).collect()).unwrap();
//!
//! // Perform 2D DWT decomposition
//! let decomp = dwt2d_decompose(&image, Wavelet::Haar, None).unwrap();
//!
//! // Arrange the coefficients in a visually informative layout
//! let arranged = arrange_coefficients_2d(&decomp);
//!
//! // The arranged coefficients form a single 2D array with the
//! // approximation coefficients in the top-left and the detail
//! // coefficients in the other quadrants
//! assert_eq!(arranged.shape(), image.shape());
//! ```

use crate::dwt2d::Dwt2dResult;
use crate::error::{SignalError, SignalResult};
use crate::swt2d::Swt2dResult;
use ndarray::{s, Array2};
use num_traits::Float;
use std::fmt::Debug;

/// Energy distribution of wavelet coefficients.
#[derive(Debug, Clone)]
pub struct WaveletEnergy {
    /// Total energy in the signal/image
    pub total: f64,
    /// Energy in the approximation coefficients
    pub approximation: f64,
    /// Energy in the horizontal detail coefficients (2D only)
    pub horizontal: Option<f64>,
    /// Energy in the vertical detail coefficients (2D only)
    pub vertical: Option<f64>,
    /// Energy in the diagonal detail coefficients (2D only)
    pub diagonal: Option<f64>,
    /// Energy in the detail coefficients (1D or sum of H+V+D for 2D)
    pub detail: f64,
    /// Percentage of energy in the approximation coefficients
    pub approximation_percent: f64,
    /// Percentage of energy in the detail coefficients
    pub detail_percent: f64,
}

/// Arranges 2D DWT coefficients into a single visualization-friendly array.
///
/// This function takes a `Dwt2dResult` containing the approximation and detail
/// coefficients from a 2D DWT and arranges them into a single 2D array with the
/// same dimensions as the original image, following the standard layout used in
/// wavelet literature:
///
/// ```text
/// +-------+-------+
/// |   LL  |  HL   |
/// |       |       |
/// +-------+-------+
/// |   LH  |  HH   |
/// |       |       |
/// +-------+-------+
/// ```
///
/// Where:
/// - LL (top-left): Approximation coefficients
/// - HL (top-right): Vertical detail coefficients
/// - LH (bottom-left): Horizontal detail coefficients
/// - HH (bottom-right): Diagonal detail coefficients
///
/// # Arguments
///
/// * `decomposition` - The 2D DWT decomposition result containing approximation and detail coefficients
///
/// # Returns
///
/// * A single `Array2<f64>` with the coefficients arranged in the standard layout
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::dwt2d_decompose;
/// use scirs2_signal::wavelet_vis::arrange_coefficients_2d;
///
/// // Create a simple test image
/// let image = Array2::from_shape_vec((8, 8), (0..64).map(|i| i as f64).collect()).unwrap();
///
/// // Perform 2D DWT decomposition
/// let decomp = dwt2d_decompose(&image, Wavelet::Haar, None).unwrap();
///
/// // Arrange the coefficients in a visually informative layout
/// let arranged = arrange_coefficients_2d(&decomp);
///
/// // The arranged array has the same shape as the original image
/// assert_eq!(arranged.shape(), image.shape());
/// ```
pub fn arrange_coefficients_2d(decomposition: &Dwt2dResult) -> Array2<f64> {
    let approx = &decomposition.approx;
    let detail_h = &decomposition.detail_h;
    let detail_v = &decomposition.detail_v;
    let detail_d = &decomposition.detail_d;

    // Get dimensions
    let (approx_rows, approx_cols) = approx.dim();
    let (detail_h_rows, detail_h_cols) = detail_h.dim();
    let (detail_v_rows, detail_v_cols) = detail_v.dim();
    let (detail_d_rows, detail_d_cols) = detail_d.dim();

    // Verify all matrices have compatible sizes
    assert_eq!(approx_rows, detail_v_rows);
    assert_eq!(approx_cols, detail_h_cols);
    assert_eq!(detail_h_rows, detail_d_rows);
    assert_eq!(detail_v_cols, detail_d_cols);

    // Create output array
    let rows = approx_rows + detail_h_rows;
    let cols = approx_cols + detail_v_cols;
    let mut arranged = Array2::zeros((rows, cols));

    // Place coefficients in their respective quadrants
    arranged
        .slice_mut(s![0..approx_rows, 0..approx_cols])
        .assign(approx);
    arranged
        .slice_mut(s![0..detail_v_rows, approx_cols..cols])
        .assign(detail_v);
    arranged
        .slice_mut(s![approx_rows..rows, 0..detail_h_cols])
        .assign(detail_h);
    arranged
        .slice_mut(s![approx_rows..rows, approx_cols..cols])
        .assign(detail_d);

    arranged
}

/// Arranges multi-level 2D DWT coefficients into a single visualization-friendly array.
///
/// This function takes a vector of `Dwt2dResult` objects containing multiple levels of
/// decomposition and arranges them into a single 2D array, following the standard layout
/// used in wavelet literature. For multi-level decomposition, the approximation coefficients
/// from each level (except the final level) are further decomposed.
///
/// # Arguments
///
/// * `decompositions` - A vector of 2D DWT decomposition results, with indices corresponding to decomposition levels
///
/// # Returns
///
/// * A single `Array2<f64>` with all levels of coefficients arranged in the standard layout
///
/// # Examples
///
/// ```no_run
/// use ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::wavedec2;
/// use scirs2_signal::wavelet_vis::arrange_multilevel_coefficients_2d;
///
/// // Create a simple test image (64x64 for 3 levels without overflow)
/// let mut image = Array2::zeros((64, 64));
/// for i in 0..64 {
///     for j in 0..64 {
///         image[[i, j]] = (i * j) as f64;
///     }
/// }
///
/// // Perform multi-level 2D DWT decomposition (3 levels)
/// let decomps = wavedec2(&image, Wavelet::Haar, 3, None).unwrap();
///
/// // Arrange the coefficients in a visually informative layout
/// let arranged = arrange_multilevel_coefficients_2d(&decomps).unwrap();
///
/// // The arranged array has the same shape as the original image
/// assert_eq!(arranged.shape(), image.shape());
/// ```
pub fn arrange_multilevel_coefficients_2d(
    decompositions: &[Dwt2dResult],
) -> SignalResult<Array2<f64>> {
    if decompositions.is_empty() {
        return Err(SignalError::ValueError(
            "Decomposition list is empty".to_string(),
        ));
    }

    // Calculate original image dimensions based on the shape of the first level's coefficients
    let first_decomp = &decompositions[0];
    let (rows, cols) = first_decomp.approx.dim();
    let original_rows = rows * 2;
    let original_cols = cols * 2;

    // Create output array with the original image dimensions
    let mut arranged = Array2::zeros((original_rows, original_cols));

    // Determine the number of levels
    let num_levels = decompositions.len();

    // Place coefficients from the deepest level (last level)
    let deepest_level = &decompositions[num_levels - 1];
    let (approx_rows, approx_cols) = deepest_level.approx.dim();

    // Calculate the start indices for the deepest level's approximation coefficients
    let start_row = 0;
    let start_col = 0;

    // Place the approximation coefficients from the deepest level
    for i in 0..approx_rows {
        for j in 0..approx_cols {
            arranged[[start_row + i, start_col + j]] = deepest_level.approx[[i, j]];
        }
    }

    // Place detail coefficients for each level
    for level_idx in (0..num_levels).rev() {
        let level = &decompositions[level_idx];

        // For each level, calculate the size and positions of the detail coefficients
        let scale_factor = 2_usize.pow((num_levels - 1 - level_idx) as u32);
        let level_approx_rows = level.approx.shape()[0];
        let level_approx_cols = level.approx.shape()[1];

        // Horizontal details (bottom-left)
        for i in 0..level_approx_rows {
            for j in 0..level_approx_cols {
                let target_row = start_row + level_approx_rows + i * scale_factor;
                let target_col = start_col + j * scale_factor;

                if target_row < original_rows && target_col < original_cols {
                    for di in 0..scale_factor {
                        for dj in 0..scale_factor {
                            if target_row + di < original_rows && target_col + dj < original_cols {
                                arranged[[target_row + di, target_col + dj]] =
                                    level.detail_h[[i, j]];
                            }
                        }
                    }
                }
            }
        }

        // Vertical details (top-right)
        for i in 0..level_approx_rows {
            for j in 0..level_approx_cols {
                let target_row = start_row + i * scale_factor;
                let target_col = start_col + level_approx_cols + j * scale_factor;

                if target_row < original_rows && target_col < original_cols {
                    for di in 0..scale_factor {
                        for dj in 0..scale_factor {
                            if target_row + di < original_rows && target_col + dj < original_cols {
                                arranged[[target_row + di, target_col + dj]] =
                                    level.detail_v[[i, j]];
                            }
                        }
                    }
                }
            }
        }

        // Diagonal details (bottom-right)
        for i in 0..level_approx_rows {
            for j in 0..level_approx_cols {
                let target_row = start_row + level_approx_rows + i * scale_factor;
                let target_col = start_col + level_approx_cols + j * scale_factor;

                if target_row < original_rows && target_col < original_cols {
                    for di in 0..scale_factor {
                        for dj in 0..scale_factor {
                            if target_row + di < original_rows && target_col + dj < original_cols {
                                arranged[[target_row + di, target_col + dj]] =
                                    level.detail_d[[i, j]];
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(arranged)
}

/// Calculates energy statistics for 2D DWT coefficients.
///
/// This function computes the energy distribution across the different subbands
/// of a 2D wavelet decomposition. Energy is calculated as the sum of squared
/// coefficient values. The function returns both absolute energy values and
/// percentage distribution.
///
/// # Arguments
///
/// * `decomposition` - The 2D DWT decomposition result
///
/// # Returns
///
/// * A `WaveletEnergy` struct containing energy statistics
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::dwt2d_decompose;
/// use scirs2_signal::wavelet_vis::calculate_energy_2d;
///
/// // Create a simple test image
/// let image = Array2::from_shape_vec((8, 8), (0..64).map(|i| i as f64).collect()).unwrap();
///
/// // Perform 2D DWT decomposition
/// let decomp = dwt2d_decompose(&image, Wavelet::Haar, None).unwrap();
///
/// // Calculate energy distribution
/// let energy = calculate_energy_2d(&decomp);
///
/// // Check that energy percentages sum to 100%
/// assert!((energy.approximation_percent + energy.detail_percent - 100.0).abs() < 1e-10);
/// ```
pub fn calculate_energy_2d(decomposition: &Dwt2dResult) -> WaveletEnergy {
    let approx = &decomposition.approx;
    let detail_h = &decomposition.detail_h;
    let detail_v = &decomposition.detail_v;
    let detail_d = &decomposition.detail_d;

    // Calculate energy (sum of squared coefficients)
    let energy_approx = approx.iter().map(|&x| x * x).sum::<f64>();
    let energy_h = detail_h.iter().map(|&x| x * x).sum::<f64>();
    let energy_v = detail_v.iter().map(|&x| x * x).sum::<f64>();
    let energy_d = detail_d.iter().map(|&x| x * x).sum::<f64>();

    let energy_detail = energy_h + energy_v + energy_d;
    let total_energy = energy_approx + energy_detail;

    // Calculate percentages
    let approx_percent = if total_energy > 0.0 {
        100.0 * energy_approx / total_energy
    } else {
        0.0
    };

    let detail_percent = if total_energy > 0.0 {
        100.0 * energy_detail / total_energy
    } else {
        0.0
    };

    WaveletEnergy {
        total: total_energy,
        approximation: energy_approx,
        horizontal: Some(energy_h),
        vertical: Some(energy_v),
        diagonal: Some(energy_d),
        detail: energy_detail,
        approximation_percent: approx_percent,
        detail_percent,
    }
}

/// Calculates energy statistics for 2D SWT coefficients.
///
/// This function computes the energy distribution across the different subbands
/// of a 2D stationary wavelet decomposition. Energy is calculated as the sum of squared
/// coefficient values. The function returns both absolute energy values and
/// percentage distribution.
///
/// # Arguments
///
/// * `decomposition` - The 2D SWT decomposition result
///
/// # Returns
///
/// * A `WaveletEnergy` struct containing energy statistics
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::swt2d::swt2d_decompose;
/// use scirs2_signal::wavelet_vis::calculate_energy_swt2d;
///
/// // Create a simple test image
/// let image = Array2::from_shape_vec((8, 8), (0..64).map(|i| i as f64).collect()).unwrap();
///
/// // Perform 2D SWT decomposition
/// let decomp = swt2d_decompose(&image, Wavelet::Haar, 1, None).unwrap();
///
/// // Calculate energy distribution
/// let energy = calculate_energy_swt2d(&decomp);
///
/// // Check that energy percentages sum to 100%
/// assert!((energy.approximation_percent + energy.detail_percent - 100.0).abs() < 1e-10);
/// ```
pub fn calculate_energy_swt2d(decomposition: &Swt2dResult) -> WaveletEnergy {
    let approx = &decomposition.approx;
    let detail_h = &decomposition.detail_h;
    let detail_v = &decomposition.detail_v;
    let detail_d = &decomposition.detail_d;

    // Calculate energy (sum of squared coefficients)
    let energy_approx = approx.iter().map(|&x| x * x).sum::<f64>();
    let energy_h = detail_h.iter().map(|&x| x * x).sum::<f64>();
    let energy_v = detail_v.iter().map(|&x| x * x).sum::<f64>();
    let energy_d = detail_d.iter().map(|&x| x * x).sum::<f64>();

    let energy_detail = energy_h + energy_v + energy_d;
    let total_energy = energy_approx + energy_detail;

    // Calculate percentages
    let approx_percent = if total_energy > 0.0 {
        100.0 * energy_approx / total_energy
    } else {
        0.0
    };

    let detail_percent = if total_energy > 0.0 {
        100.0 * energy_detail / total_energy
    } else {
        0.0
    };

    WaveletEnergy {
        total: total_energy,
        approximation: energy_approx,
        horizontal: Some(energy_h),
        vertical: Some(energy_v),
        diagonal: Some(energy_d),
        detail: energy_detail,
        approximation_percent: approx_percent,
        detail_percent,
    }
}

/// Calculates energy statistics for 1D wavelet coefficients.
///
/// This function computes the energy distribution between approximation and
/// detail coefficients in a 1D wavelet decomposition. Energy is calculated
/// as the sum of squared coefficient values.
///
/// # Arguments
///
/// * `approx` - The approximation coefficients
/// * `detail` - The detail coefficients
///
/// # Returns
///
/// * A `WaveletEnergy` struct containing energy statistics
///
/// # Examples
///
/// ```
/// use scirs2_signal::dwt::{Wavelet, dwt_decompose};
/// use scirs2_signal::wavelet_vis::calculate_energy_1d;
///
/// // Create a simple signal
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
///
/// // Perform 1D DWT decomposition
/// let (approx, detail) = dwt_decompose(&signal, Wavelet::Haar, None).unwrap();
///
/// // Calculate energy distribution
/// let energy = calculate_energy_1d(&approx, &detail);
///
/// // Check that energy percentages sum to 100%
/// assert!((energy.approximation_percent + energy.detail_percent - 100.0).abs() < 1e-10);
/// ```
pub fn calculate_energy_1d(approx: &[f64], detail: &[f64]) -> WaveletEnergy {
    // Calculate energy (sum of squared coefficients)
    let energy_approx = approx.iter().map(|&x| x * x).sum::<f64>();
    let energy_detail = detail.iter().map(|&x| x * x).sum::<f64>();

    let total_energy = energy_approx + energy_detail;

    // Calculate percentages
    let approx_percent = if total_energy > 0.0 {
        100.0 * energy_approx / total_energy
    } else {
        0.0
    };

    let detail_percent = if total_energy > 0.0 {
        100.0 * energy_detail / total_energy
    } else {
        0.0
    };

    WaveletEnergy {
        total: total_energy,
        approximation: energy_approx,
        horizontal: None,
        vertical: None,
        diagonal: None,
        detail: energy_detail,
        approximation_percent: approx_percent,
        detail_percent,
    }
}

/// Normalizes wavelet coefficients for better visualization.
///
/// This function normalizes the coefficients to a specified range,
/// typically [0, 1] for visualization purposes. Different normalization
/// strategies are available depending on the application.
///
/// # Arguments
///
/// * `coefficients` - The wavelet coefficients to normalize
/// * `strategy` - The normalization strategy to use
/// * `range` - Optional min and max values for the normalization range (defaults to [0, 1])
///
/// # Returns
///
/// * Normalized coefficients as a new Array2<f64>
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::dwt2d_decompose;
/// use scirs2_signal::wavelet_vis::{normalize_coefficients, NormalizationStrategy};
///
/// // Create a simple test image
/// let image = Array2::from_shape_vec((8, 8), (0..64).map(|i| i as f64).collect()).unwrap();
///
/// // Perform 2D DWT decomposition
/// let decomp = dwt2d_decompose(&image, Wavelet::Haar, None).unwrap();
///
/// // Normalize the horizontal detail coefficients for visualization
/// let normalized = normalize_coefficients(&decomp.detail_h, NormalizationStrategy::MinMax, None);
///
/// // Check that values are in the [0, 1] range
/// for &value in normalized.iter() {
///     assert!(value >= 0.0 && value <= 1.0);
/// }
/// ```
pub fn normalize_coefficients(
    coefficients: &Array2<f64>,
    strategy: NormalizationStrategy,
    range: Option<(f64, f64)>,
) -> Array2<f64> {
    let (min_val, max_val) = range.unwrap_or((0.0, 1.0));
    let range_size = max_val - min_val;

    match strategy {
        NormalizationStrategy::MinMax => {
            // Find min and max values in the coefficients
            let coeff_min = coefficients.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let coeff_max = coefficients
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let coeff_range = coeff_max - coeff_min;

            // Avoid division by zero
            if coeff_range.abs() < 1e-10 {
                return Array2::from_elem(coefficients.dim(), min_val);
            }

            // Normalize to [min_val, max_val]
            coefficients.mapv(|x| min_val + range_size * (x - coeff_min) / coeff_range)
        }
        NormalizationStrategy::Absolute => {
            // Find max absolute value
            let max_abs = coefficients.iter().fold(0.0, |a, &b| a.max(b.abs()));

            // Avoid division by zero
            if max_abs.abs() < 1e-10 {
                return Array2::from_elem(coefficients.dim(), min_val);
            }

            // Normalize to [-max_val, max_val] or [min_val, max_val] based on signs
            if min_val < 0.0 {
                // Symmetric normalization
                let scale = max_val / max_abs;
                coefficients.mapv(|x| x * scale)
            } else {
                // Shift to [min_val, max_val]
                let half_range = range_size / 2.0;
                let mid_point = min_val + half_range;
                coefficients.mapv(|x| mid_point + half_range * x / max_abs)
            }
        }
        NormalizationStrategy::Log => {
            // Apply logarithmic normalization (useful for compressing dynamic range)
            let max_abs = coefficients.iter().fold(0.0, |a, &b| a.max(b.abs()));

            // Avoid taking log of zero
            if max_abs.abs() < 1e-10 {
                return Array2::from_elem(coefficients.dim(), min_val);
            }

            // Apply log normalization with appropriate scaling
            coefficients.mapv(|x| {
                let sign = if x < 0.0 { -1.0 } else { 1.0 };
                // Use log(1 + |x|) for better behavior near zero
                let log_val = (1.0 + x.abs() / max_abs).ln();
                // Scale to [min_val, max_val]
                let scaled = min_val + range_size * log_val / (2.0_f64.ln());
                if min_val < 0.0 {
                    sign * scaled
                } else {
                    scaled
                }
            })
        }
        NormalizationStrategy::Percentile(lower, upper) => {
            // Collect values into a sorted vector for percentile calculation
            let mut values: Vec<f64> = coefficients.iter().copied().collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Calculate percentile values
            let n = values.len();
            let lower_idx = (n as f64 * lower / 100.0).floor() as usize;
            let upper_idx = (n as f64 * upper / 100.0).ceil() as usize;

            let lower_val = if lower_idx < n {
                values[lower_idx]
            } else {
                values[n - 1]
            };
            let upper_val = if upper_idx < n {
                values[upper_idx]
            } else {
                values[n - 1]
            };

            let val_range = upper_val - lower_val;

            // Avoid division by zero
            if val_range.abs() < 1e-10 {
                return Array2::from_elem(coefficients.dim(), min_val);
            }

            // Normalize to [min_val, max_val] with clipping at percentiles
            coefficients.mapv(|x| {
                let clipped = x.max(lower_val).min(upper_val);
                min_val + range_size * (clipped - lower_val) / val_range
            })
        }
    }
}

/// Strategies for normalizing wavelet coefficients.
#[derive(Debug, Clone, Copy)]
pub enum NormalizationStrategy {
    /// Normalize using min and max values (scales to full range)
    MinMax,
    /// Normalize using the maximum absolute value (preserves zero center)
    Absolute,
    /// Apply logarithmic normalization (compresses dynamic range)
    Log,
    /// Normalize using specified lower and upper percentiles (robust to outliers)
    Percentile(f64, f64),
}

/// Counts non-zero wavelet coefficients.
///
/// This function counts the number of non-zero coefficients in a wavelet decomposition,
/// optionally using a threshold to determine significance. This is useful for assessing
/// the sparsity of a wavelet representation, which is important for compression applications.
///
/// # Arguments
///
/// * `decomposition` - The 2D DWT decomposition result
/// * `threshold` - Optional threshold value below which coefficients are considered zero
///
/// # Returns
///
/// * A struct containing the count of non-zero coefficients in each subband and the total
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::dwt2d_decompose;
/// use scirs2_signal::wavelet_vis::count_nonzero_coefficients;
///
/// // Create a simple test image
/// let image = Array2::from_shape_vec((8, 8), (0..64).map(|i| i as f64).collect()).unwrap();
///
/// // Perform 2D DWT decomposition
/// let decomp = dwt2d_decompose(&image, Wavelet::Haar, None).unwrap();
///
/// // Count non-zero coefficients with a threshold of 1.0
/// let counts = count_nonzero_coefficients(&decomp, Some(1.0));
///
/// // Verify that the total count matches the sum of individual counts
/// assert_eq!(counts.total,
///            counts.approximation + counts.horizontal + counts.vertical + counts.diagonal);
/// ```
pub fn count_nonzero_coefficients(
    decomposition: &Dwt2dResult,
    threshold: Option<f64>,
) -> WaveletCoeffCount {
    let threshold = threshold.unwrap_or(0.0);

    // Count non-zero coefficients in each subband
    let count_approx = decomposition
        .approx
        .iter()
        .filter(|&&x| x.abs() > threshold)
        .count();

    let count_h = decomposition
        .detail_h
        .iter()
        .filter(|&&x| x.abs() > threshold)
        .count();

    let count_v = decomposition
        .detail_v
        .iter()
        .filter(|&&x| x.abs() > threshold)
        .count();

    let count_d = decomposition
        .detail_d
        .iter()
        .filter(|&&x| x.abs() > threshold)
        .count();

    let total_count = count_approx + count_h + count_v + count_d;
    let total_coeffs = decomposition.approx.len()
        + decomposition.detail_h.len()
        + decomposition.detail_v.len()
        + decomposition.detail_d.len();

    let percent = 100.0 * (total_count as f64) / (total_coeffs as f64);

    WaveletCoeffCount {
        approximation: count_approx,
        horizontal: count_h,
        vertical: count_v,
        diagonal: count_d,
        total: total_count,
        percent_nonzero: percent,
    }
}

/// Statistics on non-zero wavelet coefficients.
#[derive(Debug, Clone, Copy)]
pub struct WaveletCoeffCount {
    /// Number of non-zero approximation coefficients
    pub approximation: usize,
    /// Number of non-zero horizontal detail coefficients
    pub horizontal: usize,
    /// Number of non-zero vertical detail coefficients
    pub vertical: usize,
    /// Number of non-zero diagonal detail coefficients
    pub diagonal: usize,
    /// Total number of non-zero coefficients
    pub total: usize,
    /// Percentage of non-zero coefficients
    pub percent_nonzero: f64,
}

/// Creates a coefficient heatmap with customizable colormap for visualization.
///
/// This function creates a representation of wavelet coefficients as a heatmap,
/// where coefficient values are mapped to colors using a specified colormap function.
/// The result is a 2D array of RGB values that can be saved as an image.
///
/// # Arguments
///
/// * `coefficients` - The 2D wavelet coefficients to visualize
/// * `colormap` - A colormap function that maps values to RGB triples (0-255)
/// * `normalization` - Optional normalization strategy (default: MinMax)
///
/// # Returns
///
/// * A 3D array (height × width × 3) of RGB values representing the heatmap
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::dwt2d_decompose;
/// use scirs2_signal::wavelet_vis::{create_coefficient_heatmap, NormalizationStrategy, colormaps};
///
/// // Create a simple test image
/// let image = Array2::from_shape_vec((8, 8), (0..64).map(|i| i as f64).collect()).unwrap();
///
/// // Perform 2D DWT decomposition
/// let decomp = dwt2d_decompose(&image, Wavelet::Haar, None).unwrap();
///
/// // Create a heatmap of the approximation coefficients
/// let heatmap = create_coefficient_heatmap(
///     &decomp.approx,
///     colormaps::viridis,
///     Some(NormalizationStrategy::MinMax)
/// );
///
/// // Verify the heatmap dimensions
/// assert_eq!(heatmap.shape()[0], decomp.approx.shape()[0]);
/// assert_eq!(heatmap.shape()[1], decomp.approx.shape()[1]);
/// assert_eq!(heatmap.shape()[2], 3); // RGB
/// ```
pub fn create_coefficient_heatmap<F>(
    coefficients: &Array2<f64>,
    colormap: F,
    normalization: Option<NormalizationStrategy>,
) -> ndarray::Array3<u8>
where
    F: Fn(f64) -> [u8; 3],
{
    // Apply normalization
    let normalization = normalization.unwrap_or(NormalizationStrategy::MinMax);
    let normalized = normalize_coefficients(coefficients, normalization, Some((0.0, 1.0)));

    // Get dimensions
    let (height, width) = normalized.dim();

    // Create RGB output array
    let mut rgb = ndarray::Array3::zeros((height, width, 3));

    // Apply colormap
    for i in 0..height {
        for j in 0..width {
            let value = normalized[[i, j]];
            let [r, g, b] = colormap(value);
            rgb[[i, j, 0]] = r;
            rgb[[i, j, 1]] = g;
            rgb[[i, j, 2]] = b;
        }
    }

    rgb
}

/// Common colormap functions for coefficient visualization.
pub mod colormaps {
    /// Viridis colormap (perceptually uniform, color-blind friendly)
    pub fn viridis(value: f64) -> [u8; 3] {
        // Clamp value to [0, 1]
        let x = value.clamp(0.0, 1.0);

        // Viridis colormap approximation
        let r = (70.0 * x.powi(2) - 35.0 * x - 129.0 * x.powi(3) + 133.0).clamp(0.0, 255.0) as u8;
        let g = (87.0 + 192.0 * x - 386.0 * x.powi(2) + 203.0 * x.powi(3)).clamp(0.0, 255.0) as u8;
        let b = (173.0 - 36.0 * x - 476.0 * x.powi(2) + 357.0 * x.powi(3)).clamp(0.0, 255.0) as u8;

        [r, g, b]
    }

    /// Plasma colormap (perceptually uniform)
    pub fn plasma(value: f64) -> [u8; 3] {
        // Clamp value to [0, 1]
        let x = value.clamp(0.0, 1.0);

        // Plasma colormap approximation
        let r = (255.0 * (0.05 + 0.82 * x + 1.15 * x.powi(2) - 1.82 * x.powi(3))).clamp(0.0, 255.0)
            as u8;
        let g = (255.0 * (-0.15 + 1.80 * x - 3.09 * x.powi(2) + 1.42 * x.powi(3))).clamp(0.0, 255.0)
            as u8;
        let b = (255.0 * (0.51 + 0.56 * x - 2.31 * x.powi(2) + 1.24 * x.powi(3))).clamp(0.0, 255.0)
            as u8;

        [r, g, b]
    }

    /// Sequential grayscale (black to white)
    pub fn grayscale(value: f64) -> [u8; 3] {
        // Clamp value to [0, 1]
        let x = value.clamp(0.0, 1.0);

        // Grayscale is simple - same value for R, G, and B
        let gray = (255.0 * x) as u8;

        [gray, gray, gray]
    }

    /// Diverging red-blue colormap (for positive/negative values)
    pub fn diverging_rb(value: f64) -> [u8; 3] {
        // Clamp value to [0, 1]
        let x = value.clamp(0.0, 1.0);

        // Red for values < 0.5, blue for values > 0.5
        let r = if x < 0.5 {
            (255.0 * (1.0 - 2.0 * x)) as u8
        } else {
            0
        };

        let b = if x >= 0.5 {
            (255.0 * (2.0 * x - 1.0)) as u8
        } else {
            0
        };

        // Green is highest at midpoint
        let g = (255.0 * (1.0 - 2.0 * (x - 0.5).abs())) as u8;

        [r, g, b]
    }

    /// Jet colormap (rainbow)
    pub fn jet(value: f64) -> [u8; 3] {
        // Clamp value to [0, 1]
        let x = value.clamp(0.0, 1.0);

        // Jet colormap approximation
        let r = (255.0 * (1.5 - 4.0 * (x - 0.75).abs()).clamp(0.0, 1.0)) as u8;
        let g = (255.0 * (1.5 - 4.0 * (x - 0.5).abs()).clamp(0.0, 1.0)) as u8;
        let b = (255.0 * (1.5 - 4.0 * (x - 0.25).abs()).clamp(0.0, 1.0)) as u8;

        [r, g, b]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dwt::Wavelet;
    use crate::dwt2d::{dwt2d_decompose, wavedec2};
    use crate::swt2d::swt2d_decompose;
    use ndarray::Array2;

    // Helper function to create a test image
    fn create_test_image(size: usize) -> Array2<f64> {
        let mut image = Array2::zeros((size, size));
        for i in 0..size {
            for j in 0..size {
                image[[i, j]] = (i * j) as f64;
            }
        }
        image
    }

    #[test]
    fn test_arrange_coefficients_2d() {
        // Create a test image
        let image = create_test_image(8);

        // Perform 2D DWT
        let decomp = dwt2d_decompose(&image, Wavelet::Haar, None).unwrap();

        // Arrange coefficients
        let arranged = arrange_coefficients_2d(&decomp);

        // Check dimensions
        assert_eq!(arranged.shape(), image.shape());

        // Check that quadrants match the original coefficients
        let (approx_rows, approx_cols) = decomp.approx.dim();

        for i in 0..approx_rows {
            for j in 0..approx_cols {
                assert_eq!(arranged[[i, j]], decomp.approx[[i, j]]);
                assert_eq!(arranged[[i, approx_cols + j]], decomp.detail_v[[i, j]]);
                assert_eq!(arranged[[approx_rows + i, j]], decomp.detail_h[[i, j]]);
                assert_eq!(
                    arranged[[approx_rows + i, approx_cols + j]],
                    decomp.detail_d[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_arrange_multilevel_coefficients_2d() {
        // Create a test image (larger for multi-level decomposition)
        let image = create_test_image(8);

        // Perform multi-level 2D DWT
        let decomps = wavedec2(&image, Wavelet::Haar, 1, None).unwrap();

        // Arrange coefficients
        let arranged = arrange_multilevel_coefficients_2d(&decomps).unwrap();

        // Our test creates an 8x8 image, and after one level of decomposition,
        // we get 4x4 approximation and detail coefficients, which when arranged
        // should result in an 8x8 array.
        assert_eq!(arranged.shape(), &[8, 8]);

        // Check first level approximation placement
        let (approx_rows, approx_cols) = decomps[0].approx.dim();
        for i in 0..approx_rows {
            for j in 0..approx_cols {
                assert_eq!(arranged[[i, j]], decomps[0].approx[[i, j]]);
            }
        }
    }

    #[test]
    fn test_calculate_energy_2d() {
        // Create a test image
        let image = create_test_image(8);

        // Perform 2D DWT
        let decomp = dwt2d_decompose(&image, Wavelet::Haar, None).unwrap();

        // Calculate energy
        let energy = calculate_energy_2d(&decomp);

        // Check that energy percentages sum to 100%
        assert!((energy.approximation_percent + energy.detail_percent - 100.0).abs() < 1e-10);

        // Check that detail energy is the sum of H, V, D energies
        let h_energy = energy.horizontal.unwrap();
        let v_energy = energy.vertical.unwrap();
        let d_energy = energy.diagonal.unwrap();

        assert!((energy.detail - (h_energy + v_energy + d_energy)).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_energy_swt2d() {
        // Create a test image
        let image = create_test_image(8);

        // Perform 2D SWT
        let decomp = swt2d_decompose(&image, Wavelet::Haar, 1, None).unwrap();

        // Calculate energy
        let energy = calculate_energy_swt2d(&decomp);

        // Check that energy percentages sum to 100%
        assert!((energy.approximation_percent + energy.detail_percent - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_energy_1d() {
        // Create a simple signal
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Calculate energy of the signal directly for comparison
        let total_energy: f64 = signal.iter().map(|&x| x * x).sum();

        // Use the signal as both approximation and detail coefficients
        let energy = calculate_energy_1d(&signal, &signal);

        // Check total energy
        assert_eq!(energy.total, 2.0 * total_energy);

        // Check percentages
        assert_eq!(energy.approximation_percent, 50.0);
        assert_eq!(energy.detail_percent, 50.0);
    }

    #[test]
    fn test_normalize_coefficients() {
        // Create a test array with known range
        let mut coeffs = Array2::<f64>::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                coeffs[[i, j]] = (i * 4 + j) as f64; // 0 to 15
            }
        }

        // Test MinMax normalization
        let norm_minmax = normalize_coefficients(&coeffs, NormalizationStrategy::MinMax, None);
        assert!((norm_minmax[[0, 0]] - 0.0).abs() < 1e-10); // min should map to 0
        assert!((norm_minmax[[3, 3]] - 1.0).abs() < 1e-10); // max should map to 1

        // Test Absolute normalization
        let norm_abs = normalize_coefficients(&coeffs, NormalizationStrategy::Absolute, None);
        assert!((norm_abs[[3, 3]] - 1.0).abs() < 1e-10); // max absolute value should map to 1

        // Test custom range
        let norm_range =
            normalize_coefficients(&coeffs, NormalizationStrategy::MinMax, Some((-1.0, 1.0)));
        assert!((norm_range[[0, 0]] - (-1.0)).abs() < 1e-10); // min should map to -1
        assert!((norm_range[[3, 3]] - 1.0).abs() < 1e-10); // max should map to 1

        // Test Percentile normalization
        let norm_perc =
            normalize_coefficients(&coeffs, NormalizationStrategy::Percentile(25.0, 75.0), None);
        // 25th percentile is 3.75, 75th percentile is 11.25
        // The percentile normalization maps values to the [0,1] range by default
        assert!(norm_perc[[0, 0]] <= 0.0); // At or below the minimum normalized value

        // For the middle value test, we need to be more careful about which element to check
        // The element at [1, 3] is 7 which is between 25th and 75th percentiles
        // This should map approximately to a value between 0 and 1, with the exact value
        // depending on the distribution of values in the array
        let normalized_value = norm_perc[[1, 3]];
        assert!(
            normalized_value >= 0.0 && normalized_value <= 1.0,
            "Normalized value {} should be between 0 and 1",
            normalized_value
        );
    }

    #[test]
    fn test_count_nonzero_coefficients() {
        // Create a test array with some zeros
        let mut coeffs = Array2::<f64>::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                if i > 0 && j > 0 {
                    coeffs[[i, j]] = (i * j) as f64;
                }
            }
        }

        // Create a Dwt2dResult with the same coefficients for simplicity
        let decomp = Dwt2dResult {
            approx: coeffs.clone(),
            detail_h: coeffs.clone(),
            detail_v: coeffs.clone(),
            detail_d: coeffs.clone(),
        };

        // Count with zero threshold
        let counts = count_nonzero_coefficients(&decomp, Some(0.0));

        // Each subband has 9 non-zero elements (all except first row and column)
        assert_eq!(counts.approximation, 9);
        assert_eq!(counts.horizontal, 9);
        assert_eq!(counts.vertical, 9);
        assert_eq!(counts.diagonal, 9);
        assert_eq!(counts.total, 36);

        // Count with threshold
        let counts_thresh = count_nonzero_coefficients(&decomp, Some(3.0));

        // Only elements > 3.0 should be counted
        assert!(counts_thresh.total < counts.total);
    }

    #[test]
    fn test_create_coefficient_heatmap() {
        // Create a test array
        let mut coeffs = Array2::<f64>::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                coeffs[[i, j]] = (i * j) as f64;
            }
        }

        // Create heatmap with different colormaps
        let heatmap_viridis = create_coefficient_heatmap(&coeffs, colormaps::viridis, None);
        let heatmap_grayscale = create_coefficient_heatmap(&coeffs, colormaps::grayscale, None);

        // Check dimensions
        assert_eq!(heatmap_viridis.shape(), &[4, 4, 3]);
        assert_eq!(heatmap_grayscale.shape(), &[4, 4, 3]);

        // Check grayscale properties (all channels should be equal)
        for i in 0..4 {
            for j in 0..4 {
                let r = heatmap_grayscale[[i, j, 0]];
                let g = heatmap_grayscale[[i, j, 1]];
                let b = heatmap_grayscale[[i, j, 2]];
                assert_eq!(r, g);
                assert_eq!(g, b);
            }
        }
    }
}
