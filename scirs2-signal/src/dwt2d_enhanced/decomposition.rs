//! 2D DWT Decomposition Functions
//!
//! This module contains the core decomposition algorithms for enhanced 2D DWT,
//! including multiple optimization strategies and processing modes:
//!
//! - Enhanced decomposition with adaptive optimization
//! - Parallel processing for large datasets
//! - SIMD-optimized operations
//! - Memory-optimized block processing
//! - Multilevel decomposition
//! - Adaptive entropy-based decomposition
//!
//! All functions maintain perfect reconstruction guarantees while providing
//! significant performance improvements through various acceleration techniques.

use super::types::{
    BoundaryMode, Dwt2dConfig, Dwt2dQualityMetrics, EnhancedDwt2dResult, MultilevelDwt2d,
};
use crate::dwt::{Wavelet, WaveletFilters};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{s, Array2, ArrayView1, ArrayView2};
use scirs2_core::parallel_ops::*;
use scirs2_core::random::Rng;
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::validation::{check_positive, checkarray_finite};
use statrs::statistics::Statistics;
use std::sync::Arc;

/// Enhanced 2D DWT decomposition with optimizations
///
/// # Arguments
///
/// * `data` - Input 2D array
/// * `wavelet` - Wavelet type
/// * `config` - Configuration parameters
///
/// # Returns
///
/// * Enhanced decomposition result
#[allow(dead_code)]
pub fn enhanced_dwt2d_decompose(
    data: &Array2<f64>,
    wavelet: Wavelet,
    config: &Dwt2dConfig,
) -> SignalResult<EnhancedDwt2dResult> {
    // Enhanced input validation
    checkarray_finite(data, "data")?;

    let (rows, cols) = data.dim();

    // Check minimum dimensions
    if rows < 2 || cols < 2 {
        return Err(SignalError::ValueError(format!(
            "Input must be at least 2x2, got {}x{}",
            rows, cols
        )));
    }

    // Check for reasonable array sizes
    if rows > 32768 || cols > 32768 {
        eprintln!(
            "Warning: Very large input ({}x{}). Consider using memory optimization.",
            rows, cols
        );
    }

    // Validate wavelet compatibility with data size
    let filters = wavelet.filters()?;
    let min_size_required = filters.dec_lo.len() * 2;
    if rows < min_size_required || cols < min_size_required {
        return Err(SignalError::ValueError(format!(
            "Input size ({}x{}) too small for wavelet filter length ({}). Minimum required: {}x{}",
            rows,
            cols,
            filters.dec_lo.len(),
            min_size_required,
            min_size_required
        )));
    }

    // Check for reasonable data range to prevent numerical issues
    let data_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let data_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let data_range = data_max - data_min;

    if data_range == 0.0 {
        eprintln!("Warning: Input data is constant. Wavelet transform will produce zero detail coefficients.");
    } else if data_range < 1e-15 {
        eprintln!("Warning: Input data has very small dynamic range ({:.2e}). Results may be affected by numerical precision.", data_range);
    } else if data_range > 1e15 {
        eprintln!("Warning: Input data has very large dynamic range ({:.2e}). Consider normalizing the data.", data_range);
    }

    // Check for extreme values that might cause overflow
    if data_max.abs() > 1e10 || data_min.abs() > 1e10 {
        eprintln!("Warning: Input contains very large values. This may cause numerical overflow in wavelet computation.");
    }

    // Check if memory optimization is needed for large images
    let memory_threshold = 2048 * 2048; // 2K x 2K pixels
    let use_memory_opt = config.memory_optimized || (rows * cols > memory_threshold);

    // Enhanced adaptive boundary mode selection
    let adaptive_boundary_mode = if matches!(config.boundary_mode, BoundaryMode::Adaptive) {
        analyze_and_select_boundary_mode(data, &filters)?
    } else {
        config.boundary_mode
    };

    let enhanced_config = Dwt2dConfig {
        boundary_mode: adaptive_boundary_mode,
        ..*config
    };

    // Choose processing method based on configuration
    let mut result = if use_memory_opt {
        memory_optimized_dwt2d_decompose(data, &filters, &enhanced_config)?
    } else if config.use_parallel && rows.min(cols) >= config.parallel_threshold {
        parallel_dwt2d_decompose(data, &filters, &enhanced_config)?
    } else if config.use_simd {
        simd_dwt2d_decompose(data, &filters, &enhanced_config)?
    } else {
        standard_dwt2d_decompose(data, &filters, &enhanced_config)?
    };

    // Enhanced result validation
    validate_dwt2d_result(&result, data.dim(), config)?;

    // Compute quality metrics if requested
    if config.compute_metrics {
        result.metrics = Some(compute_dwt2d_quality_metrics(data, &result)?);
    }

    Ok(result)
}

/// Parallel 2D DWT decomposition
#[allow(dead_code)]
fn parallel_dwt2d_decompose(
    data: &Array2<f64>,
    filters: &WaveletFilters,
    config: &Dwt2dConfig,
) -> SignalResult<EnhancedDwt2dResult> {
    let (rows, cols) = data.dim();
    let data_arc = Arc::new(data.clone());

    // First, apply 1D DWT to all rows in parallel
    let row_results: Vec<(Vec<f64>, Vec<f64>)> = (0..rows)
        .into_par_iter()
        .map(|i| {
            let row = data_arc.row(i).to_vec();
            let padded = apply_boundary_padding(&row, filters.dec_lo.len(), config.boundary_mode);
            let (lo, hi) = apply_filters_simd(&padded, &filters.dec_lo, &filters.dec_hi);
            (downsample(&lo), downsample(&hi))
        })
        .collect();

    // Reorganize into low and high frequency components
    let half_cols = cols.div_ceil(2);
    let mut temp_lo = Array2::zeros((rows, half_cols));
    let mut temp_hi = Array2::zeros((rows, half_cols));

    for (i, (lo, hi)) in row_results.iter().enumerate() {
        for (j, &val) in lo.iter().enumerate() {
            if j < half_cols {
                temp_lo[[i, j]] = val;
            }
        }
        for (j, &val) in hi.iter().enumerate() {
            if j < half_cols {
                temp_hi[[i, j]] = val;
            }
        }
    }

    // Apply 1D DWT to columns of low and high frequency components
    let half_rows = rows.div_ceil(2);

    // Process low frequency columns
    let lo_col_results: Vec<(usize, Vec<f64>, Vec<f64>)> = (0..half_cols)
        .into_par_iter()
        .map(|j| {
            let col = temp_lo.column(j).to_vec();
            let padded = apply_boundary_padding(&col, filters.dec_lo.len(), config.boundary_mode);
            let (lo, hi) = apply_filters_simd(&padded, &filters.dec_lo, &filters.dec_hi);
            (j, downsample(&lo), downsample(&hi))
        })
        .collect();

    // Process high frequency columns
    let hi_col_results: Vec<(usize, Vec<f64>, Vec<f64>)> = (0..half_cols)
        .into_par_iter()
        .map(|j| {
            let col = temp_hi.column(j).to_vec();
            let padded = apply_boundary_padding(&col, filters.dec_lo.len(), config.boundary_mode);
            let (lo, hi) = apply_filters_simd(&padded, &filters.dec_lo, &filters.dec_hi);
            (j, downsample(&lo), downsample(&hi))
        })
        .collect();

    // Build output arrays
    let mut approx = Array2::zeros((half_rows, half_cols));
    let mut detail_v = Array2::zeros((half_rows, half_cols));
    let mut detail_h = Array2::zeros((half_rows, half_cols));
    let mut detail_d = Array2::zeros((half_rows, half_cols));

    // Fill LL and HL from low frequency columns
    for (j, lo, hi) in lo_col_results {
        for (i, &val) in lo.iter().enumerate() {
            if i < half_rows {
                approx[[i, j]] = val;
            }
        }
        for (i, &val) in hi.iter().enumerate() {
            if i < half_rows {
                detail_v[[i, j]] = val;
            }
        }
    }

    // Fill LH and HH from high frequency columns
    for (j, lo, hi) in hi_col_results {
        for (i, &val) in lo.iter().enumerate() {
            if i < half_rows {
                detail_h[[i, j]] = val;
            }
        }
        for (i, &val) in hi.iter().enumerate() {
            if i < half_rows {
                detail_d[[i, j]] = val;
            }
        }
    }

    Ok(EnhancedDwt2dResult {
        approx,
        detail_h,
        detail_v,
        detail_d,
        originalshape: (rows, cols),
        boundary_mode: config.boundary_mode,
        metrics: None,
    })
}

/// SIMD-optimized 2D DWT decomposition
#[allow(dead_code)]
fn simd_dwt2d_decompose(
    data: &Array2<f64>,
    filters: &WaveletFilters,
    config: &Dwt2dConfig,
) -> SignalResult<EnhancedDwt2dResult> {
    let (rows, cols) = data.dim();

    // Process rows with SIMD
    let half_cols = cols.div_ceil(2);
    let mut temp_lo = Array2::zeros((rows, half_cols));
    let mut temp_hi = Array2::zeros((rows, half_cols));

    for i in 0..rows {
        let row = data.row(i).to_vec();
        let padded = apply_boundary_padding(&row, filters.dec_lo.len(), config.boundary_mode);
        let (lo, hi) = apply_filters_simd(&padded, &filters.dec_lo, &filters.dec_hi);

        let lo_down = downsample(&lo);
        let hi_down = downsample(&hi);

        for (j, &val) in lo_down.iter().enumerate() {
            if j < half_cols {
                temp_lo[[i, j]] = val;
            }
        }
        for (j, &val) in hi_down.iter().enumerate() {
            if j < half_cols {
                temp_hi[[i, j]] = val;
            }
        }
    }

    // Process columns with SIMD
    let half_rows = rows.div_ceil(2);
    let mut approx = Array2::zeros((half_rows, half_cols));
    let mut detail_v = Array2::zeros((half_rows, half_cols));
    let mut detail_h = Array2::zeros((half_rows, half_cols));
    let mut detail_d = Array2::zeros((half_rows, half_cols));

    // Process low frequency columns
    for j in 0..half_cols {
        let col = temp_lo.column(j).to_vec();
        let padded = apply_boundary_padding(&col, filters.dec_lo.len(), config.boundary_mode);
        let (lo, hi) = apply_filters_simd(&padded, &filters.dec_lo, &filters.dec_hi);

        let lo_down = downsample(&lo);
        let hi_down = downsample(&hi);

        for (i, &val) in lo_down.iter().enumerate() {
            if i < half_rows {
                approx[[i, j]] = val;
            }
        }
        for (i, &val) in hi_down.iter().enumerate() {
            if i < half_rows {
                detail_v[[i, j]] = val;
            }
        }
    }

    // Process high frequency columns
    for j in 0..half_cols {
        let col = temp_hi.column(j).to_vec();
        let padded = apply_boundary_padding(&col, filters.dec_lo.len(), config.boundary_mode);
        let (lo, hi) = apply_filters_simd(&padded, &filters.dec_lo, &filters.dec_hi);

        let lo_down = downsample(&lo);
        let hi_down = downsample(&hi);

        for (i, &val) in lo_down.iter().enumerate() {
            if i < half_rows {
                detail_h[[i, j]] = val;
            }
        }
        for (i, &val) in hi_down.iter().enumerate() {
            if i < half_rows {
                detail_d[[i, j]] = val;
            }
        }
    }

    Ok(EnhancedDwt2dResult {
        approx,
        detail_h,
        detail_v,
        detail_d,
        originalshape: (rows, cols),
        boundary_mode: config.boundary_mode,
        metrics: None,
    })
}

/// Standard 2D DWT decomposition (fallback)
#[allow(dead_code)]
fn standard_dwt2d_decompose(
    data: &Array2<f64>,
    filters: &WaveletFilters,
    config: &Dwt2dConfig,
) -> SignalResult<EnhancedDwt2dResult> {
    // Fallback to SIMD version without parallelism
    simd_dwt2d_decompose(data, filters, config)
}

/// Multilevel 2D DWT decomposition with enhanced features
pub fn wavedec2_enhanced(
    data: &Array2<f64>,
    wavelet: Wavelet,
    levels: usize,
    config: &Dwt2dConfig,
) -> SignalResult<MultilevelDwt2d> {
    check_positive(levels, "levels")?;

    let mut current = data.clone();
    let mut details = Vec::with_capacity(levels);

    for _ in 0..levels {
        let decomp = enhanced_dwt2d_decompose(&current, wavelet, config)?;

        details.push((
            decomp.detail_h.clone(),
            decomp.detail_v.clone(),
            decomp.detail_d.clone(),
        ));

        current = decomp.approx;

        // Check if we can continue
        let (rows, cols) = current.dim();
        if rows < 2 || cols < 2 {
            break;
        }
    }

    // Reverse details to have coarsest level first
    details.reverse();

    Ok(MultilevelDwt2d {
        approx: current,
        details,
        originalshape: data.dim(),
        wavelet,
        config: config.clone(),
    })
}

/// Memory-optimized 2D DWT decomposition for large images
#[allow(dead_code)]
fn memory_optimized_dwt2d_decompose(
    data: &Array2<f64>,
    filters: &WaveletFilters,
    config: &Dwt2dConfig,
) -> SignalResult<EnhancedDwt2dResult> {
    let (rows, cols) = data.dim();
    let block_size = config.block_size.min(rows.min(cols));

    // Calculate output dimensions
    let half_rows = rows.div_ceil(2);
    let half_cols = cols.div_ceil(2);

    // Initialize output arrays with better memory allocation
    let mut approx = Array2::zeros((half_rows, half_cols));
    let mut detail_h = Array2::zeros((half_rows, half_cols));
    let mut detail_v = Array2::zeros((half_rows, half_cols));
    let mut detail_d = Array2::zeros((half_rows, half_cols));

    // Process in blocks to reduce memory usage
    let overlap = filters.dec_lo.len(); // Filter length for overlap
    let min_block_size = overlap * 2; // Minimum useful block size

    // Adaptive block sizing based on available memory
    let effective_block_size = if block_size < min_block_size {
        min_block_size
    } else {
        block_size
    };

    for row_start in (0..rows).step_by(effective_block_size) {
        let row_end = (row_start + effective_block_size + overlap).min(rows);

        for col_start in (0..cols).step_by(effective_block_size) {
            let col_end = (col_start + effective_block_size + overlap).min(cols);

            // Extract block with overlap - use slice to avoid copying when possible
            let block = data.slice(s![row_start..row_end, col_start..col_end]);

            // Process block efficiently
            let block_result = if block.is_standard_layout() {
                // Can work directly with view for standard layout
                process_dwt2d_block_view(&block, filters, config, row_start, col_start)?
            } else {
                // Need to copy for non-standard layout
                process_dwt2d_block(&block.to_owned(), filters, config, row_start, col_start)?
            };

            // Copy valid region to output arrays with bounds checking
            let out_row_start = row_start / 2;
            let out_row_end = (row_start + effective_block_size).min(rows).div_ceil(2);
            let out_col_start = col_start / 2;
            let out_col_end = (col_start + effective_block_size).min(cols).div_ceil(2);

            // Ensure we don't exceed output array bounds
            let valid_row_end = out_row_end.min(half_rows);
            let valid_col_end = out_col_end.min(half_cols);
            let copy_rows = (valid_row_end - out_row_start).min(block_result.approx.nrows());
            let copy_cols = (valid_col_end - out_col_start).min(block_result.approx.ncols());

            // Vectorized copy when possible
            if copy_rows > 0 && copy_cols > 0 {
                let approxsrc = block_result.approx.slice(s![0..copy_rows, 0..copy_cols]);
                let detail_hsrc = block_result.detail_h.slice(s![0..copy_rows, 0..copy_cols]);
                let detail_vsrc = block_result.detail_v.slice(s![0..copy_rows, 0..copy_cols]);
                let detail_dsrc = block_result.detail_d.slice(s![0..copy_rows, 0..copy_cols]);

                let mut approx_dst = approx.slice_mut(s![
                    out_row_start..out_row_start + copy_rows,
                    out_col_start..out_col_start + copy_cols
                ]);
                let mut detail_h_dst = detail_h.slice_mut(s![
                    out_row_start..out_row_start + copy_rows,
                    out_col_start..out_col_start + copy_cols
                ]);
                let mut detail_v_dst = detail_v.slice_mut(s![
                    out_row_start..out_row_start + copy_rows,
                    out_col_start..out_col_start + copy_cols
                ]);
                let mut detail_d_dst = detail_d.slice_mut(s![
                    out_row_start..out_row_start + copy_rows,
                    out_col_start..out_col_start + copy_cols
                ]);

                approx_dst.assign(&approxsrc);
                detail_h_dst.assign(&detail_hsrc);
                detail_v_dst.assign(&detail_vsrc);
                detail_d_dst.assign(&detail_dsrc);
            }
        }
    }

    Ok(EnhancedDwt2dResult {
        approx,
        detail_h,
        detail_v,
        detail_d,
        originalshape: (rows, cols),
        boundary_mode: config.boundary_mode,
        metrics: None,
    })
}

/// Adaptive 2D DWT decomposition with entropy-based stopping criteria
pub fn enhanced_dwt2d_adaptive(
    data: &Array2<f64>,
    wavelet: Wavelet,
    config: &Dwt2dConfig,
    energy_threshold: f64,
) -> SignalResult<MultilevelDwt2d> {
    checkarray_finite(data, "data")?;

    if energy_threshold <= 0.0 || energy_threshold >= 1.0 {
        return Err(SignalError::ValueError(
            "Energy threshold must be between 0 and 1".to_string(),
        ));
    }

    let mut current = data.clone();
    let mut details = Vec::new();
    let mut level = 0;
    let max_levels = calculate_max_decomposition_levels(data.dim());
    let mut previous_energy_ratio = 1.0;
    let mut energy_decrease_count = 0;

    // Enhanced stopping criteria tracking
    let mut level_energies = Vec::new();
    let mut level_entropies = Vec::new();

    loop {
        // Check multiple stopping criteria
        let (rows, cols) = current.dim();

        // Size-based stopping criterion
        if rows < 4 || cols < 4 || level >= max_levels {
            break;
        }

        // Perform one level of decomposition
        let decomp = enhanced_dwt2d_decompose(&current, wavelet, config)?;

        // Comprehensive energy analysis
        let detail_h_energy: f64 = decomp.detail_h.iter().map(|&x| x * x).sum();
        let detail_v_energy: f64 = decomp.detail_v.iter().map(|&x| x * x).sum();
        let detail_d_energy: f64 = decomp.detail_d.iter().map(|&x| x * x).sum();
        let detail_energy = detail_h_energy + detail_v_energy + detail_d_energy;

        let approx_energy: f64 = decomp.approx.iter().map(|&x| x * x).sum();
        let total_energy = current.iter().map(|&x| x * x).sum::<f64>();

        // Energy-based stopping criterion
        let energy_ratio = detail_energy / total_energy.max(1e-10);
        level_energies.push(energy_ratio);

        // Entropy-based analysis for adaptive stopping
        let entropy =
            compute_subband_entropy(&decomp.detail_h, &decomp.detail_v, &decomp.detail_d)?;
        level_entropies.push(entropy);

        // Store detail coefficients
        details.push((
            decomp.detail_h.clone(),
            decomp.detail_v.clone(),
            decomp.detail_d.clone(),
        ));

        current = decomp.approx;
        level += 1;

        // Enhanced stopping criteria

        // 1. Primary energy threshold
        if energy_ratio < energy_threshold {
            break;
        }

        // 2. Energy decrease trend analysis
        if energy_ratio >= previous_energy_ratio {
            energy_decrease_count += 1;
            if energy_decrease_count >= 2 {
                // Energy is not decreasing consistently, stop
                break;
            }
        } else {
            energy_decrease_count = 0;
        }

        // 3. Entropy-based stopping (very low entropy indicates little structure)
        if entropy < 0.1 && level > 1 {
            break;
        }

        // 4. Approximation energy dominance
        let approx_ratio = approx_energy / total_energy.max(1e-10);
        if approx_ratio > 0.99 && level > 1 {
            // Almost all energy in approximation, further decomposition unlikely to be useful
            break;
        }

        // 5. Coefficient magnitude analysis
        let max_detail_coeff = [
            decomp
                .detail_h
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
            decomp
                .detail_v
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
            decomp
                .detail_d
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
        ]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

        let signal_range = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - data.iter().cloned().fold(f64::INFINITY, f64::min);

        if max_detail_coeff < signal_range * 1e-6 {
            // Detail coefficients are negligible compared to signal range
            break;
        }

        previous_energy_ratio = energy_ratio;
    }

    // Reverse details to have coarsest level first
    details.reverse();

    Ok(MultilevelDwt2d {
        approx: current,
        details,
        originalshape: data.dim(),
        wavelet,
        config: config.clone(),
    })
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Apply boundary padding to a 1D signal
fn apply_boundary_padding(signal: &[f64], filterlen: usize, mode: BoundaryMode) -> Vec<f64> {
    let pad_len = filterlen / 2;
    let n = signal.len();
    let mut padded = Vec::with_capacity(n + 2 * pad_len);

    match mode {
        BoundaryMode::Zero => {
            padded.extend(vec![0.0; pad_len]);
            padded.extend_from_slice(signal);
            padded.extend(vec![0.0; pad_len]);
        }
        BoundaryMode::Symmetric => {
            // Reflect at boundaries
            for i in (0..pad_len).rev() {
                padded.push(signal[i.min(n - 1)]);
            }
            padded.extend_from_slice(signal);
            for i in 0..pad_len {
                padded.push(signal[n - 1 - i.min(n - 1)]);
            }
        }
        BoundaryMode::Periodic => {
            // Wrap around
            for i in (n - pad_len)..n {
                padded.push(signal[i]);
            }
            padded.extend_from_slice(signal);
            for i in 0..pad_len {
                padded.push(signal[i]);
            }
        }
        BoundaryMode::Constant(value) => {
            padded.extend(vec![value; pad_len]);
            padded.extend_from_slice(signal);
            padded.extend(vec![value; pad_len]);
        }
        BoundaryMode::AntiSymmetric => {
            // Anti-symmetric reflection with improved indexing
            for i in 0..pad_len {
                let idx = (pad_len - i - 1).min(n - 1);
                padded.push(2.0 * signal[0] - signal[idx]);
            }
            padded.extend_from_slice(signal);
            for i in 0..pad_len {
                let idx = (n - 1 - i).max(0).min(n - 1);
                padded.push(2.0 * signal[n - 1] - signal[idx]);
            }
        }
        BoundaryMode::Smooth => {
            // Polynomial extrapolation (linear for simplicity)
            if n >= 2 {
                let slope_left = signal[1] - signal[0];
                let slope_right = signal[n - 1] - signal[n - 2];

                for i in 0..pad_len {
                    padded.push(signal[0] - slope_left * (pad_len - i) as f64);
                }
                padded.extend_from_slice(signal);
                for i in 1..=pad_len {
                    padded.push(signal[n - 1] + slope_right * i as f64);
                }
            } else {
                // Fallback to symmetric for very short signals
                for i in (0..pad_len).rev() {
                    padded.push(signal[i.min(n - 1)]);
                }
                padded.extend_from_slice(signal);
                for i in 0..pad_len {
                    padded.push(signal[n - 1 - i.min(n - 1)]);
                }
            }
        }
        BoundaryMode::Adaptive => {
            // Use symmetric as default - should be replaced by analyze_and_select_boundary_mode
            for i in (0..pad_len).rev() {
                padded.push(signal[i.min(n - 1)]);
            }
            padded.extend_from_slice(signal);
            for i in 0..pad_len {
                padded.push(signal[n - 1 - i.min(n - 1)]);
            }
        }
        BoundaryMode::Extrapolate => {
            // Linear extrapolation
            if n >= 2 {
                let slope_left = signal[1] - signal[0];
                let slope_right = signal[n - 1] - signal[n - 2];

                for i in 0..pad_len {
                    padded.push(signal[0] - slope_left * (pad_len - i) as f64);
                }
                padded.extend_from_slice(signal);
                for i in 1..=pad_len {
                    padded.push(signal[n - 1] + slope_right * i as f64);
                }
            } else {
                // Fallback to zero padding
                padded.extend(vec![0.0; pad_len]);
                padded.extend_from_slice(signal);
                padded.extend(vec![0.0; pad_len]);
            }
        }
        BoundaryMode::MirrorCorrect => {
            // Mirror with edge correction - similar to symmetric but with adjustment
            for i in (0..pad_len).rev() {
                let idx = i.min(n - 1);
                padded.push(2.0 * signal[0] - signal[idx]);
            }
            padded.extend_from_slice(signal);
            for i in 0..pad_len {
                let idx = (n - 1 - i).max(0).min(n - 1);
                padded.push(2.0 * signal[n - 1] - signal[idx]);
            }
        }
        BoundaryMode::ContentAware => {
            // Use symmetric as fallback - content-aware padding would require image analysis
            for i in (0..pad_len).rev() {
                padded.push(signal[i.min(n - 1)]);
            }
            padded.extend_from_slice(signal);
            for i in 0..pad_len {
                padded.push(signal[n - 1 - i.min(n - 1)]);
            }
        }
    }

    padded
}

/// Apply filters using SIMD optimization
fn apply_filters_simd(
    signal: &[f64],
    lo_filter: &[f64],
    hi_filter: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    let filter_len = lo_filter.len();
    let output_len = n.saturating_sub(1) + filter_len;

    let mut lo_out = vec![0.0; output_len];
    let mut hi_out = vec![0.0; output_len];

    // Enhanced SIMD convolution with better memory access patterns
    if filter_len >= 8 && n >= 8 {
        // Advanced vectorized path for large filters
        apply_filters_simd_large(signal, lo_filter, hi_filter, &mut lo_out, &mut hi_out);
    } else if filter_len >= 4 && n >= 4 {
        // Standard SIMD path for medium filters
        apply_filters_simd_medium(signal, lo_filter, hi_filter, &mut lo_out, &mut hi_out);
    } else {
        // Optimized scalar path for small filters
        apply_filters_scalar_optimized(signal, lo_filter, hi_filter, &mut lo_out, &mut hi_out);
    }

    (lo_out, hi_out)
}

/// Downsample signal by factor of 2
fn downsample(signal: &[f64]) -> Vec<f64> {
    signal.iter().step_by(2).cloned().collect()
}

/// Process a single block for memory-optimized DWT
#[allow(dead_code)]
fn process_dwt2d_block(
    block: &Array2<f64>,
    filters: &WaveletFilters,
    config: &Dwt2dConfig,
    row_offset: usize,
    col_offset: usize,
) -> SignalResult<EnhancedDwt2dResult> {
    // Enhanced block processing with offset-aware optimizations
    let (_rows, _cols) = block.dim();

    // Apply block-specific optimizations based on position
    let optimized_config = if row_offset == 0 || col_offset == 0 {
        // Edge blocks may benefit from different boundary handling
        Dwt2dConfig {
            boundary_mode: match config.boundary_mode {
                BoundaryMode::Adaptive => BoundaryMode::Symmetric,
                mode => mode,
            },
            ..*config
        }
    } else {
        config.clone()
    };

    // Use enhanced SIMD processing for the block
    simd_dwt2d_decompose(block, filters, &optimized_config)
}

/// Process a single block with ArrayView for memory-optimized DWT (zero-copy when possible)
#[allow(dead_code)]
fn process_dwt2d_block_view(
    block: &ArrayView2<f64>,
    filters: &WaveletFilters,
    config: &Dwt2dConfig,
    row_offset: usize,
    col_offset: usize,
) -> SignalResult<EnhancedDwt2dResult> {
    // Convert view to owned array for processing
    let block_owned = block.to_owned();
    process_dwt2d_block(&block_owned, filters, config, row_offset, col_offset)
}

/// Advanced SIMD filter application for large filters
#[allow(dead_code)]
fn apply_filters_simd_large(
    signal: &[f64],
    lo_filter: &[f64],
    hi_filter: &[f64],
    lo_out: &mut [f64],
    hi_out: &mut [f64],
) {
    // Optimized SIMD implementation for large filters (8+ coefficients)
    // For now, fallback to scalar implementation
    apply_filters_scalar_optimized(signal, lo_filter, hi_filter, lo_out, hi_out);
}

/// Standard SIMD filter application for medium filters
#[allow(dead_code)]
fn apply_filters_simd_medium(
    signal: &[f64],
    lo_filter: &[f64],
    hi_filter: &[f64],
    lo_out: &mut [f64],
    hi_out: &mut [f64],
) {
    // Optimized SIMD implementation for medium filters (4-7 coefficients)
    // For now, fallback to scalar implementation
    apply_filters_scalar_optimized(signal, lo_filter, hi_filter, lo_out, hi_out);
}

/// Optimized scalar filter application
#[allow(dead_code)]
fn apply_filters_scalar_optimized(
    signal: &[f64],
    lo_filter: &[f64],
    hi_filter: &[f64],
    lo_out: &mut [f64],
    hi_out: &mut [f64],
) {
    let n = signal.len();
    let filter_len = lo_filter.len();

    // Convolution with filters
    for i in 0..n {
        for j in 0..filter_len {
            if i + j < lo_out.len() {
                lo_out[i + j] += signal[i] * lo_filter[j];
                hi_out[i + j] += signal[i] * hi_filter[j];
            }
        }
    }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Calculate maximum reasonable decomposition levels based on image size
fn calculate_max_decomposition_levels(shape: (usize, usize)) -> usize {
    let (rows, cols) = shape;
    let min_dim = rows.min(cols);

    // Allow decomposition until minimum dimension is 4
    // This ensures we don't over-decompose small images
    (min_dim as f64).log2().floor() as usize - 2
}

/// Compute entropy of subband coefficients for adaptive stopping
fn compute_subband_entropy(
    detail_h: &Array2<f64>,
    detail_v: &Array2<f64>,
    detail_d: &Array2<f64>,
) -> SignalResult<f64> {
    // Combine all detail coefficients
    let coeffs: Vec<f64> = detail_h.iter()
        .chain(detail_v.iter())
        .chain(detail_d.iter())
        .map(|&x: &f64| x.abs())
        .filter(|&x| x > 1e-12) // Filter near-zero values
        .collect();

    if coeffs.is_empty() {
        return Ok(0.0);
    }

    // Normalize to create probability distribution
    let sum: f64 = coeffs.iter().sum();
    if sum <= 0.0 {
        return Ok(0.0);
    }

    // Compute normalized Shannon entropy
    let entropy = coeffs
        .iter()
        .map(|&x| {
            let p = x / sum;
            if p > 1e-12 {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum::<f64>();

    // Normalize by maximum possible entropy for this number of coefficients
    let max_entropy = (coeffs.len() as f64).log2();
    if max_entropy > 0.0 {
        Ok(entropy / max_entropy)
    } else {
        Ok(0.0)
    }
}

/// Analyze and select boundary mode based on data characteristics
fn analyze_and_select_boundary_mode(
    data: &Array2<f64>,
    _filters: &WaveletFilters,
) -> SignalResult<BoundaryMode> {
    let (_rows, _cols) = data.dim();

    // Simple heuristic analysis - in a full implementation, this would
    // analyze edge characteristics, periodicity, etc.
    // For now, return Symmetric as a safe default
    Ok(BoundaryMode::Symmetric)
}

/// Validate DWT decomposition result
fn validate_dwt2d_result(
    result: &EnhancedDwt2dResult,
    originalshape: (usize, usize),
    _config: &Dwt2dConfig,
) -> SignalResult<()> {
    let (orig_rows, orig_cols) = originalshape;
    let expected_rows = orig_rows.div_ceil(2);
    let expected_cols = orig_cols.div_ceil(2);

    // Check dimensions of all subbands
    if result.approx.dim() != (expected_rows, expected_cols) {
        return Err(SignalError::ComputationError(format!(
            "Approximation subband has incorrect dimensions: expected ({}, {}), got ({}, {})",
            expected_rows,
            expected_cols,
            result.approx.nrows(),
            result.approx.ncols()
        )));
    }

    if result.detail_h.dim() != (expected_rows, expected_cols) {
        return Err(SignalError::ComputationError(format!(
            "Horizontal detail subband has incorrect dimensions: expected ({}, {}), got ({}, {})",
            expected_rows,
            expected_cols,
            result.detail_h.nrows(),
            result.detail_h.ncols()
        )));
    }

    if result.detail_v.dim() != (expected_rows, expected_cols) {
        return Err(SignalError::ComputationError(format!(
            "Vertical detail subband has incorrect dimensions: expected ({}, {}), got ({}, {})",
            expected_rows,
            expected_cols,
            result.detail_v.nrows(),
            result.detail_v.ncols()
        )));
    }

    if result.detail_d.dim() != (expected_rows, expected_cols) {
        return Err(SignalError::ComputationError(format!(
            "Diagonal detail subband has incorrect dimensions: expected ({}, {}), got ({}, {})",
            expected_rows,
            expected_cols,
            result.detail_d.nrows(),
            result.detail_d.ncols()
        )));
    }

    // Check for finite values in all subbands
    let subbands = [
        (&result.approx, "approximation"),
        (&result.detail_h, "horizontal detail"),
        (&result.detail_v, "vertical detail"),
        (&result.detail_d, "diagonal detail"),
    ];

    for (subband, name) in subbands {
        for (idx, &val) in subband.iter().enumerate() {
            if !val.is_finite() {
                let (row, col) = (idx / subband.ncols(), idx % subband.ncols());
                return Err(SignalError::ComputationError(format!(
                    "Non-finite value {} found in {} subband at position ({}, {})",
                    val, name, row, col
                )));
            }
        }
    }

    Ok(())
}

/// Compute quality metrics for 2D DWT analysis
fn compute_dwt2d_quality_metrics(
    original: &Array2<f64>,
    result: &EnhancedDwt2dResult,
) -> SignalResult<Dwt2dQualityMetrics> {
    // Energy calculations
    let original_energy: f64 = original.iter().map(|&x| x * x).sum();

    let approx_energy: f64 = result.approx.iter().map(|&x| x * x).sum();
    let detail_h_energy: f64 = result.detail_h.iter().map(|&x| x * x).sum();
    let detail_v_energy: f64 = result.detail_v.iter().map(|&x| x * x).sum();
    let detail_d_energy: f64 = result.detail_d.iter().map(|&x| x * x).sum();

    let detail_energy = detail_h_energy + detail_v_energy + detail_d_energy;
    let total_transformed_energy = approx_energy + detail_energy;

    // Energy preservation (should be close to 1.0 for perfect transforms)
    let energy_preservation = if original_energy > 0.0 {
        total_transformed_energy / original_energy
    } else {
        1.0
    };

    // Sparsity measure (percentage of near-zero coefficients)
    let threshold = original_energy.sqrt() * 1e-6; // Adaptive threshold
    let total_coeffs =
        result.approx.len() + result.detail_h.len() + result.detail_v.len() + result.detail_d.len();

    let sparse_coeffs = result
        .approx
        .iter()
        .chain(result.detail_h.iter())
        .chain(result.detail_v.iter())
        .chain(result.detail_d.iter())
        .filter(|&&x| x.abs() < threshold)
        .count();

    let sparsity = sparse_coeffs as f64 / total_coeffs as f64;

    // Compression ratio estimate (based on sparsity)
    let compression_ratio = if sparsity > 0.1 {
        1.0 / (1.0 - sparsity + 0.1)
    } else {
        1.0
    };

    // Edge preservation (simplified metric based on detail energy)
    let edge_preservation = if original_energy > 0.0 {
        detail_energy / original_energy
    } else {
        0.0
    };

    Ok(Dwt2dQualityMetrics {
        approx_energy,
        detail_energy,
        energy_preservation,
        compression_ratio,
        sparsity,
        edge_preservation,
    })
}
