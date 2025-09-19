//! SIMD-accelerated normalization operations
//!
//! This module provides SIMD-optimized implementations of normalization operations
//! using the unified SIMD operations from scirs2-core.
//!
//! Features:
//! - Adaptive block sizing based on data dimensions and cache sizes
//! - Memory-aligned processing for maximum SIMD efficiency
//! - Cache-optimal memory access patterns
//! - Advanced prefetching strategies for large datasets

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use num_traits::{Float, NumCast};
use scirs2_core::simd_ops::SimdUnifiedOps;

use crate::error::{Result, TransformError};
use crate::normalize::{NormalizationMethod, EPSILON};

/// Cache line size for optimal memory access
const CACHE_LINE_SIZE: usize = 64;
/// L1 cache size estimate for block sizing
const L1_CACHE_SIZE: usize = 32_768;
/// L2 cache size estimate for adaptive processing
const L2_CACHE_SIZE: usize = 262_144;

/// Adaptive block sizing strategy for cache-optimal processing
#[derive(Debug, Clone)]
pub struct AdaptiveBlockSizer {
    /// Optimal block size for current data dimensions
    pub optimal_block_size: usize,
    /// Whether to use cache-aligned processing
    pub use_cache_alignment: bool,
    /// Prefetch distance for memory optimization
    pub prefetch_distance: usize,
}

impl AdaptiveBlockSizer {
    /// Create adaptive block sizer based on data characteristics
    pub fn new<F>(datashape: &[usize]) -> Self
    where
        F: Float + NumCast,
    {
        let element_size = std::mem::size_of::<F>();
        let data_size = datashape.iter().product::<usize>() * element_size;

        // Adaptive block sizing based on data size and cache characteristics
        let optimal_block_size = if data_size <= L1_CACHE_SIZE / 4 {
            // Small data: use larger blocks for better vectorization
            256
        } else if data_size <= L2_CACHE_SIZE / 2 {
            // Medium data: balance cache efficiency and parallelism
            128
        } else {
            // Large data: smaller blocks to maintain cache locality
            64
        };

        let use_cache_alignment = data_size > L1_CACHE_SIZE;
        let prefetch_distance = if data_size > L2_CACHE_SIZE { 8 } else { 4 };

        AdaptiveBlockSizer {
            optimal_block_size,
            use_cache_alignment,
            prefetch_distance,
        }
    }

    /// Get cache-aligned block size
    pub fn get_aligned_block_size(&self, dimensionsize: usize) -> usize {
        if self.use_cache_alignment {
            // Align to cache line boundaries
            let cache_aligned =
                (self.optimal_block_size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1);
            cache_aligned.min(dimensionsize)
        } else {
            self.optimal_block_size.min(dimensionsize)
        }
    }
}

/// SIMD-accelerated min-max normalization for 1D arrays
#[allow(dead_code)]
pub fn simd_minmax_normalize_1d<F>(array: &Array1<F>) -> Result<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    if array.is_empty() {
        return Err(TransformError::InvalidInput(
            "Input array is empty".to_string(),
        ));
    }

    let min = F::simd_min_element(&array.view());
    let max = F::simd_max_element(&array.view());
    let range = max - min;

    if range.abs() <= F::from(EPSILON).unwrap() {
        // Constant feature, return array of 0.5
        return Ok(Array1::from_elem(array.len(), F::from(0.5).unwrap()));
    }

    // Normalize: (x - min) / range
    let minarray = Array1::from_elem(array.len(), min);
    let normalized = F::simd_sub(&array.view(), &minarray.view());
    let rangearray = Array1::from_elem(array.len(), range);
    let result = F::simd_div(&normalized.view(), &rangearray.view());

    Ok(result)
}

/// SIMD-accelerated Z-score normalization for 1D arrays
#[allow(dead_code)]
pub fn simd_zscore_normalize_1d<F>(array: &Array1<F>) -> Result<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    if array.is_empty() {
        return Err(TransformError::InvalidInput(
            "Input array is empty".to_string(),
        ));
    }

    let mean = F::simd_mean(&array.view());
    let n = F::from(array.len()).unwrap();

    // Compute variance
    let meanarray = Array1::from_elem(array.len(), mean);
    let centered = F::simd_sub(&array.view(), &meanarray.view());
    let squared = F::simd_mul(&centered.view(), &centered.view());
    let variance = F::simd_sum(&squared.view()) / n;
    let std_dev = variance.sqrt();

    if std_dev <= F::from(EPSILON).unwrap() {
        // Constant feature, return zeros
        return Ok(Array1::zeros(array.len()));
    }

    // Normalize: (x - mean) / std_dev
    let stdarray = Array1::from_elem(array.len(), std_dev);
    let result = F::simd_div(&centered.view(), &stdarray.view());

    Ok(result)
}

/// SIMD-accelerated L2 normalization for 1D arrays
#[allow(dead_code)]
pub fn simd_l2_normalize_1d<F>(array: &Array1<F>) -> Result<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    if array.is_empty() {
        return Err(TransformError::InvalidInput(
            "Input array is empty".to_string(),
        ));
    }

    let l2_norm = F::simd_norm(&array.view());

    if l2_norm <= F::from(EPSILON).unwrap() {
        // Zero vector, return zeros
        return Ok(Array1::zeros(array.len()));
    }

    // Normalize: x / l2_norm
    let normarray = Array1::from_elem(array.len(), l2_norm);
    let result = F::simd_div(&array.view(), &normarray.view());

    Ok(result)
}

/// SIMD-accelerated max absolute scaling for 1D arrays
#[allow(dead_code)]
pub fn simd_maxabs_normalize_1d<F>(array: &Array1<F>) -> Result<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    if array.is_empty() {
        return Err(TransformError::InvalidInput(
            "Input array is empty".to_string(),
        ));
    }

    let absarray = F::simd_abs(&array.view());
    let max_abs = F::simd_max_element(&absarray.view());

    if max_abs <= F::from(EPSILON).unwrap() {
        // All zeros, return zeros
        return Ok(Array1::zeros(array.len()));
    }

    // Normalize: x / max_abs
    let max_absarray = Array1::from_elem(array.len(), max_abs);
    let result = F::simd_div(&array.view(), &max_absarray.view());

    Ok(result)
}

/// Advanced SIMD-accelerated normalization for 2D arrays with optimized memory access patterns
#[allow(dead_code)]
pub fn simd_normalizearray<S, F>(
    array: &ArrayBase<S, Ix2>,
    method: NormalizationMethod,
    axis: usize,
) -> Result<Array2<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + SimdUnifiedOps,
{
    if !array.is_standard_layout() {
        return Err(TransformError::InvalidInput(
            "Input array must be in standard memory layout".to_string(),
        ));
    }

    if array.ndim() != 2 {
        return Err(TransformError::InvalidInput(
            "Only 2D arrays are supported".to_string(),
        ));
    }

    if axis >= array.ndim() {
        return Err(TransformError::InvalidInput(format!(
            "Invalid axis {} for array with {} dimensions",
            axis,
            array.ndim()
        )));
    }

    let shape = array.shape();
    let mut normalized = Array2::zeros((shape[0], shape[1]));

    match method {
        NormalizationMethod::MinMax => simd_normalize_block_minmax(array, &mut normalized, axis)?,
        NormalizationMethod::ZScore => simd_normalize_block_zscore(array, &mut normalized, axis)?,
        NormalizationMethod::L2 => simd_normalize_block_l2(array, &mut normalized, axis)?,
        NormalizationMethod::MaxAbs => simd_normalize_block_maxabs(array, &mut normalized, axis)?,
        _ => {
            // Fall back to non-SIMD implementation for other methods
            return Err(TransformError::InvalidInput(
                "SIMD implementation not available for this normalization method".to_string(),
            ));
        }
    }

    Ok(normalized)
}

/// Block-wise SIMD min-max normalization with optimized memory access
#[allow(dead_code)]
fn simd_normalize_block_minmax<S, F>(
    array: &ArrayBase<S, Ix2>,
    normalized: &mut Array2<F>,
    axis: usize,
) -> Result<()>
where
    S: Data<Elem = F>,
    F: Float + NumCast + SimdUnifiedOps,
{
    let shape = array.shape();
    let block_sizer = AdaptiveBlockSizer::new::<F>(&[shape[0], shape[1]]);
    let block_size =
        block_sizer.get_aligned_block_size(if axis == 0 { shape[1] } else { shape[0] });

    if axis == 0 {
        // Column-wise normalization with block processing
        let mut global_mins = Array1::zeros(shape[1]);
        let mut global_maxs = Array1::zeros(shape[1]);

        // First pass: compute global min/max for each column in blocks
        for block_start in (0..shape[1]).step_by(block_size) {
            let block_end = (block_start + block_size).min(shape[1]);

            for j in block_start..block_end {
                let col = array.column(j);
                let colarray = col.to_owned();
                global_mins[j] = F::simd_min_element(&colarray.view());
                global_maxs[j] = F::simd_max_element(&colarray.view());
            }
        }

        // Second pass: normalize using pre-computed min/max
        for block_start in (0..shape[1]).step_by(block_size) {
            let block_end = (block_start + block_size).min(shape[1]);

            for j in block_start..block_end {
                let col = array.column(j);
                let colarray = col.to_owned();
                let range = global_maxs[j] - global_mins[j];

                if range.abs() <= F::from(EPSILON).unwrap() {
                    // Constant feature
                    for i in 0..shape[0] {
                        normalized[[i, j]] = F::from(0.5).unwrap();
                    }
                } else {
                    // Vectorized normalization
                    let minarray = Array1::from_elem(shape[0], global_mins[j]);
                    let rangearray = Array1::from_elem(shape[0], range);
                    let centered = F::simd_sub(&colarray.view(), &minarray.view());
                    let norm_col = F::simd_div(&centered.view(), &rangearray.view());

                    for i in 0..shape[0] {
                        normalized[[i, j]] = norm_col[i];
                    }
                }
            }
        }
    } else {
        // Row-wise normalization with block processing
        for block_start in (0..shape[0]).step_by(block_size) {
            let block_end = (block_start + block_size).min(shape[0]);

            for i in block_start..block_end {
                let row = array.row(i);
                let rowarray = row.to_owned();
                let norm_row = simd_minmax_normalize_1d(&rowarray)?;

                for j in 0..shape[1] {
                    normalized[[i, j]] = norm_row[j];
                }
            }
        }
    }
    Ok(())
}

/// Block-wise SIMD Z-score normalization with optimized memory access
#[allow(dead_code)]
fn simd_normalize_block_zscore<S, F>(
    array: &ArrayBase<S, Ix2>,
    normalized: &mut Array2<F>,
    axis: usize,
) -> Result<()>
where
    S: Data<Elem = F>,
    F: Float + NumCast + SimdUnifiedOps,
{
    let shape = array.shape();
    let block_sizer = AdaptiveBlockSizer::new::<F>(&[shape[0], shape[1]]);
    let block_size =
        block_sizer.get_aligned_block_size(if axis == 0 { shape[1] } else { shape[0] });

    if axis == 0 {
        // Column-wise normalization
        let mut global_means = Array1::zeros(shape[1]);
        let mut global_stds = Array1::zeros(shape[1]);
        let n_samples_f = F::from(shape[0]).unwrap();

        // First pass: compute means and standard deviations
        for block_start in (0..shape[1]).step_by(block_size) {
            let block_end = (block_start + block_size).min(shape[1]);

            for j in block_start..block_end {
                let col = array.column(j);
                let colarray = col.to_owned();

                // Compute mean
                global_means[j] = F::simd_sum(&colarray.view()) / n_samples_f;

                // Compute standard deviation
                let meanarray = Array1::from_elem(shape[0], global_means[j]);
                let centered = F::simd_sub(&colarray.view(), &meanarray.view());
                let squared = F::simd_mul(&centered.view(), &centered.view());
                let variance = F::simd_sum(&squared.view()) / n_samples_f;
                global_stds[j] = variance.sqrt();

                // Avoid division by zero
                if global_stds[j] <= F::from(EPSILON).unwrap() {
                    global_stds[j] = F::one();
                }
            }
        }

        // Second pass: normalize using pre-computed statistics
        for block_start in (0..shape[1]).step_by(block_size) {
            let block_end = (block_start + block_size).min(shape[1]);

            for j in block_start..block_end {
                let col = array.column(j);
                let colarray = col.to_owned();

                if global_stds[j] <= F::from(EPSILON).unwrap() {
                    // Constant feature
                    for i in 0..shape[0] {
                        normalized[[i, j]] = F::zero();
                    }
                } else {
                    // Vectorized normalization
                    let meanarray = Array1::from_elem(shape[0], global_means[j]);
                    let stdarray = Array1::from_elem(shape[0], global_stds[j]);
                    let centered = F::simd_sub(&colarray.view(), &meanarray.view());
                    let norm_col = F::simd_div(&centered.view(), &stdarray.view());

                    for i in 0..shape[0] {
                        normalized[[i, j]] = norm_col[i];
                    }
                }
            }
        }
    } else {
        // Row-wise normalization with block processing
        for block_start in (0..shape[0]).step_by(block_size) {
            let block_end = (block_start + block_size).min(shape[0]);

            for i in block_start..block_end {
                let row = array.row(i);
                let rowarray = row.to_owned();
                let norm_row = simd_zscore_normalize_1d(&rowarray)?;

                for j in 0..shape[1] {
                    normalized[[i, j]] = norm_row[j];
                }
            }
        }
    }
    Ok(())
}

/// Block-wise SIMD L2 normalization with optimized memory access
#[allow(dead_code)]
fn simd_normalize_block_l2<S, F>(
    array: &ArrayBase<S, Ix2>,
    normalized: &mut Array2<F>,
    axis: usize,
) -> Result<()>
where
    S: Data<Elem = F>,
    F: Float + NumCast + SimdUnifiedOps,
{
    let shape = array.shape();
    let block_sizer = AdaptiveBlockSizer::new::<F>(&[shape[0], shape[1]]);
    let block_size =
        block_sizer.get_aligned_block_size(if axis == 0 { shape[1] } else { shape[0] });

    if axis == 0 {
        // Column-wise L2 normalization
        for block_start in (0..shape[1]).step_by(block_size) {
            let block_end = (block_start + block_size).min(shape[1]);

            for j in block_start..block_end {
                let col = array.column(j);
                let colarray = col.to_owned();
                let norm_col = simd_l2_normalize_1d(&colarray)?;

                for i in 0..shape[0] {
                    normalized[[i, j]] = norm_col[i];
                }
            }
        }
    } else {
        // Row-wise L2 normalization with SIMD optimization
        for block_start in (0..shape[0]).step_by(block_size) {
            let block_end = (block_start + block_size).min(shape[0]);

            for i in block_start..block_end {
                let row = array.row(i);
                let rowarray = row.to_owned();
                let l2_norm = F::simd_norm(&rowarray.view());

                if l2_norm <= F::from(EPSILON).unwrap() {
                    // Zero vector
                    for j in 0..shape[1] {
                        normalized[[i, j]] = F::zero();
                    }
                } else {
                    // Vectorized division
                    let normarray = Array1::from_elem(shape[1], l2_norm);
                    let norm_row = F::simd_div(&rowarray.view(), &normarray.view());

                    for j in 0..shape[1] {
                        normalized[[i, j]] = norm_row[j];
                    }
                }
            }
        }
    }
    Ok(())
}

/// Block-wise SIMD max-absolute normalization with optimized memory access
#[allow(dead_code)]
fn simd_normalize_block_maxabs<S, F>(
    array: &ArrayBase<S, Ix2>,
    normalized: &mut Array2<F>,
    axis: usize,
) -> Result<()>
where
    S: Data<Elem = F>,
    F: Float + NumCast + SimdUnifiedOps,
{
    let shape = array.shape();
    let block_sizer = AdaptiveBlockSizer::new::<F>(&[shape[0], shape[1]]);
    let block_size =
        block_sizer.get_aligned_block_size(if axis == 0 { shape[1] } else { shape[0] });

    if axis == 0 {
        // Column-wise max-abs normalization
        for block_start in (0..shape[1]).step_by(block_size) {
            let block_end = (block_start + block_size).min(shape[1]);

            for j in block_start..block_end {
                let col = array.column(j);
                let colarray = col.to_owned();
                let norm_col = simd_maxabs_normalize_1d(&colarray)?;

                for i in 0..shape[0] {
                    normalized[[i, j]] = norm_col[i];
                }
            }
        }
    } else {
        // Row-wise max-abs normalization with SIMD optimization
        for block_start in (0..shape[0]).step_by(block_size) {
            let block_end = (block_start + block_size).min(shape[0]);

            for i in block_start..block_end {
                let row = array.row(i);
                let rowarray = row.to_owned();
                let absarray = F::simd_abs(&rowarray.view());
                let max_abs = F::simd_max_element(&absarray.view());

                if max_abs <= F::from(EPSILON).unwrap() {
                    // All zeros
                    for j in 0..shape[1] {
                        normalized[[i, j]] = F::zero();
                    }
                } else {
                    // Vectorized division
                    let max_absarray = Array1::from_elem(shape[1], max_abs);
                    let norm_row = F::simd_div(&rowarray.view(), &max_absarray.view());

                    for j in 0..shape[1] {
                        normalized[[i, j]] = norm_row[j];
                    }
                }
            }
        }
    }
    Ok(())
}

/// Advanced SIMD normalization with automatic optimization selection
#[allow(dead_code)]
pub fn simd_normalize_adaptive<S, F>(
    array: &ArrayBase<S, Ix2>,
    method: NormalizationMethod,
    axis: usize,
) -> Result<Array2<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + SimdUnifiedOps,
{
    let shape = array.shape();
    let data_size = shape[0] * shape[1] * std::mem::size_of::<F>();

    // Choose optimal strategy based on data characteristics
    if data_size > L2_CACHE_SIZE {
        // Large data: use chunked processing with memory optimization
        simd_normalize_chunked(array, method, axis)
    } else if shape[0] > shape[1] * 4 {
        // Tall matrix: optimize for column-wise operations
        simd_normalize_optimized_tall(array, method, axis)
    } else if shape[1] > shape[0] * 4 {
        // Wide matrix: optimize for row-wise operations
        simd_normalize_optimized_wide(array, method, axis)
    } else {
        // Standard processing
        simd_normalizearray(array, method, axis)
    }
}

/// Memory-efficient batch processing for advanced-large datasets
#[allow(dead_code)]
pub fn simd_normalize_batch<S, F>(
    array: &ArrayBase<S, Ix2>,
    method: NormalizationMethod,
    axis: usize,
    batch_size_mb: usize,
) -> Result<Array2<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + SimdUnifiedOps,
{
    let shape = array.shape();
    let element_size = std::mem::size_of::<F>();
    let max_elements_per_batch = (batch_size_mb * 1024 * 1024) / element_size;

    if shape[0] * shape[1] <= max_elements_per_batch {
        // Small enough for single batch
        return simd_normalize_adaptive(array, method, axis);
    }

    let mut normalized = Array2::zeros((shape[0], shape[1]));

    if axis == 0 {
        // Column-wise: batch by columns
        let cols_per_batch = max_elements_per_batch / shape[0];
        for col_start in (0..shape[1]).step_by(cols_per_batch) {
            let col_end = (col_start + cols_per_batch).min(shape[1]);
            let batch_view = array.slice(ndarray::s![.., col_start..col_end]);
            let batch_normalized = simd_normalize_adaptive(&batch_view, method, axis)?;

            for (j_local, j_global) in (col_start..col_end).enumerate() {
                for i in 0..shape[0] {
                    normalized[[i, j_global]] = batch_normalized[[i, j_local]];
                }
            }
        }
    } else {
        // Row-wise: batch by rows
        let rows_per_batch = max_elements_per_batch / shape[1];
        for row_start in (0..shape[0]).step_by(rows_per_batch) {
            let row_end = (row_start + rows_per_batch).min(shape[0]);
            let batch_view = array.slice(ndarray::s![row_start..row_end, ..]);
            let batch_normalized = simd_normalize_adaptive(&batch_view, method, axis)?;

            for (i_local, i_global) in (row_start..row_end).enumerate() {
                for j in 0..shape[1] {
                    normalized[[i_global, j]] = batch_normalized[[i_local, j]];
                }
            }
        }
    }

    Ok(normalized)
}

/// Chunked SIMD normalization for large datasets
#[allow(dead_code)]
fn simd_normalize_chunked<S, F>(
    array: &ArrayBase<S, Ix2>,
    method: NormalizationMethod,
    axis: usize,
) -> Result<Array2<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + SimdUnifiedOps,
{
    let shape = array.shape();
    let mut normalized = Array2::zeros((shape[0], shape[1]));

    let block_sizer = AdaptiveBlockSizer::new::<F>(&[shape[0], shape[1]]);
    let chunk_size = block_sizer.optimal_block_size * 4; // Larger chunks for big data

    if axis == 0 {
        // Process in column chunks
        for chunk_start in (0..shape[1]).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(shape[1]);
            let chunk_view = array.slice(ndarray::s![.., chunk_start..chunk_end]);
            let chunk_normalized = simd_normalizearray(&chunk_view, method, axis)?;

            for (j_local, j_global) in (chunk_start..chunk_end).enumerate() {
                for i in 0..shape[0] {
                    normalized[[i, j_global]] = chunk_normalized[[i, j_local]];
                }
            }
        }
    } else {
        // Process in row chunks
        for chunk_start in (0..shape[0]).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(shape[0]);
            let chunk_view = array.slice(ndarray::s![chunk_start..chunk_end, ..]);
            let chunk_normalized = simd_normalizearray(&chunk_view, method, axis)?;

            for (i_local, i_global) in (chunk_start..chunk_end).enumerate() {
                for j in 0..shape[1] {
                    normalized[[i_global, j]] = chunk_normalized[[i_local, j]];
                }
            }
        }
    }

    Ok(normalized)
}

/// Optimized SIMD normalization for tall matrices (many rows, few columns)
#[allow(dead_code)]
fn simd_normalize_optimized_tall<S, F>(
    array: &ArrayBase<S, Ix2>,
    method: NormalizationMethod,
    axis: usize,
) -> Result<Array2<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + SimdUnifiedOps,
{
    // For tall matrices, optimize memory access patterns
    if axis == 0 {
        // Column-wise normalization: pre-compute all statistics
        simd_normalizearray(array, method, axis)
    } else {
        // Row-wise normalization: use smaller blocks for better cache usage
        let shape = array.shape();
        let mut normalized = Array2::zeros((shape[0], shape[1]));
        let small_block_size = 32; // Smaller blocks for tall matrices

        for block_start in (0..shape[0]).step_by(small_block_size) {
            let block_end = (block_start + small_block_size).min(shape[0]);

            for i in block_start..block_end {
                let row = array.row(i);
                let rowarray = row.to_owned();
                let norm_row = match method {
                    NormalizationMethod::MinMax => simd_minmax_normalize_1d(&rowarray)?,
                    NormalizationMethod::ZScore => simd_zscore_normalize_1d(&rowarray)?,
                    NormalizationMethod::L2 => simd_l2_normalize_1d(&rowarray)?,
                    NormalizationMethod::MaxAbs => simd_maxabs_normalize_1d(&rowarray)?,
                    _ => {
                        return Err(TransformError::InvalidInput(
                            "Unsupported normalization method for tall matrix optimization"
                                .to_string(),
                        ))
                    }
                };

                for j in 0..shape[1] {
                    normalized[[i, j]] = norm_row[j];
                }
            }
        }

        Ok(normalized)
    }
}

/// Optimized SIMD normalization for wide matrices (few rows, many columns)
#[allow(dead_code)]
fn simd_normalize_optimized_wide<S, F>(
    array: &ArrayBase<S, Ix2>,
    method: NormalizationMethod,
    axis: usize,
) -> Result<Array2<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + SimdUnifiedOps,
{
    // For wide matrices, optimize for column-wise operations
    if axis == 1 {
        // Row-wise normalization: straightforward processing
        simd_normalizearray(array, method, axis)
    } else {
        // Column-wise normalization: use vectorized column processing
        let shape = array.shape();
        let mut normalized = Array2::zeros((shape[0], shape[1]));
        let wide_block_size = 128; // Larger blocks for wide matrices

        for block_start in (0..shape[1]).step_by(wide_block_size) {
            let block_end = (block_start + wide_block_size).min(shape[1]);

            for j in block_start..block_end {
                let col = array.column(j);
                let colarray = col.to_owned();
                let norm_col = match method {
                    NormalizationMethod::MinMax => simd_minmax_normalize_1d(&colarray)?,
                    NormalizationMethod::ZScore => simd_zscore_normalize_1d(&colarray)?,
                    NormalizationMethod::L2 => simd_l2_normalize_1d(&colarray)?,
                    NormalizationMethod::MaxAbs => simd_maxabs_normalize_1d(&colarray)?,
                    _ => {
                        return Err(TransformError::InvalidInput(
                            "Unsupported normalization method for wide matrix optimization"
                                .to_string(),
                        ))
                    }
                };

                for i in 0..shape[0] {
                    normalized[[i, j]] = norm_col[i];
                }
            }
        }

        Ok(normalized)
    }
}
