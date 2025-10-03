//! Memory-efficient filter implementations using chunked processing
//!
//! This module provides memory-efficient versions of filters that can process
//! large arrays by dividing them into chunks.

use scirs2_core::ndarray::{Array, ArrayView, Dimension, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive, NumCast, Zero};
use std::fmt::Debug;

use crate::chunked::{process_chunked, ChunkConfig, ChunkPosition, ChunkProcessor};
use crate::error::{NdimageError, NdimageResult};
use crate::filters::{median_filter, uniform_filter, BorderMode};

/// Chunk processor for uniform filtering
pub struct UniformChunkProcessor<T> {
    size: Vec<usize>,
    border_mode: BorderMode,
    _marker: std::marker::PhantomData<T>,
}

impl<T> UniformChunkProcessor<T> {
    pub fn new(size: Vec<usize>, bordermode: BorderMode) -> Self {
        Self {
            size,
            border_mode: bordermode,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T, D> ChunkProcessor<T, D> for UniformChunkProcessor<T>
where
    T: Float
        + FromPrimitive
        + NumCast
        + Debug
        + Clone
        + Send
        + Sync
        + Zero
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
    D: Dimension + 'static,
{
    fn process_chunk(
        &mut self,
        chunk: ArrayView<T, D>,
        _position: &ChunkPosition,
    ) -> NdimageResult<Array<T, D>> {
        uniform_filter(&chunk.to_owned(), &self.size, Some(self.border_mode), None)
    }

    fn required_overlap(&self) -> usize {
        // Uniform filter needs overlap of (filter_size - 1) to ensure proper border handling
        // For a NxN filter, we need N-1 pixels of overlap
        self.size
            .iter()
            .max()
            .copied()
            .unwrap_or(1)
            .saturating_sub(1)
    }

    fn combine_chunks(
        &self,
        results: Vec<(Array<T, D>, ChunkPosition)>,
        outputshape: &[usize],
    ) -> NdimageResult<Array<T, D>> {
        let overlap = <Self as ChunkProcessor<T, D>>::required_overlap(self);
        combine_chunks_with_overlap(results, outputshape, overlap)
    }
}

/// Chunk processor for median filtering
pub struct MedianChunkProcessor<T> {
    size: Vec<usize>,
    border_mode: BorderMode,
    _marker: std::marker::PhantomData<T>,
}

impl<T> MedianChunkProcessor<T> {
    pub fn new(size: Vec<usize>, bordermode: BorderMode) -> Self {
        Self {
            size,
            border_mode: bordermode,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T, D> ChunkProcessor<T, D> for MedianChunkProcessor<T>
where
    T: Float
        + FromPrimitive
        + NumCast
        + Debug
        + Clone
        + Send
        + Sync
        + Zero
        + PartialOrd
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
    D: Dimension + 'static,
{
    fn process_chunk(
        &mut self,
        chunk: ArrayView<T, D>,
        _position: &ChunkPosition,
    ) -> NdimageResult<Array<T, D>> {
        median_filter(&chunk.to_owned(), &self.size, Some(self.border_mode))
    }

    fn required_overlap(&self) -> usize {
        // Median filter needs overlap of (filter_size - 1) to ensure proper border handling
        // For a NxN filter, we need N-1 pixels of overlap
        self.size
            .iter()
            .max()
            .copied()
            .unwrap_or(1)
            .saturating_sub(1)
    }

    fn combine_chunks(
        &self,
        results: Vec<(Array<T, D>, ChunkPosition)>,
        outputshape: &[usize],
    ) -> NdimageResult<Array<T, D>> {
        let overlap = <Self as ChunkProcessor<T, D>>::required_overlap(self);
        combine_chunks_with_overlap(results, outputshape, overlap)
    }
}

/// Memory-efficient uniform filter
///
/// This function applies a uniform filter to large arrays by processing them in chunks.
///
/// # Arguments
///
/// * `input` - Input array
/// * `size` - Size of the filter window in each dimension
/// * `border_mode` - How to handle borders
/// * `config` - Optional chunking configuration
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
#[allow(dead_code)]
pub fn uniform_filter_chunked<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    border_mode: BorderMode,
    config: Option<ChunkConfig>,
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + NumCast
        + Debug
        + Clone
        + Send
        + Sync
        + Zero
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
    D: Dimension + 'static,
{
    let config = config.unwrap_or_default();
    let mut processor = UniformChunkProcessor::new(size.to_vec(), border_mode);
    process_chunked(&input.view(), &mut processor, &config)
}

/// Memory-efficient median filter
///
/// This function applies a median filter to large arrays by processing them in chunks.
///
/// # Arguments
///
/// * `input` - Input array
/// * `size` - Size of the filter window in each dimension
/// * `border_mode` - How to handle borders
/// * `config` - Optional chunking configuration
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
#[allow(dead_code)]
pub fn median_filter_chunked<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    border_mode: BorderMode,
    config: Option<ChunkConfig>,
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + NumCast
        + Debug
        + Clone
        + Send
        + Sync
        + Zero
        + PartialOrd
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
    D: Dimension + 'static,
{
    let config = config.unwrap_or_default();
    let mut processor = MedianChunkProcessor::new(size.to_vec(), border_mode);
    process_chunked(&input.view(), &mut processor, &config)
}

/// Memory-efficient Gaussian filter
///
/// This function applies a Gaussian filter to large arrays by processing them in chunks.
///
/// # Arguments
///
/// * `input` - Input array
/// * `sigma` - Standard deviation for each axis
/// * `truncate` - Truncate the filter at this many standard deviations
/// * `border_mode` - How to handle borders
/// * `config` - Optional chunking configuration
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
#[allow(dead_code)]
pub fn gaussian_filter_chunked<T, D>(
    input: &Array<T, D>,
    sigma: &[T],
    truncate: Option<T>,
    border_mode: BorderMode,
    config: Option<ChunkConfig>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + Zero + 'static,
    D: Dimension + 'static,
{
    let config = config.unwrap_or_default();
    let mut processor =
        crate::chunked::GaussianChunkProcessor::new(sigma.to_vec(), truncate, border_mode);
    process_chunked(&input.view(), &mut processor, &config)
}

/// Helper function to combine chunks with overlap handling
#[allow(dead_code)]
fn combine_chunks_with_overlap<T, D>(
    results: Vec<(Array<T, D>, ChunkPosition)>,
    outputshape: &[usize],
    overlap: usize,
) -> NdimageResult<Array<T, D>>
where
    T: Clone + Zero,
    D: Dimension + 'static,
{
    use scirs2_core::ndarray::SliceInfoElem;

    // Create output array
    let mut output = Array::<T, IxDyn>::zeros(IxDyn(outputshape));

    // Copy chunks into output, handling overlap
    for (chunk_result, position) in results {
        // Calculate the region to copy (excluding overlap at boundaries)
        let mut copy_start = Vec::new();
        let mut copy_end = Vec::new();
        let mut chunk_start = Vec::new();
        let mut chunk_end = Vec::new();

        for (dim, (&start, &end)) in position.start.iter().zip(&position.end).enumerate() {
            // Output region
            let out_start = if start > 0 {
                start + overlap / 2
            } else {
                start
            };
            let out_end = if end < outputshape[dim] {
                end - overlap / 2
            } else {
                end
            };
            copy_start.push(out_start);
            copy_end.push(out_end);

            // Corresponding chunk region
            let ch_start = if start > 0 { overlap / 2 } else { 0 };
            let ch_end = chunk_result.shape()[dim]
                - if end < outputshape[dim] {
                    overlap / 2
                } else {
                    0
                };
            chunk_start.push(ch_start);
            chunk_end.push(ch_end);
        }

        // Build slice info for output
        let output_slice_info: Vec<SliceInfoElem> = copy_start
            .iter()
            .zip(&copy_end)
            .map(|(&start, &end)| SliceInfoElem::Slice {
                start: start as isize,
                end: Some(end as isize),
                step: 1,
            })
            .collect();

        // Build slice info for chunk
        let chunk_slice_info: Vec<SliceInfoElem> = chunk_start
            .iter()
            .zip(&chunk_end)
            .map(|(&start, &end)| SliceInfoElem::Slice {
                start: start as isize,
                end: Some(end as isize),
                step: 1,
            })
            .collect();

        // Copy data
        let chunk_dyn = chunk_result.view().into_dyn();
        let chunk_slice = chunk_dyn.slice(chunk_slice_info.as_slice());
        let mut output_slice = output.slice_mut(output_slice_info.as_slice());
        output_slice.assign(&chunk_slice);
    }

    // Convert back to the correct dimension type
    output
        .into_dimensionality::<D>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert output dimension".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_uniform_filter_chunked() {
        let input = Array2::<f64>::ones((100, 100));
        let size = vec![3, 3];

        let config = ChunkConfig {
            chunk_size_bytes: 800, // Force chunking
            overlap: 2,
            min_chunk_size: 10,
            parallel: false,
        };

        let result =
            uniform_filter_chunked(&input, &size, BorderMode::Constant, Some(config)).unwrap();

        // Check that the result is correct (all ones should remain ones with uniform filter)
        assert_eq!(result.shape(), input.shape());

        // Check center values (should be 1.0 for uniform input)
        for i in 10..90 {
            for j in 10..90 {
                assert_abs_diff_eq!(result[[i, j]], 1.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_median_filter_chunked() {
        let mut input = Array2::<f64>::zeros((50, 50));
        // Add some noise
        input[[25, 25]] = 10.0;

        let size = vec![3, 3];

        let config = ChunkConfig {
            chunk_size_bytes: 400, // Force chunking
            overlap: 2,
            min_chunk_size: 10,
            parallel: false,
        };

        let result =
            median_filter_chunked(&input, &size, BorderMode::Constant, Some(config)).unwrap();

        // The spike should be removed by the median filter
        assert!(result[[25, 25]] < 1.0);
    }

    #[test]
    fn test_chunked_vs_regular() {
        let input = Array2::<f64>::from_shape_fn((50, 50), |(i, j)| {
            (i as f64 * 0.1).sin() + (j as f64 * 0.1).cos()
        });

        let size = vec![5, 5];

        // Regular filter
        let regular_result =
            uniform_filter(&input, &size, Some(BorderMode::Reflect), None).unwrap();

        // Chunked filter
        let config = ChunkConfig {
            chunk_size_bytes: 400, // Force chunking
            overlap: 4,
            min_chunk_size: 10,
            parallel: false,
        };
        let chunked_result =
            uniform_filter_chunked(&input, &size, BorderMode::Reflect, Some(config)).unwrap();

        // Results should be very close (some differences at chunk boundaries are acceptable)
        let diff = (&regular_result - &chunked_result).mapv(f64::abs);
        let max_diff = diff.iter().fold(0.0, |a, &b| a.max(b));
        assert!(max_diff < 1e-10, "Max difference: {}", max_diff);
    }
}
