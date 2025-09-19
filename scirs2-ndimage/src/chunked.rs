//! Chunked processing for memory-efficient operations on large arrays
//!
//! This module provides utilities for processing large arrays in chunks
//! to reduce memory usage and enable processing of arrays that don't fit
//! in memory.

use ndarray::{Array, ArrayView, Dimension, IxDyn};
use num_traits::{Float, FromPrimitive, NumCast, Zero};
use std::fmt::Debug;
use std::sync::Mutex;

use crate::error::{NdimageError, NdimageResult};
use crate::filters::BorderMode;

/// Configuration for chunked processing
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Target chunk size in bytes
    pub chunk_size_bytes: usize,
    /// Overlap between chunks (for filters that need context)
    pub overlap: usize,
    /// Minimum chunk size in elements along each dimension
    pub min_chunk_size: usize,
    /// Whether to process chunks in parallel
    pub parallel: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size_bytes: 64 * 1024 * 1024, // 64 MB
            overlap: 0,
            min_chunk_size: 16,
            parallel: true,
        }
    }
}

/// Represents a chunk's position in the array
#[derive(Debug, Clone)]
pub struct ChunkPosition {
    /// Start indices for each dimension
    pub start: Vec<usize>,
    /// End indices for each dimension
    pub end: Vec<usize>,
}

/// Trait for operations that can be applied to chunks
pub trait ChunkProcessor<T, D>
where
    D: Dimension,
{
    /// Process a single chunk of the array
    fn process_chunk(
        &mut self,
        chunk: ArrayView<T, D>,
        position: &ChunkPosition,
    ) -> NdimageResult<Array<T, D>>;

    /// Get the required overlap for this processor
    fn required_overlap(&self) -> usize;

    /// Combine results from multiple chunks
    fn combine_chunks(
        &self,
        results: Vec<(Array<T, D>, ChunkPosition)>,
        outputshape: &[usize],
    ) -> NdimageResult<Array<T, D>>;
}

/// Process an array in chunks using the given processor
#[allow(dead_code)]
pub fn process_chunked<T, D, P>(
    input: &ArrayView<T, D>,
    processor: &mut P,
    config: &ChunkConfig,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync,
    D: Dimension,
    P: ChunkProcessor<T, D> + Send + Sync,
{
    let shape = input.shape();
    let ndim = input.ndim();
    let element_size = std::mem::size_of::<T>();

    // Calculate chunk dimensions based on target size
    let total_elements = shape.iter().product::<usize>();
    let target_elements_per_chunk = config.chunk_size_bytes / element_size;

    if total_elements <= target_elements_per_chunk {
        // Array is small enough to process as a single chunk
        let position = ChunkPosition {
            start: vec![0; ndim],
            end: shape.to_vec(),
        };
        let result = processor.process_chunk(input.clone(), &position)?;
        return Ok(result);
    }

    // Calculate chunk sizes for each dimension
    let chunk_sizes =
        calculate_chunk_sizes(shape, target_elements_per_chunk, config.min_chunk_size);
    let overlap = processor.required_overlap().max(config.overlap);

    // Generate chunk positions
    let chunks = generate_chunk_positions(shape, &chunk_sizes, overlap);

    // Process chunks
    let results = if config.parallel && chunks.len() > 1 {
        #[cfg(feature = "parallel")]
        {
            use scirs2_core::parallel_ops::*;

            let processor_mutex = Mutex::new(&mut *processor);
            chunks
                .into_par_iter()
                .map(|position| {
                    let chunk = extract_chunk(input, &position)?;
                    let result = {
                        let mut proc = processor_mutex.lock().unwrap();
                        proc.process_chunk(chunk.view(), &position)?
                    };
                    Ok((result, position))
                })
                .collect::<Result<Vec<_>, NdimageError>>()?
        }
        #[cfg(not(feature = "parallel"))]
        {
            chunks
                .into_iter()
                .map(|position| {
                    let chunk = extract_chunk(input, &position)?;
                    let result = processor.process_chunk(chunk.view(), &position)?;
                    Ok((result, position))
                })
                .collect::<Result<Vec<_>, NdimageError>>()?
        }
    } else {
        chunks
            .into_iter()
            .map(|position| {
                let chunk = extract_chunk(input, &position)?;
                let result = processor.process_chunk(chunk.view(), &position)?;
                Ok((result, position))
            })
            .collect::<Result<Vec<_>, NdimageError>>()?
    };

    // Combine results
    processor.combine_chunks(results, shape)
}

/// Calculate optimal chunk sizes for each dimension
#[allow(dead_code)]
fn calculate_chunk_sizes(
    shape: &[usize],
    target_elements: usize,
    min_chunk_size: usize,
) -> Vec<usize> {
    let ndim = shape.len();
    let mut chunk_sizes = vec![0; ndim];

    // Start with equal division
    let elements_per_dim = (target_elements as f64).powf(1.0 / ndim as f64) as usize;

    for (i, &dim_size) in shape.iter().enumerate() {
        chunk_sizes[i] = elements_per_dim.min(dim_size).max(min_chunk_size);
    }

    // Adjust chunk sizes to better match target
    let mut current_elements: usize = chunk_sizes.iter().product();

    while current_elements > target_elements * 2 {
        // Find the dimension with the largest chunk _size relative to its total _size
        let (max_idx_, _) = chunk_sizes
            .iter()
            .enumerate()
            .filter(|(i, &_size)| _size > min_chunk_size && _size < shape[*i])
            .max_by_key(|(i, &_size)| _size * 1000 / shape[*i])
            .unwrap_or((0, &1));

        if chunk_sizes[max_idx_] > min_chunk_size {
            chunk_sizes[max_idx_] = (chunk_sizes[max_idx_] / 2).max(min_chunk_size);
            current_elements = chunk_sizes.iter().product();
        } else {
            break;
        }
    }

    chunk_sizes
}

/// Generate chunk positions with overlap
#[allow(dead_code)]
fn generate_chunk_positions(
    shape: &[usize],
    chunk_sizes: &[usize],
    overlap: usize,
) -> Vec<ChunkPosition> {
    let ndim = shape.len();
    let mut positions = Vec::new();

    // Generate all combinations of chunk indices
    let mut indices = vec![0; ndim];

    loop {
        let mut position = ChunkPosition {
            start: Vec::with_capacity(ndim),
            end: Vec::with_capacity(ndim),
        };

        for dim in 0..ndim {
            let start = if indices[dim] == 0 {
                0
            } else {
                indices[dim] * chunk_sizes[dim] - overlap
            };
            let end = (start + chunk_sizes[dim] + overlap).min(shape[dim]);

            position.start.push(start);
            position.end.push(end);
        }

        positions.push(position);

        // Increment indices
        let mut carry = true;
        for dim in (0..ndim).rev() {
            if carry {
                indices[dim] += 1;
                if (indices[dim] + 1) * chunk_sizes[dim] >= shape[dim] + overlap {
                    if indices[dim] * chunk_sizes[dim] < shape[dim] {
                        carry = false;
                    } else {
                        indices[dim] = 0;
                    }
                } else {
                    carry = false;
                }
            }
        }

        if carry {
            break;
        }
    }

    positions
}

/// Extract a chunk from the array
#[allow(dead_code)]
fn extract_chunk<T, D>(
    array: &ArrayView<T, D>,
    position: &ChunkPosition,
) -> NdimageResult<Array<T, D>>
where
    T: Clone,
    D: Dimension,
{
    use ndarray::SliceInfoElem;

    // Always use dynamic slicing for any dimension
    let slice_info: Vec<SliceInfoElem> = position
        .start
        .iter()
        .zip(&position.end)
        .map(|(&start, &end)| SliceInfoElem::Slice {
            start: start as isize,
            end: Some(end as isize),
            step: 1,
        })
        .collect();

    let chunk = array.view().into_dyn().slice_move(slice_info.as_slice());
    let owned_chunk = chunk.to_owned();
    Ok(owned_chunk
        .into_dimensionality::<D>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert chunk dimension".into()))?)
}

/// Example chunk processor for Gaussian filtering
pub struct GaussianChunkProcessor<T> {
    sigma: Vec<T>,
    truncate: Option<T>,
    bordermode: BorderMode,
}

impl<T> GaussianChunkProcessor<T>
where
    T: Float + FromPrimitive,
{
    pub fn new(_sigma: Vec<T>, truncate: Option<T>, bordermode: BorderMode) -> Self {
        Self {
            sigma: _sigma,
            truncate,
            bordermode,
        }
    }
}

impl<T, D> ChunkProcessor<T, D> for GaussianChunkProcessor<T>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + Zero,
    D: Dimension,
{
    fn process_chunk(
        &mut self,
        chunk: ArrayView<T, D>,
        _position: &ChunkPosition,
    ) -> NdimageResult<Array<T, D>> {
        // Apply Gaussian filter to the chunk
        // This is a placeholder - actual implementation would call the gaussian filter
        Ok(chunk.to_owned())
    }

    fn required_overlap(&self) -> usize {
        // Calculate required overlap based on sigma and truncate
        let max_sigma = self
            .sigma
            .iter()
            .map(|&s| NumCast::from(s).unwrap_or(0.0))
            .fold(0.0f64, |a, b| a.max(b));

        let truncate = self
            .truncate
            .map(|t| NumCast::from(t).unwrap_or(4.0))
            .unwrap_or(4.0);

        ((truncate * max_sigma).ceil() as usize).max(1)
    }

    fn combine_chunks(
        &self,
        results: Vec<(Array<T, D>, ChunkPosition)>,
        outputshape: &[usize],
    ) -> NdimageResult<Array<T, D>> {
        // Create output array
        let mut output = Array::<T, IxDyn>::zeros(IxDyn(outputshape));
        let overlap = <Self as ChunkProcessor<T, D>>::required_overlap(self);

        // Copy chunks into output, handling overlap
        for (chunk_result, position) in results {
            use ndarray::SliceInfoElem;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_calculate_chunk_sizes() {
        let shape = vec![1000, 1000];
        let target_elements = 10000;
        let min_chunk_size = 10;

        let chunk_sizes = calculate_chunk_sizes(&shape, target_elements, min_chunk_size);

        assert_eq!(chunk_sizes.len(), 2);
        assert!(chunk_sizes[0] >= min_chunk_size);
        assert!(chunk_sizes[1] >= min_chunk_size);
        assert!(chunk_sizes[0] <= shape[0]);
        assert!(chunk_sizes[1] <= shape[1]);

        let total_elements: usize = chunk_sizes.iter().product();
        assert!(total_elements <= target_elements * 3); // Allow some flexibility
    }

    #[test]
    fn test_generate_chunk_positions() {
        let shape = vec![100, 100];
        let chunk_sizes = vec![50, 50];
        let overlap = 5;

        let positions = generate_chunk_positions(&shape, &chunk_sizes, overlap);

        // Should have 2x2 = 4 chunks
        assert_eq!(positions.len(), 4);

        // Check first chunk
        assert_eq!(positions[0].start, vec![0, 0]);
        assert_eq!(positions[0].end, vec![55, 55]); // 50 + 5 overlap
    }

    // Simple identity processor for testing
    struct IdentityProcessor;

    impl<T: Clone + Zero, D: Dimension> ChunkProcessor<T, D> for IdentityProcessor {
        fn process_chunk(
            &mut self,
            chunk: ArrayView<T, D>,
            _position: &ChunkPosition,
        ) -> NdimageResult<Array<T, D>> {
            Ok(chunk.to_owned())
        }

        fn required_overlap(&self) -> usize {
            0
        }

        fn combine_chunks(
            &self,
            results: Vec<(Array<T, D>, ChunkPosition)>,
            outputshape: &[usize],
        ) -> NdimageResult<Array<T, D>> {
            use ndarray::SliceInfoElem;

            let mut output = Array::zeros(IxDyn(outputshape));

            for (chunk, position) in results {
                let slice_info: Vec<SliceInfoElem> = position
                    .start
                    .iter()
                    .zip(&position.end)
                    .map(|(&start, &end)| SliceInfoElem::Slice {
                        start: start as isize,
                        end: Some(end as isize),
                        step: 1,
                    })
                    .collect();

                let mut output_slice = output.slice_mut(slice_info.as_slice());
                output_slice.assign(&chunk.view().into_dyn());
            }

            output
                .into_dimensionality::<D>()
                .map_err(|_| NdimageError::DimensionError("Dimension conversion failed".into()))
        }
    }

    #[test]
    fn test_process_chunked_identity() {
        let input = Array2::<f64>::ones((100, 100));
        let mut processor = IdentityProcessor;
        let config = ChunkConfig {
            chunk_size_bytes: 800, // Force chunking (100 elements * 8 bytes)
            overlap: 0,
            min_chunk_size: 10,
            parallel: false,
        };

        let result = process_chunked(&input.view(), &mut processor, &config).unwrap();

        assert_eq!(result.shape(), input.shape());
        assert_eq!(result, input);
    }
}
