//! Enhanced chunked processing using scirs2-core's memory-efficient features
//!
//! This module provides a more advanced implementation of chunked processing
//! that leverages scirs2-core's memory-efficient infrastructure for better
//! performance and lower memory usage.

use scirs2_core::ndarray::{Array, ArrayView, Dimension, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive, NumCast};
use std::fmt::Debug;

use scirs2_core::error::CoreResult;
use scirs2_core::memory_efficient::{
    chunk_wise_binary_op, chunk_wise_op, create_mmap, AccessMode, ChunkingStrategy,
};

use crate::error::{NdimageError, NdimageResult};
use crate::filters::BorderMode;

/// Enhanced configuration for chunked processing
#[derive(Debug, Clone)]
pub struct ChunkConfigV2 {
    /// Chunking strategy from scirs2-core
    pub strategy: ChunkingStrategy,
    /// Overlap between chunks (for filters that need context)
    pub overlap: usize,
    /// Whether to use memory-mapped arrays for very large inputs
    pub use_mmap_threshold: Option<usize>,
    /// Whether to process chunks in parallel
    pub parallel: bool,
}

impl Default for ChunkConfigV2 {
    fn default() -> Self {
        Self {
            strategy: ChunkingStrategy::Auto,
            overlap: 0,
            use_mmap_threshold: Some(100 * 1024 * 1024), // 100 MB
            parallel: true,
        }
    }
}

/// Trait for operations that can be applied to chunks using core's infrastructure
pub trait ChunkProcessorV2<T, D>
where
    T: Float + Send + Sync,
    D: Dimension,
{
    /// Create the processing closure for chunks
    fn create_processor(
        &self,
    ) -> Box<dyn Fn(&ArrayView<T, IxDyn>) -> CoreResult<Array<T, IxDyn>> + Send + Sync>;

    /// Get the required overlap for this processor
    fn required_overlap(&self) -> usize;
}

/// Process an array using scirs2-core's chunk-wise operations
#[allow(dead_code)]
pub fn process_chunked_v2<T, D, P>(
    input: &ArrayView<T, D>,
    processor: &P,
    config: &ChunkConfigV2,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension + 'static,
    P: ChunkProcessorV2<T, D>,
{
    // Get the processing operation from the processor
    let op = processor.create_processor();

    // Check if we should use memory-mapped array
    let input_size = input.len() * std::mem::size_of::<T>();

    if let Some(threshold) = config.use_mmap_threshold {
        if input_size > threshold {
            // Use memory-mapped array for very large inputs
            return process_with_mmap(input, processor, config, &*op);
        }
    }

    // Convert to dynamic dimension for chunk_wise_op
    let input_dyn = input.view().into_dyn();

    // Create a wrapper closure that unwraps the Result
    let wrapper_op = |chunk: &ArrayView<T, IxDyn>| -> Array<T, IxDyn> {
        match op(chunk) {
            Ok(result) => result,
            Err(e) => {
                // For now, we'll panic on error since chunk_wise_op doesn't support fallible operations
                panic!("Chunk processing failed: {}", e);
            }
        }
    };

    // Use scirs2-core's chunk_wise_op
    let result_dyn = chunk_wise_op(&input_dyn, wrapper_op, config.strategy.clone())
        .map_err(|e| NdimageError::ProcessingError(format!("Chunk processing failed: {}", e)))?;

    // Convert back to original dimension
    result_dyn
        .into_dimensionality::<D>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert dimension".into()))
}

/// Process using memory-mapped arrays for very large inputs
#[allow(dead_code)]
fn process_with_mmap<T, D, P>(
    input: &ArrayView<T, D>,
    processor: &P,
    config: &ChunkConfigV2,
    op: &dyn Fn(&ArrayView<T, IxDyn>) -> CoreResult<Array<T, IxDyn>>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension + 'static,
    P: ChunkProcessorV2<T, D>,
{
    use tempfile::tempdir;

    // Create temporary directory for memory-mapped files
    let temp_dir = tempdir().map_err(NdimageError::IoError)?;

    let input_path = temp_dir.path().join("input.mmap");
    let output_path = temp_dir.path().join("output.mmap");

    // Create memory-mapped input array
    let mmap_input = create_mmap(input, &input_path, AccessMode::Write, 0)
        .map_err(|e| NdimageError::ProcessingError(format!("Failed to create mmap: {}", e)))?;

    // Process using memory-mapped array
    let input_view = mmap_input
        .as_array::<D>()
        .map_err(|e| NdimageError::ProcessingError(format!("Failed to get array view: {}", e)))?;

    let input_dyn = input_view.view().into_dyn();

    // Create a wrapper closure that unwraps the Result
    let wrapper_op = |chunk: &ArrayView<T, IxDyn>| -> Array<T, IxDyn> {
        match op(chunk) {
            Ok(result) => result,
            Err(e) => {
                // For now, we'll panic on error since chunk_wise_op doesn't support fallible operations
                panic!("Chunk processing failed: {}", e);
            }
        }
    };

    // Use chunk_wise_op on the memory-mapped array
    let result_dyn = chunk_wise_op(&input_dyn, wrapper_op, config.strategy.clone())
        .map_err(|e| NdimageError::ProcessingError(format!("Chunk processing failed: {}", e)))?;

    // Convert to owned array and correct dimension
    let result = result_dyn
        .into_dimensionality::<D>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert dimension".into()))?;

    Ok(result)
}

/// Memory-efficient uniform filter using core's chunking
#[allow(dead_code)]
pub fn uniform_filter_chunked_v2<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    bordermode: BorderMode,
    config: Option<ChunkConfigV2>,
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + NumCast
        + Debug
        + Clone
        + Send
        + Sync
        + 'static
        + std::ops::AddAssign
        + std::ops::DivAssign,
    D: Dimension + 'static,
{
    let config = config.unwrap_or_default();
    let processor = UniformProcessorV2::new(size.to_vec(), bordermode);

    process_chunked_v2(&input.view(), &processor, &config)
}

/// Helper processor for uniform filter
struct UniformProcessorV2 {
    size: Vec<usize>,
    bordermode: BorderMode,
}

impl UniformProcessorV2 {
    fn new(_size: Vec<usize>, bordermode: BorderMode) -> Self {
        Self {
            size: _size,
            bordermode,
        }
    }
}

impl<T, D> ChunkProcessorV2<T, D> for UniformProcessorV2
where
    T: Float
        + FromPrimitive
        + NumCast
        + Debug
        + Clone
        + Send
        + Sync
        + 'static
        + std::ops::AddAssign
        + std::ops::DivAssign,
    D: Dimension,
{
    fn create_processor(
        &self,
    ) -> Box<dyn Fn(&ArrayView<T, IxDyn>) -> CoreResult<Array<T, IxDyn>> + Send + Sync> {
        let size_vec = self.size.clone();
        let bordermode_clone = self.bordermode;

        Box::new(
            move |chunk: &ArrayView<T, IxDyn>| -> CoreResult<Array<T, IxDyn>> {
                // Convert to owned array for processing
                let chunk_owned = chunk.to_owned();

                // Apply uniform filter to the chunk
                let result = crate::filters::uniform_filter(
                    &chunk_owned,
                    &size_vec,
                    Some(bordermode_clone),
                    None,
                )
                .map_err(|e| {
                    scirs2_core::error::CoreError::ComputationError(
                        scirs2_core::error::ErrorContext::new(e.to_string()),
                    )
                })?;

                Ok(result.into_dyn())
            },
        )
    }

    fn required_overlap(&self) -> usize {
        self.size.iter().max().copied().unwrap_or(0)
    }
}

/// Memory-efficient convolution using core's chunking
#[allow(dead_code)]
pub fn convolve_chunked_v2<T, D>(
    input: &Array<T, D>,
    kernel: &Array<T, D>,
    bordermode: BorderMode,
    config: Option<ChunkConfigV2>,
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + NumCast
        + Debug
        + Clone
        + Send
        + Sync
        + 'static
        + std::ops::AddAssign
        + std::ops::DivAssign,
    D: Dimension + 'static,
{
    let config = config.unwrap_or_default();
    let processor = ConvolveProcessorV2::new(kernel.clone(), bordermode);

    process_chunked_v2(&input.view(), &processor, &config)
}

/// Helper processor for convolution
struct ConvolveProcessorV2<T, D> {
    kernel: Array<T, D>,
    bordermode: BorderMode,
}

impl<T, D> ConvolveProcessorV2<T, D>
where
    T: Clone,
    D: Dimension,
{
    fn new(_kernel: Array<T, D>, bordermode: BorderMode) -> Self {
        Self {
            kernel: _kernel,
            bordermode,
        }
    }
}

impl<T, D> ChunkProcessorV2<T, D> for ConvolveProcessorV2<T, D>
where
    T: Float
        + FromPrimitive
        + NumCast
        + Debug
        + Clone
        + Send
        + Sync
        + 'static
        + std::ops::AddAssign
        + std::ops::DivAssign,
    D: Dimension + 'static,
{
    fn create_processor(
        &self,
    ) -> Box<dyn Fn(&ArrayView<T, IxDyn>) -> CoreResult<Array<T, IxDyn>> + Send + Sync> {
        let kernel_clone = self.kernel.clone();
        let bordermode_clone = self.bordermode;

        Box::new(
            move |chunk: &ArrayView<T, IxDyn>| -> CoreResult<Array<T, IxDyn>> {
                let chunk_owned = chunk.to_owned();

                // Apply convolution to the chunk
                let result =
                    crate::filters::convolve(&chunk_owned, &kernel_clone, Some(bordermode_clone))
                        .map_err(|e| {
                        scirs2_core::error::CoreError::ComputationError(
                            scirs2_core::error::ErrorContext::new(e.to_string()),
                        )
                    })?;

                Ok(result)
            },
        )
    }

    fn required_overlap(&self) -> usize {
        self.kernel.shape().iter().max().copied().unwrap_or(0)
    }
}

/// Binary operation between two arrays using chunked processing
#[allow(dead_code)]
pub fn binary_op_chunked_v2<T, D, F>(
    array1: &Array<T, D>,
    array2: &Array<T, D>,
    op: F,
    config: Option<ChunkConfigV2>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension + 'static,
    F: Fn(T, T) -> T + Send + Sync + Clone + 'static,
{
    let config = config.unwrap_or_default();

    // Convert to dynamic dimensions
    let array1_dyn = array1.view().into_dyn();
    let array2_dyn = array2.view().into_dyn();

    // Create a wrapper closure that applies element-wise operation
    let wrapper_op = |a1: &ArrayView<T, IxDyn>, a2: &ArrayView<T, IxDyn>| -> Array<T, IxDyn> {
        let mut result = Array::zeros(a1.raw_dim());
        for ((r, &v1), &v2) in result.iter_mut().zip(a1.iter()).zip(a2.iter()) {
            *r = op(v1, v2);
        }
        result
    };

    // Use chunk_wise_binary_op from core
    let result_dyn = chunk_wise_binary_op(&array1_dyn, &array2_dyn, wrapper_op, config.strategy)
        .map_err(|e| NdimageError::ProcessingError(format!("Binary operation failed: {}", e)))?;

    // Convert back to original dimension
    result_dyn
        .into_dimensionality::<D>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert dimension".into()))
}

/// Configuration builder for easier setup
pub struct ChunkConfigBuilder {
    config: ChunkConfigV2,
}

impl ChunkConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: ChunkConfigV2::default(),
        }
    }

    pub fn strategy(mut self, strategy: ChunkingStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn overlap(mut self, overlap: usize) -> Self {
        self.config.overlap = overlap;
        self
    }

    pub fn mmap_threshold(mut self, threshold: Option<usize>) -> Self {
        self.config.use_mmap_threshold = threshold;
        self
    }

    pub fn parallel(mut self, parallel: bool) -> Self {
        self.config.parallel = parallel;
        self
    }

    pub fn build(self) -> ChunkConfigV2 {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_uniform_filter_chunked_v2() {
        let input = Array2::<f64>::ones((100, 100));
        let size = vec![3, 3];

        let config = ChunkConfigBuilder::new()
            .strategy(ChunkingStrategy::Fixed(1000))
            .build();

        let result =
            uniform_filter_chunked_v2(&input, &size, BorderMode::Constant, Some(config)).unwrap();

        assert_eq!(result.shape(), input.shape());

        // Check center values
        for i in 10..90 {
            for j in 10..90 {
                assert_abs_diff_eq!(result[[i, j]], 1.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_binary_op_chunked() {
        let array1 = Array2::<f64>::ones((50, 50));
        let array2 = Array2::<f64>::from_elem((50, 50), 2.0);

        let config = ChunkConfigBuilder::new()
            .strategy(ChunkingStrategy::NumChunks(4))
            .build();

        let result = binary_op_chunked_v2(&array1, &array2, |a, b| a + b, Some(config)).unwrap();

        assert_eq!(result.shape(), array1.shape());

        // All values should be 3.0
        for val in result.iter() {
            assert_abs_diff_eq!(*val, 3.0, epsilon = 1e-10);
        }
    }
}
