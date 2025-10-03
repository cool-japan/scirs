//! Memory-efficient operations for large arrays
//!
//! This module provides memory-efficient implementations of special functions
//! that process large arrays in chunks to avoid memory overflow and improve
//! cache efficiency.

use crate::error::{SpecialError, SpecialResult};
use scirs2_core::ndarray::{Array, ArrayView, ArrayViewMut, Ix1};
use scirs2_core::numeric::Float;
use std::marker::PhantomData;

/// Configuration for memory-efficient processing
#[derive(Debug, Clone)]
pub struct ChunkedConfig {
    /// Maximum chunk size in bytes
    pub max_chunk_bytes: usize,
    /// Whether to use parallel processing for chunks
    pub parallel_chunks: bool,
    /// Minimum array size to trigger chunking (in elements)
    pub min_arraysize: usize,
    /// Whether to prefetch next chunk while processing current
    pub prefetch: bool,
}

impl Default for ChunkedConfig {
    fn default() -> Self {
        Self {
            // Default to 64MB chunks
            max_chunk_bytes: 64 * 1024 * 1024,
            parallel_chunks: true,
            min_arraysize: 100_000,
            prefetch: true,
        }
    }
}

/// Trait for functions that can be applied in chunks
pub trait ChunkableFunction<T> {
    /// Apply the function to a chunk of data
    fn apply_chunk(
        &self,
        input: &ArrayView<T, Ix1>,
        output: &mut ArrayViewMut<T, Ix1>,
    ) -> SpecialResult<()>;

    /// Get the name of the function for logging/debugging
    fn name(&self) -> &str;
}

/// Chunked processor for memory-efficient array operations
pub struct ChunkedProcessor<T, F> {
    config: ChunkedConfig,
    function: F,
    _phantom: PhantomData<T>,
}

impl<T, F> ChunkedProcessor<T, F>
where
    T: Float + Send + Sync,
    F: ChunkableFunction<T> + Send + Sync,
{
    /// Create a new chunked processor
    pub fn new(config: ChunkedConfig, function: F) -> Self {
        Self {
            config,
            function,
            _phantom: PhantomData,
        }
    }

    /// Calculate optimal chunk size based on element size and config
    fn calculate_chunksize(&self, totalelements: usize) -> usize {
        let elementsize = std::mem::size_of::<T>();
        let max_elements = self.config.max_chunk_bytes / elementsize;

        // Don't chunk if array is small
        if totalelements < self.config.min_arraysize {
            return totalelements;
        }

        // Find a chunk size that divides evenly if possible
        let ideal_chunk = max_elements.min(totalelements);

        // Try to find a divisor of totalelements close to ideal_chunk
        for divisor in 1..=100 {
            let chunksize = totalelements / divisor;
            if chunksize <= ideal_chunk && totalelements.is_multiple_of(divisor) {
                return chunksize;
            }
        }

        ideal_chunk
    }

    /// Process a 1D array in chunks
    pub fn process_1d(
        &self,
        input: &Array<T, Ix1>,
        output: &mut Array<T, Ix1>,
    ) -> SpecialResult<()> {
        if input.len() != output.len() {
            return Err(SpecialError::ValueError(
                "Input and output arrays must have the same length".to_string(),
            ));
        }

        let totalelements = input.len();
        let chunksize = self.calculate_chunksize(totalelements);

        if chunksize == totalelements {
            // Process without chunking
            self.function
                .apply_chunk(&input.view(), &mut output.view_mut())?;
            return Ok(());
        }

        // Process in chunks
        if self.config.parallel_chunks {
            self.process_chunks_parallel(input, output, chunksize)
        } else {
            self.process_chunks_sequential(input, output, chunksize)
        }
    }

    /// Process chunks sequentially
    fn process_chunks_sequential(
        &self,
        input: &Array<T, Ix1>,
        output: &mut Array<T, Ix1>,
        chunksize: usize,
    ) -> SpecialResult<()> {
        let totalelements = input.len();
        let mut offset = 0;

        while offset < totalelements {
            let end = (offset + chunksize).min(totalelements);
            let input_chunk = input.slice(scirs2_core::ndarray::s![offset..end]);
            let mut output_chunk = output.slice_mut(scirs2_core::ndarray::s![offset..end]);

            self.function.apply_chunk(&input_chunk, &mut output_chunk)?;

            offset = end;
        }

        Ok(())
    }

    /// Process chunks in parallel
    #[cfg(feature = "parallel")]
    fn process_chunks_parallel(
        &self,
        input: &Array<T, Ix1>,
        output: &mut Array<T, Ix1>,
        chunksize: usize,
    ) -> SpecialResult<()> {
        use scirs2_core::parallel_ops::*;

        let totalelements = input.len();
        let num_chunks = (totalelements + chunksize - 1) / chunksize;

        // Collect chunk boundaries
        let chunks: Vec<(usize, usize)> = (0..num_chunks)
            .map(|i| {
                let start = i * chunksize;
                let end = ((i + 1) * chunksize).min(totalelements);
                (start, end)
            })
            .collect();

        // Use a different approach for parallel processing with mutable output
        // Split the output into non-overlapping chunks
        use scirs2_core::parallel_ops::IndexedParallelIterator;

        let results: Vec<_> = chunks
            .par_iter()
            .enumerate()
            .map(|(idx, (start, end))| {
                let input_chunk = input.slice(scirs2_core::ndarray::s![*start..*end]);
                let mut temp_output = Array::zeros(end - start);
                let mut temp_view = temp_output.view_mut();

                match self.function.apply_chunk(&input_chunk, &mut temp_view) {
                    Ok(_) => Ok((idx, temp_output)),
                    Err(e) => Err(e),
                }
            })
            .collect();

        // Copy results back to output
        for result in results {
            match result {
                Ok((idx, temp_output)) => {
                    let (start, end) = chunks[idx];
                    output
                        .slice_mut(scirs2_core::ndarray::s![start..end])
                        .assign(&temp_output);
                }
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    fn process_chunks_parallel(
        &self,
        input: &Array<T, Ix1>,
        output: &mut Array<T, Ix1>,
        chunksize: usize,
    ) -> SpecialResult<()> {
        // Fall back to sequential processing
        self.process_chunks_sequential(input, output, chunksize)
    }
}

/// Gamma function that can be applied in chunks
pub struct ChunkedGamma;

impl Default for ChunkedGamma {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkedGamma {
    pub fn new() -> Self {
        Self
    }
}

impl<T> ChunkableFunction<T> for ChunkedGamma
where
    T: Float + scirs2_core::numeric::FromPrimitive + std::fmt::Debug + std::ops::AddAssign,
{
    fn apply_chunk(
        &self,
        input: &ArrayView<T, Ix1>,
        output: &mut ArrayViewMut<T, Ix1>,
    ) -> SpecialResult<()> {
        use crate::gamma::gamma;

        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = gamma(*inp);
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "gamma"
    }
}

/// Bessel J0 function that can be applied in chunks
pub struct ChunkedBesselJ0;

impl Default for ChunkedBesselJ0 {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkedBesselJ0 {
    pub fn new() -> Self {
        Self
    }
}

impl<T> ChunkableFunction<T> for ChunkedBesselJ0
where
    T: Float + scirs2_core::numeric::FromPrimitive + std::fmt::Debug,
{
    fn apply_chunk(
        &self,
        input: &ArrayView<T, Ix1>,
        output: &mut ArrayViewMut<T, Ix1>,
    ) -> SpecialResult<()> {
        use crate::bessel::j0;

        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = j0(*inp);
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "bessel_j0"
    }
}

/// Error function that can be applied in chunks
pub struct ChunkedErf;

impl Default for ChunkedErf {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkedErf {
    pub fn new() -> Self {
        Self
    }
}

impl<T> ChunkableFunction<T> for ChunkedErf
where
    T: Float + scirs2_core::numeric::FromPrimitive,
{
    fn apply_chunk(
        &self,
        input: &ArrayView<T, Ix1>,
        output: &mut ArrayViewMut<T, Ix1>,
    ) -> SpecialResult<()> {
        use crate::erf::erf;

        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = erf(*inp);
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "erf"
    }
}

/// Convenience functions for memory-efficient processing
/// Process gamma function on large arrays with automatic chunking
#[allow(dead_code)]
pub fn gamma_chunked<T>(
    input: &Array<T, Ix1>,
    config: Option<ChunkedConfig>,
) -> SpecialResult<Array<T, Ix1>>
where
    T: Float
        + scirs2_core::numeric::FromPrimitive
        + std::fmt::Debug
        + std::ops::AddAssign
        + Send
        + Sync,
{
    let config = config.unwrap_or_default();
    let processor = ChunkedProcessor::new(config, ChunkedGamma::new());
    let mut output = Array::zeros(input.raw_dim());
    processor.process_1d(input, &mut output)?;
    Ok(output)
}

/// Process Bessel J0 function on large arrays with automatic chunking
#[allow(dead_code)]
pub fn j0_chunked<T>(
    input: &Array<T, Ix1>,
    config: Option<ChunkedConfig>,
) -> SpecialResult<Array<T, Ix1>>
where
    T: Float + scirs2_core::numeric::FromPrimitive + std::fmt::Debug + Send + Sync,
{
    let config = config.unwrap_or_default();
    let processor = ChunkedProcessor::new(config, ChunkedBesselJ0::new());
    let mut output = Array::zeros(input.raw_dim());
    processor.process_1d(input, &mut output)?;
    Ok(output)
}

/// Process error function on large arrays with automatic chunking
#[allow(dead_code)]
pub fn erf_chunked<T>(
    input: &Array<T, Ix1>,
    config: Option<ChunkedConfig>,
) -> SpecialResult<Array<T, Ix1>>
where
    T: Float + scirs2_core::numeric::FromPrimitive + Send + Sync,
{
    let config = config.unwrap_or_default();
    let processor = ChunkedProcessor::new(config, ChunkedErf::new());
    let mut output = Array::zeros(input.raw_dim());
    processor.process_1d(input, &mut output)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_chunksize_calculation() {
        let config = ChunkedConfig::default();
        let processor: ChunkedProcessor<f64, ChunkedGamma> =
            ChunkedProcessor::new(config, ChunkedGamma::new());

        // Small array - no chunking
        assert_eq!(processor.calculate_chunksize(1000), 1000);

        // Large array - should chunk
        let chunksize = processor.calculate_chunksize(10_000_000);
        assert!(chunksize < 10_000_000);
        assert!(chunksize > 0);
    }

    #[test]
    fn test_gamma_chunked() {
        let input = Array1::linspace(0.1, 5.0, 1000);
        let result = gamma_chunked(&input, None).unwrap();

        // Verify against direct computation
        use crate::gamma::gamma;
        for i in 0..1000 {
            assert!((result[i] - gamma(input[i])).abs() < 1e-10);
        }
    }

    #[test]
    fn test_chunked_with_custom_config() {
        let config = ChunkedConfig {
            max_chunk_bytes: 1024, // Very small chunks for testing
            parallel_chunks: false,
            min_arraysize: 10,
            prefetch: false,
        };

        let input = Array1::linspace(0.1, 5.0, 100);
        let result = gamma_chunked(&input, Some(config)).unwrap();

        // Verify correctness
        use crate::gamma::gamma;
        for i in 0..100 {
            assert!((result[i] - gamma(input[i])).abs() < 1e-10);
        }
    }
}
