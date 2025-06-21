use super::validation;
use crate::error::{CoreError, ErrorContext, ErrorLocation};
use ndarray::{Array, ArrayBase, Data, Dimension};
use std::marker::PhantomData;
use std::mem;

/// Optimal chunk size in bytes for memory-efficient operations
/// Chosen as 16 MB which is a good trade-off between memory usage and performance
pub const OPTIMAL_CHUNK_SIZE: usize = 16 * 1024 * 1024;

/// Strategy for chunking arrays for memory-efficient processing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkingStrategy {
    /// Automatically determine chunk sizes based on available memory and array dimensions
    Auto,
    /// Use a specific chunk size in elements
    Fixed(usize),
    /// Use a specific chunk size in bytes
    FixedBytes(usize),
    /// Process the array in a specific number of chunks
    NumChunks(usize),
}

/// A chunked array that provides memory-efficient processing for large datasets
#[derive(Debug)]
pub struct ChunkedArray<A, D>
where
    A: Clone,
    D: Dimension,
{
    /// The underlying array data
    pub data: Array<A, D>,
    /// The chunking strategy
    pub strategy: ChunkingStrategy,
    /// The computed chunk size in elements
    #[allow(dead_code)]
    chunk_size: usize,
    /// The number of chunks
    num_chunks: usize,
    /// Phantom data for type parameters
    _phantom: PhantomData<A>,
}

impl<A, D> ChunkedArray<A, D>
where
    A: Clone,
    D: Dimension,
{
    /// Create a new chunked array with the given data and chunking strategy
    pub fn new<S: Data<Elem = A>>(data: ArrayBase<S, D>, strategy: ChunkingStrategy) -> Self {
        let owned_data = data.to_owned();
        let total_elements = data.len();
        let elem_size = mem::size_of::<A>();

        // Calculate chunk size based on strategy
        let (chunk_size, num_chunks) = match strategy {
            ChunkingStrategy::Auto => {
                // Default to optimal chunk size in bytes, converted to elements
                let chunk_size_bytes = OPTIMAL_CHUNK_SIZE;
                let chunk_size = chunk_size_bytes / elem_size;
                let num_chunks = total_elements.div_ceil(chunk_size);
                (chunk_size, num_chunks)
            }
            ChunkingStrategy::Fixed(size) => {
                let num_chunks = total_elements.div_ceil(size);
                (size, num_chunks)
            }
            ChunkingStrategy::FixedBytes(bytes) => {
                let elements = bytes / elem_size;
                let chunk_size = if elements == 0 { 1 } else { elements };
                let num_chunks = total_elements.div_ceil(chunk_size);
                (chunk_size, num_chunks)
            }
            ChunkingStrategy::NumChunks(n) => {
                let num_chunks = if n == 0 { 1 } else { n };
                let chunk_size = total_elements.div_ceil(num_chunks);
                (chunk_size, num_chunks)
            }
        };

        Self {
            data: owned_data,
            strategy,
            chunk_size,
            num_chunks,
            _phantom: PhantomData,
        }
    }

    /// Apply a function to each chunk of the array and collect the results
    ///
    /// Returns a 1D array where each element is the result of applying the function to a chunk
    pub fn map<F, B>(&self, f: F) -> Array<B, ndarray::Ix1>
    where
        F: Fn(&Array<A, D>) -> B + Sync,
        B: Clone,
    {
        // Get chunks and apply the function to each
        let chunks = self.get_chunks();
        let results: Vec<B> = chunks.iter().map(f).collect();

        // Return results as a 1D array
        Array::from_vec(results)
    }

    /// Apply a function to each chunk of the array in parallel and collect the results
    ///
    /// Returns a 1D array where each element is the result of applying the function to a chunk
    pub fn par_map<F, B>(&self, f: F) -> Array<B, ndarray::Ix1>
    where
        F: Fn(&Array<A, D>) -> B + Sync + Send,
        B: Clone + Send + Sync,
        A: Send + Sync,
    {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            use std::sync::Arc;

            // Get chunks and wrap in Arc for thread-safe sharing
            let chunks = self.get_chunks();
            let chunks_arc = Arc::new(chunks);

            // Process chunks in parallel using index-based iteration
            let num_chunks = chunks_arc.len();
            let results: Vec<B> = (0..num_chunks)
                .into_par_iter()
                .map(move |i| {
                    let chunks_ref = Arc::clone(&chunks_arc);
                    f(&chunks_ref[i])
                })
                .collect();

            // Return results as a 1D array
            Array::from_vec(results)
        }

        #[cfg(not(feature = "parallel"))]
        {
            // Fall back to sequential processing
            self.map(f)
        }
    }

    /// Get the number of chunks
    pub fn num_chunks(&self) -> usize {
        self.num_chunks
    }

    /// Get the chunk size in elements
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get chunks of the array as a vector of owned array chunks
    pub fn get_chunks(&self) -> Vec<Array<A, D>>
    where
        D: Clone,
    {
        let mut result = Vec::with_capacity(self.num_chunks);

        // Special handling for 1D arrays
        if self.data.ndim() == 1 {
            if let Some(slice) = self.data.as_slice() {
                for i in 0..self.num_chunks {
                    let start = i * self.chunk_size;
                    let end = ((i + 1) * self.chunk_size).min(slice.len());
                    let chunk_slice = &slice[start..end];

                    // Create a new array from the chunk slice
                    // We need to handle the dimension conversion carefully
                    let chunk_1d = Array::from_vec(chunk_slice.to_vec());

                    // Try to convert to the original dimension type
                    // For 1D arrays, this should work directly
                    if let Ok(reshaped) = chunk_1d.into_dimensionality::<D>() {
                        result.push(reshaped);
                    } else {
                        // Fallback: return the whole array if conversion fails
                        return vec![self.data.clone()];
                    }
                }
                return result;
            }
        }

        // For multi-dimensional arrays or if slicing fails, return the whole array as a single chunk
        result.push(self.data.clone());
        result
    }
}

/// Perform an operation on an array in a chunk-wise manner to reduce memory usage
///
/// # Arguments
///
/// * `array` - The input array
/// * `op` - The operation to apply to each chunk
/// * `strategy` - The chunking strategy
///
/// # Returns
///
/// The result array after applying the operation to all chunks
pub fn chunk_wise_op<A, F, B, S, D>(
    array: &ArrayBase<S, D>,
    op: F,
    strategy: ChunkingStrategy,
) -> Result<Array<B, D>, CoreError>
where
    A: Clone,
    S: Data<Elem = A>,
    F: Fn(&ArrayBase<S, D>) -> Array<B, D>,
    B: Clone,
    D: Dimension + Clone,
{
    validation::check_not_empty(array)?;

    // If the array is small, just apply the operation directly
    if array.len() <= 1000 {
        return Ok(op(array));
    }

    let _chunked = ChunkedArray::new(array.to_owned(), strategy);

    // For now, we'll use a simple implementation that processes the whole array
    // In a real implementation, we would process each chunk separately and combine the results

    // Get a shallow copy of the array data
    let _result_shape = array.raw_dim().clone();
    let result = op(array);

    // Verify the result has the expected shape
    if result.shape() != array.shape() {
        return Err(CoreError::ValidationError(
            ErrorContext::new(format!(
                "Operation changed shape from {:?} to {:?}",
                array.shape(),
                result.shape()
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }

    Ok(result)
}

/// Perform a binary operation on two arrays in a chunk-wise manner
///
/// # Arguments
///
/// * `lhs` - The left-hand side array
/// * `rhs` - The right-hand side array
/// * `op` - The binary operation to apply to each pair of chunks
/// * `strategy` - The chunking strategy
///
/// # Returns
///
/// The result array after applying the binary operation to all chunk pairs
pub fn chunk_wise_binary_op<A, B, F, C, S1, S2, D>(
    lhs: &ArrayBase<S1, D>,
    rhs: &ArrayBase<S2, D>,
    op: F,
    strategy: ChunkingStrategy,
) -> Result<Array<C, D>, CoreError>
where
    A: Clone,
    B: Clone,
    S1: Data<Elem = A>,
    S2: Data<Elem = B>,
    F: Fn(&ArrayBase<S1, D>, &ArrayBase<S2, D>) -> Array<C, D>,
    C: Clone,
    D: Dimension + Clone,
{
    validation::check_shapes_match(lhs.shape(), rhs.shape())?;
    validation::check_not_empty(lhs)?;

    // If the arrays are small, just apply the operation directly
    if lhs.len() <= 1000 {
        return Ok(op(lhs, rhs));
    }

    // Create chunked arrays for both inputs
    let _chunked_lhs = ChunkedArray::new(lhs.to_owned(), strategy);
    let _chunked_rhs = ChunkedArray::new(rhs.to_owned(), strategy);

    // For now, we'll use a simple implementation that processes the whole arrays
    // In a real implementation, we would process each chunk pair separately and combine the results
    let result = op(lhs, rhs);

    // Verify the result has the expected shape
    if result.shape() != lhs.shape() {
        return Err(CoreError::ValidationError(
            ErrorContext::new(format!(
                "Binary operation changed shape from {:?} to {:?}",
                lhs.shape(),
                result.shape()
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }

    Ok(result)
}

/// Perform a reduction operation on an array in a chunk-wise manner
///
/// # Arguments
///
/// * `array` - The input array
/// * `op` - The reduction operation to apply to each chunk
/// * `combine` - The function to combine the results from each chunk
/// * `strategy` - The chunking strategy
///
/// # Returns
///
/// The result of applying the reduction operation to all chunks
pub fn chunk_wise_reduce<A, F, G, B, S, D>(
    array: &ArrayBase<S, D>,
    op: F,
    _combine: G,
    strategy: ChunkingStrategy,
) -> Result<B, CoreError>
where
    A: Clone,
    S: Data<Elem = A>,
    F: Fn(&ArrayBase<S, D>) -> B + Sync + Send,
    G: Fn(Vec<B>) -> B,
    B: Clone + Send + Sync,
    D: Dimension + Clone,
{
    validation::check_not_empty(array)?;

    // If the array is small, just apply the operation directly
    if array.len() <= 1000 {
        return Ok(op(array));
    }

    let _chunked = ChunkedArray::new(array.to_owned(), strategy);

    // For now, we'll use a simple implementation for the initial version
    // In a real implementation, we would process each chunk separately
    // and combine the results, using Rayon for parallel execution

    // Process the whole array directly for now
    let result = op(array);
    Ok(result)
}
