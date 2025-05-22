use super::chunked::{ChunkedArray, ChunkingStrategy, OPTIMAL_CHUNK_SIZE};
use super::validation;
use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use bincode::{deserialize, serialize};
use ndarray::{Array, ArrayBase, Data, Dimension, Ix1, Ix2, IxDyn, ShapeBuilder};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

/// An array that stores data on disk to reduce memory usage
#[derive(Debug)]
pub struct OutOfCoreArray<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    /// The shape of the array
    pub shape: Vec<usize>,
    /// The path to the file containing the data
    pub file_path: PathBuf,
    /// The chunking strategy used for reading/writing
    pub strategy: ChunkingStrategy,
    /// The total number of elements
    pub size: usize,
    /// Whether the file is temporary and should be deleted on drop
    is_temp: bool,
    /// Phantom data for type parameters
    _phantom: PhantomData<(A, D)>,
}

impl<A, D> OutOfCoreArray<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    /// Create a new out-of-core array with the given data and file path
    pub fn new<S>(
        data: &ArrayBase<S, D>,
        file_path: &Path,
        strategy: ChunkingStrategy,
    ) -> Result<Self, CoreError>
    where
        S: Data<Elem = A>,
    {
        validation::check_not_empty(data)?;

        let shape = data.shape().to_vec();
        let size = data.len();

        // Create file and write data
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(file_path)
            .map_err(|e| CoreError::IoError(e))?;

        // Serialize data to file (in chunks if data is large)
        let _chunked = ChunkedArray::new(data.to_owned(), strategy);

        // Note: This is a simplified implementation that writes the entire array at once.
        // A real implementation would write chunks to save memory.
        let serialized = serialize(&data.to_owned()).map_err(|e| {
            CoreError::ValidationError(
                ErrorContext::new(format!("Failed to serialize data: {}", e))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        file.write_all(&serialized)
            .map_err(|e| CoreError::IoError(e))?;

        Ok(Self {
            shape,
            file_path: file_path.to_path_buf(),
            strategy,
            size,
            is_temp: false,
            _phantom: PhantomData,
        })
    }

    /// Create a new out-of-core array with a temporary file
    pub fn new_temp<S>(
        data: &ArrayBase<S, D>,
        strategy: ChunkingStrategy,
    ) -> Result<Self, CoreError>
    where
        S: Data<Elem = A>,
    {
        let temp_file = NamedTempFile::new().map_err(|e| CoreError::IoError(e))?;
        let file_path = temp_file.path().to_path_buf();

        // Manually persist the temp file so it stays around after we return
        let _file = temp_file.persist(&file_path).map_err(|e| {
            CoreError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?;

        let mut result = Self::new(data, &file_path, strategy)?;
        result.is_temp = true;

        Ok(result)
    }

    /// Load the entire array into memory
    pub fn load(&self) -> Result<Array<A, D>, CoreError> {
        let mut file = File::open(&self.file_path).map_err(|e| CoreError::IoError(e))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| CoreError::IoError(e))?;

        let array: Array<A, D> = deserialize(&buffer).map_err(|e| {
            CoreError::ValidationError(
                ErrorContext::new(format!("Failed to deserialize data: {}", e))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        Ok(array)
    }

    /// Load a chunk of the array into memory
    pub fn load_chunk(&self, chunk_index: usize) -> Result<Array<A, D>, CoreError> {
        if chunk_index >= self.num_chunks() {
            return Err(CoreError::IndexError(
                ErrorContext::new(format!(
                    "Chunk index out of bounds: {} >= {}",
                    chunk_index,
                    self.num_chunks()
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Calculate chunk size and offsets
        let chunk_size = self.get_chunk_size();
        let total_size = self.size;

        // Calculate start and end indices for this chunk
        let start_idx = chunk_index * chunk_size;
        let end_idx = std::cmp::min((chunk_index + 1) * chunk_size, total_size);
        let actual_chunk_size = end_idx - start_idx;

        // Open the file
        let mut file = File::open(&self.file_path).map_err(|e| CoreError::IoError(e))?;

        // Read the header to get overall structure
        let mut header_buf = Vec::new();
        // Read the first 1KB to extract metadata (adjust if needed)
        file.read_to_end(&mut header_buf)
            .map_err(|e| CoreError::IoError(e))?;

        // Deserialize the whole array for now
        // Note: In a more efficient implementation, we would:
        // 1. Store metadata separately in the file header
        // 2. Use custom serialization to write chunks sequentially
        // 3. Keep track of chunk offsets in the file
        let full_array: Array<A, D> = deserialize(&header_buf).map_err(|e| {
            CoreError::ValidationError(
                ErrorContext::new(format!("Failed to deserialize data: {}", e))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        // For now, extract the chunk from the full array
        // In a real implementation, we would seek to the correct position and read only the chunk

        // Create a new array with the proper shape for this chunk
        let mut chunk_shape = self.shape.clone();

        // Adjust the first dimension for this chunk
        if !chunk_shape.is_empty() {
            let first_dim_size = self.size / chunk_shape.iter().skip(1).product::<usize>().max(1);
            let first_dim_chunk_size =
                actual_chunk_size / chunk_shape.iter().skip(1).product::<usize>().max(1);
            chunk_shape[0] = first_dim_chunk_size.min(first_dim_size);
        }

        // Extract just this chunk's data from the full array
        // This is inefficient; a better implementation would read directly from disk
        let chunk = full_array.clone().into_shape(chunk_shape).map_err(|e| {
            CoreError::DimensionError(
                ErrorContext::new(format!("Failed to reshape chunk: {}", e))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        Ok(chunk)
    }

    /// Get the chunk size based on the chunking strategy
    fn get_chunk_size(&self) -> usize {
        match self.strategy {
            ChunkingStrategy::Auto => OPTIMAL_CHUNK_SIZE / std::mem::size_of::<A>(),
            ChunkingStrategy::Fixed(size) => size,
            ChunkingStrategy::FixedBytes(bytes) => bytes / std::mem::size_of::<A>(),
            ChunkingStrategy::NumChunks(n) => (self.size + n - 1) / n,
        }
    }

    /// Get the number of chunks
    pub fn num_chunks(&self) -> usize {
        let chunk_size = match self.strategy {
            ChunkingStrategy::Auto => OPTIMAL_CHUNK_SIZE / std::mem::size_of::<A>(),
            ChunkingStrategy::Fixed(size) => size,
            ChunkingStrategy::FixedBytes(bytes) => bytes / std::mem::size_of::<A>(),
            ChunkingStrategy::NumChunks(n) => (self.size + n - 1) / n,
        };

        (self.size + chunk_size - 1) / chunk_size
    }

    /// Check if the array is temporary
    pub fn is_temp(&self) -> bool {
        self.is_temp
    }

    /// Apply a function to each chunk of the array
    pub fn map<F, B, R>(&self, mut f: F) -> Result<R, CoreError>
    where
        F: FnMut(Array<A, D>) -> B,
        R: FromIterator<B>,
    {
        // Get the total number of chunks
        let num_chunks = self.num_chunks();

        if num_chunks == 0 {
            return Err(CoreError::ValueError(
                ErrorContext::new("Cannot map over an empty array".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Process each chunk and collect the results
        let mut results = Vec::with_capacity(num_chunks);

        for chunk_idx in 0..num_chunks {
            // Load the current chunk
            let chunk = self.load_chunk(chunk_idx)?;

            // Apply the function to the chunk and collect the result
            let result = f(chunk);
            results.push(result);
        }

        // Combine all results
        Ok(results.into_iter().collect())
    }

    /// Apply a function to each chunk of the array in parallel
    #[cfg(feature = "parallel")]
    pub fn par_map<F, B, R>(&self, f: F) -> Result<R, CoreError>
    where
        F: Fn(Array<A, D>) -> B + Send + Sync,
        B: Send,
        R: FromIterator<B> + Send,
        A: Send,
    {
        use rayon::prelude::*;

        // Get the total number of chunks
        let num_chunks = self.num_chunks();

        if num_chunks == 0 {
            return Err(CoreError::ValueError(
                ErrorContext::new("Cannot map over an empty array".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Create an iterator of chunk indices
        let chunk_indices: Vec<usize> = (0..num_chunks).collect();

        // Process each chunk in parallel
        let results: Result<Vec<B>, CoreError> = chunk_indices
            .into_par_iter()
            .map(|chunk_idx| {
                // Load the current chunk
                let chunk = self.load_chunk(chunk_idx)?;

                // Apply the function to the chunk and collect the result
                Ok(f(chunk))
            })
            .collect();

        // Combine all results
        results.map(|vec| vec.into_iter().collect())
    }
}

impl<A, D> Drop for OutOfCoreArray<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    fn drop(&mut self) {
        if self.is_temp {
            // Attempt to remove the temporary file
            let _ = std::fs::remove_file(&self.file_path);
        }
    }
}

/// A specialized out-of-core array that uses memory mapping for efficient access
#[derive(Debug)]
pub struct DiskBackedArray<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    /// The underlying out-of-core array
    pub array: OutOfCoreArray<A, D>,
    /// Whether the array is read-only
    pub read_only: bool,
}

impl<A, D> DiskBackedArray<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    /// Create a new disk-backed array with the given data and file path
    pub fn new<S>(
        data: &ArrayBase<S, D>,
        file_path: &Path,
        strategy: ChunkingStrategy,
        read_only: bool,
    ) -> Result<Self, CoreError>
    where
        S: Data<Elem = A>,
    {
        let array = OutOfCoreArray::new(data, file_path, strategy)?;

        Ok(Self { array, read_only })
    }

    /// Create a new disk-backed array with a temporary file
    pub fn new_temp<S>(
        data: &ArrayBase<S, D>,
        strategy: ChunkingStrategy,
        read_only: bool,
    ) -> Result<Self, CoreError>
    where
        S: Data<Elem = A>,
    {
        let array = OutOfCoreArray::new_temp(data, strategy)?;

        Ok(Self { array, read_only })
    }

    /// Load the entire array into memory
    pub fn load(&self) -> Result<Array<A, D>, CoreError> {
        self.array.load()
    }

    /// Check if the array is temporary
    pub fn is_temp(&self) -> bool {
        self.array.is_temp()
    }
}

/// Create a disk-backed array from the given data and file path
pub fn create_disk_array<A, S, D>(
    data: &ArrayBase<S, D>,
    file_path: &Path,
    strategy: ChunkingStrategy,
    read_only: bool,
) -> Result<DiskBackedArray<A, D>, CoreError>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    S: Data<Elem = A>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    DiskBackedArray::new(data, file_path, strategy, read_only)
}

/// Load chunks of an out-of-core array into memory
pub fn load_chunks<A, D>(
    array: &OutOfCoreArray<A, D>,
    chunk_indices: &[usize],
) -> Result<Vec<Array<A, D>>, CoreError>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    let num_chunks = array.num_chunks();

    // Validate chunk indices
    for &idx in chunk_indices {
        if idx >= num_chunks {
            return Err(CoreError::IndexError(
                ErrorContext::new(format!(
                    "Chunk index out of bounds: {} >= {}",
                    idx, num_chunks
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
    }

    // If no chunks are requested, return empty vector
    if chunk_indices.is_empty() {
        return Ok(Vec::new());
    }

    // Sort chunk indices to potentially optimize file reading
    // (this would matter more in a real implementation with proper chunk storage)
    let mut sorted_indices: Vec<usize> = chunk_indices.to_vec();
    sorted_indices.sort_unstable();

    // Remove duplicates
    sorted_indices.dedup();

    // Create a mapping from sorted indices to original positions
    let mut index_map = Vec::with_capacity(chunk_indices.len());
    for &idx in chunk_indices {
        index_map.push(sorted_indices.iter().position(|&x| x == idx).unwrap());
    }

    // Load chunks in optimal order
    let mut sorted_chunks = Vec::with_capacity(sorted_indices.len());
    for &idx in &sorted_indices {
        sorted_chunks.push(array.load_chunk(idx)?);
    }

    // Rearrange chunks to match the original requested order
    let mut result = Vec::with_capacity(chunk_indices.len());
    for &pos in &index_map {
        result.push(sorted_chunks[pos].clone());
    }

    Ok(result)
}
