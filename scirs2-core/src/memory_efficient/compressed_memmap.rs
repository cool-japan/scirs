//! Compressed memory-mapped arrays.
//!
//! This module provides functionality for memory-mapping arrays with transparent
//! compression and decompression. This can significantly reduce disk space
//! requirements while maintaining the benefits of memory-mapping for large data.
//!
//! The implementation uses a block-based approach, where data is split into blocks
//! that are compressed independently. This allows for efficient random access and
//! partial loading of the data.

use crate::error::{CoreError, CoreResult, ErrorContext};
use ndarray::{Array, ArrayBase, Dimension, IxDyn, RawData};

#[cfg(feature = "memory_compression")]
use lz4::{Decoder, EncoderBuilder};

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Metadata for a compressed memory-mapped file
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompressedFileMetadata {
    /// Original array shape
    pub shape: Vec<usize>,

    /// Element type information (size in bytes)
    pub element_size: usize,

    /// Total number of elements in the array
    pub num_elements: usize,

    /// Block size in elements
    pub block_size: usize,

    /// Number of blocks
    pub num_blocks: usize,

    /// Offset of each block in the compressed file
    pub block_offsets: Vec<u64>,

    /// Compressed size of each block
    pub block_compressed_sizes: Vec<usize>,

    /// Uncompressed size of each block
    pub block_uncompressed_sizes: Vec<usize>,

    /// Compression algorithm used
    pub compression_algorithm: CompressionAlgorithm,

    /// Compression level
    pub compression_level: i32,

    /// Creation timestamp
    pub creation_time: chrono::DateTime<chrono::Utc>,

    /// Optional description
    pub description: Option<String>,
}

/// Supported compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CompressionAlgorithm {
    /// LZ4 compression
    #[default]
    Lz4,
    /// Zstd compression
    Zstd,
    /// Snappy compression
    Snappy,
}

/// Builder for compressed memory-mapped arrays.
#[derive(Debug, Clone)]
pub struct CompressedMemMapBuilder {
    /// Block size in elements
    block_size: usize,

    /// Compression algorithm
    algorithm: CompressionAlgorithm,

    /// Compression level
    level: i32,

    /// Maximum cache size in blocks
    cache_size: usize,

    /// Cache time-to-live in seconds
    cache_ttl: Option<Duration>,

    /// Description
    description: Option<String>,
}

impl Default for CompressedMemMapBuilder {
    fn default() -> Self {
        Self {
            block_size: 65536, // 64K elements by default
            algorithm: CompressionAlgorithm::Lz4,
            level: 1,                                  // Default compression level
            cache_size: 32,                            // Cache 32 blocks by default
            cache_ttl: Some(Duration::from_secs(300)), // 5 minute TTL
            description: None,
        }
    }
}

impl CompressedMemMapBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the block size in elements.
    ///
    /// Larger blocks provide better compression but slower random access.
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Set the compression algorithm.
    pub fn with_algorithm(mut self, algorithm: CompressionAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the compression level.
    ///
    /// Higher levels provide better compression but slower compression speed.
    /// The valid range depends on the algorithm.
    /// - LZ4: 1-12
    /// - Zstd: 1-22
    /// - Snappy: Ignores this parameter
    pub fn with_level(mut self, level: i32) -> Self {
        self.level = level;
        self
    }

    /// Set the maximum cache size in blocks.
    ///
    /// Larger cache sizes allow for more decompressed blocks to be held in memory,
    /// potentially improving performance for repeated access patterns.
    pub fn with_cache_size(mut self, cache_size: usize) -> Self {
        self.cache_size = cache_size;
        self
    }

    /// Set the cache time-to-live duration.
    ///
    /// Blocks will be evicted from cache after this duration if not accessed.
    /// Set to None for no time-based eviction.
    pub fn with_cache_ttl(mut self, ttl: Option<Duration>) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Set an optional description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Create a compressed memory-mapped array from existing data.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to compress
    /// * `path` - The file path to save the compressed data
    ///
    /// # Returns
    ///
    /// A compressed memory-mapped array
    pub fn create<A, S, D>(
        &self,
        data: &ArrayBase<S, D>,
        path: impl AsRef<Path>,
    ) -> CoreResult<CompressedMemMappedArray<A>>
    where
        A: Clone + Copy + 'static + Send + Sync,
        S: RawData<Elem = A>,
        D: Dimension,
    {
        // Get array information
        let shape = data.shape().to_vec();
        let num_elements = data.len();
        let element_size = std::mem::size_of::<A>();

        // Calculate block information
        let block_size = self.block_size.min(num_elements);
        let num_blocks = num_elements.div_ceil(block_size);

        // Prepare metadata
        let mut metadata = CompressedFileMetadata {
            shape,
            element_size,
            num_elements,
            block_size,
            num_blocks,
            block_offsets: Vec::with_capacity(num_blocks),
            block_compressed_sizes: Vec::with_capacity(num_blocks),
            block_uncompressed_sizes: Vec::with_capacity(num_blocks),
            compression_algorithm: self.algorithm,
            compression_level: self.level,
            creation_time: chrono::Utc::now(),
            description: self.description.clone(),
        };

        // Create output file
        let path = path.as_ref();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        // Reserve space for metadata (will write it after compression)
        let metadata_placeholder = vec![0u8; 1024];
        file.write_all(&metadata_placeholder)?;

        // Compress each block
        let data_ptr = data.as_ptr() as *const u8;
        let mut current_offset = metadata_placeholder.len() as u64;

        for block_idx in 0..num_blocks {
            let start_element = block_idx * block_size;
            let end_element = (start_element + block_size).min(num_elements);
            let block_elements = end_element - start_element;
            let uncompressed_size = block_elements * element_size;

            // Record the block offset
            metadata.block_offsets.push(current_offset);
            metadata.block_uncompressed_sizes.push(uncompressed_size);

            // Get the data for this block
            let data_offset = start_element * element_size;
            let block_data =
                unsafe { std::slice::from_raw_parts(data_ptr.add(data_offset), uncompressed_size) };

            // Compress the block
            let compressed_data = match self.algorithm {
                CompressionAlgorithm::Lz4 => {
                    let mut encoder = EncoderBuilder::new()
                        .level(self.level as u32)
                        .build(Vec::new())?;
                    encoder.write_all(block_data)?;
                    let (compressed, result) = encoder.finish();
                    result?;
                    compressed
                }
                CompressionAlgorithm::Zstd => zstd::encode_all(block_data, self.level)?,
                CompressionAlgorithm::Snappy => snap::raw::Encoder::new()
                    .compress_vec(block_data)
                    .map_err(|e| {
                        CoreError::ComputationError(ErrorContext::new(format!(
                            "Snappy compression error: {}",
                            e
                        )))
                    })?,
            };

            // Record the compressed size
            metadata.block_compressed_sizes.push(compressed_data.len());

            // Write the compressed block
            file.write_all(&compressed_data)?;

            // Update the offset for the next block
            current_offset += compressed_data.len() as u64;
        }

        // Write the metadata at the beginning of the file
        let metadata_json = serde_json::to_string(&metadata).map_err(|e| {
            CoreError::ValueError(ErrorContext::new(format!(
                "Failed to serialize metadata: {}",
                e
            )))
        })?;
        let mut metadata_bytes = metadata_json.into_bytes();

        // Ensure the metadata fits in the reserved space
        if metadata_bytes.len() > metadata_placeholder.len() {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "Metadata size ({} bytes) exceeds reserved space ({} bytes)",
                metadata_bytes.len(),
                metadata_placeholder.len()
            ))));
        }

        // Pad the metadata to the reserved size
        metadata_bytes.resize(metadata_placeholder.len(), 0);

        // Write the metadata
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&metadata_bytes)?;

        // Create and return the compressed memory-mapped array
        let compressed_mmap = CompressedMemMappedArray::open_impl(
            path.to_path_buf(),
            self.cache_size,
            self.cache_ttl,
        )?;

        Ok(compressed_mmap)
    }

    /// Create a compressed memory-mapped array from raw data.
    ///
    /// # Arguments
    ///
    /// * `data` - The raw data to compress
    /// * `shape` - The shape of the array
    /// * `path` - The file path to save the compressed data
    ///
    /// # Returns
    ///
    /// A compressed memory-mapped array
    pub fn create_from_raw<A>(
        &self,
        data: &[A],
        shape: &[usize],
        path: impl AsRef<Path>,
    ) -> CoreResult<CompressedMemMappedArray<A>>
    where
        A: Clone + Copy + 'static + Send + Sync,
    {
        // Create ndarray from raw data
        let array = Array::from_shape_vec(IxDyn(shape), data.to_vec())
            .map_err(|e| CoreError::ShapeError(ErrorContext::new(format!("{}", e))))?;

        // Create compressed memory-mapped array
        self.create(&array, path)
    }
}

/// A memory-mapped array with transparent compression.
///
/// This struct provides a view into a compressed array stored on disk,
/// with transparent decompression of blocks as they are accessed.
#[derive(Debug, Clone)]
pub struct CompressedMemMappedArray<A: Clone + Copy + 'static + Send + Sync> {
    /// Path to the compressed file
    path: PathBuf,

    /// Metadata about the compressed file
    metadata: CompressedFileMetadata,

    /// Block cache for decompressed blocks
    block_cache: Arc<BlockCache<A>>,

    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Clone + Copy + 'static + Send + Sync> CompressedMemMappedArray<A> {
    /// Open a compressed memory-mapped array from a file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the compressed file
    ///
    /// # Returns
    ///
    /// A compressed memory-mapped array
    pub fn open(path: impl AsRef<Path>) -> CoreResult<Self> {
        // Use default cache settings
        let cache_size = 32;
        let cache_ttl = Some(Duration::from_secs(300));

        Self::open_impl(path.as_ref().to_path_buf(), cache_size, cache_ttl)
    }

    /// Open a compressed memory-mapped array with custom cache settings.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the compressed file
    /// * `cache_size` - The maximum number of blocks to cache
    /// * `cache_ttl` - The time-to-live for cached blocks
    ///
    /// # Returns
    ///
    /// A compressed memory-mapped array
    pub fn open_with_cache(
        path: impl AsRef<Path>,
        cache_size: usize,
        cache_ttl: Option<Duration>,
    ) -> CoreResult<Self> {
        Self::open_impl(path.as_ref().to_path_buf(), cache_size, cache_ttl)
    }

    /// Internal implementation of open.
    fn open_impl(
        path: PathBuf,
        cache_size: usize,
        cache_ttl: Option<Duration>,
    ) -> CoreResult<Self> {
        // Open the file
        let mut file = File::open(&path)?;

        // Read the metadata from the beginning of the file
        let mut metadata_bytes = vec![0u8; 1024];
        file.read_exact(&mut metadata_bytes)?;

        // Parse the metadata
        // from_utf8_lossy doesn't return an error, it replaces invalid utf8 sequences
        let metadata_json = String::from_utf8_lossy(&metadata_bytes)
            .trim_end_matches('\0')
            .to_string();
        let metadata: CompressedFileMetadata =
            serde_json::from_str(&metadata_json).map_err(|e| {
                CoreError::ValueError(ErrorContext::new(format!(
                    "Failed to deserialize metadata: {}",
                    e
                )))
            })?;

        // Check that the element size matches
        let expected_element_size = std::mem::size_of::<A>();
        if metadata.element_size != expected_element_size {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "Element size mismatch: expected {}, got {}",
                expected_element_size, metadata.element_size
            ))));
        }

        // Create the block cache
        let block_cache = Arc::new(BlockCache::new(cache_size, cache_ttl));

        // Create and return the array
        Ok(Self {
            path,
            metadata,
            block_cache,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get the shape of the array.
    pub fn shape(&self) -> &[usize] {
        &self.metadata.shape
    }

    /// Get the total number of elements in the array.
    pub fn size(&self) -> usize {
        self.metadata.num_elements
    }

    /// Get the number of dimensions of the array.
    pub fn ndim(&self) -> usize {
        self.metadata.shape.len()
    }

    /// Get the metadata for the compressed file.
    pub fn metadata(&self) -> &CompressedFileMetadata {
        &self.metadata
    }

    /// Get the block size.
    pub fn block_size(&self) -> usize {
        self.metadata.block_size
    }

    /// Get the number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.metadata.num_blocks
    }

    /// Load a specific block into memory.
    ///
    /// This is useful for preloading blocks that will be accessed soon.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The index of the block to load
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or an error
    pub fn preload_block(&self, block_idx: usize) -> CoreResult<()> {
        if block_idx >= self.metadata.num_blocks {
            return Err(CoreError::IndexError(ErrorContext::new(format!(
                "Block index {} out of bounds (max {})",
                block_idx,
                self.metadata.num_blocks - 1
            ))));
        }

        // Check if the block is already cached
        if self.block_cache.has_block(block_idx) {
            return Ok(());
        }

        // Load and decompress the block
        let block = self.load_block(block_idx)?;

        // Add to cache
        self.block_cache.put_block(block_idx, block);

        Ok(())
    }

    /// Load a block from the compressed file.
    fn load_block(&self, block_idx: usize) -> CoreResult<Vec<A>> {
        // Open the file
        let mut file = File::open(&self.path)?;

        // Get block information
        let offset = self.metadata.block_offsets[block_idx];
        let compressed_size = self.metadata.block_compressed_sizes[block_idx];
        let uncompressed_size = self.metadata.block_uncompressed_sizes[block_idx];

        // Read the compressed block
        file.seek(SeekFrom::Start(offset))?;
        let mut compressed_data = vec![0u8; compressed_size];
        file.read_exact(&mut compressed_data)?;

        // Decompress the block
        let block_bytes = match self.metadata.compression_algorithm {
            CompressionAlgorithm::Lz4 => {
                let mut decoder = Decoder::new(&compressed_data[..])?;
                let mut decompressed = Vec::with_capacity(uncompressed_size);
                decoder.read_to_end(&mut decompressed)?;
                decompressed
            }
            CompressionAlgorithm::Zstd => zstd::decode_all(&compressed_data[..])?,
            CompressionAlgorithm::Snappy => snap::raw::Decoder::new()
                .decompress_vec(&compressed_data)
                .map_err(|e| {
                    CoreError::ComputationError(ErrorContext::new(format!(
                        "Snappy decompression error: {}",
                        e
                    )))
                })?,
        };

        // Check that we got the expected number of bytes
        if block_bytes.len() != uncompressed_size {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "Block {} decompressed to {} bytes, expected {}",
                block_idx,
                block_bytes.len(),
                uncompressed_size
            ))));
        }

        // Convert bytes to elements
        let num_elements = uncompressed_size / std::mem::size_of::<A>();
        let mut elements = Vec::with_capacity(num_elements);

        // Interpret bytes as elements (this is safe because A is Copy)
        for chunk in block_bytes.chunks_exact(std::mem::size_of::<A>()) {
            let element = unsafe { *(chunk.as_ptr() as *const A) };
            elements.push(element);
        }

        Ok(elements)
    }

    /// Get a read-only view of the entire array.
    ///
    /// This will decompress all blocks and load them into memory.
    ///
    /// # Returns
    ///
    /// A read-only ndarray view of the data
    pub fn readonly_array(&self) -> CoreResult<Array<A, IxDyn>> {
        // Allocate an array for the result
        let mut result = Array::from_elem(IxDyn(&self.metadata.shape), unsafe {
            std::mem::zeroed::<A>()
        });

        // Load each block and copy it into the result
        let mut offset = 0;
        for block_idx in 0..self.metadata.num_blocks {
            // Get the block (from cache if available)
            let block = match self.block_cache.get_block(block_idx) {
                Some(block) => block,
                None => {
                    let block = self.load_block(block_idx)?;
                    self.block_cache.put_block(block_idx, block.clone());
                    block
                }
            };

            // Copy the block into the result
            let start = offset;
            let end = (start + block.len()).min(self.metadata.num_elements);
            let slice = &mut result.as_slice_mut().unwrap()[start..end];
            slice.copy_from_slice(&block[..(end - start)]);

            // Update the offset
            offset = end;
        }

        Ok(result)
    }

    /// Get a specific element from the array.
    ///
    /// This will decompress only the block containing the element.
    ///
    /// # Arguments
    ///
    /// * `indices` - The indices of the element to get
    ///
    /// # Returns
    ///
    /// The element at the specified indices
    pub fn get(&self, indices: &[usize]) -> CoreResult<A> {
        // Check that the indices are valid
        if indices.len() != self.metadata.shape.len() {
            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Expected {} indices, got {}",
                self.metadata.shape.len(),
                indices.len()
            ))));
        }

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.metadata.shape[i] {
                return Err(CoreError::IndexError(ErrorContext::new(format!(
                    "Index {} out of bounds for dimension {} (max {})",
                    idx,
                    i,
                    self.metadata.shape[i] - 1
                ))));
            }
        }

        // Calculate flat index
        let mut flat_index = 0;
        let mut stride = 1;
        for i in (0..indices.len()).rev() {
            flat_index += indices[i] * stride;
            if i > 0 {
                stride *= self.metadata.shape[i];
            }
        }

        // Calculate block index
        let block_idx = flat_index / self.metadata.block_size;
        let block_offset = flat_index % self.metadata.block_size;

        // Get the block (from cache if available)
        let block = match self.block_cache.get_block(block_idx) {
            Some(block) => block,
            None => {
                let block = self.load_block(block_idx)?;
                self.block_cache.put_block(block_idx, block.clone());
                block
            }
        };

        // Get the element
        if block_offset < block.len() {
            Ok(block[block_offset])
        } else {
            Err(CoreError::IndexError(ErrorContext::new(format!(
                "Block offset {} out of bounds for block {} (max {})",
                block_offset,
                block_idx,
                block.len() - 1
            ))))
        }
    }

    /// Get a subset of the array as a new array.
    ///
    /// This will decompress only the blocks containing the subset.
    ///
    /// # Arguments
    ///
    /// * `ranges` - The ranges of indices to get for each dimension
    ///
    /// # Returns
    ///
    /// A new array containing the subset
    pub fn slice(&self, ranges: &[(usize, usize)]) -> CoreResult<Array<A, IxDyn>> {
        // Check that the ranges are valid
        if ranges.len() != self.metadata.shape.len() {
            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Expected {} ranges, got {}",
                self.metadata.shape.len(),
                ranges.len()
            ))));
        }

        // Calculate the shape of the result
        let mut result_shape = Vec::with_capacity(ranges.len());
        for (i, &(start, end)) in ranges.iter().enumerate() {
            if start >= end {
                return Err(CoreError::ValueError(ErrorContext::new(format!(
                    "Invalid range for dimension {}: {}..{}",
                    i, start, end
                ))));
            }
            if end > self.metadata.shape[i] {
                return Err(CoreError::IndexError(ErrorContext::new(format!(
                    "Range {}..{} out of bounds for dimension {} (max {})",
                    start, end, i, self.metadata.shape[i]
                ))));
            }
            result_shape.push(end - start);
        }

        // Allocate an array for the result
        let mut result = Array::from_elem(IxDyn(&result_shape), unsafe { std::mem::zeroed::<A>() });

        // Calculate the total number of elements in the result
        let result_size = result_shape.iter().product::<usize>();

        // Create iterators for the result indices
        let mut result_indices = vec![0; ranges.len()];
        let mut source_indices = Vec::with_capacity(ranges.len());
        for &(start, _) in ranges.iter() {
            source_indices.push(start);
        }

        // Iterate through all elements in the result
        for _result_flat_idx in 0..result_size {
            // Calculate source flat index
            let mut source_flat_idx = 0;
            let mut stride = 1;
            for i in (0..source_indices.len()).rev() {
                source_flat_idx += source_indices[i] * stride;
                if i > 0 {
                    stride *= self.metadata.shape[i];
                }
            }

            // Get the element from the source
            let block_idx = source_flat_idx / self.metadata.block_size;
            let block_offset = source_flat_idx % self.metadata.block_size;

            // Get the block (from cache if available)
            let block = match self.block_cache.get_block(block_idx) {
                Some(block) => block,
                None => {
                    let block = self.load_block(block_idx)?;
                    self.block_cache.put_block(block_idx, block.clone());
                    block
                }
            };

            // Get the element and set it in the result
            if block_offset < block.len() {
                // Calculate the result flat index
                let mut result_stride = 1;
                let mut result_flat_idx = 0;
                for i in (0..result_indices.len()).rev() {
                    result_flat_idx += result_indices[i] * result_stride;
                    if i > 0 {
                        result_stride *= result_shape[i];
                    }
                }

                // Set the element in the result
                let result_slice = result.as_slice_mut().unwrap();
                result_slice[result_flat_idx] = block[block_offset];
            }

            // Increment the indices
            for i in (0..ranges.len()).rev() {
                result_indices[i] += 1;
                source_indices[i] += 1;
                if result_indices[i] < result_shape[i] {
                    break;
                }
                result_indices[i] = 0;
                source_indices[i] = ranges[i].0;
            }
        }

        Ok(result)
    }

    /// Process the array in blocks, with each block loaded and decompressed on demand.
    ///
    /// This is useful for operations that can be performed on blocks independently.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that processes a block of elements
    ///
    /// # Returns
    ///
    /// A vector of results, one for each block
    pub fn process_blocks<F, R>(&self, f: F) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A], usize) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        self.process_blocks_internal(f, false, None)
    }

    /// Process the array in blocks with a custom block size.
    ///
    /// # Arguments
    ///
    /// * `block_size` - The block size to use (in elements)
    /// * `f` - A function that processes a block of elements
    ///
    /// # Returns
    ///
    /// A vector of results, one for each block
    pub fn process_blocks_with_size<F, R>(&self, block_size: usize, f: F) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A], usize) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        self.process_blocks_internal(f, false, Some(block_size))
    }

    /// Process the array in blocks in parallel.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that processes a block of elements
    ///
    /// # Returns
    ///
    /// A vector of results, one for each block
    #[cfg(feature = "parallel")]
    pub fn process_blocks_parallel<F, R>(&self, f: F) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A], usize) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        self.process_blocks_internal(f, true, None)
    }

    /// Process the array in blocks in parallel with a custom block size.
    ///
    /// # Arguments
    ///
    /// * `block_size` - The block size to use (in elements)
    /// * `f` - A function that processes a block of elements
    ///
    /// # Returns
    ///
    /// A vector of results, one for each block
    #[cfg(feature = "parallel")]
    pub fn process_blocks_parallel_with_size<F, R>(
        &self,
        block_size: usize,
        f: F,
    ) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A], usize) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        self.process_blocks_internal(f, true, Some(block_size))
    }

    /// Internal implementation of block processing.
    #[cfg(not(feature = "parallel"))]
    fn process_blocks_internal<F, R>(
        &self,
        f: F,
        _parallel: bool,
        custom_block_size: Option<usize>,
    ) -> CoreResult<Vec<R>>
    where
        F: FnMut(&[A], usize) -> R,
    {
        // Determine block layout
        let block_size = custom_block_size.unwrap_or(self.metadata.block_size);
        let num_elements = self.metadata.num_elements;
        let num_blocks = num_elements.div_ceil(block_size);

        // Serial processing
        (0..num_blocks)
            .map(|block_idx| {
                // Calculate the range of elements for this block
                let start = block_idx * block_size;
                let end = (start + block_size).min(num_elements);

                // Load the elements for this block
                let elements = self.load_elements(start, end)?;

                // Apply the function to the block
                Ok(f(&elements, block_idx))
            })
            .collect::<Result<Vec<R>, CoreError>>()
    }

    /// Internal implementation of block processing (parallel version).
    #[cfg(feature = "parallel")]
    fn process_blocks_internal<F, R>(
        &self,
        f: F,
        parallel: bool,
        custom_block_size: Option<usize>,
    ) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A], usize) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        // Determine block layout
        let block_size = custom_block_size.unwrap_or(self.metadata.block_size);
        let num_elements = self.metadata.num_elements;
        let num_blocks = num_elements.div_ceil(block_size);

        // Process blocks
        if parallel {
            use rayon::prelude::*;

            return (0..num_blocks)
                .into_par_iter()
                .map(|block_idx| {
                    // Calculate the range of elements for this block
                    let start = block_idx * block_size;
                    let end = (start + block_size).min(num_elements);

                    // Load the elements for this block
                    let elements = match self.load_elements(start, end) {
                        Ok(elems) => elems,
                        Err(e) => return Err(e),
                    };

                    // Apply the function to the block
                    Ok(f(&elements, block_idx))
                })
                .collect::<Result<Vec<R>, CoreError>>();
        }

        // Serial processing (used when parallel=false)
        (0..num_blocks)
            .map(|block_idx| {
                // Calculate the range of elements for this block
                let start = block_idx * block_size;
                let end = (start + block_size).min(num_elements);

                // Load the elements for this block
                let elements = self.load_elements(start, end)?;

                // Apply the function to the block
                Ok(f(&elements, block_idx))
            })
            .collect::<Result<Vec<R>, CoreError>>()
    }

    /// Load a range of elements from the array.
    ///
    /// This will decompress all blocks containing the range.
    ///
    /// # Arguments
    ///
    /// * `start` - The starting element index
    /// * `end` - The ending element index (exclusive)
    ///
    /// # Returns
    ///
    /// A vector of elements in the range
    fn load_elements(&self, start: usize, end: usize) -> CoreResult<Vec<A>> {
        // Check bounds
        if start >= self.metadata.num_elements {
            return Err(CoreError::IndexError(ErrorContext::new(format!(
                "Start index {} out of bounds (max {})",
                start,
                self.metadata.num_elements - 1
            ))));
        }
        if end > self.metadata.num_elements {
            return Err(CoreError::IndexError(ErrorContext::new(format!(
                "End index {} out of bounds (max {})",
                end, self.metadata.num_elements
            ))));
        }
        if start >= end {
            return Ok(Vec::new());
        }

        // Determine which blocks we need
        let start_block = start / self.metadata.block_size;
        let end_block = (end - 1) / self.metadata.block_size;

        // Allocate space for the result
        let mut result = Vec::with_capacity(end - start);

        // Load each required block
        for block_idx in start_block..=end_block {
            // Get the block (from cache if available)
            let block = match self.block_cache.get_block(block_idx) {
                Some(block) => block,
                None => {
                    let block = self.load_block(block_idx)?;
                    self.block_cache.put_block(block_idx, block.clone());
                    block
                }
            };

            // Calculate the range of elements we need from this block
            let block_start = block_idx * self.metadata.block_size;
            let block_end = block_start + block.len();

            let range_start = start.max(block_start) - block_start;
            let range_end = end.min(block_end) - block_start;

            // Copy the elements into the result
            result.extend_from_slice(&block[range_start..range_end]);
        }

        Ok(result)
    }
}

/// Cache for decompressed blocks.
///
/// This struct provides a LRU (Least Recently Used) cache for decompressed blocks.
#[derive(Debug)]
struct BlockCache<A: Clone + Copy + 'static + Send + Sync> {
    /// Maximum number of blocks to cache
    capacity: usize,

    /// Time-to-live for cached blocks
    ttl: Option<Duration>,

    /// Cache of decompressed blocks
    cache: RwLock<HashMap<usize, CachedBlock<A>>>,
}

/// A cached block with its timestamp.
#[derive(Debug, Clone)]
struct CachedBlock<A: Clone + Copy + 'static + Send + Sync> {
    /// The decompressed block data
    data: Vec<A>,

    /// The time when the block was last accessed
    timestamp: Instant,
}

impl<A: Clone + Copy + 'static + Send + Sync> BlockCache<A> {
    /// Create a new block cache.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The maximum number of blocks to cache
    /// * `ttl` - The time-to-live for cached blocks
    fn new(capacity: usize, ttl: Option<Duration>) -> Self {
        Self {
            capacity,
            ttl,
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Check if a block is in the cache.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The index of the block to check
    ///
    /// # Returns
    ///
    /// `true` if the block is in the cache, `false` otherwise
    fn has_block(&self, block_idx: usize) -> bool {
        let cache = self.cache.read().unwrap();

        // Check if the block is in the cache
        if let Some(cached) = cache.get(&block_idx) {
            // Check if the block has expired
            if let Some(ttl) = self.ttl {
                if cached.timestamp.elapsed() > ttl {
                    return false;
                }
            }

            true
        } else {
            false
        }
    }

    /// Get a block from the cache.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The index of the block to get
    ///
    /// # Returns
    ///
    /// The block if it is in the cache, `None` otherwise
    fn get_block(&self, block_idx: usize) -> Option<Vec<A>> {
        let mut cache = self.cache.write().unwrap();

        // Check if the block is in the cache
        if let Some(mut cached) = cache.remove(&block_idx) {
            // Check if the block has expired
            if let Some(ttl) = self.ttl {
                if cached.timestamp.elapsed() > ttl {
                    return None;
                }
            }

            // Update the timestamp
            cached.timestamp = Instant::now();

            // Put the block back in the cache
            let data = cached.data.clone();
            cache.insert(block_idx, cached);

            Some(data)
        } else {
            None
        }
    }

    /// Put a block in the cache.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The index of the block to put
    /// * `block` - The block data
    fn put_block(&self, block_idx: usize, block: Vec<A>) {
        let mut cache = self.cache.write().unwrap();

        // Check if we need to evict a block
        if cache.len() >= self.capacity && !cache.contains_key(&block_idx) {
            // Find the least recently used block
            if let Some(lru_idx) = cache
                .iter()
                .min_by_key(|(_, cached)| cached.timestamp)
                .map(|(idx, _)| *idx)
            {
                cache.remove(&lru_idx);
            }
        }

        // Add the block to the cache
        cache.insert(
            block_idx,
            CachedBlock {
                data: block,
                timestamp: Instant::now(),
            },
        );
    }

    /// Clear the cache.
    #[allow(dead_code)]
    fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }

    /// Get the number of blocks in the cache.
    #[allow(dead_code)]
    fn len(&self) -> usize {
        let cache = self.cache.read().unwrap();
        cache.len()
    }

    /// Check if the cache is empty.
    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        let cache = self.cache.read().unwrap();
        cache.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use tempfile::tempdir;

    #[test]
    fn test_compressed_memmapped_array_1d() {
        // Create a temporary directory for our test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_compressed_1d.cmm");

        // Create test data
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();

        // Create a builder
        let builder = CompressedMemMapBuilder::new()
            .with_block_size(100)
            .with_algorithm(CompressionAlgorithm::Lz4)
            .with_level(1)
            .with_cache_size(4)
            .with_description("Test 1D array");

        // Create the compressed memory-mapped array
        let cmm = builder.create_from_raw(&data, &[1000], &file_path).unwrap();

        // Check metadata
        assert_eq!(cmm.shape(), &[1000]);
        assert_eq!(cmm.size(), 1000);
        assert_eq!(cmm.ndim(), 1);

        // Test random access
        for i in 0..10 {
            let val = cmm.get(&[i * 100]).unwrap();
            assert_eq!(val, (i * 100) as f64);
        }

        // Test slicing
        let slice = cmm.slice(&[(200, 300)]).unwrap();
        assert_eq!(slice.shape(), &[100]);
        for i in 0..100 {
            assert_eq!(slice[ndarray::IxDyn(&[i])], (i + 200) as f64);
        }

        // Test block processing
        let sums = cmm
            .process_blocks(|block, _| block.iter().sum::<f64>())
            .unwrap();

        assert_eq!(sums.len(), 10); // 1000 elements / 100 block size = 10 blocks

        // Test loading the entire array
        let array = cmm.readonly_array().unwrap();
        assert_eq!(array.shape(), &[1000]);
        for i in 0..1000 {
            assert_eq!(array[ndarray::IxDyn(&[i])], i as f64);
        }
    }

    #[test]
    fn test_compressed_memmapped_array_2d() {
        // Create a temporary directory for our test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_compressed_2d.cmm");

        // Create test data - 10x10 matrix
        let data = Array2::<f64>::from_shape_fn((10, 10), |(i, j)| (i * 10 + j) as f64);

        // Create a builder
        let builder = CompressedMemMapBuilder::new()
            .with_block_size(25) // 5x5 blocks
            .with_algorithm(CompressionAlgorithm::Lz4)
            .with_level(1)
            .with_cache_size(4)
            .with_description("Test 2D array");

        // Create the compressed memory-mapped array
        let cmm = builder.create(&data, &file_path).unwrap();

        // Check metadata
        assert_eq!(cmm.shape(), &[10, 10]);
        assert_eq!(cmm.size(), 100);
        assert_eq!(cmm.ndim(), 2);

        // Test random access
        for i in 0..10 {
            for j in 0..10 {
                let val = cmm.get(&[i, j]).unwrap();
                assert_eq!(val, (i * 10 + j) as f64);
            }
        }

        // Test slicing
        let slice = cmm.slice(&[(2, 5), (3, 7)]).unwrap();
        assert_eq!(slice.shape(), &[3, 4]);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(
                    slice[ndarray::IxDyn(&[i, j])],
                    ((i + 2) * 10 + (j + 3)) as f64
                );
            }
        }

        // Test loading the entire array
        let array = cmm.readonly_array().unwrap();
        assert_eq!(array.shape(), &[10, 10]);
        for i in 0..10 {
            for j in 0..10 {
                assert_eq!(array[ndarray::IxDyn(&[i, j])], (i * 10 + j) as f64);
            }
        }
    }

    #[test]
    fn test_different_compression_algorithms() {
        // Create a temporary directory for our test files
        let dir = tempdir().unwrap();

        // Create test data
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();

        // Test each compression algorithm
        for algorithm in &[
            CompressionAlgorithm::Lz4,
            CompressionAlgorithm::Zstd,
            CompressionAlgorithm::Snappy,
        ] {
            let file_path = dir.path().join(format!("test_{:?}.cmm", algorithm));

            // Create a builder
            let builder = CompressedMemMapBuilder::new()
                .with_block_size(100)
                .with_algorithm(*algorithm)
                .with_level(1)
                .with_cache_size(4);

            // Create the compressed memory-mapped array
            let cmm = builder.create_from_raw(&data, &[1000], &file_path).unwrap();

            // Test loading the entire array
            let array = cmm.readonly_array().unwrap();
            for i in 0..1000 {
                assert_eq!(array[ndarray::IxDyn(&[i])], i as f64);
            }
        }
    }

    #[test]
    fn test_block_cache() {
        // Create a temporary directory for our test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_cache.cmm");

        // Create test data
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();

        // Create compressed arrays with different cache settings
        let small_cache = CompressedMemMapBuilder::new()
            .with_block_size(100)
            .with_cache_size(2) // Very small cache
            .create_from_raw(&data, &[1000], &file_path).unwrap();

        // Test cache behavior
        // First, load all blocks to fill the cache
        for i in 0..10 {
            small_cache.preload_block(i).unwrap();
        }

        // Check the cache size - should be 2 (capacity)
        assert_eq!(small_cache.block_cache.len(), 2);

        // Now access a block that's not in the cache
        // This should evict the least recently used block
        let val = small_cache.get(&[0]).unwrap(); // Block 0
        assert_eq!(val, 0.0);

        // Check that the block is now in the cache
        assert!(small_cache.block_cache.has_block(0));
    }

    #[test]
    fn test_block_preloading() {
        // Create a temporary directory for our test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_preload.cmm");

        // Create test data
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();

        // Create the compressed memory-mapped array
        let cmm = CompressedMemMapBuilder::new()
            .with_block_size(100)
            .create_from_raw(&data, &[1000], &file_path)
            .unwrap();

        // Preload a block
        cmm.preload_block(5).unwrap();

        // Check that the block is now in the cache
        assert!(cmm.block_cache.has_block(5));

        // Access an element from the preloaded block
        let val = cmm.get(&[550]).unwrap(); // In block 5
        assert_eq!(val, 550.0);
    }
}
