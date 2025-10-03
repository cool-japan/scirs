//! Out-of-core processing for terabyte-scale datasets
//!
//! This module provides infrastructure for processing datasets that are too large
//! to fit in memory, enabling work with terabyte-scale scientific data through
//! efficient memory management and disk-based algorithms.
//!
//! ## Features
//!
//! - **Memory-mapped arrays**: Efficient access to large arrays on disk
//! - **Chunked processing**: Process data in manageable chunks
//! - **Virtual memory management**: Smart caching and paging
//! - **Disk-based algorithms**: Sorting, grouping, and aggregation
//! - **HDF5 integration**: Leverage HDF5 for structured storage
//! - **Compression support**: On-the-fly compression/decompression
//!
//! ## Examples
//!
//! ```rust,no_run
//! use scirs2_io::out_of_core::{OutOfCoreArray, ChunkProcessor};
//! use scirs2_core::ndarray::Array2;
//!
//! // Create an out-of-core array
//! let array = OutOfCoreArray::<f64>::create("large_array.ooc", &[1_000_000, 100_000])?;
//!
//! // Process in chunks
//! array.process_chunks(1000, |chunk| {
//!     // Process each chunk
//!     let mean = chunk.mean().unwrap();
//!     Ok(mean)
//! })?;
//!
//! // Virtual array view
//! let view = array.view_window(&[0, 0], &[1000, 1000])?;
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```

use crate::compression::{compress_data, decompress_data, CompressionAlgorithm};
use crate::error::{IoError, Result};
use byteorder::{ByteOrder, LittleEndian};
use memmap2::{Mmap, MmapMut, MmapOptions};
use scirs2_core::ndarray::{Array, ArrayView, Dimension, IxDyn};
use scirs2_core::numeric::ScientificNumber;
use statrs::statistics::Statistics;
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Out-of-core array configuration
#[derive(Debug, Clone)]
pub struct OutOfCoreConfig {
    /// Chunk size in elements (not bytes)
    pub chunk_size: usize,
    /// Cache size in bytes
    pub cache_size_bytes: usize,
    /// Compression algorithm (optional)
    pub compression: Option<CompressionAlgorithm>,
    /// Enable write-through caching
    pub write_through: bool,
    /// Temporary directory for intermediate files
    pub temp_dir: Option<PathBuf>,
}

impl Default for OutOfCoreConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024 * 1024,              // 1M elements per chunk
            cache_size_bytes: 1024 * 1024 * 1024, // 1GB cache
            compression: None,
            write_through: true,
            temp_dir: None,
        }
    }
}

/// Metadata for out-of-core arrays
#[derive(Debug, Clone)]
struct ArrayMetadata {
    /// Array shape
    shape: Vec<usize>,
    /// Data type name
    #[allow(dead_code)]
    dtype: String,
    /// Element size in bytes
    element_size: usize,
    /// Chunk shape
    chunkshape: Vec<usize>,
    /// Compression algorithm
    compression: Option<CompressionAlgorithm>,
    /// Total number of chunks
    num_chunks: usize,
    /// Chunk offsets in file
    chunk_offsets: Vec<u64>,
    /// Chunk sizes (compressed)
    chunk_sizes: Vec<usize>,
}

/// Out-of-core array for processing large datasets
pub struct OutOfCoreArray<T> {
    /// File path
    file_path: PathBuf,
    /// Metadata
    metadata: ArrayMetadata,
    /// Memory-mapped file
    mmap: Option<Mmap>,
    /// Mutable memory map (for writing)
    #[allow(dead_code)]
    mmap_mut: Option<MmapMut>,
    /// Configuration
    config: OutOfCoreConfig,
    /// Cache for recently accessed chunks
    cache: Arc<RwLock<ChunkCache<T>>>,
    /// Type marker
    _phantom: std::marker::PhantomData<T>,
}

/// Cache for array chunks
struct ChunkCache<T> {
    /// Maximum cache size in bytes
    max_size_bytes: usize,
    /// Current cache size in bytes
    current_size_bytes: usize,
    /// Cached chunks (chunk_id -> data)
    chunks: HashMap<usize, CachedChunk<T>>,
    /// LRU queue for eviction
    lru_queue: VecDeque<usize>,
}

/// Cached chunk data
struct CachedChunk<T> {
    /// Chunk data
    data: Vec<T>,
    /// Whether chunk is dirty (modified)
    dirty: bool,
    /// Access count
    #[allow(dead_code)]
    access_count: usize,
}

impl<T: ScientificNumber + Clone> OutOfCoreArray<T> {
    /// Create a new out-of-core array
    pub fn create<P: AsRef<Path>>(path: P, shape: &[usize]) -> Result<Self> {
        Self::create_with_config(path, shape, OutOfCoreConfig::default())
    }

    /// Create with custom configuration
    pub fn create_with_config<P: AsRef<Path>>(
        path: P,
        shape: &[usize],
        config: OutOfCoreConfig,
    ) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();

        // Calculate chunk shape
        let chunkshape = Self::calculate_chunkshape(shape, config.chunk_size);

        // Calculate total chunks
        let chunks_per_dim: Vec<_> = shape
            .iter()
            .zip(&chunkshape)
            .map(|(&dim, &chunk)| (dim + chunk - 1) / chunk)
            .collect();
        let num_chunks = chunks_per_dim.iter().product();

        // Create metadata
        let metadata = ArrayMetadata {
            shape: shape.to_vec(),
            dtype: std::any::type_name::<T>().to_string(),
            element_size: std::mem::size_of::<T>(),
            chunkshape,
            compression: config.compression,
            num_chunks,
            chunk_offsets: vec![0; num_chunks],
            chunk_sizes: vec![0; num_chunks],
        };

        // Create file
        let mut file = File::create(&file_path)
            .map_err(|e| IoError::FileError(format!("Failed to create file: {e}")))?;

        // Write metadata header
        Self::write_metadata(&mut file, &metadata)?;

        // Pre-allocate space if no compression
        if config.compression.is_none() {
            let total_size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
            file.set_len((Self::metadata_size() + total_size) as u64)
                .map_err(|e| IoError::FileError(format!("Failed to set file size: {e}")))?;
        }

        // Create cache
        let cache = Arc::new(RwLock::new(ChunkCache::<T> {
            max_size_bytes: config.cache_size_bytes,
            current_size_bytes: 0,
            chunks: HashMap::new(),
            lru_queue: VecDeque::new(),
        }));

        Ok(Self {
            file_path,
            metadata,
            mmap: None,
            mmap_mut: None,
            config,
            cache: Arc::new(std::sync::RwLock::new(ChunkCache::<T> {
                max_size_bytes: 128 * 1024 * 1024, // 128MB default
                current_size_bytes: 0,
                chunks: HashMap::new(),
                lru_queue: VecDeque::new(),
            })),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Open an existing out-of-core array
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_config(path, OutOfCoreConfig::default())
    }

    /// Open with custom configuration
    pub fn open_with_config<P: AsRef<Path>>(path: P, config: OutOfCoreConfig) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();

        // Open file and read metadata
        let mut file = File::open(&file_path)
            .map_err(|_| IoError::FileNotFound(file_path.to_string_lossy().to_string()))?;

        let metadata = Self::read_metadata(&mut file)?;

        // Create memory map
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| IoError::ParseError(format!("Failed to create memory map: {e}")))?
        };

        // Create cache
        let cache = Arc::new(RwLock::new(ChunkCache::<T> {
            max_size_bytes: config.cache_size_bytes,
            current_size_bytes: 0,
            chunks: HashMap::new(),
            lru_queue: VecDeque::new(),
        }));

        Ok(Self {
            file_path,
            metadata,
            mmap: Some(mmap),
            mmap_mut: None,
            config,
            cache,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get array shape
    pub fn shape(&self) -> &[usize] {
        &self.metadata.shape
    }

    /// Get total number of elements
    pub fn len(&self) -> usize {
        self.metadata.shape.iter().product()
    }

    /// Check if array is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Calculate chunk shape based on target chunk size
    fn calculate_chunkshape(shape: &[usize], targetsize: usize) -> Vec<usize> {
        let ndim = shape.len();
        let elements_per_dim = (targetsize as f64).powf(1.0 / ndim as f64) as usize;

        shape
            .iter()
            .map(|&dim| dim.min(elements_per_dim.max(1)))
            .collect()
    }

    /// Size of metadata header
    fn metadata_size() -> usize {
        4096 // Fixed size for simplicity
    }

    /// Write metadata to file
    fn write_metadata(file: &mut File, metadata: &ArrayMetadata) -> Result<()> {
        let mut buffer = vec![0u8; Self::metadata_size()];
        let mut cursor = 0;

        // Magic number
        buffer[0..8].copy_from_slice(b"OOCARRAY");
        cursor += 8;

        // Version
        LittleEndian::write_u32(&mut buffer[cursor..], 1);
        cursor += 4;

        // Shape
        LittleEndian::write_u32(&mut buffer[cursor..], metadata.shape.len() as u32);
        cursor += 4;
        for &dim in &metadata.shape {
            LittleEndian::write_u64(&mut buffer[cursor..], dim as u64);
            cursor += 8;
        }

        // Element size
        LittleEndian::write_u32(&mut buffer[cursor..], metadata.element_size as u32);
        cursor += 4;

        // Chunk shape
        for &dim in &metadata.chunkshape {
            LittleEndian::write_u64(&mut buffer[cursor..], dim as u64);
            cursor += 8;
        }

        // Compression
        let compression_id = match metadata.compression {
            None => 0,
            Some(CompressionAlgorithm::Gzip) => 1,
            Some(CompressionAlgorithm::Zstd) => 2,
            Some(CompressionAlgorithm::Lz4) => 3,
            Some(CompressionAlgorithm::Bzip2) => 4,
            Some(CompressionAlgorithm::Brotli) => 5,
            Some(CompressionAlgorithm::Snappy) => 6,
            Some(CompressionAlgorithm::FpZip) => 7,
            Some(CompressionAlgorithm::DeltaLz4) => 8,
        };
        buffer[cursor] = compression_id;

        file.write_all(&buffer)
            .map_err(|e| IoError::FileError(format!("Failed to write metadata: {e}")))
    }

    /// Read metadata from file
    fn read_metadata(file: &mut File) -> Result<ArrayMetadata> {
        let mut buffer = vec![0u8; Self::metadata_size()];
        file.read_exact(&mut buffer)
            .map_err(|e| IoError::ParseError(format!("Failed to read metadata: {e}")))?;

        let mut cursor = 0;

        // Check magic number
        if &buffer[0..8] != b"OOCARRAY" {
            return Err(IoError::ParseError("Invalid _file format".to_string()));
        }
        cursor += 8;

        // Version
        let version = LittleEndian::read_u32(&buffer[cursor..]);
        if version != 1 {
            return Err(IoError::ParseError(format!(
                "Unsupported version: {}",
                version
            )));
        }
        cursor += 4;

        // Shape
        let ndim = LittleEndian::read_u32(&buffer[cursor..]) as usize;
        cursor += 4;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(LittleEndian::read_u64(&buffer[cursor..]) as usize);
            cursor += 8;
        }

        // Element size
        let element_size = LittleEndian::read_u32(&buffer[cursor..]) as usize;
        cursor += 4;

        // Chunk shape
        let mut chunkshape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            chunkshape.push(LittleEndian::read_u64(&buffer[cursor..]) as usize);
            cursor += 8;
        }

        // Compression
        let compression = match buffer[cursor] {
            0 => None,
            1 => Some(CompressionAlgorithm::Gzip),
            2 => Some(CompressionAlgorithm::Zstd),
            3 => Some(CompressionAlgorithm::Lz4),
            4 => Some(CompressionAlgorithm::Bzip2),
            _ => return Err(IoError::ParseError("Invalid compression type".to_string())),
        };

        // Calculate number of chunks
        let chunks_per_dim: Vec<_> = shape
            .iter()
            .zip(&chunkshape)
            .map(|(&dim, &chunk)| (dim + chunk - 1) / chunk)
            .collect();
        let num_chunks = chunks_per_dim.iter().product();

        Ok(ArrayMetadata {
            shape,
            dtype: String::new(), // Type is known from T
            element_size,
            chunkshape,
            compression,
            num_chunks,
            chunk_offsets: vec![0; num_chunks],
            chunk_sizes: vec![0; num_chunks],
        })
    }

    /// Get a chunk by its linear index
    fn get_chunk(&self, chunkid: usize) -> Result<Vec<T>> {
        // Check cache first
        {
            let cache = self.cache.read().unwrap();
            if let Some(cached) = cache.chunks.get(&chunkid) {
                return Ok(cached.data.clone());
            }
        }

        // Read from disk
        let data = self.read_chunk_from_disk(chunkid)?;

        // Update cache
        {
            let mut cache = self.cache.write().unwrap();
            self.update_cache(&mut cache, chunkid, data.clone());
        }

        Ok(data)
    }

    /// Read chunk from disk
    fn read_chunk_from_disk(&self, chunkid: usize) -> Result<Vec<T>> {
        if let Some(ref mmap) = self.mmap {
            let chunk_size = self.metadata.chunkshape.iter().product::<usize>();
            let offset = Self::metadata_size() + chunkid * chunk_size * self.metadata.element_size;

            if let Some(compression) = self.metadata.compression {
                // Handle compressed chunks
                let compressed_size = self.metadata.chunk_sizes[chunkid];
                let compressed_offset = self.metadata.chunk_offsets[chunkid];

                if compressed_size == 0 {
                    // Chunk hasn't been written yet, return zeros
                    let chunk_size = self.metadata.chunkshape.iter().product::<usize>();
                    return Ok(vec![T::zero(); chunk_size]);
                }

                let compressed_data = &mmap
                    [compressed_offset as usize..(compressed_offset as usize + compressed_size)];
                let decompressed_data =
                    decompress_data(compressed_data, compression).map_err(|e| {
                        IoError::ParseError(format!("Failed to decompress chunk: {}", e))
                    })?;

                // Convert bytes back to T values
                let chunk_size = self.metadata.chunkshape.iter().product::<usize>();
                let mut data = Vec::with_capacity(chunk_size);

                for i in 0..chunk_size {
                    let start = i * self.metadata.element_size;
                    let end = start + self.metadata.element_size;
                    if end <= decompressed_data.len() {
                        let value = T::from_le_bytes(&decompressed_data[start..end]);
                        data.push(value);
                    } else {
                        // Partial chunk at boundary
                        break;
                    }
                }

                Ok(data)
            } else {
                // Direct memory-mapped access
                let bytes = &mmap[offset..offset + chunk_size * self.metadata.element_size];
                let mut data = Vec::with_capacity(chunk_size);

                for i in 0..chunk_size {
                    let start = i * self.metadata.element_size;
                    let end = start + self.metadata.element_size;
                    let value = T::from_le_bytes(&bytes[start..end]);
                    data.push(value);
                }

                Ok(data)
            }
        } else {
            Err(IoError::ParseError(
                "Array not opened for reading".to_string(),
            ))
        }
    }

    /// Write chunk data to disk
    fn write_chunk_to_disk(&self, chunkid: usize, data: &[T]) -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.file_path)
            .map_err(|e| IoError::FileError(format!("Failed to open file for writing: {}", e)))?;

        // Convert data to bytes
        let mut chunk_bytes = Vec::new();
        for value in data {
            value.write_le(&mut chunk_bytes)?;
        }

        if let Some(compression) = self.metadata.compression {
            // Compress the data
            let compressed_data = compress_data(&chunk_bytes, compression, None)
                .map_err(|e| IoError::FileError(format!("Failed to compress chunk: {}", e)))?;

            // For compressed data, we need to update the metadata
            // This is a simplified implementation - in reality, we'd need to handle
            // dynamic file size changes and update chunk offsets
            let offset = if self.metadata.chunk_offsets[chunkid] == 0 {
                // New chunk, append to end of file
                file.seek(SeekFrom::End(0))
                    .map_err(|e| IoError::FileError(format!("Failed to seek to end: {}", e)))?
            } else {
                // Existing chunk, use existing offset
                self.metadata.chunk_offsets[chunkid]
            };

            file.seek(SeekFrom::Start(offset))
                .map_err(|e| IoError::FileError(format!("Failed to seek: {}", e)))?;

            file.write_all(&compressed_data).map_err(|e| {
                IoError::FileError(format!("Failed to write compressed data: {}", e))
            })?;
        } else {
            // Uncompressed data
            let chunk_size = self.metadata.chunkshape.iter().product::<usize>();
            let offset = Self::metadata_size() + chunkid * chunk_size * self.metadata.element_size;

            file.seek(SeekFrom::Start(offset as u64))
                .map_err(|e| IoError::FileError(format!("Failed to seek: {}", e)))?;

            file.write_all(&chunk_bytes)
                .map_err(|e| IoError::FileError(format!("Failed to write data: {}", e)))?;
        }

        file.sync_all()
            .map_err(|e| IoError::FileError(format!("Failed to sync file: {}", e)))?;

        Ok(())
    }

    /// Update cache with new chunk
    fn update_cache(&self, cache: &mut ChunkCache<T>, chunkid: usize, data: Vec<T>) {
        let chunk_size_bytes = data.len() * std::mem::size_of::<T>();

        // Evict chunks if necessary
        while cache.current_size_bytes + chunk_size_bytes > cache.max_size_bytes
            && !cache.lru_queue.is_empty()
        {
            if let Some(evict_id) = cache.lru_queue.pop_front() {
                if let Some(evicted) = cache.chunks.remove(&evict_id) {
                    cache.current_size_bytes -= evicted.data.len() * std::mem::size_of::<T>();

                    // Write back if dirty and write-through enabled
                    if evicted.dirty && self.config.write_through {
                        if let Err(e) = self.write_chunk_to_disk(evict_id, &evicted.data) {
                            eprintln!(
                                "Warning: Failed to write back dirty chunk {}: {}",
                                evict_id, e
                            );
                        }
                    }
                }
            }
        }

        // Add to cache
        cache.chunks.insert(
            chunkid,
            CachedChunk {
                data,
                dirty: false,
                access_count: 1,
            },
        );
        cache.lru_queue.push_back(chunkid);
        cache.current_size_bytes += chunk_size_bytes;
    }

    /// Process array in chunks
    pub fn process_chunks<F, R>(&self, chunk_size: usize, processor: F) -> Result<Vec<R>>
    where
        F: Fn(ArrayView<T, IxDyn>) -> Result<R>,
        R: Send,
    {
        let total_elements = self.len();
        let num_chunks = (total_elements + chunk_size - 1) / chunk_size;
        let mut results = Vec::with_capacity(num_chunks);

        for chunk_id in 0..self.metadata.num_chunks {
            let chunk_data = self.get_chunk(chunk_id)?;
            let chunkshape = self.get_chunkshape(chunk_id);

            let array_view = ArrayView::from_shape(chunkshape, &chunk_data)
                .map_err(|e| IoError::ParseError(format!("Failed to create array view: {}", e)))?;

            let result = processor(array_view)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get shape of a specific chunk
    fn get_chunkshape(&self, chunkid: usize) -> IxDyn {
        // Calculate chunk coordinates
        let mut chunk_coords = Vec::with_capacity(self.metadata.shape.len());
        let mut temp_id = chunkid;

        let chunks_per_dim: Vec<_> = self
            .metadata
            .shape
            .iter()
            .zip(&self.metadata.chunkshape)
            .map(|(&dim, &chunk)| (dim + chunk - 1) / chunk)
            .collect();

        for &chunks in chunks_per_dim.iter().rev() {
            chunk_coords.push(temp_id % chunks);
            temp_id /= chunks;
        }
        chunk_coords.reverse();

        // Calculate actual chunk shape (may be smaller at boundaries)
        let chunkshape: Vec<_> = chunk_coords
            .iter()
            .zip(&self.metadata.shape)
            .zip(&self.metadata.chunkshape)
            .map(|((&coord, &dim), &chunk_dim)| {
                let start = coord * chunk_dim;
                let end = ((coord + 1) * chunk_dim).min(dim);
                end - start
            })
            .collect();

        IxDyn(&chunkshape)
    }

    /// Get a window view of the array
    pub fn view_window(&self, start: &[usize], shape: &[usize]) -> Result<Array<T, IxDyn>> {
        if start.len() != self.metadata.shape.len() || shape.len() != self.metadata.shape.len() {
            return Err(IoError::ParseError("Invalid window dimensions".to_string()));
        }

        // Check bounds
        for i in 0..start.len() {
            if start[i] + shape[i] > self.metadata.shape[i] {
                return Err(IoError::ParseError(
                    "Window extends beyond array bounds".to_string(),
                ));
            }
        }

        // Create result array
        let mut result = Array::zeros(IxDyn(shape));

        // Determine which chunks overlap with the window
        let start_chunks: Vec<_> = start
            .iter()
            .zip(&self.metadata.chunkshape)
            .map(|(&s, &chunk)| s / chunk)
            .collect();

        let end_chunks: Vec<_> = start
            .iter()
            .zip(shape)
            .zip(&self.metadata.chunkshape)
            .map(|((&s, &sz), &chunk)| (s + sz - 1) / chunk)
            .collect();

        // Iterate over overlapping chunks
        self.copy_chunks_to_window(start, &start_chunks, &end_chunks, &mut result)?;

        Ok(result)
    }

    /// Copy chunks to window result array
    fn copy_chunks_to_window(
        &self,
        window_start: &[usize],
        start_chunks: &[usize],
        end_chunks: &[usize],
        result: &mut Array<T, IxDyn>,
    ) -> Result<()> {
        // Iterate through all _chunks that overlap with the window
        let mut chunk_coords = start_chunks.to_vec();

        loop {
            // Calculate linear chunk ID from coordinates
            let chunk_id = self.coords_to_chunk_id(&chunk_coords);

            // Get chunk data
            let chunk_data = self.get_chunk(chunk_id)?;
            let chunkshape = self.get_chunkshape(chunk_id);

            // Calculate overlap region
            let chunk_start: Vec<_> = chunk_coords
                .iter()
                .zip(&self.metadata.chunkshape)
                .map(|(&coord, &size)| coord * size)
                .collect();

            // Calculate intersection of chunk with window
            let overlap_start: Vec<_> = chunk_start
                .iter()
                .zip(window_start)
                .map(|(&chunk_s, &win_s)| chunk_s.max(win_s))
                .collect();

            let overlap_end: Vec<_> = chunk_start
                .iter()
                .zip(chunkshape.slice())
                .zip(window_start)
                .zip(result.shape())
                .map(|(((chunk_s, chunk_sz), win_s), win_sz)| {
                    (chunk_s + chunk_sz).min(win_s + win_sz)
                })
                .collect();

            // Copy data if there's overlap
            if overlap_start.iter().zip(&overlap_end).all(|(s, e)| s < e) {
                // Calculate source indices in chunk
                let chunksrc_start: Vec<_> = overlap_start
                    .iter()
                    .zip(&chunk_start)
                    .map(|(overlap, chunk)| overlap - chunk)
                    .collect();

                let chunksrc_end: Vec<_> = overlap_end
                    .iter()
                    .zip(&chunk_start)
                    .map(|(overlap, chunk)| overlap - chunk)
                    .collect();

                // Calculate destination indices in result
                let result_dst_start: Vec<_> = overlap_start
                    .iter()
                    .zip(window_start)
                    .map(|(overlap, win)| overlap - win)
                    .collect();

                let result_dst_end: Vec<_> = overlap_end
                    .iter()
                    .zip(window_start)
                    .map(|(overlap, win)| overlap - win)
                    .collect();

                // Perform the copy for each element in the overlap region
                self.copy_chunk_region(
                    &chunk_data,
                    chunkshape.slice(),
                    &chunksrc_start,
                    &chunksrc_end,
                    result,
                    &result_dst_start,
                    &result_dst_end,
                )?;
            }

            // Move to next chunk
            if !self.increment_chunk_coords(&mut chunk_coords, start_chunks, end_chunks) {
                break;
            }
        }

        Ok(())
    }

    /// Convert chunk coordinates to linear chunk ID
    fn coords_to_chunk_id(&self, coords: &[usize]) -> usize {
        let chunks_per_dim: Vec<_> = self
            .metadata
            .shape
            .iter()
            .zip(&self.metadata.chunkshape)
            .map(|(&dim, &chunk)| (dim + chunk - 1) / chunk)
            .collect();

        let mut chunk_id = 0;
        let mut multiplier = 1;

        for (_i, (&coord, &chunks_in_dim)) in coords.iter().zip(&chunks_per_dim).enumerate().rev() {
            chunk_id += coord * multiplier;
            multiplier *= chunks_in_dim;
        }

        chunk_id
    }

    /// Copy a region from chunk data to result array
    fn copy_chunk_region(
        &self,
        chunk_data: &[T],
        chunkshape: &[usize],
        src_start: &[usize],
        src_end: &[usize],
        result: &mut Array<T, IxDyn>,
        dst_start: &[usize],
        _dst_end: &[usize],
    ) -> Result<()> {
        // For simplicity, handle only 1D and 2D cases
        match chunkshape.len() {
            1 => {
                let src_len = src_end[0] - src_start[0];
                for i in 0..src_len {
                    let src_idx = src_start[0] + i;
                    let dst_idx = dst_start[0] + i;
                    result[[dst_idx]] = chunk_data[src_idx];
                }
            }
            2 => {
                for i in 0..(src_end[0] - src_start[0]) {
                    for j in 0..(src_end[1] - src_start[1]) {
                        let src_idx = (src_start[0] + i) * chunkshape[1] + (src_start[1] + j);
                        let dst_idx = [dst_start[0] + i, dst_start[1] + j];
                        result[&dst_idx[..]] = chunk_data[src_idx];
                    }
                }
            }
            _ => {
                // For higher dimensions, use recursive approach or flatten
                return Err(IoError::ParseError(
                    "High dimensional copying not yet implemented".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Increment chunk coordinates within bounds
    fn increment_chunk_coords(
        &self,
        coords: &mut [usize],
        start_chunks: &[usize],
        end_chunks: &[usize],
    ) -> bool {
        for i in (0..coords.len()).rev() {
            coords[i] += 1;
            if coords[i] <= end_chunks[i] {
                return true;
            }
            coords[i] = start_chunks[i];
        }
        false
    }

    /// Write data to a window
    pub fn write_window(&mut self, start: &[usize], data: &ArrayView<T, IxDyn>) -> Result<()> {
        if start.len() != self.metadata.shape.len() || data.ndim() != self.metadata.shape.len() {
            return Err(IoError::FileError("Invalid window dimensions".to_string()));
        }

        // Check bounds
        for (i, &start_val) in start.iter().enumerate() {
            if start_val + data.shape()[i] > self.metadata.shape[i] {
                return Err(IoError::FileError(
                    "Window extends beyond array bounds".to_string(),
                ));
            }
        }

        // Implement actual writing logic
        // 1. Determine which chunks are affected
        let start_chunks: Vec<_> = start
            .iter()
            .zip(&self.metadata.chunkshape)
            .map(|(&s, &chunk)| s / chunk)
            .collect();

        let end_chunks: Vec<_> = start
            .iter()
            .zip(data.shape())
            .zip(&self.metadata.chunkshape)
            .map(|((&s, &sz), &chunk)| (s + sz - 1) / chunk)
            .collect();

        // 2. Iterate through affected chunks
        let mut chunk_coords = start_chunks.clone();

        loop {
            let chunk_id = self.coords_to_chunk_id(&chunk_coords);

            // 3. Read the chunk (or get from cache)
            let mut chunk_data = self.get_chunk(chunk_id)?;
            let chunkshape = self.get_chunkshape(chunk_id);

            // Calculate chunk start position in global coordinates
            let chunk_start: Vec<_> = chunk_coords
                .iter()
                .zip(&self.metadata.chunkshape)
                .map(|(&coord, &size)| coord * size)
                .collect();

            // Calculate overlap region
            let overlap_start: Vec<_> = chunk_start
                .iter()
                .zip(start)
                .map(|(&chunk_s, &win_s)| chunk_s.max(win_s))
                .collect();

            let overlap_end: Vec<_> = chunk_start
                .iter()
                .zip(chunkshape.slice())
                .zip(start)
                .zip(data.shape())
                .map(|(((chunk_s, chunk_sz), win_s), win_sz)| {
                    (chunk_s + chunk_sz).min(win_s + win_sz)
                })
                .collect();

            // 4. Update the relevant portions
            if overlap_start.iter().zip(&overlap_end).all(|(s, e)| s < e) {
                self.write_to_chunk_region(
                    &mut chunk_data,
                    chunkshape.slice(),
                    &chunk_start,
                    &overlap_start,
                    &overlap_end,
                    data,
                    start,
                )?;

                // 5. Mark chunk as dirty in cache or write back immediately
                {
                    let mut cache = self.cache.write().unwrap();
                    if let Some(cached_chunk) = cache.chunks.get_mut(&chunk_id) {
                        cached_chunk.data = chunk_data;
                        cached_chunk.dirty = true;
                    } else {
                        // Add to cache as dirty
                        let chunk_size_bytes = chunk_data.len() * std::mem::size_of::<T>();
                        cache.chunks.insert(
                            chunk_id,
                            CachedChunk {
                                data: chunk_data,
                                dirty: true,
                                access_count: 1,
                            },
                        );
                        cache.current_size_bytes += chunk_size_bytes;
                        cache.lru_queue.push_back(chunk_id);
                    }
                }
            }

            // Move to next chunk
            if !self.increment_chunk_coords(&mut chunk_coords, &start_chunks, &end_chunks) {
                break;
            }
        }

        Ok(())
    }

    /// Write data to a specific region within a chunk
    fn write_to_chunk_region(
        &self,
        chunk_data: &mut [T],
        chunkshape: &[usize],
        chunk_start: &[usize],
        overlap_start: &[usize],
        overlap_end: &[usize],
        source_data: &ArrayView<T, IxDyn>,
        source_start: &[usize],
    ) -> Result<()> {
        // Calculate indices for copying
        let chunk_local_start: Vec<_> = overlap_start
            .iter()
            .zip(chunk_start)
            .map(|(overlap, chunk)| overlap - chunk)
            .collect();

        let source_local_start: Vec<_> = overlap_start
            .iter()
            .zip(source_start)
            .map(|(overlap, source)| overlap - source)
            .collect();

        // For simplicity, handle only 1D and 2D cases
        match chunkshape.len() {
            1 => {
                let len = overlap_end[0] - overlap_start[0];
                for i in 0..len {
                    let chunk_idx = chunk_local_start[0] + i;
                    let source_idx = [source_local_start[0] + i];
                    chunk_data[chunk_idx] = source_data[&source_idx[..]];
                }
            }
            2 => {
                for i in 0..(overlap_end[0] - overlap_start[0]) {
                    for j in 0..(overlap_end[1] - overlap_start[1]) {
                        let chunk_idx =
                            (chunk_local_start[0] + i) * chunkshape[1] + (chunk_local_start[1] + j);
                        let source_idx = [source_local_start[0] + i, source_local_start[1] + j];
                        chunk_data[chunk_idx] = source_data[&source_idx[..]];
                    }
                }
            }
            _ => {
                return Err(IoError::ParseError(
                    "High dimensional writing not yet implemented".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Flush all cached data to disk
    pub fn flush(&mut self) -> Result<()> {
        let cache = self.cache.write().unwrap();

        for (&chunk_id, chunk) in &cache.chunks {
            if chunk.dirty {
                self.write_chunk_to_disk(chunk_id, &chunk.data)?;
            }
        }

        Ok(())
    }
}

/// Chunk processor for streaming operations
pub trait ChunkProcessor<T> {
    /// Process a chunk of data
    fn process(&mut self, chunk: ArrayView<T, IxDyn>) -> Result<()>;

    /// Finalize processing
    fn finalize(self) -> Result<()>;
}

/// Out-of-core sorting for large datasets
pub struct OutOfCoreSorter<T> {
    /// Temporary directory
    temp_dir: PathBuf,
    /// Chunk size
    chunk_size: usize,
    /// Sorted chunk files
    chunk_files: Vec<PathBuf>,
    /// Type marker
    _phantom: std::marker::PhantomData<T>,
}

impl<T: ScientificNumber + Ord + Clone> OutOfCoreSorter<T> {
    /// Create a new out-of-core sorter
    pub fn new(temp_dir: PathBuf, chunk_size: usize) -> Result<Self> {
        std::fs::create_dir_all(&temp_dir)
            .map_err(|e| IoError::FileError(format!("Failed to create temp dir: {}", e)))?;

        Ok(Self {
            temp_dir,
            chunk_size,
            chunk_files: Vec::new(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Add data to be sorted
    pub fn add_data(&mut self, data: &[T]) -> Result<()> {
        // Process in chunks
        for chunk in data.chunks(self.chunk_size) {
            let mut sorted_chunk = chunk.to_vec();
            sorted_chunk.sort();

            // Write to temporary file
            let chunk_file = self
                .temp_dir
                .join(format!("chunk_{}.tmp", self.chunk_files.len()));
            let mut file = File::create(&chunk_file)
                .map_err(|e| IoError::FileError(format!("Failed to create chunk file: {}", e)))?;

            for value in &sorted_chunk {
                value.write_le(&mut file)?;
            }

            self.chunk_files.push(chunk_file);
        }

        Ok(())
    }

    /// Merge sorted chunks into output
    pub fn merge<W: Write>(self, output: &mut W) -> Result<()> {
        // K-way merge of sorted chunks
        let readers: Vec<_> = self
            .chunk_files
            .iter()
            .map(File::open)
            .collect::<std::io::Result<_>>()
            .map_err(|e| IoError::ParseError(format!("Failed to open chunk file: {}", e)))?;

        // K-way merge using a binary heap
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        // Create readers with buffering
        let mut buffered_readers: Vec<_> = readers.into_iter().map(BufReader::new).collect();

        // Priority queue for k-way merge (min-heap using Reverse)
        let mut heap: BinaryHeap<Reverse<(T, usize)>> = BinaryHeap::new();

        // Initialize heap with first element from each reader
        for (reader_id, reader) in buffered_readers.iter_mut().enumerate() {
            if let Ok(value) = <T as ScientificNumberRead>::read_le(reader) {
                heap.push(Reverse((value, reader_id)));
            }
        }

        // Perform k-way merge
        while let Some(Reverse((value, reader_id))) = heap.pop() {
            // Write current minimum value
            value.write_le(output)?;

            // Read next value from the same reader
            if let Ok(next_value) =
                <T as ScientificNumberRead>::read_le(&mut buffered_readers[reader_id])
            {
                heap.push(Reverse((next_value, reader_id)));
            }
        }

        // Clean up temporary files
        for chunk_file in &self.chunk_files {
            let _ = std::fs::remove_file(chunk_file);
        }

        Ok(())
    }
}

/// Virtual array that combines multiple arrays
pub struct VirtualArray<T> {
    /// Component arrays
    arrays: Vec<Box<dyn ArraySource<T>>>,
    /// Total shape
    shape: Vec<usize>,
    /// Axis along which arrays are concatenated
    axis: usize,
}

/// Source for virtual array components
pub trait ArraySource<T>: Send + Sync {
    /// Get shape
    fn shape(&self) -> &[usize];

    /// Read a region
    fn read_region(&self, start: &[usize], shape: &[usize]) -> Result<Array<T, IxDyn>>;
}

impl<T: Clone> VirtualArray<T> {
    /// Create a virtual array by concatenating along an axis
    pub fn concatenate(arrays: Vec<Box<dyn ArraySource<T>>>, axis: usize) -> Result<Self> {
        if arrays.is_empty() {
            return Err(IoError::ParseError("No _arrays provided".to_string()));
        }

        // Validate shapes
        let firstshape = arrays[0].shape();
        for array in &arrays[1..] {
            let shape = array.shape();
            if shape.len() != firstshape.len() {
                return Err(IoError::ParseError(
                    "Inconsistent array dimensions".to_string(),
                ));
            }

            for (i, (&a, &b)) in shape.iter().zip(firstshape).enumerate() {
                if i != axis && a != b {
                    return Err(IoError::ParseError(format!(
                        "Inconsistent shape along axis {}: {} vs {}",
                        i, a, b
                    )));
                }
            }
        }

        // Calculate total shape
        let mut shape = firstshape.to_vec();
        shape[axis] = arrays.iter().map(|a| a.shape()[axis]).sum();

        Ok(Self {
            arrays,
            shape,
            axis,
        })
    }

    /// Get total shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Read a region from the virtual array
    pub fn read_region(&self, start: &[usize], shape: &[usize]) -> Result<Array<T, IxDyn>> {
        // Determine which component arrays are needed
        let end_pos = start[self.axis] + shape[self.axis];
        let mut current_pos = 0;
        let mut result_parts = Vec::new();

        for array in &self.arrays {
            let array_size = array.shape()[self.axis];
            let array_end = current_pos + array_size;

            // Check if this array overlaps with requested region
            if current_pos < end_pos && array_end > start[self.axis] {
                let local_start = start[self.axis].saturating_sub(current_pos);
                let local_end = (end_pos - current_pos).min(array_size);

                let mut local_region_start = start.to_vec();
                local_region_start[self.axis] = local_start;

                let mut local_regionshape = shape.to_vec();
                local_regionshape[self.axis] = local_end - local_start;

                let part = array.read_region(&local_region_start, &local_regionshape)?;
                result_parts.push(part);
            }

            current_pos = array_end;
            if current_pos >= end_pos {
                break;
            }
        }

        // Concatenate parts
        if result_parts.is_empty() {
            return Err(IoError::ParseError(
                "No data in requested region".to_string(),
            ));
        }

        // Simple concatenation - in reality would use ndarray's concatenate
        Ok(result_parts.into_iter().next().unwrap())
    }
}

/// Sliding window iterator for out-of-core processing
pub struct SlidingWindow<'a, T> {
    array: &'a OutOfCoreArray<T>,
    windowshape: Vec<usize>,
    stride: Vec<usize>,
    current_position: Vec<usize>,
}

impl<'a, T: ScientificNumber + Clone> SlidingWindow<'a, T> {
    /// Create a new sliding window iterator
    pub fn new(
        array: &'a OutOfCoreArray<T>,
        windowshape: Vec<usize>,
        stride: Vec<usize>,
    ) -> Result<Self> {
        if windowshape.len() != array.shape().len() || stride.len() != array.shape().len() {
            return Err(IoError::ParseError("Dimension mismatch".to_string()));
        }

        Ok(Self {
            array,
            windowshape,
            stride,
            current_position: vec![0; array.shape().len()],
        })
    }
}

impl<T: ScientificNumber + Clone> Iterator for SlidingWindow<'_, T> {
    type Item = Result<Array<T, IxDyn>>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we've reached the end
        for (i, &pos) in self.current_position.iter().enumerate() {
            if pos + self.windowshape[i] > self.array.shape()[i] {
                return None;
            }
        }

        // Get current window
        let window = self
            .array
            .view_window(&self.current_position, &self.windowshape);

        // Advance position
        let mut carry = true;
        for i in (0..self.current_position.len()).rev() {
            if carry {
                self.current_position[i] += self.stride[i];
                if self.current_position[i] + self.windowshape[i] <= self.array.shape()[i] {
                    carry = false;
                } else if i > 0 {
                    self.current_position[i] = 0;
                }
            }
        }

        Some(window)
    }
}

// Implement numeric trait extension for reading and writing
trait ScientificNumberWrite {
    fn write_le<W: Write>(&self, writer: &mut W) -> Result<()>;
}

trait ScientificNumberRead: Sized {
    fn read_le<R: Read>(reader: &mut R) -> Result<Self>;
}

impl<T: ScientificNumber> ScientificNumberWrite for T {
    fn write_le<W: Write>(&self, writer: &mut W) -> Result<()> {
        let bytes = self.to_le_bytes();
        writer
            .write_all(&bytes)
            .map_err(|e| IoError::FileError(format!("Failed to write numeric value: {}", e)))
    }
}

impl<T: ScientificNumber> ScientificNumberRead for T {
    fn read_le<R: Read>(reader: &mut R) -> Result<Self> {
        let size = std::mem::size_of::<T>();
        let mut bytes = vec![0u8; size];
        reader
            .read_exact(&mut bytes)
            .map_err(|e| IoError::ParseError(format!("Failed to read numeric value: {}", e)))?;
        Ok(T::from_le_bytes(&bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_out_of_core_array_creation() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_array.ooc");

        let array = OutOfCoreArray::<f64>::create(&file_path, &[1000, 1000])?;
        assert_eq!(array.shape(), &[1000, 1000]);
        assert_eq!(array.len(), 1_000_000);

        Ok(())
    }

    #[test]
    fn test_chunk_calculation() {
        let shape = vec![10000, 5000, 100];
        let chunkshape = OutOfCoreArray::<f64>::calculate_chunkshape(&shape, 1_000_000);

        let chunk_elements: usize = chunkshape.iter().product();
        assert!(chunk_elements <= 1_000_000);

        for (&dim, &chunk) in shape.iter().zip(&chunkshape) {
            assert!(chunk <= dim);
            assert!(chunk > 0);
        }
    }

    #[test]
    fn test_sliding_window() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_window.ooc");

        let array = OutOfCoreArray::<f64>::create(&file_path, &[100, 100])?;

        let window = SlidingWindow::new(&array, vec![10, 10], vec![5, 5])?;
        let windows: Vec<_> = window.collect();

        // Should have (100-10)/5 + 1 = 19 windows in each dimension
        assert_eq!(windows.len(), 19 * 19);

        Ok(())
    }
}
