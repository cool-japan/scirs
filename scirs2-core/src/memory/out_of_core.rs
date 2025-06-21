//! # Out-of-Core Processing
//!
//! This module provides capabilities for processing datasets that are larger than available memory
//! by streaming data from disk and processing it in chunks with intelligent caching and prefetching.

use crate::error::{CoreError, CoreResult};
use crate::memory::metrics::{track_allocation, track_deallocation};
use ndarray::{Array, IxDyn};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

/// Error types for out-of-core processing
#[derive(Debug, thiserror::Error)]
pub enum OutOfCoreError {
    /// I/O operation failed
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Chunk not found
    #[error("Chunk not found: {0}")]
    ChunkNotFound(String),

    /// Invalid chunk size
    #[error("Invalid chunk size: {0}")]
    InvalidChunkSize(String),

    /// Cache full
    #[error("Cache is full and no evictable chunks found")]
    CacheFull,

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Index out of bounds
    #[error("Index out of bounds: {index} >= {size}")]
    IndexOutOfBounds { index: usize, size: usize },
}

impl From<OutOfCoreError> for CoreError {
    fn from(err: OutOfCoreError) -> Self {
        CoreError::ComputationError(crate::error::ErrorContext::new(err.to_string()))
    }
}

/// Chunk identifier for out-of-core arrays
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChunkId {
    /// Array identifier
    pub array_id: String,
    /// Chunk coordinates in the array
    pub coordinates: Vec<usize>,
}

impl ChunkId {
    /// Create a new chunk ID
    pub fn new(array_id: String, coordinates: Vec<usize>) -> Self {
        Self {
            array_id,
            coordinates,
        }
    }
}

impl std::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}",
            self.array_id,
            self.coordinates
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

/// Metadata for an out-of-core chunk
#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    /// Chunk identifier
    pub id: ChunkId,
    /// Size in bytes
    pub size_bytes: usize,
    /// Shape of the chunk
    pub shape: Vec<usize>,
    /// File offset where chunk data starts
    pub file_offset: u64,
    /// Last access time
    pub last_accessed: Instant,
    /// Access count
    pub access_count: u64,
    /// Whether chunk has been modified
    pub is_dirty: bool,
}

impl ChunkMetadata {
    /// Create new chunk metadata
    pub fn new(id: ChunkId, shape: Vec<usize>, file_offset: u64) -> Self {
        let size_bytes = shape.iter().product::<usize>() * std::mem::size_of::<f64>(); // Assume f64 for now

        Self {
            id,
            size_bytes,
            shape,
            file_offset,
            last_accessed: Instant::now(),
            access_count: 0,
            is_dirty: false,
        }
    }

    /// Create new chunk metadata with explicit element size
    pub fn new_with_element_size(
        id: ChunkId,
        shape: Vec<usize>,
        file_offset: u64,
        element_size: usize,
    ) -> Self {
        let size_bytes = shape.iter().product::<usize>() * element_size;

        Self {
            id,
            size_bytes,
            shape,
            file_offset,
            last_accessed: Instant::now(),
            access_count: 0,
            is_dirty: false,
        }
    }

    /// Update access statistics
    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    /// Mark chunk as dirty (modified)
    pub fn mark_dirty(&mut self) {
        self.is_dirty = true;
    }
}

/// Cache replacement policy
#[derive(Debug, Clone)]
pub enum CachePolicy {
    /// Least Recently Used
    Lru,
    /// Least Frequently Used
    Lfu,
    /// First In, First Out
    Fifo,
    /// Most Recently Used (for testing)
    Mru,
}

/// Configuration for out-of-core processing
#[derive(Debug, Clone)]
pub struct OutOfCoreConfig {
    /// Maximum memory usage for cache (in bytes)
    pub max_cache_memory: usize,
    /// Maximum number of chunks in cache
    pub max_cached_chunks: usize,
    /// Chunk size for each dimension
    pub chunk_shape: Vec<usize>,
    /// Cache replacement policy
    pub cache_policy: CachePolicy,
    /// Enable prefetching
    pub enable_prefetching: bool,
    /// Number of chunks to prefetch
    pub prefetch_count: usize,
    /// Enable compression for stored chunks
    pub enable_compression: bool,
    /// I/O buffer size
    pub io_buffer_size: usize,
}

impl Default for OutOfCoreConfig {
    fn default() -> Self {
        Self {
            max_cache_memory: 1024 * 1024 * 1024, // 1 GB
            max_cached_chunks: 100,
            chunk_shape: vec![1000, 1000], // Default 2D chunks
            cache_policy: CachePolicy::Lru,
            enable_prefetching: true,
            prefetch_count: 4,
            enable_compression: false,
            io_buffer_size: 64 * 1024, // 64 KB
        }
    }
}

/// In-memory cache for chunks
pub struct ChunkCache<T> {
    /// Cached chunk data
    chunks: RwLock<HashMap<ChunkId, Array<T, IxDyn>>>,
    /// Chunk metadata
    pub(crate) metadata: RwLock<HashMap<ChunkId, ChunkMetadata>>,
    /// Access order for LRU/FIFO policies
    access_order: Mutex<VecDeque<ChunkId>>,
    /// Configuration
    config: OutOfCoreConfig,
    /// Current memory usage
    current_memory: Mutex<usize>,
}

impl<T> ChunkCache<T>
where
    T: Clone + Default + 'static + Send + Sync,
{
    /// Create a new chunk cache
    pub fn new(config: OutOfCoreConfig) -> Self {
        Self {
            chunks: RwLock::new(HashMap::new()),
            metadata: RwLock::new(HashMap::new()),
            access_order: Mutex::new(VecDeque::new()),
            config,
            current_memory: Mutex::new(0),
        }
    }

    /// Get a chunk from cache
    pub fn get(&self, chunk_id: &ChunkId) -> Option<Array<T, IxDyn>> {
        let chunks = self.chunks.read().unwrap();
        if let Some(chunk) = chunks.get(chunk_id) {
            // Update access statistics
            self.update_access_stats(chunk_id);
            Some(chunk.clone())
        } else {
            None
        }
    }

    /// Put a chunk into cache with optional writer for eviction
    pub(crate) fn put_with_writer<F>(
        &self,
        chunk_id: ChunkId,
        chunk: Array<T, IxDyn>,
        metadata: ChunkMetadata,
        writer: F,
    ) -> CoreResult<()>
    where
        F: Fn(&ChunkId) -> CoreResult<()>,
    {
        // Check if we need to evict chunks
        self.ensure_cache_space_with_writer(&metadata, writer)?;

        // Insert chunk and metadata
        let chunk_size = chunk.len() * std::mem::size_of::<T>();

        {
            let mut chunks = self.chunks.write().unwrap();
            let mut metadata_map = self.metadata.write().unwrap();
            let mut access_order = self.access_order.lock().unwrap();
            let mut current_memory = self.current_memory.lock().unwrap();

            chunks.insert(chunk_id.clone(), chunk);
            metadata_map.insert(chunk_id.clone(), metadata);
            access_order.push_back(chunk_id.clone());
            *current_memory += chunk_size;
        }

        track_allocation("OutOfCoreCache", chunk_size, 0);
        Ok(())
    }

    /// Put a chunk into cache
    pub fn put(
        &self,
        chunk_id: ChunkId,
        chunk: Array<T, IxDyn>,
        metadata: ChunkMetadata,
    ) -> CoreResult<()> {
        self.put_with_writer(chunk_id, chunk, metadata, |_| Ok(()))
    }

    /// Remove a chunk from cache
    pub fn remove(&self, chunk_id: &ChunkId) -> Option<Array<T, IxDyn>> {
        let mut chunks = self.chunks.write().unwrap();
        let mut metadata_map = self.metadata.write().unwrap();
        let mut access_order = self.access_order.lock().unwrap();
        let mut current_memory = self.current_memory.lock().unwrap();

        if let Some(chunk) = chunks.remove(chunk_id) {
            let chunk_size = chunk.len() * std::mem::size_of::<T>();
            metadata_map.remove(chunk_id);
            access_order.retain(|id| id != chunk_id);
            *current_memory = current_memory.saturating_sub(chunk_size);

            track_deallocation("OutOfCoreCache", chunk_size, 0);
            Some(chunk)
        } else {
            None
        }
    }

    /// Update access statistics for a chunk
    fn update_access_stats(&self, chunk_id: &ChunkId) {
        let mut metadata_map = self.metadata.write().unwrap();
        if let Some(metadata) = metadata_map.get_mut(chunk_id) {
            metadata.touch();
        }

        // Update access order for LRU
        let mut access_order = self.access_order.lock().unwrap();
        access_order.retain(|id| id != chunk_id);
        access_order.push_back(chunk_id.clone());
    }

    /// Ensure there's space in cache for a new chunk
    pub(crate) fn ensure_cache_space_with_writer<F>(
        &self,
        new_metadata: &ChunkMetadata,
        writer: F,
    ) -> CoreResult<()>
    where
        F: Fn(&ChunkId) -> CoreResult<()>,
    {
        let current_memory = *self.current_memory.lock().unwrap();
        let current_count = self.chunks.read().unwrap().len();

        // Check if we need to evict based on memory or count limits
        let needs_eviction = current_memory + new_metadata.size_bytes
            > self.config.max_cache_memory
            || current_count >= self.config.max_cached_chunks;

        if needs_eviction {
            self.evict_chunks_with_writer(1, writer)?;
        }

        Ok(())
    }

    /// Ensure there's space in cache for a new chunk
    #[allow(dead_code)]
    fn ensure_cache_space(&self, new_metadata: &ChunkMetadata) -> CoreResult<()> {
        self.ensure_cache_space_with_writer(new_metadata, |_| Ok(()))
    }

    /// Evict chunks based on cache policy with optional dirty chunk writer
    fn evict_chunks_with_writer<F>(&self, count: usize, writer: F) -> CoreResult<()>
    where
        F: Fn(&ChunkId) -> CoreResult<()>,
    {
        let chunks_to_evict = self.select_eviction_candidates(count)?;

        for chunk_id in chunks_to_evict {
            // Check if chunk is dirty and write back to storage if needed
            if let Some(metadata) = self.metadata.read().unwrap().get(&chunk_id) {
                if metadata.is_dirty {
                    writer(&chunk_id)?;
                }
            }
            self.remove(&chunk_id);
        }

        Ok(())
    }

    /// Evict chunks based on cache policy
    #[allow(dead_code)]
    fn evict_chunks(&self, count: usize) -> CoreResult<()> {
        self.evict_chunks_with_writer(count, |_| Ok(()))
    }

    /// Select chunks for eviction based on policy
    fn select_eviction_candidates(&self, count: usize) -> CoreResult<Vec<ChunkId>> {
        let access_order = self.access_order.lock().unwrap();
        let metadata_map = self.metadata.read().unwrap();

        let candidates: Vec<ChunkId> = match self.config.cache_policy {
            CachePolicy::Lru => {
                // Evict least recently used (front of queue)
                access_order.iter().take(count).cloned().collect()
            }
            CachePolicy::Mru => {
                // Evict most recently used (back of queue)
                access_order.iter().rev().take(count).cloned().collect()
            }
            CachePolicy::Fifo => {
                // Evict first in (front of queue)
                access_order.iter().take(count).cloned().collect()
            }
            CachePolicy::Lfu => {
                // Evict least frequently used
                let mut candidates: Vec<_> = metadata_map
                    .iter()
                    .map(|(id, metadata)| (id.clone(), metadata.access_count))
                    .collect();
                candidates.sort_by_key(|(_, count)| *count);
                candidates
                    .into_iter()
                    .take(count)
                    .map(|(id, _)| id)
                    .collect()
            }
        };

        if candidates.is_empty() {
            Err(OutOfCoreError::CacheFull.into())
        } else {
            Ok(candidates)
        }
    }

    /// Get cache statistics
    pub fn get_statistics(&self) -> CacheStatistics {
        let chunks = self.chunks.read().unwrap();
        let metadata_map = self.metadata.read().unwrap();
        let current_memory = *self.current_memory.lock().unwrap();

        let dirty_count = metadata_map.values().filter(|m| m.is_dirty).count();

        CacheStatistics {
            cached_chunks: chunks.len(),
            memory_usage: current_memory,
            dirty_chunks: dirty_count,
            hit_rate: 0.0, // Would be calculated with hit/miss counters
        }
    }

    /// Flush all dirty chunks to storage
    pub fn flush_dirty_chunks(&self) -> CoreResult<Vec<ChunkId>> {
        let metadata_map = self.metadata.read().unwrap();
        let dirty_chunks: Vec<ChunkId> = metadata_map
            .iter()
            .filter(|(_, metadata)| metadata.is_dirty)
            .map(|(id, _)| id.clone())
            .collect();

        Ok(dirty_chunks)
    }

    /// Mark a chunk as clean (not dirty)
    pub fn mark_clean(&self, chunk_id: &ChunkId) {
        let mut metadata_map = self.metadata.write().unwrap();
        if let Some(metadata) = metadata_map.get_mut(chunk_id) {
            metadata.is_dirty = false;
        }
    }

    /// Mark a chunk as dirty (modified)
    pub fn mark_dirty(&self, chunk_id: &ChunkId) {
        let mut metadata_map = self.metadata.write().unwrap();
        if let Some(metadata) = metadata_map.get_mut(chunk_id) {
            metadata.mark_dirty();
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    /// Number of chunks in cache
    pub cached_chunks: usize,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Number of dirty (modified) chunks
    pub dirty_chunks: usize,
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
}

/// Storage backend for out-of-core arrays
pub trait StorageBackend: Send + Sync {
    /// Read chunk data from storage
    fn read_chunk(&self, metadata: &ChunkMetadata) -> CoreResult<Vec<u8>>;

    /// Write chunk data to storage
    fn write_chunk(&self, metadata: &ChunkMetadata, data: &[u8]) -> CoreResult<()>;

    /// Allocate space for a new chunk
    fn allocate_chunk(&self, chunk_id: &ChunkId, size: usize) -> CoreResult<ChunkMetadata>;

    /// Deallocate chunk space
    fn deallocate_chunk(&self, chunk_id: &ChunkId) -> CoreResult<()>;

    /// Flush any pending writes
    fn flush(&self) -> CoreResult<()>;
}

/// File-based storage backend
pub struct FileStorageBackend {
    base_path: PathBuf,
    file_handles: RwLock<HashMap<String, Arc<Mutex<File>>>>,
    chunk_registry: RwLock<HashMap<ChunkId, ChunkMetadata>>,
}

impl FileStorageBackend {
    /// Create a new file storage backend
    pub fn new<P: AsRef<Path>>(base_path: P) -> CoreResult<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path)?;

        Ok(Self {
            base_path,
            file_handles: RwLock::new(HashMap::new()),
            chunk_registry: RwLock::new(HashMap::new()),
        })
    }

    /// Get file handle for an array
    fn get_file_handle(&self, array_id: &str) -> CoreResult<Arc<Mutex<File>>> {
        let mut handles = self.file_handles.write().unwrap();

        if let Some(handle) = handles.get(array_id) {
            Ok(handle.clone())
        } else {
            let file_path = self.base_path.join(format!("{}.dat", array_id));
            let file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(file_path)?;

            let handle = Arc::new(Mutex::new(file));
            handles.insert(array_id.to_string(), handle.clone());
            Ok(handle)
        }
    }
}

impl StorageBackend for FileStorageBackend {
    fn read_chunk(&self, metadata: &ChunkMetadata) -> CoreResult<Vec<u8>> {
        let file_handle = self.get_file_handle(&metadata.id.array_id)?;
        let mut file = file_handle.lock().unwrap();

        file.seek(SeekFrom::Start(metadata.file_offset))?;
        let mut buffer = vec![0u8; metadata.size_bytes];
        file.read_exact(&mut buffer)?;

        Ok(buffer)
    }

    fn write_chunk(&self, metadata: &ChunkMetadata, data: &[u8]) -> CoreResult<()> {
        let file_handle = self.get_file_handle(&metadata.id.array_id)?;
        let mut file = file_handle.lock().unwrap();

        file.seek(SeekFrom::Start(metadata.file_offset))?;
        file.write_all(data)?;

        Ok(())
    }

    fn allocate_chunk(&self, chunk_id: &ChunkId, size: usize) -> CoreResult<ChunkMetadata> {
        let file_handle = self.get_file_handle(&chunk_id.array_id)?;
        let file = file_handle.lock().unwrap();

        let file_offset = file.metadata()?.len();
        let shape = vec![size / std::mem::size_of::<f64>()]; // Simplified shape calculation

        let metadata = ChunkMetadata::new(chunk_id.clone(), shape, file_offset);

        let mut registry = self.chunk_registry.write().unwrap();
        registry.insert(chunk_id.clone(), metadata.clone());

        Ok(metadata)
    }

    fn deallocate_chunk(&self, chunk_id: &ChunkId) -> CoreResult<()> {
        let mut registry = self.chunk_registry.write().unwrap();
        registry.remove(chunk_id);
        Ok(())
    }

    fn flush(&self) -> CoreResult<()> {
        let handles = self.file_handles.read().unwrap();
        for handle in handles.values() {
            let mut file = handle.lock().unwrap();
            file.flush()?;
        }
        Ok(())
    }
}

/// Out-of-core array implementation
pub struct OutOfCoreArray<T> {
    /// Array identifier
    array_id: String,
    /// Total array shape
    shape: Vec<usize>,
    /// Chunk cache
    cache: Arc<ChunkCache<T>>,
    /// Storage backend
    storage: Arc<dyn StorageBackend>,
    /// Configuration
    config: OutOfCoreConfig,
    /// Chunk mapping (logical chunks to storage chunks)
    chunk_map: RwLock<HashMap<Vec<usize>, ChunkId>>,
}

impl<T> OutOfCoreArray<T>
where
    T: Clone + Default + 'static + Send + Sync + serde::Serialize + serde::de::DeserializeOwned,
{
    /// Create a new out-of-core array
    pub fn new(
        array_id: String,
        shape: Vec<usize>,
        storage: Arc<dyn StorageBackend>,
        config: OutOfCoreConfig,
    ) -> Self {
        let cache = Arc::new(ChunkCache::new(config.clone()));

        Self {
            array_id,
            shape,
            cache,
            storage,
            config,
            chunk_map: RwLock::new(HashMap::new()),
        }
    }

    /// Get the shape of the array
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if array is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Calculate chunk coordinates for a given index
    #[allow(dead_code)]
    fn calculate_chunk_coords(&self, indices: &[usize]) -> CoreResult<Vec<usize>> {
        if indices.len() != self.shape.len() {
            return Err(OutOfCoreError::InvalidChunkSize(format!(
                "Index dimensions {} don't match array dimensions {}",
                indices.len(),
                self.shape.len()
            ))
            .into());
        }

        let chunk_coords: Vec<usize> = indices
            .iter()
            .zip(self.config.chunk_shape.iter())
            .map(|(&idx, &chunk_size)| idx / chunk_size)
            .collect();

        Ok(chunk_coords)
    }

    /// Get chunk for given chunk coordinates
    fn get_chunk(&self, chunk_coords: &[usize]) -> CoreResult<Array<T, IxDyn>> {
        // Check if chunk exists and get its ID
        let chunk_id_opt = {
            let chunk_map = self.chunk_map.read().unwrap();
            chunk_map.get(chunk_coords).cloned()
        };

        if let Some(chunk_id) = chunk_id_opt {
            // Try to get from cache first
            if let Some(chunk) = self.cache.get(&chunk_id) {
                return Ok(chunk);
            }

            // Load from storage
            self.load_chunk_from_storage(&chunk_id)
        } else {
            // Create new chunk
            self.create_new_chunk(chunk_coords)
        }
    }

    /// Get mutable chunk for given chunk coordinates
    #[allow(dead_code)]
    fn get_chunk_mut(&self, chunk_coords: &[usize]) -> CoreResult<Array<T, IxDyn>> {
        // First get the chunk (loading from storage if needed)
        let chunk = self.get_chunk(chunk_coords)?;

        // Get the chunk ID to mark it as dirty - release lock immediately
        {
            let chunk_map = self.chunk_map.read().unwrap();
            if let Some(chunk_id) = chunk_map.get(chunk_coords) {
                // Mark chunk as dirty since it will be modified
                self.cache.mark_dirty(chunk_id);
            }
        }

        Ok(chunk)
    }

    /// Set chunk data for given chunk coordinates
    fn set_chunk(&self, chunk_coords: &[usize], data: Array<T, IxDyn>) -> CoreResult<()> {
        // Get or create the chunk
        let _ = self.get_chunk(chunk_coords)?;

        // Get the chunk ID - release lock immediately after cloning
        let chunk_id = {
            let chunk_map = self.chunk_map.read().unwrap();
            chunk_map
                .get(chunk_coords)
                .ok_or_else(|| {
                    OutOfCoreError::ChunkNotFound(format!("Chunk at {:?}", chunk_coords))
                })?
                .clone()
        };

        // Create metadata for the chunk
        let metadata = ChunkMetadata::new(
            chunk_id.clone(),
            data.shape().to_vec(),
            0, // File offset will be managed by storage backend
        );

        // Put the chunk in cache and mark as dirty
        let writer = |chunk_id: &ChunkId| self.write_chunk_to_storage(chunk_id);
        self.cache
            .put_with_writer(chunk_id.clone(), data, metadata, writer)?;
        self.cache.mark_dirty(&chunk_id);

        Ok(())
    }

    /// Load chunk from storage
    fn load_chunk_from_storage(&self, chunk_id: &ChunkId) -> CoreResult<Array<T, IxDyn>> {
        // Get metadata for this chunk
        let metadata = ChunkMetadata::new(
            chunk_id.clone(),
            self.config.chunk_shape.clone(),
            0, // Would be looked up from storage registry
        );

        // Read raw data from storage
        let data = self.storage.read_chunk(&metadata)?;

        // Deserialize data to array
        // For simplicity, assume T is f64 and data is raw bytes
        let chunk = self.deserialize_chunk_data(&data, &metadata.shape)?;

        // Cache the chunk
        let writer = |chunk_id: &ChunkId| self.write_chunk_to_storage(chunk_id);
        self.cache
            .put_with_writer(chunk_id.clone(), chunk.clone(), metadata, writer)?;

        Ok(chunk)
    }

    /// Create a new chunk
    fn create_new_chunk(&self, chunk_coords: &[usize]) -> CoreResult<Array<T, IxDyn>> {
        let chunk_id = ChunkId::new(self.array_id.clone(), chunk_coords.to_vec());

        // Calculate actual chunk shape (may be smaller at boundaries)
        let chunk_shape = self.calculate_actual_chunk_shape(chunk_coords);

        // Create zero-initialized chunk
        let chunk = Array::<T, IxDyn>::default(IxDyn(&chunk_shape));

        // Register chunk in storage
        let chunk_size = chunk.len() * std::mem::size_of::<T>();
        let metadata = self.storage.allocate_chunk(&chunk_id, chunk_size)?;

        // Update chunk mapping
        let mut chunk_map = self.chunk_map.write().unwrap();
        chunk_map.insert(chunk_coords.to_vec(), chunk_id.clone());

        // Cache the chunk
        let writer = |chunk_id: &ChunkId| self.write_chunk_to_storage(chunk_id);
        self.cache
            .put_with_writer(chunk_id, chunk.clone(), metadata, writer)?;

        Ok(chunk)
    }

    /// Calculate actual chunk shape (handles boundary chunks)
    fn calculate_actual_chunk_shape(&self, chunk_coords: &[usize]) -> Vec<usize> {
        chunk_coords
            .iter()
            .zip(self.config.chunk_shape.iter())
            .zip(self.shape.iter())
            .map(|((&coord, &chunk_size), &total_size)| {
                let start = coord * chunk_size;
                let end = ((coord + 1) * chunk_size).min(total_size);
                end - start
            })
            .collect()
    }

    /// Deserialize chunk data from bytes
    fn deserialize_chunk_data(&self, data: &[u8], shape: &[usize]) -> CoreResult<Array<T, IxDyn>> {
        // For simplicity, assuming T implements serde traits
        // In a real implementation, would use bincode or similar
        use bincode::deserialize;

        if data.is_empty() {
            // Return default initialized array if no data
            return Ok(Array::<T, IxDyn>::default(IxDyn(shape)));
        }

        // Try to deserialize the data
        match deserialize::<Vec<T>>(data) {
            Ok(vec_data) => {
                let total_elements: usize = shape.iter().product();
                if vec_data.len() != total_elements {
                    return Err(OutOfCoreError::SerializationError(format!(
                        "Data length {} does not match expected shape {:?} (total: {})",
                        vec_data.len(),
                        shape,
                        total_elements
                    ))
                    .into());
                }

                Array::from_shape_vec(IxDyn(shape), vec_data)
                    .map_err(|e| OutOfCoreError::SerializationError(e.to_string()).into())
            }
            Err(e) => Err(OutOfCoreError::SerializationError(e.to_string()).into()),
        }
    }

    /// Serialize chunk data to bytes
    fn serialize_chunk_data(&self, chunk: &Array<T, IxDyn>) -> CoreResult<Vec<u8>> {
        use bincode::serialize;

        // Convert array to vec for serialization
        let vec_data: Vec<T> = chunk.iter().cloned().collect();

        serialize(&vec_data).map_err(|e| OutOfCoreError::SerializationError(e.to_string()).into())
    }

    /// Get a view of a specific region
    pub fn view_region(&self, ranges: &[(usize, usize)]) -> CoreResult<RegionView<T>> {
        RegionView::new(self, ranges)
    }

    /// Process the array in chunks with a user-provided function
    pub fn process_chunks<F>(&self, mut processor: F) -> CoreResult<()>
    where
        F: FnMut(&Array<T, IxDyn>, &[usize]) -> CoreResult<()>,
    {
        let total_chunks: Vec<usize> = self
            .shape
            .iter()
            .zip(self.config.chunk_shape.iter())
            .map(|(&total, &chunk_size)| total.div_ceil(chunk_size))
            .collect();

        // Iterate through all possible chunk coordinates
        self.iterate_chunk_coords(
            &total_chunks,
            &mut processor,
            &mut vec![0; total_chunks.len()],
            0,
        )
    }

    /// Recursively iterate through chunk coordinates
    fn iterate_chunk_coords<F>(
        &self,
        total_chunks: &[usize],
        processor: &mut F,
        current_coords: &mut Vec<usize>,
        dimension: usize,
    ) -> CoreResult<()>
    where
        F: FnMut(&Array<T, IxDyn>, &[usize]) -> CoreResult<()>,
    {
        if dimension == total_chunks.len() {
            // Process this chunk
            let chunk = self.get_chunk(current_coords)?;
            processor(&chunk, current_coords)?;
        } else {
            // Recurse to next dimension
            for i in 0..total_chunks[dimension] {
                current_coords[dimension] = i;
                self.iterate_chunk_coords(total_chunks, processor, current_coords, dimension + 1)?;
            }
        }
        Ok(())
    }

    /// Flush all dirty chunks to storage
    pub fn flush(&self) -> CoreResult<()> {
        // Get all dirty chunks from cache
        let dirty_chunk_ids = self.cache.flush_dirty_chunks()?;

        // Write each dirty chunk to storage
        for chunk_id in dirty_chunk_ids {
            self.write_chunk_to_storage(&chunk_id)?;
        }

        // Flush storage backend
        self.storage.flush()?;
        Ok(())
    }

    /// Write a single chunk to storage
    fn write_chunk_to_storage(&self, chunk_id: &ChunkId) -> CoreResult<()> {
        // Get chunk from cache
        if let Some(chunk) = self.cache.get(chunk_id) {
            // Get metadata from chunk map or create new
            let chunk_map = self.chunk_map.read().unwrap();
            let _chunk_coords = chunk_map
                .iter()
                .find(|(_, id)| *id == chunk_id)
                .map(|(coords, _)| coords.clone())
                .ok_or_else(|| OutOfCoreError::ChunkNotFound(chunk_id.to_string()))?;

            // Create metadata for storage
            let metadata = ChunkMetadata::new(
                chunk_id.clone(),
                chunk.shape().to_vec(),
                0, // File offset will be managed by storage backend
            );

            // Serialize chunk data
            let data = self.serialize_chunk_data(&chunk)?;

            // Write to storage backend
            self.storage.write_chunk(&metadata, &data)?;

            // Mark chunk as clean in cache
            self.cache.mark_clean(chunk_id);
        }

        Ok(())
    }

    /// Get array statistics
    pub fn get_statistics(&self) -> ArrayStatistics {
        let cache_stats = self.cache.get_statistics();
        let chunk_map = self.chunk_map.read().unwrap();

        ArrayStatistics {
            array_id: self.array_id.clone(),
            total_elements: self.len(),
            total_chunks: chunk_map.len(),
            cache_stats,
        }
    }
}

/// Region view for accessing specific parts of an out-of-core array
pub struct RegionView<'a, T> {
    array: &'a OutOfCoreArray<T>,
    ranges: Vec<(usize, usize)>,
}

impl<'a, T> RegionView<'a, T>
where
    T: Clone + Default + 'static + Send + Sync + serde::Serialize + serde::de::DeserializeOwned,
{
    /// Create a new region view
    fn new(array: &'a OutOfCoreArray<T>, ranges: &[(usize, usize)]) -> CoreResult<Self> {
        if ranges.len() != array.shape.len() {
            return Err(OutOfCoreError::InvalidChunkSize(format!(
                "Range dimensions {} don't match array dimensions {}",
                ranges.len(),
                array.shape.len()
            ))
            .into());
        }

        Ok(Self {
            array,
            ranges: ranges.to_vec(),
        })
    }

    /// Get the shape of this region
    pub fn shape(&self) -> Vec<usize> {
        self.ranges.iter().map(|(start, end)| end - start).collect()
    }

    /// Process chunks that intersect with this region
    pub fn process_intersecting_chunks<F>(&self, mut processor: F) -> CoreResult<()>
    where
        F: FnMut(&Array<T, IxDyn>, &[usize], &[(usize, usize)]) -> CoreResult<()>,
    {
        // Calculate which chunks intersect with this region
        let chunk_ranges: Vec<(usize, usize)> = self
            .ranges
            .iter()
            .zip(self.array.config.chunk_shape.iter())
            .map(|((start, end), &chunk_size)| {
                let chunk_start = start / chunk_size;
                let chunk_end = (end - 1) / chunk_size + 1;
                (chunk_start, chunk_end)
            })
            .collect();

        self.iterate_region_chunks(
            &chunk_ranges,
            &mut processor,
            &mut vec![0; chunk_ranges.len()],
            0,
        )
    }

    /// Recursively iterate through chunks in the region
    fn iterate_region_chunks<F>(
        &self,
        chunk_ranges: &[(usize, usize)],
        processor: &mut F,
        current_coords: &mut Vec<usize>,
        dimension: usize,
    ) -> CoreResult<()>
    where
        F: FnMut(&Array<T, IxDyn>, &[usize], &[(usize, usize)]) -> CoreResult<()>,
    {
        if dimension == chunk_ranges.len() {
            // Process this chunk
            let chunk = self.array.get_chunk(current_coords)?;

            // Calculate intersection of chunk with region
            let intersection = self.calculate_chunk_intersection(current_coords);

            processor(&chunk, current_coords, &intersection)?;
        } else {
            // Recurse to next dimension
            let (start, end) = chunk_ranges[dimension];
            for i in start..end {
                current_coords[dimension] = i;
                self.iterate_region_chunks(chunk_ranges, processor, current_coords, dimension + 1)?;
            }
        }
        Ok(())
    }

    /// Calculate intersection of a chunk with this region
    fn calculate_chunk_intersection(&self, chunk_coords: &[usize]) -> Vec<(usize, usize)> {
        chunk_coords
            .iter()
            .zip(self.array.config.chunk_shape.iter())
            .zip(self.ranges.iter())
            .map(|((&coord, &chunk_size), &(region_start, region_end))| {
                let chunk_start = coord * chunk_size;
                let chunk_end = chunk_start + chunk_size;

                let intersect_start = chunk_start.max(region_start) - chunk_start;
                let intersect_end = chunk_end.min(region_end) - chunk_start;

                (intersect_start, intersect_end)
            })
            .collect()
    }
}

/// Statistics for an out-of-core array
#[derive(Debug, Clone)]
pub struct ArrayStatistics {
    /// Array identifier
    pub array_id: String,
    /// Total number of elements
    pub total_elements: usize,
    /// Total number of chunks
    pub total_chunks: usize,
    /// Cache statistics
    pub cache_stats: CacheStatistics,
}

/// Out-of-core array manager for handling multiple arrays
pub struct OutOfCoreManager {
    arrays: RwLock<HashMap<String, Box<dyn std::any::Any + Send + Sync>>>,
    storage_backends: RwLock<HashMap<String, Arc<dyn StorageBackend>>>,
    default_config: OutOfCoreConfig,
}

impl OutOfCoreManager {
    /// Create a new out-of-core manager
    pub fn new(default_config: OutOfCoreConfig) -> Self {
        Self {
            arrays: RwLock::new(HashMap::new()),
            storage_backends: RwLock::new(HashMap::new()),
            default_config,
        }
    }

    /// Register a storage backend
    pub fn register_storage_backend(&self, name: String, backend: Arc<dyn StorageBackend>) {
        let mut backends = self.storage_backends.write().unwrap();
        backends.insert(name, backend);
    }

    /// Create a new out-of-core array
    pub fn create_array<T>(
        &self,
        array_id: String,
        shape: Vec<usize>,
        storage_name: Option<String>,
        config: Option<OutOfCoreConfig>,
    ) -> CoreResult<Arc<OutOfCoreArray<T>>>
    where
        T: Clone + Default + 'static + Send + Sync + serde::Serialize + serde::de::DeserializeOwned,
    {
        let storage_backends = self.storage_backends.read().unwrap();
        let storage = if let Some(name) = storage_name {
            storage_backends
                .get(&name)
                .ok_or_else(|| OutOfCoreError::ChunkNotFound(format!("Storage backend: {}", name)))?
                .clone()
        } else {
            // Use default file storage
            Arc::new(FileStorageBackend::new("./out_of_core_data")?)
        };

        let config = config.unwrap_or_else(|| self.default_config.clone());
        let array = Arc::new(OutOfCoreArray::new(
            array_id.clone(),
            shape,
            storage,
            config,
        ));

        let mut arrays = self.arrays.write().unwrap();
        arrays.insert(array_id, Box::new(array.clone()));

        Ok(array)
    }

    /// Get an existing array
    pub fn get_array<T>(&self, array_id: &str) -> Option<Arc<OutOfCoreArray<T>>>
    where
        T: Clone + Default + 'static + Send + Sync + serde::Serialize + serde::de::DeserializeOwned,
    {
        let arrays = self.arrays.read().unwrap();
        arrays
            .get(array_id)
            .and_then(|boxed| boxed.downcast_ref::<Arc<OutOfCoreArray<T>>>())
            .cloned()
    }

    /// Remove an array
    pub fn remove_array(&self, array_id: &str) -> bool {
        let mut arrays = self.arrays.write().unwrap();
        arrays.remove(array_id).is_some()
    }

    /// List all array IDs
    pub fn list_arrays(&self) -> Vec<String> {
        let arrays = self.arrays.read().unwrap();
        arrays.keys().cloned().collect()
    }

    /// Get overall statistics
    pub fn get_overall_statistics(&self) -> ManagerStatistics {
        let arrays = self.arrays.read().unwrap();
        let total_arrays = arrays.len();

        // TODO: Aggregate statistics from all arrays
        ManagerStatistics {
            total_arrays,
            total_memory_usage: 0,
            total_cached_chunks: 0,
        }
    }
}

impl Default for OutOfCoreManager {
    fn default() -> Self {
        Self::new(OutOfCoreConfig::default())
    }
}

/// Manager statistics
#[derive(Debug, Clone)]
pub struct ManagerStatistics {
    /// Total number of arrays
    pub total_arrays: usize,
    /// Total memory usage for caching
    pub total_memory_usage: usize,
    /// Total cached chunks across all arrays
    pub total_cached_chunks: usize,
}

/// Global out-of-core manager instance
static GLOBAL_MANAGER: std::sync::OnceLock<Arc<OutOfCoreManager>> = std::sync::OnceLock::new();

/// Get the global out-of-core manager
pub fn global_manager() -> Arc<OutOfCoreManager> {
    GLOBAL_MANAGER
        .get_or_init(|| Arc::new(OutOfCoreManager::default()))
        .clone()
}

/// Convenience functions for out-of-core processing
pub mod utils {
    use super::*;

    /// Create a simple out-of-core array with default settings
    pub fn create_simple_array<T>(
        array_id: String,
        shape: Vec<usize>,
    ) -> CoreResult<Arc<OutOfCoreArray<T>>>
    where
        T: Clone + Default + 'static + Send + Sync + serde::Serialize + serde::de::DeserializeOwned,
    {
        let manager = global_manager();
        manager.create_array(array_id, shape, None, None)
    }

    /// Process a large dataset that doesn't fit in memory
    pub fn process_large_dataset<T, F>(
        data_path: &Path,
        shape: Vec<usize>,
        chunk_processor: F,
    ) -> CoreResult<()>
    where
        T: Clone + Default + 'static + Send + Sync + serde::Serialize + serde::de::DeserializeOwned,
        F: FnMut(&Array<T, IxDyn>, &[usize]) -> CoreResult<()>,
    {
        // Create storage backend for the data file
        let storage = Arc::new(FileStorageBackend::new(data_path.parent().unwrap())?);

        // Create out-of-core array
        let config = OutOfCoreConfig::default();
        let array = OutOfCoreArray::new(
            data_path.file_stem().unwrap().to_string_lossy().to_string(),
            shape,
            storage,
            config,
        );

        // Process all chunks
        array.process_chunks(chunk_processor)?;

        Ok(())
    }

    /// Helper function to recursively copy chunks from in-memory to out-of-core array
    fn copy_chunks_recursive<T>(
        source_array: &Array<T, IxDyn>,
        target_array: &OutOfCoreArray<T>,
        chunks_per_dim: &[usize],
        chunk_coords: &mut Vec<usize>,
        dimension: usize,
    ) -> CoreResult<()>
    where
        T: Clone + Default + 'static + Send + Sync + serde::Serialize + serde::de::DeserializeOwned,
    {
        if dimension == chunks_per_dim.len() {
            // We've reached the deepest dimension, copy this chunk

            // Calculate the slice ranges for this chunk
            let chunk_shape = &target_array.config.chunk_shape;
            let mut slices = vec![];

            for (i, (&coord, &chunk_size)) in
                chunk_coords.iter().zip(chunk_shape.iter()).enumerate()
            {
                let start = coord * chunk_size;
                let end = ((coord + 1) * chunk_size).min(source_array.shape()[i]);
                slices.push(start..end);
            }

            // Extract the chunk data from the source array
            let chunk_data = extract_chunk_data(source_array, &slices)?;

            // Set the chunk data in the target array
            target_array.set_chunk(chunk_coords, chunk_data)?;

            Ok(())
        } else {
            // Iterate through all chunks in this dimension
            for i in 0..chunks_per_dim[dimension] {
                chunk_coords[dimension] = i;
                copy_chunks_recursive(
                    source_array,
                    target_array,
                    chunks_per_dim,
                    chunk_coords,
                    dimension + 1,
                )?;
            }
            Ok(())
        }
    }

    /// Extract a chunk of data from an array given slice ranges
    fn extract_chunk_data<T>(
        array: &Array<T, IxDyn>,
        slices: &[std::ops::Range<usize>],
    ) -> CoreResult<Array<T, IxDyn>>
    where
        T: Clone,
    {
        use ndarray::{SliceInfo, SliceInfoElem};

        // Convert ranges to SliceInfoElem
        let slice_info: Vec<SliceInfoElem> = slices
            .iter()
            .map(|range| SliceInfoElem::Slice {
                start: range.start as isize,
                end: Some(range.end as isize),
                step: 1,
            })
            .collect();

        // Create SliceInfo from elements
        let slice_info = SliceInfo::<Vec<SliceInfoElem>, IxDyn, IxDyn>::try_from(slice_info)
            .map_err(|e| OutOfCoreError::InvalidChunkSize(e.to_string()))?;

        // Slice the array and convert to owned
        Ok(array.slice(slice_info).to_owned())
    }

    /// Convert an in-memory array to out-of-core format
    pub fn convert_to_out_of_core<T>(
        array: &Array<T, IxDyn>,
        array_id: String,
        chunk_shape: Vec<usize>,
    ) -> CoreResult<Arc<OutOfCoreArray<T>>>
    where
        T: Clone + Default + 'static + Send + Sync + serde::Serialize + serde::de::DeserializeOwned,
    {
        let config = OutOfCoreConfig {
            chunk_shape,
            ..Default::default()
        };

        let manager = global_manager();
        let out_of_core_array =
            manager.create_array(array_id, array.shape().to_vec(), None, Some(config))?;

        // Copy data from in-memory array to out-of-core array
        let chunk_shape = &out_of_core_array.config.chunk_shape;
        let array_shape = array.shape();

        // Calculate the number of chunks needed in each dimension
        let chunks_per_dim: Vec<usize> = array_shape
            .iter()
            .zip(chunk_shape.iter())
            .map(|(&total, &chunk)| total.div_ceil(chunk))
            .collect();

        // Iterate through all chunks and copy data
        let mut chunk_coords = vec![0; chunks_per_dim.len()];
        copy_chunks_recursive(
            array,
            &out_of_core_array,
            &chunks_per_dim,
            &mut chunk_coords,
            0,
        )?;

        // Flush to ensure all data is written
        out_of_core_array.flush()?;

        Ok(out_of_core_array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_chunk_id_creation() {
        let chunk_id = ChunkId::new("test_array".to_string(), vec![0, 1, 2]);
        assert_eq!(chunk_id.array_id, "test_array");
        assert_eq!(chunk_id.coordinates, vec![0, 1, 2]);
        assert_eq!(format!("{}", chunk_id), "test_array:0,1,2");
    }

    #[test]
    fn test_chunk_metadata() {
        let chunk_id = ChunkId::new("test".to_string(), vec![0, 0]);
        let mut metadata = ChunkMetadata::new(chunk_id, vec![100, 100], 0);

        let initial_access_count = metadata.access_count;
        metadata.touch();
        assert!(metadata.access_count > initial_access_count);
        assert!(!metadata.is_dirty);

        metadata.mark_dirty();
        assert!(metadata.is_dirty);
    }

    #[test]
    fn test_out_of_core_config() {
        let config = OutOfCoreConfig::default();
        assert_eq!(config.max_cache_memory, 1024 * 1024 * 1024);
        assert_eq!(config.max_cached_chunks, 100);
        assert_eq!(config.chunk_shape, vec![1000, 1000]);
        assert!(matches!(config.cache_policy, CachePolicy::Lru));
    }

    #[test]
    fn test_cache_policy_variants() {
        let policies = [
            CachePolicy::Lru,
            CachePolicy::Lfu,
            CachePolicy::Fifo,
            CachePolicy::Mru,
        ];

        // Test that all policies can be created and cloned
        for policy in &policies {
            let cloned = policy.clone();
            match (policy, &cloned) {
                (CachePolicy::Lru, CachePolicy::Lru) => {}
                (CachePolicy::Lfu, CachePolicy::Lfu) => {}
                (CachePolicy::Fifo, CachePolicy::Fifo) => {}
                (CachePolicy::Mru, CachePolicy::Mru) => {}
                _ => panic!("Policy clone mismatch"),
            }
        }
    }

    #[test]
    fn test_file_storage_backend() -> CoreResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileStorageBackend::new(temp_dir.path())?;

        let chunk_id = ChunkId::new("test_array".to_string(), vec![0, 0]);
        let metadata = storage.allocate_chunk(&chunk_id, 1024)?;

        assert_eq!(metadata.id, chunk_id);
        assert_eq!(metadata.size_bytes, 1024);

        // Test write and read
        let test_data = vec![1u8, 2, 3, 4, 5];
        storage.write_chunk(&metadata, &test_data)?;

        // Note: Reading would require updating metadata with correct size

        storage.flush()?;
        storage.deallocate_chunk(&chunk_id)?;

        Ok(())
    }

    #[test]
    fn test_chunk_cache() {
        let config = OutOfCoreConfig {
            max_cached_chunks: 3,   // Increase to allow both chunks
            max_cache_memory: 2048, // Increase memory limit to allow both chunks (800 bytes each)
            ..Default::default()
        };

        let cache = ChunkCache::<f64>::new(config);

        let chunk_id1 = ChunkId::new("test".to_string(), vec![0, 0]);
        let chunk_id2 = ChunkId::new("test".to_string(), vec![0, 1]);

        let chunk1 = Array::<f64, IxDyn>::zeros(IxDyn(&[10, 10]));
        let chunk2 = Array::<f64, IxDyn>::zeros(IxDyn(&[10, 10]));

        let metadata1 = ChunkMetadata::new(chunk_id1.clone(), vec![10, 10], 0);
        let metadata2 = ChunkMetadata::new(chunk_id2.clone(), vec![10, 10], 100);

        // Test putting and getting chunks
        assert!(cache.put(chunk_id1.clone(), chunk1, metadata1).is_ok());
        assert!(cache.put(chunk_id2.clone(), chunk2, metadata2).is_ok());

        assert!(cache.get(&chunk_id1).is_some());
        assert!(cache.get(&chunk_id2).is_some());

        let stats = cache.get_statistics();
        assert_eq!(stats.cached_chunks, 2);
    }

    #[test]
    fn test_chunk_cache_eviction() {
        let config = OutOfCoreConfig {
            max_cached_chunks: 2,   // Only allow 2 chunks
            max_cache_memory: 2048, // Plenty of memory
            cache_policy: CachePolicy::Lru,
            ..Default::default()
        };

        let cache = ChunkCache::<f64>::new(config);

        let chunk_id1 = ChunkId::new("test".to_string(), vec![0, 0]);
        let chunk_id2 = ChunkId::new("test".to_string(), vec![0, 1]);
        let chunk_id3 = ChunkId::new("test".to_string(), vec![0, 2]);

        let chunk1 = Array::<f64, IxDyn>::zeros(IxDyn(&[10, 10]));
        let chunk2 = Array::<f64, IxDyn>::zeros(IxDyn(&[10, 10]));
        let chunk3 = Array::<f64, IxDyn>::zeros(IxDyn(&[10, 10]));

        let metadata1 = ChunkMetadata::new(chunk_id1.clone(), vec![10, 10], 0);
        let metadata2 = ChunkMetadata::new(chunk_id2.clone(), vec![10, 10], 100);
        let metadata3 = ChunkMetadata::new(chunk_id3.clone(), vec![10, 10], 200);

        // Add first two chunks
        assert!(cache.put(chunk_id1.clone(), chunk1, metadata1).is_ok());
        assert!(cache.put(chunk_id2.clone(), chunk2, metadata2).is_ok());

        // Both should be accessible
        assert!(cache.get(&chunk_id1).is_some());
        assert!(cache.get(&chunk_id2).is_some());

        let stats = cache.get_statistics();
        assert_eq!(stats.cached_chunks, 2);

        // Add third chunk, which should evict the first one (LRU)
        assert!(cache.put(chunk_id3.clone(), chunk3, metadata3).is_ok());

        // First chunk should be evicted, others should be present
        assert!(cache.get(&chunk_id1).is_none());
        assert!(cache.get(&chunk_id2).is_some());
        assert!(cache.get(&chunk_id3).is_some());

        let stats = cache.get_statistics();
        assert_eq!(stats.cached_chunks, 2);
    }

    #[test]
    fn test_dirty_chunk_tracking() -> CoreResult<()> {
        let temp_dir = TempDir::new()?;
        let storage = Arc::new(FileStorageBackend::new(temp_dir.path())?);

        let config = OutOfCoreConfig {
            chunk_shape: vec![100, 100],
            max_cached_chunks: 2,
            ..Default::default()
        };

        let array =
            OutOfCoreArray::<f64>::new("test_dirty".to_string(), vec![200, 200], storage, config);

        // Create and set a chunk
        let chunk_coords = vec![0, 0];
        let chunk_data = Array::<f64, IxDyn>::ones(IxDyn(&[100, 100]));
        array.set_chunk(&chunk_coords, chunk_data)?;

        // Check that chunk is marked as dirty
        let cache_stats = array.cache.get_statistics();
        assert_eq!(cache_stats.dirty_chunks, 1);

        // Flush dirty chunks
        array.flush()?;

        // Check that chunk is no longer dirty
        let cache_stats = array.cache.get_statistics();
        assert_eq!(cache_stats.dirty_chunks, 0);

        Ok(())
    }

    #[test]
    fn test_out_of_core_manager() -> CoreResult<()> {
        let manager = OutOfCoreManager::default();

        // Test array creation
        let array_id = "test_array".to_string();
        let shape = vec![1000, 1000];

        let array: Arc<OutOfCoreArray<f64>> =
            manager.create_array(array_id.clone(), shape.clone(), None, None)?;

        assert_eq!(array.shape(), &shape);
        assert_eq!(array.len(), 1_000_000);

        // Test getting the array
        let retrieved: Option<Arc<OutOfCoreArray<f64>>> = manager.get_array(&array_id);
        assert!(retrieved.is_some());

        // Test listing arrays
        let array_list = manager.list_arrays();
        assert!(array_list.contains(&array_id));

        // Test removing array
        assert!(manager.remove_array(&array_id));
        assert!(!manager.list_arrays().contains(&array_id));

        Ok(())
    }

    #[test]
    fn test_global_manager() {
        let manager = global_manager();

        // Should return the same instance
        let manager2 = global_manager();
        assert!(Arc::ptr_eq(&manager, &manager2));

        let stats = manager.get_overall_statistics();
        assert_eq!(stats.total_arrays, 0);
    }
}
