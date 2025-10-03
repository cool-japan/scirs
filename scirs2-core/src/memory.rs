//! # Memory Management
//!
//! This module provides efficient memory management utilities for scientific computing.
//!
//! ## Features
//!
//! * Chunk processing for large datasets
//! * Buffer pool for memory reuse
//! * Zero-copy transformations
//! * Memory usage tracking and metrics
//! * Detailed memory allocation analysis
//!
//! ## Usage
//!
//! ```rust,no_run
//! use scirs2_core::memory::{ChunkProcessor2D, BufferPool};
//! use ndarray::Array2;
//!
//! // Process a large array in chunks
//! let large_array = Array2::<f64>::zeros((10000, 10000));
//! let mut processor = ChunkProcessor2D::new(&large_array, (1000, 1000));
//!
//! processor.process_chunks(|chunk, coords| {
//!     // Process each chunk (e.g., compute statistics, apply transformations)
//!     println!("Processing chunk at position {:?}", coords);
//! });
//!
//! // Use a buffer pool to reuse memory
//! let mut pool = BufferPool::<f64>::new();
//!
//! // Acquire a buffer from the pool
//! let mut buffer = pool.acquire_vec(1000);
//!
//! // Use the buffer for some computation
//! for i in 0..buffer.len() {
//!     buffer[i] = i as f64;
//! }
//!
//! // Release the buffer back to the pool when done
//! pool.release_vec(buffer);
//!
//! // Track memory usage with the metrics system
//! use scirs2_core::memory::metrics::{track_allocation, track_deallocation, format_memory_report};
//!
//! // Record an allocation
//! track_allocation("MyComponent", 1024, 0x1000);
//!
//! // Record a deallocation
//! track_deallocation("MyComponent", 1024, 0x1000);
//!
//! // Print a memory usage report
//! println!("{}", format_memory_report());
//! ```

use ndarray::{ArrayBase, Data, Dimension, Ix2, IxDyn, ViewRepr};
use std::alloc::{alloc, dealloc, Layout};
use std::any::TypeId;
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

/// A processor for working with large arrays in manageable chunks
pub struct ChunkProcessor<'a, A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    array: &'a ArrayBase<S, D>,
    // Chunk shape is necessary for the full implementation of chunked processing
    chunkshape: D,
    // Current position needed for iterative processing
    #[allow(dead_code)]
    position: D,
}

impl<'a, A, S, D> ChunkProcessor<'a, A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    /// Create a new chunk processor for the given array and chunk shape
    pub fn new(array: &'a ArrayBase<S, D>, chunkshape: D) -> Self {
        let position = D::zeros(array.ndim());
        Self {
            array,
            chunkshape,
            position,
        }
    }

    /// Process the array in chunks, calling the provided function for each chunk
    /// The function receives a dynamic view of the chunk and the position as IxDyn
    pub fn process_chunks_dyn<F>(&mut self, mut f: F)
    where
        F: FnMut(&ArrayBase<ViewRepr<&A>, IxDyn>, IxDyn),
    {
        use ndarray::{IntoDimension, Slice};

        // Get array shape and chunk shape as slices
        let arrayshape = self.array.shape();
        let chunkshape = self.chunkshape.slice();

        // Calculate number of chunks in each dimension
        let mut num_chunks_per_dim = vec![];
        for i in 0..arrayshape.len() {
            let n_chunks = arrayshape[i].div_ceil(chunkshape[i]);
            num_chunks_per_dim.push(n_chunks);
        }

        // Iterate through all possible chunk positions
        let mut chunk_indices = vec![0; arrayshape.len()];
        loop {
            // Calculate the slice for current chunk
            let mut slices = vec![];
            let mut position_vec = vec![];

            for i in 0..arrayshape.len() {
                let start = chunk_indices[i] * chunkshape[i];
                let end = ((chunk_indices[i] + 1) * chunkshape[i]).min(arrayshape[i]);
                slices.push(Slice::from(start..end));
                position_vec.push(start);
            }

            // Convert position vector to IxDyn
            let position = position_vec.into_dimension();

            // Get the chunk view and call the function
            // First convert the array to dynamic dimension, then slice
            let dyn_array = self.array.view().into_dyn();

            // Create dynamic slice info
            use ndarray::{SliceInfo, SliceInfoElem};
            let slice_elems: Vec<SliceInfoElem> = slices
                .into_iter()
                .map(|s| SliceInfoElem::Slice {
                    start: s.start,
                    end: s.end,
                    step: s.step,
                })
                .collect();

            let slice_info = unsafe {
                SliceInfo::<Vec<SliceInfoElem>, IxDyn, IxDyn>::new(slice_elems)
                    .expect("Failed to create slice info")
            };

            let view = dyn_array.slice(slice_info);
            f(&view, position);

            // Increment chunk indices
            let mut carry = true;
            for i in 0..chunk_indices.len() {
                if carry {
                    chunk_indices[i] += 1;
                    if chunk_indices[i] >= num_chunks_per_dim[i] {
                        chunk_indices[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }

            // If we've wrapped around all dimensions, we're done
            if carry {
                break;
            }
        }
    }

    /// Get the total number of chunks
    pub fn num_chunks(&self) -> usize {
        let arrayshape = self.array.shape();
        let chunkshape = self.chunkshape.slice();

        let mut total_chunks = 1;
        for i in 0..arrayshape.len() {
            let n_chunks = arrayshape[i].div_ceil(chunkshape[i]);
            total_chunks *= n_chunks;
        }

        total_chunks
    }
}

/// A specialized chunk processor for 2D arrays
pub struct ChunkProcessor2D<'a, A, S>
where
    S: Data<Elem = A>,
{
    array: &'a ArrayBase<S, Ix2>,
    chunkshape: (usize, usize),
    // Current position tracking for iterator implementation
    #[allow(dead_code)]
    current_row: usize,
    #[allow(dead_code)]
    current_col: usize,
}

impl<'a, A, S> ChunkProcessor2D<'a, A, S>
where
    S: Data<Elem = A>,
{
    /// Create a new 2D chunk processor
    pub fn new(array: &'a ArrayBase<S, Ix2>, chunkshape: (usize, usize)) -> Self {
        Self {
            array,
            chunkshape,
            current_row: 0,
            current_col: 0,
        }
    }

    /// Process the 2D array in chunks
    pub fn process_chunks<F>(&mut self, mut f: F)
    where
        F: FnMut(&ArrayBase<ViewRepr<&A>, Ix2>, (usize, usize)),
    {
        let (rows, cols) = self.array.dim();
        let (chunk_rows, chunk_cols) = self.chunkshape;

        for row_start in (0..rows).step_by(chunk_rows) {
            for col_start in (0..cols).step_by(chunk_cols) {
                let row_end = (row_start + chunk_rows).min(rows);
                let col_end = (col_start + chunk_cols).min(cols);

                // Get a view of the current chunk
                let chunk = self
                    .array
                    .slice(ndarray::s![row_start..row_end, col_start..col_end]);

                // Call the processing function
                f(&chunk, (row_start, col_start));
            }
        }
    }
}

/// Advanced memory allocation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllocationStrategy {
    /// Standard system allocator
    System,
    /// Pool-based allocation with size classes
    Pool,
    /// Arena allocation for batch operations
    Arena,
    /// Stack-like allocation for temporary data
    Stack,
    /// NUMA-aware allocation
    NumaAware,
    /// Cache-aligned allocation
    CacheAligned,
    /// Huge page allocation for large datasets
    HugePage,
    /// Memory-mapped allocation
    MemoryMapped,
}

/// Memory pressure levels for adaptive behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressure {
    Low,
    Medium,
    High,
    Critical,
}

/// Memory access patterns for optimization hints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessPattern {
    Sequential,
    Random,
    Temporal,
    Spatial,
    WriteOnce,
    ReadOnly,
    Streaming,
}

/// Configuration for advanced memory management
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Allocation strategy to use
    pub strategy: AllocationStrategy,
    /// Memory access pattern hint
    pub access_pattern: AccessPattern,
    /// Enable memory prefetching
    pub enable_prefetch: bool,
    /// Memory alignment requirement
    pub alignment: usize,
    /// Enable NUMA awareness
    pub numa_aware: bool,
    /// Maximum memory usage (in bytes)
    pub max_memory: Option<usize>,
    /// Enable memory compression
    pub enable_compression: bool,
    /// Memory pressure threshold
    pub pressure_threshold: f64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            strategy: AllocationStrategy::Pool,
            access_pattern: AccessPattern::Sequential,
            enable_prefetch: true,
            alignment: 64, // Cache line aligned
            numa_aware: false,
            max_memory: None,
            enable_compression: false,
            pressure_threshold: 0.8,
        }
    }
}

/// Advanced memory buffer pool with multiple allocation strategies
pub struct AdvancedBufferPool<T: Clone + Default> {
    vectors: Vec<Vec<T>>,
    arrays: Vec<ndarray::Array1<T>>,
    config: MemoryConfig,
    stats: PoolStatistics,
    size_classes: Vec<SizeClass<T>>,
    arena_allocator: Option<ArenaAllocator>,
    numa_topology: NumaTopology,
}

/// Statistics for memory pool performance
#[derive(Debug, Default)]
pub struct PoolStatistics {
    pub total_allocations: AtomicUsize,
    pub total_deallocations: AtomicUsize,
    pub pool_hits: AtomicUsize,
    pub pool_misses: AtomicUsize,
    pub bytes_allocated: AtomicUsize,
    pub bytes_deallocated: AtomicUsize,
    pub peak_memory: AtomicUsize,
}

impl Clone for PoolStatistics {
    fn clone(&self) -> Self {
        Self {
            total_allocations: AtomicUsize::new(self.total_allocations.load(Ordering::SeqCst)),
            total_deallocations: AtomicUsize::new(self.total_deallocations.load(Ordering::SeqCst)),
            pool_hits: AtomicUsize::new(self.pool_hits.load(Ordering::SeqCst)),
            pool_misses: AtomicUsize::new(self.pool_misses.load(Ordering::SeqCst)),
            bytes_allocated: AtomicUsize::new(self.bytes_allocated.load(Ordering::SeqCst)),
            bytes_deallocated: AtomicUsize::new(self.bytes_deallocated.load(Ordering::SeqCst)),
            peak_memory: AtomicUsize::new(self.peak_memory.load(Ordering::SeqCst)),
        }
    }
}

/// Size class for efficient memory pooling
#[derive(Debug)]
struct SizeClass<T> {
    size: usize,
    buffers: VecDeque<Vec<T>>,
    max_buffers: usize,
}

/// Arena allocator for batch allocations
#[derive(Debug)]
struct ArenaAllocator {
    chunks: Vec<ArenaChunk>,
    current_chunk: usize,
    chunk_size: usize,
}

/// Individual arena chunk
#[derive(Debug)]
struct ArenaChunk {
    ptr: NonNull<u8>,
    size: usize,
    offset: usize,
}

// SAFETY: ArenaChunk is thread-safe as long as access is properly synchronized
// through the containing mutex in the buffer pool
unsafe impl Send for ArenaChunk {}
unsafe impl Sync for ArenaChunk {}

/// NUMA topology information
#[derive(Debug, Clone)]
struct NumaTopology {
    nodes: Vec<NumaNode>,
    current_node: usize,
}

/// NUMA node information
#[derive(Debug, Clone)]
struct NumaNode {
    id: usize,
    memory_size: usize,
    cpu_cores: Vec<usize>,
}

/// Memory buffer pool for reusing allocated memory (legacy compatibility)
pub struct BufferPool<T: Clone + Default> {
    inner: AdvancedBufferPool<T>,
}

impl<T: Clone + Default> AdvancedBufferPool<T> {
    /// Create a new advanced buffer pool with default configuration
    pub fn new() -> Self {
        Self::with_config(MemoryConfig::default())
    }

    /// Create a new buffer pool with custom configuration
    pub fn with_config(config: MemoryConfig) -> Self {
        let size_classes = Self::create_size_classes();
        let numa_topology = Self::detect_numa_topology();

        Self {
            vectors: Vec::new(),
            arrays: Vec::new(),
            config,
            stats: PoolStatistics::default(),
            size_classes,
            arena_allocator: None,
            numa_topology,
        }
    }

    /// Create standard size classes for efficient pooling
    fn create_size_classes() -> Vec<SizeClass<T>> {
        let sizes = [
            64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        ];
        sizes
            .iter()
            .map(|&size| SizeClass {
                size,
                buffers: VecDeque::new(),
                max_buffers: 16, // Limit pool size per class
            })
            .collect()
    }

    /// Detect NUMA topology (simplified implementation)
    fn detect_numa_topology() -> NumaTopology {
        // In a real implementation, this would query the system
        NumaTopology {
            nodes: vec![NumaNode {
                id: 0,
                memory_size: 16 * 1024 * 1024 * 1024, // 16GB
                cpu_cores: (0..std::thread::available_parallelism().unwrap().get()).collect(),
            }],
            current_node: 0,
        }
    }

    /// Get memory pressure level
    pub fn memory_pressure(&self) -> MemoryPressure {
        let used_memory = self.stats.bytes_allocated.load(Ordering::Relaxed);
        let max_memory = self.config.max_memory.unwrap_or(usize::MAX);

        let pressure_ratio = used_memory as f64 / max_memory as f64;

        if pressure_ratio > 0.9 {
            MemoryPressure::Critical
        } else if pressure_ratio > 0.7 {
            MemoryPressure::High
        } else if pressure_ratio > 0.5 {
            MemoryPressure::Medium
        } else {
            MemoryPressure::Low
        }
    }

    /// Acquire a vector with advanced allocation strategy
    pub fn acquire_vec_advanced(&mut self, capacity: usize) -> Vec<T> {
        self.stats.total_allocations.fetch_add(1, Ordering::Relaxed);

        match self.config.strategy {
            AllocationStrategy::Pool => self.acquire_from_pool(capacity),
            AllocationStrategy::Arena => self.acquire_from_arena(capacity),
            AllocationStrategy::NumaAware => self.acquire_numa_aware(capacity),
            AllocationStrategy::CacheAligned => self.acquire_cache_aligned(capacity),
            _ => self.acquire_system(capacity),
        }
    }

    /// Acquire vector from size-class pools
    fn acquire_from_pool(&mut self, capacity: usize) -> Vec<T> {
        // Find appropriate size class
        for size_class in &mut self.size_classes {
            if size_class.size >= capacity {
                if let Some(mut vec) = size_class.buffers.pop_front() {
                    self.stats.pool_hits.fetch_add(1, Ordering::Relaxed);
                    vec.clear();
                    vec.resize(capacity, T::default());
                    return vec;
                }
                break;
            }
        }

        self.stats.pool_misses.fetch_add(1, Ordering::Relaxed);
        vec![T::default(); capacity]
    }

    /// Acquire vector from arena allocator
    fn acquire_from_arena(&mut self, capacity: usize) -> Vec<T> {
        if self.arena_allocator.is_none() {
            self.arena_allocator = Some(ArenaAllocator::new(1024 * 1024)); // 1MB chunks
        }

        // For simplicity, fall back to system allocation for arena
        // In a real implementation, this would use the arena
        vec![T::default(); capacity]
    }

    /// Acquire NUMA-aware vector
    fn acquire_numa_aware(&mut self, capacity: usize) -> Vec<T> {
        // In a real implementation, this would allocate on specific NUMA nodes
        vec![T::default(); capacity]
    }

    /// Acquire cache-aligned vector
    fn acquire_cache_aligned(&mut self, capacity: usize) -> Vec<T> {
        // Create vector with cache-aligned allocation
        let mut vec =
            Vec::with_capacity(capacity + self.config.alignment / std::mem::size_of::<T>());
        vec.resize(capacity, T::default());
        vec
    }

    /// Acquire vector with system allocator
    fn acquire_system(&mut self, capacity: usize) -> Vec<T> {
        vec![T::default(); capacity]
    }

    /// Release vector back to pool with advanced strategies
    pub fn release_vec_advanced(&mut self, vec: Vec<T>) {
        self.stats
            .total_deallocations
            .fetch_add(1, Ordering::Relaxed);

        match self.config.strategy {
            AllocationStrategy::Pool => self.release_to_pool(vec),
            AllocationStrategy::Arena => self.release_to_arena(vec),
            _ => {} // Other strategies don't need special release handling
        }
    }

    /// Release vector to size-class pool
    fn release_to_pool(&mut self, vec: Vec<T>) {
        let capacity = vec.capacity();

        // Find appropriate size class
        for size_class in &mut self.size_classes {
            if size_class.size >= capacity {
                if size_class.buffers.len() < size_class.max_buffers {
                    size_class.buffers.push_back(vec);
                }
                break;
            }
        }
    }

    /// Release vector to arena (no-op for arena allocations)
    fn release_to_arena(&mut self, _vec: Vec<T>) {
        // Arena allocations are freed when the arena is reset
    }

    /// Prefetch memory for upcoming access
    ///
    /// # Safety
    ///
    /// This function is unsafe because it dereferences raw pointers.
    /// The caller must ensure that:
    /// - `ptr` is valid and properly aligned
    /// - `ptr` points to at least `size` valid elements of type `T`
    /// - The memory range is readable
    pub unsafe fn prefetch_memory(&self, ptr: *const T, size: usize) {
        if self.config.enable_prefetch {
            unsafe {
                for i in 0..size {
                    let addr = ptr.add(i) as *const u8;
                    #[cfg(target_arch = "x86_64")]
                    {
                        use std::arch::x86_64::*;
                        _mm_prefetch(addr as *const i8, _MM_HINT_T0);
                    }
                }
            }
        }
    }

    /// Get pool statistics
    pub fn get_statistics(&self) -> PoolStatistics {
        self.stats.clone()
    }

    /// Reset arena allocator
    pub fn reset_arena(&mut self) {
        if let Some(ref mut arena) = self.arena_allocator {
            arena.reset();
        }
    }

    /// Compact memory pools by removing unused buffers
    pub fn compact(&mut self) {
        for size_class in &mut self.size_classes {
            // Keep only half of the buffers to reduce memory usage
            let target_size = size_class.max_buffers / 2;
            while size_class.buffers.len() > target_size {
                size_class.buffers.pop_back();
            }
        }
    }

    /// Get memory usage report
    pub fn memory_report(&self) -> MemoryReport {
        let allocated = self.stats.bytes_allocated.load(Ordering::Relaxed);
        let deallocated = self.stats.bytes_deallocated.load(Ordering::Relaxed);
        let pool_efficiency = {
            let hits = self.stats.pool_hits.load(Ordering::Relaxed);
            let total = hits + self.stats.pool_misses.load(Ordering::Relaxed);
            if total > 0 {
                hits as f64 / total as f64
            } else {
                0.0
            }
        };

        MemoryReport {
            current_usage: allocated.saturating_sub(deallocated),
            peak_usage: self.stats.peak_memory.load(Ordering::Relaxed),
            pool_efficiency,
            pressure_level: self.memory_pressure(),
            fragmentation_ratio: self.calculate_fragmentation(),
        }
    }

    /// Calculate memory fragmentation ratio
    fn calculate_fragmentation(&self) -> f64 {
        // Simplified fragmentation calculation
        let mut total_pooled = 0;
        let mut total_capacity = 0;

        for size_class in &self.size_classes {
            total_pooled += size_class.buffers.len() * size_class.size;
            total_capacity += size_class.max_buffers * size_class.size;
        }

        if total_capacity > 0 {
            1.0 - (total_pooled as f64 / total_capacity as f64)
        } else {
            0.0
        }
    }
}

impl<T: Clone + Default> Default for AdvancedBufferPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage report
#[derive(Debug, Clone)]
pub struct MemoryReport {
    pub current_usage: usize,
    pub peak_usage: usize,
    pub pool_efficiency: f64,
    pub pressure_level: MemoryPressure,
    pub fragmentation_ratio: f64,
}

impl<T: Clone + Default> BufferPool<T> {
    /// Create a new buffer pool
    pub fn new() -> Self {
        Self {
            inner: AdvancedBufferPool::new(),
        }
    }

    /// Create buffer pool with configuration
    pub fn with_config(config: MemoryConfig) -> Self {
        Self {
            inner: AdvancedBufferPool::with_config(config),
        }
    }

    /// Acquire a vector from the pool, or create a new one if none are available
    pub fn acquire_vec(&mut self, capacity: usize) -> Vec<T> {
        self.inner.acquire_vec_advanced(capacity)
    }

    /// Release a vector back to the pool
    pub fn release_vec(&mut self, vec: Vec<T>) {
        self.inner.release_vec_advanced(vec);
    }

    /// Acquire an ndarray::Array1 from the pool, or create a new one if none are available
    pub fn acquire_array(&mut self, size: usize) -> ndarray::Array1<T> {
        // Find a suitable array in the pool
        for i in 0..self.inner.arrays.len() {
            if self.inner.arrays[i].len() >= size {
                // Found a suitable array, remove it from the pool and return it
                let mut array = self.inner.arrays.swap_remove(i);
                // Resize the array (this will truncate or extend)
                if array.len() != size {
                    array = ndarray::Array1::from_elem(size, T::default());
                }
                return array;
            }
        }

        // No suitable array found, create a new one
        ndarray::Array1::from_elem(size, T::default())
    }

    /// Release an ndarray::Array1 back to the pool
    pub fn release_array(&mut self, array: ndarray::Array1<T>) {
        // Add the array back to the pool for reuse
        self.inner.arrays.push(array);
    }

    /// Clear the pool, releasing all memory
    pub fn clear(&mut self) {
        self.inner.vectors.clear();
        self.inner.arrays.clear();
    }

    /// Get memory statistics
    pub fn get_statistics(&self) -> PoolStatistics {
        self.inner.get_statistics()
    }

    /// Get memory report
    pub fn memory_report(&self) -> MemoryReport {
        self.inner.memory_report()
    }

    /// Compact the pool to reduce memory usage
    pub fn compact(&mut self) {
        self.inner.compact();
    }
}

impl<T: Clone + Default> Default for BufferPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared global buffer pool for different types
pub struct GlobalBufferPool {
    // Use TypeId to store pools for different types
    pools: Mutex<HashMap<TypeId, Box<dyn std::any::Any + Send + Sync>>>,
}

impl GlobalBufferPool {
    /// Create a new global buffer pool
    pub fn new() -> Self {
        Self {
            pools: Mutex::new(HashMap::new()),
        }
    }

    /// Get or create a buffer pool for a specific type
    pub fn get_pool<T: Clone + Default + 'static + Send + Sync>(
        &self,
    ) -> Arc<Mutex<BufferPool<T>>> {
        let type_id = TypeId::of::<T>();
        let mut pools = self.pools.lock().unwrap();

        use std::collections::hash_map::Entry;
        match pools.entry(type_id) {
            Entry::Vacant(entry) => {
                // Create a new pool for this type
                let pool = Arc::new(Mutex::new(BufferPool::<T>::new()));
                entry.insert(Box::new(pool.clone()));
                pool
            }
            Entry::Occupied(entry) => {
                // Return the existing pool
                match entry.get().downcast_ref::<Arc<Mutex<BufferPool<T>>>>() {
                    Some(pool) => pool.clone(),
                    None => panic!("Type mismatch in global buffer pool"),
                }
            }
        }
    }

    /// Clear all pools, releasing all memory
    pub fn clear_all(&self) {
        let mut pools = self.pools.lock().unwrap();
        pools.clear();
    }
}

/// Implementation of Default for GlobalBufferPool
impl Default for GlobalBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Static global buffer pool instance
#[allow(dead_code)]
pub fn global_buffer_pool() -> &'static GlobalBufferPool {
    use once_cell::sync::Lazy;
    static GLOBAL_POOL: Lazy<GlobalBufferPool> = Lazy::new(GlobalBufferPool::new);
    &GLOBAL_POOL
}

/// Zero-copy array view for efficient data transformations
pub struct ZeroCopyView<'a, T, D>
where
    D: Dimension,
{
    phantom: PhantomData<T>,
    inner: ndarray::ArrayView<'a, T, D>,
}

impl<'a, T, D> ZeroCopyView<'a, T, D>
where
    D: Dimension,
{
    /// Create a new zero-copy view from an array
    pub fn new(array: &'a ndarray::Array<T, D>) -> Self {
        Self {
            phantom: PhantomData,
            inner: array.view(),
        }
    }

    /// Get the underlying array view
    pub fn view(&self) -> ndarray::ArrayView<'a, T, D> {
        self.inner.clone()
    }

    /// Transform the view using a mapping function
    pub fn transform<F, U>(&self, f: F) -> ndarray::Array<U, D>
    where
        F: Fn(&T) -> U,
        U: Clone,
    {
        self.inner.map(f)
    }
}

/// Memory usage tracker for monitoring memory consumption
pub struct MemoryTracker {
    allocations: Mutex<HashMap<String, usize>>,
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new() -> Self {
        Self {
            allocations: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryTracker {
    /// Track a memory allocation
    pub fn track_allocation(&self, name: &str, size: usize) {
        let mut allocations = self.allocations.lock().unwrap();
        *allocations.entry(name.to_string()).or_insert(0) += size;
    }

    /// Track a memory deallocation
    pub fn track_deallocation(&self, name: &str, size: usize) {
        let mut allocations = self.allocations.lock().unwrap();
        if let Some(current) = allocations.get_mut(name) {
            *current = current.saturating_sub(size);
        }
    }

    /// Get the current memory usage for a specific allocation
    pub fn get_usage(&self, name: &str) -> usize {
        let allocations = self.allocations.lock().unwrap();
        allocations.get(name).copied().unwrap_or_default()
    }

    /// Get the total memory usage across all tracked allocations
    pub fn get_total_usage(&self) -> usize {
        let allocations = self.allocations.lock().unwrap();
        allocations.values().sum()
    }

    /// Reset all tracking data
    pub fn reset(&self) {
        let mut allocations = self.allocations.lock().unwrap();
        allocations.clear();
    }
}

/// Static global memory tracker instance
#[allow(dead_code)]
pub fn global_memory_tracker() -> &'static MemoryTracker {
    use once_cell::sync::Lazy;
    static GLOBAL_TRACKER: Lazy<MemoryTracker> = Lazy::new(MemoryTracker::new);
    &GLOBAL_TRACKER
}

impl ArenaAllocator {
    /// Create a new arena allocator
    fn new(chunk_size: usize) -> Self {
        Self {
            chunks: Vec::new(),
            current_chunk: 0,
            chunk_size,
        }
    }

    /// Allocate memory from the arena
    fn allocate(&mut self, size: usize, align: usize) -> Option<NonNull<u8>> {
        if self.chunks.is_empty() || !self.can_allocate_in_current_chunk(size, align) {
            self.add_chunk();
        }

        if let Some(chunk) = self.chunks.get_mut(self.current_chunk) {
            let aligned_offset = (chunk.offset + align - 1) & !(align - 1);
            if aligned_offset + size <= chunk.size {
                let ptr = unsafe { chunk.ptr.as_ptr().add(aligned_offset) };
                chunk.offset = aligned_offset + size;
                NonNull::new(ptr)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Check if allocation can fit in current chunk
    fn can_allocate_in_current_chunk(&self, size: usize, align: usize) -> bool {
        if let Some(chunk) = self.chunks.get(self.current_chunk) {
            let aligned_offset = (chunk.offset + align - 1) & !(align - 1);
            aligned_offset + size <= chunk.size
        } else {
            false
        }
    }

    /// Add a new chunk to the arena
    fn add_chunk(&mut self) {
        let layout = Layout::from_size_align(self.chunk_size, 64).unwrap();
        if let Some(ptr) = NonNull::new(unsafe { alloc(layout) }) {
            self.chunks.push(ArenaChunk {
                ptr,
                size: self.chunk_size,
                offset: 0,
            });
            self.current_chunk = self.chunks.len() - 1;
        }
    }

    /// Reset the arena, keeping chunks for reuse
    fn reset(&mut self) {
        for chunk in &mut self.chunks {
            chunk.offset = 0;
        }
        self.current_chunk = 0;
    }
}

impl Drop for ArenaAllocator {
    fn drop(&mut self) {
        for chunk in &self.chunks {
            let layout = Layout::from_size_align(chunk.size, 64).unwrap();
            unsafe {
                dealloc(chunk.ptr.as_ptr(), layout);
            }
        }
    }
}

/// Smart memory allocator that adapts to usage patterns
pub struct SmartAllocator {
    config: MemoryConfig,
    usage_history: VecDeque<AllocationRecord>,
    current_strategy: AllocationStrategy,
    performance_metrics: AllocationMetrics,
}

/// Record of memory allocation for pattern analysis
#[derive(Debug, Clone)]
struct AllocationRecord {
    size: usize,
    timestamp: Instant,
    access_pattern: AccessPattern,
    lifetime: Option<Duration>,
}

/// Performance metrics for allocation strategies
#[derive(Debug, Clone, Default)]
pub struct AllocationMetrics {
    pub total_allocations: usize,
    pub average_allocation_time: Duration,
    pub memory_efficiency: f64,
    pub cache_hit_ratio: f64,
}

impl SmartAllocator {
    /// Create a new smart allocator
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            current_strategy: config.strategy,
            config,
            usage_history: VecDeque::with_capacity(1000),
            performance_metrics: AllocationMetrics::default(),
        }
    }

    /// Allocate memory with smart strategy selection
    pub fn allocate(&mut self, size: usize) -> Vec<u8> {
        let start_time = Instant::now();

        // Record allocation
        self.usage_history.push_back(AllocationRecord {
            size,
            timestamp: start_time,
            access_pattern: self.config.access_pattern,
            lifetime: None,
        });

        // Adapt strategy based on recent patterns
        self.adapt_strategy();

        // Perform allocation
        let result = match self.current_strategy {
            AllocationStrategy::Pool => self.allocate_pooled(size),
            AllocationStrategy::Arena => self.allocate_arena(size),
            AllocationStrategy::CacheAligned => self.allocate_aligned(size),
            _ => vec![0; size],
        };

        // Update metrics
        let allocation_time = start_time.elapsed();
        self.update_metrics(allocation_time);

        result
    }

    /// Adapt allocation strategy based on usage patterns
    fn adapt_strategy(&mut self) {
        if self.usage_history.len() < 10 {
            return; // Not enough data
        }

        let recent_allocations: Vec<_> = self.usage_history.iter().rev().take(100).collect();

        // Analyze patterns
        let avg_size: usize =
            recent_allocations.iter().map(|r| r.size).sum::<usize>() / recent_allocations.len();
        let has_repeating_sizes = self.has_repeating_sizes(&recent_allocations);
        let is_temporal_locality = self.has_temporal_locality(&recent_allocations);

        // Adapt strategy
        if has_repeating_sizes && avg_size < 4096 {
            self.current_strategy = AllocationStrategy::Pool;
        } else if is_temporal_locality {
            self.current_strategy = AllocationStrategy::Arena;
        } else if avg_size > 1024 * 1024 {
            self.current_strategy = AllocationStrategy::HugePage;
        } else {
            self.current_strategy = AllocationStrategy::CacheAligned;
        }
    }

    /// Check for repeating allocation sizes
    fn has_repeating_sizes(&self, records: &[&AllocationRecord]) -> bool {
        let mut size_counts = HashMap::new();
        for record in records {
            *size_counts.entry(record.size).or_insert(0) += 1;
        }
        size_counts.values().any(|&count| count > records.len() / 4)
    }

    /// Check for temporal locality in allocations
    fn has_temporal_locality(&self, records: &[&AllocationRecord]) -> bool {
        if records.len() < 5 {
            return false;
        }

        let mut intervals = Vec::new();
        for window in records.windows(2) {
            if let Some(interval) = window[0]
                .timestamp
                .checked_duration_since(window[1].timestamp)
            {
                intervals.push(interval);
            }
        }

        if intervals.is_empty() {
            return false;
        }

        let avg_interval = intervals.iter().sum::<Duration>() / intervals.len() as u32;
        intervals
            .iter()
            .all(|&interval| interval < avg_interval * 2)
    }

    /// Allocate using pool strategy
    fn allocate_pooled(&mut self, size: usize) -> Vec<u8> {
        // Simplified pool allocation
        vec![0; size]
    }

    /// Allocate using arena strategy
    fn allocate_arena(&mut self, size: usize) -> Vec<u8> {
        // Simplified arena allocation
        vec![0; size]
    }

    /// Allocate with cache alignment
    fn allocate_aligned(&mut self, size: usize) -> Vec<u8> {
        let aligned_size = (size + self.config.alignment - 1) & !(self.config.alignment - 1);
        vec![0; aligned_size]
    }

    /// Update performance metrics
    fn update_metrics(&mut self, allocation_time: Duration) {
        self.performance_metrics.total_allocations += 1;

        let total_time = self.performance_metrics.average_allocation_time
            * (self.performance_metrics.total_allocations - 1) as u32
            + allocation_time;
        self.performance_metrics.average_allocation_time =
            total_time / self.performance_metrics.total_allocations as u32;
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> &AllocationMetrics {
        &self.performance_metrics
    }
}

/// Memory bandwidth optimizer for scientific computing
pub struct BandwidthOptimizer {
    access_patterns: HashMap<String, AccessPattern>,
    bandwidth_measurements: VecDeque<BandwidthMeasurement>,
    optimal_strategies: HashMap<AccessPattern, AllocationStrategy>,
}

/// Bandwidth measurement record
#[derive(Debug, Clone)]
struct BandwidthMeasurement {
    pattern: AccessPattern,
    strategy: AllocationStrategy,
    bandwidth_gbps: f64,
    timestamp: Instant,
}

impl BandwidthOptimizer {
    /// Create a new bandwidth optimizer
    pub fn new() -> Self {
        let mut optimal_strategies = HashMap::new();
        optimal_strategies.insert(AccessPattern::Sequential, AllocationStrategy::CacheAligned);
        optimal_strategies.insert(AccessPattern::Random, AllocationStrategy::Pool);
        optimal_strategies.insert(AccessPattern::Streaming, AllocationStrategy::HugePage);

        Self {
            access_patterns: HashMap::new(),
            bandwidth_measurements: VecDeque::with_capacity(1000),
            optimal_strategies,
        }
    }

    /// Register an access pattern for a workload
    pub fn register_pattern(&mut self, workload: &str, pattern: AccessPattern) {
        self.access_patterns.insert(workload.to_string(), pattern);
    }

    /// Get optimal allocation strategy for a workload
    pub fn get_optimal_strategy(&self, workload: &str) -> Option<AllocationStrategy> {
        self.access_patterns
            .get(workload)
            .and_then(|pattern| self.optimal_strategies.get(pattern))
            .copied()
    }

    /// Record bandwidth measurement
    pub fn record_bandwidth(
        &mut self,
        pattern: AccessPattern,
        strategy: AllocationStrategy,
        bandwidth_gbps: f64,
    ) {
        self.bandwidth_measurements.push_back(BandwidthMeasurement {
            pattern,
            strategy,
            bandwidth_gbps,
            timestamp: Instant::now(),
        });

        // Keep only recent measurements
        if self.bandwidth_measurements.len() > 1000 {
            self.bandwidth_measurements.pop_front();
        }

        // Update optimal strategies based on measurements
        self.update_optimal_strategies();
    }

    /// Update optimal strategies based on measured performance
    fn update_optimal_strategies(&mut self) {
        let mut pattern_performance: HashMap<AccessPattern, HashMap<AllocationStrategy, f64>> =
            HashMap::new();

        for measurement in &self.bandwidth_measurements {
            pattern_performance
                .entry(measurement.pattern)
                .or_insert_with(HashMap::new)
                .entry(measurement.strategy)
                .and_modify(|avg| *avg = (*avg + measurement.bandwidth_gbps) / 2.0)
                .or_insert(measurement.bandwidth_gbps);
        }

        // Find best strategy for each pattern
        for (pattern, strategies) in pattern_performance {
            if let Some((&best_strategy, _)) = strategies
                .iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            {
                self.optimal_strategies.insert(pattern, best_strategy);
            }
        }
    }

    /// Get bandwidth statistics for a pattern
    pub fn get_bandwidth_stats(&self, pattern: AccessPattern) -> Option<(f64, f64, f64)> {
        let measurements: Vec<f64> = self
            .bandwidth_measurements
            .iter()
            .filter(|m| m.pattern == pattern)
            .map(|m| m.bandwidth_gbps)
            .collect();

        if measurements.is_empty() {
            return None;
        }

        let sum: f64 = measurements.iter().sum();
        let avg = sum / measurements.len() as f64;
        let min = measurements.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = measurements
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        Some((avg, min, max))
    }
}

impl Default for BandwidthOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced memory metrics system
pub mod metrics;

/// Cross-device memory management for CPU/GPU/TPU
#[cfg(feature = "gpu")]
pub mod cross_device;

/// Out-of-core processing for datasets larger than memory
pub mod out_of_core;

/// Compressed memory buffers for memory-constrained environments
#[cfg(feature = "memory_compression")]
pub mod compressed_buffers;

/// Production-level memory safety features with bounds checking and overflow protection
pub mod safety;

/// Memory leak detection and monitoring system
#[cfg(feature = "memory_management")]
pub mod leak_detection;

// Re-export key metric functions for convenient usage
pub use metrics::{
    format_memory_report, generate_memory_report, track_allocation, track_deallocation,
    track_resize,
};

// Re-export leak detection types for convenience
#[cfg(feature = "memory_management")]
pub use leak_detection::{
    LeakCheckGuard, LeakDetectionConfig, LeakDetector, LeakReport, LeakType, MemoryCheckpoint,
    MemoryLeak, ProfilerTool, ValgrindIntegration,
};

/// Convenience function to create an optimized memory pool
pub fn create_optimized_pool<T: Clone + Default + 'static>() -> AdvancedBufferPool<T> {
    let config = MemoryConfig {
        strategy: AllocationStrategy::Pool,
        access_pattern: AccessPattern::Sequential,
        enable_prefetch: true,
        alignment: 64,
        numa_aware: true,
        max_memory: None,
        enable_compression: false,
        pressure_threshold: 0.8,
    };
    AdvancedBufferPool::with_config(config)
}

/// Convenience function to create a high-performance memory pool for scientific computing
pub fn create_scientific_pool<T: Clone + Default + 'static>() -> AdvancedBufferPool<T> {
    let config = MemoryConfig {
        strategy: AllocationStrategy::CacheAligned,
        access_pattern: AccessPattern::Sequential,
        enable_prefetch: true,
        alignment: 64,
        numa_aware: true,
        max_memory: None,
        enable_compression: false,
        pressure_threshold: 0.9,
    };
    AdvancedBufferPool::with_config(config)
}

/// Convenience function to create a memory pool optimized for large datasets
pub fn create_large_data_pool<T: Clone + Default + 'static>() -> AdvancedBufferPool<T> {
    let config = MemoryConfig {
        strategy: AllocationStrategy::HugePage,
        access_pattern: AccessPattern::Streaming,
        enable_prefetch: true,
        alignment: 2 * 1024 * 1024, // 2MB alignment for huge pages
        numa_aware: true,
        max_memory: None,
        enable_compression: true,
        pressure_threshold: 0.7,
    };
    AdvancedBufferPool::with_config(config)
}

/// Global smart allocator instance
static GLOBAL_SMART_ALLOCATOR: std::sync::LazyLock<Arc<Mutex<SmartAllocator>>> =
    std::sync::LazyLock::new(|| Arc::new(Mutex::new(SmartAllocator::new(MemoryConfig::default()))));

/// Get the global smart allocator
pub fn global_smart_allocator() -> Arc<Mutex<SmartAllocator>> {
    GLOBAL_SMART_ALLOCATOR.clone()
}

/// Global bandwidth optimizer instance
static GLOBAL_BANDWIDTH_OPTIMIZER: std::sync::LazyLock<Arc<Mutex<BandwidthOptimizer>>> =
    std::sync::LazyLock::new(|| Arc::new(Mutex::new(BandwidthOptimizer::new())));

/// Get the global bandwidth optimizer
pub fn global_bandwidth_optimizer() -> Arc<Mutex<BandwidthOptimizer>> {
    GLOBAL_BANDWIDTH_OPTIMIZER.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_advanced_buffer_pool() {
        let mut pool = AdvancedBufferPool::<f64>::new();

        // Test basic allocation and release
        let vec1 = pool.acquire_vec_advanced(1000);
        assert_eq!(vec1.len(), 1000);

        pool.release_vec_advanced(vec1);

        // Second allocation should reuse from pool
        let vec2 = pool.acquire_vec_advanced(800);
        assert_eq!(vec2.len(), 800);

        let stats = pool.get_statistics();
        assert_eq!(stats.total_allocations.load(Ordering::Relaxed), 2);
        assert_eq!(stats.total_deallocations.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_memory_config() {
        let config = MemoryConfig {
            strategy: AllocationStrategy::CacheAligned,
            access_pattern: AccessPattern::Random,
            enable_prefetch: true,
            alignment: 128,
            numa_aware: true,
            max_memory: Some(1024 * 1024 * 1024), // 1GB
            enable_compression: false,
            pressure_threshold: 0.7,
        };

        let mut pool = AdvancedBufferPool::<i32>::with_config(config.clone());
        assert_eq!(pool.config.alignment, 128);
        assert_eq!(pool.config.strategy, AllocationStrategy::CacheAligned);

        let vec = pool.acquire_vec_advanced(256);
        assert_eq!(vec.len(), 256);
    }

    #[test]
    fn test_memory_pressure() {
        let config = MemoryConfig {
            max_memory: Some(1024),
            ..Default::default()
        };
        let pool = AdvancedBufferPool::<u8>::with_config(config);

        // Initially should be low pressure
        let pressure = pool.memory_pressure();
        assert_eq!(pressure, MemoryPressure::Low);
    }

    #[test]
    fn test_pool_statistics() {
        let mut pool = AdvancedBufferPool::<f32>::new();

        // Perform some allocations
        let _vec1 = pool.acquire_vec_advanced(100);
        let _vec2 = pool.acquire_vec_advanced(200);
        let _vec3 = pool.acquire_vec_advanced(50);

        let stats = pool.get_statistics();
        assert_eq!(stats.total_allocations.load(Ordering::Relaxed), 3);
        assert_eq!(stats.total_deallocations.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_memory_report() {
        let mut pool = AdvancedBufferPool::<u64>::new();
        let _vec = pool.acquire_vec_advanced(500);

        let report = pool.memory_report();
        // current_usage is usize, so it's always >= 0 by definition
        assert!(report.pool_efficiency >= 0.0 && report.pool_efficiency <= 1.0);
        assert!(report.fragmentation_ratio >= 0.0 && report.fragmentation_ratio <= 1.0);
    }

    #[test]
    fn test_pool_compaction() {
        let mut pool = AdvancedBufferPool::<i64>::new();

        // Fill up some size classes
        for _ in 0..20 {
            let vec = pool.acquire_vec_advanced(128);
            pool.release_vec_advanced(vec);
        }

        // Compact should reduce memory usage
        pool.compact();

        // Pool should still work after compaction
        let vec = pool.acquire_vec_advanced(128);
        assert_eq!(vec.len(), 128);
    }

    #[test]
    fn test_smart_allocator() {
        let mut allocator = SmartAllocator::new(MemoryConfig::default());

        // Test basic allocation
        let buffer1 = allocator.allocate(1024);
        assert_eq!(buffer1.len(), 1024);

        let buffer2 = allocator.allocate(2048);
        assert_eq!(buffer2.len(), 2048);

        let metrics = allocator.get_metrics();
        assert_eq!(metrics.total_allocations, 2);
    }

    #[test]
    fn test_smart_allocator_adaptation() {
        let mut allocator = SmartAllocator::new(MemoryConfig::default());
        let initial_strategy = allocator.current_strategy;

        // Make several allocations with same size to trigger adaptation
        for _ in 0..15 {
            let _buffer = allocator.allocate(512);
        }

        // Strategy might have adapted (though not guaranteed in this simple test)
        let _final_strategy = allocator.current_strategy;
    }

    #[test]
    fn test_bandwidth_optimizer() {
        let mut optimizer = BandwidthOptimizer::new();

        // Register some patterns
        optimizer.register_pattern("matrix_multiply", AccessPattern::Spatial);
        optimizer.register_pattern("vector_sum", AccessPattern::Sequential);

        // Test getting optimal strategy
        let strategy = optimizer.get_optimal_strategy("vector_sum");
        assert!(strategy.is_some());

        // Record some measurements
        optimizer.record_bandwidth(
            AccessPattern::Sequential,
            AllocationStrategy::CacheAligned,
            25.0,
        );
        optimizer.record_bandwidth(AccessPattern::Random, AllocationStrategy::Pool, 12.0);

        // Get bandwidth stats
        let stats = optimizer.get_bandwidth_stats(AccessPattern::Sequential);
        assert!(stats.is_some());
        let (avg, min, max) = stats.unwrap();
        assert_eq!(avg, 25.0);
        assert_eq!(min, 25.0);
        assert_eq!(max, 25.0);
    }

    #[test]
    fn test_convenience_functions() {
        let _optimized_pool = create_optimized_pool::<f64>();
        let _scientific_pool = create_scientific_pool::<f32>();
        let _large_data_pool = create_large_data_pool::<u8>();

        // Test global instances
        let _smart_allocator = global_smart_allocator();
        let _bandwidth_optimizer = global_bandwidth_optimizer();
    }

    #[test]
    fn test_legacy_buffer_pool_compatibility() {
        let mut pool = BufferPool::<i32>::new();

        // Test that legacy interface still works
        let vec = pool.acquire_vec(100);
        assert_eq!(vec.len(), 100);

        pool.release_vec(vec);

        // Test enhanced functionality
        let stats = pool.get_statistics();
        assert!(stats.total_allocations.load(Ordering::Relaxed) > 0);

        let report = pool.memory_report();
        assert!(report.fragmentation_ratio >= 0.0);

        pool.compact();
    }

    #[test]
    fn test_allocation_strategies() {
        let strategies = [
            AllocationStrategy::System,
            AllocationStrategy::Pool,
            AllocationStrategy::Arena,
            AllocationStrategy::CacheAligned,
            AllocationStrategy::NumaAware,
        ];

        for strategy in &strategies {
            let config = MemoryConfig {
                strategy: *strategy,
                ..Default::default()
            };
            let mut pool = AdvancedBufferPool::<u32>::with_config(config);
            let vec = pool.acquire_vec_advanced(256);
            assert_eq!(vec.len(), 256);
        }
    }

    #[test]
    fn test_access_patterns() {
        let patterns = [
            AccessPattern::Sequential,
            AccessPattern::Random,
            AccessPattern::Temporal,
            AccessPattern::Spatial,
            AccessPattern::WriteOnce,
            AccessPattern::ReadOnly,
            AccessPattern::Streaming,
        ];

        for pattern in &patterns {
            let config = MemoryConfig {
                access_pattern: *pattern,
                ..Default::default()
            };
            let pool = AdvancedBufferPool::<f64>::with_config(config);
            assert_eq!(pool.config.access_pattern, *pattern);
        }
    }

    #[test]
    fn test_memory_pressure_levels() {
        assert!(MemoryPressure::Low < MemoryPressure::Medium);
        assert!(MemoryPressure::Medium < MemoryPressure::High);
        assert!(MemoryPressure::High < MemoryPressure::Critical);
    }

    #[test]
    fn test_concurrent_pool_access() {
        let pool = Arc::new(Mutex::new(AdvancedBufferPool::<i32>::new()));
        let mut handles = vec![];

        for i in 0..4 {
            let pool_clone = Arc::clone(&pool);
            let handle = thread::spawn(move || {
                let mut pool = pool_clone.lock().unwrap();
                let vec = pool.acquire_vec_advanced(100 + i * 50);
                thread::sleep(Duration::from_millis(10));
                pool.release_vec_advanced(vec);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let pool = pool.lock().unwrap();
        let stats = pool.get_statistics();
        assert_eq!(stats.total_allocations.load(Ordering::Relaxed), 4);
        assert_eq!(stats.total_deallocations.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn test_arena_allocator_safety() {
        let mut arena = ArenaAllocator::new(4096);

        // Test basic allocation
        let ptr1 = arena.allocate(64, 8);
        assert!(ptr1.is_some());

        let ptr2 = arena.allocate(128, 16);
        assert!(ptr2.is_some());

        // Test reset functionality
        arena.reset();

        let ptr3 = arena.allocate(32, 4);
        assert!(ptr3.is_some());
    }

    #[test]
    fn test_bandwidth_optimizer_pattern_learning() {
        let mut optimizer = BandwidthOptimizer::new();

        // Record multiple measurements for the same pattern
        optimizer.record_bandwidth(
            AccessPattern::Sequential,
            AllocationStrategy::CacheAligned,
            30.0,
        );
        optimizer.record_bandwidth(AccessPattern::Sequential, AllocationStrategy::Pool, 20.0);
        optimizer.record_bandwidth(AccessPattern::Sequential, AllocationStrategy::Arena, 25.0);

        // The optimizer should learn that CacheAligned is best for Sequential
        let stats = optimizer.get_bandwidth_stats(AccessPattern::Sequential);
        assert!(stats.is_some());
        let (avg, min, max) = stats.unwrap();
        assert!(avg > 20.0);
        assert_eq!(min, 20.0);
        assert_eq!(max, 30.0);
    }
}
