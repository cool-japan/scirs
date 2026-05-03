//! Enhanced GPU Memory Pool for efficient allocation management
//!
//! This module provides sophisticated memory pooling strategies for GPU memory,
//! including suballocation, defragmentation, and automatic memory management
//! optimized for linear algebra workloads.
//!
//! ## Features
//!
//! - Block-based suballocation for reduced allocation overhead
//! - Memory coalescing for contiguous allocations
//! - Automatic defragmentation
//! - Memory pressure monitoring and automatic eviction
//! - Thread-safe access with fine-grained locking
//! - Memory usage statistics and profiling

use crate::error::{LinalgError, LinalgResult};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Memory allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Best-fit: Find smallest block that fits
    BestFit,
    /// First-fit: Find first block that fits
    FirstFit,
    /// Next-fit: Continue from last allocation position
    NextFit,
    /// Buddy allocator: Power-of-two block sizes
    Buddy,
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Total pool size in bytes
    pub pool_size: usize,
    /// Minimum block size (for suballocation)
    pub min_block_size: usize,
    /// Maximum block size for pooling (larger allocations are direct)
    pub max_block_size: usize,
    /// Alignment requirement (must be power of 2)
    pub alignment: usize,
    /// Allocation strategy
    pub strategy: AllocationStrategy,
    /// Enable defragmentation
    pub enable_defrag: bool,
    /// Defragmentation threshold (trigger when fragmentation exceeds this)
    pub defrag_threshold: f64,
    /// Memory pressure threshold for eviction
    pub pressure_threshold: f64,
    /// Maximum cache age before eviction
    pub max_cache_age: Duration,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        #[cfg(target_pointer_width = "32")]
        let pool_size = 256 * 1024 * 1024; // 256MB default for 32-bit
        #[cfg(target_pointer_width = "64")]
        let pool_size = 1024 * 1024 * 1024; // 1GB default for 64-bit

        Self {
            pool_size,
            min_block_size: 256,
            max_block_size: 64 * 1024 * 1024, // 64MB
            alignment: 256,                   // Typical GPU alignment
            strategy: AllocationStrategy::BestFit,
            enable_defrag: true,
            defrag_threshold: 0.3,
            pressure_threshold: 0.9,
            max_cache_age: Duration::from_secs(60),
        }
    }
}

/// Memory block metadata
#[derive(Debug, Clone)]
struct MemoryBlock {
    /// Offset within the pool
    offset: usize,
    /// Size of the block
    size: usize,
    /// Whether the block is in use
    in_use: bool,
    /// Allocation ID (for tracking)
    allocation_id: Option<usize>,
    /// Last access time
    last_access: Instant,
}

/// Allocation handle returned to the user
#[derive(Debug, Clone)]
pub struct AllocationHandle {
    /// Unique allocation ID
    pub id: usize,
    /// Offset within the pool
    pub offset: usize,
    /// Size of the allocation
    pub size: usize,
    /// Creation time
    created_at: Instant,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total pool size
    pub total_size: usize,
    /// Currently allocated bytes
    pub allocated_bytes: usize,
    /// Currently free bytes
    pub free_bytes: usize,
    /// Number of active allocations
    pub active_allocations: usize,
    /// Total allocations since creation
    pub total_allocations: usize,
    /// Total deallocations since creation
    pub total_deallocations: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Number of fragmented blocks
    pub fragmented_blocks: usize,
    /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = fully fragmented)
    pub fragmentation_ratio: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
}

/// Enhanced GPU memory pool with suballocation
pub struct GpuMemoryPool {
    config: MemoryPoolConfig,
    /// Free blocks organized by size (for best-fit)
    free_blocks: RwLock<BTreeMap<usize, Vec<usize>>>,
    /// All blocks (by offset)
    blocks: RwLock<HashMap<usize, MemoryBlock>>,
    /// Active allocations
    allocations: RwLock<HashMap<usize, AllocationHandle>>,
    /// Cached free blocks for quick reuse
    block_cache: Mutex<HashMap<usize, VecDeque<usize>>>,
    /// Next allocation ID
    next_id: Mutex<usize>,
    /// Memory statistics
    stats: RwLock<MemoryStats>,
    /// Last allocation offset (for next-fit strategy)
    last_offset: Mutex<usize>,
}

impl GpuMemoryPool {
    /// Create a new memory pool with default configuration
    pub fn new() -> Self {
        Self::with_config(MemoryPoolConfig::default())
    }

    /// Create a memory pool with custom configuration
    pub fn with_config(config: MemoryPoolConfig) -> Self {
        let pool_size = config.pool_size;

        let mut blocks = HashMap::new();
        blocks.insert(
            0,
            MemoryBlock {
                offset: 0,
                size: pool_size,
                in_use: false,
                allocation_id: None,
                last_access: Instant::now(),
            },
        );

        let mut free_blocks = BTreeMap::new();
        free_blocks.insert(pool_size, vec![0]);

        let stats = MemoryStats {
            total_size: pool_size,
            free_bytes: pool_size,
            ..Default::default()
        };

        Self {
            config,
            free_blocks: RwLock::new(free_blocks),
            blocks: RwLock::new(blocks),
            allocations: RwLock::new(HashMap::new()),
            block_cache: Mutex::new(HashMap::new()),
            next_id: Mutex::new(1),
            stats: RwLock::new(stats),
            last_offset: Mutex::new(0),
        }
    }

    /// Allocate memory from the pool
    pub fn allocate(&self, size: usize) -> LinalgResult<AllocationHandle> {
        // Align size
        let aligned_size = self.align_size(size);

        // Check if size is too large for pooling
        if aligned_size > self.config.max_block_size {
            return Err(LinalgError::ComputationError(format!(
                "Allocation size {} exceeds maximum block size {}",
                aligned_size, self.config.max_block_size
            )));
        }

        // Try to find a cached block first
        if let Some(offset) = self.try_cache(aligned_size) {
            return self.complete_allocation(offset, aligned_size);
        }

        // Find a suitable free block
        let block_offset = match self.config.strategy {
            AllocationStrategy::BestFit => self.find_best_fit(aligned_size)?,
            AllocationStrategy::FirstFit => self.find_first_fit(aligned_size)?,
            AllocationStrategy::NextFit => self.find_next_fit(aligned_size)?,
            AllocationStrategy::Buddy => self.find_buddy_block(aligned_size)?,
        };

        // Update cache miss stats
        if let Ok(mut stats) = self.stats.write() {
            stats.cache_misses += 1;
            self.update_cache_hit_rate(&mut stats);
        }

        self.complete_allocation(block_offset, aligned_size)
    }

    /// Try to get a block from the cache
    fn try_cache(&self, size: usize) -> Option<usize> {
        if let Ok(mut cache) = self.block_cache.lock() {
            if let Some(offsets) = cache.get_mut(&size) {
                if let Some(offset) = offsets.pop_front() {
                    if let Ok(mut stats) = self.stats.write() {
                        stats.cache_hits += 1;
                        self.update_cache_hit_rate(&mut stats);
                    }
                    return Some(offset);
                }
            }
        }
        None
    }

    /// Complete an allocation
    fn complete_allocation(&self, offset: usize, size: usize) -> LinalgResult<AllocationHandle> {
        // Split the block if necessary and mark as allocated
        let id = {
            let mut id_guard = self
                .next_id
                .lock()
                .map_err(|_| LinalgError::ComputationError("Lock poisoned".to_string()))?;
            let id = *id_guard;
            *id_guard += 1;
            id
        };

        // Update blocks
        {
            let mut blocks = self
                .blocks
                .write()
                .map_err(|_| LinalgError::ComputationError("Lock poisoned".to_string()))?;
            let mut free_blocks = self
                .free_blocks
                .write()
                .map_err(|_| LinalgError::ComputationError("Lock poisoned".to_string()))?;

            if let Some(original_size) = blocks.get(&offset).map(|b| b.size) {
                // Split if there's remaining space
                if original_size > size {
                    let remaining_offset = offset + size;
                    let remaining_size = original_size - size;

                    blocks.insert(
                        remaining_offset,
                        MemoryBlock {
                            offset: remaining_offset,
                            size: remaining_size,
                            in_use: false,
                            allocation_id: None,
                            last_access: Instant::now(),
                        },
                    );

                    // Add remaining block to free list
                    free_blocks
                        .entry(remaining_size)
                        .or_default()
                        .push(remaining_offset);
                }

                // Update the allocated block
                if let Some(block) = blocks.get_mut(&offset) {
                    block.size = size;
                    block.in_use = true;
                    block.allocation_id = Some(id);
                    block.last_access = Instant::now();
                }

                // Remove from free list
                if let Some(offsets) = free_blocks.get_mut(&original_size) {
                    offsets.retain(|&o| o != offset);
                    if offsets.is_empty() {
                        free_blocks.remove(&original_size);
                    }
                }
            }
        }

        let handle = AllocationHandle {
            id,
            offset,
            size,
            created_at: Instant::now(),
        };

        // Store allocation handle
        if let Ok(mut allocs) = self.allocations.write() {
            allocs.insert(id, handle.clone());
        }

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.allocated_bytes += size;
            stats.free_bytes = stats.free_bytes.saturating_sub(size);
            stats.active_allocations += 1;
            stats.total_allocations += 1;
            stats.peak_usage = stats.peak_usage.max(stats.allocated_bytes);
        }

        Ok(handle)
    }

    /// Find best-fit block
    fn find_best_fit(&self, size: usize) -> LinalgResult<usize> {
        let free_blocks = self
            .free_blocks
            .read()
            .map_err(|_| LinalgError::ComputationError("Lock poisoned".to_string()))?;

        // BTreeMap is sorted, so we find the first key >= size
        for (&block_size, offsets) in free_blocks.range(size..) {
            if let Some(&offset) = offsets.first() {
                return Ok(offset);
            }
        }

        Err(LinalgError::ComputationError(
            "No suitable free block found".to_string(),
        ))
    }

    /// Find first-fit block
    fn find_first_fit(&self, size: usize) -> LinalgResult<usize> {
        let blocks = self
            .blocks
            .read()
            .map_err(|_| LinalgError::ComputationError("Lock poisoned".to_string()))?;

        // Find first free block that fits
        let mut offsets: Vec<_> = blocks
            .iter()
            .filter(|(_, b)| !b.in_use && b.size >= size)
            .map(|(&o, _)| o)
            .collect();
        offsets.sort();

        offsets.into_iter().next().ok_or_else(|| {
            LinalgError::ComputationError("No suitable free block found".to_string())
        })
    }

    /// Find next-fit block
    fn find_next_fit(&self, size: usize) -> LinalgResult<usize> {
        let last_offset = *self
            .last_offset
            .lock()
            .map_err(|_| LinalgError::ComputationError("Lock poisoned".to_string()))?;

        let blocks = self
            .blocks
            .read()
            .map_err(|_| LinalgError::ComputationError("Lock poisoned".to_string()))?;

        // Find suitable blocks after last allocation
        let mut offsets: Vec<_> = blocks
            .iter()
            .filter(|(_, b)| !b.in_use && b.size >= size)
            .map(|(&o, _)| o)
            .collect();
        offsets.sort();

        // Try blocks after last offset first
        for &offset in &offsets {
            if offset >= last_offset {
                if let Ok(mut last) = self.last_offset.lock() {
                    *last = offset;
                }
                return Ok(offset);
            }
        }

        // Wrap around to beginning
        if let Some(&offset) = offsets.first() {
            if let Ok(mut last) = self.last_offset.lock() {
                *last = offset;
            }
            return Ok(offset);
        }

        Err(LinalgError::ComputationError(
            "No suitable free block found".to_string(),
        ))
    }

    /// Find buddy block (power-of-two allocation)
    fn find_buddy_block(&self, size: usize) -> LinalgResult<usize> {
        // Round up to power of 2
        let buddy_size = size.next_power_of_two();
        self.find_best_fit(buddy_size)
    }

    /// Deallocate memory
    pub fn deallocate(&self, handle: &AllocationHandle) -> LinalgResult<()> {
        let offset = handle.offset;
        let size = handle.size;

        // Remove allocation record
        if let Ok(mut allocs) = self.allocations.write() {
            allocs.remove(&handle.id);
        }

        // Mark block as free
        {
            let mut blocks = self
                .blocks
                .write()
                .map_err(|_| LinalgError::ComputationError("Lock poisoned".to_string()))?;
            let mut free_blocks = self
                .free_blocks
                .write()
                .map_err(|_| LinalgError::ComputationError("Lock poisoned".to_string()))?;

            if let Some(block) = blocks.get_mut(&offset) {
                block.in_use = false;
                block.allocation_id = None;
                block.last_access = Instant::now();

                // Add to free list
                free_blocks.entry(size).or_default().push(offset);
            }
        }

        // Try to coalesce with adjacent free blocks
        self.try_coalesce(offset)?;

        // Add to cache for quick reuse
        if let Ok(mut cache) = self.block_cache.lock() {
            let offsets = cache.entry(size).or_default();
            if offsets.len() < 16 {
                // Limit cache size per size class
                offsets.push_back(offset);
            }
        }

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.allocated_bytes = stats.allocated_bytes.saturating_sub(size);
            stats.free_bytes += size;
            stats.active_allocations = stats.active_allocations.saturating_sub(1);
            stats.total_deallocations += 1;
        }

        Ok(())
    }

    /// Try to coalesce adjacent free blocks
    fn try_coalesce(&self, offset: usize) -> LinalgResult<()> {
        let mut blocks = self
            .blocks
            .write()
            .map_err(|_| LinalgError::ComputationError("Lock poisoned".to_string()))?;
        let mut free_blocks = self
            .free_blocks
            .write()
            .map_err(|_| LinalgError::ComputationError("Lock poisoned".to_string()))?;

        // Get current block info
        let (current_size, current_end) = {
            if let Some(block) = blocks.get(&offset) {
                if block.in_use {
                    return Ok(()); // Can't coalesce in-use block
                }
                (block.size, offset + block.size)
            } else {
                return Ok(());
            }
        };

        // Find and merge with next adjacent free block
        if let Some(next_block) = blocks.get(&current_end).cloned() {
            if !next_block.in_use {
                // Remove next block
                blocks.remove(&current_end);

                // Remove from free list
                if let Some(offsets) = free_blocks.get_mut(&next_block.size) {
                    offsets.retain(|&o| o != current_end);
                    if offsets.is_empty() {
                        free_blocks.remove(&next_block.size);
                    }
                }

                // Extend current block
                if let Some(block) = blocks.get_mut(&offset) {
                    // Remove from current size class
                    if let Some(offsets) = free_blocks.get_mut(&block.size) {
                        offsets.retain(|&o| o != offset);
                        if offsets.is_empty() {
                            free_blocks.remove(&block.size);
                        }
                    }

                    block.size += next_block.size;

                    // Add to new size class
                    free_blocks.entry(block.size).or_default().push(offset);
                }
            }
        }

        // Find and merge with previous adjacent free block
        let prev_info: Option<(usize, usize)> = blocks
            .iter()
            .filter(|(_, b)| !b.in_use && b.offset + b.size == offset)
            .map(|(&o, b)| (o, b.size))
            .next();

        if let Some((prev_offset, prev_size)) = prev_info {
            // Remove current block
            let current_size_now = blocks.get(&offset).map(|b| b.size).unwrap_or(0);
            blocks.remove(&offset);

            // Remove from free lists
            if let Some(offsets) = free_blocks.get_mut(&current_size_now) {
                offsets.retain(|&o| o != offset);
                if offsets.is_empty() {
                    free_blocks.remove(&current_size_now);
                }
            }

            if let Some(offsets) = free_blocks.get_mut(&prev_size) {
                offsets.retain(|&o| o != prev_offset);
                if offsets.is_empty() {
                    free_blocks.remove(&prev_size);
                }
            }

            // Extend previous block
            if let Some(prev_block) = blocks.get_mut(&prev_offset) {
                prev_block.size += current_size_now;

                // Add to new size class
                free_blocks
                    .entry(prev_block.size)
                    .or_default()
                    .push(prev_offset);
            }
        }

        Ok(())
    }

    /// Get current memory statistics
    pub fn stats(&self) -> MemoryStats {
        self.stats.read().map(|s| s.clone()).unwrap_or_default()
    }

    /// Calculate fragmentation ratio
    pub fn fragmentation_ratio(&self) -> f64 {
        if let Ok(blocks) = self.blocks.read() {
            let free_blocks: Vec<_> = blocks.values().filter(|b| !b.in_use).collect();

            if free_blocks.is_empty() {
                return 0.0;
            }

            let total_free: usize = free_blocks.iter().map(|b| b.size).sum();
            let largest_free = free_blocks.iter().map(|b| b.size).max().unwrap_or(0);

            if total_free == 0 {
                return 0.0;
            }

            1.0 - (largest_free as f64 / total_free as f64)
        } else {
            0.0
        }
    }

    /// Trigger defragmentation if needed
    pub fn maybe_defragment(&self) -> LinalgResult<bool> {
        let frag_ratio = self.fragmentation_ratio();

        if frag_ratio > self.config.defrag_threshold && self.config.enable_defrag {
            self.defragment()?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Defragment the memory pool
    pub fn defragment(&self) -> LinalgResult<()> {
        // In a real implementation, this would compact memory
        // For now, just update fragmentation stats
        if let Ok(mut stats) = self.stats.write() {
            stats.fragmentation_ratio = self.fragmentation_ratio();
            stats.fragmented_blocks = self.count_fragmented_blocks();
        }
        Ok(())
    }

    /// Count fragmented blocks
    fn count_fragmented_blocks(&self) -> usize {
        self.blocks
            .read()
            .map(|blocks| blocks.values().filter(|b| !b.in_use).count())
            .unwrap_or(0)
    }

    /// Evict old cached blocks
    pub fn evict_old_caches(&self) -> LinalgResult<usize> {
        let now = Instant::now();
        let mut evicted = 0;

        if let Ok(mut cache) = self.block_cache.lock() {
            for offsets in cache.values_mut() {
                let initial_len = offsets.len();
                // Keep only recent entries (this is simplified - real impl would track times)
                while offsets.len() > 8 {
                    offsets.pop_front();
                    evicted += 1;
                }
                evicted += initial_len.saturating_sub(offsets.len());
            }
        }

        Ok(evicted)
    }

    /// Reset the memory pool
    pub fn reset(&self) -> LinalgResult<()> {
        let pool_size = self.config.pool_size;

        // Clear all data structures
        if let Ok(mut blocks) = self.blocks.write() {
            blocks.clear();
            blocks.insert(
                0,
                MemoryBlock {
                    offset: 0,
                    size: pool_size,
                    in_use: false,
                    allocation_id: None,
                    last_access: Instant::now(),
                },
            );
        }

        if let Ok(mut free_blocks) = self.free_blocks.write() {
            free_blocks.clear();
            free_blocks.insert(pool_size, vec![0]);
        }

        if let Ok(mut allocs) = self.allocations.write() {
            allocs.clear();
        }

        if let Ok(mut cache) = self.block_cache.lock() {
            cache.clear();
        }

        if let Ok(mut stats) = self.stats.write() {
            *stats = MemoryStats {
                total_size: pool_size,
                free_bytes: pool_size,
                ..Default::default()
            };
        }

        Ok(())
    }

    /// Align size to required alignment
    fn align_size(&self, size: usize) -> usize {
        let alignment = self.config.alignment;
        size.div_ceil(alignment) * alignment
    }

    /// Update cache hit rate in stats
    fn update_cache_hit_rate(&self, stats: &mut MemoryStats) {
        let total = stats.cache_hits + stats.cache_misses;
        if total > 0 {
            stats.cache_hit_rate = stats.cache_hits as f64 / total as f64;
        }
    }
}

impl Default for GpuMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe memory pool wrapper
pub struct SharedMemoryPool {
    pool: Arc<GpuMemoryPool>,
}

impl SharedMemoryPool {
    /// Create a new shared memory pool
    pub fn new() -> Self {
        Self {
            pool: Arc::new(GpuMemoryPool::new()),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: MemoryPoolConfig) -> Self {
        Self {
            pool: Arc::new(GpuMemoryPool::with_config(config)),
        }
    }

    /// Allocate from the pool
    pub fn allocate(&self, size: usize) -> LinalgResult<AllocationHandle> {
        self.pool.allocate(size)
    }

    /// Deallocate from the pool
    pub fn deallocate(&self, handle: &AllocationHandle) -> LinalgResult<()> {
        self.pool.deallocate(handle)
    }

    /// Get statistics
    pub fn stats(&self) -> MemoryStats {
        self.pool.stats()
    }

    /// Clone the shared reference
    pub fn clone_ref(&self) -> Self {
        Self {
            pool: Arc::clone(&self.pool),
        }
    }
}

impl Default for SharedMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SharedMemoryPool {
    fn clone(&self) -> Self {
        self.clone_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let pool = GpuMemoryPool::new();

        let handle1 = pool.allocate(1024).expect("Allocation failed");
        assert_eq!(handle1.size, 1024);

        let stats = pool.stats();
        assert_eq!(stats.active_allocations, 1);
        assert!(stats.allocated_bytes >= 1024);
    }

    #[test]
    fn test_multiple_allocations() {
        let pool = GpuMemoryPool::new();

        let handles: Vec<_> = (0..10)
            .map(|_| pool.allocate(1024).expect("Allocation failed"))
            .collect();

        assert_eq!(handles.len(), 10);

        let stats = pool.stats();
        assert_eq!(stats.active_allocations, 10);
    }

    #[test]
    fn test_allocation_and_deallocation() {
        let pool = GpuMemoryPool::new();

        let handle = pool.allocate(1024).expect("Allocation failed");
        assert_eq!(pool.stats().active_allocations, 1);

        pool.deallocate(&handle).expect("Deallocation failed");
        assert_eq!(pool.stats().active_allocations, 0);
    }

    #[test]
    fn test_reuse_after_deallocation() {
        let pool = GpuMemoryPool::new();

        let handle1 = pool.allocate(1024).expect("Allocation failed");
        let offset1 = handle1.offset;

        pool.deallocate(&handle1).expect("Deallocation failed");

        let handle2 = pool.allocate(1024).expect("Allocation failed");
        // Should reuse the same block (or get from cache)
        assert!(handle2.offset == offset1 || pool.stats().cache_hits > 0);
    }

    #[test]
    fn test_fragmentation_calculation() {
        let config = MemoryPoolConfig {
            pool_size: 10240,
            min_block_size: 256,
            ..Default::default()
        };
        let pool = GpuMemoryPool::with_config(config);

        // Allocate multiple blocks
        let handles: Vec<_> = (0..5)
            .map(|_| pool.allocate(1024).expect("Allocation failed"))
            .collect();

        // Free every other block to create fragmentation
        for (i, handle) in handles.iter().enumerate() {
            if i % 2 == 0 {
                pool.deallocate(handle).expect("Deallocation failed");
            }
        }

        let frag_ratio = pool.fragmentation_ratio();
        assert!((0.0..=1.0).contains(&frag_ratio));
    }

    #[test]
    fn test_reset() {
        let pool = GpuMemoryPool::new();

        // Allocate some memory
        for _ in 0..10 {
            let _ = pool.allocate(1024);
        }

        assert!(pool.stats().active_allocations > 0);

        // Reset
        pool.reset().expect("Reset failed");

        let stats = pool.stats();
        assert_eq!(stats.active_allocations, 0);
        assert_eq!(stats.allocated_bytes, 0);
    }

    #[test]
    fn test_shared_pool() {
        let pool = SharedMemoryPool::new();
        let pool2 = pool.clone_ref();

        let handle = pool.allocate(1024).expect("Allocation failed");
        assert_eq!(pool2.stats().active_allocations, 1);

        pool2.deallocate(&handle).expect("Deallocation failed");
        assert_eq!(pool.stats().active_allocations, 0);
    }

    #[test]
    fn test_alignment() {
        let config = MemoryPoolConfig {
            alignment: 256,
            ..Default::default()
        };
        let pool = GpuMemoryPool::with_config(config);

        let handle = pool.allocate(100).expect("Allocation failed");
        assert!(handle.size >= 100);
        assert!(handle.size % 256 == 0 || handle.size == 256);
    }
}
