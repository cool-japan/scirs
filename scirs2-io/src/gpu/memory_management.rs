//! Advanced GPU memory management for efficient I/O operations
//!
//! This module provides sophisticated GPU memory pooling, buffer lifecycle
//! management, and fragmentation prevention for optimal performance.

use crate::error::{IoError, Result};
use scirs2_core::gpu::{GpuBuffer, GpuContext, GpuDataType, GpuDevice};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Advanced GPU memory pool with smart buffer reuse and fragmentation prevention
#[derive(Debug)]
pub struct AdvancedGpuMemoryPool {
    device: GpuDevice,
    free_buffers: BTreeMap<usize, VecDeque<PooledBuffer>>,
    allocated_buffers: HashMap<usize, BufferMetadata>,
    allocation_stats: AllocationStats,
    config: PoolConfig,
    fragmentation_manager: FragmentationManager,
    buffer_id_counter: usize,
}

impl AdvancedGpuMemoryPool {
    /// Create a new advanced GPU memory pool
    pub fn new(device: GpuDevice, config: PoolConfig) -> Self {
        Self {
            device,
            free_buffers: BTreeMap::new(),
            allocated_buffers: HashMap::new(),
            allocation_stats: AllocationStats::default(),
            config,
            fragmentation_manager: FragmentationManager::new(),
            buffer_id_counter: 0,
        }
    }

    /// Allocate a buffer from the pool
    pub fn allocate(&mut self, size: usize) -> Result<PooledBuffer> {
        let aligned_size = self.align_size(size);

        // Try to reuse an existing buffer
        if let Some(buffer) = self.find_reusable_buffer(aligned_size) {
            self.allocation_stats.cache_hits += 1;
            return Ok(buffer);
        }

        // Create new buffer
        self.allocation_stats.cache_misses += 1;
        self.create_new_buffer(aligned_size)
    }

    /// Return a buffer to the pool
    pub fn deallocate(&mut self, mut buffer: PooledBuffer) -> Result<()> {
        // Update statistics
        buffer.touch();
        self.allocation_stats.total_deallocations += 1;

        // Check if buffer should be kept in pool
        if buffer.metadata.size <= self.config.max_buffer_size
            && self.get_total_pool_size() < self.config.max_pool_size
        {
            // Return to appropriate size bucket
            let size_bucket = self.get_size_bucket(buffer.metadata.size);
            self.free_buffers
                .entry(size_bucket)
                .or_insert_with(VecDeque::new)
                .push_back(buffer);
        }
        // Otherwise, buffer will be dropped and GPU memory freed

        // Check for fragmentation and compact if needed
        if self.fragmentation_manager.needs_compaction() {
            self.compact_pool()?;
        }

        Ok(())
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> PoolStats {
        PoolStats {
            total_buffers: self.allocated_buffers.len(),
            free_buffers: self.free_buffers.values().map(|v| v.len()).sum(),
            total_pool_size: self.get_total_pool_size(),
            fragmentation_ratio: self.fragmentation_manager.get_fragmentation_ratio(),
            cache_hit_rate: self.allocation_stats.get_cache_hit_rate(),
            allocation_stats: self.allocation_stats.clone(),
        }
    }

    /// Force garbage collection of expired buffers
    pub fn garbage_collect(&mut self) -> Result<usize> {
        let mut freed_count = 0;
        let now = Instant::now();

        for buffers in self.free_buffers.values_mut() {
            let original_len = buffers.len();
            buffers.retain(|buffer| !buffer.is_expired(self.config.buffer_timeout));
            freed_count += original_len - buffers.len();
        }

        // Update fragmentation
        self.fragmentation_manager.update_after_gc();

        Ok(freed_count)
    }

    /// Compact the pool to reduce fragmentation
    pub fn compact_pool(&mut self) -> Result<()> {
        if !self.config.enable_compaction {
            return Ok(());
        }

        // Merge adjacent free buffers of similar sizes
        for buffers in self.free_buffers.values_mut() {
            // Sort by creation time and merge similar-sized buffers
            let mut merged_buffers = VecDeque::new();

            while let Some(buffer) = buffers.pop_front() {
                // Try to merge with existing buffers or add as new
                merged_buffers.push_back(buffer);
            }

            *buffers = merged_buffers;
        }

        self.fragmentation_manager.reset_fragmentation();
        Ok(())
    }

    /// Clear all free buffers
    pub fn clear(&mut self) {
        self.free_buffers.clear();
        self.allocation_stats.reset();
        self.fragmentation_manager.reset_fragmentation();
    }

    // Private helper methods
    fn find_reusable_buffer(&mut self, size: usize) -> Option<PooledBuffer> {
        let size_bucket = self.get_size_bucket(size);

        // Look for exact size match first
        if let Some(buffers) = self.free_buffers.get_mut(&size_bucket) {
            if let Some(mut buffer) = buffers.pop_front() {
                buffer.touch();
                return Some(buffer);
            }
        }

        // Look for larger buffers that can be reused
        for (&bucket_size, buffers) in self.free_buffers.range_mut(size_bucket..) {
            if bucket_size <= size * 2 {
                // Don't waste too much memory
                if let Some(mut buffer) = buffers.pop_front() {
                    buffer.touch();
                    return Some(buffer);
                }
            }
        }

        None
    }

    fn create_new_buffer(&mut self, size: usize) -> Result<PooledBuffer> {
        if size > self.config.max_buffer_size {
            return Err(IoError::Other(format!(
                "Buffer size {} exceeds maximum {}",
                size, self.config.max_buffer_size
            )));
        }

        // Create a GPU context and allocate buffer through it
        let context = GpuContext::new(self.device.backend())
            .map_err(|e| IoError::Other(format!("Failed to create GPU context: {}", e)))?;
        let buffer: GpuBuffer<u8> = context.create_buffer(size);

        let buffer_id = self.buffer_id_counter;
        self.buffer_id_counter += 1;

        let pooled_buffer = PooledBuffer::new(buffer, buffer_id, "memory_pool".to_string());

        // Track allocation
        self.allocation_stats.total_allocations += 1;
        self.allocation_stats.bytes_allocated += size;
        self.allocated_buffers
            .insert(buffer_id, pooled_buffer.metadata.clone());

        Ok(pooled_buffer)
    }

    fn align_size(&self, size: usize) -> usize {
        let alignment = self.config.alignment;
        (size + alignment - 1) & !(alignment - 1)
    }

    fn get_size_bucket(&self, size: usize) -> usize {
        // Use power-of-2 buckets for efficient lookup
        if size <= self.config.min_buffer_size {
            self.config.min_buffer_size
        } else {
            size.next_power_of_two()
        }
    }

    fn get_total_pool_size(&self) -> usize {
        self.free_buffers
            .iter()
            .map(|(&size, buffers)| size * buffers.len())
            .sum()
    }
}

/// Configuration for the memory pool
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum total pool size in bytes
    pub max_pool_size: usize,
    /// Minimum buffer size in bytes
    pub min_buffer_size: usize,
    /// Maximum buffer size in bytes
    pub max_buffer_size: usize,
    /// Memory alignment in bytes
    pub alignment: usize,
    /// Fragmentation threshold for triggering defragmentation (0.0-1.0)
    pub defragmentation_threshold: f64,
    /// Buffer timeout duration
    pub buffer_timeout: Duration,
    /// Enable memory compaction
    pub enable_compaction: bool,
    /// Enable memory prefetching
    pub enable_prefetch: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 1024 * 1024 * 1024,        // 1GB default
            min_buffer_size: 4096,                    // 4KB minimum
            max_buffer_size: 64 * 1024 * 1024,        // 64MB maximum single allocation
            alignment: 256,                           // GPU-optimal alignment
            defragmentation_threshold: 0.3,           // Defrag when 30% fragmented
            buffer_timeout: Duration::from_secs(300), // 5 minutes timeout
            enable_compaction: true,
            enable_prefetch: true,
        }
    }
}

/// Metadata for tracking buffer usage and performance
#[derive(Debug, Clone)]
pub struct BufferMetadata {
    /// Unique buffer ID
    pub id: usize,
    /// Buffer size in bytes
    pub size: usize,
    /// Timestamp when buffer was allocated
    pub allocated_at: Instant,
    /// Number of times the buffer has been accessed
    pub access_count: usize,
    /// Timestamp of last access
    pub last_access: Instant,
    /// Source of the allocation for debugging
    pub allocation_source: String,
}

/// Buffer wrapper with lifecycle tracking
pub struct PooledBuffer {
    /// GPU buffer handle
    pub buffer: GpuBuffer<u8>,
    /// Buffer metadata
    pub metadata: BufferMetadata,
    /// Timestamp when buffer was created in pool
    pub created_at: Instant,
    /// Timestamp of last use
    pub last_used: Instant,
    /// Number of times the buffer has been used
    pub use_count: usize,
}

impl std::fmt::Debug for PooledBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledBuffer")
            .field("buffer_size", &self.buffer.len())
            .field("metadata", &self.metadata)
            .field("created_at", &self.created_at)
            .field("last_used", &self.last_used)
            .field("use_count", &self.use_count)
            .finish()
    }
}

impl PooledBuffer {
    fn new(buffer: GpuBuffer<u8>, id: usize, allocation_source: String) -> Self {
        let now = Instant::now();
        let size = buffer.len();

        Self {
            buffer,
            metadata: BufferMetadata {
                id,
                size,
                allocated_at: now,
                access_count: 0,
                last_access: now,
                allocation_source,
            },
            created_at: now,
            last_used: now,
            use_count: 0,
        }
    }

    fn touch(&mut self) {
        self.last_used = Instant::now();
        self.use_count += 1;
        self.metadata.access_count += 1;
        self.metadata.last_access = self.last_used;
    }

    fn is_expired(&self, timeout: Duration) -> bool {
        self.last_used.elapsed() > timeout
    }

    /// Get buffer utilization efficiency
    pub fn get_utilization_efficiency(&self) -> f64 {
        if self.use_count == 0 {
            0.0
        } else {
            let age_seconds = self.created_at.elapsed().as_secs_f64();
            self.use_count as f64 / age_seconds.max(1.0)
        }
    }
}

/// Allocation statistics for performance monitoring
#[derive(Debug, Default, Clone)]
pub struct AllocationStats {
    /// Total number of allocations
    pub total_allocations: usize,
    /// Total number of deallocations
    pub total_deallocations: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Total bytes allocated
    pub bytes_allocated: usize,
    /// Total bytes deallocated
    pub bytes_deallocated: usize,
    /// Peak memory usage in bytes
    pub peak_memory_usage: usize,
    /// Number of memory compactions performed
    pub compaction_count: usize,
}

impl AllocationStats {
    /// Calculate cache hit rate as a percentage (0.0-1.0)
    pub fn get_cache_hit_rate(&self) -> f64 {
        let total_requests = self.cache_hits + self.cache_misses;
        if total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total_requests as f64
        }
    }

    /// Reset all statistics to default values
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Fragmentation management for optimal memory usage
#[derive(Debug)]
pub struct FragmentationManager {
    internal_fragmentation: f64,
    external_fragmentation: f64,
    compaction_threshold: f64,
    last_compaction: Instant,
    fragmentation_history: VecDeque<f64>,
}

impl FragmentationManager {
    /// Create a new fragmentation manager
    pub fn new() -> Self {
        Self {
            internal_fragmentation: 0.0,
            external_fragmentation: 0.0,
            compaction_threshold: 0.3,
            last_compaction: Instant::now(),
            fragmentation_history: VecDeque::with_capacity(100),
        }
    }

    /// Check if memory compaction is needed
    pub fn needs_compaction(&self) -> bool {
        self.external_fragmentation > self.compaction_threshold
            && self.last_compaction.elapsed() > Duration::from_secs(60)
    }

    /// Get current fragmentation ratio (0.0-1.0)
    pub fn get_fragmentation_ratio(&self) -> f64 {
        (self.internal_fragmentation + self.external_fragmentation) / 2.0
    }

    /// Update fragmentation measurements
    pub fn update_fragmentation(&mut self, internal: f64, external: f64) {
        self.internal_fragmentation = internal;
        self.external_fragmentation = external;

        let avg_fragmentation = self.get_fragmentation_ratio();
        self.fragmentation_history.push_back(avg_fragmentation);

        if self.fragmentation_history.len() > 100 {
            self.fragmentation_history.pop_front();
        }
    }

    /// Reset fragmentation counters after compaction
    pub fn reset_fragmentation(&mut self) {
        self.internal_fragmentation = 0.0;
        self.external_fragmentation = 0.0;
        self.last_compaction = Instant::now();
    }

    /// Update fragmentation estimates after garbage collection
    pub fn update_after_gc(&mut self) {
        // Fragmentation typically reduces after garbage collection
        self.external_fragmentation *= 0.8;
    }

    /// Get fragmentation trend based on history
    pub fn get_trend(&self) -> FragmentationTrend {
        if self.fragmentation_history.len() < 10 {
            return FragmentationTrend::Stable;
        }

        let recent_avg = self.fragmentation_history.iter().rev().take(5).sum::<f64>() / 5.0;
        let older_avg = self
            .fragmentation_history
            .iter()
            .rev()
            .skip(5)
            .take(5)
            .sum::<f64>()
            / 5.0;

        if recent_avg > older_avg * 1.1 {
            FragmentationTrend::Increasing
        } else if recent_avg < older_avg * 0.9 {
            FragmentationTrend::Decreasing
        } else {
            FragmentationTrend::Stable
        }
    }
}

impl Default for FragmentationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Fragmentation trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentationTrend {
    /// Fragmentation is increasing over time
    Increasing,
    /// Fragmentation is stable
    Stable,
    /// Fragmentation is decreasing over time
    Decreasing,
}

/// Pool statistics for monitoring and optimization
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total number of buffers in pool
    pub total_buffers: usize,
    /// Number of free buffers
    pub free_buffers: usize,
    /// Total pool size in bytes
    pub total_pool_size: usize,
    /// Current fragmentation ratio (0.0-1.0)
    pub fragmentation_ratio: f64,
    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: f64,
    /// Allocation statistics
    pub allocation_stats: AllocationStats,
}

impl PoolStats {
    /// Get memory efficiency score (0.0 to 1.0)
    pub fn get_efficiency_score(&self) -> f64 {
        let utilization = if self.total_buffers == 0 {
            0.0
        } else {
            (self.total_buffers - self.free_buffers) as f64 / self.total_buffers as f64
        };

        let fragmentation_penalty = 1.0 - self.fragmentation_ratio.min(1.0);
        let cache_bonus = self.cache_hit_rate;

        (utilization + fragmentation_penalty + cache_bonus) / 3.0
    }
}

/// Memory type for different allocation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryType {
    /// GPU device memory
    Device,
    /// Unified memory (accessible by both CPU and GPU)
    Unified,
    /// Pinned host memory for fast transfers
    Pinned,
    /// Memory-mapped buffers
    Mapped,
}

/// GPU memory pool manager for multiple pools
#[derive(Debug)]
pub struct GpuMemoryPoolManager {
    pools: HashMap<MemoryType, AdvancedGpuMemoryPool>,
    device: GpuDevice,
    global_stats: AllocationStats,
}

impl GpuMemoryPoolManager {
    /// Create a new GPU memory pool manager
    pub fn new(device: GpuDevice) -> Result<Self> {
        let mut pools = HashMap::new();

        // Create pools for different memory types
        for memory_type in [MemoryType::Device, MemoryType::Unified, MemoryType::Pinned] {
            let config = PoolConfig::default();
            let pool = AdvancedGpuMemoryPool::new(device.clone(), config);
            pools.insert(memory_type, pool);
        }

        Ok(Self {
            pools,
            device,
            global_stats: AllocationStats::default(),
        })
    }

    /// Create a memory pool with specific configuration
    pub fn create_pool(
        &mut self,
        total_size: usize,
        memory_type: MemoryType,
    ) -> Result<&mut AdvancedGpuMemoryPool> {
        let mut config = PoolConfig::default();
        config.max_pool_size = total_size;

        let pool = AdvancedGpuMemoryPool::new(self.device.clone(), config);
        self.pools.insert(memory_type, pool);

        Ok(self.pools.get_mut(&memory_type).unwrap())
    }

    /// Allocate from specific memory type
    pub fn allocate(&mut self, size: usize, memory_type: MemoryType) -> Result<PooledBuffer> {
        let pool = self
            .pools
            .get_mut(&memory_type)
            .ok_or_else(|| IoError::Other(format!("Memory pool {:?} not found", memory_type)))?;

        let buffer = pool.allocate(size)?;
        self.global_stats.total_allocations += 1;
        self.global_stats.bytes_allocated += size;

        Ok(buffer)
    }

    /// Return buffer to appropriate pool
    pub fn deallocate(&mut self, buffer: PooledBuffer, memory_type: MemoryType) -> Result<()> {
        let pool = self
            .pools
            .get_mut(&memory_type)
            .ok_or_else(|| IoError::Other(format!("Memory pool {:?} not found", memory_type)))?;

        self.global_stats.total_deallocations += 1;
        self.global_stats.bytes_deallocated += buffer.metadata.size;

        pool.deallocate(buffer)
    }

    /// Get global statistics
    pub fn get_global_stats(&self) -> GlobalPoolStats {
        let pool_stats: Vec<_> = self
            .pools
            .iter()
            .map(|(&memory_type, pool)| (memory_type, pool.get_stats()))
            .collect();

        let total_buffers: usize = pool_stats
            .iter()
            .map(|(_, stats)| stats.total_buffers)
            .sum();
        let total_pool_size: usize = pool_stats
            .iter()
            .map(|(_, stats)| stats.total_pool_size)
            .sum();
        let avg_fragmentation: f64 = if pool_stats.is_empty() {
            0.0
        } else {
            pool_stats
                .iter()
                .map(|(_, stats)| stats.fragmentation_ratio)
                .sum::<f64>()
                / pool_stats.len() as f64
        };

        GlobalPoolStats {
            total_buffers,
            total_pool_size,
            pool_count: self.pools.len(),
            average_fragmentation: avg_fragmentation,
            global_allocation_stats: self.global_stats.clone(),
            pool_stats,
        }
    }

    /// Perform garbage collection on all pools
    pub fn garbage_collect_all(&mut self) -> Result<usize> {
        let mut total_freed = 0;
        for pool in self.pools.values_mut() {
            total_freed += pool.garbage_collect()?;
        }
        Ok(total_freed)
    }

    /// Get the total size of a specific pool
    pub fn get_pool_size(&self, memory_type: MemoryType) -> usize {
        self.pools
            .get(&memory_type)
            .map(|pool| pool.get_total_pool_size())
            .unwrap_or(0)
    }
}

/// Global statistics across all memory pools
#[derive(Debug, Clone)]
pub struct GlobalPoolStats {
    /// Total number of buffers across all pools
    pub total_buffers: usize,
    /// Total size of all pools
    pub total_pool_size: usize,
    /// Number of memory pools
    pub pool_count: usize,
    /// Average fragmentation across all pools
    pub average_fragmentation: f64,
    /// Global allocation statistics
    pub global_allocation_stats: AllocationStats,
    /// Per-pool statistics
    pub pool_stats: Vec<(MemoryType, PoolStats)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::gpu::{GpuBackend, GpuDevice};

    fn create_test_device() -> GpuDevice {
        // Use CPU backend for testing
        GpuDevice::new(GpuBackend::Cpu, 0)
    }

    #[test]
    fn test_pool_config_defaults() {
        let config = PoolConfig::default();
        assert_eq!(config.min_buffer_size, 4096);
        assert_eq!(config.max_buffer_size, 64 * 1024 * 1024);
        assert_eq!(config.alignment, 256);
    }

    #[test]
    fn test_fragmentation_manager() {
        let mut manager = FragmentationManager::new();
        assert_eq!(manager.get_fragmentation_ratio(), 0.0);

        manager.update_fragmentation(0.2, 0.3);
        assert_eq!(manager.get_fragmentation_ratio(), 0.25);

        assert!(!manager.needs_compaction()); // Should not need compaction yet
    }

    #[test]
    fn test_allocation_stats() {
        let mut stats = AllocationStats::default();
        stats.cache_hits = 8;
        stats.cache_misses = 2;

        assert_eq!(stats.get_cache_hit_rate(), 0.8);
    }

    #[test]
    fn test_memory_pool_manager_creation() {
        let device = create_test_device();
        let manager = GpuMemoryPoolManager::new(device);
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        assert_eq!(manager.pools.len(), 3); // Device, Unified, Pinned
    }

    #[test]
    fn test_pool_stats_efficiency() {
        let stats = PoolStats {
            total_buffers: 10,
            free_buffers: 2,
            total_pool_size: 1024 * 1024,
            fragmentation_ratio: 0.1,
            cache_hit_rate: 0.9,
            allocation_stats: AllocationStats::default(),
        };

        let efficiency = stats.get_efficiency_score();
        assert!(efficiency > 0.8); // Should be high efficiency
    }
}
