//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::pool::{MemoryError, MemoryResult};
use crate::gpu::GpuError;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Compaction allocator statistics
#[derive(Debug, Clone)]
pub struct CompactionAllocatorStatistics {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub total_compactions: usize,
    pub total_relocations: usize,
    pub bytes_relocated: usize,
    pub current_allocated_bytes: usize,
    pub free_region_count: usize,
    pub allocated_blocks: usize,
    pub total_free_bytes: usize,
    pub largest_free_block: usize,
    pub fragmentation_ratio: f64,
    pub total_memory: usize,
}
/// Hybrid allocator statistics
#[derive(Debug, Clone)]
pub struct HybridAllocatorStatistics {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub small_allocations: usize,
    pub medium_allocations: usize,
    pub large_allocations: usize,
    pub slab_statistics: SlabAllocatorStatistics,
    pub buddy_statistics: BuddyAllocatorStatistics,
    pub large_allocation_count: usize,
    pub large_allocation_bytes: usize,
    pub small_threshold: usize,
    pub medium_threshold: usize,
}
/// Slab allocator for fixed-size block allocation
///
/// The slab allocator maintains multiple pools of fixed-size blocks, providing
/// O(1) allocation and deallocation with zero fragmentation for common sizes.
/// This is particularly useful for GPU kernels with known buffer sizes.
///
/// # Time Complexity
/// - Allocation: O(1)
/// - Deallocation: O(1)
///
/// # Space Complexity
/// - O(n) where n is the number of slabs
///
/// # Fragmentation
/// - Zero internal fragmentation within each slab size
/// - Minimal external fragmentation due to fixed sizes
#[derive(Debug)]
pub struct SlabAllocator {
    slab_sizes: Vec<usize>,
    free_lists: Arc<Mutex<HashMap<usize, Vec<usize>>>>,
    allocated_blocks: Arc<Mutex<HashMap<usize, usize>>>,
    slabs_per_size: HashMap<usize, usize>,
    total_memory_per_size: HashMap<usize, usize>,
    allocations_per_size: Arc<Mutex<HashMap<usize, usize>>>,
    deallocations_per_size: Arc<Mutex<HashMap<usize, usize>>>,
    total_allocations: Arc<AtomicUsize>,
    total_deallocations: Arc<AtomicUsize>,
    current_allocated_bytes: Arc<AtomicUsize>,
}
impl SlabAllocator {
    /// Create a new slab allocator with default size classes
    ///
    /// Default sizes: 64B, 256B, 1KB, 4KB, 16KB, 64KB
    pub fn new() -> Self {
        Self::with_sizes(vec![64, 256, 1024, 4096, 16384, 65536])
    }
    /// Create a slab allocator with custom size classes
    ///
    /// # Arguments
    /// * `slab_sizes` - List of slab sizes in bytes
    ///
    /// # Example
    /// ```ignore
    /// let allocator = SlabAllocator::with_sizes(vec![128, 512, 2048]);
    /// ```
    pub fn with_sizes(slab_sizes: Vec<usize>) -> Self {
        let mut free_lists = HashMap::new();
        let mut slabs_per_size = HashMap::new();
        let mut total_memory_per_size = HashMap::new();
        let mut allocations_per_size = HashMap::new();
        let mut deallocations_per_size = HashMap::new();
        let slabs_count = 1024;
        for (idx, &size) in slab_sizes.iter().enumerate() {
            let mut free_list = Vec::new();
            for i in 0..slabs_count {
                let base_offset = idx * slabs_count * 65536;
                let offset = base_offset + i * size;
                free_list.push(offset);
            }
            free_lists.insert(size, free_list);
            slabs_per_size.insert(size, slabs_count);
            total_memory_per_size.insert(size, slabs_count * size);
            allocations_per_size.insert(size, 0);
            deallocations_per_size.insert(size, 0);
        }
        Self {
            slab_sizes,
            free_lists: Arc::new(Mutex::new(free_lists)),
            allocated_blocks: Arc::new(Mutex::new(HashMap::new())),
            slabs_per_size,
            total_memory_per_size,
            allocations_per_size: Arc::new(Mutex::new(allocations_per_size)),
            deallocations_per_size: Arc::new(Mutex::new(deallocations_per_size)),
            total_allocations: Arc::new(AtomicUsize::new(0)),
            total_deallocations: Arc::new(AtomicUsize::new(0)),
            current_allocated_bytes: Arc::new(AtomicUsize::new(0)),
        }
    }
    /// Find the best slab size for a requested size
    fn find_slab_size(&self, size: usize) -> Option<usize> {
        self.slab_sizes.iter().find(|&&s| s >= size).copied()
    }
    /// Allocate a block of the specified size
    ///
    /// # Arguments
    /// * `size` - Size to allocate in bytes
    ///
    /// # Returns
    /// Offset of the allocated block or error
    pub fn allocate(&self, size: usize) -> MemoryResult<usize> {
        let slab_size = self
            .find_slab_size(size)
            .ok_or(MemoryError::InvalidSize(size))?;
        let mut free_lists = self.free_lists.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other("Failed to lock free lists".to_string()))
        })?;
        let offset = free_lists
            .get_mut(&slab_size)
            .and_then(|list: &mut Vec<usize>| list.pop())
            .ok_or(MemoryError::OutOfMemory(format!(
                "No free slabs of size {}",
                slab_size
            )))?;
        drop(free_lists);
        let mut allocated = self.allocated_blocks.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other(
                "Failed to lock allocated blocks".to_string(),
            ))
        })?;
        allocated.insert(offset, slab_size);
        drop(allocated);
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.current_allocated_bytes
            .fetch_add(slab_size, Ordering::Relaxed);
        if let Ok(mut stats) = self.allocations_per_size.lock() {
            *stats.entry(slab_size).or_insert(0) += 1;
        }
        Ok(offset)
    }
    /// Deallocate a previously allocated block
    ///
    /// # Arguments
    /// * `offset` - Offset of the block to deallocate
    pub fn deallocate(&self, offset: usize) -> MemoryResult<()> {
        let mut allocated = self.allocated_blocks.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other(
                "Failed to lock allocated blocks".to_string(),
            ))
        })?;
        let slab_size = allocated
            .remove(&offset)
            .ok_or(MemoryError::BufferNotFound(offset as u64))?;
        drop(allocated);
        let mut free_lists = self.free_lists.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other("Failed to lock free lists".to_string()))
        })?;
        if let Some(list) = free_lists.get_mut(&slab_size) {
            let list: &mut Vec<usize> = list;
            list.push(offset);
        }
        drop(free_lists);
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);
        self.current_allocated_bytes
            .fetch_sub(slab_size, Ordering::Relaxed);
        if let Ok(mut stats) = self.deallocations_per_size.lock() {
            *stats.entry(slab_size).or_insert(0) += 1;
        }
        Ok(())
    }
    /// Get allocator statistics
    pub fn statistics(&self) -> SlabAllocatorStatistics {
        let free_lists = self.free_lists.lock().ok();
        let allocated_blocks = self.allocated_blocks.lock().ok();
        let allocations = self.allocations_per_size.lock().ok();
        let deallocations = self.deallocations_per_size.lock().ok();
        let mut per_size_stats = Vec::new();
        for &size in &self.slab_sizes {
            let total_slabs = self.slabs_per_size.get(&size).copied().unwrap_or(0);
            let free_slabs = free_lists
                .as_ref()
                .and_then(|fl| fl.get(&size).map(|v| v.len()))
                .unwrap_or(0);
            let allocated_slabs = total_slabs - free_slabs;
            let alloc_count = allocations
                .as_ref()
                .and_then(|a| a.get(&size).copied())
                .unwrap_or(0);
            let dealloc_count = deallocations
                .as_ref()
                .and_then(|d| d.get(&size).copied())
                .unwrap_or(0);
            per_size_stats.push(SlabSizeStatistics {
                slab_size: size,
                total_slabs,
                free_slabs,
                allocated_slabs,
                allocations: alloc_count,
                deallocations: dealloc_count,
                utilization: if total_slabs > 0 {
                    allocated_slabs as f64 / total_slabs as f64
                } else {
                    0.0
                },
            });
        }
        let total_memory: usize = self.total_memory_per_size.values().sum();
        SlabAllocatorStatistics {
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
            total_deallocations: self.total_deallocations.load(Ordering::Relaxed),
            current_allocated_bytes: self.current_allocated_bytes.load(Ordering::Relaxed),
            total_memory,
            per_size_stats,
            fragmentation_ratio: 0.0,
        }
    }
    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_allocated_bytes.load(Ordering::Relaxed)
    }
    /// Get total memory capacity
    pub fn total_memory(&self) -> usize {
        self.total_memory_per_size.values().sum()
    }
}
/// Allocation strategy selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AllocationStrategy {
    Slab,
    Buddy,
    Direct,
}
/// Compaction allocator with memory defragmentation support
///
/// The compaction allocator manages memory with the ability to defragment
/// by relocating buffers during idle periods. This helps maintain low
/// fragmentation over time, especially for long-running applications.
///
/// # Time Complexity
/// - Allocation: O(n) where n is the number of free blocks
/// - Deallocation: O(1)
/// - Compaction: O(m) where m is the number of allocated blocks
///
/// # Space Complexity
/// - O(n) where n is the number of blocks
///
/// # Fragmentation
/// - Can achieve near-zero fragmentation with periodic compaction
/// - Automatic compaction triggered by fragmentation threshold
#[derive(Debug)]
pub struct CompactionAllocator {
    total_memory: usize,
    free_regions: Arc<Mutex<Vec<(usize, usize)>>>,
    allocated_blocks: Arc<Mutex<HashMap<usize, (usize, bool)>>>,
    compaction_threshold: f64,
    total_allocations: Arc<AtomicUsize>,
    total_deallocations: Arc<AtomicUsize>,
    total_compactions: Arc<AtomicUsize>,
    total_relocations: Arc<AtomicUsize>,
    bytes_relocated: Arc<AtomicUsize>,
    current_allocated_bytes: Arc<AtomicUsize>,
}
impl CompactionAllocator {
    /// Create a new compaction allocator
    ///
    /// # Arguments
    /// * `total_memory` - Total memory to manage
    /// * `compaction_threshold` - Fragmentation threshold (0.0 - 1.0)
    ///
    /// # Example
    /// ```ignore
    /// let allocator = CompactionAllocator::new(1024 * 1024, 0.3); // 30% fragmentation threshold
    /// ```
    pub fn new(total_memory: usize, compaction_threshold: f64) -> Self {
        let free_regions = vec![(0, total_memory)];
        Self {
            total_memory,
            free_regions: Arc::new(Mutex::new(free_regions)),
            allocated_blocks: Arc::new(Mutex::new(HashMap::new())),
            compaction_threshold,
            total_allocations: Arc::new(AtomicUsize::new(0)),
            total_deallocations: Arc::new(AtomicUsize::new(0)),
            total_compactions: Arc::new(AtomicUsize::new(0)),
            total_relocations: Arc::new(AtomicUsize::new(0)),
            bytes_relocated: Arc::new(AtomicUsize::new(0)),
            current_allocated_bytes: Arc::new(AtomicUsize::new(0)),
        }
    }
    /// Allocate memory of the specified size
    ///
    /// # Arguments
    /// * `size` - Size to allocate in bytes
    /// * `can_relocate` - Whether this buffer can be relocated during compaction
    ///
    /// # Returns
    /// Offset of the allocated block or error
    pub fn allocate(&self, size: usize, can_relocate: bool) -> MemoryResult<usize> {
        if size == 0 {
            return Err(MemoryError::InvalidSize(size));
        }
        let mut free_regions = self.free_regions.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other("Failed to lock free regions".to_string()))
        })?;
        let region_idx = free_regions
            .iter()
            .position(|(_, region_size)| *region_size >= size)
            .ok_or(MemoryError::OutOfMemory(format!(
                "Cannot allocate {} bytes",
                size
            )))?;
        let (offset, region_size) = free_regions[region_idx];
        if region_size > size {
            free_regions[region_idx] = (offset + size, region_size - size);
        } else {
            free_regions.remove(region_idx);
        }
        drop(free_regions);
        let mut allocated = self.allocated_blocks.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other(
                "Failed to lock allocated blocks".to_string(),
            ))
        })?;
        allocated.insert(offset, (size, can_relocate));
        drop(allocated);
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.current_allocated_bytes
            .fetch_add(size, Ordering::Relaxed);
        Ok(offset)
    }
    /// Deallocate a previously allocated block
    ///
    /// # Arguments
    /// * `offset` - Offset of the block to deallocate
    pub fn deallocate(&self, offset: usize) -> MemoryResult<()> {
        let mut allocated = self.allocated_blocks.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other(
                "Failed to lock allocated blocks".to_string(),
            ))
        })?;
        let (size, _) = allocated
            .remove(&offset)
            .ok_or(MemoryError::BufferNotFound(offset as u64))?;
        drop(allocated);
        let mut free_regions = self.free_regions.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other("Failed to lock free regions".to_string()))
        })?;
        free_regions.push((offset, size));
        self.coalesce_free_regions(&mut free_regions);
        drop(free_regions);
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);
        self.current_allocated_bytes
            .fetch_sub(size, Ordering::Relaxed);
        Ok(())
    }
    /// Coalesce adjacent free regions to reduce fragmentation
    fn coalesce_free_regions(&self, regions: &mut Vec<(usize, usize)>) {
        if regions.len() <= 1 {
            return;
        }
        regions.sort_by_key(|(offset, _)| *offset);
        let mut i = 0;
        while i < regions.len() - 1 {
            let (offset1, size1) = regions[i];
            let (offset2, size2) = regions[i + 1];
            if offset1 + size1 == offset2 {
                regions[i] = (offset1, size1 + size2);
                regions.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }
    /// Calculate current fragmentation ratio
    fn calculate_fragmentation(&self) -> f64 {
        let free_regions = match self.free_regions.lock() {
            Ok(regions) => regions,
            Err(_) => return 0.0,
        };
        if free_regions.is_empty() {
            return 0.0;
        }
        let total_free: usize = free_regions.iter().map(|(_, size)| size).sum();
        let largest_free = free_regions
            .iter()
            .map(|(_, size)| size)
            .max()
            .unwrap_or(&0);
        if total_free == 0 {
            return 0.0;
        }
        1.0 - (*largest_free as f64 / total_free as f64)
    }
    /// Check if compaction is needed
    pub fn needs_compaction(&self) -> bool {
        self.calculate_fragmentation() > self.compaction_threshold
    }
    /// Perform memory compaction
    ///
    /// This relocates buffers to reduce fragmentation. Only relocates buffers
    /// that were marked as `can_relocate` during allocation.
    ///
    /// # Returns
    /// Number of buffers relocated
    pub fn compact(&self) -> MemoryResult<usize> {
        let mut allocated = self.allocated_blocks.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other(
                "Failed to lock allocated blocks".to_string(),
            ))
        })?;
        let mut free_regions = self.free_regions.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other("Failed to lock free regions".to_string()))
        })?;
        let mut relocatable: Vec<(usize, usize, bool)> = allocated
            .iter()
            .filter_map(|(offset, (size, can_relocate))| {
                if *can_relocate {
                    Some((*offset, *size, *can_relocate))
                } else {
                    None
                }
            })
            .collect();
        relocatable.sort_by_key(|(offset, _, _)| *offset);
        let mut relocated_count = 0;
        let mut total_bytes_relocated = 0;
        let mut current_offset = 0;
        for (old_offset, size, can_relocate) in relocatable {
            if current_offset < old_offset {
                allocated.remove(&old_offset);
                allocated.insert(current_offset, (size, can_relocate));
                free_regions.push((old_offset, size));
                relocated_count += 1;
                total_bytes_relocated += size;
            }
            current_offset += size;
        }
        if relocated_count > 0 {
            self.coalesce_free_regions(&mut free_regions);
        }
        drop(allocated);
        drop(free_regions);
        if relocated_count > 0 {
            self.total_compactions.fetch_add(1, Ordering::Relaxed);
            self.total_relocations
                .fetch_add(relocated_count, Ordering::Relaxed);
            self.bytes_relocated
                .fetch_add(total_bytes_relocated, Ordering::Relaxed);
        }
        Ok(relocated_count)
    }
    /// Get allocator statistics
    pub fn statistics(&self) -> CompactionAllocatorStatistics {
        let free_regions = self.free_regions.lock().ok();
        let allocated_blocks = self.allocated_blocks.lock().ok();
        let free_region_count = free_regions.as_ref().map(|fr| fr.len()).unwrap_or(0);
        let allocated_count = allocated_blocks.as_ref().map(|ab| ab.len()).unwrap_or(0);
        let total_free: usize = free_regions
            .as_ref()
            .map(|fr| fr.iter().map(|(_, size)| size).sum())
            .unwrap_or(0);
        let largest_free = free_regions
            .as_ref()
            .and_then(|fr| fr.iter().map(|(_, size)| size).max().copied())
            .unwrap_or(0);
        // Compute fragmentation inline using already-held lock guard.
        // Do NOT call self.calculate_fragmentation() here: that method acquires
        // self.free_regions.lock() again, which deadlocks because std::sync::Mutex
        // is non-reentrant and the guard above is still live.
        let fragmentation_ratio = if total_free == 0 {
            0.0
        } else {
            1.0 - (largest_free as f64 / total_free as f64)
        };
        CompactionAllocatorStatistics {
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
            total_deallocations: self.total_deallocations.load(Ordering::Relaxed),
            total_compactions: self.total_compactions.load(Ordering::Relaxed),
            total_relocations: self.total_relocations.load(Ordering::Relaxed),
            bytes_relocated: self.bytes_relocated.load(Ordering::Relaxed),
            current_allocated_bytes: self.current_allocated_bytes.load(Ordering::Relaxed),
            free_region_count,
            allocated_blocks: allocated_count,
            total_free_bytes: total_free,
            largest_free_block: largest_free,
            fragmentation_ratio,
            total_memory: self.total_memory,
        }
    }
    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_allocated_bytes.load(Ordering::Relaxed)
    }
    /// Get available memory
    pub fn available_memory(&self) -> usize {
        self.total_memory.saturating_sub(self.current_usage())
    }
}
/// Hybrid allocator strategy combining multiple allocators
///
/// The hybrid strategy combines different allocators based on allocation size:
/// - Small buffers (< 64KB): SlabAllocator for O(1) allocation
/// - Medium buffers (64KB - 16MB): BuddyAllocator for balanced performance
/// - Large buffers (> 16MB): Direct allocation for minimal overhead
///
/// # Time Complexity
/// - Small: O(1) via slab allocator
/// - Medium: O(log n) via buddy allocator
/// - Large: O(1) direct allocation
///
/// # Space Complexity
/// - O(n) where n is the total number of blocks across all strategies
///
/// # Fragmentation
/// - Near-zero for small allocations (slab)
/// - Low for medium allocations (buddy)
/// - Minimal overhead for large allocations
#[derive(Debug)]
pub struct HybridAllocator {
    slab_allocator: SlabAllocator,
    buddy_allocator: BuddyAllocator,
    large_allocations: Arc<Mutex<HashMap<usize, usize>>>,
    small_threshold: usize,
    medium_threshold: usize,
    large_offset_counter: Arc<AtomicUsize>,
    small_allocations: Arc<AtomicUsize>,
    medium_allocations: Arc<AtomicUsize>,
    large_allocations_count: Arc<AtomicUsize>,
    total_allocations: Arc<AtomicUsize>,
    total_deallocations: Arc<AtomicUsize>,
}
impl HybridAllocator {
    /// Create a new hybrid allocator with default thresholds
    ///
    /// Default thresholds:
    /// - Small: < 64KB (slab)
    /// - Medium: 64KB - 16MB (buddy)
    /// - Large: > 16MB (direct)
    pub fn new() -> MemoryResult<Self> {
        Self::with_thresholds(65536, 16 * 1024 * 1024)
    }
    /// Create a hybrid allocator with custom thresholds
    ///
    /// # Arguments
    /// * `small_threshold` - Maximum size for slab allocator
    /// * `medium_threshold` - Maximum size for buddy allocator
    ///
    /// # Example
    /// ```ignore
    /// let allocator = HybridAllocator::with_thresholds(32768, 8 * 1024 * 1024)?;
    /// ```
    pub fn with_thresholds(small_threshold: usize, medium_threshold: usize) -> MemoryResult<Self> {
        let slab_sizes = vec![64, 256, 1024, 4096, 16384, small_threshold];
        let slab_allocator = SlabAllocator::with_sizes(slab_sizes);
        let buddy_total = medium_threshold * 16;
        let buddy_allocator = BuddyAllocator::new(buddy_total, small_threshold)?;
        Ok(Self {
            slab_allocator,
            buddy_allocator,
            large_allocations: Arc::new(Mutex::new(HashMap::new())),
            small_threshold,
            medium_threshold,
            large_offset_counter: Arc::new(AtomicUsize::new(1_000_000_000)),
            small_allocations: Arc::new(AtomicUsize::new(0)),
            medium_allocations: Arc::new(AtomicUsize::new(0)),
            large_allocations_count: Arc::new(AtomicUsize::new(0)),
            total_allocations: Arc::new(AtomicUsize::new(0)),
            total_deallocations: Arc::new(AtomicUsize::new(0)),
        })
    }
    /// Determine which strategy to use for a given size
    fn select_strategy(&self, size: usize) -> AllocationStrategy {
        if size <= self.small_threshold {
            AllocationStrategy::Slab
        } else if size <= self.medium_threshold {
            AllocationStrategy::Buddy
        } else {
            AllocationStrategy::Direct
        }
    }
    /// Allocate memory using the appropriate strategy
    ///
    /// # Arguments
    /// * `size` - Size to allocate in bytes
    ///
    /// # Returns
    /// Offset of the allocated block or error
    pub fn allocate(&self, size: usize) -> MemoryResult<usize> {
        if size == 0 {
            return Err(MemoryError::InvalidSize(size));
        }
        let strategy = self.select_strategy(size);
        let offset = match strategy {
            AllocationStrategy::Slab => {
                self.small_allocations.fetch_add(1, Ordering::Relaxed);
                self.slab_allocator.allocate(size)?
            }
            AllocationStrategy::Buddy => {
                self.medium_allocations.fetch_add(1, Ordering::Relaxed);
                self.buddy_allocator.allocate(size)?
            }
            AllocationStrategy::Direct => {
                self.large_allocations_count.fetch_add(1, Ordering::Relaxed);
                let offset = self.large_offset_counter.fetch_add(size, Ordering::Relaxed);
                let mut large = self.large_allocations.lock().map_err(|_| {
                    MemoryError::GpuError(GpuError::Other(
                        "Failed to lock large allocations".to_string(),
                    ))
                })?;
                large.insert(offset, size);
                offset
            }
        };
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        Ok(offset)
    }
    /// Deallocate a previously allocated block
    ///
    /// # Arguments
    /// * `offset` - Offset of the block to deallocate
    /// * `size` - Size of the block (needed to determine strategy)
    pub fn deallocate(&self, offset: usize, size: usize) -> MemoryResult<()> {
        let strategy = self.select_strategy(size);
        match strategy {
            AllocationStrategy::Slab => {
                self.slab_allocator.deallocate(offset)?;
            }
            AllocationStrategy::Buddy => {
                self.buddy_allocator.deallocate(offset)?;
            }
            AllocationStrategy::Direct => {
                let mut large = self.large_allocations.lock().map_err(|_| {
                    MemoryError::GpuError(GpuError::Other(
                        "Failed to lock large allocations".to_string(),
                    ))
                })?;
                large
                    .remove(&offset)
                    .ok_or(MemoryError::BufferNotFound(offset as u64))?;
            }
        }
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    /// Get allocator statistics
    pub fn statistics(&self) -> HybridAllocatorStatistics {
        let slab_stats = self.slab_allocator.statistics();
        let buddy_stats = self.buddy_allocator.statistics();
        let large = self.large_allocations.lock().ok();
        let large_count = large.as_ref().map(|l| l.len()).unwrap_or(0);
        let large_bytes: usize = large.as_ref().map(|l| l.values().sum()).unwrap_or(0);
        HybridAllocatorStatistics {
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
            total_deallocations: self.total_deallocations.load(Ordering::Relaxed),
            small_allocations: self.small_allocations.load(Ordering::Relaxed),
            medium_allocations: self.medium_allocations.load(Ordering::Relaxed),
            large_allocations: self.large_allocations_count.load(Ordering::Relaxed),
            slab_statistics: slab_stats,
            buddy_statistics: buddy_stats,
            large_allocation_count: large_count,
            large_allocation_bytes: large_bytes,
            small_threshold: self.small_threshold,
            medium_threshold: self.medium_threshold,
        }
    }
    /// Get current total memory usage across all strategies
    pub fn current_usage(&self) -> usize {
        let slab_usage = self.slab_allocator.current_usage();
        let buddy_usage = self.buddy_allocator.current_usage();
        let large_usage = self
            .large_allocations
            .lock()
            .ok()
            .map(|l| l.values().sum())
            .unwrap_or(0);
        slab_usage + buddy_usage + large_usage
    }
}
/// Buddy allocator statistics
#[derive(Debug, Clone)]
pub struct BuddyAllocatorStatistics {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub total_splits: usize,
    pub total_merges: usize,
    pub current_allocated_bytes: usize,
    pub free_blocks: usize,
    pub allocated_blocks: usize,
    pub fragmentation_ratio: f64,
    pub total_memory: usize,
}
#[derive(Debug)]
pub struct BuddyAllocator {
    min_block_size: usize,
    max_block_size: usize,
    free_lists: Arc<Mutex<BTreeMap<usize, Vec<usize>>>>,
    allocated_blocks: Arc<Mutex<HashMap<usize, usize>>>,
    total_memory: usize,
    total_allocations: Arc<AtomicUsize>,
    total_deallocations: Arc<AtomicUsize>,
    total_splits: Arc<AtomicUsize>,
    total_merges: Arc<AtomicUsize>,
    current_allocated_bytes: Arc<AtomicUsize>,
}
impl BuddyAllocator {
    /// Create a new buddy allocator
    ///
    /// # Arguments
    /// * `total_memory` - Total memory to manage (will be rounded to power of 2)
    /// * `min_block_size` - Minimum allocation size (must be power of 2)
    ///
    /// # Example
    /// ```ignore
    /// let allocator = BuddyAllocator::new(1024 * 1024, 64); // 1MB total, 64B min
    /// ```
    pub fn new(total_memory: usize, min_block_size: usize) -> MemoryResult<Self> {
        if !min_block_size.is_power_of_two() {
            return Err(MemoryError::InvalidSize(min_block_size));
        }
        if !total_memory.is_power_of_two() {
            return Err(MemoryError::InvalidSize(total_memory));
        }
        if min_block_size > total_memory {
            return Err(MemoryError::InvalidSize(min_block_size));
        }
        let max_block_size = total_memory;
        let max_order =
            (max_block_size.trailing_zeros() - min_block_size.trailing_zeros()) as usize;
        let mut free_lists = BTreeMap::new();
        free_lists.insert(max_order, vec![0]);
        Ok(Self {
            min_block_size,
            max_block_size,
            free_lists: Arc::new(Mutex::new(free_lists)),
            allocated_blocks: Arc::new(Mutex::new(HashMap::new())),
            total_memory,
            total_allocations: Arc::new(AtomicUsize::new(0)),
            total_deallocations: Arc::new(AtomicUsize::new(0)),
            total_splits: Arc::new(AtomicUsize::new(0)),
            total_merges: Arc::new(AtomicUsize::new(0)),
            current_allocated_bytes: Arc::new(AtomicUsize::new(0)),
        })
    }
    /// Get the order (log2 of block size / min block size) for a size
    fn size_to_order(&self, size: usize) -> Option<usize> {
        if size == 0 || size > self.max_block_size {
            return None;
        }
        let required_size = size.max(self.min_block_size).next_power_of_two();
        Some((required_size.trailing_zeros() - self.min_block_size.trailing_zeros()) as usize)
    }
    /// Get the block size for an order
    fn order_to_size(&self, order: usize) -> usize {
        self.min_block_size << order
    }
    /// Calculate the buddy offset for a given block
    fn buddy_offset(&self, offset: usize, order: usize) -> usize {
        let block_size = self.order_to_size(order);
        offset ^ block_size
    }
    /// Split a block into two buddies
    fn split_block(
        &self,
        free_lists: &mut BTreeMap<usize, Vec<usize>>,
        order: usize,
    ) -> Option<usize> {
        if order == 0 {
            return None;
        }
        let max_order =
            (self.max_block_size.trailing_zeros() - self.min_block_size.trailing_zeros()) as usize;
        if order > max_order {
            return None;
        }
        if let Some(list) = free_lists.get_mut(&order) {
            if let Some(offset) = list.pop() {
                self.total_splits.fetch_add(1, Ordering::Relaxed);
                let lower_order = order - 1;
                let buddy1 = offset;
                let buddy2 = offset + self.order_to_size(lower_order);
                free_lists
                    .entry(lower_order)
                    .or_insert_with(Vec::new)
                    .extend(&[buddy1, buddy2]);
                return Some(lower_order);
            }
        }
        if let Some(_) = self.split_block(free_lists, order + 1) {
            self.split_block(free_lists, order)
        } else {
            None
        }
    }
    /// Allocate memory of the specified size
    ///
    /// # Arguments
    /// * `size` - Size to allocate in bytes
    ///
    /// # Returns
    /// Offset of the allocated block or error
    pub fn allocate(&self, size: usize) -> MemoryResult<usize> {
        let order = self
            .size_to_order(size)
            .ok_or(MemoryError::InvalidSize(size))?;
        let mut free_lists = self.free_lists.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other("Failed to lock free lists".to_string()))
        })?;
        let offset = if let Some(list) = free_lists.get_mut(&order) {
            let list: &mut Vec<usize> = list;
            list.pop()
        } else {
            None
        };
        let offset = if let Some(offset) = offset {
            offset
        } else {
            self.split_block(&mut free_lists, order + 1);
            free_lists
                .get_mut(&order)
                .and_then(|list: &mut Vec<usize>| list.pop())
                .ok_or(MemoryError::OutOfMemory(format!(
                    "Cannot allocate {} bytes",
                    size
                )))?
        };
        drop(free_lists);
        let mut allocated = self.allocated_blocks.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other(
                "Failed to lock allocated blocks".to_string(),
            ))
        })?;
        allocated.insert(offset, order);
        drop(allocated);
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        let block_size = self.order_to_size(order);
        self.current_allocated_bytes
            .fetch_add(block_size, Ordering::Relaxed);
        Ok(offset)
    }
    /// Deallocate a previously allocated block
    ///
    /// # Arguments
    /// * `offset` - Offset of the block to deallocate
    pub fn deallocate(&self, offset: usize) -> MemoryResult<()> {
        let mut allocated = self.allocated_blocks.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other(
                "Failed to lock allocated blocks".to_string(),
            ))
        })?;
        let order = allocated
            .remove(&offset)
            .ok_or(MemoryError::BufferNotFound(offset as u64))?;
        drop(allocated);
        let block_size = self.order_to_size(order);
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);
        self.current_allocated_bytes
            .fetch_sub(block_size, Ordering::Relaxed);
        self.coalesce(offset, order)?;
        Ok(())
    }
    /// Coalesce a free block with its buddy
    fn coalesce(&self, mut offset: usize, mut order: usize) -> MemoryResult<()> {
        let mut free_lists = self.free_lists.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other("Failed to lock free lists".to_string()))
        })?;
        let max_order =
            (self.max_block_size.trailing_zeros() - self.min_block_size.trailing_zeros()) as usize;
        loop {
            if order >= max_order {
                free_lists
                    .entry(order)
                    .or_insert_with(Vec::new)
                    .push(offset);
                break;
            }
            let buddy = self.buddy_offset(offset, order);
            let buddy_free = free_lists
                .get_mut(&order)
                .and_then(|list: &mut Vec<usize>| {
                    list.iter().position(|&o| o == buddy).map(|pos| {
                        list.swap_remove(pos);
                        true
                    })
                })
                .unwrap_or(false);
            if !buddy_free {
                free_lists
                    .entry(order)
                    .or_insert_with(Vec::new)
                    .push(offset);
                break;
            }
            self.total_merges.fetch_add(1, Ordering::Relaxed);
            offset = offset.min(buddy);
            order += 1;
        }
        Ok(())
    }
    /// Get allocator statistics
    pub fn statistics(&self) -> BuddyAllocatorStatistics {
        let free_lists = self.free_lists.lock().ok();
        let allocated_blocks = self.allocated_blocks.lock().ok();
        let free_blocks: usize = free_lists
            .as_ref()
            .map(|fl| fl.values().map(|v| v.len()).sum())
            .unwrap_or(0);
        let allocated_count = allocated_blocks.as_ref().map(|ab| ab.len()).unwrap_or(0);
        let current_allocated = self.current_allocated_bytes.load(Ordering::Relaxed);
        // Fragmentation is the ratio of unusable free space to total free space.
        // When nothing is allocated, all free space is contiguous (no fragmentation).
        // When something is allocated, fragmentation = 1 - (largest_free / total_free).
        // As a simplified metric: use ratio of free blocks to what could be coalesced.
        let fragmentation = if current_allocated == 0 || self.total_memory == 0 {
            0.0
        } else {
            // Count free blocks: if there's more than 1 free block, there is fragmentation
            let num_free_blocks = free_lists
                .as_ref()
                .map(|fl| fl.values().map(|v| v.len()).sum::<usize>())
                .unwrap_or(0);
            if num_free_blocks <= 1 {
                0.0
            } else {
                // Fragmentation proportional to how many free blocks exist vs ideal (1)
                1.0 - (1.0 / num_free_blocks as f64)
            }
        };
        BuddyAllocatorStatistics {
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
            total_deallocations: self.total_deallocations.load(Ordering::Relaxed),
            total_splits: self.total_splits.load(Ordering::Relaxed),
            total_merges: self.total_merges.load(Ordering::Relaxed),
            current_allocated_bytes: current_allocated,
            free_blocks,
            allocated_blocks: allocated_count,
            fragmentation_ratio: fragmentation,
            total_memory: self.total_memory,
        }
    }
    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_allocated_bytes.load(Ordering::Relaxed)
    }
    /// Get available memory
    pub fn available_memory(&self) -> usize {
        self.total_memory.saturating_sub(self.current_usage())
    }
}
/// Per-size statistics for slab allocator
#[derive(Debug, Clone)]
pub struct SlabSizeStatistics {
    pub slab_size: usize,
    pub total_slabs: usize,
    pub free_slabs: usize,
    pub allocated_slabs: usize,
    pub allocations: usize,
    pub deallocations: usize,
    pub utilization: f64,
}
/// Slab allocator statistics
#[derive(Debug, Clone)]
pub struct SlabAllocatorStatistics {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub current_allocated_bytes: usize,
    pub total_memory: usize,
    pub per_size_stats: Vec<SlabSizeStatistics>,
    pub fragmentation_ratio: f64,
}
