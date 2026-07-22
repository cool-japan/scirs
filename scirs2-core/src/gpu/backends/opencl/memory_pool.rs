//! OpenCL device-memory pool.
//!
//! This is the advanced size-class bucketing pool that backs
//! [`super::OpenCLContext`]. The algorithm is unchanged from the previous
//! `opencl3`-based implementation; only the buffer type was swapped to the
//! runtime-loader's [`ClBuffer`] and the buffer-size query now uses the
//! tracked size instead of `Buffer::size()`.

use std::collections::HashMap;
use std::time::Duration;

use super::ffi::ClBuffer;

/// Advanced OpenCL memory pool with advanced-optimization for efficient buffer management
///
/// Features:
/// - Size-class bucketing for O(1) allocation/deallocation
/// - Memory pressure monitoring and adaptive allocation
/// - Memory defragmentation and compaction
/// - Statistics tracking for optimization insights
/// - Cache-aware allocation patterns
pub(super) struct OpenCLMemoryPool {
    available_buffers: HashMap<usize, Vec<ClBuffer>>,

    // Advanced memory management features
    size_classes: Vec<usize>,
    allocation_stats: HashMap<usize, AllocationStats>,
    memory_pressure_threshold: f64,
    total_size: usize,
    used_size: usize,
    peak_used_size: usize,
    fragmentation_ratio: f64,

    // Cache-aware allocation tracking
    recent_allocations: std::collections::VecDeque<(usize, std::time::Instant)>,
    hot_sizes: std::collections::BTreeSet<usize>,
}

/// Statistics for tracking allocation patterns
#[derive(Debug, Clone)]
pub struct AllocationStats {
    total_allocations: u64,
    total_deallocations: u64,
    total_bytes_allocated: u64,
    #[allow(dead_code)]
    average_lifetime: Duration,
    peak_concurrent_allocations: u64,
    current_allocations: u64,
}

/// Pool statistics for monitoring and optimization
#[derive(Debug, Clone)]
pub struct PoolStatistics {
    pub total_size: usize,
    pub used_size: usize,
    pub peak_used_size: usize,
    pub fragmentation_ratio: f64,
    pub available_buffer_count: usize,
    pub hot_size_classes: usize,
    pub allocation_stats: HashMap<usize, AllocationStats>,
}

impl OpenCLMemoryPool {
    pub(super) fn new(totalsize: usize) -> Self {
        // Define power-of-2 size classes for optimal bucketing
        let size_classes = (0..32)
            .map(|i| 1usize << i)
            .filter(|&size| size <= totalsize)
            .collect();

        Self {
            available_buffers: HashMap::new(),
            size_classes,
            allocation_stats: HashMap::new(),
            memory_pressure_threshold: 0.85, // Trigger cleanup at 85% usage
            total_size: totalsize,
            used_size: 0,
            peak_used_size: 0,
            fragmentation_ratio: 0.0,
            recent_allocations: std::collections::VecDeque::with_capacity(1000),
            hot_sizes: std::collections::BTreeSet::new(),
        }
    }

    /// Get the appropriate size class for a requested size
    fn get_size_class(&self, requestedsize: usize) -> usize {
        self.size_classes
            .iter()
            .find(|&&class_size| class_size >= requestedsize)
            .copied()
            .unwrap_or_else(|| {
                // For sizes larger than our classes, round up to the nearest 4KB boundary
                ((requestedsize + 4095) / 4096) * 4096
            })
    }

    /// Update memory pressure and trigger cleanup if needed
    fn update_memory_pressure(&mut self) {
        let pressure = self.used_size as f64 / self.total_size as f64;

        if pressure > self.memory_pressure_threshold {
            self.cleanup_cold_buffers();
            self.defragment_if_needed();
        }

        // Update fragmentation ratio
        let total_available = self
            .available_buffers
            .values()
            .map(|buffers| buffers.len())
            .sum::<usize>();

        if total_available > 0 {
            self.fragmentation_ratio =
                1.0 - (self.used_size as f64 / (self.used_size + total_available * 1024) as f64);
        }
    }

    /// Remove buffers that haven't been used recently
    fn cleanup_cold_buffers(&mut self) {
        let now = std::time::Instant::now();
        let cold_threshold = Duration::from_secs(30); // 30 seconds

        // Clean up old allocation tracking
        while let Some(&(_, timestamp)) = self.recent_allocations.front() {
            if now.duration_since(timestamp) > cold_threshold {
                self.recent_allocations.pop_front();
            } else {
                break;
            }
        }

        // Update hot sizes based on recent allocations
        self.hot_sizes.clear();
        for &(size, _) in &self.recent_allocations {
            self.hot_sizes.insert(self.get_size_class(size));
        }

        // Remove buffers for size classes that are not hot
        for (size_class, buffers) in &mut self.available_buffers {
            if !self.hot_sizes.contains(size_class) && buffers.len() > 2 {
                // Keep only 2 buffers for cold size classes
                let excess = buffers.len() - 2;
                for _ in 0..excess {
                    buffers.pop();
                }
            }
        }
    }

    /// Defragment memory if fragmentation ratio is too high
    fn defragment_if_needed(&mut self) {
        if self.fragmentation_ratio > 0.3 {
            // High fragmentation - perform compaction
            for buffers in self.available_buffers.values_mut() {
                // Sort buffers by some criteria if possible
                // For now, just shuffle to redistribute
                if buffers.len() > 4 {
                    buffers.truncate(buffers.len() / 2);
                }
            }
        }
    }

    /// Update allocation statistics
    fn update_allocation_stats(&mut self, sizeclass: usize, allocated: bool) {
        let stats = self
            .allocation_stats
            .entry(sizeclass)
            .or_insert_with(|| AllocationStats {
                total_allocations: 0,
                total_deallocations: 0,
                total_bytes_allocated: 0,
                average_lifetime: Duration::new(0, 0),
                peak_concurrent_allocations: 0,
                current_allocations: 0,
            });

        if allocated {
            stats.total_allocations += 1;
            stats.total_bytes_allocated += sizeclass as u64;
            stats.current_allocations += 1;
            stats.peak_concurrent_allocations = stats
                .peak_concurrent_allocations
                .max(stats.current_allocations);
        } else {
            stats.total_deallocations += 1;
            stats.current_allocations = stats.current_allocations.saturating_sub(1);
        }
    }

    /// Get memory pool statistics for monitoring
    #[allow(dead_code)]
    pub(super) fn get_pool_statistics(&self) -> PoolStatistics {
        PoolStatistics {
            total_size: self.total_size,
            used_size: self.used_size,
            peak_used_size: self.peak_used_size,
            fragmentation_ratio: self.fragmentation_ratio,
            available_buffer_count: self.available_buffers.values().map(|v| v.len()).sum(),
            hot_size_classes: self.hot_sizes.len(),
            allocation_stats: self.allocation_stats.clone(),
        }
    }

    pub(super) fn allocate(&mut self, size: usize) -> Option<ClBuffer> {
        let size_class = self.get_size_class(size);

        // Try to find a suitable buffer in the pool
        if let Some(buffers) = self.available_buffers.get_mut(&size_class) {
            if let Some(buffer) = buffers.pop() {
                self.used_size += size_class;
                self.peak_used_size = self.peak_used_size.max(self.used_size);

                // Track this allocation
                self.recent_allocations
                    .push_back((size, std::time::Instant::now()));
                if self.recent_allocations.len() > 1000 {
                    self.recent_allocations.pop_front();
                }
                self.hot_sizes.insert(size_class);

                // Update statistics
                self.update_allocation_stats(size_class, true);
                self.update_memory_pressure();

                return Some(buffer);
            }
        }
        None
    }

    pub(super) fn deallocate(&mut self, buffer: ClBuffer) {
        // Return buffer to pool. The tracked byte size replaces the previous
        // `Buffer::size().unwrap_or(0)` device query; buffers that are not
        // returned here are released through `ClBuffer`'s `Drop`.
        let size = buffer.size;
        self.available_buffers
            .entry(size)
            .or_insert_with(Vec::new)
            .push(buffer);
        self.used_size = self.used_size.saturating_sub(size);
    }

    #[allow(dead_code)]
    pub(super) fn get_memory_usage(&self) -> (usize, usize) {
        (self.used_size, self.total_size)
    }
}
