//! GPU memory management and allocation strategies
//!
//! This module provides efficient GPU memory management including memory pooling,
//! allocation strategies, and memory usage tracking.

use crate::error::{ClusteringError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// GPU memory allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryStrategy {
    /// Use unified memory (CUDA/HIP)
    Unified,
    /// Explicit host-device transfers
    Explicit,
    /// Memory pooling for reuse
    Pooled { pool_size_mb: usize },
    /// Zero-copy memory (if supported)
    ZeroCopy,
    /// Adaptive strategy based on data size
    Adaptive,
}

impl Default for MemoryStrategy {
    fn default() -> Self {
        MemoryStrategy::Adaptive
    }
}

/// GPU memory pool for efficient allocation reuse
#[derive(Debug)]
pub struct GpuMemoryManager {
    /// Memory pools indexed by size ranges
    pools: HashMap<usize, Vec<GpuMemoryBlock>>,
    /// Total allocated memory
    total_allocated: usize,
    /// Peak memory usage
    peak_usage: usize,
    /// Memory alignment requirement
    alignment: usize,
    /// Maximum pool size per allocation size
    max_pool_size: usize,
    /// Memory statistics
    stats: MemoryStats,
}

/// GPU memory block for pooling
#[derive(Debug, Clone)]
pub struct GpuMemoryBlock {
    /// Device pointer
    pub device_ptr: usize,
    /// Size in bytes
    pub size: usize,
    /// Whether currently in use
    pub in_use: bool,
    /// Allocation timestamp
    pub allocated_at: Instant,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total bytes allocated
    pub total_allocated: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Number of deallocations
    pub deallocation_count: usize,
    /// Number of pool hits
    pub pool_hits: usize,
    /// Number of pool misses
    pub pool_misses: usize,
}

impl GpuMemoryManager {
    /// Create new memory manager
    pub fn new(alignment: usize, max_pool_size: usize) -> Self {
        Self {
            pools: HashMap::new(),
            total_allocated: 0,
            peak_usage: 0,
            alignment,
            max_pool_size,
            stats: MemoryStats::default(),
        }
    }

    /// Allocate memory with efficient pooling
    pub fn allocate(&mut self, size: usize) -> Result<GpuMemoryBlock> {
        let aligned_size = (size + self.alignment - 1) & !(self.alignment - 1);
        let size_class = self.get_size_class(aligned_size);

        self.stats.allocation_count += 1;

        // Try to find existing block in pool
        if let Some(pool) = self.pools.get_mut(&size_class) {
            for block in pool.iter_mut() {
                if !block.in_use && block.size >= aligned_size {
                    block.in_use = true;
                    self.stats.pool_hits += 1;
                    return Ok(GpuMemoryBlock {
                        device_ptr: block.device_ptr,
                        size: block.size,
                        in_use: true,
                        allocated_at: Instant::now(),
                    });
                }
            }
        }

        // Pool miss - allocate new block
        self.stats.pool_misses += 1;
        let device_ptr = self.allocate_device_memory(aligned_size)?;
        self.total_allocated += aligned_size;
        self.peak_usage = self.peak_usage.max(self.total_allocated);
        self.stats.total_allocated = self.total_allocated;
        self.stats.peak_usage = self.peak_usage;

        Ok(GpuMemoryBlock {
            device_ptr,
            size: aligned_size,
            in_use: true,
            allocated_at: Instant::now(),
        })
    }

    /// Return memory block to pool
    pub fn deallocate(&mut self, mut block: GpuMemoryBlock) -> Result<()> {
        block.in_use = false;
        self.stats.deallocation_count += 1;

        let size_class = self.get_size_class(block.size);

        let pool = self.pools.entry(size_class).or_insert_with(Vec::new);
        if pool.len() < self.max_pool_size {
            pool.push(block);
        } else {
            // Pool full, actually free memory
            self.free_device_memory(block.device_ptr, block.size)?;
            self.total_allocated -= block.size;
            self.stats.total_allocated = self.total_allocated;
        }

        Ok(())
    }

    /// Clear all memory pools
    pub fn clear_pools(&mut self) -> Result<()> {
        for pool in self.pools.values() {
            for block in pool {
                if !block.in_use {
                    self.free_device_memory(block.device_ptr, block.size)?;
                    self.total_allocated -= block.size;
                }
            }
        }
        self.pools.clear();
        self.stats.total_allocated = self.total_allocated;
        Ok(())
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Get memory efficiency (pool hit rate)
    pub fn pool_efficiency(&self) -> f64 {
        if self.stats.allocation_count == 0 {
            0.0
        } else {
            self.stats.pool_hits as f64 / self.stats.allocation_count as f64
        }
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.total_allocated
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    /// Get size class for pooling
    fn get_size_class(&self, size: usize) -> usize {
        // Round up to next power of 2 for size classing
        if size == 0 {
            return 1;
        }
        let mut class = 1;
        while class < size {
            class <<= 1;
        }
        class
    }

    /// Platform-specific device memory allocation (stub)
    fn allocate_device_memory(&self, size: usize) -> Result<usize> {
        // This is a stub implementation
        // Real implementation would call CUDA malloc, OpenCL buffer creation, etc.

        if size == 0 {
            return Err(ClusteringError::InvalidInput(
                "Cannot allocate zero bytes".to_string(),
            ));
        }

        if size > 16 * 1024 * 1024 * 1024 {
            return Err(ClusteringError::InvalidInput(
                "Allocation too large".to_string(),
            ));
        }

        // Simulate device memory allocation by returning a fake pointer
        Ok(0x1000_0000 + size) // Fake device pointer
    }

    /// Platform-specific device memory deallocation (stub)
    fn free_device_memory(&self, _device_ptr: usize, _size: usize) -> Result<()> {
        // This is a stub implementation
        // Real implementation would call CUDA free, OpenCL release, etc.
        Ok(())
    }
}

impl MemoryStats {
    /// Get allocation efficiency (deallocations / allocations)
    pub fn allocation_efficiency(&self) -> f64 {
        if self.allocation_count == 0 {
            1.0
        } else {
            self.deallocation_count as f64 / self.allocation_count as f64
        }
    }

    /// Get average allocation size
    pub fn average_allocation_size(&self) -> f64 {
        if self.allocation_count == 0 {
            0.0
        } else {
            self.total_allocated as f64 / self.allocation_count as f64
        }
    }

    /// Check if there are potential memory leaks
    pub fn has_potential_leaks(&self) -> bool {
        self.allocation_count > self.deallocation_count // Any unmatched allocation is a potential leak
    }
}

/// Memory transfer operations
#[derive(Debug, Clone)]
pub enum MemoryTransfer {
    /// Host to device transfer
    HostToDevice {
        /// Source host pointer
        host_ptr: *const u8,
        /// Destination device pointer
        device_ptr: usize,
        /// Size in bytes
        size: usize,
    },
    /// Device to host transfer
    DeviceToHost {
        /// Source device pointer
        device_ptr: usize,
        /// Destination host pointer
        host_ptr: *mut u8,
        /// Size in bytes
        size: usize,
    },
    /// Device to device transfer
    DeviceToDevice {
        /// Source device pointer
        src_device_ptr: usize,
        /// Destination device pointer
        dst_device_ptr: usize,
        /// Size in bytes
        size: usize,
    },
}

impl MemoryTransfer {
    /// Get transfer size in bytes
    pub fn size(&self) -> usize {
        match self {
            MemoryTransfer::HostToDevice { size, .. } => *size,
            MemoryTransfer::DeviceToHost { size, .. } => *size,
            MemoryTransfer::DeviceToDevice { size, .. } => *size,
        }
    }

    /// Execute the memory transfer (stub)
    pub fn execute(&self) -> Result<()> {
        // This is a stub implementation
        // Real implementation would call platform-specific transfer functions
        match self {
            MemoryTransfer::HostToDevice { .. } => {
                // Simulate host-to-device transfer
                Ok(())
            }
            MemoryTransfer::DeviceToHost { .. } => {
                // Simulate device-to-host transfer
                Ok(())
            }
            MemoryTransfer::DeviceToDevice { .. } => {
                // Simulate device-to-device transfer
                Ok(())
            }
        }
    }
}

/// Memory bandwidth monitoring
#[derive(Debug, Clone, Default)]
pub struct BandwidthMonitor {
    /// Total bytes transferred
    pub total_transferred: usize,
    /// Number of transfers
    pub transfer_count: usize,
    /// Total transfer time in microseconds
    pub total_time_us: u64,
}

impl BandwidthMonitor {
    /// Record a memory transfer
    pub fn record_transfer(&mut self, size: usize, duration_us: u64) {
        self.total_transferred += size;
        self.transfer_count += 1;
        self.total_time_us += duration_us;
    }

    /// Get average bandwidth in GB/s
    pub fn average_bandwidth_gbps(&self) -> f64 {
        if self.total_time_us == 0 {
            0.0
        } else {
            let total_gb = self.total_transferred as f64 / (1024.0 * 1024.0 * 1024.0);
            let total_seconds = self.total_time_us as f64 / 1_000_000.0;
            total_gb / total_seconds
        }
    }

    /// Get average transfer size
    pub fn average_transfer_size(&self) -> f64 {
        if self.transfer_count == 0 {
            0.0
        } else {
            self.total_transferred as f64 / self.transfer_count as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager_creation() {
        let manager = GpuMemoryManager::new(256, 10);
        assert_eq!(manager.alignment, 256);
        assert_eq!(manager.max_pool_size, 10);
        assert_eq!(manager.current_usage(), 0);
    }

    #[test]
    fn test_memory_allocation() {
        let mut manager = GpuMemoryManager::new(256, 10);

        let block = manager.allocate(1024).unwrap();
        assert!(block.size >= 1024);
        assert!(block.in_use);

        assert_eq!(manager.get_stats().allocation_count, 1);
        assert_eq!(manager.get_stats().pool_misses, 1);
    }

    #[test]
    fn test_memory_pooling() {
        let mut manager = GpuMemoryManager::new(256, 10);

        // Allocate and deallocate
        let block = manager.allocate(1024).unwrap();
        manager.deallocate(block).unwrap();

        // Second allocation should hit pool
        let _block2 = manager.allocate(1024).unwrap();
        assert!(manager.get_stats().pool_hits > 0);
    }

    #[test]
    fn test_memory_stats() {
        let stats = MemoryStats {
            allocation_count: 10,
            deallocation_count: 8,
            total_allocated: 1024,
            pool_hits: 5,
            ..Default::default()
        };

        assert_eq!(stats.allocation_efficiency(), 0.8);
        assert_eq!(stats.average_allocation_size(), 102.4);
        assert!(stats.has_potential_leaks());
    }

    #[test]
    fn test_bandwidth_monitor() {
        let mut monitor = BandwidthMonitor::default();

        // Record a 1GB transfer in 1 second
        monitor.record_transfer(1024 * 1024 * 1024, 1_000_000);

        assert_eq!(monitor.average_bandwidth_gbps(), 1.0);
        assert_eq!(monitor.average_transfer_size(), 1024.0 * 1024.0 * 1024.0);
    }

    #[test]
    fn test_memory_transfer() {
        let transfer = MemoryTransfer::HostToDevice {
            host_ptr: std::ptr::null(),
            device_ptr: 0x1000,
            size: 1024,
        };

        assert_eq!(transfer.size(), 1024);
        assert!(transfer.execute().is_ok());
    }
}
