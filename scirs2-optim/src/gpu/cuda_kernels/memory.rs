//! CUDA memory management and allocation strategies
//!
//! This module provides sophisticated memory management for CUDA kernel operations,
//! including memory pool allocation, smart caching, unified memory management,
//! and memory usage optimization strategies.

use crate::gpu::cuda_kernels::config::*;
use scirs2_core::error::{Result, ScirsMlError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, DevicePtr, CudaSlice};

/// Memory allocation strategy for CUDA operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Direct allocation for each operation
    Direct,
    /// Pre-allocated memory pools
    PooledFixed { pool_size_mb: usize },
    /// Dynamic memory pools with growth
    PooledDynamic {
        initial_size_mb: usize,
        max_size_mb: usize,
        growth_factor: f32
    },
    /// Unified memory management (CUDA 6.0+)
    UnifiedMemory,
    /// Memory-mapped allocation
    MemoryMapped,
}

/// Memory pool configuration and management
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Allocation strategy
    pub strategy: AllocationStrategy,
    /// Alignment requirements in bytes
    pub alignment: usize,
    /// Enable memory recycling
    pub enable_recycling: bool,
    /// Maximum number of cached allocations
    pub max_cached_allocations: usize,
    /// Memory usage threshold for cleanup (percentage)
    pub cleanup_threshold: f32,
    /// Enable memory usage tracking
    pub enable_tracking: bool,
    /// Prefetch hint for unified memory
    pub prefetch_hint: bool,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            strategy: AllocationStrategy::PooledDynamic {
                initial_size_mb: 256,
                max_size_mb: 2048,
                growth_factor: 1.5,
            },
            alignment: 256, // CUDA alignment recommendation
            enable_recycling: true,
            max_cached_allocations: 1000,
            cleanup_threshold: 85.0,
            enable_tracking: true,
            prefetch_hint: true,
        }
    }
}

/// Memory allocation metadata
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Allocation size in bytes
    pub size: usize,
    /// Allocation timestamp
    pub timestamp: Instant,
    /// Last access timestamp
    pub last_access: Instant,
    /// Reference count
    pub ref_count: usize,
    /// Allocation alignment
    pub alignment: usize,
    /// Whether allocation is currently in use
    pub in_use: bool,
    /// Memory type (device, host, unified)
    pub memory_type: MemoryType,
    /// Optional tag for debugging
    pub tag: Option<String>,
}

/// Type of memory allocation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryType {
    /// Device memory (GPU VRAM)
    Device,
    /// Host memory (system RAM)
    Host,
    /// Pinned host memory (page-locked)
    HostPinned,
    /// Unified memory (CUDA managed)
    Unified,
    /// Memory-mapped
    Mapped,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total allocated memory in bytes
    pub total_allocated: usize,
    /// Currently used memory in bytes
    pub currently_used: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Number of active allocations
    pub active_allocations: usize,
    /// Number of cached allocations
    pub cached_allocations: usize,
    /// Total number of allocations made
    pub total_allocations: u64,
    /// Total number of deallocations
    pub total_deallocations: u64,
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
    /// Average allocation size
    pub avg_allocation_size: usize,
    /// Memory fragmentation percentage
    pub fragmentation: f32,
}

/// Managed memory allocation handle
pub struct ManagedAllocation<T> {
    /// Raw device pointer
    #[cfg(feature = "cuda")]
    device_ptr: Option<DevicePtr<T>>,
    /// Host pointer for unified/mapped memory
    host_ptr: Option<*mut T>,
    /// Allocation information
    info: AllocationInfo,
    /// Reference to the memory manager
    manager: Arc<CudaMemoryManager>,
    /// Unique allocation ID
    allocation_id: u64,
}

#[cfg(feature = "cuda")]
unsafe impl<T: Send> Send for ManagedAllocation<T> {}
#[cfg(feature = "cuda")]
unsafe impl<T: Sync> Sync for ManagedAllocation<T> {}

impl<T> Drop for ManagedAllocation<T> {
    fn drop(&mut self) {
        if let Err(e) = self.manager.deallocate(self.allocation_id) {
            eprintln!("Warning: Failed to deallocate memory: {}", e);
        }
    }
}

impl<T> ManagedAllocation<T> {
    /// Gets the device pointer (CUDA feature only)
    #[cfg(feature = "cuda")]
    pub fn device_ptr(&self) -> Option<&DevicePtr<T>> {
        self.device_ptr.as_ref()
    }

    /// Gets the host pointer
    pub fn host_ptr(&self) -> Option<*mut T> {
        self.host_ptr
    }

    /// Gets allocation information
    pub fn info(&self) -> &AllocationInfo {
        &self.info
    }

    /// Gets the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.info.size
    }

    /// Gets the element count
    pub fn element_count(&self) -> usize {
        self.info.size / std::mem::size_of::<T>()
    }

    /// Updates last access time
    pub fn touch(&mut self) {
        self.info.last_access = Instant::now();
        self.manager.update_access_time(self.allocation_id, self.info.last_access);
    }
}

/// CUDA memory manager with sophisticated allocation strategies
pub struct CudaMemoryManager {
    /// Memory pool configuration
    config: MemoryPoolConfig,
    /// Memory usage statistics
    stats: Arc<RwLock<MemoryStats>>,
    /// Active allocations tracking
    allocations: Arc<RwLock<HashMap<u64, AllocationInfo>>>,
    /// Memory pool for recycling
    memory_pool: Arc<Mutex<HashMap<usize, VecDeque<CachedAllocation>>>>,
    /// CUDA device reference
    #[cfg(feature = "cuda")]
    device: Arc<CudaDevice>,
    /// Next allocation ID
    next_id: Arc<Mutex<u64>>,
    /// Memory cleanup thread handle
    cleanup_handle: Option<std::thread::JoinHandle<()>>,
}

/// Cached allocation for memory pool
struct CachedAllocation {
    /// Allocation ID
    id: u64,
    /// Device pointer
    #[cfg(feature = "cuda")]
    device_ptr: Option<Box<dyn std::any::Any + Send + Sync>>,
    /// Host pointer
    host_ptr: Option<*mut u8>,
    /// Size in bytes
    size: usize,
    /// Cache timestamp
    cached_at: Instant,
    /// Memory type
    memory_type: MemoryType,
}

impl CudaMemoryManager {
    /// Creates a new CUDA memory manager
    pub fn new(config: MemoryPoolConfig) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let device = Arc::new(CudaDevice::new(0)?);

        let manager = Self {
            config,
            stats: Arc::new(RwLock::new(MemoryStats::default())),
            allocations: Arc::new(RwLock::new(HashMap::new())),
            memory_pool: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(feature = "cuda")]
            device,
            next_id: Arc::new(Mutex::new(1)),
            cleanup_handle: None,
        };

        Ok(manager)
    }

    /// Allocates managed memory with the specified configuration
    pub fn allocate<T>(&self, element_count: usize, memory_type: MemoryType, tag: Option<String>) -> Result<ManagedAllocation<T>> {
        let size = element_count * std::mem::size_of::<T>();
        let aligned_size = self.align_size(size);

        // Try to get from cache first
        if self.config.enable_recycling {
            if let Some(cached) = self.try_get_from_cache::<T>(aligned_size, &memory_type)? {
                return Ok(cached);
            }
        }

        // Allocate new memory
        let allocation = self.allocate_new::<T>(aligned_size, memory_type, tag)?;

        // Update statistics
        self.update_stats_allocation(aligned_size);

        Ok(allocation)
    }

    /// Attempts to get allocation from cache
    fn try_get_from_cache<T>(&self, size: usize, memory_type: &MemoryType) -> Result<Option<ManagedAllocation<T>>> {
        let mut pool = self.memory_pool.lock().unwrap();

        if let Some(cached_list) = pool.get_mut(&size) {
            while let Some(cached) = cached_list.pop_front() {
                if cached.memory_type == *memory_type {
                    // Found suitable cached allocation
                    let allocation_id = self.get_next_id();

                    let info = AllocationInfo {
                        size,
                        timestamp: Instant::now(),
                        last_access: Instant::now(),
                        ref_count: 1,
                        alignment: self.config.alignment,
                        in_use: true,
                        memory_type: memory_type.clone(),
                        tag: None,
                    };

                    // Update statistics
                    {
                        let mut stats = self.stats.write().unwrap();
                        stats.cache_hits += 1;
                        stats.currently_used += size;
                        stats.active_allocations += 1;
                    }

                    let allocation = ManagedAllocation {
                        #[cfg(feature = "cuda")]
                        device_ptr: match cached.device_ptr {
                            Some(ptr) => {
                                // This is a simplified approach - in reality, we'd need proper type casting
                                None
                            },
                            None => None,
                        },
                        host_ptr: if cached.host_ptr.is_some() {
                            Some(cached.host_ptr.unwrap() as *mut T)
                        } else {
                            None
                        },
                        info,
                        manager: Arc::new(self.clone()),
                        allocation_id,
                    };

                    return Ok(Some(allocation));
                }
            }
        }

        // Cache miss
        {
            let mut stats = self.stats.write().unwrap();
            stats.cache_misses += 1;
        }

        Ok(None)
    }

    /// Allocates new memory
    fn allocate_new<T>(&self, size: usize, memory_type: MemoryType, tag: Option<String>) -> Result<ManagedAllocation<T>> {
        let allocation_id = self.get_next_id();

        let info = AllocationInfo {
            size,
            timestamp: Instant::now(),
            last_access: Instant::now(),
            ref_count: 1,
            alignment: self.config.alignment,
            in_use: true,
            memory_type: memory_type.clone(),
            tag,
        };

        #[cfg(feature = "cuda")]
        let (device_ptr, host_ptr) = match memory_type {
            MemoryType::Device => {
                let ptr: DevicePtr<T> = self.device.alloc_zeros(size / std::mem::size_of::<T>())?;
                (Some(ptr), None)
            },
            MemoryType::Host => {
                // Allocate host memory
                let layout = std::alloc::Layout::from_size_align(size, self.config.alignment)
                    .map_err(|e| ScirsMlError::InvalidArgument(format!("Invalid layout: {}", e)))?;
                let ptr = unsafe { std::alloc::alloc(layout) as *mut T };
                if ptr.is_null() {
                    return Err(ScirsMlError::OutOfMemory("Failed to allocate host memory".into()));
                }
                (None, Some(ptr))
            },
            MemoryType::HostPinned => {
                // Allocate pinned host memory using CUDA
                let ptr: DevicePtr<T> = self.device.alloc_zeros(size / std::mem::size_of::<T>())?;
                (Some(ptr), None)
            },
            MemoryType::Unified => {
                // Allocate unified memory
                let ptr: DevicePtr<T> = self.device.alloc_zeros(size / std::mem::size_of::<T>())?;
                (Some(ptr), None)
            },
            MemoryType::Mapped => {
                // Memory-mapped allocation (simplified)
                let layout = std::alloc::Layout::from_size_align(size, self.config.alignment)
                    .map_err(|e| ScirsMlError::InvalidArgument(format!("Invalid layout: {}", e)))?;
                let ptr = unsafe { std::alloc::alloc(layout) as *mut T };
                if ptr.is_null() {
                    return Err(ScirsMlError::OutOfMemory("Failed to allocate mapped memory".into()));
                }
                (None, Some(ptr))
            },
        };

        #[cfg(not(feature = "cuda"))]
        let (device_ptr, host_ptr) = {
            // Fallback to host allocation when CUDA is not available
            let layout = std::alloc::Layout::from_size_align(size, self.config.alignment)
                .map_err(|e| ScirsMlError::InvalidArgument(format!("Invalid layout: {}", e)))?;
            let ptr = unsafe { std::alloc::alloc(layout) as *mut T };
            if ptr.is_null() {
                return Err(ScirsMlError::OutOfMemory("Failed to allocate memory".into()));
            }
            (None, Some(ptr))
        };

        // Track the allocation
        {
            let mut allocations = self.allocations.write().unwrap();
            allocations.insert(allocation_id, info.clone());
        }

        let allocation = ManagedAllocation {
            #[cfg(feature = "cuda")]
            device_ptr,
            host_ptr,
            info,
            manager: Arc::new(self.clone()),
            allocation_id,
        };

        Ok(allocation)
    }

    /// Deallocates managed memory
    pub fn deallocate(&self, allocation_id: u64) -> Result<()> {
        let info = {
            let mut allocations = self.allocations.write().unwrap();
            allocations.remove(&allocation_id)
        };

        if let Some(info) = info {
            if self.config.enable_recycling && info.size <= (self.config.max_cached_allocations * 1024 * 1024) {
                // Add to cache
                self.add_to_cache(allocation_id, info.size, info.memory_type)?;
            }

            // Update statistics
            self.update_stats_deallocation(info.size);
        }

        Ok(())
    }

    /// Adds allocation to cache for reuse
    fn add_to_cache(&self, allocation_id: u64, size: usize, memory_type: MemoryType) -> Result<()> {
        let mut pool = self.memory_pool.lock().unwrap();
        let cached_list = pool.entry(size).or_insert_with(VecDeque::new);

        // Limit cache size
        while cached_list.len() >= self.config.max_cached_allocations {
            if let Some(old) = cached_list.pop_front() {
                // Actually free the old allocation
                self.free_cached_allocation(old)?;
            }
        }

        let cached = CachedAllocation {
            id: allocation_id,
            #[cfg(feature = "cuda")]
            device_ptr: None, // Simplified - would store actual pointer
            host_ptr: None,   // Simplified - would store actual pointer
            size,
            cached_at: Instant::now(),
            memory_type,
        };

        cached_list.push_back(cached);

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.cached_allocations += 1;
            stats.currently_used -= size;
            stats.active_allocations -= 1;
        }

        Ok(())
    }

    /// Frees a cached allocation
    fn free_cached_allocation(&self, cached: CachedAllocation) -> Result<()> {
        // Implementation would actually free the memory based on type
        // This is simplified for the example

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.cached_allocations = stats.cached_allocations.saturating_sub(1);
            stats.total_deallocations += 1;
        }

        Ok(())
    }

    /// Updates statistics for new allocation
    fn update_stats_allocation(&self, size: usize) {
        let mut stats = self.stats.write().unwrap();
        stats.total_allocated += size;
        stats.currently_used += size;
        stats.peak_usage = stats.peak_usage.max(stats.currently_used);
        stats.active_allocations += 1;
        stats.total_allocations += 1;

        if stats.total_allocations > 0 {
            stats.avg_allocation_size = stats.total_allocated / stats.total_allocations as usize;
        }
    }

    /// Updates statistics for deallocation
    fn update_stats_deallocation(&self, size: usize) {
        let mut stats = self.stats.write().unwrap();
        stats.currently_used = stats.currently_used.saturating_sub(size);
        stats.active_allocations = stats.active_allocations.saturating_sub(1);
        stats.total_deallocations += 1;
    }

    /// Updates access time for an allocation
    pub fn update_access_time(&self, allocation_id: u64, access_time: Instant) {
        if let Ok(mut allocations) = self.allocations.write() {
            if let Some(info) = allocations.get_mut(&allocation_id) {
                info.last_access = access_time;
            }
        }
    }

    /// Gets the next unique allocation ID
    fn get_next_id(&self) -> u64 {
        let mut id = self.next_id.lock().unwrap();
        let current = *id;
        *id += 1;
        current
    }

    /// Aligns size according to configuration
    fn align_size(&self, size: usize) -> usize {
        let alignment = self.config.alignment;
        (size + alignment - 1) & !(alignment - 1)
    }

    /// Gets current memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        self.stats.read().unwrap().clone()
    }

    /// Performs memory cleanup based on usage thresholds
    pub fn cleanup(&self) -> Result<usize> {
        let stats = self.get_stats();
        let usage_percent = if stats.total_allocated > 0 {
            (stats.currently_used as f32 / stats.total_allocated as f32) * 100.0
        } else {
            0.0
        };

        if usage_percent > self.config.cleanup_threshold {
            return self.force_cleanup();
        }

        Ok(0)
    }

    /// Forces cleanup of old cached allocations
    fn force_cleanup(&self) -> Result<usize> {
        let mut pool = self.memory_pool.lock().unwrap();
        let mut cleaned = 0;
        let cutoff_time = Instant::now() - Duration::from_secs(300); // 5 minutes

        for cached_list in pool.values_mut() {
            while let Some(cached) = cached_list.front() {
                if cached.cached_at < cutoff_time {
                    let old = cached_list.pop_front().unwrap();
                    self.free_cached_allocation(old)?;
                    cleaned += 1;
                } else {
                    break; // Newer allocations, stop cleanup
                }
            }
        }

        Ok(cleaned)
    }

    /// Resets all statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write().unwrap();
        *stats = MemoryStats::default();
    }

    /// Generates memory usage report
    pub fn generate_report(&self) -> MemoryReport {
        let stats = self.get_stats();
        let allocations = self.allocations.read().unwrap();

        let mut allocation_summary = HashMap::new();
        for info in allocations.values() {
            let entry = allocation_summary.entry(info.memory_type.clone())
                .or_insert((0, 0usize));
            entry.0 += 1;
            entry.1 += info.size;
        }

        MemoryReport {
            stats,
            allocation_summary,
            cache_efficiency: if stats.cache_hits + stats.cache_misses > 0 {
                (stats.cache_hits as f32 / (stats.cache_hits + stats.cache_misses) as f32) * 100.0
            } else {
                0.0
            },
            fragmentation: self.calculate_fragmentation(),
            recommendations: self.generate_memory_recommendations(&stats),
        }
    }

    /// Calculates memory fragmentation percentage
    fn calculate_fragmentation(&self) -> f32 {
        // Simplified fragmentation calculation
        // Real implementation would analyze actual memory layout
        let pool = self.memory_pool.lock().unwrap();
        let total_cached = pool.values().map(|v| v.len()).sum::<usize>();

        if total_cached > 0 {
            let stats = self.get_stats();
            (total_cached as f32 / stats.active_allocations.max(1) as f32) * 10.0
        } else {
            0.0
        }
    }

    /// Generates memory optimization recommendations
    fn generate_memory_recommendations(&self, stats: &MemoryStats) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check cache efficiency
        let cache_hit_rate = if stats.cache_hits + stats.cache_misses > 0 {
            (stats.cache_hits as f32 / (stats.cache_hits + stats.cache_misses) as f32) * 100.0
        } else {
            0.0
        };

        if cache_hit_rate < 50.0 {
            recommendations.push("Low cache hit rate - consider adjusting allocation sizes or patterns".to_string());
        }

        // Check memory usage
        let usage_ratio = if stats.total_allocated > 0 {
            stats.currently_used as f32 / stats.total_allocated as f32
        } else {
            0.0
        };

        if usage_ratio > 0.9 {
            recommendations.push("High memory usage - consider increasing memory pool size".to_string());
        }

        // Check fragmentation
        if stats.fragmentation > 25.0 {
            recommendations.push("High memory fragmentation - consider defragmentation or larger allocation sizes".to_string());
        }

        // Check allocation patterns
        if stats.avg_allocation_size < 1024 {
            recommendations.push("Many small allocations detected - consider batching allocations".to_string());
        }

        recommendations
    }
}

impl Clone for CudaMemoryManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            stats: Arc::clone(&self.stats),
            allocations: Arc::clone(&self.allocations),
            memory_pool: Arc::clone(&self.memory_pool),
            #[cfg(feature = "cuda")]
            device: Arc::clone(&self.device),
            next_id: Arc::clone(&self.next_id),
            cleanup_handle: None, // New instance doesn't inherit cleanup thread
        }
    }
}

/// Memory usage and performance report
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Memory statistics
    pub stats: MemoryStats,
    /// Allocation summary by memory type (count, total_size)
    pub allocation_summary: HashMap<MemoryType, (u32, usize)>,
    /// Cache efficiency percentage
    pub cache_efficiency: f32,
    /// Memory fragmentation percentage
    pub fragmentation: f32,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
}

impl MemoryReport {
    /// Formats the report as a human-readable string
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== CUDA Memory Management Report ===\n\n");

        // Statistics
        report.push_str(&format!("Memory Statistics:\n"));
        report.push_str(&format!("  Total Allocated: {:.2} MB\n", self.stats.total_allocated as f64 / 1024.0 / 1024.0));
        report.push_str(&format!("  Currently Used: {:.2} MB\n", self.stats.currently_used as f64 / 1024.0 / 1024.0));
        report.push_str(&format!("  Peak Usage: {:.2} MB\n", self.stats.peak_usage as f64 / 1024.0 / 1024.0));
        report.push_str(&format!("  Active Allocations: {}\n", self.stats.active_allocations));
        report.push_str(&format!("  Cached Allocations: {}\n", self.stats.cached_allocations));
        report.push_str(&format!("  Cache Efficiency: {:.2}%\n", self.cache_efficiency));
        report.push_str(&format!("  Fragmentation: {:.2}%\n", self.fragmentation));
        report.push_str("\n");

        // Allocation summary
        if !self.allocation_summary.is_empty() {
            report.push_str("Allocation Summary by Type:\n");
            for (memory_type, (count, size)) in &self.allocation_summary {
                report.push_str(&format!("  {:?}: {} allocations, {:.2} MB\n",
                    memory_type, count, *size as f64 / 1024.0 / 1024.0));
            }
            report.push_str("\n");
        }

        // Recommendations
        if !self.recommendations.is_empty() {
            report.push_str("Recommendations:\n");
            for recommendation in &self.recommendations {
                report.push_str(&format!("  • {}\n", recommendation));
            }
        }

        report
    }
}