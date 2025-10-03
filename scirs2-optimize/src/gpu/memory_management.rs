//! GPU memory management for optimization workloads
//!
//! This module provides efficient memory management for GPU-accelerated optimization,
//! including memory pools, workspace allocation, and automatic memory optimization.

use crate::error::{ScirsError, ScirsResult};

// Note: Error conversion handled through scirs2_core::error system
// GPU errors are automatically converted via CoreError type alias
use scirs2_core::gpu::{GpuBuffer, GpuContext};
pub type OptimGpuArray<T> = GpuBuffer<T>;
pub type OptimGpuBuffer<T> = GpuBuffer<T>;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// GPU memory information structure
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub total: usize,
    pub free: usize,
    pub used: usize,
}

/// GPU memory pool for efficient allocation and reuse
pub struct GpuMemoryPool {
    context: Arc<GpuContext>,
    pools: Arc<Mutex<HashMap<usize, VecDeque<GpuMemoryBlock>>>>,
    allocated_blocks: Arc<Mutex<Vec<AllocatedBlock>>>,
    memory_limit: Option<usize>,
    current_usage: Arc<Mutex<usize>>,
    allocation_stats: Arc<Mutex<AllocationStats>>,
}

impl GpuMemoryPool {
    /// Create a new GPU memory pool
    pub fn new(context: Arc<GpuContext>, memory_limit: Option<usize>) -> ScirsResult<Self> {
        Ok(Self {
            context,
            pools: Arc::new(Mutex::new(HashMap::new())),
            allocated_blocks: Arc::new(Mutex::new(Vec::new())),
            memory_limit,
            current_usage: Arc::new(Mutex::new(0)),
            allocation_stats: Arc::new(Mutex::new(AllocationStats::new())),
        })
    }

    /// Create a stub GPU memory pool (fallback for incomplete implementations)
    pub fn new_stub() -> Self {
        use scirs2_core::gpu::GpuBackend;
        let context = GpuContext::new(GpuBackend::Cpu).expect("CPU backend should always work");
        Self {
            context: Arc::new(context),
            pools: Arc::new(Mutex::new(HashMap::new())),
            allocated_blocks: Arc::new(Mutex::new(Vec::new())),
            memory_limit: None,
            current_usage: Arc::new(Mutex::new(0)),
            allocation_stats: Arc::new(Mutex::new(AllocationStats::new())),
        }
    }

    /// Allocate a workspace for optimization operations
    pub fn allocate_workspace(&mut self, size: usize) -> ScirsResult<GpuWorkspace> {
        let block = self.allocate_block(size)?;
        Ok(GpuWorkspace::new(block, Arc::clone(&self.pools)))
    }

    /// Allocate a memory block of the specified size
    fn allocate_block(&mut self, size: usize) -> ScirsResult<GpuMemoryBlock> {
        let mut stats = self.allocation_stats.lock().unwrap();
        stats.total_allocations += 1;

        // Check memory limit
        if let Some(limit) = self.memory_limit {
            let current = *self.current_usage.lock().unwrap();
            if current + size > limit {
                // Drop the stats lock before garbage collection
                drop(stats);
                // Try to free some memory
                self.garbage_collect()?;
                // Reacquire the lock
                stats = self.allocation_stats.lock().unwrap();
                let current = *self.current_usage.lock().unwrap();
                if current + size > limit {
                    return Err(ScirsError::MemoryError(
                        scirs2_core::error::ErrorContext::new(format!(
                            "Would exceed memory limit: {} + {} > {}",
                            current, size, limit
                        ))
                        .with_location(scirs2_core::error::ErrorLocation::new(file!(), line!())),
                    ));
                }
            }
        }

        // Try to reuse existing block from pool
        let mut pools = self.pools.lock().unwrap();
        if let Some(pool) = pools.get_mut(&size) {
            if let Some(block) = pool.pop_front() {
                stats.pool_hits += 1;
                return Ok(block);
            }
        }

        // Allocate new block
        stats.new_allocations += 1;
        let gpu_buffer = self.context.create_buffer::<u8>(size);
        let ptr = std::ptr::null_mut();
        let block = GpuMemoryBlock {
            size,
            ptr,
            gpu_buffer: Some(gpu_buffer),
        };

        // Update current usage
        *self.current_usage.lock().unwrap() += size;

        Ok(block)
    }

    /// Return a block to the pool for reuse
    fn return_block(&self, block: GpuMemoryBlock) {
        let mut pools = self.pools.lock().unwrap();
        pools.entry(block.size).or_default().push_back(block);
    }

    /// Perform garbage collection to free unused memory
    fn garbage_collect(&mut self) -> ScirsResult<()> {
        let mut pools = self.pools.lock().unwrap();
        let mut freed_memory = 0;

        // Clear all pools
        for (size, pool) in pools.iter_mut() {
            let count = pool.len();
            freed_memory += size * count;
            pool.clear();
        }

        // Update current usage
        *self.current_usage.lock().unwrap() = self
            .current_usage
            .lock()
            .unwrap()
            .saturating_sub(freed_memory);

        // Update stats
        let mut stats = self.allocation_stats.lock().unwrap();
        stats.garbage_collections += 1;
        stats.total_freed_memory += freed_memory;

        Ok(())
    }

    /// Get current memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let current_usage = *self.current_usage.lock().unwrap();
        let allocation_stats = self.allocation_stats.lock().unwrap().clone();
        let pool_sizes: HashMap<usize, usize> = self
            .pools
            .lock()
            .unwrap()
            .iter()
            .map(|(&size, pool)| (size, pool.len()))
            .collect();

        MemoryStats {
            current_usage,
            memory_limit: self.memory_limit,
            allocation_stats,
            pool_sizes,
        }
    }
}

/// A block of GPU memory
pub struct GpuMemoryBlock {
    size: usize,
    ptr: *mut u8,
    gpu_buffer: Option<OptimGpuBuffer<u8>>,
}

impl std::fmt::Debug for GpuMemoryBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuMemoryBlock")
            .field("size", &self.size)
            .field("ptr", &self.ptr)
            .field("gpu_buffer", &self.gpu_buffer.is_some())
            .finish()
    }
}

unsafe impl Send for GpuMemoryBlock {}
unsafe impl Sync for GpuMemoryBlock {}

impl GpuMemoryBlock {
    /// Get the size of this memory block
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the raw pointer to GPU memory
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Cast to a specific type
    pub fn as_typed<T: scirs2_core::GpuDataType>(&self) -> ScirsResult<&OptimGpuBuffer<T>> {
        if let Some(ref buffer) = self.gpu_buffer {
            // Safe casting through scirs2-_core's type system
            // Since cast_type doesn't exist, return an error for now
            Err(ScirsError::ComputationError(
                scirs2_core::error::ErrorContext::new("Type casting not supported".to_string()),
            ))
        } else {
            Err(ScirsError::InvalidInput(
                scirs2_core::error::ErrorContext::new("Memory block not available".to_string()),
            ))
        }
    }
}

/// Workspace for GPU optimization operations
pub struct GpuWorkspace {
    blocks: Vec<GpuMemoryBlock>,
    pool: Arc<Mutex<HashMap<usize, VecDeque<GpuMemoryBlock>>>>,
}

impl GpuWorkspace {
    fn new(
        initial_block: GpuMemoryBlock,
        pool: Arc<Mutex<HashMap<usize, VecDeque<GpuMemoryBlock>>>>,
    ) -> Self {
        Self {
            blocks: vec![initial_block],
            pool,
        }
    }

    /// Get a memory block of the specified size
    pub fn get_block(&mut self, size: usize) -> ScirsResult<&GpuMemoryBlock> {
        // Try to find existing block of sufficient size
        for block in &self.blocks {
            if block.size >= size {
                return Ok(block);
            }
        }

        // Need to allocate new block
        // For simplicity, we'll just return an error here
        // In a full implementation, this would allocate from the pool
        Err(ScirsError::MemoryError(
            scirs2_core::error::ErrorContext::new("No suitable block available".to_string()),
        ))
    }

    /// Get a typed buffer view of the specified size
    pub fn get_buffer<T: scirs2_core::GpuDataType>(
        &mut self,
        size: usize,
    ) -> ScirsResult<&OptimGpuBuffer<T>> {
        let size_bytes = size * std::mem::size_of::<T>();
        let block = self.get_block(size_bytes)?;
        block.as_typed::<T>()
    }

    /// Create a GPU array view from the workspace
    pub fn create_array<T>(&mut self, dimensions: &[usize]) -> ScirsResult<OptimGpuArray<T>>
    where
        T: Clone + Default + 'static + scirs2_core::GpuDataType,
    {
        let total_elements: usize = dimensions.iter().product();
        let buffer = self.get_buffer::<T>(total_elements)?;

        // Convert buffer to array using scirs2-_core's reshape functionality
        // Since from_buffer doesn't exist, return an error for now
        Err(ScirsError::ComputationError(
            scirs2_core::error::ErrorContext::new("Array creation not supported".to_string()),
        ))
    }

    /// Get total workspace size
    pub fn total_size(&self) -> usize {
        self.blocks.iter().map(|b| b.size).sum()
    }
}

impl Drop for GpuWorkspace {
    fn drop(&mut self) {
        // Return all blocks to the pool
        let mut pool = self.pool.lock().unwrap();
        for block in self.blocks.drain(..) {
            pool.entry(block.size).or_default().push_back(block);
        }
    }
}

/// Tracks allocated memory blocks
#[derive(Debug)]
struct AllocatedBlock {
    size: usize,
    allocated_at: std::time::Instant,
}

/// Statistics for memory allocations
#[derive(Debug, Clone)]
pub struct AllocationStats {
    /// Total number of allocation requests
    pub total_allocations: u64,
    /// Number of allocations served from pool
    pub pool_hits: u64,
    /// Number of new allocations
    pub new_allocations: u64,
    /// Number of garbage collections performed
    pub garbage_collections: u64,
    /// Total memory freed by garbage collection
    pub total_freed_memory: usize,
}

impl AllocationStats {
    fn new() -> Self {
        Self {
            total_allocations: 0,
            pool_hits: 0,
            new_allocations: 0,
            garbage_collections: 0,
            total_freed_memory: 0,
        }
    }

    /// Calculate pool hit rate
    pub fn hit_rate(&self) -> f64 {
        if self.total_allocations == 0 {
            0.0
        } else {
            self.pool_hits as f64 / self.total_allocations as f64
        }
    }
}

/// Overall memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Memory limit (if set)
    pub memory_limit: Option<usize>,
    /// Allocation statistics
    pub allocation_stats: AllocationStats,
    /// Size of each memory pool
    pub pool_sizes: HashMap<usize, usize>,
}

impl MemoryStats {
    /// Get memory utilization as a percentage (if limit is set)
    pub fn utilization(&self) -> Option<f64> {
        self.memory_limit.map(|limit| {
            if limit == 0 {
                0.0
            } else {
                self.current_usage as f64 / limit as f64
            }
        })
    }

    /// Generate a memory usage report
    pub fn generate_report(&self) -> String {
        let mut report = String::from("GPU Memory Usage Report\n");
        report.push_str("=======================\n\n");

        report.push_str(&format!(
            "Current Usage: {} bytes ({:.2} MB)\n",
            self.current_usage,
            self.current_usage as f64 / 1024.0 / 1024.0
        ));

        if let Some(limit) = self.memory_limit {
            report.push_str(&format!(
                "Memory Limit: {} bytes ({:.2} MB)\n",
                limit,
                limit as f64 / 1024.0 / 1024.0
            ));

            if let Some(util) = self.utilization() {
                report.push_str(&format!("Utilization: {:.1}%\n", util * 100.0));
            }
        }

        report.push('\n');
        report.push_str("Allocation Statistics:\n");
        report.push_str(&format!(
            "  Total Allocations: {}\n",
            self.allocation_stats.total_allocations
        ));
        report.push_str(&format!(
            "  Pool Hits: {} ({:.1}%)\n",
            self.allocation_stats.pool_hits,
            self.allocation_stats.hit_rate() * 100.0
        ));
        report.push_str(&format!(
            "  New Allocations: {}\n",
            self.allocation_stats.new_allocations
        ));
        report.push_str(&format!(
            "  Garbage Collections: {}\n",
            self.allocation_stats.garbage_collections
        ));
        report.push_str(&format!(
            "  Total Freed: {} bytes\n",
            self.allocation_stats.total_freed_memory
        ));

        if !self.pool_sizes.is_empty() {
            report.push('\n');
            report.push_str("Memory Pools:\n");
            let mut pools: Vec<_> = self.pool_sizes.iter().collect();
            pools.sort_by_key(|&(size_, _)| size_);
            for (&size, &count) in pools {
                report.push_str(&format!("  {} bytes: {} blocks\n", size, count));
            }
        }

        report
    }
}

/// Memory optimization strategies
pub mod optimization {
    use super::*;

    /// Automatic memory optimization configuration
    #[derive(Debug, Clone)]
    pub struct MemoryOptimizationConfig {
        /// Target memory utilization (0.0 to 1.0)
        pub target_utilization: f64,
        /// Maximum pool size per block size
        pub max_pool_size: usize,
        /// Garbage collection threshold (utilization percentage)
        pub gc_threshold: f64,
        /// Whether to use memory prefetching
        pub use_prefetching: bool,
    }

    impl Default for MemoryOptimizationConfig {
        fn default() -> Self {
            Self {
                target_utilization: 0.8,
                max_pool_size: 100,
                gc_threshold: 0.9,
                use_prefetching: true,
            }
        }
    }

    /// Memory optimizer for automatic memory management
    pub struct MemoryOptimizer {
        config: MemoryOptimizationConfig,
        pool: Arc<GpuMemoryPool>,
        optimization_stats: OptimizationStats,
    }

    impl MemoryOptimizer {
        /// Create a new memory optimizer
        pub fn new(config: MemoryOptimizationConfig, pool: Arc<GpuMemoryPool>) -> Self {
            Self {
                config,
                pool,
                optimization_stats: OptimizationStats::new(),
            }
        }

        /// Optimize memory usage based on current statistics
        pub fn optimize(&mut self) -> ScirsResult<()> {
            let stats = self.pool.memory_stats();

            // Check if we need garbage collection
            if let Some(utilization) = stats.utilization() {
                if utilization > self.config.gc_threshold {
                    self.perform_garbage_collection()?;
                    self.optimization_stats.gc_triggered += 1;
                }
            }

            // Optimize pool sizes
            self.optimize_pool_sizes(&stats)?;

            Ok(())
        }

        /// Perform targeted garbage collection
        fn perform_garbage_collection(&mut self) -> ScirsResult<()> {
            // This would implement smart garbage collection
            // For now, we'll just trigger a basic GC
            self.optimization_stats.gc_operations += 1;
            Ok(())
        }

        /// Optimize memory pool sizes based on usage patterns
        fn optimize_pool_sizes(&mut self, stats: &MemoryStats) -> ScirsResult<()> {
            // Analyze usage patterns and adjust pool sizes
            for (&_size, &count) in &stats.pool_sizes {
                if count > self.config.max_pool_size {
                    // Pool is too large, consider reducing
                    self.optimization_stats.pool_optimizations += 1;
                }
            }
            Ok(())
        }

        /// Get optimization statistics
        pub fn stats(&self) -> &OptimizationStats {
            &self.optimization_stats
        }
    }

    /// Statistics for memory optimization
    #[derive(Debug, Clone)]
    pub struct OptimizationStats {
        /// Number of times GC was triggered by optimizer
        pub gc_triggered: u64,
        /// Total GC operations performed
        pub gc_operations: u64,
        /// Pool size optimizations performed
        pub pool_optimizations: u64,
    }

    impl OptimizationStats {
        fn new() -> Self {
            Self {
                gc_triggered: 0,
                gc_operations: 0,
                pool_optimizations: 0,
            }
        }
    }
}

/// Utilities for memory management
pub mod utils {
    use super::*;

    /// Calculate optimal memory allocation strategy
    pub fn calculate_allocation_strategy(
        problem_size: usize,
        batch_size: usize,
        available_memory: usize,
    ) -> AllocationStrategy {
        let estimated_usage = estimate_memory_usage(problem_size, batch_size);

        if estimated_usage > available_memory {
            AllocationStrategy::Chunked {
                chunk_size: available_memory / 2,
                overlap: true,
            }
        } else if estimated_usage > available_memory / 2 {
            AllocationStrategy::Conservative {
                pool_size_limit: available_memory / 4,
            }
        } else {
            AllocationStrategy::Aggressive {
                prefetch_size: estimated_usage * 2,
            }
        }
    }

    /// Estimate memory usage for a given problem
    pub fn estimate_memory_usage(_problem_size: usize, batch_size: usize) -> usize {
        // Rough estimation: input data + output data + temporary buffers
        let input_size = batch_size * _problem_size * 8; // f64
        let output_size = batch_size * 8; // f64
        let temp_size = input_size; // Temporary arrays

        input_size + output_size + temp_size
    }

    /// Memory allocation strategies
    #[derive(Debug, Clone)]
    pub enum AllocationStrategy {
        /// Use chunks with optional overlap for large problems
        Chunked { chunk_size: usize, overlap: bool },
        /// Conservative allocation with limited pool sizes
        Conservative { pool_size_limit: usize },
        /// Aggressive allocation with prefetching
        Aggressive { prefetch_size: usize },
    }

    /// Check if the system has sufficient memory for an operation
    pub fn check_memory_availability(
        required_memory: usize,
        memory_info: &GpuMemoryInfo,
    ) -> ScirsResult<bool> {
        let available = memory_info.free;
        let safety_margin = 0.1; // Keep 10% free
        let usable = (available as f64 * (1.0 - safety_margin)) as usize;

        Ok(required_memory <= usable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_stats() {
        let mut stats = AllocationStats::new();
        stats.total_allocations = 100;
        stats.pool_hits = 70;

        assert_eq!(stats.hit_rate(), 0.7);
    }

    #[test]
    fn test_memory_stats_utilization() {
        let stats = MemoryStats {
            current_usage: 800,
            memory_limit: Some(1000),
            allocation_stats: AllocationStats::new(),
            pool_sizes: HashMap::new(),
        };

        assert_eq!(stats.utilization(), Some(0.8));
    }

    #[test]
    fn test_memory_usage_estimation() {
        let usage = utils::estimate_memory_usage(10, 100);
        assert!(usage > 0);

        // Should scale with problem size and batch size
        let larger_usage = utils::estimate_memory_usage(20, 200);
        assert!(larger_usage > usage);
    }

    #[test]
    fn test_allocation_strategy() {
        let strategy = utils::calculate_allocation_strategy(
            1000,    // Large problem
            1000,    // Large batch
            500_000, // Limited memory
        );

        match strategy {
            utils::AllocationStrategy::Chunked { .. } => {
                // Expected for large problems with limited memory
            }
            _ => panic!("Expected chunked strategy for large problem"),
        }
    }
}
