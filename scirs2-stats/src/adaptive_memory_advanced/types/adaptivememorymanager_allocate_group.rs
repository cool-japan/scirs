//! # AdaptiveMemoryManager - allocate_group Methods
//!
//! This module contains method implementations for `AdaptiveMemoryManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{StatsError, StatsResult};
use scirs2_core::numeric::{Float, NumCast, One, Zero};
use scirs2_core::{
    parallel_ops::*,
    simd_ops::{PlatformCapabilities, SimdUnifiedOps},
};

use super::types::{AllocationContext, AllocationStrategy};

use super::adaptivememorymanager_type::AdaptiveMemoryManager;

impl<F> AdaptiveMemoryManager<F>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + 'static
        + std::fmt::Display,
{
    /// Allocate memory with optimal strategy
    pub fn allocate(&self, size: usize) -> StatsResult<*mut u8> {
        let allocation_context = self.analyze_allocation_request(size)?;
        let strategy = self.select_allocation_strategy(&allocation_context)?;
        let ptr = match strategy {
            AllocationStrategy::System => self.allocate_system(size),
            AllocationStrategy::Pool => self.allocate_pool(size),
            AllocationStrategy::NumaAware => self.allocate_numa_aware(size, &allocation_context),
            AllocationStrategy::MemoryMapped => self.allocate_memory_mapped(size),
            AllocationStrategy::Adaptive => self.allocate_adaptive(size, &allocation_context),
            AllocationStrategy::ZeroCopy => self.allocate_zero_copy(size),
        }?;

        // Remember which allocator produced this pointer so `deallocate` can
        // return it to the same one. `strategy` is the *resolved* strategy,
        // which for an `Adaptive` config is whatever the predictor chose.
        self.allocation_strategies
            .write()
            .expect("Operation failed")
            .insert(ptr as usize, strategy);

        Ok(ptr)
    }
    /// Analyze allocation request context
    fn analyze_allocation_request(&self, size: usize) -> StatsResult<AllocationContext> {
        let current_thread = std::thread::current().id();

        // Use hash-based approach instead of transmute for platform compatibility
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        current_thread.hash(&mut hasher);
        let thread_id = hasher.finish() as usize;

        let current_pressure = self.pressure_monitor.get_current_pressure();
        let predicted_usage = self
            .predictive_engine
            .predict_memory_usage(size, thread_id)?;
        let numa_node = self.numa_manager.get_optimal_node(thread_id);
        Ok(AllocationContext {
            size,
            thread_id,
            current_pressure,
            predicted_usage,
            numa_node,
            allocation_type: self.infer_allocation_type(size),
            urgency: self.calculate_urgency(size, current_pressure),
        })
    }
    /// Select optimal allocation strategy
    fn select_allocation_strategy(
        &self,
        context: &AllocationContext,
    ) -> StatsResult<AllocationStrategy> {
        match self.config.allocation_strategy {
            AllocationStrategy::Adaptive => {
                let features = self.extract_allocation_features(context);
                let predicted_strategy = self
                    .predictive_engine
                    .predict_allocation_strategy(&features)?;
                Ok(predicted_strategy)
            }
            strategy => Ok(strategy),
        }
    }
    /// System allocator
    pub fn allocate_system(&self, size: usize) -> StatsResult<*mut u8> {
        use std::alloc::{alloc, Layout};
        let layout = Layout::from_size_align(size, std::mem::align_of::<F>())
            .map_err(|e| StatsError::InvalidArgument(format!("Invalid layout: {}", e)))?;
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            Err(StatsError::ComputationError(
                "Memory allocation failed".to_string(),
            ))
        } else {
            Ok(ptr)
        }
    }
    /// Pool allocator
    pub fn allocate_pool(&self, size: usize) -> StatsResult<*mut u8> {
        let poolsize = self.calculate_poolsize(size);
        let pool = self.get_or_create_pool(poolsize)?;
        pool.allocate()
    }
    /// NUMA-aware allocator
    pub fn allocate_numa_aware(
        &self,
        size: usize,
        context: &AllocationContext,
    ) -> StatsResult<*mut u8> {
        let numa_node = context.numa_node.unwrap_or(0);
        self.numa_manager.allocate_on_node(size, numa_node)
    }
    /// Memory-mapped allocator
    pub fn allocate_memory_mapped(&self, size: usize) -> StatsResult<*mut u8> {
        self.out_of_core_manager.allocate_mapped(size)
    }
    /// Adaptive allocator
    pub fn allocate_adaptive(
        &self,
        size: usize,
        context: &AllocationContext,
    ) -> StatsResult<*mut u8> {
        let performance_metrics = self.performance_monitor.get_current_metrics();
        if performance_metrics.memory_bandwidth < 0.5 {
            self.allocate_pool(size)
        } else if performance_metrics.numa_locality < 0.7 {
            self.allocate_numa_aware(size, context)
        } else if performance_metrics.cache_hit_ratio < 0.8 {
            self.allocate_system(size)
        } else {
            self.allocate_pool(size)
        }
    }
    /// Zero-copy allocator
    pub fn allocate_zero_copy(&self, size: usize) -> StatsResult<*mut u8> {
        self.allocate_memory_mapped(size)
    }
}
