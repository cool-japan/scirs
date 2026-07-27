//! # AdaptiveMemoryManager - deallocate_group Methods
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

use super::types::AllocationStrategy;

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
    /// Deallocate memory
    pub fn deallocate(&self, ptr: *mut u8, size: usize) -> StatsResult<()> {
        let strategy = self.infer_deallocation_strategy(ptr, size);
        match strategy {
            AllocationStrategy::System => self.deallocate_system(ptr, size),
            AllocationStrategy::Pool => self.deallocate_pool(ptr, size),
            AllocationStrategy::NumaAware => self.deallocate_numa_aware(ptr, size),
            AllocationStrategy::MemoryMapped => self.deallocate_memory_mapped(ptr, size),
            AllocationStrategy::Adaptive => self.deallocate_adaptive(ptr, size),
            AllocationStrategy::ZeroCopy => self.deallocate_zero_copy(ptr, size),
        }
    }
    /// Infer deallocation strategy from pointer.
    ///
    /// Looks up (and forgets) the strategy recorded by `allocate` for this exact
    /// pointer. Falling back to `self.config.allocation_strategy` — as this used
    /// to do unconditionally — is wrong whenever the config says `Adaptive`,
    /// because `allocate` resolves that to a concrete strategy such as `Pool`.
    /// Pool memory then got handed to `dealloc` with a different size and
    /// alignment than it was allocated with, corrupting the heap.
    pub fn infer_deallocation_strategy(&self, ptr: *mut u8, size: usize) -> AllocationStrategy {
        if let Some(strategy) = self
            .allocation_strategies
            .write()
            .expect("Operation failed")
            .remove(&(ptr as usize))
        {
            return strategy;
        }

        // Pointer we never handed out (or already freed): fall back to the
        // configured strategy, resolving `Adaptive` the way `allocate` would.
        match self.config.allocation_strategy {
            AllocationStrategy::Adaptive => AllocationStrategy::System,
            strategy => strategy,
        }
    }
    /// System deallocation
    pub fn deallocate_system(&self, ptr: *mut u8, size: usize) -> StatsResult<()> {
        use std::alloc::{dealloc, Layout};
        let layout = Layout::from_size_align(size, std::mem::align_of::<F>())
            .map_err(|e| StatsError::InvalidArgument(format!("Invalid layout: {}", e)))?;
        unsafe { dealloc(ptr, layout) };
        Ok(())
    }
    /// NUMA-aware deallocation
    pub fn deallocate_numa_aware(&self, ptr: *mut u8, size: usize) -> StatsResult<()> {
        self.numa_manager.deallocate(ptr, size)
    }
    /// Memory-mapped deallocation
    pub fn deallocate_memory_mapped(&self, ptr: *mut u8, size: usize) -> StatsResult<()> {
        self.out_of_core_manager.deallocate_mapped(ptr, size)
    }
    /// Adaptive deallocation
    pub fn deallocate_adaptive(&self, ptr: *mut u8, size: usize) -> StatsResult<()> {
        self.deallocate_system(ptr, size)
    }
    /// Zero-copy deallocation
    pub fn deallocate_zero_copy(&self, ptr: *mut u8, size: usize) -> StatsResult<()> {
        self.deallocate_memory_mapped(ptr, size)
    }
}
