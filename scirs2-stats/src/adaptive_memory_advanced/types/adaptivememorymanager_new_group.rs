//! # AdaptiveMemoryManager - new_group Methods
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
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, RwLock, Weak};

use super::types::{
    AdaptiveMemoryConfig, CacheManager, GCManager, MemoryPerformanceMonitor, MemoryPool,
    NumaManager, OutOfCoreManager, PredictiveEngine, PressureMonitor,
};

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
    /// Create new adaptive memory manager
    pub fn new() -> Self {
        Self::with_config(AdaptiveMemoryConfig::default())
    }
    /// Create with custom configuration
    pub fn with_config(config: AdaptiveMemoryConfig) -> Self {
        let memory_pools = Arc::new(RwLock::new(HashMap::new()));
        let cache_manager = Arc::new(CacheManager::new(&config.cache_optimization));
        let numa_manager = Arc::new(NumaManager::new(&config.numa_config));
        let predictive_engine = Arc::new(PredictiveEngine::new(&config.predictive_config));
        let pressure_monitor = Arc::new(PressureMonitor::new(&config.pressure_config));
        let out_of_core_manager = Arc::new(OutOfCoreManager::new(&config.out_of_core_config));
        let gc_manager = Arc::new(GCManager::new(&config.gc_config));
        let performance_monitor = Arc::new(MemoryPerformanceMonitor::new());
        Self {
            config,
            memory_pools,
            allocation_strategies: Arc::new(RwLock::new(HashMap::new())),
            cache_manager,
            numa_manager,
            predictive_engine,
            pressure_monitor,
            out_of_core_manager,
            gc_manager,
            performance_monitor,
            _phantom: PhantomData,
        }
    }
    /// Get or create memory pool
    pub fn get_or_create_pool(&self, poolsize: usize) -> StatsResult<Arc<MemoryPool>> {
        {
            let pools = self.memory_pools.read().expect("Operation failed");
            if let Some(pool) = pools.get(&poolsize) {
                return Ok(Arc::clone(pool));
            }
        }
        let mut pools = self.memory_pools.write().expect("Operation failed");
        if let Some(pool) = pools.get(&poolsize) {
            return Ok(Arc::clone(pool));
        }
        let pool = Arc::new(MemoryPool::new(poolsize, self.config.allocation_strategy));
        pools.insert(poolsize, Arc::clone(&pool));
        Ok(pool)
    }
}
