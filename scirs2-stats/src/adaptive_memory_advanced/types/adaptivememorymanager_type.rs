//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::{
    parallel_ops::*,
    simd_ops::{PlatformCapabilities, SimdUnifiedOps},
};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, RwLock, Weak};

use super::types::{
    AdaptiveMemoryConfig, AllocationStrategy, CacheManager, GCManager, MemoryPerformanceMonitor,
    MemoryPool, NumaManager, OutOfCoreManager, PredictiveEngine, PressureMonitor,
};

/// Advanced-advanced adaptive memory manager
pub struct AdaptiveMemoryManager<F> {
    pub(super) config: AdaptiveMemoryConfig,
    pub(super) memory_pools: Arc<RwLock<HashMap<usize, Arc<MemoryPool>>>>,
    /// Strategy actually used for each live allocation, keyed by pointer address.
    ///
    /// `allocate` may resolve `AllocationStrategy::Adaptive` into a different
    /// concrete strategy (e.g. `Pool`), so the deallocation path cannot re-derive
    /// the strategy from the config — it must free through the same allocator
    /// that produced the pointer. Freeing pool memory through the system
    /// allocator (or with a different layout) is undefined behaviour and aborts
    /// with STATUS_HEAP_CORRUPTION on Windows.
    pub(super) allocation_strategies: Arc<RwLock<HashMap<usize, AllocationStrategy>>>,
    pub(super) cache_manager: Arc<CacheManager>,
    pub(super) numa_manager: Arc<NumaManager>,
    pub(super) predictive_engine: Arc<PredictiveEngine>,
    pub(super) pressure_monitor: Arc<PressureMonitor>,
    pub(super) out_of_core_manager: Arc<OutOfCoreManager>,
    pub(super) gc_manager: Arc<GCManager>,
    pub(super) performance_monitor: Arc<MemoryPerformanceMonitor>,
    pub(super) _phantom: PhantomData<F>,
}
