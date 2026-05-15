//! Advanced parallel processing and scheduling
//!
//! This module provides comprehensive parallel processing capabilities including:
//! - Work-stealing scheduler for efficient thread utilization
//! - Custom partitioning strategies for different data distributions
//! - Nested parallelism with controlled resource usage
//! - Load balancing and adaptive scheduling
//! - NUMA-locality chunk map (`par_map_chunks`)

mod nested;
pub mod numa;
mod partitioning;
mod scheduler;

// Re-export scheduler functionality
pub use scheduler::{
    create_work_stealing_scheduler, create_work_stealing_scheduler_with_workers, get_workerid,
    CloneableTask, ParallelTask, SchedulerConfig, SchedulerConfigBuilder, SchedulerStats,
    SchedulingPolicy, TaskHandle, TaskPriority, TaskStatus, WorkStealingArray,
    WorkStealingScheduler,
};

// Re-export partitioning functionality
pub use partitioning::{
    DataDistribution, DataPartitioner, LoadBalancer, PartitionStrategy, PartitionerConfig,
};

// Re-export nested parallelism functionality
pub use nested::{
    adaptive_par_for_each, adaptive_par_map, current_nesting_level, is_nested_parallelism_allowed,
    nested_scope, nested_scope_with_limits, with_nested_policy, NestedConfig, NestedContext,
    NestedPolicy, NestedScope, ResourceLimits, ResourceManager, ResourceUsageStats,
};

// Re-export NUMA par_map_chunks at parallel module level
pub use numa::par_map_chunks;

// Note: parallel_map is now provided by parallel_ops module for simpler usage
