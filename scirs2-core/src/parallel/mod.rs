//! Advanced parallel processing and scheduling
//!
//! This module re-exports all functionality from the main parallel.rs file
//! and adds a work-stealing scheduler for more efficient thread utilization.

mod scheduler;

pub use scheduler::{
    create_work_stealing_scheduler, create_work_stealing_scheduler_with_workers, get_worker_id,
    CloneableTask, ParallelTask, SchedulerConfig, SchedulerConfigBuilder, SchedulerStats,
    SchedulingPolicy, TaskHandle, TaskPriority, TaskStatus, WorkStealingArray,
    WorkStealingScheduler,
};

// Re-export the parallel_map function from the scheduler module
pub use self::scheduler::parallel::par_map as parallel_map;
