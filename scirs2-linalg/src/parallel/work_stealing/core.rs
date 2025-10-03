//! Core types and enums for the work-stealing scheduler
//!
//! This module contains the fundamental data structures and types used
//! by the work-stealing scheduler implementation.

use std::time::Duration;

/// Type alias for complex work item types used in QR decomposition
use scirs2_core::ndarray::{Array1, Array2};
pub type QRWorkItem<F> = super::WorkItem<(usize, Array1<F>, Array2<F>)>;

/// Type alias for complex work item types used in band matrix solving
pub type BandSolveWorkItem<F> = super::WorkItem<(usize, usize, usize, Array2<F>, Array1<F>)>;

/// Work item for the work-stealing scheduler
#[derive(Debug, Clone)]
pub struct WorkItem<T>
where
    T: Clone,
{
    /// Unique identifier for the work item
    pub id: usize,
    /// The actual work payload
    pub payload: T,
    /// Expected execution time (for scheduling optimization)
    pub estimated_time: Option<Duration>,
}

impl<T: Clone> WorkItem<T> {
    /// Create a new work item
    pub fn new(id: usize, payload: T) -> Self {
        Self {
            id,
            payload,
            estimated_time: None,
        }
    }

    /// Create a work item with estimated execution time
    pub fn with_estimate(_id: usize, payload: T, estimatedtime: Duration) -> Self {
        Self {
            id: _id,
            payload,
            estimated_time: Some(estimatedtime),
        }
    }
}

/// Work-stealing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StealingStrategy {
    /// Random victim selection
    Random,
    /// Round-robin victim selection
    RoundRobin,
    /// Target the most loaded worker
    MostLoaded,
    /// Target based on work locality
    LocalityAware,
    /// Adaptive strategy that learns from history
    #[default]
    Adaptive,
}

/// Load balancing parameters for adaptive optimization
#[derive(Debug, Clone)]
pub struct LoadBalancingParams {
    /// Minimum work items before attempting to steal
    pub steal_threshold: usize,
    /// Maximum steal attempts per worker
    pub max_steal_attempts: usize,
    /// Exponential backoff base for failed steals
    pub backoff_base: Duration,
    /// Maximum backoff time
    pub max_backoff: Duration,
    /// Work chunk size for splitting large tasks
    pub chunksize: usize,
    /// Enable work item priority scheduling
    pub priority_scheduling: bool,
}

impl Default for LoadBalancingParams {
    fn default() -> Self {
        Self {
            steal_threshold: 2,
            max_steal_attempts: 3,
            backoff_base: Duration::from_micros(10),
            max_backoff: Duration::from_millis(1),
            chunksize: 100,
            priority_scheduling: false,
        }
    }
}

/// Priority levels for work items
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum WorkPriority {
    Low,
    #[default]
    Normal,
    High,
    Critical,
}

/// Matrix operation types for scheduler optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixOperationType {
    MatrixVectorMultiplication,
    MatrixMatrixMultiplication,
    Decomposition,
    EigenComputation,
    IterativeSolver,
}

/// Workload characteristics for adaptive optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadCharacteristics {
    HighVariance,
    LowVariance,
    MemoryBound,
    ComputeBound,
}

/// Work complexity patterns for execution time prediction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkComplexity {
    Constant,
    Linear,
    Quadratic,
    Variable,
}

/// Scheduler performance statistics
#[derive(Debug, Default, Clone)]
pub struct SchedulerStats {
    /// Total items processed
    pub total_items: usize,
    /// Total execution time across all workers
    pub total_execution_time: Duration,
    /// Number of successful steals
    pub successful_steals: usize,
    /// Number of failed steal attempts
    pub failed_steals: usize,
    /// Load balancing efficiency (0.0 to 1.0)
    pub load_balance_efficiency: f64,
    /// Time variance across workers
    pub time_variance: f64,
    /// Average work stealing latency
    pub avg_steal_latency: Duration,
    /// Work distribution histogram
    pub work_distribution: Vec<usize>,
    /// Thread utilization rates
    pub thread_utilization: Vec<f64>,
}
