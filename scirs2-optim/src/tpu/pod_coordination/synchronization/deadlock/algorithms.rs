//! Deadlock Detection Algorithms
//!
//! This module contains various algorithms for detecting deadlocks in distributed systems,
//! including graph-based, banker's algorithm, and machine learning approaches.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Deadlock detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeadlockDetectionAlgorithm {
    /// Wait-for graph algorithm
    WaitForGraph {
        cycle_detection: CycleDetectionMethod,
        optimization: GraphOptimization,
    },
    /// Banker's algorithm
    Bankers {
        safe_state_checking: SafeStateMethod,
        resource_allocation: ResourceAllocationMethod,
    },
    /// Resource allocation graph
    ResourceAllocationGraph {
        reduction_method: GraphReductionMethod,
        deadlock_criteria: DeadlockCriteria,
    },
    /// Timestamp-based detection
    TimestampBased {
        ordering: TimestampOrdering,
        conflict_resolution: ConflictResolution,
    },
    /// Probe-based detection
    ProbeBased {
        probe_propagation: PropagationStrategy,
        response_handling: ResponseHandling,
    },
    /// Machine learning detection
    MachineLearning {
        model_type: super::ml::MLModelType,
        feature_extraction: super::ml::FeatureExtraction,
        prediction_threshold: f64,
    },
    /// Hybrid detection
    Hybrid {
        algorithms: Vec<DeadlockDetectionAlgorithm>,
        combination_strategy: CombinationStrategy,
    },
    /// Custom algorithm
    Custom {
        algorithm_name: String,
        parameters: HashMap<String, String>,
    },
}

/// Cycle detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CycleDetectionMethod {
    /// Depth-first search
    DepthFirstSearch,
    /// Tarjan's strongly connected components
    TarjanSCC,
    /// Johnson's algorithm
    Johnson,
    /// Floyd-Warshall based
    FloydWarshall,
    /// Matrix multiplication
    MatrixMultiplication,
}

/// Graph optimization techniques
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphOptimization {
    /// Enable graph compression
    pub compression: bool,
    /// Use adjacency matrix
    pub adjacency_matrix: bool,
    /// Cache optimization
    pub cache_optimization: CacheOptimization,
    /// Parallel processing
    pub parallel_processing: ParallelProcessing,
}

/// Cache optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheOptimization {
    /// Cache size
    pub cache_size: usize,
    /// Cache policy
    pub policy: CachePolicy,
    /// Prefetching strategy
    pub prefetching: PrefetchingStrategy,
}

/// Cache policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CachePolicy {
    /// Least recently used
    LRU,
    /// Least frequently used
    LFU,
    /// Most recently used
    MRU,
    /// Random replacement
    Random,
    /// Optimal (Belady's algorithm)
    Optimal,
}

/// Prefetching strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrefetchingStrategy {
    /// No prefetching
    None,
    /// Sequential prefetching
    Sequential { lookahead: usize },
    /// Adaptive prefetching
    Adaptive { learning_rate: f64 },
    /// Pattern-based prefetching
    PatternBased { patterns: Vec<String> },
}

/// Parallel processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelProcessing {
    /// Enable parallel processing
    pub enabled: bool,
    /// Number of threads
    pub thread_count: usize,
    /// Work distribution strategy
    pub distribution: WorkDistribution,
    /// Synchronization method
    pub synchronization: SynchronizationMethod,
}

/// Work distribution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkDistribution {
    /// Static partitioning
    Static,
    /// Dynamic load balancing
    Dynamic,
    /// Work stealing
    WorkStealing,
    /// Hierarchical decomposition
    Hierarchical,
}

/// Synchronization methods for parallel processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynchronizationMethod {
    /// Barrier synchronization
    Barrier,
    /// Lock-based synchronization
    LockBased,
    /// Lock-free synchronization
    LockFree,
    /// Message passing
    MessagePassing,
}

/// Safe state checking methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafeStateMethod {
    /// Basic safe state algorithm
    Basic,
    /// Optimized safe state checking
    Optimized,
    /// Incremental safe state checking
    Incremental,
    /// Probabilistic safe state checking
    Probabilistic { confidence: f64 },
}

/// Resource allocation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceAllocationMethod {
    /// First fit
    FirstFit,
    /// Best fit
    BestFit,
    /// Worst fit
    WorstFit,
    /// Next fit
    NextFit,
    /// Buddy system
    BuddySystem,
}

/// Graph reduction methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphReductionMethod {
    /// Sequential reduction
    Sequential,
    /// Parallel reduction
    Parallel,
    /// Hierarchical reduction
    Hierarchical,
    /// Optimized reduction
    Optimized,
}

/// Deadlock criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlockCriteria {
    /// Mutual exclusion requirement
    pub mutual_exclusion: bool,
    /// Hold and wait requirement
    pub hold_and_wait: bool,
    /// No preemption requirement
    pub no_preemption: bool,
    /// Circular wait requirement
    pub circular_wait: bool,
    /// Custom criteria
    pub custom_criteria: Vec<String>,
}

/// Timestamp ordering methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimestampOrdering {
    /// Wait-die
    WaitDie,
    /// Wound-wait
    WoundWait,
    /// Conservative timestamp ordering
    Conservative,
    /// Optimistic timestamp ordering
    Optimistic,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Abort younger transaction
    AbortYounger,
    /// Abort older transaction
    AbortOlder,
    /// Priority-based resolution
    PriorityBased,
    /// Random resolution
    Random,
    /// Custom resolution
    Custom { strategy: String },
}

/// Probe propagation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropagationStrategy {
    /// Breadth-first propagation
    BreadthFirst,
    /// Depth-first propagation
    DepthFirst,
    /// Priority-based propagation
    PriorityBased,
    /// Optimized propagation
    Optimized,
}

/// Response handling for probe-based detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseHandling {
    /// Response timeout
    pub timeout: Duration,
    /// Retry policy
    pub retry_policy: RetryPolicy,
    /// Error handling
    pub error_handling: ErrorHandling,
}

/// Retry policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum retries
    pub max_retries: usize,
    /// Retry delay
    pub delay: Duration,
    /// Backoff strategy
    pub backoff: BackoffStrategy,
}

/// Backoff strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    /// Fixed delay
    Fixed,
    /// Exponential backoff
    Exponential { factor: f64 },
    /// Linear backoff
    Linear { increment: Duration },
    /// Jittered exponential
    JitteredExponential { factor: f64, jitter: f64 },
}

/// Error handling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorHandling {
    /// Ignore errors
    Ignore,
    /// Log errors
    Log,
    /// Escalate errors
    Escalate,
    /// Custom handling
    Custom { handler: String },
}

/// Combination strategies for hybrid algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CombinationStrategy {
    /// Majority voting
    MajorityVoting,
    /// Weighted voting
    WeightedVoting { weights: Vec<f64> },
    /// Consensus-based
    Consensus,
    /// First positive result
    FirstPositive,
    /// All must agree
    AllAgree,
    /// Custom combination
    Custom { strategy: String },
}

impl Default for GraphOptimization {
    fn default() -> Self {
        Self {
            compression: true,
            adjacency_matrix: false,
            cache_optimization: CacheOptimization::default(),
            parallel_processing: ParallelProcessing::default(),
        }
    }
}

impl Default for CacheOptimization {
    fn default() -> Self {
        Self {
            cache_size: 1024,
            policy: CachePolicy::LRU,
            prefetching: PrefetchingStrategy::None,
        }
    }
}

impl Default for ParallelProcessing {
    fn default() -> Self {
        Self {
            enabled: true,
            thread_count: num_cpus::get(),
            distribution: WorkDistribution::Dynamic,
            synchronization: SynchronizationMethod::LockFree,
        }
    }
}

impl Default for DeadlockCriteria {
    fn default() -> Self {
        Self {
            mutual_exclusion: true,
            hold_and_wait: true,
            no_preemption: true,
            circular_wait: true,
            custom_criteria: Vec::new(),
        }
    }
}

impl Default for ResponseHandling {
    fn default() -> Self {
        Self {
            timeout: Duration::from_millis(100),
            retry_policy: RetryPolicy::default(),
            error_handling: ErrorHandling::Log,
        }
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            delay: Duration::from_millis(10),
            backoff: BackoffStrategy::Exponential { factor: 2.0 },
        }
    }
}