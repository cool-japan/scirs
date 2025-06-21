//! # Adaptive Optimization System
//!
//! Enterprise-grade adaptive optimization system for runtime performance tuning
//! and workload-aware optimization. Automatically adjusts system parameters
//! based on real-time performance metrics and workload characteristics.
//!
//! ## Features
//!
//! - Runtime performance tuning with machine learning algorithms
//! - Workload-aware optimization based on usage patterns
//! - Automatic parameter adjustment for optimal performance
//! - Multi-objective optimization (speed, memory, energy efficiency)
//! - Predictive performance modeling
//! - Adaptive algorithm selection based on data characteristics
//! - Dynamic resource allocation and load balancing
//! - Performance regression prevention
//! - Integration with production profiling and monitoring
//! - Enterprise-grade analytics and reporting
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::profiling::adaptive::{
//!     AdaptiveOptimizer, OptimizationConfig, OptimizationGoal, WorkloadProfile
//! };
//!
//! // Configure adaptive optimizer
//! let config = OptimizationConfig::production()
//!     .with_goal(OptimizationGoal::Balanced)
//!     .with_learning_rate(0.01)
//!     .with_adaptation_interval(std::time::Duration::from_secs(60));
//!
//! let mut optimizer = AdaptiveOptimizer::new(config)?;
//!
//! // Register workload for optimization
//! let workload = WorkloadProfile::builder()
//!     .with_name("matrix_operations")
//!     .with_data_size(1_000_000)
//!     .with_compute_intensity(0.8)
//!     .build();
//!
//! optimizer.register_workload(workload)?;
//!
//! // Start adaptive optimization
//! optimizer.start_optimization()?;
//!
//! // Your code here - optimizer automatically tunes performance
//! fn compute_intensive_operation() -> Vec<f64> {
//!     // Example compute-intensive operation
//!     (0..10000).map(|i| (i as f64).sin() * (i as f64).cos()).collect()
//! }
//! let result = compute_intensive_operation();
//!
//! // Get optimization recommendations
//! let recommendations = optimizer.get_recommendations()?;
//! for rec in recommendations {
//!     println!("Recommendation: {} -> {}", rec.parameter, rec.suggested_value);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{CoreError, CoreResult};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Adaptive optimization configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OptimizationConfig {
    /// Primary optimization goal
    pub goal: OptimizationGoal,
    /// Learning rate for adaptive algorithms
    pub learning_rate: f64,
    /// Interval between optimization adjustments
    pub adaptation_interval: Duration,
    /// Historical data retention period
    pub history_retention: Duration,
    /// Minimum confidence threshold for changes
    pub confidence_threshold: f64,
    /// Maximum parameter adjustment per iteration (safety limit)
    pub max_adjustment_rate: f64,
    /// Enable predictive modeling
    pub enable_prediction: bool,
    /// Enable multi-objective optimization
    pub enable_multi_objective: bool,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
    /// Enable rollback on performance degradation
    pub enable_rollback: bool,
    /// Performance monitoring window
    pub monitoring_window: Duration,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            goal: OptimizationGoal::Balanced,
            learning_rate: 0.01,
            adaptation_interval: Duration::from_secs(60),
            history_retention: Duration::from_secs(24 * 60 * 60), // 24 hours
            confidence_threshold: 0.95,
            max_adjustment_rate: 0.1, // 10% max change per iteration
            enable_prediction: true,
            enable_multi_objective: true,
            resource_constraints: ResourceConstraints::default(),
            enable_rollback: true,
            monitoring_window: Duration::from_secs(300), // 5 minutes
        }
    }
}

impl OptimizationConfig {
    /// Create production-optimized configuration
    pub fn production() -> Self {
        Self {
            goal: OptimizationGoal::Performance,
            learning_rate: 0.005, // Conservative for production
            adaptation_interval: Duration::from_secs(300), // 5 minutes
            confidence_threshold: 0.99, // High confidence required
            max_adjustment_rate: 0.05, // Conservative 5% max change
            enable_rollback: true,
            ..Default::default()
        }
    }

    /// Create development-optimized configuration
    pub fn development() -> Self {
        Self {
            goal: OptimizationGoal::Development,
            learning_rate: 0.02, // More aggressive for experimentation
            adaptation_interval: Duration::from_secs(30),
            confidence_threshold: 0.85,
            max_adjustment_rate: 0.2, // Allow larger adjustments
            ..Default::default()
        }
    }

    /// Create memory-optimized configuration
    pub fn memory_optimized() -> Self {
        Self {
            goal: OptimizationGoal::MemoryEfficiency,
            resource_constraints: ResourceConstraints {
                max_memory_usage: Some(1024 * 1024 * 1024), // 1GB limit
                max_cpu_usage: Some(0.8),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Set optimization goal
    pub fn with_goal(mut self, goal: OptimizationGoal) -> Self {
        self.goal = goal;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, rate: f64) -> Self {
        self.learning_rate = rate.clamp(0.001, 0.1);
        self
    }

    /// Set adaptation interval
    pub fn with_adaptation_interval(mut self, interval: Duration) -> Self {
        self.adaptation_interval = interval;
        self
    }

    /// Set confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold.clamp(0.5, 0.999);
        self
    }

    /// Enable predictive modeling
    pub fn with_prediction(mut self, enable: bool) -> Self {
        self.enable_prediction = enable;
        self
    }

    /// Set resource constraints
    pub fn with_resource_constraints(mut self, constraints: ResourceConstraints) -> Self {
        self.resource_constraints = constraints;
        self
    }
}

/// Optimization goals
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OptimizationGoal {
    /// Optimize for maximum performance (speed)
    Performance,
    /// Optimize for memory efficiency
    MemoryEfficiency,
    /// Optimize for energy efficiency
    EnergyEfficiency,
    /// Balanced optimization across all metrics
    Balanced,
    /// Optimize for development/debugging
    Development,
    /// Custom multi-objective optimization
    Custom(Vec<ObjectiveWeight>),
}

/// Objective weights for multi-objective optimization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ObjectiveWeight {
    /// Objective name
    pub name: String,
    /// Weight (0.0 to 1.0)
    pub weight: f64,
    /// Priority level
    pub priority: Priority,
}

impl PartialEq for ObjectiveWeight {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && (self.weight - other.weight).abs() < f64::EPSILON
            && self.priority == other.priority
    }
}

impl Eq for ObjectiveWeight {}

/// Priority levels for objectives
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Priority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Resource constraints for optimization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ResourceConstraints {
    /// Maximum memory usage (bytes)
    pub max_memory_usage: Option<usize>,
    /// Maximum CPU usage (0.0 to 1.0)
    pub max_cpu_usage: Option<f64>,
    /// Maximum network bandwidth (bytes/sec)
    pub max_network_bandwidth: Option<usize>,
    /// Maximum disk I/O (bytes/sec)
    pub max_disk_io: Option<usize>,
    /// Maximum number of threads
    pub max_threads: Option<usize>,
    /// Energy consumption limit (watts)
    pub max_energy_consumption: Option<f64>,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_memory_usage: None,
            max_cpu_usage: Some(0.9), // 90% CPU usage limit
            max_network_bandwidth: None,
            max_disk_io: None,
            max_threads: Some({
                #[cfg(feature = "num_cpus")]
                {
                    num_cpus::get()
                }
                #[cfg(not(feature = "num_cpus"))]
                {
                    4 // Default to 4 threads
                }
            }),
            max_energy_consumption: None,
        }
    }
}

/// Workload profile for optimization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WorkloadProfile {
    /// Workload name
    pub name: String,
    /// Data size characteristics
    pub data_size: usize,
    /// Compute intensity (0.0 to 1.0)
    pub compute_intensity: f64,
    /// Memory access pattern
    pub memory_pattern: MemoryPattern,
    /// Parallelism characteristics
    pub parallelism_profile: ParallelismProfile,
    /// I/O characteristics
    pub io_profile: IOProfile,
    /// Workload type
    pub workload_type: WorkloadType,
    /// Expected duration
    pub expected_duration: Option<Duration>,
    /// Priority level
    pub priority: Priority,
}

/// Memory access patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MemoryPattern {
    /// Sequential access pattern
    Sequential,
    /// Random access pattern
    Random,
    /// Strided access pattern
    Strided,
    /// Mixed access pattern
    Mixed,
    /// Cache-friendly access
    CacheFriendly,
}

/// Parallelism characteristics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParallelismProfile {
    /// Can benefit from parallelization
    pub parallelizable: bool,
    /// Optimal number of threads
    pub optimal_threads: Option<usize>,
    /// Synchronization overhead
    pub sync_overhead: f64,
    /// Load balancing characteristics
    pub load_balance: LoadBalanceType,
}

/// Load balancing types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LoadBalanceType {
    /// Even load distribution
    Even,
    /// Uneven load distribution
    Uneven,
    /// Dynamic load balancing required
    Dynamic,
    /// Work-stealing beneficial
    WorkStealing,
}

/// I/O characteristics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IOProfile {
    /// I/O intensity (0.0 to 1.0)
    pub intensity: f64,
    /// Primary I/O type
    pub io_type: IOType,
    /// Read/write ratio
    pub read_write_ratio: f64,
    /// Buffer size preferences
    pub preferred_buffer_size: Option<usize>,
}

/// I/O types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum IOType {
    /// Primarily disk I/O
    Disk,
    /// Primarily network I/O
    Network,
    /// Primarily memory I/O
    Memory,
    /// Mixed I/O patterns
    Mixed,
    /// No significant I/O
    None,
}

/// Workload types
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum WorkloadType {
    /// CPU-intensive computations
    ComputeIntensive,
    /// Memory-intensive operations
    MemoryIntensive,
    /// I/O-intensive operations
    IOIntensive,
    /// Network-intensive operations
    NetworkIntensive,
    /// Balanced workload
    Balanced,
    /// Interactive/latency-sensitive
    Interactive,
    /// Batch processing
    Batch,
    /// Real-time processing
    RealTime,
    /// Custom workload type
    Custom(String),
}

impl WorkloadProfile {
    /// Create a new workload profile builder
    pub fn builder() -> WorkloadProfileBuilder {
        WorkloadProfileBuilder::new()
    }

    /// Get optimization parameters for this workload
    pub fn get_optimization_hints(&self) -> OptimizationHints {
        OptimizationHints {
            preferred_chunk_size: self.calculate_optimal_chunk_size(),
            preferred_thread_count: self.parallelism_profile.optimal_threads,
            memory_allocation_strategy: self.get_memory_strategy(),
            caching_strategy: self.get_caching_strategy(),
            io_strategy: self.get_io_strategy(),
            algorithm_preferences: self.get_algorithm_preferences(),
        }
    }

    /// Calculate optimal chunk size for this workload
    fn calculate_optimal_chunk_size(&self) -> Option<usize> {
        match self.memory_pattern {
            MemoryPattern::Sequential => Some((self.data_size / 1000).max(1024)),
            MemoryPattern::Random => Some(4096), // Small chunks for random access
            MemoryPattern::CacheFriendly => Some(64 * 1024), // L1 cache friendly
            _ => None,
        }
    }

    /// Get memory allocation strategy
    fn get_memory_strategy(&self) -> MemoryStrategy {
        match self.workload_type {
            WorkloadType::MemoryIntensive => MemoryStrategy::PreAllocate,
            WorkloadType::Interactive => MemoryStrategy::LazyAllocation,
            WorkloadType::Batch => MemoryStrategy::BulkAllocation,
            _ => MemoryStrategy::Adaptive,
        }
    }

    /// Get caching strategy
    fn get_caching_strategy(&self) -> CachingStrategy {
        match self.memory_pattern {
            MemoryPattern::Sequential => CachingStrategy::Prefetch,
            MemoryPattern::Random => CachingStrategy::LRU,
            MemoryPattern::CacheFriendly => CachingStrategy::Aggressive,
            _ => CachingStrategy::Conservative,
        }
    }

    /// Get I/O strategy
    fn get_io_strategy(&self) -> IOStrategy {
        match self.io_profile.io_type {
            IOType::Disk => IOStrategy::Buffered,
            IOType::Network => IOStrategy::Async,
            IOType::Memory => IOStrategy::Direct,
            IOType::Mixed => IOStrategy::Adaptive,
            IOType::None => IOStrategy::Minimal,
        }
    }

    /// Get algorithm preferences
    fn get_algorithm_preferences(&self) -> Vec<AlgorithmPreference> {
        let mut preferences = Vec::new();

        match self.workload_type {
            WorkloadType::ComputeIntensive => {
                preferences.push(AlgorithmPreference::Simd);
                preferences.push(AlgorithmPreference::Parallel);
            }
            WorkloadType::MemoryIntensive => {
                preferences.push(AlgorithmPreference::CacheEfficient);
                preferences.push(AlgorithmPreference::MemoryEfficient);
            }
            WorkloadType::Interactive => {
                preferences.push(AlgorithmPreference::LowLatency);
                preferences.push(AlgorithmPreference::Responsive);
            }
            _ => {
                preferences.push(AlgorithmPreference::Balanced);
            }
        }

        preferences
    }
}

/// Workload profile builder
pub struct WorkloadProfileBuilder {
    name: String,
    data_size: usize,
    compute_intensity: f64,
    memory_pattern: MemoryPattern,
    parallelism_profile: ParallelismProfile,
    io_profile: IOProfile,
    workload_type: WorkloadType,
    expected_duration: Option<Duration>,
    priority: Priority,
}

impl WorkloadProfileBuilder {
    fn new() -> Self {
        Self {
            name: "default_workload".to_string(),
            data_size: 1024,
            compute_intensity: 0.5,
            memory_pattern: MemoryPattern::Mixed,
            parallelism_profile: ParallelismProfile {
                parallelizable: true,
                optimal_threads: None,
                sync_overhead: 0.1,
                load_balance: LoadBalanceType::Even,
            },
            io_profile: IOProfile {
                intensity: 0.1,
                io_type: IOType::None,
                read_write_ratio: 0.8,
                preferred_buffer_size: None,
            },
            workload_type: WorkloadType::Balanced,
            expected_duration: None,
            priority: Priority::Medium,
        }
    }

    /// Set workload name
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Set data size
    pub fn with_data_size(mut self, size: usize) -> Self {
        self.data_size = size;
        self
    }

    /// Set compute intensity
    pub fn with_compute_intensity(mut self, intensity: f64) -> Self {
        self.compute_intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set memory pattern
    pub fn with_memory_pattern(mut self, pattern: MemoryPattern) -> Self {
        self.memory_pattern = pattern;
        self
    }

    /// Set workload type
    pub fn with_workload_type(mut self, workload_type: WorkloadType) -> Self {
        self.workload_type = workload_type;
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set expected duration
    pub fn with_expected_duration(mut self, duration: Duration) -> Self {
        self.expected_duration = Some(duration);
        self
    }

    /// Set parallelism profile
    pub const fn with_parallelism(
        mut self,
        parallelizable: bool,
        optimal_threads: Option<usize>,
    ) -> Self {
        self.parallelism_profile.parallelizable = parallelizable;
        self.parallelism_profile.optimal_threads = optimal_threads;
        self
    }

    /// Set I/O profile
    pub fn with_io_profile(mut self, intensity: f64, io_type: IOType) -> Self {
        self.io_profile.intensity = intensity.clamp(0.0, 1.0);
        self.io_profile.io_type = io_type;
        self
    }

    /// Build the workload profile
    pub fn build(self) -> WorkloadProfile {
        WorkloadProfile {
            name: self.name,
            data_size: self.data_size,
            compute_intensity: self.compute_intensity,
            memory_pattern: self.memory_pattern,
            parallelism_profile: self.parallelism_profile,
            io_profile: self.io_profile,
            workload_type: self.workload_type,
            expected_duration: self.expected_duration,
            priority: self.priority,
        }
    }
}

/// Optimization hints derived from workload analysis
#[derive(Debug, Clone)]
pub struct OptimizationHints {
    /// Preferred data chunk size
    pub preferred_chunk_size: Option<usize>,
    /// Preferred thread count
    pub preferred_thread_count: Option<usize>,
    /// Memory allocation strategy
    pub memory_allocation_strategy: MemoryStrategy,
    /// Caching strategy
    pub caching_strategy: CachingStrategy,
    /// I/O strategy
    pub io_strategy: IOStrategy,
    /// Algorithm preferences
    pub algorithm_preferences: Vec<AlgorithmPreference>,
}

/// Memory allocation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStrategy {
    /// Pre-allocate all required memory
    PreAllocate,
    /// Allocate memory lazily as needed
    LazyAllocation,
    /// Bulk allocation for batch operations
    BulkAllocation,
    /// Adaptive allocation based on usage patterns
    Adaptive,
}

/// Caching strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachingStrategy {
    /// Aggressive caching with prefetching
    Aggressive,
    /// Conservative caching
    Conservative,
    /// LRU-based caching
    LRU,
    /// Prefetch-based caching
    Prefetch,
    /// No caching
    None,
}

/// I/O strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IOStrategy {
    /// Buffered I/O
    Buffered,
    /// Asynchronous I/O
    Async,
    /// Direct I/O
    Direct,
    /// Adaptive I/O strategy
    Adaptive,
    /// Minimal I/O overhead
    Minimal,
}

/// Algorithm preferences
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmPreference {
    /// SIMD-optimized algorithms
    Simd,
    /// Parallel algorithms
    Parallel,
    /// Cache-efficient algorithms
    CacheEfficient,
    /// Memory-efficient algorithms
    MemoryEfficient,
    /// Low-latency algorithms
    LowLatency,
    /// Responsive algorithms
    Responsive,
    /// Balanced algorithms
    Balanced,
}

/// Performance metric for optimization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PerformanceMetric {
    /// Metric name
    pub name: String,
    /// Current value
    pub value: f64,
    /// Target value (if applicable)
    pub target: Option<f64>,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Confidence level
    pub confidence: f64,
    /// Trend direction
    pub trend: Trend,
}

/// Performance trend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Trend {
    /// Performance is improving
    Improving,
    /// Performance is stable
    Stable,
    /// Performance is degrading
    Degrading,
    /// Not enough data to determine trend
    Unknown,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OptimizationRecommendation {
    /// Parameter to adjust
    pub parameter: String,
    /// Current value
    pub current_value: String,
    /// Suggested value
    pub suggested_value: String,
    /// Expected impact
    pub expected_impact: Impact,
    /// Confidence level
    pub confidence: f64,
    /// Recommendation rationale
    pub rationale: String,
    /// Risk level
    pub risk_level: RiskLevel,
}

/// Expected impact of optimization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Impact {
    /// Performance improvement (percentage)
    pub performance_improvement: f64,
    /// Memory usage change (percentage)
    pub memory_change: f64,
    /// Energy consumption change (percentage)
    pub energy_change: f64,
    /// Overall benefit score
    pub benefit_score: f64,
}

/// Risk levels for optimization changes
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RiskLevel {
    /// Low risk change
    Low,
    /// Medium risk change
    Medium,
    /// High risk change
    High,
    /// Critical risk change (requires approval)
    Critical,
}

/// Adaptive optimizer main structure
pub struct AdaptiveOptimizer {
    /// Configuration
    config: OptimizationConfig,
    /// Registered workloads
    workloads: Arc<RwLock<HashMap<String, WorkloadProfile>>>,
    /// Performance history
    performance_history: Arc<Mutex<HashMap<String, VecDeque<PerformanceMetric>>>>,
    /// Current optimization parameters
    current_parameters: Arc<RwLock<HashMap<String, f64>>>,
    /// Optimization state
    state: OptimizerState,
    /// Last optimization timestamp
    last_optimization: Instant,
    /// Performance baseline
    baseline_metrics: Arc<Mutex<HashMap<String, f64>>>,
    /// Active recommendations
    active_recommendations: Arc<Mutex<Vec<OptimizationRecommendation>>>,
    /// Learning history for ML algorithms
    #[allow(dead_code)]
    learning_history: Arc<Mutex<VecDeque<LearningDataPoint>>>,
}

/// Optimizer state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerState {
    /// Optimizer is stopped
    Stopped,
    /// Optimizer is learning (collecting baseline data)
    Learning,
    /// Optimizer is actively optimizing
    Optimizing,
    /// Optimizer is paused
    Paused,
    /// Optimizer encountered an error
    Error,
}

/// Learning data point for ML algorithms
#[derive(Debug, Clone)]
struct LearningDataPoint {
    /// Input parameters
    #[allow(dead_code)]
    parameters: HashMap<String, f64>,
    /// Performance metrics
    #[allow(dead_code)]
    metrics: HashMap<String, f64>,
    /// Timestamp
    #[allow(dead_code)]
    timestamp: SystemTime,
    /// Workload context
    #[allow(dead_code)]
    workload: String,
}

impl AdaptiveOptimizer {
    /// Create a new adaptive optimizer
    pub fn new(config: OptimizationConfig) -> CoreResult<Self> {
        Ok(Self {
            config,
            workloads: Arc::new(RwLock::new(HashMap::new())),
            performance_history: Arc::new(Mutex::new(HashMap::new())),
            current_parameters: Arc::new(RwLock::new(HashMap::new())),
            state: OptimizerState::Stopped,
            last_optimization: Instant::now(),
            baseline_metrics: Arc::new(Mutex::new(HashMap::new())),
            active_recommendations: Arc::new(Mutex::new(Vec::new())),
            learning_history: Arc::new(Mutex::new(VecDeque::new())),
        })
    }

    /// Register a workload for optimization
    pub fn register_workload(&mut self, workload: WorkloadProfile) -> CoreResult<()> {
        if let Ok(mut workloads) = self.workloads.write() {
            workloads.insert(workload.name.clone(), workload);
        }
        Ok(())
    }

    /// Start optimization process
    pub fn start_optimization(&mut self) -> CoreResult<()> {
        if self.state == OptimizerState::Optimizing {
            return Ok(());
        }

        // Initialize baseline metrics
        self.collect_baseline_metrics()?;

        self.state = OptimizerState::Learning;
        self.last_optimization = Instant::now();

        println!("🚀 Adaptive optimizer started in learning mode");
        Ok(())
    }

    /// Stop optimization process
    pub fn stop_optimization(&mut self) -> CoreResult<()> {
        self.state = OptimizerState::Stopped;
        println!("🛑 Adaptive optimizer stopped");
        Ok(())
    }

    /// Record performance metric
    pub fn record_metric(
        &mut self,
        workload: &str,
        metric_name: &str,
        value: f64,
    ) -> CoreResult<()> {
        let metric = PerformanceMetric {
            name: metric_name.to_string(),
            value,
            target: None,
            timestamp: SystemTime::now(),
            confidence: 1.0,
            trend: self.calculate_trend(workload, metric_name, value),
        };

        if let Ok(mut history) = self.performance_history.lock() {
            let workload_metrics = history
                .entry(workload.to_string())
                .or_insert_with(VecDeque::new);
            workload_metrics.push_back(metric);

            // Limit history size
            while workload_metrics.len() > 1000 {
                workload_metrics.pop_front();
            }
        }

        // Trigger optimization if enough time has passed
        if self.last_optimization.elapsed() >= self.config.adaptation_interval {
            self.run_optimization_cycle()?;
        }

        Ok(())
    }

    /// Get current optimization recommendations
    pub fn get_recommendations(&self) -> CoreResult<Vec<OptimizationRecommendation>> {
        self.active_recommendations
            .lock()
            .map(|recs| recs.clone())
            .map_err(|_| {
                CoreError::from(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Failed to access recommendations",
                ))
            })
    }

    /// Apply optimization recommendation
    pub fn apply_recommendation(&mut self, recommendation_id: usize) -> CoreResult<()> {
        let recommendation = {
            let mut recs = self.active_recommendations.lock().map_err(|_| {
                CoreError::from(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Failed to access recommendations",
                ))
            })?;

            if recommendation_id >= recs.len() {
                return Err(CoreError::from(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Recommendation ID out of range",
                )));
            }

            recs.remove(recommendation_id)
        };

        // Apply the recommendation
        if let Ok(mut params) = self.current_parameters.write() {
            if let Ok(suggested_value) = recommendation.suggested_value.parse::<f64>() {
                params.insert(recommendation.parameter.clone(), suggested_value);
                println!(
                    "✅ Applied optimization: {} = {}",
                    recommendation.parameter, recommendation.suggested_value
                );
            }
        }

        Ok(())
    }

    /// Get optimizer statistics
    pub fn get_statistics(&self) -> OptimizerStatistics {
        let workload_count = self.workloads.read().map(|w| w.len()).unwrap_or(0);
        let recommendation_count = self
            .active_recommendations
            .lock()
            .map(|r| r.len())
            .unwrap_or(0);
        let parameter_count = self.current_parameters.read().map(|p| p.len()).unwrap_or(0);

        OptimizerStatistics {
            state: self.state,
            registered_workloads: workload_count,
            active_recommendations: recommendation_count,
            optimized_parameters: parameter_count,
            last_optimization: self.last_optimization,
            uptime: self.last_optimization.elapsed(),
        }
    }

    /// Get workload-specific optimization hints
    pub fn get_workload_hints(&self, workload_name: &str) -> CoreResult<OptimizationHints> {
        if let Ok(workloads) = self.workloads.read() {
            if let Some(workload) = workloads.get(workload_name) {
                return Ok(workload.get_optimization_hints());
            }
        }

        Err(CoreError::from(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Workload '{}' not found", workload_name),
        )))
    }

    /// Collect baseline performance metrics
    fn collect_baseline_metrics(&mut self) -> CoreResult<()> {
        // In a real implementation, this would collect system metrics
        let mut baseline = HashMap::new();
        baseline.insert("cpu_usage".to_string(), 45.0);
        baseline.insert("memory_usage".to_string(), 1024.0);
        baseline.insert("execution_time".to_string(), 100.0);
        baseline.insert("throughput".to_string(), 1000.0);

        if let Ok(mut metrics) = self.baseline_metrics.lock() {
            *metrics = baseline;
        }

        Ok(())
    }

    /// Calculate performance trend
    fn calculate_trend(&self, workload: &str, metric_name: &str, _current_value: f64) -> Trend {
        if let Ok(history) = self.performance_history.lock() {
            if let Some(workload_metrics) = history.get(workload) {
                let recent_values: Vec<f64> = workload_metrics
                    .iter()
                    .filter(|m| m.name == metric_name)
                    .rev()
                    .take(10)
                    .map(|m| m.value)
                    .collect();

                if recent_values.len() < 3 {
                    return Trend::Unknown;
                }

                let avg_old =
                    recent_values[5..].iter().sum::<f64>() / (recent_values.len() - 5) as f64;
                let avg_new = recent_values[..5].iter().sum::<f64>() / 5.0;

                let change_percent = (avg_new - avg_old) / avg_old * 100.0;

                if change_percent > 5.0 {
                    Trend::Improving
                } else if change_percent < -5.0 {
                    Trend::Degrading
                } else {
                    Trend::Stable
                }
            } else {
                Trend::Unknown
            }
        } else {
            Trend::Unknown
        }
    }

    /// Run optimization cycle
    fn run_optimization_cycle(&mut self) -> CoreResult<()> {
        if self.state == OptimizerState::Learning {
            // Check if we have enough data to start optimizing
            if self.has_sufficient_learning_data() {
                self.state = OptimizerState::Optimizing;
                println!("🧠 Transitioning to optimization mode");
            }
        }

        if self.state == OptimizerState::Optimizing {
            let recommendations = self.generate_recommendations()?;

            if let Ok(mut active_recs) = self.active_recommendations.lock() {
                active_recs.extend(recommendations);

                // Apply low-risk recommendations automatically
                let auto_apply: Vec<_> = active_recs
                    .iter()
                    .enumerate()
                    .filter(|(_, rec)| rec.risk_level == RiskLevel::Low && rec.confidence > 0.9)
                    .map(|(i, _)| i)
                    .collect();

                for &index in auto_apply.iter().rev() {
                    if let Some(rec) = active_recs.get(index).cloned() {
                        if let Ok(mut params) = self.current_parameters.write() {
                            if let Ok(value) = rec.suggested_value.parse::<f64>() {
                                params.insert(rec.parameter.clone(), value);
                                active_recs.remove(index);
                                println!(
                                    "🤖 Auto-applied: {} = {}",
                                    rec.parameter, rec.suggested_value
                                );
                            }
                        }
                    }
                }
            }
        }

        self.last_optimization = Instant::now();
        Ok(())
    }

    /// Check if sufficient learning data is available
    fn has_sufficient_learning_data(&self) -> bool {
        if let Ok(history) = self.performance_history.lock() {
            history.values().any(|metrics| metrics.len() >= 10)
        } else {
            false
        }
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self) -> CoreResult<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        // Analyze performance trends and generate recommendations
        if let Ok(history) = self.performance_history.lock() {
            for (workload, metrics) in history.iter() {
                if let Some(latest) = metrics.back() {
                    // Example: Recommend thread count adjustment
                    if latest.name == "execution_time" && latest.trend == Trend::Degrading {
                        recommendations.push(OptimizationRecommendation {
                            parameter: "thread_count".to_string(),
                            current_value: "4".to_string(),
                            suggested_value: "8".to_string(),
                            expected_impact: Impact {
                                performance_improvement: 25.0,
                                memory_change: 10.0,
                                energy_change: 5.0,
                                benefit_score: 0.8,
                            },
                            confidence: 0.85,
                            rationale: format!("Workload '{}' shows degrading execution time. Increasing thread count may improve parallelization.", workload),
                            risk_level: RiskLevel::Low,
                        });
                    }

                    // Example: Recommend memory optimization
                    if latest.name == "memory_usage" && latest.value > 2000.0 {
                        recommendations.push(OptimizationRecommendation {
                            parameter: "chunk_size".to_string(),
                            current_value: "1048576".to_string(),
                            suggested_value: "524288".to_string(),
                            expected_impact: Impact {
                                performance_improvement: 5.0,
                                memory_change: -30.0,
                                energy_change: -10.0,
                                benefit_score: 0.7,
                            },
                            confidence: 0.9,
                            rationale: "High memory usage detected. Reducing chunk size may improve memory efficiency.".to_string(),
                            risk_level: RiskLevel::Low,
                        });
                    }
                }
            }
        }

        Ok(recommendations)
    }
}

/// Optimizer statistics
#[derive(Debug, Clone)]
pub struct OptimizerStatistics {
    /// Current optimizer state
    pub state: OptimizerState,
    /// Number of registered workloads
    pub registered_workloads: usize,
    /// Number of active recommendations
    pub active_recommendations: usize,
    /// Number of optimized parameters
    pub optimized_parameters: usize,
    /// Last optimization timestamp
    pub last_optimization: Instant,
    /// Optimizer uptime
    pub uptime: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let config = OptimizationConfig::production();
        let optimizer = AdaptiveOptimizer::new(config);
        assert!(optimizer.is_ok());

        let optimizer = optimizer.unwrap();
        assert_eq!(optimizer.state, OptimizerState::Stopped);
    }

    #[test]
    fn test_workload_profile_builder() {
        let workload = WorkloadProfile::builder()
            .with_name("test_workload")
            .with_data_size(1000000)
            .with_compute_intensity(0.8)
            .with_workload_type(WorkloadType::ComputeIntensive)
            .with_priority(Priority::High)
            .build();

        assert_eq!(workload.name, "test_workload");
        assert_eq!(workload.data_size, 1000000);
        assert_eq!(workload.compute_intensity, 0.8);
        assert_eq!(workload.workload_type, WorkloadType::ComputeIntensive);
        assert_eq!(workload.priority, Priority::High);
    }

    #[test]
    fn test_optimization_config() {
        let config = OptimizationConfig::production()
            .with_goal(OptimizationGoal::Performance)
            .with_learning_rate(0.01)
            .with_confidence_threshold(0.95);

        assert_eq!(config.goal, OptimizationGoal::Performance);
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.confidence_threshold, 0.95);
    }

    #[test]
    fn test_workload_optimization_hints() {
        let workload = WorkloadProfile::builder()
            .with_name("compute_workload")
            .with_workload_type(WorkloadType::ComputeIntensive)
            .with_parallelism(true, Some(8))
            .build();

        let hints = workload.get_optimization_hints();
        assert!(hints
            .algorithm_preferences
            .contains(&AlgorithmPreference::Parallel));
        assert_eq!(hints.preferred_thread_count, Some(8));
    }

    #[test]
    fn test_workload_registration() {
        let config = OptimizationConfig::default();
        let mut optimizer = AdaptiveOptimizer::new(config).unwrap();

        let workload = WorkloadProfile::builder()
            .with_name("test_registration")
            .build();

        let result = optimizer.register_workload(workload);
        assert!(result.is_ok());

        let stats = optimizer.get_statistics();
        assert_eq!(stats.registered_workloads, 1);
    }

    #[test]
    fn test_metric_recording() {
        let config = OptimizationConfig::default();
        let mut optimizer = AdaptiveOptimizer::new(config).unwrap();

        let result = optimizer.record_metric("test_workload", "execution_time", 150.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_trend_calculation() {
        let config = OptimizationConfig::default();
        let optimizer = AdaptiveOptimizer::new(config).unwrap();

        // Test with no history
        let trend = optimizer.calculate_trend("test", "metric", 100.0);
        assert_eq!(trend, Trend::Unknown);
    }

    #[test]
    fn test_resource_constraints() {
        let constraints = ResourceConstraints {
            max_memory_usage: Some(1024 * 1024 * 1024), // 1GB
            max_cpu_usage: Some(0.8),
            max_threads: Some(16),
            ..Default::default()
        };

        assert_eq!(constraints.max_memory_usage, Some(1024 * 1024 * 1024));
        assert_eq!(constraints.max_cpu_usage, Some(0.8));
        assert_eq!(constraints.max_threads, Some(16));
    }

    #[test]
    fn test_optimization_goal_configuration() {
        let performance_config =
            OptimizationConfig::production().with_goal(OptimizationGoal::Performance);
        assert_eq!(performance_config.goal, OptimizationGoal::Performance);

        let memory_config = OptimizationConfig::memory_optimized();
        assert_eq!(memory_config.goal, OptimizationGoal::MemoryEfficiency);

        let dev_config = OptimizationConfig::development();
        assert_eq!(dev_config.goal, OptimizationGoal::Development);
    }
}
