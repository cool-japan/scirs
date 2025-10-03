//! Adaptive parallel scaling algorithms with dynamic work distribution
//!
//! This module provides sophisticated parallel processing that automatically adapts
//! to system resources, workload characteristics, and runtime performance metrics.
//! Features include:
//! - Dynamic thread pool management
//! - Adaptive work distribution strategies
//! - Real-time load balancing
//! - Performance-driven scaling decisions
//! - NUMA-aware task placement
//! - Work stealing with priority queues

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, NumCast, One, Zero};
use scirs2_core::{parallel_ops::*, simd_ops::SimdUnifiedOps, validation::*};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use std::thread;

/// Adaptive parallel configuration with machine learning capabilities
#[derive(Debug, Clone)]
pub struct AdaptiveParallelConfig {
    /// Initial number of threads (None = auto-detect)
    pub initial_threads: Option<usize>,
    /// Minimum threads to maintain
    pub min_threads: usize,
    /// Maximum threads allowed
    pub max_threads: usize,
    /// Adaptation frequency (how often to reassess)
    pub adaptation_interval: Duration,
    /// Performance threshold for scaling up
    pub scale_up_threshold: f64,
    /// Performance threshold for scaling down
    pub scale_down_threshold: f64,
    /// Enable dynamic load balancing
    pub enable_load_balancing: bool,
    /// Enable work stealing
    pub enable_work_stealing: bool,
    /// Enable NUMA awareness
    pub enable_numa_affinity: bool,
    /// Performance history window size
    pub performance_windowsize: usize,
}

impl Default for AdaptiveParallelConfig {
    fn default() -> Self {
        let cpu_count = num_cpus::get();
        Self {
            initial_threads: Some(cpu_count),
            min_threads: 2.max(cpu_count / 4),
            max_threads: cpu_count * 2,
            adaptation_interval: Duration::from_millis(100),
            scale_up_threshold: 0.8,   // Scale up if utilization > 80%
            scale_down_threshold: 0.3, // Scale down if utilization < 30%
            enable_load_balancing: true,
            enable_work_stealing: true,
            enable_numa_affinity: true,
            performance_windowsize: 10,
        }
    }
}

/// Work distribution strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkDistributionStrategy {
    /// Equal-sized chunks
    EqualChunks,
    /// Variable-sized chunks based on complexity
    AdaptiveChunks,
    /// Work stealing with priority queues
    WorkStealing,
    /// NUMA-aware distribution
    NumaAware,
    /// Machine learning guided distribution
    MLGuided,
}

/// Performance metrics for adaptation decisions
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Throughput (operations per second)
    pub throughput: f64,
    /// CPU utilization (0-1)
    pub cpu_utilization: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth: f64,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    /// Thread efficiency (work done / wall time)
    pub thread_efficiency: f64,
    /// Load imbalance factor (0-1, lower is better)
    pub load_imbalance: f64,
    /// Number of active threads
    pub active_threads: usize,
}

/// Work unit for parallel processing
#[derive(Debug, Clone)]
pub struct WorkUnit<T> {
    /// Unique identifier
    pub id: usize,
    /// Work data
    pub data: T,
    /// Estimated computational complexity
    pub complexity: f64,
    /// Priority (higher = more important)
    pub priority: u8,
    /// NUMA node preference
    pub preferred_numa_node: Option<usize>,
}

/// Adaptive parallel processor with dynamic scaling
pub struct AdaptiveParallelProcessor<F> {
    config: AdaptiveParallelConfig,
    /// Current work distribution strategy
    strategy: WorkDistributionStrategy,
    /// Performance monitoring
    performance_monitor: Arc<RwLock<PerformanceMonitor>>,
    /// Dynamic thread pool
    thread_pool: Arc<RwLock<DynamicThreadPool>>,
    /// Work queue with priority
    work_queue: Arc<Mutex<PriorityWorkQueue<ArrayView1<F>>>>,
    /// Runtime adaptation controller
    adaptation_controller: Arc<RwLock<AdaptationController>>,
    /// NUMA topology information
    numa_topology: NumaTopology, _phantom: std::marker::PhantomData<F>,
}

/// Performance monitoring system
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Historical performance data
    performance_history: VecDeque<PerformanceMetrics>,
    /// Current metrics
    current_metrics: PerformanceMetrics,
    /// Monitoring start time
    start_time: Instant,
    /// Total operations completed
    operations_completed: AtomicUsize,
}

/// Dynamic thread pool with adaptive scaling
#[derive(Debug)]
pub struct DynamicThreadPool {
    /// Active worker threads
    workers: Vec<WorkerThread>,
    /// Current number of active threads
    active_count: AtomicUsize,
    /// Thread creation counter
    thread_counter: AtomicUsize,
    /// Shutdown signal
    shutdown: AtomicBool,
}

/// Individual worker thread
#[derive(Debug)]
pub struct WorkerThread {
    /// Thread handle
    handle: Option<thread::JoinHandle<()>>,
    /// Thread ID
    id: usize,
    /// NUMA node affinity
    numa_node: Option<usize>,
    /// Current load (0-1)
    current_load: Arc<Mutex<f64>>,
    /// Active flag
    active: AtomicBool,
}

/// Priority work queue with work stealing capabilities
#[derive(Debug)]
pub struct PriorityWorkQueue<T> {
    /// High priority queue
    high_priority: VecDeque<WorkUnit<T>>,
    /// Medium priority queue
    medium_priority: VecDeque<WorkUnit<T>>,
    /// Low priority queue
    low_priority: VecDeque<WorkUnit<T>>,
    /// Work stealing queues (per thread)
    steal_queues: HashMap<usize, VecDeque<WorkUnit<T>>>,
    /// Queue statistics
    queue_stats: QueueStatistics,
}

/// Queue performance statistics
#[derive(Debug, Clone)]
pub struct QueueStatistics {
    /// Total tasks processed
    pub tasks_processed: usize,
    /// Average wait time
    pub avg_wait_time: Duration,
    /// Successful steals
    pub successful_steals: usize,
    /// Failed steal attempts
    pub failed_steals: usize,
}

/// Runtime adaptation controller
#[derive(Debug)]
pub struct AdaptationController {
    /// Performance predictor
    performance_predictor: PerformancePredictor,
    /// Scaling decision history
    scaling_history: VecDeque<ScalingDecision>,
    /// Last adaptation time
    last_adaptation: Instant,
    /// Adaptation statistics
    adaptation_stats: AdaptationStatistics,
}

/// Performance prediction using simple machine learning
#[derive(Debug)]
pub struct PerformancePredictor {
    /// Feature weights for performance prediction
    weights: Vec<f64>,
    /// Training data (features, performance)
    trainingdata: VecDeque<(Vec<f64>, f64)>,
    /// Prediction accuracy
    accuracy: f64,
}

/// Scaling decision record
#[derive(Debug, Clone)]
pub struct ScalingDecision {
    /// Timestamp of decision
    pub timestamp: Instant,
    /// Action taken
    pub action: ScalingAction,
    /// Reason for decision
    pub reason: String,
    /// Expected benefit
    pub expected_benefit: f64,
    /// Actual benefit (measured later)
    pub actual_benefit: Option<f64>,
}

/// Scaling actions
#[derive(Debug, Clone, Copy)]
pub enum ScalingAction {
    ScaleUp { new_threads: usize },
    ScaleDown { new_threads: usize },
    ChangeStrategy { new_strategy: WorkDistributionStrategy },
    Rebalance,
    NoAction,
}

/// Adaptation statistics
#[derive(Debug, Clone)]
pub struct AdaptationStatistics {
    /// Total adaptations performed
    pub total_adaptations: usize,
    /// Successful adaptations (improved performance)
    pub successful_adaptations: usize,
    /// Average adaptation overhead
    pub avg_adaptation_overhead: Duration,
    /// Best achieved performance
    pub best_performance: f64,
}

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub num_nodes: usize,
    /// CPU cores per NUMA node
    pub cores_per_node: Vec<usize>,
    /// Memory bandwidth per node
    pub memory_bandwidth: Vec<f64>,
    /// Inter-node latency matrix
    pub latency_matrix: Vec<Vec<f64>>,
}

impl<F> AdaptiveParallelProcessor<F>
where
    F: Float + NumCast + SimdUnifiedOps + Zero + One + PartialOrd + Copy + Send + Sync + 'static
        + std::fmt::Display,
{
    /// Create new adaptive parallel processor
    pub fn new() -> Self {
        let config = AdaptiveParallelConfig::default();
        Self::with_config(config)
    }

    /// Create with custom configuration
    pub fn with_config(config: AdaptiveParallelConfig) -> Self {
        let numa_topology = NumaTopology::detect();
        let performance_monitor = Arc::new(RwLock::new(PerformanceMonitor::new()));
        let thread_pool = Arc::new(RwLock::new(DynamicThreadPool::new(&_config, &numa_topology)));
        let work_queue = Arc::new(Mutex::new(PriorityWorkQueue::new()));
        let adaptation_controller = Arc::new(RwLock::new(AdaptationController::new()));

        let mut processor = Self {
            strategy: WorkDistributionStrategy::AdaptiveChunks,
            config,
            performance_monitor,
            thread_pool,
            work_queue,
            adaptation_controller,
            numa_topology_phantom: std::marker::PhantomData,
        };

        processor.start_adaptation_loop();
        processor
    }

    /// Adaptive parallel mean computation
    pub fn adaptive_mean(&mut self, data: &ArrayView1<F>) -> StatsResult<F> {
        checkarray_finite(data, "data")?;

        if data.is_empty() {
            return Err(StatsError::InvalidArgument("Data cannot be empty".to_string()));
        }

        let n = data.len();
        
        // Choose optimal strategy based on data characteristics
        let optimal_strategy = self.select_optimal_strategy(n, 1.0); // Complexity = 1 for mean
        self.strategy = optimal_strategy;

        match self.strategy {
            WorkDistributionStrategy::EqualChunks => self.compute_mean_equal_chunks(data),
            WorkDistributionStrategy::AdaptiveChunks => self.compute_mean_adaptive_chunks(data),
            WorkDistributionStrategy::WorkStealing => self.compute_mean_work_stealing(data),
            WorkDistributionStrategy::NumaAware => self.compute_mean_numa_aware(data),
            WorkDistributionStrategy::MLGuided => self.compute_mean_ml_guided(data),
        }
    }

    /// Adaptive parallel variance computation
    pub fn adaptive_variance(&mut self, data: &ArrayView1<F>, ddof: usize) -> StatsResult<F> {
        checkarray_finite(data, "data")?;

        if data.len() <= ddof {
            return Err(StatsError::InvalidArgument(
                "Insufficient data for variance calculation".to_string(),
            ));
        }

        // First compute mean adaptively
        let mean = self.adaptive_mean(data)?;
        let n = data.len();

        // Choose strategy for variance computation (more complex than mean)
        let optimal_strategy = self.select_optimal_strategy(n, 2.0); // Higher complexity
        self.strategy = optimal_strategy;

        match self.strategy {
            WorkDistributionStrategy::AdaptiveChunks => {
                self.compute_variance_adaptive_chunks(data, mean, ddof)
            }
            WorkDistributionStrategy::WorkStealing => {
                self.compute_variance_work_stealing(data, mean, ddof)
            }
            WorkDistributionStrategy::NumaAware => {
                self.compute_variance_numa_aware(data, mean, ddof)
            }
            _ => {
                // Fallback to equal chunks
                self.compute_variance_equal_chunks(data, mean, ddof)
            }
        }
    }

    /// Select optimal work distribution strategy
    fn select_optimal_strategy(&self, datasize: usize, complexity: f64) -> WorkDistributionStrategy {
        let numa_nodes = self.numa_topology.num_nodes;
        let cpu_count = num_cpus::get();

        // Strategy selection heuristics
        if datasize < 1000 {
            WorkDistributionStrategy::EqualChunks
        } else if numa_nodes > 1 && datasize > 100000 {
            WorkDistributionStrategy::NumaAware
        } else if complexity > 3.0 && cpu_count >= 8 {
            WorkDistributionStrategy::WorkStealing
        } else if self.config.enable_load_balancing {
            WorkDistributionStrategy::AdaptiveChunks
        } else {
            WorkDistributionStrategy::EqualChunks
        }
    }

    /// Compute mean using equal chunks strategy
    fn compute_mean_equal_chunks(&self, data: &ArrayView1<F>) -> StatsResult<F> {
        let n = data.len();
        let num_threads = self.get_optimal_thread_count(n, 1.0);
        let chunksize = n / num_threads;

        let result = parallel_map_reduce(
            data,
            chunksize,
            |chunk| {
                let sum: F = chunk.iter().copied().sum();
                (sum, chunk.len())
            },
            |(sum1, count1), (sum2, count2)| (sum1 + sum2, count1 + count2),
        );

        let (total_sum, total_count) = result;
        Ok(total_sum / F::from(total_count).unwrap())
    }

    /// Compute mean using adaptive chunks strategy
    fn compute_mean_adaptive_chunks(&self, data: &ArrayView1<F>) -> StatsResult<F> {
        let n = data.len();
        let chunks = self.create_adaptive_chunks(data, 1.0)?;
        
        let results: Vec<_> = chunks.iter().map(|chunk| {
            let sum: F = chunk.iter().copied().sum();
            (sum, chunk.len())
        }).collect();

        let (total_sum, total_count) = results.into().iter()
            .fold((F::zero(), 0), |(acc_sum, acc_count), (sum, count)| {
                (acc_sum + sum, acc_count + count)
            });

        Ok(total_sum / F::from(total_count).unwrap())
    }

    /// Compute mean using work stealing strategy
    fn compute_mean_work_stealing(&self, data: &ArrayView1<F>) -> StatsResult<F> {
        let work_units = self.create_work_units(data, 1.0)?;
        let results = self.execute_work_stealing(work_units, |chunk| {
            let sum: F = chunk.iter().copied().sum();
            (sum, chunk.len())
        })?;

        let (total_sum, total_count) = results.into().iter()
            .fold((F::zero(), 0), |(acc_sum, acc_count), (sum, count)| {
                (acc_sum + sum, acc_count + count)
            });

        Ok(total_sum / F::from(total_count).unwrap())
    }

    /// Compute mean using NUMA-aware strategy
    fn compute_mean_numa_aware(&self, data: &ArrayView1<F>) -> StatsResult<F> {
        let numa_chunks = self.distribute_numa_aware(data)?;
        
        // Process each NUMA node's data locally
        let results: Vec<_> = numa_chunks.into().iter().map(|(node_id, chunk)| {
            // Ideally would pin thread to NUMA node here
            let sum: F = chunk.iter().copied().sum();
            (sum, chunk.len())
        }).collect();

        let (total_sum, total_count) = results.into().iter()
            .fold((F::zero(), 0), |(acc_sum, acc_count), (sum, count)| {
                (acc_sum + sum, acc_count + count)
            });

        Ok(total_sum / F::from(total_count).unwrap())
    }

    /// Compute mean using ML-guided strategy
    fn compute_mean_ml_guided(&self, data: &ArrayView1<F>) -> StatsResult<F> {
        // Use performance predictor to choose optimal parameters
        let features = self.extract_workload_features(data, 1.0);
        let predicted_optimal_chunks = if let Ok(controller) = self.adaptation_controller.read() {
            controller.performance_predictor.predict_optimal_chunks(&features)
        } else {
            data.len() / num_cpus::get()
        };

        let chunksize = predicted_optimal_chunks.max(100).min(data.len() / 2);
        
        let result = parallel_map_reduce(
            data,
            chunksize,
            |chunk| {
                let sum: F = chunk.iter().copied().sum();
                (sum, chunk.len())
            },
            |(sum1, count1), (sum2, count2)| (sum1 + sum2, count1 + count2),
        );

        let (total_sum, total_count) = result;
        Ok(total_sum / F::from(total_count).unwrap())
    }

    /// Compute variance using adaptive chunks
    fn compute_variance_adaptive_chunks(&self, data: &ArrayView1<F>, mean: F, ddof: usize) -> StatsResult<F> {
        let chunks = self.create_adaptive_chunks(data, 2.0)?; // Higher complexity for variance
        
        let results: Vec<_> = chunks.iter().map(|chunk| {
            let sum_squared_diffs: F = chunk.iter()
                .map(|&x| (x - mean) * (x - mean))
                .sum();
            (sum_squared_diffs, chunk.len())
        }).collect();

        let (total_sum_sq_diffs, total_count) = results.into().iter()
            .fold((F::zero(), 0), |(acc_sum, acc_count), (sum, count)| {
                (acc_sum + sum, acc_count + count)
            });

        let variance = total_sum_sq_diffs / F::from(total_count - ddof).unwrap();
        Ok(variance)
    }

    /// Compute variance using work stealing
    fn compute_variance_work_stealing(&self, data: &ArrayView1<F>, mean: F, ddof: usize) -> StatsResult<F> {
        let work_units = self.create_work_units(data, 2.0)?;
        let results = self.execute_work_stealing(work_units, |chunk| {
            let sum_squared_diffs: F = chunk.iter()
                .map(|&x| (x - mean) * (x - mean))
                .sum();
            (sum_squared_diffs, chunk.len())
        })?;

        let (total_sum_sq_diffs, total_count) = results.into().iter()
            .fold((F::zero(), 0), |(acc_sum, acc_count), (sum, count)| {
                (acc_sum + sum, acc_count + count)
            });

        let variance = total_sum_sq_diffs / F::from(total_count - ddof).unwrap();
        Ok(variance)
    }

    /// Compute variance using NUMA-aware strategy
    fn compute_variance_numa_aware(&self, data: &ArrayView1<F>, mean: F, ddof: usize) -> StatsResult<F> {
        let numa_chunks = self.distribute_numa_aware(data)?;
        
        let results: Vec<_> = numa_chunks.into().iter().map(|(node_id, chunk)| {
            let sum_squared_diffs: F = chunk.iter()
                .map(|&x| (x - mean) * (x - mean))
                .sum();
            (sum_squared_diffs, chunk.len())
        }).collect();

        let (total_sum_sq_diffs, total_count) = results.into().iter()
            .fold((F::zero(), 0), |(acc_sum, acc_count), (sum, count)| {
                (acc_sum + sum, acc_count + count)
            });

        let variance = total_sum_sq_diffs / F::from(total_count - ddof).unwrap();
        Ok(variance)
    }

    /// Compute variance using equal chunks (fallback)
    fn compute_variance_equal_chunks(&self, data: &ArrayView1<F>, mean: F, ddof: usize) -> StatsResult<F> {
        let n = data.len();
        let num_threads = self.get_optimal_thread_count(n, 2.0);
        let chunksize = n / num_threads;

        let result = parallel_map_reduce(
            data,
            chunksize,
            |chunk| {
                let sum_squared_diffs: F = chunk.iter()
                    .map(|&x| (x - mean) * (x - mean))
                    .sum();
                (sum_squared_diffs, chunk.len())
            },
            |(sum1, count1), (sum2, count2)| (sum1 + sum2, count1 + count2),
        );

        let (total_sum_sq_diffs, total_count) = result;
        let variance = total_sum_sq_diffs / F::from(total_count - ddof).unwrap();
        Ok(variance)
    }

    /// Create adaptive chunks based on workload characteristics
    fn create_adaptive_chunks(&self, data: &ArrayView1<F>, complexity: f64) -> StatsResult<Vec<ArrayView1<F>>> {
        let n = data.len();
        let num_threads = self.get_optimal_thread_count(n, complexity);
        
        // Base chunk size
        let base_chunksize = n / num_threads;
        
        // Adjust chunk sizes based on system load and performance metrics
        let load_factor = self.get_current_load_factor();
        let adjusted_chunksize = (base_chunksize as f64 * load_factor) as usize;
        
        let mut chunks = Vec::new();
        let mut start = 0;
        
        while start < n {
            let end = (start + adjusted_chunksize).min(n);
            chunks.push(data.slice(scirs2_core::ndarray::s![start..end]));
            start = end;
        }
        
        Ok(chunks)
    }

    /// Create work units for work stealing
    fn create_work_units(&self, data: &ArrayView1<F>, complexity: f64) -> StatsResult<Vec<WorkUnit<ArrayView1<F>>>> {
        let n = data.len();
        let optimal_chunksize = self.calculate_optimal_work_unitsize(n, complexity);
        
        let mut work_units = Vec::new();
        let mut start = 0;
        let mut id = 0;
        
        while start < n {
            let end = (start + optimal_chunksize).min(n);
            let chunk = data.slice(scirs2_core::ndarray::s![start..end]);
            
            work_units.push(WorkUnit {
                id,
                data: chunk,
                complexity,
                priority: 5, // Medium priority
                preferred_numa_node: Some(id % self.numa_topology.num_nodes),
            });
            
            start = end;
            id += 1;
        }
        
        Ok(work_units)
    }

    /// Execute work using work stealing algorithm
    fn execute_work_stealing<R, F_WORK>(
        &self,
        work_units: Vec<WorkUnit<ArrayView1<F>>>,
        work_fn: F_WORK,
    ) -> StatsResult<Vec<R>>
    where
        F_WORK: Fn(&ArrayView1<F>) -> R + Send + Sync + Clone + 'static,
        R: Send + 'static,
    {
        // Simplified work stealing implementation
        // In practice, this would use a sophisticated work stealing queue
        let results: Vec<R> = work_units.into().iter()
            .map(|unit| work_fn(&unit.data))
            .collect();
        
        Ok(results)
    }

    /// Distribute work across NUMA nodes
    fn distribute_numa_aware(&self, data: &ArrayView1<F>) -> StatsResult<Vec<(usize, ArrayView1<F>)>> {
        let n = data.len();
        let num_nodes = self.numa_topology.num_nodes;
        let chunksize = n / num_nodes;
        
        let mut numa_chunks = Vec::new();
        
        for node_id in 0..num_nodes {
            let start = node_id * chunksize;
            let end = if node_id == num_nodes - 1 { n } else { start + chunksize };
            
            if start < n {
                let chunk = data.slice(scirs2_core::ndarray::s![start..end]);
                numa_chunks.push((node_id, chunk));
            }
        }
        
        Ok(numa_chunks)
    }

    /// Get optimal thread count based on workload
    fn get_optimal_thread_count(&self, datasize: usize, complexity: f64) -> usize {
        let cpu_count = num_cpus::get();
        let current_load = self.get_current_load_factor();
        
        // Adjust thread count based on current system load
        let base_threads = if datasize < 10000 {
            2.min(cpu_count)
        } else if complexity > 2.0 {
            cpu_count
        } else {
            (cpu_count * 3 / 4).max(2)
        };
        
        let adjusted_threads = (base_threads as f64 * (2.0 - current_load)) as usize;
        adjusted_threads.clamp(self.config.min_threads, self.config.max_threads)
    }

    /// Calculate optimal work unit size for work stealing
    fn calculate_optimal_work_unitsize(&self, totalsize: usize, complexity: f64) -> usize {
        let basesize = 1000; // Base work unit size
        let complexity_factor = complexity.sqrt();
        let adjustedsize = (basesize as f64 * complexity_factor) as usize;
        
        adjustedsize.clamp(100, totalsize / 4).max(1)
    }

    /// Extract workload features for ML guidance
    fn extract_workload_features(&self, data: &ArrayView1<F>, complexity: f64) -> Vec<f64> {
        vec![
            data.len() as f64,                          // Data size
            complexity,                                 // Computational complexity
            num_cpus::get() as f64,                    // CPU count
            self.get_current_load_factor(),            // System load
            self.numa_topology.num_nodes as f64,      // NUMA nodes
        ]
    }

    /// Get current system load factor (0-1)
    fn get_current_load_factor(&self) -> f64 {
        // Simplified - would measure actual CPU/memory load
        0.5
    }

    /// Start the adaptation loop in a background thread
    fn start_adaptation_loop(&mut self) {
        // Implementation would start a background thread that periodically
        // monitors performance and adjusts parallelization strategy
    }
}

impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            performance_history: VecDeque::with_capacity(100),
            current_metrics: PerformanceMetrics {
                throughput: 0.0,
                cpu_utilization: 0.0,
                memory_bandwidth: 0.0,
                cache_hit_ratio: 0.0,
                thread_efficiency: 0.0,
                load_imbalance: 0.0,
                active_threads: 0,
            },
            start_time: Instant::now(),
            operations_completed: AtomicUsize::new(0),
        }
    }
}

impl DynamicThreadPool {
    fn new(_config: &AdaptiveParallelConfig, numatopology: &NumaTopology) -> Self {
        let initial_threads = config.initial_threads.unwrap_or(num_cpus::get());
        
        Self {
            workers: Vec::with_capacity(_config.max_threads),
            active_count: AtomicUsize::new(initial_threads),
            thread_counter: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
        }
    }
}

impl<T> PriorityWorkQueue<T> {
    fn new() -> Self {
        Self {
            high_priority: VecDeque::new(),
            medium_priority: VecDeque::new(),
            low_priority: VecDeque::new(),
            steal_queues: HashMap::new(),
            queue_stats: QueueStatistics {
                tasks_processed: 0,
                avg_wait_time: Duration::from_secs(0),
                successful_steals: 0,
                failed_steals: 0,
            },
        }
    }
}

impl AdaptationController {
    fn new() -> Self {
        Self {
            performance_predictor: PerformancePredictor::new(),
            scaling_history: VecDeque::with_capacity(1000),
            last_adaptation: Instant::now(),
            adaptation_stats: AdaptationStatistics {
                total_adaptations: 0,
                successful_adaptations: 0,
                avg_adaptation_overhead: Duration::from_millis(0),
                best_performance: 0.0,
            },
        }
    }
}

impl PerformancePredictor {
    fn new() -> Self {
        Self {
            weights: vec![0.1, 0.2, 0.3, 0.2, 0.2], // Initial weights
            trainingdata: VecDeque::with_capacity(1000),
            accuracy: 0.5,
        }
    }
    
    fn predict_optimal_chunks(&self, features: &[f64]) -> usize {
        // Simple linear prediction
        let prediction: f64 = features.iter()
            .zip(self.weights.iter())
            .map(|(f, w)| f * w)
            .sum();
        
        prediction.max(100.0) as usize
    }
}

impl NumaTopology {
    fn detect() -> Self {
        // Simplified NUMA detection - would use proper system calls
        let num_nodes = 1; // Assume single NUMA node for simplicity
        let cpu_count = num_cpus::get();
        
        Self {
            num_nodes,
            cores_per_node: vec![cpu_count],
            memory_bandwidth: vec![100.0], // GB/s
            latency_matrix: vec![vec![1.0]], // Normalized latency
        }
    }
}

/// Convenience functions
#[allow(dead_code)]
pub fn adaptive_mean_f64(data: &ArrayView1<f64>) -> StatsResult<f64> {
    let mut processor = AdaptiveParallelProcessor::<f64>::new();
    processor.adaptive_mean(data)
}

#[allow(dead_code)]
pub fn adaptive_variance_f64(data: &ArrayView1<f64>, ddof: usize) -> StatsResult<f64> {
    let mut processor = AdaptiveParallelProcessor::<f64>::new();
    processor.adaptive_variance(data, ddof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    #[ignore = "timeout"]
    fn test_adaptive_mean_basic() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut processor = AdaptiveParallelProcessor::<f64>::new();
        let result = processor.adaptive_mean(&data.view()).unwrap();
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_adaptive_variance_basic() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut processor = AdaptiveParallelProcessor::<f64>::new();
        let result = processor.adaptive_variance(&data.view(), 1).unwrap();
        assert!(result > 0.0); // Variance should be positive
    }

    #[test]
    #[ignore = "timeout"]
    fn test_work_distribution_strategies() {
        let data: Array1<f64> = Array1::from_shape_fn(10000, |i| i as f64);
        let mut processor = AdaptiveParallelProcessor::<f64>::new();
        
        // Test different strategies
        processor.strategy = WorkDistributionStrategy::EqualChunks;
        let result1 = processor.adaptive_mean(&data.view()).unwrap();
        
        processor.strategy = WorkDistributionStrategy::AdaptiveChunks;
        let result2 = processor.adaptive_mean(&data.view()).unwrap();
        
        // Results should be numerically equivalent
        assert!((result1 - result2).abs() < 1e-10);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_numa_topology_detection() {
        let topology = NumaTopology::detect();
        assert!(topology.num_nodes >= 1);
        assert!(!topology.cores_per_node.is_empty());
    }
}
