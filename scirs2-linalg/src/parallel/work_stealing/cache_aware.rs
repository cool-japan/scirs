//! Advanced Cache-Aware and NUMA-Aware Work-Stealing
//!
//! This module provides advanced work-stealing implementations that optimize for
//! cache locality, NUMA topology, and adaptive performance tuning.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, One, Zero};
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};
use std::iter::Sum;
use std::sync::{Arc, Mutex};

use super::core::{MatrixOperationType, WorkItem};
use super::scheduler::WorkStealingScheduler;

/// Cache-aware work-stealing scheduler with memory locality optimization
pub struct CacheAwareWorkStealer<T: Clone + Send + 'static> {
    /// Standard work-stealing scheduler
    #[allow(dead_code)]
    base_scheduler: WorkStealingScheduler<T>,
    /// Cache line size for optimization
    #[allow(dead_code)]
    cache_linesize: usize,
    /// Memory affinity mapping for workers
    worker_affinity: Vec<usize>,
    /// Cache miss rate tracking per worker
    cache_miss_rates: Arc<Mutex<Vec<f64>>>,
    /// NUMA node topology
    numa_topology: NumaTopology,
}

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub node_count: usize,
    /// CPUs per NUMA node
    pub cpus_per_node: Vec<Vec<usize>>,
    /// Memory bandwidth between nodes (relative)
    pub bandwidthmatrix: Array2<f64>,
    /// Latency between nodes (nanoseconds)
    pub latencymatrix: Array2<f64>,
}

impl NumaTopology {
    /// Create a default NUMA topology for systems without NUMA
    pub fn default_single_node() -> Self {
        let cpu_count = num_cpus::get();
        Self {
            node_count: 1,
            cpus_per_node: vec![(0..cpu_count).collect()],
            bandwidthmatrix: Array2::from_elem((1, 1), 1.0),
            latencymatrix: Array2::from_elem((1, 1), 0.0),
        }
    }

    /// Detect NUMA topology (simplified version)
    pub fn detect() -> Self {
        // This is a simplified implementation
        // In practice, you'd use system calls to detect actual NUMA topology
        let cpu_count = num_cpus::get();

        if cpu_count <= 4 {
            Self::default_single_node()
        } else {
            // Assume dual-socket system for larger CPU counts
            let nodes = 2;
            let cpus_per_socket = cpu_count / nodes;
            let mut cpus_per_node = Vec::new();

            for i in 0..nodes {
                let start = i * cpus_per_socket;
                let end = if i == nodes - 1 {
                    cpu_count
                } else {
                    (i + 1) * cpus_per_socket
                };
                cpus_per_node.push((start..end).collect());
            }

            // Default bandwidth and latency matrices for dual-socket
            let mut bandwidthmatrix = Array2::from_elem((nodes, nodes), 0.6); // Cross-node bandwidth
            let mut latencymatrix = Array2::from_elem((nodes, nodes), 100.0); // Cross-node latency

            for i in 0..nodes {
                bandwidthmatrix[[i, i]] = 1.0; // Local bandwidth
                latencymatrix[[i, i]] = 0.0; // Local latency
            }

            Self {
                node_count: nodes,
                cpus_per_node,
                bandwidthmatrix,
                latencymatrix,
            }
        }
    }
}

/// Cache-aware work distribution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheAwareStrategy {
    /// Distribute work to minimize cache misses
    LocalityFirst,
    /// Balance between locality and load balancing
    Balanced,
    /// Prioritize load balancing over locality
    LoadFirst,
    /// Adaptive strategy based on cache miss rates
    Adaptive,
}

impl<T: Clone + Send + 'static> CacheAwareWorkStealer<T> {
    /// Create a new cache-aware work stealer
    pub fn new(_num_workers: usize, strategy: CacheAwareStrategy) -> LinalgResult<Self> {
        let base_scheduler = WorkStealingScheduler::new(_num_workers);
        let numa_topology = NumaTopology::detect();

        // Assign _workers to NUMA nodes in round-robin fashion
        let mut worker_affinity = Vec::with_capacity(_num_workers);
        for i in 0.._num_workers {
            let node = i % numa_topology.node_count;
            let cpu_idx = i / numa_topology.node_count;
            let cpu = numa_topology.cpus_per_node[node]
                .get(cpu_idx)
                .copied()
                .unwrap_or(numa_topology.cpus_per_node[node][0]);
            worker_affinity.push(cpu);
        }

        Ok(Self {
            base_scheduler,
            cache_linesize: 64, // Common cache line size
            worker_affinity,
            cache_miss_rates: Arc::new(Mutex::new(vec![0.0; _num_workers])),
            numa_topology,
        })
    }

    /// Execute work with cache-aware distribution
    pub fn execute_cache_aware<F, R>(
        &self,
        work_items: Vec<WorkItem<T>>,
        worker_fn: F,
        strategy: CacheAwareStrategy,
    ) -> LinalgResult<Vec<R>>
    where
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + Clone + 'static,
    {
        let redistributed_work = self.redistribute_for_cache_locality(work_items, strategy)?;
        self.base_scheduler.submit_work(redistributed_work)?;
        self.base_scheduler.execute(worker_fn)
    }

    /// Redistribute work items to optimize cache locality
    fn redistribute_for_cache_locality(
        &self,
        mut work_items: Vec<WorkItem<T>>,
        strategy: CacheAwareStrategy,
    ) -> LinalgResult<Vec<WorkItem<T>>> {
        match strategy {
            CacheAwareStrategy::LocalityFirst => {
                // Sort work _items by estimated memory access patterns
                work_items.sort_by_key(|item| self.estimate_memory_footprint(&item.payload));
                Ok(work_items)
            }
            CacheAwareStrategy::Balanced => {
                // Interleave local and distributed work
                let chunksize = work_items.len() / self.numa_topology.node_count;
                let mut redistributed = Vec::new();

                for node in 0..self.numa_topology.node_count {
                    let start = node * chunksize;
                    let end = if node == self.numa_topology.node_count - 1 {
                        work_items.len()
                    } else {
                        (node + 1) * chunksize
                    };

                    redistributed.extend(work_items.drain(start..end));
                }

                Ok(redistributed)
            }
            CacheAwareStrategy::LoadFirst => {
                // Use standard load balancing
                Ok(work_items)
            }
            CacheAwareStrategy::Adaptive => {
                // Choose strategy based on current cache miss rates
                let miss_rates = self.cache_miss_rates.lock().unwrap();
                let avg_miss_rate: f64 = miss_rates.iter().sum::<f64>() / miss_rates.len() as f64;

                if avg_miss_rate > 0.1 {
                    // High miss rate - prioritize locality
                    drop(miss_rates);
                    self.redistribute_for_cache_locality(
                        work_items,
                        CacheAwareStrategy::LocalityFirst,
                    )
                } else {
                    // Low miss rate - prioritize load balancing
                    Ok(work_items)
                }
            }
        }
    }

    /// Estimate memory footprint of work item (simplified)
    fn estimate_memory_footprint(&self, payload: &T) -> usize {
        // This is a placeholder - in practice you'd analyze the _payload
        // to estimate its memory access pattern
        64 // Default cache line size
    }

    /// Update cache miss rate for a worker
    pub fn update_cache_miss_rate(&self, worker_id: usize, missrate: f64) -> LinalgResult<()> {
        if worker_id >= self.worker_affinity.len() {
            return Err(LinalgError::InvalidInput("Invalid worker ID".to_string()));
        }

        let mut rates = self.cache_miss_rates.lock().unwrap();
        rates[worker_id] = missrate;
        Ok(())
    }

    /// Get NUMA-aware worker assignment for a task
    pub fn get_numa_optimal_worker(&self, memorynode: usize) -> usize {
        if memorynode >= self.numa_topology.node_count {
            return 0;
        }

        // Find a worker on the same NUMA _node
        for (worker_id, &cpu) in self.worker_affinity.iter().enumerate() {
            for _node in 0..self.numa_topology.node_count {
                if self.numa_topology.cpus_per_node[_node].contains(&cpu) && _node == memorynode {
                    return worker_id;
                }
            }
        }

        // Fallback to any worker
        0
    }
}

/// Advanced parallel matrix multiplication with cache-aware optimization
#[allow(dead_code)]
pub fn parallel_gemm_cache_aware<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    workers: usize,
    cache_strategy: CacheAwareStrategy,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (m, k) = a.dim();
    let (k2, n) = b.dim();

    if k != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions incompatible: {m}x{k} * {k2}x{n}"
        )));
    }

    let cache_stealer = CacheAwareWorkStealer::new(workers, cache_strategy)?;
    let mut result = Array2::zeros((m, n));

    // Create work items for cache-optimized block multiplication
    let blocksize = 64; // Optimize for L1 cache
    let mut work_items = Vec::new();
    let mut work_id = 0;

    for i in (0..m).step_by(blocksize) {
        for j in (0..n).step_by(blocksize) {
            for kk in (0..k).step_by(blocksize) {
                let i_end = (i + blocksize).min(m);
                let j_end = (j + blocksize).min(n);
                let k_end = (kk + blocksize).min(k);

                let block_work = BlockMultiplyWork {
                    i_start: i,
                    i_end,
                    j_start: j,
                    j_end,
                    k_start: kk,
                    k_end,
                    a_block: a.slice(s![i..i_end, kk..k_end]).to_owned(),
                    b_block: b.slice(s![kk..k_end, j..j_end]).to_owned(),
                };

                work_items.push(WorkItem::new(work_id, block_work));
                work_id += 1;
            }
        }
    }

    // Execute cache-aware multiplication
    let block_results: Vec<LinalgResult<BlockMultiplyResult<F>>> = cache_stealer
        .execute_cache_aware(
            work_items,
            |work| {
                let mut block_result =
                    Array2::zeros((work.i_end - work.i_start, work.j_end - work.j_start));

                // Perform block multiplication
                for i in 0..(work.i_end - work.i_start) {
                    for j in 0..(work.j_end - work.j_start) {
                        let mut sum = F::zero();
                        for k in 0..(work.k_end - work.k_start) {
                            sum += work.a_block[[i, k]] * work.b_block[[k, j]];
                        }
                        block_result[[i, j]] = sum;
                    }
                }

                Ok(BlockMultiplyResult {
                    i_start: work.i_start,
                    j_start: work.j_start,
                    result: block_result,
                })
            },
            cache_strategy,
        )?;

    // Accumulate results
    for block_result in block_results {
        let block_result = block_result?; // Handle the Result
        let i_end = block_result.i_start + block_result.result.nrows();
        let j_end = block_result.j_start + block_result.result.ncols();

        let mut result_slice =
            result.slice_mut(s![block_result.i_start..i_end, block_result.j_start..j_end]);

        result_slice += &block_result.result;
    }

    Ok(result)
}

/// Work item for block matrix multiplication
#[derive(Clone)]
struct BlockMultiplyWork<F: Clone> {
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize,
    k_start: usize,
    k_end: usize,
    a_block: Array2<F>,
    b_block: Array2<F>,
}

/// Result of block matrix multiplication
#[derive(Clone)]
struct BlockMultiplyResult<F> {
    i_start: usize,
    j_start: usize,
    result: Array2<F>,
}

/// Adaptive work chunk sizing based on workload characteristics
#[derive(Debug, Clone)]
pub struct AdaptiveChunking {
    /// Minimum chunk size
    min_chunksize: usize,
    /// Maximum chunk size
    max_chunksize: usize,
    /// Current optimal chunk size
    current_chunksize: usize,
    /// Performance history for adaptation
    performance_history: Vec<ChunkPerformance>,
    /// Maximum history entries to maintain
    max_history: usize,
}

/// Performance metrics for a chunk execution
#[derive(Debug, Clone)]
pub struct ChunkPerformance {
    /// Chunk size used
    pub chunksize: usize,
    /// Execution time in nanoseconds
    pub execution_time_ns: u64,
    /// Work complexity estimate
    pub work_complexity: f64,
    /// Cache miss rate (if available)
    pub cache_miss_rate: Option<f64>,
    /// Thread utilization percentage
    pub thread_utilization: f64,
}

impl AdaptiveChunking {
    /// Create a new adaptive chunking strategy
    pub fn new(_minsize: usize, maxsize: usize) -> Self {
        Self {
            min_chunksize: _minsize,
            max_chunksize: maxsize,
            current_chunksize: (_minsize + maxsize) / 2,
            performance_history: Vec::new(),
            max_history: 50,
        }
    }

    /// Record performance for a chunk execution
    pub fn record_performance(&mut self, performance: ChunkPerformance) {
        self.performance_history.push(performance);

        // Maintain history size limit
        if self.performance_history.len() > self.max_history {
            self.performance_history.remove(0);
        }

        // Adapt chunk size based on recent performance
        self.adapt_chunksize();
    }

    /// Enhanced adaptive chunk size optimization with statistical analysis
    fn adapt_chunksize(&mut self) {
        if self.performance_history.len() < 5 {
            return;
        }

        // Analyze performance metrics with statistical approach
        let recent_entries =
            &self.performance_history[self.performance_history.len().saturating_sub(10)..];

        // Group entries by chunk size and calculate statistics
        let mut chunk_performance: HashMap<usize, Vec<f64>> = HashMap::new();

        for entry in recent_entries {
            let throughput =
                entry.work_complexity / (entry.execution_time_ns as f64 / 1_000_000_000.0);
            chunk_performance
                .entry(entry.chunksize)
                .or_default()
                .push(throughput);
        }

        // Find optimal chunk size considering both performance and stability
        let mut best_score = f64::NEG_INFINITY;
        let mut best_chunksize = self.current_chunksize;

        for (&chunksize, throughputs) in &chunk_performance {
            if throughputs.len() < 2 {
                continue; // Need at least 2 samples for variance calculation
            }

            let mean_throughput: f64 = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
            let variance: f64 = throughputs
                .iter()
                .map(|&t| (t - mean_throughput).powi(2))
                .sum::<f64>()
                / throughputs.len() as f64;
            let std_dev = variance.sqrt();

            // Score considering both performance (mean) and stability (inverse of std_dev)
            // Higher throughput is better, lower variance is better
            let stability_factor = 1.0 / (1.0 + std_dev / mean_throughput); // Coefficient of variation
            let score = mean_throughput * stability_factor;

            if score > best_score {
                best_score = score;
                best_chunksize = chunksize;
            }
        }

        // Enhanced adaptive adjustment with momentum and exploration
        let adjustment_factor = if self.performance_history.len() > 20 {
            0.3 // More aggressive when we have more data
        } else {
            0.15 // Conservative when learning
        };

        // Add small exploration component to avoid local optima
        let exploration_factor = 0.05;
        // Use deterministic pseudo-random based on history length for exploration
        let mut hasher = DefaultHasher::new();
        self.performance_history.len().hash(&mut hasher);
        let pseudo_random = (hasher.finish() % 1000) as f64 / 1000.0;
        let exploration_offset = (pseudo_random - 0.5) * exploration_factor * best_chunksize as f64;

        let targetsize = best_chunksize as f64 + exploration_offset;
        let currentsize = self.current_chunksize as f64;
        let newsize = currentsize + (targetsize - currentsize) * adjustment_factor;

        self.current_chunksize = (newsize as usize)
            .max(self.min_chunksize)
            .min(self.max_chunksize);

        // Adaptive bounds adjustment - expand search space if we're hitting boundaries
        if self.current_chunksize == self.min_chunksize && best_score > 0.0 {
            self.min_chunksize = (self.min_chunksize as f64 * 0.8) as usize;
        }
        if self.current_chunksize == self.max_chunksize && best_score > 0.0 {
            self.max_chunksize = (self.max_chunksize as f64 * 1.2) as usize;
        }
    }

    /// Get the current optimal chunk size
    pub fn get_chunksize(&self) -> usize {
        self.current_chunksize
    }

    /// Predict optimal chunk size for a given matrix operation without execution
    pub fn predict_optimal_chunksize(
        &self,
        matrixsize: (usize, usize),
        operation_type: MatrixOperationType,
        num_workers: usize,
    ) -> usize {
        // Base prediction using matrix characteristics
        let (rows, cols) = matrixsize;
        let total_elements = rows * cols;

        let base_chunksize = match operation_type {
            MatrixOperationType::MatrixVectorMultiplication => {
                // For matvec, chunk by rows to maintain cache locality
                (rows / num_workers).clamp(16, 1024)
            }
            MatrixOperationType::MatrixMatrixMultiplication => {
                // For matmul, consider both dimensions and target block sizes for cache efficiency
                let target_block_elements = 4096; // Good for L1 cache (32KB / 8 bytes)
                let elements_per_worker = total_elements / num_workers;
                elements_per_worker.min(target_block_elements).max(64)
            }
            MatrixOperationType::Decomposition => {
                // Decompositions have irregular patterns, use smaller chunks for better load balancing
                (rows / (num_workers * 4)).clamp(8, 256)
            }
            MatrixOperationType::EigenComputation => {
                // Eigenvalue computations are typically iterative and memory-intensive
                (rows / (num_workers * 2)).clamp(16, 512)
            }
            MatrixOperationType::IterativeSolver => {
                // Iterative solvers benefit from larger chunks to amortize synchronization costs
                (rows / num_workers).clamp(32, 2048)
            }
        };

        // Adjust based on historical performance if available
        if self.performance_history.len() > 5 {
            // Find similar matrix sizes in history
            let mut similar_performance = Vec::new();
            for entry in &self.performance_history {
                // Consider operations on matrices within 20% size difference as "similar"
                let size_ratio = (total_elements as f64) / entry.work_complexity;
                if size_ratio > 0.8 && size_ratio < 1.2 {
                    let throughput =
                        entry.work_complexity / (entry.execution_time_ns as f64 / 1_000_000_000.0);
                    similar_performance.push((entry.chunksize, throughput));
                }
            }

            if !similar_performance.is_empty() {
                // Weight historical performance with base prediction
                let historical_optimum = similar_performance
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|&(chunksize_, _)| chunksize_)
                    .unwrap_or(base_chunksize);

                // Blend base prediction with historical optimum
                let blend_factor = 0.7; // Favor historical data
                let predicted = (base_chunksize as f64 * (1.0 - blend_factor)
                    + historical_optimum as f64 * blend_factor)
                    as usize;

                return predicted.max(self.min_chunksize).min(self.max_chunksize);
            }
        }

        base_chunksize
            .max(self.min_chunksize)
            .min(self.max_chunksize)
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> AdaptiveChunkingStats {
        if self.performance_history.is_empty() {
            return AdaptiveChunkingStats::default();
        }

        let total_entries = self.performance_history.len();
        let avg_execution_time = self
            .performance_history
            .iter()
            .map(|p| p.execution_time_ns)
            .sum::<u64>() as f64
            / total_entries as f64;

        let avg_utilization = self
            .performance_history
            .iter()
            .map(|p| p.thread_utilization)
            .sum::<f64>()
            / total_entries as f64;

        let cache_miss_rate = self
            .performance_history
            .iter()
            .filter_map(|p| p.cache_miss_rate)
            .fold(None, |acc, x| Some(acc.unwrap_or(0.0) + x))
            .map(|rate| {
                rate / self
                    .performance_history
                    .iter()
                    .filter(|p| p.cache_miss_rate.is_some())
                    .count() as f64
            });

        AdaptiveChunkingStats {
            current_chunksize: self.current_chunksize,
            avg_execution_time_ms: avg_execution_time / 1_000_000.0,
            avg_thread_utilization: avg_utilization,
            avg_cache_miss_rate: cache_miss_rate,
            total_adaptations: total_entries,
        }
    }
}

/// Statistics for adaptive chunking performance
#[derive(Debug, Clone, Default)]
pub struct AdaptiveChunkingStats {
    /// Current chunk size
    pub current_chunksize: usize,
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// Average thread utilization percentage
    pub avg_thread_utilization: f64,
    /// Average cache miss rate (if available)
    pub avg_cache_miss_rate: Option<f64>,
    /// Total number of adaptations performed
    pub total_adaptations: usize,
}

/// Enhanced work-stealing scheduler with adaptive optimizations
pub struct OptimizedWorkStealingScheduler<T: Clone + Send + 'static> {
    /// Base work-stealing scheduler
    #[allow(dead_code)]
    base_scheduler: WorkStealingScheduler<T>,
    /// Adaptive chunking strategy
    adaptive_chunking: Arc<Mutex<AdaptiveChunking>>,
    /// Performance monitoring
    performance_monitor: Arc<Mutex<PerformanceMonitor>>,
    /// Cache locality optimizer
    cache_optimizer: Arc<Mutex<CacheLocalityOptimizer>>,
}

/// Performance monitoring for work-stealing operations
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Total tasks executed
    total_tasks: u64,
    /// Total execution time
    total_execution_time_ns: u64,
    /// Work stealing events
    steal_events: u64,
    /// Failed steal attempts
    failed_steals: u64,
    /// Queue contentions
    queue_contentions: u64,
    /// Load imbalance measurements
    load_imbalance_history: Vec<f64>,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            total_tasks: 0,
            total_execution_time_ns: 0,
            steal_events: 0,
            failed_steals: 0,
            queue_contentions: 0,
            load_imbalance_history: Vec::new(),
        }
    }

    /// Record task execution
    pub fn record_task(&mut self, execution_timens: u64) {
        self.total_tasks += 1;
        self.total_execution_time_ns += execution_timens;
    }

    /// Record work stealing event
    pub fn record_steal(&mut self, successful: bool) {
        if successful {
            self.steal_events += 1;
        } else {
            self.failed_steals += 1;
        }
    }

    /// Record queue contention
    pub fn record_contention(&mut self) {
        self.queue_contentions += 1;
    }

    /// Record load imbalance measurement
    pub fn record_load_imbalance(&mut self, imbalance: f64) {
        self.load_imbalance_history.push(imbalance);
        // Keep only recent measurements
        if self.load_imbalance_history.len() > 100 {
            self.load_imbalance_history.remove(0);
        }
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> PerformanceStats {
        let avg_task_time = if self.total_tasks > 0 {
            self.total_execution_time_ns as f64 / self.total_tasks as f64
        } else {
            0.0
        };

        let steal_success_rate = if self.steal_events + self.failed_steals > 0 {
            self.steal_events as f64 / (self.steal_events + self.failed_steals) as f64
        } else {
            0.0
        };

        let avg_load_imbalance = if !self.load_imbalance_history.is_empty() {
            self.load_imbalance_history.iter().sum::<f64>()
                / self.load_imbalance_history.len() as f64
        } else {
            0.0
        };

        PerformanceStats {
            total_tasks: self.total_tasks,
            avg_task_time_ns: avg_task_time,
            steal_success_rate,
            queue_contentions: self.queue_contentions,
            avg_load_imbalance,
        }
    }
}

/// Performance statistics summary
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Total number of tasks executed
    pub total_tasks: u64,
    /// Average task execution time in nanoseconds
    pub avg_task_time_ns: f64,
    /// Work stealing success rate (0.0 to 1.0)
    pub steal_success_rate: f64,
    /// Number of queue contentions
    pub queue_contentions: u64,
    /// Average load imbalance factor
    pub avg_load_imbalance: f64,
}

/// Cache locality optimizer for work distribution
#[derive(Debug)]
pub struct CacheLocalityOptimizer {
    /// Memory access patterns
    access_patterns: Vec<MemoryAccessPattern>,
    /// Cache line size (typically 64 bytes)
    cache_linesize: usize,
    /// L1 cache size estimate
    l1_cachesize: usize,
    /// L2 cache size estimate
    l2_cachesize: usize,
}

/// Memory access pattern for cache optimization
#[derive(Debug, Clone)]
pub struct MemoryAccessPattern {
    /// Memory address range start
    pub address_start: usize,
    /// Memory address range end
    pub address_end: usize,
    /// Access frequency
    pub access_frequency: u64,
    /// Sequential vs random access ratio
    pub sequential_ratio: f64,
}

impl CacheLocalityOptimizer {
    /// Create a new cache locality optimizer
    pub fn new() -> Self {
        Self {
            access_patterns: Vec::new(),
            cache_linesize: 64,       // Common cache line size
            l1_cachesize: 32 * 1024,  // 32KB typical L1
            l2_cachesize: 256 * 1024, // 256KB typical L2
        }
    }

    /// Record memory access pattern
    pub fn record_access_pattern(&mut self, pattern: MemoryAccessPattern) {
        self.access_patterns.push(pattern);

        // Maintain reasonable history size
        if self.access_patterns.len() > 1000 {
            self.access_patterns.remove(0);
        }
    }

    /// Optimize work distribution based on cache locality
    pub fn optimize_work_distribution(
        &self,
        work_items: &[usize],
        num_workers: usize,
    ) -> Vec<Vec<usize>> {
        let mut worker_assignments = vec![Vec::new(); num_workers];

        if work_items.is_empty() {
            return worker_assignments;
        }

        // Simple locality-aware distribution
        // Group adjacent work _items to the same worker to improve cache locality
        let chunksize = work_items.len().div_ceil(num_workers);

        for (i, &work_item) in work_items.iter().enumerate() {
            let worker_id = (i / chunksize).min(num_workers - 1);
            worker_assignments[worker_id].push(work_item);
        }

        worker_assignments
    }

    /// Get cache optimization recommendations
    pub fn get_recommendations(&self) -> CacheOptimizationRecommendations {
        let total_accesses = self
            .access_patterns
            .iter()
            .map(|p| p.access_frequency)
            .sum::<u64>();

        let avg_sequential_ratio = if !self.access_patterns.is_empty() {
            self.access_patterns
                .iter()
                .map(|p| p.sequential_ratio * p.access_frequency as f64)
                .sum::<f64>()
                / total_accesses as f64
        } else {
            0.5
        };

        let working_setsize = self
            .access_patterns
            .iter()
            .map(|p| p.address_end - p.address_start)
            .sum::<usize>();

        CacheOptimizationRecommendations {
            recommended_blocksize: if avg_sequential_ratio > 0.7 {
                self.cache_linesize * 4 // Larger blocks for sequential access
            } else {
                self.cache_linesize // Smaller blocks for random access
            },
            locality_friendly: avg_sequential_ratio > 0.5,
            working_set_fits_l1: working_setsize <= self.l1_cachesize,
            working_set_fits_l2: working_setsize <= self.l2_cachesize,
            prefetch_beneficial: avg_sequential_ratio > 0.6,
        }
    }
}

/// Cache optimization recommendations
#[derive(Debug, Clone)]
pub struct CacheOptimizationRecommendations {
    /// Recommended block size for optimal cache usage
    pub recommended_blocksize: usize,
    /// Whether the access pattern is locality-friendly
    pub locality_friendly: bool,
    /// Whether the working set fits in L1 cache
    pub working_set_fits_l1: bool,
    /// Whether the working set fits in L2 cache
    pub working_set_fits_l2: bool,
    /// Whether prefetching would be beneficial
    pub prefetch_beneficial: bool,
}

impl<T: Clone + Send + 'static> OptimizedWorkStealingScheduler<T> {
    /// Create a new optimized work-stealing scheduler
    pub fn new(_numworkers: usize) -> Self {
        Self {
            base_scheduler: WorkStealingScheduler::new(_numworkers),
            adaptive_chunking: Arc::new(Mutex::new(AdaptiveChunking::new(8, 1024))),
            performance_monitor: Arc::new(Mutex::new(PerformanceMonitor::new())),
            cache_optimizer: Arc::new(Mutex::new(CacheLocalityOptimizer::new())),
        }
    }

    /// Execute work with adaptive optimization
    pub fn execute_optimized<F, R>(&self, work_items: Vec<T>, workfn: F) -> LinalgResult<Vec<R>>
    where
        F: Fn(T) -> R + Send + Sync + Clone + 'static,
        R: Send + Clone + 'static,
    {
        let start_time = std::time::Instant::now();

        // Get current chunk size from adaptive chunking
        let chunksize = {
            let chunking = self.adaptive_chunking.lock().unwrap();
            chunking.get_chunksize()
        };

        // Use parallel processing from scirs2-core as per project policy
        use scirs2_core::parallel_ops::*;

        // Execute work _items in parallel using proper parallel processing
        let results: Vec<R> = work_items.into_par_iter().map(workfn).collect();

        // Record performance metrics
        let execution_time = start_time.elapsed();
        {
            let mut monitor = self.performance_monitor.lock().unwrap();
            monitor.record_task(execution_time.as_nanos() as u64);
        }

        // Record chunk performance for adaptation
        {
            let mut chunking = self.adaptive_chunking.lock().unwrap();
            chunking.record_performance(ChunkPerformance {
                chunksize,
                execution_time_ns: execution_time.as_nanos() as u64,
                work_complexity: results.len() as f64, // Simple complexity estimate
                cache_miss_rate: None,                 // Would need hardware performance counters
                thread_utilization: 0.8, // Placeholder - would need actual measurement
            });
        }

        Ok(results)
    }

    /// Get comprehensive performance statistics
    pub fn get_performance_stats(&self) -> OptimizedSchedulerStats {
        let chunking_stats = {
            let chunking = self.adaptive_chunking.lock().unwrap();
            chunking.get_stats()
        };

        let performance_stats = {
            let monitor = self.performance_monitor.lock().unwrap();
            monitor.get_stats()
        };

        let cache_recommendations = {
            let optimizer = self.cache_optimizer.lock().unwrap();
            optimizer.get_recommendations()
        };

        OptimizedSchedulerStats {
            chunking_stats,
            performance_stats,
            cache_recommendations,
        }
    }
}

/// Comprehensive statistics for the optimized scheduler
#[derive(Debug, Clone)]
pub struct OptimizedSchedulerStats {
    /// Adaptive chunking statistics
    pub chunking_stats: AdaptiveChunkingStats,
    /// Performance monitoring statistics
    pub performance_stats: PerformanceStats,
    /// Cache optimization recommendations
    pub cache_recommendations: CacheOptimizationRecommendations,
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CacheLocalityOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod optimization_tests {
    use super::*;

    #[test]
    fn test_adaptive_chunking() {
        let mut chunking = AdaptiveChunking::new(8, 512);
        assert_eq!(chunking.get_chunksize(), 260); // (8 + 512) / 2

        // Record some performance data
        chunking.record_performance(ChunkPerformance {
            chunksize: 64,
            execution_time_ns: 1_000_000,
            work_complexity: 100.0,
            cache_miss_rate: Some(0.05),
            thread_utilization: 0.9,
        });

        let stats = chunking.get_stats();
        assert_eq!(stats.total_adaptations, 1);
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();

        monitor.record_task(1_000_000);
        monitor.record_steal(true);
        monitor.record_steal(false);
        monitor.record_contention();

        let stats = monitor.get_stats();
        assert_eq!(stats.total_tasks, 1);
        assert_eq!(stats.steal_success_rate, 0.5);
        assert_eq!(stats.queue_contentions, 1);
    }

    #[test]
    fn test_cache_locality_optimizer() {
        let mut optimizer = CacheLocalityOptimizer::new();

        optimizer.record_access_pattern(MemoryAccessPattern {
            address_start: 0,
            address_end: 1024,
            access_frequency: 100,
            sequential_ratio: 0.8,
        });

        let recommendations = optimizer.get_recommendations();
        assert!(recommendations.locality_friendly);
        assert!(recommendations.prefetch_beneficial);
    }

    #[test]
    fn test_optimized_scheduler_creation() {
        let scheduler = OptimizedWorkStealingScheduler::<i32>::new(4);
        let stats = scheduler.get_performance_stats();

        // Check that stats are properly initialized
        assert_eq!(stats.performance_stats.total_tasks, 0);
        assert!(stats.cache_recommendations.recommended_blocksize > 0);
    }
}
