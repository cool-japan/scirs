//! Advanced parallel processing optimization for numerical algorithms
//!
//! This module provides sophisticated parallel processing strategies including
//! work-stealing task distribution, NUMA-aware memory allocation, vectorized
//! operations, and dynamic load balancing for numerical integration algorithms.

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Advanced parallel execution engine
pub struct ParallelOptimizer {
    /// Number of worker threads
    pub num_threads: usize,
    /// Thread pool for task execution
    thread_pool: Option<ThreadPool>,
    /// NUMA topology information
    pub numa_info: NumaTopology,
    /// Load balancing strategy
    pub load_balancer: LoadBalancingStrategy,
    /// Work-stealing configuration
    pub work_stealing_config: WorkStealingConfig,
}

/// Thread pool with advanced features
pub struct ThreadPool {
    workers: Vec<Worker>,
    task_queue: Arc<Mutex<TaskQueue>>,
    shutdown: Arc<AtomicUsize>,
}

/// Individual worker thread
struct Worker {
    id: usize,
    thread: Option<JoinHandle<()>>,
    local_queue: Arc<Mutex<LocalTaskQueue>>,
}

/// Main task queue for work distribution
struct TaskQueue {
    global_tasks: Vec<Box<dyn ParallelTask + Send>>,
    pending_tasks: usize,
}

/// Local task queue for each worker
struct LocalTaskQueue {
    tasks: Vec<Box<dyn ParallelTask + Send>>,
    steals_attempted: usize,
    steals_successful: usize,
}

/// NUMA (Non-Uniform Memory Access) topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub num_nodes: usize,
    /// CPU cores per NUMA node
    pub cores_per_node: Vec<usize>,
    /// Memory bandwidth per node
    pub bandwidth_per_node: Vec<f64>,
    /// Memory latency between nodes
    pub inter_node_latency: Vec<Vec<f64>>,
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    /// Static load balancing
    Static,
    /// Dynamic load balancing based on runtime metrics
    Dynamic,
    /// Work-stealing between threads
    WorkStealing,
    /// NUMA-aware load balancing
    NumaAware,
    /// Adaptive strategy that switches based on workload
    Adaptive,
}

/// Work-stealing configuration
#[derive(Debug, Clone)]
pub struct WorkStealingConfig {
    /// Maximum number of steal attempts before yielding
    pub max_steal_attempts: usize,
    /// Steal ratio (fraction of work to steal)
    pub steal_ratio: f64,
    /// Minimum task size to enable stealing
    pub min_steal_size: usize,
    /// Backoff strategy for failed steals
    pub backoff_strategy: BackoffStrategy,
}

/// Backoff strategies for work stealing
#[derive(Debug, Clone, Copy)]
pub enum BackoffStrategy {
    /// No backoff
    None,
    /// Linear backoff
    Linear(Duration),
    /// Exponential backoff
    Exponential { initial: Duration, max: Duration },
    /// Random jitter backoff
    RandomJitter { min: Duration, max: Duration },
}

/// Trait for parallel tasks
pub trait ParallelTask: Send {
    /// Execute the task
    fn execute(&self) -> ParallelResult;

    /// Estimate computational cost
    fn estimated_cost(&self) -> f64;

    /// Check if task can be subdivided
    fn can_subdivide(&self) -> bool;

    /// Subdivide task into smaller tasks
    fn subdivide(&self) -> Vec<Box<dyn ParallelTask + Send>>;

    /// Get task priority
    fn priority(&self) -> TaskPriority {
        TaskPriority::Normal
    }

    /// Get preferred NUMA node
    fn preferred_numa_node(&self) -> Option<usize> {
        None
    }
}

/// Task execution result
pub type ParallelResult = IntegrateResult<Box<dyn std::any::Any + Send>>;

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Vectorized computation task
pub struct VectorizedComputeTask<F: IntegrateFloat> {
    /// Input data
    pub input: Array2<F>,
    /// Operation to perform
    pub operation: VectorOperation<F>,
    /// Chunk size for processing
    pub chunk_size: usize,
    /// SIMD preference
    pub prefer_simd: bool,
}

/// Types of vectorized operations
#[derive(Clone)]
pub enum VectorOperation<F: IntegrateFloat> {
    /// Element-wise arithmetic
    ElementWise(ArithmeticOp),
    /// Matrix-vector operations
    MatrixVector(Array1<F>),
    /// Reduction operations
    Reduction(ReductionOp),
    /// Custom function
    Custom(Arc<dyn Fn(&ArrayView2<F>) -> Array2<F> + Send + Sync>),
}

/// Arithmetic operations
#[derive(Debug, Clone, Copy)]
pub enum ArithmeticOp {
    Add(f64),
    Multiply(f64),
    Power(f64),
    Exp,
    Log,
    Sin,
    Cos,
}

/// Reduction operations
#[derive(Debug, Clone, Copy)]
pub enum ReductionOp {
    Sum,
    Product,
    Max,
    Min,
    Mean,
    Variance,
}

/// NUMA-aware memory allocator
pub struct NumaAllocator {
    /// Node affinities for threads
    node_affinities: Vec<usize>,
    /// Memory usage per node
    memory_usage: Vec<AtomicUsize>,
    /// Allocation strategy
    strategy: NumaAllocationStrategy,
}

/// NUMA allocation strategies
#[derive(Debug, Clone, Copy)]
pub enum NumaAllocationStrategy {
    /// First-touch allocation
    FirstTouch,
    /// Round-robin allocation
    RoundRobin,
    /// Local allocation (preferred node)
    Local,
    /// Interleaved allocation
    Interleaved,
}

/// Parallel execution statistics
#[derive(Debug, Clone)]
pub struct ParallelExecutionStats {
    /// Total execution time
    pub total_time: Duration,
    /// Time per thread
    pub thread_times: Vec<Duration>,
    /// Load balance efficiency
    pub load_balance_efficiency: f64,
    /// Work stealing statistics
    pub work_stealing_stats: WorkStealingStats,
    /// NUMA affinity hits
    pub numa_affinity_hits: usize,
    /// Cache performance metrics
    pub cache_performance: CachePerformanceMetrics,
    /// SIMD utilization
    pub simd_utilization: f64,
}

/// Work stealing performance statistics
#[derive(Debug, Clone)]
pub struct WorkStealingStats {
    /// Total steal attempts
    pub steal_attempts: usize,
    /// Successful steals
    pub successful_steals: usize,
    /// Average steal success rate
    pub success_rate: f64,
    /// Time spent on stealing vs working
    pub steal_time_ratio: f64,
}

/// Cache performance metrics.
///
/// These fields require CPU performance counters to populate. The
/// [`ParallelOptimizer`] execution path does not sample such counters, so it
/// reports them as zero ("not measured") rather than fabricating values.
#[derive(Debug, Clone)]
pub struct CachePerformanceMetrics {
    /// Estimated cache hit rate (0.0 when not measured)
    pub hit_rate: f64,
    /// Memory bandwidth utilization (0.0 when not measured)
    pub bandwidth_utilization: f64,
    /// Cache-friendly access patterns detected (0 when not measured)
    pub cache_friendly_accesses: usize,
}

impl ParallelOptimizer {
    /// Create new parallel optimizer
    pub fn new(_numthreads: usize) -> Self {
        Self {
            num_threads: _numthreads,
            thread_pool: None,
            numa_info: NumaTopology::detect(),
            load_balancer: LoadBalancingStrategy::Adaptive,
            work_stealing_config: WorkStealingConfig::default(),
        }
    }

    /// Initialize thread pool
    pub fn initialize(&mut self) -> IntegrateResult<()> {
        let thread_pool = ThreadPool::new(self.num_threads, &self.work_stealing_config)?;
        self.thread_pool = Some(thread_pool);
        Ok(())
    }

    /// Execute tasks in parallel with optimization
    pub fn execute_parallel<T: ParallelTask + Send + 'static>(
        &mut self,
        tasks: Vec<Box<T>>,
    ) -> IntegrateResult<(Vec<ParallelResult>, ParallelExecutionStats)> {
        let start_time = Instant::now();

        if self.thread_pool.is_none() {
            self.initialize()?;
        }

        // Optimize task distribution based on strategy
        let optimized_tasks = self.optimize_task_distribution(tasks)?;

        // Execute tasks
        let results = self
            .thread_pool
            .as_ref()
            .expect("Failed to create parallel plan")
            .execute_tasks(optimized_tasks)?;

        // Collect statistics
        let stats = self.collect_execution_stats(
            start_time,
            self.thread_pool.as_ref().expect("Operation failed"),
        )?;

        Ok((results, stats))
    }

    /// Optimize task distribution based on load balancing strategy
    fn optimize_task_distribution<T: ParallelTask + Send + 'static>(
        &mut self,
        mut tasks: Vec<Box<T>>,
    ) -> IntegrateResult<Vec<Box<dyn ParallelTask + Send>>> {
        match self.load_balancer {
            LoadBalancingStrategy::Static => {
                // Simple round-robin distribution
                Ok(tasks
                    .into_iter()
                    .map(|t| t as Box<dyn ParallelTask + Send>)
                    .collect())
            }
            LoadBalancingStrategy::Dynamic => {
                // Sort by estimated cost and distribute
                tasks.sort_by(|a, b| {
                    b.estimated_cost()
                        .partial_cmp(&a.estimated_cost())
                        .expect("Operation failed")
                });
                Ok(tasks
                    .into_iter()
                    .map(|t| t as Box<dyn ParallelTask + Send>)
                    .collect())
            }
            LoadBalancingStrategy::WorkStealing => {
                // Enable subdivisions for large tasks
                let mut optimized_tasks = Vec::new();
                for task in tasks {
                    if task.can_subdivide() && task.estimated_cost() > 1000.0 {
                        let subtasks = task.subdivide();
                        optimized_tasks.extend(subtasks);
                    } else {
                        optimized_tasks.push(task as Box<dyn ParallelTask + Send>);
                    }
                }
                Ok(optimized_tasks)
            }
            LoadBalancingStrategy::NumaAware => {
                // Group tasks by preferred NUMA node
                let mut numa_groups: Vec<Vec<Box<dyn ParallelTask + Send>>> =
                    (0..self.numa_info.num_nodes).map(|_| Vec::new()).collect();
                let mut no_preference = Vec::new();

                for task in tasks {
                    if let Some(preferred_node) = task.preferred_numa_node() {
                        if preferred_node < numa_groups.len() {
                            numa_groups[preferred_node].push(task as Box<dyn ParallelTask + Send>);
                        } else {
                            no_preference.push(task as Box<dyn ParallelTask + Send>);
                        }
                    } else {
                        no_preference.push(task as Box<dyn ParallelTask + Send>);
                    }
                }

                // Distribute no-preference tasks evenly
                for (i, task) in no_preference.into_iter().enumerate() {
                    let group_idx = i % numa_groups.len();
                    numa_groups[group_idx].push(task);
                }

                Ok(numa_groups.into_iter().flatten().collect())
            }
            LoadBalancingStrategy::Adaptive => {
                // Choose strategy based on task characteristics
                let total_cost: f64 = tasks.iter().map(|t| t.estimated_cost()).sum();
                let avg_cost = total_cost / tasks.len() as f64;

                if avg_cost > 1000.0 {
                    // Use work-stealing for expensive tasks
                    self.load_balancer = LoadBalancingStrategy::WorkStealing;
                } else if tasks.iter().any(|t| t.preferred_numa_node().is_some()) {
                    // Use NUMA-aware for tasks with locality preferences
                    self.load_balancer = LoadBalancingStrategy::NumaAware;
                } else {
                    // Use dynamic for other cases
                    self.load_balancer = LoadBalancingStrategy::Dynamic;
                }

                self.optimize_task_distribution(tasks)
            }
        }
    }

    /// Collect execution statistics.
    ///
    /// Only genuinely observable quantities are reported here; nothing is
    /// fabricated. Concretely:
    ///
    /// * `total_time` is the real wall-clock duration of the parallel section.
    /// * `work_stealing_stats` are read from each worker's live counters. Note
    ///   that the data-parallel execution path ([`ThreadPool::execute_tasks`])
    ///   dispatches work through the core `parallel_ops` runtime rather than the
    ///   persistent worker loop, so these counters report the actual steal
    ///   activity of the explicit work-stealing path (which is zero when all
    ///   work was handled by the data-parallel runtime).
    /// * Per-thread wall-clock attribution (`thread_times`) is **not** available
    ///   for the data-parallel execution path, so an empty vector is returned
    ///   rather than a fabricated per-thread duration.
    /// * Hardware-level quantities (NUMA affinity hits, cache hit/bandwidth,
    ///   SIMD utilisation) require CPU performance counters that are not
    ///   sampled here; they are reported as zero and documented as "not
    ///   measured" rather than fabricated with plausible numbers.
    fn collect_execution_stats(
        &self,
        start_time: Instant,
        thread_pool: &ThreadPool,
    ) -> IntegrateResult<ParallelExecutionStats> {
        let total_time = start_time.elapsed();

        // Aggregate the real work-stealing counters maintained by the worker
        // local queues. These are honest measurements: they reflect exactly the
        // number of steal attempts/successes recorded by the explicit
        // work-stealing path.
        let mut steal_attempts = 0usize;
        let mut successful_steals = 0usize;
        for worker in &thread_pool.workers {
            if let Ok(local_q) = worker.local_queue.lock() {
                steal_attempts += local_q.steals_attempted;
                successful_steals += local_q.steals_successful;
            }
        }
        let success_rate = if steal_attempts > 0 {
            successful_steals as f64 / steal_attempts as f64
        } else {
            0.0
        };

        let work_stealing_stats = WorkStealingStats {
            steal_attempts,
            successful_steals,
            success_rate,
            // The fraction of time spent stealing versus working is not
            // instrumented; reported as zero rather than fabricated.
            steal_time_ratio: 0.0,
        };

        // Per-thread wall-clock times are not observable for the data-parallel
        // execution path; return an empty set rather than fabricate durations.
        // With no per-thread samples there is no measured imbalance, so report
        // perfect (1.0) load-balance efficiency as the neutral value.
        let thread_times: Vec<Duration> = Vec::new();
        let load_balance_efficiency = 1.0;

        Ok(ParallelExecutionStats {
            total_time,
            thread_times,
            load_balance_efficiency,
            work_stealing_stats,
            // NUMA affinity hits require hardware/OS counters that are not
            // sampled here (not measured).
            numa_affinity_hits: 0,
            cache_performance: CachePerformanceMetrics {
                // Cache hit-rate / bandwidth / friendly-access counts require
                // CPU performance counters that are not sampled here
                // (not measured).
                hit_rate: 0.0,
                bandwidth_utilization: 0.0,
                cache_friendly_accesses: 0,
            },
            // SIMD utilisation requires instruction-level profiling that is not
            // performed here (not measured).
            simd_utilization: 0.0,
        })
    }

    /// Execute vectorized computation with SIMD optimization
    pub fn execute_vectorized<F: IntegrateFloat>(
        &self,
        task: VectorizedComputeTask<F>,
    ) -> IntegrateResult<Array2<F>> {
        let chunk_size = task.chunk_size.max(1);
        let inputshape = task.input.dim();
        let mut result = Array2::zeros(inputshape);

        // Process in chunks for cache efficiency
        for chunk_start in (0..inputshape.0).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(inputshape.0);
            let chunk = task.input.slice(s![chunk_start..chunk_end, ..]);

            let chunk_result = match &task.operation {
                VectorOperation::ElementWise(op) => {
                    self.apply_elementwise_operation(&chunk, *op)?
                }
                VectorOperation::MatrixVector(vec) => self.apply_matvec_operation(&chunk, vec)?,
                VectorOperation::Reduction(op) => {
                    let reduced = self.apply_reduction_operation(&chunk, *op)?;
                    // Broadcast back to chunk shape
                    Array2::from_elem(chunk.dim(), reduced[[0, 0]])
                }
                VectorOperation::Custom(func) => func(&chunk),
            };

            result
                .slice_mut(s![chunk_start..chunk_end, ..])
                .assign(&chunk_result);
        }

        Ok(result)
    }

    /// Apply element-wise operation with SIMD optimization
    fn apply_elementwise_operation<F: IntegrateFloat>(
        &self,
        input: &ArrayView2<F>,
        op: ArithmeticOp,
    ) -> IntegrateResult<Array2<F>> {
        use ArithmeticOp::*;

        let result = match op {
            Add(value) => input.mapv(|x| x + F::from(value).expect("Failed to convert to float")),
            Multiply(value) => {
                input.mapv(|x| x * F::from(value).expect("Failed to convert to float"))
            }
            Power(exp) => input.mapv(|x| x.powf(F::from(exp).expect("Failed to convert to float"))),
            Exp => input.mapv(|x| x.exp()),
            Log => input.mapv(|x| x.ln()),
            Sin => input.mapv(|x| x.sin()),
            Cos => input.mapv(|x| x.cos()),
        };

        Ok(result)
    }

    /// Apply matrix-vector operation
    fn apply_matvec_operation<F: IntegrateFloat>(
        &self,
        matrix: &ArrayView2<F>,
        vector: &Array1<F>,
    ) -> IntegrateResult<Array2<F>> {
        if matrix.ncols() != vector.len() {
            return Err(IntegrateError::DimensionMismatch(
                "Matrix columns must match vector length".to_string(),
            ));
        }

        let mut result = Array2::zeros(matrix.dim());

        // Parallel matrix-vector multiplication
        for (i, mut row) in result.axis_iter_mut(Axis(0)).enumerate() {
            let matrix_row = matrix.row(i);
            let dot_product = matrix_row.dot(vector);
            row.fill(dot_product);
        }

        Ok(result)
    }

    /// Apply reduction operation
    fn apply_reduction_operation<F: IntegrateFloat>(
        &self,
        input: &ArrayView2<F>,
        op: ReductionOp,
    ) -> IntegrateResult<Array2<F>> {
        let result_value = match op {
            ReductionOp::Sum => input.sum(),
            ReductionOp::Product => input.fold(F::one(), |acc, &x| acc * x),
            ReductionOp::Max => input.fold(F::neg_infinity(), |acc, &x| acc.max(x)),
            ReductionOp::Min => input.fold(F::infinity(), |acc, &x| acc.min(x)),
            ReductionOp::Mean => input.sum() / F::from(input.len()).expect("Operation failed"),
            ReductionOp::Variance => {
                let mean = input.sum() / F::from(input.len()).expect("Operation failed");

                input.mapv(|x| (x - mean).powi(2)).sum()
                    / F::from(input.len()).expect("Operation failed")
            }
        };

        Ok(Array2::from_elem((1, 1), result_value))
    }
}

impl NumaTopology {
    /// Detect NUMA topology
    pub fn detect() -> Self {
        // Simplified NUMA detection - in practice would use hwloc or similar
        let num_cores = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let num_nodes = (num_cores / 4).max(1); // Assume 4 cores per node

        Self {
            num_nodes,
            cores_per_node: vec![4; num_nodes],
            bandwidth_per_node: vec![100.0; num_nodes], // GB/s
            inter_node_latency: vec![vec![1.0; num_nodes]; num_nodes], // μs
        }
    }

    /// Get preferred NUMA node for current thread
    pub fn get_preferred_node(&self) -> usize {
        // Simple round-robin assignment
        // Simple thread-to-NUMA mapping
        0 // Simplified - would use proper thread ID mapping
    }
}

impl Default for WorkStealingConfig {
    fn default() -> Self {
        Self {
            max_steal_attempts: 10,
            steal_ratio: 0.5,
            min_steal_size: 100,
            backoff_strategy: BackoffStrategy::Exponential {
                initial: Duration::from_micros(1),
                max: Duration::from_millis(1),
            },
        }
    }
}

impl ThreadPool {
    /// Create new thread pool
    pub fn new(num_threads: usize, config: &WorkStealingConfig) -> IntegrateResult<Self> {
        let task_queue = Arc::new(Mutex::new(TaskQueue {
            global_tasks: Vec::new(),
            pending_tasks: 0,
        }));

        let shutdown = Arc::new(AtomicUsize::new(0));
        let mut workers = Vec::with_capacity(num_threads);

        for id in 0..num_threads {
            let worker_queue = Arc::new(Mutex::new(LocalTaskQueue {
                tasks: Vec::new(),
                steals_attempted: 0,
                steals_successful: 0,
            }));

            let task_queue_clone = Arc::clone(&task_queue);
            let worker_queue_clone = Arc::clone(&worker_queue);
            let shutdown_clone = Arc::clone(&shutdown);

            let thread_handle = thread::spawn(move || {
                Self::worker_thread_loop(id, worker_queue_clone, task_queue_clone, shutdown_clone);
            });

            let worker = Worker {
                id,
                thread: Some(thread_handle),
                local_queue: worker_queue,
            };
            workers.push(worker);
        }

        Ok(Self {
            workers,
            task_queue,
            shutdown,
        })
    }

    /// Execute tasks in parallel and collect their real results.
    ///
    /// Large tasks are subdivided (when `can_subdivide()` and the estimated
    /// cost warrants it) and the resulting work items are sorted largest-first
    /// for better load balancing before being dispatched across the available
    /// CPU threads via the data-parallel runtime.  Each returned
    /// [`ParallelResult`] is the genuine output of `ParallelTask::execute`;
    /// there is one entry per executed (sub)task, in dispatch order.
    pub fn execute_tasks(
        &self,
        tasks: Vec<Box<dyn ParallelTask + Send>>,
    ) -> IntegrateResult<Vec<ParallelResult>> {
        use scirs2_core::parallel_ops::*;

        if tasks.is_empty() {
            return Ok(Vec::new());
        }

        // Subdivide large tasks first for better load distribution.
        let mut all_tasks: Vec<Box<dyn ParallelTask + Send>> = Vec::with_capacity(tasks.len());
        for task in tasks {
            if task.can_subdivide() && task.estimated_cost() > 10.0 {
                all_tasks.extend(task.subdivide());
            } else {
                all_tasks.push(task);
            }
        }

        // Sort work items by estimated cost (largest first) so that the most
        // expensive items are scheduled before the cheap ones.
        all_tasks.sort_by(|a, b| {
            b.estimated_cost()
                .partial_cmp(&a.estimated_cost())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Execute every work item and collect the genuine result of each one.
        // `execute()` cannot fail at the dispatch level, so the only errors are
        // those reported by the tasks themselves (preserved inside each
        // `ParallelResult`).
        let results: Vec<ParallelResult> = all_tasks
            .into_par_iter()
            .map(|task| task.execute())
            .collect();

        Ok(results)
    }

    /// Shutdown the thread pool and wait for all threads to complete
    pub fn shutdown(&mut self) -> IntegrateResult<()> {
        // Signal all threads to shutdown
        self.shutdown.store(1, Ordering::Relaxed);

        // Wait for all threads to complete
        for worker in self.workers.drain(..) {
            if let Some(thread) = worker.thread {
                if thread.join().is_err() {
                    return Err(IntegrateError::ComputationError(
                        "Failed to join worker thread".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Try to steal work from other workers (simplified implementation)
    fn try_work_stealing(
        _worker_id: usize,
        local_queue: &Arc<Mutex<LocalTaskQueue>>,
        global_queue: &Arc<Mutex<TaskQueue>>,
    ) -> Option<Box<dyn ParallelTask + Send>> {
        // In a full implementation, we'd need access to other workers' queues
        // For now, increment steal attempts counter and try global _queue again
        if let Ok(mut local_q) = local_queue.lock() {
            local_q.steals_attempted += 1;
        }

        // Try global _queue one more time as fallback
        if let Ok(mut global_q) = global_queue.lock() {
            let task = global_q.global_tasks.pop();
            if task.is_some() {
                global_q.pending_tasks = global_q.pending_tasks.saturating_sub(1);
                if let Ok(mut local_q) = local_queue.lock() {
                    local_q.steals_successful += 1;
                }
            }
            task
        } else {
            None
        }
    }

    /// Worker thread main loop
    fn worker_thread_loop(
        _worker_id: usize,
        local_queue: Arc<Mutex<LocalTaskQueue>>,
        global_queue: Arc<Mutex<TaskQueue>>,
        shutdown: Arc<AtomicUsize>,
    ) {
        loop {
            // Check for shutdown signal
            if shutdown.load(Ordering::Relaxed) == 1 {
                break;
            }

            // Try to get a task from local _queue first
            let mut task_option = None;
            if let Ok(mut local_q) = local_queue.lock() {
                task_option = local_q.tasks.pop();
            }

            // If no local task, try global _queue
            if task_option.is_none() {
                if let Ok(mut global_q) = global_queue.lock() {
                    task_option = global_q.global_tasks.pop();
                    if task_option.is_some() {
                        global_q.pending_tasks = global_q.pending_tasks.saturating_sub(1);
                    }
                }
            }

            // If still no task, try work stealing from other workers
            if task_option.is_none() {
                task_option = Self::try_work_stealing(_worker_id, &local_queue, &global_queue);
            }

            // Execute task if found
            if let Some(task) = task_option {
                let _result = task.execute();
                // Task executed successfully
            } else {
                // No work available, sleep briefly
                thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Signal shutdown
        self.shutdown.store(1, Ordering::Relaxed);

        // Wait for threads to complete
        for worker in self.workers.drain(..) {
            if let Some(thread) = worker.thread {
                let _ = thread.join(); // Ignore errors during cleanup
            }
        }
    }
}

impl<F: IntegrateFloat + Send + Sync> ParallelTask for VectorizedComputeTask<F> {
    fn execute(&self) -> ParallelResult {
        // Perform actual vectorized computation based on operation type
        let result: Array2<F> = match &self.operation {
            VectorOperation::ElementWise(op) => match op {
                ArithmeticOp::Add(value) => self
                    .input
                    .mapv(|x| x + F::from(*value).expect("Failed to convert to float")),
                ArithmeticOp::Multiply(value) => self
                    .input
                    .mapv(|x| x * F::from(*value).expect("Failed to convert to float")),
                ArithmeticOp::Power(exp) => self
                    .input
                    .mapv(|x| x.powf(F::from(*exp).expect("Failed to convert to float"))),
                ArithmeticOp::Exp => self.input.mapv(|x| x.exp()),
                ArithmeticOp::Log => self.input.mapv(|x| x.ln()),
                ArithmeticOp::Sin => self.input.mapv(|x| x.sin()),
                ArithmeticOp::Cos => self.input.mapv(|x| x.cos()),
            },
            VectorOperation::MatrixVector(vector) => {
                if self.input.ncols() != vector.len() {
                    return Err(IntegrateError::DimensionMismatch(
                        "Matrix columns must match vector length".to_string(),
                    ));
                }

                let mut result = Array2::zeros(self.input.dim());
                for (i, mut row) in result.axis_iter_mut(Axis(0)).enumerate() {
                    let matrix_row = self.input.row(i);
                    let dot_product = matrix_row.dot(vector);
                    row.fill(dot_product);
                }
                result
            }
            VectorOperation::Reduction(op) => {
                let result_value = match op {
                    ReductionOp::Sum => self.input.sum(),
                    ReductionOp::Product => self.input.fold(F::one(), |acc, &x| acc * x),
                    ReductionOp::Max => self.input.fold(F::neg_infinity(), |acc, &x| acc.max(x)),
                    ReductionOp::Min => self.input.fold(F::infinity(), |acc, &x| acc.min(x)),
                    ReductionOp::Mean => {
                        self.input.sum() / F::from(self.input.len()).expect("Operation failed")
                    }
                    ReductionOp::Variance => {
                        let mean =
                            self.input.sum() / F::from(self.input.len()).expect("Operation failed");
                        self.input.mapv(|x| (x - mean).powi(2)).sum()
                            / F::from(self.input.len()).expect("Operation failed")
                    }
                };
                Array2::from_elem((1, 1), result_value)
            }
            VectorOperation::Custom(func) => func(&self.input.view()),
        };

        Ok(Box::new(result) as Box<dyn std::any::Any + Send>)
    }

    fn estimated_cost(&self) -> f64 {
        (self.input.len() as f64) / (self.chunk_size as f64)
    }

    fn can_subdivide(&self) -> bool {
        self.input.nrows() > self.chunk_size * 2
    }

    fn subdivide(&self) -> Vec<Box<dyn ParallelTask + Send>> {
        // Only subdivide if the task is large enough and can benefit from parallelization
        if self.input.len() < self.chunk_size * 2 {
            return vec![];
        }

        let num_chunks = self.input.nrows().div_ceil(self.chunk_size);
        let mut subtasks = Vec::with_capacity(num_chunks);

        for i in 0..num_chunks {
            let start_row = i * self.chunk_size;
            let end_row = ((i + 1) * self.chunk_size).min(self.input.nrows());

            if start_row < self.input.nrows() {
                let chunk = self.input.slice(s![start_row..end_row, ..]).to_owned();

                let subtask = VectorizedComputeTask {
                    input: chunk,
                    operation: self.operation.clone(),
                    chunk_size: self.chunk_size,
                    prefer_simd: self.prefer_simd,
                };

                subtasks.push(Box::new(subtask) as Box<dyn ParallelTask + Send>);
            }
        }

        subtasks
    }
}

#[cfg(test)]
mod tests {
    use crate::parallel_optimization::ArithmeticOp;
    use crate::{NumaTopology, ParallelOptimizer, VectorOperation, VectorizedComputeTask};
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_parallel_optimizer_creation() {
        let optimizer = ParallelOptimizer::new(4);
        assert_eq!(optimizer.num_threads, 4);
    }

    #[test]
    fn test_numa_topology_detection() {
        let topology = NumaTopology::detect();
        assert!(topology.num_nodes > 0);
        assert!(!topology.cores_per_node.is_empty());
    }

    #[test]
    fn test_vectorized_computation() {
        let optimizer = ParallelOptimizer::new(2);
        let input = Array2::from_elem((4, 4), 1.0);

        let task = VectorizedComputeTask {
            input,
            operation: VectorOperation::ElementWise(ArithmeticOp::Add(2.0)),
            chunk_size: 2,
            prefer_simd: true,
        };

        let result = optimizer.execute_vectorized(task);
        assert!(result.is_ok());

        let output = result.expect("Test: parallel integration failed");
        assert_eq!(output.dim(), (4, 4));
        assert!((output[[0, 0]] - 3.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_execute_parallel_returns_real_results() {
        // `execute_parallel` must return the genuine output of each task, not a
        // placeholder `()`. We submit a small (non-subdivided) element-wise
        // task and verify the real computed array comes back.
        let mut optimizer = ParallelOptimizer::new(2);
        let input = Array2::from_elem((2, 2), 5.0_f64);
        let task = VectorizedComputeTask {
            input,
            operation: VectorOperation::ElementWise(ArithmeticOp::Add(3.0)),
            chunk_size: 8, // larger than the input so the task is not subdivided
            prefer_simd: false,
        };

        let (results, _stats) = optimizer
            .execute_parallel(vec![Box::new(task)])
            .expect("parallel execution should succeed");

        assert_eq!(results.len(), 1, "exactly one task was submitted");
        let any = results
            .into_iter()
            .next()
            .expect("one result")
            .expect("task itself should succeed");
        let array = any
            .downcast_ref::<Array2<f64>>()
            .expect("result must be the real Array2 produced by the task, not a placeholder");
        assert_eq!(array.dim(), (2, 2));
        for &v in array.iter() {
            assert!((v - 8.0_f64).abs() < 1e-10, "5 + 3 should equal 8, got {v}");
        }
    }
}
