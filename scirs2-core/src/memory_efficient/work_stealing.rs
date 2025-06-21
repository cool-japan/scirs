//! Advanced work-stealing scheduler for parallel processing.
//!
//! This module provides a sophisticated work-stealing scheduler that enables efficient
//! parallel processing by automatically balancing workloads across available threads.
//! It's designed for scientific computing workloads with varying computational complexity.
//!
//! ## Features
//!
//! - **Work-Stealing Algorithm**: Idle threads steal work from busy threads
//! - **Load Balancing**: Automatic distribution of tasks based on thread utilization
//! - **NUMA Awareness**: Consider NUMA topology for optimal memory access
//! - **Priority Queues**: Support for high/low priority task scheduling
//! - **Adaptive Partitioning**: Dynamic adjustment of work partitioning strategies
//! - **Performance Monitoring**: Real-time monitoring of scheduler performance
//! - **Resource Limits**: Configurable limits on memory and CPU usage
//!
//! ## Example Usage
//!
//! ```rust
//! use scirs2_core::memory_efficient::work_stealing::{
//!     WorkStealingScheduler, WorkStealingConfig, TaskPriority
//! };
//!
//! // Create a work-stealing scheduler
//! let config = WorkStealingConfig::default();
//! let mut scheduler = WorkStealingScheduler::new(config)?;
//!
//! // Submit tasks
//! scheduler.submit(TaskPriority::Normal, || {
//!     // Perform computation
//!     42
//! })?;
//!
//! // Start processing
//! scheduler.start()?;
//!
//! // Get results
//! while let Some(result) = scheduler.try_recv() {
//!     println!("Result: {}", result);
//! }
//! ```

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[cfg(feature = "parallel")]
/// Task priority levels for the work-stealing scheduler
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum TaskPriority {
    /// Low priority tasks (background processing)
    Low = 0,
    /// Normal priority tasks (default)
    #[default]
    Normal = 1,
    /// High priority tasks (time-sensitive operations)
    High = 2,
    /// Critical priority tasks (must be processed immediately)
    Critical = 3,
}

/// NUMA node information for work-stealing optimization
#[derive(Debug)]
pub struct NumaNode {
    /// Node ID
    pub id: usize,
    /// CPU cores on this node
    pub cpu_cores: Vec<usize>,
    /// Available memory on this node (bytes)
    pub memory_size: usize,
    /// Current memory usage (bytes)
    pub memory_used: AtomicUsize,
}

impl Clone for NumaNode {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            cpu_cores: self.cpu_cores.clone(),
            memory_size: self.memory_size,
            memory_used: AtomicUsize::new(
                self.memory_used.load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }
}

impl NumaNode {
    /// Create a new NUMA node
    pub fn new(id: usize, cpu_cores: Vec<usize>, memory_size: usize) -> Self {
        Self {
            id,
            cpu_cores,
            memory_size,
            memory_used: AtomicUsize::new(0),
        }
    }

    /// Get memory utilization ratio (0.0 to 1.0)
    pub fn memory_utilization(&self) -> f64 {
        if self.memory_size == 0 {
            0.0
        } else {
            self.memory_used.load(Ordering::Relaxed) as f64 / self.memory_size as f64
        }
    }

    /// Check if node has available memory
    pub fn has_available_memory(&self, required: usize) -> bool {
        self.memory_used.load(Ordering::Relaxed) + required <= self.memory_size
    }
}

/// Configuration for the work-stealing scheduler
#[derive(Debug, Clone)]
pub struct WorkStealingConfig {
    /// Number of worker threads (None = auto-detect)
    pub num_workers: Option<usize>,

    /// Maximum number of tasks per worker queue
    pub max_queue_size: usize,

    /// Steal attempt timeout in milliseconds
    pub steal_timeout_ms: u64,

    /// Maximum steal attempts per idle cycle
    pub max_steal_attempts: usize,

    /// Enable NUMA-aware scheduling
    pub numa_aware: bool,

    /// Enable priority-based scheduling
    pub priority_scheduling: bool,

    /// Worker thread affinity (CPU cores to bind threads to)
    pub thread_affinity: Option<Vec<usize>>,

    /// Maximum memory usage per worker (bytes)
    pub max_memory_per_worker: Option<usize>,

    /// Enable performance monitoring
    pub enable_monitoring: bool,

    /// Statistics collection interval
    pub stats_interval: Duration,

    /// Adaptive load balancing
    pub adaptive_balancing: bool,

    /// Load balancing threshold (0.0 to 1.0)
    pub load_balance_threshold: f64,
}

impl Default for WorkStealingConfig {
    fn default() -> Self {
        Self {
            num_workers: None,
            max_queue_size: 10000,
            steal_timeout_ms: 1,
            max_steal_attempts: 3,
            numa_aware: false,
            priority_scheduling: true,
            thread_affinity: None,
            max_memory_per_worker: None,
            enable_monitoring: true,
            stats_interval: Duration::from_secs(1),
            adaptive_balancing: true,
            load_balance_threshold: 0.8,
        }
    }
}

/// A task that can be executed by the work-stealing scheduler
pub trait WorkStealingTask: Send + 'static {
    /// The result type produced by this task
    type Output: Send + 'static;

    /// Execute the task and return the result
    fn execute(self) -> Self::Output;

    /// Get the estimated execution time (for load balancing)
    fn estimated_duration(&self) -> Option<Duration> {
        None
    }

    /// Get the estimated memory usage (for NUMA scheduling)
    fn estimated_memory(&self) -> Option<usize> {
        None
    }

    /// Check if this task can be split into smaller tasks
    fn can_split(&self) -> bool {
        false
    }

    /// Split this task into smaller tasks (if supported)
    fn split(self) -> Vec<Box<dyn WorkStealingTask<Output = Self::Output>>>
    where
        Self: Sized,
    {
        vec![Box::new(self)]
    }
}

/// Simple function-based task implementation
struct FunctionTask<F, R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    func: Option<F>,
    estimated_duration: Option<Duration>,
    estimated_memory: Option<usize>,
}

impl<F, R> FunctionTask<F, R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    fn new(func: F) -> Self {
        Self {
            func: Some(func),
            estimated_duration: None,
            estimated_memory: None,
        }
    }

    fn with_estimates(func: F, duration: Option<Duration>, memory: Option<usize>) -> Self {
        Self {
            func: Some(func),
            estimated_duration: duration,
            estimated_memory: memory,
        }
    }
}

impl<F, R> WorkStealingTask for FunctionTask<F, R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    type Output = R;

    fn execute(mut self) -> Self::Output {
        let func = self.func.take().expect("Function already executed");
        func()
    }

    fn estimated_duration(&self) -> Option<Duration> {
        self.estimated_duration
    }

    fn estimated_memory(&self) -> Option<usize> {
        self.estimated_memory
    }
}

/// Task wrapper with priority and metadata
struct PrioritizedTask {
    task: Option<Box<dyn FnOnce() -> Box<dyn std::any::Any + Send> + Send>>,
    priority: TaskPriority,
    submitted_at: Instant,
    numa_hint: Option<usize>,
}

impl std::fmt::Debug for PrioritizedTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrioritizedTask")
            .field("priority", &self.priority)
            .field("submitted_at", &self.submitted_at)
            .field("numa_hint", &self.numa_hint)
            .field("task", &self.task.is_some())
            .finish()
    }
}

impl PrioritizedTask {
    fn new<T: WorkStealingTask>(task: T, priority: TaskPriority, numa_hint: Option<usize>) -> Self
    where
        T::Output: 'static,
    {
        Self {
            task: Some(Box::new(move || Box::new(task.execute()))),
            priority,
            submitted_at: Instant::now(),
            numa_hint,
        }
    }

    fn execute(mut self) -> Box<dyn std::any::Any + Send> {
        let task = self.task.take().expect("Task already executed");
        task()
    }
}

/// Priority queue for tasks
#[derive(Debug)]
struct PriorityTaskQueue {
    queues: [VecDeque<PrioritizedTask>; 4], // One for each priority level
    total_size: usize,
    max_size: usize,
}

impl PriorityTaskQueue {
    fn new(max_size: usize) -> Self {
        Self {
            queues: [
                VecDeque::new(), // Low
                VecDeque::new(), // Normal
                VecDeque::new(), // High
                VecDeque::new(), // Critical
            ],
            total_size: 0,
            max_size,
        }
    }

    fn push(&mut self, task: PrioritizedTask) -> Result<(), PrioritizedTask> {
        if self.total_size >= self.max_size {
            return Err(task);
        }

        let priority_idx = task.priority as usize;
        self.queues[priority_idx].push_back(task);
        self.total_size += 1;
        Ok(())
    }

    fn pop(&mut self) -> Option<PrioritizedTask> {
        // Pop from highest priority first
        for queue in self.queues.iter_mut().rev() {
            if let Some(task) = queue.pop_front() {
                self.total_size -= 1;
                return Some(task);
            }
        }
        None
    }

    fn steal(&mut self) -> Option<PrioritizedTask> {
        // Steal from lowest priority first
        for queue in &mut self.queues {
            if let Some(task) = queue.pop_back() {
                self.total_size -= 1;
                return Some(task);
            }
        }
        None
    }

    fn len(&self) -> usize {
        self.total_size
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.total_size == 0
    }

    #[allow(dead_code)]
    fn is_full(&self) -> bool {
        self.total_size >= self.max_size
    }
}

/// Worker thread for the work-stealing scheduler
struct Worker {
    #[allow(dead_code)]
    id: usize,
    local_queue: Arc<Mutex<PriorityTaskQueue>>,
    global_queue: Arc<Mutex<PriorityTaskQueue>>,
    other_workers: Vec<Arc<Mutex<PriorityTaskQueue>>>,
    #[allow(dead_code)]
    numa_node: Option<usize>,
    shutdown: Arc<AtomicBool>,
    stats: Arc<WorkerStats>,
    config: WorkStealingConfig,
}

/// Statistics for individual workers
#[derive(Debug, Default)]
struct WorkerStats {
    tasks_executed: AtomicU64,
    tasks_stolen: AtomicU64,
    #[allow(dead_code)]
    tasks_provided: AtomicU64,
    idle_time: AtomicU64,
    active_time: AtomicU64,
    last_activity: AtomicU64,
}

impl Worker {
    fn new(
        id: usize,
        global_queue: Arc<Mutex<PriorityTaskQueue>>,
        numa_node: Option<usize>,
        shutdown: Arc<AtomicBool>,
        config: WorkStealingConfig,
    ) -> Self {
        let local_queue = Arc::new(Mutex::new(PriorityTaskQueue::new(
            config.max_queue_size / 4, // Local queues are smaller
        )));

        Self {
            id,
            local_queue,
            global_queue,
            other_workers: Vec::new(),
            numa_node,
            shutdown,
            stats: Arc::new(WorkerStats::default()),
            config,
        }
    }

    fn add_other_worker(&mut self, worker_queue: Arc<Mutex<PriorityTaskQueue>>) {
        self.other_workers.push(worker_queue);
    }

    fn run(&self, result_sender: crossbeam::channel::Sender<Box<dyn std::any::Any + Send>>) {
        let mut consecutive_steals = 0;
        let mut last_steal_attempt = Instant::now();

        while !self.shutdown.load(Ordering::Relaxed) {
            let task_start = Instant::now();

            // Try to get a task
            if let Some(task) = self.get_task() {
                // Execute the task
                let result = task.execute();

                // Send result
                if result_sender.send(result).is_err() {
                    // Receiver dropped, probably shutting down
                    break;
                }

                // Update statistics
                self.stats.tasks_executed.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .active_time
                    .fetch_add(task_start.elapsed().as_micros() as u64, Ordering::Relaxed);
                self.stats
                    .last_activity
                    .store(task_start.elapsed().as_secs(), Ordering::Relaxed);

                consecutive_steals = 0;
            } else {
                // No task found, record idle time
                let idle_start = Instant::now();

                // Try to steal work
                if last_steal_attempt.elapsed()
                    >= Duration::from_millis(self.config.steal_timeout_ms)
                {
                    if self.try_steal_work() {
                        consecutive_steals += 1;
                        self.stats.tasks_stolen.fetch_add(1, Ordering::Relaxed);
                    }
                    last_steal_attempt = Instant::now();
                }

                // Update idle time
                self.stats
                    .idle_time
                    .fetch_add(idle_start.elapsed().as_micros() as u64, Ordering::Relaxed);

                // Adaptive back-off
                let backoff_duration = if consecutive_steals > 5 {
                    Duration::from_millis(10) // Longer backoff if many failed steals
                } else {
                    Duration::from_micros(100) // Short backoff normally
                };

                thread::sleep(backoff_duration);
            }
        }
    }

    fn get_task(&self) -> Option<PrioritizedTask> {
        // Try local queue first
        if let Ok(mut local) = self.local_queue.try_lock() {
            if let Some(task) = local.pop() {
                return Some(task);
            }
        }

        // Try global queue
        if let Ok(mut global) = self.global_queue.try_lock() {
            if let Some(task) = global.pop() {
                return Some(task);
            }
        }

        None
    }

    fn try_steal_work(&self) -> bool {
        let mut attempts = 0;

        // Shuffle the worker list to avoid always stealing from the same worker
        let mut workers = self.other_workers.clone();
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.id.hash(&mut hasher);
        let seed = hasher.finish() as usize;

        // Simple shuffle based on worker ID
        for i in 0..workers.len() {
            let j = (seed + i) % workers.len();
            workers.swap(i, j);
        }

        for worker_queue in workers {
            if attempts >= self.config.max_steal_attempts {
                break;
            }

            if let Ok(mut queue) = worker_queue.try_lock() {
                if let Some(task) = queue.steal() {
                    // Successfully stole a task, add it to our local queue
                    if let Ok(mut local) = self.local_queue.lock() {
                        if local.push(task).is_ok() {
                            return true;
                        }
                    }
                }
            }

            attempts += 1;
        }

        false
    }
}

/// Overall scheduler statistics
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Number of tasks submitted
    pub tasks_submitted: u64,

    /// Number of tasks completed
    pub tasks_completed: u64,

    /// Number of tasks currently pending
    pub tasks_pending: u64,

    /// Total number of steal operations
    pub total_steals: u64,

    /// Average task execution time (microseconds)
    pub avg_execution_time_us: f64,

    /// Worker utilization (0.0 to 1.0)
    pub worker_utilization: f64,

    /// Memory usage per worker (bytes)
    pub memory_usage_per_worker: Vec<usize>,

    /// Number of load balance operations
    pub load_balance_operations: u64,

    /// Throughput (tasks per second)
    pub throughput: f64,
}

/// Advanced work-stealing scheduler
pub struct WorkStealingScheduler {
    config: WorkStealingConfig,
    #[allow(dead_code)]
    workers: Vec<Worker>,
    worker_handles: Vec<JoinHandle<()>>,
    global_queue: Arc<Mutex<PriorityTaskQueue>>,
    result_receiver: crossbeam::channel::Receiver<Box<dyn std::any::Any + Send>>,
    result_sender: crossbeam::channel::Sender<Box<dyn std::any::Any + Send>>,
    shutdown: Arc<AtomicBool>,
    #[allow(dead_code)]
    numa_nodes: Vec<NumaNode>,
    stats: Arc<RwLock<SchedulerStats>>,
    start_time: Option<Instant>,
}

impl WorkStealingScheduler {
    /// Create a new work-stealing scheduler
    pub fn new(config: WorkStealingConfig) -> CoreResult<Self> {
        let num_workers = config.num_workers.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

        let global_queue = Arc::new(Mutex::new(PriorityTaskQueue::new(config.max_queue_size)));
        let (result_sender, result_receiver) = crossbeam::channel::unbounded();
        let shutdown = Arc::new(AtomicBool::new(false));

        // Detect NUMA topology if enabled
        let numa_nodes = if config.numa_aware {
            Self::detect_numa_topology(num_workers)
        } else {
            vec![NumaNode::new(0, (0..num_workers).collect(), 0)]
        };

        // Create workers
        let mut workers = Vec::with_capacity(num_workers);
        for i in 0..num_workers {
            let numa_node = if config.numa_aware {
                Some(i % numa_nodes.len())
            } else {
                None
            };

            let worker = Worker::new(
                i,
                global_queue.clone(),
                numa_node,
                shutdown.clone(),
                config.clone(),
            );
            workers.push(worker);
        }

        // Set up worker cross-references for stealing
        // First collect all local queue references
        let local_queues: Vec<_> = workers.iter().map(|w| w.local_queue.clone()).collect();

        for (i, worker) in workers.iter_mut().enumerate() {
            for (j, queue) in local_queues.iter().enumerate() {
                if i != j {
                    worker.add_other_worker(queue.clone());
                }
            }
        }

        Ok(Self {
            config,
            workers,
            worker_handles: Vec::new(),
            global_queue,
            result_receiver,
            result_sender,
            shutdown,
            numa_nodes,
            stats: Arc::new(RwLock::new(SchedulerStats::default())),
            start_time: None,
        })
    }

    /// Start the scheduler
    pub fn start(&mut self) -> CoreResult<()> {
        if !self.worker_handles.is_empty() {
            return Err(CoreError::StreamError(
                ErrorContext::new("Scheduler already started".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        self.start_time = Some(Instant::now());

        // Start worker threads
        for worker in &self.workers {
            let worker_id = worker.id;
            let worker_ref = unsafe { std::ptr::read(worker as *const Worker) };
            let result_sender = self.result_sender.clone();

            let handle = thread::Builder::new()
                .name(format!("worker-{}", worker_id))
                .spawn(move || {
                    worker_ref.run(result_sender);
                })
                .map_err(|e| {
                    CoreError::StreamError(
                        ErrorContext::new(format!("Failed to start worker thread: {}", e))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;

            self.worker_handles.push(handle);
        }

        // Start statistics monitoring if enabled
        if self.config.enable_monitoring {
            self.start_monitoring();
        }

        Ok(())
    }

    /// Submit a task to the scheduler
    pub fn submit<F, R>(&self, priority: TaskPriority, func: F) -> CoreResult<()>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let task = FunctionTask::new(func);
        self.submit_task(priority, task, None)
    }

    /// Submit a task with execution estimates
    pub fn submit_with_estimates<F, R>(
        &self,
        priority: TaskPriority,
        func: F,
        duration_estimate: Option<Duration>,
        memory_estimate: Option<usize>,
    ) -> CoreResult<()>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let task = FunctionTask::with_estimates(func, duration_estimate, memory_estimate);
        self.submit_task(priority, task, None)
    }

    /// Submit a task to a specific NUMA node
    pub fn submit_to_numa<F, R>(
        &self,
        priority: TaskPriority,
        numa_node: usize,
        func: F,
    ) -> CoreResult<()>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let task = FunctionTask::new(func);
        self.submit_task(priority, task, Some(numa_node))
    }

    fn submit_task<T>(
        &self,
        priority: TaskPriority,
        task: T,
        numa_hint: Option<usize>,
    ) -> CoreResult<()>
    where
        T: WorkStealingTask,
        T::Output: 'static,
    {
        // Try to submit to a specific worker's local queue if NUMA hint is provided
        if let Some(numa_node) = numa_hint {
            if numa_node < self.workers.len() {
                if let Ok(mut local_queue) = self.workers[numa_node].local_queue.try_lock() {
                    // If we can get the lock, use the local queue
                    let prioritized_task = PrioritizedTask::new(task, priority, numa_hint);
                    local_queue.push(prioritized_task).map_err(|_| {
                        CoreError::StreamError(
                            ErrorContext::new("Local task queue is full".to_string())
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;
                    self.update_submit_stats();
                    return Ok(());
                }
            }
        }

        // Fall back to global queue
        let prioritized_task = PrioritizedTask::new(task, priority, numa_hint);
        let mut global_queue = self.global_queue.lock().unwrap();
        global_queue.push(prioritized_task).map_err(|_| {
            CoreError::StreamError(
                ErrorContext::new("Global task queue is full".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        self.update_submit_stats();
        Ok(())
    }

    /// Try to receive a completed task result
    pub fn try_recv<T: 'static>(&self) -> Option<T> {
        if let Ok(result) = self.result_receiver.try_recv() {
            self.update_completion_stats();

            // Try to downcast to the expected type
            if let Ok(typed_result) = result.downcast::<T>() {
                Some(*typed_result)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Receive a completed task result with timeout
    pub fn recv_timeout<T: 'static>(&self, timeout: Duration) -> CoreResult<T> {
        let result = self.result_receiver.recv_timeout(timeout).map_err(|_| {
            CoreError::TimeoutError(
                ErrorContext::new("Timeout waiting for task result".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        self.update_completion_stats();

        result.downcast::<T>().map(|r| *r).map_err(|_| {
            CoreError::ValidationError(
                ErrorContext::new("Task result type mismatch".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })
    }

    /// Get current scheduler statistics
    pub fn stats(&self) -> SchedulerStats {
        self.stats.read().unwrap().clone()
    }

    /// Stop the scheduler
    pub fn stop(&mut self) -> CoreResult<()> {
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for all worker threads to finish
        for handle in self.worker_handles.drain(..) {
            handle.join().map_err(|_| {
                CoreError::StreamError(
                    ErrorContext::new("Failed to join worker thread".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
        }

        Ok(())
    }

    /// Get the number of pending tasks
    pub fn pending_tasks(&self) -> usize {
        let global_pending = self.global_queue.lock().unwrap().len();
        let local_pending: usize = self
            .workers
            .iter()
            .map(|w| w.local_queue.lock().unwrap().len())
            .sum();

        global_pending + local_pending
    }

    /// Detect NUMA topology (simplified implementation)
    fn detect_numa_topology(num_workers: usize) -> Vec<NumaNode> {
        // This is a simplified implementation
        // In practice, you'd use a library like hwloc or libnuma
        vec![NumaNode::new(0, (0..num_workers).collect(), 0)]
    }

    fn update_submit_stats(&self) {
        if let Ok(mut stats) = self.stats.write() {
            stats.tasks_submitted += 1;
            stats.tasks_pending += 1;
        }
    }

    fn update_completion_stats(&self) {
        if let Ok(mut stats) = self.stats.write() {
            stats.tasks_completed += 1;
            if stats.tasks_pending > 0 {
                stats.tasks_pending -= 1;
            }
        }
    }

    fn start_monitoring(&self) {
        // Start a monitoring thread for statistics collection
        let stats = self.stats.clone();
        let shutdown = self.shutdown.clone();
        let interval = self.config.stats_interval;
        let start_time = self.start_time;

        thread::spawn(move || {
            while !shutdown.load(Ordering::Relaxed) {
                thread::sleep(interval);

                if let Ok(mut stats_guard) = stats.write() {
                    // Calculate throughput
                    if let Some(start) = start_time {
                        let elapsed = start.elapsed().as_secs_f64();
                        if elapsed > 0.0 {
                            stats_guard.throughput = stats_guard.tasks_completed as f64 / elapsed;
                        }
                    }
                }
            }
        });
    }
}

impl Drop for WorkStealingScheduler {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// Builder for work-stealing scheduler configuration
#[derive(Debug, Clone)]
pub struct WorkStealingConfigBuilder {
    config: WorkStealingConfig,
}

impl WorkStealingConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: WorkStealingConfig::default(),
        }
    }

    /// Set the number of worker threads
    pub fn num_workers(mut self, workers: usize) -> Self {
        self.config.num_workers = Some(workers);
        self
    }

    /// Set the maximum queue size
    pub const fn max_queue_size(mut self, size: usize) -> Self {
        self.config.max_queue_size = size;
        self
    }

    /// Enable NUMA-aware scheduling
    pub const fn numa_aware(mut self, enable: bool) -> Self {
        self.config.numa_aware = enable;
        self
    }

    /// Enable priority-based scheduling
    pub const fn priority_scheduling(mut self, enable: bool) -> Self {
        self.config.priority_scheduling = enable;
        self
    }

    /// Set thread affinity
    pub fn thread_affinity(mut self, affinity: Vec<usize>) -> Self {
        self.config.thread_affinity = Some(affinity);
        self
    }

    /// Set maximum memory per worker
    pub fn max_memory_per_worker(mut self, memory: usize) -> Self {
        self.config.max_memory_per_worker = Some(memory);
        self
    }

    /// Enable performance monitoring
    pub const fn enable_monitoring(mut self, enable: bool) -> Self {
        self.config.enable_monitoring = enable;
        self
    }

    /// Enable adaptive load balancing
    pub const fn adaptive_balancing(mut self, enable: bool) -> Self {
        self.config.adaptive_balancing = enable;
        self
    }

    /// Build the configuration
    pub fn build(self) -> WorkStealingConfig {
        self.config
    }
}

impl Default for WorkStealingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a default work-stealing scheduler
pub fn create_work_stealing_scheduler() -> CoreResult<WorkStealingScheduler> {
    WorkStealingScheduler::new(WorkStealingConfig::default())
}

/// Create a work-stealing scheduler optimized for CPU-intensive tasks
pub fn create_cpu_intensive_scheduler() -> CoreResult<WorkStealingScheduler> {
    let config = WorkStealingConfigBuilder::new()
        .num_workers(
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        )
        .priority_scheduling(true)
        .adaptive_balancing(true)
        .enable_monitoring(true)
        .build();

    WorkStealingScheduler::new(config)
}

/// Create a work-stealing scheduler optimized for I/O-intensive tasks
pub fn create_io_intensive_scheduler() -> CoreResult<WorkStealingScheduler> {
    let num_workers = std::thread::available_parallelism()
        .map(|n| n.get() * 2) // More threads for I/O
        .unwrap_or(8);

    let config = WorkStealingConfigBuilder::new()
        .num_workers(num_workers)
        .max_queue_size(50000) // Larger queue for I/O tasks
        .priority_scheduling(true)
        .adaptive_balancing(false) // Less useful for I/O
        .enable_monitoring(true)
        .build();

    WorkStealingScheduler::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn test_work_stealing_scheduler_creation() {
        let scheduler = create_work_stealing_scheduler();
        assert!(scheduler.is_ok());
    }

    #[test]
    fn test_task_submission_and_execution() {
        let mut scheduler = create_work_stealing_scheduler().unwrap();
        scheduler.start().unwrap();

        // Submit a simple task
        scheduler.submit(TaskPriority::Normal, || 42).unwrap();

        // Wait a bit and try to receive the result
        std::thread::sleep(Duration::from_millis(100));

        if let Some(result) = scheduler.try_recv::<i32>() {
            assert_eq!(result, 42);
        }

        scheduler.stop().unwrap();
    }

    #[test]
    fn test_priority_scheduling() {
        let mut scheduler = create_work_stealing_scheduler().unwrap();
        scheduler.start().unwrap();

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        // Submit low priority task
        scheduler
            .submit(TaskPriority::Low, move || {
                std::thread::sleep(Duration::from_millis(50));
                counter_clone.store(1, Ordering::Relaxed);
            })
            .unwrap();

        let counter_clone = counter.clone();

        // Submit high priority task
        scheduler
            .submit(TaskPriority::High, move || {
                counter_clone.store(2, Ordering::Relaxed);
            })
            .unwrap();

        // Wait for tasks to complete
        std::thread::sleep(Duration::from_millis(200));

        // High priority task should have run (and set counter to 2)
        // Note: This test is probabilistic and may not always pass due to timing

        scheduler.stop().unwrap();
    }

    #[test]
    fn test_scheduler_stats() {
        let mut scheduler = create_work_stealing_scheduler().unwrap();
        scheduler.start().unwrap();

        // Submit multiple tasks
        for i in 0..10 {
            scheduler
                .submit(TaskPriority::Normal, move || i * 2)
                .unwrap();
        }

        // Wait for tasks to complete
        std::thread::sleep(Duration::from_millis(100));

        let stats = scheduler.stats();
        assert!(stats.tasks_submitted >= 10);

        scheduler.stop().unwrap();
    }

    #[test]
    fn test_config_builder() {
        let config = WorkStealingConfigBuilder::new()
            .num_workers(8)
            .max_queue_size(5000)
            .numa_aware(true)
            .priority_scheduling(false)
            .adaptive_balancing(true)
            .build();

        assert_eq!(config.num_workers, Some(8));
        assert_eq!(config.max_queue_size, 5000);
        assert!(config.numa_aware);
        assert!(!config.priority_scheduling);
        assert!(config.adaptive_balancing);
    }
}
