//! Parallel processing and thread pool optimizations
//!
//! This module provides thread pool management and parallel execution
//! optimizations for tensor operations, particularly targeting CPU performance
//! improvements for large-scale computations.

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub mod parallel_ops;
pub mod thread_pool;
pub mod work_stealing;

/// Global thread pool manager
static GLOBAL_THREAD_POOL: std::sync::LazyLock<Arc<Mutex<Option<ThreadPool>>>> =
    std::sync::LazyLock::new(|| Arc::new(Mutex::new(None)));

/// Configuration for thread pool optimization
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Number of worker threads
    pub num_threads: usize,
    /// Maximum queue size per thread
    pub max_queue_size: usize,
    /// Enable work stealing between threads
    pub work_stealing: bool,
    /// Thread priority level
    pub priority: ThreadPriority,
    /// CPU affinity settings
    pub cpu_affinity: CpuAffinity,
    /// Idle timeout for threads
    pub idle_timeout: Duration,
    /// Enable adaptive scheduling
    pub adaptive_scheduling: bool,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            num_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            max_queue_size: 1000,
            work_stealing: true,
            priority: ThreadPriority::Normal,
            cpu_affinity: CpuAffinity::Auto,
            idle_timeout: Duration::from_secs(60),
            adaptive_scheduling: true,
        }
    }
}

/// Thread priority levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThreadPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// CPU affinity configuration
#[derive(Debug, Clone)]
pub enum CpuAffinity {
    /// Automatic assignment
    Auto,
    /// Specific CPU cores
    Cores(Vec<usize>),
    /// NUMA-aware assignment
    Numa,
}

/// Thread pool for parallel execution
pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: Sender<Job>,
    config: ThreadPoolConfig,
    stats: Arc<Mutex<ThreadPoolStats>>,
}

impl ThreadPool {
    /// Create a new thread pool with default configuration
    pub fn new() -> Self {
        Self::with_config(ThreadPoolConfig::default())
    }

    /// Create a new thread pool with custom configuration
    pub fn with_config(config: ThreadPoolConfig) -> Self {
        let (sender, receiver) = channel();
        let receiver = Arc::new(Mutex::new(receiver));

        // Initialize stats with proper worker_stats vector
        let mut stats_data = ThreadPoolStats::new();
        stats_data.worker_stats = (0..config.num_threads).map(WorkerStats::new).collect();
        let stats = Arc::new(Mutex::new(stats_data));

        let mut workers = Vec::with_capacity(config.num_threads);

        for id in 0..config.num_threads {
            workers.push(Worker::new(
                id,
                Arc::clone(&receiver),
                Arc::clone(&stats),
                config.clone(),
            ));
        }

        ThreadPool {
            workers,
            sender,
            config,
            stats,
        }
    }

    /// Execute a closure on the thread pool
    pub fn execute<F>(&self, f: F) -> Result<(), ThreadPoolError>
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender
            .send(job)
            .map_err(|_| ThreadPoolError::QueueFull)
    }

    /// Execute a closure and wait for completion
    pub fn execute_and_wait<F, R>(&self, f: F) -> Result<R, ThreadPoolError>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::channel();

        self.execute(move || {
            let result = f();
            let _ = tx.send(result);
        })?;

        rx.recv().map_err(|_| ThreadPoolError::ExecutionFailed)
    }

    /// Execute multiple tasks in parallel
    pub fn execute_parallel<F, I>(&self, tasks: I) -> Result<Vec<()>, ThreadPoolError>
    where
        F: FnOnce() + Send + 'static,
        I: IntoIterator<Item = F>,
    {
        let tasks: Vec<F> = tasks.into_iter().collect();
        let mut handles = Vec::with_capacity(tasks.len());

        for task in tasks {
            let (tx, rx) = std::sync::mpsc::channel();

            self.execute(move || {
                task();
                let _ = tx.send(());
            })?;

            handles.push(rx);
        }

        // Wait for all tasks to complete
        let num_handles = handles.len();
        for rx in handles {
            rx.recv().map_err(|_| ThreadPoolError::ExecutionFailed)?;
        }

        Ok(vec![(); num_handles])
    }

    /// Get thread pool statistics
    pub fn get_stats(&self) -> ThreadPoolStats {
        self.stats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Get current configuration
    pub fn get_config(&self) -> &ThreadPoolConfig {
        &self.config
    }

    /// Resize the thread pool
    pub fn resize(&mut self, new_size: usize) -> Result<(), ThreadPoolError> {
        if new_size == 0 {
            return Err(ThreadPoolError::InvalidConfiguration(
                "Thread pool size cannot be zero".into(),
            ));
        }

        // Implementation would recreate the thread pool with new size
        // For now, just update the config
        self.config.num_threads = new_size;
        Ok(())
    }

    /// Shutdown the thread pool gracefully
    pub fn shutdown(self) -> Result<(), ThreadPoolError> {
        // Drop the sender to signal shutdown
        drop(self.sender);

        // Wait for all workers to finish
        for worker in self.workers {
            if let Some(thread) = worker.thread {
                thread.join().map_err(|_| ThreadPoolError::ShutdownFailed)?;
            }
        }

        Ok(())
    }
}

impl Default for ThreadPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Worker thread in the thread pool
struct Worker {
    #[allow(dead_code)]
    id: usize,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    fn new(
        id: usize,
        receiver: Arc<Mutex<Receiver<Job>>>,
        stats: Arc<Mutex<ThreadPoolStats>>,
        config: ThreadPoolConfig,
    ) -> Worker {
        let thread = thread::spawn(move || {
            // Set thread priority if supported
            Self::set_thread_priority(config.priority);

            // Set CPU affinity if specified
            Self::set_cpu_affinity(id, &config.cpu_affinity);

            loop {
                let job = {
                    let receiver = receiver.lock().unwrap();
                    receiver.recv()
                };

                match job {
                    Ok(job) => {
                        let start = Instant::now();
                        job();
                        let duration = start.elapsed();

                        // Update statistics
                        {
                            if let Ok(mut stats) = stats.lock() {
                                stats.tasks_completed += 1;
                                stats.total_execution_time += duration;
                                if id < stats.worker_stats.len() {
                                    stats.worker_stats[id].tasks_completed += 1;
                                    stats.worker_stats[id].total_time += duration;
                                }
                            }
                        }
                    }
                    Err(_) => {
                        // Channel closed, shutdown
                        break;
                    }
                }
            }
        });

        Worker {
            id,
            thread: Some(thread),
        }
    }

    fn set_thread_priority(_priority: ThreadPriority) {
        // Platform-specific thread priority setting would go here
        // For now, this is a no-op
    }

    fn set_cpu_affinity(_worker_id: usize, _affinity: &CpuAffinity) {
        // Platform-specific CPU affinity setting would go here
        // For now, this is a no-op
    }
}

/// Job type for the thread pool
type Job = Box<dyn FnOnce() + Send + 'static>;

/// Statistics for thread pool performance monitoring
#[derive(Debug, Clone)]
pub struct ThreadPoolStats {
    /// Total tasks completed
    pub tasks_completed: u64,
    /// Total execution time across all threads
    pub total_execution_time: Duration,
    /// Current queue size
    pub current_queue_size: usize,
    /// Maximum queue size reached
    pub max_queue_size: usize,
    /// Number of active threads
    pub active_threads: usize,
    /// Per-worker statistics
    pub worker_stats: Vec<WorkerStats>,
    /// Load balancing efficiency
    pub load_balance_ratio: f32,
}

impl ThreadPoolStats {
    fn new() -> Self {
        Self {
            tasks_completed: 0,
            total_execution_time: Duration::ZERO,
            current_queue_size: 0,
            max_queue_size: 0,
            active_threads: 0,
            worker_stats: Vec::new(),
            load_balance_ratio: 1.0,
        }
    }

    /// Calculate average task execution time
    pub fn average_execution_time(&self) -> Duration {
        if self.tasks_completed == 0 {
            Duration::ZERO
        } else {
            self.total_execution_time / self.tasks_completed as u32
        }
    }

    /// Calculate tasks per second
    pub fn tasks_per_second(&self) -> f64 {
        if self.total_execution_time.is_zero() {
            0.0
        } else {
            self.tasks_completed as f64 / self.total_execution_time.as_secs_f64()
        }
    }

    /// Calculate thread utilization
    pub fn thread_utilization(&self) -> f64 {
        if self.worker_stats.is_empty() {
            return 0.0;
        }

        let total_time: Duration = self.worker_stats.iter().map(|stats| stats.total_time).sum();

        let max_time = self
            .worker_stats
            .iter()
            .map(|stats| stats.total_time)
            .max()
            .unwrap_or(Duration::ZERO);

        if max_time.is_zero() {
            0.0
        } else {
            total_time.as_secs_f64() / (max_time.as_secs_f64() * self.worker_stats.len() as f64)
        }
    }
}

/// Statistics for individual worker threads
#[derive(Debug, Clone)]
pub struct WorkerStats {
    /// Worker ID
    pub worker_id: usize,
    /// Tasks completed by this worker
    pub tasks_completed: u64,
    /// Total execution time for this worker
    pub total_time: Duration,
    /// Current queue size for this worker
    pub queue_size: usize,
    /// Last activity timestamp
    pub last_activity: Option<Instant>,
}

impl WorkerStats {
    fn new(worker_id: usize) -> Self {
        Self {
            worker_id,
            tasks_completed: 0,
            total_time: Duration::ZERO,
            queue_size: 0,
            last_activity: None,
        }
    }
}

/// Parallel execution scheduler for tensor operations
pub struct ParallelScheduler {
    thread_pool: Arc<ThreadPool>,
    config: SchedulerConfig,
}

impl ParallelScheduler {
    /// Create a new parallel scheduler
    pub fn new() -> Self {
        let thread_pool = Arc::new(ThreadPool::new());
        Self {
            thread_pool,
            config: SchedulerConfig::default(),
        }
    }

    /// Create a scheduler with custom thread pool
    pub fn with_thread_pool(thread_pool: Arc<ThreadPool>) -> Self {
        Self {
            thread_pool,
            config: SchedulerConfig::default(),
        }
    }

    /// Schedule a parallel tensor operation
    pub fn schedule_operation<F, R>(&self, operation: F) -> Result<R, ThreadPoolError>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        if self.should_parallelize(&operation) {
            self.thread_pool.execute_and_wait(operation)
        } else {
            // Execute on current thread for small operations
            Ok(operation())
        }
    }

    /// Schedule multiple parallel operations
    pub fn schedule_batch<F>(&self, operations: Vec<F>) -> Result<Vec<()>, ThreadPoolError>
    where
        F: FnOnce() + Send + 'static,
    {
        if operations.len() <= 1 || !self.config.enable_batching {
            // Execute sequentially for small batches
            for op in operations {
                op();
            }
            Ok(vec![])
        } else {
            self.thread_pool.execute_parallel(operations)
        }
    }

    /// Check if an operation should be parallelized
    fn should_parallelize<F, R>(&self, _operation: &F) -> bool
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        // Heuristics for deciding whether to parallelize:
        // - Operation complexity
        // - Data size
        // - Current thread pool load
        // - Overhead considerations

        true // Simplified decision
    }

    /// Get scheduler statistics
    pub fn get_stats(&self) -> ThreadPoolStats {
        self.thread_pool.get_stats()
    }
}

impl Default for ParallelScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the parallel scheduler
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Enable automatic batching of operations
    pub enable_batching: bool,
    /// Minimum operation size for parallelization
    pub min_parallel_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            enable_batching: true,
            min_parallel_size: 1000,
            max_batch_size: 100,
            load_balancing: LoadBalancingStrategy::RoundRobin,
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    /// Simple round-robin assignment
    RoundRobin,
    /// Assign to least loaded worker
    LeastLoaded,
    /// Work stealing between workers
    WorkStealing,
    /// Adaptive based on operation characteristics
    Adaptive,
}

/// Errors that can occur in thread pool operations
#[derive(Debug, thiserror::Error)]
pub enum ThreadPoolError {
    #[error("Thread pool queue is full")]
    QueueFull,
    #[error("Task execution failed")]
    ExecutionFailed,
    #[error("Thread pool shutdown failed")]
    ShutdownFailed,
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    #[error("Worker thread panicked")]
    WorkerPanic,
}

/// Public API functions for thread pool management
/// Initialize the global thread pool with default configuration
pub fn init_thread_pool() -> Result<(), ThreadPoolError> {
    let mut pool = GLOBAL_THREAD_POOL.lock().unwrap();
    if pool.is_none() {
        *pool = Some(ThreadPool::new());
    }
    Ok(())
}

/// Initialize the global thread pool with custom configuration
pub fn init_thread_pool_with_config(config: ThreadPoolConfig) -> Result<(), ThreadPoolError> {
    let mut pool = GLOBAL_THREAD_POOL.lock().unwrap();
    *pool = Some(ThreadPool::with_config(config));
    Ok(())
}

/// Execute a task on the global thread pool
pub fn execute_global<F>(f: F) -> Result<(), ThreadPoolError>
where
    F: FnOnce() + Send + 'static,
{
    let pool = GLOBAL_THREAD_POOL.lock().unwrap();
    if let Some(ref pool) = *pool {
        pool.execute(f)
    } else {
        Err(ThreadPoolError::InvalidConfiguration(
            "Thread pool not initialized".into(),
        ))
    }
}

/// Execute a task and wait for completion on the global thread pool
pub fn execute_and_wait_global<F, R>(f: F) -> Result<R, ThreadPoolError>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let pool = GLOBAL_THREAD_POOL.lock().unwrap();
    if let Some(ref pool) = *pool {
        pool.execute_and_wait(f)
    } else {
        Err(ThreadPoolError::InvalidConfiguration(
            "Thread pool not initialized".into(),
        ))
    }
}

/// Get global thread pool statistics
pub fn get_global_thread_pool_stats() -> Option<ThreadPoolStats> {
    let pool = GLOBAL_THREAD_POOL.lock().unwrap();
    pool.as_ref().map(|p| p.get_stats())
}

/// Shutdown the global thread pool
pub fn shutdown_global_thread_pool() -> Result<(), ThreadPoolError> {
    let mut pool = GLOBAL_THREAD_POOL.lock().unwrap();
    if let Some(pool) = pool.take() {
        pool.shutdown()
    } else {
        Ok(())
    }
}

/// Set the number of threads for the global thread pool
pub fn set_global_thread_count(count: usize) -> Result<(), ThreadPoolError> {
    let config = ThreadPoolConfig {
        num_threads: count,
        ..Default::default()
    };
    init_thread_pool_with_config(config)
}

/// Get the current number of threads in the global thread pool
pub fn get_global_thread_count() -> usize {
    let pool = GLOBAL_THREAD_POOL.lock().unwrap();
    pool.as_ref()
        .map(|p| p.get_config().num_threads)
        .unwrap_or(0)
}

/// Check if the global thread pool is initialized
pub fn is_thread_pool_initialized() -> bool {
    let pool = GLOBAL_THREAD_POOL.lock().unwrap();
    pool.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_thread_pool_creation() {
        let pool = ThreadPool::new();
        assert!(pool.get_config().num_threads > 0);

        let config = ThreadPoolConfig {
            num_threads: 2,
            work_stealing: false,
            ..Default::default()
        };
        let custom_pool = ThreadPool::with_config(config);
        assert_eq!(custom_pool.get_config().num_threads, 2);
        assert!(!custom_pool.get_config().work_stealing);
    }

    #[test]
    fn test_thread_pool_execution() {
        let pool = ThreadPool::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        pool.execute(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap();

        // Give the task time to execute
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_thread_pool_execute_and_wait() {
        let pool = ThreadPool::new();

        let result = pool.execute_and_wait(|| 42).unwrap();

        assert_eq!(result, 42);
    }

    #[test]
    fn test_thread_pool_parallel_execution() {
        let pool = ThreadPool::new();
        let counter = Arc::new(AtomicUsize::new(0));

        let tasks: Vec<_> = (0..5)
            .map(|_| {
                let counter_clone = Arc::clone(&counter);
                move || {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                }
            })
            .collect();

        pool.execute_parallel(tasks).unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_thread_pool_stats() {
        let pool = ThreadPool::new();
        let stats = pool.get_stats();

        // Initially no tasks completed
        assert_eq!(stats.tasks_completed, 0);
        assert_eq!(stats.average_execution_time(), Duration::ZERO);
    }

    #[test]
    fn test_parallel_scheduler() {
        let scheduler = ParallelScheduler::new();

        let result = scheduler.schedule_operation(|| 100).unwrap();

        assert_eq!(result, 100);
    }

    #[test]
    fn test_global_thread_pool() {
        // Clean shutdown first in case of previous test failures
        let _ = shutdown_global_thread_pool();

        // Initialize fresh thread pool
        init_thread_pool().unwrap();
        assert!(is_thread_pool_initialized());

        // Test execute and wait (more reliable than async execute)
        let result = execute_and_wait_global(|| 42);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        // Get stats (handle potential poisoned mutex gracefully)
        let stats = get_global_thread_pool_stats();
        assert!(stats.is_some());

        // Get thread count
        let thread_count = get_global_thread_count();
        assert!(thread_count > 0);

        // Shutdown
        shutdown_global_thread_pool().unwrap();
        assert!(!is_thread_pool_initialized());
    }

    #[test]
    fn test_thread_pool_config() {
        let config = ThreadPoolConfig::default();
        assert!(config.num_threads > 0);
        assert!(config.work_stealing);
        assert_eq!(config.priority, ThreadPriority::Normal);

        let custom_config = ThreadPoolConfig {
            num_threads: 8,
            max_queue_size: 500,
            work_stealing: false,
            priority: ThreadPriority::High,
            cpu_affinity: CpuAffinity::Cores(vec![0, 1, 2, 3]),
            idle_timeout: Duration::from_secs(30),
            adaptive_scheduling: false,
        };

        assert_eq!(custom_config.num_threads, 8);
        assert_eq!(custom_config.max_queue_size, 500);
        assert!(!custom_config.work_stealing);
        assert_eq!(custom_config.priority, ThreadPriority::High);
    }

    #[test]
    fn test_scheduler_config() {
        let config = SchedulerConfig::default();
        assert!(config.enable_batching);
        assert_eq!(config.min_parallel_size, 1000);
        assert_eq!(config.max_batch_size, 100);
        assert!(matches!(
            config.load_balancing,
            LoadBalancingStrategy::RoundRobin
        ));
    }

    #[test]
    fn test_worker_stats() {
        let stats = WorkerStats::new(0);
        assert_eq!(stats.worker_id, 0);
        assert_eq!(stats.tasks_completed, 0);
        assert_eq!(stats.total_time, Duration::ZERO);
        assert_eq!(stats.queue_size, 0);
        assert!(stats.last_activity.is_none());
    }

    #[test]
    fn test_thread_pool_stats_calculations() {
        let mut stats = ThreadPoolStats::new();
        stats.tasks_completed = 10;
        stats.total_execution_time = Duration::from_secs(5);

        assert_eq!(stats.average_execution_time(), Duration::from_millis(500));
        assert_eq!(stats.tasks_per_second(), 2.0);
    }
}
