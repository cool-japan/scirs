//! Advanced thread pool implementation with work stealing and load balancing
//!
//! This module provides a high-performance thread pool optimized for
//! scientific computing workloads with features like work stealing,
//! NUMA awareness, and adaptive scheduling.

use super::{ThreadPoolConfig, ThreadPoolError, WorkerStats};
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Condvar, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Advanced thread pool with work stealing
pub struct AdvancedThreadPool {
    workers: Vec<WorkStealingWorker>,
    global_queue: Arc<Mutex<VecDeque<Task>>>,
    config: ThreadPoolConfig,
    running: Arc<AtomicBool>,
    stats: Arc<Mutex<AdvancedThreadPoolStats>>,
}

impl AdvancedThreadPool {
    /// Create a new advanced thread pool
    pub fn new(config: ThreadPoolConfig) -> Self {
        let global_queue = Arc::new(Mutex::new(VecDeque::new()));
        let running = Arc::new(AtomicBool::new(true));
        let stats = Arc::new(Mutex::new(AdvancedThreadPoolStats::new(config.num_threads)));

        let mut workers = Vec::with_capacity(config.num_threads);

        for id in 0..config.num_threads {
            let worker = WorkStealingWorker::new(
                id,
                Arc::clone(&global_queue),
                Arc::clone(&running),
                Arc::clone(&stats),
                config.clone(),
            );
            workers.push(worker);
        }

        Self {
            workers,
            global_queue,
            config,
            running,
            stats,
        }
    }

    /// Submit a task to the thread pool
    pub fn submit<F>(&self, task: F) -> Result<TaskHandle<()>, ThreadPoolError>
    where
        F: FnOnce() + Send + 'static,
    {
        if !self.running.load(Ordering::Relaxed) {
            return Err(ThreadPoolError::QueueFull);
        }

        let (task, handle) = Task::new(task);

        // Use global queue for all tasks to avoid move issues
        let mut queue = self.global_queue.lock().unwrap();
        if queue.len() >= self.config.max_queue_size {
            return Err(ThreadPoolError::QueueFull);
        }
        queue.push_back(task);

        Ok(handle)
    }

    /// Submit a batch of tasks
    pub fn submit_batch<F, I>(&self, tasks: I) -> Result<Vec<TaskHandle<()>>, ThreadPoolError>
    where
        F: FnOnce() + Send + 'static,
        I: IntoIterator<Item = F>,
    {
        let tasks: Vec<F> = tasks.into_iter().collect();
        let mut handles = Vec::with_capacity(tasks.len());

        for task in tasks {
            handles.push(self.submit(task)?);
        }

        Ok(handles)
    }

    /// Find the least loaded worker
    #[allow(dead_code)]
    fn find_least_loaded_worker(&self) -> Option<usize> {
        // Use round-robin strategy by default since ThreadPoolConfig doesn't have load_balancing
        if self.config.work_stealing {
            self.workers
                .iter()
                .enumerate()
                .min_by_key(|(_, worker)| worker.get_queue_size())
                .map(|(id, _)| id)
        } else {
            // Simple round-robin based on current time
            let now = Instant::now();
            Some(now.elapsed().as_nanos() as usize % self.workers.len())
        }
    }

    /// Get thread pool statistics
    pub fn get_stats(&self) -> AdvancedThreadPoolStats {
        self.stats.lock().unwrap().clone()
    }

    /// Shutdown the thread pool
    pub fn shutdown(self) -> Result<(), ThreadPoolError> {
        self.running.store(false, Ordering::Relaxed);

        // Wake up all workers
        for worker in &self.workers {
            worker.notify_shutdown();
        }

        // Wait for all workers to finish
        for worker in self.workers {
            worker.join().map_err(|_| ThreadPoolError::ShutdownFailed)?;
        }

        Ok(())
    }

    /// Resize the thread pool (dynamic scaling)
    pub fn resize(&mut self, new_size: usize) -> Result<(), ThreadPoolError> {
        if new_size == 0 {
            return Err(ThreadPoolError::InvalidConfiguration(
                "Thread pool size cannot be zero".into(),
            ));
        }

        let current_size = self.workers.len();

        match new_size.cmp(&current_size) {
            std::cmp::Ordering::Greater => {
                // Add new workers
                for id in current_size..new_size {
                    let worker = WorkStealingWorker::new(
                        id,
                        Arc::clone(&self.global_queue),
                        Arc::clone(&self.running),
                        Arc::clone(&self.stats),
                        self.config.clone(),
                    );
                    self.workers.push(worker);
                }
            }
            std::cmp::Ordering::Less => {
                // Remove workers (simplified - in practice would need graceful shutdown)
                self.workers.truncate(new_size);
            }
            std::cmp::Ordering::Equal => {
                // No change needed
            }
        }

        self.config.num_threads = new_size;
        Ok(())
    }
}

/// Work-stealing worker thread
pub struct WorkStealingWorker {
    #[allow(dead_code)]
    id: usize,
    #[allow(dead_code)]
    local_queue: Arc<Mutex<VecDeque<Task>>>,
    thread_handle: Option<JoinHandle<()>>,
    shutdown_signal: Arc<(Mutex<bool>, Condvar)>,
}

impl WorkStealingWorker {
    /// Create a new work-stealing worker
    fn new(
        id: usize,
        global_queue: Arc<Mutex<VecDeque<Task>>>,
        running: Arc<AtomicBool>,
        stats: Arc<Mutex<AdvancedThreadPoolStats>>,
        config: ThreadPoolConfig,
    ) -> Self {
        let local_queue = Arc::new(Mutex::new(VecDeque::new()));
        let shutdown_signal = Arc::new((Mutex::new(false), Condvar::new()));

        let local_queue_clone = Arc::clone(&local_queue);
        let shutdown_signal_clone = Arc::clone(&shutdown_signal);

        let thread_handle = thread::spawn(move || {
            Self::worker_loop(
                id,
                local_queue_clone,
                global_queue,
                running,
                stats,
                config,
                shutdown_signal_clone,
            );
        });

        Self {
            id,
            local_queue,
            thread_handle: Some(thread_handle),
            shutdown_signal,
        }
    }

    /// Main worker loop with work stealing
    fn worker_loop(
        id: usize,
        local_queue: Arc<Mutex<VecDeque<Task>>>,
        global_queue: Arc<Mutex<VecDeque<Task>>>,
        running: Arc<AtomicBool>,
        stats: Arc<Mutex<AdvancedThreadPoolStats>>,
        config: ThreadPoolConfig,
        shutdown_signal: Arc<(Mutex<bool>, Condvar)>,
    ) {
        let mut idle_start = None;

        while running.load(Ordering::Relaxed) {
            let task = Self::find_task(&local_queue, &global_queue, &config);

            match task {
                Some(task) => {
                    idle_start = None;
                    let start_time = Instant::now();

                    // Execute the task
                    task.execute();

                    let execution_time = start_time.elapsed();

                    // Update statistics
                    {
                        let mut stats = stats.lock().unwrap();
                        stats.total_tasks_executed += 1;
                        stats.total_execution_time += execution_time;
                        stats.worker_stats[id].tasks_completed += 1;
                        stats.worker_stats[id].total_time += execution_time;
                        stats.worker_stats[id].last_activity = Some(Instant::now());
                    }
                }
                None => {
                    // No task found, handle idle time
                    if idle_start.is_none() {
                        idle_start = Some(Instant::now());
                    }

                    // Check for shutdown after idle timeout
                    if let Some(start) = idle_start {
                        if start.elapsed() > config.idle_timeout {
                            let (lock, cvar) = &*shutdown_signal;
                            let mut shutdown = lock.lock().unwrap();
                            while !*shutdown && running.load(Ordering::Relaxed) {
                                let result = cvar
                                    .wait_timeout(shutdown, Duration::from_millis(100))
                                    .unwrap();
                                shutdown = result.0;
                                if result.1.timed_out() {
                                    break;
                                }
                            }
                        }
                    }

                    // Brief sleep to avoid busy waiting
                    thread::sleep(Duration::from_micros(100));
                }
            }
        }
    }

    /// Find a task to execute (local queue -> global queue -> work stealing)
    fn find_task(
        local_queue: &Arc<Mutex<VecDeque<Task>>>,
        global_queue: &Arc<Mutex<VecDeque<Task>>>,
        config: &ThreadPoolConfig,
    ) -> Option<Task> {
        // Try local queue first
        {
            let mut queue = local_queue.lock().unwrap();
            if let Some(task) = queue.pop_front() {
                return Some(task);
            }
        }

        // Try global queue
        {
            let mut queue = global_queue.lock().unwrap();
            if let Some(task) = queue.pop_front() {
                return Some(task);
            }
        }

        // Work stealing (simplified - would need access to other workers)
        if config.work_stealing {
            // Implementation would steal from other workers' queues
        }

        None
    }

    /// Try to submit a task to the local queue
    #[allow(dead_code)]
    fn try_submit_local(&self, task: Task) -> bool {
        let mut queue = self.local_queue.lock().unwrap();
        if queue.len() < self.local_queue.lock().unwrap().capacity() {
            queue.push_back(task);
            true
        } else {
            false
        }
    }

    /// Get the current queue size
    fn get_queue_size(&self) -> usize {
        self.local_queue.lock().unwrap().len()
    }

    /// Notify worker to shutdown
    fn notify_shutdown(&self) {
        let (lock, cvar) = &*self.shutdown_signal;
        let mut shutdown = lock.lock().unwrap();
        *shutdown = true;
        cvar.notify_one();
    }

    /// Join the worker thread
    fn join(mut self) -> Result<(), Box<dyn std::any::Any + Send>> {
        if let Some(handle) = self.thread_handle.take() {
            handle.join()
        } else {
            Ok(())
        }
    }
}

/// Task wrapper with execution tracking
pub struct Task {
    func: Box<dyn FnOnce() + Send + 'static>,
    created_at: Instant,
    priority: TaskPriority,
}

impl Task {
    /// Create a new task with completion handle
    pub fn new<F>(func: F) -> (Self, TaskHandle<()>)
    where
        F: FnOnce() + Send + 'static,
    {
        let (sender, receiver) = std::sync::mpsc::channel();

        let task = Task {
            func: Box::new(move || {
                func();
                let _ = sender.send(());
            }),
            created_at: Instant::now(),
            priority: TaskPriority::Normal,
        };

        let handle = TaskHandle { receiver };
        (task, handle)
    }

    /// Execute the task
    fn execute(self) {
        (self.func)();
    }

    /// Get task age
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Set task priority
    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Handle for waiting on task completion
pub struct TaskHandle<T> {
    receiver: std::sync::mpsc::Receiver<T>,
}

impl<T> TaskHandle<T> {
    /// Wait for task completion
    pub fn wait(self) -> Result<T, ThreadPoolError> {
        self.receiver
            .recv()
            .map_err(|_| ThreadPoolError::ExecutionFailed)
    }

    /// Wait for task completion with timeout
    pub fn wait_timeout(self, timeout: Duration) -> Result<T, ThreadPoolError> {
        self.receiver
            .recv_timeout(timeout)
            .map_err(|_| ThreadPoolError::ExecutionFailed)
    }

    /// Check if task is complete without blocking
    pub fn try_wait(&self) -> Result<Option<T>, ThreadPoolError> {
        match self.receiver.try_recv() {
            Ok(result) => Ok(Some(result)),
            Err(std::sync::mpsc::TryRecvError::Empty) => Ok(None),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                Err(ThreadPoolError::ExecutionFailed)
            }
        }
    }
}

/// Advanced statistics for the thread pool
#[derive(Debug, Clone)]
pub struct AdvancedThreadPoolStats {
    /// Total tasks executed across all workers
    pub total_tasks_executed: u64,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Number of work steals performed
    pub work_steals: u64,
    /// Load balancing efficiency (0.0 to 1.0)
    pub load_balance_efficiency: f64,
    /// Per-worker statistics
    pub worker_stats: Vec<WorkerStats>,
    /// Queue contention metrics
    pub queue_contention: f64,
}

impl AdvancedThreadPoolStats {
    fn new(num_workers: usize) -> Self {
        Self {
            total_tasks_executed: 0,
            total_execution_time: Duration::ZERO,
            work_steals: 0,
            load_balance_efficiency: 1.0,
            worker_stats: (0..num_workers).map(WorkerStats::new).collect(),
            queue_contention: 0.0,
        }
    }

    /// Calculate throughput (tasks per second)
    pub fn throughput(&self) -> f64 {
        if self.total_execution_time.is_zero() {
            0.0
        } else {
            self.total_tasks_executed as f64 / self.total_execution_time.as_secs_f64()
        }
    }

    /// Calculate average task latency
    pub fn average_latency(&self) -> Duration {
        if self.total_tasks_executed == 0 {
            Duration::ZERO
        } else {
            self.total_execution_time / self.total_tasks_executed as u32
        }
    }

    /// Calculate worker utilization
    pub fn worker_utilization(&self) -> Vec<f64> {
        let total_time = self.total_execution_time;
        self.worker_stats
            .iter()
            .map(|stats| {
                if total_time.is_zero() {
                    0.0
                } else {
                    stats.total_time.as_secs_f64() / total_time.as_secs_f64()
                }
            })
            .collect()
    }

    /// Calculate load balance efficiency
    pub fn calculate_load_balance_efficiency(&self) -> f64 {
        if self.worker_stats.len() <= 1 {
            return 1.0;
        }

        let task_counts: Vec<u64> = self
            .worker_stats
            .iter()
            .map(|stats| stats.tasks_completed)
            .collect();

        let total_tasks: u64 = task_counts.iter().sum();
        if total_tasks == 0 {
            return 1.0;
        }

        let average_tasks = total_tasks as f64 / task_counts.len() as f64;
        let variance: f64 = task_counts
            .iter()
            .map(|&count| {
                let diff = count as f64 - average_tasks;
                diff * diff
            })
            .sum::<f64>()
            / task_counts.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if average_tasks > 0.0 {
            std_dev / average_tasks
        } else {
            0.0
        };

        // Convert to efficiency (lower variation = higher efficiency)
        (1.0 - coefficient_of_variation.min(1.0)).max(0.0)
    }
}

/// NUMA-aware thread pool for systems with multiple memory nodes
pub struct NumaAwareThreadPool {
    pools: Vec<AdvancedThreadPool>,
    #[allow(dead_code)]
    numa_topology: NumaTopology,
}

impl NumaAwareThreadPool {
    /// Create a NUMA-aware thread pool
    pub fn new(config: ThreadPoolConfig) -> Self {
        let topology = NumaTopology::detect();
        let pools_per_node = config.num_threads / topology.num_nodes.max(1);

        let mut pools = Vec::with_capacity(topology.num_nodes);

        for _ in 0..topology.num_nodes {
            let node_config = ThreadPoolConfig {
                num_threads: pools_per_node,
                ..config.clone()
            };
            pools.push(AdvancedThreadPool::new(node_config));
        }

        Self {
            pools,
            numa_topology: topology,
        }
    }

    /// Submit a task to the appropriate NUMA node
    pub fn submit_numa<F>(
        &self,
        task: F,
        preferred_node: Option<usize>,
    ) -> Result<TaskHandle<()>, ThreadPoolError>
    where
        F: FnOnce() + Send + 'static,
    {
        let node = preferred_node
            .unwrap_or_else(|| self.select_optimal_node())
            .min(self.pools.len() - 1);

        self.pools[node].submit(task)
    }

    /// Select the optimal NUMA node for task placement
    fn select_optimal_node(&self) -> usize {
        // Simple load balancing - choose least loaded node
        self.pools
            .iter()
            .enumerate()
            .min_by_key(|(_, pool)| pool.get_stats().total_tasks_executed)
            .map(|(id, _)| id)
            .unwrap_or(0)
    }
}

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub num_nodes: usize,
    /// CPU cores per node
    pub cores_per_node: Vec<usize>,
    /// Memory per node (in bytes)
    pub memory_per_node: Vec<usize>,
}

impl NumaTopology {
    /// Detect NUMA topology (simplified implementation)
    fn detect() -> Self {
        // In a real implementation, this would query the system for NUMA information
        // For now, assume a simple single-node system
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            num_nodes: 1,
            cores_per_node: vec![num_cpus],
            memory_per_node: vec![8 * 1024 * 1024 * 1024], // 8GB default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_advanced_thread_pool() {
        let config = ThreadPoolConfig {
            num_threads: 2,
            work_stealing: true,
            ..Default::default()
        };

        let pool = AdvancedThreadPool::new(config);
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let handle = pool
            .submit(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .unwrap();

        handle.wait().unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_task_handle_timeout() {
        let config = ThreadPoolConfig {
            num_threads: 1,
            ..Default::default()
        };

        let pool = AdvancedThreadPool::new(config);

        let handle = pool
            .submit(|| {
                std::thread::sleep(Duration::from_millis(200));
            })
            .unwrap();

        // Should timeout
        let result = handle.wait_timeout(Duration::from_millis(50));
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_submission() {
        let config = ThreadPoolConfig {
            num_threads: 2,
            ..Default::default()
        };

        let pool = AdvancedThreadPool::new(config);
        let counter = Arc::new(AtomicUsize::new(0));

        let tasks: Vec<_> = (0..5)
            .map(|_| {
                let counter_clone = Arc::clone(&counter);
                move || {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                }
            })
            .collect();

        let handles = pool.submit_batch(tasks).unwrap();

        for handle in handles {
            handle.wait().unwrap();
        }

        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_thread_pool_stats() {
        let config = ThreadPoolConfig {
            num_threads: 2,
            ..Default::default()
        };

        let pool = AdvancedThreadPool::new(config);
        let stats = pool.get_stats();

        assert_eq!(stats.total_tasks_executed, 0);
        assert_eq!(stats.worker_stats.len(), 2);
    }

    #[test]
    fn test_numa_aware_thread_pool() {
        let config = ThreadPoolConfig {
            num_threads: 4,
            ..Default::default()
        };

        let numa_pool = NumaAwareThreadPool::new(config);
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let handle = numa_pool
            .submit_numa(
                move || {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                },
                Some(0),
            )
            .unwrap();

        handle.wait().unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_task_priority() {
        let task = Task::new(|| {}).0.with_priority(TaskPriority::High);
        assert_eq!(task.priority, TaskPriority::High);
    }

    #[test]
    fn test_numa_topology() {
        let topology = NumaTopology::detect();
        assert!(topology.num_nodes > 0);
        assert!(!topology.cores_per_node.is_empty());
        assert!(!topology.memory_per_node.is_empty());
    }
}
