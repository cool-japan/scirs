//! Work-stealing schedulers for adaptive algorithms
//!
//! This module provides work-stealing task schedulers optimized for adaptive
//! numerical algorithms. These schedulers dynamically balance workload across
//! threads, which is particularly important for algorithms with irregular
//! computational patterns.
//!
//! # Work-Stealing Concepts
//!
//! Work-stealing is a scheduling technique where idle threads "steal" work
//! from busy threads' task queues. This is especially effective for:
//! - Adaptive algorithms with unpredictable work distribution
//! - Recursive divide-and-conquer algorithms
//! - Dynamic load balancing scenarios
//!
//! # Examples
//!
//! ```
//! use scirs2_integrate::scheduling::{WorkStealingPool, Task};
//!
//! // Create work-stealing pool with 4 threads
//! let pool = WorkStealingPool::new(4);
//!
//! // Submit a simple task
//! let task = Task::new(|| 0.5 * 0.5); // Simple computation
//! pool.submit(task);
//! ```

use crate::common::IntegrateFloat;
use crate::error::IntegrateResult;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Generic task that can be executed by the work-stealing scheduler
pub trait WorkStealingTask: Send + 'static {
    type Output: Send;

    /// Execute the task
    fn execute(self) -> Self::Output;

    /// Estimate computational cost (for load balancing)
    fn estimated_cost(&self) -> f64 {
        1.0
    }

    /// Check if task can be subdivided for better load balancing
    fn can_subdivide(&self) -> bool {
        false
    }

    /// Subdivide task into smaller tasks (if possible)
    fn subdivide(self) -> Vec<Box<dyn WorkStealingTask<Output = Self::Output>>>
    where
        Self: Sized,
    {
        vec![]
    }
}

/// Simple boxed task for closures
pub struct Task<F, R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    func: Option<F>,
    cost_estimate: f64,
}

impl<F, R> Task<F, R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    /// Create a new task from closure
    pub fn new(func: F) -> Self {
        Self {
            func: Some(func),
            cost_estimate: 1.0,
        }
    }

    /// Create task with cost estimate
    pub fn with_cost(func: F, cost: f64) -> Self {
        Self {
            func: Some(func),
            cost_estimate: cost,
        }
    }
}

impl<F, R> WorkStealingTask for Task<F, R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    type Output = R;

    fn execute(mut self) -> Self::Output {
        (self.func.take().unwrap())()
    }

    fn estimated_cost(&self) -> f64 {
        self.cost_estimate
    }
}

/// Work-stealing deque for efficient task queue operations
#[derive(Debug)]
struct WorkStealingDeque<T> {
    items: VecDeque<T>,
    total_cost: f64,
}

impl<T: WorkStealingTask> WorkStealingDeque<T> {
    fn new() -> Self {
        Self {
            items: VecDeque::new(),
            total_cost: 0.0,
        }
    }

    fn push_back(&mut self, task: T) {
        self.total_cost += task.estimated_cost();
        self.items.push_back(task);
    }

    fn pop_back(&mut self) -> Option<T> {
        if let Some(task) = self.items.pop_back() {
            self.total_cost -= task.estimated_cost();
            Some(task)
        } else {
            None
        }
    }

    fn steal_front(&mut self) -> Option<T> {
        if let Some(task) = self.items.pop_front() {
            self.total_cost -= task.estimated_cost();
            Some(task)
        } else {
            None
        }
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.items.len()
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn total_cost(&self) -> f64 {
        self.total_cost
    }
}

/// Worker thread state
struct WorkerState<T: WorkStealingTask> {
    /// Local task queue
    local_queue: Mutex<WorkStealingDeque<T>>,
    /// Number of tasks completed by this worker
    completed_tasks: AtomicUsize,
    /// Total computation time for this worker
    computation_time: Mutex<Duration>,
}

impl<T: WorkStealingTask> WorkerState<T> {
    fn new() -> Self {
        Self {
            local_queue: Mutex::new(WorkStealingDeque::new()),
            completed_tasks: AtomicUsize::new(0),
            computation_time: Mutex::new(Duration::ZERO),
        }
    }
}

/// Work-stealing thread pool for adaptive algorithms
pub struct WorkStealingPool<T: WorkStealingTask> {
    /// Worker threads
    workers: Vec<JoinHandle<()>>,
    /// Worker states (shared between threads)
    worker_states: Arc<Vec<WorkerState<T>>>,
    /// Global task queue for initial distribution
    global_queue: Arc<Mutex<WorkStealingDeque<T>>>,
    /// Number of tasks currently being executed
    active_tasks: Arc<AtomicUsize>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// Condition variable for thread coordination
    cv: Arc<Condvar>,
    /// Mutex for condition variable
    #[allow(dead_code)]
    cv_mutex: Arc<Mutex<()>>,
    /// Pool statistics
    stats: Arc<Mutex<PoolStatistics>>,
}

/// Statistics about pool performance
#[derive(Debug, Clone, Default)]
pub struct PoolStatistics {
    /// Total tasks executed
    pub total_tasks: usize,
    /// Total computation time across all threads
    pub total_computation_time: Duration,
    /// Number of work-stealing operations
    pub steal_attempts: usize,
    /// Successful steals
    pub successful_steals: usize,
    /// Load balancing efficiency (0.0 to 1.0)
    pub load_balance_efficiency: f64,
}

impl<T: WorkStealingTask + 'static> WorkStealingPool<T> {
    /// Create new work-stealing pool with specified number of threads
    pub fn new(num_threads: usize) -> Self {
        let num_threads = num_threads.max(1);

        let worker_states = Arc::new(
            (0..num_threads)
                .map(|_| WorkerState::new())
                .collect::<Vec<_>>(),
        );

        let global_queue = Arc::new(Mutex::new(WorkStealingDeque::new()));
        let active_tasks = Arc::new(AtomicUsize::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));
        let cv = Arc::new(Condvar::new());
        let cv_mutex = Arc::new(Mutex::new(()));
        let stats = Arc::new(Mutex::new(PoolStatistics::default()));

        let workers = (0..num_threads)
            .map(|worker_id| {
                let worker_states = Arc::clone(&worker_states);
                let global_queue = Arc::clone(&global_queue);
                let active_tasks = Arc::clone(&active_tasks);
                let shutdown = Arc::clone(&shutdown);
                let cv = Arc::clone(&cv);
                let cv_mutex = Arc::clone(&cv_mutex);
                let stats = Arc::clone(&stats);

                thread::spawn(move || {
                    Self::worker_thread(
                        worker_id,
                        worker_states,
                        global_queue,
                        active_tasks,
                        shutdown,
                        cv,
                        cv_mutex,
                        stats,
                    );
                })
            })
            .collect();

        Self {
            workers,
            worker_states,
            global_queue,
            active_tasks,
            shutdown,
            cv,
            cv_mutex,
            stats,
        }
    }

    /// Submit a single task for execution
    pub fn submit(&self, task: T) {
        let mut global_queue = self.global_queue.lock().unwrap();
        global_queue.push_back(task);
        drop(global_queue);

        // Notify workers
        self.cv.notify_one();
    }

    /// Submit multiple tasks for execution
    pub fn submit_all(&self, tasks: Vec<T>) {
        let mut global_queue = self.global_queue.lock().unwrap();
        for task in tasks {
            global_queue.push_back(task);
        }
        drop(global_queue);

        // Notify all workers
        self.cv.notify_all();
    }

    /// Execute all submitted tasks and wait for completion
    pub fn execute_and_wait(&self) -> IntegrateResult<()> {
        // Wait for all tasks to complete
        loop {
            // Check if all queues are empty AND no tasks are currently being executed
            let global_empty = self.global_queue.lock().unwrap().is_empty();
            let locals_empty = self
                .worker_states
                .iter()
                .all(|state| state.local_queue.lock().unwrap().is_empty());
            let no_active_tasks = self.active_tasks.load(Ordering::Relaxed) == 0;

            if global_empty && locals_empty && no_active_tasks {
                break;
            }

            // Small delay to avoid busy waiting
            thread::sleep(Duration::from_micros(100));
        }

        Ok(())
    }

    /// Get current pool statistics
    pub fn statistics(&self) -> PoolStatistics {
        let mut stats = self.stats.lock().unwrap();

        // Update statistics from worker states
        stats.total_tasks = self
            .worker_states
            .iter()
            .map(|state| state.completed_tasks.load(Ordering::Relaxed))
            .sum();

        stats.total_computation_time = self
            .worker_states
            .iter()
            .map(|state| *state.computation_time.lock().unwrap())
            .sum();

        // Calculate load balance efficiency
        if stats.total_tasks > 0 {
            let worker_loads: Vec<f64> = self
                .worker_states
                .iter()
                .map(|state| {
                    let completed = state.completed_tasks.load(Ordering::Relaxed);
                    completed as f64 / stats.total_tasks as f64
                })
                .collect();

            let ideal_load = 1.0 / self.worker_states.len() as f64;
            let load_variance: f64 = worker_loads
                .iter()
                .map(|&load| (load - ideal_load).powi(2))
                .sum::<f64>()
                / self.worker_states.len() as f64;

            stats.load_balance_efficiency = (1.0 - load_variance).max(0.0);
        }

        stats.clone()
    }

    /// Worker thread main loop
    fn worker_thread(
        worker_id: usize,
        worker_states: Arc<Vec<WorkerState<T>>>,
        global_queue: Arc<Mutex<WorkStealingDeque<T>>>,
        active_tasks: Arc<AtomicUsize>,
        shutdown: Arc<AtomicBool>,
        cv: Arc<Condvar>,
        cv_mutex: Arc<Mutex<()>>,
        stats: Arc<Mutex<PoolStatistics>>,
    ) {
        let my_state = &worker_states[worker_id];

        while !shutdown.load(Ordering::Relaxed) {
            // Try to get work from local queue first
            let mut task_opt = my_state.local_queue.lock().unwrap().pop_back();

            // If no local work, try global queue
            if task_opt.is_none() {
                task_opt = global_queue.lock().unwrap().pop_back();
            }

            // If still no work, try stealing from other workers
            if task_opt.is_none() {
                task_opt = Self::try_steal_work(worker_id, &worker_states, &stats);
            }

            if let Some(task) = task_opt {
                // Mark task as active
                active_tasks.fetch_add(1, Ordering::Relaxed);

                // Execute the task
                let start_time = Instant::now();
                let _result = task.execute();
                let computation_time = start_time.elapsed();

                // Mark task as completed
                active_tasks.fetch_sub(1, Ordering::Relaxed);

                // Update statistics
                my_state.completed_tasks.fetch_add(1, Ordering::Relaxed);
                *my_state.computation_time.lock().unwrap() += computation_time;
            } else {
                // No work available, wait for notification
                let _guard = cv
                    .wait_timeout(cv_mutex.lock().unwrap(), Duration::from_millis(10))
                    .unwrap();
            }
        }
    }

    /// Try to steal work from other workers
    fn try_steal_work(
        worker_id: usize,
        worker_states: &[WorkerState<T>],
        stats: &Arc<Mutex<PoolStatistics>>,
    ) -> Option<T> {
        // Update steal attempt counter
        stats.lock().unwrap().steal_attempts += 1;

        // Find worker with most work (highest cost)
        let mut best_victim = None;
        let mut best_cost = 0.0;

        for (victim_id, victim_state) in worker_states.iter().enumerate() {
            if victim_id == worker_id {
                continue; // Don't steal from ourselves
            }

            let queue = victim_state.local_queue.lock().unwrap();
            let cost = queue.total_cost();

            if cost > best_cost && !queue.is_empty() {
                best_cost = cost;
                best_victim = Some(victim_id);
            }
        }

        // Try to steal from the best victim
        if let Some(victim_id) = best_victim {
            let victim_state = &worker_states[victim_id];
            let mut victim_queue = victim_state.local_queue.lock().unwrap();

            if let Some(stolen_task) = victim_queue.steal_front() {
                // Update successful steal counter
                stats.lock().unwrap().successful_steals += 1;
                return Some(stolen_task);
            }
        }

        None
    }
}

impl<T: WorkStealingTask> Drop for WorkStealingPool<T> {
    fn drop(&mut self) {
        // Signal shutdown
        self.shutdown.store(true, Ordering::Relaxed);
        self.cv.notify_all();

        // Wait for all workers to finish
        while let Some(worker) = self.workers.pop() {
            let _ = worker.join();
        }
    }
}

/// Adaptive integration task for work-stealing scheduler
pub struct AdaptiveIntegrationTask<F: IntegrateFloat, Func> {
    /// Function to integrate
    integrand: Func,
    /// Integration interval
    interval: (F, F),
    /// Tolerance for this region
    tolerance: F,
    /// Current depth (for subdivision control)
    depth: usize,
    /// Maximum subdivision depth
    max_depth: usize,
}

impl<F: IntegrateFloat, Func> AdaptiveIntegrationTask<F, Func>
where
    Func: Fn(F) -> F + Send + Clone + 'static,
{
    /// Create new adaptive integration task
    pub fn new(integrand: Func, interval: (F, F), tolerance: F, max_depth: usize) -> Self {
        Self {
            integrand,
            interval,
            tolerance,
            depth: 0,
            max_depth,
        }
    }

    /// Simple trapezoidal rule integration
    fn integrate_region(&self) -> F {
        let (a, b) = self.interval;
        let h = b - a;
        let fa = (self.integrand)(a);
        let fb = (self.integrand)(b);
        h * (fa + fb) / F::from(2.0).unwrap()
    }

    /// Estimate integration error using subdivision
    fn estimate_error(&self) -> F {
        let (a, b) = self.interval;
        let mid = (a + b) / F::from(2.0).unwrap();

        // Coarse estimate (full interval)
        let coarse = self.integrate_region();

        // Fine estimate (two half intervals)
        let left_task = AdaptiveIntegrationTask {
            integrand: self.integrand.clone(),
            interval: (a, mid),
            tolerance: self.tolerance,
            depth: self.depth + 1,
            max_depth: self.max_depth,
        };

        let right_task = AdaptiveIntegrationTask {
            integrand: self.integrand.clone(),
            interval: (mid, b),
            tolerance: self.tolerance,
            depth: self.depth + 1,
            max_depth: self.max_depth,
        };

        let fine = left_task.integrate_region() + right_task.integrate_region();

        (fine - coarse).abs()
    }
}

impl<F: IntegrateFloat + Send, Func> WorkStealingTask for AdaptiveIntegrationTask<F, Func>
where
    Func: Fn(F) -> F + Send + Clone + 'static,
{
    type Output = IntegrateResult<F>;

    fn execute(self) -> Self::Output {
        let result = self.integrate_region();
        Ok(result)
    }

    fn estimated_cost(&self) -> f64 {
        let (a, b) = self.interval;
        (b - a).to_f64().unwrap_or(1.0)
    }

    fn can_subdivide(&self) -> bool {
        self.depth < self.max_depth && self.estimate_error() > self.tolerance
    }

    fn subdivide(self) -> Vec<Box<dyn WorkStealingTask<Output = Self::Output>>> {
        let (a, b) = self.interval;
        let mid = (a + b) / F::from(2.0).unwrap();

        let left_task = AdaptiveIntegrationTask {
            integrand: self.integrand.clone(),
            interval: (a, mid),
            tolerance: self.tolerance / F::from(2.0).unwrap(),
            depth: self.depth + 1,
            max_depth: self.max_depth,
        };

        let right_task = AdaptiveIntegrationTask {
            integrand: self.integrand,
            interval: (mid, b),
            tolerance: self.tolerance / F::from(2.0).unwrap(),
            depth: self.depth + 1,
            max_depth: self.max_depth,
        };

        vec![
            Box::new(left_task) as Box<dyn WorkStealingTask<Output = Self::Output>>,
            Box::new(right_task) as Box<dyn WorkStealingTask<Output = Self::Output>>,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicI32;
    use std::time::Duration;

    #[test]
    fn test_work_stealing_pool_basic() {
        let pool: WorkStealingPool<Task<_, i32>> = WorkStealingPool::new(2);

        // Submit some simple tasks
        for i in 0..10 {
            let task = Task::new(move || i * 2);
            pool.submit(task);
        }

        // Wait for completion
        assert!(pool.execute_and_wait().is_ok());

        // Check statistics
        let stats = pool.statistics();
        assert_eq!(stats.total_tasks, 10);
        assert!(stats.load_balance_efficiency >= 0.0);
    }

    #[test]
    fn test_task_subdivision() {
        let integrand = |x: f64| x * x;
        let task = AdaptiveIntegrationTask::new(integrand, (0.0, 1.0), 1e-6, 5);

        assert!(task.can_subdivide());

        let subtasks = task.subdivide();
        assert_eq!(subtasks.len(), 2);
    }

    #[test]
    fn test_load_balancing() {
        let pool: WorkStealingPool<Task<_, ()>> = WorkStealingPool::new(4);
        let counter = Arc::new(AtomicI32::new(0));

        // Submit tasks with varying computational cost
        for i in 0..20 {
            let counter_clone = Arc::clone(&counter);
            let sleep_time = (i % 5) * 10; // Variable work

            let task = Task::with_cost(
                move || {
                    thread::sleep(Duration::from_millis(sleep_time));
                    counter_clone.fetch_add(1, Ordering::Relaxed);
                },
                sleep_time as f64,
            );

            pool.submit(task);
        }

        pool.execute_and_wait().unwrap();

        assert_eq!(counter.load(Ordering::Relaxed), 20);

        let stats = pool.statistics();
        assert_eq!(stats.total_tasks, 20);
        assert!(stats.steal_attempts > 0); // Should have attempted work stealing
    }
}
