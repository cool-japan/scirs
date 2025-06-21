//! Work-stealing scheduler for efficient thread utilization
//!
//! This module provides a work-stealing scheduler that distributes tasks
//! efficiently across available threads, optimizing CPU utilization.
//! It includes adaptivity to workload characteristics and dynamic task
//! prioritization.

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use crossbeam_utils::sync::{Parker, Unparker};
use num_cpus;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Type alias for task completion notification
type TaskCompletionNotify = Arc<(Mutex<bool>, Condvar)>;

/// Type alias for task completion map
type TaskCompletionMap = Arc<Mutex<HashMap<usize, TaskCompletionNotify>>>;

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum TaskPriority {
    /// Background tasks (lowest priority)
    Background = 0,
    /// Normal tasks (default priority)
    #[default]
    Normal = 1,
    /// High priority tasks
    High = 2,
    /// Critical tasks (highest priority)
    Critical = 3,
}

/// Task scheduling policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchedulingPolicy {
    /// First-in-first-out (queue)
    Fifo,
    /// Last-in-first-out (stack)
    Lifo,
    /// Priority-based scheduling
    #[default]
    Priority,
    /// Weighted fair queuing
    WeightedFair,
}

/// Work-stealing scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Number of worker threads
    pub num_workers: usize,
    /// Scheduling policy
    pub policy: SchedulingPolicy,
    /// Maximum task queue size
    pub max_queue_size: usize,
    /// Whether to use adaptive scheduling
    pub adaptive: bool,
    /// Whether to enable stealing heuristics
    pub enable_stealing_heuristics: bool,
    /// Whether to enable task priorities
    pub enable_priorities: bool,
    /// Stealing threshold (number of tasks before stealing is allowed)
    pub stealing_threshold: usize,
    /// Worker sleep time when idle (milliseconds)
    pub sleep_ms: u64,
    /// Minimum batch size for task scheduling
    pub min_batch_size: usize,
    /// Maximum batch size for task scheduling
    pub max_batch_size: usize,
    /// Task timeout (milliseconds, 0 for no timeout)
    pub task_timeout_ms: u64,
    /// Maximum number of retries for failed tasks
    pub max_retries: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
            policy: SchedulingPolicy::Priority,
            max_queue_size: 10000,
            adaptive: true,
            enable_stealing_heuristics: true,
            enable_priorities: true,
            stealing_threshold: 4,
            sleep_ms: 1,
            min_batch_size: 1,
            max_batch_size: 100,
            task_timeout_ms: 0,
            max_retries: 3,
        }
    }
}

/// Builder for scheduler configuration
#[derive(Debug, Clone, Default)]
pub struct SchedulerConfigBuilder {
    config: SchedulerConfig,
}

impl SchedulerConfigBuilder {
    /// Create a new scheduler configuration builder with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of worker threads
    pub const fn num_workers(mut self, num_workers: usize) -> Self {
        self.config.num_workers = num_workers;
        self
    }

    /// Set the scheduling policy
    pub const fn policy(mut self, policy: SchedulingPolicy) -> Self {
        self.config.policy = policy;
        self
    }

    /// Set the maximum queue size
    pub const fn max_queue_size(mut self, size: usize) -> Self {
        self.config.max_queue_size = size;
        self
    }

    /// Enable or disable adaptive scheduling
    pub const fn adaptive(mut self, enable: bool) -> Self {
        self.config.adaptive = enable;
        self
    }

    /// Enable or disable stealing heuristics
    pub const fn enable_stealing_heuristics(mut self, enable: bool) -> Self {
        self.config.enable_stealing_heuristics = enable;
        self
    }

    /// Enable or disable task priorities
    pub const fn enable_priorities(mut self, enable: bool) -> Self {
        self.config.enable_priorities = enable;
        self
    }

    /// Set the stealing threshold
    pub const fn stealing_threshold(mut self, threshold: usize) -> Self {
        self.config.stealing_threshold = threshold;
        self
    }

    /// Set the worker sleep time
    pub const fn sleep_ms(mut self, ms: u64) -> Self {
        self.config.sleep_ms = ms;
        self
    }

    /// Set the minimum batch size
    pub const fn min_batch_size(mut self, size: usize) -> Self {
        self.config.min_batch_size = size;
        self
    }

    /// Set the maximum batch size
    pub const fn max_batch_size(mut self, size: usize) -> Self {
        self.config.max_batch_size = size;
        self
    }

    /// Set the task timeout
    pub const fn task_timeout_ms(mut self, timeout: u64) -> Self {
        self.config.task_timeout_ms = timeout;
        self
    }

    /// Set the maximum number of retries
    pub const fn max_retries(mut self, retries: usize) -> Self {
        self.config.max_retries = retries;
        self
    }

    /// Build the configuration
    pub fn build(self) -> SchedulerConfig {
        self.config
    }
}

/// Task trait for work-stealing scheduler
pub trait Task: Send + 'static {
    /// Execute the task
    fn execute(&mut self) -> Result<(), CoreError>;

    /// Get the task priority
    fn priority(&self) -> TaskPriority {
        TaskPriority::Normal
    }

    /// Get the task weight (for weighted fair queuing)
    fn weight(&self) -> usize {
        1
    }

    /// Get the estimated cost of the task
    fn estimated_cost(&self) -> usize {
        1
    }

    /// Clone the task
    fn clone_task(&self) -> Box<dyn Task>;

    /// Get the task name
    fn name(&self) -> &str {
        "unnamed"
    }
}

/// Task handle for tracking task status
#[derive(Clone)]
pub struct TaskHandle {
    /// Task ID
    id: usize,
    /// Task status
    status: Arc<Mutex<TaskStatus>>,
    /// Result notification condvar
    result_notify: TaskCompletionNotify,
}

/// Task status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task is pending execution
    Pending,
    /// Task is currently running
    Running,
    /// Task has completed successfully
    Completed,
    /// Task has failed
    Failed(usize), // retry count
    /// Task has been cancelled
    Cancelled,
    /// Task has timed out
    TimedOut,
}

impl TaskHandle {
    #[allow(dead_code)]
    /// Create a new task handle
    fn new(id: usize) -> Self {
        Self {
            id,
            status: Arc::new(Mutex::new(TaskStatus::Pending)),
            result_notify: Arc::new((Mutex::new(false), Condvar::new())),
        }
    }

    /// Get the task ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get the task status
    pub fn status(&self) -> TaskStatus {
        *self.status.lock().unwrap()
    }

    /// Wait for the task to complete
    pub fn wait(&self) -> TaskStatus {
        let (lock, cvar) = &*self.result_notify;
        let completed = lock.lock().unwrap();

        // Wait if the task is not complete
        if !*completed {
            let _unused = cvar.wait(completed).unwrap();
        }

        self.status()
    }

    /// Wait for the task to complete with a timeout
    pub fn wait_timeout(&self, timeout: Duration) -> Result<TaskStatus, CoreError> {
        let (lock, cvar) = &*self.result_notify;
        let completed = lock.lock().unwrap();

        // Wait if the task is not complete
        if !*completed {
            let result = cvar.wait_timeout(completed, timeout).unwrap();

            if result.1.timed_out() {
                return Err(CoreError::TimeoutError(
                    ErrorContext::new(format!("Timeout waiting for task {}", self.id))
                        .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }
        }

        Ok(self.status())
    }

    /// Cancel the task
    pub fn cancel(&self) -> bool {
        let mut status = self.status.lock().unwrap();

        if *status == TaskStatus::Pending {
            *status = TaskStatus::Cancelled;

            // Notify waiters
            let (lock, cvar) = &*self.result_notify;
            let mut completed = lock.lock().unwrap();
            *completed = true;
            cvar.notify_all();

            true
        } else {
            false
        }
    }
}

/// Task wrapper for internal use
struct TaskWrapper {
    /// Task ID
    id: usize,
    /// The task to execute
    task: Box<dyn Task>,
    /// Task priority
    priority: TaskPriority,
    /// Task weight
    weight: usize,
    /// Estimated cost
    #[allow(dead_code)]
    cost: usize,
    /// Task status
    status: Arc<Mutex<TaskStatus>>,
    /// Result notification condvar
    result_notify: TaskCompletionNotify,
    /// Submission time
    #[allow(dead_code)]
    submission_time: Instant,
    /// Retry count
    retry_count: usize,
    /// Task name
    #[allow(dead_code)]
    name: String,
}

impl TaskWrapper {
    /// Create a new task wrapper
    fn new(id: usize, task: Box<dyn Task>) -> Self {
        let priority = task.priority();
        let weight = task.weight();
        let cost = task.estimated_cost();
        let name = task.name().to_string();

        Self {
            id,
            task,
            priority,
            weight,
            cost,
            status: Arc::new(Mutex::new(TaskStatus::Pending)),
            result_notify: Arc::new((Mutex::new(false), Condvar::new())),
            submission_time: Instant::now(),
            retry_count: 0,
            name,
        }
    }

    /// Create a task handle for this task
    fn create_handle(&self) -> TaskHandle {
        TaskHandle {
            id: self.id,
            status: self.status.clone(),
            result_notify: self.result_notify.clone(),
        }
    }

    /// Execute the task
    fn execute(&mut self) -> Result<(), CoreError> {
        // Update status
        {
            let mut status = self.status.lock().unwrap();
            *status = TaskStatus::Running;
        }

        // Execute the task
        let result = self.task.execute();

        // Update status
        {
            let mut status = self.status.lock().unwrap();
            *status = match result {
                Ok(_) => TaskStatus::Completed,
                Err(_) => TaskStatus::Failed(self.retry_count),
            };
        }

        // Notify waiters
        let (lock, cvar) = &*self.result_notify;
        let mut completed = lock.lock().unwrap();
        *completed = true;
        cvar.notify_all();

        result
    }

    /// Increment the retry count
    fn increment_retry(&mut self) {
        self.retry_count += 1;
    }
}

thread_local! {
    /// Thread-local storage for worker state
    static WORKER_ID: UnsafeCell<Option<usize>> = const { UnsafeCell::new(None) };
}

/// Set the worker ID for the current thread
fn set_worker_id(id: usize) {
    WORKER_ID.with(|cell| unsafe {
        *cell.get() = Some(id);
    });
}

/// Get the worker ID for the current thread
pub fn get_worker_id() -> Option<usize> {
    WORKER_ID.with(|cell| unsafe { *cell.get() })
}

/// Worker thread state
struct WorkerState {
    /// Worker ID
    #[allow(dead_code)]
    id: usize,
    /// Worker-local task queue - NOT shared between threads
    #[allow(clippy::type_complexity)]
    local_queue: UnsafeCell<Worker<TaskWrapper>>,
    /// Stealers for other workers
    stealers: Vec<Stealer<TaskWrapper>>,
    /// Shared task injector for global tasks
    injector: Arc<Injector<TaskWrapper>>,
    /// Whether the worker is active
    #[allow(dead_code)]
    active: Arc<AtomicBool>,
    /// Parker for sleeping when idle - NOT shared between threads
    parker: UnsafeCell<Parker>,
    /// Unparker for waking up the worker
    unparker: Unparker,
    /// Number of tasks processed
    tasks_processed: AtomicUsize,
    /// Number of tasks stolen
    tasks_stolen: AtomicUsize,
    /// Number of failed steal attempts
    failed_steals: AtomicUsize,
    /// Last active time
    last_active: Mutex<Instant>,
    /// Number of tasks in the local queue
    local_queue_size: AtomicUsize,
    /// Adaptive batch size
    adaptive_batch_size: AtomicUsize,
}

// Explicitly implement Send and Sync for WorkerState to make the compiler happy
// This is safe because we're careful about accessing the UnsafeCell contents
unsafe impl Send for WorkerState {}
unsafe impl Sync for WorkerState {}

impl WorkerState {
    /// Create a new worker state
    fn new(
        id: usize,
        stealers: Vec<Stealer<TaskWrapper>>,
        injector: Arc<Injector<TaskWrapper>>,
    ) -> Self {
        let parker = Parker::new();
        let unparker = parker.unparker().clone();

        Self {
            id,
            local_queue: UnsafeCell::new(Worker::new_fifo()),
            stealers,
            injector,
            active: Arc::new(AtomicBool::new(true)),
            parker: UnsafeCell::new(parker),
            unparker,
            tasks_processed: AtomicUsize::new(0),
            tasks_stolen: AtomicUsize::new(0),
            failed_steals: AtomicUsize::new(0),
            last_active: Mutex::new(Instant::now()),
            local_queue_size: AtomicUsize::new(0),
            adaptive_batch_size: AtomicUsize::new(1),
        }
    }

    /// Get the worker ID
    #[allow(dead_code)]
    fn id(&self) -> usize {
        self.id
    }

    /// Get the local queue size
    fn local_queue_size(&self) -> usize {
        self.local_queue_size.load(Ordering::Relaxed)
    }

    /// Push a task to the local queue
    fn push_local(&self, task: TaskWrapper) {
        // Safety: Each worker thread only accesses its own local_queue
        unsafe {
            (*self.local_queue.get()).push(task);
        }
        self.local_queue_size.fetch_add(1, Ordering::Relaxed);
    }

    /// Pop a task from the local queue
    fn pop_local(&self) -> Option<TaskWrapper> {
        // Safety: Each worker thread only accesses its own local_queue
        let result = unsafe { (*self.local_queue.get()).pop() };

        if result.is_some() {
            self.local_queue_size.fetch_sub(1, Ordering::Relaxed);
        }

        result
    }

    /// Steal a task from another worker
    fn steal(&self) -> Option<TaskWrapper> {
        // First, try to steal from the global injector
        match self.injector.steal() {
            Steal::Success(task) => {
                self.tasks_stolen.fetch_add(1, Ordering::Relaxed);
                return Some(task);
            }
            Steal::Empty => {}
            Steal::Retry => {}
        }

        // Then, try to steal from other workers
        for stealer in &self.stealers {
            match stealer.steal() {
                Steal::Success(task) => {
                    self.tasks_stolen.fetch_add(1, Ordering::Relaxed);
                    return Some(task);
                }
                Steal::Empty => {}
                Steal::Retry => {}
            }
        }

        // If we got here, no tasks were stolen
        self.failed_steals.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Update the last active time
    fn update_last_active(&self) {
        let mut last_active = self.last_active.lock().unwrap();
        *last_active = Instant::now();
    }

    /// Get the time since the worker was last active
    fn time_since_last_active(&self) -> Duration {
        let last_active = self.last_active.lock().unwrap();
        last_active.elapsed()
    }

    /// Update the adaptive batch size based on recent performance
    fn update_adaptive_batch_size(&self, config: &SchedulerConfig) {
        if !config.adaptive {
            // Adaptive scheduling disabled, use default batch size
            self.adaptive_batch_size
                .store(config.min_batch_size, Ordering::Relaxed);
            return;
        }

        // Get the current statistics
        let _tasks_processed = self.tasks_processed.load(Ordering::Relaxed);
        let tasks_stolen = self.tasks_stolen.load(Ordering::Relaxed);
        let failed_steals = self.failed_steals.load(Ordering::Relaxed);

        // Calculate the steal success rate
        let steal_attempts = tasks_stolen + failed_steals;
        let steal_success_rate = if steal_attempts > 0 {
            tasks_stolen as f64 / steal_attempts as f64
        } else {
            0.0
        };

        // Calculate the ideal batch size based on stealing behavior
        let current_batch_size = self.adaptive_batch_size.load(Ordering::Relaxed);
        let new_batch_size = if steal_success_rate > 0.8 {
            // High stealing success rate, increase batch size to reduce stealing overhead
            (current_batch_size * 2).min(config.max_batch_size)
        } else if steal_success_rate < 0.2 {
            // Low stealing success rate, decrease batch size to increase stealing opportunities
            (current_batch_size / 2).max(config.min_batch_size)
        } else {
            // Reasonable success rate, keep current batch size
            current_batch_size
        };

        // Update the batch size
        self.adaptive_batch_size
            .store(new_batch_size, Ordering::Relaxed);
    }
}

/// Work-stealing scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    /// Number of tasks submitted
    pub tasks_submitted: usize,
    /// Number of tasks completed
    pub tasks_completed: usize,
    /// Number of tasks failed
    pub tasks_failed: usize,
    /// Number of tasks cancelled
    pub tasks_cancelled: usize,
    /// Number of tasks timed out
    pub tasks_timed_out: usize,
    /// Number of task retries
    pub task_retries: usize,
    /// Number of workers
    pub num_workers: usize,
    /// Average queue size
    pub avg_queue_size: f64,
    /// Average task latency (milliseconds)
    pub avg_task_latency_ms: f64,
    /// Average task execution time (milliseconds)
    pub avg_task_execution_ms: f64,
    /// Number of successful steals
    pub successful_steals: usize,
    /// Number of failed steal attempts
    pub failed_steals: usize,
    /// Worker utilization rate
    pub worker_utilization: Vec<f64>,
    /// Scheduler uptime in seconds
    pub uptime_seconds: f64,
    /// Tasks per second
    pub tasks_per_second: f64,
}

impl Default for SchedulerStats {
    fn default() -> Self {
        Self {
            tasks_submitted: 0,
            tasks_completed: 0,
            tasks_failed: 0,
            tasks_cancelled: 0,
            tasks_timed_out: 0,
            task_retries: 0,
            num_workers: 0,
            avg_queue_size: 0.0,
            avg_task_latency_ms: 0.0,
            avg_task_execution_ms: 0.0,
            successful_steals: 0,
            failed_steals: 0,
            worker_utilization: Vec::new(),
            uptime_seconds: 0.0,
            tasks_per_second: 0.0,
        }
    }
}

/// Work-stealing scheduler
pub struct WorkStealingScheduler {
    /// Scheduler configuration
    config: SchedulerConfig,
    /// Shared task injector for global tasks
    injector: Arc<Injector<TaskWrapper>>,
    /// Worker threads
    workers: Vec<JoinHandle<()>>,
    /// Worker state
    worker_states: Vec<Arc<WorkerState>>,
    /// Scheduler state
    state: Arc<RwLock<SchedulerState>>,
    /// Next task ID
    next_task_id: Arc<AtomicUsize>,
    /// Task completion map
    task_completion: TaskCompletionMap,
    /// Task submission times
    task_submissions: Arc<Mutex<HashMap<usize, Instant>>>,
    /// Task execution times
    task_executions: Arc<Mutex<HashMap<usize, Duration>>>,
    /// Scheduler start time
    start_time: Instant,
}

/// Scheduler state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchedulerState {
    /// Scheduler is running
    Running,
    /// Scheduler is shutting down
    ShuttingDown,
    /// Scheduler has shut down
    ShutDown,
}

impl WorkStealingScheduler {
    /// Create a new work-stealing scheduler
    pub fn new(config: SchedulerConfig) -> Self {
        // Create the scheduler
        let injector = Arc::new(Injector::new());
        let state = Arc::new(RwLock::new(SchedulerState::Running));
        let next_task_id = Arc::new(AtomicUsize::new(1));
        let task_completion = Arc::new(Mutex::new(HashMap::new()));
        let task_submissions = Arc::new(Mutex::new(HashMap::new()));
        let task_executions = Arc::new(Mutex::new(HashMap::new()));

        // Create workers and stealers
        let mut worker_states = Vec::with_capacity(config.num_workers);
        let mut workers = Vec::with_capacity(config.num_workers);

        // Create worker queues and stealers
        let worker_queues: Vec<_> = (0..config.num_workers)
            .map(|_| Worker::new_fifo())
            .collect();

        let stealers: Vec<_> = worker_queues
            .iter()
            .map(|worker| worker.stealer())
            .collect();

        // Create worker threads
        for i in 0..config.num_workers {
            // Create worker state
            let worker_state = Arc::new(WorkerState::new(i, stealers.clone(), injector.clone()));

            worker_states.push(worker_state.clone());

            // Create worker thread
            let state_clone = state.clone();
            let config_clone = config.clone();
            let task_completion_clone = task_completion.clone();
            let task_executions_clone = task_executions.clone();

            let worker_thread = thread::spawn(move || {
                // Set the worker ID in thread-local storage
                set_worker_id(i);

                // Run the worker loop
                Self::worker_loop(
                    worker_state,
                    state_clone,
                    config_clone,
                    task_completion_clone, // renamed to _task_completion in worker_loop
                    task_executions_clone,
                );
            });

            workers.push(worker_thread);
        }

        Self {
            config,
            injector,
            workers,
            worker_states,
            state,
            next_task_id,
            task_completion,
            task_submissions,
            task_executions,
            start_time: Instant::now(),
        }
    }

    /// Worker loop for processing tasks
    fn worker_loop(
        worker_state: Arc<WorkerState>,
        state: Arc<RwLock<SchedulerState>>,
        config: SchedulerConfig,
        _task_completion: TaskCompletionMap,
        task_executions: Arc<Mutex<HashMap<usize, Duration>>>,
    ) {
        // Main worker loop
        while let SchedulerState::Running = *state.read().unwrap() {
            // Try to get a task
            let task = worker_state.pop_local().or_else(|| worker_state.steal());

            if let Some(mut task) = task {
                // Update the last active time
                worker_state.update_last_active();

                // Execute the task
                let start_time = Instant::now();
                let result = task.execute();
                let execution_time = start_time.elapsed();

                // Record task execution time
                let task_id = task.id;
                task_executions
                    .lock()
                    .unwrap()
                    .insert(task_id, execution_time);

                // Update worker statistics
                worker_state.tasks_processed.fetch_add(1, Ordering::Relaxed);

                // Handle task result
                match result {
                    Ok(_) => {
                        // Task completed successfully, nothing more to do
                    }
                    Err(_) => {
                        // Task failed, check if we should retry
                        if task.retry_count < config.max_retries {
                            // Retry the task
                            task.increment_retry();

                            // Reset task status
                            {
                                let mut status = task.status.lock().unwrap();
                                *status = TaskStatus::Pending;
                            }

                            // Push task back to local queue
                            worker_state.push_local(task);
                        }
                    }
                }

                // Update the adaptive batch size
                worker_state.update_adaptive_batch_size(&config);
            } else {
                // No tasks available, sleep for a bit
                if config.sleep_ms > 0 {
                    // Safety: Each worker thread only accesses its own parker
                    unsafe {
                        (*worker_state.parker.get())
                            .park_timeout(Duration::from_millis(config.sleep_ms));
                    }
                } else {
                    // Just yield instead of sleeping
                    thread::yield_now();
                }
            }
        }

        // The scheduler is shutting down, process remaining local tasks
        while let Some(mut task) = worker_state.pop_local() {
            // Execute the task
            let start_time = Instant::now();
            let _ = task.execute();
            let execution_time = start_time.elapsed();

            // Record task execution time
            let task_id = task.id;
            task_executions
                .lock()
                .unwrap()
                .insert(task_id, execution_time);

            // Update worker statistics
            worker_state.tasks_processed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Submit a task to the scheduler
    pub fn submit<T: Task>(&self, task: T) -> TaskHandle {
        self.submit_boxed(Box::new(task))
    }

    /// Submit a boxed task to the scheduler
    pub fn submit_boxed(&self, task: Box<dyn Task>) -> TaskHandle {
        // Check if the scheduler is running
        if *self.state.read().unwrap() != SchedulerState::Running {
            panic!("Cannot submit tasks to a stopped scheduler");
        }

        // Get a new task ID
        let task_id = self.next_task_id.fetch_add(1, Ordering::SeqCst);

        // Wrap the task
        let wrapper = TaskWrapper::new(task_id, task);

        // Create task handle
        let handle = wrapper.create_handle();

        // Store task completion info
        self.task_completion
            .lock()
            .unwrap()
            .insert(task_id, wrapper.result_notify.clone());

        // Store submission time
        self.task_submissions
            .lock()
            .unwrap()
            .insert(task_id, Instant::now());

        // Submit the task based on the scheduling policy
        match self.config.policy {
            SchedulingPolicy::Fifo | SchedulingPolicy::Lifo => {
                // Submit to global queue
                self.injector.push(wrapper);
            }
            SchedulingPolicy::Priority => {
                // High priority tasks go to worker queues directly
                if wrapper.priority >= TaskPriority::High {
                    // Round-robin assignment to worker queues
                    let queue_idx = task_id % self.worker_states.len();
                    self.worker_states[queue_idx].push_local(wrapper);
                    self.worker_states[queue_idx].unparker.unpark();
                } else {
                    // Normal and background tasks go to global queue
                    self.injector.push(wrapper);
                }
            }
            SchedulingPolicy::WeightedFair => {
                // Higher weight tasks go to worker queues directly
                if wrapper.weight > 1 {
                    // Assign to worker with the least work
                    let min_queue_idx = self
                        .worker_states
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, state)| state.local_queue_size())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    self.worker_states[min_queue_idx].push_local(wrapper);
                    self.worker_states[min_queue_idx].unparker.unpark();
                } else {
                    // Normal weight tasks go to global queue
                    self.injector.push(wrapper);
                }
            }
        }

        // Wake up any sleeping workers
        self.wake_idle_workers();

        handle
    }

    /// Submit a batch of tasks to the scheduler
    pub fn submit_batch<T: Task + Clone>(&self, tasks: &[T]) -> Vec<TaskHandle> {
        let mut handles = Vec::with_capacity(tasks.len());

        for task in tasks {
            handles.push(self.submit(task.clone()));
        }

        handles
    }

    /// Submit a function to be executed as a task
    pub fn submit_fn<F, R>(&self, f: F) -> TaskHandle
    where
        F: FnOnce() -> Result<R, CoreError> + Send + 'static,
        R: Send + 'static,
    {
        // Create a simple task from the function
        struct FnTask<F, R> {
            f: Option<F>,
            _phantom: std::marker::PhantomData<R>,
        }

        impl<F, R> Task for FnTask<F, R>
        where
            F: FnOnce() -> Result<R, CoreError> + Send + 'static,
            R: Send + 'static,
        {
            fn execute(&mut self) -> Result<(), CoreError> {
                if let Some(f) = self.f.take() {
                    f()?;
                    Ok(())
                } else {
                    Err(CoreError::SchedulerError(
                        ErrorContext::new("Task function was already called".to_string())
                            .with_location(ErrorLocation::new(file!(), line!())),
                    ))
                }
            }

            fn clone_task(&self) -> Box<dyn Task> {
                panic!("FnTask cannot be cloned")
            }
        }

        self.submit_boxed(Box::new(FnTask {
            f: Some(f),
            _phantom: std::marker::PhantomData,
        }))
    }

    /// Wake up any idle workers
    fn wake_idle_workers(&self) {
        for worker in &self.worker_states {
            if worker.time_since_last_active() > Duration::from_millis(self.config.sleep_ms) {
                worker.unparker.unpark();
            }
        }
    }

    /// Wait for all tasks to complete
    pub fn wait_all(&self) {
        // Collect all task IDs
        let task_ids: Vec<_> = self
            .task_completion
            .lock()
            .unwrap()
            .keys()
            .copied()
            .collect();

        // Wait for each task
        for id in task_ids {
            if let Some(notify) = self.task_completion.lock().unwrap().get(&id) {
                let (lock, cvar) = &**notify;
                let completed = lock.lock().unwrap();

                if !*completed {
                    let _unused = cvar.wait(completed);
                }
            }
        }
    }

    /// Wait for all tasks to complete with a timeout
    pub fn wait_all_timeout(&self, timeout: Duration) -> Result<(), CoreError> {
        let deadline = Instant::now() + timeout;

        // Collect all task IDs
        let task_ids: Vec<_> = self
            .task_completion
            .lock()
            .unwrap()
            .keys()
            .copied()
            .collect();

        // Wait for each task
        for id in task_ids {
            let remaining = deadline.saturating_duration_since(Instant::now());

            if remaining.as_secs() == 0 && remaining.subsec_nanos() == 0 {
                return Err(CoreError::TimeoutError(
                    ErrorContext::new("Timeout waiting for tasks".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }

            if let Some(notify) = self.task_completion.lock().unwrap().get(&id) {
                let (lock, cvar) = &**notify;
                let completed = lock.lock().unwrap();

                if !*completed {
                    let result = cvar.wait_timeout(completed, remaining).unwrap();

                    if result.1.timed_out() && !*result.0 {
                        return Err(CoreError::TimeoutError(
                            ErrorContext::new("Timeout waiting for tasks".to_string())
                                .with_location(ErrorLocation::new(file!(), line!())),
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Get statistics for the scheduler
    pub fn stats(&self) -> SchedulerStats {
        let mut stats = SchedulerStats {
            tasks_submitted: self.next_task_id.load(Ordering::Relaxed) - 1,
            num_workers: self.config.num_workers,
            ..SchedulerStats::default()
        };

        // Task latency and execution time
        let mut total_latency = Duration::from_secs(0);
        let mut total_execution = Duration::from_secs(0);
        let mut completed_tasks = 0;

        let submissions = self.task_submissions.lock().unwrap();
        let executions = self.task_executions.lock().unwrap();

        for (id, submission_time) in submissions.iter() {
            if let Some(execution_time) = executions.get(id) {
                // Calculate latency as the time from submission to execution
                let latency = submission_time.elapsed() - *execution_time;

                total_latency += latency;
                total_execution += *execution_time;
                completed_tasks += 1;
            }
        }

        // Calculate averages
        stats.tasks_completed = completed_tasks;

        if completed_tasks > 0 {
            stats.avg_task_latency_ms = total_latency.as_millis() as f64 / completed_tasks as f64;
            stats.avg_task_execution_ms =
                total_execution.as_millis() as f64 / completed_tasks as f64;
        }

        // Worker statistics
        let mut total_queue_size = 0;
        let mut total_successful_steals = 0;
        let mut total_failed_steals = 0;
        let mut worker_utils = Vec::with_capacity(self.worker_states.len());

        for worker in &self.worker_states {
            total_queue_size += worker.local_queue_size();
            total_successful_steals += worker.tasks_stolen.load(Ordering::Relaxed);
            total_failed_steals += worker.failed_steals.load(Ordering::Relaxed);

            // Calculate worker utilization based on task processing
            let tasks_processed = worker.tasks_processed.load(Ordering::Relaxed);
            let utilization = if stats.tasks_submitted > 0 {
                tasks_processed as f64 / stats.tasks_submitted as f64
            } else {
                0.0
            };

            worker_utils.push(utilization);
        }

        stats.avg_queue_size = total_queue_size as f64 / self.worker_states.len() as f64;
        stats.successful_steals = total_successful_steals;
        stats.failed_steals = total_failed_steals;
        stats.worker_utilization = worker_utils;

        // Overall statistics
        stats.uptime_seconds = self.start_time.elapsed().as_secs_f64();

        if stats.uptime_seconds > 0.0 {
            stats.tasks_per_second = stats.tasks_completed as f64 / stats.uptime_seconds;
        }

        stats
    }

    /// Shutdown the scheduler
    pub fn shutdown(&mut self) {
        // Update state
        {
            let mut state = self.state.write().unwrap();
            *state = SchedulerState::ShuttingDown;
        }

        // Wake up all workers
        for worker in &self.worker_states {
            worker.unparker.unpark();
        }

        // Wait for workers to finish
        while let Some(worker) = self.workers.pop() {
            let _ = worker.join();
        }

        // Update state
        {
            let mut state = self.state.write().unwrap();
            *state = SchedulerState::ShutDown;
        }
    }

    /// Get the number of workers
    pub fn num_workers(&self) -> usize {
        self.worker_states.len()
    }

    /// Get the current worker ID
    pub fn current_worker_id(&self) -> Option<usize> {
        get_worker_id()
    }

    /// Get the number of pending tasks
    pub fn pending_tasks(&self) -> usize {
        // Count tasks in worker queues
        let mut total = 0;

        for worker in &self.worker_states {
            total += worker.local_queue_size();
        }

        total
    }
}

impl Drop for WorkStealingScheduler {
    fn drop(&mut self) {
        if *self.state.read().unwrap() != SchedulerState::ShutDown {
            self.shutdown();
        }
    }
}

/// CloneableTask for simple function-based tasks
pub struct CloneableTask<F, R>
where
    F: Fn() -> Result<R, CoreError> + Send + Sync + Clone + 'static,
    R: Send + 'static,
{
    /// The function to execute
    func: F,
    /// Task name
    name: String,
    /// Task priority
    priority: TaskPriority,
    /// Task weight
    weight: usize,
}

impl<F, R> CloneableTask<F, R>
where
    F: Fn() -> Result<R, CoreError> + Send + Sync + Clone + 'static,
    R: Send + 'static,
{
    /// Create a new cloneable task
    pub fn new(func: F) -> Self {
        Self {
            func,
            name: "unnamed".to_string(),
            priority: TaskPriority::Normal,
            weight: 1,
        }
    }

    /// Set the task name
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Set the task priority
    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set the task weight
    pub fn with_weight(mut self, weight: usize) -> Self {
        self.weight = weight;
        self
    }
}

impl<F, R> Task for CloneableTask<F, R>
where
    F: Fn() -> Result<R, CoreError> + Send + Sync + Clone + 'static,
    R: Send + 'static,
{
    fn execute(&mut self) -> Result<(), CoreError> {
        (self.func)().map(|_| ())
    }

    fn priority(&self) -> TaskPriority {
        self.priority
    }

    fn weight(&self) -> usize {
        self.weight
    }

    fn clone_task(&self) -> Box<dyn Task> {
        Box::new(Self {
            func: self.func.clone(),
            name: self.name.clone(),
            priority: self.priority,
            weight: self.weight,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Create a new work-stealing scheduler with default configuration
pub fn create_work_stealing_scheduler() -> WorkStealingScheduler {
    WorkStealingScheduler::new(SchedulerConfig::default())
}

/// Create a new work-stealing scheduler with the specified number of workers
pub fn create_work_stealing_scheduler_with_workers(workers: usize) -> WorkStealingScheduler {
    let config = SchedulerConfigBuilder::new().num_workers(workers).build();

    WorkStealingScheduler::new(config)
}

/// ParallelTask for executing tasks in parallel with work stealing
pub struct ParallelTask<T, F, R>
where
    T: Clone + Send + Sync + 'static,
    F: Fn(&T) -> Result<R, CoreError> + Send + Sync + Clone + 'static,
    R: Send + 'static,
{
    /// The items to process
    items: Vec<T>,
    /// The function to apply to each item
    func: F,
    /// Task name
    name: String,
    /// Task priority
    priority: TaskPriority,
    /// Whether to continue on errors
    continue_on_error: bool,
}

impl<T, F, R> ParallelTask<T, F, R>
where
    T: Clone + Send + Sync + 'static,
    F: Fn(&T) -> Result<R, CoreError> + Send + Sync + Clone + 'static,
    R: Send + 'static,
{
    /// Create a new parallel task
    pub fn new(items: Vec<T>, func: F) -> Self {
        Self {
            items,
            func,
            name: "parallel".to_string(),
            priority: TaskPriority::Normal,
            continue_on_error: false,
        }
    }

    /// Set the task name
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Set the task priority
    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set whether to continue on errors
    pub const fn continue_on_error(mut self, continue_on_error: bool) -> Self {
        self.continue_on_error = continue_on_error;
        self
    }

    /// Execute the tasks using a work-stealing scheduler
    pub fn execute(self) -> Result<Vec<R>, CoreError>
    where
        R: Clone,
    {
        // Create a scheduler
        let scheduler = create_work_stealing_scheduler();

        // Get the item count before we move them
        let items_len = self.items.len();

        // Submit tasks for each item
        let mut handles = Vec::with_capacity(items_len);
        let results = Arc::new(Mutex::new(Vec::with_capacity(items_len)));

        for (i, item) in self.items.into_iter().enumerate() {
            let func = self.func.clone();
            let results_clone = results.clone();
            let task_name = format!("{}-{}", self.name, i);
            let priority = self.priority;

            // Create a closure with moved ownership of item
            let task = CloneableTask::new(move || {
                let result = func(&item)?;
                results_clone.lock().unwrap().push((i, result));
                Ok(())
            })
            .with_name(&task_name)
            .with_priority(priority);

            handles.push(scheduler.submit(task));
        }

        // Wait for all tasks to complete
        for handle in &handles {
            match handle.wait() {
                TaskStatus::Completed => {}
                TaskStatus::Failed(_) if self.continue_on_error => {}
                status => {
                    return Err(CoreError::SchedulerError(
                        ErrorContext::new(format!(
                            "Task {} failed with status {:?}",
                            handle.id(),
                            status
                        ))
                        .with_location(ErrorLocation::new(file!(), line!())),
                    ));
                }
            }
        }

        // Get results in the original order
        let mut result_map = Vec::with_capacity(items_len);
        {
            let results_guard = results.lock().unwrap();
            for (i, result) in results_guard.iter() {
                result_map.push((*i, result.clone()));
            }
        }

        result_map.sort_by_key(|(i, _)| *i);

        let results = result_map.into_iter().map(|(_, r)| r).collect();

        Ok(results)
    }
}

/// Extension to the parallel module for using work-stealing scheduler
pub mod parallel {
    use super::*;
    use crate::error::CoreResult;

    /// Map a function over a collection in parallel using the work-stealing scheduler
    pub fn par_map<T, U, F>(items: &[T], f: F) -> CoreResult<Vec<U>>
    where
        T: Clone + Send + Sync + 'static,
        U: Clone + Send + 'static,
        F: Fn(&T) -> Result<U, CoreError> + Send + Sync + Clone + 'static,
    {
        // Convert slice to owned data since we need 'static lifetime
        let owned_items = items.to_vec();
        let task = ParallelTask::new(owned_items, f);
        task.execute()
    }

    /// Filter a collection in parallel using the work-stealing scheduler
    #[allow(dead_code)]
    pub fn par_filter<T, F>(items: &[T], predicate: F) -> CoreResult<Vec<T>>
    where
        T: Clone + Send + Sync + 'static,
        F: Fn(&T) -> Result<bool, CoreError> + Send + Sync + Clone + 'static,
    {
        let task = ParallelTask::new(items.to_vec(), move |item| {
            let include = predicate(item)?;
            if include {
                Ok(Some(item.clone()))
            } else {
                Ok(None)
            }
        });

        let results = task.execute()?;

        // Filter out None values
        let filtered: Vec<_> = results.into_iter().flatten().collect();

        Ok(filtered)
    }

    /// Apply a function to each element of a collection in parallel with no return value
    #[allow(dead_code)]
    pub fn par_for_each<T, F>(items: &[T], f: F) -> CoreResult<()>
    where
        T: Clone + Send + Sync + 'static,
        F: Fn(&T) -> Result<(), CoreError> + Send + Sync + Clone + 'static,
    {
        let task = ParallelTask::new(items.to_vec(), f);
        task.execute()?;
        Ok(())
    }

    /// Reduce a collection in parallel using the work-stealing scheduler
    #[allow(dead_code)]
    pub fn par_reduce<T, F>(items: &[T], init: T, f: F) -> CoreResult<T>
    where
        T: Clone + Send + Sync + 'static,
        F: Fn(T, &T) -> Result<T, CoreError> + Send + Sync + Clone + 'static,
    {
        if items.is_empty() {
            return Ok(init);
        }

        // Convert slices to owned data to avoid lifetime issues
        let items_owned: Vec<T> = items.to_vec();

        // Split the collection into chunks
        let num_chunks = std::cmp::min(items_owned.len(), num_cpus::get() * 4);
        let chunk_size = std::cmp::max(1, items_owned.len() / num_chunks);

        let mut chunks = Vec::with_capacity(num_chunks);
        for chunk_start in (0..items_owned.len()).step_by(chunk_size) {
            let chunk_end = std::cmp::min(chunk_start + chunk_size, items_owned.len());
            chunks.push(items_owned[chunk_start..chunk_end].to_vec());
        }

        // Reduce each chunk
        let f_clone = f.clone();
        let init_clone = init.clone();
        let chunk_results = par_map(&chunks, move |chunk| {
            let mut result = init_clone.clone();
            for item in chunk {
                result = f_clone(result, item)?;
            }
            Ok(result)
        })?;

        // Reduce the chunk results
        let mut final_result = init;
        for result in chunk_results {
            final_result = f(final_result, &result)?;
        }

        Ok(final_result)
    }
}

// Extension trait for arrays to use work-stealing scheduler
pub trait WorkStealingArray<A, S, D>
where
    A: Clone + Send + Sync + 'static,
    S: ndarray::RawData<Elem = A>,
    D: ndarray::Dimension,
{
    /// Map a function over the array elements in parallel
    fn work_stealing_map<F, B>(&self, f: F) -> CoreResult<ndarray::Array<B, D>>
    where
        B: Clone + Send + 'static,
        F: Fn(&A) -> Result<B, CoreError> + Send + Sync + Clone + 'static;
}

impl<A, S, D> WorkStealingArray<A, S, D> for ndarray::ArrayBase<S, D>
where
    A: Clone + Send + Sync + 'static,
    S: ndarray::RawData<Elem = A> + ndarray::Data,
    D: ndarray::Dimension + Clone + Send + 'static,
{
    fn work_stealing_map<F, B>(&self, f: F) -> CoreResult<ndarray::Array<B, D>>
    where
        B: Clone + Send + 'static,
        F: Fn(&A) -> Result<B, CoreError> + Send + Sync + Clone + 'static,
    {
        // Convert the array to a vector
        let shape = self.raw_dim();
        let flat_view = self
            .view()
            .into_shape_with_order(ndarray::IxDyn(&[self.len()]))
            .unwrap();
        let flat = flat_view.to_slice().unwrap();

        // Process in parallel
        let results = parallel::par_map(flat, f)?;

        // Convert back to array
        let result_array = ndarray::Array::from_shape_vec(shape, results).map_err(|e| {
            CoreError::DimensionError(
                ErrorContext::new(format!("Failed to reshape results: {}", e))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        Ok(result_array)
    }
}

/// Extension to CoreError for scheduler errors
impl CoreError {
    /// Create a new scheduler error
    pub fn scheduler_error(message: &str) -> Self {
        CoreError::SchedulerError(
            ErrorContext::new(message.to_string())
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    }
}
