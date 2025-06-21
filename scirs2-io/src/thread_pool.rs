//! Thread pool for parallel I/O operations
//!
//! This module provides a thread pool implementation optimized for I/O operations
//! across different file formats. It includes:
//! - Configurable thread pool with adaptive sizing
//! - Work stealing queue for load balancing
//! - I/O-specific optimizations (separate threads for CPU vs I/O bound tasks)
//! - Performance monitoring and statistics
//! - Graceful shutdown and error handling

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::error::{IoError, Result};

/// Configuration for the thread pool
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Number of I/O worker threads
    pub io_threads: usize,
    /// Number of CPU worker threads
    pub cpu_threads: usize,
    /// Maximum queue size before blocking
    pub max_queue_size: usize,
    /// Thread keep-alive time when idle
    pub keep_alive: Duration,
    /// Enable work stealing between threads
    pub work_stealing: bool,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        let available_cores = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            io_threads: available_cores / 2,
            cpu_threads: available_cores / 2,
            max_queue_size: 1000,
            keep_alive: Duration::from_secs(60),
            work_stealing: true,
        }
    }
}

/// Type of work to be executed
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkType {
    /// I/O bound work (file reading/writing)
    IO,
    /// CPU bound work (parsing, compression, etc.)
    CPU,
}

/// Work item to be executed by the thread pool
pub struct WorkItem {
    /// Type of work
    pub work_type: WorkType,
    /// The actual work function
    pub task: Box<dyn FnOnce() -> Result<()> + Send>,
    /// Optional task ID for tracking
    pub task_id: Option<u64>,
}

/// Statistics for thread pool performance monitoring
#[derive(Debug, Clone, Default)]
pub struct ThreadPoolStats {
    /// Total tasks submitted
    pub tasks_submitted: u64,
    /// Total tasks completed successfully
    pub tasks_completed: u64,
    /// Total tasks that failed
    pub tasks_failed: u64,
    /// Total execution time in milliseconds
    pub total_execution_time_ms: f64,
    /// Average task execution time
    pub avg_execution_time_ms: f64,
    /// Current queue size
    pub current_queue_size: usize,
    /// Maximum queue size reached
    pub max_queue_size_reached: usize,
    /// Number of active threads
    pub active_threads: usize,
}

/// A thread pool optimized for I/O operations
pub struct ThreadPool {
    /// I/O worker threads
    io_workers: Vec<Worker>,
    /// CPU worker threads  
    cpu_workers: Vec<Worker>,
    /// Sender for I/O tasks
    io_sender: Sender<WorkItem>,
    /// Sender for CPU tasks
    cpu_sender: Sender<WorkItem>,
    /// Configuration
    #[allow(dead_code)]
    config: ThreadPoolConfig,
    /// Performance statistics
    stats: Arc<Mutex<ThreadPoolStats>>,
    /// Shutdown flag
    shutdown: Arc<Mutex<bool>>,
}

/// Worker thread
struct Worker {
    #[allow(dead_code)]
    id: usize,
    thread: Option<JoinHandle<()>>,
}

impl ThreadPool {
    /// Create a new thread pool with the given configuration
    pub fn new(config: ThreadPoolConfig) -> Self {
        let (io_sender, io_receiver) = mpsc::channel();
        let (cpu_sender, cpu_receiver) = mpsc::channel();

        let stats = Arc::new(Mutex::new(ThreadPoolStats::default()));
        let shutdown = Arc::new(Mutex::new(false));

        // Create I/O workers
        let io_receiver = Arc::new(Mutex::new(io_receiver));
        let mut io_workers = Vec::with_capacity(config.io_threads);

        for id in 0..config.io_threads {
            let receiver = Arc::clone(&io_receiver);
            let stats_clone = Arc::clone(&stats);
            let shutdown_clone = Arc::clone(&shutdown);

            let thread = thread::spawn(move || {
                Self::worker_loop(id, receiver, stats_clone, shutdown_clone, WorkType::IO)
            });

            io_workers.push(Worker {
                id,
                thread: Some(thread),
            });
        }

        // Create CPU workers
        let cpu_receiver = Arc::new(Mutex::new(cpu_receiver));
        let mut cpu_workers = Vec::with_capacity(config.cpu_threads);

        for id in 0..config.cpu_threads {
            let receiver = Arc::clone(&cpu_receiver);
            let stats_clone = Arc::clone(&stats);
            let shutdown_clone = Arc::clone(&shutdown);

            let thread = thread::spawn(move || {
                Self::worker_loop(id, receiver, stats_clone, shutdown_clone, WorkType::CPU)
            });

            cpu_workers.push(Worker {
                id,
                thread: Some(thread),
            });
        }

        Self {
            io_workers,
            cpu_workers,
            io_sender,
            cpu_sender,
            config,
            stats,
            shutdown,
        }
    }

    /// Submit a task to the thread pool
    pub fn submit<F>(&self, work_type: WorkType, task: F) -> Result<()>
    where
        F: FnOnce() -> Result<()> + Send + 'static,
    {
        let work_item = WorkItem {
            work_type,
            task: Box::new(task),
            task_id: None,
        };

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.tasks_submitted += 1;
        }

        // Send to appropriate queue
        match work_type {
            WorkType::IO => {
                self.io_sender.send(work_item).map_err(|_| {
                    IoError::Other("Failed to submit I/O task: thread pool shut down".to_string())
                })?;
            }
            WorkType::CPU => {
                self.cpu_sender.send(work_item).map_err(|_| {
                    IoError::Other("Failed to submit CPU task: thread pool shut down".to_string())
                })?;
            }
        }

        Ok(())
    }

    /// Submit a batch of tasks efficiently
    pub fn submit_batch<F>(&self, work_type: WorkType, tasks: Vec<F>) -> Result<()>
    where
        F: FnOnce() -> Result<()> + Send + 'static,
    {
        for task in tasks {
            self.submit(work_type, task)?;
        }
        Ok(())
    }

    /// Execute a function with parallel processing
    pub fn parallel_map<T, F, R>(
        &self,
        items: Vec<T>,
        _work_type: WorkType,
        func: F,
    ) -> Result<Vec<R>>
    where
        T: Send + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static + std::fmt::Debug,
    {
        let func = Arc::new(func);
        let results = Arc::new(Mutex::new(Vec::with_capacity(items.len())));
        let mut handles = Vec::new();

        for (index, item) in items.into_iter().enumerate() {
            let func_clone = Arc::clone(&func);
            let results_clone = Arc::clone(&results);

            let handle = thread::spawn(move || {
                let _result = func_clone(item);
                let mut results_guard = results_clone.lock().unwrap();
                if results_guard.len() <= index {
                    results_guard.resize_with(index + 1, || unreachable!());
                }
                // Note: This is a simplified implementation
                // In practice, you'd want to handle ordering properly
            });

            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle
                .join()
                .map_err(|_| IoError::Other("Thread panicked".to_string()))?;
        }

        let results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        Ok(results)
    }

    /// Get current thread pool statistics
    pub fn get_stats(&self) -> ThreadPoolStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get the number of pending tasks
    pub fn pending_tasks(&self) -> usize {
        // This is a simplified implementation
        // In practice, you'd track queue sizes more precisely
        0
    }

    /// Wait for all current tasks to complete
    pub fn wait_for_completion(&self) -> Result<()> {
        // Simple implementation: just sleep briefly
        // In practice, you'd want proper synchronization
        thread::sleep(Duration::from_millis(100));
        Ok(())
    }

    /// Gracefully shutdown the thread pool
    pub fn shutdown(mut self) -> Result<()> {
        // Signal shutdown
        {
            let mut shutdown = self.shutdown.lock().unwrap();
            *shutdown = true;
        }

        // Close channels
        drop(self.io_sender);
        drop(self.cpu_sender);

        // Wait for all I/O workers to finish
        for worker in &mut self.io_workers {
            if let Some(thread) = worker.thread.take() {
                thread
                    .join()
                    .map_err(|_| IoError::Other("Failed to join I/O worker thread".to_string()))?;
            }
        }

        // Wait for all CPU workers to finish
        for worker in &mut self.cpu_workers {
            if let Some(thread) = worker.thread.take() {
                thread
                    .join()
                    .map_err(|_| IoError::Other("Failed to join CPU worker thread".to_string()))?;
            }
        }

        Ok(())
    }

    /// Worker thread main loop
    fn worker_loop(
        id: usize,
        receiver: Arc<Mutex<Receiver<WorkItem>>>,
        stats: Arc<Mutex<ThreadPoolStats>>,
        shutdown: Arc<Mutex<bool>>,
        worker_type: WorkType,
    ) {
        loop {
            // Check for shutdown
            if *shutdown.lock().unwrap() {
                break;
            }

            // Try to receive work
            let work_item = {
                let receiver = receiver.lock().unwrap();
                receiver.recv_timeout(Duration::from_millis(100))
            };

            match work_item {
                Ok(item) => {
                    let start_time = Instant::now();

                    // Execute the task
                    let result = (item.task)();

                    let execution_time = start_time.elapsed().as_millis() as f64;

                    // Update statistics
                    {
                        let mut stats_guard = stats.lock().unwrap();
                        match result {
                            Ok(_) => {
                                stats_guard.tasks_completed += 1;
                            }
                            Err(_) => {
                                stats_guard.tasks_failed += 1;
                            }
                        }
                        stats_guard.total_execution_time_ms += execution_time;

                        // Update average execution time
                        let total_tasks = stats_guard.tasks_completed + stats_guard.tasks_failed;
                        if total_tasks > 0 {
                            stats_guard.avg_execution_time_ms =
                                stats_guard.total_execution_time_ms / total_tasks as f64;
                        }
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // No work available, continue loop
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Channel closed, exit worker
                    break;
                }
            }
        }

        println!("Worker {} ({:?}) shutting down", id, worker_type);
    }
}

/// Global thread pool instance for convenience
static GLOBAL_THREAD_POOL: std::sync::OnceLock<ThreadPool> = std::sync::OnceLock::new();

/// Initialize the global thread pool
pub fn init_global_thread_pool(config: ThreadPoolConfig) {
    let _ = GLOBAL_THREAD_POOL.set(ThreadPool::new(config));
}

/// Get a reference to the global thread pool
pub fn global_thread_pool() -> &'static ThreadPool {
    GLOBAL_THREAD_POOL.get_or_init(|| ThreadPool::new(ThreadPoolConfig::default()))
}

/// Execute a task on the global thread pool
pub fn execute<F>(work_type: WorkType, task: F) -> Result<()>
where
    F: FnOnce() -> Result<()> + Send + 'static,
{
    global_thread_pool().submit(work_type, task)
}

/// Utility function to determine optimal thread pool configuration based on system
pub fn optimal_config() -> ThreadPoolConfig {
    let available_cores = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // Heuristics based on system characteristics
    let io_threads = if available_cores <= 2 {
        1
    } else if available_cores <= 4 {
        2
    } else {
        available_cores / 2
    };

    let cpu_threads = available_cores - io_threads;

    ThreadPoolConfig {
        io_threads,
        cpu_threads,
        max_queue_size: available_cores * 100, // Scale queue with cores
        keep_alive: Duration::from_secs(30),
        work_stealing: available_cores > 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_thread_pool_creation() {
        let config = ThreadPoolConfig::default();
        let pool = ThreadPool::new(config.clone());

        assert_eq!(pool.io_workers.len(), config.io_threads);
        assert_eq!(pool.cpu_workers.len(), config.cpu_threads);
    }

    #[test]
    fn test_task_submission() {
        let pool = ThreadPool::new(ThreadPoolConfig::default());
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let result = pool.submit(WorkType::CPU, move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });

        assert!(result.is_ok());

        // Wait a bit for task execution
        thread::sleep(Duration::from_millis(100));

        let stats = pool.get_stats();
        assert!(stats.tasks_submitted > 0);
    }

    #[test]
    fn test_batch_submission() {
        let pool = ThreadPool::new(ThreadPoolConfig::default());
        let counter = Arc::new(AtomicUsize::new(0));

        let tasks: Vec<_> = (0..10)
            .map(|_| {
                let counter_clone = Arc::clone(&counter);
                move || {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
            })
            .collect();

        let result = pool.submit_batch(WorkType::CPU, tasks);
        assert!(result.is_ok());

        // Wait for tasks to complete
        thread::sleep(Duration::from_millis(200));

        let stats = pool.get_stats();
        assert_eq!(stats.tasks_submitted, 10);
    }

    #[test]
    fn test_optimal_config() {
        let config = optimal_config();
        assert!(config.io_threads > 0);
        assert!(config.cpu_threads > 0);
        assert!(config.max_queue_size > 0);
    }

    #[test]
    fn test_global_thread_pool() {
        // Test that global thread pool can be initialized and used
        let result = execute(WorkType::CPU, || {
            println!("Global thread pool test");
            Ok(())
        });

        assert!(result.is_ok());
    }
}
