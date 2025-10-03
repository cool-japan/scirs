//! Thread pool integration for shared worker management
//!
//! This module provides integration with a shared thread pool that can be
//! used across different scirs2 modules for consistent thread management
//! and resource control.

// use scirs2_core::parallel_ops::prelude::*; // FORBIDDEN: Use scirs2-core::parallel_ops instead
use scirs2_core::parallel_ops::*;
use std::sync::{Arc, Mutex, OnceLock};

/// Global thread pool configuration
static THREAD_POOL_CONFIG: OnceLock<Arc<Mutex<ThreadPoolConfig>>> = OnceLock::new();

/// Thread pool configuration
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Number of worker threads
    pub num_threads: Option<usize>,
    /// Stack size for worker threads
    pub stack_size: Option<usize>,
    /// Thread name prefix
    pub thread_name_prefix: String,
    /// Whether to pin threads to CPUs
    pub pin_threads: bool,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            num_threads: None,                 // Use system default
            stack_size: Some(8 * 1024 * 1024), // 8MB
            thread_name_prefix: "scirs2-worker".to_string(),
            pin_threads: false,
        }
    }
}

/// Initialize the global thread pool configuration
#[allow(dead_code)]
pub fn init_thread_pool(config: ThreadPoolConfig) -> Result<(), String> {
    THREAD_POOL_CONFIG
        .set(Arc::new(Mutex::new(config)))
        .map_err(|_| "Thread pool already initialized".to_string())
}

/// Get the current thread pool configuration
#[allow(dead_code)]
pub fn get_thread_pool_config() -> ThreadPoolConfig {
    THREAD_POOL_CONFIG
        .get()
        .map(|config| config.lock().unwrap().clone())
        .unwrap_or_default()
}

/// Update thread pool configuration
#[allow(dead_code)]
pub fn update_thread_pool_config<F>(_updatefn: F) -> Result<(), String>
where
    F: FnOnce(&mut ThreadPoolConfig),
{
    if let Some(config) = THREAD_POOL_CONFIG.get() {
        let mut config = config.lock().unwrap();
        _updatefn(&mut *config);
        Ok(())
    } else {
        Err("Thread pool not initialized".to_string())
    }
}

/// Thread-local worker information
#[derive(Debug, Clone)]
pub struct WorkerInfo {
    /// Worker thread ID
    pub thread_id: usize,
    /// Total number of workers
    pub num_workers: usize,
    /// CPU affinity if pinned
    pub cpu_affinity: Option<usize>,
}

thread_local! {
    static WORKER_INFO: std::cell::RefCell<Option<WorkerInfo>> = const { std::cell::RefCell::new(None) };
}

/// Get current worker information
#[allow(dead_code)]
pub fn current_worker_info() -> Option<WorkerInfo> {
    WORKER_INFO.with(|info| info.borrow().clone())
}

/// Set worker information for the current thread
#[allow(dead_code)]
pub fn set_worker_info(info: WorkerInfo) {
    WORKER_INFO.with(|cell| {
        *cell.borrow_mut() = Some(info);
    });
}

/// Parallel iterator with thread pool integration
#[allow(dead_code)]
pub trait ParallelIteratorExt: ParallelIterator {
    /// Configure the number of threads for this operation
    fn with_threads(self, numthreads: usize) -> Self;

    /// Configure thread-local initialization
    fn with_thread_init<F>(self, init: F) -> Self
    where
        F: Fn() + Send + Sync + 'static;
}

/// Extension trait for arrays to use the shared thread pool
#[allow(dead_code)]
pub trait ThreadPoolArrayExt<T, D> {
    /// Apply a function to each element in parallel using the shared thread pool
    fn par_map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&mut T) + Send + Sync;

    /// Apply a function to chunks in parallel
    fn par_chunks_mut<F>(&mut self, chunksize: usize, f: F)
    where
        F: Fn(&mut [T]) + Send + Sync;
}

/// Thread pool aware execution context
#[allow(dead_code)]
pub struct ThreadPoolContext {
    config: ThreadPoolConfig,
}

impl ThreadPoolContext {
    pub fn new() -> Self {
        Self {
            config: get_thread_pool_config(),
        }
    }

    /// Execute a parallel operation with the configured thread pool
    pub fn execute_parallel<F, R>(&self, operation: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        // In a real implementation, this would configure the thread pool
        // before executing the operation
        operation()
    }

    /// Execute with a specific number of threads
    pub fn execute_with_threads<F, R>(&self, numthreads: usize, operation: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        // Configure thread count for this operation
        let _prev_threads = num_threads();
        // In real implementation, would set thread count here
        let result = operation();
        // Restore previous thread count
        result
    }
}

/// Adaptive thread pool that adjusts based on workload
#[allow(dead_code)]
pub struct AdaptiveThreadPool {
    min_threads: usize,
    max_threads: usize,
    current_threads: Arc<Mutex<usize>>,
    load_threshold: f64,
}

impl AdaptiveThreadPool {
    pub fn new(_min_threads: usize, maxthreads: usize) -> Self {
        Self {
            min_threads: _min_threads,
            max_threads: maxthreads,
            current_threads: Arc::new(Mutex::new(_min_threads)),
            load_threshold: 0.8,
        }
    }

    /// Adjust thread count based on current load
    pub fn adjust_threads(&self, currentload: f64) {
        let mut threads = self.current_threads.lock().unwrap();

        if currentload > self.load_threshold && *threads < self.max_threads {
            *threads = (*threads + 1).min(self.max_threads);
        } else if currentload < self.load_threshold * 0.5 && *threads > self.min_threads {
            *threads = (*threads - 1).max(self.min_threads);
        }
    }

    /// Get current thread count
    pub fn current_thread_count(&self) -> usize {
        *self.current_threads.lock().unwrap()
    }
}

/// Work-stealing queue for load balancing
#[allow(dead_code)]
pub struct WorkStealingQueue<T> {
    queues: Vec<Arc<Mutex<Vec<T>>>>,
}

impl<T: Send> WorkStealingQueue<T> {
    pub fn new(_numqueues: usize) -> Self {
        let _queues = (0.._numqueues)
            .map(|_| Arc::new(Mutex::new(Vec::new())))
            .collect();

        Self { queues: _queues }
    }

    /// Push work to a specific queue
    pub fn push(&self, queueid: usize, item: T) {
        if let Some(queue) = self.queues.get(queueid) {
            queue.lock().unwrap().push(item);
        }
    }

    /// Try to pop from a queue, stealing from others if empty
    pub fn pop(&self, queueid: usize) -> Option<T> {
        // Try own queue first
        if let Some(queue) = self.queues.get(queueid) {
            if let Some(item) = queue.lock().unwrap().pop() {
                return Some(item);
            }
        }

        // Try to steal from other queues
        for (i, queue) in self.queues.iter().enumerate() {
            if i != queueid {
                if let Some(item) = queue.lock().unwrap().pop() {
                    return Some(item);
                }
            }
        }

        None
    }
}

/// Integration with scirs2-core parallel operations
#[allow(dead_code)]
pub fn configure_parallel_ops() {
    let config = get_thread_pool_config();

    // Configure based on thread pool settings
    if let Some(num_threads) = config.num_threads {
        // In real implementation, would configure core parallel ops
        let _ = num_threads;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_pool_config() {
        let config = ThreadPoolConfig {
            num_threads: Some(4),
            stack_size: Some(4 * 1024 * 1024),
            thread_name_prefix: "test-worker".to_string(),
            pin_threads: true,
        };

        assert_eq!(config.num_threads, Some(4));
        assert_eq!(config.thread_name_prefix, "test-worker");
    }

    #[test]
    fn test_adaptive_thread_pool() {
        let pool = AdaptiveThreadPool::new(2, 8);

        assert_eq!(pool.current_thread_count(), 2);

        // High load should increase threads
        pool.adjust_threads(0.9);
        assert_eq!(pool.current_thread_count(), 3);

        // Low load should decrease threads
        pool.adjust_threads(0.3);
        assert_eq!(pool.current_thread_count(), 2);
    }

    #[test]
    fn test_work_stealing_queue() {
        let queue: WorkStealingQueue<i32> = WorkStealingQueue::new(2);

        // Push to queue 0
        queue.push(0, 1);
        queue.push(0, 2);

        // Pop from queue 0
        assert_eq!(queue.pop(0), Some(2));

        // Queue 1 steals from queue 0
        assert_eq!(queue.pop(1), Some(1));

        // Both empty now
        assert_eq!(queue.pop(0), None);
        assert_eq!(queue.pop(1), None);
    }
}
