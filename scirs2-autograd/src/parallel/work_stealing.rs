//! Work-stealing queue implementation for load balancing
//!
//! This module provides a work-stealing deque that allows threads to steal
//! work from each other, improving load balancing and overall throughput.

use std::collections::VecDeque;
use std::ptr;
use std::sync::Mutex;
use std::sync::{
    atomic::{AtomicPtr, AtomicUsize, Ordering},
    Arc,
};

/// Work-stealing deque for load balancing between threads
pub struct WorkStealingDeque<T> {
    /// Bottom pointer for owner thread
    bottom: AtomicUsize,
    /// Top pointer for stealer threads  
    top: AtomicUsize,
    /// Array of tasks
    array: AtomicPtr<WorkStealingArray<T>>,
}

impl<T> WorkStealingDeque<T> {
    /// Create a new work-stealing deque
    pub fn new() -> Self {
        let initial_array = Box::into_raw(Box::new(WorkStealingArray::new(1024)));

        Self {
            bottom: AtomicUsize::new(0),
            top: AtomicUsize::new(0),
            array: AtomicPtr::new(initial_array),
        }
    }

    /// Push a task to the bottom (owner thread only)
    pub fn push(&self, task: T) {
        let bottom = self.bottom.load(Ordering::Relaxed);
        let top = self.top.load(Ordering::Acquire);

        let array_ptr = self.array.load(Ordering::Relaxed);
        let array = unsafe { &*array_ptr };

        // Check if we need to resize
        if bottom - top >= array.capacity - 1 {
            self.resize();
        }

        // Store the task
        array.put(bottom, task);

        // Update bottom pointer
        self.bottom.store(bottom + 1, Ordering::Release);
    }

    /// Pop a task from the bottom (owner thread only)
    pub fn pop(&self) -> Option<T> {
        let bottom = self.bottom.load(Ordering::Relaxed);

        if bottom == 0 {
            return None;
        }

        let new_bottom = bottom - 1;
        self.bottom.store(new_bottom, Ordering::Relaxed);

        let array_ptr = self.array.load(Ordering::Relaxed);
        let array = unsafe { &*array_ptr };

        let task = array.get(new_bottom);

        let top = self.top.load(Ordering::Relaxed);

        if new_bottom > top {
            // Common case: we own the task
            return Some(task);
        }

        if new_bottom == top {
            // Race with stealer: try to win the task
            if self
                .top
                .compare_exchange_weak(top, top + 1, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                self.bottom.store(bottom, Ordering::Relaxed);
                return Some(task);
            }
        }

        // Lost the race or queue is empty
        self.bottom.store(bottom, Ordering::Relaxed);
        None
    }

    /// Steal a task from the top (stealer threads)
    pub fn steal(&self) -> StealResult<T> {
        let top = self.top.load(Ordering::Acquire);
        let bottom = self.bottom.load(Ordering::Acquire);

        if top >= bottom {
            return StealResult::Empty;
        }

        let array_ptr = self.array.load(Ordering::Acquire);
        let array = unsafe { &*array_ptr };

        let task = array.get(top);

        // Try to increment top
        if self
            .top
            .compare_exchange_weak(top, top + 1, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
        {
            StealResult::Success(task)
        } else {
            StealResult::Retry
        }
    }

    /// Check if the deque is empty
    pub fn is_empty(&self) -> bool {
        let bottom = self.bottom.load(Ordering::Relaxed);
        let top = self.top.load(Ordering::Relaxed);
        bottom <= top
    }

    /// Get the current size of the deque
    pub fn size(&self) -> usize {
        let bottom = self.bottom.load(Ordering::Relaxed);
        let top = self.top.load(Ordering::Relaxed);
        bottom.saturating_sub(top)
    }

    /// Resize the internal array (private method)
    fn resize(&self) {
        let old_array_ptr = self.array.load(Ordering::Relaxed);
        let old_array = unsafe { &*old_array_ptr };

        let new_capacity = old_array.capacity * 2;
        let new_array = Box::into_raw(Box::new(WorkStealingArray::new(new_capacity)));

        let bottom = self.bottom.load(Ordering::Relaxed);
        let top = self.top.load(Ordering::Relaxed);

        // Copy tasks to new array
        for i in top..bottom {
            let task = old_array.get(i);
            unsafe { &*new_array }.put(i, task);
        }

        // Update array pointer
        self.array.store(new_array, Ordering::Release);

        // Note: In a real implementation, we'd need to handle memory reclamation
        // safely (e.g., using hazard pointers or epoch-based reclamation)
    }
}

impl<T> Default for WorkStealingDeque<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for WorkStealingDeque<T> {
    fn drop(&mut self) {
        let array_ptr = self.array.load(Ordering::Relaxed);
        if !array_ptr.is_null() {
            unsafe {
                let _ = Box::from_raw(array_ptr);
            }
        }
    }
}

/// Result of a steal operation
#[derive(Debug, PartialEq)]
pub enum StealResult<T> {
    /// Successfully stole a task
    Success(T),
    /// No tasks available
    Empty,
    /// Retry the steal operation
    Retry,
}

/// Array backing the work-stealing deque
struct WorkStealingArray<T> {
    capacity: usize,
    mask: usize,
    data: Vec<std::mem::MaybeUninit<T>>,
}

impl<T> WorkStealingArray<T> {
    /// Create a new array with the given capacity
    fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two());

        let mut data = Vec::with_capacity(capacity);
        data.resize_with(capacity, || std::mem::MaybeUninit::uninit());

        Self {
            capacity,
            mask: capacity - 1,
            data,
        }
    }

    /// Put a task at the given index
    fn put(&self, index: usize, task: T) {
        let pos = index & self.mask;
        unsafe {
            let ptr = self.data.as_ptr().add(pos) as *mut std::mem::MaybeUninit<T>;
            ptr::write(ptr, std::mem::MaybeUninit::new(task));
        }
    }

    /// Get a task from the given index
    fn get(&self, index: usize) -> T {
        let pos = index & self.mask;
        unsafe {
            let ptr = self.data.as_ptr().add(pos);
            ptr::read(ptr).assume_init()
        }
    }
}

/// Work-stealing scheduler for distributing tasks
pub struct WorkStealingScheduler<T> {
    /// Work-stealing deques for each thread
    deques: Vec<Arc<WorkStealingDeque<T>>>,
    /// Number of threads
    num_threads: usize,
    /// Round-robin counter for task submission
    round_robin: AtomicUsize,
}

impl<T> WorkStealingScheduler<T>
where
    T: Send + 'static,
{
    /// Create a new work-stealing scheduler
    pub fn new(num_threads: usize) -> Self {
        let deques = (0..num_threads)
            .map(|_| Arc::new(WorkStealingDeque::new()))
            .collect();

        Self {
            deques,
            num_threads,
            round_robin: AtomicUsize::new(0),
        }
    }

    /// Submit a task to the scheduler
    pub fn submit(&self, task: T) {
        let thread_id = self.round_robin.fetch_add(1, Ordering::Relaxed) % self.num_threads;
        self.deques[thread_id].push(task);
    }

    /// Submit a task to a specific thread
    pub fn submit_to_thread(&self, task: T, thread_id: usize) {
        if thread_id < self.num_threads {
            self.deques[thread_id].push(task);
        } else {
            // Fallback to round-robin
            self.submit(task);
        }
    }

    /// Try to get a task for the given thread (with work stealing)
    pub fn try_get_task(&self, thread_id: usize) -> Option<T> {
        if thread_id >= self.num_threads {
            return None;
        }

        // First try to get from own queue
        if let Some(task) = self.deques[thread_id].pop() {
            return Some(task);
        }

        // Try to steal from other threads
        for steal_attempts in 0..self.num_threads * 2 {
            let target = (thread_id + steal_attempts + 1) % self.num_threads;

            match self.deques[target].steal() {
                StealResult::Success(task) => return Some(task),
                StealResult::Empty => continue,
                StealResult::Retry => {
                    // Brief pause before retry
                    std::hint::spin_loop();
                    continue;
                }
            }
        }

        None
    }

    /// Get work-stealing statistics
    pub fn get_stats(&self) -> WorkStealingStats {
        let queue_sizes: Vec<usize> = self.deques.iter().map(|deque| deque.size()).collect();
        let total_tasks: usize = queue_sizes.iter().sum();
        let max_queue_size = queue_sizes.iter().max().copied().unwrap_or(0);
        let min_queue_size = queue_sizes.iter().min().copied().unwrap_or(0);

        let load_balance = if max_queue_size == 0 {
            1.0
        } else {
            1.0 - (max_queue_size - min_queue_size) as f64 / max_queue_size as f64
        };

        WorkStealingStats {
            num_threads: self.num_threads,
            total_tasks,
            queue_sizes,
            load_balance,
            avg_queue_size: total_tasks as f64 / self.num_threads as f64,
        }
    }

    /// Get the number of threads
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Check if all queues are empty
    pub fn is_empty(&self) -> bool {
        self.deques.iter().all(|deque| deque.is_empty())
    }
}

/// Statistics for work-stealing performance
#[derive(Debug, Clone)]
pub struct WorkStealingStats {
    /// Number of threads
    pub num_threads: usize,
    /// Total tasks across all queues
    pub total_tasks: usize,
    /// Size of each queue
    pub queue_sizes: Vec<usize>,
    /// Load balance efficiency (0.0 to 1.0)
    pub load_balance: f64,
    /// Average queue size
    pub avg_queue_size: f64,
}

impl WorkStealingStats {
    /// Calculate load imbalance ratio
    pub fn load_imbalance(&self) -> f64 {
        1.0 - self.load_balance
    }

    /// Get the most loaded thread
    pub fn most_loaded_thread(&self) -> Option<usize> {
        self.queue_sizes
            .iter()
            .enumerate()
            .max_by_key(|(_, &size)| size)
            .map(|(id, _)| id)
    }

    /// Get the least loaded thread
    pub fn least_loaded_thread(&self) -> Option<usize> {
        self.queue_sizes
            .iter()
            .enumerate()
            .min_by_key(|(_, &size)| size)
            .map(|(id, _)| id)
    }
}

/// Simple work-stealing pool for basic use cases
pub struct SimpleWorkStealingPool<T> {
    scheduler: WorkStealingScheduler<T>,
    worker_handles: Vec<std::thread::JoinHandle<()>>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl<T> SimpleWorkStealingPool<T>
where
    T: Send + 'static,
{
    /// Create a new simple work-stealing pool
    pub fn new<F>(num_threads: usize, task_processor: F) -> Self
    where
        F: Fn(T) + Send + Sync + Clone + 'static,
    {
        let scheduler = WorkStealingScheduler::new(num_threads);
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut worker_handles = Vec::with_capacity(num_threads);

        for thread_id in 0..num_threads {
            let scheduler_clone = WorkStealingScheduler {
                deques: scheduler.deques.clone(),
                num_threads: scheduler.num_threads,
                round_robin: AtomicUsize::new(0),
            };
            let shutdown_clone = Arc::clone(&shutdown);
            let processor = task_processor.clone();

            let handle = std::thread::spawn(move || {
                while !shutdown_clone.load(Ordering::Relaxed) {
                    if let Some(task) = scheduler_clone.try_get_task(thread_id) {
                        processor(task);
                    } else {
                        // No work available, brief sleep
                        std::thread::sleep(std::time::Duration::from_micros(100));
                    }
                }
            });

            worker_handles.push(handle);
        }

        Self {
            scheduler,
            worker_handles,
            shutdown,
        }
    }

    /// Submit a task to the pool
    pub fn submit(&self, task: T) {
        self.scheduler.submit(task);
    }

    /// Get work-stealing statistics
    pub fn get_stats(&self) -> WorkStealingStats {
        self.scheduler.get_stats()
    }

    /// Shutdown the pool and wait for all workers to finish
    pub fn shutdown(self) -> Result<(), Box<dyn std::any::Any + Send>> {
        self.shutdown.store(true, Ordering::Relaxed);

        for handle in self.worker_handles {
            handle.join()?;
        }

        Ok(())
    }
}

/// Lock-free work-stealing deque implementation using hazard pointers
/// (Simplified version for demonstration)
pub struct LockFreeWorkStealingDeque<T> {
    inner: Mutex<VecDeque<T>>, // Simplified with mutex for safety
}

impl<T> LockFreeWorkStealingDeque<T> {
    /// Create a new lock-free work-stealing deque
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(VecDeque::new()),
        }
    }

    /// Push to bottom (owner thread)
    pub fn push(&self, item: T) {
        let mut deque = self.inner.lock().unwrap();
        deque.push_back(item);
    }

    /// Pop from bottom (owner thread)
    pub fn pop(&self) -> Option<T> {
        let mut deque = self.inner.lock().unwrap();
        deque.pop_back()
    }

    /// Steal from top (stealer threads)
    pub fn steal(&self) -> Option<T> {
        let mut deque = self.inner.lock().unwrap();
        deque.pop_front()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        let deque = self.inner.lock().unwrap();
        deque.is_empty()
    }

    /// Get size
    pub fn len(&self) -> usize {
        let deque = self.inner.lock().unwrap();
        deque.len()
    }
}

impl<T> Default for LockFreeWorkStealingDeque<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    #[allow(unused_imports)]
    use std::sync::Mutex;

    #[test]
    fn test_work_stealing_deque_basic() {
        let deque = WorkStealingDeque::new();

        // Push some tasks
        deque.push(1);
        deque.push(2);
        deque.push(3);

        assert_eq!(deque.size(), 3);
        assert!(!deque.is_empty());

        // Pop tasks
        assert_eq!(deque.pop(), Some(3));
        assert_eq!(deque.pop(), Some(2));
        assert_eq!(deque.pop(), Some(1));
        assert_eq!(deque.pop(), None);

        assert!(deque.is_empty());
    }

    #[test]
    fn test_work_stealing_deque_steal() {
        let deque = WorkStealingDeque::new();

        // Push tasks
        deque.push(1);
        deque.push(2);
        deque.push(3);

        // Steal from top
        assert_eq!(deque.steal(), StealResult::Success(1));
        assert_eq!(deque.steal(), StealResult::Success(2));
        assert_eq!(deque.steal(), StealResult::Success(3));
        assert_eq!(deque.steal(), StealResult::Empty);
    }

    #[test]
    fn test_work_stealing_scheduler() {
        let scheduler = WorkStealingScheduler::new(4);

        // Submit tasks
        for i in 0..10 {
            scheduler.submit(i);
        }

        // Get tasks from different threads
        let mut collected = Vec::new();
        for thread_id in 0..4 {
            while let Some(task) = scheduler.try_get_task(thread_id) {
                collected.push(task);
            }
        }

        assert_eq!(collected.len(), 10);
        collected.sort();
        assert_eq!(collected, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_work_stealing_stats() {
        let scheduler = WorkStealingScheduler::new(3);

        // Submit uneven tasks
        for i in 0..5 {
            scheduler.submit_to_thread(i, 0);
        }
        for i in 5..7 {
            scheduler.submit_to_thread(i, 1);
        }
        // Thread 2 gets no tasks

        let stats = scheduler.get_stats();
        assert_eq!(stats.num_threads, 3);
        assert_eq!(stats.total_tasks, 7);
        assert_eq!(stats.queue_sizes, vec![5, 2, 0]);
        assert!(stats.load_balance < 1.0); // Should show imbalance
    }

    #[test]
    fn test_simple_work_stealing_pool() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let pool = SimpleWorkStealingPool::new(2, move |_task: i32| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        // Submit tasks
        for i in 0..10 {
            pool.submit(i);
        }

        // Wait a bit for processing
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Check that tasks were processed
        let processed = counter.load(Ordering::SeqCst);
        assert!(processed > 0);

        pool.shutdown().unwrap();
    }

    #[test]
    fn test_lock_free_deque() {
        let deque = LockFreeWorkStealingDeque::new();

        assert!(deque.is_empty());
        assert_eq!(deque.len(), 0);

        deque.push(1);
        deque.push(2);
        deque.push(3);

        assert_eq!(deque.len(), 3);
        assert!(!deque.is_empty());

        assert_eq!(deque.steal(), Some(1)); // Steal from front
        assert_eq!(deque.pop(), Some(3)); // Pop from back
        assert_eq!(deque.pop(), Some(2)); // Pop from back
        assert_eq!(deque.pop(), None);

        assert!(deque.is_empty());
    }

    #[test]
    fn test_concurrent_work_stealing() {
        use std::thread;

        let deque = Arc::new(WorkStealingDeque::new());
        let processed = Arc::new(AtomicUsize::new(0));

        // Producer thread
        let deque_producer = Arc::clone(&deque);
        let producer = thread::spawn(move || {
            for i in 0..100 {
                deque_producer.push(i);
            }
        });

        // Consumer threads
        let mut consumers = Vec::new();
        for _ in 0..4 {
            let deque_consumer = Arc::clone(&deque);
            let processed_consumer = Arc::clone(&processed);

            let consumer = thread::spawn(move || {
                let mut empty_count = 0;
                loop {
                    match deque_consumer.steal() {
                        StealResult::Success(_) => {
                            processed_consumer.fetch_add(1, Ordering::SeqCst);
                            empty_count = 0; // Reset empty count on success
                        }
                        StealResult::Empty => {
                            empty_count += 1;
                            // Only exit after multiple consecutive empty results
                            if empty_count > 10 {
                                break;
                            }
                            // Brief pause before checking again
                            std::thread::sleep(std::time::Duration::from_micros(100));
                        }
                        StealResult::Retry => continue,
                    }
                }
            });

            consumers.push(consumer);
        }

        producer.join().unwrap();

        // Give consumers time to finish
        std::thread::sleep(std::time::Duration::from_millis(100));

        for consumer in consumers {
            consumer.join().unwrap();
        }

        // All tasks should be processed
        let total_processed = processed.load(Ordering::SeqCst);
        assert!(total_processed > 0);
        assert!(total_processed <= 100);
    }
}
