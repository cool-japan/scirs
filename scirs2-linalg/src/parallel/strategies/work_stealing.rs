//! Work-stealing scheduler optimizations
//!
//! This module provides advanced scheduling strategies for parallel algorithms
//! using work-stealing techniques to improve load balancing and performance.

use super::super::WorkerConfig;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Work-stealing task scheduler
///
/// Implements a work-stealing scheduler that dynamically balances work
/// across threads for improved performance on irregular workloads.
pub struct WorkStealingScheduler {
    pub(crate) num_workers: usize,
    chunksize: usize,
    adaptive_chunking: bool,
}

impl WorkStealingScheduler {
    /// Create a new work-stealing scheduler
    pub fn new(config: &WorkerConfig) -> Self {
        let num_workers = config.workers.unwrap_or_else(|| {
            // Default to available parallelism or 4 threads
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });
        Self {
            num_workers,
            chunksize: config.chunksize,
            adaptive_chunking: true,
        }
    }

    /// Set whether to use adaptive chunking
    pub fn with_adaptive_chunking(mut self, adaptive: bool) -> Self {
        self.adaptive_chunking = adaptive;
        self
    }

    /// Execute work items using work-stealing strategy
    ///
    /// This function divides work into chunks and uses atomic counters
    /// to allow threads to steal work from a global queue when they
    /// finish their assigned chunks early.
    pub fn execute<T, R, F>(&self, items: &[T], f: F) -> Vec<R>
    where
        T: Send + Sync,
        R: Send + Default + Clone,
        F: Fn(&T) -> R + Send + Sync,
    {
        let n = items.len();
        if n == 0 {
            return Vec::new();
        }

        // Determine chunk size based on workload characteristics
        let chunksize = if self.adaptive_chunking {
            self.adaptive_chunksize(n)
        } else {
            self.chunksize
        };

        // Create shared work counter
        let work_counter = Arc::new(AtomicUsize::new(0));
        let results = Arc::new(Mutex::new(vec![R::default(); n]));

        // Use scoped threads to process work items
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..self.num_workers)
                .map(|_| {
                    let work_counter = work_counter.clone();
                    let results = results.clone();
                    let items_ref = items;
                    let f_ref = &f;

                    s.spawn(move || {
                        loop {
                            // Steal a chunk of work
                            let start = work_counter.fetch_add(chunksize, Ordering::SeqCst);
                            if start >= n {
                                break;
                            }

                            let end = std::cmp::min(start + chunksize, n);

                            // Process the chunk
                            for i in start..end {
                                let result = f_ref(&items_ref[i]);
                                let mut results_guard = results.lock().unwrap();
                                results_guard[i] = result;
                            }
                        }
                    })
                })
                .collect();

            // Wait for all threads to complete
            for handle in handles {
                handle.join().unwrap();
            }
        });

        // Extract results
        Arc::try_unwrap(results)
            .unwrap_or_else(|_| panic!("Failed to extract results"))
            .into_inner()
            .unwrap_or_else(|_| panic!("Failed to extract mutex inner value"))
    }

    /// Determine adaptive chunk size based on workload size
    fn adaptive_chunksize(&self, totalitems: usize) -> usize {
        // Use smaller chunks for better load balancing on smaller workloads
        // and larger chunks for better cache efficiency on larger workloads
        let items_per_worker = totalitems / self.num_workers;

        if items_per_worker < 100 {
            // Small workload: use fine-grained chunks
            std::cmp::max(1, items_per_worker / 4)
        } else if items_per_worker < 1000 {
            // Medium workload: balance between overhead and load balancing
            items_per_worker / 8
        } else {
            // Large workload: prioritize cache efficiency
            std::cmp::min(self.chunksize, items_per_worker / 16)
        }
    }

    /// Execute matrix operations with work-stealing
    ///
    /// Specialized version for matrix operations that takes into account
    /// cache line sizes and memory access patterns.
    pub fn execute_matrix<R, F>(&self, rows: usize, cols: usize, f: F) -> scirs2_core::ndarray::Array2<R>
    where
        R: Send + Default + Clone,
        F: Fn(usize, usize) -> R + Send + Sync,
    {
        // Use block partitioning for better cache efficiency
        let blocksize = 64; // Typical cache line aligned block
        let work_items: Vec<(usize, usize)> = (0..rows)
            .step_by(blocksize)
            .flat_map(|i| (0..cols).step_by(blocksize).map(move |j| (i, j)))
            .collect();

        // Process blocks using work-stealing and collect results
        let work_counter = Arc::new(AtomicUsize::new(0));
        let results_vec = Arc::new(Mutex::new(Vec::new()));

        std::thread::scope(|s| {
            let handles: Vec<_> = (0..self.num_workers)
                .map(|_| {
                    let work_counter = work_counter.clone();
                    let results_vec = results_vec.clone();
                    let work_items_ref = &work_items;
                    let f_ref = &f;

                    s.spawn(move || {
                        let mut local_results = Vec::new();

                        loop {
                            let idx = work_counter.fetch_add(1, Ordering::SeqCst);
                            if idx >= work_items_ref.len() {
                                break;
                            }

                            let (block_i, block_j) = work_items_ref[idx];
                            let i_end = std::cmp::min(block_i + blocksize, rows);
                            let j_end = std::cmp::min(block_j + blocksize, cols);

                            // Process the block
                            for i in block_i..i_end {
                                for j in block_j..j_end {
                                    local_results.push((i, j, f_ref(i, j)));
                                }
                            }
                        }

                        // Add local results to global results
                        if !local_results.is_empty() {
                            let mut global_results = results_vec.lock().unwrap();
                            global_results.extend(local_results);
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        });

        // Create result matrix from collected results
        let mut result = scirs2_core::ndarray::Array2::default((rows, cols));
        let results = Arc::try_unwrap(results_vec)
            .unwrap_or_else(|_| panic!("Failed to extract results"))
            .into_inner()
            .unwrap_or_else(|_| panic!("Failed to extract mutex inner value"));

        for (i, j, val) in results {
            result[[i, j]] = val;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_stealing_scheduler() {
        let config = WorkerConfig::default();
        let scheduler = WorkStealingScheduler::new(&config);
        
        let items = vec![1, 2, 3, 4, 5];
        let results = scheduler.execute(&items, |x| x * 2);
        
        assert_eq!(results, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_adaptive_chunking() {
        let config = WorkerConfig::default();
        let scheduler = WorkStealingScheduler::new(&config).with_adaptive_chunking(true);
        
        // Test with small workload
        assert!(scheduler.adaptive_chunksize(10) >= 1);
        
        // Test with large workload
        assert!(scheduler.adaptive_chunksize(10000) > 10);
    }

    #[test]
    fn test_matrix_execution() {
        let config = WorkerConfig::default();
        let scheduler = WorkStealingScheduler::new(&config);
        
        let result = scheduler.execute_matrix(3, 3, |i, j| i + j);
        
        assert_eq!(result.shape(), &[3, 3]);
        assert_eq!(result[[0, 0]], 0);
        assert_eq!(result[[1, 1]], 2);
        assert_eq!(result[[2, 2]], 4);
    }
}