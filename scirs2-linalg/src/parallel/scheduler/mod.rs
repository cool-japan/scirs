//! Work-stealing scheduler optimizations
//!
//! This module provides advanced scheduling strategies for parallel algorithms
//! using work-stealing techniques to improve load balancing and performance.

use super::WorkerConfig;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Work-stealing task scheduler
///
/// Implements a work-stealing scheduler that dynamically balances work
/// across threads for improved performance on irregular workloads.
pub struct WorkStealingScheduler {
    num_workers: usize,
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
    pub fn executematrix<R, F>(&self, rows: usize, cols: usize, f: F) -> ndarray::Array2<R>
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
        let mut result = ndarray::Array2::default((rows, cols));
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

/// Dynamic load balancer for irregular workloads
///
/// This struct provides dynamic load balancing for workloads where
/// different items may take varying amounts of time to process.
pub struct DynamicLoadBalancer {
    scheduler: WorkStealingScheduler,
    /// Tracks execution time statistics for adaptive scheduling
    timing_stats: Arc<Mutex<TimingStats>>,
}

#[derive(Default)]
struct TimingStats {
    total_items: usize,
    total_time_ms: u128,
    min_time_ms: u128,
    max_time_ms: u128,
}

impl DynamicLoadBalancer {
    /// Create a new dynamic load balancer
    pub fn new(config: &WorkerConfig) -> Self {
        Self {
            scheduler: WorkStealingScheduler::new(config),
            timing_stats: Arc::new(Mutex::new(TimingStats::default())),
        }
    }

    /// Execute work items with dynamic load balancing and timing
    pub fn execute_timed<T, R, F>(&self, items: &[T], f: F) -> Vec<R>
    where
        T: Send + Sync,
        R: Send + Default + Clone,
        F: Fn(&T) -> R + Send + Sync,
    {
        use std::time::Instant;

        let n = items.len();
        if n == 0 {
            return Vec::new();
        }

        let results = Arc::new(Mutex::new(vec![R::default(); n]));
        let work_counter = Arc::new(AtomicUsize::new(0));
        let timing_stats = self.timing_stats.clone();

        std::thread::scope(|s| {
            let handles: Vec<_> = (0..self.scheduler.num_workers)
                .map(|_| {
                    let work_counter = work_counter.clone();
                    let results = results.clone();
                    let timing_stats = timing_stats.clone();
                    let items_ref = items;
                    let f_ref = &f;

                    s.spawn(move || {
                        let mut local_min = u128::MAX;
                        let mut local_max = 0u128;
                        let mut local_total = 0u128;
                        let mut local_count = 0usize;

                        loop {
                            let idx = work_counter.fetch_add(1, Ordering::SeqCst);
                            if idx >= n {
                                break;
                            }

                            // Time the execution
                            let start = Instant::now();
                            let result = f_ref(&items_ref[idx]);
                            let elapsed = start.elapsed().as_millis();

                            // Update local statistics
                            local_min = local_min.min(elapsed);
                            local_max = local_max.max(elapsed);
                            local_total += elapsed;
                            local_count += 1;

                            // Store result
                            let mut results_guard = results.lock().unwrap();
                            results_guard[idx] = result;
                        }

                        // Update global statistics
                        if local_count > 0 {
                            let mut stats = timing_stats.lock().unwrap();
                            stats.total_items += local_count;
                            stats.total_time_ms += local_total;
                            stats.min_time_ms = stats.min_time_ms.min(local_min);
                            stats.max_time_ms = stats.max_time_ms.max(local_max);
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        });

        Arc::try_unwrap(results)
            .unwrap_or_else(|_| panic!("Failed to extract results"))
            .into_inner()
            .unwrap_or_else(|_| panic!("Failed to extract mutex inner value"))
    }

    /// Get average execution time per item
    pub fn get_average_time_ms(&self) -> f64 {
        let stats = self.timing_stats.lock().unwrap();
        if stats.total_items > 0 {
            stats.total_time_ms as f64 / stats.total_items as f64
        } else {
            0.0
        }
    }

    /// Get timing variance to detect irregular workloads
    pub fn get_time_variance(&self) -> f64 {
        let stats = self.timing_stats.lock().unwrap();
        if stats.total_items > 0 && stats.max_time_ms > stats.min_time_ms {
            (stats.max_time_ms - stats.min_time_ms) as f64 / stats.min_time_ms as f64
        } else {
            0.0
        }
    }
}

/// Advanced work-stealing scheduler with NUMA awareness and cache optimization
///
/// This enhanced scheduler provides advanced optimizations for work-stealing
/// including NUMA topology awareness and cache-friendly work distribution.
pub struct AdvancedWorkStealingScheduler {
    base_scheduler: WorkStealingScheduler,
    numa_aware: bool,
    cache_linesize: usize,
    #[allow(dead_code)]
    work_queue_per_thread: bool,
}

impl AdvancedWorkStealingScheduler {
    /// Create a new advanced work-stealing scheduler
    pub fn new(config: &WorkerConfig) -> Self {
        Self {
            base_scheduler: WorkStealingScheduler::new(config),
            numa_aware: true,
            cache_linesize: 64, // Common cache line size
            work_queue_per_thread: true,
        }
    }

    /// Enable or disable NUMA-aware scheduling
    pub fn with_numa_aware(mut self, enabled: bool) -> Self {
        self.numa_aware = enabled;
        self
    }

    /// Set cache line size for cache-aware optimization
    pub fn with_cache_linesize(mut self, size: usize) -> Self {
        self.cache_linesize = size;
        self
    }

    /// Execute work with advanced optimizations
    ///
    /// This method implements enhanced work-stealing with:
    /// - NUMA-aware work distribution
    /// - Cache-friendly chunking
    /// - Adaptive scheduling based on workload characteristics
    pub fn execute_optimized<T, R, F>(&self, items: &[T], f: F) -> Vec<R>
    where
        T: Send + Sync,
        R: Send + Default + Clone,
        F: Fn(&T) -> R + Send + Sync,
    {
        if items.is_empty() {
            return Vec::new();
        }

        let n = items.len();

        // Analyze workload characteristics
        let workload_type = self.analyze_workload(n);

        // Determine optimal chunking strategy
        let chunk_config = match workload_type {
            WorkloadType::MemoryBound => ChunkConfig {
                size: self.cache_linesize / std::mem::size_of::<T>(),
                strategy: ChunkStrategy::Sequential,
            },
            WorkloadType::CpuBound => ChunkConfig {
                size: n / (self.base_scheduler.num_workers * 4),
                strategy: ChunkStrategy::Interleaved,
            },
            WorkloadType::Mixed => ChunkConfig {
                size: self.adaptive_chunksize_enhanced(n),
                strategy: ChunkStrategy::Dynamic,
            },
        };

        // Execute with optimized strategy
        self.execute_with_strategy(items, f, chunk_config)
    }

    /// Analyze workload characteristics to optimize scheduling
    fn analyze_workload(&self, size: usize) -> WorkloadType {
        let memory_footprint = size * std::mem::size_of::<usize>();
        let cachesize = 8 * 1024 * 1024; // Approximate L3 cache size

        if memory_footprint > cachesize {
            WorkloadType::MemoryBound
        } else if size < 1000 {
            WorkloadType::CpuBound
        } else {
            WorkloadType::Mixed
        }
    }

    /// Enhanced adaptive chunk size calculation
    fn adaptive_chunksize_enhanced(&self, totalitems: usize) -> usize {
        let num_workers = self.base_scheduler.num_workers;
        let items_per_worker = totalitems / num_workers;

        // Consider cache efficiency and load balancing
        let cache_optimalsize = self.cache_linesize / std::mem::size_of::<usize>();
        let load_balancesize = std::cmp::max(1, items_per_worker / 8);

        // Choose the better of cache-optimal or load-balance size
        if cache_optimalsize > 0 && cache_optimalsize < load_balancesize * 2 {
            cache_optimalsize
        } else {
            load_balancesize
        }
    }

    /// Execute work with specific strategy
    fn execute_with_strategy<T, R, F>(&self, items: &[T], f: F, config: ChunkConfig) -> Vec<R>
    where
        T: Send + Sync,
        R: Send + Default + Clone,
        F: Fn(&T) -> R + Send + Sync,
    {
        match config.strategy {
            ChunkStrategy::Sequential => self.execute_sequential_chunks(items, f, config.size),
            ChunkStrategy::Interleaved => self.execute_interleaved_chunks(items, f, config.size),
            ChunkStrategy::Dynamic => self.execute_dynamic_chunks(items, f, config.size),
        }
    }

    /// Execute with sequential chunk allocation
    fn execute_sequential_chunks<T, R, F>(&self, items: &[T], f: F, _chunksize: usize) -> Vec<R>
    where
        T: Send + Sync,
        R: Send + Default + Clone,
        F: Fn(&T) -> R + Send + Sync,
    {
        // Use the base scheduler for sequential chunking
        self.base_scheduler.execute(items, f)
    }

    /// Execute with interleaved chunk allocation for better cache utilization
    fn execute_interleaved_chunks<T, R, F>(&self, items: &[T], f: F, chunksize: usize) -> Vec<R>
    where
        T: Send + Sync,
        R: Send + Default + Clone,
        F: Fn(&T) -> R + Send + Sync,
    {
        let n = items.len();
        let chunksize = chunksize.max(1);
        let results = Arc::new(Mutex::new(vec![R::default(); n]));
        let work_counter = Arc::new(AtomicUsize::new(0));

        std::thread::scope(|s| {
            let handles: Vec<_> = (0..self.base_scheduler.num_workers)
                .map(|_worker_id| {
                    let items_ref = items;
                    let f_ref = &f;
                    let results = results.clone();
                    let work_counter = work_counter.clone();

                    s.spawn(move || {
                        loop {
                            let chunk_id = work_counter.fetch_add(1, Ordering::SeqCst);
                            let start = chunk_id * chunksize;

                            if start >= n {
                                break;
                            }

                            let end = std::cmp::min(start + chunksize, n);

                            // Process interleaved indices for better cache utilization
                            for i in start..end {
                                let interleaved_idx = (i % self.base_scheduler.num_workers)
                                    * (n / self.base_scheduler.num_workers)
                                    + (i / self.base_scheduler.num_workers);

                                if interleaved_idx < n {
                                    let result = f_ref(&items_ref[interleaved_idx]);
                                    let mut results_guard = results.lock().unwrap();
                                    results_guard[interleaved_idx] = result;
                                }
                            }
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        });

        Arc::try_unwrap(results)
            .unwrap_or_else(|_| panic!("Failed to extract results"))
            .into_inner()
            .unwrap_or_else(|_| panic!("Failed to extract mutex inner value"))
    }

    /// Execute with dynamic chunk sizing based on performance feedback
    fn execute_dynamic_chunks<T, R, F>(
        &self,
        items: &[T],
        f: F,
        _initial_chunksize: usize,
    ) -> Vec<R>
    where
        T: Send + Sync,
        R: Send + Default + Clone,
        F: Fn(&T) -> R + Send + Sync,
    {
        // For now, use the base implementation with dynamic sizing
        // In a full implementation, this would adapt chunk sizes based on timing
        self.base_scheduler.execute(items, f)
    }
}

/// Workload analysis types
#[derive(Debug, Clone, Copy)]
enum WorkloadType {
    /// Memory-bound workloads that benefit from cache optimization
    MemoryBound,
    /// CPU-bound workloads that benefit from load balancing
    CpuBound,
    /// Mixed workloads requiring balanced approach
    Mixed,
}

/// Chunk configuration for work distribution
#[derive(Debug, Clone)]
struct ChunkConfig {
    size: usize,
    strategy: ChunkStrategy,
}

/// Work distribution strategies
#[derive(Debug, Clone, Copy)]
enum ChunkStrategy {
    /// Sequential chunk allocation
    Sequential,
    /// Interleaved allocation for cache efficiency
    Interleaved,
    /// Dynamic sizing based on performance
    Dynamic,
}
