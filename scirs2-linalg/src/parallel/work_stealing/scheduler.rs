//! Work-stealing scheduler implementation with dynamic load balancing
//!
//! This module provides the WorkStealingScheduler struct and its implementation,
//! handling work distribution, stealing strategies, and execution management.

use crate::error::{LinalgError, LinalgResult};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use super::core::{
    LoadBalancingParams, MatrixOperationType, SchedulerStats, StealingStrategy, WorkComplexity,
    WorkItem, WorkloadCharacteristics,
};
use super::queue::WorkQueue;

/// Simple parallel map utility function using rayon
#[allow(dead_code)]
fn parallel_map<T, U, F>(items: &[T], func: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
{
    #[allow(unused_imports)]
    use scirs2_core::parallel_ops::*;

    // Use rayon's parallel iterator if available, otherwise sequential
    #[cfg(feature = "parallel")]
    {
        items.par_iter().map(func).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        items.iter().map(func).collect()
    }
}

/// Work-stealing scheduler with dynamic load balancing
pub struct WorkStealingScheduler<T: Clone>
where
    T: Send + 'static,
{
    /// Worker queues (one per thread)
    worker_queues: Vec<Arc<Mutex<WorkQueue<T>>>>,
    /// Number of worker threads
    num_workers: usize,
    /// Condition variable for worker synchronization
    worker_sync: Arc<(Mutex<bool>, Condvar)>,
    /// Statistics collection
    stats: Arc<Mutex<SchedulerStats>>,
    /// Work-stealing strategy
    stealing_strategy: StealingStrategy,
    /// Adaptive load balancing parameters
    load_balancing_params: LoadBalancingParams,
}

impl<T: Send + 'static + Clone> WorkStealingScheduler<T> {
    /// Create a new work-stealing scheduler
    pub fn new(_numworkers: usize) -> Self {
        Self::with_strategy(
            _numworkers,
            StealingStrategy::default(),
            LoadBalancingParams::default(),
        )
    }

    /// Create a new work-stealing scheduler with custom strategy
    pub fn with_strategy(
        num_workers: usize,
        strategy: StealingStrategy,
        params: LoadBalancingParams,
    ) -> Self {
        let worker_queues = (0..num_workers)
            .map(|_| Arc::new(Mutex::new(WorkQueue::default())))
            .collect();

        Self {
            worker_queues,
            num_workers,
            worker_sync: Arc::new((Mutex::new(false), Condvar::new())),
            stats: Arc::new(Mutex::new(SchedulerStats::default())),
            stealing_strategy: strategy,
            load_balancing_params: params,
        }
    }

    /// Create optimized scheduler for specific matrix operations
    pub fn formatrix_operation(
        num_workers: usize,
        operation_type: MatrixOperationType,
        matrixsize: (usize, usize),
    ) -> Self {
        let (strategy, params) = match operation_type {
            MatrixOperationType::MatrixVectorMultiplication => {
                // Matrix-vector operations benefit from locality-aware stealing
                (
                    StealingStrategy::LocalityAware,
                    LoadBalancingParams {
                        steal_threshold: 4,
                        max_steal_attempts: 2,
                        chunksize: matrixsize.0 / num_workers,
                        priority_scheduling: false,
                        ..LoadBalancingParams::default()
                    },
                )
            }
            MatrixOperationType::MatrixMatrixMultiplication => {
                // Matrix-matrix operations benefit from adaptive stealing
                (
                    StealingStrategy::Adaptive,
                    LoadBalancingParams {
                        steal_threshold: 2,
                        max_steal_attempts: 4,
                        chunksize: (matrixsize.0 * matrixsize.1) / (num_workers * 8),
                        priority_scheduling: true,
                        ..LoadBalancingParams::default()
                    },
                )
            }
            MatrixOperationType::Decomposition => {
                // Decompositions have irregular workloads, use adaptive approach
                (
                    StealingStrategy::Adaptive,
                    LoadBalancingParams {
                        steal_threshold: 1,
                        max_steal_attempts: 6,
                        chunksize: matrixsize.0 / (num_workers * 2),
                        priority_scheduling: true,
                        backoff_base: Duration::from_micros(5),
                        max_backoff: Duration::from_millis(2),
                    },
                )
            }
            MatrixOperationType::EigenComputation => {
                // Eigenvalue computations have sequential dependencies
                (
                    StealingStrategy::MostLoaded,
                    LoadBalancingParams {
                        steal_threshold: 8,
                        max_steal_attempts: 2,
                        chunksize: matrixsize.0 / num_workers,
                        priority_scheduling: false,
                        ..LoadBalancingParams::default()
                    },
                )
            }
            MatrixOperationType::IterativeSolver => {
                // Iterative solvers need balanced load distribution
                (
                    StealingStrategy::RoundRobin,
                    LoadBalancingParams {
                        steal_threshold: 3,
                        max_steal_attempts: 3,
                        chunksize: matrixsize.0 / (num_workers * 4),
                        priority_scheduling: false,
                        ..LoadBalancingParams::default()
                    },
                )
            }
        };

        Self::with_strategy(num_workers, strategy, params)
    }

    /// Submit work items to the scheduler
    pub fn submit_work(&self, items: Vec<WorkItem<T>>) -> LinalgResult<()> {
        if items.is_empty() {
            return Ok(());
        }

        // Advanced work distribution based on strategy
        self.distribute_work_optimally(items)?;

        // Wake up all workers
        let (lock, cvar) = &*self.worker_sync;
        if let Ok(mut started) = lock.lock() {
            *started = true;
            cvar.notify_all();
        }

        Ok(())
    }

    /// Optimally distribute work items based on current load and strategy
    fn distribute_work_optimally(&self, items: Vec<WorkItem<T>>) -> LinalgResult<()> {
        match self.stealing_strategy {
            StealingStrategy::Random => {
                // Random distribution
                for (i, item) in items.into_iter().enumerate() {
                    let mut hasher = DefaultHasher::new();
                    i.hash(&mut hasher);
                    let worker_id = (hasher.finish() as usize) % self.num_workers;

                    if let Ok(mut queue) = self.worker_queues[worker_id].lock() {
                        queue.push_front(item);
                    }
                }
            }
            StealingStrategy::RoundRobin => {
                // Round-robin distribution (default)
                for (i, item) in items.into_iter().enumerate() {
                    let worker_id = i % self.num_workers;
                    if let Ok(mut queue) = self.worker_queues[worker_id].lock() {
                        queue.push_front(item);
                    }
                }
            }
            StealingStrategy::MostLoaded => {
                // Distribute to least loaded workers first
                let load_info = self.get_worker_loads();
                let mut sorted_workers: Vec<usize> = (0..self.num_workers).collect();
                sorted_workers.sort_by_key(|&i| load_info[i]);

                for (i, item) in items.into_iter().enumerate() {
                    let worker_id = sorted_workers[i % self.num_workers];
                    if let Ok(mut queue) = self.worker_queues[worker_id].lock() {
                        queue.push_front(item);
                    }
                }
            }
            StealingStrategy::LocalityAware => {
                // Try to maintain work locality (simplified implementation)
                let chunksize = self.load_balancing_params.chunksize;
                for chunk in items.chunks(chunksize) {
                    let worker_id = (chunk.as_ptr() as usize / chunksize) % self.num_workers;
                    if let Ok(mut queue) = self.worker_queues[worker_id].lock() {
                        for item in chunk {
                            queue.push_front(item.clone());
                        }
                    }
                }
            }
            StealingStrategy::Adaptive => {
                // Use adaptive strategy based on historical performance
                self.adaptive_work_distribution(items)?;
            }
        }

        Ok(())
    }

    /// Get current load (number of work items) for each worker
    fn get_worker_loads(&self) -> Vec<usize> {
        let mut loads = Vec::with_capacity(self.num_workers);

        for queue in &self.worker_queues {
            if let Ok(queue) = queue.lock() {
                loads.push(queue.items.len());
            } else {
                loads.push(0);
            }
        }

        loads
    }

    /// Adaptive work distribution based on historical performance
    fn adaptive_work_distribution(&self, items: Vec<WorkItem<T>>) -> LinalgResult<()> {
        // Get current worker utilization
        let loads = self.get_worker_loads();
        let total_load: usize = loads.iter().sum();

        if total_load == 0 {
            // No existing load, use round-robin
            for (i, item) in items.into_iter().enumerate() {
                let worker_id = i % self.num_workers;
                if let Ok(mut queue) = self.worker_queues[worker_id].lock() {
                    queue.push_front(item);
                }
            }
        } else {
            // Distribute inversely proportional to current load
            let mut worker_weights = Vec::with_capacity(self.num_workers);
            let max_load = loads.iter().max().unwrap_or(&1);

            for &load in &loads {
                // Higher load = lower weight
                worker_weights.push(max_load + 1 - load);
            }

            let total_weight: usize = worker_weights.iter().sum();
            let mut cumulative_weights = Vec::with_capacity(self.num_workers);
            let mut sum = 0;
            for &weight in &worker_weights {
                sum += weight;
                cumulative_weights.push(sum);
            }

            // Distribute items based on weights
            let items_len = items.len();
            for (i, item) in items.into_iter().enumerate() {
                let target = (i * total_weight / items_len).min(total_weight - 1);
                let worker_id = cumulative_weights
                    .iter()
                    .position(|&w| w > target)
                    .unwrap_or(self.num_workers - 1);

                if let Ok(mut queue) = self.worker_queues[worker_id].lock() {
                    queue.push_front(item);
                }
            }
        }

        Ok(())
    }

    /// Advanced work stealing with different victim selection strategies
    #[allow(dead_code)]
    fn steal_work(&self, thiefid: usize) -> Option<WorkItem<T>> {
        let mut attempts = 0;
        let max_attempts = self.load_balancing_params.max_steal_attempts;

        while attempts < max_attempts {
            let victim_id = self.select_victim(thiefid, attempts);

            if let Some(victim_id) = victim_id {
                if let Ok(mut victim_queue) = self.worker_queues[victim_id].try_lock() {
                    if let Some(stolen_item) = victim_queue.steal_back() {
                        // Update statistics
                        if let Ok(mut stats) = self.stats.lock() {
                            stats.successful_steals += 1;
                        }
                        return Some(stolen_item);
                    }
                }
            }

            attempts += 1;

            // Exponential backoff
            let backoff_duration =
                self.load_balancing_params.backoff_base * 2_u32.pow(attempts.min(10) as u32);
            let capped_backoff = backoff_duration.min(self.load_balancing_params.max_backoff);

            thread::sleep(capped_backoff);
        }

        // Update failed steal statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.failed_steals += max_attempts;
        }

        None
    }

    /// Select victim for work stealing based on strategy
    #[allow(dead_code)]
    fn select_victim(&self, thiefid: usize, attempt: usize) -> Option<usize> {
        match self.stealing_strategy {
            StealingStrategy::Random => {
                let mut hasher = DefaultHasher::new();
                (thiefid + attempt).hash(&mut hasher);
                let victim = (hasher.finish() as usize) % self.num_workers;

                if victim != thiefid {
                    Some(victim)
                } else {
                    Some((victim + 1) % self.num_workers)
                }
            }
            StealingStrategy::RoundRobin => Some((thiefid + attempt + 1) % self.num_workers),
            StealingStrategy::MostLoaded => {
                // Target the worker with the most work
                let loads = self.get_worker_loads();
                let max_load_worker = loads
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != thiefid)
                    .max_by_key(|(_, &load)| load)
                    .map(|(i, _)| i);

                max_load_worker
            }
            StealingStrategy::LocalityAware => {
                // Try to steal from nearby workers first
                let distance = (attempt % (self.num_workers / 2)) + 1;
                Some((thiefid + distance) % self.num_workers)
            }
            StealingStrategy::Adaptive => {
                // Combine strategies based on historical success rates
                if attempt < 2 {
                    // First try most loaded
                    self.select_victim_most_loaded(thiefid)
                } else {
                    // Then try random (fix infinite recursion)
                    let mut hasher = DefaultHasher::new();
                    (thiefid + attempt).hash(&mut hasher);
                    let victim = (hasher.finish() as usize) % self.num_workers;

                    if victim != thiefid {
                        Some(victim)
                    } else {
                        Some((victim + 1) % self.num_workers)
                    }
                }
            }
        }
    }

    /// Helper for most-loaded victim selection
    #[allow(dead_code)]
    fn select_victim_most_loaded(&self, thiefid: usize) -> Option<usize> {
        let loads = self.get_worker_loads();
        loads
            .iter()
            .enumerate()
            .filter(|(i_, _)| *i_ != thiefid)
            .max_by_key(|(_, &load)| load)
            .map(|(i_, _)| i_)
    }

    /// Execute all work items using the work-stealing scheduler
    pub fn execute<F, R>(&self, workfn: F) -> LinalgResult<Vec<R>>
    where
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + Clone + 'static,
        T: Send + 'static,
    {
        let work_fn = Arc::new(workfn);
        let results = Arc::new(Mutex::new(Vec::new()));

        // Start worker threads
        let mut handles = Vec::new();
        for worker_id in 0..self.num_workers {
            let queue = Arc::clone(&self.worker_queues[worker_id]);
            let all_queues = self.worker_queues.clone();
            let work_fn = Arc::clone(&work_fn);
            let results = Arc::clone(&results);
            let stats = Arc::clone(&self.stats);
            let sync = Arc::clone(&self.worker_sync);

            let handle = thread::spawn(move || {
                Self::worker_loop(worker_id, queue, all_queues, work_fn, results, stats, sync);
            });
            handles.push(handle);
        }

        // Wait for all workers to complete
        for handle in handles {
            handle.join().map_err(|_| {
                crate::error::LinalgError::ComputationError("Worker thread panicked".to_string())
            })?;
        }

        // Extract results
        let results = results.lock().unwrap();
        Ok((*results).clone())
    }

    /// Worker thread main loop
    fn worker_loop<F, R>(
        worker_id: usize,
        my_queue: Arc<Mutex<WorkQueue<T>>>,
        all_queues: Vec<Arc<Mutex<WorkQueue<T>>>>,
        work_fn: Arc<F>,
        results: Arc<Mutex<Vec<R>>>,
        stats: Arc<Mutex<SchedulerStats>>,
        sync: Arc<(Mutex<bool>, Condvar)>,
    ) where
        F: Fn(T) -> R + Send + Sync,
        R: Send,
    {
        let (lock, cvar) = &*sync;

        // Wait for work to be available
        let _started = cvar
            .wait_while(lock.lock().unwrap(), |&mut started| !started)
            .unwrap();

        loop {
            let work_item = {
                // Try to get work from own _queue first
                if let Ok(mut queue) = my_queue.lock() {
                    queue.pop_front()
                } else {
                    None
                }
            };

            let work_item = match work_item {
                Some(item) => item,
                None => {
                    // Try to steal work from other workers
                    match Self::steal_work_global(worker_id, &all_queues, &stats) {
                        Some(item) => item,
                        None => {
                            // No work available, check if all _queues are empty
                            if Self::all_queues_empty(&all_queues) {
                                break;
                            }
                            // Brief pause before trying again
                            thread::sleep(Duration::from_micros(10));
                            continue;
                        }
                    }
                }
            };

            // Execute the work item
            let start_time = Instant::now();
            let result = work_fn(work_item.payload);
            let execution_time = start_time.elapsed();

            // Update timing statistics
            if let Ok(mut queue) = my_queue.lock() {
                queue.update_timing(execution_time);
            }

            // Store the result
            if let Ok(mut results) = results.lock() {
                results.push(result);
            }

            // Update global statistics
            if let Ok(mut stats) = stats.lock() {
                stats.total_items += 1;
                stats.total_execution_time += execution_time;
            }
        }
    }

    /// Attempt to steal work from other workers
    fn steal_work_global(
        worker_id: usize,
        all_queues: &[Arc<Mutex<WorkQueue<T>>>],
        stats: &Arc<Mutex<SchedulerStats>>,
    ) -> Option<WorkItem<T>> {
        // Try to steal from the most loaded worker
        let mut best_target = None;
        let mut max_load = Duration::ZERO;

        for (i, queue) in all_queues.iter().enumerate() {
            if i == worker_id {
                continue; // Don't steal from ourselves
            }

            if let Ok(queue) = queue.lock() {
                let load = queue.estimated_load();
                if load > max_load {
                    max_load = load;
                    best_target = Some(i);
                }
            }
        }

        if let Some(target_id) = best_target {
            if let Ok(mut target_queue) = all_queues[target_id].lock() {
                if let Some(stolen_item) = target_queue.steal_back() {
                    // Update steal statistics
                    if let Ok(mut stats) = stats.lock() {
                        stats.successful_steals += 1;
                    }
                    return Some(stolen_item);
                }
            }
        }

        // Update failed steal statistics
        if let Ok(mut stats) = stats.lock() {
            stats.failed_steals += 1;
        }

        None
    }

    /// Check if all worker queues are empty
    fn all_queues_empty(queues: &[Arc<Mutex<WorkQueue<T>>>]) -> bool {
        queues.iter().all(|queue| {
            if let Ok(queue) = queue.lock() {
                queue.items.is_empty()
            } else {
                true // Assume empty if we can't lock
            }
        })
    }

    /// Get current scheduler statistics
    pub fn get_stats(&self) -> SchedulerStats {
        if let Ok(stats) = self.stats.lock() {
            let mut stats = stats.clone();
            stats.load_balance_efficiency = self.calculate_load_balance_efficiency();
            stats.time_variance = self.calculate_time_variance();
            stats
        } else {
            SchedulerStats::default()
        }
    }

    /// Adaptive performance monitoring and load balancing optimization
    pub fn optimize_for_workload(
        &self,
        workload_characteristics: WorkloadCharacteristics,
    ) -> LinalgResult<()> {
        let mut stats = self.stats.lock().map_err(|_| {
            crate::error::LinalgError::ComputationError("Failed to acquire stats lock".to_string())
        })?;

        // Analyze current performance metrics
        let load_imbalance = self.calculate_load_imbalance();
        let steal_success_rate = if stats.successful_steals + stats.failed_steals > 0 {
            stats.successful_steals as f64 / (stats.successful_steals + stats.failed_steals) as f64
        } else {
            0.5
        };

        // Adapt strategy based on workload _characteristics and performance
        let _suggested_strategy =
            match (workload_characteristics, load_imbalance, steal_success_rate) {
                (WorkloadCharacteristics::HighVariance, imbalance_, _) if imbalance_ > 0.3 => {
                    StealingStrategy::Adaptive
                }
                (WorkloadCharacteristics::LowVariance, _, success_rate) if success_rate < 0.2 => {
                    StealingStrategy::RoundRobin
                }
                (WorkloadCharacteristics::MemoryBound, _, _) => StealingStrategy::LocalityAware,
                (WorkloadCharacteristics::ComputeBound, _, success_rate) if success_rate > 0.8 => {
                    StealingStrategy::MostLoaded
                }
                _ => StealingStrategy::Adaptive,
            };

        // Update performance recommendations
        stats.load_balance_efficiency = 1.0 - load_imbalance;

        Ok(())
    }

    /// Calculate load imbalance across workers
    fn calculate_load_imbalance(&self) -> f64 {
        let loads = self.get_worker_loads();
        if loads.is_empty() {
            return 0.0;
        }

        let total_load: usize = loads.iter().sum();
        let avg_load = total_load as f64 / loads.len() as f64;

        if avg_load == 0.0 {
            return 0.0;
        }

        let variance: f64 = loads
            .iter()
            .map(|&load| (load as f64 - avg_load).powi(2))
            .sum::<f64>()
            / loads.len() as f64;

        let std_dev = variance.sqrt();
        std_dev / avg_load // Coefficient of variation
    }

    /// Dynamic chunk size adjustment based on performance history
    pub fn adaptive_chunk_sizing(
        &self,
        base_worksize: usize,
        worker_efficiency: &[f64],
    ) -> Vec<usize> {
        let total_efficiency: f64 = worker_efficiency.iter().sum();
        let avg_efficiency = total_efficiency / worker_efficiency.len() as f64;

        // Adjust chunk sizes based on relative worker _efficiency
        worker_efficiency
            .iter()
            .map(|&_efficiency| {
                let efficiency_ratio = _efficiency / avg_efficiency;
                let chunksize = (base_worksize as f64 * efficiency_ratio) as usize;
                chunksize.max(1).min(base_worksize) // Clamp to reasonable bounds
            })
            .collect()
    }

    /// Advanced workload prediction based on execution history
    pub fn predict_execution_time(&self, workcomplexity: WorkComplexity) -> Duration {
        let stats = self.stats.lock().unwrap();

        let base_time = if stats.total_items > 0 {
            stats.total_execution_time / stats.total_items as u32
        } else {
            Duration::from_millis(1)
        };

        match workcomplexity {
            WorkComplexity::Constant => base_time,
            WorkComplexity::Linear => base_time * 2,
            WorkComplexity::Quadratic => base_time * 4,
            WorkComplexity::Variable => {
                // Use historical variance to estimate
                Duration::from_nanos(
                    (base_time.as_nanos() as f64 * (1.0 + stats.time_variance)).max(1.0) as u64,
                )
            }
        }
    }

    /// Calculate load balancing efficiency
    fn calculate_load_balance_efficiency(&self) -> f64 {
        let worker_times: Vec<Duration> = self
            .worker_queues
            .iter()
            .filter_map(|queue| queue.lock().ok().map(|q| q.total_time))
            .collect();

        if worker_times.is_empty() {
            return 1.0;
        }

        let max_time = worker_times.iter().max().unwrap().as_nanos() as f64;
        let min_time = worker_times.iter().min().unwrap().as_nanos() as f64;

        if max_time == 0.0 {
            1.0
        } else {
            min_time / max_time
        }
    }

    /// Calculate time variance across workers
    fn calculate_time_variance(&self) -> f64 {
        let worker_times: Vec<f64> = self
            .worker_queues
            .iter()
            .filter_map(|queue| queue.lock().ok().map(|q| q.total_time.as_nanos() as f64))
            .collect();

        if worker_times.len() < 2 {
            return 0.0;
        }

        let mean = worker_times.iter().sum::<f64>() / worker_times.len() as f64;
        let variance = worker_times
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / worker_times.len() as f64;

        variance.sqrt()
    }
}
