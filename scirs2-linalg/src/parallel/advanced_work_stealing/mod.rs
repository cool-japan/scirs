//! Advanced work-stealing algorithms with intelligent scheduling
//!
//! This module provides sophisticated work-stealing implementations with
//! priority-based scheduling, predictive load balancing, and adaptive chunking.

use super::*;
use crate::parallel::numa::NumaTopology;
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Work item with priority for the advanced scheduler
#[derive(Debug, Clone)]
pub struct PriorityWorkItem<T> {
    pub data: T,
    pub priority: u32,
    pub estimated_cost: Duration,
    pub dependencies: Vec<usize>,
    pub task_id: usize,
}

impl<T> PartialEq for PriorityWorkItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl<T> Eq for PriorityWorkItem<T> {}

impl<T> PartialOrd for PriorityWorkItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for PriorityWorkItem<T> {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Higher priority values get processed first
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.estimated_cost.cmp(&self.estimated_cost))
    }
}

/// Advanced work-stealing queue with priority and prediction capabilities
pub struct AdvancedWorkStealingQueue<T> {
    /// High priority work items (processed first)
    high_priority: Mutex<BinaryHeap<PriorityWorkItem<T>>>,
    /// Normal priority work items
    normal_priority: Mutex<VecDeque<PriorityWorkItem<T>>>,
    /// Low priority work items (processed when idle)
    low_priority: Mutex<VecDeque<PriorityWorkItem<T>>>,
    /// Completion time history for prediction
    completion_history: Mutex<VecDeque<(usize, Duration)>>,
    /// Number of active workers
    #[allow(dead_code)]
    active_workers: AtomicUsize,
    /// Queue statistics
    stats: Mutex<WorkStealingStats>,
}

/// Statistics for work-stealing performance analysis
#[derive(Debug, Clone, Default)]
pub struct WorkStealingStats {
    pub tasks_completed: usize,
    pub successful_steals: usize,
    pub failed_steals: usize,
    pub average_completion_time: Duration,
    pub load_imbalance_ratio: f64,
    pub prediction_accuracy: f64,
}

impl<T> Default for AdvancedWorkStealingQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AdvancedWorkStealingQueue<T> {
    /// Create a new advanced work-stealing queue
    pub fn new() -> Self {
        Self {
            high_priority: Mutex::new(BinaryHeap::new()),
            normal_priority: Mutex::new(VecDeque::new()),
            low_priority: Mutex::new(VecDeque::new()),
            completion_history: Mutex::new(VecDeque::with_capacity(1000)),
            active_workers: AtomicUsize::new(0),
            stats: Mutex::new(WorkStealingStats::default()),
        }
    }

    /// Add work item with automatic priority classification
    pub fn push(&self, item: T, estimatedcost: Duration, dependencies: Vec<usize>) -> usize {
        let task_id = self.generate_task_id();
        let priority = self.classify_priority(&estimatedcost, &dependencies);

        let work_item = PriorityWorkItem {
            data: item,
            priority,
            estimated_cost: estimatedcost,
            dependencies,
            task_id,
        };

        match priority {
            0..=33 => {
                self.low_priority.lock().unwrap().push_back(work_item);
            }
            34..=66 => {
                self.normal_priority.lock().unwrap().push_back(work_item);
            }
            _ => {
                self.high_priority.lock().unwrap().push(work_item);
            }
        }

        task_id
    }

    /// Try to pop work item using intelligent scheduling
    pub fn try_pop(&self) -> Option<PriorityWorkItem<T>> {
        // First try high priority tasks
        if let Ok(mut high_queue) = self.high_priority.try_lock() {
            if let Some(item) = high_queue.pop() {
                return Some(item);
            }
        }

        // Then try normal priority tasks
        if let Ok(mut normal_queue) = self.normal_priority.try_lock() {
            if let Some(item) = normal_queue.pop_front() {
                return Some(item);
            }
        }

        // Finally try low priority tasks if we're idle
        if let Ok(mut low_queue) = self.low_priority.try_lock() {
            if let Some(item) = low_queue.pop_front() {
                return Some(item);
            }
        }

        None
    }

    /// Attempt to steal work from other queues (for work-stealing)
    pub fn try_steal(&self) -> Option<PriorityWorkItem<T>> {
        // Record steal attempt
        if let Ok(mut stats) = self.stats.try_lock() {
            // Try stealing from normal priority first (better balance)
            if let Ok(mut normal_queue) = self.normal_priority.try_lock() {
                if let Some(item) = normal_queue.pop_back() {
                    stats.successful_steals += 1;
                    return Some(item);
                }
            }

            // Then try low priority
            if let Ok(mut low_queue) = self.low_priority.try_lock() {
                if let Some(item) = low_queue.pop_back() {
                    stats.successful_steals += 1;
                    return Some(item);
                }
            }

            stats.failed_steals += 1;
        }

        None
    }

    /// Classify task priority based on cost and dependencies
    fn classify_priority(&self, estimatedcost: &Duration, dependencies: &[usize]) -> u32 {
        let base_priority: u32 = if estimatedcost.as_millis() > 100 {
            80 // High _cost tasks get high priority
        } else if estimatedcost.as_millis() > 10 {
            50 // Medium _cost tasks
        } else {
            20 // Low _cost tasks
        };

        // Adjust for dependencies (more dependencies = lower priority)
        let dependency_penalty = (dependencies.len() as u32 * 5).min(30);
        base_priority.saturating_sub(dependency_penalty)
    }

    /// Generate unique task ID
    fn generate_task_id(&self) -> usize {
        static TASK_COUNTER: AtomicUsize = AtomicUsize::new(0);
        TASK_COUNTER.fetch_add(1, Ordering::Relaxed)
    }

    /// Record task completion for performance prediction
    pub fn record_completion(&self, task_id: usize, actualduration: Duration) {
        if let Ok(mut history) = self.completion_history.try_lock() {
            history.push_back((task_id, actualduration));

            // Keep history bounded
            if history.len() > 1000 {
                history.pop_front();
            }
        }
    }

    /// Get current queue statistics
    pub fn get_stats(&self) -> WorkStealingStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get estimated remaining work
    pub fn estimated_remaining_work(&self) -> Duration {
        let high_count = self.high_priority.lock().unwrap().len();
        let normal_count = self.normal_priority.lock().unwrap().len();
        let low_count = self.low_priority.lock().unwrap().len();

        // Rough estimates based on priority
        Duration::from_millis((high_count * 100 + normal_count * 50 + low_count * 10) as u64)
    }
}

/// Matrix-specific adaptive chunking strategy
pub struct MatrixAdaptiveChunking {
    /// Cache line size for optimal memory access
    #[allow(dead_code)]
    cache_linesize: usize,
    /// NUMA node information
    #[allow(dead_code)]
    numa_info: Option<NumaTopology>,
    /// Historical performance data
    performance_history: Mutex<VecDeque<ChunkingPerformance>>,
}

#[derive(Debug, Clone)]
struct ChunkingPerformance {
    chunksize: usize,
    matrix_dimensions: (usize, usize),
    throughput: f64, // operations per second
    #[allow(dead_code)]
    cache_misses: usize,
    #[allow(dead_code)]
    timestamp: Instant,
}

impl Default for MatrixAdaptiveChunking {
    fn default() -> Self {
        Self::new()
    }
}

impl MatrixAdaptiveChunking {
    /// Create new adaptive chunking strategy
    pub fn new() -> Self {
        Self {
            cache_linesize: 64, // typical cache line size
            numa_info: Some(NumaTopology::detect()),
            performance_history: Mutex::new(VecDeque::with_capacity(100)),
        }
    }

    /// Calculate optimal chunk size for matrix operation
    pub fn optimal_chunksize(
        &self,
        matrix_dims: (usize, usize),
        operation_type: MatrixOperation,
    ) -> usize {
        let (rows, cols) = matrix_dims;

        // Base chunk size calculation
        let base_chunk = match operation_type {
            MatrixOperation::MatrixMultiply => {
                // For matrix multiplication, consider cache blocking
                let l1_cachesize = 32 * 1024; // 32KB typical L1 cache
                let elementsize = std::mem::size_of::<f64>();
                let elements_per_cache = l1_cachesize / elementsize;

                // Aim for square blocks that fit in cache
                ((elements_per_cache as f64).sqrt() as usize).clamp(32, 512)
            }
            MatrixOperation::ElementWise => {
                // For element-wise operations, optimize for memory bandwidth
                let memory_bandwidth = self.estimate_memory_bandwidth();
                (memory_bandwidth / 8).clamp(64, 1024) // 8 bytes per f64
            }
            MatrixOperation::Reduction => {
                // For reductions, use smaller chunks to balance load
                let num_cores = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);
                rows.max(cols) / (num_cores * 4)
            }
            MatrixOperation::Decomposition => {
                // For decompositions, larger chunks for better locality
                let num_cores = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);
                rows.min(cols) / num_cores.max(1)
            }
        };

        // Adjust based on historical performance
        self.adjust_for_history(base_chunk, matrix_dims, operation_type)
    }

    /// Estimate memory bandwidth (simplified)
    fn estimate_memory_bandwidth(&self) -> usize {
        // This is a simplified estimation - in practice, this would
        // involve actual benchmarking
        match std::env::var("SCIRS_MEMORY_BANDWIDTH") {
            Ok(val) => val.parse().unwrap_or(100_000), // MB/s
            Err(_) => 100_000,                         // Default assumption: 100 GB/s
        }
    }

    /// Adjust chunk size based on historical performance
    fn adjust_for_history(
        &self,
        base_chunk: usize,
        matrix_dims: (usize, usize),
        _operation_type: MatrixOperation,
    ) -> usize {
        if let Ok(history) = self.performance_history.lock() {
            // Find similar operations in history
            let similar_ops: Vec<_> = history
                .iter()
                .filter(|perf| {
                    let (h_rows, h_cols) = perf.matrix_dimensions;
                    // Consider operations on similar-sized matrices
                    (h_rows as f64 / matrix_dims.0 as f64).abs() < 2.0
                        && (h_cols as f64 / matrix_dims.1 as f64).abs() < 2.0
                })
                .collect();

            if !similar_ops.is_empty() {
                // Find the _chunk size with best throughput
                let best_perf = similar_ops
                    .iter()
                    .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap());

                if let Some(best) = best_perf {
                    // Interpolate between base _chunk and historically best
                    let weight = 0.7; // Favor historical data
                    return (base_chunk as f64 * (1.0 - weight) + best.chunksize as f64 * weight)
                        as usize;
                }
            }
        }

        base_chunk
    }

    /// Record performance data for future optimization
    pub fn record_performance(
        &self,
        chunksize: usize,
        matrix_dims: (usize, usize),
        throughput: f64,
    ) {
        if let Ok(mut history) = self.performance_history.lock() {
            let perf = ChunkingPerformance {
                chunksize,
                matrix_dimensions: matrix_dims,
                throughput,
                cache_misses: 0, // Would be measured in practice
                timestamp: Instant::now(),
            };

            history.push_back(perf);

            // Keep history bounded
            if history.len() > 100 {
                history.pop_front();
            }
        }
    }
}

/// Types of matrix operations for chunking optimization
#[derive(Debug, Clone, Copy)]
pub enum MatrixOperation {
    MatrixMultiply,
    ElementWise,
    Reduction,
    Decomposition,
}

/// Predictive load balancer using machine learning-like predictions
pub struct PredictiveLoadBalancer {
    /// Historical execution times for different task types
    execution_history: Mutex<std::collections::HashMap<String, Vec<Duration>>>,
    /// Current load per worker
    worker_loads: Mutex<Vec<f64>>,
    /// Prediction model weights (simplified linear model)
    model_weights: Mutex<Vec<f64>>,
}

impl PredictiveLoadBalancer {
    /// Create new predictive load balancer
    pub fn new(_numworkers: usize) -> Self {
        Self {
            execution_history: Mutex::new(std::collections::HashMap::new()),
            worker_loads: Mutex::new(vec![0.0; _numworkers]),
            model_weights: Mutex::new(vec![1.0; 4]), // Simple 4-feature model
        }
    }

    /// Predict execution time for a task
    pub fn predict_execution_time(&self, taskfeatures: &TaskFeatures) -> Duration {
        let weights = self.model_weights.lock().unwrap();

        // Extract _features
        let _features = [
            taskfeatures.datasize as f64,
            taskfeatures.complexity_factor,
            taskfeatures.memory_access_pattern as f64,
            taskfeatures.arithmetic_intensity,
        ];

        // Simple linear prediction
        let predicted_ms = _features
            .iter()
            .zip(weights.iter())
            .map(|(f, w)| f * w)
            .sum::<f64>()
            .max(1.0); // Minimum 1ms

        Duration::from_millis(predicted_ms as u64)
    }

    /// Assign task to optimal worker based on predicted load
    pub fn assign_task(&self, taskfeatures: &TaskFeatures) -> usize {
        let predicted_time = self.predict_execution_time(taskfeatures);
        let mut loads = self.worker_loads.lock().unwrap();

        // Find worker with minimum predicted finish time
        let (best_worker, min_load) = loads
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // Update predicted load
        loads[best_worker] += predicted_time.as_secs_f64();

        best_worker
    }

    /// Update model with actual execution time
    pub fn update_model(&self, task_features: &TaskFeatures, actualtime: Duration) {
        // Record execution _time
        let task_type = format!(
            "{}_{}",
            task_features.datasize, task_features.complexity_factor as u32
        );

        if let Ok(mut history) = self.execution_history.lock() {
            history
                .entry(task_type)
                .or_insert_with(Vec::new)
                .push(actualtime);
        }

        // Simple model update (in practice, would use more sophisticated ML)
        self.update_weights(task_features, actualtime);
    }

    /// Update worker load (when task completes)
    pub fn update_worker_load(&self, worker_id: usize, completedtime: Duration) {
        if let Ok(mut loads) = self.worker_loads.lock() {
            if worker_id < loads.len() {
                loads[worker_id] -= completedtime.as_secs_f64();
                loads[worker_id] = loads[worker_id].max(0.0);
            }
        }
    }

    /// Simple weight update using gradient descent-like approach
    fn update_weights(&self, task_features: &TaskFeatures, actualtime: Duration) {
        let predicted_time = self.predict_execution_time(task_features);
        let error = actualtime.as_secs_f64() - predicted_time.as_secs_f64();

        if let Ok(mut weights) = self.model_weights.lock() {
            let learning_rate = 0.001;
            let _features = [
                task_features.datasize as f64,
                task_features.complexity_factor,
                task_features.memory_access_pattern as f64,
                task_features.arithmetic_intensity,
            ];

            // Update weights based on error
            for (weight, feature) in weights.iter_mut().zip(_features.iter()) {
                *weight += learning_rate * error * feature;
            }
        }
    }
}

/// Features describing a computational task for prediction
#[derive(Debug, Clone)]
pub struct TaskFeatures {
    pub datasize: usize,
    pub complexity_factor: f64,
    pub memory_access_pattern: u32, // 0=sequential, 1=random, 2=strided
    pub arithmetic_intensity: f64,  // operations per byte
}

impl TaskFeatures {
    /// Create task features for matrix operation
    pub fn formatrix_operation(matrix_dims: (usize, usize), operation: MatrixOperation) -> Self {
        let (rows, cols) = matrix_dims;
        let datasize = rows * cols;

        let (complexity_factor, memory_pattern, arithmetic_intensity) = match operation {
            MatrixOperation::MatrixMultiply => {
                (rows as f64 * cols as f64 * 2.0, 1, 2.0) // O(n²) complexity, random access, 2 ops per element
            }
            MatrixOperation::ElementWise => {
                (datasize as f64, 0, 1.0) // O(n) complexity, sequential access, 1 op per element
            }
            MatrixOperation::Reduction => {
                (datasize as f64, 0, 1.0) // O(n) complexity, sequential access
            }
            MatrixOperation::Decomposition => {
                (datasize as f64 * 1.5, 2, 3.0) // Higher complexity, strided access
            }
        };

        Self {
            datasize,
            complexity_factor,
            memory_access_pattern: memory_pattern,
            arithmetic_intensity,
        }
    }
}
