//! Adaptive chunking strategies for memory-efficient operations.
//!
//! This module provides algorithms that dynamically determine optimal
//! chunk sizes based on workload characteristics, memory constraints,
//! and data distribution patterns. Adaptive chunking can significantly
//! improve performance by balancing memory usage with processing efficiency.

use super::chunked::ChunkingStrategy;
use super::memmap::MemoryMappedArray;
use super::memmap_chunks::MemoryMappedChunks;
use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
// use ndarray::Dimension; // Currently unused
use std::time::Duration;

/// Alpha 6: Workload types for optimized chunking strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadType {
    /// Memory-intensive workloads that need smaller chunks
    MemoryIntensive,
    /// Compute-intensive workloads that can benefit from larger chunks
    ComputeIntensive,
    /// I/O-intensive workloads that need optimized for throughput
    IoIntensive,
    /// Balanced workloads with mixed requirements
    Balanced,
}

/// Parameters for configuring adaptive chunking behavior.
#[derive(Debug, Clone)]
pub struct AdaptiveChunkingParams {
    /// Target memory usage per chunk (in bytes)
    pub target_memory_usage: usize,

    /// Maximum chunk size (in elements)
    pub max_chunk_size: usize,

    /// Minimum chunk size (in elements)
    pub min_chunk_size: usize,

    /// Target processing time per chunk (for time-based adaptation)
    pub target_chunk_duration: Option<Duration>,

    /// Whether to consider data distribution (can be expensive to calculate)
    pub consider_distribution: bool,

    /// Whether to adjust for parallel processing
    pub optimize_for_parallel: bool,

    /// Number of worker threads to optimize for (when parallel is enabled)
    pub num_workers: Option<usize>,
}

impl Default for AdaptiveChunkingParams {
    fn default() -> Self {
        // Alpha 6: Enhanced defaults based on system detection
        let available_memory = Self::detect_available_memory();
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Target 1/8 of available memory per chunk, with reasonable bounds
        let target_memory = if let Some(mem) = available_memory {
            (mem / 8).clamp(16 * 1024 * 1024, 256 * 1024 * 1024) // 16MB to 256MB
        } else {
            64 * 1024 * 1024 // Default to 64MB
        };

        Self {
            target_memory_usage: target_memory,
            max_chunk_size: usize::MAX,
            min_chunk_size: 1024,
            target_chunk_duration: Some(Duration::from_millis(100)), // Alpha 6: Default target 100ms per chunk
            consider_distribution: true,                             // Alpha 6: Enable by default
            optimize_for_parallel: cpu_cores > 1,                    // Alpha 6: Auto-detect
            num_workers: Some(cpu_cores),
        }
    }
}

impl AdaptiveChunkingParams {
    /// Alpha 6: Detect available system memory
    fn detect_available_memory() -> Option<usize> {
        // Simplified memory detection - in a real implementation this would be more robust
        #[cfg(unix)]
        {
            if let Ok(output) = std::process::Command::new("sh")
                .args([
                    "-c",
                    "cat /proc/meminfo | grep MemAvailable | awk '{print $2}'",
                ])
                .output()
            {
                if let Ok(mem_str) = String::from_utf8(output.stdout) {
                    if let Ok(mem_kb) = mem_str.trim().parse::<usize>() {
                        return Some(mem_kb * 1024); // Convert from KB to bytes
                    }
                }
            }
        }
        None
    }

    /// Alpha 6: Create optimized parameters for specific workload types
    pub fn for_workload_type(workload: WorkloadType) -> Self {
        let mut params = Self::default();

        match workload {
            WorkloadType::MemoryIntensive => {
                params.target_memory_usage /= 2; // Use smaller chunks
                params.consider_distribution = false; // Skip expensive analysis
            }
            WorkloadType::ComputeIntensive => {
                params.target_chunk_duration = Some(Duration::from_millis(500)); // Longer chunks
                params.optimize_for_parallel = true;
            }
            WorkloadType::IoIntensive => {
                params.target_memory_usage *= 2; // Larger chunks for I/O
                params.min_chunk_size = 64 * 1024; // Larger minimum for I/O efficiency
            }
            WorkloadType::Balanced => {
                // Use defaults
            }
        }

        params
    }
}

/// Result of adaptive chunking analysis.
#[derive(Debug, Clone)]
pub struct AdaptiveChunkingResult {
    /// Recommended chunking strategy
    pub strategy: ChunkingStrategy,

    /// Estimated memory usage per chunk (in bytes)
    pub estimated_memory_per_chunk: usize,

    /// Factors that influenced the chunking decision
    pub decision_factors: Vec<String>,
}

/// Trait for adaptive chunking capabilities.
pub trait AdaptiveChunking<A: Clone + Copy + 'static + Send + Sync> {
    /// Calculate an optimal chunking strategy based on array characteristics.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to guide the adaptive chunking process
    ///
    /// # Returns
    ///
    /// A result containing the recommended chunking strategy and metadata
    fn adaptive_chunking(
        &self,
        params: AdaptiveChunkingParams,
    ) -> CoreResult<AdaptiveChunkingResult>;

    /// Process chunks using an automatically determined optimal chunking strategy.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to guide the adaptive chunking process
    /// * `f` - Function to process each chunk
    ///
    /// # Returns
    ///
    /// A vector of results, one for each chunk
    fn process_chunks_adaptive<F, R>(
        &self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A], usize) -> R;

    /// Process chunks mutably using an automatically determined optimal chunking strategy.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to guide the adaptive chunking process
    /// * `f` - Function to process each chunk
    fn process_chunks_mut_adaptive<F>(
        &mut self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<()>
    where
        F: Fn(&mut [A], usize);

    /// Process chunks in parallel using an automatically determined optimal chunking strategy.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to guide the adaptive chunking process
    /// * `f` - Function to process each chunk
    ///
    /// # Returns
    ///
    /// A vector of results, one for each chunk
    #[cfg(feature = "parallel")]
    fn process_chunks_parallel_adaptive<F, R>(
        &self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A], usize) -> R + Send + Sync,
        R: Send,
        A: Send + Sync;
}

impl<A: Clone + Copy + 'static + Send + Sync + Send + Sync> AdaptiveChunking<A>
    for MemoryMappedArray<A>
{
    fn adaptive_chunking(
        &self,
        params: AdaptiveChunkingParams,
    ) -> CoreResult<AdaptiveChunkingResult> {
        // Get total number of elements in the array
        let total_elements = self.size;

        // Calculate element size
        let element_size = std::mem::size_of::<A>();

        // Prevent division by zero for zero-sized types
        if element_size == 0 {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new("Cannot chunk zero-sized type".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Calculate initial chunk size based on target memory usage
        // Use checked division to prevent arithmetic overflow
        let mut chunk_size = params
            .target_memory_usage
            .checked_div(element_size)
            .ok_or_else(|| {
                CoreError::ComputationError(
                    ErrorContext::new("Arithmetic overflow in chunk size calculation".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        // Apply min/max constraints
        chunk_size = chunk_size.clamp(params.min_chunk_size, params.max_chunk_size);

        // Ensure we don't exceed total elements
        chunk_size = chunk_size.min(total_elements);

        // Consider dimensionality-specific adjustments
        let (chunk_size, decision_factors) =
            self.optimize_for_dimensionality(chunk_size, &params)?;

        // Factor in parallel processing if requested
        let (chunk_size, decision_factors) = if params.optimize_for_parallel {
            let (parallel_chunk_size, parallel_factors) =
                self.optimize_for_parallel_processing(chunk_size, decision_factors, &params);
            // Re-apply dimensionality optimization after parallel adjustment
            let (final_chunk_size, mut final_factors) =
                self.optimize_for_dimensionality(parallel_chunk_size, &params)?;
            final_factors.extend(parallel_factors);
            (final_chunk_size, final_factors)
        } else {
            (chunk_size, decision_factors)
        };

        // Create final chunking strategy
        let strategy = ChunkingStrategy::Fixed(chunk_size);

        // Calculate estimated memory per chunk using checked multiplication
        let estimated_memory = chunk_size.checked_mul(element_size).ok_or_else(|| {
            CoreError::ComputationError(
                ErrorContext::new("Arithmetic overflow in memory estimation".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        Ok(AdaptiveChunkingResult {
            strategy,
            estimated_memory_per_chunk: estimated_memory,
            decision_factors,
        })
    }

    fn process_chunks_adaptive<F, R>(
        &self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A], usize) -> R,
    {
        // Determine optimal chunking strategy
        let adaptive_result = self.adaptive_chunking(params)?;

        // Use determined strategy to process chunks - wrap with Ok
        Ok(self.process_chunks(adaptive_result.strategy, f))
    }

    fn process_chunks_mut_adaptive<F>(
        &mut self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<()>
    where
        F: Fn(&mut [A], usize),
    {
        // Determine optimal chunking strategy
        let adaptive_result = self.adaptive_chunking(params)?;

        // Use determined strategy to process chunks - wrap with Ok
        self.process_chunks_mut(adaptive_result.strategy, f);
        Ok(())
    }

    #[cfg(feature = "parallel")]
    fn process_chunks_parallel_adaptive<F, R>(
        &self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A], usize) -> R + Send + Sync,
        R: Send,
        A: Send + Sync,
    {
        // Make sure parameters are optimized for parallel processing
        let mut parallel_params = params;
        parallel_params.optimize_for_parallel = true;

        // Set default number of workers if not specified
        if parallel_params.num_workers.is_none() {
            parallel_params.num_workers = Some(rayon::current_num_threads());
        }

        // Determine optimal chunking strategy for parallel processing
        let adaptive_result = self.adaptive_chunking(parallel_params)?;

        // Use determined strategy to process chunks in parallel
        use super::memmap_chunks::MemoryMappedChunksParallel;
        Ok(self.process_chunks_parallel(adaptive_result.strategy, f))
    }
}

impl<A: Clone + Copy + 'static + Send + Sync + Send + Sync> MemoryMappedArray<A> {
    /// Optimize chunking based on array dimensionality.
    fn optimize_for_dimensionality(
        &self,
        initial_chunk_size: usize,
        params: &AdaptiveChunkingParams,
    ) -> CoreResult<(usize, Vec<String>)> {
        let mut decision_factors = Vec::new();
        let mut chunk_size = initial_chunk_size;

        match self.shape.len() {
            1 => {
                // For 1D arrays, we can use the initial chunk size directly
                decision_factors.push("1D array: Using direct chunking".to_string());
            }
            2 => {
                // For 2D arrays, try to align with rows when possible
                let row_length = self.shape[1];

                if chunk_size >= row_length {
                    // If chunk size is larger than row length, adjust to be a multiple
                    if chunk_size % row_length != 0 {
                        // Adjust to a multiple of row length for better cache behavior using checked arithmetic
                        let new_size = (chunk_size / row_length)
                            .checked_mul(row_length)
                            .unwrap_or(chunk_size); // Fallback to original size on overflow
                        if new_size >= params.min_chunk_size {
                            chunk_size = new_size;
                            decision_factors.push(format!(
                                "2D array: Adjusted chunk size to {} (multiple of row length {})",
                                chunk_size, row_length
                            ));
                        }
                    }
                } else {
                    // If chunk size is smaller than row length, round up to row length
                    // to ensure we process complete rows
                    if row_length <= params.max_chunk_size {
                        chunk_size = row_length;
                        decision_factors.push(format!(
                            "2D array: Adjusted chunk size to row length {}",
                            row_length
                        ));
                    } else {
                        // Row length exceeds max chunk size, keep original chunk size
                        decision_factors.push(format!(
                            "2D array: Row length {} exceeds max chunk size, keeping chunk size {}",
                            row_length, chunk_size
                        ));
                    }
                }
            }
            3 => {
                // For 3D arrays, try to align with planes or rows using checked arithmetic
                let plane_size = self.shape[1].checked_mul(self.shape[2]).unwrap_or_else(|| {
                    decision_factors.push(
                        "3D array: Overflow in plane size calculation, using row alignment"
                            .to_string(),
                    );
                    self.shape[2] // Fallback to row-based chunking
                });
                let row_length = self.shape[2];

                if chunk_size >= plane_size && chunk_size % plane_size != 0 {
                    // Adjust to a multiple of plane size for better cache behavior using checked arithmetic
                    let new_size = (chunk_size / plane_size)
                        .checked_mul(plane_size)
                        .unwrap_or(chunk_size); // Fallback to original size on overflow
                    if new_size >= params.min_chunk_size {
                        chunk_size = new_size;
                        decision_factors.push(format!(
                            "3D array: Adjusted chunk size to {} (multiple of plane size {})",
                            chunk_size, plane_size
                        ));
                    }
                } else if chunk_size >= row_length && chunk_size % row_length != 0 {
                    // Adjust to a multiple of row length using checked arithmetic
                    let new_size = (chunk_size / row_length)
                        .checked_mul(row_length)
                        .unwrap_or(chunk_size); // Fallback to original size on overflow
                    if new_size >= params.min_chunk_size {
                        chunk_size = new_size;
                        decision_factors.push(format!(
                            "3D array: Adjusted chunk size to {} (multiple of row length {})",
                            chunk_size, row_length
                        ));
                    }
                }
            }
            n => {
                decision_factors.push(format!("{}D array: Using default chunking strategy", n));
            }
        }

        Ok((chunk_size, decision_factors))
    }

    /// Optimize chunking for parallel processing.
    fn optimize_for_parallel_processing(
        &self,
        initial_chunk_size: usize,
        mut decision_factors: Vec<String>,
        params: &AdaptiveChunkingParams,
    ) -> (usize, Vec<String>) {
        let mut chunk_size = initial_chunk_size;

        if let Some(num_workers) = params.num_workers {
            let total_elements = self.size;

            // Ideally, we want at least num_workers * 2 chunks for good load balancing
            // Use checked arithmetic to prevent overflow
            let target_num_chunks = num_workers.checked_mul(2).unwrap_or(num_workers);
            let ideal_chunk_size = if target_num_chunks > 0 {
                total_elements / target_num_chunks
            } else {
                total_elements // Fallback for edge cases
            };

            if ideal_chunk_size >= params.min_chunk_size
                && ideal_chunk_size <= params.max_chunk_size
            {
                // Use the ideal chunk size for parallel processing
                chunk_size = ideal_chunk_size;
                decision_factors.push(format!(
                    "Parallel optimization: Adjusted chunk size to {} for {} workers",
                    chunk_size, num_workers
                ));
            } else if ideal_chunk_size < params.min_chunk_size {
                // If ideal size is too small, use minimum size
                chunk_size = params.min_chunk_size;
                let actual_chunks = total_elements / chunk_size
                    + if total_elements % chunk_size != 0 {
                        1
                    } else {
                        0
                    };
                decision_factors.push(format!(
                    "Parallel optimization: Using minimum chunk size {}, resulting in {} chunks for {} workers",
                    chunk_size, actual_chunks, num_workers
                ));
            }
        } else {
            decision_factors.push(
                "Parallel optimization requested but no worker count specified, using default chunking".to_string()
            );
        }

        (chunk_size, decision_factors)
    }
}

/// Builder for creating adaptive chunking parameters with a fluent API.
#[derive(Debug, Clone)]
pub struct AdaptiveChunkingBuilder {
    params: AdaptiveChunkingParams,
}

impl AdaptiveChunkingBuilder {
    /// Create a new builder with default parameters.
    pub fn new() -> Self {
        Self {
            params: AdaptiveChunkingParams::default(),
        }
    }

    /// Set the target memory usage per chunk.
    pub const fn with_target_memory(mut self, bytes: usize) -> Self {
        self.params.target_memory_usage = bytes;
        self
    }

    /// Set the maximum chunk size.
    pub const fn with_max_chunk_size(mut self, size: usize) -> Self {
        self.params.max_chunk_size = size;
        self
    }

    /// Set the minimum chunk size.
    pub const fn with_min_chunk_size(mut self, size: usize) -> Self {
        self.params.min_chunk_size = size;
        self
    }

    /// Set the target chunk processing duration.
    pub fn with_target_duration(mut self, duration: Duration) -> Self {
        self.params.target_chunk_duration = Some(duration);
        self
    }

    /// Enable consideration of data distribution.
    pub const fn consider_distribution(mut self, enable: bool) -> Self {
        self.params.consider_distribution = enable;
        self
    }

    /// Enable optimization for parallel processing.
    pub const fn optimize_for_parallel(mut self, enable: bool) -> Self {
        self.params.optimize_for_parallel = enable;
        self
    }

    /// Set the number of worker threads to optimize for.
    pub fn with_num_workers(mut self, workers: usize) -> Self {
        self.params.num_workers = Some(workers);
        self
    }

    /// Build the parameters.
    pub fn build(self) -> AdaptiveChunkingParams {
        self.params
    }
}

impl Default for AdaptiveChunkingBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Alpha 6: Advanced adaptive chunking algorithms and load balancing
pub mod alpha6_enhancements {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    /// Performance metrics collector for adaptive optimization
    #[derive(Debug, Clone, Default)]
    #[allow(dead_code)]
    pub struct ChunkingPerformanceMetrics {
        pub chunk_processing_times: Vec<Duration>,
        pub memory_usage_per_chunk: Vec<usize>,
        pub throughput_mbps: Vec<f64>,
        pub cpu_utilization: Vec<f64>,
    }

    /// Alpha 6: Dynamic load balancer for heterogeneous computing environments
    #[allow(dead_code)]
    pub struct DynamicLoadBalancer {
        worker_performance: Vec<f64>,         // Relative performance scores
        current_loads: Arc<Vec<AtomicUsize>>, // Current load per worker
        target_efficiency: f64,               // Target CPU utilization (0.0-1.0)
    }

    #[allow(dead_code)]
    impl DynamicLoadBalancer {
        /// Create a new load balancer for the specified number of workers
        pub fn new(num_workers: usize) -> Self {
            Self {
                worker_performance: vec![1.0; num_workers], // Start with equal performance
                current_loads: Arc::new((0..num_workers).map(|_| AtomicUsize::new(0)).collect()),
                target_efficiency: 0.85, // Target 85% CPU utilization
            }
        }

        /// Calculate optimal chunk distribution based on worker performance
        pub fn calculate_chunk_distribution(&self, total_work: usize) -> Vec<usize> {
            let total_performance: f64 = self.worker_performance.iter().sum();
            let mut distribution = Vec::new();
            let mut remaining_work = total_work;

            // Distribute work proportionally to performance, except for the last worker
            for (i, &performance) in self.worker_performance.iter().enumerate() {
                if i == self.worker_performance.len() - 1 {
                    // Give all remaining work to the last worker
                    distribution.push(remaining_work);
                } else {
                    let work_share = (total_work as f64 * performance / total_performance) as usize;
                    distribution.push(work_share);
                    remaining_work = remaining_work.saturating_sub(work_share);
                }
            }

            distribution
        }

        /// Update worker performance metrics based on observed execution times
        pub fn update_performance_metrics(
            &mut self,
            worker_id: usize,
            execution_time: Duration,
            work_amount: usize,
        ) {
            if worker_id < self.worker_performance.len() {
                // Calculate performance as work/time (higher is better)
                let performance = work_amount as f64 / execution_time.as_secs_f64();

                // Exponential moving average to adapt to changing conditions
                let alpha = 0.1; // Learning rate
                self.worker_performance[worker_id] =
                    (1.0 - alpha) * self.worker_performance[worker_id] + alpha * performance;
            }
        }
    }

    /// Alpha 6: Intelligent chunk size predictor using historical data
    #[allow(dead_code)]
    pub struct ChunkSizePredictor {
        historical_metrics: Vec<ChunkingPerformanceMetrics>,
        workload_characteristics: Vec<(WorkloadType, usize)>, // (workload_type, optimal_chunk_size)
    }

    #[allow(dead_code)]
    impl ChunkSizePredictor {
        pub fn new() -> Self {
            Self {
                historical_metrics: Vec::new(),
                workload_characteristics: Vec::new(),
            }
        }

        /// Predict optimal chunk size based on workload characteristics and history
        pub fn predict_optimal_chunk_size(
            &self,
            workload: WorkloadType,
            data_size: usize,
            available_memory: usize,
        ) -> usize {
            // Start with base predictions from historical data
            let historical_prediction = self.get_historical_prediction(workload);

            // Apply memory constraints
            let memory_constrained = (available_memory / 4).max(1024); // Use 1/4 of available memory

            // Apply data size constraints
            let data_constrained = (data_size / 8).max(1024); // At least 8 chunks

            // Combine predictions with weighting
            let base_prediction = historical_prediction.unwrap_or(64 * 1024); // 64KB default
            let memory_weight = 0.4;
            let data_weight = 0.4;
            let historical_weight = 0.2;

            let predicted_size = (memory_weight * memory_constrained as f64
                + data_weight * data_constrained as f64
                + historical_weight * base_prediction as f64)
                as usize;

            // Ensure reasonable bounds
            predicted_size.clamp(1024, 256 * 1024 * 1024) // 1KB to 256MB
        }

        fn get_historical_prediction(&self, workload: WorkloadType) -> Option<usize> {
            // Find the most recent matching workload
            self.workload_characteristics
                .iter()
                .rev() // Start from most recent
                .find(|(wl, _)| *wl == workload)
                .map(|(_, size)| *size)
        }

        /// Record performance metrics for future predictions
        pub fn record_performance(
            &mut self,
            workload: WorkloadType,
            chunk_size: usize,
            metrics: ChunkingPerformanceMetrics,
        ) {
            self.historical_metrics.push(metrics);
            self.workload_characteristics.push((workload, chunk_size));

            // Keep only the last 100 entries to prevent unbounded growth
            if self.historical_metrics.len() > 100 {
                self.historical_metrics.remove(0);
                self.workload_characteristics.remove(0);
            }
        }
    }

    /// Alpha 6: NUMA-aware chunking for large multi-socket systems
    #[allow(dead_code)]
    pub fn calculate_numa_aware_chunking(
        data_size: usize,
        num_numa_nodes: usize,
    ) -> ChunkingStrategy {
        if num_numa_nodes <= 1 {
            return ChunkingStrategy::Auto;
        }

        // Try to align chunks with NUMA boundaries
        let base_chunk_size = data_size / (num_numa_nodes * 2); // 2 chunks per NUMA node
        let aligned_chunk_size = align_to_cache_line(base_chunk_size);

        ChunkingStrategy::Fixed(aligned_chunk_size)
    }

    /// Align chunk size to cache line boundaries for better performance
    #[allow(dead_code)]
    fn align_to_cache_line(size: usize) -> usize {
        const CACHE_LINE_SIZE: usize = 64; // Typical cache line size
        ((size / CACHE_LINE_SIZE) + 1) * CACHE_LINE_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_adaptive_chunking_1d() {
        // Create a temporary directory for our test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_adaptive_1d.bin");

        // Create a test array and save it to a file
        let data: Vec<f64> = (0..100_000).map(|i| i as f64).collect();
        let mut file = File::create(&file_path).unwrap();
        for val in &data {
            file.write_all(&val.to_ne_bytes()).unwrap();
        }
        drop(file);

        // Create a memory-mapped array
        let mmap = MemoryMappedArray::<f64>::open(&file_path, &[100_000]).unwrap();

        // Create adaptive chunking parameters
        let params = AdaptiveChunkingBuilder::new()
            .with_target_memory(1024 * 1024) // 1MB chunks
            .with_min_chunk_size(1000)
            .with_max_chunk_size(50000)
            .optimize_for_parallel(false) // Disable parallel optimization for this test
            .build();

        // Calculate adaptive chunking
        let result = mmap.adaptive_chunking(params).unwrap();

        // Verify results
        match result.strategy {
            ChunkingStrategy::Fixed(chunk_size) => {
                // The chunk size should be close to 1MB / 8 bytes = 131072 elements,
                // but capped at our max of 50000
                assert_eq!(chunk_size, 50000);
            }
            _ => panic!("Expected fixed chunking strategy"),
        }

        // Verify that the estimated memory per chunk is reasonable
        assert!(result.estimated_memory_per_chunk > 0);

        // The decision factors should mention that it's a 1D array
        assert!(result
            .decision_factors
            .iter()
            .any(|s| s.contains("1D array")));
    }

    #[test]
    fn test_adaptive_chunking_2d() {
        // Create a temporary directory for our test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_adaptive_2d.bin");

        // Create dimensions that will test row alignment
        let rows = 1000;
        let cols = 120;

        // Create a test 2D array and save it to a file
        let data = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| (i * cols + j) as f64);
        let mut file = File::create(&file_path).unwrap();
        for val in data.iter() {
            file.write_all(&val.to_ne_bytes()).unwrap();
        }
        drop(file);

        // Create a memory-mapped array
        let mmap = MemoryMappedArray::<f64>::open(&file_path, &[rows, cols]).unwrap();

        // Create adaptive chunking parameters
        let params = AdaptiveChunkingBuilder::new()
            .with_target_memory(100 * 1024) // 100KB chunks
            .with_min_chunk_size(1000)
            .with_max_chunk_size(50000)
            .build();

        // Calculate adaptive chunking
        let result = mmap.adaptive_chunking(params).unwrap();

        // Verify results
        match result.strategy {
            ChunkingStrategy::Fixed(chunk_size) => {
                // The chunk size should be adjusted to be a multiple of the row length (120)
                assert_eq!(
                    chunk_size % cols,
                    0,
                    "Chunk size should be a multiple of row length"
                );
            }
            _ => panic!("Expected fixed chunking strategy"),
        }

        // The decision factors should mention that it's a 2D array
        assert!(result
            .decision_factors
            .iter()
            .any(|s| s.contains("2D array")));
    }

    #[test]
    fn test_adaptive_chunking_parallel() {
        // Create a temporary directory for our test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_adaptive_parallel.bin");

        // Create a large test array
        let data: Vec<f64> = (0..1_000_000).map(|i| i as f64).collect();
        let mut file = File::create(&file_path).unwrap();
        for val in &data {
            file.write_all(&val.to_ne_bytes()).unwrap();
        }
        drop(file);

        // Create a memory-mapped array
        let mmap = MemoryMappedArray::<f64>::open(&file_path, &[1_000_000]).unwrap();

        // Create adaptive chunking parameters optimized for parallel processing
        let params = AdaptiveChunkingBuilder::new()
            .with_target_memory(10 * 1024 * 1024) // 10MB chunks
            .optimize_for_parallel(true)
            .with_num_workers(4)
            .build();

        // Calculate adaptive chunking
        let result = mmap.adaptive_chunking(params).unwrap();

        // Verify results
        match result.strategy {
            ChunkingStrategy::Fixed(chunk_size) => {
                // With 4 workers and desiring 8 chunks (2*workers), each chunk should handle ~125,000 elements
                // But it might be adjusted based on other factors
                assert!(chunk_size > 0, "Chunk size should be positive");
            }
            _ => panic!("Expected fixed chunking strategy"),
        }

        // The decision factors should mention parallel optimization
        assert!(result
            .decision_factors
            .iter()
            .any(|s| s.contains("Parallel optimization")));
    }
}
