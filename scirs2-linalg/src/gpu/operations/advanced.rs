//! Advanced GPU-accelerated linear algebra operations

use super::super::AutoGpuSelector;
use super::dispatcher::GpuOperationDispatcher;
use super::kernels::GpuKernelManager;
use super::optimization::{BatchPerformanceRecord, BatchSizeOptimizer};
use super::profiling::GpuPerformanceProfiler;
use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign, Zero};
use std::collections::HashMap;
use std::fmt::Debug;

/// Advanced GPU-accelerated linear algebra operations
pub struct AdvancedGpuOperations<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    dispatcher: GpuOperationDispatcher<T>,
    kernel_manager: GpuKernelManager,
    profiler: GpuPerformanceProfiler,
    batchsize_optimizer: BatchSizeOptimizer,
}

impl<T> AdvancedGpuOperations<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    /// Create new advanced GPU operations handler
    pub fn new() -> Self {
        Self {
            dispatcher: GpuOperationDispatcher::new(),
            kernel_manager: GpuKernelManager::new(),
            profiler: GpuPerformanceProfiler::new(),
            batchsize_optimizer: BatchSizeOptimizer::new(1024 * 1024 * 1024), // 1GB default
        }
    }

    /// Advanced batched matrix multiplication with optimal batching
    pub fn batched_matmul_optimized(
        &mut self,
        matrices_a: &[ArrayView2<T>],
        matrices_b: &[ArrayView2<T>],
    ) -> LinalgResult<Vec<Array2<T>>> {
        if matrices_a.len() != matrices_b.len() {
            return Err(LinalgError::InvalidInput(
                "Number of A and B matrices must match".to_string(),
            ));
        }

        let batch_count = matrices_a.len();
        let optimal_batchsize = self
            .batchsize_optimizer
            .optimize_batchsize("batched_matmul", batch_count);

        let mut results = Vec::with_capacity(batch_count);

        // Process in optimal-sized batches
        for batch_start in (0..batch_count).step_by(optimal_batchsize) {
            let batch_end = (batch_start + optimal_batchsize).min(batch_count);
            let batchsize = batch_end - batch_start;

            let start_time = std::time::Instant::now();

            // Process batch
            for i in batch_start..batch_end {
                let result = self
                    .dispatcher
                    .auto_matmul(&matrices_a[i], &matrices_b[i], None)?;
                results.push(result);
            }

            let execution_time = start_time.elapsed().as_secs_f64();

            // Record performance
            let record = BatchPerformanceRecord {
                operation: "batched_matmul".to_string(),
                batchsize,
                execution_time,
                memory_usage: batchsize * 1000, // Estimate
                throughput: batchsize as f64 / execution_time,
            };

            self.batchsize_optimizer.record_performance(record);
        }

        Ok(results)
    }

    /// GPU-accelerated tensor contraction (Einstein summation)
    pub fn gpu_tensor_contraction(
        &mut self,
        tensors: &[ArrayView2<T>],
        contraction_indices: &[(usize, usize)],
    ) -> LinalgResult<Array2<T>> {
        if tensors.is_empty() {
            return Err(LinalgError::InvalidInput("No tensors provided".to_string()));
        }

        let start_time = std::time::Instant::now();

        // For this simplified implementation, we'll do pairwise contractions
        let mut result = tensors[0].to_owned();

        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if i - 1 < contraction_indices.len() {
                result = self.contract_pair(&result.view(), tensor, contraction_indices[i - 1])?;
            }
        }

        let execution_time = start_time.elapsed().as_secs_f64();
        self.profiler.record("tensor_contraction", execution_time);

        Ok(result)
    }

    /// Contract two matrices along specified indices
    fn contract_pair(
        &self,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
        indices: (usize, usize),
    ) -> LinalgResult<Array2<T>> {
        let (a_contract_idx, b_contract_idx) = indices;

        // Validate indices
        if a_contract_idx >= 2 || b_contract_idx >= 2 {
            return Err(LinalgError::InvalidInput(
                "Contraction indices out of bounds".to_string(),
            ));
        }

        // Determine result dimensions
        let _a_dim = a.dim();
        let _b_dim = b.dim();

        // For 2D tensors, this is essentially matrix multiplication with potential transposition
        match (a_contract_idx, b_contract_idx) {
            (1, 0) => self.dispatcher.cpu_matmul(a, b), // Standard matrix multiplication
            (0, 0) => {
                // Need to transpose a
                let a_t = a.t();
                self.dispatcher.cpu_matmul(&a_t, b)
            }
            (1, 1) => {
                // Need to transpose b
                let b_t = b.t();
                self.dispatcher.cpu_matmul(a, &b_t)
            }
            (0, 1) => {
                // Need to transpose both
                let a_t = a.t();
                let b_t = b.t();
                self.dispatcher.cpu_matmul(&a_t, &b_t)
            }
            _ => Err(LinalgError::InvalidInput(
                "Invalid contraction pattern".to_string(),
            )),
        }
    }

    /// Adaptive GPU memory management
    pub fn optimize_memory_usage(&mut self, operation_sequence: &[&str]) -> LinalgResult<()> {
        // Analyze operation sequence to optimize memory allocation patterns
        let mut memory_requirements = std::collections::HashMap::new();

        for &op in operation_sequence {
            let requirement = match op {
                "matmul" => 1000000, // Estimate based on typical matrix sizes
                "matvec" => 100000,
                "decomposition" => 2000000,
                "solve" => 1500000,
                _ => 500000,
            };

            memory_requirements.insert(op.to_string(), requirement);
        }

        // Update batch size optimizer with new requirements
        for (op, req) in memory_requirements {
            let optimal_batch = (self.batchsize_optimizer.memory_limit / req).max(1);
            self.batchsize_optimizer
                .optimalsizes
                .insert(op, optimal_batch);
        }

        Ok(())
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> std::collections::HashMap<String, (f64, f64)> {
        let mut stats = std::collections::HashMap::new();

        for op in self.profiler.operations() {
            if let (Some(avg), Some(best)) = (
                self.profiler.average_time(&op),
                self.profiler.best_time(&op),
            ) {
                stats.insert(op.to_string(), (avg, best));
            }
        }

        stats
    }

    /// Get reference to the dispatcher for direct access
    pub fn dispatcher(&self) -> &GpuOperationDispatcher<T> {
        &self.dispatcher
    }

    /// Get mutable reference to the dispatcher for configuration
    pub fn dispatcher_mut(&mut self) -> &mut GpuOperationDispatcher<T> {
        &mut self.dispatcher
    }

    /// Get reference to the kernel manager for direct access
    pub fn kernel_manager(&self) -> &GpuKernelManager {
        &self.kernel_manager
    }

    /// Get mutable reference to the kernel manager for configuration
    pub fn kernel_manager_mut(&mut self) -> &mut GpuKernelManager {
        &mut self.kernel_manager
    }

    /// Get reference to the profiler for direct access
    pub fn profiler(&self) -> &GpuPerformanceProfiler {
        &self.profiler
    }

    /// Get mutable reference to the profiler for configuration
    pub fn profiler_mut(&mut self) -> &mut GpuPerformanceProfiler {
        &mut self.profiler
    }

    /// Get reference to the batch size optimizer for direct access
    pub fn batch_optimizer(&self) -> &BatchSizeOptimizer {
        &self.batchsize_optimizer
    }

    /// Get mutable reference to the batch size optimizer for configuration
    pub fn batch_optimizer_mut(&mut self) -> &mut BatchSizeOptimizer {
        &mut self.batchsize_optimizer
    }

    /// Perform a complete pipeline of operations with optimization
    pub fn optimized_pipeline(
        &mut self,
        operations: &[&str],
        matrices: &[ArrayView2<T>],
    ) -> LinalgResult<Vec<Array2<T>>> {
        // First optimize memory usage for the operation sequence
        self.optimize_memory_usage(operations)?;

        let mut results = Vec::new();

        for (&operation, matrix) in operations.iter().zip(matrices.iter()) {
            let start_time = std::time::Instant::now();

            let result = match operation {
                "transpose" => {
                    let transposed = matrix.t().to_owned();
                    transposed
                }
                "copy" => matrix.to_owned(),
                _ => {
                    return Err(LinalgError::InvalidInput(format!(
                        "Unsupported operation: {}",
                        operation
                    )))
                }
            };

            let execution_time = start_time.elapsed().as_secs_f64();
            self.profiler.record(operation, execution_time);

            results.push(result);
        }

        Ok(results)
    }
}

impl<T> Default for AdvancedGpuOperations<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}
