//! Heterogeneous computing support for CPU-GPU hybrid operations
//!
//! This module provides capabilities for efficiently distributing work between
//! CPU and GPU resources, including automatic workload balancing, data migration,
//! and coordinated execution strategies.

use crate::gpu::{async_execution::*, GpuBackend, GpuError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Error types for heterogeneous computing operations
#[derive(Error, Debug)]
pub enum HeterogeneousError {
    /// No suitable compute device found
    #[error("No suitable compute device found for workload")]
    NoSuitableDevice,

    /// Workload balancing failed
    #[error("Workload balancing failed: {0}")]
    BalancingFailed(String),

    /// Data migration error
    #[error("Data migration error: {0}")]
    DataMigration(String),

    /// Execution coordination error
    #[error("Execution coordination error: {0}")]
    ExecutionCoordination(String),

    /// Resource exhaustion
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Underlying GPU error
    #[error("GPU error: {0}")]
    GpuError(#[from] GpuError),
}

/// Compute device types in a heterogeneous system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeDevice {
    /// CPU device
    Cpu,
    /// GPU device with specific backend
    Gpu(GpuBackend),
    /// Neural processing unit
    Npu,
    /// Field-programmable gate array
    Fpga,
    /// Digital signal processor
    Dsp,
}

impl ComputeDevice {
    /// Check if the device is available on the current system
    pub fn is_available(&self) -> bool {
        match self {
            ComputeDevice::Cpu => true,
            ComputeDevice::Gpu(backend) => backend.is_available(),
            ComputeDevice::Npu => false, // Would need specific detection
            ComputeDevice::Fpga => false, // Would need specific detection
            ComputeDevice::Dsp => false, // Would need specific detection
        }
    }

    /// Get the relative performance factor for different operation types
    pub fn performance_factor(&self, op_type: &WorkloadType) -> f64 {
        match (self, op_type) {
            (ComputeDevice::Cpu, WorkloadType::Sequential) => 1.0,
            (ComputeDevice::Cpu, WorkloadType::Parallel) => 0.3,
            (ComputeDevice::Cpu, WorkloadType::VectorizedMath) => 0.2,
            (ComputeDevice::Cpu, WorkloadType::MatrixOperations) => 0.1,
            (ComputeDevice::Cpu, WorkloadType::ConvolutionalNN) => 0.05,

            (ComputeDevice::Gpu(_), WorkloadType::Sequential) => 0.1,
            (ComputeDevice::Gpu(_), WorkloadType::Parallel) => 1.0,
            (ComputeDevice::Gpu(_), WorkloadType::VectorizedMath) => 1.0,
            (ComputeDevice::Gpu(_), WorkloadType::MatrixOperations) => 1.0,
            (ComputeDevice::Gpu(_), WorkloadType::ConvolutionalNN) => 1.0,

            (ComputeDevice::Npu, WorkloadType::ConvolutionalNN) => 1.5,
            (ComputeDevice::Npu, WorkloadType::MatrixOperations) => 1.2,
            (ComputeDevice::Npu, _) => 0.8,

            _ => 0.5, // Default conservative estimate
        }
    }
}

/// Types of computational workloads
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WorkloadType {
    /// Sequential operations
    Sequential,
    /// Embarrassingly parallel operations
    Parallel,
    /// Vectorized mathematical operations
    VectorizedMath,
    /// Dense matrix operations
    MatrixOperations,
    /// Sparse matrix operations
    SparseOperations,
    /// Convolutional neural network operations
    ConvolutionalNN,
    /// Memory-intensive operations
    MemoryIntensive,
    /// Custom workload type
    Custom(String),
}

/// Workload characteristics for heterogeneous scheduling
#[derive(Debug, Clone)]
pub struct WorkloadCharacteristics {
    /// Type of workload
    pub workload_type: WorkloadType,
    /// Problem size (number of elements or operations)
    pub problem_size: usize,
    /// Memory requirements in bytes
    pub memory_requirement: usize,
    /// Computational intensity (operations per memory access)
    pub computational_intensity: f64,
    /// Data locality score (0.0 = random access, 1.0 = sequential)
    pub data_locality: f64,
    /// Parallelization factor (how well it scales with cores)
    pub parallelization_factor: f64,
    /// Preferred data types
    pub preferred_data_types: Vec<String>,
}

impl WorkloadCharacteristics {
    /// Create characteristics for a matrix multiplication workload
    pub fn matrix_multiply(m: usize, n: usize, k: usize) -> Self {
        Self {
            workload_type: WorkloadType::MatrixOperations,
            problem_size: m * n * k,
            memory_requirement: (m * k + k * n + m * n) * 8, // Assume f64
            computational_intensity: (2.0 * k as f64) / 3.0, // 2*K ops per 3 memory accesses
            data_locality: 0.7,                              // Good with proper blocking
            parallelization_factor: 0.9,                     // Scales well with many cores
            preferred_data_types: vec!["f32".to_string(), "f16".to_string()],
        }
    }

    /// Create characteristics for a convolution workload
    pub fn convolution(
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
    ) -> Self {
        let input_size = batch_size * channels * height * width;
        let output_size = batch_size * channels * height * width; // Simplified

        Self {
            workload_type: WorkloadType::ConvolutionalNN,
            problem_size: input_size * kernel_size * kernel_size,
            memory_requirement: (input_size + output_size) * 4, // Assume f32
            computational_intensity: (kernel_size * kernel_size * 2) as f64,
            data_locality: 0.8,           // Good spatial locality in convolutions
            parallelization_factor: 0.95, // Excellent parallelization
            preferred_data_types: vec!["f16".to_string(), "i8".to_string()],
        }
    }

    /// Create characteristics for an element-wise operation
    pub fn element_wise(size: usize, ops_per_element: usize) -> Self {
        Self {
            workload_type: WorkloadType::VectorizedMath,
            problem_size: size,
            memory_requirement: size * 8, // Assume f64
            computational_intensity: ops_per_element as f64 / 2.0, // Read + write
            data_locality: 1.0,           // Perfect sequential access
            parallelization_factor: 1.0,  // Perfect parallelization
            preferred_data_types: vec!["f32".to_string(), "f64".to_string()],
        }
    }
}

/// Device performance characteristics
#[derive(Debug, Clone)]
pub struct DeviceCharacteristics {
    /// Device type
    pub device: ComputeDevice,
    /// Peak computational throughput (GFLOPS)
    pub peak_gflops: f64,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,
    /// Available memory (bytes)
    pub available_memory: usize,
    /// Number of compute units
    pub compute_units: usize,
    /// Power consumption (watts)
    pub power_consumption: f64,
    /// Data transfer overhead to/from device
    pub transfer_overhead: Duration,
}

impl DeviceCharacteristics {
    /// Create characteristics for a typical CPU
    pub fn typical_cpu() -> Self {
        Self {
            device: ComputeDevice::Cpu,
            peak_gflops: 200.0,                         // Modern CPU with AVX
            memory_bandwidth: 50.0,                     // DDR4-3200
            available_memory: 16 * 1024 * 1024 * 1024,  // 16 GB
            compute_units: 8,                           // 8 cores
            power_consumption: 95.0,                    // 95W TDP
            transfer_overhead: Duration::from_nanos(0), // No transfer needed
        }
    }

    /// Create characteristics for a typical discrete GPU
    pub fn typical_gpu() -> Self {
        Self {
            device: ComputeDevice::Gpu(GpuBackend::Cuda),
            peak_gflops: 10000.0,                         // High-end GPU
            memory_bandwidth: 900.0,                      // GDDR6X
            available_memory: 12 * 1024 * 1024 * 1024,    // 12 GB VRAM
            compute_units: 80,                            // Streaming multiprocessors
            power_consumption: 350.0,                     // 350W TGP
            transfer_overhead: Duration::from_micros(10), // PCIe transfer
        }
    }

    /// Estimate execution time for a workload
    pub fn estimate_execution_time(&self, workload: &WorkloadCharacteristics) -> Duration {
        let performance_factor = self.device.performance_factor(&workload.workload_type);

        // Simple performance model combining compute and memory bounds
        let compute_time =
            (workload.problem_size as f64) / (self.peak_gflops * 1e9 * performance_factor);

        let memory_time = (workload.memory_requirement as f64) / (self.memory_bandwidth * 1e9);

        // Take the maximum (bottleneck) and add transfer overhead
        let execution_time = compute_time.max(memory_time) + self.transfer_overhead.as_secs_f64();

        Duration::from_secs_f64(execution_time)
    }
}

/// Heterogeneous execution strategy
#[derive(Debug, Clone)]
pub enum ExecutionStrategy {
    /// Execute entirely on CPU
    CpuOnly,
    /// Execute entirely on GPU
    GpuOnly(GpuBackend),
    /// Split workload between CPU and GPU
    CpuGpuSplit {
        cpu_fraction: f64,
        gpu_backend: GpuBackend,
    },
    /// Use multiple devices with custom distribution
    MultiDevice(HashMap<ComputeDevice, f64>),
    /// Automatic selection based on characteristics
    Automatic,
}

/// Heterogeneous compute scheduler
pub struct HeterogeneousScheduler {
    available_devices: Vec<DeviceCharacteristics>,
    performance_history: Arc<Mutex<HashMap<String, Duration>>>,
    #[allow(dead_code)]
    async_manager: AsyncGpuManager,
}

impl HeterogeneousScheduler {
    /// Create a new heterogeneous scheduler
    pub fn new() -> Self {
        let mut available_devices = vec![DeviceCharacteristics::typical_cpu()];

        // Detect available GPUs
        for backend in [GpuBackend::Cuda, GpuBackend::Rocm, GpuBackend::Metal] {
            if backend.is_available() {
                let mut gpu_chars = DeviceCharacteristics::typical_gpu();
                gpu_chars.device = ComputeDevice::Gpu(backend);
                available_devices.push(gpu_chars);
            }
        }

        Self {
            available_devices,
            performance_history: Arc::new(Mutex::new(HashMap::new())),
            async_manager: AsyncGpuManager::new(),
        }
    }

    /// Get available compute devices
    pub fn available_devices(&self) -> &[DeviceCharacteristics] {
        &self.available_devices
    }

    /// Select optimal execution strategy for a workload
    pub fn select_strategy(
        &self,
        workload: &WorkloadCharacteristics,
    ) -> Result<ExecutionStrategy, HeterogeneousError> {
        if self.available_devices.is_empty() {
            return Err(HeterogeneousError::NoSuitableDevice);
        }

        // Estimate execution time on each device
        let mut device_times: Vec<_> = self
            .available_devices
            .iter()
            .map(|device| {
                let time = device.estimate_execution_time(workload);
                (device.device, time)
            })
            .collect();

        // Sort by execution time
        device_times.sort_by_key(|(_, time)| *time);

        let best_device = device_times[0].0;
        let best_time = device_times[0].1;

        // Check if splitting between CPU and GPU would be beneficial
        if device_times.len() >= 2 {
            let second_best_time = device_times[1].1;

            // If the times are close, consider splitting
            if best_time.as_secs_f64() * 1.5 > second_best_time.as_secs_f64() {
                if let (ComputeDevice::Cpu, ComputeDevice::Gpu(backend)) =
                    (device_times[0].0, device_times[1].0)
                {
                    return Ok(ExecutionStrategy::CpuGpuSplit {
                        cpu_fraction: 0.3,
                        gpu_backend: backend,
                    });
                } else if let (ComputeDevice::Gpu(backend), ComputeDevice::Cpu) =
                    (device_times[0].0, device_times[1].0)
                {
                    return Ok(ExecutionStrategy::CpuGpuSplit {
                        cpu_fraction: 0.3,
                        gpu_backend: backend,
                    });
                }
            }
        }

        // Default to best single device
        match best_device {
            ComputeDevice::Cpu => Ok(ExecutionStrategy::CpuOnly),
            ComputeDevice::Gpu(backend) => Ok(ExecutionStrategy::GpuOnly(backend)),
            _ => Err(HeterogeneousError::NoSuitableDevice),
        }
    }

    /// Execute a workload using the specified strategy
    pub fn execute_workload<F, R>(
        &self,
        workload: &WorkloadCharacteristics,
        strategy: ExecutionStrategy,
        work_fn: F,
    ) -> Result<R, HeterogeneousError>
    where
        F: FnOnce(&ExecutionStrategy) -> Result<R, HeterogeneousError>,
    {
        let start_time = Instant::now();

        let result = work_fn(&strategy)?;

        let execution_time = start_time.elapsed();

        // Store performance history for future optimization
        let key = format!("{:?}_{}", workload.workload_type, workload.problem_size);
        self.performance_history
            .lock()
            .unwrap()
            .insert(key, execution_time);

        Ok(result)
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> HeterogeneousStats {
        let history = self.performance_history.lock().unwrap();

        let total_executions = history.len();
        let avg_execution_time = if total_executions > 0 {
            let total_time: Duration = history.values().sum();
            total_time / total_executions as u32
        } else {
            Duration::ZERO
        };

        HeterogeneousStats {
            available_devices: self.available_devices.len(),
            total_executions,
            avg_execution_time,
            device_utilization: self.calculate_device_utilization(),
        }
    }

    /// Calculate device utilization statistics
    fn calculate_device_utilization(&self) -> HashMap<ComputeDevice, f64> {
        // In a real implementation, this would track actual device usage
        // For now, return mock data
        let mut utilization = HashMap::new();
        for device in &self.available_devices {
            utilization.insert(device.device, 0.0);
        }
        utilization
    }

    /// Optimize execution strategy based on historical performance
    pub fn optimize_strategy(
        &self,
        workload: &WorkloadCharacteristics,
        current_strategy: ExecutionStrategy,
    ) -> ExecutionStrategy {
        let key = format!("{:?}_{}", workload.workload_type, workload.problem_size);
        let history = self.performance_history.lock().unwrap();

        // If we have historical data, use it to refine the strategy
        if let Some(&_historical_time) = history.get(&key) {
            // Simple heuristic: if historical time is much better than estimated,
            // stick with the historical strategy
            // This is a simplified version - real implementation would be more sophisticated
            return current_strategy;
        }

        current_strategy
    }
}

impl Default for HeterogeneousScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for heterogeneous computing operations
#[derive(Debug, Clone)]
pub struct HeterogeneousStats {
    /// Number of available compute devices
    pub available_devices: usize,
    /// Total number of executions
    pub total_executions: usize,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Device utilization percentages
    pub device_utilization: HashMap<ComputeDevice, f64>,
}

/// Workload distribution for multi-device execution
#[derive(Debug, Clone)]
pub struct WorkloadDistribution {
    /// Device assignments with work fractions
    pub assignments: HashMap<ComputeDevice, f64>,
    /// Data partitioning strategy
    pub partitioning: PartitioningStrategy,
    /// Coordination strategy
    pub coordination: CoordinationStrategy,
}

/// Data partitioning strategies for multi-device execution
#[derive(Debug, Clone)]
pub enum PartitioningStrategy {
    /// Split data by rows
    RowSplit,
    /// Split data by columns
    ColumnSplit,
    /// Split data by blocks
    BlockSplit { block_size: (usize, usize) },
    /// Custom partitioning function
    Custom(String),
}

/// Coordination strategies for multi-device execution
#[derive(Debug, Clone)]
pub enum CoordinationStrategy {
    /// Bulk synchronous parallel
    BulkSynchronous,
    /// Asynchronous with events
    AsyncWithEvents,
    /// Pipeline parallel
    Pipeline,
    /// Custom coordination
    Custom(String),
}

/// Helper functions for common heterogeneous computing patterns
pub mod patterns {
    use super::*;

    /// Execute a map operation across heterogeneous devices
    pub fn heterogeneous_map<T, F>(
        scheduler: &HeterogeneousScheduler,
        data: &[T],
        map_fn: F,
    ) -> Result<Vec<T>, HeterogeneousError>
    where
        T: Clone + Send + Sync,
        F: Fn(&T) -> T + Send + Sync,
    {
        let workload = WorkloadCharacteristics::element_wise(data.len(), 1);
        let strategy = scheduler.select_strategy(&workload)?;

        scheduler.execute_workload(&workload, strategy, |_strategy| {
            // Simple implementation - in practice would distribute across devices
            Ok(data.iter().map(map_fn).collect())
        })
    }

    /// Execute a reduction operation across heterogeneous devices
    pub fn heterogeneous_reduce<T, F>(
        scheduler: &HeterogeneousScheduler,
        data: &[T],
        initial: T,
        reduce_fn: F,
    ) -> Result<T, HeterogeneousError>
    where
        T: Clone + Send + Sync,
        F: Fn(T, &T) -> T + Send + Sync,
    {
        let workload = WorkloadCharacteristics::element_wise(data.len(), 1);
        let strategy = scheduler.select_strategy(&workload)?;

        scheduler.execute_workload(&workload, strategy, |_strategy| {
            Ok(data.iter().fold(initial, reduce_fn))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_availability() {
        assert!(ComputeDevice::Cpu.is_available());
        // GPU availability depends on system configuration
    }

    #[test]
    fn test_workload_characteristics() {
        let gemm = WorkloadCharacteristics::matrix_multiply(1000, 1000, 1000);
        assert_eq!(gemm.workload_type, WorkloadType::MatrixOperations);
        assert!(gemm.computational_intensity > 0.0);
    }

    #[test]
    fn test_device_characteristics() {
        let cpu = DeviceCharacteristics::typical_cpu();
        let gpu = DeviceCharacteristics::typical_gpu();

        assert_eq!(cpu.device, ComputeDevice::Cpu);
        assert!(matches!(gpu.device, ComputeDevice::Gpu(_)));
        assert!(gpu.peak_gflops > cpu.peak_gflops);
    }

    #[test]
    fn test_execution_time_estimation() {
        let cpu = DeviceCharacteristics::typical_cpu();
        let workload = WorkloadCharacteristics::element_wise(1000000, 1);

        let time = cpu.estimate_execution_time(&workload);
        assert!(time > Duration::ZERO);
    }

    #[test]
    fn test_scheduler_creation() {
        let scheduler = HeterogeneousScheduler::new();
        assert!(!scheduler.available_devices().is_empty());
    }

    #[test]
    fn test_strategy_selection() {
        let scheduler = HeterogeneousScheduler::new();
        let workload = WorkloadCharacteristics::matrix_multiply(100, 100, 100);

        let strategy = scheduler.select_strategy(&workload);
        assert!(strategy.is_ok());
    }
}
