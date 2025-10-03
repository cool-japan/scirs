//! GPU acceleration foundations for linear algebra operations
//!
//! This module provides the foundation for GPU-accelerated linear algebra operations
//! including CUDA, OpenCL, and ROCm support. It defines traits and abstractions
//! for different GPU backends while maintaining a unified interface.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign, Zero};
use std::fmt::Debug;

pub mod acceleration;
pub mod advanced;
pub mod backends;
pub mod device_info;
pub mod memory;
pub mod operations;

// Re-export operations
pub use operations::{
    AdvancedGpuOperations, BatchSizeOptimizer, GpuOperationDispatcher, DEFAULT_GPU_THRESHOLD,
};

// Re-export acceleration framework
pub use acceleration::{
    get_global_gpu_framework, initialize_global_gpu_acceleration, AccelerationConfig,
    GpuAccelerationFramework, GpuPerformanceProfiler,
};

/// GPU device types supported by the library
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuDeviceType {
    /// NVIDIA GPU with CUDA support
    Cuda,
    /// OpenCL-compatible GPU
    OpenCl,
    /// AMD GPU with ROCm support
    Rocm,
    /// Vulkan compute support
    Vulkan,
    /// Apple Metal GPU
    Metal,
    /// Intel GPU with OneAPI
    OneApi,
    /// WebGPU for browser support
    WebGpu,
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device type
    pub device_type: GpuDeviceType,
    /// Device name/description
    pub name: String,
    /// Total global memory in bytes
    pub total_memory: usize,
    /// Number of compute units/cores
    pub compute_units: u32,
    /// Clock frequency in MHz
    pub clock_frequency: u32,
    /// Whether the device supports double precision
    pub supports_fp64: bool,
    /// Whether the device supports half precision
    pub supports_fp16: bool,
    /// Maximum work group size
    pub max_work_groupsize: usize,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth: f64,
    /// L2 cache size in bytes
    pub l2_cachesize: usize,
    /// Shared memory per work group in bytes
    pub shared_memory_per_block: usize,
    /// Number of registers per work group
    pub registers_per_block: u32,
    /// Warp/wavefront size
    pub warpsize: u32,
    /// Maximum number of threads per multiprocessor
    pub max_threads_per_mp: u32,
    /// Number of multiprocessors
    pub multiprocessor_count: u32,
    /// Tensor core support (for AI workloads)
    pub supports_tensor_cores: bool,
    /// Mixed precision support
    pub supports_mixed_precision: bool,
    /// Device vendor
    pub vendor: String,
}

/// GPU performance characteristics for optimization
#[derive(Debug, Clone)]
pub struct GpuPerformanceProfile {
    /// Peak theoretical FLOPS (single precision)
    pub peak_flops_sp: f64,
    /// Peak theoretical FLOPS (double precision)
    pub peak_flops_dp: f64,
    /// Peak memory bandwidth utilization efficiency (0.0 to 1.0)
    pub memory_efficiency: f64,
    /// Compute efficiency (0.0 to 1.0)
    pub compute_efficiency: f64,
    /// Optimal work group sizes for different operations
    pub optimal_work_groupsizes: std::collections::HashMap<GpuOperation, usize>,
    /// Thread occupancy targets
    pub target_occupancy: f64,
}

/// GPU operation types for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuOperation {
    MatrixMultiplication,
    ElementWise,
    Reduction,
    Transpose,
    Decomposition,
    IterativeSolver,
    FFT,
    Convolution,
}

/// GPU memory management strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMemoryStrategy {
    /// Always allocate on GPU
    GpuOnly,
    /// Use unified memory
    Unified,
    /// Explicit CPU-GPU transfers
    Explicit,
    /// Adaptive based on data size
    Adaptive,
    /// Stream-based overlapped transfers
    Streaming,
}

/// GPU computation precision mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPrecisionMode {
    /// Single precision (32-bit)
    Single,
    /// Double precision (64-bit)
    Double,
    /// Half precision (16-bit)
    Half,
    /// Mixed precision (automatic selection)
    Mixed,
    /// Tensor core optimized mixed precision
    TensorCore,
}

/// GPU acceleration strategy for different matrix sizes
#[derive(Debug, Clone)]
pub struct GpuAccelerationStrategy {
    /// Minimum matrix size for GPU acceleration
    pub minsize_threshold: usize,
    /// Maximum matrix size for single GPU
    pub max_single_gpusize: usize,
    /// Preferred memory strategy
    pub memory_strategy: GpuMemoryStrategy,
    /// Preferred precision mode
    pub precision_mode: GpuPrecisionMode,
    /// Enable multi-GPU if available
    pub multi_gpu_enabled: bool,
    /// Overlap computation with memory transfers
    pub overlap_compute_transfer: bool,
    /// Use streams for concurrent operations
    pub use_streams: bool,
    /// Number of streams to use
    pub num_streams: usize,
}

impl Default for GpuAccelerationStrategy {
    fn default() -> Self {
        Self {
            minsize_threshold: 512,
            max_single_gpusize: 50000,
            memory_strategy: GpuMemoryStrategy::Adaptive,
            precision_mode: GpuPrecisionMode::Double,
            multi_gpu_enabled: true,
            overlap_compute_transfer: true,
            use_streams: true,
            num_streams: 4,
        }
    }
}

/// GPU memory buffer abstraction
pub trait GpuBuffer<T>: Send + Sync + std::fmt::Debug {
    /// Get the size of the buffer in elements
    fn len(&self) -> usize;

    /// Check if the buffer is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy data from host to device
    fn copy_from_host(&mut self, data: &[T]) -> LinalgResult<()>;

    /// Copy data from device to host
    fn copy_to_host(&self, data: &mut [T]) -> LinalgResult<()>;

    /// Get device pointer (platform-specific)
    fn device_ptr(&self) -> *mut std::ffi::c_void;
}

/// GPU context abstraction for managing device state (dyn compatible)
pub trait GpuContext: Send + Sync + std::fmt::Debug {
    /// Get device information
    fn device_info(&self) -> &GpuDeviceInfo;

    /// Synchronize all operations
    fn synchronize(&self) -> LinalgResult<()>;

    /// Get available memory in bytes
    fn available_memory(&self) -> LinalgResult<usize>;

    /// Get total memory in bytes
    fn total_memory(&self) -> usize {
        self.device_info().total_memory
    }
}

/// GPU context with generic operations (separate trait for dyn compatibility)
pub trait GpuContextAlloc: GpuContext {
    /// Allocate buffer on GPU
    fn allocate_buffer<T: Clone + Send + Sync + Copy + 'static + std::fmt::Debug>(
        &self,
        size: usize,
    ) -> LinalgResult<Box<dyn GpuBuffer<T>>>;
}

/// GPU linear algebra operations trait
pub trait GpuLinearAlgebra<T>: Send + Sync
where
    T: Float + NumAssign + Zero + Send + Sync + Clone + Debug,
{
    /// Matrix-matrix multiplication: C = alpha * A * B + beta * C
    fn gemm(
        &self,
        a: &dyn GpuBuffer<T>,
        b: &dyn GpuBuffer<T>,
        c: &mut dyn GpuBuffer<T>,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        beta: T,
    ) -> LinalgResult<()>;

    /// Matrix-vector multiplication: y = alpha * A * x + beta * y
    fn gemv(
        &self,
        a: &dyn GpuBuffer<T>,
        x: &dyn GpuBuffer<T>,
        y: &mut dyn GpuBuffer<T>,
        m: usize,
        n: usize,
        alpha: T,
        beta: T,
    ) -> LinalgResult<()>;

    /// Element-wise operations
    fn elementwise_add(
        &self,
        a: &dyn GpuBuffer<T>,
        b: &dyn GpuBuffer<T>,
        result: &mut dyn GpuBuffer<T>,
        size: usize,
    ) -> LinalgResult<()>;

    /// Vector dot product
    fn dot(&self, a: &dyn GpuBuffer<T>, b: &dyn GpuBuffer<T>, size: usize) -> LinalgResult<T>;

    /// Matrix transpose
    fn transpose(
        &self,
        input: &dyn GpuBuffer<T>,
        output: &mut dyn GpuBuffer<T>,
        rows: usize,
        cols: usize,
    ) -> LinalgResult<()>;

    /// Cholesky decomposition
    fn cholesky(&self, matrix: &mut dyn GpuBuffer<T>, n: usize) -> LinalgResult<()>;

    /// LU decomposition with partial pivoting
    fn lu_decomposition(
        &self,
        matrix: &mut dyn GpuBuffer<T>,
        pivots: &mut dyn GpuBuffer<i32>,
        n: usize,
    ) -> LinalgResult<()>;

    /// QR decomposition
    fn qr_decomposition(
        &self,
        matrix: &mut dyn GpuBuffer<T>,
        q: &mut dyn GpuBuffer<T>,
        r: &mut dyn GpuBuffer<T>,
        m: usize,
        n: usize,
    ) -> LinalgResult<()>;

    /// Eigenvalue computation (simplified interface)
    fn eigenvalues(
        &self,
        matrix: &dyn GpuBuffer<T>,
        eigenvals: &mut dyn GpuBuffer<T>,
        n: usize,
    ) -> LinalgResult<()>;
}

/// GPU manager for coordinating multiple devices and strategies
pub struct GpuManager {
    /// Available GPU contexts
    devices: Vec<Box<dyn GpuContext>>,
    /// Current acceleration strategy
    strategy: GpuAccelerationStrategy,
    /// Performance profiles for each device
    #[allow(dead_code)]
    performance_profiles: std::collections::HashMap<usize, GpuPerformanceProfile>,
}

impl GpuManager {
    /// Create a new GPU manager
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
            strategy: GpuAccelerationStrategy::default(),
            performance_profiles: std::collections::HashMap::new(),
        }
    }

    /// Add a GPU device context
    pub fn add_device(&mut self, device: Box<dyn GpuContext>) {
        self.devices.push(device);
    }

    /// Get number of available devices
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get device information for a specific device
    pub fn device_info(&self, device_id: usize) -> Option<&GpuDeviceInfo> {
        self.devices.get(device_id).map(|d| d.device_info())
    }

    /// Set acceleration strategy
    pub fn set_strategy(&mut self, strategy: GpuAccelerationStrategy) {
        self.strategy = strategy;
    }

    /// Recommend GPU usage for a given matrix operation
    pub fn recommend_gpu_usage(
        &self,
        operation: GpuOperation,
        matrixsize: usize,
        data_typesize: usize,
    ) -> GpuRecommendation {
        let total_elements = matrixsize * matrixsize;
        let memory_required = total_elements * data_typesize;

        // Check if matrix is large enough for GPU acceleration
        if matrixsize < self.strategy.minsize_threshold {
            return GpuRecommendation::UseCpu {
                reason: "Matrix too small for GPU acceleration".to_string(),
            };
        }

        // Check if any GPU has enough memory
        let suitable_devices: Vec<usize> = self
            .devices
            .iter()
            .enumerate()
            .filter(|(_, device)| device.device_info().total_memory > memory_required)
            .map(|(idx, _)| idx)
            .collect();

        if suitable_devices.is_empty() {
            return GpuRecommendation::UseCpu {
                reason: "No GPU with sufficient memory".to_string(),
            };
        }

        // Select best device based on operation type and performance
        let best_device = self.select_best_device(&suitable_devices, operation);

        if matrixsize > self.strategy.max_single_gpusize && self.strategy.multi_gpu_enabled {
            GpuRecommendation::UseMultiGpu {
                devices: suitable_devices,
                primary_device: best_device,
                partition_strategy: MultiGpuPartition::RowWise,
            }
        } else {
            GpuRecommendation::UseSingleGpu {
                device_id: best_device,
                memory_strategy: self.strategy.memory_strategy,
                precision_mode: self.strategy.precision_mode,
            }
        }
    }

    /// Select the best device for a specific operation
    fn select_best_device(&self, candidates: &[usize], operation: GpuOperation) -> usize {
        // Simple heuristic: select device with most memory for now
        // In a full implementation, this would consider performance profiles
        candidates
            .iter()
            .max_by_key(|&&device_id| self.devices[device_id].device_info().total_memory)
            .copied()
            .unwrap_or(0)
    }

    /// Get performance statistics for all devices
    pub fn get_performance_stats(&self) -> Vec<GpuPerformanceStats> {
        self.devices
            .iter()
            .enumerate()
            .map(|(idx, device)| {
                GpuPerformanceStats {
                    device_id: idx,
                    device_info: device.device_info().clone(),
                    operations_per_second: 0.0, // Would be updated from real metrics
                    memory_utilization: 0.0,
                    compute_utilization: 0.0,
                    power_consumption: 0.0,
                }
            })
            .collect()
    }
}

/// GPU acceleration recommendation
#[derive(Debug, Clone)]
pub enum GpuRecommendation {
    /// Use CPU for computation
    UseCpu { reason: String },
    /// Use single GPU
    UseSingleGpu {
        device_id: usize,
        memory_strategy: GpuMemoryStrategy,
        precision_mode: GpuPrecisionMode,
    },
    /// Use multiple GPUs
    UseMultiGpu {
        devices: Vec<usize>,
        primary_device: usize,
        partition_strategy: MultiGpuPartition,
    },
}

/// Multi-GPU partitioning strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiGpuPartition {
    /// Split rows across devices
    RowWise,
    /// Split columns across devices
    ColumnWise,
    /// 2D block partitioning
    Block2D,
    /// Replicate data across devices
    Replicated,
}

/// GPU performance statistics
#[derive(Debug, Clone)]
pub struct GpuPerformanceStats {
    /// Device identifier
    pub device_id: usize,
    /// Device information
    pub device_info: GpuDeviceInfo,
    /// Operations per second achieved
    pub operations_per_second: f64,
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f64,
    /// Compute utilization (0.0 to 1.0)
    pub compute_utilization: f64,
    /// Power consumption in watts
    pub power_consumption: f64,
}

impl Default for GpuManager {
    fn default() -> Self {
        Self::new()
    }
}
pub trait GpuLinalgOps<T>: Send + Sync
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    /// GPU matrix-vector multiplication
    fn gpu_matvec(
        &self,
        ctx: &dyn GpuContext,
        a: &ArrayView2<T>,
        x: &ArrayView1<T>,
    ) -> LinalgResult<Array1<T>>;

    /// GPU matrix-matrix multiplication  
    fn gpu_matmul(
        &self,
        ctx: &dyn GpuContext,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
    ) -> LinalgResult<Array2<T>>;

    /// GPU vector dot product
    fn gpu_dot(
        &self,
        ctx: &dyn GpuContext,
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
    ) -> LinalgResult<T>;

    /// GPU vector norm computation
    fn gpu_norm(&self, ctx: &dyn GpuContext, x: &ArrayView1<T>) -> LinalgResult<T>;

    /// GPU element-wise operations
    fn gpu_elementwise_add(
        &self,
        ctx: &dyn GpuContext,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
    ) -> LinalgResult<Array2<T>>;

    /// GPU element-wise multiplication
    fn gpu_elementwise_mul(
        &self,
        ctx: &dyn GpuContext,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
    ) -> LinalgResult<Array2<T>>;
}

/// GPU backend factory for creating contexts
pub trait GpuBackend: Send + Sync {
    /// Get backend name
    fn name(&self) -> &str;

    /// Check if backend is available
    fn is_available(&self) -> bool;

    /// List available devices
    fn list_devices(&self) -> LinalgResult<Vec<GpuDeviceInfo>>;

    /// Create context for specified device
    fn create_context(&self, deviceid: usize) -> LinalgResult<Box<dyn GpuContext>>;

    /// Create context for best available device
    fn create_best_context(&self) -> LinalgResult<Box<dyn GpuContext>> {
        let devices = self.list_devices()?;
        if devices.is_empty() {
            return Err(LinalgError::ComputationError(
                "No GPU devices available".to_string(),
            ));
        }

        // Find device with most memory as a simple heuristic
        let best_device = devices
            .iter()
            .enumerate()
            .max_by_key(|(_, device)| device.total_memory)
            .map(|(idx, _)| idx)
            .unwrap();

        self.create_context(best_device)
    }
}

/// GPU manager for handling multiple backends
#[derive(Default)]
pub struct GpuBackendManager {
    backends: Vec<Box<dyn GpuBackend>>,
}

impl GpuBackendManager {
    /// Create a new GPU backend manager
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    /// Register a GPU backend
    pub fn register_backend(&mut self, backend: Box<dyn GpuBackend>) {
        if backend.is_available() {
            self.backends.push(backend);
        }
    }

    /// Get all available backends
    pub fn available_backends(&self) -> &[Box<dyn GpuBackend>] {
        &self.backends
    }

    /// Get backend by name
    pub fn get_backend(&self, name: &str) -> Option<&dyn GpuBackend> {
        self.backends
            .iter()
            .find(|backend| backend.name() == name)
            .map(|b| b.as_ref())
    }

    /// Create context using the best available backend
    pub fn create_best_context(&self) -> LinalgResult<Box<dyn GpuContext>> {
        if self.backends.is_empty() {
            return Err(LinalgError::ComputationError(
                "No GPU backends available".to_string(),
            ));
        }

        // Try backends in order of preference
        for backend in &self.backends {
            if let Ok(context) = backend.create_best_context() {
                return Ok(context);
            }
        }

        Err(LinalgError::ComputationError(
            "Failed to create GPU context with any backend".to_string(),
        ))
    }

    /// List all available devices across all backends
    pub fn list_all_devices(&self) -> LinalgResult<Vec<(String, Vec<GpuDeviceInfo>)>> {
        let mut all_devices = Vec::new();

        for backend in &self.backends {
            let devices = backend.list_devices()?;
            all_devices.push((backend.name().to_string(), devices));
        }

        Ok(all_devices)
    }
}

/// Initialize GPU manager with all available backends
#[allow(dead_code)]
pub fn initialize_gpu_manager() -> LinalgResult<GpuBackendManager> {
    let mut manager = GpuBackendManager::new();

    // Register CUDA backend if available
    #[cfg(feature = "cuda")]
    {
        if let Ok(cuda_backend) = backends::cuda::CudaBackend::new() {
            manager.register_backend(Box::new(cuda_backend));
        }
    }

    // Register OpenCL backend if available
    #[cfg(feature = "opencl")]
    {
        if let Ok(opencl_backend) = backends::opencl::OpenClBackend::new() {
            manager.register_backend(Box::new(opencl_backend));
        }
    }

    // Register ROCm backend if available
    #[cfg(feature = "rocm")]
    {
        if let Ok(rocm_backend) = backends::rocm::RocmBackend::new() {
            manager.register_backend(Box::new(rocm_backend));
        }
    }

    // Register Metal backend if available
    #[cfg(feature = "metal")]
    {
        if let Ok(metal_backend) = backends::metal::MetalBackend::new() {
            manager.register_backend(Box::new(metal_backend));
        }
    }

    Ok(manager)
}

/// Determine if GPU acceleration should be used based on problem size
#[allow(dead_code)]
pub fn should_use_gpu(
    matrix_elements: usize,
    threshold: usize,
    gpu_context: Option<&dyn GpuContext>,
) -> bool {
    // GPU is beneficial for larger problems and when GPU _context is available
    gpu_context.is_some() && matrix_elements > threshold
}

/// Auto-select between CPU and GPU based on problem characteristics
pub trait AutoGpuSelector<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    /// Automatically choose the best implementation for matrix-vector multiplication
    fn auto_matvec(
        &self,
        a: &ArrayView2<T>,
        x: &ArrayView1<T>,
        gpu_context: Option<&dyn GpuContext>,
    ) -> LinalgResult<Array1<T>>;

    /// Automatically choose the best implementation for matrix-matrix multiplication
    fn auto_matmul(
        &self,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
        gpu_context: Option<&dyn GpuContext>,
    ) -> LinalgResult<Array2<T>>;
}

/// Default thresholds for GPU usage (number of elements)
pub const DEFAULT_GPU_THRESHOLD_MATVEC: usize = 10_000;
pub const DEFAULT_GPU_THRESHOLD_MATMUL: usize = 100_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_manager_creation() {
        let manager = GpuBackendManager::new();
        assert_eq!(manager.available_backends().len(), 0);
    }

    #[test]
    fn test_should_use_gpu_threshold() {
        // Below threshold should not use GPU
        assert!(!should_use_gpu(100, 1000, None));

        // Above threshold but no GPU _context
        assert!(!should_use_gpu(2000, 1000, None));

        // Would use GPU if _context was available
        // (We can't test with actual _context without GPU backends)
    }

    #[test]
    fn test_gpu_device_type_equality() {
        assert_eq!(GpuDeviceType::Cuda, GpuDeviceType::Cuda);
        assert_ne!(GpuDeviceType::Cuda, GpuDeviceType::OpenCl);
    }
}
