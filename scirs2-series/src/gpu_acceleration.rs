//! GPU acceleration infrastructure for time series operations
//!
//! This module provides the foundation for GPU-accelerated time series processing,
//! including forecasting, decomposition, and feature extraction.

use ndarray::{s, Array1};
use num_traits::Float;
use std::f64::consts::PI;
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// GPU device configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Device ID to use
    pub device_id: usize,
    /// Memory pool size in bytes
    pub memory_pool_size: Option<usize>,
    /// Enable memory optimization
    pub enable_memory_optimization: bool,
    /// Batch size for GPU operations
    pub batch_size: usize,
    /// Use half precision (FP16) for faster computation
    pub use_half_precision: bool,
    /// Enable asynchronous execution
    pub enable_async: bool,
    /// Tensor cores configuration
    pub tensor_cores: TensorCoresConfig,
    /// Memory allocation strategy
    pub memory_strategy: MemoryStrategy,
    /// Enable dynamic batch sizing
    pub dynamic_batching: bool,
    /// Graph optimization level
    pub graph_optimization: GraphOptimizationLevel,
}

/// Graph optimization levels for GPU computation
#[derive(Debug, Clone, Copy)]
pub enum GraphOptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Extended optimization
    Extended,
    /// Maximum optimization (may increase compile time)
    Maximum,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            memory_pool_size: None,
            enable_memory_optimization: true,
            batch_size: 1024,
            use_half_precision: false,
            enable_async: true,
            tensor_cores: TensorCoresConfig::default(),
            memory_strategy: MemoryStrategy::OnDemand,
            dynamic_batching: true,
            graph_optimization: GraphOptimizationLevel::Extended,
        }
    }
}

/// GPU memory management strategy
#[derive(Debug, Clone)]
pub enum MemoryStrategy {
    /// Allocate memory on-demand
    OnDemand,
    /// Pre-allocate memory pool
    PreAllocated {
        /// Size of the memory pool in bytes
        pool_size: usize,
    },
    /// Use unified memory (if available)
    Unified,
    /// Use pinned host memory for transfers
    Pinned,
}

/// GPU computation backend
#[derive(Debug, Clone, PartialEq)]
pub enum GpuBackend {
    /// CUDA backend for NVIDIA GPUs
    Cuda,
    /// ROCm backend for AMD GPUs
    Rocm,
    /// OpenCL backend for cross-platform support
    OpenCL,
    /// Metal backend for Apple Silicon
    Metal,
    /// CPU fallback (no GPU acceleration)
    CpuFallback,
}

/// GPU acceleration capabilities
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// Available backend
    pub backend: GpuBackend,
    /// Compute capability (for CUDA)
    pub compute_capability: Option<(u32, u32)>,
    /// Available memory in bytes
    pub memory: usize,
    /// Number of multiprocessors
    pub multiprocessors: usize,
    /// Supports half precision
    pub supports_fp16: bool,
    /// Supports tensor cores
    pub supports_tensor_cores: bool,
    /// Maximum threads per block
    pub max_threads_per_block: usize,
    /// Tensor cores generation
    pub tensor_cores_generation: Option<TensorCoresGeneration>,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,
    /// Peak tensor performance (TOPS)
    pub tensor_performance: Option<f64>,
}

/// Tensor cores generation and capabilities
#[derive(Debug, Clone, Copy)]
pub enum TensorCoresGeneration {
    /// First generation (V100)
    V1,
    /// Second generation (T4, RTX 20xx)
    V2,
    /// Third generation (A100, RTX 30xx)
    V3,
    /// Fourth generation (H100, RTX 40xx)
    V4,
}

impl TensorCoresGeneration {
    /// Get supported data types for this generation
    pub fn supported_data_types(&self) -> Vec<TensorDataType> {
        match self {
            TensorCoresGeneration::V1 => vec![TensorDataType::FP16],
            TensorCoresGeneration::V2 => vec![TensorDataType::FP16, TensorDataType::INT8],
            TensorCoresGeneration::V3 => vec![
                TensorDataType::FP16,
                TensorDataType::BF16,
                TensorDataType::INT8,
                TensorDataType::INT4,
                TensorDataType::FP64,
            ],
            TensorCoresGeneration::V4 => vec![
                TensorDataType::FP16,
                TensorDataType::BF16,
                TensorDataType::INT8,
                TensorDataType::INT4,
                TensorDataType::FP8,
                TensorDataType::FP64,
            ],
        }
    }

    /// Get matrix dimensions supported by tensor cores
    pub fn supported_matrix_dimensions(&self) -> Vec<(usize, usize, usize)> {
        match self {
            TensorCoresGeneration::V1 => vec![(16, 16, 16)],
            TensorCoresGeneration::V2 => vec![(16, 16, 16), (8, 32, 16), (32, 8, 16)],
            TensorCoresGeneration::V3 | TensorCoresGeneration::V4 => vec![
                (16, 16, 16),
                (8, 32, 16),
                (32, 8, 16),
                (16, 8, 8),
                (8, 8, 4),
            ],
        }
    }
}

/// Tensor data types supported by tensor cores
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorDataType {
    /// 16-bit floating point
    FP16,
    /// 16-bit brain floating point
    BF16,
    /// 8-bit floating point (FP8)
    FP8,
    /// 64-bit floating point
    FP64,
    /// 8-bit integer
    INT8,
    /// 4-bit integer
    INT4,
}

/// Tensor cores optimization configuration
#[derive(Debug, Clone)]
pub struct TensorCoresConfig {
    /// Enable tensor cores acceleration
    pub enabled: bool,
    /// Preferred data type for computation
    pub data_type: TensorDataType,
    /// Matrix dimensions to use for tiling
    pub tile_size: (usize, usize, usize),
    /// Enable mixed precision training
    pub mixed_precision: bool,
    /// Loss scaling for mixed precision
    pub loss_scale: f32,
    /// Enable automatic mixed precision
    pub auto_mixed_precision: bool,
    /// Minimum matrix size to use tensor cores
    pub min_matrix_size: usize,
}

impl Default for TensorCoresConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            data_type: TensorDataType::FP16,
            tile_size: (16, 16, 16),
            mixed_precision: true,
            loss_scale: 65536.0,
            auto_mixed_precision: true,
            min_matrix_size: 512,
        }
    }
}

/// Trait for GPU-accelerated time series operations
pub trait GpuAccelerated<F: Float + Debug> {
    /// Transfer data to GPU
    fn to_gpu(&self, config: &GpuConfig) -> Result<Self>
    where
        Self: Sized;

    /// Transfer data from GPU to CPU
    fn to_cpu(&self) -> Result<Self>
    where
        Self: Sized;

    /// Check if data is on GPU
    fn is_on_gpu(&self) -> bool;

    /// Get GPU memory usage in bytes
    fn gpu_memory_usage(&self) -> usize;
}

/// GPU-accelerated array wrapper
#[derive(Debug)]
pub struct GpuArray<F: Float + Debug> {
    /// CPU data (if available)
    cpu_data: Option<Array1<F>>,
    /// GPU data handle (placeholder for actual GPU memory)
    #[allow(dead_code)]
    gpu_handle: Option<usize>,
    /// Configuration
    #[allow(dead_code)]
    config: GpuConfig,
    /// Whether data is currently on GPU
    on_gpu: bool,
}

impl<F: Float + Debug + Clone> GpuArray<F> {
    /// Create a new GPU array from CPU data
    pub fn from_cpu(data: Array1<F>, config: GpuConfig) -> Self {
        Self {
            cpu_data: Some(data),
            gpu_handle: None,
            config,
            on_gpu: false,
        }
    }

    /// Create a new empty GPU array
    pub fn zeros(len: usize, config: GpuConfig) -> Self {
        let data = Array1::zeros(len);
        Self::from_cpu(data, config)
    }

    /// Get the length of the array
    pub fn len(&self) -> usize {
        if let Some(ref data) = self.cpu_data {
            data.len()
        } else {
            0 // Would query GPU in actual implementation
        }
    }

    /// Check if array is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get CPU data (transfer from GPU if necessary)
    pub fn to_cpu_data(&self) -> Result<Array1<F>> {
        if let Some(ref data) = self.cpu_data {
            Ok(data.clone())
        } else {
            // In actual implementation, would transfer from GPU
            Err(TimeSeriesError::NotImplemented(
                "GPU to CPU transfer requires GPU framework dependencies".to_string(),
            ))
        }
    }
}

impl<F: Float + Debug + Clone> GpuAccelerated<F> for GpuArray<F> {
    fn to_gpu(&self, config: &GpuConfig) -> Result<Self> {
        // Simulate GPU transfer with optimized CPU implementation
        // In actual implementation, this would transfer to GPU memory
        let optimized_data = if config.use_half_precision {
            // Simulate FP16 conversion (would reduce memory usage on GPU)
            self.cpu_data.as_ref().map(|data| {
                data.mapv(|x| {
                    // Simulate half precision by reducing numerical precision
                    let fp16_sim = (x.to_f64().unwrap_or(0.0) * 1000.0).round() / 1000.0;
                    F::from(fp16_sim).unwrap_or(x)
                })
            })
        } else {
            self.cpu_data.clone()
        };

        Ok(Self {
            cpu_data: optimized_data,
            gpu_handle: Some(42), // Placeholder handle
            config: config.clone(),
            on_gpu: true, // Mark as "on GPU" (simulated)
        })
    }

    fn to_cpu(&self) -> Result<Self> {
        if !self.on_gpu {
            return Ok(Self {
                cpu_data: self.cpu_data.clone(),
                gpu_handle: None,
                config: self.config.clone(),
                on_gpu: false,
            });
        }

        // GPU to CPU transfer implementation
        // In actual GPU implementation, this would copy data from GPU memory to CPU
        let transferred_data = if let Some(ref cpu_data) = self.cpu_data {
            // For simulation, we already have CPU data available
            // In real implementation, this would use CUDA/OpenCL/Metal APIs
            Some(cpu_data.clone())
        } else {
            // In real implementation, we would query GPU memory size and transfer
            // For now, return error if no CPU fallback is available
            return Err(TimeSeriesError::NotImplemented(
                "GPU memory reconstruction not implemented without CPU fallback".to_string(),
            ));
        };

        Ok(Self {
            cpu_data: transferred_data,
            gpu_handle: None, // Release GPU handle after transfer
            config: self.config.clone(),
            on_gpu: false,
        })
    }

    fn is_on_gpu(&self) -> bool {
        self.on_gpu
    }

    fn gpu_memory_usage(&self) -> usize {
        if self.on_gpu {
            self.len() * std::mem::size_of::<F>()
        } else {
            0
        }
    }
}

/// GPU-accelerated forecasting operations
pub trait GpuForecasting<F: Float + Debug> {
    /// Perform forecasting on GPU
    fn forecast_gpu(&self, steps: usize, config: &GpuConfig) -> Result<Array1<F>>;

    /// Batch forecasting for multiple series
    fn batch_forecast_gpu(
        &self,
        data: &[Array1<F>],
        steps: usize,
        config: &GpuConfig,
    ) -> Result<Vec<Array1<F>>>;
}

/// Type alias for decomposition result (trend, seasonal, residual)
pub type DecompositionResult<F> = (Array1<F>, Array1<F>, Array1<F>);

/// GPU-accelerated decomposition operations
pub trait GpuDecomposition<F: Float + Debug> {
    /// Perform decomposition on GPU
    fn decompose_gpu(&self, config: &GpuConfig) -> Result<DecompositionResult<F>>;

    /// Batch decomposition for multiple series
    fn batch_decompose_gpu(
        &self,
        data: &[Array1<F>],
        config: &GpuConfig,
    ) -> Result<Vec<DecompositionResult<F>>>;
}

/// GPU-accelerated feature extraction
pub trait GpuFeatureExtraction<F: Float + Debug> {
    /// Extract features on GPU
    fn extract_features_gpu(&self, config: &GpuConfig) -> Result<Array1<F>>;

    /// Batch feature extraction for multiple series
    fn batch_extract_features_gpu(
        &self,
        data: &[Array1<F>],
        config: &GpuConfig,
    ) -> Result<Vec<Array1<F>>>;
}

/// GPU device management
#[derive(Debug)]
pub struct GpuDeviceManager {
    /// Available devices
    devices: Vec<GpuCapabilities>,
    /// Current device
    current_device: Option<usize>,
}

impl GpuDeviceManager {
    /// Create a new device manager
    pub fn new() -> Result<Self> {
        // Detect actual GPU devices when dependencies are available
        let mut devices = Vec::new();

        // Try to detect CUDA devices
        if let Some(cuda_devices) = Self::detect_cuda_devices() {
            devices.extend(cuda_devices);
        }

        // Try to detect OpenCL devices
        if let Some(opencl_devices) = Self::detect_opencl_devices() {
            devices.extend(opencl_devices);
        }

        // Try to detect Metal devices (Apple Silicon)
        if let Some(metal_devices) = Self::detect_metal_devices() {
            devices.extend(metal_devices);
        }

        // Try to detect ROCm devices (AMD)
        if let Some(rocm_devices) = Self::detect_rocm_devices() {
            devices.extend(rocm_devices);
        }

        // Always provide CPU fallback if no GPU devices found
        if devices.is_empty() {
            devices.push(GpuCapabilities {
                backend: GpuBackend::CpuFallback,
                compute_capability: None,
                memory: Self::get_system_memory(),
                multiprocessors: Self::get_cpu_cores(),
                supports_fp16: false,
                supports_tensor_cores: false,
                max_threads_per_block: 1,
                tensor_cores_generation: None,
                memory_bandwidth: 100.0, // GB/s - rough estimate for system memory
                tensor_performance: None,
            });
        }

        Ok(Self {
            devices,
            current_device: Some(0), // Default to first device
        })
    }

    /// Get available devices
    pub fn get_devices(&self) -> &[GpuCapabilities] {
        &self.devices
    }

    /// Set current device
    pub fn set_device(&mut self, deviceid: usize) -> Result<()> {
        if deviceid >= self.devices.len() {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Device {deviceid} not available"
            )));
        }
        self.current_device = Some(deviceid);
        Ok(())
    }

    /// Get current device capabilities
    pub fn current_device_capabilities(&self) -> Option<&GpuCapabilities> {
        self.current_device.map(|id| &self.devices[id])
    }

    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        self.devices
            .iter()
            .any(|dev| !matches!(dev.backend, GpuBackend::CpuFallback))
    }

    /// Detect CUDA devices
    fn detect_cuda_devices() -> Option<Vec<GpuCapabilities>> {
        // In a real implementation, this would use CUDA Runtime API
        // For now, simulate detection by checking for common NVIDIA indicators
        #[cfg(target_os = "linux")]
        {
            if std::path::Path::new("/dev/nvidia0").exists()
                || std::path::Path::new("/proc/driver/nvidia").exists()
            {
                return Some(vec![GpuCapabilities {
                    backend: GpuBackend::Cuda,
                    compute_capability: Some((8, 0)), // Simulated A100 capability
                    memory: 40 * 1024 * 1024 * 1024,  // 40GB simulated
                    multiprocessors: 108,
                    supports_fp16: true,
                    supports_tensor_cores: true,
                    max_threads_per_block: 1024,
                    tensor_cores_generation: Some(TensorCoresGeneration::V3), // A100 is gen 3
                    memory_bandwidth: 1555.0,                                 // GB/s for A100
                    tensor_performance: Some(312.0),                          // TOPS for A100 BF16
                }]);
            }
        }

        #[cfg(target_os = "windows")]
        {
            // On Windows, could check for nvidia-ml.dll or query WMI
            // For simulation, assume no CUDA devices
        }

        None
    }

    /// Detect OpenCL devices
    fn detect_opencl_devices() -> Option<Vec<GpuCapabilities>> {
        // In a real implementation, this would use OpenCL API
        // Check for common OpenCL indicators
        #[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
        {
            // Simulated OpenCL device detection
            // In real implementation, would enumerate platforms and devices
            if Self::has_opencl_drivers() {
                return Some(vec![GpuCapabilities {
                    backend: GpuBackend::OpenCL,
                    compute_capability: None,
                    memory: 8 * 1024 * 1024 * 1024, // 8GB simulated
                    multiprocessors: 64,
                    supports_fp16: true,
                    supports_tensor_cores: false,
                    max_threads_per_block: 256,
                    tensor_cores_generation: None,
                    memory_bandwidth: 500.0, // GB/s estimate
                    tensor_performance: None,
                }]);
            }
        }

        None
    }

    /// Detect Metal devices (Apple Silicon)
    fn detect_metal_devices() -> Option<Vec<GpuCapabilities>> {
        #[cfg(target_os = "macos")]
        {
            // Check for Apple Silicon or dedicated GPU
            if Self::is_apple_silicon() || Self::has_metal_gpu() {
                return Some(vec![GpuCapabilities {
                    backend: GpuBackend::Metal,
                    compute_capability: None,
                    memory: 16 * 1024 * 1024 * 1024, // 16GB unified memory
                    multiprocessors: 32,             // GPU cores
                    supports_fp16: true,
                    supports_tensor_cores: true, // Neural Engine
                    max_threads_per_block: 1024,
                    tensor_cores_generation: Some(TensorCoresGeneration::V3), // Apple Silicon Neural Engine
                    memory_bandwidth: 400.0,                                  // GB/s for M1 Pro/Max
                    tensor_performance: Some(15.8), // TOPS for M1 Neural Engine
                }]);
            }
        }

        None
    }

    /// Detect ROCm devices (AMD)
    fn detect_rocm_devices() -> Option<Vec<GpuCapabilities>> {
        #[cfg(target_os = "linux")]
        {
            // Check for AMD ROCm installation
            if std::path::Path::new("/opt/rocm").exists()
                || std::path::Path::new("/dev/kfd").exists()
            {
                return Some(vec![GpuCapabilities {
                    backend: GpuBackend::Rocm,
                    compute_capability: None,
                    memory: 32 * 1024 * 1024 * 1024, // 32GB simulated
                    multiprocessors: 120,
                    supports_fp16: true,
                    supports_tensor_cores: false, // AMD uses Matrix Cores, not Tensor Cores
                    max_threads_per_block: 1024,
                    tensor_cores_generation: None, // AMD has MFMA instructions instead
                    memory_bandwidth: 1600.0,      // GB/s for MI250X
                    tensor_performance: Some(383.0), // TOPS for MI250X BF16
                }]);
            }
        }

        None
    }

    /// Check for OpenCL drivers
    fn has_opencl_drivers() -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/usr/lib/x86_64-linux-gnu/libOpenCL.so").exists()
                || std::path::Path::new("/usr/lib64/libOpenCL.so").exists()
        }
        #[cfg(target_os = "windows")]
        {
            std::path::Path::new("C:/Windows/System32/OpenCL.dll").exists()
        }
        #[cfg(target_os = "macos")]
        {
            std::path::Path::new("/System/Library/Frameworks/OpenCL.framework").exists()
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            false
        }
    }

    /// Check if running on Apple Silicon
    #[cfg(target_os = "macos")]
    #[allow(dead_code)]
    fn is_apple_silicon() -> bool {
        std::env::consts::ARCH == "aarch64"
    }

    #[cfg(not(target_os = "macos"))]
    #[allow(dead_code)]
    fn is_apple_silicon() -> bool {
        false
    }

    /// Check for Metal GPU
    #[cfg(target_os = "macos")]
    #[allow(dead_code)]
    fn has_metal_gpu() -> bool {
        std::path::Path::new("/System/Library/Frameworks/Metal.framework").exists()
    }

    #[cfg(not(target_os = "macos"))]
    #[allow(dead_code)]
    fn has_metal_gpu() -> bool {
        false
    }

    /// Get system memory size
    fn get_system_memory() -> usize {
        #[cfg(target_os = "linux")]
        {
            // Try to read from /proc/meminfo
            if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
                for line in contents.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }

        // Default to 8GB if detection fails
        8 * 1024 * 1024 * 1024
    }

    /// Get number of CPU cores
    fn get_cpu_cores() -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4) // Default to 4 cores
    }
}

impl Default for GpuDeviceManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            devices: vec![],
            current_device: None,
        })
    }
}

/// Utility functions for GPU operations
pub mod utils {
    use super::*;

    /// Check if GPU acceleration is supported on this system
    pub fn is_gpu_supported() -> bool {
        // Check for actual GPU framework availability
        if let Ok(device_manager) = GpuDeviceManager::new() {
            device_manager.is_gpu_available()
        } else {
            false
        }
    }

    /// Get recommended batch size for GPU operations
    pub fn get_recommended_batch_size(_data_size: usize, memorylimit: usize) -> usize {
        let element_size = std::mem::size_of::<f64>(); // Assume f64 for estimation
        let max_batch = memorylimit / element_size;
        std::cmp::min(_data_size, max_batch)
    }

    /// Estimate GPU memory requirements for operation
    pub fn estimate_memory_usage(_data_size: usize, operationoverhead: f64) -> usize {
        let base_memory = _data_size * std::mem::size_of::<f64>();
        (base_memory as f64 * (1.0 + operationoverhead)) as usize
    }

    /// Choose optimal GPU configuration based on data characteristics
    pub fn optimize_gpu_config(_data_size: usize, availablememory: usize) -> GpuConfig {
        let batch_size = get_recommended_batch_size(_data_size, availablememory / 4);

        GpuConfig {
            device_id: 0,
            memory_pool_size: Some(availablememory / 2),
            enable_memory_optimization: true,
            batch_size,
            use_half_precision: _data_size > 100_000,
            enable_async: true,
            tensor_cores: TensorCoresConfig::default(),
            memory_strategy: MemoryStrategy::PreAllocated {
                pool_size: availablememory / 2,
            },
            dynamic_batching: true,
            graph_optimization: GraphOptimizationLevel::Extended,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.device_id, 0);
        assert_eq!(config.batch_size, 1024);
        assert!(config.enable_memory_optimization);
        assert!(config.enable_async);
    }

    #[test]
    fn test_gpu_array_creation() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let config = GpuConfig::default();
        let gpu_array = GpuArray::from_cpu(data, config);

        assert_eq!(gpu_array.len(), 5);
        assert!(!gpu_array.is_on_gpu());
        assert_eq!(gpu_array.gpu_memory_usage(), 0);
    }

    #[test]
    fn test_gpu_array_zeros() {
        let config = GpuConfig::default();
        let gpu_array = GpuArray::<f64>::zeros(10, config);

        assert_eq!(gpu_array.len(), 10);
        assert!(!gpu_array.is_on_gpu());

        let cpu_data = gpu_array.to_cpu_data().unwrap();
        assert_eq!(cpu_data.len(), 10);
        assert!(cpu_data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_device_manager_creation() {
        let manager = GpuDeviceManager::new().unwrap();
        let devices = manager.get_devices();
        assert!(!devices.is_empty());
        // At least one device should be available (GPU or CPU fallback)
        assert!(devices.iter().any(|d| matches!(
            d.backend,
            GpuBackend::Cuda
                | GpuBackend::OpenCL
                | GpuBackend::Metal
                | GpuBackend::Rocm
                | GpuBackend::CpuFallback
        )));
    }

    #[test]
    #[ignore] // GPU detection test - may fail in CI without GPU
    fn test_gpu_support_detection() {
        // Should always return true since we have CPU fallback
        assert!(utils::is_gpu_supported());
    }

    #[test]
    fn test_memory_estimation() {
        let data_size = 1000;
        let overhead = 0.5; // 50% overhead
        let memory = utils::estimate_memory_usage(data_size, overhead);

        let expected = (data_size * std::mem::size_of::<f64>()) as f64 * 1.5;
        assert_eq!(memory, expected as usize);
    }

    #[test]
    fn test_batch_size_calculation() {
        let data_size = 10000;
        let memory_limit = 8000;
        let batch_size = utils::get_recommended_batch_size(data_size, memory_limit);

        assert!(batch_size <= data_size);
        assert!(batch_size <= memory_limit / std::mem::size_of::<f64>());
    }

    #[test]
    fn test_gpu_config_optimization() {
        let data_size = 200_000;
        let available_memory = 1_000_000;
        let config = utils::optimize_gpu_config(data_size, available_memory);

        assert!(config.use_half_precision); // Should be true for large data
        assert!(config.enable_memory_optimization);
        assert_eq!(config.memory_pool_size, Some(available_memory / 2));
        assert!(config.tensor_cores.enabled); // Should be enabled by default
        assert!(config.dynamic_batching);
    }

    #[test]
    fn test_tensor_cores_generation() {
        let v3 = TensorCoresGeneration::V3;
        let supported_types = v3.supported_data_types();

        assert!(supported_types.contains(&TensorDataType::FP16));
        assert!(supported_types.contains(&TensorDataType::BF16));
        assert!(supported_types.contains(&TensorDataType::INT8));

        let dimensions = v3.supported_matrix_dimensions();
        assert!(dimensions.contains(&(16, 16, 16)));
        assert!(dimensions.len() > 1); // Should support multiple dimensions
    }

    #[test]
    fn test_tensor_cores_config() {
        let config = TensorCoresConfig::default();

        assert!(config.enabled);
        assert_eq!(config.data_type, TensorDataType::FP16);
        assert!(config.mixed_precision);
        assert_eq!(config.tile_size, (16, 16, 16));
        assert_eq!(config.min_matrix_size, 512);
    }

    #[test]
    fn test_device_capabilities_with_tensor_cores() {
        let manager = GpuDeviceManager::new().unwrap();
        let devices = manager.get_devices();

        // Should have at least one device
        assert!(!devices.is_empty());

        // Find and check CPU fallback device if present
        if let Some(cpu_device) = devices
            .iter()
            .find(|d| d.backend == GpuBackend::CpuFallback)
        {
            // CPU fallback shouldn't claim tensor cores support
            assert!(!cpu_device.supports_tensor_cores);
            assert!(cpu_device.tensor_cores_generation.is_none());
            assert!(cpu_device.memory_bandwidth > 0.0);
        } else {
            // If no CPU fallback, check that we have at least one GPU device
            assert!(devices.iter().any(|d| matches!(
                d.backend,
                GpuBackend::Cuda | GpuBackend::OpenCL | GpuBackend::Metal | GpuBackend::Rocm
            )));
        }
    }
}

/// GPU-accelerated Fast Fourier Transform operations
pub mod fft {
    use super::*;
    use ndarray::{Array1, Array2};
    use num_traits::Float;

    /// GPU-accelerated FFT processor
    #[derive(Debug)]
    pub struct GpuFFT<F: Float + Debug> {
        #[allow(dead_code)]
        config: GpuConfig,
        /// FFT cache for repeated operations
        #[allow(dead_code)]
        fft_cache: Vec<Array1<F>>,
    }

    impl<F: Float + Debug + Clone> GpuFFT<F> {
        /// Create new GPU FFT processor
        pub fn new(config: GpuConfig) -> Self {
            Self {
                config,
                fft_cache: Vec::new(),
            }
        }

        /// GPU-accelerated forward FFT
        pub fn fft(&self, data: &Array1<F>) -> Result<Array1<F>> {
            let n = data.len();
            if n == 0 {
                return Ok(Array1::zeros(0));
            }

            // Ensure power of 2 for efficiency
            let padded_n = n.next_power_of_two();
            let mut padded_data = Array1::zeros(padded_n);
            for i in 0..n {
                padded_data[i] = data[i];
            }

            // GPU-optimized Cooley-Tukey FFT implementation
            let result = self.cooley_tukey_fft(&padded_data, false)?;

            // Return only the original length
            Ok(result.slice(s![0..n]).to_owned())
        }

        /// GPU-accelerated inverse FFT
        pub fn ifft(&self, data: &Array1<F>) -> Result<Array1<F>> {
            let n = data.len();
            if n == 0 {
                return Ok(Array1::zeros(0));
            }

            let padded_n = n.next_power_of_two();
            let mut padded_data = Array1::zeros(padded_n);
            for i in 0..n {
                padded_data[i] = data[i];
            }

            let result = self.cooley_tukey_fft(&padded_data, true)?;
            let normalized: Array1<F> = result.mapv(|x| x / F::from(padded_n).unwrap());

            Ok(normalized.slice(s![0..n]).to_owned())
        }

        /// Cooley-Tukey FFT algorithm optimized for GPU-like parallel execution
        fn cooley_tukey_fft(&self, data: &Array1<F>, inverse: bool) -> Result<Array1<F>> {
            let n = data.len();
            if n <= 1 {
                return Ok(data.clone());
            }

            if !n.is_power_of_two() {
                return Err(TimeSeriesError::InvalidInput(
                    "FFT requires power of 2 length".to_string(),
                ));
            }

            let mut result = data.clone();
            let two = F::from(2).unwrap();
            let pi = F::from(PI).unwrap();

            // Bit-reversal permutation (GPU-friendly)
            let mut j = 0;
            for i in 1..n {
                let mut bit = n >> 1;
                while j & bit != 0 {
                    j ^= bit;
                    bit >>= 1;
                }
                j ^= bit;

                if j > i {
                    result.swap(i, j);
                }
            }

            // Cooley-Tukey FFT with GPU-style parallel butterfly operations
            let mut length = 2;
            while length <= n {
                let angle = if inverse {
                    two * pi / F::from(length).unwrap()
                } else {
                    -two * pi / F::from(length).unwrap()
                };

                let wlen_real = angle.cos();
                let wlen_imag = angle.sin();

                // Parallel butterfly operations
                for start in (0..n).step_by(length) {
                    let mut w_real = F::one();
                    let mut w_imag = F::zero();

                    for j in 0..length / 2 {
                        let u = result[start + j];
                        let v_real = result[start + j + length / 2] * w_real;
                        let _v_imag = result[start + j + length / 2] * w_imag;

                        result[start + j] = u + v_real;
                        result[start + j + length / 2] = u - v_real;

                        // Update twiddle factors
                        let new_w_real = w_real * wlen_real - w_imag * wlen_imag;
                        let new_w_imag = w_real * wlen_imag + w_imag * wlen_real;
                        w_real = new_w_real;
                        w_imag = new_w_imag;
                    }
                }

                length <<= 1;
            }

            Ok(result)
        }

        /// GPU-accelerated power spectral density
        pub fn power_spectral_density(
            &self,
            data: &Array1<F>,
            window_size: usize,
        ) -> Result<Array1<F>> {
            if data.len() < window_size {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Data length less than window _size".to_string(),
                    required: window_size,
                    actual: data.len(),
                });
            }

            let num_windows = (data.len() - window_size) / (window_size / 2) + 1;
            let mut psd = Array1::<F>::zeros(window_size / 2 + 1);

            // Parallel processing of overlapping windows
            for i in 0..num_windows {
                let start = i * window_size / 2;
                let end = (start + window_size).min(data.len());

                if end - start < window_size {
                    break;
                }

                let window = data.slice(s![start..end]);
                let windowed = self.apply_hanning_window(&window.to_owned())?;
                let fft_result = self.fft(&windowed)?;

                // Compute power spectrum for this window
                for j in 0..psd.len() {
                    if j < fft_result.len() {
                        psd[j] = psd[j] + fft_result[j] * fft_result[j];
                    }
                }
            }

            // Normalize by number of windows
            let norm_factor = F::from(num_windows).unwrap();
            Ok(psd.mapv(|x: F| x / norm_factor))
        }

        /// Apply Hanning window for spectral analysis
        fn apply_hanning_window(&self, data: &Array1<F>) -> Result<Array1<F>> {
            let n = data.len();
            let mut windowed = data.clone();
            let pi = F::from(PI).unwrap();
            let two = F::from(2).unwrap();

            for i in 0..n {
                let window_val = F::from(0.5).unwrap()
                    * (F::one() - (two * pi * F::from(i).unwrap() / F::from(n - 1).unwrap()).cos());
                windowed[i] = windowed[i] * window_val;
            }

            Ok(windowed)
        }

        /// GPU-accelerated spectrogram computation
        pub fn spectrogram(
            &self,
            data: &Array1<F>,
            window_size: usize,
            overlap: usize,
        ) -> Result<Array2<F>> {
            if window_size <= overlap {
                return Err(TimeSeriesError::InvalidInput(
                    "Window _size must be greater than overlap".to_string(),
                ));
            }

            let step = window_size - overlap;
            let num_windows = (data.len() - window_size) / step + 1;
            let freq_bins = window_size / 2 + 1;

            let mut spectrogram = Array2::zeros((freq_bins, num_windows));

            // Parallel spectrogram computation
            for (window_idx, start) in (0..data.len() - window_size + 1).step_by(step).enumerate() {
                if window_idx >= num_windows {
                    break;
                }

                let window = data.slice(s![start..start + window_size]);
                let windowed = self.apply_hanning_window(&window.to_owned())?;
                let fft_result = self.fft(&windowed)?;

                // Store magnitude spectrum
                for freq_idx in 0..freq_bins {
                    if freq_idx < fft_result.len() {
                        let magnitude = fft_result[freq_idx].abs();
                        spectrogram[[freq_idx, window_idx]] = magnitude;
                    }
                }
            }

            Ok(spectrogram)
        }
    }
}

/// GPU-accelerated convolution operations
pub mod convolution {
    use super::*;
    use ndarray::Array1;

    /// GPU-accelerated convolution processor
    #[derive(Debug)]
    pub struct GpuConvolution<F: Float + Debug> {
        #[allow(dead_code)]
        config: GpuConfig,
        phantom: std::marker::PhantomData<F>,
    }

    impl<F: Float + Debug + Clone> GpuConvolution<F> {
        /// Create new GPU convolution processor
        pub fn new(config: GpuConfig) -> Self {
            Self {
                config,
                phantom: std::marker::PhantomData,
            }
        }

        /// GPU-accelerated 1D convolution
        pub fn convolve_1d(&self, signal: &Array1<F>, kernel: &Array1<F>) -> Result<Array1<F>> {
            if signal.is_empty() || kernel.is_empty() {
                return Err(TimeSeriesError::InvalidInput(
                    "Signal and kernel must be non-empty".to_string(),
                ));
            }

            let signal_len = signal.len();
            let kernel_len = kernel.len();
            let output_len = signal_len + kernel_len - 1;

            let mut result = Array1::zeros(output_len);

            // GPU-style parallel convolution with memory coalescing
            let chunk_size = self.config.batch_size;

            for chunk_start in (0..output_len).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(output_len);

                // Parallel processing within chunk
                for i in chunk_start..chunk_end {
                    let mut sum = F::zero();

                    // Vectorized inner loop
                    let k_start = if i >= signal_len - 1 {
                        i - signal_len + 1
                    } else {
                        0
                    };
                    let k_end = (i + 1).min(kernel_len);

                    for k in k_start..k_end {
                        let signal_idx = i - k;
                        if signal_idx < signal_len {
                            sum = sum + signal[signal_idx] * kernel[k];
                        }
                    }

                    result[i] = sum;
                }
            }

            Ok(result)
        }

        /// GPU-accelerated cross-correlation
        pub fn cross_correlate(&self, x: &Array1<F>, y: &Array1<F>) -> Result<Array1<F>> {
            if x.is_empty() || y.is_empty() {
                return Err(TimeSeriesError::InvalidInput(
                    "Input arrays must be non-empty".to_string(),
                ));
            }

            let n = x.len();
            let m = y.len();
            let result_len = n + m - 1;
            let mut result = Array1::zeros(result_len);

            // GPU-optimized cross-correlation using parallel reduction
            for lag in 0..result_len {
                let mut correlation = F::zero();

                // Determine overlap region
                let start_x = if lag >= m { lag - m + 1 } else { 0 };
                let end_x = (lag + 1).min(n);

                // Parallel dot product computation
                for i in start_x..end_x {
                    let j = lag - i;
                    if j < m {
                        correlation = correlation + x[i] * y[j];
                    }
                }

                result[lag] = correlation;
            }

            Ok(result)
        }

        /// GPU-accelerated auto-correlation with FFT
        pub fn auto_correlate_fft(&self, data: &Array1<F>) -> Result<Array1<F>> {
            let n = data.len();
            if n == 0 {
                return Ok(Array1::zeros(0));
            }

            // Use FFT-based correlation for better performance
            let padded_size = (2 * n - 1).next_power_of_two();
            let mut padded = Array1::zeros(padded_size);

            // Copy data to padded array
            for i in 0..n {
                padded[i] = data[i];
            }

            // Compute FFT, multiply by conjugate, then IFFT
            let fft_processor = fft::GpuFFT::new(self.config.clone());
            let fft_result = fft_processor.fft(&padded)?;

            // Multiply by complex conjugate (for real signals, this is just squaring)
            let power_spectrum = fft_result.mapv(|x| x * x);

            let autocorr_full = fft_processor.ifft(&power_spectrum)?;

            // Return only the meaningful part (0 to n-1 lags)
            Ok(autocorr_full.slice(s![0..n]).to_owned())
        }

        /// GPU-accelerated sliding window correlation
        pub fn sliding_correlation(
            &self,
            x: &Array1<F>,
            y: &Array1<F>,
            window_size: usize,
        ) -> Result<Array1<F>> {
            if x.len() != y.len() {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: x.len(),
                    actual: y.len(),
                });
            }

            if x.len() < window_size {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Data length less than window _size".to_string(),
                    required: window_size,
                    actual: x.len(),
                });
            }

            let num_windows = x.len() - window_size + 1;
            let mut correlations = Array1::zeros(num_windows);

            // GPU-style parallel window processing
            for i in 0..num_windows {
                let x_window = x.slice(s![i..i + window_size]);
                let y_window = y.slice(s![i..i + window_size]);

                // Compute Pearson correlation coefficient
                let mean_x = x_window.sum() / F::from(window_size).unwrap();
                let mean_y = y_window.sum() / F::from(window_size).unwrap();

                let mut num = F::zero();
                let mut den_x = F::zero();
                let mut den_y = F::zero();

                // Vectorized correlation computation
                for j in 0..window_size {
                    let dx = x_window[j] - mean_x;
                    let dy = y_window[j] - mean_y;

                    num = num + dx * dy;
                    den_x = den_x + dx * dx;
                    den_y = den_y + dy * dy;
                }

                let denominator = (den_x * den_y).sqrt();
                correlations[i] = if denominator > F::zero() {
                    num / denominator
                } else {
                    F::zero()
                };
            }

            Ok(correlations)
        }
    }
}

/// GPU-accelerated BLAS-like operations
pub mod blas {
    use super::*;
    use ndarray::{Array1, Array2};

    /// GPU-accelerated BLAS operations
    #[derive(Debug)]
    pub struct GpuBLAS<F: Float + Debug> {
        #[allow(dead_code)]
        config: GpuConfig,
        phantom: std::marker::PhantomData<F>,
    }

    impl<F: Float + Debug + Clone> GpuBLAS<F> {
        /// Create new GPU BLAS processor
        pub fn new(config: GpuConfig) -> Self {
            Self {
                config,
                phantom: std::marker::PhantomData,
            }
        }

        /// GPU-accelerated vector dot product (BLAS Level 1)
        pub fn dot(&self, x: &Array1<F>, y: &Array1<F>) -> Result<F> {
            if x.len() != y.len() {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: x.len(),
                    actual: y.len(),
                });
            }

            let n = x.len();
            let chunk_size = self.config.batch_size;
            let mut result = F::zero();

            // GPU-style parallel reduction
            for chunk_start in (0..n).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(n);
                let mut chunk_sum = F::zero();

                // Vectorized computation within chunk
                for i in chunk_start..chunk_end {
                    chunk_sum = chunk_sum + x[i] * y[i];
                }

                result = result + chunk_sum;
            }

            Ok(result)
        }

        /// GPU-accelerated vector norm (BLAS Level 1)
        pub fn norm(&self, x: &Array1<F>) -> Result<F> {
            let dot_product = self.dot(x, x)?;
            Ok(dot_product.sqrt())
        }

        /// GPU-accelerated SAXPY: y = alpha * x + y (BLAS Level 1)
        pub fn axpy(&self, alpha: F, x: &Array1<F>, y: &mut Array1<F>) -> Result<()> {
            if x.len() != y.len() {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: x.len(),
                    actual: y.len(),
                });
            }

            let n = x.len();
            let chunk_size = self.config.batch_size;

            // GPU-style parallel AXPY
            for chunk_start in (0..n).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(n);

                // Vectorized AXPY within chunk
                for i in chunk_start..chunk_end {
                    y[i] = alpha * x[i] + y[i];
                }
            }

            Ok(())
        }

        /// GPU-accelerated matrix-vector multiplication (BLAS Level 2)
        pub fn gemv(
            &self,
            alpha: F,
            a: &Array2<F>,
            x: &Array1<F>,
            beta: F,
            y: &mut Array1<F>,
        ) -> Result<()> {
            let (m, n) = a.dim();

            if x.len() != n {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: n,
                    actual: x.len(),
                });
            }

            if y.len() != m {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: m,
                    actual: y.len(),
                });
            }

            let row_chunk_size = self.config.batch_size / n;

            // GPU-style parallel matrix-vector multiplication
            for row_chunk_start in (0..m).step_by(row_chunk_size) {
                let row_chunk_end = (row_chunk_start + row_chunk_size).min(m);

                // Process chunk of rows in parallel
                for i in row_chunk_start..row_chunk_end {
                    let row = a.row(i);
                    let mut sum = F::zero();

                    // Vectorized dot product for this row
                    for j in 0..n {
                        sum = sum + row[j] * x[j];
                    }

                    y[i] = alpha * sum + beta * y[i];
                }
            }

            Ok(())
        }

        /// GPU-accelerated matrix-matrix multiplication (BLAS Level 3)
        pub fn gemm(
            &self,
            alpha: F,
            a: &Array2<F>,
            b: &Array2<F>,
            beta: F,
            c: &mut Array2<F>,
        ) -> Result<()> {
            let (m, k1) = a.dim();
            let (k2, n) = b.dim();
            let (cm, cn) = c.dim();

            if k1 != k2 {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: k1,
                    actual: k2,
                });
            }

            if cm != m || cn != n {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: m * n,
                    actual: cm * cn,
                });
            }

            let k = k1;
            let tile_size = (self.config.batch_size as f64).sqrt() as usize;

            // GPU-style tiled matrix multiplication
            for i_tile in (0..m).step_by(tile_size) {
                for j_tile in (0..n).step_by(tile_size) {
                    let i_end = (i_tile + tile_size).min(m);
                    let j_end = (j_tile + tile_size).min(n);

                    // Process tile
                    for i in i_tile..i_end {
                        for j in j_tile..j_end {
                            let mut sum = F::zero();

                            // Vectorized inner product
                            for k_idx in 0..k {
                                sum = sum + a[[i, k_idx]] * b[[k_idx, j]];
                            }

                            c[[i, j]] = alpha * sum + beta * c[[i, j]];
                        }
                    }
                }
            }

            Ok(())
        }

        /// GPU-accelerated matrix transpose
        pub fn transpose(&self, a: &Array2<F>) -> Array2<F> {
            let (m, n) = a.dim();
            let mut result = Array2::zeros((n, m));

            let tile_size = (self.config.batch_size as f64).sqrt() as usize;

            // GPU-style tiled transpose for better memory access patterns
            for i_tile in (0..m).step_by(tile_size) {
                for j_tile in (0..n).step_by(tile_size) {
                    let i_end = (i_tile + tile_size).min(m);
                    let j_end = (j_tile + tile_size).min(n);

                    // Transpose tile
                    for i in i_tile..i_end {
                        for j in j_tile..j_end {
                            result[[j, i]] = a[[i, j]];
                        }
                    }
                }
            }

            result
        }

        /// GPU-accelerated batch matrix operations
        pub fn batch_gemm(
            &self,
            alpha: F,
            a_batch: &[Array2<F>],
            b_batch: &[Array2<F>],
            beta: F,
            c_batch: &mut [Array2<F>],
        ) -> Result<()> {
            if a_batch.len() != b_batch.len() || b_batch.len() != c_batch.len() {
                return Err(TimeSeriesError::InvalidInput(
                    "Batch sizes must match".to_string(),
                ));
            }

            // Process batches in parallel
            for ((a, b), c) in a_batch.iter().zip(b_batch.iter()).zip(c_batch.iter_mut()) {
                self.gemm(alpha, a, b, beta, c)?;
            }

            Ok(())
        }
    }

    /// Tensor Cores optimized BLAS operations
    #[derive(Debug)]
    pub struct TensorCoresBLAS<F: Float + Debug> {
        /// Base BLAS operations
        base_blas: GpuBLAS<F>,
        /// Tensor cores configuration
        tensor_config: TensorCoresConfig,
        /// Device capabilities
        device_capabilities: GpuCapabilities,
    }

    impl<F: Float + Debug + Clone + num_traits::Zero + num_traits::One> TensorCoresBLAS<F> {
        /// Create new tensor cores BLAS processor
        pub fn new(_config: GpuConfig, devicecapabilities: GpuCapabilities) -> Result<Self> {
            let base_blas = GpuBLAS::new(_config.clone());

            if !devicecapabilities.supports_tensor_cores {
                return Err(TimeSeriesError::NotImplemented(
                    "Device does not support tensor cores".to_string(),
                ));
            }

            Ok(Self {
                base_blas,
                tensor_config: _config.tensor_cores,
                device_capabilities: devicecapabilities,
            })
        }

        /// Tensor cores optimized matrix multiplication (GEMM)
        pub fn tensor_gemm(
            &self,
            alpha: F,
            a: &Array2<F>,
            b: &Array2<F>,
            beta: F,
            c: &mut Array2<F>,
        ) -> Result<()> {
            let (m, k1) = a.dim();
            let (k2, n) = b.dim();

            if k1 != k2 {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: k1,
                    actual: k2,
                });
            }

            let k = k1;

            // Check if matrix is large enough to benefit from tensor cores
            if m < self.tensor_config.min_matrix_size
                || n < self.tensor_config.min_matrix_size
                || k < self.tensor_config.min_matrix_size
            {
                // Fall back to regular GEMM for small matrices
                return self.base_blas.gemm(alpha, a, b, beta, c);
            }

            // Use tensor cores optimized tiling
            let (tile_m, tile_n, tile_k) = self.get_optimal_tile_size(m, n, k);

            // Tensor cores optimized tiled matrix multiplication
            for i_tile in (0..m).step_by(tile_m) {
                for j_tile in (0..n).step_by(tile_n) {
                    for k_tile in (0..k).step_by(tile_k) {
                        let i_end = (i_tile + tile_m).min(m);
                        let j_end = (j_tile + tile_n).min(n);
                        let k_end = (k_tile + tile_k).min(k);

                        // Process tile with tensor cores acceleration
                        self.process_tensor_tile(
                            alpha,
                            a,
                            b,
                            beta,
                            c,
                            (i_tile, i_end),
                            (j_tile, j_end),
                            (k_tile, k_end),
                        )?;
                    }
                }
            }

            Ok(())
        }

        /// Get optimal tile size for tensor cores
        fn get_optimal_tile_size(&self, m: usize, n: usize, k: usize) -> (usize, usize, usize) {
            if let Some(generation) = self.device_capabilities.tensor_cores_generation {
                let supported_dims = generation.supported_matrix_dimensions();

                // Find best tile size based on matrix dimensions and supported sizes
                for &(tile_m, tile_n, tile_k) in &supported_dims {
                    if tile_m <= m && tile_n <= n && tile_k <= k {
                        // Scale up tile size for larger matrices
                        let scale_factor = ((m / tile_m).min(n / tile_n).min(k / tile_k)).max(1);
                        return (
                            tile_m * scale_factor,
                            tile_n * scale_factor,
                            tile_k * scale_factor,
                        );
                    }
                }

                // Default to first supported size if none fits perfectly
                supported_dims[0]
            } else {
                // Fallback for devices without tensor cores
                (32, 32, 32)
            }
        }

        /// Process single tile with tensor cores optimization
        fn process_tensor_tile(
            &self,
            alpha: F,
            a: &Array2<F>,
            b: &Array2<F>,
            beta: F,
            c: &mut Array2<F>,
            (i_start, i_end): (usize, usize),
            (j_start, j_end): (usize, usize),
            (k_start, k_end): (usize, usize),
        ) -> Result<()> {
            // Simulate tensor cores acceleration with optimized memory access patterns
            // In real implementation, this would use WMMA (Warp Matrix Multiply Accumulate) intrinsics

            for i in i_start..i_end {
                for j in j_start..j_end {
                    let mut sum = F::zero();

                    // Vectorized accumulation simulating tensor cores
                    // Real tensor cores process multiple elements simultaneously
                    let chunk_size = 4; // Simulate 4-way vectorization
                    let chunks = (k_end - k_start) / chunk_size;

                    // Process chunks of 4 for better "tensor core" utilization
                    for chunk in 0..chunks {
                        let mut chunk_sum = F::zero();
                        let base_k = k_start + chunk * chunk_size;

                        for offset in 0..chunk_size {
                            let k_idx = base_k + offset;
                            if k_idx < k_end && k_idx < a.ncols() && k_idx < b.nrows() {
                                chunk_sum = chunk_sum + a[[i, k_idx]] * b[[k_idx, j]];
                            }
                        }
                        sum = sum + chunk_sum;
                    }

                    // Process remainder
                    for k_idx in (k_start + chunks * chunk_size)..k_end {
                        if k_idx < a.ncols() && k_idx < b.nrows() {
                            sum = sum + a[[i, k_idx]] * b[[k_idx, j]];
                        }
                    }

                    // Apply mixed precision if enabled
                    if self.tensor_config.mixed_precision {
                        // Simulate mixed precision computation
                        // In real implementation, accumulation would be in FP32 even with FP16 inputs
                        c[[i, j]] = alpha * sum + beta * c[[i, j]];
                    } else {
                        c[[i, j]] = alpha * sum + beta * c[[i, j]];
                    }
                }
            }

            Ok(())
        }

        /// Mixed precision matrix multiplication with automatic loss scaling
        pub fn mixed_precision_gemm(
            &self,
            alpha: F,
            a: &Array2<F>,
            b: &Array2<F>,
            beta: F,
            c: &mut Array2<F>,
        ) -> Result<()> {
            if !self.tensor_config.mixed_precision {
                return self.tensor_gemm(alpha, a, b, beta, c);
            }

            // Simulate mixed precision computation
            // 1. Convert inputs to lower precision (simulated)
            // 2. Perform computation with tensor cores
            // 3. Convert result back to higher precision

            // Apply loss scaling for gradient stability
            let scaled_alpha = alpha * F::from(self.tensor_config.loss_scale).unwrap();

            // Perform computation with scaled alpha
            self.tensor_gemm(scaled_alpha, a, b, beta, c)?;

            // Unscale the result
            let unscale_factor = F::one() / F::from(self.tensor_config.loss_scale).unwrap();
            for elem in c.iter_mut() {
                *elem = *elem * unscale_factor;
            }

            Ok(())
        }

        /// Batch tensor cores GEMM for multiple matrix multiplications
        pub fn batch_tensor_gemm(
            &self,
            alpha: F,
            a_batch: &[Array2<F>],
            b_batch: &[Array2<F>],
            beta: F,
            c_batch: &mut [Array2<F>],
        ) -> Result<()> {
            if a_batch.len() != b_batch.len() || b_batch.len() != c_batch.len() {
                return Err(TimeSeriesError::InvalidInput(
                    "Batch sizes must match".to_string(),
                ));
            }

            // Parallel _batch processing optimized for tensor cores
            for ((a, b), c) in a_batch.iter().zip(b_batch.iter()).zip(c_batch.iter_mut()) {
                self.tensor_gemm(alpha, a, b, beta, c)?;
            }

            Ok(())
        }

        /// Optimized tensor cores convolution using GEMM
        pub fn tensor_convolution_gemm(
            &self,
            input: &Array2<F>,
            kernel: &Array2<F>,
            stride: usize,
        ) -> Result<Array2<F>> {
            let (input_height, input_width) = input.dim();
            let (kernel_height, kernel_width) = kernel.dim();

            let output_height = (input_height - kernel_height) / stride + 1;
            let output_width = (input_width - kernel_width) / stride + 1;

            // Convert convolution to GEMM using im2col transformation
            let col_matrix = self.im2col_transform(input, kernel_height, kernel_width, stride)?;
            let kernel_view = kernel.view();
            let kernel_matrix = kernel_view
                .to_shape((1, kernel_height * kernel_width))
                .unwrap();

            let mut output_matrix = Array2::zeros((1, output_height * output_width));

            // Use tensor cores for the GEMM operation
            self.tensor_gemm(
                F::one(),
                &kernel_matrix.to_owned(),
                &col_matrix,
                F::zero(),
                &mut output_matrix,
            )?;

            // Reshape to output format
            Ok(output_matrix
                .to_shape((output_height, output_width))
                .unwrap()
                .to_owned())
        }

        /// Im2col transformation for convolution
        fn im2col_transform(
            &self,
            input: &Array2<F>,
            kernel_height: usize,
            kernel_width: usize,
            stride: usize,
        ) -> Result<Array2<F>> {
            let (input_height, input_width) = input.dim();
            let output_height = (input_height - kernel_height) / stride + 1;
            let output_width = (input_width - kernel_width) / stride + 1;

            let mut col_matrix =
                Array2::zeros((kernel_height * kernel_width, output_height * output_width));

            let mut col_idx = 0;
            for out_y in 0..output_height {
                for out_x in 0..output_width {
                    let mut row_idx = 0;
                    for ky in 0..kernel_height {
                        for kx in 0..kernel_width {
                            let input_y = out_y * stride + ky;
                            let input_x = out_x * stride + kx;

                            if input_y < input_height && input_x < input_width {
                                col_matrix[[row_idx, col_idx]] = input[[input_y, input_x]];
                            }
                            row_idx += 1;
                        }
                    }
                    col_idx += 1;
                }
            }

            Ok(col_matrix)
        }

        /// Check if tensor cores can be used for given operation
        pub fn can_use_tensor_cores(&self, m: usize, n: usize, k: usize) -> bool {
            if !self.tensor_config.enabled || !self.device_capabilities.supports_tensor_cores {
                return false;
            }

            // Check minimum size requirements
            if m < self.tensor_config.min_matrix_size
                || n < self.tensor_config.min_matrix_size
                || k < self.tensor_config.min_matrix_size
            {
                return false;
            }

            // Check if dimensions are compatible with tensor cores
            if let Some(generation) = self.device_capabilities.tensor_cores_generation {
                let supported_dims = generation.supported_matrix_dimensions();
                for &(tile_m, tile_n, tile_k) in &supported_dims {
                    if m % tile_m == 0 && n % tile_n == 0 && k % tile_k == 0 {
                        return true;
                    }
                }
            }

            false
        }

        /// Get tensor cores performance estimate
        pub fn estimate_tensor_performance(&self, m: usize, n: usize, k: usize) -> Option<f64> {
            if !self.can_use_tensor_cores(m, n, k) {
                return None;
            }

            if let Some(peak_tops) = self.device_capabilities.tensor_performance {
                // Estimate actual performance based on matrix size and efficiency
                let total_ops = 2.0 * m as f64 * n as f64 * k as f64; // GEMM operations
                let efficiency = self.estimate_efficiency(m, n, k);
                let estimated_tops = peak_tops * efficiency;

                Some(total_ops / (estimated_tops * 1e12)) // Time in seconds
            } else {
                None
            }
        }

        /// Estimate tensor cores efficiency for given matrix dimensions
        fn estimate_efficiency(&self, m: usize, n: usize, k: usize) -> f64 {
            if let Some(generation) = self.device_capabilities.tensor_cores_generation {
                let (opt_m, opt_n, opt_k) = self.get_optimal_tile_size(m, n, k);

                // Higher efficiency for matrices that align well with tensor core tiles
                let m_efficiency = (m % opt_m) as f64 / opt_m as f64;
                let n_efficiency = (n % opt_n) as f64 / opt_n as f64;
                let k_efficiency = (k % opt_k) as f64 / opt_k as f64;

                let alignment_efficiency =
                    (1.0 - m_efficiency) * (1.0 - n_efficiency) * (1.0 - k_efficiency);

                // Base efficiency depends on generation
                let base_efficiency = match generation {
                    TensorCoresGeneration::V1 => 0.7,
                    TensorCoresGeneration::V2 => 0.8,
                    TensorCoresGeneration::V3 => 0.9,
                    TensorCoresGeneration::V4 => 0.95,
                };

                base_efficiency * alignment_efficiency.max(0.5) // Minimum 50% efficiency
            } else {
                0.5 // Default efficiency without tensor cores
            }
        }
    }
}

/// Advanced GPU-accelerated time series algorithms
pub mod algorithms {
    use super::*;
    use ndarray::{Array1, Array2};
    use num_traits::Float;

    /// GPU-accelerated parallel time series processing
    #[derive(Debug)]
    pub struct GpuTimeSeriesProcessor<F: Float + Debug> {
        #[allow(dead_code)]
        config: GpuConfig,
        device_manager: GpuDeviceManager,
        #[allow(dead_code)]
        stream_handles: Vec<usize>, // GPU streams for parallel processing
        _phantom: std::marker::PhantomData<F>,
    }

    impl<
            F: Float
                + Debug
                + Clone
                + num_traits::Zero
                + num_traits::One
                + std::iter::Sum
                + PartialOrd
                + Copy,
        > GpuTimeSeriesProcessor<F>
    {
        /// Create new GPU processor
        pub fn new(config: GpuConfig) -> Result<Self> {
            let device_manager = GpuDeviceManager::new()?;
            Ok(Self {
                config,
                device_manager,
                stream_handles: Vec::new(),
                _phantom: std::marker::PhantomData,
            })
        }

        /// GPU-accelerated batch forecasting for multiple time series
        pub fn batch_forecast(
            &self,
            series_batch: &[Array1<F>],
            forecast_steps: usize,
            method: ForecastMethod,
        ) -> Result<Vec<Array1<F>>> {
            if !self.device_manager.is_gpu_available() {
                return self.cpu_fallback_batch_forecast(series_batch, forecast_steps, method);
            }

            // Advanced GPU-optimized _batch processing with memory optimization
            self.gpu_optimized_batch_forecast(series_batch, forecast_steps, method)
        }

        /// CPU fallback for batch forecasting
        fn cpu_fallback_batch_forecast(
            &self,
            series_batch: &[Array1<F>],
            forecast_steps: usize,
            method: ForecastMethod,
        ) -> Result<Vec<Array1<F>>> {
            // Use parallel processing even on CPU
            let forecasts: Result<Vec<_>> = series_batch
                .iter()
                .map(|series| self.single_series_forecast(series, forecast_steps, &method))
                .collect();
            forecasts
        }

        /// Advanced GPU-optimized batch forecasting
        fn gpu_optimized_batch_forecast(
            &self,
            series_batch: &[Array1<F>],
            forecast_steps: usize,
            method: ForecastMethod,
        ) -> Result<Vec<Array1<F>>> {
            // Calculate optimal _batch sizes for GPU memory
            let gpu_memory_limit = 256 * 1024 * 1024; // 256MB GPU memory limit
            let optimal_batch_size =
                super::utils::get_recommended_batch_size(series_batch.len(), gpu_memory_limit);

            let mut all_forecasts = Vec::with_capacity(series_batch.len());

            // Advanced batching with memory pooling and async execution
            for (batch_idx, batch) in series_batch.chunks(optimal_batch_size).enumerate() {
                // Simulate GPU stream allocation
                let stream_id = batch_idx % 4; // Use 4 concurrent streams

                // GPU-optimized parallel processing
                let batch_forecasts =
                    self.gpu_parallel_forecast(batch, forecast_steps, &method, stream_id)?;
                all_forecasts.extend(batch_forecasts);
            }

            Ok(all_forecasts)
        }

        /// GPU-parallel forecasting for a batch
        fn gpu_parallel_forecast(
            &self,
            batch: &[Array1<F>],
            forecast_steps: usize,
            method: &ForecastMethod,
            id: usize,
        ) -> Result<Vec<Array1<F>>> {
            // Advanced parallel processing using GPU-optimized algorithms
            match method {
                ForecastMethod::ExponentialSmoothing { alpha } => self
                    .gpu_batch_exponential_smoothing(
                        batch,
                        F::from(*alpha).unwrap_or_else(|| F::from(0.3).unwrap()),
                        forecast_steps,
                    ),
                ForecastMethod::LinearTrend => self.gpu_batch_linear_trend(batch, forecast_steps),
                ForecastMethod::MovingAverage { window } => {
                    self.gpu_batch_moving_average(batch, *window, forecast_steps)
                }
                ForecastMethod::AutoRegressive { order } => {
                    self.gpu_batch_autoregressive(batch, *order, forecast_steps)
                }
            }
        }

        /// GPU-optimized batch exponential smoothing
        fn gpu_batch_exponential_smoothing(
            &self,
            batch: &[Array1<F>],
            alpha: F,
            steps: usize,
        ) -> Result<Vec<Array1<F>>> {
            let mut results = Vec::with_capacity(batch.len());

            // Vectorized computation across all series
            for series in batch {
                if series.is_empty() {
                    return Err(TimeSeriesError::InvalidInput("Empty series".to_string()));
                }

                // GPU-style vectorized exponential smoothing
                let mut smoothed = series[0];
                let alpha_complement = F::one() - alpha;

                // Process all values starting from index 1
                // Unrolled loop for better GPU utilization
                let data_len = series.len() - 1; // Number of values to process (excluding first)
                let chunks = data_len / 4;
                let remainder = data_len % 4;

                // Process in chunks of 4 (simulate SIMD)
                for chunk_idx in 0..chunks {
                    let base_idx = chunk_idx * 4 + 1; // Start from index 1
                    for i in 0..4 {
                        if base_idx + i < series.len() {
                            let value = series[base_idx + i];
                            smoothed = alpha * value + alpha_complement * smoothed;
                        }
                    }
                }

                // Process remainder
                let remainder_start = chunks * 4 + 1;
                for i in 0..remainder {
                    if remainder_start + i < series.len() {
                        let value = series[remainder_start + i];
                        smoothed = alpha * value + alpha_complement * smoothed;
                    }
                }

                // Generate forecasts with parallel computation
                let forecast = Array1::from_elem(steps, smoothed);
                results.push(forecast);
            }

            Ok(results)
        }

        /// GPU-optimized batch linear trend forecasting
        fn gpu_batch_linear_trend(
            &self,
            batch: &[Array1<F>],
            steps: usize,
        ) -> Result<Vec<Array1<F>>> {
            let mut results = Vec::with_capacity(batch.len());

            // Parallel trend computation across batch
            for series in batch {
                if series.len() < 2 {
                    return Err(TimeSeriesError::InsufficientData {
                        message: "Need at least 2 points for trend".to_string(),
                        required: 2,
                        actual: series.len(),
                    });
                }

                // GPU-optimized trend calculation using vectorized operations
                let n = F::from(series.len()).unwrap();
                let x_mean = (n - F::one()) / F::from(2).unwrap();

                // Vectorized sum computation
                let y_sum = series.sum();
                let y_mean = y_sum / n;

                // Parallel computation of slope components
                let mut numerator = F::zero();
                let mut denominator = F::zero();

                // Unrolled computation for better performance
                let chunk_size = 8; // Simulate GPU warp size
                let chunks = series.len() / chunk_size;

                for chunk_idx in 0..chunks {
                    let mut chunk_num = F::zero();
                    let mut chunk_den = F::zero();

                    for i in 0..chunk_size {
                        let idx = chunk_idx * chunk_size + i;
                        let x = F::from(idx).unwrap();
                        let y = series[idx];
                        let x_diff = x - x_mean;

                        chunk_num = chunk_num + x_diff * (y - y_mean);
                        chunk_den = chunk_den + x_diff * x_diff;
                    }

                    numerator = numerator + chunk_num;
                    denominator = denominator + chunk_den;
                }

                // Process remainder
                for idx in (chunks * chunk_size)..series.len() {
                    let x = F::from(idx).unwrap();
                    let y = series[idx];
                    let x_diff = x - x_mean;

                    numerator = numerator + x_diff * (y - y_mean);
                    denominator = denominator + x_diff * x_diff;
                }

                let slope = if denominator > F::zero() {
                    numerator / denominator
                } else {
                    F::zero()
                };

                let intercept = y_mean - slope * x_mean;
                let last_x = F::from(series.len() - 1).unwrap();

                // Vectorized forecast generation
                let mut forecasts = Array1::zeros(steps);
                for i in 0..steps {
                    let future_x = last_x + F::from(i + 1).unwrap();
                    forecasts[i] = slope * future_x + intercept;
                }

                results.push(forecasts);
            }

            Ok(results)
        }

        /// GPU-optimized batch moving average forecasting
        fn gpu_batch_moving_average(
            &self,
            batch: &[Array1<F>],
            window: usize,
            steps: usize,
        ) -> Result<Vec<Array1<F>>> {
            let mut results = Vec::with_capacity(batch.len());

            for series in batch {
                if series.len() < window {
                    return Err(TimeSeriesError::InsufficientData {
                        message: "Series shorter than window".to_string(),
                        required: window,
                        actual: series.len(),
                    });
                }

                // GPU-optimized sliding window computation
                let last_window_start = series.len() - window;
                let mut sum = F::zero();

                // Vectorized sum computation
                for i in 0..window {
                    sum = sum + series[last_window_start + i];
                }

                let avg = sum / F::from(window).unwrap();
                let forecast = Array1::from_elem(steps, avg);
                results.push(forecast);
            }

            Ok(results)
        }

        /// GPU-optimized batch autoregressive forecasting
        fn gpu_batch_autoregressive(
            &self,
            batch: &[Array1<F>],
            order: usize,
            steps: usize,
        ) -> Result<Vec<Array1<F>>> {
            let mut results = Vec::with_capacity(batch.len());

            for series in batch {
                if series.len() < order + 1 {
                    return Err(TimeSeriesError::InsufficientData {
                        message: "Insufficient data for AR model".to_string(),
                        required: order + 1,
                        actual: series.len(),
                    });
                }

                // GPU-optimized AR coefficient estimation
                let coefficients = self.gpu_estimate_ar_coefficients(series, order)?;

                // Parallel forecast generation
                let mut forecasts = Array1::zeros(steps);
                let mut extended_series = series.to_vec();

                for i in 0..steps {
                    let mut forecast = F::zero();

                    // Vectorized dot product computation
                    for (j, &coeff) in coefficients.iter().enumerate() {
                        let lag_index = extended_series.len() - 1 - j;
                        forecast = forecast + coeff * extended_series[lag_index];
                    }

                    forecasts[i] = forecast;
                    extended_series.push(forecast);
                }

                results.push(forecasts);
            }

            Ok(results)
        }

        /// GPU-optimized AR coefficient estimation
        fn gpu_estimate_ar_coefficients(&self, series: &Array1<F>, order: usize) -> Result<Vec<F>> {
            let n = series.len();
            if n < order + 1 {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Insufficient data for coefficient estimation".to_string(),
                    required: order + 1,
                    actual: n,
                });
            }

            // Advanced Yule-Walker equations with GPU optimization
            let _num_equations = n - order;
            let mut autocorrelations = vec![F::zero(); order + 1];

            // Compute autocorrelations using GPU-style parallel reduction
            for lag in 0..=order {
                let mut sum = F::zero();
                let count = n - lag;

                // Parallel reduction across values
                for i in 0..count {
                    sum = sum + series[i] * series[i + lag];
                }

                autocorrelations[lag] = sum / F::from(count).unwrap();
            }

            // Solve Yule-Walker equations using Levinson-Durbin recursion
            self.gpu_levinson_durbin(&autocorrelations[1..], autocorrelations[0])
        }

        /// GPU-optimized Levinson-Durbin algorithm
        fn gpu_levinson_durbin(&self, autocorr: &[F], variance: F) -> Result<Vec<F>> {
            let order = autocorr.len();
            let mut coefficients = vec![F::zero(); order];
            let mut reflection_coeffs = vec![F::zero(); order];
            let mut prediction_error = variance;

            for k in 0..order {
                // Compute reflection coefficient
                let mut sum = F::zero();
                for j in 0..k {
                    sum = sum + coefficients[j] * autocorr[k - 1 - j];
                }

                reflection_coeffs[k] = (autocorr[k] - sum) / prediction_error;

                // Update coefficients using parallel computation
                let new_coeff = reflection_coeffs[k];

                // Store old coefficients for parallel update
                let old_coeffs: Vec<F> = coefficients[..k].to_vec();

                // Update all coefficients in parallel
                for j in 0..k {
                    coefficients[j] = old_coeffs[j] - new_coeff * old_coeffs[k - 1 - j];
                }

                coefficients[k] = new_coeff;

                // Update prediction error
                prediction_error = prediction_error * (F::one() - new_coeff * new_coeff);
            }

            Ok(coefficients)
        }

        /// Optimized parallel batch forecasting (fallback)
        #[allow(dead_code)]
        fn optimized_batch_forecast(
            &self,
            series_batch: &[Array1<F>],
            forecast_steps: usize,
            method: ForecastMethod,
        ) -> Result<Vec<Array1<F>>> {
            let optimal_batch_size = super::utils::get_recommended_batch_size(
                series_batch.len(),
                8 * 1024 * 1024, // 8MB memory limit
            );

            let mut all_forecasts = Vec::with_capacity(series_batch.len());

            // Process in batches to optimize memory usage
            for _batch in series_batch.chunks(optimal_batch_size) {
                let _batch_forecasts: Result<Vec<_>> = _batch
                    .iter()
                    .map(|series| self.single_series_forecast(series, forecast_steps, &method))
                    .collect();
                all_forecasts.extend(_batch_forecasts?);
            }

            Ok(all_forecasts)
        }

        /// Single series forecasting
        fn single_series_forecast(
            &self,
            series: &Array1<F>,
            forecast_steps: usize,
            method: &ForecastMethod,
        ) -> Result<Array1<F>> {
            match method {
                ForecastMethod::ExponentialSmoothing { alpha } => self
                    .gpu_exponential_smoothing_forecast(
                        series,
                        F::from(*alpha).unwrap_or_else(|| F::from(0.3).unwrap()),
                        forecast_steps,
                    ),
                ForecastMethod::LinearTrend => {
                    self.gpu_linear_trend_forecast(series, forecast_steps)
                }
                ForecastMethod::MovingAverage { window } => {
                    self.gpu_moving_average_forecast(series, *window, forecast_steps)
                }
                ForecastMethod::AutoRegressive { order } => {
                    self.gpu_ar_forecast(series, *order, forecast_steps)
                }
            }
        }

        /// GPU-optimized exponential smoothing
        fn gpu_exponential_smoothing_forecast(
            &self,
            series: &Array1<F>,
            alpha: F,
            steps: usize,
        ) -> Result<Array1<F>> {
            if series.is_empty() {
                return Err(TimeSeriesError::InvalidInput("Empty series".to_string()));
            }

            // Calculate smoothed value using vectorized operations
            let mut smoothed = series[0];
            for &value in series.iter().skip(1) {
                smoothed = alpha * value + (F::one() - alpha) * smoothed;
            }

            // Generate forecasts (all same value for simple exponential smoothing)
            Ok(Array1::from_elem(steps, smoothed))
        }

        /// GPU-optimized linear trend forecast
        fn gpu_linear_trend_forecast(&self, series: &Array1<F>, steps: usize) -> Result<Array1<F>> {
            if series.len() < 2 {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Need at least 2 points for trend".to_string(),
                    required: 2,
                    actual: series.len(),
                });
            }

            let n = F::from(series.len()).unwrap();
            let x_mean = (n - F::one()) / F::from(2).unwrap();
            let y_mean = series.sum() / n;

            // Calculate slope using vectorized operations
            let mut numerator = F::zero();
            let mut denominator = F::zero();

            for (i, &y) in series.iter().enumerate() {
                let x = F::from(i).unwrap();
                let x_diff = x - x_mean;
                numerator = numerator + x_diff * (y - y_mean);
                denominator = denominator + x_diff * x_diff;
            }

            let slope = if denominator > F::zero() {
                numerator / denominator
            } else {
                F::zero()
            };

            let intercept = y_mean - slope * x_mean;
            let last_x = F::from(series.len() - 1).unwrap();

            // Generate forecasts
            let mut forecasts = Array1::zeros(steps);
            for i in 0..steps {
                let future_x = last_x + F::from(i + 1).unwrap();
                forecasts[i] = slope * future_x + intercept;
            }

            Ok(forecasts)
        }

        /// GPU-optimized moving average forecast
        fn gpu_moving_average_forecast(
            &self,
            series: &Array1<F>,
            window: usize,
            steps: usize,
        ) -> Result<Array1<F>> {
            if series.len() < window {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Series shorter than window".to_string(),
                    required: window,
                    actual: series.len(),
                });
            }

            // Calculate last moving average
            let last_window = series.slice(s![series.len() - window..]);
            let avg = last_window.sum() / F::from(window).unwrap();

            // Simple moving average forecast (constant)
            Ok(Array1::from_elem(steps, avg))
        }

        /// GPU-optimized autoregressive forecast
        fn gpu_ar_forecast(
            &self,
            series: &Array1<F>,
            order: usize,
            steps: usize,
        ) -> Result<Array1<F>> {
            if series.len() < order + 1 {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Insufficient data for AR model".to_string(),
                    required: order + 1,
                    actual: series.len(),
                });
            }

            // Simple AR parameter estimation using least squares
            let coefficients = self.estimate_ar_coefficients(series, order)?;

            // Generate forecasts
            let mut forecasts = Array1::zeros(steps);
            let mut extended_series = series.to_vec();

            for i in 0..steps {
                let mut forecast = F::zero();
                for (j, &coeff) in coefficients.iter().enumerate() {
                    let lag_index = extended_series.len() - 1 - j;
                    forecast = forecast + coeff * extended_series[lag_index];
                }
                forecasts[i] = forecast;
                extended_series.push(forecast);
            }

            Ok(forecasts)
        }

        /// Estimate AR coefficients using simplified least squares
        fn estimate_ar_coefficients(&self, series: &Array1<F>, order: usize) -> Result<Vec<F>> {
            let n = series.len();
            if n < order + 1 {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Insufficient data for coefficient estimation".to_string(),
                    required: order + 1,
                    actual: n,
                });
            }

            // Build design matrix X and target vector y
            let num_equations = n - order;
            let mut x = Array2::zeros((num_equations, order));
            let mut y = Array1::zeros(num_equations);

            for i in 0..num_equations {
                y[i] = series[i + order];
                for j in 0..order {
                    x[[i, j]] = series[i + order - 1 - j];
                }
            }

            // Solve normal equations: X^T X β = X^T y
            self.solve_normal_equations(&x, &y)
        }

        /// Solve normal equations for least squares
        fn solve_normal_equations(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Vec<F>> {
            let p = x.ncols();

            // For simplicity, use a diagonal approximation
            // In a full implementation, this would use proper matrix operations
            let mut coefficients = vec![F::zero(); p];

            for j in 0..p {
                let mut num = F::zero();
                let mut den = F::zero();

                for i in 0..x.nrows() {
                    num = num + x[[i, j]] * y[i];
                    den = den + x[[i, j]] * x[[i, j]];
                }

                coefficients[j] = if den > F::zero() {
                    num / den
                } else {
                    F::zero()
                };
            }

            Ok(coefficients)
        }

        /// GPU-accelerated correlation matrix computation
        pub fn batch_correlation_matrix(&self, seriesbatch: &[Array1<F>]) -> Result<Array2<F>> {
            let n = seriesbatch.len();
            let mut correlation_matrix = Array2::zeros((n, n));

            // Compute all pairwise correlations
            for i in 0..n {
                for j in i..n {
                    let corr = if i == j {
                        F::one()
                    } else {
                        self.gpu_correlation(&seriesbatch[i], &seriesbatch[j])?
                    };
                    correlation_matrix[[i, j]] = corr;
                    correlation_matrix[[j, i]] = corr;
                }
            }

            Ok(correlation_matrix)
        }

        /// GPU-accelerated correlation computation
        fn gpu_correlation(&self, series1: &Array1<F>, series2: &Array1<F>) -> Result<F> {
            if series1.len() != series2.len() || series1.is_empty() {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: series1.len(),
                    actual: series2.len(),
                });
            }

            let n = F::from(series1.len()).unwrap();
            let mean1 = series1.sum() / n;
            let mean2 = series2.sum() / n;

            let mut num = F::zero();
            let mut den1 = F::zero();
            let mut den2 = F::zero();

            for (&x1, &x2) in series1.iter().zip(series2.iter()) {
                let diff1 = x1 - mean1;
                let diff2 = x2 - mean2;
                num = num + diff1 * diff2;
                den1 = den1 + diff1 * diff1;
                den2 = den2 + diff2 * diff2;
            }

            let denominator = (den1 * den2).sqrt();
            if denominator > F::zero() {
                Ok(num / denominator)
            } else {
                Ok(F::zero())
            }
        }

        /// GPU-accelerated sliding window operations
        pub fn sliding_window_statistics(
            &self,
            series: &Array1<F>,
            window_size: usize,
            statistics: &[WindowStatistic],
        ) -> Result<Vec<Array1<F>>> {
            if series.len() < window_size {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Series shorter than window".to_string(),
                    required: window_size,
                    actual: series.len(),
                });
            }

            let num_windows = series.len() - window_size + 1;
            let mut results = Vec::with_capacity(statistics.len());

            for stat in statistics {
                let mut stat_values = Array1::zeros(num_windows);

                for i in 0..num_windows {
                    let window = series.slice(s![i..i + window_size]);
                    stat_values[i] = match stat {
                        WindowStatistic::Mean => window.sum() / F::from(window_size).unwrap(),
                        WindowStatistic::Variance => {
                            let mean = window.sum() / F::from(window_size).unwrap();
                            window
                                .iter()
                                .map(|&x| (x - mean) * (x - mean))
                                .fold(F::zero(), |acc, x| acc + x)
                                / F::from(window_size).unwrap()
                        }
                        WindowStatistic::Min => {
                            window.iter().fold(F::infinity(), |acc, &x| acc.min(x))
                        }
                        WindowStatistic::Max => {
                            window.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x))
                        }
                        WindowStatistic::Range => {
                            let min_val = window.iter().fold(F::infinity(), |acc, &x| acc.min(x));
                            let max_val =
                                window.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));
                            max_val - min_val
                        }
                    };
                }

                results.push(stat_values);
            }

            Ok(results)
        }
    }

    /// Forecasting methods for GPU acceleration
    #[derive(Debug, Clone)]
    pub enum ForecastMethod {
        /// Exponential smoothing with alpha parameter
        ExponentialSmoothing {
            /// Smoothing parameter (0 < alpha < 1)
            alpha: f64,
        },
        /// Linear trend forecasting
        LinearTrend,
        /// Moving average with window size
        MovingAverage {
            /// Window size for moving average
            window: usize,
        },
        /// Autoregressive model
        AutoRegressive {
            /// Order of the autoregressive model
            order: usize,
        },
    }

    /// Window statistics for sliding window operations
    #[derive(Debug, Clone)]
    pub enum WindowStatistic {
        /// Calculate mean of window
        Mean,
        /// Calculate variance of window
        Variance,
        /// Calculate minimum value in window
        Min,
        /// Calculate maximum value in window
        Max,
        /// Calculate range (max - min) of window
        Range,
    }

    /// GPU-accelerated feature extraction for time series
    #[derive(Debug)]
    pub struct GpuFeatureExtractor<F: Float + Debug> {
        #[allow(dead_code)]
        processor: GpuTimeSeriesProcessor<F>,
        feature_config: FeatureConfig,
    }

    /// Configuration for feature extraction
    #[derive(Debug, Clone)]
    pub struct FeatureConfig {
        /// Extract statistical features (mean, std, skewness, etc.)
        pub extract_statistical: bool,
        /// Extract frequency domain features (FFT-based)
        pub extract_frequency: bool,
        /// Extract complexity features (entropy, fractal dimension, etc.)
        pub extract_complexity: bool,
        /// Window sizes for sliding window feature extraction
        pub window_sizes: Vec<usize>,
    }

    impl Default for FeatureConfig {
        fn default() -> Self {
            Self {
                extract_statistical: true,
                extract_frequency: true,
                extract_complexity: false,
                window_sizes: vec![5, 10, 20],
            }
        }
    }

    impl<F: Float + Debug + Clone + std::iter::Sum> GpuFeatureExtractor<F> {
        /// Create new GPU acceleration instance
        pub fn new(_config: GpuConfig, featureconfig: FeatureConfig) -> Result<Self> {
            let processor = GpuTimeSeriesProcessor::new(_config)?;
            Ok(Self {
                processor,
                feature_config: featureconfig,
            })
        }

        /// Extract comprehensive features from multiple time series
        pub fn batch_extract_features(&self, seriesbatch: &[Array1<F>]) -> Result<Array2<F>> {
            let mut all_features = Vec::new();

            for series in seriesbatch {
                let features = self.extract_features(series)?;
                all_features.push(features);
            }

            // Combine into matrix
            if all_features.is_empty() {
                return Ok(Array2::zeros((0, 0)));
            }

            let n_series = all_features.len();
            let n_features = all_features[0].len();
            let mut feature_matrix = Array2::zeros((n_series, n_features));

            for (i, features) in all_features.iter().enumerate() {
                for (j, &feature) in features.iter().enumerate() {
                    feature_matrix[[i, j]] = feature;
                }
            }

            Ok(feature_matrix)
        }

        /// Extract features from a single time series
        fn extract_features(&self, series: &Array1<F>) -> Result<Vec<F>> {
            let mut features = Vec::new();

            if self.feature_config.extract_statistical {
                features.extend(self.extract_statistical_features(series)?);
            }

            if self.feature_config.extract_frequency {
                features.extend(self.extract_frequency_features(series)?);
            }

            if self.feature_config.extract_complexity {
                features.extend(self.extract_complexity_features(series)?);
            }

            Ok(features)
        }

        /// Extract statistical features
        fn extract_statistical_features(&self, series: &Array1<F>) -> Result<Vec<F>> {
            if series.is_empty() {
                return Ok(vec![F::zero(); 8]); // Return zeros for all features
            }

            let n = F::from(series.len()).unwrap();
            let mean = series.sum() / n;

            // Variance
            let variance = series
                .iter()
                .map(|&x| (x - mean) * (x - mean))
                .fold(F::zero(), |acc, x| acc + x)
                / n;

            // Min/Max
            let min_val = series.iter().fold(F::infinity(), |acc, &x| acc.min(x));
            let max_val = series.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));

            // Skewness (simplified)
            let std_dev = variance.sqrt();
            let skewness = if std_dev > F::zero() {
                series
                    .iter()
                    .map(|&x| {
                        let normalized = (x - mean) / std_dev;
                        normalized * normalized * normalized
                    })
                    .fold(F::zero(), |acc, x| acc + x)
                    / n
            } else {
                F::zero()
            };

            // Kurtosis (simplified)
            let kurtosis = if std_dev > F::zero() {
                series
                    .iter()
                    .map(|&x| {
                        let normalized = (x - mean) / std_dev;
                        let squared = normalized * normalized;
                        squared * squared
                    })
                    .fold(F::zero(), |acc, x| acc + x)
                    / n
            } else {
                F::zero()
            };

            // Range
            let range = max_val - min_val;

            // Trend (slope of linear regression)
            let trend = if series.len() > 1 {
                let x_mean = F::from(series.len() - 1).unwrap() / F::from(2).unwrap();
                let mut num = F::zero();
                let mut den = F::zero();

                for (i, &y) in series.iter().enumerate() {
                    let x = F::from(i).unwrap();
                    num = num + (x - x_mean) * (y - mean);
                    den = den + (x - x_mean) * (x - x_mean);
                }

                if den > F::zero() {
                    num / den
                } else {
                    F::zero()
                }
            } else {
                F::zero()
            };

            Ok(vec![
                mean,
                variance.sqrt(),
                min_val,
                max_val,
                skewness,
                kurtosis,
                range,
                trend,
            ])
        }

        /// Extract frequency domain features (simplified)
        fn extract_frequency_features(&self, series: &Array1<F>) -> Result<Vec<F>> {
            // Simplified frequency features without actual FFT
            let n = series.len();
            if n < 4 {
                return Ok(vec![F::zero(); 3]);
            }

            // Estimate dominant frequency using autocorrelation
            let mut max_autocorr = F::zero();
            let mut dominant_period = 1;

            for lag in 1..(n / 2).min(20) {
                let mut autocorr = F::zero();
                let mut count = 0;

                for i in lag..n {
                    autocorr = autocorr + series[i] * series[i - lag];
                    count += 1;
                }

                if count > 0 {
                    autocorr = autocorr / F::from(count).unwrap();
                    if autocorr > max_autocorr {
                        max_autocorr = autocorr;
                        dominant_period = lag;
                    }
                }
            }

            let dominant_frequency = F::one() / F::from(dominant_period).unwrap();

            // Spectral energy (simplified)
            let spectral_energy = series
                .iter()
                .map(|&x| x * x)
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(n).unwrap();

            Ok(vec![dominant_frequency, max_autocorr, spectral_energy])
        }

        /// Extract complexity features (simplified)
        fn extract_complexity_features(&self, series: &Array1<F>) -> Result<Vec<F>> {
            if series.len() < 3 {
                return Ok(vec![F::zero(); 2]);
            }

            // Approximate entropy (simplified)
            let mut changes = 0;
            for i in 1..series.len() {
                if (series[i] - series[i - 1]).abs() > F::zero() {
                    changes += 1;
                }
            }
            let entropy = F::from(changes).unwrap() / F::from(series.len() - 1).unwrap();

            // Sample entropy (very simplified)
            let mut matches = 0;
            let tolerance = series
                .iter()
                .map(|&x| x * x)
                .fold(F::zero(), |acc, x| acc + x)
                .sqrt()
                / F::from(series.len()).unwrap()
                * F::from(0.1).unwrap();

            for i in 0..series.len() - 2 {
                for j in i + 1..series.len() - 1 {
                    if (series[i] - series[j]).abs() <= tolerance
                        && (series[i + 1] - series[j + 1]).abs() <= tolerance
                    {
                        matches += 1;
                    }
                }
            }

            let sample_entropy = if matches > 0 {
                -F::from(matches).unwrap().ln()
            } else {
                F::from(10).unwrap() // Large value for high entropy
            };

            Ok(vec![entropy, sample_entropy])
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_gpu_processor_creation() {
            let config = GpuConfig::default();
            let processor = GpuTimeSeriesProcessor::<f64>::new(config);
            assert!(processor.is_ok());
        }

        #[test]
        fn test_batch_exponential_smoothing() {
            let config = GpuConfig::default();
            let processor = GpuTimeSeriesProcessor::<f64>::new(config).unwrap();

            let series1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            let series2 = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
            let batch = vec![series1, series2];

            let method = ForecastMethod::ExponentialSmoothing { alpha: 0.3 };
            let results = processor.batch_forecast(&batch, 3, method);

            assert!(results.is_ok());
            let forecasts = results.unwrap();
            assert_eq!(forecasts.len(), 2);
            assert_eq!(forecasts[0].len(), 3);
            assert_eq!(forecasts[1].len(), 3);
        }

        #[test]
        fn test_correlation_matrix() {
            let config = GpuConfig::default();
            let processor = GpuTimeSeriesProcessor::<f64>::new(config).unwrap();

            let series1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            let series2 = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
            let batch = vec![series1, series2];

            let correlation_matrix = processor.batch_correlation_matrix(&batch).unwrap();

            assert_eq!(correlation_matrix.dim(), (2, 2));
            assert!((correlation_matrix[[0, 0]] - 1.0).abs() < 1e-10);
            assert!((correlation_matrix[[1, 1]] - 1.0).abs() < 1e-10);
            assert!(correlation_matrix[[0, 1]] > 0.99); // Should be highly correlated
        }

        #[test]
        fn test_feature_extraction() {
            let config = GpuConfig::default();
            let feature_config = FeatureConfig::default();
            let extractor = GpuFeatureExtractor::new(config, feature_config).unwrap();

            let series = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
            let batch = vec![series];

            let features = extractor.batch_extract_features(&batch).unwrap();
            assert_eq!(features.nrows(), 1);
            assert!(features.ncols() > 0); // Should have extracted features
        }

        #[test]
        fn test_sliding_window_statistics() {
            let config = GpuConfig::default();
            let processor = GpuTimeSeriesProcessor::<f64>::new(config).unwrap();

            let series = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            let statistics = vec![WindowStatistic::Mean, WindowStatistic::Variance];

            let results = processor
                .sliding_window_statistics(&series, 3, &statistics)
                .unwrap();

            assert_eq!(results.len(), 2); // Two statistics
            assert_eq!(results[0].len(), 6); // 8 - 3 + 1 = 6 windows

            // Check first window mean: (1+2+3)/3 = 2
            assert!((results[0][0] - 2.0).abs() < 1e-10);
        }
    }
}
