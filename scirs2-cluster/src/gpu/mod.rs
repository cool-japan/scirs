//! GPU acceleration interfaces and implementations for clustering algorithms
//!
//! This module provides GPU acceleration capabilities for clustering algorithms,
//! supporting multiple backends including CUDA, OpenCL, ROCm, Intel OneAPI, and Metal.
//! When GPU acceleration is not available, algorithms automatically fall back to
//! optimized CPU implementations.
//!
//! # Features
//!
//! * **Multiple GPU Backends**: Support for CUDA, OpenCL, ROCm, Intel OneAPI, and Metal
//! * **Automatic Fallback**: Seamless fallback to CPU when GPU is not available
//! * **Memory Management**: Efficient GPU memory allocation and pooling
//! * **Distance Computations**: Optimized distance matrix calculations
//! * **Performance Monitoring**: Built-in benchmarking and performance statistics
//! * **Device Selection**: Automatic or manual GPU device selection strategies
//!
//! # Examples
//!
//! ## Basic GPU Configuration
//!
//! ```rust
//! use scirs2_cluster::gpu::{GpuConfig, GpuBackend, DeviceSelection};
//!
//! // Create CUDA configuration with automatic device selection
//! let config = GpuConfig::cuda()
//!     .with_device_selection(DeviceSelection::Auto)
//!     .with_memory_pool_size(1024 * 1024 * 1024); // 1GB pool
//!
//! // OpenCL configuration for cross-platform support
//! let opencl_config = GpuConfig::opencl()
//!     .with_device_selection(DeviceSelection::MostMemory);
//! ```
//!
//! ## GPU Distance Matrix Computation
//!
//! ```rust
//! use scirs2_cluster::gpu::{GpuDistanceMatrix, DistanceMetric, GpuConfig};
//! use scirs2_core::ndarray::Array2;
//!
//! // Create sample data
//! let data = Array2::from_shape_vec((1000, 10), (0..10000).map(|x| x as f64).collect()).unwrap();
//!
//! // Create GPU distance matrix
//! let config = GpuConfig::default();
//! let mut gpu_matrix = GpuDistanceMatrix::new(config, DistanceMetric::Euclidean, None).unwrap();
//!
//! // Preload data to GPU for faster repeated computations
//! gpu_matrix.preload_data(data.view()).unwrap();
//!
//! // Compute distance matrix
//! let distances = gpu_matrix.compute_distance_matrix(data.view()).unwrap();
//! ```
//!
//! ## Memory Management
//!
//! ```rust
//! use scirs2_cluster::gpu::{GpuMemoryManager, MemoryStrategy};
//!
//! // Create memory manager with 256-byte alignment and pool size of 100
//! let mut memory_manager = GpuMemoryManager::new(256, 100);
//!
//! // Allocate GPU memory
//! let block = memory_manager.allocate(1024 * 1024).unwrap(); // 1MB
//!
//! // Memory is automatically pooled for reuse
//! memory_manager.deallocate(block).unwrap();
//!
//! // Check memory statistics
//! let stats = memory_manager.get_stats();
//! println!("Pool efficiency: {:.2}%", memory_manager.pool_efficiency() * 100.0);
//! ```

pub mod core;
pub mod distance;
pub mod memory;

// Re-export main types for convenience
pub use core::{BackendContext, DeviceSelection, GpuBackend, GpuConfig, GpuContext, GpuDevice};

pub use distance::{DistanceMetric, GpuArray, GpuDistanceMatrix};

pub use memory::{
    BandwidthMonitor, GpuMemoryBlock, GpuMemoryManager, MemoryStats, MemoryStrategy, MemoryTransfer,
};

// Additional convenience functions and types

/// Create a GPU configuration for the best available backend
pub fn auto_config() -> GpuConfig {
    // Try backends in order of preference
    let preferred_backends = [
        GpuBackend::Cuda,
        GpuBackend::OpenCl,
        GpuBackend::Rocm,
        GpuBackend::Metal,
        GpuBackend::OneApi,
    ];

    for &backend in &preferred_backends {
        if is_backend_available(backend) {
            return GpuConfig::new(backend);
        }
    }

    // Fallback to CPU
    GpuConfig::default()
}

/// Check if a specific GPU backend is available
pub fn is_backend_available(backend: GpuBackend) -> bool {
    match backend {
        GpuBackend::CpuFallback => true,
        _ => {
            // This is a stub implementation
            // Real implementation would check for:
            // - CUDA: nvidia-ml-py, cupy, or pycuda availability
            // - OpenCL: pyopencl availability
            // - ROCm: rocm installation
            // - Metal: Metal framework availability (macOS)
            // - OneAPI: Intel OneAPI toolkit installation
            false
        }
    }
}

/// List all available GPU devices
pub fn list_devices() -> Vec<GpuDevice> {
    // This is a stub implementation
    // Real implementation would enumerate actual devices
    vec![GpuDevice::new(
        0,
        "Integrated GPU".to_string(),
        4_000_000_000, // 4GB
        3_500_000_000, // 3.5GB available
        "1.0".to_string(),
        512,
        GpuBackend::CpuFallback,
        false,
    )]
}

/// Get the best available GPU device
pub fn get_best_device() -> Option<GpuDevice> {
    let devices = list_devices();
    devices
        .into_iter()
        .filter(|d| d.backend != GpuBackend::CpuFallback)
        .max_by(|a, b| {
            a.get_device_score()
                .partial_cmp(&b.get_device_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Benchmark GPU vs CPU performance for distance computations
pub fn benchmark_gpu_vs_cpu(
    data_size: usize,
    n_features: usize,
    metric: DistanceMetric,
) -> Result<BenchmarkResult, crate::error::ClusteringError> {
    use scirs2_core::ndarray::Array2;
    use std::time::Instant;

    // Generate test data
    let data = Array2::from_shape_fn((data_size, n_features), |(i, j)| {
        (i * n_features + j) as f64 / 1000.0
    });

    // CPU benchmark
    let cpu_start = Instant::now();
    let cpu_config = GpuConfig::new(GpuBackend::CpuFallback);
    let cpu_matrix = GpuDistanceMatrix::new(cpu_config, metric, None)?;
    let _cpu_result = cpu_matrix.compute_distance_matrix_cpu(data.view())?;
    let cpu_duration = cpu_start.elapsed();

    // GPU benchmark (will fallback to CPU in stub implementation)
    let gpu_start = Instant::now();
    let gpu_config = auto_config();
    let mut gpu_matrix = GpuDistanceMatrix::new(gpu_config, metric, None)?;
    let _gpu_result = gpu_matrix.compute_distance_matrix(data.view())?;
    let gpu_duration = gpu_start.elapsed();

    Ok(BenchmarkResult {
        cpu_duration_ms: cpu_duration.as_millis() as f64,
        gpu_duration_ms: gpu_duration.as_millis() as f64,
        speedup: cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64(),
        data_size,
        n_features,
        metric,
    })
}

/// Result of GPU vs CPU benchmark
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// CPU computation time in milliseconds
    pub cpu_duration_ms: f64,
    /// GPU computation time in milliseconds
    pub gpu_duration_ms: f64,
    /// Speedup factor (CPU time / GPU time)
    pub speedup: f64,
    /// Size of test data
    pub data_size: usize,
    /// Number of features
    pub n_features: usize,
    /// Distance metric used
    pub metric: DistanceMetric,
}

impl BenchmarkResult {
    /// Get performance summary
    pub fn summary(&self) -> String {
        format!(
            "GPU vs CPU Benchmark Results:\n\
             Data size: {} samples x {} features\n\
             Distance metric: {}\n\
             CPU time: {:.2} ms\n\
             GPU time: {:.2} ms\n\
             Speedup: {:.2}x",
            self.data_size,
            self.n_features,
            self.metric,
            self.cpu_duration_ms,
            self.gpu_duration_ms,
            self.speedup
        )
    }

    /// Check if GPU provided a speedup
    pub fn gpu_is_faster(&self) -> bool {
        self.speedup > 1.0
    }

    /// Get efficiency rating
    pub fn efficiency_rating(&self) -> &'static str {
        match self.speedup {
            x if x >= 10.0 => "Excellent",
            x if x >= 5.0 => "Very Good",
            x if x >= 2.0 => "Good",
            x if x >= 1.1 => "Marginal",
            _ => "No Benefit",
        }
    }
}

/// GPU feature detection and capabilities
pub struct GpuCapabilities {
    /// Available backends
    pub available_backends: Vec<GpuBackend>,
    /// Best device for each backend
    pub best_devices: std::collections::HashMap<GpuBackend, GpuDevice>,
    /// Total GPU memory across all devices
    pub total_gpu_memory: usize,
    /// Supports unified memory
    pub supports_unified_memory: bool,
    /// Supports double precision
    pub supports_double_precision: bool,
}

impl GpuCapabilities {
    /// Detect GPU capabilities
    pub fn detect() -> Self {
        let available_backends: Vec<GpuBackend> = [
            GpuBackend::Cuda,
            GpuBackend::OpenCl,
            GpuBackend::Rocm,
            GpuBackend::Metal,
            GpuBackend::OneApi,
        ]
        .iter()
        .cloned()
        .filter(|&backend| is_backend_available(backend))
        .collect();

        let mut best_devices = std::collections::HashMap::new();
        let mut total_memory = 0;
        let mut supports_unified = false;
        let mut supports_double = false;

        // Stub implementation
        for backend in available_backends.iter() {
            if let Some(device) = Self::get_best_device_for_backend(*backend) {
                total_memory += device.total_memory;
                supports_unified |= *backend == GpuBackend::Cuda; // CUDA typically supports unified memory
                supports_double |= device.supports_double_precision;
                best_devices.insert(*backend, device);
            }
        }

        Self {
            available_backends,
            best_devices,
            total_gpu_memory: total_memory,
            supports_unified_memory: supports_unified,
            supports_double_precision: supports_double,
        }
    }

    /// Get summary of GPU capabilities
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str("GPU Capabilities Summary:\n");
        summary.push_str(&format!(
            "Available backends: {:?}\n",
            self.available_backends
        ));
        summary.push_str(&format!(
            "Total GPU memory: {:.2} GB\n",
            self.total_gpu_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        ));
        summary.push_str(&format!(
            "Unified memory support: {}\n",
            self.supports_unified_memory
        ));
        summary.push_str(&format!(
            "Double precision support: {}\n",
            self.supports_double_precision
        ));

        for (backend, device) in &self.best_devices {
            summary.push_str(&format!(
                "Best {} device: {} ({:.2} GB)\n",
                backend,
                device.name,
                device.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
            ));
        }

        summary
    }

    fn get_best_device_for_backend(backend: GpuBackend) -> Option<GpuDevice> {
        // Stub implementation
        match backend {
            GpuBackend::CpuFallback => None,
            _ => Some(GpuDevice::new(
                0,
                format!("{} Device", backend),
                8_000_000_000,
                7_000_000_000,
                "1.0".to_string(),
                1024,
                backend,
                true,
            )),
        }
    }
}

/// Convenience function to check if GPU acceleration is recommended for a given problem size
pub fn is_gpu_recommended(n_samples: usize, n_features: usize) -> bool {
    // Simple heuristic: GPU typically beneficial for larger problems
    let problem_size = n_samples * n_features;
    problem_size > 10_000 && n_samples > 100
}

/// Get recommended tile size for GPU computations
pub fn get_recommended_tile_size(device: &GpuDevice, element_size: usize) -> usize {
    // Calculate based on available memory and compute units
    let memory_per_tile = device.available_memory / 16; // Use 1/16 of available memory per tile
    let elements_per_tile = memory_per_tile / element_size;
    let sqrt_elements = (elements_per_tile as f64).sqrt() as usize;

    // Clamp to reasonable range and align to compute units
    let base_tile_size = sqrt_elements.max(32).min(1024);
    let compute_aligned = ((base_tile_size + device.compute_units as usize - 1)
        / device.compute_units as usize)
        * device.compute_units as usize;

    compute_aligned.min(1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_config() {
        let config = auto_config();
        assert!(config.auto_fallback);
    }

    #[test]
    fn test_backend_availability() {
        assert!(is_backend_available(GpuBackend::CpuFallback));
        // Other backends return false in stub implementation
        assert!(!is_backend_available(GpuBackend::Cuda));
    }

    #[test]
    fn test_list_devices() {
        let devices = list_devices();
        assert!(!devices.is_empty());
    }

    #[test]
    fn test_gpu_recommendation() {
        assert!(!is_gpu_recommended(10, 10)); // Small problem
        assert!(is_gpu_recommended(1000, 100)); // Large problem
    }

    #[test]
    fn test_capabilities_detection() {
        let caps = GpuCapabilities::detect();
        assert!(!caps.summary().is_empty());
    }

    #[test]
    fn test_recommended_tile_size() {
        let device = GpuDevice::new(
            0,
            "Test".to_string(),
            8_000_000_000,
            6_000_000_000,
            "1.0".to_string(),
            1024,
            GpuBackend::Cuda,
            true,
        );

        let tile_size = get_recommended_tile_size(&device, 8);
        assert!(tile_size >= 32);
        assert!(tile_size <= 1024);
    }

    #[test]
    fn test_benchmark_result() {
        let result = BenchmarkResult {
            cpu_duration_ms: 100.0,
            gpu_duration_ms: 20.0,
            speedup: 5.0,
            data_size: 1000,
            n_features: 10,
            metric: DistanceMetric::Euclidean,
        };

        assert!(result.gpu_is_faster());
        assert_eq!(result.efficiency_rating(), "Very Good");
        assert!(!result.summary().is_empty());
    }
}
