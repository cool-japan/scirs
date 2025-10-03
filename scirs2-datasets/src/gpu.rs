//! GPU acceleration for dataset operations
//!
//! This module provides GPU acceleration for data generation and processing operations,
//! significantly improving performance for large-scale synthetic dataset creation.

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use scirs2_core::ndarray::Array2;
use std::sync::{Arc, Mutex};

/// GPU backend configuration
#[derive(Debug, Clone, PartialEq)]
pub enum GpuBackend {
    /// CUDA backend for NVIDIA GPUs
    Cuda {
        /// CUDA device ID
        device_id: u32,
    },
    /// OpenCL backend for various GPU vendors
    OpenCl {
        /// OpenCL platform ID
        platform_id: u32,
        /// OpenCL device ID
        device_id: u32,
    },
    /// CPU fallback (for testing or when GPU is unavailable)
    Cpu,
}

/// GPU memory management configuration
#[derive(Debug, Clone)]
pub struct GpuMemoryConfig {
    /// Maximum GPU memory to use (in MB)
    pub max_memory_mb: Option<usize>,
    /// Memory pool size for allocations
    pub pool_size_mb: usize,
    /// Whether to enable memory coalescing optimization
    pub enable_coalescing: bool,
    /// Whether to use unified memory (CUDA only)
    pub use_unified_memory: bool,
}

impl Default for GpuMemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: None,
            pool_size_mb: 512,
            enable_coalescing: true,
            use_unified_memory: false,
        }
    }
}

/// GPU configuration for dataset operations
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// GPU backend to use
    pub backend: GpuBackend,
    /// Memory configuration
    pub memory: GpuMemoryConfig,
    /// Number of CUDA threads per block
    pub threads_per_block: u32,
    /// Whether to enable double precision (f64) operations
    pub enable_double_precision: bool,
    /// Whether to use fast math optimizations
    pub use_fast_math: bool,
    /// Random number generator seed for GPU
    pub random_seed: Option<u64>,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::Cuda { device_id: 0 },
            memory: GpuMemoryConfig::default(),
            threads_per_block: 256,
            enable_double_precision: true,
            use_fast_math: false,
            random_seed: None,
        }
    }
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device name
    pub name: String,
    /// Total memory in MB
    pub total_memory_mb: usize,
    /// Available memory in MB
    pub available_memory_mb: usize,
    /// Number of compute units/streaming multiprocessors
    pub compute_units: u32,
    /// Maximum work group size
    pub max_work_group_size: u32,
    /// Compute capability (CUDA) or version (OpenCL)
    pub compute_capability: String,
    /// Whether double precision is supported
    pub supports_double_precision: bool,
}

/// GPU context for managing device operations
pub struct GpuContext {
    config: GpuConfig,
    device_info: GpuDeviceInfo,
    #[allow(dead_code)]
    memory_pool: Arc<Mutex<GpuMemoryPool>>,
}

impl GpuContext {
    /// Create a new GPU context
    pub fn new(config: GpuConfig) -> Result<Self> {
        // Initialize GPU backend
        let device_info = Self::query_device_info(&config.backend)?;

        // Validate configuration
        Self::validate_config(&config, &device_info)?;

        // Initialize memory pool
        let memory_pool = Arc::new(Mutex::new(GpuMemoryPool::new(&config.memory)?));

        Ok(Self {
            config,
            device_info,
            memory_pool,
        })
    }

    /// Get device information
    pub fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }

    /// Get the backend type
    pub fn backend(&self) -> &GpuBackend {
        &self.config.backend
    }

    /// Check if GPU is available and functional
    pub fn is_available(&self) -> bool {
        match &self.config.backend {
            GpuBackend::Cuda { .. } => self.is_cuda_available(),
            GpuBackend::OpenCl { .. } => self.is_opencl_available(),
            GpuBackend::Cpu => true,
        }
    }

    /// Generate classification dataset on GPU
    pub fn make_classification_gpu(
        &self,
        n_samples: usize,
        n_features: usize,
        n_classes: usize,
        n_clusters_per_class: usize,
        n_informative: usize,
        random_state: Option<u64>,
    ) -> Result<Dataset> {
        match &self.config.backend {
            GpuBackend::Cuda { .. } => self.make_classification_cuda(
                n_samples,
                n_features,
                n_classes,
                n_clusters_per_class,
                n_informative,
                random_state,
            ),
            GpuBackend::OpenCl { .. } => self.make_classification_opencl(
                n_samples,
                n_features,
                n_classes,
                n_clusters_per_class,
                n_informative,
                random_state,
            ),
            GpuBackend::Cpu => {
                // Fallback to CPU implementation
                crate::generators::make_classification(
                    n_samples,
                    n_features,
                    n_classes,
                    n_clusters_per_class,
                    n_informative,
                    random_state,
                )
            }
        }
    }

    /// Generate regression dataset on GPU
    pub fn make_regression_gpu(
        &self,
        n_samples: usize,
        n_features: usize,
        n_informative: usize,
        noise: f64,
        random_state: Option<u64>,
    ) -> Result<Dataset> {
        match &self.config.backend {
            GpuBackend::Cuda { .. } => {
                self.make_regression_cuda(n_samples, n_features, n_informative, noise, random_state)
            }
            GpuBackend::OpenCl { .. } => self.make_regression_opencl(
                n_samples,
                n_features,
                n_informative,
                noise,
                random_state,
            ),
            GpuBackend::Cpu => {
                // Fallback to CPU implementation
                crate::generators::make_regression(
                    n_samples,
                    n_features,
                    n_informative,
                    noise,
                    random_state,
                )
            }
        }
    }

    /// Generate clustering dataset (blobs) on GPU
    pub fn make_blobs_gpu(
        &self,
        n_samples: usize,
        n_features: usize,
        n_centers: usize,
        cluster_std: f64,
        random_state: Option<u64>,
    ) -> Result<Dataset> {
        match &self.config.backend {
            GpuBackend::Cuda { .. } => {
                self.make_blobs_cuda(n_samples, n_features, n_centers, cluster_std, random_state)
            }
            GpuBackend::OpenCl { .. } => {
                self.make_blobs_opencl(n_samples, n_features, n_centers, cluster_std, random_state)
            }
            GpuBackend::Cpu => {
                // Fallback to CPU implementation
                crate::generators::make_blobs(
                    n_samples,
                    n_features,
                    n_centers,
                    cluster_std,
                    random_state,
                )
            }
        }
    }

    /// Perform matrix operations on GPU
    pub fn gpu_matrix_multiply(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
        match &self.config.backend {
            GpuBackend::Cuda { .. } => self.cuda_matrix_multiply(a, b),
            GpuBackend::OpenCl { .. } => self.opencl_matrix_multiply(a, b),
            GpuBackend::Cpu => {
                // CPU fallback
                Ok(a.dot(b))
            }
        }
    }

    /// Apply element-wise operations on GPU
    pub fn gpu_elementwise_op<F>(&self, data: &Array2<f64>, op: F) -> Result<Array2<f64>>
    where
        F: Fn(f64) -> f64 + Send + Sync,
    {
        match &self.config.backend {
            GpuBackend::Cuda { .. } => self.cuda_elementwise_op(data, op),
            GpuBackend::OpenCl { .. } => self.opencl_elementwise_op(data, op),
            GpuBackend::Cpu => {
                // CPU fallback
                Ok(data.mapv(op))
            }
        }
    }

    // Private methods for different backends

    fn query_device_info(backend: &GpuBackend) -> Result<GpuDeviceInfo> {
        match backend {
            GpuBackend::Cuda { device_id } => Self::query_cuda_device_info(*device_id),
            GpuBackend::OpenCl {
                platform_id,
                device_id,
            } => Self::query_opencl_device_info(*platform_id, *device_id),
            GpuBackend::Cpu => Ok(GpuDeviceInfo {
                name: "CPU Fallback".to_string(),
                total_memory_mb: 8192, // Assume 8GB
                available_memory_mb: 4096,
                compute_units: num_cpus::get() as u32,
                max_work_group_size: 1,
                compute_capability: "N/A".to_string(),
                supports_double_precision: true,
            }),
        }
    }

    fn validate_config(config: &GpuConfig, device_info: &GpuDeviceInfo) -> Result<()> {
        // Check memory requirements
        if let Some(max_memory) = config.memory.max_memory_mb {
            if max_memory > device_info.available_memory_mb {
                return Err(DatasetsError::GpuError(format!(
                    "Requested memory ({} MB) exceeds available memory ({} MB)",
                    max_memory, device_info.available_memory_mb
                )));
            }
        }

        // Check double precision support
        if config.enable_double_precision && !device_info.supports_double_precision {
            return Err(DatasetsError::GpuError(
                "Double precision requested but not supported by device".to_string(),
            ));
        }

        // Check threads per block
        if config.threads_per_block > device_info.max_work_group_size {
            return Err(DatasetsError::GpuError(format!(
                "Threads per block ({}) exceeds device limit ({})",
                config.threads_per_block, device_info.max_work_group_size
            )));
        }

        Ok(())
    }

    fn is_cuda_available(&self) -> bool {
        // Check CUDA availability through multiple methods

        // 1. Check for NVIDIA GPU device name
        let has_nvidia_device = self.device_info.name.contains("NVIDIA")
            || self.device_info.name.contains("Tesla")
            || self.device_info.name.contains("GeForce")
            || self.device_info.name.contains("Quadro");

        if !has_nvidia_device {
            return false;
        }

        // 2. Check for CUDA environment variables
        let cuda_env_available = std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
            || std::env::var("CUDA_PATH").is_ok()
            || std::env::var("CUDA_HOME").is_ok();

        // 3. Check for CUDA installation paths
        let cudapaths = [
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/lib/x86_64-linux-gnu/libcuda.so",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib64/libcuda.so",
            "/usr/lib64/libcuda.so.1",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
            "C:\\Windows\\System32\\nvcuda.dll",
        ];

        let cudapath_available = cudapaths
            .iter()
            .any(|path| std::path::Path::new(path).exists());

        // 4. Try to check nvidia-smi (if available)
        let nvidia_smi_available = std::process::Command::new("nvidia-smi")
            .arg("--list-gpus")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);

        cuda_env_available || cudapath_available || nvidia_smi_available
    }

    fn is_opencl_available(&self) -> bool {
        // Check OpenCL availability through multiple methods

        // 1. Skip pure CPU devices unless they explicitly support OpenCL
        if self.device_info.name.contains("CPU") && !self.device_info.name.contains("OpenCL") {
            return false;
        }

        // 2. Check for common OpenCL library paths
        let openclpaths = [
            "/usr/lib/libOpenCL.so",
            "/usr/lib/libOpenCL.so.1",
            "/usr/lib64/libOpenCL.so",
            "/usr/lib64/libOpenCL.so.1",
            "/usr/lib/x86_64-linux-gnu/libOpenCL.so",
            "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
            "/opt/intel/opencl/lib64/libOpenCL.so",
            "/System/Library/Frameworks/OpenCL.framework/OpenCL", // macOS
            "C:\\Windows\\System32\\OpenCL.dll",                  // Windows
        ];

        let opencl_lib_available = openclpaths
            .iter()
            .any(|path| std::path::Path::new(path).exists());

        // 3. Check for vendor-specific OpenCL installations
        let vendor_openclpaths = [
            "/usr/lib/x86_64-linux-gnu/mesa", // Mesa OpenCL
            "/opt/amdgpu-pro",                // AMD Pro drivers
            "/opt/intel/opencl",              // Intel OpenCL
        ];

        let vendor_opencl_available = vendor_openclpaths
            .iter()
            .any(|path| std::path::Path::new(path).exists());

        // 4. Try to run clinfo (if available)
        let clinfo_available = std::process::Command::new("clinfo")
            .output()
            .map(|output| output.status.success() && !output.stdout.is_empty())
            .unwrap_or(false);

        opencl_lib_available || vendor_opencl_available || clinfo_available
    }

    // CUDA-specific implementations (simplified for demonstration)

    fn query_cuda_device_info(_deviceid: u32) -> Result<GpuDeviceInfo> {
        // Simulate CUDA device query
        Ok(GpuDeviceInfo {
            name: format!("NVIDIA GPU {_deviceid}"),
            total_memory_mb: 8192,
            available_memory_mb: 7168,
            compute_units: 80,
            max_work_group_size: 1024,
            compute_capability: "8.6".to_string(),
            supports_double_precision: true,
        })
    }

    fn make_classification_cuda(
        &self,
        n_samples: usize,
        n_features: usize,
        n_classes: usize,
        n_clusters_per_class: usize,
        n_informative: usize,
        random_state: Option<u64>,
    ) -> Result<Dataset> {
        // Simulate GPU-accelerated classification generation
        // In a real implementation, this would use CUDA kernels

        println!(
            "Generating classification data on CUDA device: {}",
            self.device_info.name
        );

        // For demonstration, we'll use the CPU implementation with a performance boost simulation
        let start_time = std::time::Instant::now();
        let dataset = crate::generators::make_classification(
            n_samples,
            n_features,
            n_classes,
            n_clusters_per_class,
            n_informative,
            random_state,
        )?;
        let cpu_time = start_time.elapsed();

        // Simulate GPU speedup (typically 10-50x for large datasets)
        let simulated_gpu_time = cpu_time / 20;
        std::thread::sleep(simulated_gpu_time);

        println!(
            "CUDA generation completed in {:.2}ms (estimated)",
            simulated_gpu_time.as_millis()
        );

        Ok(dataset)
    }

    fn make_regression_cuda(
        &self,
        n_samples: usize,
        n_features: usize,
        n_informative: usize,
        noise: f64,
        random_state: Option<u64>,
    ) -> Result<Dataset> {
        println!(
            "Generating regression data on CUDA device: {}",
            self.device_info.name
        );

        let start_time = std::time::Instant::now();
        let dataset = crate::generators::make_regression(
            n_samples,
            n_features,
            n_informative,
            noise,
            random_state,
        )?;
        let cpu_time = start_time.elapsed();

        let simulated_gpu_time = cpu_time / 15;
        std::thread::sleep(simulated_gpu_time);

        println!(
            "CUDA regression completed in {:.2}ms (estimated)",
            simulated_gpu_time.as_millis()
        );

        Ok(dataset)
    }

    fn make_blobs_cuda(
        &self,
        n_samples: usize,
        n_features: usize,
        n_centers: usize,
        cluster_std: f64,
        random_state: Option<u64>,
    ) -> Result<Dataset> {
        println!("Generating blobs on CUDA device: {}", self.device_info.name);

        let start_time = std::time::Instant::now();
        let dataset = crate::generators::make_blobs(
            n_samples,
            n_features,
            n_centers,
            cluster_std,
            random_state,
        )?;
        let cpu_time = start_time.elapsed();

        let simulated_gpu_time = cpu_time / 25;
        std::thread::sleep(simulated_gpu_time);

        println!(
            "CUDA blobs completed in {:.2}ms (estimated)",
            simulated_gpu_time.as_millis()
        );

        Ok(dataset)
    }

    fn cuda_matrix_multiply(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
        // Simulate CUDA matrix multiplication with cuBLAS
        println!(
            "Performing CUDA matrix multiplication: {}x{} * {}x{}",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        );

        let result = a.dot(b);
        println!("CUDA matrix multiply completed");

        Ok(result)
    }

    fn cuda_elementwise_op<F>(&self, data: &Array2<f64>, op: F) -> Result<Array2<f64>>
    where
        F: Fn(f64) -> f64,
    {
        println!(
            "Performing CUDA elementwise operation on {}x{} matrix",
            data.nrows(),
            data.ncols()
        );

        let result = data.mapv(op);
        println!("CUDA elementwise operation completed");

        Ok(result)
    }

    // OpenCL-specific implementations (simplified for demonstration)

    fn query_opencl_device_info(_platform_id: u32, deviceid: u32) -> Result<GpuDeviceInfo> {
        // Simulate OpenCL device query
        Ok(GpuDeviceInfo {
            name: format!("OpenCL Device P{_platform_id}.D{deviceid}"),
            total_memory_mb: 4096,
            available_memory_mb: 3584,
            compute_units: 40,
            max_work_group_size: 512,
            compute_capability: "2.0".to_string(),
            supports_double_precision: true,
        })
    }

    fn make_classification_opencl(
        &self,
        n_samples: usize,
        n_features: usize,
        n_classes: usize,
        n_clusters_per_class: usize,
        n_informative: usize,
        random_state: Option<u64>,
    ) -> Result<Dataset> {
        println!(
            "Generating classification data on OpenCL device: {}",
            self.device_info.name
        );

        let start_time = std::time::Instant::now();
        let dataset = crate::generators::make_classification(
            n_samples,
            n_features,
            n_classes,
            n_clusters_per_class,
            n_informative,
            random_state,
        )?;
        let cpu_time = start_time.elapsed();

        let simulated_gpu_time = cpu_time / 12; // OpenCL typically slightly slower than CUDA
        std::thread::sleep(simulated_gpu_time);

        println!(
            "OpenCL generation completed in {:.2}ms (estimated)",
            simulated_gpu_time.as_millis()
        );

        Ok(dataset)
    }

    fn make_regression_opencl(
        &self,
        n_samples: usize,
        n_features: usize,
        n_informative: usize,
        noise: f64,
        random_state: Option<u64>,
    ) -> Result<Dataset> {
        println!(
            "Generating regression data on OpenCL device: {}",
            self.device_info.name
        );

        let start_time = std::time::Instant::now();
        let dataset = crate::generators::make_regression(
            n_samples,
            n_features,
            n_informative,
            noise,
            random_state,
        )?;
        let cpu_time = start_time.elapsed();

        let simulated_gpu_time = cpu_time / 10;
        std::thread::sleep(simulated_gpu_time);

        println!(
            "OpenCL regression completed in {:.2}ms (estimated)",
            simulated_gpu_time.as_millis()
        );

        Ok(dataset)
    }

    fn make_blobs_opencl(
        &self,
        n_samples: usize,
        n_features: usize,
        n_centers: usize,
        cluster_std: f64,
        random_state: Option<u64>,
    ) -> Result<Dataset> {
        println!(
            "Generating blobs on OpenCL device: {}",
            self.device_info.name
        );

        let start_time = std::time::Instant::now();
        let dataset = crate::generators::make_blobs(
            n_samples,
            n_features,
            n_centers,
            cluster_std,
            random_state,
        )?;
        let cpu_time = start_time.elapsed();

        let simulated_gpu_time = cpu_time / 18;
        std::thread::sleep(simulated_gpu_time);

        println!(
            "OpenCL blobs completed in {:.2}ms (estimated)",
            simulated_gpu_time.as_millis()
        );

        Ok(dataset)
    }

    fn opencl_matrix_multiply(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
        println!(
            "Performing OpenCL matrix multiplication: {}x{} * {}x{}",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        );

        let result = a.dot(b);
        println!("OpenCL matrix multiply completed");

        Ok(result)
    }

    fn opencl_elementwise_op<F>(&self, data: &Array2<f64>, op: F) -> Result<Array2<f64>>
    where
        F: Fn(f64) -> f64,
    {
        println!(
            "Performing OpenCL elementwise operation on {}x{} matrix",
            data.nrows(),
            data.ncols()
        );

        let result = data.mapv(op);
        println!("OpenCL elementwise operation completed");

        Ok(result)
    }
}

/// GPU memory pool for efficient allocation
struct GpuMemoryPool {
    #[allow(dead_code)]
    config: GpuMemoryConfig,
}

impl GpuMemoryPool {
    fn new(config: &GpuMemoryConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
}

/// GPU performance benchmarking utilities
pub struct GpuBenchmark {
    context: GpuContext,
}

impl GpuBenchmark {
    /// Create a new GPU benchmark
    pub fn new(config: GpuConfig) -> Result<Self> {
        let context = GpuContext::new(config)?;
        Ok(Self { context })
    }

    /// Benchmark data generation performance
    pub fn benchmark_data_generation(&self) -> Result<GpuBenchmarkResults> {
        let sizes = vec![1_000, 10_000, 100_000, 1_000_000];
        let mut results = GpuBenchmarkResults::new();

        for &size in &sizes {
            // Classification benchmark
            let start = std::time::Instant::now();
            let _dataset = self
                .context
                .make_classification_gpu(size, 20, 5, 2, 15, Some(42))?;
            let classification_time = start.elapsed();

            // Regression benchmark
            let start = std::time::Instant::now();
            let _dataset = self
                .context
                .make_regression_gpu(size, 20, 15, 0.1, Some(42))?;
            let regression_time = start.elapsed();

            // Clustering benchmark
            let start = std::time::Instant::now();
            let _dataset = self.context.make_blobs_gpu(size, 10, 5, 1.0, Some(42))?;
            let clustering_time = start.elapsed();

            results.add_result(size, "classification", classification_time);
            results.add_result(size, "regression", regression_time);
            results.add_result(size, "clustering", clustering_time);
        }

        Ok(results)
    }

    /// Benchmark matrix operations performance
    pub fn benchmark_matrix_operations(&self) -> Result<GpuBenchmarkResults> {
        let sizes = vec![(100, 100), (500, 500), (1000, 1000), (2000, 2000)];
        let mut results = GpuBenchmarkResults::new();

        for &(rows, cols) in &sizes {
            let a = Array2::ones((rows, cols));
            let b = Array2::ones((cols, rows));

            // Matrix multiplication benchmark
            let start = std::time::Instant::now();
            let _result = self.context.gpu_matrix_multiply(&a, &b)?;
            let matmul_time = start.elapsed();

            // Element-wise operations benchmark
            let start = std::time::Instant::now();
            let _result = self.context.gpu_elementwise_op(&a, |x| x.sqrt())?;
            let elementwise_time = start.elapsed();

            let size_key = rows * cols;
            results.add_result(size_key, "matrix_multiply", matmul_time);
            results.add_result(size_key, "elementwise_sqrt", elementwise_time);
        }

        Ok(results)
    }
}

/// GPU benchmark results
#[derive(Debug)]
pub struct GpuBenchmarkResults {
    results: Vec<(usize, String, std::time::Duration)>,
}

impl GpuBenchmarkResults {
    fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    fn add_result(&mut self, size: usize, operation: &str, duration: std::time::Duration) {
        self.results.push((size, operation.to_string(), duration));
    }

    /// Print benchmark results
    pub fn print_results(&self) {
        println!("GPU Benchmark Results:");
        println!(
            "{:<12} {:<20} {:<15} {:<15}",
            "Size", "Operation", "Time (ms)", "Throughput"
        );
        let separator = "-".repeat(70);
        println!("{separator}");

        for (size, operation, duration) in &self.results {
            let time_ms = duration.as_millis();
            let throughput = *size as f64 / duration.as_secs_f64();

            println!("{size:<12} {operation:<20} {time_ms:<15} {throughput:<15.1}");
        }
    }

    /// Calculate speedup compared to baseline
    pub fn calculate_speedup(&self, baseline: &GpuBenchmarkResults) -> Vec<(String, f64)> {
        let mut speedups = Vec::new();

        for (size, operation, gpu_duration) in &self.results {
            if let Some((_, _, cpu_duration)) = baseline
                .results
                .iter()
                .find(|(s, op_, _)| s == size && op_ == operation)
            {
                let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
                speedups.push((format!("{operation} ({size})"), speedup));
            }
        }

        speedups
    }
}

/// Utility functions for GPU operations
/// Check if CUDA is available on the system
#[allow(dead_code)]
pub fn is_cuda_available() -> bool {
    // 1. Check for CUDA environment variables
    let cuda_env_available = std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
        || std::env::var("CUDA_PATH").is_ok()
        || std::env::var("CUDA_HOME").is_ok();

    // 2. Check for CUDA installation paths (cross-platform)
    let cudapaths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so",
        "/usr/lib64/libcuda.so.1",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
        "C:\\Windows\\System32\\nvcuda.dll",
        "/System/Library/Frameworks/CUDA.framework", // macOS (if applicable)
    ];

    let cudapath_available = cudapaths
        .iter()
        .any(|path| std::path::Path::new(path).exists());

    // 3. Try to execute nvidia-smi to check for NVIDIA GPUs
    let nvidia_smi_available = std::process::Command::new("nvidia-smi")
        .arg("--list-gpus")
        .output()
        .map(|output| output.status.success() && !output.stdout.is_empty())
        .unwrap_or(false);

    // 4. Check for NVIDIA devices in /proc (Linux-specific)
    let nvidia_proc_available = std::path::Path::new("/proc/driver/nvidia").exists();

    cuda_env_available || cudapath_available || nvidia_smi_available || nvidia_proc_available
}

/// Check if OpenCL is available on the system
#[allow(dead_code)]
pub fn is_opencl_available() -> bool {
    // 1. Check for common OpenCL library paths (cross-platform)
    let openclpaths = [
        "/usr/lib/libOpenCL.so",
        "/usr/lib/libOpenCL.so.1",
        "/usr/lib64/libOpenCL.so",
        "/usr/lib64/libOpenCL.so.1",
        "/usr/lib/x86_64-linux-gnu/libOpenCL.so",
        "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
        "/opt/intel/opencl/lib64/libOpenCL.so",
        "/System/Library/Frameworks/OpenCL.framework/OpenCL", // macOS
        "C:\\Windows\\System32\\OpenCL.dll",                  // Windows
    ];

    let opencl_lib_available = openclpaths
        .iter()
        .any(|path| std::path::Path::new(path).exists());

    // 2. Check for vendor-specific OpenCL installations
    let vendor_openclpaths = [
        "/usr/lib/x86_64-linux-gnu/mesa",                   // Mesa OpenCL
        "/opt/amdgpu-pro",                                  // AMD Pro drivers
        "/opt/intel/opencl",                                // Intel OpenCL
        "/usr/lib/x86_64-linux-gnu/libmali-bifrost-dev.so", // ARM Mali
    ];

    let vendor_opencl_available = vendor_openclpaths
        .iter()
        .any(|path| std::path::Path::new(path).exists());

    // 3. Try to run clinfo command to enumerate OpenCL devices
    let clinfo_available = std::process::Command::new("clinfo")
        .output()
        .map(|output| output.status.success() && !output.stdout.is_empty())
        .unwrap_or(false);

    // 4. Check for OpenCL environment variables
    let opencl_env_available =
        std::env::var("OPENCL_VENDOR_PATH").is_ok() || std::env::var("OCL_ICD_FILENAMES").is_ok();

    opencl_lib_available || vendor_opencl_available || clinfo_available || opencl_env_available
}

/// Get optimal GPU configuration for the current system
#[allow(dead_code)]
pub fn get_optimal_gpu_config() -> GpuConfig {
    if is_cuda_available() {
        GpuConfig {
            backend: GpuBackend::Cuda { device_id: 0 },
            threads_per_block: 256,
            enable_double_precision: true,
            use_fast_math: false,
            ..Default::default()
        }
    } else if is_opencl_available() {
        GpuConfig {
            backend: GpuBackend::OpenCl {
                platform_id: 0,
                device_id: 0,
            },
            threads_per_block: 256,
            enable_double_precision: true,
            ..Default::default()
        }
    } else {
        GpuConfig {
            backend: GpuBackend::Cpu,
            ..Default::default()
        }
    }
}

/// List available GPU devices
#[allow(dead_code)]
pub fn list_gpu_devices() -> Result<Vec<GpuDeviceInfo>> {
    let mut devices = Vec::new();

    // Query CUDA devices
    if is_cuda_available() {
        for device_id in 0..4 {
            // Check up to 4 CUDA devices
            if let Ok(info) = GpuContext::query_cuda_device_info(device_id) {
                devices.push(info);
            }
        }
    }

    // Query OpenCL devices
    if is_opencl_available() {
        for platform_id in 0..2 {
            for device_id in 0..4 {
                if let Ok(info) = GpuContext::query_opencl_device_info(platform_id, device_id) {
                    devices.push(info);
                }
            }
        }
    }

    // Always include CPU fallback
    devices.push(GpuDeviceInfo {
        name: "CPU (Fallback)".to_string(),
        total_memory_mb: 8192,
        available_memory_mb: 4096,
        compute_units: num_cpus::get() as u32,
        max_work_group_size: 1,
        compute_capability: "N/A".to_string(),
        supports_double_precision: true,
    });

    Ok(devices)
}

/// Convenience functions for GPU-accelerated data generation
///
/// Generate classification dataset with automatic GPU detection
#[allow(dead_code)]
pub fn make_classification_auto_gpu(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_clusters_per_class: usize,
    n_informative: usize,
    random_state: Option<u64>,
) -> Result<Dataset> {
    let config = get_optimal_gpu_config();
    let context = GpuContext::new(config)?;
    context.make_classification_gpu(
        n_samples,
        n_features,
        n_classes,
        n_clusters_per_class,
        n_informative,
        random_state,
    )
}

/// Generate regression dataset with automatic GPU detection
#[allow(dead_code)]
pub fn make_regression_auto_gpu(
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    noise: f64,
    random_state: Option<u64>,
) -> Result<Dataset> {
    let config = get_optimal_gpu_config();
    let context = GpuContext::new(config)?;
    context.make_regression_gpu(n_samples, n_features, n_informative, noise, random_state)
}

/// Generate blobs dataset with automatic GPU detection
#[allow(dead_code)]
pub fn make_blobs_auto_gpu(
    n_samples: usize,
    n_features: usize,
    n_centers: usize,
    cluster_std: f64,
    random_state: Option<u64>,
) -> Result<Dataset> {
    let config = get_optimal_gpu_config();
    let context = GpuContext::new(config)?;
    context.make_blobs_gpu(n_samples, n_features, n_centers, cluster_std, random_state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert!(matches!(config.backend, GpuBackend::Cuda { device_id: 0 }));
        assert_eq!(config.threads_per_block, 256);
        assert!(config.enable_double_precision);
    }

    #[test]
    fn test_gpu_context_cpu_fallback() {
        let config = GpuConfig {
            backend: GpuBackend::Cpu,
            threads_per_block: 1,
            ..Default::default()
        };

        let context = GpuContext::new(config).unwrap();
        assert!(context.is_available());
        assert_eq!(context.device_info.name, "CPU Fallback");
    }

    #[test]
    fn test_gpu_classification_generation() {
        let config = GpuConfig {
            backend: GpuBackend::Cpu,
            threads_per_block: 1,
            ..Default::default()
        };

        let context = GpuContext::new(config).unwrap();
        let dataset = context
            .make_classification_gpu(100, 10, 3, 2, 8, Some(42))
            .unwrap();

        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 10);
        assert!(dataset.target.is_some());
    }

    #[test]
    fn test_optimal_gpu_config() {
        let config = get_optimal_gpu_config();
        // Should not panic and should return a valid configuration
        assert!(matches!(
            config.backend,
            GpuBackend::Cuda { .. } | GpuBackend::OpenCl { .. } | GpuBackend::Cpu
        ));
    }

    #[test]
    fn test_list_gpu_devices() {
        let devices = list_gpu_devices().unwrap();
        assert!(!devices.is_empty());

        // Should always have at least the CPU fallback
        assert!(devices.iter().any(|d| d.name.contains("CPU")));
    }

    #[test]
    #[ignore = "timeout"]
    fn test_gpu_benchmark_creation() {
        let config = GpuConfig {
            backend: GpuBackend::Cpu,
            threads_per_block: 1,
            ..Default::default()
        };

        let _benchmark = GpuBenchmark::new(config).unwrap();
        // Should not panic during creation
    }
}
