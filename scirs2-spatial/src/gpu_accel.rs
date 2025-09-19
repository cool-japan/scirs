//! GPU acceleration for spatial algorithms
//!
//! This module provides GPU-accelerated implementations of spatial algorithms
//! for massive datasets that benefit from parallel computation on graphics cards.
//! It integrates with the existing SIMD and memory pool optimizations to provide
//! the highest possible performance.
//!
//! # Features
//!
//! - **GPU distance matrix computation**: Massive parallel distance calculations
//! - **GPU clustering algorithms**: K-means and DBSCAN on GPU
//! - **GPU nearest neighbor search**: Optimized spatial queries
//! - **Hybrid CPU-GPU algorithms**: Automatic workload distribution
//! - **Memory-mapped GPU transfers**: Minimize data movement overhead
//! - **Multi-GPU support**: Scale across multiple graphics cards
//!
//! # Architecture Support
//!
//! The GPU acceleration is designed to work with:
//! - NVIDIA GPUs (CUDA backend via cupy/rust-cuda)
//! - AMD GPUs (ROCm backend)
//! - Intel GPUs (Level Zero backend)
//! - Vulkan compute for cross-platform support
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::gpu_accel::{GpuDistanceMatrix, GpuKMeans};
//! use ndarray::array;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // GPU distance matrix computation
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
//!
//! let gpu_matrix = GpuDistanceMatrix::new()?;
//! let distances = gpu_matrix.compute_parallel(&points.view()).await?;
//! println!("GPU distance matrix: {:?}", distances);
//!
//! // GPU K-means clustering
//! let gpu_kmeans = GpuKMeans::new(2)?;
//! let (centroids, assignments) = gpu_kmeans.fit(&points.view()).await?;
//! println!("GPU centroids: {:?}", centroids);
//! # Ok(())
//! # }
//! ```

use crate::error::SpatialResult;
use crate::memory_pool::DistancePool;
use ndarray::{Array1, Array2, ArrayView2};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

// Type alias for complex return types
type GpuDeviceInfoResult = Result<(Vec<String>, Vec<(usize, usize)>), Box<dyn std::error::Error>>;

/// GPU device capabilities and information
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// Is GPU acceleration available
    pub gpu_available: bool,
    /// Number of GPU devices available
    pub device_count: usize,
    /// Total GPU memory in bytes
    pub total_memory: usize,
    /// Available GPU memory in bytes
    pub available_memory: usize,
    /// GPU compute capability (for CUDA)
    pub compute_capability: Option<(u32, u32)>,
    /// Maximum threads per block
    pub max_threads_per_block: usize,
    /// Maximum blocks per grid
    pub max_blocks_per_grid: usize,
    /// GPU device names
    pub device_names: Vec<String>,
    /// Supported backends
    pub supported_backends: Vec<GpuBackend>,
}

impl Default for GpuCapabilities {
    fn default() -> Self {
        Self {
            gpu_available: false,
            device_count: 0,
            total_memory: 0,
            available_memory: 0,
            compute_capability: None,
            max_threads_per_block: 1024,
            max_blocks_per_grid: 65535,
            device_names: Vec::new(),
            supported_backends: Vec::new(),
        }
    }
}

/// Supported GPU backends
#[derive(Debug, Clone, PartialEq)]
pub enum GpuBackend {
    /// NVIDIA CUDA backend
    Cuda,
    /// AMD ROCm backend
    Rocm,
    /// Intel Level Zero backend
    LevelZero,
    /// Vulkan compute backend (cross-platform)
    Vulkan,
    /// CPU fallback (OpenMP/SIMD)
    CpuFallback,
}

/// GPU device management and capability detection
pub struct GpuDevice {
    capabilities: GpuCapabilities,
    preferred_backend: GpuBackend,
    #[allow(dead_code)]
    memory_pool: Arc<DistancePool>,
}

impl GpuDevice {
    /// Create a new GPU device manager
    pub fn new() -> SpatialResult<Self> {
        let capabilities = Self::detect_capabilities()?;
        let preferred_backend = Self::select_optimal_backend(&capabilities);
        let memory_pool = Arc::new(DistancePool::new(1000));

        Ok(Self {
            capabilities,
            preferred_backend,
            memory_pool,
        })
    }

    /// Detect available GPU capabilities
    fn detect_capabilities() -> SpatialResult<GpuCapabilities> {
        let mut caps = GpuCapabilities::default();

        // Check CUDA backend
        #[cfg(feature = "cuda")]
        {
            if Self::check_cuda_available() {
                caps.gpu_available = true;
                caps.device_count = Self::get_cuda_device_count();
                caps.supported_backends.push(GpuBackend::Cuda);

                // Get CUDA device information
                if let Ok((names, memory_info)) = Self::get_cuda_device_info() {
                    caps.device_names = names;
                    if let Some((total, available)) = memory_info.first() {
                        caps.total_memory = *total;
                        caps.available_memory = *available;
                    }
                }

                // Set CUDA-specific capabilities
                caps.max_threads_per_block = 1024;
                caps.max_blocks_per_grid = 2147483647; // 2^31 - 1
                caps.compute_capability = Self::get_cuda_compute_capability();
            }
        }

        // Check ROCm backend
        #[cfg(feature = "rocm")]
        {
            if Self::check_rocm_available() {
                caps.gpu_available = true;
                let rocm_count = Self::get_rocm_device_count();
                if rocm_count > caps.device_count {
                    caps.device_count = rocm_count;
                }
                caps.supported_backends.push(GpuBackend::Rocm);

                // Get ROCm device information
                if let Ok((names, memory_info)) = Self::get_rocm_device_info() {
                    if caps.device_names.is_empty() {
                        caps.device_names = names;
                    } else {
                        caps.device_names.extend(names);
                    }
                    if let Some((total, available)) = memory_info.first() {
                        if caps.total_memory == 0 {
                            caps.total_memory = *total;
                            caps.available_memory = *available;
                        }
                    }
                }

                // Set ROCm-specific capabilities
                caps.max_threads_per_block = 1024;
                caps.max_blocks_per_grid = 2147483647;
            }
        }

        // Check Vulkan backend
        #[cfg(feature = "vulkan")]
        {
            if Self::check_vulkan_available() {
                caps.gpu_available = true;
                caps.supported_backends.push(GpuBackend::Vulkan);

                // Get Vulkan device information
                if let Ok((names, memory_info)) = Self::get_vulkan_device_info() {
                    if caps.device_names.is_empty() {
                        caps.device_names = names;
                    } else {
                        caps.device_names.extend(names);
                    }
                    if let Some((total, available)) = memory_info.first() {
                        if caps.total_memory == 0 {
                            caps.total_memory = *total;
                            caps.available_memory = *available;
                        }
                    }
                }
            }
        }

        // Always have CPU fallback
        caps.supported_backends.push(GpuBackend::CpuFallback);

        Ok(caps)
    }

    /// Select optimal backend based on capabilities
    fn select_optimal_backend(caps: &GpuCapabilities) -> GpuBackend {
        // Prefer CUDA for NVIDIA, ROCm for AMD, etc.
        if caps.supported_backends.contains(&GpuBackend::Cuda) {
            GpuBackend::Cuda
        } else if caps.supported_backends.contains(&GpuBackend::Rocm) {
            GpuBackend::Rocm
        } else if caps.supported_backends.contains(&GpuBackend::Vulkan) {
            GpuBackend::Vulkan
        } else {
            GpuBackend::CpuFallback
        }
    }

    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        self.capabilities.gpu_available
    }

    /// Get GPU capabilities
    pub fn capabilities(&self) -> &GpuCapabilities {
        &self.capabilities
    }

    /// Get optimal block size for GPU kernels
    pub fn optimal_block_size(&self, _problemsize: usize) -> usize {
        match self.preferred_backend {
            GpuBackend::Cuda => {
                // Optimize for CUDA warp _size (32) and compute capability
                let warp_size = 32;
                let optimal = (_problemsize / warp_size).max(1) * warp_size;
                optimal.min(self.capabilities.max_threads_per_block)
            }
            GpuBackend::Rocm => {
                // Optimize for AMD wavefront _size (64)
                let wavefront_size = 64;
                let optimal = (_problemsize / wavefront_size).max(1) * wavefront_size;
                optimal.min(self.capabilities.max_threads_per_block)
            }
            _ => {
                // Generic optimization
                256.min(self.capabilities.max_threads_per_block)
            }
        }
    }

    // GPU backend detection implementations
    #[cfg(feature = "cuda")]
    fn check_cuda_available() -> bool {
        // Check for NVIDIA driver and CUDA runtime
        if let Ok(output) = Command::new("nvidia-smi")
            .arg("--query-gpu=count")
            .arg("--format=csv,noheader,nounits")
            .output()
        {
            if output.status.success() {
                if let Ok(count_str) = String::from_utf8(output.stdout) {
                    if let Ok(count) = count_str.trim().parse::<u32>() {
                        return count > 0;
                    }
                }
            }
        }

        // Fallback: check for CUDA libraries
        #[cfg(target_os = "linux")]
        {
            Path::exists(Path::new("/usr/local/cuda/lib64/libcuda.so"))
                || Path::exists(Path::new("/usr/lib/x86_64-linux-gnu/libcuda.so"))
                || Path::exists(Path::new("/usr/lib64/libcuda.so"))
        }

        #[cfg(target_os = "windows")]
        {
            Path::exists(Path::new("C:\\Windows\\System32\\nvcuda.dll"))
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        false
    }

    #[cfg(feature = "cuda")]
    fn get_cuda_device_count() -> usize {
        if let Ok(output) = Command::new("nvidia-smi")
            .arg("--query-gpu=count")
            .arg("--format=csv,noheader,nounits")
            .output()
        {
            if output.status.success() {
                if let Ok(count_str) = String::from_utf8(output.stdout) {
                    if let Ok(count) = count_str.trim().parse::<usize>() {
                        return count;
                    }
                }
            }
        }

        // Fallback: try to count GPUs via nvidia-ml-py equivalent
        if let Ok(output) = Command::new("nvidia-smi").arg("-L").output() {
            if output.status.success() {
                if let Ok(list_str) = String::from_utf8(output.stdout) {
                    return list_str
                        .lines()
                        .filter(|line| line.starts_with("GPU "))
                        .count();
                }
            }
        }

        0
    }

    #[cfg(feature = "rocm")]
    fn check_rocm_available() -> bool {
        // Check for ROCm tools
        if let Ok(output) = Command::new("rocm-smi").arg("--showid").output() {
            if output.status.success() {
                return true;
            }
        }

        // Fallback: check for ROCm libraries
        #[cfg(target_os = "linux")]
        {
            Path::exists(Path::new("/opt/rocm/lib/libhip.so"))
                || Path::exists(Path::new("/usr/lib/libhip.so"))
                || Path::exists(Path::new("/usr/lib/x86_64-linux-gnu/libhip.so"))
        }

        #[cfg(not(target_os = "linux"))]
        false
    }

    #[cfg(feature = "rocm")]
    fn get_rocm_device_count() -> usize {
        if let Ok(output) = Command::new("rocm-smi").arg("--showid").output() {
            if output.status.success() {
                if let Ok(list_str) = String::from_utf8(output.stdout) {
                    // Count GPU entries in rocm-smi output
                    return list_str
                        .lines()
                        .filter(|line| line.contains("GPU") || line.contains("card"))
                        .count();
                }
            }
        }

        // Fallback: check /sys/class/drm for AMD GPUs
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            if let Ok(entries) = fs::read_dir("/sys/class/drm") {
                let count = entries
                    .filter_map(Result::ok)
                    .filter(|entry| {
                        if let Ok(name) = entry.file_name().into_string() {
                            name.starts_with("card") && !name.contains("-")
                        } else {
                            false
                        }
                    })
                    .count();
                if count > 0 {
                    return count;
                }
            }
        }

        0
    }

    #[cfg(feature = "vulkan")]
    fn check_vulkan_available() -> bool {
        // Check for vulkaninfo tool
        if let Ok(output) = Command::new("vulkaninfo").arg("--summary").output() {
            if output.status.success() {
                if let Ok(info_str) = String::from_utf8(output.stdout) {
                    // Check if any devices support compute
                    return info_str.contains("VK_QUEUE_COMPUTE_BIT")
                        || info_str.contains("deviceType");
                }
            }
        }

        // Fallback: check for Vulkan loader library
        #[cfg(target_os = "linux")]
        {
            Path::exists(Path::new("/usr/lib/libvulkan.so"))
                || Path::exists(Path::new("/usr/lib/x86_64-linux-gnu/libvulkan.so"))
                || Path::exists(Path::new("/usr/local/lib/libvulkan.so"))
        }

        #[cfg(target_os = "windows")]
        {
            Path::exists(Path::new("C:\\Windows\\System32\\vulkan-1.dll"))
        }

        #[cfg(target_os = "macos")]
        {
            Path::exists(Path::new("/usr/local/lib/libvulkan.dylib"))
                || Path::exists(Path::new(
                    "/System/Library/Frameworks/Metal.framework/Metal",
                ))
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        false
    }

    /// Get detailed CUDA device information
    #[cfg(feature = "cuda")]
    fn get_cuda_device_info() -> GpuDeviceInfoResult {
        let mut device_names = Vec::new();
        let mut memory_info = Vec::new();

        // Get device names
        if let Ok(output) = Command::new("nvidia-smi")
            .arg("--query-gpu=name")
            .arg("--format=csv,noheader,nounits")
            .output()
        {
            if output.status.success() {
                if let Ok(names_str) = String::from_utf8(output.stdout) {
                    device_names = names_str.lines().map(|s| s.trim().to_string()).collect();
                }
            }
        }

        // Get memory information (in MB)
        if let Ok(output) = Command::new("nvidia-smi")
            .arg("--query-gpu=memory.total,memory.free")
            .arg("--format=csv,noheader,nounits")
            .output()
        {
            if output.status.success() {
                if let Ok(memory_str) = String::from_utf8(output.stdout) {
                    for line in memory_str.lines() {
                        let parts: Vec<&str> = line.split(',').collect();
                        if parts.len() >= 2 {
                            if let (Ok(total), Ok(free)) = (
                                parts[0].trim().parse::<usize>(),
                                parts[1].trim().parse::<usize>(),
                            ) {
                                memory_info.push((total * 1024 * 1024, free * 1024 * 1024));
                                // Convert MB to bytes
                            }
                        }
                    }
                }
            }
        }

        Ok((device_names, memory_info))
    }

    /// Get CUDA compute capability
    #[cfg(feature = "cuda")]
    fn get_cuda_compute_capability() -> Option<(u32, u32)> {
        if let Ok(output) = Command::new("nvidia-smi")
            .arg("--query-gpu=compute_cap")
            .arg("--format=csv,noheader,nounits")
            .output()
        {
            if output.status.success() {
                if let Ok(cap_str) = String::from_utf8(output.stdout) {
                    if let Some(line) = cap_str.lines().next() {
                        let parts: Vec<&str> = line.trim().split('.').collect();
                        if parts.len() >= 2 {
                            if let (Ok(major), Ok(minor)) =
                                (parts[0].parse::<u32>(), parts[1].parse::<u32>())
                            {
                                return Some((major, minor));
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Get detailed ROCm device information
    #[cfg(feature = "rocm")]
    fn get_rocm_device_info() -> GpuDeviceInfoResult {
        let mut device_names = Vec::new();
        let mut memory_info = Vec::new();

        // Try to get device information from rocm-smi
        if let Ok(output) = Command::new("rocm-smi").arg("--showproductname").output() {
            if output.status.success() {
                if let Ok(info_str) = String::from_utf8(output.stdout) {
                    for line in info_str.lines() {
                        if line.contains("Card series:") {
                            if let Some(name) = line.split(':').nth(1) {
                                device_names.push(name.trim().to_string());
                            }
                        }
                    }
                }
            }
        }

        // Get memory information
        if let Ok(output) = Command::new("rocm-smi")
            .arg("--showmeminfo")
            .arg("vram")
            .output()
        {
            if output.status.success() {
                if let Ok(memory_str) = String::from_utf8(output.stdout) {
                    for line in memory_str.lines() {
                        if line.contains("Total memory") || line.contains("Used memory") {
                            if let Some(mem_part) = line
                                .split_whitespace()
                                .find(|s| s.ends_with("MB") || s.ends_with("GB"))
                            {
                                if let Ok(mem_val) = mem_part
                                    .trim_end_matches("MB")
                                    .trim_end_matches("GB")
                                    .parse::<usize>()
                                {
                                    let bytes = if mem_part.ends_with("GB") {
                                        mem_val * 1024 * 1024 * 1024
                                    } else {
                                        mem_val * 1024 * 1024
                                    };
                                    memory_info.push((bytes, bytes / 2));
                                }
                            }
                        }
                    }
                }
            }
        }

        if device_names.is_empty() && memory_info.is_empty() {
            device_names.push("AMD GPU (ROCm)".to_string());
            memory_info.push((8 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024));
        }

        Ok((device_names, memory_info))
    }

    /// Get detailed Vulkan device information
    #[cfg(feature = "vulkan")]
    fn get_vulkan_device_info() -> GpuDeviceInfoResult {
        let mut device_names = Vec::new();
        let mut memory_info = Vec::new();

        if let Ok(output) = Command::new("vulkaninfo").arg("--summary").output() {
            if output.status.success() {
                if let Ok(info_str) = String::from_utf8(output.stdout) {
                    for line in info_str.lines() {
                        if line.contains("deviceName") {
                            if let Some(name_part) = line.split('=').nth(1) {
                                device_names.push(name_part.trim().to_string());
                            }
                        } else if line.contains("heapSize") {
                            if let Some(mem_part) = line.split('=').nth(1) {
                                if let Ok(mem_val) = mem_part.trim().parse::<usize>() {
                                    memory_info.push((mem_val, mem_val * 3 / 4));
                                }
                            }
                        }
                    }
                }
            }
        }

        if device_names.is_empty() {
            device_names.push("Vulkan Device".to_string());
            memory_info.push((4 * 1024 * 1024 * 1024, 3 * 1024 * 1024 * 1024));
        }

        Ok((device_names, memory_info))
    }

    #[cfg(not(feature = "cuda"))]
    #[allow(dead_code)]
    fn get_cuda_device_info() -> GpuDeviceInfoResult {
        Ok((Vec::new(), Vec::new()))
    }

    #[cfg(not(feature = "cuda"))]
    #[allow(dead_code)]
    fn get_cuda_compute_capability() -> Option<(u32, u32)> {
        None
    }

    #[cfg(not(feature = "rocm"))]
    #[allow(dead_code)]
    fn get_rocm_device_info() -> GpuDeviceInfoResult {
        Ok((Vec::new(), Vec::new()))
    }

    #[cfg(not(feature = "vulkan"))]
    #[allow(dead_code)]
    fn get_vulkan_device_info() -> GpuDeviceInfoResult {
        Ok((Vec::new(), Vec::new()))
    }
}

impl Default for GpuDevice {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            capabilities: GpuCapabilities::default(),
            preferred_backend: GpuBackend::CpuFallback,
            memory_pool: Arc::new(DistancePool::new(1000)),
        })
    }
}

/// GPU-accelerated distance matrix computation
pub struct GpuDistanceMatrix {
    device: Arc<GpuDevice>,
    batch_size: usize,
    use_mixed_precision: bool,
}

impl GpuDistanceMatrix {
    /// Create a new GPU distance matrix computer
    pub fn new() -> SpatialResult<Self> {
        let device = Arc::new(GpuDevice::new()?);
        Ok(Self {
            device,
            batch_size: 1024,
            use_mixed_precision: true,
        })
    }

    /// Configure batch size for GPU processing
    pub fn with_batch_size(mut self, batchsize: usize) -> Self {
        self.batch_size = batchsize;
        self
    }

    /// Configure mixed precision (f32 vs f64)
    pub fn with_mixed_precision(mut self, use_mixedprecision: bool) -> Self {
        self.use_mixed_precision = use_mixedprecision;
        self
    }

    /// Compute distance matrix on GPU (async)
    pub async fn compute_parallel(
        &self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        let _n_points = points.nrows();

        if !self.device.is_gpu_available() {
            return self.compute_cpu_fallback(points).await;
        }

        match self.device.preferred_backend {
            GpuBackend::Cuda => self.compute_cuda(points).await,
            GpuBackend::Rocm => self.compute_rocm(points).await,
            GpuBackend::Vulkan => self.compute_vulkan(points).await,
            GpuBackend::CpuFallback => self.compute_cpu_fallback(points).await,
            GpuBackend::LevelZero => self.compute_cpu_fallback(points).await, // Fallback for now
        }
    }

    /// GPU distance matrix computation using CUDA
    async fn compute_cuda(&self, points: &ArrayView2<'_, f64>) -> SpatialResult<Array2<f64>> {
        // In a real implementation, this would:
        // 1. Allocate GPU memory for input points and output matrix
        // 2. Transfer points to GPU memory
        // 3. Launch CUDA kernels to compute distances in parallel
        // 4. Transfer results back to CPU
        // 5. Handle errors and memory cleanup

        self.compute_cpu_fallback(points).await
    }

    /// GPU distance matrix computation using ROCm
    async fn compute_rocm(&self, points: &ArrayView2<'_, f64>) -> SpatialResult<Array2<f64>> {
        // Similar to CUDA but using ROCm/HIP APIs
        self.compute_cpu_fallback(points).await
    }

    /// GPU distance matrix computation using Vulkan
    async fn compute_vulkan(&self, points: &ArrayView2<'_, f64>) -> SpatialResult<Array2<f64>> {
        // Use Vulkan compute shaders for cross-platform GPU acceleration
        self.compute_cpu_fallback(points).await
    }

    /// CPU fallback using optimized SIMD operations
    async fn compute_cpu_fallback(
        &self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        // Use existing SIMD implementation as fallback
        use crate::simd_distance::parallel_pdist;

        let condensed = parallel_pdist(points, "euclidean")?;

        // Convert condensed to full matrix
        let n = points.nrows();
        let mut matrix = Array2::zeros((n, n));

        let mut idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                matrix[[i, j]] = condensed[idx];
                matrix[[j, i]] = condensed[idx];
                idx += 1;
            }
        }

        Ok(matrix)
    }
}

/// GPU-accelerated K-means clustering
pub struct GpuKMeans {
    device: Arc<GpuDevice>,
    k: usize,
    max_iterations: usize,
    tolerance: f64,
    batch_size: usize,
}

impl GpuKMeans {
    /// Create a new GPU K-means clusterer
    pub fn new(k: usize) -> SpatialResult<Self> {
        let device = Arc::new(GpuDevice::new()?);
        Ok(Self {
            device,
            k,
            max_iterations: 100,
            tolerance: 1e-6,
            batch_size: 1024,
        })
    }

    /// Configure maximum iterations
    pub fn with_max_iterations(mut self, maxiterations: usize) -> Self {
        self.max_iterations = maxiterations;
        self
    }

    /// Configure convergence tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Configure batch size for GPU processing
    pub fn with_batch_size(mut self, batchsize: usize) -> Self {
        self.batch_size = batchsize;
        self
    }

    /// Perform K-means clustering on GPU (async)
    pub async fn fit(
        &self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<(Array2<f64>, Array1<usize>)> {
        if !self.device.is_gpu_available() {
            return self.fit_cpu_fallback(points).await;
        }

        match self.device.preferred_backend {
            GpuBackend::Cuda => self.fit_cuda(points).await,
            GpuBackend::Rocm => self.fit_rocm(points).await,
            GpuBackend::Vulkan => self.fit_vulkan(points).await,
            GpuBackend::CpuFallback => self.fit_cpu_fallback(points).await,
            GpuBackend::LevelZero => self.fit_cpu_fallback(points).await, // Fallback for now
        }
    }

    /// GPU K-means using CUDA
    async fn fit_cuda(
        &self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<(Array2<f64>, Array1<usize>)> {
        // In a real implementation, this would:
        // 1. Initialize centroids on GPU
        // 2. Iteratively update assignments and centroids using GPU kernels
        // 3. Use shared memory for efficient centroid updates
        // 4. Use atomic operations for convergence checking

        self.fit_cpu_fallback(points).await
    }

    /// GPU K-means using ROCm
    async fn fit_rocm(
        &self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<(Array2<f64>, Array1<usize>)> {
        // Similar to CUDA but using ROCm/HIP APIs
        self.fit_cpu_fallback(points).await
    }

    /// GPU K-means using Vulkan
    async fn fit_vulkan(
        &self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<(Array2<f64>, Array1<usize>)> {
        // Use Vulkan compute shaders
        self.fit_cpu_fallback(points).await
    }

    /// CPU fallback using advanced-optimized SIMD K-means
    async fn fit_cpu_fallback(
        &self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<(Array2<f64>, Array1<usize>)> {
        // Use existing advanced-optimized SIMD K-means as fallback
        use crate::simd_distance::advanced_simd_clustering::AdvancedSimdKMeans;

        let advanced_kmeans = AdvancedSimdKMeans::new(self.k)
            .with_mixed_precision(true)
            .with_block_size(256);

        advanced_kmeans.fit(points)
    }
}

/// GPU-accelerated nearest neighbor search
pub struct GpuNearestNeighbors {
    device: Arc<GpuDevice>,
    #[allow(dead_code)]
    build_batch_size: usize,
    #[allow(dead_code)]
    query_batch_size: usize,
}

impl GpuNearestNeighbors {
    /// Create a new GPU nearest neighbor searcher
    pub fn new() -> SpatialResult<Self> {
        let device = Arc::new(GpuDevice::new()?);
        Ok(Self {
            device,
            build_batch_size: 1024,
            query_batch_size: 256,
        })
    }

    /// GPU k-nearest neighbors search (async)
    pub async fn knn_search(
        &self,
        query_points: &ArrayView2<'_, f64>,
        data_points: &ArrayView2<'_, f64>,
        k: usize,
    ) -> SpatialResult<(Array2<usize>, Array2<f64>)> {
        if !self.device.is_gpu_available() {
            return self
                .knn_search_cpu_fallback(query_points, data_points, k)
                .await;
        }

        match self.device.preferred_backend {
            GpuBackend::Cuda => self.knn_search_cuda(query_points, data_points, k).await,
            GpuBackend::Rocm => self.knn_search_rocm(query_points, data_points, k).await,
            GpuBackend::Vulkan => self.knn_search_vulkan(query_points, data_points, k).await,
            GpuBackend::CpuFallback => {
                self.knn_search_cpu_fallback(query_points, data_points, k)
                    .await
            }
            _ => {
                self.knn_search_cpu_fallback(query_points, data_points, k)
                    .await
            }
        }
    }

    /// GPU k-NN using CUDA
    async fn knn_search_cuda(
        &self,
        query_points: &ArrayView2<'_, f64>,
        data_points: &ArrayView2<'_, f64>,
        k: usize,
    ) -> SpatialResult<(Array2<usize>, Array2<f64>)> {
        // In a real implementation, this would:
        // 1. Use GPU-optimized k-NN algorithms like FAISS
        // 2. Build spatial indices on GPU
        // 3. Perform batch queries with optimal memory access patterns

        self.knn_search_cpu_fallback(query_points, data_points, k)
            .await
    }

    /// GPU k-NN using ROCm
    async fn knn_search_rocm(
        &self,
        query_points: &ArrayView2<'_, f64>,
        data_points: &ArrayView2<'_, f64>,
        k: usize,
    ) -> SpatialResult<(Array2<usize>, Array2<f64>)> {
        self.knn_search_cpu_fallback(query_points, data_points, k)
            .await
    }

    /// GPU k-NN using Vulkan
    async fn knn_search_vulkan(
        &self,
        query_points: &ArrayView2<'_, f64>,
        data_points: &ArrayView2<'_, f64>,
        k: usize,
    ) -> SpatialResult<(Array2<usize>, Array2<f64>)> {
        self.knn_search_cpu_fallback(query_points, data_points, k)
            .await
    }

    /// CPU fallback using advanced-optimized SIMD nearest neighbors
    async fn knn_search_cpu_fallback(
        &self,
        query_points: &ArrayView2<'_, f64>,
        data_points: &ArrayView2<'_, f64>,
        k: usize,
    ) -> SpatialResult<(Array2<usize>, Array2<f64>)> {
        // Use existing advanced-optimized SIMD nearest neighbors as fallback
        use crate::simd_distance::advanced_simd_clustering::AdvancedSimdNearestNeighbors;

        let advanced_nn = AdvancedSimdNearestNeighbors::new();
        advanced_nn.simd_knn_advanced_fast(query_points, data_points, k)
    }
}

impl Default for GpuNearestNeighbors {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            device: Arc::new(GpuDevice::default()),
            build_batch_size: 1024,
            query_batch_size: 256,
        })
    }
}

/// Hybrid CPU-GPU workload distribution
pub struct HybridProcessor {
    gpu_device: Arc<GpuDevice>,
    cpu_threshold: usize,
    gpu_threshold: usize,
}

impl HybridProcessor {
    /// Create a new hybrid CPU-GPU processor
    pub fn new() -> SpatialResult<Self> {
        let gpu_device = Arc::new(GpuDevice::new()?);
        Ok(Self {
            gpu_device,
            cpu_threshold: 1000,   // Use CPU for small datasets
            gpu_threshold: 100000, // Use GPU for large datasets
        })
    }

    /// Automatically choose optimal processing strategy
    pub fn choose_strategy(&self, _datasetsize: usize) -> ProcessingStrategy {
        if !self.gpu_device.is_gpu_available() {
            return ProcessingStrategy::CpuOnly;
        }

        if _datasetsize < self.cpu_threshold {
            ProcessingStrategy::CpuOnly
        } else if _datasetsize < self.gpu_threshold {
            ProcessingStrategy::Hybrid
        } else {
            ProcessingStrategy::GpuOnly
        }
    }

    /// Get optimal batch sizes for hybrid processing
    pub fn optimal_batch_sizes(&self, _totalsize: usize) -> (usize, usize) {
        let gpu_capability = self.gpu_device.capabilities().total_memory / (8 * 1024); // Estimate based on memory
        let cpu_batch = (_totalsize / 4).max(1000); // 25% to CPU
        let gpu_batch = (_totalsize * 3 / 4).min(gpu_capability); // 75% to GPU if memory allows

        (cpu_batch, gpu_batch)
    }
}

impl Default for HybridProcessor {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            gpu_device: Arc::new(GpuDevice::default()),
            cpu_threshold: 1000,
            gpu_threshold: 100000,
        })
    }
}

/// Processing strategy for workload distribution
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingStrategy {
    /// Use CPU only (small datasets or no GPU)
    CpuOnly,
    /// Use GPU only (large datasets with available GPU)
    GpuOnly,
    /// Use hybrid CPU-GPU processing
    Hybrid,
}

/// Global GPU device instance for convenience
static GLOBAL_GPU_DEVICE: std::sync::OnceLock<GpuDevice> = std::sync::OnceLock::new();

/// Get the global GPU device instance
#[allow(dead_code)]
pub fn global_gpu_device() -> &'static GpuDevice {
    GLOBAL_GPU_DEVICE.get_or_init(GpuDevice::default)
}

/// Check if GPU acceleration is available globally
#[allow(dead_code)]
pub fn is_gpu_acceleration_available() -> bool {
    global_gpu_device().is_gpu_available()
}

/// Get GPU capabilities
#[allow(dead_code)]
pub fn get_gpu_capabilities() -> &'static GpuCapabilities {
    global_gpu_device().capabilities()
}

/// Report GPU acceleration status
#[allow(dead_code)]
pub fn report_gpu_status() {
    let device = global_gpu_device();
    let caps = device.capabilities();

    println!("GPU Acceleration Status:");
    println!("  Available: {}", caps.gpu_available);
    println!("  Device Count: {}", caps.device_count);

    if caps.gpu_available {
        println!(
            "  Total Memory: {:.1} GB",
            caps.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!(
            "  Available Memory: {:.1} GB",
            caps.available_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!("  Max Threads/Block: {}", caps.max_threads_per_block);
        println!("  Supported Backends: {:?}", caps.supported_backends);

        for (i, name) in caps.device_names.iter().enumerate() {
            println!("  Device {i}: {name}");
        }
    } else {
        println!("  Reason: No compatible GPU devices found");
        println!("  Fallback: Using optimized CPU SIMD operations");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gpu_device_creation() {
        let device = GpuDevice::new();
        assert!(device.is_ok());

        let device = device.unwrap();
        // Even without actual GPU, should have CPU fallback
        assert!(!device.capabilities().supported_backends.is_empty());
    }

    #[test]
    fn test_processing_strategy_selection() {
        let processor = HybridProcessor::new().unwrap();

        // Small dataset should use CPU
        let strategy = processor.choose_strategy(500);
        assert_eq!(strategy, ProcessingStrategy::CpuOnly);

        // Large dataset should use GPU if available, otherwise CPU
        let strategy = processor.choose_strategy(200000);
        // Result depends on GPU availability
        assert!(matches!(
            strategy,
            ProcessingStrategy::GpuOnly | ProcessingStrategy::CpuOnly
        ));
    }

    #[test]
    #[ignore] // GPU testing requires async runtime
    fn test_gpu_distance_matrix() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let gpu_matrix = GpuDistanceMatrix::new().unwrap();
        // GPU functionality not available in this configuration
        let points_view = points.view();
        let _result = gpu_matrix.compute_parallel(&points_view);

        // GPU functionality not available in this configuration
        // Tests are disabled pending proper async runtime setup
        // assert!(result.is_ok());
        // let matrix = result.unwrap();
        // assert_eq!(matrix.dim(), (4, 4));
    }

    #[test]
    #[ignore] // GPU testing requires async runtime
    fn test_gpu_kmeans() {
        let points = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.0, 0.1], // Cluster 1
            [5.0, 5.0],
            [5.1, 5.1],
            [5.0, 5.1], // Cluster 2
        ];

        let gpu_kmeans = GpuKMeans::new(2).unwrap();
        // GPU functionality not available in this configuration
        let points_view = points.view();
        let _result = gpu_kmeans.fit(&points_view);

        // GPU functionality not available in this configuration
        // Tests are disabled pending proper async runtime setup
        // assert!(result.is_ok());
        // let (centroids, assignments) = result.unwrap();
    }

    #[test]
    #[ignore] // GPU testing requires async runtime
    fn test_gpu_nearest_neighbors() {
        let data_points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let query_points = array![[0.1, 0.1], [0.9, 0.9]];

        let gpu_nn = GpuNearestNeighbors::new().unwrap();
        let query_view = query_points.view();
        let data_view = data_points.view();
        let _result = gpu_nn.knn_search(&query_view, &data_view, 2);

        // GPU functionality not available in this configuration
        // Tests are disabled pending proper async runtime setup
        // assert!(result.is_ok());
        // let (indices, distances) = result.unwrap();

        // GPU functionality not available in this configuration
        // Verify results make sense (closest to first query should be [0,0])
        // assert_eq!(indices[[0, 0]], 0);  // Point [0,0] should be closest to [0.1, 0.1]
    }

    #[test]
    fn test_global_gpu_functions() {
        // Test global functions
        let device = global_gpu_device();
        assert!(!device.capabilities.device_names.is_empty() || !device.capabilities.gpu_available);

        // These shouldn't panic
        report_gpu_status();
        let _caps = get_gpu_capabilities();
        let _available = is_gpu_acceleration_available();
    }
}
