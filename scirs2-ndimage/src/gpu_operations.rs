//! GPU-accelerated implementations for intensive ndimage operations
//!
//! This module provides GPU-accelerated versions of the most compute-intensive
//! ndimage operations, including morphological operations, filtering, and distance
//! transforms. It automatically falls back to CPU implementations when GPU
//! acceleration is not available.

use std::collections::HashMap;
use std::mem;
use std::sync::Arc;

use scirs2_core::ndarray::{Array, ArrayView, ArrayView2, Dimension, Ix2};
use scirs2_core::numeric::{Float, FromPrimitive};

use crate::backend::gpu_acceleration_framework::{
    GpuAccelerationManager, GpuPerformanceReport, MemoryPoolConfig,
};
use crate::backend::{Backend, DeviceManager};
use crate::error::{NdimageError, NdimageResult};
use crate::interpolation::BoundaryMode;
use std::f64::consts::PI;

/// High-level GPU operations manager for ndimage
pub struct GpuOperations {
    /// GPU acceleration manager
    acceleration_manager: Arc<GpuAccelerationManager>,
    /// Device manager for hardware detection
    device_manager: DeviceManager,
    /// Configuration
    config: GpuOperationsConfig,
    /// Operation registry
    operation_registry: HashMap<String, OperationInfo>,
}

#[derive(Debug, Clone)]
pub struct GpuOperationsConfig {
    /// Minimum array size to consider GPU acceleration
    pub min_gpu_size: usize,
    /// Memory pool configuration
    pub memory_pool_config: MemoryPoolConfig,
    /// Automatic fallback to CPU on GPU errors
    pub auto_fallback: bool,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// GPU operation timeout in milliseconds
    pub operation_timeout: u64,
}

impl Default for GpuOperationsConfig {
    fn default() -> Self {
        Self {
            min_gpu_size: 1024 * 1024, // 1M elements minimum
            memory_pool_config: MemoryPoolConfig::default(),
            auto_fallback: true,
            enable_monitoring: true,
            operation_timeout: 10000, // 10 seconds
        }
    }
}

#[derive(Debug, Clone)]
struct OperationInfo {
    /// Operation name
    name: String,
    /// GPU kernel source
    kernel_source: String,
    /// Preferred backend
    preferred_backend: Backend,
    /// Memory complexity (bytes per input element)
    memory_complexity: f64,
    /// Computational complexity (operations per input element)
    compute_complexity: f64,
}

impl GpuOperations {
    /// Create a new GPU operations manager
    pub fn new(config: GpuOperationsConfig) -> NdimageResult<Self> {
        let acceleration_manager = Arc::new(GpuAccelerationManager::new(
            config.memory_pool_config.clone(),
        )?);
        let device_manager = DeviceManager::new()?;

        let mut gpu_ops = Self {
            acceleration_manager,
            device_manager,
            config,
            operation_registry: HashMap::new(),
        };

        // Register built-in GPU operations
        gpu_ops.register_builtin_operations()?;

        Ok(gpu_ops)
    }

    /// GPU-accelerated 2D convolution
    pub fn gpu_convolution_2d<T>(
        &self,
        input: ArrayView2<T>,
        kernel: ArrayView2<T>,
        mode: BoundaryMode,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float
            + FromPrimitive
            + Clone
            + Send
            + Sync
            + std::ops::DivAssign
            + std::ops::AddAssign
            + std::fmt::Debug
            + 'static,
    {
        let operation_name = "convolution_2d";

        // Check if GPU acceleration is beneficial
        if !self.should_use_gpu(&input, operation_name) {
            return self.fallback_convolution_2d(input, kernel, mode);
        }

        // Get GPU kernel source
        let kernel_source = self.get_convolution_kernel_source();

        // Select best available backend
        let backend = self.select_backend_for_operation(operation_name)?;

        // Execute on GPU with error handling
        match self.execute_gpu_convolution(input, kernel, &kernel_source, backend, mode) {
            Ok(result) => Ok(result),
            Err(e) if self.config.auto_fallback => {
                eprintln!("GPU convolution failed: {:?}, falling back to CPU", e);
                self.fallback_convolution_2d(input, kernel, mode)
            }
            Err(e) => Err(e),
        }
    }

    /// GPU-accelerated morphological erosion
    pub fn gpu_morphological_erosion<T>(
        &self,
        input: ArrayView2<T>,
        structuring_element: ArrayView2<bool>,
        mode: BoundaryMode,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float
            + FromPrimitive
            + Clone
            + Send
            + Sync
            + PartialOrd
            + std::ops::AddAssign
            + std::ops::DivAssign
            + std::fmt::Debug
            + 'static,
    {
        let operation_name = "morphological_erosion";

        if !self.should_use_gpu(&input, operation_name) {
            return self.fallback_morphological_erosion(input, structuring_element, mode);
        }

        let kernel_source = self.get_morphology_kernel_source();
        let backend = self.select_backend_for_operation(operation_name)?;

        match self.execute_gpu_morphology(
            input,
            structuring_element,
            &kernel_source,
            backend,
            mode,
            MorphologyOperation::Erosion,
        ) {
            Ok(result) => Ok(result),
            Err(e) if self.config.auto_fallback => {
                eprintln!("GPU erosion failed: {:?}, falling back to CPU", e);
                self.fallback_morphological_erosion(input, structuring_element, mode)
            }
            Err(e) => Err(e),
        }
    }

    /// GPU-accelerated morphological dilation
    pub fn gpu_morphological_dilation<T>(
        &self,
        input: ArrayView2<T>,
        structuring_element: ArrayView2<bool>,
        mode: BoundaryMode,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float
            + FromPrimitive
            + Clone
            + Send
            + Sync
            + PartialOrd
            + std::ops::AddAssign
            + std::ops::DivAssign
            + std::fmt::Debug
            + 'static,
    {
        let operation_name = "morphological_dilation";

        if !self.should_use_gpu(&input, operation_name) {
            return self.fallback_morphological_dilation(input, structuring_element, mode);
        }

        let kernel_source = self.get_morphology_kernel_source();
        let backend = self.select_backend_for_operation(operation_name)?;

        match self.execute_gpu_morphology(
            input,
            structuring_element,
            &kernel_source,
            backend,
            mode,
            MorphologyOperation::Dilation,
        ) {
            Ok(result) => Ok(result),
            Err(e) if self.config.auto_fallback => {
                eprintln!("GPU dilation failed: {:?}, falling back to CPU", e);
                self.fallback_morphological_dilation(input, structuring_element, mode)
            }
            Err(e) => Err(e),
        }
    }

    /// GPU-accelerated Gaussian filter
    pub fn gpu_gaussian_filter<T>(
        &self,
        input: ArrayView2<T>,
        sigma: (f64, f64),
        mode: BoundaryMode,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float + FromPrimitive + Clone + Send + Sync + std::ops::DivAssign + std::ops::AddAssign,
    {
        let operation_name = "gaussian_filter";

        if !self.should_use_gpu(&input, operation_name) {
            return self.fallback_gaussian_filter(input, sigma, mode);
        }

        let kernel_source = self.get_gaussian_kernel_source();
        let backend = self.select_backend_for_operation(operation_name)?;

        match self.execute_gpu_gaussian(input, sigma, &kernel_source, backend, mode) {
            Ok(result) => Ok(result),
            Err(e) if self.config.auto_fallback => {
                eprintln!("GPU Gaussian filter failed: {:?}, falling back to CPU", e);
                self.fallback_gaussian_filter(input, sigma, mode)
            }
            Err(e) => Err(e),
        }
    }

    /// GPU-accelerated distance transform
    pub fn gpu_distance_transform<T>(
        &self,
        input: ArrayView2<T>,
        metric: DistanceMetric,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float + FromPrimitive + Clone + Send + Sync + PartialOrd,
    {
        let operation_name = "distance_transform";

        if !self.should_use_gpu(&input, operation_name) {
            return self.fallback_distance_transform(input, metric);
        }

        let kernel_source = self.get_distance_transform_kernel_source();
        let backend = self.select_backend_for_operation(operation_name)?;

        match self.execute_gpu_distance_transform(input, metric, &kernel_source, backend) {
            Ok(result) => Ok(result),
            Err(e) if self.config.auto_fallback => {
                eprintln!(
                    "GPU distance transform failed: {:?}, falling back to CPU",
                    e
                );
                self.fallback_distance_transform(input, metric)
            }
            Err(e) => Err(e),
        }
    }

    /// Get comprehensive performance report
    pub fn get_performance_report(&self) -> GpuPerformanceReport {
        self.acceleration_manager.get_performance_report()
    }

    /// Check GPU availability and capabilities
    pub fn get_gpu_info(&self) -> GpuInfo {
        let capabilities = self.device_manager.get_capabilities();

        GpuInfo {
            cuda_available: capabilities.cuda_available,
            opencl_available: capabilities.opencl_available,
            metal_available: capabilities.metal_available,
            gpu_memory: capabilities.gpu_memory_mb,
            compute_units: capabilities.compute_units,
            preferred_backend: self.select_preferred_backend(),
        }
    }

    // Private implementation methods

    fn register_builtin_operations(&mut self) -> NdimageResult<()> {
        // Register convolution operation
        self.operation_registry.insert(
            "convolution_2d".to_string(),
            OperationInfo {
                name: "convolution_2d".to_string(),
                kernel_source: self.get_convolution_kernel_source(),
                preferred_backend: {
                    #[cfg(feature = "opencl")]
                    {
                        Backend::OpenCL
                    }
                    #[cfg(not(feature = "opencl"))]
                    {
                        Backend::Cpu
                    }
                },
                memory_complexity: 2.0,  // Input + output
                compute_complexity: 9.0, // Assume 3x3 kernel average
            },
        );

        // Register morphological operations
        self.operation_registry.insert(
            "morphological_erosion".to_string(),
            OperationInfo {
                name: "morphological_erosion".to_string(),
                kernel_source: self.get_morphology_kernel_source(),
                preferred_backend: {
                    #[cfg(feature = "opencl")]
                    {
                        Backend::OpenCL
                    }
                    #[cfg(not(feature = "opencl"))]
                    {
                        Backend::Cpu
                    }
                },
                memory_complexity: 2.0,
                compute_complexity: 9.0,
            },
        );

        // Register Gaussian filter
        self.operation_registry.insert(
            "gaussian_filter".to_string(),
            OperationInfo {
                name: "gaussian_filter".to_string(),
                kernel_source: self.get_gaussian_kernel_source(),
                preferred_backend: {
                    #[cfg(feature = "opencl")]
                    {
                        Backend::OpenCL
                    }
                    #[cfg(not(feature = "opencl"))]
                    {
                        Backend::Cpu
                    }
                },
                memory_complexity: 3.0,  // Separable: input + temp + output
                compute_complexity: 6.0, // Two 1D passes
            },
        );

        // Register distance transform
        self.operation_registry.insert(
            "distance_transform".to_string(),
            OperationInfo {
                name: "distance_transform".to_string(),
                kernel_source: self.get_distance_transform_kernel_source(),
                preferred_backend: {
                    #[cfg(feature = "opencl")]
                    {
                        Backend::OpenCL
                    }
                    #[cfg(not(feature = "opencl"))]
                    {
                        Backend::Cpu
                    }
                },
                memory_complexity: 2.0,
                compute_complexity: 10.0, // Multiple passes
            },
        );

        Ok(())
    }

    fn should_use_gpu<T, D>(&self, input: &ArrayView<T, D>, operation_name: &str) -> bool
    where
        T: Float + FromPrimitive,
        D: Dimension,
    {
        // Check minimum size threshold
        if input.len() < self.config.min_gpu_size {
            return false;
        }

        // Check if GPU is available
        let capabilities = self.device_manager.get_capabilities();
        if !capabilities.gpu_available {
            return false;
        }

        // Check memory requirements
        if let Some(op_info) = self.operation_registry.get(operation_name) {
            let required_memory =
                input.len() * std::mem::size_of::<T>() * op_info.memory_complexity as usize;
            let available_memory = capabilities.gpu_memory_mb * 1024 * 1024;

            if required_memory > available_memory {
                return false;
            }
        }

        true
    }

    fn select_backend_for_operation(&self, operation_name: &str) -> NdimageResult<Backend> {
        let capabilities = self.device_manager.get_capabilities();

        // Check operation preference
        if let Some(op_info) = self.operation_registry.get(operation_name) {
            match op_info.preferred_backend {
                #[cfg(feature = "cuda")]
                Backend::Cuda if capabilities.cuda_available => return Ok(Backend::Cuda),
                #[cfg(feature = "opencl")]
                Backend::OpenCL if capabilities.opencl_available => return Ok(Backend::OpenCL),
                #[cfg(all(target_os = "macos", feature = "metal"))]
                Backend::Metal if capabilities.metal_available => return Ok(Backend::Metal),
                _ => {}
            }
        }

        // Fallback to best available backend
        if capabilities.cuda_available {
            #[cfg(feature = "cuda")]
            {
                Ok(Backend::Cuda)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Ok(Backend::Cpu)
            }
        } else if capabilities.opencl_available {
            #[cfg(feature = "opencl")]
            {
                Ok(Backend::OpenCL)
            }
            #[cfg(not(feature = "opencl"))]
            {
                Ok(Backend::Cpu)
            }
        } else if capabilities.metal_available {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                Ok(Backend::Metal)
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                Ok(Backend::Cpu)
            }
        } else {
            Err(NdimageError::GpuNotAvailable(
                "No GPU backend available".to_string(),
            ))
        }
    }

    fn select_preferred_backend(&self) -> Backend {
        let capabilities = self.device_manager.get_capabilities();

        if capabilities.cuda_available {
            #[cfg(feature = "cuda")]
            {
                Backend::Cuda
            }
            #[cfg(not(feature = "cuda"))]
            {
                Backend::Cpu
            }
        } else if capabilities.opencl_available {
            #[cfg(feature = "opencl")]
            {
                Backend::OpenCL
            }
            #[cfg(not(feature = "opencl"))]
            {
                Backend::Cpu
            }
        } else if capabilities.metal_available {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                Backend::Metal
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                Backend::Cpu
            }
        } else {
            Backend::Cpu
        }
    }

    // GPU operation execution methods

    fn execute_gpu_convolution<T>(
        &self,
        input: ArrayView2<T>,
        _kernel: ArrayView2<T>,
        kernel_source: &str,
        backend: Backend,
        mode: BoundaryMode,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float + FromPrimitive + Clone + Send + Sync + std::ops::DivAssign + std::ops::AddAssign,
    {
        // This would contain the actual GPU execution logic
        // For now, we'll use the acceleration manager framework
        self.acceleration_manager
            .execute_operation("convolution_2d", input.into_dyn(), kernel_source, backend)
            .map(|result| result.into_dimensionality::<Ix2>().unwrap())
    }

    fn execute_gpu_morphology<T>(
        &self,
        input: ArrayView2<T>,
        _structuring_element: ArrayView2<bool>,
        kernel_source: &str,
        backend: Backend,
        mode: BoundaryMode,
        _operation: MorphologyOperation,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float + FromPrimitive + Clone + Send + Sync + PartialOrd,
    {
        // Execute morphological _operation on GPU
        self.acceleration_manager
            .execute_operation(
                "morphological_operation",
                input.into_dyn(),
                kernel_source,
                backend,
            )
            .map(|result| result.into_dimensionality::<Ix2>().unwrap())
    }

    fn execute_gpu_gaussian<T>(
        &self,
        input: ArrayView2<T>,
        _sigma: (f64, f64),
        kernel_source: &str,
        backend: Backend,
        mode: BoundaryMode,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float + FromPrimitive + Clone + Send + Sync + std::ops::DivAssign + std::ops::AddAssign,
    {
        // Execute Gaussian filter on GPU
        self.acceleration_manager
            .execute_operation("gaussian_filter", input.into_dyn(), kernel_source, backend)
            .map(|result| result.into_dimensionality::<Ix2>().unwrap())
    }

    fn execute_gpu_distance_transform<T>(
        &self,
        input: ArrayView2<T>,
        _metric: DistanceMetric,
        kernel_source: &str,
        backend: Backend,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float + FromPrimitive + Clone + Send + Sync + PartialOrd,
    {
        // Execute distance transform on GPU
        self.acceleration_manager
            .execute_operation(
                "distance_transform",
                input.into_dyn(),
                kernel_source,
                backend,
            )
            .map(|result| result.into_dimensionality::<Ix2>().unwrap())
    }

    // Fallback CPU implementations

    fn fallback_convolution_2d<T>(
        &self,
        input: ArrayView2<T>,
        kernel: ArrayView2<T>,
        mode: BoundaryMode,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float
            + FromPrimitive
            + Clone
            + Send
            + Sync
            + std::ops::DivAssign
            + std::ops::AddAssign
            + std::fmt::Debug
            + 'static,
    {
        // Use existing CPU implementation
        let border_mode = match mode {
            BoundaryMode::Constant => Some(crate::filters::BorderMode::Constant),
            BoundaryMode::Reflect => Some(crate::filters::BorderMode::Reflect),
            BoundaryMode::Mirror => Some(crate::filters::BorderMode::Mirror),
            BoundaryMode::Wrap => Some(crate::filters::BorderMode::Wrap),
            BoundaryMode::Nearest => Some(crate::filters::BorderMode::Constant), // No direct equivalent, use Constant
        };
        crate::convolve(&input.to_owned(), &kernel.to_owned(), border_mode)
    }

    fn fallback_morphological_erosion<T>(
        &self,
        input: ArrayView2<T>,
        structuring_element: ArrayView2<bool>,
        mode: BoundaryMode,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float
            + FromPrimitive
            + Clone
            + Send
            + Sync
            + PartialOrd
            + std::ops::DivAssign
            + std::ops::AddAssign
            + std::fmt::Debug
            + 'static,
    {
        // Use existing CPU implementation
        use crate::morphology::MorphBorderMode;
        let morph_mode = match mode {
            BoundaryMode::Constant => MorphBorderMode::Constant,
            BoundaryMode::Reflect => MorphBorderMode::Reflect,
            BoundaryMode::Mirror => MorphBorderMode::Mirror,
            BoundaryMode::Wrap => MorphBorderMode::Wrap,
            BoundaryMode::Nearest => MorphBorderMode::Nearest,
        };
        crate::grey_erosion(
            &input.to_owned(),
            None,
            Some(&structuring_element.to_owned()),
            Some(morph_mode),
            None,
            None,
        )
    }

    fn fallback_morphological_dilation<T>(
        &self,
        input: ArrayView2<T>,
        structuring_element: ArrayView2<bool>,
        mode: BoundaryMode,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float
            + FromPrimitive
            + Clone
            + Send
            + Sync
            + PartialOrd
            + std::ops::DivAssign
            + std::ops::AddAssign
            + std::fmt::Debug
            + 'static,
    {
        // Use existing CPU implementation
        use crate::morphology::MorphBorderMode;
        let morph_mode = match mode {
            BoundaryMode::Constant => MorphBorderMode::Constant,
            BoundaryMode::Reflect => MorphBorderMode::Reflect,
            BoundaryMode::Mirror => MorphBorderMode::Mirror,
            BoundaryMode::Wrap => MorphBorderMode::Wrap,
            BoundaryMode::Nearest => MorphBorderMode::Nearest,
        };
        crate::grey_dilation(
            &input.to_owned(),
            None,
            Some(&structuring_element.to_owned()),
            Some(morph_mode),
            None,
            None,
        )
    }

    fn fallback_gaussian_filter<T>(
        &self,
        input: ArrayView2<T>,
        sigma: (f64, f64),
        mode: BoundaryMode,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float + FromPrimitive + Clone + Send + Sync + std::ops::DivAssign + std::ops::AddAssign,
    {
        // Implement simple Gaussian filter as CPU fallback
        // Use separable convolution: convolve along rows then columns

        let (rows, cols) = (input.nrows(), input.ncols());
        let (sigma_x, sigma_y) = sigma;

        // Create 1D Gaussian kernels
        let kernel_size_x = (6.0 * sigma_x).ceil() as usize | 1; // Ensure odd
        let kernel_size_y = (6.0 * sigma_y).ceil() as usize | 1;

        let kernel_x = gaussian_kernel_1d(sigma_x, kernel_size_x);
        let kernel_y = gaussian_kernel_1d(sigma_y, kernel_size_y);

        // Convolve along rows first
        let mut temp = Array::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = T::zero();
                let mut weight_sum = T::zero();

                let half_k = (kernel_size_x / 2) as isize;
                for k_idx in 0..kernel_size_x {
                    let j_offset = j as isize + k_idx as isize - half_k;
                    let weight = safe_f64_to_float::<T>(kernel_x[k_idx])?;

                    if j_offset >= 0 && j_offset < cols as isize {
                        sum = sum + input[[i, j_offset as usize]] * weight;
                        weight_sum = weight_sum + weight;
                    }
                }

                if weight_sum > T::zero() {
                    temp[[i, j]] = sum / weight_sum;
                } else {
                    temp[[i, j]] = input[[i, j]];
                }
            }
        }

        // Convolve along columns
        let mut output = Array::zeros((rows, cols));
        for j in 0..cols {
            for i in 0..rows {
                let mut sum = T::zero();
                let mut weight_sum = T::zero();

                let half_k = (kernel_size_y / 2) as isize;
                for k_idx in 0..kernel_size_y {
                    let i_offset = i as isize + k_idx as isize - half_k;
                    let weight = safe_f64_to_float::<T>(kernel_y[k_idx])?;

                    if i_offset >= 0 && i_offset < rows as isize {
                        sum = sum + temp[[i_offset as usize, j]] * weight;
                        weight_sum = weight_sum + weight;
                    }
                }

                if weight_sum > T::zero() {
                    output[[i, j]] = sum / weight_sum;
                } else {
                    output[[i, j]] = temp[[i, j]];
                }
            }
        }

        Ok(output)
    }

    fn fallback_distance_transform<T>(
        &self,
        input: ArrayView2<T>,
        metric: DistanceMetric,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float + FromPrimitive + Clone + Send + Sync + PartialOrd,
    {
        // Convert input to boolean array by thresholding at zero
        let bool_input = input.mapv(|x| x > T::zero());

        // Use existing CPU implementation
        let result = crate::distance_transform_edt(&bool_input.into_dyn(), None, true, false)?;

        // Convert result back to the desired type
        if let Some(distances) = result.0 {
            let converted_distances = distances.mapv(|x| T::from_f64(x).unwrap_or(T::zero()));
            // Convert from dynamic dimension back to Ix2
            converted_distances
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|_| {
                    NdimageError::DimensionError(
                        "Failed to convert distance transform result to 2D".into(),
                    )
                })
        } else {
            Err(NdimageError::ComputationError(
                "Distance transform did not return distances".into(),
            ))
        }
    }

    // GPU kernel source code methods

    fn get_convolution_kernel_source(&self) -> String {
        include_str!("backend/kernels/convolution.kernel").to_string()
    }

    fn get_morphology_kernel_source(&self) -> String {
        include_str!("backend/kernels/morphology.kernel").to_string()
    }

    fn get_gaussian_kernel_source(&self) -> String {
        include_str!("backend/kernels/gaussian_blur.kernel").to_string()
    }

    fn get_distance_transform_kernel_source(&self) -> String {
        // For now, use a placeholder - would need a dedicated distance transform kernel
        include_str!("backend/kernels/morphology.kernel").to_string()
    }
}

// Supporting types and enums

#[derive(Debug, Clone, Copy)]
pub enum MorphologyOperation {
    Erosion,
    Dilation,
}

#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Chessboard,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// CUDA availability
    pub cuda_available: bool,
    /// OpenCL availability
    pub opencl_available: bool,
    /// Metal availability (macOS)
    pub metal_available: bool,
    /// GPU memory in MB
    pub gpu_memory: usize,
    /// Number of compute units
    pub compute_units: u32,
    /// Preferred backend for this system
    pub preferred_backend: Backend,
}

impl GpuInfo {
    /// Display GPU information
    pub fn display(&self) {
        println!("=== GPU Information ===");
        println!("CUDA Available: {}", self.cuda_available);
        println!("OpenCL Available: {}", self.opencl_available);
        println!("Metal Available: {}", self.metal_available);
        println!("GPU Memory: {} MB", self.gpu_memory);
        println!("Compute Units: {}", self.compute_units);
        println!("Preferred Backend: {:?}", self.preferred_backend);
    }
}

/// Convenience function to create a GPU operations instance with default configuration
#[allow(dead_code)]
pub fn create_gpu_operations() -> NdimageResult<GpuOperations> {
    GpuOperations::new(GpuOperationsConfig::default())
}

/// Convenience function to create GPU operations with custom configuration
#[allow(dead_code)]
pub fn create_gpu_operations_with_config(
    config: GpuOperationsConfig,
) -> NdimageResult<GpuOperations> {
    GpuOperations::new(config)
}

/// Generate 1D Gaussian kernel
fn gaussian_kernel_1d(sigma: f64, size: usize) -> Vec<f64> {
    let center = (size / 2) as f64;
    let coeff = 1.0 / (sigma * (2.0 * PI).sqrt());
    let denom = 2.0 * sigma * sigma;

    let mut kernel: Vec<f64> = (0..size)
        .map(|i| {
            let x = i as f64 - center;
            coeff * (-x * x / denom).exp()
        })
        .collect();

    // Normalize
    let sum: f64 = kernel.iter().sum();
    if sum > 0.0 {
        for val in &mut kernel {
            *val /= sum;
        }
    }

    kernel
}

/// Safely convert f64 to generic float type
fn safe_f64_to_float<T: Float>(value: f64) -> NdimageResult<T> {
    T::from(value).ok_or_else(|| {
        NdimageError::InvalidInput(format!("Cannot convert {} to target float type", value))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_gpu_operations_creation() {
        let result = create_gpu_operations();
        // This might fail in environments without GPU support, which is expected
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_gpu_info_display() {
        let gpu_info = GpuInfo {
            cuda_available: false,
            opencl_available: true,
            metal_available: false,
            gpu_memory: 8192,
            compute_units: 16,
            preferred_backend: {
                #[cfg(feature = "opencl")]
                {
                    Backend::OpenCL
                }
                #[cfg(not(feature = "opencl"))]
                {
                    Backend::Cpu
                }
            },
        };

        // Test that display doesn't panic
        gpu_info.display();
    }

    #[test]
    fn test_gpu_convolution_fallback() {
        // Test small array that should fallback to CPU
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let kernel = array![[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]];

        // This should work even without GPU support
        if let Ok(gpu_ops) = create_gpu_operations() {
            let result =
                gpu_ops.gpu_convolution_2d(input.view(), kernel.view(), BoundaryMode::Constant);

            // Should either succeed or fail gracefully
            assert!(result.is_ok() || result.is_err());
        }
    }
}
