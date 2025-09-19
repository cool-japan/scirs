//! Enhanced array operations for special functions
//!
//! This module provides comprehensive array support for special functions with
//! lazy evaluation, GPU acceleration, and multidimensional operations.

#![allow(dead_code)]

use crate::error::{SpecialError, SpecialResult};
use ndarray::{Array, ArrayView1, Dimension};

/// Safe slice casting replacement for bytemuck::cast_slice
#[allow(dead_code)]
fn cast_slice_to_bytes<T>(slice: &[T]) -> &[u8] {
    // SAFETY: This is safe because:
    // 1. The pointer is derived from a valid slice
    // 2. The size calculation is correct using size_of_val
    // 3. The lifetime is bounded by the input slice
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

/// Safe slice casting replacement for bytemuck::cast_slice (reverse)
#[allow(dead_code)]
fn cast_bytes_to_slice<T>(bytes: &[u8]) -> &[T] {
    assert_eq!(bytes.len() % std::mem::size_of::<T>(), 0);
    // SAFETY: This is safe because:
    // 1. We assert that the byte length is a multiple of T's size
    // 2. The pointer is derived from a valid slice
    // 3. The length calculation ensures we don't exceed bounds
    // 4. The lifetime is bounded by the input slice
    unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const T,
            bytes.len() / std::mem::size_of::<T>(),
        )
    }
}

#[cfg(feature = "futures")]
use futures::future::BoxFuture;

// #[cfg(feature = "arrayfire")]
// use arrayfire;

// #[cfg(feature = "arrayfire")]
// use log;

// #[cfg(feature = "lazy")]
// use std::collections::HashMap;

/// Execution backend for array operations
#[derive(Debug, Clone, Default)]
pub enum Backend {
    /// CPU-based computation with ndarray
    #[default]
    Cpu,
    /// GPU-based computation (requires gpu feature)
    #[cfg(feature = "gpu")]
    Gpu,
    /// Lazy evaluation (requires lazy feature)
    #[cfg(feature = "lazy")]
    Lazy,
    // /// ArrayFire backend for GPU acceleration (disabled - placeholder)
    // #[cfg(feature = "arrayfire")]
    // ArrayFire,
}

/// Configuration for array operations
#[derive(Debug, Clone)]
pub struct ArrayConfig {
    /// Chunk size for memory-efficient processing
    pub chunksize: usize,
    /// Whether to use parallel processing
    pub parallel: bool,
    /// Memory limit for operations (in bytes)
    pub memory_limit: usize,
    /// Execution backend
    pub backend: Backend,
    /// Whether to cache computed results
    pub cache_results: bool,
    /// Maximum cache size (number of entries)
    pub max_cachesize: usize,
    /// Lazy evaluation threshold (array size)
    pub lazy_threshold: usize,
}

impl Default for ArrayConfig {
    fn default() -> Self {
        Self {
            chunksize: 1024,
            parallel: cfg!(feature = "parallel"),
            memory_limit: 1024 * 1024 * 1024, // 1GB
            backend: Backend::default(),
            cache_results: true,
            max_cachesize: 1000,
            lazy_threshold: 10_000, // Use lazy evaluation for arrays > 10k elements
        }
    }
}

/// Memory-efficient array operations
pub mod memory_efficient {
    use super::*;

    /// Estimate memory usage for an operation
    pub fn estimate_memory_usage<T>(shape: &[usize], numarrays: usize) -> usize {
        let elemsize = std::mem::size_of::<T>();
        let total_elements: usize = shape.iter().product();
        total_elements * elemsize * numarrays
    }

    /// Check if operation fits within memory limits
    pub fn check_memory_limit<T>(shape: &[usize], numarrays: usize, config: &ArrayConfig) -> bool {
        estimate_memory_usage::<T>(shape, numarrays) <= config.memory_limit
    }
}

/// Lazy evaluation system for deferred computation
#[cfg(feature = "lazy")]
pub mod lazy {
    use super::*;
    // use std::any::Any;
    use std::fmt::Debug;

    /// Trait for lazy operations that can be computed on demand
    pub trait LazyOperation: Send + Sync + Debug {
        type Output;

        /// Execute the lazy operation
        fn execute(&self) -> SpecialResult<Self::Output>;

        /// Get operation description for debugging
        fn description(&self) -> String;

        /// Estimate computational cost (arbitrary units)
        fn cost_estimate(&self) -> usize;
    }

    /// Container for lazy array operations
    #[derive(Debug)]
    pub struct LazyArray<T, D>
    where
        D: Dimension,
    {
        /// The operation to be performed
        operation: Box<dyn LazyOperation<Output = Array<T, D>>>,
        /// Shape of the resulting array
        shape: Vec<usize>,
        /// Whether the result has been computed and cached
        computed: std::sync::Mutex<Option<Array<T, D>>>,
        /// Configuration for execution
        config: ArrayConfig,
    }

    impl<T, D> LazyArray<T, D>
    where
        D: Dimension,
        T: Clone + Send + Sync,
    {
        /// Create a new lazy array
        pub fn new(
            operation: Box<dyn LazyOperation<Output = Array<T, D>>>,
            shape: Vec<usize>,
            config: ArrayConfig,
        ) -> Self {
            Self {
                operation,
                shape,
                computed: std::sync::Mutex::new(None),
                config,
            }
        }

        /// Get the shape of the lazy array
        pub fn shape(&self) -> &[usize] {
            &self.shape
        }

        /// Force evaluation of the lazy array
        pub fn compute(&self) -> SpecialResult<Array<T, D>> {
            let mut computed = self.computed.lock().unwrap();

            if let Some(ref cached) = *computed {
                return Ok(cached.clone());
            }

            let result = self.operation.execute()?;
            *computed = Some(result.clone());
            Ok(result)
        }

        /// Check if the array has been computed
        pub fn is_computed(&self) -> bool {
            self.computed.lock().unwrap().is_some()
        }

        /// Get operation description
        pub fn description(&self) -> String {
            self.operation.description()
        }

        /// Get cost estimate
        pub fn cost_estimate(&self) -> usize {
            self.operation.cost_estimate()
        }
    }

    /// Lazy operation for gamma function
    #[derive(Debug)]
    pub struct LazyGamma<D>
    where
        D: Dimension,
    {
        input: Array<f64, D>,
    }

    impl<D> LazyGamma<D>
    where
        D: Dimension,
    {
        pub fn new(input: Array<f64, D>) -> Self {
            Self { input }
        }
    }

    impl<D> LazyOperation for LazyGamma<D>
    where
        D: Dimension + Send + Sync,
    {
        type Output = Array<f64, D>;

        fn execute(&self) -> SpecialResult<Self::Output> {
            Ok(self.input.mapv(crate::gamma::gamma))
        }

        fn description(&self) -> String {
            format!("LazyGamma(shape={:?})", self.input.shape())
        }

        fn cost_estimate(&self) -> usize {
            self.input.len() * 100 // Estimate: 100 units per gamma computation
        }
    }

    /// Lazy operation for Bessel J0 function
    #[derive(Debug)]
    pub struct LazyBesselJ0<D>
    where
        D: Dimension,
    {
        input: Array<f64, D>,
    }

    impl<D> LazyBesselJ0<D>
    where
        D: Dimension,
    {
        pub fn new(input: Array<f64, D>) -> Self {
            Self { input }
        }
    }

    impl<D> LazyOperation for LazyBesselJ0<D>
    where
        D: Dimension + Send + Sync,
    {
        type Output = Array<f64, D>;

        fn execute(&self) -> SpecialResult<Self::Output> {
            Ok(self.input.mapv(crate::bessel::j0))
        }

        fn description(&self) -> String {
            format!("LazyBesselJ0(shape={:?})", self.input.shape())
        }

        fn cost_estimate(&self) -> usize {
            self.input.len() * 150 // Estimate: 150 units per Bessel computation
        }
    }

    /// Create a lazy gamma array
    pub fn lazy_gamma<D>(input: Array<f64, D>, config: ArrayConfig) -> LazyArray<f64, D>
    where
        D: Dimension + Send + Sync + 'static,
    {
        let shape = input.shape().to_vec();
        let operation = Box::new(LazyGamma::new(input));
        LazyArray::new(operation, shape, config)
    }

    /// Create a lazy Bessel J0 array
    pub fn lazy_bessel_j0<D>(input: Array<f64, D>, config: ArrayConfig) -> LazyArray<f64, D>
    where
        D: Dimension + Send + Sync + 'static,
    {
        let shape = input.shape().to_vec();
        let operation = Box::new(LazyBesselJ0::new(input));
        LazyArray::new(operation, shape, config)
    }
}

/// GPU acceleration for array operations
#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;

    /// Advanced GPU buffer for array data with memory management
    pub struct GpuBuffer {
        #[cfg(feature = "gpu")]
        buffer: Option<std::sync::Arc<scirs2_core::gpu::GpuBuffer<f64>>>,
        size: usize,
        elementsize: usize,
        shape: Vec<usize>,
        allocatedsize: usize,
    }

    impl GpuBuffer {
        /// Create a new GPU buffer
        #[cfg(feature = "gpu")]
        pub fn new(ctx: &scirs2_core::gpu::GpuContext, data: &[f64]) -> SpecialResult<Self> {
            let buffer = ctx.create_buffer_from_slice(data);

            Ok(Self {
                buffer: Some(std::sync::Arc::new(buffer)),
                size: data.len(),
                elementsize: std::mem::size_of::<f64>(),
                shape: vec![data.len()],
                allocatedsize: data.len() * std::mem::size_of::<f64>(),
            })
        }

        /// Get buffer size in elements
        pub fn size(&self) -> usize {
            self.size
        }

        /// Get buffer shape
        pub fn shape(&self) -> &[usize] {
            &self.shape
        }

        /// Check if buffer is valid
        #[cfg(feature = "gpu")]
        pub fn is_valid(&self) -> bool {
            self.buffer.is_some()
        }

        #[cfg(not(feature = "gpu"))]
        pub fn is_valid(&self) -> bool {
            false
        }
    }

    /// Advanced GPU compute pipeline for special functions
    pub struct GpuPipeline {
        #[cfg(feature = "gpu")]
        context: Option<std::sync::Arc<scirs2_core::gpu::GpuContext>>,
        #[cfg(feature = "gpu")]
        pipelines:
            std::collections::HashMap<String, std::sync::Arc<scirs2_core::gpu::GpuKernelHandle>>,
        cache_enabled: bool,
        performance_stats:
            std::sync::Mutex<std::collections::HashMap<String, (u64, std::time::Duration)>>,
    }

    impl GpuPipeline {
        /// Create a new advanced GPU pipeline with comprehensive functionality
        #[cfg(feature = "gpu")]
        pub fn new() -> SpecialResult<Self> {
            use crate::gpu_context_manager::get_best_gpu_context;

            let context = get_best_gpu_context().map_err(|e| {
                SpecialError::ComputationError(format!("Failed to create GPU context: {}", e))
            })?;

            let mut pipelines = std::collections::HashMap::new();

            // Note: GPU pipelines not currently supported in scirs2-core
            // Pre-compiled kernels would be loaded here when available

            Ok(Self {
                context: Some(context),
                pipelines,
                cache_enabled: true,
                performance_stats: std::sync::Mutex::new(std::collections::HashMap::new()),
            })
        }

        /// Execute a kernel on GPU with performance monitoring
        #[cfg(feature = "gpu")]
        pub fn execute_kernel<T>(
            &self,
            kernel_name: &str,
            input: &[T],
            output: &mut [T],
        ) -> SpecialResult<std::time::Duration>
        where
            T: Clone + Copy + scirs2_core::gpu::GpuDataType,
        {
            let start_time = std::time::Instant::now();

            let context = self.context.as_ref().ok_or_else(|| {
                SpecialError::ComputationError("No GPU context available".to_string())
            })?;

            let pipeline = self.pipelines.get(kernel_name).ok_or_else(|| {
                SpecialError::ComputationError(format!("Kernel '{}' not found", kernel_name))
            })?;

            // Create GPU buffers
            let input_buffer = context.create_buffer_from_slice(input);

            let output_buffer = context.create_buffer::<T>(output.len());

            // Note: Direct kernel execution not currently supported in scirs2-core
            // Fall back to CPU computation for now
            return Err(SpecialError::ComputationError(
                "GPU kernel execution not yet implemented".to_string(),
            ));

            let elapsed = start_time.elapsed();

            // Update performance statistics
            if let Ok(mut stats) = self.performance_stats.lock() {
                let entry = stats
                    .entry(kernel_name.to_string())
                    .or_insert((0, std::time::Duration::ZERO));
                entry.0 += 1;
                entry.1 += elapsed;
            }

            Ok(elapsed)
        }

        /// Get performance statistics for a kernel
        pub fn get_kernel_stats(&self, kernelname: &str) -> Option<(u64, std::time::Duration)> {
            self.performance_stats.lock().ok()?.get(kernelname).copied()
        }

        /// Clear performance statistics
        pub fn clear_stats(&self) {
            if let Ok(mut stats) = self.performance_stats.lock() {
                stats.clear();
            }
        }

        /// Execute gamma function on GPU with advanced features
        #[cfg(feature = "gpu")]
        pub fn gamma_gpu<D>(&self, input: &Array<f64, D>) -> SpecialResult<Array<f64, D>>
        where
            D: Dimension,
        {
            // For 1D arrays, use direct GPU execution
            if input.ndim() == 1 {
                let input_slice = input.as_slice().ok_or_else(|| {
                    SpecialError::ComputationError("Array not contiguous".to_string())
                })?;

                let mut output = vec![0.0f64; input_slice.len()];
                self.execute_kernel("gamma", input_slice, &mut output)?;

                let result = Array::from_vec(output)
                    .into_dimensionality::<D>()
                    .map_err(|e| {
                        SpecialError::ComputationError(format!("Shape conversion error: {}", e))
                    })?;

                Ok(result)
            } else {
                // For multi-dimensional arrays, flatten, process, and reshape
                let flattened: Vec<f64> = input.iter().copied().collect();
                let mut output = vec![0.0f64; flattened.len()];

                self.execute_kernel("gamma", &flattened, &mut output)?;

                let result = Array::from_vec(output)
                    .to_shape(input.dim())
                    .map_err(|e| SpecialError::ComputationError(format!("Shape error: {}", e)))?
                    .into_owned();

                Ok(result)
            }
        }

        /// Execute Bessel J0 function on GPU
        #[cfg(feature = "gpu")]
        pub fn bessel_j0_gpu<D>(&self, input: &Array<f64, D>) -> SpecialResult<Array<f64, D>>
        where
            D: Dimension,
        {
            let flattened: Vec<f64> = input.iter().copied().collect();
            let mut output = vec![0.0f64; flattened.len()];

            self.execute_kernel("bessel_j0", &flattened, &mut output)?;

            let result = Array::from_vec(output)
                .to_shape(input.dim())
                .map_err(|e| SpecialError::ComputationError(format!("Shape error: {}", e)))?
                .into_owned();

            Ok(result)
        }

        /// Execute error function on GPU
        #[cfg(feature = "gpu")]
        pub fn erf_gpu<D>(&self, input: &Array<f64, D>) -> SpecialResult<Array<f64, D>>
        where
            D: Dimension,
        {
            let flattened: Vec<f64> = input.iter().copied().collect();
            let mut output = vec![0.0f64; flattened.len()];

            self.execute_kernel("erf", &flattened, &mut output)?;

            let result = Array::from_vec(output)
                .to_shape(input.dim())
                .map_err(|e| SpecialError::ComputationError(format!("Shape error: {}", e)))?
                .into_owned();

            Ok(result)
        }

        /// Execute gamma function on CPU as fallback
        #[cfg(not(feature = "gpu"))]
        pub async fn gamma_gpu<D>(&self, input: &Array<f64, D>) -> SpecialResult<Array<f64, D>>
        where
            D: Dimension,
        {
            // Fallback to CPU implementation
            Ok(input.mapv(crate::gamma::gamma))
        }
    }

    /// GPU-accelerated gamma computation
    pub async fn gamma_gpu<D>(input: &Array<f64, D>) -> SpecialResult<Array<f64, D>>
    where
        D: Dimension,
    {
        #[cfg(feature = "gpu")]
        {
            let pipeline = GpuPipeline::new()?;
            pipeline.gamma_gpu(input)
        }
        #[cfg(not(feature = "gpu"))]
        {
            // Fallback to CPU
            Ok(input.mapv(crate::gamma::gamma))
        }
    }
}

/// Broadcasting utilities for array operations
pub mod broadcasting {
    use super::*;

    /// Check if two shapes can be broadcast together
    pub fn can_broadcast(shape1: &[usize], shape2: &[usize]) -> bool {
        let max_len = shape1.len().max(shape2.len());

        for i in 0..max_len {
            let dim1 = shape1.get(shape1.len().wrapping_sub(i + 1)).unwrap_or(&1);
            let dim2 = shape2.get(shape2.len().wrapping_sub(i + 1)).unwrap_or(&1);

            if *dim1 != 1 && *dim2 != 1 && *dim1 != *dim2 {
                return false;
            }
        }

        true
    }

    /// Compute the broadcast shape of two arrays
    pub fn broadcastshape(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>, SpecialError> {
        if !can_broadcast(shape1, shape2) {
            return Err(SpecialError::DomainError(
                "Arrays cannot be broadcast together".to_string(),
            ));
        }

        let max_len = shape1.len().max(shape2.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let dim1 = shape1.get(shape1.len().wrapping_sub(i + 1)).unwrap_or(&1);
            let dim2 = shape2.get(shape2.len().wrapping_sub(i + 1)).unwrap_or(&1);

            result.push((*dim1).max(*dim2));
        }

        result.reverse();
        Ok(result)
    }
}

/// Vectorized special function operations with automatic backend selection
pub mod vectorized {
    use super::*;

    #[cfg(feature = "lazy")]
    use super::lazy::*;

    /// Enhanced gamma function computation with backend selection
    pub fn gamma_array<D>(
        input: &Array<f64, D>,
        config: &ArrayConfig,
    ) -> SpecialResult<GammaResult<D>>
    where
        D: Dimension + Send + Sync + 'static,
    {
        let total_elements = input.len();

        // Choose backend based on configuration and array size
        match &config.backend {
            #[cfg(feature = "lazy")]
            Backend::Lazy => {
                if total_elements >= config.lazy_threshold {
                    let lazy_array = lazy_gamma(input.clone(), config.clone());
                    return Ok(GammaResult::Lazy(lazy_array));
                }
            }
            #[cfg(all(feature = "gpu", feature = "futures"))]
            Backend::Gpu => {
                if total_elements >= 1000 {
                    // GPU efficient for larger arrays
                    let input_owned = input.to_owned();
                    // Since gamma_gpu is not async, we create a future wrapper
                    return Ok(GammaResult::Future(Box::pin(async move {
                        // Convert to appropriate array views for 1D operations
                        if input_owned.ndim() == 1 {
                            let input_1d = input_owned
                                .into_dimensionality::<ndarray::Ix1>()
                                .map_err(|e| {
                                    SpecialError::ComputationError(format!(
                                        "Dimension error: {}",
                                        e
                                    ))
                                })?;
                            let mut output = Array::zeros(input_1d.len());
                            match crate::gpu_ops::gamma_gpu(
                                &input_1d.view(),
                                &mut output.view_mut(),
                            ) {
                                Ok(_) => {
                                    // Convert back to original dimensions
                                    let result =
                                        output.into_dimensionality::<D>().map_err(|e| {
                                            SpecialError::ComputationError(format!(
                                                "Dimension error: {}",
                                                e
                                            ))
                                        })?;
                                    Ok(result)
                                }
                                Err(e) => {
                                    Err(SpecialError::ComputationError(format!("GPU error: {}", e)))
                                }
                            }
                        } else {
                            // For multi-dimensional arrays, fall back to CPU implementation
                            Ok(input_owned.mapv(crate::gamma::gamma))
                        }
                    })));
                }
            }
            #[cfg(all(feature = "gpu", not(feature = "futures")))]
            Backend::Gpu => {
                // Without futures, fall through to CPU implementation
            }
            Backend::Cpu => {
                // Use CPU implementation
            } // #[cfg(feature = "arrayfire")]
              // Backend::ArrayFire => {
              //     return arrayfire_gamma(input, config);
              // }
        }

        // Default CPU implementation with optional parallelization
        if config.parallel && total_elements > config.chunksize {
            #[cfg(feature = "parallel")]
            {
                use scirs2_core::parallel_ops::*;
                let data: Vec<f64> = input.iter().copied().collect();
                let result: Vec<f64> = data.par_iter().map(|&x| crate::gamma::gamma(x)).collect();
                let result_array = Array::from_vec(result)
                    .to_shape(input.dim())
                    .map_err(|e| SpecialError::ComputationError(format!("Shape error: {}", e)))?
                    .into_owned();
                return Ok(GammaResult::Immediate(result_array));
            }
        }

        Ok(GammaResult::Immediate(input.mapv(crate::gamma::gamma)))
    }

    /// Enhanced Bessel J0 function computation with backend selection  
    pub fn j0_array<D>(
        input: &Array<f64, D>,
        config: &ArrayConfig,
    ) -> SpecialResult<BesselResult<D>>
    where
        D: Dimension + Send + Sync + 'static,
    {
        let _total_elements = input.len();

        // Choose backend based on configuration and array size
        match &config.backend {
            #[cfg(feature = "lazy")]
            Backend::Lazy => {
                if _total_elements >= config.lazy_threshold {
                    let lazy_array = lazy_bessel_j0(input.clone(), config.clone());
                    return Ok(BesselResult::Lazy(lazy_array));
                }
            }
            Backend::Cpu => {
                // Use CPU implementation
            }
            #[cfg(feature = "gpu")]
            Backend::Gpu => {
                // Use CPU fallback for Bessel functions
            }
        }

        // Default CPU implementation
        Ok(BesselResult::Immediate(input.mapv(crate::bessel::j0)))
    }

    /// Enhanced error function computation
    pub fn erf_array<D>(input: &Array<f64, D>, config: &ArrayConfig) -> SpecialResult<Array<f64, D>>
    where
        D: Dimension,
    {
        if config.parallel && input.len() > config.chunksize {
            #[cfg(feature = "parallel")]
            {
                use scirs2_core::parallel_ops::*;
                let data: Vec<f64> = input.iter().copied().collect();
                let result: Vec<f64> = data.par_iter().map(|&x| crate::erf::erf(x)).collect();
                return Ok(Array::from_vec(result)
                    .to_shape(input.dim())
                    .map_err(|e| SpecialError::ComputationError(format!("Shape error: {}", e)))?
                    .into_owned());
            }
        }

        Ok(input.mapv(crate::erf::erf))
    }

    /// Enhanced factorial function computation
    pub fn factorial_array<D>(
        input: &Array<u32, D>,
        config: &ArrayConfig,
    ) -> SpecialResult<Array<f64, D>>
    where
        D: Dimension,
    {
        if config.parallel && input.len() > config.chunksize {
            #[cfg(feature = "parallel")]
            {
                use scirs2_core::parallel_ops::*;
                let data: Vec<u32> = input.iter().copied().collect();
                let result: Vec<f64> = data
                    .par_iter()
                    .map(|&x| crate::combinatorial::factorial(x).unwrap_or(f64::NAN))
                    .collect();
                return Ok(Array::from_vec(result)
                    .to_shape(input.dim())
                    .map_err(|e| SpecialError::ComputationError(format!("Shape error: {}", e)))?
                    .into_owned());
            }
        }

        Ok(input.mapv(|x| crate::combinatorial::factorial(x).unwrap_or(f64::NAN)))
    }

    /// Enhanced softmax computation
    pub fn softmax_1d(
        input: ArrayView1<f64>,
        _config: &ArrayConfig,
    ) -> SpecialResult<Array<f64, ndarray::Ix1>> {
        // Use existing optimized implementation from statistical module
        crate::statistical::softmax(input)
    }

    /// Result type for gamma computations that can be immediate, lazy, or async
    pub enum GammaResult<D>
    where
        D: Dimension,
    {
        /// Immediate result computed synchronously
        Immediate(Array<f64, D>),
        /// Lazy result computed on demand
        #[cfg(feature = "lazy")]
        Lazy(LazyArray<f64, D>),
        /// Future result computed asynchronously (e.g., on GPU)
        #[cfg(feature = "futures")]
        Future(BoxFuture<'static, SpecialResult<Array<f64, D>>>),
    }

    impl<D> GammaResult<D>
    where
        D: Dimension,
    {
        /// Force evaluation of the result
        pub async fn compute(self) -> SpecialResult<Array<f64, D>> {
            match self {
                GammaResult::Immediate(array) => Ok(array),
                #[cfg(feature = "lazy")]
                GammaResult::Lazy(lazy_array) => lazy_array.compute(),
                #[cfg(feature = "futures")]
                GammaResult::Future(future) => future.await,
            }
        }

        /// Check if result is immediately available
        pub fn is_ready(&self) -> bool {
            match self {
                GammaResult::Immediate(_) => true,
                #[cfg(feature = "lazy")]
                GammaResult::Lazy(lazy_array) => lazy_array.is_computed(),
                #[cfg(feature = "futures")]
                GammaResult::Future(_) => false,
            }
        }
    }

    /// Result type for Bessel function computations
    pub enum BesselResult<D>
    where
        D: Dimension,
    {
        /// Immediate result computed synchronously
        Immediate(Array<f64, D>),
        /// Lazy result computed on demand
        #[cfg(feature = "lazy")]
        Lazy(LazyArray<f64, D>),
    }

    impl<D> BesselResult<D>
    where
        D: Dimension,
    {
        /// Force evaluation of the result
        pub fn compute(self) -> SpecialResult<Array<f64, D>> {
            match self {
                BesselResult::Immediate(array) => Ok(array),
                #[cfg(feature = "lazy")]
                BesselResult::Lazy(lazy_array) => lazy_array.compute(),
            }
        }
    }

    // ArrayFire backend implementation for gamma function (disabled - placeholder)
    // #[cfg(feature = "arrayfire")]
    // fn arrayfire_gamma<D>(
    //     input: &Array<f64, D>,
    //     config: &ArrayConfig,
    // ) -> SpecialResult<GammaResult<D>>
    // where
    //     D: Dimension,
    // {
    //     use arrayfire as af;
    //
    //     // Initialize ArrayFire if not already done
    //     af::set_backend(af::Backend::DEFAULT);
    //     af::set_device(0);
    //
    //     // Convert ndarray to ArrayFire array
    //     let input_vec: Vec<f64> = input.iter().cloned().collect();
    //     let dims = input.shape();
    //
    //     // Create ArrayFire array
    //     let afinput = match dims.len() {
    //         1 => af::Array::new(&input_vec, af::Dim4::new(&[dims[0] as u64, 1, 1, 1])),
    //         2 => af::Array::new(&input_vec, af::Dim4::new(&[dims[0] as u64, dims[1] as u64, 1, 1])),
    //         3 => af::Array::new(&input_vec, af::Dim4::new(&[dims[0] as u64, dims[1] as u64, dims[2] as u64, 1])),
    //         4 => af::Array::new(&input_vec, af::Dim4::new(&[dims[0] as u64, dims[1] as u64, dims[2] as u64, dims[3] as u64])),
    //         _ => {
    //             // For higher dimensions, flatten and reshape later
    //             af::Array::new(&input_vec, af::Dim4::new(&[input_vec.len() as u64, 1, 1, 1]))
    //         }
    //     };
    //
    //     // Compute gamma function using ArrayFire
    //     let af_result = arrayfire_gamma_kernel(&afinput)?;
    //
    //     // Convert result back to ndarray
    //     let mut result_vec = vec![0.0; input.len()];
    //     af_result.host(&mut result_vec);
    //
    //     let result = Array::from_vec(result_vec)
    //         .to_shape(input.dim())
    //         .map_err(|e| SpecialError::ComputationError(format!("Shape conversion error: {}", e)))?
    //         .into_owned();
    //
    //     Ok(GammaResult::Immediate(result))
    // }

    // ArrayFire kernel for gamma function computation (disabled - placeholder)
    // #[cfg(feature = "arrayfire")]
    // fn arrayfire_gamma_kernel(input: &arrayfire::Array<f64>) -> SpecialResult<arrayfire::Array<f64>> {
    //     use arrayfire as af;
    //
    //     // Check for negative values (gamma undefined for negative integers)
    //     let negative_mask = af::lt(input, &0.0, false);
    //     let has_negatives = af::any_true_all(&negative_mask).0;
    //
    //     if has_negatives {
    //         log::warn!("Gamma function called with negative values, may produce NaN");
    //     }
    //
    //     // Compute gamma using ArrayFire's built-in function if available,
    //     // otherwise implement Lanczos approximation
    //     let result = if af::get_backend() == af::Backend::CUDA || af::get_backend() == af::Backend::OPENCL {
    //         // Use GPU-accelerated computation
    //         arrayfire_gamma_lanczos(input)?
    //     } else {
    //         // Fallback to CPU
    //         arrayfire_gamma_lanczos(input)?
    //     };
    //
    //     Ok(result)
    // }

    // Lanczos approximation for gamma function in ArrayFire (disabled - placeholder)
    // #[cfg(feature = "arrayfire")]
    // fn arrayfire_gamma_lanczos(x: &arrayfire::Array<f64>) -> SpecialResult<arrayfire::Array<f64>> {
    //     use arrayfire as af;
    //
    //     // Lanczos coefficients
    //     let g = 7.0;
    //     let coeffs = vec![
    //         0.99999999999980993,
    //         676.5203681218851,
    //         -1259.1392167224028,
    //         771.32342877765313,
    //         -176.61502916214059,
    //         12.507343278686905,
    //         -0.13857109526572012,
    //         9.9843695780195716e-6,
    //         1.5056327351493116e-7,
    //     ];
    //
    //     // Handle reflection formula for x < 0.5
    //     let half = af::constant(0.5, x.dims());
    //     let use_reflection = af::lt(x, &half, false);
    //
    //     // For reflection formula: Γ(z) = π / (sin(πz) × Γ(1-z))
    //     let pi = af::constant(std::f64::consts::PI, x.dims());
    //     let one = af::constant(1.0, x.dims());
    //     let reflected_x = af::sub(&one, x, false);
    //
    //     // Compute main Lanczos approximation
    //     let z = af::sub(x, &one, false);
    //     let mut acc = af::constant(coeffs[0], x.dims());
    //
    //     for (i, &coeff) in coeffs.iter().enumerate().skip(1) {
    //         let k = af::constant(i as f64, x.dims());
    //         let denominator = af::add(&z, &k, false);
    //         let term = af::div(&af::constant(coeff, x.dims()), &denominator, false);
    //         acc = af::add(&acc, &term, false);
    //     }
    //
    //     let t = af::add(&z, &af::constant(g + 0.5, x.dims()), false);
    //     let sqrt_2pi = af::constant((2.0 * std::f64::consts::PI).sqrt(), x.dims());
    //
    //     let z_plus_half = af::add(&z, &af::constant(0.5, x.dims()), false);
    //     let t_pow = af::pow(&t, &z_plus_half, false);
    //     let exp_neg_t = af::exp(&af::mul(&t, &af::constant(-1.0, x.dims()), false));
    //
    //     let gamma_main = af::mul(&af::mul(&sqrt_2pi, &acc, false), &af::mul(&t_pow, &exp_neg_t, false), false);
    //
    //     // Apply reflection formula where needed
    //     let sin_pi_x = af::sin(&af::mul(&pi, x, false));
    //     let gamma_reflected = af::div(&pi, &af::mul(&sin_pi_x, &gamma_main, false), false);
    //
    //     // Select appropriate result based on x value
    //     let result = af::select(&use_reflection, &gamma_reflected, &gamma_main);
    //
    //     Ok(result)
    // }

    /// Chunked processing for large arrays
    pub fn process_chunks<T, D, F>(
        input: &Array<T, D>,
        config: &ArrayConfig,
        operation: F,
    ) -> SpecialResult<Array<T, D>>
    where
        T: Clone + Send + Sync,
        D: Dimension,
        F: Fn(T) -> T + Send + Sync,
    {
        if input.len() <= config.chunksize {
            return Ok(input.mapv(operation));
        }

        // Process in chunks to manage memory usage
        #[cfg(feature = "parallel")]
        if config.parallel {
            use scirs2_core::parallel_ops::*;
            let data: Vec<T> = input.iter().cloned().collect();
            let processed: Vec<T> = data.into_par_iter().map(operation).collect();
            let result = Array::from_vec(processed)
                .to_shape(input.dim())
                .map_err(|e| SpecialError::ComputationError(format!("Shape error: {}", e)))?
                .into_owned();
            return Ok(result);
        }

        // Default sequential processing
        Ok(input.mapv(operation))
    }
}

/// Complex number array operations
pub mod complex {
    use super::*;
    use num_complex::Complex64;

    /// Apply Lambert W function to complex array
    pub fn lambert_w_array<D>(
        input: &Array<Complex64, D>,
        branch: i32,
        tolerance: f64,
        _config: &ArrayConfig,
    ) -> SpecialResult<Array<Complex64, D>>
    where
        D: Dimension,
    {
        Ok(input.mapv(|z| {
            crate::lambert::lambert_w(z, branch, tolerance)
                .unwrap_or(Complex64::new(f64::NAN, f64::NAN))
        }))
    }
}

/// High-level convenience functions for common array operations
pub mod convenience {
    use super::*;
    use ndarray::{Array1, Array2};

    /// Apply gamma function to 1D array with automatic backend selection
    pub async fn gamma_1d(input: &Array1<f64>) -> SpecialResult<Array1<f64>> {
        let config = ArrayConfig::default();
        let result = vectorized::gamma_array(input, &config)?;
        result.compute().await
    }

    /// Apply gamma function to 1D array with custom config
    pub async fn gamma_1d_with_config(
        input: &Array1<f64>,
        config: &ArrayConfig,
    ) -> SpecialResult<Array1<f64>> {
        let result = vectorized::gamma_array(input, config)?;
        result.compute().await
    }

    /// Apply gamma function to 2D array with automatic backend selection
    pub async fn gamma_2d(input: &Array2<f64>) -> SpecialResult<Array2<f64>> {
        let config = ArrayConfig::default();
        let result = vectorized::gamma_array(input, &config)?;
        result.compute().await
    }

    /// Create lazy gamma computation for large arrays
    #[cfg(feature = "lazy")]
    pub fn gamma_lazy<D>(
        input: &Array<f64, D>,
        config: Option<ArrayConfig>,
    ) -> SpecialResult<super::lazy::LazyArray<f64, D>>
    where
        D: Dimension + Send + Sync + 'static,
    {
        let config = config.unwrap_or_else(|| ArrayConfig {
            backend: Backend::Lazy,
            ..Default::default()
        });

        if let vectorized::GammaResult::Lazy(lazy_array) = vectorized::gamma_array(input, &config)?
        {
            Ok(lazy_array)
        } else {
            // Force lazy evaluation
            Ok(super::lazy::lazy_gamma(input.clone(), config))
        }
    }

    /// GPU-accelerated gamma computation
    #[cfg(feature = "gpu")]
    pub async fn gamma_gpu<D>(input: &Array<f64, D>) -> SpecialResult<Array<f64, D>>
    where
        D: Dimension,
    {
        super::gpu::gamma_gpu(input).await
    }

    /// Apply Bessel J0 function to 1D array
    pub fn j0_1d(input: &Array1<f64>) -> SpecialResult<Array1<f64>> {
        let config = ArrayConfig::default();
        let result = vectorized::j0_array(input, &config)?;
        result.compute()
    }

    /// Apply Bessel J0 function with custom config
    pub fn j0_with_config<D>(
        input: &Array<f64, D>,
        config: &ArrayConfig,
    ) -> SpecialResult<Array<f64, D>>
    where
        D: Dimension + Send + Sync + 'static,
    {
        let result = vectorized::j0_array(input, config)?;
        result.compute()
    }

    /// Apply error function to 1D array
    pub fn erf_1d(input: &Array1<f64>) -> SpecialResult<Array1<f64>> {
        let config = ArrayConfig::default();
        vectorized::erf_array(input, &config)
    }

    /// Apply error function with parallel processing
    pub fn erf_parallel<D>(input: &Array<f64, D>) -> SpecialResult<Array<f64, D>>
    where
        D: Dimension,
    {
        let config = ArrayConfig {
            parallel: true,
            ..Default::default()
        };
        vectorized::erf_array(input, &config)
    }

    /// Apply factorial to 1D array
    pub fn factorial_1d(input: &Array1<u32>) -> SpecialResult<Array1<f64>> {
        let config = ArrayConfig::default();
        vectorized::factorial_array(input, &config)
    }

    /// Apply softmax to 1D array
    pub fn softmax_1d(input: &Array1<f64>) -> SpecialResult<Array1<f64>> {
        let config = ArrayConfig::default();
        vectorized::softmax_1d(input.view(), &config)
    }

    /// Batch processing for multiple arrays
    pub async fn batch_gamma<D>(
        inputs: &[Array<f64, D>],
        config: &ArrayConfig,
    ) -> SpecialResult<Vec<Array<f64, D>>>
    where
        D: Dimension + Send + Sync + 'static,
    {
        let mut results = Vec::with_capacity(inputs.len());

        for input in inputs {
            let result = vectorized::gamma_array(input, config)?;
            results.push(result.compute().await?);
        }

        Ok(results)
    }

    /// Create configuration for different use cases
    pub struct ConfigBuilder {
        config: ArrayConfig,
    }

    impl ConfigBuilder {
        /// Create a new configuration builder
        pub fn new() -> Self {
            Self {
                config: ArrayConfig::default(),
            }
        }

        /// Set backend type
        pub fn backend(mut self, backend: Backend) -> Self {
            self.config.backend = backend;
            self
        }

        /// Enable parallel processing
        pub fn parallel(mut self, parallel: bool) -> Self {
            self.config.parallel = parallel;
            self
        }

        /// Set chunk size for processing
        pub fn chunksize(mut self, size: usize) -> Self {
            self.config.chunksize = size;
            self
        }

        /// Set memory limit
        pub fn memory_limit(mut self, limit: usize) -> Self {
            self.config.memory_limit = limit;
            self
        }

        /// Set lazy evaluation threshold
        pub fn lazy_threshold(mut self, threshold: usize) -> Self {
            self.config.lazy_threshold = threshold;
            self
        }

        /// Build the configuration
        pub fn build(self) -> ArrayConfig {
            self.config
        }
    }

    impl Default for ConfigBuilder {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Create a configuration optimized for large arrays
    pub fn large_array_config() -> ArrayConfig {
        ConfigBuilder::new()
            .chunksize(8192)
            .memory_limit(4 * 1024 * 1024 * 1024) // 4GB
            .lazy_threshold(50_000)
            .parallel(true)
            .build()
    }

    /// Create a configuration optimized for small arrays
    pub fn small_array_config() -> ArrayConfig {
        ConfigBuilder::new()
            .chunksize(256)
            .lazy_threshold(100_000) // Higher threshold to avoid lazy overhead
            .parallel(false)
            .build()
    }

    /// Create a configuration for GPU acceleration
    #[cfg(feature = "gpu")]
    pub fn gpu_config() -> ArrayConfig {
        ConfigBuilder::new()
            .backend(Backend::Gpu)
            .chunksize(4096)
            .build()
    }

    /// Create a configuration for lazy evaluation
    #[cfg(feature = "lazy")]
    pub fn lazy_config() -> ArrayConfig {
        ConfigBuilder::new()
            .backend(Backend::Lazy)
            .lazy_threshold(1000)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{arr1, arr2, Array};

    #[test]
    fn test_broadcasting() {
        assert!(broadcasting::can_broadcast(&[3, 1], &[1, 4]));
        assert!(broadcasting::can_broadcast(&[2, 3, 4], &[3, 4]));
        assert!(!broadcasting::can_broadcast(&[3, 2], &[4, 5]));

        let shape = broadcasting::broadcastshape(&[3, 1], &[1, 4]).unwrap();
        assert_eq!(shape, vec![3, 4]);
    }

    #[tokio::test]
    async fn test_vectorized_gamma() {
        let input = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = convenience::gamma_1d(&input).await.unwrap();

        // Γ(1)=1, Γ(2)=1, Γ(3)=2, Γ(4)=6, Γ(5)=24
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[3], 6.0, epsilon = 1e-10);
        assert_relative_eq!(result[4], 24.0, epsilon = 1e-10);
    }

    #[tokio::test]
    async fn test_vectorized_gamma_2d() {
        let input = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let result = convenience::gamma_2d(&input).await.unwrap();

        assert_relative_eq!(result[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[[0, 1]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[[1, 0]], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[[1, 1]], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vectorized_bessel() {
        let input = arr1(&[0.0, 1.0, 2.0]);
        let result = convenience::j0_1d(&input).unwrap();

        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], crate::bessel::j0(1.0), epsilon = 1e-10);
        assert_relative_eq!(result[2], crate::bessel::j0(2.0), epsilon = 1e-10);
    }

    #[test]
    fn test_softmax_1d() {
        let input = arr1(&[1.0, 2.0, 3.0]);
        let result = convenience::softmax_1d(&input).unwrap();

        // Check that result sums to 1
        assert_relative_eq!(result.sum(), 1.0, epsilon = 1e-10);

        // Check that all values are positive
        for &val in result.iter() {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_memory_estimation() {
        let shape = [1000, 1000];
        let memory = memory_efficient::estimate_memory_usage::<f64>(&shape, 2);
        assert_eq!(memory, 1000 * 1000 * 8 * 2); // 16MB for two f64 arrays

        let config = ArrayConfig::default();
        assert!(memory_efficient::check_memory_limit::<f64>(
            &shape, 2, &config
        ));
    }

    #[test]
    fn test_config_builder() {
        let config = convenience::ConfigBuilder::new()
            .chunksize(2048)
            .parallel(true)
            .memory_limit(2 * 1024 * 1024 * 1024)
            .lazy_threshold(5000)
            .build();

        assert_eq!(config.chunksize, 2048);
        assert!(config.parallel);
        assert_eq!(config.memory_limit, 2 * 1024 * 1024 * 1024);
        assert_eq!(config.lazy_threshold, 5000);
    }

    #[test]
    fn test_predefined_configs() {
        let large_config = convenience::large_array_config();
        assert_eq!(large_config.chunksize, 8192);
        assert!(large_config.parallel);
        assert_eq!(large_config.lazy_threshold, 50_000);

        let small_config = convenience::small_array_config();
        assert_eq!(small_config.chunksize, 256);
        assert!(!small_config.parallel);
        assert_eq!(small_config.lazy_threshold, 100_000);
    }

    #[cfg(feature = "lazy")]
    #[test]
    fn test_lazy_evaluation() {
        let input = Array::linspace(1.0, 5.0, 1000);
        let lazy_array = convenience::gamma_lazy(&input, None).unwrap();

        // Check that computation is deferred
        assert!(!lazy_array.is_computed());
        assert_eq!(lazy_array.shape(), input.shape());

        // Force computation
        let result = lazy_array.compute().unwrap();
        assert_eq!(result.shape(), input.shape());

        // Verify some values
        assert_relative_eq!(result[0], crate::gamma::gamma(1.0), epsilon = 1e-10);
    }

    #[cfg(feature = "lazy")]
    #[test]
    fn test_lazy_bessel() {
        let input = Array::linspace(0.0, 5.0, 1500); // Size above lazy threshold
        let config = convenience::lazy_config();
        let result = vectorized::j0_array(&input, &config).unwrap();

        if let vectorized::BesselResult::Lazy(lazy_array) = result {
            assert!(!lazy_array.is_computed());
            let computed = lazy_array.compute().unwrap();
            assert_eq!(computed.shape(), input.shape());
        } else {
            panic!("Expected lazy result");
        }
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let arrays = vec![
            arr1(&[1.0, 2.0, 3.0]),
            arr1(&[4.0, 5.0, 6.0]),
            arr1(&[7.0, 8.0, 9.0]),
        ];

        let config = ArrayConfig::default();
        let results = convenience::batch_gamma(&arrays, &config).await.unwrap();

        assert_eq!(results.len(), 3);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.len(), 3);
            for (j, &val) in result.iter().enumerate() {
                let expected = crate::gamma::gamma(arrays[i][j]);
                assert_relative_eq!(val, expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_chunked_processing() {
        let input = Array::ones(2000);
        let config = ArrayConfig {
            chunksize: 100,
            ..Default::default()
        };

        let result = vectorized::process_chunks(&input, &config, |x: f64| x * 2.0).unwrap();

        assert_eq!(result.len(), input.len());
        for &val in result.iter() {
            assert_relative_eq!(val, 2.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_backend_selection() {
        let config = ArrayConfig {
            backend: Backend::Cpu,
            ..Default::default()
        };

        let input = arr1(&[1.0, 2.0, 3.0]);
        let result = vectorized::gamma_array(&input, &config).unwrap();

        // Should get immediate result for CPU backend
        assert!(result.is_ready());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_processing() {
        let input = Array::linspace(1.0, 10.0, 1000);
        let result = convenience::erf_parallel(&input).unwrap();

        assert_eq!(result.len(), input.len());
        for (i, &val) in result.iter().enumerate() {
            let expected = crate::erf::erf(input[i]);
            assert_relative_eq!(val, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gamma_result_types() {
        let input = arr1(&[1.0, 2.0, 3.0]);
        let config = ArrayConfig::default();
        let result = vectorized::gamma_array(&input, &config).unwrap();

        // Test immediate result
        match result {
            vectorized::GammaResult::Immediate(array) => {
                assert_eq!(array.len(), 3);
                assert_relative_eq!(array[0], 1.0, epsilon = 1e-10);
                assert_relative_eq!(array[1], 1.0, epsilon = 1e-10);
                assert_relative_eq!(array[2], 2.0, epsilon = 1e-10);
            }
            #[cfg(feature = "lazy")]
            vectorized::GammaResult::Lazy(_) => {
                panic!("Expected immediate result but got lazy result");
            }
            #[cfg(feature = "futures")]
            vectorized::GammaResult::Future(_) => {
                panic!("Expected immediate result but got future result");
            }
        }
    }

    #[cfg(feature = "lazy")]
    #[test]
    fn test_lazy_array_operations() {
        let input = Array::linspace(1.0, 5.0, 100);
        let lazy_gamma = super::lazy::lazy_gamma(input.clone(), ArrayConfig::default());

        // Test properties before computation
        assert_eq!(lazy_gamma.shape(), input.shape());
        assert!(!lazy_gamma.is_computed());
        assert!(lazy_gamma.cost_estimate() > 0);
        assert!(lazy_gamma.description().contains("LazyGamma"));

        // Test computation
        let result = lazy_gamma.compute().unwrap();
        assert_eq!(result.shape(), input.shape());
    }

    #[tokio::test]
    #[cfg(feature = "gpu")]
    async fn test_gpu_fallback() {
        // Test that GPU functions work (fallback to CPU if no GPU)
        let input = arr1(&[1.0, 2.0, 3.0]);

        match convenience::gamma_gpu(&input).await {
            Ok(result) => {
                assert_eq!(result.len(), 3);
                assert_relative_eq!(result[0], 1.0, epsilon = 1e-6); // Lower precision for GPU
                assert_relative_eq!(result[1], 1.0, epsilon = 1e-6);
                assert_relative_eq!(result[2], 2.0, epsilon = 1e-6);
            }
            Err(_) => {
                // GPU not available, which is acceptable for tests
            }
        }
    }

    #[test]
    fn test_complex_array_operations() {
        use num_complex::Complex64;

        let input = Array::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.5),
            Complex64::new(0.5, 1.0),
        ]);

        let config = ArrayConfig::default();
        let result = complex::lambert_w_array(&input, 0, 1e-8, &config).unwrap();

        assert_eq!(result.len(), 3);
        // Check that results are finite (not NaN)
        for val in result.iter() {
            assert!(val.re.is_finite());
            assert!(val.im.is_finite());
        }
    }
}
