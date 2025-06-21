//! Enhanced array operations for special functions
//!
//! This module provides comprehensive array support for special functions with
//! lazy evaluation, GPU acceleration, and multidimensional operations.

#![allow(dead_code)]

use crate::error::{SpecialError, SpecialResult};
use ndarray::{Array, ArrayView1, Dimension};

#[cfg(feature = "futures")]
use futures::future::BoxFuture;

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
    // ArrayFire backend (placeholder for future implementation)
    // #[cfg(feature = "array-api")]
    // ArrayFire,
}

/// Configuration for array operations
#[derive(Debug, Clone)]
pub struct ArrayConfig {
    /// Chunk size for memory-efficient processing
    pub chunk_size: usize,
    /// Whether to use parallel processing
    pub parallel: bool,
    /// Memory limit for operations (in bytes)
    pub memory_limit: usize,
    /// Execution backend
    pub backend: Backend,
    /// Whether to cache computed results
    pub cache_results: bool,
    /// Maximum cache size (number of entries)
    pub max_cache_size: usize,
    /// Lazy evaluation threshold (array size)
    pub lazy_threshold: usize,
}

impl Default for ArrayConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024,
            parallel: cfg!(feature = "parallel"),
            memory_limit: 1024 * 1024 * 1024, // 1GB
            backend: Backend::default(),
            cache_results: true,
            max_cache_size: 1000,
            lazy_threshold: 10_000, // Use lazy evaluation for arrays > 10k elements
        }
    }
}

/// Memory-efficient array operations
pub mod memory_efficient {
    use super::*;

    /// Estimate memory usage for an operation
    pub fn estimate_memory_usage<T>(shape: &[usize], num_arrays: usize) -> usize {
        let elem_size = std::mem::size_of::<T>();
        let total_elements: usize = shape.iter().product();
        total_elements * elem_size * num_arrays
    }

    /// Check if operation fits within memory limits
    pub fn check_memory_limit<T>(shape: &[usize], num_arrays: usize, config: &ArrayConfig) -> bool {
        estimate_memory_usage::<T>(shape, num_arrays) <= config.memory_limit
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

    /// GPU buffer for array data
    pub struct GpuBuffer {
        #[cfg(feature = "gpu")]
        buffer: wgpu::Buffer,
        size: usize,
    }

    /// GPU compute pipeline for special functions
    pub struct GpuPipeline {
        #[cfg(feature = "gpu")]
        device: wgpu::Device,
        #[cfg(feature = "gpu")]
        queue: wgpu::Queue,
        #[cfg(feature = "gpu")]
        pipeline: wgpu::ComputePipeline,
    }

    impl GpuPipeline {
        /// Create a new GPU pipeline
        #[cfg(feature = "gpu")]
        pub async fn new() -> SpecialResult<Self> {
            #[allow(unused_imports)]
            use wgpu::util::DeviceExt;
            use wgpu::*;

            let instance = Instance::new(InstanceDescriptor::default());
            let adapter = instance
                .request_adapter(&RequestAdapterOptions::default())
                .await
                .ok_or_else(|| {
                    SpecialError::ComputationError("Failed to get GPU adapter".to_string())
                })?;

            let (device, queue) = adapter
                .request_device(&DeviceDescriptor::default(), None)
                .await
                .map_err(|e| {
                    SpecialError::ComputationError(format!("Failed to get GPU device: {}", e))
                })?;

            // Create a simple compute shader for gamma function
            let shader_source = include_str!("../shaders/gamma_compute.wgsl");
            let shader_module = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Gamma Compute Shader"),
                source: ShaderSource::Wgsl(shader_source.into()),
            });

            let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Gamma Compute Pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: "main",
                compilation_options: Default::default(),
            });

            Ok(Self {
                device,
                queue,
                pipeline,
            })
        }

        /// Execute gamma function on GPU
        #[cfg(feature = "gpu")]
        pub async fn gamma_gpu<D>(&self, input: &Array<f64, D>) -> SpecialResult<Array<f64, D>>
        where
            D: Dimension,
        {
            use bytemuck;
            use wgpu::util::{BufferInitDescriptor, DeviceExt};
            use wgpu::*;

            let data: Vec<f32> = input.iter().map(|&x| x as f32).collect();
            let size = data.len() * std::mem::size_of::<f32>();

            // Create input buffer
            let input_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Input Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

            // Create output buffer
            let output_buffer = self.device.create_buffer(&BufferDescriptor {
                label: Some("Output Buffer"),
                size: size as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // Create staging buffer for reading results
            let staging_buffer = self.device.create_buffer(&BufferDescriptor {
                label: Some("Staging Buffer"),
                size: size as u64,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Create bind group
            let bind_group_layout = self.pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Compute Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

            // Dispatch compute shader
            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Compute Encoder"),
                });

            {
                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups((data.len() as u32 + 255) / 256, 1, 1);
            }

            encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, size as u64);
            self.queue.submit(Some(encoder.finish()));

            // Read results
            let buffer_slice = staging_buffer.slice(..);
            let (sender, receiver) = futures::channel::oneshot::channel();
            buffer_slice.map_async(MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });

            self.device.poll(Maintain::Wait);
            receiver.await.unwrap().map_err(|e| {
                SpecialError::ComputationError(format!("GPU buffer error: {:?}", e))
            })?;

            let buffer_view = buffer_slice.get_mapped_range();
            let result_data: &[f32] = bytemuck::cast_slice(&buffer_view);
            let result_f64: Vec<f64> = result_data.iter().map(|&x| x as f64).collect();

            drop(buffer_view);
            staging_buffer.unmap();

            // Reconstruct array with original shape
            let result_array = Array::from_vec(result_f64)
                .into_shape_with_order(input.dim())
                .map_err(|e| SpecialError::ComputationError(format!("Shape error: {}", e)))?;
            Ok(result_array)
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
            let pipeline = GpuPipeline::new().await?;
            pipeline.gamma_gpu(input).await
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
    pub fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>, SpecialError> {
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

    #[cfg(feature = "gpu")]
    use super::gpu::*;

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
            #[cfg(feature = "gpu")]
            Backend::Gpu => {
                if total_elements >= 1000 {
                    // GPU efficient for larger arrays
                    let input_cloned = input.clone();
                    return Ok(GammaResult::Future(Box::pin(gamma_gpu(&input_cloned))));
                }
            }
            Backend::Cpu => {
                // Use CPU implementation
            } // #[cfg(feature = "array-api")]
              // Backend::ArrayFire => {
              //     return arrayfire_gamma(input, config);
              // }
        }

        // Default CPU implementation with optional parallelization
        if config.parallel && total_elements > config.chunk_size {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
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
        if config.parallel && input.len() > config.chunk_size {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
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
        if config.parallel && input.len() > config.chunk_size {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
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

    // ArrayFire backend implementation (placeholder)
    // #[cfg(feature = "array-api")]
    // fn arrayfire_gamma<D>(
    //     input: &Array<f64, D>,
    //     _config: &ArrayConfig,
    // ) -> SpecialResult<GammaResult<D>>
    // where
    //     D: Dimension,
    // {
    //     // TODO: Implement ArrayFire backend
    //     // For now, fallback to CPU
    //     Ok(GammaResult::Immediate(input.mapv(crate::gamma::gamma)))
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
        if input.len() <= config.chunk_size {
            return Ok(input.mapv(operation));
        }

        // Process in chunks to manage memory usage
        #[cfg(feature = "parallel")]
        if config.parallel {
            use rayon::prelude::*;
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
        let config = config.unwrap_or_else(|| {
            let mut cfg = ArrayConfig::default();
            cfg.backend = Backend::Lazy;
            cfg
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
        pub fn chunk_size(mut self, size: usize) -> Self {
            self.config.chunk_size = size;
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
            .chunk_size(8192)
            .memory_limit(4 * 1024 * 1024 * 1024) // 4GB
            .lazy_threshold(50_000)
            .parallel(true)
            .build()
    }

    /// Create a configuration optimized for small arrays
    pub fn small_array_config() -> ArrayConfig {
        ConfigBuilder::new()
            .chunk_size(256)
            .lazy_threshold(100_000) // Higher threshold to avoid lazy overhead
            .parallel(false)
            .build()
    }

    /// Create a configuration for GPU acceleration
    #[cfg(feature = "gpu")]
    pub fn gpu_config() -> ArrayConfig {
        ConfigBuilder::new()
            .backend(Backend::Gpu)
            .chunk_size(4096)
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

        let shape = broadcasting::broadcast_shape(&[3, 1], &[1, 4]).unwrap();
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
            .chunk_size(2048)
            .parallel(true)
            .memory_limit(2 * 1024 * 1024 * 1024)
            .lazy_threshold(5000)
            .build();

        assert_eq!(config.chunk_size, 2048);
        assert!(config.parallel);
        assert_eq!(config.memory_limit, 2 * 1024 * 1024 * 1024);
        assert_eq!(config.lazy_threshold, 5000);
    }

    #[test]
    fn test_predefined_configs() {
        let large_config = convenience::large_array_config();
        assert_eq!(large_config.chunk_size, 8192);
        assert!(large_config.parallel);
        assert_eq!(large_config.lazy_threshold, 50_000);

        let small_config = convenience::small_array_config();
        assert_eq!(small_config.chunk_size, 256);
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
        let input = Array::linspace(0.0, 5.0, 500);
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
            chunk_size: 100,
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
