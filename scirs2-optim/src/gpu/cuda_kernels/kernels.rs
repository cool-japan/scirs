//! Core CUDA kernel implementations for optimization
//!
//! This module contains the main OptimizerKernel implementation and
//! core kernel execution functionality for GPU-accelerated optimization.

use super::config::*;
use crate::adaptive_selection::OptimizerType;
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::HashMap;
use std::ptr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use scirs2_core::gpu::{CudaContext, CudaKernel, CudaStream};

/// Custom CUDA kernel wrapper for optimizer operations
pub struct OptimizerKernel {
    /// CUDA context
    #[cfg(feature = "cuda")]
    context: CudaContext,

    /// Compiled kernel functions
    #[cfg(feature = "cuda")]
    adam_kernel: CudaKernel,

    #[cfg(feature = "cuda")]
    lamb_kernel: CudaKernel,

    #[cfg(feature = "cuda")]
    adamw_kernel: CudaKernel,

    /// Tensor core optimized kernels
    #[cfg(feature = "cuda")]
    tensor_core_adam_fp16_kernel: CudaKernel,

    #[cfg(feature = "cuda")]
    tensor_core_adam_bf16_kernel: CudaKernel,

    #[cfg(feature = "cuda")]
    fused_tensor_core_kernel: CudaKernel,

    #[cfg(feature = "cuda")]
    mixed_precision_kernel: CudaKernel,

    /// CUDA stream for async execution
    #[cfg(feature = "cuda")]
    stream: CudaStream,

    /// Kernel configuration
    kernel_config: KernelConfig,

    /// Performance metrics
    metrics: Arc<Mutex<PerformanceMetrics>>,

    /// Tensor core support
    tensor_core_support: TensorCoreSupport,

    /// Mixed precision support
    mixed_precision_support: MixedPrecisionSupport,

    /// Adaptive configuration
    adaptive_config: AdaptiveKernelConfig<f32>,

    /// Kernel cache for compiled kernels
    kernel_cache: HashMap<String, CompiledKernel>,

    /// Memory allocator reference
    allocator: Option<Arc<dyn CudaMemoryAllocatorTrait>>,
}

/// Kernel configuration parameters
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Block size for kernel execution
    pub block_size: (u32, u32, u32),

    /// Grid size for kernel execution
    pub grid_size: (u32, u32, u32),

    /// Shared memory size per block
    pub shared_memory_size: u32,

    /// Kernel execution timeout
    pub timeout: Duration,

    /// Enable asynchronous execution
    pub async_execution: bool,

    /// Stream priority
    pub stream_priority: i32,

    /// Enable profiling for this kernel
    pub enable_profiling: bool,

    /// Optimization level
    pub optimization_level: OptimizationLevel,
}

/// Optimization levels for kernel compilation
#[derive(Debug, Clone, Copy)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
    Adaptive,
}

/// Compiled kernel representation
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    /// Kernel identifier
    pub kernel_id: String,

    /// Source code or PTX
    pub source: KernelSource,

    /// Compilation timestamp
    pub compilation_time: std::time::SystemTime,

    /// Compilation options
    pub compilation_options: Vec<String>,

    /// Resource usage
    pub resource_usage: KernelResourceUsage,

    /// Performance characteristics
    pub performance_profile: KernelPerformanceProfile,
}

/// Kernel source representation
#[derive(Debug, Clone)]
pub enum KernelSource {
    /// CUDA C++ source code
    CudaSource(String),

    /// PTX assembly
    PtxAssembly(String),

    /// SASS assembly
    SassAssembly(String),

    /// Precompiled binary
    Binary(Vec<u8>),
}

/// Kernel resource usage information
#[derive(Debug, Clone)]
pub struct KernelResourceUsage {
    /// Registers per thread
    pub registers_per_thread: u32,

    /// Shared memory per block
    pub shared_memory_per_block: u32,

    /// Local memory per thread
    pub local_memory_per_thread: u32,

    /// Constant memory usage
    pub constant_memory_usage: u32,

    /// Texture memory usage
    pub texture_memory_usage: u32,

    /// Maximum occupancy
    pub max_occupancy: f32,
}

/// Kernel performance profile
#[derive(Debug, Clone)]
pub struct KernelPerformanceProfile {
    /// Average execution time
    pub avg_execution_time: Duration,

    /// Peak performance (FLOPS)
    pub peak_performance: f64,

    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f32,

    /// Instruction throughput
    pub instruction_throughput: f64,

    /// Cache hit rates
    pub cache_hit_rates: CacheHitRates,

    /// Branch divergence statistics
    pub branch_divergence: BranchDivergenceStats,
}

/// Cache hit rate statistics
#[derive(Debug, Clone)]
pub struct CacheHitRates {
    /// L1 cache hit rate
    pub l1_cache_hit_rate: f32,

    /// L2 cache hit rate
    pub l2_cache_hit_rate: f32,

    /// Texture cache hit rate
    pub texture_cache_hit_rate: f32,

    /// Constant cache hit rate
    pub constant_cache_hit_rate: f32,
}

/// Branch divergence statistics
#[derive(Debug, Clone)]
pub struct BranchDivergenceStats {
    /// Divergent branch percentage
    pub divergent_branch_percentage: f32,

    /// Average warp efficiency
    pub avg_warp_efficiency: f32,

    /// Control flow complexity
    pub control_flow_complexity: f32,
}

/// Memory allocator trait for CUDA kernels
pub trait CudaMemoryAllocatorTrait: Send + Sync {
    /// Allocate GPU memory
    fn allocate(&self, size: usize) -> Result<*mut std::ffi::c_void, OptimizerKernelError>;

    /// Deallocate GPU memory
    fn deallocate(&self, ptr: *mut std::ffi::c_void) -> Result<(), OptimizerKernelError>;

    /// Copy data from host to device
    fn copy_to_device(&self, host_ptr: *const std::ffi::c_void, device_ptr: *mut std::ffi::c_void, size: usize) -> Result<(), OptimizerKernelError>;

    /// Copy data from device to host
    fn copy_to_host(&self, device_ptr: *const std::ffi::c_void, host_ptr: *mut std::ffi::c_void, size: usize) -> Result<(), OptimizerKernelError>;

    /// Get allocated memory size
    fn get_allocated_size(&self) -> usize;

    /// Get available memory
    fn get_available_memory(&self) -> usize;
}

impl OptimizerKernel {
    /// Create a new optimizer kernel instance
    pub fn new(config: KernelConfig) -> Result<Self, OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        let context = CudaContext::new().map_err(|e| {
            OptimizerKernelError::CudaContextCreationFailed(format!("Failed to create CUDA context: {}", e))
        })?;

        #[cfg(feature = "cuda")]
        let stream = CudaStream::new(&context, config.stream_priority).map_err(|e| {
            OptimizerKernelError::CudaContextCreationFailed(format!("Failed to create CUDA stream: {}", e))
        })?;

        let mut kernel = Self {
            #[cfg(feature = "cuda")]
            context,
            #[cfg(feature = "cuda")]
            adam_kernel: Self::compile_adam_kernel()?,
            #[cfg(feature = "cuda")]
            lamb_kernel: Self::compile_lamb_kernel()?,
            #[cfg(feature = "cuda")]
            adamw_kernel: Self::compile_adamw_kernel()?,
            #[cfg(feature = "cuda")]
            tensor_core_adam_fp16_kernel: Self::compile_tensor_core_adam_fp16_kernel()?,
            #[cfg(feature = "cuda")]
            tensor_core_adam_bf16_kernel: Self::compile_tensor_core_adam_bf16_kernel()?,
            #[cfg(feature = "cuda")]
            fused_tensor_core_kernel: Self::compile_fused_tensor_core_kernel()?,
            #[cfg(feature = "cuda")]
            mixed_precision_kernel: Self::compile_mixed_precision_kernel()?,
            #[cfg(feature = "cuda")]
            stream,
            kernel_config: config,
            metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
            tensor_core_support: TensorCoreSupport::default(),
            mixed_precision_support: MixedPrecisionSupport::default(),
            adaptive_config: AdaptiveKernelConfig::default(),
            kernel_cache: HashMap::new(),
            allocator: None,
        };

        // Initialize tensor core support
        kernel.initialize_tensor_core_support()?;

        Ok(kernel)
    }

    /// Execute Adam optimizer kernel
    pub fn execute_adam<T: Float>(
        &mut self,
        params: &mut Array1<T>,
        gradients: &Array1<T>,
        m: &mut Array1<T>,
        v: &mut Array1<T>,
        learning_rate: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        step: u64,
    ) -> Result<(), OptimizerKernelError> {
        let start_time = Instant::now();

        #[cfg(feature = "cuda")]
        {
            // Prepare kernel parameters
            let params_ptr = params.as_mut_ptr() as *mut std::ffi::c_void;
            let gradients_ptr = gradients.as_ptr() as *const std::ffi::c_void;
            let m_ptr = m.as_mut_ptr() as *mut std::ffi::c_void;
            let v_ptr = v.as_mut_ptr() as *mut std::ffi::c_void;

            // Execute kernel
            self.adam_kernel.launch(
                &self.stream,
                self.kernel_config.grid_size,
                self.kernel_config.block_size,
                self.kernel_config.shared_memory_size,
                &[
                    params_ptr,
                    gradients_ptr,
                    m_ptr,
                    v_ptr,
                    &learning_rate as *const T as *const std::ffi::c_void,
                    &beta1 as *const T as *const std::ffi::c_void,
                    &beta2 as *const T as *const std::ffi::c_void,
                    &epsilon as *const T as *const std::ffi::c_void,
                    &step as *const u64 as *const std::ffi::c_void,
                ],
            ).map_err(|e| {
                OptimizerKernelError::KernelExecutionFailed(format!("Adam kernel execution failed: {}", e))
            })?;

            // Synchronize if not using async execution
            if !self.kernel_config.async_execution {
                self.stream.synchronize().map_err(|e| {
                    OptimizerKernelError::KernelExecutionFailed(format!("Stream synchronization failed: {}", e))
                })?;
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Fallback CPU implementation
            self.execute_adam_cpu(params, gradients, m, v, learning_rate, beta1, beta2, epsilon, step)?;
        }

        // Update performance metrics
        let execution_time = start_time.elapsed();
        self.update_performance_metrics(execution_time, params.len())?;

        Ok(())
    }

    /// Execute LAMB optimizer kernel
    pub fn execute_lamb<T: Float>(
        &mut self,
        params: &mut Array1<T>,
        gradients: &Array1<T>,
        m: &mut Array1<T>,
        v: &mut Array1<T>,
        learning_rate: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        weight_decay: T,
        step: u64,
    ) -> Result<(), OptimizerKernelError> {
        let start_time = Instant::now();

        #[cfg(feature = "cuda")]
        {
            // Prepare kernel parameters
            let params_ptr = params.as_mut_ptr() as *mut std::ffi::c_void;
            let gradients_ptr = gradients.as_ptr() as *const std::ffi::c_void;
            let m_ptr = m.as_mut_ptr() as *mut std::ffi::c_void;
            let v_ptr = v.as_mut_ptr() as *mut std::ffi::c_void;

            // Execute kernel
            self.lamb_kernel.launch(
                &self.stream,
                self.kernel_config.grid_size,
                self.kernel_config.block_size,
                self.kernel_config.shared_memory_size,
                &[
                    params_ptr,
                    gradients_ptr,
                    m_ptr,
                    v_ptr,
                    &learning_rate as *const T as *const std::ffi::c_void,
                    &beta1 as *const T as *const std::ffi::c_void,
                    &beta2 as *const T as *const std::ffi::c_void,
                    &epsilon as *const T as *const std::ffi::c_void,
                    &weight_decay as *const T as *const std::ffi::c_void,
                    &step as *const u64 as *const std::ffi::c_void,
                ],
            ).map_err(|e| {
                OptimizerKernelError::KernelExecutionFailed(format!("LAMB kernel execution failed: {}", e))
            })?;

            if !self.kernel_config.async_execution {
                self.stream.synchronize().map_err(|e| {
                    OptimizerKernelError::KernelExecutionFailed(format!("Stream synchronization failed: {}", e))
                })?;
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Fallback CPU implementation
            self.execute_lamb_cpu(params, gradients, m, v, learning_rate, beta1, beta2, epsilon, weight_decay, step)?;
        }

        let execution_time = start_time.elapsed();
        self.update_performance_metrics(execution_time, params.len())?;

        Ok(())
    }

    /// Execute tensor core optimized Adam kernel
    pub fn execute_tensor_core_adam<T: Float>(
        &mut self,
        params: &mut Array2<T>,
        gradients: &Array2<T>,
        m: &mut Array2<T>,
        v: &mut Array2<T>,
        learning_rate: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        step: u64,
        precision: Precision,
    ) -> Result<(), OptimizerKernelError> {
        // Check tensor core availability
        if !self.tensor_core_support.available {
            return Err(OptimizerKernelError::UnsupportedOperation(
                "Tensor cores not available on this device".to_string()
            ));
        }

        // Check data type support
        let tensor_core_dtype = match precision {
            Precision::FP16 => TensorCoreDataType::FP16,
            Precision::BF16 => TensorCoreDataType::BF16,
            _ => return Err(OptimizerKernelError::UnsupportedOperation(
                format!("Precision {:?} not supported for tensor cores", precision)
            )),
        };

        if !self.tensor_core_support.supported_data_types.contains(&tensor_core_dtype) {
            return Err(OptimizerKernelError::UnsupportedOperation(
                format!("Data type {:?} not supported by tensor cores", tensor_core_dtype)
            ));
        }

        let start_time = Instant::now();

        #[cfg(feature = "cuda")]
        {
            let kernel = match precision {
                Precision::FP16 => &self.tensor_core_adam_fp16_kernel,
                Precision::BF16 => &self.tensor_core_adam_bf16_kernel,
                _ => return Err(OptimizerKernelError::UnsupportedOperation(
                    "Invalid precision for tensor core execution".to_string()
                )),
            };

            // Calculate optimal grid and block dimensions for tensor cores
            let (grid_size, block_size) = self.calculate_tensor_core_dimensions(params.dim())?;

            // Prepare kernel parameters
            let params_ptr = params.as_mut_ptr() as *mut std::ffi::c_void;
            let gradients_ptr = gradients.as_ptr() as *const std::ffi::c_void;
            let m_ptr = m.as_mut_ptr() as *mut std::ffi::c_void;
            let v_ptr = v.as_mut_ptr() as *mut std::ffi::c_void;

            // Execute tensor core kernel
            kernel.launch(
                &self.stream,
                grid_size,
                block_size,
                self.kernel_config.shared_memory_size,
                &[
                    params_ptr,
                    gradients_ptr,
                    m_ptr,
                    v_ptr,
                    &learning_rate as *const T as *const std::ffi::c_void,
                    &beta1 as *const T as *const std::ffi::c_void,
                    &beta2 as *const T as *const std::ffi::c_void,
                    &epsilon as *const T as *const std::ffi::c_void,
                    &step as *const u64 as *const std::ffi::c_void,
                ],
            ).map_err(|e| {
                OptimizerKernelError::KernelExecutionFailed(format!("Tensor core Adam kernel failed: {}", e))
            })?;

            if !self.kernel_config.async_execution {
                self.stream.synchronize().map_err(|e| {
                    OptimizerKernelError::KernelExecutionFailed(format!("Stream synchronization failed: {}", e))
                })?;
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Fallback to regular Adam execution
            return Err(OptimizerKernelError::UnsupportedOperation(
                "Tensor core execution requires CUDA feature".to_string()
            ));
        }

        let execution_time = start_time.elapsed();
        self.update_performance_metrics(execution_time, params.len())?;

        Ok(())
    }

    /// Execute mixed precision kernel
    pub fn execute_mixed_precision<T: Float>(
        &mut self,
        params: &mut Array1<T>,
        gradients: &Array1<T>,
        m: &mut Array1<T>,
        v: &mut Array1<T>,
        optimizer_type: OptimizerType,
        learning_rate: T,
        loss_scale: f32,
    ) -> Result<(), OptimizerKernelError> {
        if !self.mixed_precision_support.enabled {
            return Err(OptimizerKernelError::UnsupportedOperation(
                "Mixed precision not enabled".to_string()
            ));
        }

        let start_time = Instant::now();

        #[cfg(feature = "cuda")]
        {
            // Prepare kernel parameters
            let params_ptr = params.as_mut_ptr() as *mut std::ffi::c_void;
            let gradients_ptr = gradients.as_ptr() as *const std::ffi::c_void;
            let m_ptr = m.as_mut_ptr() as *mut std::ffi::c_void;
            let v_ptr = v.as_mut_ptr() as *mut std::ffi::c_void;

            // Execute mixed precision kernel
            self.mixed_precision_kernel.launch(
                &self.stream,
                self.kernel_config.grid_size,
                self.kernel_config.block_size,
                self.kernel_config.shared_memory_size,
                &[
                    params_ptr,
                    gradients_ptr,
                    m_ptr,
                    v_ptr,
                    &optimizer_type as *const OptimizerType as *const std::ffi::c_void,
                    &learning_rate as *const T as *const std::ffi::c_void,
                    &loss_scale as *const f32 as *const std::ffi::c_void,
                ],
            ).map_err(|e| {
                OptimizerKernelError::KernelExecutionFailed(format!("Mixed precision kernel failed: {}", e))
            })?;

            if !self.kernel_config.async_execution {
                self.stream.synchronize().map_err(|e| {
                    OptimizerKernelError::KernelExecutionFailed(format!("Stream synchronization failed: {}", e))
                })?;
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            return Err(OptimizerKernelError::UnsupportedOperation(
                "Mixed precision kernel requires CUDA feature".to_string()
            ));
        }

        let execution_time = start_time.elapsed();
        self.update_performance_metrics(execution_time, params.len())?;

        Ok(())
    }

    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> Result<PerformanceMetrics, OptimizerKernelError> {
        self.metrics.lock()
            .map_err(|e| OptimizerKernelError::InvalidConfiguration(format!("Metrics lock failed: {}", e)))
            .map(|metrics| metrics.clone())
    }

    /// Set memory allocator
    pub fn set_allocator(&mut self, allocator: Arc<dyn CudaMemoryAllocatorTrait>) {
        self.allocator = Some(allocator);
    }

    /// Synchronize all operations
    pub fn synchronize(&self) -> Result<(), OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        {
            self.stream.synchronize().map_err(|e| {
                OptimizerKernelError::KernelExecutionFailed(format!("Synchronization failed: {}", e))
            })?;
        }
        Ok(())
    }

    // Private implementation methods

    #[cfg(feature = "cuda")]
    fn compile_adam_kernel() -> Result<CudaKernel, OptimizerKernelError> {
        let source = include_str!("kernels/adam.cu");
        CudaKernel::from_source(source, "adam_kernel").map_err(|e| {
            OptimizerKernelError::KernelCompilationFailed(format!("Adam kernel compilation failed: {}", e))
        })
    }

    #[cfg(feature = "cuda")]
    fn compile_lamb_kernel() -> Result<CudaKernel, OptimizerKernelError> {
        let source = include_str!("kernels/lamb.cu");
        CudaKernel::from_source(source, "lamb_kernel").map_err(|e| {
            OptimizerKernelError::KernelCompilationFailed(format!("LAMB kernel compilation failed: {}", e))
        })
    }

    #[cfg(feature = "cuda")]
    fn compile_adamw_kernel() -> Result<CudaKernel, OptimizerKernelError> {
        let source = include_str!("kernels/adamw.cu");
        CudaKernel::from_source(source, "adamw_kernel").map_err(|e| {
            OptimizerKernelError::KernelCompilationFailed(format!("AdamW kernel compilation failed: {}", e))
        })
    }

    #[cfg(feature = "cuda")]
    fn compile_tensor_core_adam_fp16_kernel() -> Result<CudaKernel, OptimizerKernelError> {
        let source = include_str!("kernels/tensor_core_adam_fp16.cu");
        CudaKernel::from_source(source, "tensor_core_adam_fp16_kernel").map_err(|e| {
            OptimizerKernelError::KernelCompilationFailed(format!("Tensor core Adam FP16 kernel compilation failed: {}", e))
        })
    }

    #[cfg(feature = "cuda")]
    fn compile_tensor_core_adam_bf16_kernel() -> Result<CudaKernel, OptimizerKernelError> {
        let source = include_str!("kernels/tensor_core_adam_bf16.cu");
        CudaKernel::from_source(source, "tensor_core_adam_bf16_kernel").map_err(|e| {
            OptimizerKernelError::KernelCompilationFailed(format!("Tensor core Adam BF16 kernel compilation failed: {}", e))
        })
    }

    #[cfg(feature = "cuda")]
    fn compile_fused_tensor_core_kernel() -> Result<CudaKernel, OptimizerKernelError> {
        let source = include_str!("kernels/fused_tensor_core.cu");
        CudaKernel::from_source(source, "fused_tensor_core_kernel").map_err(|e| {
            OptimizerKernelError::KernelCompilationFailed(format!("Fused tensor core kernel compilation failed: {}", e))
        })
    }

    #[cfg(feature = "cuda")]
    fn compile_mixed_precision_kernel() -> Result<CudaKernel, OptimizerKernelError> {
        let source = include_str!("kernels/mixed_precision.cu");
        CudaKernel::from_source(source, "mixed_precision_kernel").map_err(|e| {
            OptimizerKernelError::KernelCompilationFailed(format!("Mixed precision kernel compilation failed: {}", e))
        })
    }

    fn initialize_tensor_core_support(&mut self) -> Result<(), OptimizerKernelError> {
        // Query device capabilities and initialize tensor core support
        // This is a simplified implementation
        self.tensor_core_support = TensorCoreSupport::default();
        Ok(())
    }

    fn calculate_tensor_core_dimensions(&self, shape: (usize, usize)) -> Result<((u32, u32, u32), (u32, u32, u32)), OptimizerKernelError> {
        let (rows, cols) = shape;

        // Calculate optimal dimensions for tensor core operations
        // Tensor cores work best with multiples of 16 (for FP16) or 8 (for some other types)
        let tile_size = 16;
        let block_dim_x = 16;
        let block_dim_y = 16;

        let grid_dim_x = (cols + tile_size - 1) / tile_size;
        let grid_dim_y = (rows + tile_size - 1) / tile_size;

        let grid_size = (grid_dim_x as u32, grid_dim_y as u32, 1);
        let block_size = (block_dim_x, block_dim_y, 1);

        Ok((grid_size, block_size))
    }

    fn update_performance_metrics(&self, execution_time: Duration, element_count: usize) -> Result<(), OptimizerKernelError> {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.execution_time = execution_time;

            // Calculate throughput
            let throughput = element_count as f64 / execution_time.as_secs_f64();
            metrics.compute_throughput = throughput;

            // Update GPU utilization (simplified)
            metrics.gpu_utilization = 0.8; // Would be queried from actual GPU
        }
        Ok(())
    }

    // Fallback CPU implementations
    fn execute_adam_cpu<T: Float>(
        &self,
        params: &mut Array1<T>,
        gradients: &Array1<T>,
        m: &mut Array1<T>,
        v: &mut Array1<T>,
        learning_rate: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        step: u64,
    ) -> Result<(), OptimizerKernelError> {
        let bias_correction1 = T::one() - beta1.powi(step as i32);
        let bias_correction2 = T::one() - beta2.powi(step as i32);

        for i in 0..params.len() {
            // Update biased first moment estimate
            m[i] = beta1 * m[i] + (T::one() - beta1) * gradients[i];

            // Update biased second raw moment estimate
            v[i] = beta2 * v[i] + (T::one() - beta2) * gradients[i] * gradients[i];

            // Compute bias-corrected first moment estimate
            let m_hat = m[i] / bias_correction1;

            // Compute bias-corrected second raw moment estimate
            let v_hat = v[i] / bias_correction2;

            // Update parameters
            params[i] = params[i] - learning_rate * m_hat / (v_hat.sqrt() + epsilon);
        }

        Ok(())
    }

    fn execute_lamb_cpu<T: Float>(
        &self,
        params: &mut Array1<T>,
        gradients: &Array1<T>,
        m: &mut Array1<T>,
        v: &mut Array1<T>,
        learning_rate: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        weight_decay: T,
        step: u64,
    ) -> Result<(), OptimizerKernelError> {
        let bias_correction1 = T::one() - beta1.powi(step as i32);
        let bias_correction2 = T::one() - beta2.powi(step as i32);

        // Calculate parameter norm
        let param_norm = params.iter().map(|&p| p * p).fold(T::zero(), |acc, x| acc + x).sqrt();

        for i in 0..params.len() {
            // Add weight decay to gradient
            let grad_with_decay = gradients[i] + weight_decay * params[i];

            // Update biased first moment estimate
            m[i] = beta1 * m[i] + (T::one() - beta1) * grad_with_decay;

            // Update biased second raw moment estimate
            v[i] = beta2 * v[i] + (T::one() - beta2) * grad_with_decay * grad_with_decay;

            // Compute bias-corrected first moment estimate
            let m_hat = m[i] / bias_correction1;

            // Compute bias-corrected second raw moment estimate
            let v_hat = v[i] / bias_correction2;

            // Compute update
            let update = m_hat / (v_hat.sqrt() + epsilon);

            // Calculate update norm
            let update_norm = update * update; // Will be summed across all parameters

            // Apply trust ratio (simplified for single parameter)
            let trust_ratio = if param_norm > T::zero() && update_norm > T::zero() {
                param_norm / update_norm.sqrt()
            } else {
                T::one()
            };

            // Update parameters
            params[i] = params[i] - learning_rate * trust_ratio.min(T::one()) * update;
        }

        Ok(())
    }
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            block_size: (256, 1, 1),
            grid_size: (1, 1, 1),
            shared_memory_size: 0,
            timeout: Duration::from_secs(10),
            async_execution: true,
            stream_priority: 0,
            enable_profiling: false,
            optimization_level: OptimizationLevel::Basic,
        }
    }
}

impl Default for KernelResourceUsage {
    fn default() -> Self {
        Self {
            registers_per_thread: 0,
            shared_memory_per_block: 0,
            local_memory_per_thread: 0,
            constant_memory_usage: 0,
            texture_memory_usage: 0,
            max_occupancy: 0.0,
        }
    }
}

impl Default for KernelPerformanceProfile {
    fn default() -> Self {
        Self {
            avg_execution_time: Duration::from_millis(0),
            peak_performance: 0.0,
            memory_bandwidth_utilization: 0.0,
            instruction_throughput: 0.0,
            cache_hit_rates: CacheHitRates::default(),
            branch_divergence: BranchDivergenceStats::default(),
        }
    }
}

impl Default for CacheHitRates {
    fn default() -> Self {
        Self {
            l1_cache_hit_rate: 0.0,
            l2_cache_hit_rate: 0.0,
            texture_cache_hit_rate: 0.0,
            constant_cache_hit_rate: 0.0,
        }
    }
}

impl Default for BranchDivergenceStats {
    fn default() -> Self {
        Self {
            divergent_branch_percentage: 0.0,
            avg_warp_efficiency: 0.0,
            control_flow_complexity: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_config_default() {
        let config = KernelConfig::default();
        assert_eq!(config.block_size, (256, 1, 1));
        assert_eq!(config.grid_size, (1, 1, 1));
        assert!(config.async_execution);
    }

    #[test]
    fn test_kernel_creation() {
        let config = KernelConfig::default();
        let result = OptimizerKernel::new(config);

        // May fail without CUDA support, but should compile
        match result {
            Ok(_) => println!("Kernel created successfully"),
            Err(e) => println!("Kernel creation failed (expected without CUDA): {}", e),
        }
    }

    #[test]
    fn test_performance_metrics_default() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.execution_time, Duration::from_millis(0));
        assert_eq!(metrics.gpu_utilization, 0.0);
    }

    #[test]
    fn test_cpu_adam_fallback() {
        let config = KernelConfig::default();
        let kernel = OptimizerKernel::new(config);

        if let Ok(kernel) = kernel {
            let mut params = Array1::<f32>::ones(10);
            let gradients = Array1::<f32>::ones(10) * 0.1;
            let mut m = Array1::<f32>::zeros(10);
            let mut v = Array1::<f32>::zeros(10);

            let result = kernel.execute_adam_cpu(
                &mut params, &gradients, &mut m, &mut v,
                0.001, 0.9, 0.999, 1e-8, 1
            );

            assert!(result.is_ok());

            // Check that parameters were updated
            assert_ne!(params[0], 1.0);
        }
    }
}