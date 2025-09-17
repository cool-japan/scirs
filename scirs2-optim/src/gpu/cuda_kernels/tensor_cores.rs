//! NVIDIA Tensor Core support for accelerated matrix operations
//!
//! This module provides comprehensive support for NVIDIA Tensor Cores across
//! different GPU generations (Volta, Turing, Ampere, Ada Lovelace, Hopper),
//! enabling mixed-precision training and high-performance matrix operations.

use crate::gpu::cuda_kernels::config::*;
use scirs2_core::error::{Result, ScirsMlError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaFunction, CudaModule};

/// Tensor Core operation types supported across generations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorCoreOperation {
    /// Matrix multiplication: C = A * B + C
    GEMM,
    /// Batched matrix multiplication
    BatchedGEMM,
    /// Convolution operation
    Convolution,
    /// Fused matrix operations with activation
    FusedGEMM,
    /// Sparse matrix operations (Ampere+)
    SparseGEMM,
    /// Transformer attention (Hopper+)
    AttentionQKV,
    /// Element-wise operations with broadcasting
    ElementWise,
}

/// Precision modes supported by Tensor Cores
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorCorePrecision {
    /// FP16 input, FP16 accumulate
    FP16,
    /// FP16 input, FP32 accumulate
    Mixed16_32,
    /// BF16 input, FP32 accumulate (Ampere+)
    BF16,
    /// TF32 input, FP32 accumulate (Ampere+)
    TF32,
    /// FP8 precision (Hopper+)
    FP8,
    /// INT8 precision
    INT8,
    /// INT4 precision (Ada Lovelace+)
    INT4,
}

/// Matrix layout for Tensor Core operations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatrixLayout {
    /// Row-major layout
    RowMajor,
    /// Column-major layout
    ColumnMajor,
    /// Tensor Core optimized layout
    TensorCoreOptimized,
}

/// Tensor Core capability information
#[derive(Debug, Clone)]
pub struct TensorCoreCapability {
    /// Tensor Core generation
    pub generation: TensorCoreGeneration,
    /// Supported operations
    pub supported_operations: Vec<TensorCoreOperation>,
    /// Supported precision modes
    pub supported_precisions: Vec<TensorCorePrecision>,
    /// Maximum matrix dimensions
    pub max_dimensions: (usize, usize, usize), // (M, N, K)
    /// Warp size for operations
    pub warp_size: usize,
    /// Number of Tensor Cores per SM
    pub cores_per_sm: usize,
    /// Peak throughput in TOPS
    pub peak_throughput: f64,
}

/// Tensor Core operation configuration
#[derive(Debug, Clone)]
pub struct TensorCoreConfig {
    /// Operation type
    pub operation: TensorCoreOperation,
    /// Precision mode
    pub precision: TensorCorePrecision,
    /// Matrix dimensions (M, N, K)
    pub dimensions: (usize, usize, usize),
    /// Batch size for batched operations
    pub batch_size: Option<usize>,
    /// Matrix layouts
    pub layout_a: MatrixLayout,
    pub layout_b: MatrixLayout,
    pub layout_c: MatrixLayout,
    /// Alpha and beta scaling factors
    pub alpha: f32,
    pub beta: f32,
    /// Use automatic mixed precision
    pub use_amp: bool,
    /// Enable operation fusion
    pub enable_fusion: bool,
}

/// Tensor Core operation descriptor
pub struct TensorCoreDescriptor {
    /// Operation configuration
    config: TensorCoreConfig,
    /// Compiled CUDA kernel
    #[cfg(feature = "cuda")]
    kernel: Option<CudaFunction>,
    /// Kernel source code
    kernel_source: String,
    /// Optimal grid and block dimensions
    grid_dims: (u32, u32, u32),
    block_dims: (u32, u32, u32),
    /// Shared memory size requirement
    shared_mem_bytes: u32,
}

/// Tensor Core manager for handling operations across GPU generations
pub struct TensorCoreManager {
    /// Current GPU capabilities
    capabilities: TensorCoreCapability,
    /// Compiled operation descriptors cache
    descriptors: Arc<Mutex<HashMap<String, TensorCoreDescriptor>>>,
    /// CUDA device reference
    #[cfg(feature = "cuda")]
    device: Arc<CudaDevice>,
    /// Performance metrics
    metrics: Arc<Mutex<TensorCoreMetrics>>,
}

/// Performance metrics for Tensor Core operations
#[derive(Debug, Clone, Default)]
pub struct TensorCoreMetrics {
    /// Total operations executed
    pub total_operations: u64,
    /// Total time spent in Tensor Core operations
    pub total_time_ms: f64,
    /// Average operation time
    pub avg_operation_time_ms: f64,
    /// Peak throughput achieved (TOPS)
    pub peak_throughput_tops: f64,
    /// Average throughput (TOPS)
    pub avg_throughput_tops: f64,
    /// Utilization percentage
    pub utilization_percent: f64,
    /// Error count
    pub error_count: u64,
    /// Per-operation metrics
    pub operation_metrics: HashMap<TensorCoreOperation, OperationMetrics>,
}

/// Metrics for specific operation types
#[derive(Debug, Clone, Default)]
pub struct OperationMetrics {
    /// Number of executions
    pub execution_count: u64,
    /// Total execution time
    pub total_time_ms: f64,
    /// Best execution time
    pub best_time_ms: f64,
    /// Average throughput for this operation
    pub avg_throughput_tops: f64,
}

impl TensorCoreManager {
    /// Creates a new Tensor Core manager
    pub fn new() -> Result<Self> {
        #[cfg(feature = "cuda")]
        let device = Arc::new(CudaDevice::new(0)?);

        let capabilities = Self::detect_capabilities(
            #[cfg(feature = "cuda")]
            &device
        )?;

        Ok(Self {
            capabilities,
            descriptors: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(feature = "cuda")]
            device,
            metrics: Arc::new(Mutex::new(TensorCoreMetrics::default())),
        })
    }

    /// Detects Tensor Core capabilities of the current GPU
    fn detect_capabilities(
        #[cfg(feature = "cuda")]
        device: &CudaDevice
    ) -> Result<TensorCoreCapability> {
        #[cfg(feature = "cuda")]
        {
            // Get GPU architecture information
            let major = device.compute_capability().0;
            let minor = device.compute_capability().1;

            let generation = match (major, minor) {
                (7, 0) => TensorCoreGeneration::V1, // Volta V100
                (7, 5) => TensorCoreGeneration::V2, // Turing RTX 20xx
                (8, 0) | (8, 6) => TensorCoreGeneration::V3, // Ampere A100, RTX 30xx
                (8, 9) => TensorCoreGeneration::V4, // Ada Lovelace RTX 40xx
                (9, 0) => TensorCoreGeneration::V4, // Hopper H100
                _ => return Err(ScirsMlError::UnsupportedOperation(
                    format!("Tensor Cores not supported on compute capability {}.{}", major, minor)
                )),
            };

            Ok(Self::get_capability_for_generation(generation))
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Return minimal capabilities for CPU fallback
            Ok(TensorCoreCapability {
                generation: TensorCoreGeneration::V1,
                supported_operations: vec![TensorCoreOperation::GEMM],
                supported_precisions: vec![TensorCorePrecision::FP16],
                max_dimensions: (16, 16, 16),
                warp_size: 32,
                cores_per_sm: 0,
                peak_throughput: 0.0,
            })
        }
    }

    /// Gets capability information for a specific Tensor Core generation
    fn get_capability_for_generation(generation: TensorCoreGeneration) -> TensorCoreCapability {
        match generation {
            TensorCoreGeneration::V1 => TensorCoreCapability {
                generation,
                supported_operations: vec![
                    TensorCoreOperation::GEMM,
                    TensorCoreOperation::BatchedGEMM,
                    TensorCoreOperation::Convolution,
                ],
                supported_precisions: vec![
                    TensorCorePrecision::FP16,
                    TensorCorePrecision::Mixed16_32,
                ],
                max_dimensions: (256, 256, 256),
                warp_size: 32,
                cores_per_sm: 8,
                peak_throughput: 125.0, // TOPS for V100
            },
            TensorCoreGeneration::V2 => TensorCoreCapability {
                generation,
                supported_operations: vec![
                    TensorCoreOperation::GEMM,
                    TensorCoreOperation::BatchedGEMM,
                    TensorCoreOperation::Convolution,
                    TensorCoreOperation::FusedGEMM,
                ],
                supported_precisions: vec![
                    TensorCorePrecision::FP16,
                    TensorCorePrecision::Mixed16_32,
                    TensorCorePrecision::INT8,
                ],
                max_dimensions: (256, 256, 256),
                warp_size: 32,
                cores_per_sm: 8,
                peak_throughput: 130.0, // TOPS for RTX 2080 Ti
            },
            TensorCoreGeneration::V3 => TensorCoreCapability {
                generation,
                supported_operations: vec![
                    TensorCoreOperation::GEMM,
                    TensorCoreOperation::BatchedGEMM,
                    TensorCoreOperation::Convolution,
                    TensorCoreOperation::FusedGEMM,
                    TensorCoreOperation::SparseGEMM,
                ],
                supported_precisions: vec![
                    TensorCorePrecision::FP16,
                    TensorCorePrecision::Mixed16_32,
                    TensorCorePrecision::BF16,
                    TensorCorePrecision::TF32,
                    TensorCorePrecision::INT8,
                ],
                max_dimensions: (512, 512, 512),
                warp_size: 32,
                cores_per_sm: 4,
                peak_throughput: 312.0, // TOPS for A100
            },
            TensorCoreGeneration::V4 => TensorCoreCapability {
                generation,
                supported_operations: vec![
                    TensorCoreOperation::GEMM,
                    TensorCoreOperation::BatchedGEMM,
                    TensorCoreOperation::Convolution,
                    TensorCoreOperation::FusedGEMM,
                    TensorCoreOperation::SparseGEMM,
                    TensorCoreOperation::AttentionQKV,
                    TensorCoreOperation::ElementWise,
                ],
                supported_precisions: vec![
                    TensorCorePrecision::FP16,
                    TensorCorePrecision::Mixed16_32,
                    TensorCorePrecision::BF16,
                    TensorCorePrecision::TF32,
                    TensorCorePrecision::FP8,
                    TensorCorePrecision::INT8,
                    TensorCorePrecision::INT4,
                ],
                max_dimensions: (1024, 1024, 1024),
                warp_size: 32,
                cores_per_sm: 4,
                peak_throughput: 1000.0, // TOPS for H100
            },
        }
    }

    /// Prepares a Tensor Core operation for execution
    pub fn prepare_operation(&self, config: TensorCoreConfig) -> Result<String> {
        // Validate operation is supported
        if !self.capabilities.supported_operations.contains(&config.operation) {
            return Err(ScirsMlError::UnsupportedOperation(
                format!("Operation {:?} not supported on {:?}", config.operation, self.capabilities.generation)
            ));
        }

        // Validate precision is supported
        if !self.capabilities.supported_precisions.contains(&config.precision) {
            return Err(ScirsMlError::UnsupportedOperation(
                format!("Precision {:?} not supported on {:?}", config.precision, self.capabilities.generation)
            ));
        }

        // Validate dimensions
        let (m, n, k) = config.dimensions;
        let (max_m, max_n, max_k) = self.capabilities.max_dimensions;
        if m > max_m || n > max_n || k > max_k {
            return Err(ScirsMlError::InvalidArgument(
                format!("Matrix dimensions ({}, {}, {}) exceed maximum ({}, {}, {})",
                    m, n, k, max_m, max_n, max_k)
            ));
        }

        // Generate descriptor key
        let descriptor_key = format!("{:?}_{:?}_{}x{}x{}",
            config.operation, config.precision, m, n, k);

        // Check if descriptor already exists
        {
            let descriptors = self.descriptors.lock().unwrap();
            if descriptors.contains_key(&descriptor_key) {
                return Ok(descriptor_key);
            }
        }

        // Generate and compile kernel
        let descriptor = self.generate_kernel_descriptor(config)?;

        // Cache the descriptor
        {
            let mut descriptors = self.descriptors.lock().unwrap();
            descriptors.insert(descriptor_key.clone(), descriptor);
        }

        Ok(descriptor_key)
    }

    /// Generates CUDA kernel descriptor for the operation
    fn generate_kernel_descriptor(&self, config: TensorCoreConfig) -> Result<TensorCoreDescriptor> {
        let kernel_source = self.generate_kernel_source(&config)?;
        let (grid_dims, block_dims, shared_mem) = self.calculate_launch_parameters(&config);

        #[cfg(feature = "cuda")]
        let kernel = {
            let module = self.device.load_ptx(
                kernel_source.as_bytes(),
                "tensor_core_kernel",
                &[]
            )?;
            Some(module.get_func("tensor_core_kernel")?)
        };

        Ok(TensorCoreDescriptor {
            config,
            #[cfg(feature = "cuda")]
            kernel,
            kernel_source,
            grid_dims,
            block_dims,
            shared_mem_bytes: shared_mem,
        })
    }

    /// Generates optimized CUDA kernel source code
    fn generate_kernel_source(&self, config: &TensorCoreConfig) -> Result<String> {
        let mut source = String::new();

        // Add headers and includes
        source.push_str("#include <cuda_runtime.h>\n");
        source.push_str("#include <mma.h>\n");
        source.push_str("using namespace nvcuda;\n\n");

        // Add precision-specific types
        match config.precision {
            TensorCorePrecision::FP16 => {
                source.push_str("using InputType = half;\n");
                source.push_str("using AccumType = half;\n");
            },
            TensorCorePrecision::Mixed16_32 => {
                source.push_str("using InputType = half;\n");
                source.push_str("using AccumType = float;\n");
            },
            TensorCorePrecision::BF16 => {
                source.push_str("using InputType = __nv_bfloat16;\n");
                source.push_str("using AccumType = float;\n");
            },
            TensorCorePrecision::TF32 => {
                source.push_str("using InputType = float;\n");
                source.push_str("using AccumType = float;\n");
            },
            _ => return Err(ScirsMlError::UnsupportedOperation(
                format!("Precision {:?} kernel generation not implemented", config.precision)
            )),
        }

        // Generate operation-specific kernel
        match config.operation {
            TensorCoreOperation::GEMM => {
                source.push_str(&self.generate_gemm_kernel(config)?);
            },
            TensorCoreOperation::BatchedGEMM => {
                source.push_str(&self.generate_batched_gemm_kernel(config)?);
            },
            TensorCoreOperation::FusedGEMM => {
                source.push_str(&self.generate_fused_gemm_kernel(config)?);
            },
            _ => return Err(ScirsMlError::UnsupportedOperation(
                format!("Kernel generation for {:?} not implemented", config.operation)
            )),
        }

        Ok(source)
    }

    /// Generates GEMM kernel source
    fn generate_gemm_kernel(&self, config: &TensorCoreConfig) -> Result<String> {
        let (m, n, k) = config.dimensions;
        let wmma_m = 16; // Standard Tensor Core tile size
        let wmma_n = 16;
        let wmma_k = 16;

        let source = format!(r#"
extern "C" __global__ void tensor_core_kernel(
    const InputType* A, const InputType* B, AccumType* C,
    int M, int N, int K, AccumType alpha, AccumType beta) {{

    // Warp and lane IDs
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Block indices
    int block_row = blockIdx.y * blockDim.y;
    int block_col = blockIdx.x * blockDim.x;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, {wmma_m}, {wmma_n}, {wmma_k}, InputType, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, {wmma_m}, {wmma_n}, {wmma_k}, InputType, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, {wmma_m}, {wmma_n}, {wmma_k}, AccumType> c_frag;

    // Initialize accumulator fragment
    wmma::fill_fragment(c_frag, 0.0f);

    // Perform matrix multiplication
    for (int k_step = 0; k_step < K; k_step += {wmma_k}) {{
        // Load fragments
        wmma::load_matrix_sync(a_frag, A + (block_row * K + k_step), K);
        wmma::load_matrix_sync(b_frag, B + (k_step * N + block_col), N);

        // Perform matrix multiply-accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }}

    // Scale and store result
    for (int i = 0; i < c_frag.num_elements; i++) {{
        c_frag.x[i] = alpha * c_frag.x[i] + beta * c_frag.x[i];
    }}

    // Store result
    wmma::store_matrix_sync(C + (block_row * N + block_col), c_frag, N, wmma::mem_row_major);
}}
"#, wmma_m = wmma_m, wmma_n = wmma_n, wmma_k = wmma_k);

        Ok(source)
    }

    /// Generates batched GEMM kernel source
    fn generate_batched_gemm_kernel(&self, config: &TensorCoreConfig) -> Result<String> {
        let batch_size = config.batch_size.unwrap_or(1);
        let gemm_kernel = self.generate_gemm_kernel(config)?;

        let source = format!(r#"
{gemm_kernel}

extern "C" __global__ void batched_tensor_core_kernel(
    const InputType** A_array, const InputType** B_array, AccumType** C_array,
    int M, int N, int K, AccumType alpha, AccumType beta, int batch_count) {{

    int batch_id = blockIdx.z;
    if (batch_id >= batch_count) return;

    // Call single GEMM for this batch
    tensor_core_kernel(A_array[batch_id], B_array[batch_id], C_array[batch_id],
                      M, N, K, alpha, beta);
}}
"#, gemm_kernel = gemm_kernel);

        Ok(source)
    }

    /// Generates fused GEMM kernel with activation
    fn generate_fused_gemm_kernel(&self, config: &TensorCoreConfig) -> Result<String> {
        let gemm_kernel = self.generate_gemm_kernel(config)?;

        let source = format!(r#"
{gemm_kernel}

__device__ __forceinline__ AccumType relu_activation(AccumType x) {{
    return fmaxf(x, 0.0f);
}}

__device__ __forceinline__ AccumType gelu_activation(AccumType x) {{
    return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}}

extern "C" __global__ void fused_tensor_core_kernel(
    const InputType* A, const InputType* B, AccumType* C,
    int M, int N, int K, AccumType alpha, AccumType beta, int activation_type) {{

    // Perform standard GEMM first
    tensor_core_kernel(A, B, C, M, N, K, alpha, beta);

    // Apply activation function
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * N;

    if (tid < total_elements) {{
        switch (activation_type) {{
            case 1: // ReLU
                C[tid] = relu_activation(C[tid]);
                break;
            case 2: // GELU
                C[tid] = gelu_activation(C[tid]);
                break;
            default:
                // No activation
                break;
        }}
    }}
}}
"#, gemm_kernel = gemm_kernel);

        Ok(source)
    }

    /// Calculates optimal launch parameters for the kernel
    fn calculate_launch_parameters(&self, config: &TensorCoreConfig) -> ((u32, u32, u32), (u32, u32, u32), u32) {
        let (m, n, _k) = config.dimensions;
        let wmma_tile = 16;

        // Calculate grid dimensions
        let grid_x = (n + wmma_tile - 1) / wmma_tile;
        let grid_y = (m + wmma_tile - 1) / wmma_tile;
        let grid_z = config.batch_size.unwrap_or(1);

        // Block dimensions (one warp per block for Tensor Cores)
        let block_x = 32;
        let block_y = 1;
        let block_z = 1;

        // Shared memory requirements
        let shared_mem = 0; // Tensor Cores use registers primarily

        (
            (grid_x as u32, grid_y as u32, grid_z as u32),
            (block_x, block_y, block_z),
            shared_mem
        )
    }

    /// Executes a prepared Tensor Core operation
    pub fn execute_operation(&self, descriptor_key: &str, a_ptr: *const f32, b_ptr: *const f32, c_ptr: *mut f32) -> Result<()> {
        let descriptor = {
            let descriptors = self.descriptors.lock().unwrap();
            descriptors.get(descriptor_key)
                .ok_or_else(|| ScirsMlError::InvalidArgument("Operation descriptor not found".into()))?
                .clone()
        };

        #[cfg(feature = "cuda")]
        {
            if let Some(kernel) = &descriptor.kernel {
                let start_time = std::time::Instant::now();

                // Launch kernel
                let result = unsafe {
                    kernel.launch(
                        descriptor.grid_dims,
                        descriptor.block_dims,
                        descriptor.shared_mem_bytes,
                        &[&a_ptr, &b_ptr, &c_ptr,
                          &(descriptor.config.dimensions.0 as i32),
                          &(descriptor.config.dimensions.1 as i32),
                          &(descriptor.config.dimensions.2 as i32),
                          &descriptor.config.alpha,
                          &descriptor.config.beta]
                    )
                };

                let elapsed = start_time.elapsed();
                self.record_operation_metrics(&descriptor.config.operation, elapsed.as_secs_f64() * 1000.0);

                result?;
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // CPU fallback implementation
            self.execute_cpu_fallback(&descriptor.config, a_ptr, b_ptr, c_ptr)?;
        }

        Ok(())
    }

    /// CPU fallback for Tensor Core operations
    fn execute_cpu_fallback(&self, config: &TensorCoreConfig, a_ptr: *const f32, b_ptr: *const f32, c_ptr: *mut f32) -> Result<()> {
        let (m, n, k) = config.dimensions;

        // Simple CPU GEMM implementation
        unsafe {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        sum += *a_ptr.add(i * k + l) * *b_ptr.add(l * n + j);
                    }
                    let c_idx = i * n + j;
                    *c_ptr.add(c_idx) = config.alpha * sum + config.beta * *c_ptr.add(c_idx);
                }
            }
        }

        Ok(())
    }

    /// Records performance metrics for an operation
    fn record_operation_metrics(&self, operation: &TensorCoreOperation, execution_time_ms: f64) {
        let mut metrics = self.metrics.lock().unwrap();

        metrics.total_operations += 1;
        metrics.total_time_ms += execution_time_ms;
        metrics.avg_operation_time_ms = metrics.total_time_ms / metrics.total_operations as f64;

        // Update per-operation metrics
        let op_metrics = metrics.operation_metrics.entry(operation.clone())
            .or_insert_with(OperationMetrics::default);

        op_metrics.execution_count += 1;
        op_metrics.total_time_ms += execution_time_ms;

        if op_metrics.best_time_ms == 0.0 || execution_time_ms < op_metrics.best_time_ms {
            op_metrics.best_time_ms = execution_time_ms;
        }
    }

    /// Gets current performance metrics
    pub fn get_metrics(&self) -> TensorCoreMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Gets Tensor Core capabilities
    pub fn get_capabilities(&self) -> &TensorCoreCapability {
        &self.capabilities
    }

    /// Checks if an operation is supported
    pub fn is_operation_supported(&self, operation: &TensorCoreOperation, precision: &TensorCorePrecision) -> bool {
        self.capabilities.supported_operations.contains(operation) &&
        self.capabilities.supported_precisions.contains(precision)
    }

    /// Gets optimal configuration for given matrix dimensions
    pub fn get_optimal_config(&self, m: usize, n: usize, k: usize) -> TensorCoreConfig {
        // Choose best precision based on capabilities
        let precision = if self.capabilities.supported_precisions.contains(&TensorCorePrecision::TF32) {
            TensorCorePrecision::TF32
        } else if self.capabilities.supported_precisions.contains(&TensorCorePrecision::Mixed16_32) {
            TensorCorePrecision::Mixed16_32
        } else {
            TensorCorePrecision::FP16
        };

        TensorCoreConfig {
            operation: TensorCoreOperation::GEMM,
            precision,
            dimensions: (m, n, k),
            batch_size: None,
            layout_a: MatrixLayout::RowMajor,
            layout_b: MatrixLayout::ColumnMajor,
            layout_c: MatrixLayout::RowMajor,
            alpha: 1.0,
            beta: 0.0,
            use_amp: true,
            enable_fusion: true,
        }
    }

    /// Generates a performance report
    pub fn generate_report(&self) -> TensorCoreReport {
        let metrics = self.get_metrics();

        TensorCoreReport {
            capabilities: self.capabilities.clone(),
            metrics,
            utilization_analysis: self.analyze_utilization(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Analyzes Tensor Core utilization patterns
    fn analyze_utilization(&self) -> String {
        let metrics = self.metrics.lock().unwrap();

        if metrics.total_operations == 0 {
            return "No operations executed".to_string();
        }

        let efficiency = if metrics.avg_operation_time_ms > 0.0 {
            // Rough efficiency calculation
            (1.0 / metrics.avg_operation_time_ms).min(1.0) * 100.0
        } else {
            0.0
        };

        format!("Average efficiency: {:.1}%, {} operations executed",
                efficiency, metrics.total_operations)
    }

    /// Generates optimization recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        let metrics = self.metrics.lock().unwrap();

        if metrics.total_operations == 0 {
            recommendations.push("No Tensor Core operations detected - consider using matrix operations that benefit from Tensor Cores".to_string());
            return recommendations;
        }

        // Check average operation time
        if metrics.avg_operation_time_ms > 10.0 {
            recommendations.push("High average operation time - consider optimizing matrix dimensions or reducing precision".to_string());
        }

        // Check for underutilized capabilities
        if self.capabilities.generation >= TensorCoreGeneration::V3 &&
           !metrics.operation_metrics.contains_key(&TensorCoreOperation::SparseGEMM) {
            recommendations.push("Consider using sparse operations available on Ampere+ architecture".to_string());
        }

        // Check precision usage
        let has_mixed_precision = metrics.operation_metrics.keys()
            .any(|_| true); // In real implementation, would track precision usage

        if !has_mixed_precision && self.capabilities.supported_precisions.contains(&TensorCorePrecision::Mixed16_32) {
            recommendations.push("Consider using mixed-precision training for better performance".to_string());
        }

        recommendations
    }
}

/// Comprehensive Tensor Core report
#[derive(Debug, Clone)]
pub struct TensorCoreReport {
    /// GPU capabilities
    pub capabilities: TensorCoreCapability,
    /// Performance metrics
    pub metrics: TensorCoreMetrics,
    /// Utilization analysis
    pub utilization_analysis: String,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
}

impl TensorCoreReport {
    /// Formats the report as human-readable text
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== NVIDIA Tensor Core Report ===\n\n");

        // Capabilities
        report.push_str(&format!("GPU Generation: {:?}\n", self.capabilities.generation));
        report.push_str(&format!("Peak Throughput: {:.1} TOPS\n", self.capabilities.peak_throughput));
        report.push_str(&format!("Cores per SM: {}\n", self.capabilities.cores_per_sm));
        report.push_str("\n");

        // Performance metrics
        report.push_str("Performance Metrics:\n");
        report.push_str(&format!("  Total Operations: {}\n", self.metrics.total_operations));
        report.push_str(&format!("  Average Time: {:.3} ms\n", self.metrics.avg_operation_time_ms));
        report.push_str(&format!("  Peak Throughput: {:.1} TOPS\n", self.metrics.peak_throughput_tops));
        report.push_str(&format!("  Average Throughput: {:.1} TOPS\n", self.metrics.avg_throughput_tops));
        report.push_str("\n");

        // Utilization
        report.push_str(&format!("Utilization: {}\n\n", self.utilization_analysis));

        // Recommendations
        if !self.recommendations.is_empty() {
            report.push_str("Recommendations:\n");
            for rec in &self.recommendations {
                report.push_str(&format!("  • {}\n", rec));
            }
        }

        report
    }
}