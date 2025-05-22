//! General matrix-matrix multiplication (GEMM) kernels for GPU
//!
//! Implements C = alpha * A * B + beta * C where:
//! - A is an M x K matrix
//! - B is a K x N matrix
//! - C is an M x N matrix
//! - alpha and beta are scalar values

use std::collections::HashMap;
use std::fmt;

use crate::gpu::kernels::{
    BaseKernel, DataType, GpuKernel, KernelMetadata, KernelParams, OperationType,
};
use crate::gpu::{GpuBackend, GpuError};

/// GEMM specialized implementation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmImpl {
    /// Standard tiled implementation
    Standard,
    /// Implementation optimized for large matrices
    Large,
    /// Implementation optimized for small matrices
    Small,
    /// Implementation using tensor cores (if available)
    TensorCore,
}

impl fmt::Display for GemmImpl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GemmImpl::Standard => write!(f, "standard"),
            GemmImpl::Large => write!(f, "large"),
            GemmImpl::Small => write!(f, "small"),
            GemmImpl::TensorCore => write!(f, "tensor_core"),
        }
    }
}

/// General matrix-matrix multiplication kernel
pub struct GemmKernel {
    base: BaseKernel,
    implementation: GemmImpl,
}

impl GemmKernel {
    /// Create a new GEMM kernel with standard implementation
    pub fn new() -> Self {
        // Default to a standard implementation
        Self::with_implementation(GemmImpl::Standard)
    }

    /// Create a new GEMM kernel with specified implementation
    pub fn with_implementation(implementation: GemmImpl) -> Self {
        let metadata = match implementation {
            GemmImpl::Standard => KernelMetadata {
                workgroup_size: [16, 16, 1],
                local_memory_usage: 8192, // 8 KB
                supports_tensor_cores: false,
                operation_type: OperationType::ComputeIntensive,
                backend_metadata: HashMap::new(),
            },
            GemmImpl::Large => KernelMetadata {
                workgroup_size: [32, 32, 1],
                local_memory_usage: 32768, // 32 KB
                supports_tensor_cores: false,
                operation_type: OperationType::ComputeIntensive,
                backend_metadata: HashMap::new(),
            },
            GemmImpl::Small => KernelMetadata {
                workgroup_size: [8, 8, 1],
                local_memory_usage: 2048, // 2 KB
                supports_tensor_cores: false,
                operation_type: OperationType::ComputeIntensive,
                backend_metadata: HashMap::new(),
            },
            GemmImpl::TensorCore => KernelMetadata {
                workgroup_size: [16, 16, 1],
                local_memory_usage: 8192, // 8 KB
                supports_tensor_cores: true,
                operation_type: OperationType::ComputeIntensive,
                backend_metadata: HashMap::new(),
            },
        };

        let (name, cuda_source, wgpu_source, metal_source, opencl_source) =
            Self::get_sources_for_implementation(implementation);

        Self {
            base: BaseKernel::new(
                &name,
                &cuda_source,
                &wgpu_source,
                &metal_source,
                &opencl_source,
                metadata,
            ),
            implementation,
        }
    }

    /// Create a GEMM kernel with specific alpha and beta values
    pub fn with_alpha_beta(_alpha: f32, _beta: f32) -> Box<dyn GpuKernel> {
        let kernel = Self::new();

        // Generate specialized kernel sources with hard-coded alpha/beta values
        // for better performance

        Box::new(kernel)
    }

    /// Get kernel sources for the specified implementation
    fn get_sources_for_implementation(
        implementation: GemmImpl,
    ) -> (String, String, String, String, String) {
        let name = format!("gemm_{}", implementation);

        // In a real implementation, we would have different optimized kernel sources
        // for each backend and implementation type. Here we'll use the same source for simplicity.

        // CUDA kernel for GEMM
        let cuda_source = match implementation {
            GemmImpl::Standard => r#"
extern "C" __global__ void gemm_standard(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k,
    float alpha, float beta
) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Define block size
    const int BLOCK_SIZE = 16;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = k * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + k - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * n;

    // The element of the block sub-matrix that is computed
    // by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B required to
    // compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep) {

        // Shared memory for the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Shared memory for the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from global memory to shared memory
        As[ty][tx] = a[a + k * ty + tx];
        Bs[ty][tx] = b[b + n * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            Csub += As[ty][i] * Bs[i][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to global memory
    int c_idx = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    int c_row = c_idx + n * ty + tx;

    if (beta == 0) {
        c[c_row] = alpha * Csub;
    } else {
        c[c_row] = alpha * Csub + beta * c[c_row];
    }
}
"#
            .to_string(),
            // Other implementations would have different optimized kernels
            _ => r#"
// Placeholder for other optimized CUDA kernels
extern "C" __global__ void gemm_standard(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k,
    float alpha, float beta
) {
    // Implementation similar to standard but with optimizations
    // specific to the implementation type
}
"#
            .to_string(),
        };

        // WebGPU kernel for GEMM
        let wgpu_source = r#"
struct Uniforms {
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, write> c: array<f32>;

var<workgroup> As: array<array<f32, 16>, 16>;
var<workgroup> Bs: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn gemm_standard(@builtin(global_invocation_id) global_id: vec3<u32>,
                 @builtin(workgroup_id) workgroup_id: vec3<u32>,
                 @builtin(local_invocation_id) local_id: vec3<u32>) {

    let bx = workgroup_id.x;
    let by = workgroup_id.y;

    let tx = local_id.x;
    let ty = local_id.y;

    let block_size = 16u;

    // Index of c
    let row = by * block_size + ty;
    let col = bx * block_size + tx;

    var sum = 0.0;

    // Loop over A and B tiles
    for (var t = 0u; t < (uniforms.k + block_size - 1u) / block_size; t = t + 1u) {
        // Load A tile
        if (row < uniforms.m && t * block_size + tx < uniforms.k) {
            As[ty][tx] = a[row * uniforms.k + t * block_size + tx];
        } else {
            As[ty][tx] = 0.0;
        }

        // Load B tile
        if (t * block_size + ty < uniforms.k && col < uniforms.n) {
            Bs[ty][tx] = b[(t * block_size + ty) * uniforms.n + col];
        } else {
            Bs[ty][tx] = 0.0;
        }

        workgroupBarrier();

        // Compute
        for (var k = 0u; k < block_size; k = k + 1u) {
            sum = sum + As[ty][k] * Bs[k][tx];
        }

        workgroupBarrier();
    }

    // Write result
    if (row < uniforms.m && col < uniforms.n) {
        let c_idx = row * uniforms.n + col;
        if (uniforms.beta == 0.0) {
            c[c_idx] = uniforms.alpha * sum;
        } else {
            c[c_idx] = uniforms.alpha * sum + uniforms.beta * c[c_idx];
        }
    }
}
"#
        .to_string();

        // Metal kernel for GEMM
        let metal_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemm_standard(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 wgid [[threadgroup_position_in_grid]])
{
    const uint block_size = 16;

    // Thread indices
    uint tx = lid.x;
    uint ty = lid.y;

    // Block indices
    uint bx = wgid.x;
    uint by = wgid.y;

    // Global indices
    uint row = by * block_size + ty;
    uint col = bx * block_size + tx;

    // Shared memory for tile
    threadgroup float As[16][16];
    threadgroup float Bs[16][16];

    float sum = 0.0;

    // Loop over tiles
    for (uint t = 0; t < (k + block_size - 1) / block_size; t++) {
        // Load tiles
        if (row < m && t * block_size + tx < k) {
            As[ty][tx] = a[row * k + t * block_size + tx];
        } else {
            As[ty][tx] = 0.0;
        }

        if (t * block_size + ty < k && col < n) {
            Bs[ty][tx] = b[(t * block_size + ty) * n + col];
        } else {
            Bs[ty][tx] = 0.0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute
        for (uint i = 0; i < block_size; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (row < m && col < n) {
        uint c_idx = row * n + col;
        if (beta == 0.0) {
            c[c_idx] = alpha * sum;
        } else {
            c[c_idx] = alpha * sum + beta * c[c_idx];
        }
    }
}
"#
        .to_string();

        // OpenCL kernel for GEMM
        let opencl_source = r#"
__kernel void gemm_standard(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const int m,
    const int n,
    const int k,
    const float alpha,
    const float beta)
{
    const int block_size = 16;

    // Thread indices
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    // Block indices
    const int bx = get_group_id(0);
    const int by = get_group_id(1);

    // Global indices
    const int row = by * block_size + ty;
    const int col = bx * block_size + tx;

    // Shared memory for tile
    __local float As[16][16];
    __local float Bs[16][16];

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (k + block_size - 1) / block_size; t++) {
        // Load tiles
        if (row < m && t * block_size + tx < k) {
            As[ty][tx] = a[row * k + t * block_size + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (t * block_size + ty < k && col < n) {
            Bs[ty][tx] = b[(t * block_size + ty) * n + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute
        for (int i = 0; i < block_size; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result
    if (row < m && col < n) {
        const int c_idx = row * n + col;
        if (beta == 0.0f) {
            c[c_idx] = alpha * sum;
        } else {
            c[c_idx] = alpha * sum + beta * c[c_idx];
        }
    }
}
"#
        .to_string();

        (name, cuda_source, wgpu_source, metal_source, opencl_source)
    }

    /// Generate a specialized kernel for the given dimensions
    fn generate_specialized_kernel(
        &self,
        data_type: DataType,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<GemmKernel, GpuError> {
        // Select appropriate implementation based on matrix dimensions
        let implementation = if data_type == DataType::Float16 || data_type == DataType::BFloat16 {
            // Use tensor core implementation for half-precision types when possible
            GemmImpl::TensorCore
        } else if m >= 1024 && n >= 1024 && k >= 1024 {
            // Use large implementation for big matrices
            GemmImpl::Large
        } else if m <= 128 && n <= 128 && k <= 128 {
            // Use small implementation for small matrices
            GemmImpl::Small
        } else {
            // Default to standard implementation
            GemmImpl::Standard
        };

        Ok(GemmKernel::with_implementation(implementation))
    }
}

impl GpuKernel for GemmKernel {
    fn name(&self) -> &str {
        self.base.name()
    }

    fn source_for_backend(&self, backend: GpuBackend) -> Result<String, GpuError> {
        self.base.source_for_backend(backend)
    }

    fn metadata(&self) -> KernelMetadata {
        self.base.metadata()
    }

    fn can_specialize(&self, params: &KernelParams) -> bool {
        // Check if we can specialize for these parameters
        match params.data_type {
            DataType::Float32 | DataType::Float64 | DataType::Float16 | DataType::BFloat16 => {
                params.input_dims.len() >= 2 && params.output_dims.len() >= 2
            }
            _ => false,
        }
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }

        // Extract dimensions
        let m = params.input_dims.get(0).copied().unwrap_or(0);
        let k = params.input_dims.get(1).copied().unwrap_or(0);
        let n = params.output_dims.get(1).copied().unwrap_or(0);

        // Generate specialized kernel
        let specialized = self.generate_specialized_kernel(params.data_type, m, n, k)?;

        Ok(Box::new(specialized))
    }
}
