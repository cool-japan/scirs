//! Softmax activation function kernel
//!
//! Implements the softmax function for neural networks.

use std::collections::HashMap;

use crate::gpu::kernels::{
    BaseKernel, DataType, GpuKernel, KernelMetadata, KernelParams, OperationType,
};
use crate::gpu::{GpuBackend, GpuError};

/// Softmax activation function kernel
pub struct SoftmaxKernel {
    base: BaseKernel,
}

impl Default for SoftmaxKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftmaxKernel {
    /// Create a new softmax kernel
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 2048, // Need extra memory for reductions
            supports_tensor_cores: false,
            operation_type: OperationType::ComputeIntensive,
            backend_metadata: HashMap::new(),
        };

        let (cuda_source, rocm_source, wgpu_source, metal_source, opencl_source) =
            Self::get_kernel_sources();

        Self {
            base: BaseKernel::new(
                "softmax",
                &cuda_source,
                &rocm_source,
                &wgpu_source,
                &metal_source,
                &opencl_source,
                metadata,
            ),
        }
    }

    /// Get kernel sources for different backends
    fn get_kernel_sources() -> (String, String, String, String, String) {
        // CUDA kernel for softmax
        let cuda_source = r#"
// Three-pass softmax implementation for numerical stability

// First pass: find maximum value
extern "C" __global__ void softmax_find_max(
    const float* __restrict__ input,
    float* __restrict__ max_vals,
    int n,
    int batch_size
) {
    __shared__ float sdata[256];
    
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    int i = batch_idx * n + blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Initialize with -infinity
    sdata[tid] = -INFINITY;
    
    // Load and compare first element
    if (blockIdx.x * blockDim.x + threadIdx.x < n) {
        sdata[tid] = input[i];
    }
    
    // Load and compare second element
    if (blockIdx.x * blockDim.x + blockDim.x + threadIdx.x < n) {
        sdata[tid] = fmaxf(sdata[tid], input[i + blockDim.x]);
    }
    
    __syncthreads();
    
    // Reduce to find max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Write partial max
    if (tid == 0) {
        max_vals[batch_idx * gridDim.x + blockIdx.x] = sdata[0];
    }
}

// Second pass: compute sum of exponentials
extern "C" __global__ void softmax_compute_sum(
    const float* __restrict__ input,
    const float* __restrict__ max_val,
    float* __restrict__ sum_vals,
    int n,
    int batch_size
) {
    __shared__ float sdata[256];
    
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    int i = batch_idx * n + blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    sdata[tid] = 0.0f;
    
    // Compute exp(x - max) for first element
    if (blockIdx.x * blockDim.x + threadIdx.x < n) {
        sdata[tid] = expf(input[i] - max_val[batch_idx]);
    }
    
    // Compute exp(x - max) for second element
    if (blockIdx.x * blockDim.x + blockDim.x + threadIdx.x < n) {
        sdata[tid] += expf(input[i + blockDim.x] - max_val[batch_idx]);
    }
    
    __syncthreads();
    
    // Reduce to find sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write partial sum
    if (tid == 0) {
        sum_vals[batch_idx * gridDim.x + blockIdx.x] = sdata[0];
    }
}

// Third pass: compute final softmax values
extern "C" __global__ void softmax_finalize(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ max_val,
    const float* __restrict__ sum_val,
    int n,
    int batch_size
) {
    int batch_idx = blockIdx.y;
    int i = batch_idx * n + blockIdx.x * blockDim.x + threadIdx.x;
    
    if (blockIdx.x * blockDim.x + threadIdx.x < n) {
        output[i] = expf(input[i] - max_val[batch_idx]) / sum_val[batch_idx];
    }
}
"#
        .to_string();

        // WebGPU kernel for softmax
        let wgpu_source = r#"
struct Uniforms {
    n: u32,
    batch_size: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, write> output: array<f32>;
@group(0) @binding(3) var<storage, read_write> max_vals: array<f32>;
@group(0) @binding(4) var<storage, read_write> sum_vals: array<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax_find_max(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let batch_idx = workgroup_id.y;
    let tid = local_id.x;
    let i = batch_idx * uniforms.n + workgroup_id.x * 256u * 2u + local_id.x;
    
    // Initialize
    sdata[tid] = -3.4028235e+38; // f32::NEG_INFINITY
    
    if (workgroup_id.x * 256u + local_id.x < uniforms.n) {
        sdata[tid] = input[i];
    }
    
    if (workgroup_id.x * 256u + 256u + local_id.x < uniforms.n) {
        sdata[tid] = max(sdata[tid], input[i + 256u]);
    }
    
    workgroupBarrier();
    
    // Reduce to find max
    var s = 256u / 2u;
    for (var j = 0u; s > 0u; j = j + 1u) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        s = s / 2u;
        workgroupBarrier();
    }
    
    if (tid == 0u) {
        max_vals[batch_idx * 32u + workgroup_id.x] = sdata[0]; // Assuming max 32 workgroups
    }
}

@compute @workgroup_size(256)
fn softmax_compute_sum(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let batch_idx = workgroup_id.y;
    let tid = local_id.x;
    let i = batch_idx * uniforms.n + workgroup_id.x * 256u * 2u + local_id.x;
    
    sdata[tid] = 0.0;
    
    if (workgroup_id.x * 256u + local_id.x < uniforms.n) {
        sdata[tid] = exp(input[i] - max_vals[batch_idx]);
    }
    
    if (workgroup_id.x * 256u + 256u + local_id.x < uniforms.n) {
        sdata[tid] += exp(input[i + 256u] - max_vals[batch_idx]);
    }
    
    workgroupBarrier();
    
    // Reduce to find sum
    var s = 256u / 2u;
    for (var j = 0u; s > 0u; j = j + 1u) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        s = s / 2u;
        workgroupBarrier();
    }
    
    if (tid == 0u) {
        sum_vals[batch_idx * 32u + workgroup_id.x] = sdata[0];
    }
}

@compute @workgroup_size(256)
fn softmax_finalize(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let batch_idx = workgroup_id.y;
    let i = batch_idx * uniforms.n + workgroup_id.x * 256u + global_id.x % 256u;
    
    if (workgroup_id.x * 256u + global_id.x % 256u < uniforms.n) {
        output[i] = exp(input[i] - max_vals[batch_idx]) / sum_vals[batch_idx];
    }
}
"#
        .to_string();

        // Metal kernel for softmax
        let metal_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void softmax_find_max(
    const device float* input [[buffer(0)]],
    device float* max_vals [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    uint global_id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]])
{
    threadgroup float sdata[256];
    
    uint batch_idx = group_id / 32; // Assuming max 32 groups per batch
    uint tid = local_id;
    uint i = batch_idx * n + (group_id % 32) * 256 * 2 + local_id;
    
    sdata[tid] = -INFINITY;
    
    if ((group_id % 32) * 256 + local_id < n) {
        sdata[tid] = input[i];
    }
    
    if ((group_id % 32) * 256 + 256 + local_id < n) {
        sdata[tid] = max(sdata[tid], input[i + 256]);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 256 / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        max_vals[group_id] = sdata[0];
    }
}

kernel void softmax_compute_sum(
    const device float* input [[buffer(0)]],
    const device float* max_vals [[buffer(1)]],
    device float* sum_vals [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    uint global_id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]])
{
    threadgroup float sdata[256];
    
    uint batch_idx = group_id / 32;
    uint tid = local_id;
    uint i = batch_idx * n + (group_id % 32) * 256 * 2 + local_id;
    
    sdata[tid] = 0.0f;
    
    if ((group_id % 32) * 256 + local_id < n) {
        sdata[tid] = exp(input[i] - max_vals[batch_idx]);
    }
    
    if ((group_id % 32) * 256 + 256 + local_id < n) {
        sdata[tid] += exp(input[i + 256] - max_vals[batch_idx]);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 256 / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        sum_vals[group_id] = sdata[0];
    }
}

kernel void softmax_finalize(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* max_vals [[buffer(2)]],
    const device float* sum_vals [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    uint global_id [[thread_position_in_grid]],
    uint group_id [[threadgroup_position_in_grid]])
{
    uint batch_idx = group_id / 32;
    uint i = batch_idx * n + (group_id % 32) * 256 + global_id % 256;
    
    if ((group_id % 32) * 256 + global_id % 256 < n) {
        output[i] = exp(input[i] - max_vals[batch_idx]) / sum_vals[batch_idx];
    }
}
"#
        .to_string();

        // OpenCL kernel for softmax
        let opencl_source = r#"
__kernel void softmax_find_max(
    __global const float* input,
    __global float* max_vals,
    const int n,
    const int batch_size)
{
    __local float sdata[256];
    
    int batch_idx = get_group_id(1);
    int tid = get_local_id(0);
    int i = batch_idx * n + get_group_id(0) * get_local_size(0) * 2 + get_local_id(0);
    
    sdata[tid] = -INFINITY;
    
    if (get_group_id(0) * get_local_size(0) + get_local_id(0) < n) {
        sdata[tid] = input[i];
    }
    
    if (get_group_id(0) * get_local_size(0) + get_local_size(0) + get_local_id(0) < n) {
        sdata[tid] = max(sdata[tid], input[i + get_local_size(0)]);
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (tid == 0) {
        max_vals[batch_idx * get_num_groups(0) + get_group_id(0)] = sdata[0];
    }
}

__kernel void softmax_compute_sum(
    __global const float* input,
    __global const float* max_vals,
    __global float* sum_vals,
    const int n,
    const int batch_size)
{
    __local float sdata[256];
    
    int batch_idx = get_group_id(1);
    int tid = get_local_id(0);
    int i = batch_idx * n + get_group_id(0) * get_local_size(0) * 2 + get_local_id(0);
    
    sdata[tid] = 0.0f;
    
    if (get_group_id(0) * get_local_size(0) + get_local_id(0) < n) {
        sdata[tid] = exp(input[i] - max_vals[batch_idx]);
    }
    
    if (get_group_id(0) * get_local_size(0) + get_local_size(0) + get_local_id(0) < n) {
        sdata[tid] += exp(input[i + get_local_size(0)] - max_vals[batch_idx]);
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (tid == 0) {
        sum_vals[batch_idx * get_num_groups(0) + get_group_id(0)] = sdata[0];
    }
}

__kernel void softmax_finalize(
    __global const float* input,
    __global float* output,
    __global const float* max_vals,
    __global const float* sum_vals,
    const int n,
    const int batch_size)
{
    int batch_idx = get_group_id(1);
    int i = batch_idx * n + get_group_id(0) * get_local_size(0) + get_local_id(0);
    
    if (get_group_id(0) * get_local_size(0) + get_local_id(0) < n) {
        output[i] = exp(input[i] - max_vals[batch_idx]) / sum_vals[batch_idx];
    }
}
"#
        .to_string();

        // ROCm (HIP) kernel - similar to CUDA
        let rocm_source = cuda_source.clone();

        (
            cuda_source,
            rocm_source,
            wgpu_source,
            metal_source,
            opencl_source,
        )
    }
}

impl GpuKernel for SoftmaxKernel {
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
        matches!(
            params.data_type,
            DataType::Float32 | DataType::Float64 | DataType::Float16 | DataType::BFloat16
        )
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }

        // No specialization needed for Softmax
        Ok(Box::new(Self::new()))
    }
}
