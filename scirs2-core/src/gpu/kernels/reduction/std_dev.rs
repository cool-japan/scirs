//! Standard deviation reduction kernel
//!
//! Computes the standard deviation of all elements in an array.

use std::collections::HashMap;

use crate::gpu::kernels::{
    BaseKernel, DataType, GpuKernel, KernelMetadata, KernelParams, OperationType,
};
use crate::gpu::{GpuBackend, GpuError};

/// Standard deviation reduction kernel
pub struct StdDevKernel {
    base: BaseKernel,
}

impl StdDevKernel {
    /// Create a new standard deviation reduction kernel
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 1024, // 256 * sizeof(float)
            supports_tensor_cores: false,
            operation_type: OperationType::ComputeIntensive,
            backend_metadata: HashMap::new(),
        };

        let (cuda_source, rocm_source, wgpu_source, metal_source, opencl_source) =
            Self::get_kernel_sources();

        Self {
            base: BaseKernel::new(
                "std_dev_reduce",
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
        // CUDA kernel for standard deviation - two-pass implementation
        let cuda_source = r#"
// First pass: compute sum
extern "C" __global__ void std_dev_reduce_sum(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    sdata[tid] = 0.0f;

    if (i < n) {
        sdata[tid] = input[i];
    }

    if (i + blockDim.x < n) {
        sdata[tid] += input[i + blockDim.x];
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Second pass: compute sum of squared differences from mean
extern "C" __global__ void std_dev_reduce_variance(
    const float* __restrict__ input,
    float* __restrict__ output,
    float mean,
    int n
) {
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    sdata[tid] = 0.0f;

    if (i < n) {
        float diff = input[i] - mean;
        sdata[tid] = diff * diff;
    }

    if (i + blockDim.x < n) {
        float diff = input[i + blockDim.x] - mean;
        sdata[tid] += diff * diff;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Third pass: finalize standard deviation
extern "C" __global__ void std_dev_reduce_finalize(
    const float* __restrict__ variances,
    float* __restrict__ output,
    int num_blocks,
    int total_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i == 0) {
        float total_variance = 0.0f;
        for (int j = 0; j < num_blocks; j++) {
            total_variance += variances[j];
        }
        
        float variance = total_variance / (float)(total_elements - 1); // Sample variance
        output[0] = sqrtf(variance);
    }
}
"#
        .to_string();

        // WebGPU kernel for standard deviation
        let wgpu_source = r#"
struct Uniforms {
    n: u32,
    total_elements: u32,
    mean: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, write> output: array<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn std_dev_reduce_sum(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let i = workgroup_id.x * 256u * 2u + local_id.x;

    sdata[tid] = 0.0;

    if (i < uniforms.n) {
        sdata[tid] = input[i];
    }

    if (i + 256u < uniforms.n) {
        sdata[tid] = sdata[tid] + input[i + 256u];
    }

    workgroupBarrier();

    var s = 256u / 2u;
    for (var j = 0u; s > 0u; j = j + 1u) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }

        s = s / 2u;
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[workgroup_id.x] = sdata[0];
    }
}

@compute @workgroup_size(256)
fn std_dev_reduce_variance(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let i = workgroup_id.x * 256u * 2u + local_id.x;

    sdata[tid] = 0.0;

    if (i < uniforms.n) {
        let diff = input[i] - uniforms.mean;
        sdata[tid] = diff * diff;
    }

    if (i + 256u < uniforms.n) {
        let diff = input[i + 256u] - uniforms.mean;
        sdata[tid] = sdata[tid] + (diff * diff);
    }

    workgroupBarrier();

    var s = 256u / 2u;
    for (var j = 0u; s > 0u; j = j + 1u) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }

        s = s / 2u;
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[workgroup_id.x] = sdata[0];
    }
}

@compute @workgroup_size(1)
fn std_dev_reduce_finalize(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    if (global_id.x == 0u) {
        var total_variance = 0.0;
        
        for (var i = 0u; i < arrayLength(&output); i = i + 1u) {
            total_variance = total_variance + output[i];
        }
        
        let variance = total_variance / f32(uniforms.total_elements - 1u);
        output[0] = sqrt(variance);
    }
}
"#
        .to_string();

        // Metal kernel for standard deviation
        let metal_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void std_dev_reduce_sum(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint global_id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]])
{
    threadgroup float sdata[256];

    uint tid = local_id;
    uint i = group_id * 256 * 2 + local_id;

    sdata[tid] = 0.0f;

    if (i < n) {
        sdata[tid] = input[i];
    }

    if (i + 256 < n) {
        sdata[tid] += input[i + 256];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 256 / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[group_id] = sdata[0];
    }
}

kernel void std_dev_reduce_variance(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& mean [[buffer(3)]],
    uint global_id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]])
{
    threadgroup float sdata[256];

    uint tid = local_id;
    uint i = group_id * 256 * 2 + local_id;

    sdata[tid] = 0.0f;

    if (i < n) {
        float diff = input[i] - mean;
        sdata[tid] = diff * diff;
    }

    if (i + 256 < n) {
        float diff = input[i + 256] - mean;
        sdata[tid] += diff * diff;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 256 / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[group_id] = sdata[0];
    }
}

kernel void std_dev_reduce_finalize(
    const device float* variances [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& num_blocks [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint global_id [[thread_position_in_grid]])
{
    if (global_id == 0) {
        float total_variance = 0.0f;
        
        for (uint i = 0; i < num_blocks; i++) {
            total_variance += variances[i];
        }
        
        float variance = total_variance / float(total_elements - 1);
        output[0] = sqrt(variance);
    }
}
"#
        .to_string();

        // OpenCL kernel for standard deviation
        let opencl_source = r#"
__kernel void std_dev_reduce_sum(
    __global const float* input,
    __global float* output,
    const int n)
{
    __local float sdata[256];

    unsigned int tid = get_local_id(0);
    unsigned int i = get_group_id(0) * get_local_size(0) * 2 + get_local_id(0);

    sdata[tid] = 0.0f;

    if (i < n) {
        sdata[tid] = input[i];
    }

    if (i + get_local_size(0) < n) {
        sdata[tid] += input[i + get_local_size(0)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        output[get_group_id(0)] = sdata[0];
    }
}

__kernel void std_dev_reduce_variance(
    __global const float* input,
    __global float* output,
    const float mean,
    const int n)
{
    __local float sdata[256];

    unsigned int tid = get_local_id(0);
    unsigned int i = get_group_id(0) * get_local_size(0) * 2 + get_local_id(0);

    sdata[tid] = 0.0f;

    if (i < n) {
        float diff = input[i] - mean;
        sdata[tid] = diff * diff;
    }

    if (i + get_local_size(0) < n) {
        float diff = input[i + get_local_size(0)] - mean;
        sdata[tid] += diff * diff;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        output[get_group_id(0)] = sdata[0];
    }
}

__kernel void std_dev_reduce_finalize(
    __global const float* variances,
    __global float* output,
    const int num_blocks,
    const int total_elements)
{
    int i = get_global_id(0);
    
    if (i == 0) {
        float total_variance = 0.0f;
        
        for (int j = 0; j < num_blocks; j++) {
            total_variance += variances[j];
        }
        
        float variance = total_variance / (float)(total_elements - 1);
        output[0] = sqrt(variance);
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

impl Default for StdDevKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for StdDevKernel {
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
        matches!(params.data_type, DataType::Float32 | DataType::Float64)
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }

        Ok(Box::new(Self::new()))
    }
}
