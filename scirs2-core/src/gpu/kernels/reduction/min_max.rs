//! Min and Max reduction kernels
//!
//! Computes the minimum and maximum values in an array.

use std::collections::HashMap;

use crate::gpu::kernels::{
    BaseKernel, DataType, GpuKernel, KernelMetadata, KernelParams, OperationType,
};
use crate::gpu::{GpuBackend, GpuError};

/// Min reduction kernel
pub struct MinKernel {
    base: BaseKernel,
}

impl MinKernel {
    /// Create a new min reduction kernel
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 1024, // 256 * sizeof(float)
            supports_tensor_cores: false,
            operation_type: OperationType::Balanced,
            backend_metadata: HashMap::new(),
        };

        let (cuda_source, rocm_source, wgpu_source, metal_source, opencl_source) =
            Self::get_kernel_sources();

        Self {
            base: BaseKernel::new(
                "min_reduce",
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
        // CUDA kernel
        let cuda_source = r#"
extern "C" __global__ void min_reduce(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float sdata[256];

    // Each block loads data into shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Initialize with first element or +infinity
    if (i < n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = INFINITY;
    }

    // Load and compare second element
    if (i + blockDim.x < n) {
        sdata[tid] = fminf(sdata[tid], input[i + blockDim.x]);
    }

    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to output
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
"#
        .to_string();

        // WebGPU kernel
        let wgpu_source = r#"
struct Uniforms {
    n: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, write> output: array<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn min_reduce(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let i = workgroup_id.x * 256u * 2u + local_id.x;

    // Initialize with first element or +infinity
    if (i < uniforms.n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = 3.4028235e+38; // f32::INFINITY
    }

    // Load and compare second element
    if (i + 256u < uniforms.n) {
        sdata[tid] = min(sdata[tid], input[i + 256u]);
    }

    workgroupBarrier();

    // Do reduction in shared memory
    var s = 256u / 2u;
    for (var j = 0u; s > 0u; j = j + 1u) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }

        s = s / 2u;
        workgroupBarrier();
    }

    // Write result for this workgroup
    if (tid == 0u) {
        output[workgroup_id.x] = sdata[0];
    }
}
"#
        .to_string();

        // Metal kernel
        let metal_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void min_reduce(
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

    // Initialize with first element or +infinity
    if (i < n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = INFINITY;
    }

    // Load and compare second element
    if (i + 256 < n) {
        sdata[tid] = min(sdata[tid], input[i + 256]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Do reduction in shared memory
    for (uint s = 256 / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result for this threadgroup
    if (tid == 0) {
        output[group_id] = sdata[0];
    }
}
"#
        .to_string();

        // OpenCL kernel
        let opencl_source = r#"
__kernel void min_reduce(
    __global const float* input,
    __global float* output,
    const int n)
{
    __local float sdata[256];

    unsigned int tid = get_local_id(0);
    unsigned int i = get_group_id(0) * get_local_size(0) * 2 + get_local_id(0);

    // Initialize with first element or +infinity
    if (i < n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = INFINITY;
    }

    // Load and compare second element
    if (i + get_local_size(0) < n) {
        sdata[tid] = min(sdata[tid], input[i + get_local_size(0)]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Do reduction in shared memory
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result for this workgroup
    if (tid == 0) {
        output[get_group_id(0)] = sdata[0];
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

impl Default for MinKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for MinKernel {
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
            DataType::Float32 | DataType::Float64 | DataType::Int32 | DataType::UInt32
        )
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }

        Ok(Box::new(Self::new()))
    }
}

/// Max reduction kernel
pub struct MaxKernel {
    base: BaseKernel,
}

impl MaxKernel {
    /// Create a new max reduction kernel
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 1024, // 256 * sizeof(float)
            supports_tensor_cores: false,
            operation_type: OperationType::Balanced,
            backend_metadata: HashMap::new(),
        };

        let (cuda_source, rocm_source, wgpu_source, metal_source, opencl_source) =
            Self::get_kernel_sources();

        Self {
            base: BaseKernel::new(
                "max_reduce",
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
        // CUDA kernel
        let cuda_source = r#"
extern "C" __global__ void max_reduce(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float sdata[256];

    // Each block loads data into shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Initialize with first element or -infinity
    if (i < n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = -INFINITY;
    }

    // Load and compare second element
    if (i + blockDim.x < n) {
        sdata[tid] = fmaxf(sdata[tid], input[i + blockDim.x]);
    }

    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to output
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
"#
        .to_string();

        // WebGPU kernel
        let wgpu_source = r#"
struct Uniforms {
    n: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, write> output: array<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn max_reduce(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let i = workgroup_id.x * 256u * 2u + local_id.x;

    // Initialize with first element or -infinity
    if (i < uniforms.n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = -3.4028235e+38; // f32::NEG_INFINITY
    }

    // Load and compare second element
    if (i + 256u < uniforms.n) {
        sdata[tid] = max(sdata[tid], input[i + 256u]);
    }

    workgroupBarrier();

    // Do reduction in shared memory
    var s = 256u / 2u;
    for (var j = 0u; s > 0u; j = j + 1u) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }

        s = s / 2u;
        workgroupBarrier();
    }

    // Write result for this workgroup
    if (tid == 0u) {
        output[workgroup_id.x] = sdata[0];
    }
}
"#
        .to_string();

        // Metal kernel
        let metal_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void max_reduce(
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

    // Initialize with first element or -infinity
    if (i < n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = -INFINITY;
    }

    // Load and compare second element
    if (i + 256 < n) {
        sdata[tid] = max(sdata[tid], input[i + 256]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Do reduction in shared memory
    for (uint s = 256 / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result for this threadgroup
    if (tid == 0) {
        output[group_id] = sdata[0];
    }
}
"#
        .to_string();

        // OpenCL kernel
        let opencl_source = r#"
__kernel void max_reduce(
    __global const float* input,
    __global float* output,
    const int n)
{
    __local float sdata[256];

    unsigned int tid = get_local_id(0);
    unsigned int i = get_group_id(0) * get_local_size(0) * 2 + get_local_id(0);

    // Initialize with first element or -infinity
    if (i < n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = -INFINITY;
    }

    // Load and compare second element
    if (i + get_local_size(0) < n) {
        sdata[tid] = max(sdata[tid], input[i + get_local_size(0)]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Do reduction in shared memory
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result for this workgroup
    if (tid == 0) {
        output[get_group_id(0)] = sdata[0];
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

impl Default for MaxKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for MaxKernel {
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
            DataType::Float32 | DataType::Float64 | DataType::Int32 | DataType::UInt32
        )
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }

        Ok(Box::new(Self::new()))
    }
}
