//! Norm reduction kernels
//!
//! Computes vector norms (L1, L2, etc.).

use std::collections::HashMap;

use crate::gpu::kernels::{
    BaseKernel, DataType, GpuKernel, KernelMetadata, KernelParams, OperationType,
};
use crate::gpu::{GpuBackend, GpuError};

/// Norm type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// L1 norm (sum of absolute values)
    L1,
    /// L2 norm (Euclidean norm, sqrt of sum of squares)
    L2,
    /// Infinity norm (maximum absolute value)
    Inf,
}

/// Norm reduction kernel
pub struct NormKernel {
    base: BaseKernel,
    norm_type: NormType,
}

impl NormKernel {
    /// Create a new norm kernel for L2 norm (default)
    pub fn new() -> Self {
        Self::with_type(NormType::L2)
    }

    /// Create a new norm kernel with the specified norm type
    pub fn with_type(norm_type: NormType) -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 1024, // 256 * sizeof(float)
            supports_tensor_cores: false,
            operation_type: OperationType::Balanced,
            backend_metadata: HashMap::new(),
        };

        let name = match norm_type {
            NormType::L1 => "norm_l1",
            NormType::L2 => "norm_l2",
            NormType::Inf => "norm_inf",
        };

        let (cuda_source, rocm_source, wgpu_source, metal_source, opencl_source) =
            Self::get_kernel_sources(norm_type);

        Self {
            base: BaseKernel::new(
                name,
                &cuda_source,
                &rocm_source,
                &wgpu_source,
                &metal_source,
                &opencl_source,
                metadata,
            ),
            norm_type,
        }
    }

    /// Get kernel sources for different backends and norm types
    fn get_kernel_sources(norm_type: NormType) -> (String, String, String, String, String) {
        match norm_type {
            NormType::L2 => {
                // CUDA kernel for L2 norm
                let cuda_source = r#"
extern "C" __global__ void norm_l2(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float sdata[256];

    // Each block loads data into shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Initialize with identity value
    sdata[tid] = 0.0f;

    // Load and square first element
    if (i < n) {
        sdata[tid] = input[i] * input[i];
    }

    // Load and square second element
    if (i + blockDim.x < n) {
        sdata[tid] += input[i + blockDim.x] * input[i + blockDim.x];
    }

    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
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

                // WebGPU kernel for L2 norm
                let wgpu_source = r#"
struct Uniforms {
    n: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, write> output: array<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn norm_l2(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let i = workgroup_id.x * 256u * 2u + local_id.x;

    // Initialize
    sdata[tid] = 0.0;

    // Load and square first element
    if (i < uniforms.n) {
        sdata[tid] = input[i] * input[i];
    }

    // Load and square second element
    if (i + 256u < uniforms.n) {
        sdata[tid] = sdata[tid] + input[i + 256u] * input[i + 256u];
    }

    workgroupBarrier();

    // Do reduction in shared memory
    var s = 256u / 2u;
    for (var j = 0u; s > 0u; j = j + 1u) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
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

                // Metal kernel for L2 norm
                let metal_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void norm_l2(
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

    // Initialize
    sdata[tid] = 0.0f;

    // Load and square first element
    if (i < n) {
        sdata[tid] = input[i] * input[i];
    }

    // Load and square second element
    if (i + 256 < n) {
        sdata[tid] += input[i + 256] * input[i + 256];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Do reduction in shared memory
    for (uint s = 256 / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
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

                // OpenCL kernel for L2 norm
                let opencl_source = r#"
__kernel void norm_l2(
    __global const float* input,
    __global float* output,
    const int n)
{
    __local float sdata[256];

    unsigned int tid = get_local_id(0);
    unsigned int i = get_group_id(0) * get_local_size(0) * 2 + get_local_id(0);

    // Initialize
    sdata[tid] = 0.0f;

    // Load and square first element
    if (i < n) {
        sdata[tid] = input[i] * input[i];
    }

    // Load and square second element
    if (i + get_local_size(0) < n) {
        sdata[tid] += input[i + get_local_size(0)] * input[i + get_local_size(0)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Do reduction in shared memory
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
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
            // For brevity, we'll just sketch the structure for L1 and Inf norms
            // In a real implementation, these would be fully implemented
            NormType::L1 => {
                // Similar to L2 but using abs(x) instead of x*x
                let placeholder = r#"
// L1 norm kernel (sum of absolute values)
// In a real implementation, this would contain the full kernel code
"#
                .to_string();

                (
                    placeholder.clone(),
                    placeholder.clone(),
                    placeholder.clone(),
                    placeholder.clone(),
                    placeholder,
                )
            }
            NormType::Inf => {
                // Similar to reduce but using max(abs(x)) instead of sum
                let placeholder = r#"
// Infinity norm kernel (maximum absolute value)
// In a real implementation, this would contain the full kernel code
"#
                .to_string();

                (
                    placeholder.clone(),
                    placeholder.clone(),
                    placeholder.clone(),
                    placeholder.clone(),
                    placeholder,
                )
            }
        }
    }
}

impl Default for NormKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for NormKernel {
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

        // Check for norm type in parameters
        if let Some(norm_param) = params.string_params.get("norm_type") {
            let norm_type = match norm_param.as_str() {
                "l1" => NormType::L1,
                "l2" => NormType::L2,
                "inf" => NormType::Inf,
                _ => return Err(GpuError::InvalidParameter("norm_type".to_string())),
            };

            return Ok(Box::new(Self::with_type(norm_type)));
        }

        // Default to same norm type as this kernel
        Ok(Box::new(Self::with_type(self.norm_type)))
    }
}
