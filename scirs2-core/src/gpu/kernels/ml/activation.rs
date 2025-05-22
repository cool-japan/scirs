//! Activation function kernels for neural networks
//!
//! Implements common activation functions (ReLU, Sigmoid, etc.)

use std::collections::HashMap;

use crate::gpu::kernels::{
    BaseKernel, DataType, GpuKernel, KernelMetadata, KernelParams, OperationType,
};
use crate::gpu::{GpuBackend, GpuError};

/// ReLU activation function kernel
pub struct ReluKernel {
    base: BaseKernel,
}

impl ReluKernel {
    /// Create a new ReLU kernel
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operation_type: OperationType::MemoryIntensive,
            backend_metadata: HashMap::new(),
        };

        let cuda_source = r#"
extern "C" __global__ void relu(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = max(0.0f, input[i]);
    }
}
"#
        .to_string();

        let wgpu_source = r#"
struct Uniforms {
    n: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, write> output: array<f32>;

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if (i < uniforms.n) {
        output[i] = max(0.0, input[i]);
    }
}
"#
        .to_string();

        let metal_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void relu(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        output[gid] = max(0.0f, input[gid]);
    }
}
"#
        .to_string();

        let opencl_source = r#"
__kernel void relu(
    __global const float* input,
    __global float* output,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        output[i] = max(0.0f, input[i]);
    }
}
"#
        .to_string();

        Self {
            base: BaseKernel::new(
                "relu",
                &cuda_source,
                &wgpu_source,
                &metal_source,
                &opencl_source,
                metadata,
            ),
        }
    }
}

impl GpuKernel for ReluKernel {
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
        match params.data_type {
            DataType::Float32 | DataType::Float64 | DataType::Float16 | DataType::BFloat16 => true,
            _ => false,
        }
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }

        // No specialization needed for ReLU
        Ok(Box::new(Self::new()))
    }
}

/// Sigmoid activation function kernel
pub struct SigmoidKernel {
    base: BaseKernel,
}

impl SigmoidKernel {
    /// Create a new Sigmoid kernel
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operation_type: OperationType::ComputeIntensive,
            backend_metadata: HashMap::new(),
        };

        let cuda_source = r#"
extern "C" __global__ void sigmoid(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}
"#
        .to_string();

        let wgpu_source = r#"
struct Uniforms {
    n: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, write> output: array<f32>;

@compute @workgroup_size(256)
fn sigmoid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if (i < uniforms.n) {
        output[i] = 1.0 / (1.0 + exp(-input[i]));
    }
}
"#
        .to_string();

        let metal_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sigmoid(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        output[gid] = 1.0f / (1.0f + exp(-input[gid]));
    }
}
"#
        .to_string();

        let opencl_source = r#"
__kernel void sigmoid(
    __global const float* input,
    __global float* output,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        output[i] = 1.0f / (1.0f + exp(-input[i]));
    }
}
"#
        .to_string();

        Self {
            base: BaseKernel::new(
                "sigmoid",
                &cuda_source,
                &wgpu_source,
                &metal_source,
                &opencl_source,
                metadata,
            ),
        }
    }
}

impl GpuKernel for SigmoidKernel {
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
        match params.data_type {
            DataType::Float32 | DataType::Float64 | DataType::Float16 | DataType::BFloat16 => true,
            _ => false,
        }
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }

        // No specialization needed for Sigmoid
        Ok(Box::new(Self::new()))
    }
}
