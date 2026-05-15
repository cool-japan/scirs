//! Element-wise operation kernels for GPU
//!
//! These kernels implement basic element-wise operations that are fundamental
//! to tensor computations: addition, multiplication, division, subtraction.

use std::collections::HashMap;

use crate::gpu::kernels::{
    BaseKernel, DataType, GpuKernel, KernelMetadata, KernelParams, OperationType,
};
use crate::gpu::{GpuBackend, GpuError};

/// Element-wise addition kernel (a + b)
pub struct ElementwiseAddKernel {
    base: BaseKernel,
}

impl Default for ElementwiseAddKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementwiseAddKernel {
    /// Create a new element-wise addition kernel
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operationtype: OperationType::MemoryIntensive,
            backend_metadata: HashMap::new(),
        };

        let cuda_source = r#"
extern "C" __global__ void elementwise_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ result,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = a[i] + b[i];
    }
}
"#
        .to_string();

        let rocm_source = cuda_source.clone();

        let wgpu_source = r#"
struct Uniforms {
    n: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, write> result: array<f32>;

@compute @workgroup_size(256)
fn elementwise_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if (i < uniforms.n) {
        result[i] = a[i] + b[i];
    }
}
"#
        .to_string();

        let metal_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_add(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        result[gid] = a[gid] + b[gid];
    }
}
"#
        .to_string();

        let opencl_source = r#"
__kernel void elementwise_add(
    __global const float* a,
    __global const float* b,
    __global float* result,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        result[i] = a[i] + b[i];
    }
}
"#
        .to_string();

        Self {
            base: BaseKernel::new(
                "elementwise_add",
                &cuda_source,
                &rocm_source,
                &wgpu_source,
                &metal_source,
                &opencl_source,
                metadata,
            ),
        }
    }
}

impl GpuKernel for ElementwiseAddKernel {
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
            params.datatype,
            DataType::Float32 | DataType::Float64 | DataType::Int32
        )
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }

        // For now, return the same kernel (no type specialization implemented yet)
        Ok(Box::new(Self::new()))
    }
}

/// Element-wise multiplication kernel (a * b)
pub struct ElementwiseMulKernel {
    base: BaseKernel,
}

impl Default for ElementwiseMulKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementwiseMulKernel {
    /// Create a new element-wise multiplication kernel
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operationtype: OperationType::MemoryIntensive,
            backend_metadata: HashMap::new(),
        };

        let cuda_source = r#"
extern "C" __global__ void elementwise_mul(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ result,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = a[i] * b[i];
    }
}
"#
        .to_string();

        let rocm_source = cuda_source.clone();

        let wgpu_source = r#"
struct Uniforms {
    n: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, write> result: array<f32>;

@compute @workgroup_size(256)
fn elementwise_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if (i < uniforms.n) {
        result[i] = a[i] * b[i];
    }
}
"#
        .to_string();

        let metal_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_mul(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        result[gid] = a[gid] * b[gid];
    }
}
"#
        .to_string();

        let opencl_source = r#"
__kernel void elementwise_mul(
    __global const float* a,
    __global const float* b,
    __global float* result,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        result[i] = a[i] * b[i];
    }
}
"#
        .to_string();

        Self {
            base: BaseKernel::new(
                "elementwise_mul",
                &cuda_source,
                &rocm_source,
                &wgpu_source,
                &metal_source,
                &opencl_source,
                metadata,
            ),
        }
    }
}

impl GpuKernel for ElementwiseMulKernel {
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
            params.datatype,
            DataType::Float32 | DataType::Float64 | DataType::Int32
        )
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }

        Ok(Box::new(Self::new()))
    }
}

/// Element-wise division kernel (a / b)
pub struct ElementwiseDivKernel {
    base: BaseKernel,
}

impl Default for ElementwiseDivKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementwiseDivKernel {
    /// Create a new element-wise division kernel
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operationtype: OperationType::ComputeIntensive, // Division is more compute-intensive
            backend_metadata: HashMap::new(),
        };

        let cuda_source = r#"
extern "C" __global__ void elementwise_div(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ result,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = a[i] / b[i];
    }
}
"#
        .to_string();

        let rocm_source = cuda_source.clone();

        let wgpu_source = r#"
struct Uniforms {
    n: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, write> result: array<f32>;

@compute @workgroup_size(256)
fn elementwise_div(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if (i < uniforms.n) {
        result[i] = a[i] / b[i];
    }
}
"#
        .to_string();

        let metal_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_div(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        result[gid] = a[gid] / b[gid];
    }
}
"#
        .to_string();

        let opencl_source = r#"
__kernel void elementwise_div(
    __global const float* a,
    __global const float* b,
    __global float* result,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        result[i] = a[i] / b[i];
    }
}
"#
        .to_string();

        Self {
            base: BaseKernel::new(
                "elementwise_div",
                &cuda_source,
                &rocm_source,
                &wgpu_source,
                &metal_source,
                &opencl_source,
                metadata,
            ),
        }
    }
}

impl GpuKernel for ElementwiseDivKernel {
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
        matches!(params.datatype, DataType::Float32 | DataType::Float64)
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }

        Ok(Box::new(Self::new()))
    }
}

/// Scalar multiplication kernel (a * scalar)
pub struct ScalarMulKernel {
    base: BaseKernel,
}

impl Default for ScalarMulKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarMulKernel {
    /// Create a new scalar multiplication kernel
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operationtype: OperationType::MemoryIntensive,
            backend_metadata: HashMap::new(),
        };

        let cuda_source = r#"
extern "C" __global__ void scalar_mul(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scalar,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i] * scalar;
    }
}
"#
        .to_string();

        let rocm_source = cuda_source.clone();

        let wgpu_source = r#"
struct Uniforms {
    n: u32,
    scalar: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, write> output: array<f32>;

@compute @workgroup_size(256)
fn scalar_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if (i < uniforms.n) {
        output[i] = input[i] * uniforms.scalar;
    }
}
"#
        .to_string();

        let metal_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void scalar_mul(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        output[gid] = input[gid] * scalar;
    }
}
"#
        .to_string();

        let opencl_source = r#"
__kernel void scalar_mul(
    __global const float* input,
    __global float* output,
    const float scalar,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        output[i] = input[i] * scalar;
    }
}
"#
        .to_string();

        Self {
            base: BaseKernel::new(
                "scalar_mul",
                &cuda_source,
                &rocm_source,
                &wgpu_source,
                &metal_source,
                &opencl_source,
                metadata,
            ),
        }
    }
}

impl GpuKernel for ScalarMulKernel {
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
            params.datatype,
            DataType::Float32 | DataType::Float64 | DataType::Int32
        )
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }

        Ok(Box::new(Self::new()))
    }
}
/// Element-wise subtraction kernel (a - b)
pub struct ElementwiseSubKernel {
    base: BaseKernel,
}

impl Default for ElementwiseSubKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementwiseSubKernel {
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operationtype: OperationType::MemoryIntensive,
            backend_metadata: HashMap::new(),
        };

        let cuda_source = r#"
extern \"C\" __global__ void elementwise_sub(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ result,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = a[i] - b[i];
    }
}
"#
        .to_string();

        let rocm_source = cuda_source.clone();

        let wgpu_source = r#"
struct Uniforms {
    n: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if (i < uniforms.n) {
        result[i] = a[i] - b[i];
    }
}
"#
        .to_string();

        let metal_source = String::new();
        let opencl_source = String::new();

        Self {
            base: BaseKernel::new(
                "elementwise_sub",
                &cuda_source,
                &rocm_source,
                &wgpu_source,
                &metal_source,
                &opencl_source,
                metadata,
            ),
        }
    }
}

impl GpuKernel for ElementwiseSubKernel {
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
            params.datatype,
            DataType::Float32 | DataType::Float64 | DataType::Int32
        )
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }
        Ok(Box::new(Self::new()))
    }
}
/// Element-wise power kernel (pow(a, b))
pub struct ElementwisePowKernel {
    base: BaseKernel,
}

impl Default for ElementwisePowKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementwisePowKernel {
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operationtype: OperationType::ComputeIntensive,
            backend_metadata: HashMap::new(),
        };

        let cuda_source = r#"
extern "C" __global__ void elementwise_pow(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ result,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = powf(a[i], b[i]);
    }
}
"#
        .to_string();

        let rocm_source = cuda_source.clone();

        let wgpu_source = r#"
struct Uniforms {
    n: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if (i < uniforms.n) {
        result[i] = pow(a[i], b[i]);
    }
}
"#
        .to_string();

        let metal_source = String::new();
        let opencl_source = String::new();

        Self {
            base: BaseKernel::new(
                "elementwise_pow",
                &cuda_source,
                &rocm_source,
                &wgpu_source,
                &metal_source,
                &opencl_source,
                metadata,
            ),
        }
    }
}

impl GpuKernel for ElementwisePowKernel {
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
        matches!(params.datatype, DataType::Float32 | DataType::Float64)
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }
        Ok(Box::new(Self::new()))
    }
}

/// Element-wise square root kernel (sqrt(a))
pub struct ElementwiseSqrtKernel {
    base: BaseKernel,
}

impl Default for ElementwiseSqrtKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementwiseSqrtKernel {
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operationtype: OperationType::ComputeIntensive,
            backend_metadata: HashMap::new(),
        };

        let cuda_source = r#"
extern "C" __global__ void elementwise_sqrt(
    const float* __restrict__ a,
    float* __restrict__ result,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = sqrtf(a[i]);
    }
}
"#
        .to_string();

        let rocm_source = cuda_source.clone();

        let wgpu_source = r#"
struct Uniforms {
    n: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if (i < uniforms.n) {
        result[i] = sqrt(input[i]);
    }
}
"#
        .to_string();

        let metal_source = String::new();
        let opencl_source = String::new();

        Self {
            base: BaseKernel::new(
                "elementwise_sqrt",
                &cuda_source,
                &rocm_source,
                &wgpu_source,
                &metal_source,
                &opencl_source,
                metadata,
            ),
        }
    }
}

impl GpuKernel for ElementwiseSqrtKernel {
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
        matches!(params.datatype, DataType::Float32 | DataType::Float64)
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }
        Ok(Box::new(Self::new()))
    }
}

/// Element-wise exponential kernel (exp(a))
pub struct ElementwiseExpKernel {
    base: BaseKernel,
}

impl Default for ElementwiseExpKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementwiseExpKernel {
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operationtype: OperationType::ComputeIntensive,
            backend_metadata: HashMap::new(),
        };

        let cuda_source = r#"
extern "C" __global__ void elementwise_exp(
    const float* __restrict__ a,
    float* __restrict__ result,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = expf(a[i]);
    }
}
"#
        .to_string();

        let rocm_source = cuda_source.clone();

        let wgpu_source = r#"
struct Uniforms {
    n: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if (i < uniforms.n) {
        result[i] = exp(input[i]);
    }
}
"#
        .to_string();

        let metal_source = String::new();
        let opencl_source = String::new();

        Self {
            base: BaseKernel::new(
                "elementwise_exp",
                &cuda_source,
                &rocm_source,
                &wgpu_source,
                &metal_source,
                &opencl_source,
                metadata,
            ),
        }
    }
}

impl GpuKernel for ElementwiseExpKernel {
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
        matches!(params.datatype, DataType::Float32 | DataType::Float64)
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }
        Ok(Box::new(Self::new()))
    }
}

/// Element-wise logarithm kernel (log(a))
pub struct ElementwiseLogKernel {
    base: BaseKernel,
}

impl Default for ElementwiseLogKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementwiseLogKernel {
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operationtype: OperationType::ComputeIntensive,
            backend_metadata: HashMap::new(),
        };

        let cuda_source = r#"
extern "C" __global__ void elementwise_log(
    const float* __restrict__ a,
    float* __restrict__ result,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = logf(a[i]);
    }
}
"#
        .to_string();

        let rocm_source = cuda_source.clone();

        let wgpu_source = r#"
struct Uniforms {
    n: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if (i < uniforms.n) {
        result[i] = log(input[i]);
    }
}
"#
        .to_string();

        let metal_source = String::new();
        let opencl_source = String::new();

        Self {
            base: BaseKernel::new(
                "elementwise_log",
                &cuda_source,
                &rocm_source,
                &wgpu_source,
                &metal_source,
                &opencl_source,
                metadata,
            ),
        }
    }
}

impl GpuKernel for ElementwiseLogKernel {
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
        matches!(params.datatype, DataType::Float32 | DataType::Float64)
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }
        Ok(Box::new(Self::new()))
    }
}
