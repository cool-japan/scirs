//! GPU kernel library for common scientific computing operations
//!
//! This module provides optimized GPU kernels for various operations used in
//! scientific computing, with support for multiple GPU backends.

use std::collections::HashMap;
use std::fmt;

pub mod blas;
pub mod complex;
pub mod elementwise;
pub mod ml;
pub mod reduction;
pub mod transform;

use crate::gpu::{GpuBackend, GpuError};

/// Supported data types for GPU kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// 32-bit floating point (f32)
    Float32,
    /// 64-bit floating point (f64)
    Float64,
    /// 32-bit signed integer (i32)
    Int32,
    /// 32-bit unsigned integer (u32)
    UInt32,
    /// 16-bit floating point (f16)
    Float16,
    /// Brain floating point (bfloat16)
    BFloat16,
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Float32 => write!(f, "f32"),
            DataType::Float64 => write!(f, "f64"),
            DataType::Int32 => write!(f, "i32"),
            DataType::UInt32 => write!(f, "u32"),
            DataType::Float16 => write!(f, "f16"),
            DataType::BFloat16 => write!(f, "bf16"),
        }
    }
}

/// The type of operation performed by the kernel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    /// Primarily compute-intensive operations
    ComputeIntensive,
    /// Primarily memory-intensive operations
    MemoryIntensive,
    /// Balanced between compute and memory
    Balanced,
}

/// Metadata for kernel execution
#[derive(Debug, Clone)]
pub struct KernelMetadata {
    /// Recommended workgroup size
    pub workgroup_size: [u32; 3],
    /// Local memory usage in bytes
    pub local_memory_usage: usize,
    /// Whether the kernel supports tensor cores (NVIDIA) or similar
    pub supports_tensor_cores: bool,
    /// Operation type (compute intensive, memory intensive, balanced)
    pub operationtype: OperationType,
    /// Additional backend-specific metadata
    pub backend_metadata: HashMap<String, String>,
}

impl Default for KernelMetadata {
    fn default() -> Self {
        Self {
            workgroup_size: [16, 16, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operationtype: OperationType::Balanced,
            backend_metadata: HashMap::new(),
        }
    }
}

/// Parameters for kernel specialization
#[derive(Debug, Clone)]
pub struct KernelParams {
    /// Numeric type (f32, f64, etc.)
    pub datatype: DataType,
    /// Input dimensions
    pub input_dims: Vec<usize>,
    /// Output dimensions
    pub output_dims: Vec<usize>,
    /// Additional numeric parameters
    pub numeric_params: HashMap<String, f64>,
    /// Additional string parameters
    pub string_params: HashMap<String, String>,
}

impl KernelParams {
    /// Create new kernel parameters
    pub fn new(datatype: DataType) -> Self {
        Self {
            datatype,
            input_dims: Vec::new(),
            output_dims: Vec::new(),
            numeric_params: HashMap::new(),
            string_params: HashMap::new(),
        }
    }

    /// Set input dimensions
    pub fn with_input_dims(mut self, dims: Vec<usize>) -> Self {
        self.input_dims = dims;
        self
    }

    /// Set output dimensions
    pub fn with_output_dims(mut self, dims: Vec<usize>) -> Self {
        self.output_dims = dims;
        self
    }

    /// Add a numeric parameter
    pub fn with_numeric_param(mut self, name: &str, value: f64) -> Self {
        self.numeric_params.insert(name.to_string(), value);
        self
    }

    /// Add a string parameter
    pub fn with_string_param(mut self, name: &str, value: &str) -> Self {
        self.string_params
            .insert(name.to_string(), value.to_string());
        self
    }
}

/// GPU Kernel interface
pub trait GpuKernel: Send + Sync {
    /// The name of the kernel
    fn name(&self) -> &str;

    /// Get kernel source for the specified backend
    fn source_for_backend(&self, backend: GpuBackend) -> Result<String, GpuError>;

    /// Get kernel metadata (workgroup size, memory requirements, etc.)
    fn metadata(&self) -> KernelMetadata;

    /// Can this kernel be specialized for the given parameters?
    fn can_specialize(&self, params: &KernelParams) -> bool;

    /// Create a specialized version of this kernel for the given parameters
    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError>;
}

/// Base kernel implementation that can be used by specialized kernels
pub struct BaseKernel {
    name: String,
    cuda_source: String,
    rocm_source: String,
    wgpu_source: String,
    metal_source: String,
    opencl_source: String,
    metadata: KernelMetadata,
}

impl BaseKernel {
    /// Create a new base kernel
    pub fn new(
        name: &str,
        cuda_source: &str,
        rocm_source: &str,
        wgpu_source: &str,
        metal_source: &str,
        opencl_source: &str,
        metadata: KernelMetadata,
    ) -> Self {
        Self {
            name: name.to_string(),
            cuda_source: cuda_source.to_string(),
            rocm_source: rocm_source.to_string(),
            wgpu_source: wgpu_source.to_string(),
            metal_source: metal_source.to_string(),
            opencl_source: opencl_source.to_string(),
            metadata,
        }
    }
}

impl GpuKernel for BaseKernel {
    fn name(&self) -> &str {
        &self.name
    }

    fn source_for_backend(&self, backend: GpuBackend) -> Result<String, GpuError> {
        match backend {
            GpuBackend::Cuda => Ok(self.cuda_source.clone()),
            GpuBackend::Rocm => Ok(self.rocm_source.clone()),
            GpuBackend::Wgpu => Ok(self.wgpu_source.clone()),
            GpuBackend::Metal => Ok(self.metal_source.clone()),
            GpuBackend::OpenCL => Ok(self.opencl_source.clone()),
            _ => Err(GpuError::UnsupportedBackend(backend)),
        }
    }

    fn metadata(&self) -> KernelMetadata {
        self.metadata.clone()
    }

    fn can_specialize(&self, params: &KernelParams) -> bool {
        false // Base implementation doesn't support specialization
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        Err(GpuError::SpecializationNotSupported)
    }
}

/// Registry of available GPU kernels
pub struct KernelRegistry {
    kernels: HashMap<String, Box<dyn GpuKernel>>,
}

impl KernelRegistry {
    /// Create a new kernel registry
    pub fn new() -> Self {
        Self {
            kernels: HashMap::new(),
        }
    }

    /// Create a registry with all default kernels
    pub fn with_default_kernels() -> Self {
        let mut registry = Self::new();

        // Register BLAS kernels
        registry.register(Box::new(blas::gemm::GemmKernel::new()));
        registry.register(Box::new(blas::axpy::AxpyKernel::new()));
        registry.register(Box::new(blas::gemv::GemvKernel::new()));

        // Register elementwise kernels
        registry.register(Box::new(elementwise::ElementwiseAddKernel::new()));
        registry.register(Box::new(elementwise::ElementwiseSubKernel::new()));
        registry.register(Box::new(elementwise::ElementwiseMulKernel::new()));
        registry.register(Box::new(elementwise::ElementwiseDivKernel::new()));
        registry.register(Box::new(elementwise::ElementwisePowKernel::new()));
        registry.register(Box::new(elementwise::ElementwiseSqrtKernel::new()));
        registry.register(Box::new(elementwise::ElementwiseExpKernel::new()));
        registry.register(Box::new(elementwise::ElementwiseLogKernel::new()));

        // Register optimization kernels
        registry.register(Box::new(create_adam_optimizer_kernel()));
        registry.register(Box::new(create_sgd_optimizer_kernel()));
        registry.register(Box::new(create_rmsprop_optimizer_kernel()));
        registry.register(Box::new(create_adagrad_optimizer_kernel()));
        registry.register(Box::new(create_lamb_optimizer_kernel()));

        // Register utility kernels
        registry.register(Box::new(create_memcpy_kernel()));
        registry.register(Box::new(create_fill_kernel()));
        registry.register(Box::new(create_reduce_sum_kernel()));
        registry.register(Box::new(create_reduce_max_kernel()));

        // Register transform kernels
        registry.register(Box::new(transform::fft::FftKernel::new()));
        registry.register(Box::new(transform::convolution::Conv1dKernel::new()));
        registry.register(Box::new(transform::convolution::Conv2dKernel::new()));

        // Register reduction kernels
        registry.register(Box::new(reduction::sum::SumKernel::new()));
        registry.register(Box::new(reduction::norm::NormKernel::new()));
        registry.register(Box::new(reduction::min_max::MinKernel::new()));
        registry.register(Box::new(reduction::min_max::MaxKernel::new()));
        registry.register(Box::new(reduction::mean::MeanKernel::new()));
        registry.register(Box::new(reduction::std_dev::StdDevKernel::new()));

        // Register ML kernels
        registry.register(Box::new(ml::activation::ReluKernel::new()));
        registry.register(Box::new(ml::activation::SigmoidKernel::new()));
        registry.register(Box::new(ml::activation::TanhKernel::new()));
        registry.register(Box::new(ml::softmax::SoftmaxKernel::new()));
        registry.register(Box::new(ml::pooling::MaxPoolKernel::new()));
        registry.register(Box::new(ml::pooling::AvgPoolKernel::new()));

        // Register complex number kernels
        registry.register(Box::new(complex::ComplexMultiplyKernel::new()));
        registry.register(Box::new(complex::ComplexConjugateKernel::new()));
        registry.register(Box::new(complex::ComplexMatMulKernel::new()));

        // Register RK4 integration kernels for advanced mode
        registry.register(Box::new(create_rk4_stage1_kernel()));
        registry.register(Box::new(create_rk4_stage2_kernel()));
        registry.register(Box::new(create_rk4_stage3_kernel()));
        registry.register(Box::new(create_rk4_stage4_kernel()));
        registry.register(Box::new(create_rk4_combine_kernel()));
        registry.register(Box::new(createerror_estimate_kernel()));

        registry
    }

    /// Register a kernel
    pub fn register(&mut self, kernel: Box<dyn GpuKernel>) {
        self.kernels.insert(kernel.name().to_string(), kernel);
    }

    /// Get a kernel by name
    pub fn get(&self, name: &str) -> Option<&dyn GpuKernel> {
        self.kernels.get(name).map(|k| k.as_ref())
    }

    /// Get a specialized kernel
    pub fn get_specialized(
        &self,
        name: &str,
        params: &KernelParams,
    ) -> Result<Box<dyn GpuKernel>, GpuError> {
        let kernel = self
            .get(name)
            .ok_or_else(|| GpuError::KernelNotFound(name.to_string()))?;

        if kernel.can_specialize(params) {
            kernel.specialize(params)
        } else {
            Err(GpuError::SpecializationNotSupported)
        }
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::with_default_kernels()
    }
}

// ─── WGSL shader sources for all kernels ─────────────────────────────────────

/// WGSL source for the Adam optimizer kernel (workgroup 256).
///
/// Buffers (all at group 0):
///   0 → params (read_write), 1 → grads (read), 2 → m (read_write),
///   3 → v (read_write), 4 → uniforms (uniform)
const ADAM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;

struct AdamUniforms {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    n: u32,
};

@group(0) @binding(4) var<uniform> uniforms: AdamUniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.n { return; }

    var grad = grads[idx];
    if uniforms.weight_decay > 0.0 {
        grad += uniforms.weight_decay * params[idx];
    }

    // Update biased first moment estimate
    m[idx] = uniforms.beta1 * m[idx] + (1.0 - uniforms.beta1) * grad;

    // Update biased second raw moment estimate
    v[idx] = uniforms.beta2 * v[idx] + (1.0 - uniforms.beta2) * grad * grad;

    // Bias-corrected moment estimates
    let m_hat = m[idx] / uniforms.bias_correction1;
    let v_hat = v[idx] / uniforms.bias_correction2;

    // Parameter update
    params[idx] -= uniforms.lr * m_hat / (sqrt(v_hat) + uniforms.eps);
}
"#;

/// WGSL source for the SGD optimizer kernel (workgroup 256, with momentum).
///
/// Buffers: 0 → params (rw), 1 → grads (r), 2 → momentum_buf (rw)
/// Uniforms: lr, momentum_factor, n
const SGD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> momentum_buf: array<f32>;

struct SgdUniforms {
    lr: f32,
    momentum_factor: f32,
    n: u32,
};

@group(0) @binding(3) var<uniform> uniforms: SgdUniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.n { return; }

    let grad = grads[idx];
    if uniforms.momentum_factor > 0.0 {
        // SGD with momentum: buf = momentum * buf + grad; param -= lr * buf
        let buf = uniforms.momentum_factor * momentum_buf[idx] + grad;
        momentum_buf[idx] = buf;
        params[idx] -= uniforms.lr * buf;
    } else {
        params[idx] -= uniforms.lr * grad;
    }
}
"#;

/// WGSL source for the RMSprop optimizer kernel (workgroup 256).
///
/// Buffers: 0 → params (rw), 1 → grads (r), 2 → cache (rw)
/// Uniforms: lr, decay (alpha), epsilon, n
const RMSPROP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> cache: array<f32>;

struct RmspropUniforms {
    lr: f32,
    decay: f32,
    epsilon: f32,
    n: u32,
};

@group(0) @binding(3) var<uniform> uniforms: RmspropUniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.n { return; }

    let grad = grads[idx];
    // cache = decay * cache + (1 - decay) * grad^2
    let new_cache = uniforms.decay * cache[idx] + (1.0 - uniforms.decay) * grad * grad;
    cache[idx] = new_cache;
    // params -= lr * grad / (sqrt(cache) + epsilon)
    params[idx] -= uniforms.lr * grad / (sqrt(new_cache) + uniforms.epsilon);
}
"#;

/// WGSL source for the Adagrad optimizer kernel (workgroup 256).
///
/// Buffers: 0 → params (rw), 1 → grads (r), 2 → cache (rw, accumulated sq grads)
/// Uniforms: lr, epsilon, n
const ADAGRAD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> cache: array<f32>;

struct AdagradUniforms {
    lr: f32,
    epsilon: f32,
    n: u32,
};

@group(0) @binding(3) var<uniform> uniforms: AdagradUniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.n { return; }

    let grad = grads[idx];
    // Accumulate squared gradient
    let new_cache = cache[idx] + grad * grad;
    cache[idx] = new_cache;
    // Adaptive update
    params[idx] -= uniforms.lr * grad / (sqrt(new_cache) + uniforms.epsilon);
}
"#;

/// WGSL source for the LAMB optimizer kernel (workgroup 256, uniform-norm variant).
///
/// The caller pre-computes param_norm and grad_norm (L2 norms) and passes them
/// as uniforms so a single pass can perform the layer-wise ratio scaling.
///
/// Buffers: 0 → params (rw), 1 → grads (r)
/// Uniforms: lr, weight_decay, param_norm, grad_norm, n
const LAMB_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;

struct LambUniforms {
    lr: f32,
    weight_decay: f32,
    param_norm: f32,
    grad_norm: f32,
    n: u32,
};

@group(0) @binding(2) var<uniform> uniforms: LambUniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.n { return; }

    // update = grad + weight_decay * param
    let update = grads[idx] + uniforms.weight_decay * params[idx];

    // Layer-wise adaptive ratio: trust_ratio = param_norm / (grad_norm + eps)
    // Guard against zero norms (use 1.0 as neutral ratio)
    let eps = 1e-6;
    let denom = uniforms.grad_norm + eps;
    let trust_ratio = select(1.0, uniforms.param_norm / denom, uniforms.param_norm > 0.0 && uniforms.grad_norm > 0.0);

    params[idx] -= uniforms.lr * trust_ratio * update;
}
"#;

/// WGSL source for the memcpy kernel (workgroup 256).
///
/// Buffers: 0 → src (read), 1 → dst (read_write)
/// Uniforms: n
const MEMCPY_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

struct MemcpyUniforms {
    n: u32,
};

@group(0) @binding(2) var<uniform> uniforms: MemcpyUniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.n { return; }
    dst[idx] = src[idx];
}
"#;

/// WGSL source for the fill kernel (workgroup 256).
///
/// Buffers: 0 → dst (read_write)
/// Uniforms: value, n
const FILL_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;

struct FillUniforms {
    value: f32,
    n: u32,
};

@group(0) @binding(1) var<uniform> uniforms: FillUniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.n { return; }
    dst[idx] = uniforms.value;
}
"#;

/// WGSL source for the reduce_sum kernel (single-pass workgroup reduction, workgroup 256).
///
/// Each workgroup reduces its slice into one partial sum written to `output[workgroup_id]`.
/// The host must dispatch `ceil(n / 256)` workgroups and sum `output` on the CPU or with a
/// second dispatch.
///
/// Buffers: 0 → input (read), 1 → output (read_write)
/// Uniforms: n
const REDUCE_SUM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct ReduceUniforms {
    n: u32,
};

@group(0) @binding(2) var<uniform> uniforms: ReduceUniforms;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         wg_id:     vec3<u32>,
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    // Load with bounds guard
    if gidx < uniforms.n {
        scratch[lidx] = input[gidx];
    } else {
        scratch[lidx] = 0.0;
    }
    workgroupBarrier();

    // Tree reduction within the workgroup
    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if lidx < stride {
            scratch[lidx] += scratch[lidx + stride];
        }
        workgroupBarrier();
        if stride == 1u { break; }
        stride = stride >> 1u;
    }

    // Thread 0 writes the partial sum for this workgroup
    if lidx == 0u {
        output[wg_id.x] = scratch[0];
    }
}
"#;

/// WGSL source for the reduce_max kernel (single-pass workgroup reduction, workgroup 256).
///
/// Uses the same two-pass convention as reduce_sum; each workgroup writes a partial maximum.
///
/// Buffers: 0 → input (read), 1 → output (read_write)
/// Uniforms: n
const REDUCE_MAX_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct ReduceUniforms {
    n: u32,
};

@group(0) @binding(2) var<uniform> uniforms: ReduceUniforms;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         wg_id:     vec3<u32>,
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    // Load with bounds guard; use -f32::MAX as neutral element for max
    if gidx < uniforms.n {
        scratch[lidx] = input[gidx];
    } else {
        scratch[lidx] = -3.402823e+38; // -FLT_MAX
    }
    workgroupBarrier();

    // Tree reduction within the workgroup
    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if lidx < stride {
            scratch[lidx] = max(scratch[lidx], scratch[lidx + stride]);
        }
        workgroupBarrier();
        if stride == 1u { break; }
        stride = stride >> 1u;
    }

    if lidx == 0u {
        output[wg_id.x] = scratch[0];
    }
}
"#;

/// WGSL for RK4 stage 1: k1[i] = h * f(t, y[i])  where f(t, y) = -y  (exponential decay placeholder).
///
/// Buffers: 0 → y (read), 1 → k1 (read_write)
/// Uniforms: t, h, n
const RK4_STAGE1_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read_write> k1: array<f32>;

struct Rk4Uniforms {
    t: f32,
    h: f32,
    n: u32,
};

@group(0) @binding(2) var<uniform> uniforms: Rk4Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.n { return; }
    // Placeholder ODE: dy/dt = -y  (exponential decay)
    let dydt = -y[idx];
    k1[idx] = uniforms.h * dydt;
}
"#;

/// WGSL for RK4 stage 2: k2[i] = h * f(t + h/2, y[i] + k1[i]/2).
///
/// Buffers: 0 → y (read), 1 → k1 (read), 2 → k2 (read_write)
/// Uniforms: t, h, n
const RK4_STAGE2_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read> k1: array<f32>;
@group(0) @binding(2) var<storage, read_write> k2: array<f32>;

struct Rk4Uniforms {
    t: f32,
    h: f32,
    n: u32,
};

@group(0) @binding(3) var<uniform> uniforms: Rk4Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.n { return; }
    let y_mid = y[idx] + 0.5 * k1[idx];
    // Placeholder ODE: dy/dt = -y
    let dydt = -y_mid;
    k2[idx] = uniforms.h * dydt;
}
"#;

/// WGSL for RK4 stage 3: k3[i] = h * f(t + h/2, y[i] + k2[i]/2).
///
/// Buffers: 0 → y (read), 1 → k2 (read), 2 → k3 (read_write)
/// Uniforms: t, h, n
const RK4_STAGE3_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read> k2: array<f32>;
@group(0) @binding(2) var<storage, read_write> k3: array<f32>;

struct Rk4Uniforms {
    t: f32,
    h: f32,
    n: u32,
};

@group(0) @binding(3) var<uniform> uniforms: Rk4Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.n { return; }
    let y_mid = y[idx] + 0.5 * k2[idx];
    // Placeholder ODE: dy/dt = -y
    let dydt = -y_mid;
    k3[idx] = uniforms.h * dydt;
}
"#;

/// WGSL for RK4 stage 4: k4[i] = h * f(t + h, y[i] + k3[i]).
///
/// Buffers: 0 → y (read), 1 → k3 (read), 2 → k4 (read_write)
/// Uniforms: t, h, n
const RK4_STAGE4_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read> k3: array<f32>;
@group(0) @binding(2) var<storage, read_write> k4: array<f32>;

struct Rk4Uniforms {
    t: f32,
    h: f32,
    n: u32,
};

@group(0) @binding(3) var<uniform> uniforms: Rk4Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.n { return; }
    let y_next = y[idx] + k3[idx];
    // Placeholder ODE: dy/dt = -y
    let dydt = -y_next;
    k4[idx] = uniforms.h * dydt;
}
"#;

/// WGSL for RK4 final combination: y_new[i] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6.
///
/// Buffers: 0 → y (read), 1 → k1 (read), 2 → k2 (read), 3 → k3 (read), 4 → k4 (read),
///          5 → y_new (read_write)
/// Uniforms: n
const RK4_COMBINE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read> k1: array<f32>;
@group(0) @binding(2) var<storage, read> k2: array<f32>;
@group(0) @binding(3) var<storage, read> k3: array<f32>;
@group(0) @binding(4) var<storage, read> k4: array<f32>;
@group(0) @binding(5) var<storage, read_write> y_new: array<f32>;

struct Rk4CombineUniforms {
    n: u32,
};

@group(0) @binding(6) var<uniform> uniforms: Rk4CombineUniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.n { return; }
    let weighted = k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx];
    y_new[idx] = y[idx] + weighted * (1.0 / 6.0);
}
"#;

/// WGSL for the error estimate kernel: err[i] = |y1[i] - y2[i]| / max(scale, eps).
///
/// scale = atol + rtol * max(|y1|, |y2|) — matches CUDA error_estimate.cu semantics.
///
/// Buffers: 0 → y1 (read), 1 → y2 (read), 2 → err (read_write)
/// Uniforms: rtol, atol, n
const ERROR_ESTIMATE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> y1: array<f32>;
@group(0) @binding(1) var<storage, read> y2: array<f32>;
@group(0) @binding(2) var<storage, read_write> err: array<f32>;

struct ErrorUniforms {
    rtol: f32,
    atol: f32,
    n: u32,
};

@group(0) @binding(3) var<uniform> uniforms: ErrorUniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.n { return; }

    let v1 = y1[idx];
    let v2 = y2[idx];
    let abs_err = abs(v1 - v2);
    let y_scale = max(abs(v1), abs(v2));
    let scale = uniforms.atol + uniforms.rtol * y_scale;
    err[idx] = abs_err / max(scale, 1e-7);
}
"#;

// ─── Kernel factory functions ─────────────────────────────────────────────────

/// Create RK4 Stage 1 kernel for advanced mode GPU acceleration
fn create_rk4_stage1_kernel() -> BaseKernel {
    let cuda_source = include_str!("rk4_stage1.cu");
    let metadata = KernelMetadata {
        workgroup_size: [256, 1, 1],
        local_memory_usage: 0,
        supports_tensor_cores: false,
        operationtype: OperationType::ComputeIntensive,
        backend_metadata: HashMap::new(),
    };

    BaseKernel::new(
        "rk4_stage1",
        cuda_source,
        cuda_source, // Use CUDA source for ROCm (HIP compatible)
        RK4_STAGE1_WGSL,
        "",          // Metal source not implemented yet
        cuda_source, // Use CUDA source for OpenCL (with minor modifications)
        metadata,
    )
}

/// Create RK4 Stage 2 kernel for advanced mode GPU acceleration
fn create_rk4_stage2_kernel() -> BaseKernel {
    let cuda_source = include_str!("rk4_stage2.cu");
    let metadata = KernelMetadata {
        workgroup_size: [256, 1, 1],
        local_memory_usage: 0,
        supports_tensor_cores: false,
        operationtype: OperationType::ComputeIntensive,
        backend_metadata: HashMap::new(),
    };

    BaseKernel::new(
        "rk4_stage2",
        cuda_source,
        cuda_source,
        RK4_STAGE2_WGSL,
        "",
        cuda_source,
        metadata,
    )
}

/// Create RK4 Stage 3 kernel for advanced mode GPU acceleration
fn create_rk4_stage3_kernel() -> BaseKernel {
    let cuda_source = include_str!("rk4_stage3.cu");
    let metadata = KernelMetadata {
        workgroup_size: [256, 1, 1],
        local_memory_usage: 0,
        supports_tensor_cores: false,
        operationtype: OperationType::ComputeIntensive,
        backend_metadata: HashMap::new(),
    };

    BaseKernel::new(
        "rk4_stage3",
        cuda_source,
        cuda_source,
        RK4_STAGE3_WGSL,
        "",
        cuda_source,
        metadata,
    )
}

/// Create RK4 Stage 4 kernel for advanced mode GPU acceleration
fn create_rk4_stage4_kernel() -> BaseKernel {
    let cuda_source = include_str!("rk4_stage4.cu");
    let metadata = KernelMetadata {
        workgroup_size: [256, 1, 1],
        local_memory_usage: 0,
        supports_tensor_cores: false,
        operationtype: OperationType::ComputeIntensive,
        backend_metadata: HashMap::new(),
    };

    BaseKernel::new(
        "rk4_stage4",
        cuda_source,
        cuda_source,
        RK4_STAGE4_WGSL,
        "",
        cuda_source,
        metadata,
    )
}

/// Create RK4 Combination kernel for advanced mode GPU acceleration
fn create_rk4_combine_kernel() -> BaseKernel {
    let cuda_source = include_str!("rk4_combine.cu");
    let metadata = KernelMetadata {
        workgroup_size: [256, 1, 1],
        local_memory_usage: 0,
        supports_tensor_cores: false,
        operationtype: OperationType::MemoryIntensive,
        backend_metadata: HashMap::new(),
    };

    BaseKernel::new(
        "rk4_combine",
        cuda_source,
        cuda_source,
        RK4_COMBINE_WGSL,
        "",
        cuda_source,
        metadata,
    )
}

/// Create Error Estimation kernel for adaptive step size control
fn createerror_estimate_kernel() -> BaseKernel {
    let cuda_source = include_str!("error_estimate.cu");
    let metadata = KernelMetadata {
        workgroup_size: [256, 1, 1],
        local_memory_usage: 1024, // Shared memory for reduction
        supports_tensor_cores: false,
        operationtype: OperationType::ComputeIntensive,
        backend_metadata: HashMap::new(),
    };

    BaseKernel::new(
        "error_estimate",
        cuda_source,
        cuda_source,
        ERROR_ESTIMATE_WGSL,
        "",
        cuda_source,
        metadata,
    )
}

/// Create Adam optimizer kernel for GPU acceleration
fn create_adam_optimizer_kernel() -> BaseKernel {
    let cuda_source = include_str!("adam_optimizer.cu");
    let metadata = KernelMetadata {
        workgroup_size: [256, 1, 1],
        local_memory_usage: 0,
        supports_tensor_cores: false,
        operationtype: OperationType::ComputeIntensive,
        backend_metadata: HashMap::new(),
    };

    BaseKernel::new(
        "adam_optimizer",
        cuda_source,
        cuda_source,
        ADAM_WGSL,
        "",
        cuda_source,
        metadata,
    )
}

/// Create SGD optimizer kernel for GPU acceleration
fn create_sgd_optimizer_kernel() -> BaseKernel {
    let cuda_source = include_str!("sgd_optimizer.cu");
    let metadata = KernelMetadata {
        workgroup_size: [256, 1, 1],
        local_memory_usage: 0,
        supports_tensor_cores: false,
        operationtype: OperationType::MemoryIntensive,
        backend_metadata: HashMap::new(),
    };

    BaseKernel::new(
        "sgd_optimizer",
        cuda_source,
        cuda_source,
        SGD_WGSL,
        "",
        cuda_source,
        metadata,
    )
}

/// Create RMSprop optimizer kernel for GPU acceleration
fn create_rmsprop_optimizer_kernel() -> BaseKernel {
    let cuda_source = include_str!("rmsprop_optimizer.cu");
    let metadata = KernelMetadata {
        workgroup_size: [256, 1, 1],
        local_memory_usage: 0,
        supports_tensor_cores: false,
        operationtype: OperationType::ComputeIntensive,
        backend_metadata: HashMap::new(),
    };

    BaseKernel::new(
        "rmsprop_optimizer",
        cuda_source,
        cuda_source,
        RMSPROP_WGSL,
        "",
        cuda_source,
        metadata,
    )
}

/// Create Adagrad optimizer kernel for GPU acceleration
fn create_adagrad_optimizer_kernel() -> BaseKernel {
    let cuda_source = include_str!("adagrad_optimizer.cu");
    let metadata = KernelMetadata {
        workgroup_size: [256, 1, 1],
        local_memory_usage: 0,
        supports_tensor_cores: false,
        operationtype: OperationType::ComputeIntensive,
        backend_metadata: HashMap::new(),
    };

    BaseKernel::new(
        "adagrad_optimizer",
        cuda_source,
        cuda_source,
        ADAGRAD_WGSL,
        "",
        cuda_source,
        metadata,
    )
}

/// Create LAMB optimizer kernel for GPU acceleration
fn create_lamb_optimizer_kernel() -> BaseKernel {
    let cuda_source = include_str!("lamb_optimizer.cu");
    let metadata = KernelMetadata {
        workgroup_size: [256, 1, 1],
        local_memory_usage: 0,
        supports_tensor_cores: false,
        operationtype: OperationType::ComputeIntensive,
        backend_metadata: HashMap::new(),
    };

    BaseKernel::new(
        "lamb_optimizer",
        cuda_source,
        cuda_source,
        LAMB_WGSL,
        "",
        cuda_source,
        metadata,
    )
}

/// Create memory copy kernel for GPU acceleration
fn create_memcpy_kernel() -> BaseKernel {
    let cuda_source = include_str!("memcpy.cu");
    let metadata = KernelMetadata {
        workgroup_size: [256, 1, 1],
        local_memory_usage: 0,
        supports_tensor_cores: false,
        operationtype: OperationType::MemoryIntensive,
        backend_metadata: HashMap::new(),
    };

    BaseKernel::new(
        "memcpy",
        cuda_source,
        cuda_source,
        MEMCPY_WGSL,
        "",
        cuda_source,
        metadata,
    )
}

/// Create fill kernel for GPU acceleration
fn create_fill_kernel() -> BaseKernel {
    let cuda_source = include_str!("fill.cu");
    let metadata = KernelMetadata {
        workgroup_size: [256, 1, 1],
        local_memory_usage: 0,
        supports_tensor_cores: false,
        operationtype: OperationType::MemoryIntensive,
        backend_metadata: HashMap::new(),
    };

    BaseKernel::new(
        "fill",
        cuda_source,
        cuda_source,
        FILL_WGSL,
        "",
        cuda_source,
        metadata,
    )
}

/// Create reduce sum kernel for GPU acceleration
fn create_reduce_sum_kernel() -> BaseKernel {
    let cuda_source = include_str!("reduce_sum.cu");
    let metadata = KernelMetadata {
        workgroup_size: [256, 1, 1],
        local_memory_usage: 1024, // Shared memory for reduction
        supports_tensor_cores: false,
        operationtype: OperationType::ComputeIntensive,
        backend_metadata: HashMap::new(),
    };

    BaseKernel::new(
        "reduce_sum",
        cuda_source,
        cuda_source,
        REDUCE_SUM_WGSL,
        "",
        cuda_source,
        metadata,
    )
}

/// Create reduce max kernel for GPU acceleration
fn create_reduce_max_kernel() -> BaseKernel {
    let cuda_source = include_str!("reduce_max.cu");
    let metadata = KernelMetadata {
        workgroup_size: [256, 1, 1],
        local_memory_usage: 1024, // Shared memory for reduction
        supports_tensor_cores: false,
        operationtype: OperationType::ComputeIntensive,
        backend_metadata: HashMap::new(),
    };

    BaseKernel::new(
        "reduce_max",
        cuda_source,
        cuda_source,
        REDUCE_MAX_WGSL,
        "",
        cuda_source,
        metadata,
    )
}
