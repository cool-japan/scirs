//! GPU-accelerated batch statistical distribution computations via WebGPU (wgpu).
//!
//! This module provides batch evaluation of common distribution functions
//! (Normal and Exponential) using WGSL compute shaders dispatched through
//! the wgpu backend.  Each public function has a CPU fallback that is always
//! available, regardless of whether the `gpu_wgpu` feature is enabled.
//!
//! # Feature flags
//!
//! * `gpu`       — enables GPU abstraction layer via `scirs2-core/gpu`
//! * `gpu_wgpu`  — enables WGSL/wgpu compute path (implies `gpu`)
//!
//! # Examples
//!
//! ```rust
//! use scirs2_stats::gpu::{normal_log_pdf_batch, exponential_cdf_batch};
//!
//! let xs = vec![0.0_f64, 1.0, -1.0];
//! let log_pdfs = normal_log_pdf_batch(&xs, 0.0, 1.0);
//! assert_eq!(log_pdfs.len(), 3);
//!
//! let cdfs = exponential_cdf_batch(&xs, 1.0);
//! assert!(cdfs[2].abs() < 1e-10);  // x = -1 → CDF = 0
//! ```

// ---------------------------------------------------------------------------
// WGSL shader constants
// ---------------------------------------------------------------------------

/// WGSL compute shader: batch log-PDF of Normal distribution.
///
/// Computes `log_pdf(xᵢ; µ, σ) = -½((xᵢ−µ)/σ)² − ln(σ) − ln(√(2π))`
/// for each element in the input array, given scalar µ and σ in a uniform
/// buffer.
///
/// Bindings:
///   - `@group(0) @binding(0)`: read-only `array<f32>` input x values
///   - `@group(0) @binding(1)`: read-write `array<f32>` output values
///   - `@group(0) @binding(2)`: uniform `NormalParams { mu, sigma, n, _pad }`
const NORMAL_LOG_PDF_WGSL: &str = r#"
struct NormalParams {
    mu: f32,
    sigma: f32,
    n: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: NormalParams;

const LOG_SQRT_2PI: f32 = 0.9189385332046727;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    let z = (x[i] - params.mu) / params.sigma;
    out[i] = -0.5 * z * z - LOG_SQRT_2PI - log(params.sigma);
}
"#;

/// WGSL compute shader: batch CDF of Normal distribution.
///
/// Computes `Φ((xᵢ−µ)/(σ√2))` using the Horner-form A&S erf approximation
/// (Abramowitz & Stegun 7.1.26), which has a maximum absolute error of ≈ 1.5×10⁻⁷.
///
/// Bindings: same layout as [`NORMAL_LOG_PDF_WGSL`].
const NORMAL_CDF_WGSL: &str = r#"
struct NormalParams {
    mu: f32,
    sigma: f32,
    n: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: NormalParams;

fn approx_erf(v: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * abs(v));
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t
        + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-v * v));
    return select(-y, y, v >= 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    let z = (x[i] - params.mu) / (params.sigma * 1.41421356237f);
    out[i] = 0.5 * (1.0 + approx_erf(z));
}
"#;

/// WGSL compute shader: batch log-PDF of Exponential distribution.
///
/// For `xᵢ ≥ 0`: `log_pdf(xᵢ; λ) = ln(λ) − λ·xᵢ`.
/// For `xᵢ < 0`: outputs `-1e30` (representing −∞).
///
/// Bindings:
///   - `@group(0) @binding(0)`: read-only `array<f32>` input x values
///   - `@group(0) @binding(1)`: read-write `array<f32>` output values
///   - `@group(0) @binding(2)`: uniform `ExponParams { lambda, n, _pad0, _pad1 }`
const EXPONENTIAL_LOG_PDF_WGSL: &str = r#"
struct ExponParams {
    lambda: f32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: ExponParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    let xi = x[i];
    out[i] = select(-1e30, log(params.lambda) - params.lambda * xi, xi >= 0.0);
}
"#;

/// WGSL compute shader: batch CDF of Exponential distribution.
///
/// For `xᵢ ≥ 0`: `CDF(xᵢ; λ) = 1 − exp(−λ·xᵢ)`.
/// For `xᵢ < 0`: outputs `0.0`.
///
/// Bindings: same layout as [`EXPONENTIAL_LOG_PDF_WGSL`].
const EXPONENTIAL_CDF_WGSL: &str = r#"
struct ExponParams {
    lambda: f32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: ExponParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    let xi = x[i];
    out[i] = select(0.0, 1.0 - exp(-params.lambda * xi), xi >= 0.0);
}
"#;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise from GPU-accelerated distribution evaluation.
#[derive(Debug, Clone)]
pub enum GpuStatsError {
    /// No wgpu-capable GPU adapter is available on this system.
    GpuNotAvailable,
    /// A runtime error occurred during GPU pipeline setup or dispatch.
    RuntimeError(String),
    /// The `gpu_wgpu` feature flag is not enabled in this build.
    FeatureNotEnabled,
}

impl std::fmt::Display for GpuStatsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuStatsError::GpuNotAvailable => {
                write!(f, "wgpu GPU adapter not available on this system")
            }
            GpuStatsError::RuntimeError(msg) => {
                write!(f, "GPU runtime error: {msg}")
            }
            GpuStatsError::FeatureNotEnabled => {
                write!(f, "gpu_wgpu feature is not enabled in this build")
            }
        }
    }
}

impl std::error::Error for GpuStatsError {}

// ---------------------------------------------------------------------------
// GPU dispatch helper — real path (gpu_wgpu feature)
// ---------------------------------------------------------------------------

/// Upload `xs` as `f32`, dispatch the WGSL shader with a uniform parameter
/// buffer (`params_bytes`), and return the resulting `f32` values.
///
/// The shader is expected to declare exactly three bindings:
///   - `@group(0) @binding(0)` — read-only storage `array<f32>` (input)
///   - `@group(0) @binding(1)` — read-write storage `array<f32>` (output)
///   - `@group(0) @binding(2)` — uniform buffer (distribution parameters)
///
/// `params_bytes` must be 16-byte aligned (wgpu requirement for uniform
/// buffers); all parameter structs in this module use 4 × f32/u32 fields
/// (= 16 bytes) to satisfy this constraint.
#[cfg(feature = "gpu_wgpu")]
fn dispatch_with_params_f32(
    wgsl: &str,
    xs: &[f32],
    params_bytes: &[u8],
) -> Result<Vec<f32>, GpuStatsError> {
    use wgpu::{
        util::{BufferInitDescriptor, DeviceExt as _},
        Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
        BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
        CommandEncoderDescriptor, ComputePassDescriptor, DeviceDescriptor, Features, Instance,
        InstanceDescriptor, Limits, MapMode, PowerPreference, RequestAdapterOptions,
        ShaderModuleDescriptor, ShaderSource, ShaderStages,
    };

    let n = xs.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // ── Adapter / device ─────────────────────────────────────────────────────
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });

    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|_| GpuStatsError::GpuNotAvailable)?;

    let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("scirs2-stats-gpu"),
        required_features: Features::empty(),
        required_limits: Limits::default(),
        ..Default::default()
    }))
    .map_err(|e| GpuStatsError::RuntimeError(e.to_string()))?;

    // ── Shader / pipeline ─────────────────────────────────────────────────────
    let shader_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("scirs2-stats-shader"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });

    // Three-binding layout: storage-readonly, storage-readwrite, uniform
    let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("scirs2-stats-bgl"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("scirs2-stats-layout"),
        bind_group_layouts: &[Some(&bgl)],
        ..Default::default()
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("scirs2-stats-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // ── Buffers ───────────────────────────────────────────────────────────────
    let input_bytes: Vec<u8> = xs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let byte_len = (n as u64) * 4;

    let buf_input = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("scirs2-stats-input"),
        contents: &input_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    let buf_output = device.create_buffer(&BufferDescriptor {
        label: Some("scirs2-stats-output"),
        size: byte_len,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let buf_params = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("scirs2-stats-params"),
        contents: params_bytes,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let buf_staging = device.create_buffer(&BufferDescriptor {
        label: Some("scirs2-stats-staging"),
        size: byte_len,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ── Bind group ────────────────────────────────────────────────────────────
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("scirs2-stats-bg"),
        layout: &bgl,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: buf_input.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: buf_output.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: buf_params.as_entire_binding(),
            },
        ],
    });

    // ── Encode / dispatch ─────────────────────────────────────────────────────
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("scirs2-stats-encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("scirs2-stats-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (n as u32 + 63) / 64;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buf_output, 0, &buf_staging, 0, byte_len);
    queue.submit(Some(encoder.finish()));

    // ── Readback ──────────────────────────────────────────────────────────────
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| GpuStatsError::RuntimeError(format!("GPU poll error: {e:?}")))?;

    let slice = buf_staging.slice(0..byte_len);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(MapMode::Read, move |r| {
        let _ = tx.send(r);
    });

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| GpuStatsError::RuntimeError(format!("GPU poll during map: {e:?}")))?;

    rx.recv()
        .map_err(|_| GpuStatsError::RuntimeError("channel closed in map_async".into()))?
        .map_err(|e| GpuStatsError::RuntimeError(format!("map_async failed: {e:?}")))?;

    let mapped = slice.get_mapped_range();
    let result: Vec<f32> = mapped
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    drop(mapped);
    buf_staging.unmap();

    Ok(result)
}

// ---------------------------------------------------------------------------
// Encode NormalParams as little-endian bytes for the uniform buffer.
// Layout: mu (f32), sigma (f32), n (u32), _pad (u32) → 16 bytes.
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu_wgpu")]
fn encode_normal_params(mu: f32, sigma: f32, n: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&mu.to_le_bytes());
    out[4..8].copy_from_slice(&sigma.to_le_bytes());
    out[8..12].copy_from_slice(&n.to_le_bytes());
    // _pad = 0
    out
}

// Encode ExponParams: lambda (f32), n (u32), _pad0, _pad1 → 16 bytes.
#[cfg(feature = "gpu_wgpu")]
fn encode_expon_params(lambda: f32, n: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&lambda.to_le_bytes());
    out[4..8].copy_from_slice(&n.to_le_bytes());
    // _pad0, _pad1 = 0
    out
}

// ---------------------------------------------------------------------------
// WGPU dispatch wrappers (gpu_wgpu feature)
// ---------------------------------------------------------------------------

/// Attempt batch Normal log-PDF evaluation via WebGPU.
///
/// Returns `Err(GpuStatsError::GpuNotAvailable)` when no adapter is found.
#[cfg(feature = "gpu_wgpu")]
fn normal_log_pdf_wgpu(xs: &[f64], mu: f64, sigma: f64) -> Result<Vec<f64>, GpuStatsError> {
    let xs_f32: Vec<f32> = xs.iter().map(|&v| v as f32).collect();
    let params = encode_normal_params(mu as f32, sigma as f32, xs_f32.len() as u32);
    let out_f32 = dispatch_with_params_f32(NORMAL_LOG_PDF_WGSL, &xs_f32, &params)?;
    Ok(out_f32.iter().map(|&v| v as f64).collect())
}

/// Attempt batch Normal CDF evaluation via WebGPU.
#[cfg(feature = "gpu_wgpu")]
fn normal_cdf_wgpu(xs: &[f64], mu: f64, sigma: f64) -> Result<Vec<f64>, GpuStatsError> {
    let xs_f32: Vec<f32> = xs.iter().map(|&v| v as f32).collect();
    let params = encode_normal_params(mu as f32, sigma as f32, xs_f32.len() as u32);
    let out_f32 = dispatch_with_params_f32(NORMAL_CDF_WGSL, &xs_f32, &params)?;
    Ok(out_f32.iter().map(|&v| v as f64).collect())
}

/// Attempt batch Exponential log-PDF evaluation via WebGPU.
#[cfg(feature = "gpu_wgpu")]
fn exponential_log_pdf_wgpu(xs: &[f64], lambda: f64) -> Result<Vec<f64>, GpuStatsError> {
    let xs_f32: Vec<f32> = xs.iter().map(|&v| v as f32).collect();
    let params = encode_expon_params(lambda as f32, xs_f32.len() as u32);
    let out_f32 = dispatch_with_params_f32(EXPONENTIAL_LOG_PDF_WGSL, &xs_f32, &params)?;
    Ok(out_f32.iter().map(|&v| v as f64).collect())
}

/// Attempt batch Exponential CDF evaluation via WebGPU.
#[cfg(feature = "gpu_wgpu")]
fn exponential_cdf_wgpu(xs: &[f64], lambda: f64) -> Result<Vec<f64>, GpuStatsError> {
    let xs_f32: Vec<f32> = xs.iter().map(|&v| v as f32).collect();
    let params = encode_expon_params(lambda as f32, xs_f32.len() as u32);
    let out_f32 = dispatch_with_params_f32(EXPONENTIAL_CDF_WGSL, &xs_f32, &params)?;
    Ok(out_f32.iter().map(|&v| v as f64).collect())
}

// ---------------------------------------------------------------------------
// CPU implementations
// ---------------------------------------------------------------------------

/// Compute the error function using the A&S 7.1.26 polynomial approximation.
///
/// Maximum absolute error: ≈ 1.5 × 10⁻⁷.
#[inline]
fn erf_cpu(x: f64) -> f64 {
    // Handle negative input by symmetry
    if x < 0.0 {
        return -erf_cpu(-x);
    }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    // Horner evaluation of A&S 7.1.26
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    1.0 - poly * (-x * x).exp()
}

/// Compute the standard-normal CDF Φ(z) = 0.5·(1 + erf(z/√2)).
#[inline]
fn phi_cpu(z: f64) -> f64 {
    0.5 * (1.0 + erf_cpu(z / std::f64::consts::SQRT_2))
}

/// CPU scalar Normal log-PDF.
#[inline]
fn normal_log_pdf_scalar(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    -0.5 * z * z - (2.0 * std::f64::consts::PI).sqrt().ln() - sigma.ln()
}

/// CPU scalar Normal CDF.
#[inline]
fn normal_cdf_scalar(x: f64, mu: f64, sigma: f64) -> f64 {
    phi_cpu((x - mu) / sigma)
}

/// CPU scalar Exponential log-PDF.
///
/// Returns `f64::NEG_INFINITY` (represented as a very large negative number)
/// for `x < 0`.
#[inline]
fn exponential_log_pdf_scalar(x: f64, lambda: f64) -> f64 {
    if x < 0.0 {
        f64::NEG_INFINITY
    } else {
        lambda.ln() - lambda * x
    }
}

/// CPU scalar Exponential CDF.
#[inline]
fn exponential_cdf_scalar(x: f64, lambda: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        1.0 - (-lambda * x).exp()
    }
}

/// Minimum array length to trigger GPU dispatch.
/// Arrays smaller than this always use the CPU path (GPU launch overhead
/// is not worth it for short inputs, and CPU delivers full f64 precision).
const MIN_GPU_SIZE: usize = 1024;

// ---------------------------------------------------------------------------
// Public batch API
// ---------------------------------------------------------------------------

/// Batch compute Normal log-PDF for an array of `x` values.
///
/// Computes `log_pdf(xᵢ; µ, σ) = -½·((xᵢ−µ)/σ)² − ln(σ) − ln(√(2π))`
/// for each element.
///
/// When the `gpu_wgpu` feature is enabled and a compatible GPU is available,
/// the computation is dispatched to a WGSL compute shader; otherwise the
/// function silently falls back to a vectorised CPU loop.
///
/// # Arguments
///
/// * `xs`    — Input points at which to evaluate the log-PDF.
/// * `mu`    — Distribution mean µ.
/// * `sigma` — Distribution standard deviation σ (must be positive).
///
/// # Returns
///
/// A `Vec<f64>` of the same length as `xs`.
pub fn normal_log_pdf_batch(xs: &[f64], mu: f64, sigma: f64) -> Vec<f64> {
    #[cfg(feature = "gpu_wgpu")]
    {
        if xs.len() >= MIN_GPU_SIZE {
            if let Ok(result) = normal_log_pdf_wgpu(xs, mu, sigma) {
                return result;
            }
        }
    }
    xs.iter()
        .map(|&x| normal_log_pdf_scalar(x, mu, sigma))
        .collect()
}

/// Batch compute Normal CDF for an array of `x` values.
///
/// Computes `Φ((xᵢ−µ)/σ)` where Φ is the standard-normal CDF.
///
/// When the `gpu_wgpu` feature is enabled and a compatible GPU is available,
/// computation is dispatched to a WGSL compute shader; otherwise the
/// function falls back to a CPU loop using the A&S erf approximation.
///
/// # Arguments
///
/// * `xs`    — Input points at which to evaluate the CDF.
/// * `mu`    — Distribution mean µ.
/// * `sigma` — Distribution standard deviation σ (must be positive).
///
/// # Returns
///
/// A `Vec<f64>` of the same length as `xs`, with values in `[0, 1]`.
pub fn normal_cdf_batch(xs: &[f64], mu: f64, sigma: f64) -> Vec<f64> {
    #[cfg(feature = "gpu_wgpu")]
    {
        if xs.len() >= MIN_GPU_SIZE {
            if let Ok(result) = normal_cdf_wgpu(xs, mu, sigma) {
                return result;
            }
        }
    }
    xs.iter()
        .map(|&x| normal_cdf_scalar(x, mu, sigma))
        .collect()
}

/// Batch compute Exponential log-PDF for an array of `x` values.
///
/// For `xᵢ ≥ 0`: `log_pdf(xᵢ; λ) = ln(λ) − λ·xᵢ`.
/// For `xᵢ < 0`: returns `f64::NEG_INFINITY`.
///
/// When the `gpu_wgpu` feature is enabled and a compatible GPU is available,
/// computation is dispatched to a WGSL compute shader; otherwise the
/// function falls back to a CPU loop.
///
/// # Arguments
///
/// * `xs`     — Input points at which to evaluate the log-PDF.
/// * `lambda` — Rate parameter λ (must be positive).
///
/// # Returns
///
/// A `Vec<f64>` of the same length as `xs`.
pub fn exponential_log_pdf_batch(xs: &[f64], lambda: f64) -> Vec<f64> {
    #[cfg(feature = "gpu_wgpu")]
    {
        if xs.len() >= MIN_GPU_SIZE {
            if let Ok(result) = exponential_log_pdf_wgpu(xs, lambda) {
                return result;
            }
        }
    }
    xs.iter()
        .map(|&x| exponential_log_pdf_scalar(x, lambda))
        .collect()
}

/// Batch compute Exponential CDF for an array of `x` values.
///
/// For `xᵢ ≥ 0`: `CDF(xᵢ; λ) = 1 − exp(−λ·xᵢ)`.
/// For `xᵢ < 0`: returns `0.0`.
///
/// When the `gpu_wgpu` feature is enabled and a compatible GPU is available,
/// computation is dispatched to a WGSL compute shader; otherwise the
/// function falls back to a CPU loop.
///
/// # Arguments
///
/// * `xs`     — Input points at which to evaluate the CDF.
/// * `lambda` — Rate parameter λ (must be positive).
///
/// # Returns
///
/// A `Vec<f64>` of the same length as `xs`, with values in `[0, 1]`.
pub fn exponential_cdf_batch(xs: &[f64], lambda: f64) -> Vec<f64> {
    #[cfg(feature = "gpu_wgpu")]
    {
        if xs.len() >= MIN_GPU_SIZE {
            if let Ok(result) = exponential_cdf_wgpu(xs, lambda) {
                return result;
            }
        }
    }
    xs.iter()
        .map(|&x| exponential_cdf_scalar(x, lambda))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// ln(√(2π))
    fn log_sqrt_2pi() -> f64 {
        (2.0 * std::f64::consts::PI).sqrt().ln()
    }

    #[test]
    fn test_normal_log_pdf_batch_cpu() {
        let xs = vec![0.0_f64, 1.0, -1.0, 2.0];
        let result = normal_log_pdf_batch(&xs, 0.0, 1.0);
        let lsp = log_sqrt_2pi();
        assert_eq!(result.len(), xs.len());
        for (r, &x) in result.iter().zip(xs.iter()) {
            let expected = -0.5 * x * x - lsp;
            assert!(
                (r - expected).abs() < 1e-10,
                "normal_log_pdf mismatch at x={x}: got {r}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_normal_log_pdf_batch_nonstandard() {
        // Verify that mu/sigma parameters are applied correctly.
        let xs = vec![2.0_f64, 3.0, 4.0];
        let mu = 2.0;
        let sigma = 2.0;
        let result = normal_log_pdf_batch(&xs, mu, sigma);
        let lsp = log_sqrt_2pi();
        for (r, &x) in result.iter().zip(xs.iter()) {
            let z = (x - mu) / sigma;
            let expected = -0.5 * z * z - lsp - sigma.ln();
            assert!(
                (r - expected).abs() < 1e-10,
                "nonstandard normal_log_pdf mismatch at x={x}"
            );
        }
    }

    #[test]
    fn test_normal_log_pdf_batch_empty() {
        let result = normal_log_pdf_batch(&[], 0.0, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_normal_cdf_batch_cpu() {
        let xs = vec![-1e6_f64, -1.0, 0.0, 1.0, 1e6_f64];
        let result = normal_cdf_batch(&xs, 0.0, 1.0);
        assert_eq!(result.len(), xs.len());
        // Φ(−∞) ≈ 0
        assert!(result[0] < 1e-6, "Φ(-1e6) should be ~0, got {}", result[0]);
        // Φ(+∞) ≈ 1
        assert!(
            result[4] > 1.0 - 1e-6,
            "Φ(+1e6) should be ~1, got {}",
            result[4]
        );
        // Φ(0) ≈ 0.5; the A&S polynomial residual at x=0 gives ~5e-10 error
        assert!(
            (result[2] - 0.5).abs() < 1e-8,
            "Φ(0) should be 0.5, got {}",
            result[2]
        );
        // Φ(-1) ≈ 0.1587 (to within 0.001)
        assert!(
            (result[1] - 0.158_655_253_931_457_05).abs() < 1e-3,
            "Φ(-1) should be ≈0.1587, got {}",
            result[1]
        );
        // Φ(1) ≈ 0.8413
        assert!(
            (result[3] - 0.841_344_746_068_543).abs() < 1e-3,
            "Φ(1) should be ≈0.8413, got {}",
            result[3]
        );
    }

    #[test]
    fn test_normal_cdf_batch_symmetry() {
        // Φ(-z) + Φ(z) = 1 for any z ≠ 0
        // At z=0 the polynomial residual prevents exact cancellation; use 1e-8.
        let xs = vec![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
        let result = normal_cdf_batch(&xs, 0.0, 1.0);
        // Φ(-2) + Φ(2) = 1
        assert!(
            (result[0] + result[4] - 1.0).abs() < 1e-7,
            "Φ(-2)+Φ(2) should be 1, got {}",
            result[0] + result[4]
        );
        // Φ(-1) + Φ(1) = 1
        assert!(
            (result[1] + result[3] - 1.0).abs() < 1e-7,
            "Φ(-1)+Φ(1) should be 1, got {}",
            result[1] + result[3]
        );
        // Φ(0) ≈ 0.5 (A&S polynomial residual gives ~5e-10 at x=0)
        assert!(
            (result[2] - 0.5).abs() < 1e-8,
            "Φ(0) should be ~0.5, got {}",
            result[2]
        );
    }

    #[test]
    fn test_normal_cdf_batch_empty() {
        let result = normal_cdf_batch(&[], 0.0, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_exponential_log_pdf_batch_cpu() {
        let xs = vec![0.0_f64, 1.0, 2.0, -1.0];
        let lambda = 2.0_f64;
        let result = exponential_log_pdf_batch(&xs, lambda);
        assert_eq!(result.len(), xs.len());

        // log_pdf(0; λ=2) = ln(2)
        assert!(
            (result[0] - lambda.ln()).abs() < 1e-10,
            "log_pdf(0) should be ln(2), got {}",
            result[0]
        );
        // log_pdf(1; λ=2) = ln(2) - 2
        let expected_1 = lambda.ln() - lambda * 1.0;
        assert!(
            (result[1] - expected_1).abs() < 1e-10,
            "log_pdf(1) should be {expected_1}, got {}",
            result[1]
        );
        // log_pdf(2; λ=2) = ln(2) - 4
        let expected_2 = lambda.ln() - lambda * 2.0;
        assert!(
            (result[2] - expected_2).abs() < 1e-10,
            "log_pdf(2) should be {expected_2}, got {}",
            result[2]
        );
        // x < 0 → -inf
        assert!(
            result[3] < -1e20,
            "log_pdf(-1) should be -inf, got {}",
            result[3]
        );
    }

    #[test]
    fn test_exponential_log_pdf_batch_unit_rate() {
        // λ = 1: log_pdf(x) = -x for x >= 0
        let xs: Vec<f64> = (0..=5).map(|i| i as f64).collect();
        let result = exponential_log_pdf_batch(&xs, 1.0);
        for (i, (&x, &r)) in xs.iter().zip(result.iter()).enumerate() {
            let expected = -x; // ln(1) - 1*x = -x
            assert!(
                (r - expected).abs() < 1e-10,
                "unit-rate log_pdf mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_exponential_log_pdf_batch_empty() {
        let result = exponential_log_pdf_batch(&[], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_exponential_cdf_batch_cpu() {
        let xs = vec![0.0_f64, 1.0, -1.0];
        let result = exponential_cdf_batch(&xs, 1.0);
        assert_eq!(result.len(), xs.len());

        // CDF(0; λ=1) = 0
        assert!(
            (result[0] - 0.0).abs() < 1e-10,
            "CDF(0) should be 0, got {}",
            result[0]
        );
        // CDF(1; λ=1) = 1 - exp(-1)
        let expected_1 = 1.0 - (-1.0_f64).exp();
        assert!(
            (result[1] - expected_1).abs() < 1e-10,
            "CDF(1) should be {expected_1}, got {}",
            result[1]
        );
        // CDF(-1; λ=1) = 0
        assert!(
            (result[2] - 0.0).abs() < 1e-10,
            "CDF(-1) should be 0, got {}",
            result[2]
        );
    }

    #[test]
    fn test_exponential_cdf_batch_large_x() {
        // CDF(x; λ) → 1 as x → +∞
        let xs = vec![100.0_f64, 1000.0];
        let result = exponential_cdf_batch(&xs, 1.0);
        assert!(result[0] > 1.0 - 1e-10);
        assert!(result[1] > 1.0 - 1e-10);
    }

    #[test]
    fn test_exponential_cdf_batch_empty() {
        let result = exponential_cdf_batch(&[], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_erf_cpu_symmetry() {
        // erf is an odd function: erf(-x) = -erf(x)
        for &x in &[0.5_f64, 1.0, 1.5, 2.0, 3.0] {
            let pos = erf_cpu(x);
            let neg = erf_cpu(-x);
            assert!(
                (pos + neg).abs() < 1e-12,
                "erf symmetry failed at x={x}: erf(x)={pos}, erf(-x)={neg}"
            );
        }
    }

    #[test]
    fn test_erf_cpu_known_values() {
        // erf(0) = 0: the A&S polynomial has a residual of ~1e-9 at x=0
        // due to the polynomial not cancelling exactly; use a generous tolerance.
        assert!(
            erf_cpu(0.0).abs() < 1e-8,
            "erf(0) should be ~0, got {}",
            erf_cpu(0.0)
        );
        // erf(1) ≈ 0.8427007929; A&S 7.1.26 has max absolute error ~1.5e-7
        assert!(
            (erf_cpu(1.0) - 0.842_700_792_949_715).abs() < 2e-7,
            "erf(1) mismatch: {}",
            erf_cpu(1.0)
        );
        // erf(2) ≈ 0.9953222650; same tolerance
        assert!(
            (erf_cpu(2.0) - 0.995_322_265_018_953).abs() < 2e-7,
            "erf(2) mismatch: {}",
            erf_cpu(2.0)
        );
    }

    // ── GPU test (skipped when wgpu adapter unavailable) ─────────────────────

    #[cfg(feature = "gpu_wgpu")]
    #[test]
    fn test_normal_log_pdf_wgpu_or_skip() {
        let xs = vec![0.0_f64, 1.0, -1.0];
        let gpu_result = normal_log_pdf_wgpu(&xs, 0.0, 1.0);
        match gpu_result {
            Err(GpuStatsError::GpuNotAvailable) => {
                // No GPU adapter available — acceptable in CI
                eprintln!("test_normal_log_pdf_wgpu_or_skip: GPU not available, skipping");
            }
            Err(e) => panic!("GPU error: {e}"),
            Ok(gpu) => {
                let cpu: Vec<f64> = xs
                    .iter()
                    .map(|&x| normal_log_pdf_scalar(x, 0.0, 1.0))
                    .collect();
                for (g, c) in gpu.iter().zip(cpu.iter()) {
                    // f32 GPU vs f64 CPU: allow 1e-4 relative tolerance
                    assert!((g - c).abs() < 1e-4, "GPU/CPU mismatch: gpu={g}, cpu={c}");
                }
            }
        }
    }

    #[cfg(feature = "gpu_wgpu")]
    #[test]
    fn test_normal_cdf_wgpu_or_skip() {
        let xs = vec![-1.0_f64, 0.0, 1.0];
        let gpu_result = normal_cdf_wgpu(&xs, 0.0, 1.0);
        match gpu_result {
            Err(GpuStatsError::GpuNotAvailable) => {
                eprintln!("test_normal_cdf_wgpu_or_skip: GPU not available, skipping");
            }
            Err(e) => panic!("GPU error: {e}"),
            Ok(gpu) => {
                let cpu: Vec<f64> = xs.iter().map(|&x| normal_cdf_scalar(x, 0.0, 1.0)).collect();
                for (g, c) in gpu.iter().zip(cpu.iter()) {
                    assert!((g - c).abs() < 1e-4, "GPU/CPU mismatch: gpu={g}, cpu={c}");
                }
            }
        }
    }

    #[cfg(feature = "gpu_wgpu")]
    #[test]
    fn test_exponential_log_pdf_wgpu_or_skip() {
        let xs = vec![0.0_f64, 1.0, 2.0];
        let lambda = 2.0_f64;
        let gpu_result = exponential_log_pdf_wgpu(&xs, lambda);
        match gpu_result {
            Err(GpuStatsError::GpuNotAvailable) => {
                eprintln!("test_exponential_log_pdf_wgpu_or_skip: GPU not available, skipping");
            }
            Err(e) => panic!("GPU error: {e}"),
            Ok(gpu) => {
                let cpu: Vec<f64> = xs
                    .iter()
                    .map(|&x| exponential_log_pdf_scalar(x, lambda))
                    .collect();
                for (g, c) in gpu.iter().zip(cpu.iter()) {
                    assert!((g - c).abs() < 1e-4, "GPU/CPU mismatch: gpu={g}, cpu={c}");
                }
            }
        }
    }

    #[cfg(feature = "gpu_wgpu")]
    #[test]
    fn test_exponential_cdf_wgpu_or_skip() {
        let xs = vec![0.0_f64, 1.0, 2.0];
        let gpu_result = exponential_cdf_wgpu(&xs, 1.0);
        match gpu_result {
            Err(GpuStatsError::GpuNotAvailable) => {
                eprintln!("test_exponential_cdf_wgpu_or_skip: GPU not available, skipping");
            }
            Err(e) => panic!("GPU error: {e}"),
            Ok(gpu) => {
                let cpu: Vec<f64> = xs.iter().map(|&x| exponential_cdf_scalar(x, 1.0)).collect();
                for (g, c) in gpu.iter().zip(cpu.iter()) {
                    assert!((g - c).abs() < 1e-4, "GPU/CPU mismatch: gpu={g}, cpu={c}");
                }
            }
        }
    }
}
