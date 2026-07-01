//! WGSL compute-shader kernels for WebGPU-backed dispatch of batch special functions.
//!
//! Each constant holds the WGSL source for a `@compute` shader that evaluates a batch
//! of special-function values.  The shaders operate on `array<f32>` — inputs must be
//! cast from `f64` before upload and cast back after download.
//!
//! The host-side dispatch functions (`gamma_batch_wgpu`, `erf_batch_wgpu`,
//! `bessel_j0_batch_wgpu`, `lgamma_batch_wgpu`) perform real GPU dispatch when the
//! `wgpu` feature is enabled and a wgpu adapter is found.  When the feature
//! is disabled, or no adapter is available at runtime, they return
//! [`WgslDispatchError::GpuNotAvailable`] so the caller can fall back to CPU.
//!
//! # Feature gating
//!
//! The WGSL shader source constants are always compiled (useful for documentation and
//! validation tooling).  The GPU dispatch paths are gated behind
//! `#[cfg(feature = "wgpu")]`.

// ---------------------------------------------------------------------------
// WGSL shader sources
// ---------------------------------------------------------------------------

/// WGSL compute shader for batch Gamma evaluation (Lanczos g=7 approximation).
///
/// Workgroup size 64.  Each invocation reads one `f32` from `input` and
/// writes the approximated `Γ(x)` into `output`.
/// The reflection formula `Γ(x) = π / (sin(π x) · Γ(1-x))` is applied when
/// `x < 0.5`.
pub const GAMMA_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const PI: f32 = 3.14159265358979323846;

// Lanczos g=7 coefficients (Spouge's form, 9 terms)
fn lanczos_gamma(x_in: f32) -> f32 {
    var x = x_in;
    var sign = 1.0f;
    if x < 0.5 {
        sign = PI / (sin(PI * x));
        x = 1.0 - x;
    }
    let g: f32 = 7.0;
    x = x - 1.0;

    let c0: f32 =  0.99999999999980993;
    let c1: f32 =  676.5203681218851;
    let c2: f32 = -1259.1392167224028;
    let c3: f32 =  771.32342877765313;
    let c4: f32 = -176.61502916214059;
    let c5: f32 =  12.507343278686905;
    let c6: f32 = -0.13857109526572012;
    let c7: f32 =  9.9843695780195716e-6;
    let c8: f32 =  1.5056327351493116e-7;

    let s = c0
        + c1 / (x + 1.0)
        + c2 / (x + 2.0)
        + c3 / (x + 3.0)
        + c4 / (x + 4.0)
        + c5 / (x + 5.0)
        + c6 / (x + 6.0)
        + c7 / (x + 7.0)
        + c8 / (x + 8.0);

    let t = x + g + 0.5;
    let result = sqrt(2.0 * PI) * pow(t, x + 0.5) * exp(-t) * s;
    if sign != 1.0 { return sign / result; }
    return result;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = lanczos_gamma(input[idx]);
}
"#;

/// WGSL compute shader for batch `erf` evaluation.
///
/// Uses the Abramowitz & Stegun 7.1.26 approximation (max error ≈ 1.5 × 10⁻⁷).
pub const ERF_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

fn approx_erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * abs(x));
    let y = 1.0 - (((((
          1.061405429 * t
        - 1.453152027) * t
        + 1.421413741) * t
        - 0.284496736) * t
        + 0.254829592) * t * exp(-x * x));
    return select(-y, y, x >= 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = approx_erf(input[idx]);
}
"#;

/// WGSL compute shader for batch Bessel J₀ evaluation.
///
/// Uses the polynomial approximation from Abramowitz & Stegun §9.4 for
/// |x| < 8 and the asymptotic expansion for |x| ≥ 8.
pub const BESSEL_J0_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const PI: f32 = 3.14159265358979323846;

fn bessel_j0(x_in: f32) -> f32 {
    let x = abs(x_in);
    if x < 8.0 {
        let y = x * x;
        let p1: f32 =  57568490574.0;
        let p2: f32 = -13362590354.0;
        let p3: f32 =  651619640.7;
        let p4: f32 = -11214424.18;
        let p5: f32 =  77392.33017;
        let p6: f32 = -184.9052456;
        let q1: f32 =  57568490411.0;
        let q2: f32 =  1029532985.0;
        let q3: f32 =  9494680.718;
        let q4: f32 =  59272.64853;
        let q5: f32 =  267.8532712;
        let p = p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * p6))));
        let q = q1 + y * (q2 + y * (q3 + y * (q4 + y * (q5 + y))));
        return p / q;
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - 0.785398164;
        let pv = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
                 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let qv = -0.1562499995e-1 + y * (0.1430488765e-3
                 + y * (-0.6911147651e-5 + y * (0.7621095161e-6
                 - y * 0.934945152e-7)));
        return sqrt(0.636619772 / x) * (cos(xx) * pv - z * sin(xx) * qv);
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = bessel_j0(input[idx]);
}
"#;

/// WGSL compute shader for batch `erfc` evaluation.
///
/// Computes `erfc(x) = 1 - erf(x)` using the same Abramowitz & Stegun 7.1.26
/// rational approximation as [`ERF_WGSL`].  The subtraction `1 - approx_erf(x)`
/// is used throughout; edge cases at |x| > 6 short-circuit to exact 0 or 2.
pub const ERFC_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

fn approx_erf_inner(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * abs(x));
    let y = 1.0 - (((((
          1.061405429 * t
        - 1.453152027) * t
        + 1.421413741) * t
        - 0.284496736) * t
        + 0.254829592) * t * exp(-x * x));
    return select(-y, y, x >= 0.0);
}

fn approx_erfc(x: f32) -> f32 {
    // erfc saturates quickly: |erfc(x)| < f32_epsilon for |x| > ~6
    if abs(x) > 6.0 {
        return select(0.0, 2.0, x < 0.0);
    }
    return 1.0 - approx_erf_inner(x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = approx_erfc(input[idx]);
}
"#;

/// WGSL compute shader for batch inverse-erf evaluation.
///
/// Uses the Winitzki (2008) rational approximation to `erfinv`, which achieves
/// a maximum absolute error of approximately 5 × 10⁻⁴ for |p| < 1 in f32.
/// Inputs with |p| ≥ 1 return ±1 × 10¹⁰ (representable large f32 values).
pub const ERFINV_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const PI_F: f32 = 3.14159265358979323846;
const WINITZKI_A: f32 = 0.147;
const INV_WINITZKI_A: f32 = 6.802721088;  // 1.0 / 0.147

fn approx_erfinv(p: f32) -> f32 {
    let ap = abs(p);
    if ap >= 1.0 {
        // Return signed large value for |p| = 1 boundary
        return select(1e10, -1e10, p < 0.0);
    }
    if p == 0.0 {
        return 0.0;
    }

    let sign_p = select(-1.0f, 1.0f, p >= 0.0);
    // Winitzki (2008): erfinv(p) ≈ sign(p) * sqrt(sqrt(c^2 - ln(1-p^2)/a) - c)
    // where c = 2/(π·a) + ln(1-p^2)/2
    let ln_term = log(1.0 - p * p);
    let two_over_pia = 2.0 / (PI_F * WINITZKI_A);
    let c = two_over_pia + ln_term * 0.5;
    let discriminant = c * c - ln_term * INV_WINITZKI_A;
    // discriminant is always non-negative for |p| < 1
    let inner = sqrt(max(discriminant, 0.0)) - c;
    return sign_p * sqrt(max(inner, 0.0));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = approx_erfinv(input[idx]);
}
"#;

/// WGSL compute shader for batch log-gamma evaluation.
///
/// Uses the same Lanczos g=7 coefficients as [`GAMMA_WGSL`], but computes
/// `ln Γ(x)` directly to avoid overflow for large `x`.
/// The reflection formula `ln Γ(x) = ln(π/|sin(πx)|) - ln Γ(1-x)` handles `x < 0.5`.
pub const LGAMMA_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const PI: f32 = 3.14159265358979323846;

fn lanczos_lgamma(x_in: f32) -> f32 {
    var x = x_in;
    var log_sign: f32 = 0.0;
    if x < 0.5 {
        log_sign = log(PI / abs(sin(PI * x)));
        x = 1.0 - x;
    }
    let g: f32 = 7.0;
    x = x - 1.0;
    let c0: f32 =  0.99999999999980993;
    let c1: f32 =  676.5203681218851;
    let c2: f32 = -1259.1392167224028;
    let c3: f32 =  771.32342877765313;
    let c4: f32 = -176.61502916214059;
    let c5: f32 =  12.507343278686905;
    let c6: f32 = -0.13857109526572012;
    let c7: f32 =  9.9843695780195716e-6;
    let c8: f32 =  1.5056327351493116e-7;
    let s = c0 + c1/(x+1.0) + c2/(x+2.0) + c3/(x+3.0) + c4/(x+4.0)
              + c5/(x+5.0) + c6/(x+6.0) + c7/(x+7.0) + c8/(x+8.0);
    let t = x + g + 0.5;
    let lgamma = 0.5 * log(2.0 * PI) + (x + 0.5) * log(t) - t + log(s);
    if log_sign != 0.0 {
        return log_sign - lgamma;
    }
    return lgamma;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = lanczos_lgamma(input[idx]);
}
"#;

// ---------------------------------------------------------------------------
// Dispatch error
// ---------------------------------------------------------------------------

/// Error type for WGSL/WebGPU dispatch.
#[derive(Debug, Clone)]
pub enum WgslDispatchError {
    /// No wgpu device is available (headless or non-WASM build).
    GpuNotAvailable,
    /// The wgpu pipeline setup or execution failed.
    RuntimeError(String),
}

impl std::fmt::Display for WgslDispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WgslDispatchError::GpuNotAvailable => {
                write!(f, "wgpu GPU device not available")
            }
            WgslDispatchError::RuntimeError(msg) => {
                write!(f, "wgpu runtime error: {msg}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Real GPU dispatch (wgpu feature)
// ---------------------------------------------------------------------------

/// Upload `xs_f32` to a read-only storage buffer, dispatch `shader_src` over it,
/// and return the resulting `f32` values.
///
/// This is the shared implementation used by all four batch dispatch functions.
/// The shader is expected to have exactly two bindings:
///   - `@group(0) @binding(0)` — read-only input `array<f32>`
///   - `@group(0) @binding(1)` — read-write output `array<f32>`
#[cfg(feature = "wgpu")]
fn dispatch_unary_f32(shader_src: &str, xs_f32: &[f32]) -> Result<Vec<f32>, WgslDispatchError> {
    use wgpu::{
        util::BufferInitDescriptor, util::DeviceExt as _, Backends, BindGroupDescriptor,
        BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
        BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
        ComputePassDescriptor, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
        MapMode, PowerPreference, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
        ShaderStages,
    };

    let n = xs_f32.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // ── Adapter / device acquisition ──────────────────────────────────────────
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
    .map_err(|_| WgslDispatchError::GpuNotAvailable)?;

    let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("scirs2-special"),
        required_features: Features::empty(),
        required_limits: Limits::default(),
        ..Default::default()
    }))
    .map_err(|e| WgslDispatchError::RuntimeError(e.to_string()))?;

    // ── Shader / pipeline ─────────────────────────────────────────────────────
    let shader_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("scirs2-special-shader"),
        source: ShaderSource::Wgsl(shader_src.into()),
    });

    let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("scirs2-special-bgl"),
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
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("scirs2-special-layout"),
        bind_group_layouts: &[Some(&bgl)],
        ..Default::default()
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("scirs2-special-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // ── Buffers ───────────────────────────────────────────────────────────────
    // Encode f32 slice to raw bytes manually (avoids bytemuck dependency)
    let input_bytes: Vec<u8> = xs_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let byte_len = (n * 4) as u64;

    let buf_input = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("scirs2-special-input"),
        contents: &input_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    let buf_output = device.create_buffer(&BufferDescriptor {
        label: Some("scirs2-special-output"),
        size: byte_len,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let buf_staging = device.create_buffer(&BufferDescriptor {
        label: Some("scirs2-special-staging"),
        size: byte_len,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ── Bind group ────────────────────────────────────────────────────────────
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("scirs2-special-bg"),
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
        ],
    });

    // ── Encode / dispatch ─────────────────────────────────────────────────────
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("scirs2-special-encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("scirs2-special-pass"),
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
        .map_err(|e| WgslDispatchError::RuntimeError(format!("GPU poll error: {e:?}")))?;

    let slice = buf_staging.slice(0..byte_len);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(MapMode::Read, move |r| {
        let _ = tx.send(r);
    });

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| WgslDispatchError::RuntimeError(format!("GPU poll during map: {e:?}")))?;

    rx.recv()
        .map_err(|_| WgslDispatchError::RuntimeError("channel closed in map_async".into()))?
        .map_err(|e| WgslDispatchError::RuntimeError(format!("map_async failed: {e:?}")))?;

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
// Host-side dispatch functions — real (wgpu) path
// ---------------------------------------------------------------------------

/// Attempt batch Gamma evaluation on a WebGPU device.
///
/// When `wgpu` is enabled and a wgpu adapter is found, uploads `xs` as
/// `f32`, dispatches the Lanczos WGSL shader, and returns `f64` results.
/// Falls back to [`WgslDispatchError::GpuNotAvailable`] otherwise.
#[cfg(feature = "wgpu")]
pub fn gamma_batch_wgpu(xs: &[f64]) -> Result<Vec<f64>, WgslDispatchError> {
    let xs_f32: Vec<f32> = xs.iter().map(|&x| x as f32).collect();
    let result_f32 = dispatch_unary_f32(GAMMA_WGSL, &xs_f32)?;
    Ok(result_f32.iter().map(|&v| v as f64).collect())
}

/// Attempt batch `erf` evaluation on a WebGPU device.
///
/// When `wgpu` is enabled and a wgpu adapter is found, uploads `xs` as
/// `f32`, dispatches the A&S WGSL erf shader, and returns `f64` results.
#[cfg(feature = "wgpu")]
pub fn erf_batch_wgpu(xs: &[f64]) -> Result<Vec<f64>, WgslDispatchError> {
    let xs_f32: Vec<f32> = xs.iter().map(|&x| x as f32).collect();
    let result_f32 = dispatch_unary_f32(ERF_WGSL, &xs_f32)?;
    Ok(result_f32.iter().map(|&v| v as f64).collect())
}

/// Attempt batch Bessel J₀ evaluation on a WebGPU device.
///
/// When `wgpu` is enabled and a wgpu adapter is found, uploads `xs` as
/// `f32`, dispatches the A&S polynomial WGSL shader, and returns `f64` results.
#[cfg(feature = "wgpu")]
pub fn bessel_j0_batch_wgpu(xs: &[f64]) -> Result<Vec<f64>, WgslDispatchError> {
    let xs_f32: Vec<f32> = xs.iter().map(|&x| x as f32).collect();
    let result_f32 = dispatch_unary_f32(BESSEL_J0_WGSL, &xs_f32)?;
    Ok(result_f32.iter().map(|&v| v as f64).collect())
}

/// Attempt batch log-Gamma evaluation on a WebGPU device.
///
/// When `wgpu` is enabled and a wgpu adapter is found, uploads `xs` as
/// `f32`, dispatches the Lanczos log-gamma WGSL shader, and returns `f64` results.
#[cfg(feature = "wgpu")]
pub fn lgamma_batch_wgpu(xs: &[f64]) -> Result<Vec<f64>, WgslDispatchError> {
    let xs_f32: Vec<f32> = xs.iter().map(|&x| x as f32).collect();
    let result_f32 = dispatch_unary_f32(LGAMMA_WGSL, &xs_f32)?;
    Ok(result_f32.iter().map(|&v| v as f64).collect())
}

/// Attempt batch `erfc` evaluation on a WebGPU device.
///
/// When `wgpu` is enabled and a wgpu adapter is found, uploads `xs` as
/// `f32`, dispatches the A&S erfc WGSL shader, and returns `f64` results.
/// Falls back to [`WgslDispatchError::GpuNotAvailable`] otherwise.
#[cfg(feature = "wgpu")]
pub fn erfc_batch_wgpu(xs: &[f64]) -> Result<Vec<f64>, WgslDispatchError> {
    let xs_f32: Vec<f32> = xs.iter().map(|&x| x as f32).collect();
    let result_f32 = dispatch_unary_f32(ERFC_WGSL, &xs_f32)?;
    Ok(result_f32.iter().map(|&v| v as f64).collect())
}

/// Attempt batch `erfinv` evaluation on a WebGPU device.
///
/// When `wgpu` is enabled and a wgpu adapter is found, uploads `xs` as
/// `f32`, dispatches the Winitzki erfinv WGSL shader, and returns `f64` results.
/// Falls back to [`WgslDispatchError::GpuNotAvailable`] otherwise.
#[cfg(feature = "wgpu")]
pub fn erfinv_batch_wgpu(xs: &[f64]) -> Result<Vec<f64>, WgslDispatchError> {
    let xs_f32: Vec<f32> = xs.iter().map(|&x| x as f32).collect();
    let result_f32 = dispatch_unary_f32(ERFINV_WGSL, &xs_f32)?;
    Ok(result_f32.iter().map(|&v| v as f64).collect())
}

// ---------------------------------------------------------------------------
// Host-side dispatch functions — stub (no wgpu) path
// ---------------------------------------------------------------------------

/// Stub: returns [`WgslDispatchError::GpuNotAvailable`] when `wgpu` is off.
#[cfg(not(feature = "wgpu"))]
pub fn gamma_batch_wgpu(_xs: &[f64]) -> Result<Vec<f64>, WgslDispatchError> {
    Err(WgslDispatchError::GpuNotAvailable)
}

/// Stub: returns [`WgslDispatchError::GpuNotAvailable`] when `wgpu` is off.
#[cfg(not(feature = "wgpu"))]
pub fn erf_batch_wgpu(_xs: &[f64]) -> Result<Vec<f64>, WgslDispatchError> {
    Err(WgslDispatchError::GpuNotAvailable)
}

/// Stub: returns [`WgslDispatchError::GpuNotAvailable`] when `wgpu` is off.
#[cfg(not(feature = "wgpu"))]
pub fn bessel_j0_batch_wgpu(_xs: &[f64]) -> Result<Vec<f64>, WgslDispatchError> {
    Err(WgslDispatchError::GpuNotAvailable)
}

/// Stub: returns [`WgslDispatchError::GpuNotAvailable`] when `wgpu` is off.
#[cfg(not(feature = "wgpu"))]
pub fn lgamma_batch_wgpu(_xs: &[f64]) -> Result<Vec<f64>, WgslDispatchError> {
    Err(WgslDispatchError::GpuNotAvailable)
}

/// Stub: returns [`WgslDispatchError::GpuNotAvailable`] when `wgpu` is off.
#[cfg(not(feature = "wgpu"))]
pub fn erfc_batch_wgpu(_xs: &[f64]) -> Result<Vec<f64>, WgslDispatchError> {
    Err(WgslDispatchError::GpuNotAvailable)
}

/// Stub: returns [`WgslDispatchError::GpuNotAvailable`] when `wgpu` is off.
#[cfg(not(feature = "wgpu"))]
pub fn erfinv_batch_wgpu(_xs: &[f64]) -> Result<Vec<f64>, WgslDispatchError> {
    Err(WgslDispatchError::GpuNotAvailable)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_wgsl_source_is_non_empty() {
        assert!(!GAMMA_WGSL.is_empty());
        assert!(GAMMA_WGSL.contains("@compute"));
        assert!(GAMMA_WGSL.contains("workgroup_size"));
        assert!(GAMMA_WGSL.contains("lanczos_gamma"));
    }

    #[test]
    fn test_erf_wgsl_source_is_non_empty() {
        assert!(!ERF_WGSL.is_empty());
        assert!(ERF_WGSL.contains("@compute"));
        assert!(ERF_WGSL.contains("approx_erf"));
    }

    #[test]
    fn test_bessel_j0_wgsl_source_is_non_empty() {
        assert!(!BESSEL_J0_WGSL.is_empty());
        assert!(BESSEL_J0_WGSL.contains("@compute"));
        assert!(BESSEL_J0_WGSL.contains("bessel_j0"));
    }

    #[test]
    fn test_lgamma_wgsl_source_is_non_empty() {
        assert!(!LGAMMA_WGSL.is_empty());
        assert!(LGAMMA_WGSL.contains("@compute"));
        assert!(LGAMMA_WGSL.contains("lanczos_lgamma"));
    }

    #[test]
    fn test_erfc_wgsl_source_is_non_empty() {
        assert!(!ERFC_WGSL.is_empty());
        assert!(ERFC_WGSL.contains("@compute"));
        assert!(ERFC_WGSL.contains("approx_erfc"));
        assert!(ERFC_WGSL.contains("workgroup_size"));
    }

    #[test]
    fn test_erfinv_wgsl_source_is_non_empty() {
        assert!(!ERFINV_WGSL.is_empty());
        assert!(ERFINV_WGSL.contains("@compute"));
        assert!(ERFINV_WGSL.contains("approx_erfinv"));
        assert!(ERFINV_WGSL.contains("workgroup_size"));
    }

    #[test]
    fn test_gamma_batch_wgpu_returns_not_available() {
        // Without wgpu feature, always returns GpuNotAvailable.
        // With wgpu feature on headless CI, also returns GpuNotAvailable.
        let xs = vec![1.0_f64, 2.0, 3.0];
        let result = gamma_batch_wgpu(&xs);
        // Either Ok (GPU present) or GpuNotAvailable (no GPU)
        match result {
            Ok(_) | Err(WgslDispatchError::GpuNotAvailable) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn test_erf_batch_wgpu_returns_not_available() {
        let xs = vec![0.0_f64, 1.0];
        let result = erf_batch_wgpu(&xs);
        match result {
            Ok(_) | Err(WgslDispatchError::GpuNotAvailable) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn test_bessel_j0_batch_wgpu_returns_not_available() {
        let xs = vec![0.0_f64, 2.405];
        let result = bessel_j0_batch_wgpu(&xs);
        match result {
            Ok(_) | Err(WgslDispatchError::GpuNotAvailable) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn test_lgamma_batch_wgpu_returns_not_available() {
        let xs = vec![1.0_f64, 2.0, 3.0];
        let result = lgamma_batch_wgpu(&xs);
        match result {
            Ok(_) | Err(WgslDispatchError::GpuNotAvailable) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn test_erfc_batch_wgpu_returns_not_available() {
        let xs = vec![0.0_f64, 1.0, -1.0];
        let result = erfc_batch_wgpu(&xs);
        match result {
            Ok(_) | Err(WgslDispatchError::GpuNotAvailable) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn test_erfinv_batch_wgpu_returns_not_available() {
        let xs = vec![0.0_f64, 0.5, -0.5];
        let result = erfinv_batch_wgpu(&xs);
        match result {
            Ok(_) | Err(WgslDispatchError::GpuNotAvailable) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn test_wgsl_dispatch_error_display() {
        let e = WgslDispatchError::GpuNotAvailable;
        assert!(e.to_string().contains("not available"));
        let e2 = WgslDispatchError::RuntimeError("buffer overflow".into());
        assert!(e2.to_string().contains("buffer overflow"));
    }
}
