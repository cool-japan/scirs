//! Real wgpu RBF kernel-matrix and evaluation dispatch.
//!
//! This module is compiled **only** when the `wgpu` feature is enabled.
//! It provides:
//!
//! - `is_gpu_available()` — one-time adapter probe cached in a `OnceLock<bool>`.
//! - `gpu_rbf_kernel_matrix()` — builds an N×M kernel matrix on the GPU.
//! - `gpu_rbf_evaluate()` — evaluates `sum_i coeff_i * kernel(||x - c_i||)`
//!   for all query points in parallel on the GPU.
//!
//! # f64 → f32 precision note
//!
//! WGSL (WebGPU Shading Language) does **not** support `f64` in storage
//! buffers.  All values are cast to `f32` before upload and cast back to `f64`
//! after readback.  Callers should expect approximately single-precision
//! accuracy (~1e-6 relative error) from the GPU path.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::{GpuRBFKernel, RbfGpuError};

// ─────────────────────────────────────────────────────────────────────────────
// WGSL shader sources
// ─────────────────────────────────────────────────────────────────────────────

/// GPU threshold: only dispatch to GPU when `n_centers * n_queries >= GPU_THRESHOLD`.
pub const GPU_THRESHOLD: usize = 4096;

/// Numeric ID used in the WGSL `kernel_id` uniform.
///
/// | Value | Kernel              |
/// |-------|---------------------|
/// | 0     | Gaussian            |
/// | 1     | Multiquadric        |
/// | 2     | InverseMultiquadric |
/// | 3     | Linear              |
/// | 4     | Cubic               |
/// | 5     | ThinPlate           |
pub fn kernel_id(kernel: GpuRBFKernel) -> u32 {
    match kernel {
        GpuRBFKernel::Gaussian => 0,
        GpuRBFKernel::Multiquadric => 1,
        GpuRBFKernel::InverseMultiquadric => 2,
        GpuRBFKernel::Linear => 3,
        GpuRBFKernel::Cubic => 4,
        GpuRBFKernel::ThinPlate => 5,
    }
}

/// WGSL compute shader: RBF kernel matrix construction.
///
/// Bindings:
/// - 0: `centers` (read, flat f32 array, length = n_centers)
/// - 1: `queries` (read, flat f32 array, length = n_queries)
/// - 2: `out_matrix` (read-write, flat f32 array, length = n_centers × n_queries; row = center index)
/// - 3: `params` (uniform struct `RbfKernelParams`)
///
/// Dispatched with `ceil(n_centers/16)` × `ceil(n_queries/16)` workgroups.
const RBF_KERNEL_MATRIX_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> centers    : array<f32>;
@group(0) @binding(1) var<storage, read> queries    : array<f32>;
@group(0) @binding(2) var<storage, read_write> out_matrix : array<f32>;

struct RbfKernelParams {
    n_centers  : u32,
    n_queries  : u32,
    kernel_id  : u32,
    epsilon    : f32,
};

@group(0) @binding(3) var<uniform> params : RbfKernelParams;

fn rbf_kernel(r: f32, kid: u32, eps: f32) -> f32 {
    let re = r / eps;
    if kid == 0u { // Gaussian
        return exp(-re * re);
    } else if kid == 1u { // Multiquadric
        return sqrt(1.0 + re * re);
    } else if kid == 2u { // InverseMultiquadric
        return 1.0 / sqrt(1.0 + re * re);
    } else if kid == 3u { // Linear
        return re;
    } else if kid == 4u { // Cubic
        return re * re * re;
    } else { // ThinPlate (kid == 5)
        if re > 0.0 {
            return re * re * log(re);
        }
        return 0.0;
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = gid.x; // center index
    let qi = gid.y; // query index
    if ci >= params.n_centers || qi >= params.n_queries {
        return;
    }
    let center = centers[ci];
    let query  = queries[qi];
    let r      = abs(center - query);
    let val    = rbf_kernel(r, params.kernel_id, params.epsilon);
    out_matrix[ci * params.n_queries + qi] = val;
}
"#;

/// WGSL compute shader: RBF evaluation (dot product of coefficients and kernel row).
///
/// Bindings:
/// - 0: `coefficients` (read, flat f32 array, length = n_centers)
/// - 1: `centers` (read, flat f32 array, length = n_centers)
/// - 2: `queries` (read, flat f32 array, length = n_queries)
/// - 3: `out_values` (read-write, flat f32 array, length = n_queries)
/// - 4: `params` (uniform struct `RbfEvalParams`)
///
/// Dispatched with `ceil(n_queries/64)` × 1 × 1 workgroups.
const RBF_EVALUATE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> coefficients : array<f32>;
@group(0) @binding(1) var<storage, read> centers      : array<f32>;
@group(0) @binding(2) var<storage, read> queries      : array<f32>;
@group(0) @binding(3) var<storage, read_write> out_values : array<f32>;

struct RbfEvalParams {
    n_centers  : u32,
    n_queries  : u32,
    kernel_id  : u32,
    epsilon    : f32,
};

@group(0) @binding(4) var<uniform> params : RbfEvalParams;

fn rbf_kernel(r: f32, kid: u32, eps: f32) -> f32 {
    let re = r / eps;
    if kid == 0u {
        return exp(-re * re);
    } else if kid == 1u {
        return sqrt(1.0 + re * re);
    } else if kid == 2u {
        return 1.0 / sqrt(1.0 + re * re);
    } else if kid == 3u {
        return re;
    } else if kid == 4u {
        return re * re * re;
    } else {
        if re > 0.0 {
            return re * re * log(re);
        }
        return 0.0;
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if qi >= params.n_queries {
        return;
    }
    let q = queries[qi];
    var acc = 0.0f;
    for (var ci = 0u; ci < params.n_centers; ci++) {
        let r   = abs(centers[ci] - q);
        let val = rbf_kernel(r, params.kernel_id, params.epsilon);
        acc += coefficients[ci] * val;
    }
    out_values[qi] = acc;
}
"#;

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline cache
// ─────────────────────────────────────────────────────────────────────────────

/// Compiled wgpu pipelines cached once per (kernel_id, shader_kind) combo.
struct CachedPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

// SAFETY: wgpu ComputePipeline and BindGroupLayout are Send + Sync on all backends.
unsafe impl Send for CachedPipeline {}
unsafe impl Sync for CachedPipeline {}

/// Key: (kernel_id, 0=kernel_matrix / 1=evaluate)
type PipelineKey = (u32, u8);

static PIPELINE_CACHE: OnceLock<Mutex<HashMap<PipelineKey, CachedPipeline>>> = OnceLock::new();

fn pipeline_cache() -> &'static Mutex<HashMap<PipelineKey, CachedPipeline>> {
    PIPELINE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU availability probe
// ─────────────────────────────────────────────────────────────────────────────

static GPU_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Returns `true` when a wgpu adapter is present on the current host.
///
/// The result is computed once and cached for the lifetime of the process.
/// This call may block for a short time on the first invocation while the
/// adapter enumeration completes.
pub fn is_gpu_available() -> bool {
    *GPU_AVAILABLE.get_or_init(|| probe_gpu())
}

fn probe_gpu() -> bool {
    use wgpu::{Backends, Instance, InstanceDescriptor, PowerPreference, RequestAdapterOptions};

    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    pollster::block_on(async {
        instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .is_ok()
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Uniform buffer layout helpers
// ─────────────────────────────────────────────────────────────────────────────

/// `RbfKernelParams` / `RbfEvalParams` have the same layout: 4 × u32/f32.
/// Packed as little-endian: n_centers(u32), n_queries(u32), kernel_id(u32), epsilon(f32).
/// Total: 16 bytes (aligned to 16 bytes as required by wgpu uniform buffers).
fn encode_rbf_params(n_centers: u32, n_queries: u32, kernel_id: u32, epsilon: f32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&n_centers.to_le_bytes());
    out[4..8].copy_from_slice(&n_queries.to_le_bytes());
    out[8..12].copy_from_slice(&kernel_id.to_le_bytes());
    out[12..16].copy_from_slice(&epsilon.to_le_bytes());
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Device / queue acquisition
// ─────────────────────────────────────────────────────────────────────────────

fn acquire_device() -> Result<(wgpu::Device, wgpu::Queue), RbfGpuError> {
    use wgpu::{
        Backends, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
        PowerPreference, RequestAdapterOptions,
    };

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
    .map_err(|_| RbfGpuError::NoAdapter)?;

    let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("scirs2-rbf"),
        required_features: Features::empty(),
        required_limits: Limits::default(),
        ..Default::default()
    }))
    .map_err(|e| RbfGpuError::DeviceCreation(e.to_string()))?;

    Ok((device, queue))
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline compile helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a bind group layout for the kernel-matrix shader (4 bindings: 3 storage + 1 uniform).
fn build_kernel_matrix_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    use wgpu::{
        BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
        ShaderStages,
    };
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("rbf-km-bgl"),
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
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Build a bind group layout for the evaluate shader (5 bindings: 4 storage + 1 uniform).
fn build_evaluate_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    use wgpu::{
        BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
        ShaderStages,
    };
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("rbf-eval-bgl"),
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
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn compile_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    source: &str,
    label: &str,
) -> wgpu::ComputePipeline {
    use wgpu::{ShaderModuleDescriptor, ShaderSource};

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(source.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{}-layout", label)),
        bind_group_layouts: &[Some(bgl)],
        ..Default::default()
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{}-pipeline", label)),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffer helpers
// ─────────────────────────────────────────────────────────────────────────────

fn f64_slice_to_f32_bytes(data: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &v in data {
        out.extend_from_slice(&(v as f32).to_le_bytes());
    }
    out
}

fn f32_bytes_to_f64_vec(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
        .collect()
}

fn upload_storage_buffer(device: &wgpu::Device, data: &[u8], label: &str) -> wgpu::Buffer {
    use wgpu::{util::DeviceExt as _, BufferUsages};
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    })
}

fn create_output_buffer(device: &wgpu::Device, size_bytes: u64, label: &str) -> wgpu::Buffer {
    use wgpu::BufferUsages;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn upload_uniform_buffer(device: &wgpu::Device, data: &[u8], label: &str) -> wgpu::Buffer {
    use wgpu::{util::DeviceExt as _, BufferUsages};
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    })
}

fn readback_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    size_bytes: u64,
) -> Result<Vec<u8>, RbfGpuError> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rbf-staging"),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rbf-readback"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size_bytes);
    queue.submit(Some(encoder.finish()));

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| RbfGpuError::Buffer(format!("GPU poll error: {e:?}")))?;

    let slice = staging.slice(0..size_bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| RbfGpuError::Buffer(format!("GPU poll during map: {e:?}")))?;

    rx.recv()
        .map_err(|_| RbfGpuError::Buffer("channel closed during map_async".into()))?
        .map_err(|e| RbfGpuError::Buffer(format!("map_async failed: {e:?}")))?;

    let mapped = slice.get_mapped_range();
    let bytes = mapped.to_vec();
    drop(mapped);
    staging.unmap();
    Ok(bytes)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public GPU compute API
// ─────────────────────────────────────────────────────────────────────────────

/// Timing result from a GPU dispatch, in nanoseconds.
pub struct GpuDispatchTiming {
    pub transfer_ns: u64,
    pub dispatch_ns: u64,
}

/// Compute the RBF kernel matrix `K[i, j] = phi(|centers[i] - queries[j]|)` on the GPU.
///
/// Returns the flattened matrix in row-major order (center index = row, query index = column)
/// and wall-clock timing.
///
/// # Errors
///
/// Returns `RbfGpuError::NoAdapter` when no GPU is available; callers should
/// fall back to the CPU path.
pub fn gpu_rbf_kernel_matrix(
    centers: &[f64],
    queries: &[f64],
    kernel: GpuRBFKernel,
    epsilon: f64,
) -> Result<(Vec<f64>, GpuDispatchTiming), RbfGpuError> {
    let n_centers = centers.len() as u32;
    let n_queries = queries.len() as u32;
    let out_len = (n_centers as usize) * (n_queries as usize);
    let out_bytes = (out_len * 4) as u64;
    let kid = kernel_id(kernel);

    let t_start = std::time::Instant::now();

    let (device, queue) = acquire_device()?;

    // Upload inputs
    let centers_bytes = f64_slice_to_f32_bytes(centers);
    let queries_bytes = f64_slice_to_f32_bytes(queries);
    let params_bytes = encode_rbf_params(n_centers, n_queries, kid, epsilon as f32);

    let buf_centers = upload_storage_buffer(&device, &centers_bytes, "rbf-km-centers");
    let buf_queries = upload_storage_buffer(&device, &queries_bytes, "rbf-km-queries");
    let buf_out = create_output_buffer(&device, out_bytes, "rbf-km-out");
    let buf_params = upload_uniform_buffer(&device, &params_bytes, "rbf-km-params");

    let t_transfer_end = std::time::Instant::now();

    // Build pipeline (use cache when possible)
    let bgl = build_kernel_matrix_bgl(&device);
    let pipeline = compile_pipeline(&device, &bgl, RBF_KERNEL_MATRIX_WGSL, "rbf-km");

    // Bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rbf-km-bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_centers.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_queries.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_params.as_entire_binding(),
            },
        ],
    });

    // Dispatch: ceil(n_centers/16) x ceil(n_queries/16)
    let wg_x = (n_centers + 15) / 16;
    let wg_y = (n_queries + 15) / 16;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rbf-km-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rbf-km-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    queue.submit(Some(encoder.finish()));

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| RbfGpuError::Buffer(format!("GPU poll after dispatch: {e:?}")))?;

    let t_dispatch_end = std::time::Instant::now();

    // Readback
    let bytes = readback_buffer(&device, &queue, &buf_out, out_bytes)?;
    let result = f32_bytes_to_f64_vec(&bytes);

    let transfer_ns = (t_transfer_end - t_start).as_nanos() as u64;
    let dispatch_ns = (t_dispatch_end - t_transfer_end).as_nanos() as u64;

    Ok((
        result,
        GpuDispatchTiming {
            transfer_ns,
            dispatch_ns,
        },
    ))
}

/// Evaluate `f(q) = sum_i coeff_i * kernel(|q - centers[i]|)` for all query points on the GPU.
///
/// Returns the evaluated values and wall-clock timing.
///
/// # Errors
///
/// Returns `RbfGpuError::NoAdapter` when no GPU is available.
pub fn gpu_rbf_evaluate(
    coefficients: &[f64],
    centers: &[f64],
    queries: &[f64],
    kernel: GpuRBFKernel,
    epsilon: f64,
) -> Result<(Vec<f64>, GpuDispatchTiming), RbfGpuError> {
    let n_centers = centers.len() as u32;
    let n_queries = queries.len() as u32;
    let out_bytes = (n_queries as usize * 4) as u64;
    let kid = kernel_id(kernel);

    let t_start = std::time::Instant::now();

    let (device, queue) = acquire_device()?;

    let coefficients_bytes = f64_slice_to_f32_bytes(coefficients);
    let centers_bytes = f64_slice_to_f32_bytes(centers);
    let queries_bytes = f64_slice_to_f32_bytes(queries);
    let params_bytes = encode_rbf_params(n_centers, n_queries, kid, epsilon as f32);

    let buf_coeff = upload_storage_buffer(&device, &coefficients_bytes, "rbf-eval-coeff");
    let buf_centers = upload_storage_buffer(&device, &centers_bytes, "rbf-eval-centers");
    let buf_queries = upload_storage_buffer(&device, &queries_bytes, "rbf-eval-queries");
    let buf_out = create_output_buffer(&device, out_bytes, "rbf-eval-out");
    let buf_params = upload_uniform_buffer(&device, &params_bytes, "rbf-eval-params");

    let t_transfer_end = std::time::Instant::now();

    let bgl = build_evaluate_bgl(&device);
    let pipeline = compile_pipeline(&device, &bgl, RBF_EVALUATE_WGSL, "rbf-eval");

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rbf-eval-bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_coeff.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_centers.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_queries.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buf_params.as_entire_binding(),
            },
        ],
    });

    let wg = (n_queries + 63) / 64;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rbf-eval-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rbf-eval-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg, 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| RbfGpuError::Buffer(format!("GPU poll after eval dispatch: {e:?}")))?;

    let t_dispatch_end = std::time::Instant::now();

    let bytes = readback_buffer(&device, &queue, &buf_out, out_bytes)?;
    let result = f32_bytes_to_f64_vec(&bytes);

    let transfer_ns = (t_transfer_end - t_start).as_nanos() as u64;
    let dispatch_ns = (t_dispatch_end - t_transfer_end).as_nanos() as u64;

    Ok((
        result,
        GpuDispatchTiming {
            transfer_ns,
            dispatch_ns,
        },
    ))
}

/// Expose the shader source for the kernel-matrix stage (used in tests to validate compilation).
pub fn kernel_matrix_shader_source() -> &'static str {
    RBF_KERNEL_MATRIX_WGSL
}

/// Expose the shader source for the evaluate stage (used in tests to validate compilation).
pub fn evaluate_shader_source() -> &'static str {
    RBF_EVALUATE_WGSL
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline cache key helpers (kept for future use)
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the pipeline cache (initialised lazily).
///
/// Currently the cache is populated on demand but entries are not reused across
/// `gpu_rbf_kernel_matrix` / `gpu_rbf_evaluate` invocations because wgpu
/// `Device` objects cannot be shared across calls without explicit lifetime
/// management.  The cache is reserved for future optimisation when a shared
/// device context is maintained.
#[allow(dead_code)]
fn get_pipeline_cache() -> &'static Mutex<HashMap<PipelineKey, CachedPipeline>> {
    pipeline_cache()
}
