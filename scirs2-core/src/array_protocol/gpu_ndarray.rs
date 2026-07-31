// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under the Apache License, Version 2.0
// (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
//

//! Real wgpu-backed `GpuNdarray<T>` that implements `ArrayProtocol`.
//!
//! Enabled only with the `array_protocol_wgpu` feature, which implies `wgpu`.
//!
//! ## Supported operations (GPU dispatch)
//! - `add`, `subtract`, `multiply` — elementwise binary, workgroup (256,1,1), uses `arrayLength`
//! - `multiply_by_scalar_f32` — elementwise scalar multiply, workgroup (256,1,1)
//! - `matmul` — naive (one thread per output element), workgroup (16,16,1)
//! - `sum(axis=None)` — two-pass reduce, workgroup (256,1,1)
//! - `transpose` (2-D) — 16×16 bank-conflict-padded tile, workgroup (16,16,1)
//!   (32×32 exceeds Metal's 256-invocation-per-workgroup limit)
//! - `concatenate(axis=0)` — via `copy_buffer_to_buffer`, no shader
//! - `concatenate(axis>0)` — WGSL gather kernel (`CONCAT_AXISN_WGSL`), storage-buf strides
//! - `sum(axis=Some(ax))` — WGSL per-output-element axis reduction (`REDUCE_SUM_AXIS_WGSL`)
//! - `reshape` — zero-copy (clone `Arc<Buffer>`, new shape/strides)
//!
//! ## CPU-fallback operations
//! - `svd` — falls back to CPU `NdarrayWrapper`
//! - `inverse` — falls back to CPU `NdarrayWrapper`
//! - `multiply_by_scalar_f64`, `divide_by_scalar_f64` — convert to f32, then GPU
//! - GPU kernel errors on axis ops — fallback to CPU (graceful degradation)
//!
//! ## GPU threshold
//! Arrays with fewer than 4096 elements skip GPU dispatch entirely and fall back to CPU.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

use crate::array_protocol::{
    ArrayFunction, ArrayProtocol, GPUArray, NdarrayWrapper, NotImplemented,
};
use crate::error::{CoreError, CoreResult, ErrorContext};
use crate::gpu::backends::WebGPUContext;
use crate::gpu::GpuError;

// ──────────────────────────────────────────────────────────────────────
// 1. GpuScalar — sealed trait, blanket impl for f32 only
// ──────────────────────────────────────────────────────────────────────

mod sealed {
    pub trait Sealed {}
}

/// Marker trait for element types that wgpu-29 supports natively (f32 only;
/// f64 is not supported in WGSL without extensions).
pub trait GpuScalar: sealed::Sealed + Clone + Send + Sync + 'static {}

impl sealed::Sealed for f32 {}
impl GpuScalar for f32 {}

// ──────────────────────────────────────────────────────────────────────
// 2. GPU dispatch threshold, availability cache, and singleton context
// ──────────────────────────────────────────────────────────────────────

/// Arrays smaller than this skip GPU dispatch entirely.
const GPU_THRESHOLD: usize = 4096;

/// Cached GPU availability flag (computed once per process).
static GPU_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Cached singleton `WebGPUContext` — all `GpuNdarray` share one device.
static GPU_CONTEXT: OnceLock<Option<Arc<WebGPUContext>>> = OnceLock::new();

/// Returns the shared `WebGPUContext`, or `None` if no adapter is available.
///
/// Initializes the singleton device once; all subsequent calls return the cached value.
pub fn global_context() -> Option<Arc<WebGPUContext>> {
    GPU_CONTEXT
        .get_or_init(|| match WebGPUContext::new() {
            Ok(ctx) => Some(Arc::new(ctx)),
            Err(_) => None,
        })
        .clone()
}

/// Returns `true` if a wgpu adapter was found when first called; cached afterwards.
pub fn is_gpu_available() -> bool {
    *GPU_AVAILABLE.get_or_init(|| global_context().is_some())
}

// ──────────────────────────────────────────────────────────────────────
// 3. GpuNdarray<T>
// ──────────────────────────────────────────────────────────────────────

/// A GPU-backed n-dimensional array backed by a real wgpu `Buffer`.
///
/// Created via [`GpuNdarray::from_ndarray_data`] or [`GpuNdarray::from_data`].
/// Converted back with [`GpuNdarray::to_ndarray`].
pub struct GpuNdarray<T: GpuScalar> {
    /// The live wgpu `Buffer` on the device.  Shared under `Arc` to make
    /// `reshape` zero-copy and `Clone` cheap.
    buffer: Arc<wgpu::Buffer>,

    /// Logical shape (row-major).
    shape: Vec<usize>,

    /// Row-major strides in *elements* (not bytes).
    strides: Vec<usize>,

    /// Shared device/queue context.
    context: Arc<WebGPUContext>,

    _phantom: PhantomData<T>,
}

impl<T: GpuScalar> std::fmt::Debug for GpuNdarray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuNdarray")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .finish_non_exhaustive()
    }
}

impl<T: GpuScalar> Clone for GpuNdarray<T> {
    fn clone(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            context: Arc::clone(&self.context),
            _phantom: PhantomData,
        }
    }
}

impl<T: GpuScalar> GpuNdarray<T> {
    /// Expose the underlying `Arc<wgpu::Buffer>` for zero-copy checks in tests.
    #[must_use]
    pub fn buffer_arc(&self) -> &Arc<wgpu::Buffer> {
        &self.buffer
    }

    /// Total number of elements.
    #[must_use]
    fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Row-major strides for the given shape.
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}

// ──────────────────────────────────────────────────────────────────────
// 4. Low-level pipeline builder helper (not part of GpuNdarray impl)
// ──────────────────────────────────────────────────────────────────────

/// Build a compute pipeline with an explicit bind group layout.
///
/// `bgl_entries` must match the WGSL `@group(0)` bindings exactly.
fn build_pipeline(
    ctx: &WebGPUContext,
    wgsl: &str,
    bgl_entries: &[wgpu::BindGroupLayoutEntry],
    label: &str,
) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout), GpuError> {
    let device = ctx.device();
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}_bgl")),
        entries: bgl_entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_layout")),
        bind_group_layouts: &[Some(&bgl)],
        ..Default::default()
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{label}_pipeline")),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    Ok((pipeline, bgl))
}

/// Shorthand BGL entry for a read-only storage buffer.
fn storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Shorthand BGL entry for a read-write storage buffer.
fn storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Shorthand BGL entry for a uniform buffer.
fn uniform_buf(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ──────────────────────────────────────────────────────────────────────
// 5. from_ndarray / to_ndarray for f32
// ──────────────────────────────────────────────────────────────────────

impl GpuNdarray<f32> {
    /// Upload a CPU `NdarrayWrapper<f32, _>` to the GPU.
    ///
    /// Returns `Err(GpuError)` if no adapter is available or upload fails.
    pub fn from_ndarray_data(
        data: &[f32],
        shape: Vec<usize>,
        context: Arc<WebGPUContext>,
    ) -> Result<Self, GpuError> {
        use wgpu::util::DeviceExt as _;
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buffer = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuNdarray<f32>"),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        let strides = Self::compute_strides(&shape);
        Ok(Self {
            buffer: Arc::new(buffer),
            shape,
            strides,
            context,
            _phantom: PhantomData,
        })
    }

    /// Download the GPU buffer to a flat `Vec<f32>`.
    pub fn to_vec(&self) -> Result<Vec<f32>, GpuError> {
        let byte_size = (self.numel() * std::mem::size_of::<f32>()) as u64;
        let staging = self
            .context
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("GpuNdarray-readback"),
                size: byte_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        let mut encoder =
            self.context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("GpuNdarray-readback-encoder"),
                });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, byte_size);
        self.context.queue().submit(Some(encoder.finish()));

        self.context
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Other(format!("poll error: {e:?}")))?;

        let slice = staging.slice(0..byte_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });

        self.context
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Other(format!("poll-map error: {e:?}")))?;

        rx.recv()
            .map_err(|_| GpuError::Other("channel closed".into()))?
            .map_err(|e| GpuError::Other(format!("map_async failed: {e:?}")))?;

        let mapped = slice.get_mapped_range();
        let result: Vec<f32> = mapped
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        drop(mapped);
        staging.unmap();
        Ok(result)
    }

    /// Download to a dynamic ndarray.
    pub fn to_ndarray(&self) -> Result<ndarray::ArrayD<f32>, GpuError> {
        let flat = self.to_vec()?;
        ndarray::ArrayD::<f32>::from_shape_vec(self.shape.clone(), flat)
            .map_err(|e| GpuError::Other(format!("shape_vec error: {e}")))
    }

    /// Construct from a `WebGPUContext`, uploading `data`.
    ///
    /// Uses the process-wide singleton `WebGPUContext` so that all arrays
    /// created via this function share the same `wgpu::Device` and can be
    /// combined freely in kernel dispatches.
    pub fn from_data(data: &[f32], shape: Vec<usize>) -> Result<Self, GpuError> {
        let ctx =
            global_context().ok_or_else(|| GpuError::Other("No wgpu adapter available".into()))?;
        Self::from_ndarray_data(data, shape, ctx)
    }

    // ─── public high-level ops ─────────────────────────────────────────

    /// Elementwise add: `self + other`.  Both must have the same shape.
    pub fn add(&self, other: &GpuNdarray<f32>) -> Result<GpuNdarray<f32>, GpuError> {
        self.dispatch_elementwise_binary(other, 0)
    }

    /// Elementwise subtract: `self - other`.  Both must have the same shape.
    pub fn subtract(&self, other: &GpuNdarray<f32>) -> Result<GpuNdarray<f32>, GpuError> {
        self.dispatch_elementwise_binary(other, 1)
    }

    /// Elementwise multiply: `self * other`.  Both must have the same shape.
    pub fn multiply(&self, other: &GpuNdarray<f32>) -> Result<GpuNdarray<f32>, GpuError> {
        self.dispatch_elementwise_binary(other, 2)
    }

    /// Multiply every element by a scalar: `self * scalar`.
    pub fn multiply_by_scalar_f32(&self, scalar: f32) -> Result<GpuNdarray<f32>, GpuError> {
        self.dispatch_scalar_multiply(scalar)
    }

    /// Compute the sum of all elements (dot product via `a.multiply(b)?.sum_all()`).
    pub fn sum_all(&self) -> Result<f32, GpuError> {
        self.dispatch_sum_all()
    }

    /// Compute the dot product `self · other = sum(self * other)`.
    pub fn dot_gpu(&self, other: &GpuNdarray<f32>) -> Result<f32, GpuError> {
        let prod = self.dispatch_elementwise_binary(other, 2)?;
        prod.dispatch_sum_all()
    }

    /// Tiled 16x16 matrix multiplication: self x other.
    ///
    /// Both arrays must be 2-D.  For a matrix-vector product, upload the
    /// vector as a [n, 1] column; the result will have shape [m, 1].
    pub fn matmul(&self, other: &GpuNdarray<f32>) -> Result<GpuNdarray<f32>, GpuError> {
        self.dispatch_matmul(other)
    }

    // ─── internal GPU kernel dispatchers ───────────────────────────────

    /// Dispatch an elementwise binary kernel (`add`, `subtract`, `multiply`).
    ///
    /// `op_id`: 0=add, 1=subtract, 2=multiply.
    fn dispatch_elementwise_binary(
        &self,
        other: &GpuNdarray<f32>,
        op_id: u32,
    ) -> Result<GpuNdarray<f32>, GpuError> {
        let n = self.numel();
        if n != other.numel() {
            return Err(GpuError::InvalidParameter(
                "shape mismatch in elementwise binary".into(),
            ));
        }
        let wgsl = match op_id {
            0 => ELEMENTWISE_ADD_WGSL,
            1 => ELEMENTWISE_SUB_WGSL,
            _ => ELEMENTWISE_MUL_WGSL,
        };
        let byte_size = (n * 4) as u64;
        let result_buf = self
            .context
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("elementwise-result"),
                size: byte_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        let bgl_entries = [storage_ro(0), storage_ro(1), storage_rw(2)];
        let (pipeline, bgl) = build_pipeline(&self.context, wgsl, &bgl_entries, "elementwise")?;

        let bind_group = self
            .context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("elementwise-bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: other.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: result_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            self.context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("elementwise-encoder"),
                });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("elementwise-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n as u32 + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        self.context
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Other(format!("poll error: {e:?}")))?;

        Ok(GpuNdarray {
            buffer: Arc::new(result_buf),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            context: Arc::clone(&self.context),
            _phantom: PhantomData,
        })
    }

    /// Dispatch scalar multiply: each element * scalar.
    fn dispatch_scalar_multiply(&self, scalar: f32) -> Result<GpuNdarray<f32>, GpuError> {
        let n = self.numel();
        let byte_size = (n * 4) as u64;
        let result_buf = self
            .context
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("scalar-mul-result"),
                size: byte_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        // Uniform: scalar (f32), n (u32) — 8 bytes padded to 16
        let mut unif: Vec<u8> = Vec::with_capacity(16);
        unif.extend_from_slice(&scalar.to_le_bytes());
        unif.extend_from_slice(&(n as u32).to_le_bytes());
        while unif.len() % 16 != 0 {
            unif.push(0);
        }
        use wgpu::util::DeviceExt as _;
        let uniform_buffer =
            self.context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scalar-mul-uniform"),
                    contents: &unif,
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bgl_entries = [storage_ro(0), storage_rw(1), uniform_buf(2)];
        let (pipeline, bgl) =
            build_pipeline(&self.context, SCALAR_MUL_WGSL, &bgl_entries, "scalar-mul")?;

        let bind_group = self
            .context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scalar-mul-bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: result_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            self.context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("scalar-mul-encoder"),
                });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scalar-mul-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n as u32 + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        self.context
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Other(format!("poll error: {e:?}")))?;

        Ok(GpuNdarray {
            buffer: Arc::new(result_buf),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            context: Arc::clone(&self.context),
            _phantom: PhantomData,
        })
    }

    /// Dispatch tiled 16×16 matmul for 2-D arrays.
    fn dispatch_matmul(&self, other: &GpuNdarray<f32>) -> Result<GpuNdarray<f32>, GpuError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(GpuError::InvalidParameter(
                "matmul requires 2-D arrays".into(),
            ));
        }
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        if k != k2 {
            return Err(GpuError::InvalidParameter(format!(
                "matmul shape mismatch: [{m},{k}] x [{k2},{n}]"
            )));
        }

        let byte_size = (m * n * 4) as u64;
        let result_buf = self
            .context
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("matmul-result"),
                size: byte_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        let uniform_data: [u32; 3] = [m as u32, n as u32, k as u32];
        let uniform_bytes: Vec<u8> = uniform_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Pad to 16-byte alignment (wgpu uniform requirement)
        let mut uniform_padded = uniform_bytes;
        while uniform_padded.len() % 16 != 0 {
            uniform_padded.push(0);
        }
        use wgpu::util::DeviceExt as _;
        let uniform_buffer =
            self.context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("matmul-uniform"),
                    contents: &uniform_padded,
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bgl_entries = [storage_ro(0), storage_ro(1), storage_rw(2), uniform_buf(3)];
        let (pipeline, bgl) = build_pipeline(&self.context, MATMUL_WGSL, &bgl_entries, "matmul")?;
        let bind_group = self
            .context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("matmul-bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: other.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: result_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

        let wg_x = (n as u32 + 15) / 16;
        let wg_y = (m as u32 + 15) / 16;
        let mut encoder =
            self.context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("matmul-encoder"),
                });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        self.context
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Other(format!("poll error: {e:?}")))?;

        Ok(GpuNdarray {
            buffer: Arc::new(result_buf),
            shape: vec![m, n],
            strides: Self::compute_strides(&[m, n]),
            context: Arc::clone(&self.context),
            _phantom: PhantomData,
        })
    }

    /// Dispatch two-pass sum reduction (axis=None).
    fn dispatch_sum_all(&self) -> Result<f32, GpuError> {
        let n = self.numel();
        // First pass: workgroup partial sums
        let workgroup_count = (n as u32 + 255) / 256;
        let partial_byte_size = (workgroup_count as usize * 4) as u64;
        let partial_buf = self
            .context
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("sum-partial"),
                size: partial_byte_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        let n_bytes = (n as u32).to_le_bytes();
        let mut uniform_bytes = n_bytes.to_vec();
        while uniform_bytes.len() % 16 != 0 {
            uniform_bytes.push(0);
        }
        use wgpu::util::DeviceExt as _;
        let uniform_buffer =
            self.context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sum-uniform"),
                    contents: &uniform_bytes,
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bgl_entries = [storage_ro(0), storage_rw(1), uniform_buf(2)];
        let (pipeline, bgl) =
            build_pipeline(&self.context, SUM_REDUCE_WGSL, &bgl_entries, "sum-reduce")?;
        let bind_group = self
            .context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sum-bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: partial_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            self.context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sum-encoder"),
                });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sum-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        self.context
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Other(format!("poll error: {e:?}")))?;

        // Read back partial sums and sum on CPU (second pass)
        let staging = self
            .context
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("sum-staging"),
                size: partial_byte_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        let mut encoder2 =
            self.context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sum-copy-encoder"),
                });
        encoder2.copy_buffer_to_buffer(&partial_buf, 0, &staging, 0, partial_byte_size);
        self.context.queue().submit(Some(encoder2.finish()));

        self.context
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Other(format!("poll error: {e:?}")))?;

        let slice = staging.slice(0..partial_byte_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.context
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Other(format!("map poll error: {e:?}")))?;
        rx.recv()
            .map_err(|_| GpuError::Other("channel closed".into()))?
            .map_err(|e| GpuError::Other(format!("map_async: {e:?}")))?;

        let mapped = slice.get_mapped_range();
        let total: f32 = mapped
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .sum();
        drop(mapped);
        staging.unmap();
        Ok(total)
    }

    /// Dispatch tiled 32×32 2-D transpose.
    fn dispatch_transpose_2d(&self) -> Result<GpuNdarray<f32>, GpuError> {
        if self.shape.len() != 2 {
            return Err(GpuError::InvalidParameter(
                "transpose_2d requires a 2-D array".into(),
            ));
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let byte_size = (rows * cols * 4) as u64;

        let result_buf = self
            .context
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("transpose-result"),
                size: byte_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        // Uniform: rows (u32), cols (u32)
        let uniform_data: [u32; 2] = [rows as u32, cols as u32];
        let uniform_bytes: Vec<u8> = uniform_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut uniform_padded = uniform_bytes;
        while uniform_padded.len() % 16 != 0 {
            uniform_padded.push(0);
        }
        use wgpu::util::DeviceExt as _;
        let uniform_buffer =
            self.context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("transpose-uniform"),
                    contents: &uniform_padded,
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        // Use 16×16 workgroup — Metal limit is 256 total invocations
        let bgl_entries = [storage_ro(0), storage_rw(1), uniform_buf(2)];
        let (pipeline, bgl) =
            build_pipeline(&self.context, TRANSPOSE_WGSL, &bgl_entries, "transpose")?;
        let bind_group = self
            .context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("transpose-bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: result_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

        // Workgroup size is 16×16 tiles
        let wg_x = (cols as u32 + 15) / 16;
        let wg_y = (rows as u32 + 15) / 16;
        let mut encoder =
            self.context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("transpose-encoder"),
                });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("transpose-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        self.context
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Other(format!("poll error: {e:?}")))?;

        Ok(GpuNdarray {
            buffer: Arc::new(result_buf),
            shape: vec![cols, rows],
            strides: Self::compute_strides(&[cols, rows]),
            context: Arc::clone(&self.context),
            _phantom: PhantomData,
        })
    }

    /// Concatenate along axis=0 via `copy_buffer_to_buffer` (no shader).
    fn dispatch_concatenate_axis0(
        arrays: &[&GpuNdarray<f32>],
    ) -> Result<GpuNdarray<f32>, GpuError> {
        if arrays.is_empty() {
            return Err(GpuError::InvalidParameter("empty array list".into()));
        }
        // Validate trailing dimensions match
        let trailing = &arrays[0].shape[1..];
        for arr in arrays.iter().skip(1) {
            if arr.shape[1..] != *trailing {
                return Err(GpuError::InvalidParameter(
                    "concatenate axis=0: trailing dimensions must match".into(),
                ));
            }
        }
        let ctx = Arc::clone(&arrays[0].context);
        let trailing_elems: usize = trailing.iter().product::<usize>().max(1);

        let total_rows: usize = arrays.iter().map(|a| a.shape[0]).sum();
        let total_elems = total_rows * trailing_elems;
        let total_bytes = (total_elems * 4) as u64;

        let result_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("concat-result"),
            size: total_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("concat-encoder"),
            });
        let mut offset: u64 = 0;
        for arr in arrays {
            let arr_bytes = (arr.numel() * 4) as u64;
            encoder.copy_buffer_to_buffer(&arr.buffer, 0, &result_buf, offset, arr_bytes);
            offset += arr_bytes;
        }
        ctx.queue().submit(Some(encoder.finish()));
        ctx.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Other(format!("poll error: {e:?}")))?;

        let new_shape = {
            let mut s = vec![total_rows];
            s.extend_from_slice(trailing);
            s
        };
        let new_strides = Self::compute_strides(&new_shape);
        Ok(GpuNdarray {
            buffer: Arc::new(result_buf),
            shape: new_shape,
            strides: new_strides,
            context: ctx,
            _phantom: PhantomData,
        })
    }

    /// Concatenate two arrays along an arbitrary axis (axis > 0) using a GPU gather kernel.
    ///
    /// Each output element is handled by one invocation; the kernel decomposes the flat output
    /// index into multi-dim coordinates, branches on the axis coordinate, and gathers from A or B.
    fn dispatch_concatenate_axisn(
        a: &GpuNdarray<f32>,
        b: &GpuNdarray<f32>,
        axis: usize,
    ) -> Result<GpuNdarray<f32>, GpuError> {
        let ndim = a.shape.len();
        if ndim > 8 {
            return Err(GpuError::InvalidParameter(
                "concat_axisn: ndim must be <= 8".into(),
            ));
        }

        // Compute output shape
        let mut out_shape = a.shape.clone();
        out_shape[axis] += b.shape[axis];

        let out_strides = GpuNdarray::<f32>::compute_strides(&out_shape);
        let a_strides = GpuNdarray::<f32>::compute_strides(&a.shape);
        let b_strides = GpuNdarray::<f32>::compute_strides(&b.shape);

        let total_out = out_shape.iter().product::<usize>();
        let byte_out = (total_out * 4) as u64;

        let ctx = Arc::clone(&a.context);
        let result_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("concat-axisn-result"),
            size: byte_out,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pack uniform: axis (u32), dim_a (u32), ndim (u32), _pad (u32) — 16 bytes
        let dim_a = a.shape[axis] as u32;
        let mut unif_bytes: Vec<u8> = Vec::with_capacity(16);
        unif_bytes.extend_from_slice(&(axis as u32).to_le_bytes());
        unif_bytes.extend_from_slice(&dim_a.to_le_bytes());
        unif_bytes.extend_from_slice(&(ndim as u32).to_le_bytes());
        unif_bytes.extend_from_slice(&0u32.to_le_bytes()); // _pad
        debug_assert_eq!(unif_bytes.len(), 16);

        // Pack shape/strides as storage buffers (avoid WGSL uniform 16-byte-per-element alignment)
        // Each: array of u32, length = ndim (padded to multiple of 4 u32s for buffer sizing)
        let pack_u32_slice = |vals: &[usize]| -> Vec<u8> {
            let mut out: Vec<u8> = Vec::with_capacity(vals.len() * 4);
            for &v in vals {
                out.extend_from_slice(&(v as u32).to_le_bytes());
            }
            // Pad to 16-byte boundary
            while out.len() % 16 != 0 {
                out.extend_from_slice(&0u32.to_le_bytes());
            }
            out
        };

        let out_shape_bytes = pack_u32_slice(&out_shape);
        let out_strides_bytes = pack_u32_slice(&out_strides);
        let a_strides_bytes = pack_u32_slice(&a_strides);
        let b_strides_bytes = pack_u32_slice(&b_strides);

        use wgpu::util::DeviceExt as _;
        let make_storage_buf = |bytes: &[u8], label: &str| -> wgpu::Buffer {
            ctx.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let unif_buf = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("concat-axisn-uniform"),
                contents: &unif_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let out_shape_buf = make_storage_buf(&out_shape_bytes, "concat-axisn-out-shape");
        let out_strides_buf = make_storage_buf(&out_strides_bytes, "concat-axisn-out-strides");
        let a_strides_buf = make_storage_buf(&a_strides_bytes, "concat-axisn-a-strides");
        let b_strides_buf = make_storage_buf(&b_strides_bytes, "concat-axisn-b-strides");

        // Bindings:
        // 0=a (storage_ro), 1=b (storage_ro), 2=result (storage_rw),
        // 3=uniform, 4=out_shape (storage_ro), 5=out_strides (storage_ro),
        // 6=a_strides (storage_ro), 7=b_strides (storage_ro)
        let bgl_entries = [
            storage_ro(0),
            storage_ro(1),
            storage_rw(2),
            uniform_buf(3),
            storage_ro(4),
            storage_ro(5),
            storage_ro(6),
            storage_ro(7),
        ];
        let (pipeline, bgl) =
            build_pipeline(&ctx, CONCAT_AXISN_WGSL, &bgl_entries, "concat-axisn")?;

        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("concat-axisn-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: unif_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_shape_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out_strides_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: a_strides_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: b_strides_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups = (total_out as u32 + 255) / 256;
        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("concat-axisn-encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("concat-axisn-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        ctx.queue().submit(Some(encoder.finish()));
        ctx.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Other(format!("poll error: {e:?}")))?;

        let new_strides = GpuNdarray::<f32>::compute_strides(&out_shape);
        Ok(GpuNdarray {
            buffer: Arc::new(result_buf),
            shape: out_shape,
            strides: new_strides,
            context: ctx,
            _phantom: PhantomData,
        })
    }

    /// Reduce one axis via GPU: output[out_idx] = sum over j in 0..shape[axis] of input[...j...].
    ///
    /// Each invocation handles one output element.  Shape/strides are passed as
    /// `storage_ro` buffers to avoid WGSL uniform 16-byte-per-element alignment.
    fn dispatch_sum_axis(&self, axis: usize) -> Result<GpuNdarray<f32>, GpuError> {
        let ndim = self.shape.len();
        if ndim > 8 {
            return Err(GpuError::InvalidParameter(
                "sum_axis: ndim must be <= 8".into(),
            ));
        }

        let axis_size = self.shape[axis];

        // Output shape: remove `axis` dimension
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &d)| d)
            .collect();
        let out_strides = GpuNdarray::<f32>::compute_strides(&out_shape);
        let in_strides = &self.strides;

        let total_out = out_shape.iter().product::<usize>().max(1);
        let byte_out = (total_out * 4) as u64;

        let result_buf = self
            .context
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("sum-axis-result"),
                size: byte_out,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        // Uniform: axis (u32), axis_size (u32), ndim (u32), in_axis_stride (u32) — 16 bytes
        let in_axis_stride = self.strides[axis] as u32;
        let mut unif_bytes: Vec<u8> = Vec::with_capacity(16);
        unif_bytes.extend_from_slice(&(axis as u32).to_le_bytes());
        unif_bytes.extend_from_slice(&(axis_size as u32).to_le_bytes());
        unif_bytes.extend_from_slice(&(ndim as u32).to_le_bytes());
        unif_bytes.extend_from_slice(&in_axis_stride.to_le_bytes());
        debug_assert_eq!(unif_bytes.len(), 16);

        let pack_u32_slice = |vals: &[usize]| -> Vec<u8> {
            let mut out: Vec<u8> = Vec::with_capacity(vals.len() * 4);
            for &v in vals {
                out.extend_from_slice(&(v as u32).to_le_bytes());
            }
            while out.len() % 16 != 0 {
                out.extend_from_slice(&0u32.to_le_bytes());
            }
            out
        };

        let in_shape_bytes = pack_u32_slice(&self.shape);
        let in_strides_bytes = pack_u32_slice(in_strides);
        let out_shape_bytes = pack_u32_slice(&out_shape);
        let out_strides_bytes = pack_u32_slice(&out_strides);

        use wgpu::util::DeviceExt as _;
        let unif_buf =
            self.context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sum-axis-uniform"),
                    contents: &unif_bytes,
                    usage: wgpu::BufferUsages::UNIFORM,
                });
        let make_storage_buf = |bytes: &[u8], label: &str| -> wgpu::Buffer {
            self.context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let in_shape_buf = make_storage_buf(&in_shape_bytes, "sum-axis-in-shape");
        let in_strides_buf = make_storage_buf(&in_strides_bytes, "sum-axis-in-strides");
        let out_shape_buf = make_storage_buf(&out_shape_bytes, "sum-axis-out-shape");
        let out_strides_buf = make_storage_buf(&out_strides_bytes, "sum-axis-out-strides");

        // Bindings:
        // 0=input (storage_ro), 1=result (storage_rw), 2=uniform,
        // 3=in_shape (storage_ro), 4=in_strides (storage_ro),
        // 5=out_shape (storage_ro), 6=out_strides (storage_ro)
        let bgl_entries = [
            storage_ro(0),
            storage_rw(1),
            uniform_buf(2),
            storage_ro(3),
            storage_ro(4),
            storage_ro(5),
            storage_ro(6),
        ];
        let (pipeline, bgl) = build_pipeline(
            &self.context,
            REDUCE_SUM_AXIS_WGSL,
            &bgl_entries,
            "sum-axis",
        )?;

        let bind_group = self
            .context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sum-axis-bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: result_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: unif_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: in_shape_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: in_strides_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: out_shape_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: out_strides_buf.as_entire_binding(),
                    },
                ],
            });

        let workgroups = (total_out as u32 + 255) / 256;
        let mut encoder =
            self.context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sum-axis-encoder"),
                });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sum-axis-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        self.context
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Other(format!("poll error: {e:?}")))?;

        let new_strides = GpuNdarray::<f32>::compute_strides(&out_shape);
        Ok(GpuNdarray {
            buffer: Arc::new(result_buf),
            shape: out_shape,
            strides: new_strides,
            context: Arc::clone(&self.context),
            _phantom: PhantomData,
        })
    }

    /// Zero-copy reshape: clone `Arc<Buffer>`, new shape/strides.
    fn dispatch_reshape(&self, new_shape: Vec<usize>) -> Result<GpuNdarray<f32>, GpuError> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(GpuError::InvalidParameter(format!(
                "reshape: element count mismatch: {} vs {}",
                self.numel(),
                new_numel
            )));
        }
        let new_strides = Self::compute_strides(&new_shape);
        Ok(GpuNdarray {
            buffer: Arc::clone(&self.buffer),
            shape: new_shape,
            strides: new_strides,
            context: Arc::clone(&self.context),
            _phantom: PhantomData,
        })
    }

    /// CPU fallback: download, run ndarray op, re-upload.
    fn cpu_fallback_unary<F>(&self, f: F) -> Result<GpuNdarray<f32>, GpuError>
    where
        F: FnOnce(ndarray::ArrayD<f32>) -> Result<ndarray::ArrayD<f32>, GpuError>,
    {
        let arr = self.to_ndarray()?;
        let result = f(arr)?;
        let shape = result.shape().to_vec();
        let flat: Vec<f32> = result.into_iter().collect();
        Self::from_ndarray_data(&flat, shape, Arc::clone(&self.context))
    }
}

// ──────────────────────────────────────────────────────────────────────
// 5. ArrayProtocol impl
// ──────────────────────────────────────────────────────────────────────

impl ArrayProtocol for GpuNdarray<f32> {
    fn array_function(
        &self,
        func: &ArrayFunction,
        _types: &[TypeId],
        args: &[Box<dyn Any>],
        kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, NotImplemented> {
        // Helper: extract GpuNdarray<f32> from args[idx]
        macro_rules! gpu_arg {
            ($idx:expr) => {{
                let boxed_ap = args[$idx]
                    .downcast_ref::<Box<dyn ArrayProtocol>>()
                    .ok_or(NotImplemented)?;
                boxed_ap
                    .as_any()
                    .downcast_ref::<GpuNdarray<f32>>()
                    .ok_or(NotImplemented)?
            }};
        }

        // Helper: fall back to CPU if below threshold or GPU unavailable
        let n = self.numel();
        let use_gpu = n >= GPU_THRESHOLD && is_gpu_available();

        match func.name {
            // ── elementwise add ─────────────────────────────────────────
            "scirs2::array_protocol::operations::add" => {
                let a = gpu_arg!(0);
                let b = gpu_arg!(1);
                if use_gpu {
                    // OP_ID 0 = add
                    let result = a
                        .dispatch_elementwise_binary(b, 0)
                        .map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                } else {
                    let ra = a.to_ndarray().map_err(|_| NotImplemented)?;
                    let rb = b.to_ndarray().map_err(|_| NotImplemented)?;
                    let rc = ra + rb;
                    let flat: Vec<f32> = rc.into_iter().collect();
                    let result = GpuNdarray::<f32>::from_ndarray_data(
                        &flat,
                        a.shape.clone(),
                        Arc::clone(&a.context),
                    )
                    .map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                }
            }

            // ── elementwise subtract ─────────────────────────────────────
            "scirs2::array_protocol::operations::subtract" => {
                let a = gpu_arg!(0);
                let b = gpu_arg!(1);
                if use_gpu {
                    // OP_ID 1 = subtract
                    let result = a
                        .dispatch_elementwise_binary(b, 1)
                        .map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                } else {
                    let ra = a.to_ndarray().map_err(|_| NotImplemented)?;
                    let rb = b.to_ndarray().map_err(|_| NotImplemented)?;
                    let rc = ra - rb;
                    let flat: Vec<f32> = rc.into_iter().collect();
                    let result = GpuNdarray::<f32>::from_ndarray_data(
                        &flat,
                        a.shape.clone(),
                        Arc::clone(&a.context),
                    )
                    .map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                }
            }

            // ── elementwise multiply ─────────────────────────────────────
            "scirs2::array_protocol::operations::multiply" => {
                let a = gpu_arg!(0);
                let b = gpu_arg!(1);
                if use_gpu {
                    // OP_ID 2 = multiply
                    let result = a
                        .dispatch_elementwise_binary(b, 2)
                        .map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                } else {
                    let ra = a.to_ndarray().map_err(|_| NotImplemented)?;
                    let rb = b.to_ndarray().map_err(|_| NotImplemented)?;
                    let rc = ra * rb;
                    let flat: Vec<f32> = rc.into_iter().collect();
                    let result = GpuNdarray::<f32>::from_ndarray_data(
                        &flat,
                        a.shape.clone(),
                        Arc::clone(&a.context),
                    )
                    .map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                }
            }

            // ── multiply_by_scalar_f32 ────────────────────────────────────
            "scirs2::array_protocol::operations::multiply_by_scalar_f32" => {
                let a = gpu_arg!(0);
                // scalar value is stored as kwarg with key = scalar.to_string()
                let scalar = kwargs
                    .values()
                    .find_map(|v| v.downcast_ref::<f32>().copied())
                    .ok_or(NotImplemented)?;
                if use_gpu {
                    let result = a
                        .dispatch_scalar_multiply(scalar)
                        .map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                } else {
                    let ra = a.to_ndarray().map_err(|_| NotImplemented)?;
                    let rc = ra * scalar;
                    let flat: Vec<f32> = rc.into_iter().collect();
                    let result = GpuNdarray::<f32>::from_ndarray_data(
                        &flat,
                        a.shape.clone(),
                        Arc::clone(&a.context),
                    )
                    .map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                }
            }

            // ── multiply_by_scalar_f64 — convert scalar to f32 ────────────
            "scirs2::array_protocol::operations::multiply_by_scalar_f64" => {
                let a = gpu_arg!(0);
                let scalar = kwargs
                    .values()
                    .find_map(|v| v.downcast_ref::<f64>().copied())
                    .ok_or(NotImplemented)? as f32;
                if use_gpu {
                    let result = a
                        .dispatch_scalar_multiply(scalar)
                        .map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                } else {
                    let ra = a.to_ndarray().map_err(|_| NotImplemented)?;
                    let rc = ra * scalar;
                    let flat: Vec<f32> = rc.into_iter().collect();
                    let result = GpuNdarray::<f32>::from_ndarray_data(
                        &flat,
                        a.shape.clone(),
                        Arc::clone(&a.context),
                    )
                    .map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                }
            }

            // ── divide_by_scalar_f64 — convert to f32 scalar multiply ─────
            "scirs2::array_protocol::operations::divide_by_scalar_f64" => {
                let a = gpu_arg!(0);
                let scalar = kwargs
                    .values()
                    .find_map(|v| v.downcast_ref::<f64>().copied())
                    .ok_or(NotImplemented)?;
                if scalar == 0.0 {
                    return Err(NotImplemented);
                }
                let inv = (1.0 / scalar) as f32;
                if use_gpu {
                    let result = a
                        .dispatch_scalar_multiply(inv)
                        .map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                } else {
                    let ra = a.to_ndarray().map_err(|_| NotImplemented)?;
                    let rc = ra * inv;
                    let flat: Vec<f32> = rc.into_iter().collect();
                    let result = GpuNdarray::<f32>::from_ndarray_data(
                        &flat,
                        a.shape.clone(),
                        Arc::clone(&a.context),
                    )
                    .map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                }
            }

            // ── matmul ─────────────────────────────────────────────────────
            "scirs2::array_protocol::operations::matmul" => {
                let a = gpu_arg!(0);
                let b = gpu_arg!(1);
                if use_gpu {
                    let result = a.dispatch_matmul(b).map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                } else {
                    // CPU fallback via ndarray matmul
                    let ra = a.to_ndarray().map_err(|_| NotImplemented)?;
                    let rb = b.to_ndarray().map_err(|_| NotImplemented)?;
                    if ra.ndim() != 2 || rb.ndim() != 2 {
                        return Err(NotImplemented);
                    }
                    let ra2 = ra
                        .into_dimensionality::<ndarray::Ix2>()
                        .map_err(|_| NotImplemented)?;
                    let rb2 = rb
                        .into_dimensionality::<ndarray::Ix2>()
                        .map_err(|_| NotImplemented)?;
                    let rc = ra2.dot(&rb2);
                    let new_shape = vec![rc.nrows(), rc.ncols()];
                    let flat: Vec<f32> = rc.into_iter().collect();
                    let result = GpuNdarray::<f32>::from_ndarray_data(
                        &flat,
                        new_shape,
                        Arc::clone(&a.context),
                    )
                    .map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                }
            }

            // ── sum ─────────────────────────────────────────────────────────
            "scirs2::array_protocol::operations::sum" => {
                let a = gpu_arg!(0);
                let axis = kwargs
                    .get("axis")
                    .and_then(|v| v.downcast_ref::<usize>().copied());

                match axis {
                    None => {
                        // Full reduction — return scalar
                        if use_gpu {
                            let total = a.dispatch_sum_all().map_err(|_| NotImplemented)?;
                            Ok(Box::new(total) as Box<dyn Any>)
                        } else {
                            let arr = a.to_ndarray().map_err(|_| NotImplemented)?;
                            let total: f32 = arr.sum();
                            Ok(Box::new(total) as Box<dyn Any>)
                        }
                    }
                    Some(ax) => {
                        // GPU axis reduction when above threshold; otherwise CPU fallback
                        let try_gpu = use_gpu && ax < a.shape.len();
                        if try_gpu {
                            match a.dispatch_sum_axis(ax) {
                                Ok(result) => {
                                    return Ok(
                                        Box::new(Box::new(result) as Box<dyn ArrayProtocol>)
                                            as Box<dyn Any>,
                                    );
                                }
                                Err(_) => {
                                    // fall through to CPU
                                }
                            }
                        }
                        let arr = a.to_ndarray().map_err(|_| NotImplemented)?;
                        let reduced = arr.sum_axis(ndarray::Axis(ax));
                        let new_shape = reduced.shape().to_vec();
                        let flat: Vec<f32> = reduced.into_iter().collect();
                        let result = GpuNdarray::<f32>::from_ndarray_data(
                            &flat,
                            new_shape,
                            Arc::clone(&a.context),
                        )
                        .map_err(|_| NotImplemented)?;
                        Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                    }
                }
            }

            // ── transpose ──────────────────────────────────────────────────
            "scirs2::array_protocol::operations::transpose" => {
                let a = gpu_arg!(0);
                if use_gpu && a.shape.len() == 2 {
                    let result = a.dispatch_transpose_2d().map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                } else {
                    // CPU fallback for non-2D or below threshold
                    let arr = a.to_ndarray().map_err(|_| NotImplemented)?;
                    let transposed = arr.t().to_owned();
                    let new_shape = transposed.shape().to_vec();
                    let flat: Vec<f32> = transposed.into_iter().collect();
                    let result = GpuNdarray::<f32>::from_ndarray_data(
                        &flat,
                        new_shape,
                        Arc::clone(&a.context),
                    )
                    .map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                }
            }

            // ── concatenate ─────────────────────────────────────────────────
            "scirs2::array_protocol::operations::concatenate" => {
                // axis is kwarg key = axis_value.to_string()
                let axis = kwargs
                    .values()
                    .find_map(|v| v.downcast_ref::<usize>().copied())
                    .unwrap_or(0);

                let gpu_arrays: Vec<&GpuNdarray<f32>> = args
                    .iter()
                    .filter_map(|arg| {
                        arg.downcast_ref::<Box<dyn ArrayProtocol>>()
                            .and_then(|ap| ap.as_any().downcast_ref::<GpuNdarray<f32>>())
                    })
                    .collect();

                if gpu_arrays.is_empty() {
                    return Err(NotImplemented);
                }

                // Helper: CPU fallback for concatenate
                let cpu_concat_fallback = |gpu_arrays: &[&GpuNdarray<f32>],
                                           axis: usize|
                 -> Result<Box<dyn Any>, NotImplemented> {
                    let arrs: Vec<ndarray::ArrayD<f32>> = gpu_arrays
                        .iter()
                        .map(|g| g.to_ndarray())
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| NotImplemented)?;
                    let views: Vec<ndarray::ArrayViewD<f32>> =
                        arrs.iter().map(|a| a.view()).collect();
                    let concatenated = ndarray::concatenate(ndarray::Axis(axis), &views)
                        .map_err(|_| NotImplemented)?;
                    let ctx = Arc::clone(&gpu_arrays[0].context);
                    let new_shape = concatenated.shape().to_vec();
                    let flat: Vec<f32> = concatenated.into_iter().collect();
                    let result = GpuNdarray::<f32>::from_ndarray_data(&flat, new_shape, ctx)
                        .map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                };

                if axis == 0 && use_gpu {
                    let result = GpuNdarray::<f32>::dispatch_concatenate_axis0(&gpu_arrays)
                        .map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                } else if axis > 0
                    && use_gpu
                    && gpu_arrays.len() >= 2
                    && gpu_arrays[0].shape.len() <= 8
                    && gpu_arrays[0].shape.iter().product::<usize>() >= GPU_THRESHOLD
                {
                    // GPU gather kernel for axis > 0; fold pair-wise for >2 arrays
                    let mut acc = gpu_arrays[0].clone();
                    let mut gpu_failed = false;
                    for next in gpu_arrays.iter().skip(1) {
                        match GpuNdarray::<f32>::dispatch_concatenate_axisn(&acc, next, axis) {
                            Ok(r) => acc = r,
                            Err(_) => {
                                gpu_failed = true;
                                break;
                            }
                        }
                    }
                    if gpu_failed {
                        cpu_concat_fallback(&gpu_arrays, axis)
                    } else {
                        Ok(Box::new(Box::new(acc) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                    }
                } else {
                    // CPU fallback
                    cpu_concat_fallback(&gpu_arrays, axis)
                }
            }

            // ── reshape — zero-copy ─────────────────────────────────────────
            "scirs2::array_protocol::operations::reshape" => {
                let a = gpu_arg!(0);
                let new_shape = kwargs
                    .get("shape")
                    .and_then(|v| v.downcast_ref::<Vec<usize>>().cloned())
                    .ok_or(NotImplemented)?;
                let result = a.dispatch_reshape(new_shape).map_err(|_| NotImplemented)?;
                Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
            }

            // ── svd — CPU fallback (download, run a real decomposition,
            // re-upload) ─────────────────────────────────────────────────
            "scirs2::array_protocol::operations::svd" => {
                let a = gpu_arg!(0);
                let arr = a.to_ndarray().map_err(|_| NotImplemented)?;
                if arr.ndim() != 2 {
                    return Err(NotImplemented);
                }
                let arr2 = arr
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| NotImplemented)?;
                let ctx = Arc::clone(&a.context);

                #[cfg(feature = "linalg")]
                {
                    let svd_result =
                        crate::linalg::svd_ndarray(&arr2).map_err(|_| NotImplemented)?;
                    let (u_rows, u_cols) = svd_result.u.dim();
                    let (vt_rows, vt_cols) = svd_result.vt.dim();
                    let s_len = svd_result.s.len();
                    let u_data: Vec<f32> = svd_result.u.into_iter().collect();
                    let s_data: Vec<f32> = svd_result.s.into_iter().collect();
                    let vt_data: Vec<f32> = svd_result.vt.into_iter().collect();

                    let u_gpu = GpuNdarray::<f32>::from_ndarray_data(
                        &u_data,
                        vec![u_rows, u_cols],
                        Arc::clone(&ctx),
                    )
                    .map_err(|_| NotImplemented)?;
                    let s_gpu = GpuNdarray::<f32>::from_ndarray_data(
                        &s_data,
                        vec![s_len],
                        Arc::clone(&ctx),
                    )
                    .map_err(|_| NotImplemented)?;
                    let vt_gpu =
                        GpuNdarray::<f32>::from_ndarray_data(&vt_data, vec![vt_rows, vt_cols], ctx)
                            .map_err(|_| NotImplemented)?;

                    Ok(Box::new((
                        Box::new(u_gpu) as Box<dyn ArrayProtocol>,
                        Box::new(s_gpu) as Box<dyn ArrayProtocol>,
                        Box::new(vt_gpu) as Box<dyn ArrayProtocol>,
                    )) as Box<dyn Any>)
                }
                #[cfg(not(feature = "linalg"))]
                {
                    // Without the `linalg` feature there is no in-crate real
                    // decomposition available — an honest `NotImplemented`
                    // rather than a fabricated identity/ones placeholder.
                    let _ = (arr2, ctx);
                    Err(NotImplemented)
                }
            }

            // ── inverse — CPU fallback (download, invert for real,
            // re-upload) ─────────────────────────────────────────────────
            "scirs2::array_protocol::operations::inverse" => {
                let a = gpu_arg!(0);
                let arr = a.to_ndarray().map_err(|_| NotImplemented)?;
                if arr.ndim() != 2 || arr.shape()[0] != arr.shape()[1] {
                    return Err(NotImplemented);
                }
                let m = arr.shape()[0];
                let arr2 = arr
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| NotImplemented)?;
                let ctx = Arc::clone(&a.context);

                #[cfg(feature = "linalg")]
                {
                    let inv = crate::linalg::inv_ndarray(&arr2).map_err(|_| NotImplemented)?;
                    let inv_data: Vec<f32> = inv.into_iter().collect();
                    let result = GpuNdarray::<f32>::from_ndarray_data(&inv_data, vec![m, m], ctx)
                        .map_err(|_| NotImplemented)?;
                    Ok(Box::new(Box::new(result) as Box<dyn ArrayProtocol>) as Box<dyn Any>)
                }
                #[cfg(not(feature = "linalg"))]
                {
                    // Without the `linalg` feature there is no in-crate real
                    // matrix inversion available — an honest `NotImplemented`
                    // rather than a fabricated identity placeholder.
                    let _ = (arr2, ctx, m);
                    Err(NotImplemented)
                }
            }

            _ => Err(NotImplemented),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> TypeId {
        TypeId::of::<f32>()
    }

    fn box_clone(&self) -> Box<dyn ArrayProtocol> {
        Box::new(self.clone())
    }
}

// ──────────────────────────────────────────────────────────────────────
// 6. GPUArray trait impl
// ──────────────────────────────────────────────────────────────────────

impl GPUArray for GpuNdarray<f32> {
    fn to_gpu(&self) -> CoreResult<Box<dyn GPUArray>> {
        // Already on GPU; cheap clone
        Ok(Box::new(self.clone()))
    }

    fn to_cpu(&self) -> CoreResult<Box<dyn ArrayProtocol>> {
        let arr = self.to_ndarray().map_err(|e| {
            CoreError::ComputationError(ErrorContext::new(format!("GPU→CPU readback: {e}")))
        })?;
        Ok(Box::new(NdarrayWrapper::new(arr)))
    }

    fn is_on_gpu(&self) -> bool {
        true
    }

    fn device_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("backend".to_string(), "wgpu".to_string());
        info.insert("dtype".to_string(), "f32".to_string());
        info.insert("shape".to_string(), format!("{:?}", self.shape));
        info
    }
}

// ──────────────────────────────────────────────────────────────────────
// 7. WGSL kernels — see gpu_ndarray_shaders.rs
// ──────────────────────────────────────────────────────────────────────
use super::gpu_ndarray_shaders::{
    CONCAT_AXISN_WGSL, ELEMENTWISE_ADD_WGSL, ELEMENTWISE_MUL_WGSL, ELEMENTWISE_SUB_WGSL,
    MATMUL_WGSL, REDUCE_SUM_AXIS_WGSL, SCALAR_MUL_WGSL, SUM_REDUCE_WGSL, TRANSPOSE_WGSL,
};

// ──────────────────────────────────────────────────────────────────────
// 8. Tests — value-correctness for the svd/inverse CPU-fallback dispatch
// (split out to keep gpu_ndarray.rs under the workspace's 2000-line-per-file
// limit — see neural.rs/neural_backward_tests.rs for the same pattern).
// ──────────────────────────────────────────────────────────────────────
#[cfg(all(test, feature = "linalg"))]
#[path = "gpu_ndarray_dispatch_tests.rs"]
mod dispatch_correctness_tests;
