//! GPU device capability information.
//!
//! This module defines [`GpuDeviceInfo`], a backend-agnostic, queryable
//! description of a GPU device's capabilities. It is produced by
//! [`crate::gpu::GpuDevice::get_info`] and is intended to be consumed by callers
//! (such as `scirs2-io`'s backend management layer) that need to validate device
//! capabilities before scheduling work.
//!
//! In the default Pure-Rust build no GPU SDKs are linked, so the values returned
//! for hardware backends are conservative, deterministic placeholders rather
//! than live adapter queries. The struct and method form a stable API surface
//! that can later be wired to real adapter introspection without changing the
//! public contract.

use crate::gpu::GpuBackend;

/// Capability information describing a single GPU device.
///
/// All fields are public so callers can perform capability validation directly.
/// Values are deterministic for a given [`GpuBackend`]; for hardware backends in
/// the default build they are conservative placeholders (see the module-level
/// documentation).
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Human-readable device name (for example `"CPU"` or `"WebGPU Device"`).
    pub device_name: String,

    /// High-level device classification (for example `"CPU"` or `"Discrete GPU"`).
    pub device_type: String,

    /// The backend this device is exposed through.
    pub backend: GpuBackend,

    /// Backend-specific compute-capability descriptor.
    ///
    /// For CUDA this mirrors the SM version string (for example `"8.0"`); for
    /// other backends it is a backend-appropriate descriptor or `"unknown"`.
    pub compute_capability: String,

    /// Total device memory in bytes. A value of `0` means "unknown".
    pub total_memory: u64,

    /// Currently available device memory in bytes. A value of `0` means
    /// "unknown".
    pub available_memory: u64,

    /// Maximum number of work-items in a single work group (CUDA block /
    /// OpenCL work-group / compute workgroup invocation count).
    pub max_work_group_size: u32,

    /// Whether the device supports 64-bit floating point (`f64`) computation.
    pub supports_fp64: bool,

    /// Whether the device supports 16-bit floating point (`f16`) computation.
    pub supports_fp16: bool,
}

impl GpuDeviceInfo {
    /// Build a deterministic [`GpuDeviceInfo`] for the given backend.
    ///
    /// The returned values are conservative placeholders suitable for capability
    /// validation in the default Pure-Rust build, where no GPU SDK is linked.
    /// They are stable for a given backend so callers can rely on them in tests
    /// and configuration logic.
    #[must_use]
    pub fn for_backend(backend: GpuBackend) -> Self {
        match backend {
            GpuBackend::Cpu => Self {
                device_name: "CPU".to_string(),
                device_type: "CPU".to_string(),
                backend,
                compute_capability: "host".to_string(),
                // 0 means "unknown" here: the host memory size is not probed in
                // the default build to keep this query side-effect free.
                total_memory: 0,
                available_memory: 0,
                max_work_group_size: 1,
                // A scalar host fallback can always emulate both precisions.
                supports_fp64: true,
                supports_fp16: true,
            },
            GpuBackend::Cuda => Self {
                device_name: "CUDA Device".to_string(),
                device_type: "Discrete GPU".to_string(),
                backend,
                compute_capability: "unknown".to_string(),
                total_memory: 0,
                available_memory: 0,
                max_work_group_size: 1024,
                supports_fp64: true,
                supports_fp16: true,
            },
            GpuBackend::Rocm => Self {
                device_name: "ROCm Device".to_string(),
                device_type: "Discrete GPU".to_string(),
                backend,
                compute_capability: "unknown".to_string(),
                total_memory: 0,
                available_memory: 0,
                max_work_group_size: 1024,
                supports_fp64: true,
                supports_fp16: true,
            },
            GpuBackend::Wgpu => Self {
                device_name: "WebGPU Device".to_string(),
                device_type: "GPU".to_string(),
                backend,
                compute_capability: "wgsl".to_string(),
                total_memory: 0,
                available_memory: 0,
                // WebGPU's guaranteed minimum maxComputeInvocationsPerWorkgroup.
                max_work_group_size: 256,
                // f64 is not part of core WebGPU; f16 is gated behind an
                // optional feature, so report conservatively.
                supports_fp64: false,
                supports_fp16: false,
            },
            GpuBackend::Metal => Self {
                device_name: "Metal Device".to_string(),
                device_type: "Integrated GPU".to_string(),
                backend,
                compute_capability: "msl".to_string(),
                total_memory: 0,
                available_memory: 0,
                max_work_group_size: 1024,
                // Metal does not expose double precision in shaders.
                supports_fp64: false,
                supports_fp16: true,
            },
            GpuBackend::OpenCL => Self {
                device_name: "OpenCL Device".to_string(),
                device_type: "GPU".to_string(),
                backend,
                compute_capability: "unknown".to_string(),
                total_memory: 0,
                available_memory: 0,
                max_work_group_size: 256,
                // fp64/fp16 are optional OpenCL extensions; report conservatively.
                supports_fp64: false,
                supports_fp16: false,
            },
        }
    }
}
