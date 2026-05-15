//! GPU kernel implementations for batch special-function evaluation.
//!
//! This module houses compute-shader sources and host-side dispatch stubs for
//! two GPU backends:
//!
//! | Sub-module | Backend | Feature flag | Default |
//! |------------|---------|--------------|---------|
//! | [`wgsl`]   | WebGPU (WGSL shaders) | `gpu` via `scirs2-core/gpu` | off |
//! | [`cuda`]   | CUDA / ROCm           | `cuda_kernels`              | off |
//!
//! Both backends currently provide stub implementations that return
//! "not available" errors.  The actual dispatch logic lives in
//! [`crate::gpu_dispatch`] which calls these stubs under
//! `DispatchTarget::Gpu` and falls back to the CPU rayon path automatically.
//!
//! # Pure Rust policy
//!
//! The `cuda_kernels` feature is off by default.  CUDA depends on C dynamic
//! libraries (`libcuda.so`, `libcudart.so`) which violate the Pure Rust
//! Policy unless feature-gated.  WebGPU is Pure Rust (via `wgpu`) but the
//! host-side integration with `scirs2-core/gpu` is not yet stabilised.

pub mod cuda;
pub mod wgsl;

pub use cuda::{
    bessel_j0_batch_cuda, erf_batch_cuda, gamma_batch_cuda, CudaDispatchError, BESSEL_J0_PTX_STUB,
    ERF_PTX_STUB, GAMMA_PTX_STUB,
};
pub use wgsl::{
    bessel_j0_batch_wgpu, erf_batch_wgpu, erfc_batch_wgpu, erfinv_batch_wgpu, gamma_batch_wgpu,
    lgamma_batch_wgpu, WgslDispatchError, BESSEL_J0_WGSL, ERFC_WGSL, ERFINV_WGSL, ERF_WGSL,
    GAMMA_WGSL, LGAMMA_WGSL,
};
