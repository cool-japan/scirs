//! CUDA/ROCm kernel dispatch stubs for batch special-function evaluation.
//!
//! This module is feature-gated behind `cuda_kernels` (default: off) to
//! satisfy the Pure Rust Policy.  A CUDA build requires linking against
//! `libcuda.so` / `libcudart.so`, which are C dynamic libraries and must
//! therefore remain optional.
//!
//! # Current state
//!
//! All public functions are stubs that immediately return
//! [`CudaDispatchError::CudaNotAvailable`].  The dispatcher in
//! [`crate::gpu_dispatch`] calls these under the `DispatchTarget::Gpu`
//! branch and falls back to CPU when it receives this error.
//!
//! # Future integration
//!
//! When a `cuda-sys` (or `cudarc`) binding is introduced behind this feature:
//!
//! 1. Compile PTX kernels at build time with `nvcc` (or inline PTX strings).
//! 2. Allocate device memory via `cuMemAlloc`.
//! 3. Copy host → device, launch the kernel, copy device → host.
//! 4. Map the `f64` arrays through `f32` or use `__nv_bfloat16` depending
//!    on the target architecture.
//!
//! ROCm / HIP support follows an analogous path through `hipMalloc` /
//! `hipMemcpy` / `hipLaunchKernelGGL`.

// ---------------------------------------------------------------------------
// Dispatch error
// ---------------------------------------------------------------------------

/// Error type for CUDA/ROCm dispatch.
#[derive(Debug, Clone)]
pub enum CudaDispatchError {
    /// CUDA runtime is not available on this host.
    CudaNotAvailable,
    /// A CUDA runtime call failed (driver / cuBLAS / cuDNN error code).
    RuntimeError(String),
    /// The feature flag `cuda_kernels` is not enabled.
    FeatureNotEnabled,
}

impl std::fmt::Display for CudaDispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaDispatchError::CudaNotAvailable => {
                write!(f, "CUDA runtime not available on this host")
            }
            CudaDispatchError::RuntimeError(msg) => {
                write!(f, "CUDA runtime error: {msg}")
            }
            CudaDispatchError::FeatureNotEnabled => {
                write!(f, "feature `cuda_kernels` not enabled at compile time")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PTX kernel source stubs
// ---------------------------------------------------------------------------

/// Inline PTX stub for batch Gamma evaluation on CUDA devices.
///
/// A production implementation would replace this with a compiled PTX or
/// CUBIN binary embedded via `include_bytes!` or generated at build time
/// by `nvcc`.
pub const GAMMA_PTX_STUB: &str = r#"
// gamma_batch.cu  (stub — not compiled)
//
// __global__ void gamma_batch_kernel(const float* in, float* out, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= n) return;
//     out[idx] = tgammaf(in[idx]);   // CUDA built-in
// }
"#;

/// Inline PTX stub for batch `erf` evaluation.
pub const ERF_PTX_STUB: &str = r#"
// erf_batch.cu  (stub — not compiled)
//
// __global__ void erf_batch_kernel(const float* in, float* out, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= n) return;
//     out[idx] = erff(in[idx]);   // CUDA built-in
// }
"#;

/// Inline PTX stub for batch Bessel J₀ evaluation.
pub const BESSEL_J0_PTX_STUB: &str = r#"
// bessel_j0_batch.cu  (stub — not compiled)
//
// __global__ void bessel_j0_batch_kernel(const float* in, float* out, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= n) return;
//     out[idx] = j0f(in[idx]);   // POSIX math built-in (available in libm / CUDA)
// }
"#;

// ---------------------------------------------------------------------------
// Dispatch stubs
// ---------------------------------------------------------------------------

/// Attempt batch Gamma evaluation on a CUDA device.
///
/// Always returns [`CudaDispatchError::CudaNotAvailable`] until the
/// `cuda_kernels` feature is implemented.
#[allow(unused_variables)]
pub fn gamma_batch_cuda(xs: &[f64]) -> Result<Vec<f64>, CudaDispatchError> {
    #[cfg(feature = "cuda_kernels")]
    {
        // Future: allocate, copy, launch GAMMA_PTX kernel, download.
        return Err(CudaDispatchError::RuntimeError(
            "cuda_kernels feature is declared but not yet implemented".into(),
        ));
    }
    #[allow(unreachable_code)]
    Err(CudaDispatchError::FeatureNotEnabled)
}

/// Attempt batch `erf` evaluation on a CUDA device.
///
/// Always returns [`CudaDispatchError::FeatureNotEnabled`] until the
/// `cuda_kernels` feature is implemented.
#[allow(unused_variables)]
pub fn erf_batch_cuda(xs: &[f64]) -> Result<Vec<f64>, CudaDispatchError> {
    #[cfg(feature = "cuda_kernels")]
    {
        return Err(CudaDispatchError::RuntimeError(
            "cuda_kernels feature is declared but not yet implemented".into(),
        ));
    }
    #[allow(unreachable_code)]
    Err(CudaDispatchError::FeatureNotEnabled)
}

/// Attempt batch Bessel J₀ evaluation on a CUDA device.
///
/// Always returns [`CudaDispatchError::FeatureNotEnabled`] until the
/// `cuda_kernels` feature is implemented.
#[allow(unused_variables)]
pub fn bessel_j0_batch_cuda(xs: &[f64]) -> Result<Vec<f64>, CudaDispatchError> {
    #[cfg(feature = "cuda_kernels")]
    {
        return Err(CudaDispatchError::RuntimeError(
            "cuda_kernels feature is declared but not yet implemented".into(),
        ));
    }
    #[allow(unreachable_code)]
    Err(CudaDispatchError::FeatureNotEnabled)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_cuda_stub_returns_feature_not_enabled() {
        let xs = vec![1.0_f64, 2.0, 3.0];
        let result = gamma_batch_cuda(&xs);
        // Without cuda_kernels feature: FeatureNotEnabled
        // With cuda_kernels feature (not yet impl): RuntimeError
        assert!(
            matches!(
                result,
                Err(CudaDispatchError::FeatureNotEnabled) | Err(CudaDispatchError::RuntimeError(_))
            ),
            "unexpected result: {:?}",
            result
        );
    }

    #[test]
    fn test_erf_cuda_stub_returns_error() {
        let xs = vec![0.0_f64, 1.0];
        let result = erf_batch_cuda(&xs);
        assert!(result.is_err());
    }

    #[test]
    fn test_bessel_j0_cuda_stub_returns_error() {
        let xs = vec![0.0_f64, 2.405];
        let result = bessel_j0_batch_cuda(&xs);
        assert!(result.is_err());
    }

    #[test]
    fn test_cuda_dispatch_error_display() {
        let e = CudaDispatchError::CudaNotAvailable;
        assert!(e.to_string().contains("not available"));

        let e2 = CudaDispatchError::RuntimeError("device OOM".into());
        assert!(e2.to_string().contains("device OOM"));

        let e3 = CudaDispatchError::FeatureNotEnabled;
        assert!(e3.to_string().contains("cuda_kernels"));
    }

    #[test]
    fn test_ptx_stub_sources_are_non_empty() {
        assert!(!GAMMA_PTX_STUB.is_empty());
        assert!(!ERF_PTX_STUB.is_empty());
        assert!(!BESSEL_J0_PTX_STUB.is_empty());
    }
}
