#![cfg(feature = "cuda")]
//! Optional, off-by-default, NVIDIA-only CUDA acceleration for the dense
//! regression-target matmul at the heart of synthetic dataset generation,
//! calling the pure-Rust `oxicuda-*` library crates directly.
//!
//! This module is ADDITIVE and entirely feature-gated behind the `cuda` feature
//! (off by default). It does not replace or modify the existing GPU subsystem
//! (`src/gpu.rs`, `src/gpu_optimization.rs`) or the wgpu-backed `wgpu` path.
//! Every public function is `f64`-native — there is no silent `f32` downcast at
//! the boundary, which is the advantage of the oxicuda CUDA path over the f32
//! wgpu path.
//!
//! Each entry point degrades safely when no NVIDIA device is present:
//! [`cuda_is_available`] never panics, and the compute functions return a
//! [`crate::error::DatasetsError::ComputationError`] rather than aborting.
//!
//! # Library-call pattern
//!
//! The regression-target generator at the heart of `make_regression` forms the
//! dense product `y = X · coef` (`[n_samples × n_features] × [n_features] →
//! [n_samples]`), an `O(n_samples · n_features)` matrix×vector multiply. This
//! module hands that product off to the pre-built, battle-tested `oxicuda-blas`
//! GEMM rather than authoring any bespoke kernels, mirroring the proven
//! matrix×vector layout of the Phase-1 interpolate reference
//! (`scirs2-interpolate/src/gpu_cuda.rs::cuda_eval_gemm`) and the optimize
//! Phase-2 reference (`scirs2-optimize/src/gpu_cuda.rs`).
//!
//! # Scoping note
//!
//! This is a standalone additive function, deliberately NOT wired into
//! `make_regression` / `generators/basic.rs` — exactly like the Phase-1 crates
//! kept their `cuda_*` functions standalone.

use crate::error::{DatasetsError, Result};
use oxicuda_blas::types::{Layout, MatrixDesc, MatrixDescMut, Transpose};
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};

/// Returns `true` iff the CUDA driver initializes and at least one NVIDIA device
/// is visible. Never panics.
///
/// On a non-NVIDIA platform such as macOS the CUDA driver fails to initialise
/// (or reports zero devices) and this returns `false`, allowing callers to fall
/// back to the CPU path.
pub fn cuda_is_available() -> bool {
    oxicuda_driver::init().is_ok()
        && oxicuda_driver::device::Device::count()
            .map(|c| c > 0)
            .unwrap_or(false)
}

/// Maps an `oxicuda-blas` error onto the crate's honest `ComputationError`.
fn blas_err(e: oxicuda_blas::error::BlasError) -> DatasetsError {
    DatasetsError::ComputationError(format!("oxicuda-blas: {e}"))
}

/// Maps an `oxicuda` CUDA driver/memory error onto `ComputationError`.
fn cuda_err(e: oxicuda_driver::CudaError) -> DatasetsError {
    DatasetsError::ComputationError(format!("oxicuda CUDA driver: {e}"))
}

/// Initialises the CUDA driver, selects device 0, and builds a shared context
/// wrapped in an `Arc` as required by the oxicuda BLAS handle constructor.
///
/// Returns a [`ComputationError`](DatasetsError::ComputationError) — never a
/// fabricated success — when CUDA is unavailable or no device is present.
fn build_context() -> Result<std::sync::Arc<oxicuda_driver::Context>> {
    oxicuda_driver::init()
        .map_err(|e| DatasetsError::ComputationError(format!("CUDA unavailable: {e}")))?;
    let count = oxicuda_driver::device::Device::count()
        .map_err(|e| DatasetsError::ComputationError(format!("device count: {e}")))?;
    if count <= 0 {
        return Err(DatasetsError::ComputationError(
            "no NVIDIA CUDA device available".into(),
        ));
    }
    let dev = oxicuda_driver::device::Device::get(0).map_err(cuda_err)?;
    Ok(std::sync::Arc::new(
        oxicuda_driver::Context::new(&dev).map_err(cuda_err)?,
    ))
}

/// Compute the regression target `y = X · coef` on the GPU in `f64`.
///
/// This is the dense matmul at the heart of `make_regression`: `X` is the
/// `[n_samples × n_features]` design matrix and `coef` is the length-`n_features`
/// coefficient vector; the returned target vector has length `n_samples`.
///
/// Layout correctness (cannot be runtime-checked on a non-NVIDIA host): ndarray
/// is row-major. We materialize a contiguous row-major copy of `X` via
/// [`as_standard_layout`](scirs2_core::ndarray::ArrayBase::as_standard_layout)
/// and describe it as a [`Layout::RowMajor`] `MatrixDesc`
/// (`n_samples`×`n_features`). Mirroring the proven optimize/interpolate
/// matrix×vector layout, `coef` is described as a column-vector
/// [`Layout::ColMajor`] `MatrixDesc` (`n_features`×1) and the output as a
/// [`Layout::ColMajor`] `MatrixDescMut` (`n_samples`×1). Under
/// [`Transpose::NoTrans`] (`alpha = 1.0`, `beta = 0.0`) this yields
/// `M = n_samples`, `K = n_features`, `N = 1`.
pub fn cuda_regression_target(x: &ArrayView2<f64>, coef: &ArrayView1<f64>) -> Result<Array1<f64>> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    if coef.len() != n_features {
        return Err(DatasetsError::ValidationError(format!(
            "cuda_regression_target: coef length {} does not match X column count {n_features}",
            coef.len()
        )));
    }
    if n_samples == 0 {
        return Ok(Array1::zeros(0));
    }
    if n_features == 0 {
        // No features => the linear combination is empty, so every target is 0.
        return Ok(Array1::zeros(n_samples));
    }

    // Contiguous row-major copy so `.as_slice()` is `Some` and the device upload
    // matches the `Layout::RowMajor` descriptor below.
    let x_std = x.as_standard_layout();
    let x_slice = x_std.as_slice().ok_or_else(|| {
        DatasetsError::ComputationError("cuda_regression_target: X not contiguous".into())
    })?;
    let coef_vec: Vec<f64> = coef.iter().copied().collect();

    let ctx = build_context()?;
    let handle = oxicuda_blas::BlasHandle::new(&ctx).map_err(blas_err)?;

    let d_x = oxicuda_memory::DeviceBuffer::from_host(x_slice).map_err(cuda_err)?;
    let d_coef = oxicuda_memory::DeviceBuffer::from_host(&coef_vec).map_err(cuda_err)?;
    let mut d_y = oxicuda_memory::DeviceBuffer::<f64>::alloc(n_samples).map_err(cuda_err)?;

    // X as RowMajor (n_samples×n_features), coef as a ColMajor column
    // (n_features×1), output ColMajor (n_samples×1) — the exact operand layout
    // of optimize's `cuda_hessian_vector_product` / interpolate's
    // `cuda_eval_gemm`.
    let a_desc =
        MatrixDesc::from_buffer(&d_x, n_samples as u32, n_features as u32, Layout::RowMajor)
            .map_err(blas_err)?;
    let b_desc = MatrixDesc::from_buffer(&d_coef, n_features as u32, 1, Layout::ColMajor)
        .map_err(blas_err)?;
    let mut c_desc = MatrixDescMut::from_buffer(&mut d_y, n_samples as u32, 1, Layout::ColMajor)
        .map_err(blas_err)?;

    oxicuda_blas::level3::gemm_api::gemm::<f64>(
        &handle,
        Transpose::NoTrans,
        Transpose::NoTrans,
        1.0,
        &a_desc,
        &b_desc,
        0.0,
        &mut c_desc,
    )
    .map_err(blas_err)?;

    let mut y = vec![0.0f64; n_samples];
    d_y.copy_to_host(&mut y).map_err(cuda_err)?;
    Ok(Array1::from_vec(y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn mat(rows: usize, cols: usize, data: Vec<f64>) -> Array2<f64> {
        Array2::from_shape_vec((rows, cols), data).expect("valid test matrix shape")
    }

    #[test]
    fn cuda_regression_target_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }
        // X (3x2) row-major; coef [1, 1] => row sums [3, 7, 11].
        let x = mat(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let coef = Array1::from_vec(vec![1.0, 1.0]);
        let y =
            cuda_regression_target(&x.view(), &coef.view()).expect("cuda_regression_target failed");
        let expected = x.dot(&coef);
        let max_diff = y
            .iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-9, "max abs diff {max_diff} exceeds 1e-9");
    }

    #[test]
    fn cuda_regression_target_shape_mismatch_errors() {
        // Runs without a GPU: validation rejects before any device call.
        // X has 2 columns but coef has length 3.
        let x = mat(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let coef_bad = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(cuda_regression_target(&x.view(), &coef_bad.view()).is_err());
    }
}
