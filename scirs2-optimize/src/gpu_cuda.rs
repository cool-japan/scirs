#![cfg(feature = "cuda")]
//! Optional, off-by-default, NVIDIA-only CUDA acceleration for the
//! Hessian-vector products at the heart of the Newton-CG inner loop, calling
//! the pure-Rust `oxicuda-*` library crates directly.
//!
//! This module is ADDITIVE and entirely feature-gated behind the `cuda` feature
//! (off by default). It does not replace or modify the existing `gpu` subsystem
//! (`src/gpu/`) or the wgpu Newton-CG / CG / L-BFGS paths
//! (`src/unconstrained/{newton_gpu,cg_gpu,lbfgs_gpu}.rs`). Every public function
//! is `f64`-native — there is no silent `f32` downcast at the boundary, which is
//! the advantage of the oxicuda CUDA path over the f32 wgpu path.
//!
//! Each entry point degrades safely when no NVIDIA device is present:
//! [`cuda_is_available`] never panics, and the compute functions return an
//! [`crate::error::OptimizeError::ComputationError`] rather than aborting.
//!
//! # Library-call pattern
//!
//! The inner CG loop of the Newton-CG subsystem performs, per iteration, a
//! Hessian-vector product `H · p` (a dense matrix×vector multiply, `O(n²)`)
//! that dominates the per-iteration cost for large `n`. This module hands that
//! product off to the pre-built, battle-tested `oxicuda-blas` GEMM rather than
//! authoring any bespoke kernels, mirroring the proven matrix×vector layout of
//! the Phase-1 interpolate reference
//! (`scirs2-interpolate/src/gpu_cuda.rs::cuda_eval_gemm`).
//!
//! # Scoping note
//!
//! This is a standalone additive function, deliberately NOT wired into
//! `minimize_newton_cg` / `newton.rs` — exactly like the Phase-1 crates kept
//! their `cuda_*` functions standalone.

use crate::error::{OptimizeError, OptimizeResult};
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
fn blas_err(e: oxicuda_blas::error::BlasError) -> OptimizeError {
    OptimizeError::ComputationError(format!("oxicuda-blas: {e}"))
}

/// Maps an `oxicuda` CUDA driver/memory error onto `ComputationError`.
fn cuda_err(e: oxicuda_driver::CudaError) -> OptimizeError {
    OptimizeError::ComputationError(format!("oxicuda CUDA driver: {e}"))
}

/// Initialises the CUDA driver, selects device 0, and builds a shared context
/// wrapped in an `Arc` as required by the oxicuda BLAS handle constructor.
///
/// Returns a [`ComputationError`](OptimizeError::ComputationError) — never a
/// fabricated success — when CUDA is unavailable or no device is present.
fn build_context() -> OptimizeResult<std::sync::Arc<oxicuda_driver::Context>> {
    oxicuda_driver::init()
        .map_err(|e| OptimizeError::ComputationError(format!("CUDA unavailable: {e}")))?;
    let count = oxicuda_driver::device::Device::count()
        .map_err(|e| OptimizeError::ComputationError(format!("device count: {e}")))?;
    if count <= 0 {
        return Err(OptimizeError::ComputationError(
            "no NVIDIA CUDA device available".into(),
        ));
    }
    let dev = oxicuda_driver::device::Device::get(0).map_err(cuda_err)?;
    Ok(std::sync::Arc::new(
        oxicuda_driver::Context::new(&dev).map_err(cuda_err)?,
    ))
}

/// Compute the Hessian-vector product `H · v` on the GPU in `f64`.
///
/// This is the dominant per-iteration cost of the Newton-CG inner loop. `H` is
/// the `n`×`n` Hessian and `v` is a length-`n` vector; the returned vector has
/// length `n`.
///
/// Layout correctness (cannot be runtime-checked on a non-NVIDIA host): ndarray
/// is row-major. We materialize a contiguous row-major copy of `H` via
/// `as_standard_layout`
/// and describe it as a [`Layout::RowMajor`] `MatrixDesc` (`n`×`n`). Mirroring
/// the proven interpolate `cuda_eval_gemm` matrix×vector layout, `v` is
/// described as a column-vector [`Layout::RowMajor`] `MatrixDesc` (`n`×1) and
/// the output as a [`Layout::RowMajor`] `MatrixDescMut` (`n`×1) — a
/// single-column matrix is byte-identical under either layout tag, and
/// `oxicuda-blas`'s GEMM dispatcher requires RowMajor on every operand. Under
/// [`Transpose::NoTrans`] (`alpha = 1.0`, `beta = 0.0`) this yields `M = n`,
/// `K = n`, `N = 1`.
pub fn cuda_hessian_vector_product(
    h: &ArrayView2<f64>,
    v: &ArrayView1<f64>,
) -> OptimizeResult<Array1<f64>> {
    let n = h.nrows();
    if !h.is_square() {
        return Err(OptimizeError::ValueError(format!(
            "cuda_hessian_vector_product: Hessian must be square, got {n}x{}",
            h.ncols()
        )));
    }
    if v.len() != n {
        return Err(OptimizeError::ValueError(format!(
            "cuda_hessian_vector_product: vector length {} does not match Hessian dimension {n}",
            v.len()
        )));
    }
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    // Contiguous row-major copy so `.as_slice()` is `Some` and the device upload
    // matches the `Layout::RowMajor` descriptor below.
    let h_std = h.as_standard_layout();
    let h_slice = h_std.as_slice().ok_or_else(|| {
        OptimizeError::ComputationError("cuda_hessian_vector_product: H not contiguous".into())
    })?;
    let v_vec: Vec<f64> = v.iter().copied().collect();

    let ctx = build_context()?;
    let handle = oxicuda_blas::BlasHandle::new(&ctx).map_err(blas_err)?;

    let d_h = oxicuda_memory::DeviceBuffer::from_host(h_slice).map_err(cuda_err)?;
    let d_v = oxicuda_memory::DeviceBuffer::from_host(&v_vec).map_err(cuda_err)?;
    let mut d_c = oxicuda_memory::DeviceBuffer::<f64>::alloc(n).map_err(cuda_err)?;

    // H as RowMajor (n×n), v as a RowMajor column (n×1), output RowMajor (n×1).
    // A single-column matrix has one possible packed layout regardless of the
    // tag, but oxicuda-blas's GEMM dispatcher hard-rejects any non-RowMajor
    // descriptor, so all three must be tagged RowMajor.
    let a_desc =
        MatrixDesc::from_buffer(&d_h, n as u32, n as u32, Layout::RowMajor).map_err(blas_err)?;
    let b_desc = MatrixDesc::from_buffer(&d_v, n as u32, 1, Layout::RowMajor).map_err(blas_err)?;
    let mut c_desc =
        MatrixDescMut::from_buffer(&mut d_c, n as u32, 1, Layout::RowMajor).map_err(blas_err)?;

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

    // The GEMM runs on the BLAS handle's `CU_STREAM_NON_BLOCKING` stream, but
    // `copy_to_host` issues its `cuMemcpyDtoH_v2` on the legacy default stream,
    // which a non-blocking stream does NOT implicitly synchronise with. Without
    // this wait the copy races the kernel and reads the (uninitialised / zero)
    // output before the GEMM finishes — the longer the kernel runs (large n),
    // the more often it reads all-zeros. Synchronise the compute stream first.
    handle.stream().synchronize().map_err(cuda_err)?;

    let mut hv = vec![0.0f64; n];
    d_c.copy_to_host(&mut hv).map_err(cuda_err)?;
    Ok(Array1::from_vec(hv))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn mat(rows: usize, cols: usize, data: Vec<f64>) -> Array2<f64> {
        Array2::from_shape_vec((rows, cols), data).expect("valid test matrix shape")
    }

    #[test]
    fn cuda_hessian_vector_product_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }
        // Symmetric 3x3 Hessian and a vector; verify H·v against a CPU dot.
        let h = mat(3, 3, vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]);
        let v = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let hv = cuda_hessian_vector_product(&h.view(), &v.view())
            .expect("cuda_hessian_vector_product failed");
        let expected = h.dot(&v);
        let max_diff = hv
            .iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-9, "max abs diff {max_diff} exceeds 1e-9");
    }

    #[test]
    fn cuda_hessian_vector_product_nonsymmetric_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }
        // Non-symmetric 4×4 matrix — catches any row/col-major transpose bug that
        // a symmetric Hessian (H = Hᵀ) would silently hide because swapping rows
        // and columns of a symmetric matrix yields the same result.
        let h = mat(
            4,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        );
        let v = Array1::from_vec(vec![1.0, -1.0, 2.0, -2.0]);
        let hv = cuda_hessian_vector_product(&h.view(), &v.view())
            .expect("cuda_hessian_vector_product nonsymmetric failed");
        let expected = h.dot(&v);
        let max_diff = hv
            .iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 1e-9,
            "nonsymmetric 4x4 HVP max abs diff {max_diff} exceeds 1e-9"
        );
    }

    #[test]
    fn cuda_hessian_vector_product_shape_mismatch_errors() {
        // Runs without a GPU: validation rejects before any device call.
        // Non-square Hessian.
        let nonsquare = mat(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(cuda_hessian_vector_product(&nonsquare.view(), &v.view()).is_err());

        // Square Hessian but mismatched vector length.
        let square = mat(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let v_bad = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(cuda_hessian_vector_product(&square.view(), &v_bad.view()).is_err());
    }
}
