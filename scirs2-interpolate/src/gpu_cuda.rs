#![cfg(feature = "cuda")]
//! Optional, off-by-default, NVIDIA-only CUDA acceleration for the dense linear
//! algebra at the heart of RBF interpolation.
//!
//! This module is compiled only when the `cuda` feature is enabled. Even then it
//! is *runtime-probed*: [`cuda_is_available`] returns `false` on any platform
//! without an NVIDIA CUDA device (for example macOS), so callers can compile
//! everywhere and dispatch to the CPU path when no device is present. Everything
//! here is `f64`-native — there is no silent `f32` downcast at the boundary.
//!
//! # Library-call pattern (vs. custom-kernel pattern)
//!
//! Unlike `scirs2-symbolic`'s CUDA path — which *emits custom kernels* through
//! the `oxicuda-ptx` builder — this module follows the **library-call pattern**:
//! it hands the heavy dense linear algebra off to the pre-built, battle-tested
//! oxicuda *library* crates (`oxicuda-solver` for the Cholesky factor/solve and
//! `oxicuda-blas` for the evaluation GEMM). No bespoke kernels are authored here.
//!
//! # Scoping note: kernel-matrix construction stays on the CPU (PENDING)
//!
//! Building the RBF kernel *matrix* itself on the GPU — the entrywise
//! `gaussian` `exp(-r^2)`, the `multiquadric` `sqrt`, and so on — would require
//! custom transcendental kernels and runs straight into the *same*
//! `oxicuda-ptx` DSL gap already documented in `scirs2-symbolic`'s CUDA path:
//! the public `oxicuda-ptx` builder DSL exposes neither `f64` transcendentals
//! nor division. Therefore this slice deliberately keeps kernel-matrix
//! **construction** on the CPU and accelerates **only** the dense linear
//! algebra that follows it: the symmetric-positive-definite kernel-system
//! **solve** (`oxicuda-solver` Cholesky) and the evaluation **GEMM**
//! (`oxicuda-blas`). The kernel-matrix-construction step is **PENDING**
//! `oxicuda-ptx` transcendental support.

use crate::error::{InterpolateError, InterpolateResult};
use oxicuda_blas::types::{FillMode, Layout, MatrixDesc, MatrixDescMut, Transpose};

/// Returns `true` only when an NVIDIA CUDA device is present and the driver
/// initialises successfully.
///
/// This never panics: on a non-NVIDIA platform such as macOS the CUDA driver
/// fails to initialise (or reports zero devices) and this returns `false`,
/// allowing callers to fall back to the CPU path.
pub fn cuda_is_available() -> bool {
    oxicuda_driver::init().is_ok()
        && oxicuda_driver::device::Device::count()
            .map(|c| c > 0)
            .unwrap_or(false)
}

/// Maps an `oxicuda-solver` error onto the crate's honest `ComputationError`.
fn solver_err(e: oxicuda_solver::error::SolverError) -> InterpolateError {
    InterpolateError::ComputationError(format!("oxicuda-solver: {e}"))
}

/// Maps an `oxicuda-blas` error onto the crate's honest `ComputationError`.
fn blas_err(e: oxicuda_blas::error::BlasError) -> InterpolateError {
    InterpolateError::ComputationError(format!("oxicuda-blas: {e}"))
}

/// Maps an `oxicuda` CUDA driver/memory error onto `ComputationError`.
fn cuda_err(e: oxicuda_driver::CudaError) -> InterpolateError {
    InterpolateError::ComputationError(format!("oxicuda CUDA driver: {e}"))
}

/// Initialises the CUDA driver, selects device 0, and builds a shared context.
///
/// Returns a [`ComputationError`](InterpolateError::ComputationError) — never a
/// fabricated success — when CUDA is unavailable or no device is present.
fn build_context() -> InterpolateResult<std::sync::Arc<oxicuda_driver::Context>> {
    oxicuda_driver::init()
        .map_err(|e| InterpolateError::ComputationError(format!("CUDA unavailable: {e}")))?;
    let count = oxicuda_driver::device::Device::count()
        .map_err(|e| InterpolateError::ComputationError(format!("device count: {e}")))?;
    if count <= 0 {
        return Err(InterpolateError::ComputationError(
            "no NVIDIA CUDA device available".into(),
        ));
    }
    let dev = oxicuda_driver::device::Device::get(0).map_err(cuda_err)?;
    Ok(std::sync::Arc::new(
        oxicuda_driver::Context::new(&dev).map_err(cuda_err)?,
    ))
}

/// Solves the dense symmetric-positive-definite RBF weight system `A w = f` on
/// the GPU and returns the weight vector `w`.
///
/// `A` is the `n`×`n` kernel matrix, built on the CPU by the caller and passed
/// in **row-major** order (length `n * n`). Because `A` is **symmetric**, its
/// column-major reinterpretation is the *same* matrix, so it is uploaded as-is
/// and treated as [`Layout::ColMajor`] by the solver. `f` is the right-hand
/// side of length `n`. The returned `w` has length `n`.
///
/// The factorisation uses `oxicuda-solver`'s dense Cholesky
/// (`A = L Lᵀ`, lower triangle) followed by a Cholesky triangular solve.
pub fn cuda_rbf_solve(a: &[f64], n: usize, f: &[f64]) -> InterpolateResult<Vec<f64>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if a.len() != n * n {
        return Err(InterpolateError::ComputationError(format!(
            "cuda_rbf_solve: matrix length {} does not match n*n = {}",
            a.len(),
            n * n
        )));
    }
    if f.len() != n {
        return Err(InterpolateError::ComputationError(format!(
            "cuda_rbf_solve: rhs length {} does not match n = {}",
            f.len(),
            n
        )));
    }

    let ctx = build_context()?;
    let mut handle = oxicuda_solver::SolverHandle::new(&ctx).map_err(solver_err)?;

    let mut d_a = oxicuda_memory::DeviceBuffer::from_host(a).map_err(cuda_err)?;
    let mut d_f = oxicuda_memory::DeviceBuffer::from_host(f).map_err(cuda_err)?;

    // Factor A = L Lᵀ in place (lower triangle holds L); borrow &mut handle + &mut d_a.
    oxicuda_solver::dense::cholesky::<f64>(
        &mut handle,
        FillMode::Lower,
        &mut d_a,
        n as u32,
        n as u32,
    )
    .map_err(solver_err)?;

    // Solve A X = B (nrhs = 1); shared &handle + shared &d_a, B overwritten with solution.
    oxicuda_solver::dense::cholesky_solve::<f64>(
        &handle,
        FillMode::Lower,
        &d_a,
        &mut d_f,
        n as u32,
        1,
    )
    .map_err(solver_err)?;

    // The Cholesky factor/solve run on the solver handle's
    // `CU_STREAM_NON_BLOCKING` stream; `copy_to_host` copies on the legacy
    // default stream, which does not implicitly wait on a non-blocking stream.
    // Synchronise so the solution is fully written before the device-to-host
    // copy reads it (otherwise the copy races the solve and can read zeros).
    handle.stream().synchronize().map_err(cuda_err)?;

    let mut w = vec![0.0f64; n];
    d_f.copy_to_host(&mut w).map_err(cuda_err)?;
    Ok(w)
}

/// Computes the RBF evaluation product `y = Phi_query @ weights` on the GPU.
///
/// `phi_query` is the `(n_query × n_centers)` evaluation matrix in **row-major**
/// order (length `n_query * n_centers`), `weights` has length `n_centers`, and
/// the returned `y` has length `n_query`.
///
/// The multiply is routed through `oxicuda-blas`'s dense GEMM. `Phi_query` is
/// described as a [`Layout::RowMajor`] `MatrixDesc` (`rows = n_query`,
/// `cols = n_centers`), `weights` as a column-vector [`Layout::RowMajor`]
/// `MatrixDesc` (`n_centers`×1), and the output as a [`Layout::RowMajor`]
/// `MatrixDescMut` (`n_query`×1) — a single-column matrix is byte-identical
/// under either layout tag, and `oxicuda-blas`'s GEMM dispatcher requires
/// RowMajor on every operand. Under [`Transpose::NoTrans`] this yields
/// `M = n_query`, `K = n_centers`, `N = 1`.
pub fn cuda_eval_gemm(
    phi_query: &[f64],
    n_query: usize,
    n_centers: usize,
    weights: &[f64],
) -> InterpolateResult<Vec<f64>> {
    if n_query == 0 {
        return Ok(Vec::new());
    }
    // No centers => the RBF sum is empty, so every query evaluates to 0.
    if n_centers == 0 {
        return Ok(vec![0.0; n_query]);
    }
    if phi_query.len() != n_query * n_centers {
        return Err(InterpolateError::ComputationError(format!(
            "cuda_eval_gemm: phi_query length {} does not match n_query*n_centers = {}",
            phi_query.len(),
            n_query * n_centers
        )));
    }
    if weights.len() != n_centers {
        return Err(InterpolateError::ComputationError(format!(
            "cuda_eval_gemm: weights length {} does not match n_centers = {}",
            weights.len(),
            n_centers
        )));
    }

    let ctx = build_context()?;
    let handle = oxicuda_blas::BlasHandle::new(&ctx).map_err(blas_err)?;

    let d_phi = oxicuda_memory::DeviceBuffer::from_host(phi_query).map_err(cuda_err)?;
    let d_w = oxicuda_memory::DeviceBuffer::from_host(weights).map_err(cuda_err)?;
    let mut d_c = oxicuda_memory::DeviceBuffer::<f64>::alloc(n_query).map_err(cuda_err)?;

    let a_desc =
        MatrixDesc::from_buffer(&d_phi, n_query as u32, n_centers as u32, Layout::RowMajor)
            .map_err(blas_err)?;
    // A column vector (N x 1) is byte-identical whether tagged RowMajor or
    // ColMajor — there is only one way to pack N elements contiguously — but
    // oxicuda-blas's GEMM dispatcher now hard-rejects any non-RowMajor operand
    // (its generated kernels are tight row-major only and ignore `ld`/`layout`
    // otherwise), so both must be tagged RowMajor to satisfy that check.
    let b_desc =
        MatrixDesc::from_buffer(&d_w, n_centers as u32, 1, Layout::RowMajor).map_err(blas_err)?;
    let mut c_desc = MatrixDescMut::from_buffer(&mut d_c, n_query as u32, 1, Layout::RowMajor)
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

    // The GEMM runs on the BLAS handle's `CU_STREAM_NON_BLOCKING` stream, but
    // `copy_to_host` issues its `cuMemcpyDtoH_v2` on the legacy default stream,
    // which a non-blocking stream does NOT implicitly synchronise with. Without
    // this wait the copy races the kernel and reads the (uninitialised / zero)
    // output before the GEMM finishes — the longer the kernel runs (large
    // n_centers = K), the more often it reads all-zeros. Synchronise first.
    handle.stream().synchronize().map_err(cuda_err)?;

    let mut y = vec![0.0f64; n_query];
    d_c.copy_to_host(&mut y).map_err(cuda_err)?;
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny in-test dense solver: Gaussian elimination with partial pivoting.
    /// `a` is `n`×`n` row-major, `b` length `n`; returns the solution length `n`.
    fn solve_dense(a: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
        let mut m = a.to_vec();
        let mut rhs = b.to_vec();
        for col in 0..n {
            // Partial pivot: find the row with the largest magnitude in this column.
            let mut pivot = col;
            let mut best = m[col * n + col].abs();
            for row in (col + 1)..n {
                let v = m[row * n + col].abs();
                if v > best {
                    best = v;
                    pivot = row;
                }
            }
            assert!(best > 0.0, "solve_dense: singular matrix");
            if pivot != col {
                for k in 0..n {
                    m.swap(col * n + k, pivot * n + k);
                }
                rhs.swap(col, pivot);
            }
            let diag = m[col * n + col];
            for row in (col + 1)..n {
                let factor = m[row * n + col] / diag;
                if factor != 0.0 {
                    for k in col..n {
                        m[row * n + k] -= factor * m[col * n + k];
                    }
                    rhs[row] -= factor * rhs[col];
                }
            }
        }
        // Back-substitution.
        let mut x = vec![0.0f64; n];
        for col in (0..n).rev() {
            let mut acc = rhs[col];
            for k in (col + 1)..n {
                acc -= m[col * n + k] * x[k];
            }
            x[col] = acc / m[col * n + col];
        }
        x
    }

    #[test]
    fn cuda_rbf_solve_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }
        let n = 3;
        // Symmetric-positive-definite tridiagonal system, row-major.
        let a = [4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let f = [1.0, 2.0, 3.0];
        let w = cuda_rbf_solve(&a, n, &f).expect("cuda_rbf_solve failed");
        let expected = solve_dense(&a, n, &f);
        let max_diff = w
            .iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-9, "max abs diff {max_diff} exceeds 1e-9");
    }

    #[test]
    fn solve_empty_is_ok() {
        let v = cuda_rbf_solve(&[], 0, &[]).expect("empty solve should be Ok");
        assert!(v.is_empty());
    }

    #[test]
    fn solve_shape_mismatch_errors() {
        // n = 3 needs a 9-element matrix; a length-3 slice is a shape mismatch.
        let res = cuda_rbf_solve(&[1.0, 2.0, 3.0], 3, &[1.0, 2.0, 3.0]);
        assert!(res.is_err());
    }

    #[test]
    fn cuda_eval_gemm_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }
        // Phi (3x2) row-major; weights [1, 1] => row sums [3, 7, 11].
        let phi = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = [1.0, 1.0];
        let y = cuda_eval_gemm(&phi, 3, 2, &weights).expect("cuda_eval_gemm failed");
        let expected = [3.0, 7.0, 11.0];
        for (g, e) in y.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-9, "got {g}, expected {e}");
        }
    }

    #[test]
    fn eval_gemm_empty_is_ok() {
        let y = cuda_eval_gemm(&[], 0, 2, &[1.0, 2.0]).expect("empty eval should be Ok");
        assert!(y.is_empty());
    }

    #[test]
    fn eval_gemm_shape_mismatch_errors() {
        // n_query = 3, n_centers = 2 needs 6 entries; a length-3 slice mismatches.
        let res = cuda_eval_gemm(&[1.0, 2.0, 3.0], 3, 2, &[1.0, 1.0]);
        assert!(res.is_err());
    }
}
