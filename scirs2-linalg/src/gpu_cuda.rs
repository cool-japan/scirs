#![cfg(feature = "cuda")]
//! Optional, off-by-default, NVIDIA-only CUDA acceleration for dense linear
//! algebra, calling the pure-Rust `oxicuda-*` library crates directly.
//!
//! This module is ADDITIVE and entirely feature-gated behind the `cuda` feature
//! (off by default). It does not replace or modify the existing `gpu`,
//! `gpu_linalg`, `gpu_accel`, or `gpu_gemm` subsystems. Every public function is
//! f64-native, which is the advantage of the oxicuda CUDA path over the f32
//! wgpu path.
//!
//! Each entry point degrades safely when no NVIDIA device is present:
//! [`cuda_is_available`] never panics, and the compute functions return a
//! [`crate::error::LinalgError::ComputationError`] rather than aborting.
//!
//! See `crate::gpu` for the self-contained local GPU abstraction (its own
//! `GpuContext` trait) and `crate::gpu_linalg` for the portable f32 wgpu path built
//! on `scirs2_core::gpu`.

use crate::error::{LinalgError, LinalgResult};
use oxicuda_blas::types::{FillMode, Layout, MatrixDesc, MatrixDescMut, Transpose};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Returns `true` iff the CUDA driver initializes and at least one NVIDIA device
/// is visible. Never panics.
pub fn cuda_is_available() -> bool {
    oxicuda_driver::init().is_ok()
        && oxicuda_driver::device::Device::count()
            .map(|c| c > 0)
            .unwrap_or(false)
}

/// Build a CUDA context bound to device 0, wrapped in an `Arc` as required by the
/// oxicuda BLAS/solver handle constructors.
fn build_context() -> LinalgResult<std::sync::Arc<oxicuda_driver::Context>> {
    oxicuda_driver::init()
        .map_err(|e| LinalgError::ComputationError(format!("CUDA unavailable: {e}")))?;
    let count = oxicuda_driver::device::Device::count()
        .map_err(|e| LinalgError::ComputationError(format!("device count: {e}")))?;
    if count <= 0 {
        return Err(LinalgError::ComputationError(
            "no NVIDIA CUDA device available".into(),
        ));
    }
    let dev = oxicuda_driver::device::Device::get(0).map_err(cuda_err)?;
    Ok(std::sync::Arc::new(
        oxicuda_driver::Context::new(&dev).map_err(cuda_err)?,
    ))
}

fn solver_err(e: oxicuda_solver::error::SolverError) -> LinalgError {
    LinalgError::ComputationError(format!("oxicuda-solver: {e}"))
}

fn blas_err(e: oxicuda_blas::error::BlasError) -> LinalgError {
    LinalgError::ComputationError(format!("oxicuda-blas: {e}"))
}

fn cuda_err(e: oxicuda_driver::CudaError) -> LinalgError {
    LinalgError::ComputationError(format!("oxicuda CUDA driver: {e}"))
}

/// Compute `C = A · B` on the GPU in f64.
///
/// Layout correctness (cannot be runtime-checked on a non-NVIDIA host): ndarray
/// is row-major. We materialize contiguous row-major slices for both operands
/// and describe all three matrices as `Layout::RowMajor`, exactly mirroring the
/// oxicuda-blas row-major GEMM setup in
/// `oxicuda/crates/oxicuda-blas/benches/gemm_f64_4096.rs` (N×N, all
/// `Layout::RowMajor`, `Transpose::NoTrans`). With `op(A)` = M×K and `op(B)` =
/// K×N, the result `C` is M×N per `gemm`'s dimension contract, so we pass
/// A as (m × k), B as (k × n), C as (m × n).
pub fn cuda_gemm(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    let (m, k) = (a.nrows(), a.ncols());
    let (k_b, n) = (b.nrows(), b.ncols());
    if k != k_b {
        return Err(LinalgError::ShapeError(format!(
            "cuda_gemm: inner dimensions disagree: A is {m}x{k}, B is {k_b}x{n}"
        )));
    }
    if m == 0 || k == 0 || n == 0 {
        return Ok(Array2::zeros((m, n)));
    }

    // Contiguous row-major copies so `.as_slice()` is `Some` and the device
    // upload matches the `Layout::RowMajor` descriptors below.
    let a_std = a.as_standard_layout();
    let b_std = b.as_standard_layout();
    let a_slice = a_std
        .as_slice()
        .ok_or_else(|| LinalgError::ComputationError("cuda_gemm: A not contiguous".into()))?;
    let b_slice = b_std
        .as_slice()
        .ok_or_else(|| LinalgError::ComputationError("cuda_gemm: B not contiguous".into()))?;

    let ctx = build_context()?;
    let handle = oxicuda_blas::BlasHandle::new(&ctx).map_err(blas_err)?;

    let d_a = oxicuda_memory::DeviceBuffer::from_host(a_slice).map_err(cuda_err)?;
    let d_b = oxicuda_memory::DeviceBuffer::from_host(b_slice).map_err(cuda_err)?;
    let mut d_c = oxicuda_memory::DeviceBuffer::<f64>::alloc(m * n).map_err(cuda_err)?;

    let a_desc =
        MatrixDesc::from_buffer(&d_a, m as u32, k as u32, Layout::RowMajor).map_err(blas_err)?;
    let b_desc =
        MatrixDesc::from_buffer(&d_b, k as u32, n as u32, Layout::RowMajor).map_err(blas_err)?;
    let mut c_desc = MatrixDescMut::from_buffer(&mut d_c, m as u32, n as u32, Layout::RowMajor)
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

    let mut c_host = vec![0.0f64; m * n];
    d_c.copy_to_host(&mut c_host).map_err(cuda_err)?;
    Array2::from_shape_vec((m, n), c_host)
        .map_err(|e| LinalgError::ComputationError(format!("cuda_gemm: reshape failed: {e}")))
}

/// Solve `A · x = b` for a symmetric positive-definite `A` (n×n) on the GPU in
/// f64, via Cholesky factorization plus triangular solve.
///
/// Mirrors the proven layout of the Phase-1 interpolate reference
/// (`scirs2-interpolate/src/gpu_cuda.rs::cuda_rbf_solve`): upload the row-major
/// `A` and `b`, factor `A = L Lᵀ` in place (`FillMode::Lower`), then
/// `cholesky_solve` with `nrhs = 1`, overwriting `b` with the solution. For a
/// symmetric matrix the row-major and column-major interpretations coincide, so
/// the oxicuda solver's column-major expectation is satisfied automatically.
pub fn cuda_solve_spd(a: &ArrayView2<f64>, b: &ArrayView1<f64>) -> LinalgResult<Array1<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(format!(
            "cuda_solve_spd: A must be square, got {n}x{}",
            a.ncols()
        )));
    }
    if b.len() != n {
        return Err(LinalgError::DimensionError(format!(
            "cuda_solve_spd: b length {} does not match A dimension {n}",
            b.len()
        )));
    }
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let a_std = a.as_standard_layout();
    let a_slice = a_std
        .as_slice()
        .ok_or_else(|| LinalgError::ComputationError("cuda_solve_spd: A not contiguous".into()))?;
    let b_vec: Vec<f64> = b.iter().copied().collect();

    let ctx = build_context()?;
    let mut handle = oxicuda_solver::SolverHandle::new(&ctx).map_err(solver_err)?;

    let mut d_a = oxicuda_memory::DeviceBuffer::from_host(a_slice).map_err(cuda_err)?;
    let mut d_b = oxicuda_memory::DeviceBuffer::from_host(&b_vec).map_err(cuda_err)?;

    // Factor A = L Lᵀ in place (lower triangle holds L); borrow &mut handle + &mut d_a.
    oxicuda_solver::dense::cholesky::<f64>(
        &mut handle,
        FillMode::Lower,
        &mut d_a,
        n as u32,
        n as u32,
    )
    .map_err(solver_err)?;

    // Solve A X = b (nrhs = 1); shared &handle + shared &d_a, d_b overwritten with x.
    oxicuda_solver::dense::cholesky_solve::<f64>(
        &handle,
        FillMode::Lower,
        &d_a,
        &mut d_b,
        n as u32,
        1,
    )
    .map_err(solver_err)?;

    let mut x = vec![0.0f64; n];
    d_b.copy_to_host(&mut x).map_err(cuda_err)?;
    Ok(Array1::from_vec(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(rows: usize, cols: usize, data: Vec<f64>) -> Array2<f64> {
        Array2::from_shape_vec((rows, cols), data).expect("valid test matrix shape")
    }

    #[test]
    fn cuda_gemm_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }
        // A (2x3) · B (3x2) = C (2x2) = [[58, 64], [139, 154]].
        let a = mat(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = mat(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = cuda_gemm(&a.view(), &b.view()).expect("cuda_gemm failed");
        let expected = mat(2, 2, vec![58.0, 64.0, 139.0, 154.0]);
        let max_diff = c
            .iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-9, "max abs diff {max_diff} exceeds 1e-9");
    }

    #[test]
    fn cuda_gemm_nonsquare_stress_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }
        // 17×5 · 5×31: clearly non-square, odd M/N dimensions — guards against
        // row/col-major layout bugs that only surface when M ≠ K ≠ N.  The
        // oxicuda GEMM kernel was rewritten to a geometry-independent grid-stride
        // loop; this is the first runtime check of that path on non-square shapes.
        let m = 17usize;
        let k = 5usize;
        let n = 31usize;
        let a_data: Vec<f64> = (0..m * k).map(|idx| (idx as f64 + 1.0) * 0.1).collect();
        let b_data: Vec<f64> = (0..k * n)
            .map(|idx| (idx as f64 * 1.3 + 0.5) * 0.07)
            .collect();
        let a = mat(m, k, a_data);
        let b = mat(k, n, b_data);
        let c_gpu = cuda_gemm(&a.view(), &b.view()).expect("cuda_gemm failed on 17x5 . 5x31");
        let c_cpu = a.dot(&b);
        assert_eq!(c_gpu.shape(), &[m, n], "output shape must be {m}x{n}");
        let max_diff = c_gpu
            .iter()
            .zip(c_cpu.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 1e-9,
            "17x5 . 5x31 max abs diff {max_diff} exceeds 1e-9"
        );
    }

    #[test]
    fn cuda_solve_spd_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }
        // Symmetric positive-definite tridiagonal system.
        let a = mat(3, 3, vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]);
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let x = cuda_solve_spd(&a.view(), &b.view()).expect("cuda_solve_spd failed");
        // Verify A·x == b.
        let ax = a.dot(&x);
        let max_diff = ax
            .iter()
            .zip(b.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-9, "max abs diff {max_diff} exceeds 1e-9");
    }

    #[test]
    fn cuda_gemm_shape_mismatch_errors() {
        // Runs without a GPU: validation rejects before any device call.
        let a = mat(1, 3, vec![1.0, 2.0, 3.0]);
        let b = mat(1, 2, vec![1.0, 2.0]);
        assert!(cuda_gemm(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn cuda_solve_spd_shape_mismatch_errors() {
        // Runs without a GPU: validation rejects before any device call.
        let a = mat(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(cuda_solve_spd(&a.view(), &b.view()).is_err());
        let nonsquare = mat(1, 3, vec![1.0, 2.0, 3.0]);
        let b2 = Array1::from_vec(vec![1.0]);
        assert!(cuda_solve_spd(&nonsquare.view(), &b2.view()).is_err());
    }

    /// End-to-end blocked Cholesky regression guard.
    ///
    /// `n = 80` and `n = 128` both exceed the oxicuda-solver block-size threshold
    /// (64), forcing the blocked factorisation path that was previously broken
    /// (wrong leading dimensions in the TRSM/SYRK device-kernel calls, producing
    /// silently corrupted results). This test proves the fixed path is correct
    /// through the full `scirs2-linalg::cuda_solve_spd` call stack.
    ///
    /// Matrix construction (CPU only, deterministic, no external RNG):
    ///   M[i][j] = (i + j + 1) / (2·n)           -- entries in (0, 1]
    ///   A = Mᵀ M + n·I                           -- strictly SPD, λ_min ≥ n
    ///   b = A · 1 (all-ones)                     -- exact solution is known
    ///
    /// Condition number of A ≈ 1 + n/3 (very well-conditioned), so the f64
    /// Cholesky backward error stays comfortably below n·1e-10 for both sizes.
    #[test]
    fn cuda_solve_spd_large_n_blocked_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }
        for &n in &[80_usize, 128_usize] {
            // Deterministic M: M[i][j] = (i + j + 1) / (2·n).
            let m_flat: Vec<f64> = (0..n * n)
                .map(|idx| {
                    let i = idx / n;
                    let j = idx % n;
                    (i + j + 1) as f64 / (2 * n) as f64
                })
                .collect();
            let m = Array2::from_shape_vec((n, n), m_flat).expect("deterministic M must be valid");
            // Mᵀ M is PSD; adding n·I makes it strictly SPD.
            let mut a = m.t().dot(&m);
            for i in 0..n {
                a[[i, i]] += n as f64;
            }
            // b = A · ones  =>  exact solution is the all-ones vector.
            let ones = Array1::from_elem(n, 1.0_f64);
            let b = a.dot(&ones);
            // GPU Cholesky solve — exercises the (now-fixed) blocked path for n > 64.
            let x = cuda_solve_spd(&a.view(), &b.view())
                .unwrap_or_else(|e| panic!("cuda_solve_spd n={n}: {e}"));
            // Residual check: A·x ≈ b within n·1e-10.
            let ax = a.dot(&x);
            let tol = (n as f64) * 1e-10;
            let max_diff = ax
                .iter()
                .zip(b.iter())
                .map(|(gpu_val, expected)| (gpu_val - expected).abs())
                .fold(0.0_f64, f64::max);
            eprintln!("n={n}: blocked-Cholesky residual max_diff={max_diff:.3e}  tol={tol:.3e}");
            assert!(
                max_diff < tol,
                "n={n}: blocked Cholesky residual {max_diff:.3e} exceeds tight tol {tol:.3e}"
            );
        }
    }
}
