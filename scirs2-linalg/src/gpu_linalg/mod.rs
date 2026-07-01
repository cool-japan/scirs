//! GPU-accelerated linear algebra operations.
//!
//! This module provides GPU-accelerated versions of key linear algebra operations
//! using scirs2-core's portable wgpu/WebGPU backend (f32, with CPU fallback).
//!
//! # Which GPU module is this? (PORTABILITY path)
//!
//! This is the **portability** path. It builds on scirs2-core's wgpu/WebGPU
//! portability layer — `scirs2_core::gpu::{GpuBackend, GpuContext}` — a shared,
//! cross-platform (f32, CPU-fallback) foundation; that coupling to core is
//! intentional. The `GpuContext` used here is **core's**, NOT the self-contained
//! `crate::gpu::GpuContext` trait. Gated on the `gpu` feature.
//!
//! For the NVIDIA-only, f64, real-CUDA **performance** path (via `oxicuda-*`, no
//! fallback) see `crate::gpu_cuda` (feature `cuda`); for the self-contained local
//! GPU abstraction with its own trait, see `crate::gpu`.
//!
//! # Architecture
//!
//! The module is organized into three sub-modules:
//!
//! - **`types`**: Result types for decompositions (`LuFactors`, `QrFactors`, etc.)
//! - **`matmul`**: GPU-accelerated matrix multiplication (single and batched)
//! - **`decompositions`**: GPU-accelerated LU, QR, Cholesky, and SVD
//!
//! All operations automatically fall back to CPU when GPU is unavailable or
//! when the matrix is too small to benefit from GPU transfer overhead.
//!
//! # Usage
//!
//! The primary entry point is [`GpuLinalgSolver`], which manages a GPU context
//! and provides a unified API for all operations:
//!
//! ```rust,ignore
//! use scirs2_linalg::gpu_linalg::GpuLinalgSolver;
//! use scirs2_core::ndarray::array;
//!
//! let solver = GpuLinalgSolver::new();
//!
//! let a = array![[4.0_f64, 2.0], [2.0, 5.0]];
//! let chol = solver.cholesky(&a.view()).expect("Cholesky failed");
//! // chol.l is the lower-triangular factor
//!
//! let b = array![[1.0_f64, 2.0], [3.0, 4.0]];
//! let lu = solver.lu(&b.view()).expect("LU failed");
//! // lu.p, lu.l, lu.u
//! ```
//!
//! You can also use the free-standing functions directly if you manage
//! the `GpuContext` yourself or want to opt out of GPU:
//!
//! ```rust,ignore
//! use scirs2_linalg::gpu_linalg::decompositions::gpu_svd;
//!
//! let factors = gpu_svd(None, &matrix.view())?; // CPU-only
//! ```

pub mod decompositions;
pub mod matmul;
pub mod types;

// Re-export types for convenience
pub use decompositions::{gpu_cholesky, gpu_lu, gpu_qr, gpu_svd};
pub use matmul::{gpu_batched_matmul, gpu_matmul};
pub use types::{
    BatchMatmulResult, CholeskyFactors, ExecutionBackend, LuFactors, QrFactors, SvdFactors,
};

use scirs2_core::gpu::{GpuBackend, GpuContext};
use scirs2_core::ndarray::{Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, One};
use std::iter::Sum;

use crate::error::LinalgResult;

/// Unified GPU-accelerated linear algebra solver.
///
/// `GpuLinalgSolver` manages a GPU context and provides methods for performing
/// GPU-accelerated linear algebra operations. When a GPU is available, operations
/// are dispatched to the GPU for large matrices; otherwise, they seamlessly fall
/// back to efficient CPU implementations.
///
/// # Construction
///
/// ```rust,ignore
/// // Auto-detect the best available GPU backend
/// let solver = GpuLinalgSolver::new();
///
/// // Force a specific backend
/// let cpu_solver = GpuLinalgSolver::with_backend(GpuBackend::Cpu);
///
/// // Use CPU-only mode explicitly
/// let cpu_solver = GpuLinalgSolver::cpu_only();
/// ```
///
/// # Thread Safety
///
/// `GpuLinalgSolver` is `Send` but not `Sync` due to the underlying GPU context.
/// If you need concurrent access, wrap it in a `Mutex` or create one per thread.
pub struct GpuLinalgSolver {
    /// The GPU context, if available.
    context: Option<GpuContext>,
    /// The backend being used.
    backend: GpuBackend,
}

impl GpuLinalgSolver {
    /// Create a new `GpuLinalgSolver` with automatic GPU detection.
    ///
    /// Attempts to find and use the best available GPU backend. If no GPU is
    /// available, falls back to CPU transparently.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let solver = GpuLinalgSolver::new();
    /// println!("Using backend: {}", solver.backend());
    /// ```
    pub fn new() -> Self {
        let preferred = GpuBackend::preferred();
        match GpuContext::new(preferred) {
            Ok(ctx) => Self {
                backend: preferred,
                context: Some(ctx),
            },
            Err(_) => {
                // Try CPU fallback
                match GpuContext::new(GpuBackend::Cpu) {
                    Ok(ctx) => Self {
                        backend: GpuBackend::Cpu,
                        context: Some(ctx),
                    },
                    Err(_) => Self {
                        backend: GpuBackend::Cpu,
                        context: None,
                    },
                }
            }
        }
    }

    /// Create a `GpuLinalgSolver` with a specific GPU backend.
    ///
    /// # Arguments
    ///
    /// * `backend` - The GPU backend to use
    ///
    /// # Returns
    ///
    /// A solver using the specified backend, or CPU fallback if the backend
    /// is unavailable.
    pub fn with_backend(backend: GpuBackend) -> Self {
        match GpuContext::new(backend) {
            Ok(ctx) => Self {
                backend,
                context: Some(ctx),
            },
            Err(_) => {
                // Fall back to CPU
                match GpuContext::new(GpuBackend::Cpu) {
                    Ok(ctx) => Self {
                        backend: GpuBackend::Cpu,
                        context: Some(ctx),
                    },
                    Err(_) => Self {
                        backend: GpuBackend::Cpu,
                        context: None,
                    },
                }
            }
        }
    }

    /// Create a CPU-only solver (no GPU acceleration).
    ///
    /// This is useful for testing, benchmarking against CPU, or in
    /// environments where GPU access is undesirable.
    pub fn cpu_only() -> Self {
        match GpuContext::new(GpuBackend::Cpu) {
            Ok(ctx) => Self {
                backend: GpuBackend::Cpu,
                context: Some(ctx),
            },
            Err(_) => Self {
                backend: GpuBackend::Cpu,
                context: None,
            },
        }
    }

    /// Get the backend being used by this solver.
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    /// Check whether GPU acceleration is available.
    ///
    /// Returns `true` if the solver has a GPU context and the backend is not CPU.
    pub fn has_gpu(&self) -> bool {
        self.context.is_some() && self.backend != GpuBackend::Cpu
    }

    /// Get a reference to the GPU context, if available.
    pub fn context(&self) -> Option<&GpuContext> {
        self.context.as_ref()
    }

    // =========================================================================
    // Matrix Multiplication
    // =========================================================================

    /// GPU-accelerated matrix multiplication: C = A * B
    ///
    /// # Arguments
    ///
    /// * `a` - Left matrix (m x k)
    /// * `b` - Right matrix (k x n)
    ///
    /// # Returns
    ///
    /// Result matrix C (m x n)
    ///
    /// # Errors
    ///
    /// Returns error if matrices have incompatible dimensions.
    pub fn matmul<F>(&self, a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    {
        gpu_matmul(self.context.as_ref(), a, b)
    }

    /// GPU-accelerated batched matrix multiplication.
    ///
    /// Multiplies corresponding pairs: C_i = A_i * B_i for each i.
    ///
    /// # Arguments
    ///
    /// * `a_batch` - Batch of left matrices
    /// * `b_batch` - Batch of right matrices (same length as `a_batch`)
    ///
    /// # Returns
    ///
    /// `BatchMatmulResult` with one result per pair.
    ///
    /// # Errors
    ///
    /// Returns error if batch sizes differ or any pair has incompatible shapes.
    pub fn batched_matmul<F>(
        &self,
        a_batch: &[Array2<F>],
        b_batch: &[Array2<F>],
    ) -> LinalgResult<BatchMatmulResult<F>>
    where
        F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    {
        gpu_batched_matmul(self.context.as_ref(), a_batch, b_batch)
    }

    // =========================================================================
    // Decompositions
    // =========================================================================

    /// GPU-accelerated LU decomposition: PA = LU
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix
    ///
    /// # Returns
    ///
    /// `LuFactors` containing P, L, and U matrices.
    pub fn lu<F>(&self, a: &ArrayView2<F>) -> LinalgResult<LuFactors<F>>
    where
        F: Float + NumAssign + One + Sum + Send + Sync + ScalarOperand + 'static,
    {
        gpu_lu(self.context.as_ref(), a)
    }

    /// GPU-accelerated QR decomposition: A = QR
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix
    ///
    /// # Returns
    ///
    /// `QrFactors` containing Q and R matrices.
    pub fn qr<F>(&self, a: &ArrayView2<F>) -> LinalgResult<QrFactors<F>>
    where
        F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    {
        gpu_qr(self.context.as_ref(), a)
    }

    /// GPU-accelerated Cholesky decomposition: A = LL^T
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric positive-definite matrix
    ///
    /// # Returns
    ///
    /// `CholeskyFactors` containing the lower-triangular factor L.
    pub fn cholesky<F>(&self, a: &ArrayView2<F>) -> LinalgResult<CholeskyFactors<F>>
    where
        F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    {
        gpu_cholesky(self.context.as_ref(), a)
    }

    /// GPU-accelerated thin SVD: A = U * diag(S) * V^T
    ///
    /// Computes the economy-size singular value decomposition.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix
    ///
    /// # Returns
    ///
    /// `SvdFactors` containing U, S, and V^T.
    pub fn svd<F>(&self, a: &ArrayView2<F>) -> LinalgResult<SvdFactors<F>>
    where
        F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    {
        gpu_svd(self.context.as_ref(), a)
    }

    // =========================================================================
    // Convenience: Solve linear systems via decomposition
    // =========================================================================

    /// Solve a linear system Ax = b using LU decomposition.
    ///
    /// # Arguments
    ///
    /// * `a` - Coefficient matrix (n x n)
    /// * `b` - Right-hand side vector (as n x 1 matrix)
    ///
    /// # Returns
    ///
    /// Solution vector x (as n x 1 matrix).
    ///
    /// # Errors
    ///
    /// Returns error if A is singular or dimensions are incompatible.
    pub fn solve<F>(&self, a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + One + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let n = a.nrows();
        if a.nrows() != a.ncols() {
            return Err(crate::error::LinalgError::ShapeError(
                "Solve requires a square coefficient matrix".to_string(),
            ));
        }
        if b.nrows() != n {
            return Err(crate::error::LinalgError::DimensionError(format!(
                "Right-hand side has {} rows but coefficient matrix has {} rows",
                b.nrows(),
                n
            )));
        }

        let factors = self.lu(a)?;
        let n_rhs = b.ncols();

        // Convention: P * A = L * U, so A = P^T * L * U
        // To solve A * x = b: P^T * L * U * x = b => L * U * x = P * b
        // 1. Compute P * b
        let pb = factors.p.dot(b);

        // 2. Forward substitution: Ly = P * b
        let mut y = Array2::<F>::zeros((n, n_rhs));
        for col in 0..n_rhs {
            for i in 0..n {
                let mut sum = pb[[i, col]];
                for j in 0..i {
                    sum -= factors.l[[i, j]] * y[[j, col]];
                }
                // L has unit diagonal
                y[[i, col]] = sum;
            }
        }

        // 3. Back substitution: Ux = y
        let min_dim = n.min(factors.u.ncols());
        let mut x = Array2::<F>::zeros((n, n_rhs));
        for col in 0..n_rhs {
            for i in (0..min_dim).rev() {
                let mut sum = y[[i, col]];
                for j in (i + 1)..factors.u.ncols().min(n) {
                    sum -= factors.u[[i, j]] * x[[j, col]];
                }
                let diag = factors.u[[i, i]];
                if diag.abs() < F::epsilon() {
                    return Err(crate::error::LinalgError::SingularMatrixError(
                        "Matrix is singular: zero diagonal in U factor".to_string(),
                    ));
                }
                x[[i, col]] = sum / diag;
            }
        }

        Ok(x)
    }
}

impl std::fmt::Debug for GpuLinalgSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuLinalgSolver")
            .field("backend", &self.backend)
            .field("has_context", &self.context.is_some())
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_solver_creation() {
        let solver = GpuLinalgSolver::new();
        // Should always succeed, possibly with CPU backend
        let _backend = solver.backend();
    }

    #[test]
    fn test_solver_cpu_only() {
        let solver = GpuLinalgSolver::cpu_only();
        assert_eq!(solver.backend(), GpuBackend::Cpu);
    }

    #[test]
    fn test_solver_with_backend() {
        let solver = GpuLinalgSolver::with_backend(GpuBackend::Cpu);
        assert_eq!(solver.backend(), GpuBackend::Cpu);
    }

    #[test]
    fn test_solver_debug() {
        let solver = GpuLinalgSolver::cpu_only();
        let debug_str = format!("{:?}", solver);
        assert!(debug_str.contains("GpuLinalgSolver"));
        assert!(debug_str.contains("Cpu"));
    }

    #[test]
    fn test_solver_matmul() {
        let solver = GpuLinalgSolver::cpu_only();
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
        let c = solver.matmul(&a.view(), &b.view()).expect("matmul failed");
        assert_relative_eq!(c[[0, 0]], 19.0, epsilon = 1e-10);
        assert_relative_eq!(c[[1, 1]], 50.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solver_batched_matmul() {
        let solver = GpuLinalgSolver::cpu_only();
        let a1 = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b1 = array![[3.0_f64, 4.0], [5.0, 6.0]];
        let batch = solver
            .batched_matmul(&[a1], &[b1])
            .expect("batch matmul failed");
        assert_eq!(batch.results.len(), 1);
        assert_relative_eq!(batch.results[0][[0, 0]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solver_lu() {
        let solver = GpuLinalgSolver::cpu_only();
        let a = array![[2.0_f64, 1.0], [4.0, 3.0]];
        let factors = solver.lu(&a.view()).expect("LU failed");
        // Convention: P * A = L * U
        let pa = factors.p.dot(&a);
        let lu_product = factors.l.dot(&factors.u);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(pa[[i, j]], lu_product[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_solver_qr() {
        let solver = GpuLinalgSolver::cpu_only();
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let factors = solver.qr(&a.view()).expect("QR failed");
        let qr_product = factors.q.dot(&factors.r);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(qr_product[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_solver_cholesky() {
        let solver = GpuLinalgSolver::cpu_only();
        let a = array![[4.0_f64, 2.0], [2.0, 5.0]];
        let factors = solver.cholesky(&a.view()).expect("Cholesky failed");
        let llt = factors.l.dot(&factors.l.t());
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(llt[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_solver_svd() {
        let solver = GpuLinalgSolver::cpu_only();
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let factors = solver.svd(&a.view()).expect("SVD failed");
        let s_diag = Array2::from_diag(&factors.s);
        let usv = factors.u.dot(&s_diag).dot(&factors.vt);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(usv[[i, j]], a[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_solver_solve() {
        let solver = GpuLinalgSolver::cpu_only();
        // Solve [[2,1],[4,3]] * x = [[5],[13]]
        let a = array![[2.0_f64, 1.0], [4.0, 3.0]];
        let b = array![[5.0_f64], [13.0]];
        let x = solver.solve(&a.view(), &b.view()).expect("solve failed");
        // x should be [[1], [3]]
        assert_relative_eq!(x[[0, 0]], 1.0, epsilon = 1e-8);
        assert_relative_eq!(x[[1, 0]], 3.0, epsilon = 1e-8);
    }

    #[test]
    fn test_solver_solve_non_square() {
        let solver = GpuLinalgSolver::cpu_only();
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[1.0_f64], [2.0]];
        let result = solver.solve(&a.view(), &b.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_solver_solve_dimension_mismatch() {
        let solver = GpuLinalgSolver::cpu_only();
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[1.0_f64], [2.0], [3.0]];
        let result = solver.solve(&a.view(), &b.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_solver_solve_identity() {
        let solver = GpuLinalgSolver::cpu_only();
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![[7.0_f64], [11.0]];
        let x = solver
            .solve(&a.view(), &b.view())
            .expect("identity solve failed");
        assert_relative_eq!(x[[0, 0]], 7.0, epsilon = 1e-10);
        assert_relative_eq!(x[[1, 0]], 11.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solver_solve_3x3() {
        let solver = GpuLinalgSolver::cpu_only();
        let a = array![[1.0_f64, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];
        let b = array![[1.0_f64], [2.0], [3.0]];
        let x = solver
            .solve(&a.view(), &b.view())
            .expect("3x3 solve failed");

        // Verify A * x ≈ b
        let ax = a.dot(&x);
        for i in 0..3 {
            assert_relative_eq!(ax[[i, 0]], b[[i, 0]], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_solver_matmul_f32() {
        let solver = GpuLinalgSolver::cpu_only();
        let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
        let c = solver
            .matmul(&a.view(), &b.view())
            .expect("f32 matmul failed");
        assert!((c[[0, 0]] - 19.0).abs() < 1e-4);
    }

    #[test]
    fn test_execution_backend_display() {
        assert_eq!(ExecutionBackend::Gpu.to_string(), "GPU");
        assert_eq!(ExecutionBackend::CpuFallback.to_string(), "CPU (fallback)");
    }

    #[test]
    fn test_has_gpu_cpu_only() {
        let solver = GpuLinalgSolver::cpu_only();
        assert!(!solver.has_gpu());
    }

    #[test]
    fn test_solver_context_ref() {
        let solver = GpuLinalgSolver::cpu_only();
        // CPU solver should have a context
        assert!(solver.context().is_some() || solver.context().is_none());
    }

    #[test]
    fn test_solver_multiple_operations() {
        // Test using the same solver for multiple operations
        let solver = GpuLinalgSolver::cpu_only();

        let a = array![[4.0_f64, 2.0], [2.0, 5.0]];

        // Cholesky
        let chol = solver.cholesky(&a.view()).expect("cholesky failed");
        let llt = chol.l.dot(&chol.l.t());
        assert_relative_eq!(llt[[0, 0]], a[[0, 0]], epsilon = 1e-10);

        // LU: convention P * A = L * U
        let lu = solver.lu(&a.view()).expect("lu failed");
        let pa = lu.p.dot(&a);
        let lu_product = lu.l.dot(&lu.u);
        assert_relative_eq!(pa[[0, 0]], lu_product[[0, 0]], epsilon = 1e-10);

        // SVD
        let svd = solver.svd(&a.view()).expect("svd failed");
        let s_diag = Array2::from_diag(&svd.s);
        let usv = svd.u.dot(&s_diag).dot(&svd.vt);
        assert_relative_eq!(usv[[0, 0]], a[[0, 0]], epsilon = 1e-8);
    }
}
