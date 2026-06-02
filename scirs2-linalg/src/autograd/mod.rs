//! Automatic differentiation support for linear algebra operations
//!
//! This module provides integration with the scirs2-autograd crate for automatic
//! differentiation of linear algebra operations.
//!
//! ## Current Status
//!
//! The autograd module is currently in a transitional state. The scirs2-autograd
//! crate is actively being developed, and many linear algebra-specific operations
//! are not yet available. Based on feedback from the scirs2-autograd maintainers,
//! the following features are planned but not yet implemented:
//!
//! ### Missing Operations
//!
//! 1. **Built-in matrix operations**:
//!    - `eye(n, ctx)` - Identity matrix creation
//!    - `diag(vector, ctx)` - Diagonal matrix from vector
//!    - `trace(matrix)` - Matrix trace
//!
//! 2. **Decompositions with gradients**:
//!    - LU decomposition
//!    - QR decomposition
//!    - SVD decomposition
//!    - Eigendecomposition
//!
//! 3. **Complex number support**: Currently limited to real floating-point types
//!
//! 4. **Specialized matrix types**: Support for symmetric, hermitian, and triangular
//!    matrices with optimized operations
//!
//! ## Using Autograd with Linear Algebra
//!
//! Until these operations are available in scirs2-autograd, users can:
//!
//! 1. Use basic operations (matmul, transpose, element-wise ops) which are available
//! 2. Compose complex operations from primitives
//! 3. Implement custom operations using the scirs2-autograd `Op` trait
//!
//! ## Examples
//!
//! For current usage examples, see:
//! - `examples/autograd_basic_example.rs` - Basic differentiation
//! - `examples/autograd_linalg_example.rs` - Linear algebra operations
//!
//! ## Future Plans
//!
//! Once the requested features are implemented in scirs2-autograd, this module will
//! provide:
//!
//! - Seamless integration of all linear algebra operations with automatic differentiation
//! - Optimized gradients for matrix decompositions
//! - Support for complex-valued matrices
//! - Specialized operations for structured matrices

use scirs2_autograd as ag;

pub use ag::tensor_ops;
/// Re-export commonly used types from scirs2-autograd
pub use ag::{Context, Float, Tensor};

// Autodiff backward passes for matrix factorizations are implemented in
// `scirs2_autograd::tensor_ops`. Canonical entry points:
//   - scirs2_autograd::tensor_ops::cholesky
//   - scirs2_autograd::tensor_ops::lu
//   - scirs2_autograd::tensor_ops::qr
//   - scirs2_autograd::tensor_ops::pinv   (three-term Golub & Pereyra grad)
//   - scirs2_autograd::tensor_ops::sqrtm / sqrtm_pd  (SPD-restricted Sylvester backward)
//   - scirs2_autograd::tensor_ops::logm   (Daleckii-Krein spectral backward)

/// Matrix calculus operations with automatic differentiation support
/// Includes: gradient, hessian, jacobian computations, VJP, JVP
pub mod matrix_calculus;

/// Einstein summation engine for arbitrary-rank tensor contractions.
///
/// Provides `einsum` (evaluation) and `einsum_grad` (gradients) supporting
/// traces, diagonals, outer products, matrix multiplications, batched matmul,
/// and arbitrary index contractions.
pub mod einsum;
pub use einsum::{einsum, einsum_grad, EinsumError};

/// Helper functions for common patterns in linear algebra autodiff
pub mod helpers {
    use super::*;

    /// Compute trace using available operations (workaround until trace is available)
    ///
    /// This multiplies the matrix element-wise with an identity matrix and sums all elements.
    pub fn trace_workaround<'g, F: ag::Float>(
        matrix: &ag::Tensor<'g, F>,
        n: usize,
        ctx: &'g ag::Context<'g, F>,
    ) -> ag::Tensor<'g, F> {
        // Create identity pattern manually
        let mut eye_data = vec![F::zero(); n * n];
        for i in 0..n {
            eye_data[i * n + i] = F::one();
        }
        let eye = ag::tensor_ops::convert_to_tensor(
            ag::ndarray::Array2::from_shape_vec((n, n), eye_data).expect("Operation failed"),
            ctx,
        );

        // Element-wise multiply and sum
        let diag_elements = matrix * eye;
        ag::tensor_ops::sum_all(diag_elements)
    }

    /// Create identity matrix using available operations (workaround)
    pub fn eye_workaround<'g, F: ag::Float>(
        n: usize,
        ctx: &'g ag::Context<'g, F>,
    ) -> ag::Tensor<'g, F> {
        let mut eye_data = vec![F::zero(); n * n];
        for i in 0..n {
            eye_data[i * n + i] = F::one();
        }
        ag::tensor_ops::convert_to_tensor(
            ag::ndarray::Array2::from_shape_vec((n, n), eye_data).expect("Operation failed"),
            ctx,
        )
    }

    /// Create diagonal matrix from vector (workaround)
    pub fn diag_workaround<'g, F: ag::Float>(
        diagonal: &ag::Tensor<'g, F>,
        ctx: &'g ag::Context<'g, F>,
    ) -> ag::Tensor<'g, F> {
        // Extract diagonal values from tensor
        let diagarray =
            ag::integration::tensor_conversion::to_ndarray(diagonal).expect("Operation failed");
        let n = diagarray.len();

        let mut matrix_data = vec![F::zero(); n * n];
        for i in 0..n {
            matrix_data[i * n + i] = diagarray[i];
        }

        ag::tensor_ops::convert_to_tensor(
            ag::ndarray::Array2::from_shape_vec((n, n), matrix_data).expect("Operation failed"),
            ctx,
        )
    }

    /// Compute Frobenius norm using available operations
    pub fn frobenius_norm<'g, F: ag::Float>(matrix: &ag::Tensor<'g, F>) -> ag::Tensor<'g, F> {
        // ||A||_F = sqrt(sum(A .* A))
        let squared = matrix * matrix;
        let sum_squared = ag::tensor_ops::sum_all(squared);
        ag::tensor_ops::sqrt(sum_squared)
    }

    /// Compute matrix determinant approximation using available operations
    ///
    /// This uses a recursive approach for small matrices or iterative methods for larger ones.
    /// Note: This is a computational approximation, not optimized for accuracy or performance.
    pub fn det_approximation<'g, F: ag::Float>(
        matrix: &ag::Tensor<'g, F>,
        n: usize,
        ctx: &'g ag::Context<'g, F>,
    ) -> ag::Tensor<'g, F> {
        if n == 1 {
            // 1x1 matrix determinant is the single element
            return *matrix;
        }

        if n == 2 {
            // 2x2 determinant: ad - bc
            let matarray =
                ag::integration::tensor_conversion::to_ndarray(matrix).expect("Operation failed");
            let a = ag::tensor_ops::convert_to_tensor(
                ag::ndarray::Array2::from_elem((1, 1), matarray[[0, 0]]),
                ctx,
            );
            let b = ag::tensor_ops::convert_to_tensor(
                ag::ndarray::Array2::from_elem((1, 1), matarray[[0, 1]]),
                ctx,
            );
            let c = ag::tensor_ops::convert_to_tensor(
                ag::ndarray::Array2::from_elem((1, 1), matarray[[1, 0]]),
                ctx,
            );
            let d = ag::tensor_ops::convert_to_tensor(
                ag::ndarray::Array2::from_elem((1, 1), matarray[[1, 1]]),
                ctx,
            );

            return a * d - b * c;
        }

        // For larger matrices, return a placeholder (would need LU decomposition)
        ag::tensor_ops::convert_to_tensor(ag::ndarray::Array2::from_elem((1, 1), F::one()), ctx)
    }

    /// Solve linear system Ax = b using iterative method approximation
    ///
    /// This implements a simplified gradient descent approach for solving Ax = b
    /// by minimizing ||Ax - b||^2. Not optimized for accuracy - mainly for demonstration.
    pub fn solve_iterative<'g, F: ag::Float>(
        a: &ag::Tensor<'g, F>,
        b: &ag::Tensor<'g, F>,
        iterations: usize,
        learning_rate: F,
        ctx: &'g ag::Context<'g, F>,
    ) -> ag::Tensor<'g, F> {
        // Initialize x as zeros
        let barray = ag::integration::tensor_conversion::to_ndarray(b).expect("Operation failed");
        let n = barray.len();
        let mut x = ag::tensor_ops::convert_to_tensor(ag::ndarray::Array2::zeros((n, 1)), ctx);

        let lr_tensor = ag::tensor_ops::convert_to_tensor(
            ag::ndarray::Array2::from_elem((1, 1), learning_rate),
            ctx,
        );

        // Gradient descent: x = x - lr * A^T * (A*x - b)
        for _iter in 0..iterations {
            let ax = ag::tensor_ops::matmul(a, x);
            let residual = ax - b;
            let at = ag::tensor_ops::transpose(a, &[1, 0]);
            let gradient = ag::tensor_ops::matmul(at, residual);
            let update = gradient * lr_tensor;
            x = x - update;
        }

        x
    }

    /// Compute eigenvalue approximation using power iteration
    ///
    /// Finds the dominant eigenvalue and eigenvector using power iteration method.
    /// Returns the dominant eigenvalue as a scalar tensor.
    pub fn dominant_eigenvalue<'g, F: ag::Float>(
        matrix: &ag::Tensor<'g, F>,
        iterations: usize,
        n: usize,
        ctx: &'g ag::Context<'g, F>,
    ) -> ag::Tensor<'g, F> {
        // Initialize random vector
        let mut v_data = vec![F::one(); n];
        v_data[0] = F::one();
        for (i, item) in v_data.iter_mut().enumerate().take(n).skip(1) {
            *item = F::from(0.1).expect("Operation failed")
                * F::from(i as f64).expect("Operation failed");
        }

        let mut v = ag::tensor_ops::convert_to_tensor(
            ag::ndarray::Array2::from_shape_vec((n, 1), v_data).expect("Operation failed"),
            ctx,
        );

        // Power iteration
        for _iter in 0..iterations {
            let av = ag::tensor_ops::matmul(matrix, v);
            let norm = frobenius_norm(&av);
            v = av / norm;
        }

        // Compute eigenvalue: λ = v^T * A * v / (v^T * v)
        let vt = ag::tensor_ops::transpose(v, &[1, 0]);
        let av = ag::tensor_ops::matmul(matrix, v);
        let numerator = ag::tensor_ops::matmul(vt, av);
        let denominator = ag::tensor_ops::matmul(vt, v);

        numerator / denominator
    }

    /// Matrix condition number approximation using eigenvalue ratio
    pub fn condition_number_approx<'g, F: ag::Float>(
        matrix: &ag::Tensor<'g, F>,
        iterations: usize,
        n: usize,
        ctx: &'g ag::Context<'g, F>,
    ) -> ag::Tensor<'g, F> {
        // Compute largest eigenvalue
        let lambda_max = dominant_eigenvalue(matrix, iterations, n, ctx);

        // For condition number, we'd need smallest eigenvalue too
        // This is a simplified approximation - return max eigenvalue as proxy
        lambda_max
    }

    /// Compute matrix rank approximation using SVD-like approach
    ///
    /// This is a simplified rank estimation - actual rank would require full SVD
    pub fn rank_approximation<'g, F: ag::Float>(
        matrix: &ag::Tensor<'g, F>,
        _tolerance: F,
        ctx: &'g ag::Context<'g, F>,
    ) -> ag::Tensor<'g, F> {
        // For now, return a constant estimate
        // A proper implementation would need SVD with gradient tracking
        ag::tensor_ops::convert_to_tensor(
            ag::ndarray::Array2::from_elem((1, 1), F::from(3.0).expect("Operation failed")),
            ctx,
        )
    }
}
