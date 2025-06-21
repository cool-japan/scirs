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

/// Placeholder for future batch operations module
/// Will include: batch_matmul, batch_inv, batch_det, etc.
pub mod batch {
    // Batch matrix operations (coming soon)
}

/// Placeholder for future factorizations module
/// Will include: lu, qr, svd, cholesky with automatic differentiation
pub mod factorizations {
    // Matrix factorizations with gradients (coming soon)
}

/// Matrix calculus operations with automatic differentiation support
/// Includes: gradient, hessian, jacobian computations, VJP, JVP
pub mod matrix_calculus;

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
            ag::ndarray::Array2::from_shape_vec((n, n), eye_data).unwrap(),
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
            ag::ndarray::Array2::from_shape_vec((n, n), eye_data).unwrap(),
            ctx,
        )
    }
}
