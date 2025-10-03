//! High-level distributed linear algebra operations

use crate::distributed::{DistributedMatrix, DistributedVector, solvers, decomposition};
use crate::error::{LinalgError, LinalgResult};

/// High-level distributed linear algebra operations
pub struct DistributedLinalgOps;

impl DistributedLinalgOps {
    /// Distributed matrix multiplication: C = A * B
    pub fn distributed_matmul<T>(
        a: &DistributedMatrix<T>,
        b: &DistributedMatrix<T>,
    ) -> LinalgResult<DistributedMatrix<T>>
    where
        T: scirs2_core::numeric::Float + Send + Sync + 'static,
    {
        // Check matrix dimensions
        let (m, k) = a.global_shape();
        let (k2, n) = b.global_shape();

        if k != k2 {
            return Err(LinalgError::DimensionError(format!(
                "Matrix dimensions don't match for multiplication: ({}, {}) x ({}, {})",
                m, k, k2, n
            )));
        }

        // Execute distributed matrix multiplication
        a.multiply(b)
    }

    /// Distributed matrix addition: C = A + B
    pub fn distributed_add<T>(
        a: &DistributedMatrix<T>,
        b: &DistributedMatrix<T>,
    ) -> LinalgResult<DistributedMatrix<T>>
    where
        T: scirs2_core::numeric::Float + Send + Sync + 'static,
    {
        // Check matrix dimensions
        if a.global_shape() != b.global_shape() {
            return Err(LinalgError::DimensionError(format!(
                "Matrix dimensions don't match for addition: {:?} vs {:?}",
                a.global_shape(),
                b.global_shape()
            )));
        }

        // Execute distributed matrix addition
        a.add(b)
    }

    /// Distributed matrix transpose: B = A^T
    pub fn distributed_transpose<T>(
        matrix: &DistributedMatrix<T>,
    ) -> LinalgResult<DistributedMatrix<T>>
    where
        T: scirs2_core::numeric::Float + Send + Sync + 'static,
    {
        matrix.transpose()
    }

    /// Distributed solve linear system: Ax = b
    pub fn distributed_solve<T>(
        a: &DistributedMatrix<T>,
        b: &DistributedVector<T>,
    ) -> LinalgResult<DistributedVector<T>>
    where
        T: scirs2_core::numeric::Float + Send + Sync + 'static,
    {
        solvers::solve_linear_system(a, b)
    }

    /// Distributed LU decomposition
    pub fn distributed_lu<T>(
        matrix: &DistributedMatrix<T>,
    ) -> LinalgResult<(DistributedMatrix<T>, DistributedMatrix<T>)>
    where
        T: scirs2_core::numeric::Float + Send + Sync + 'static,
    {
        decomposition::lu_decomposition(matrix)
    }

    /// Distributed QR decomposition
    pub fn distributed_qr<T>(
        matrix: &DistributedMatrix<T>,
    ) -> LinalgResult<(DistributedMatrix<T>, DistributedMatrix<T>)>
    where
        T: scirs2_core::numeric::Float + Send + Sync + 'static,
    {
        decomposition::qr_decomposition(matrix)
    }
}