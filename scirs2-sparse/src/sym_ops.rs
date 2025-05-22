// Optimized operations for symmetric sparse matrices
//
// This module provides specialized, optimized implementations of common
// operations for symmetric sparse matrices, including matrix-vector products
// and other computations that can take advantage of symmetry.

use ndarray::{Array1, ArrayView1};
use num_traits::Float;
use std::fmt::Debug;
use std::ops::{Add, Mul};

use crate::error::SparseResult;
use crate::sym_coo::SymCooMatrix;
use crate::sym_csr::SymCsrMatrix;

/// Computes a matrix-vector product for symmetric CSR matrices.
///
/// This function computes `y = A * x` where `A` is a symmetric matrix
/// in CSR format, taking advantage of the symmetry. Only the lower (or upper)
/// triangular part of the matrix is stored, but the full matrix is used
/// in the computation.
///
/// # Arguments
///
/// * `matrix` - The symmetric matrix in CSR format
/// * `x` - The input vector
///
/// # Returns
///
/// The result vector `y = A * x`
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_sparse::sym_csr::SymCsrMatrix;
/// use scirs2_sparse::sym_ops::sym_csr_matvec;
///
/// // Create a symmetric matrix
/// let data = vec![2.0, 1.0, 2.0, 3.0, 1.0];
/// let indices = vec![0, 0, 1, 1, 2];
/// let indptr = vec![0, 1, 3, 5];
/// let matrix = SymCsrMatrix::new(data, indptr, indices, (3, 3)).unwrap();
///
/// // Create a vector
/// let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
///
/// // Compute the product
/// let y = sym_csr_matvec(&matrix, &x.view()).unwrap();
///
/// // Verify the result: [2*1 + 1*2 + 0*3, 1*1 + 2*2 + 3*3, 0*1 + 3*2 + 1*3] = [4, 14, 9]
/// assert_eq!(y[0], 4.0);
/// assert_eq!(y[1], 14.0);
/// assert_eq!(y[2], 9.0);
/// ```
pub fn sym_csr_matvec<T>(matrix: &SymCsrMatrix<T>, x: &ArrayView1<T>) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + Add<Output = T>,
{
    let (n, _) = matrix.shape();
    if x.len() != n {
        return Err(crate::error::SparseError::DimensionMismatch {
            expected: n,
            found: x.len(),
        });
    }

    let mut y = Array1::zeros(n);

    // First pass: Process the diagonal and lower triangular parts
    for i in 0..n {
        for j in matrix.indptr[i]..matrix.indptr[i + 1] {
            let col = matrix.indices[j];
            let val = matrix.data[j];

            y[i] = y[i] + val * x[col];

            // If not on the diagonal, also update the upper triangular part
            if i != col {
                y[col] = y[col] + val * x[i];
            }
        }
    }

    Ok(y)
}

/// Computes a matrix-vector product for symmetric COO matrices.
///
/// This function computes `y = A * x` where `A` is a symmetric matrix
/// in COO format, taking advantage of the symmetry. Only the lower (or upper)
/// triangular part of the matrix is stored, but the full matrix is used
/// in the computation.
///
/// # Arguments
///
/// * `matrix` - The symmetric matrix in COO format
/// * `x` - The input vector
///
/// # Returns
///
/// The result vector `y = A * x`
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_sparse::sym_coo::SymCooMatrix;
/// use scirs2_sparse::sym_ops::sym_coo_matvec;
///
/// // Create a symmetric matrix
/// let rows = vec![0, 1, 1, 2, 2];
/// let cols = vec![0, 0, 1, 1, 2];
/// let data = vec![2.0, 1.0, 2.0, 3.0, 1.0];
/// let matrix = SymCooMatrix::new(data, rows, cols, (3, 3)).unwrap();
///
/// // Create a vector
/// let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
///
/// // Compute the product
/// let y = sym_coo_matvec(&matrix, &x.view()).unwrap();
///
/// // Verify the result: [2*1 + 1*2 + 0*3, 1*1 + 2*2 + 3*3, 0*1 + 3*2 + 1*3] = [4, 14, 9]
/// assert_eq!(y[0], 4.0);
/// assert_eq!(y[1], 14.0);
/// assert_eq!(y[2], 9.0);
/// ```
pub fn sym_coo_matvec<T>(matrix: &SymCooMatrix<T>, x: &ArrayView1<T>) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + Add<Output = T>,
{
    let (n, _) = matrix.shape();
    if x.len() != n {
        return Err(crate::error::SparseError::DimensionMismatch {
            expected: n,
            found: x.len(),
        });
    }

    let mut y = Array1::zeros(n);

    // Process each non-zero element in the lower triangular part
    for i in 0..matrix.data.len() {
        let row = matrix.rows[i];
        let col = matrix.cols[i];
        let val = matrix.data[i];

        y[row] = y[row] + val * x[col];

        // If not on the diagonal, also update the upper triangular part
        if row != col {
            y[col] = y[col] + val * x[row];
        }
    }

    Ok(y)
}

/// Performs a symmetric rank-1 update of a symmetric CSR matrix.
///
/// This computes `A = A + alpha * x * x^T` where `A` is a symmetric matrix,
/// `alpha` is a scalar, and `x` is a vector.
///
/// # Arguments
///
/// * `matrix` - The symmetric matrix to update (will be modified in-place)
/// * `x` - The vector to use for the update
/// * `alpha` - The scalar multiplier
///
/// # Returns
///
/// Result with `()` on success
///
/// # Note
///
/// This operation preserves symmetry but may change the sparsity pattern of the matrix.
/// Currently only implemented for dense updates (all elements of x*x^T are considered).
/// For sparse updates, additional optimizations would be possible.
pub fn sym_csr_rank1_update<T>(
    matrix: &mut SymCsrMatrix<T>,
    x: &ArrayView1<T>,
    alpha: T,
) -> SparseResult<()>
where
    T: Float + Debug + Copy + Add<Output = T> + Mul<Output = T> + std::ops::AddAssign,
{
    let (n, _) = matrix.shape();
    if x.len() != n {
        return Err(crate::error::SparseError::DimensionMismatch {
            expected: n,
            found: x.len(),
        });
    }

    // For now, the easiest approach is to:
    // 1. Convert to a dense matrix
    // 2. Perform the rank-1 update
    // 3. Convert back to symmetric CSR format

    // Convert to dense
    let mut dense = matrix.to_dense();

    // Perform rank-1 update
    for i in 0..n {
        for j in 0..=i {
            // Only update lower triangular (including diagonal)
            let update = alpha * x[i] * x[j];
            dense[i][j] += update;
        }
    }

    // Convert back to CSR format (preserving symmetry)
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0];

    for (i, row) in dense.iter().enumerate().take(n) {
        for (j, &val) in row.iter().enumerate().take(i + 1) {
            // Only include lower triangular (including diagonal)
            if val != T::zero() {
                data.push(val);
                indices.push(j);
            }
        }
        indptr.push(data.len());
    }

    // Replace the matrix data
    matrix.data = data;
    matrix.indices = indices;
    matrix.indptr = indptr;

    Ok(())
}

/// Calculates the quadratic form `x^T * A * x` for a symmetric matrix `A`.
///
/// This computation takes advantage of symmetry for efficiency.
///
/// # Arguments
///
/// * `matrix` - The symmetric matrix
/// * `x` - The vector
///
/// # Returns
///
/// The scalar result of `x^T * A * x`
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_sparse::sym_csr::SymCsrMatrix;
/// use scirs2_sparse::sym_ops::sym_csr_quadratic_form;
///
/// // Create a symmetric matrix
/// let data = vec![2.0, 1.0, 2.0, 3.0, 1.0];
/// let indices = vec![0, 0, 1, 1, 2];
/// let indptr = vec![0, 1, 3, 5];
/// let matrix = SymCsrMatrix::new(data, indptr, indices, (3, 3)).unwrap();
///
/// // Create a vector
/// let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
///
/// // Compute the quadratic form
/// let result = sym_csr_quadratic_form(&matrix, &x.view()).unwrap();
///
/// // Verify: [1,2,3] * [2,1,0; 1,2,3; 0,3,1] * [1;2;3] = [1,2,3] * [4,14,9] = 4 + 28 + 27 = 59
/// assert_eq!(result, 59.0);
/// ```
pub fn sym_csr_quadratic_form<T>(matrix: &SymCsrMatrix<T>, x: &ArrayView1<T>) -> SparseResult<T>
where
    T: Float + Debug + Copy + Add<Output = T> + Mul<Output = T>,
{
    // First compute A * x
    let ax = sym_csr_matvec(matrix, x)?;

    // Then compute x^T * (A * x)
    let mut result = T::zero();
    for i in 0..ax.len() {
        result = result + x[i] * ax[i];
    }

    Ok(result)
}

/// Calculates the trace of a symmetric matrix.
///
/// The trace is the sum of the diagonal elements.
///
/// # Arguments
///
/// * `matrix` - The symmetric matrix
///
/// # Returns
///
/// The trace of the matrix
///
/// # Example
///
/// ```
/// use scirs2_sparse::sym_csr::SymCsrMatrix;
/// use scirs2_sparse::sym_ops::sym_csr_trace;
///
/// // Create a symmetric matrix
/// let data = vec![2.0, 1.0, 2.0, 3.0, 1.0];
/// let indices = vec![0, 0, 1, 1, 2];
/// let indptr = vec![0, 1, 3, 5];
/// let matrix = SymCsrMatrix::new(data, indptr, indices, (3, 3)).unwrap();
///
/// // Compute the trace
/// let trace = sym_csr_trace(&matrix);
///
/// // Verify: 2 + 2 + 1 = 5
/// assert_eq!(trace, 5.0);
/// ```
pub fn sym_csr_trace<T>(matrix: &SymCsrMatrix<T>) -> T
where
    T: Float + Debug + Copy + Add<Output = T>,
{
    let (n, _) = matrix.shape();
    let mut trace = T::zero();

    // Sum the diagonal elements
    for i in 0..n {
        for j in matrix.indptr[i]..matrix.indptr[i + 1] {
            let col = matrix.indices[j];
            if col == i {
                trace = trace + matrix.data[j];
                break;
            }
        }
    }

    trace
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sym_coo::SymCooMatrix;
    use crate::sym_csr::SymCsrMatrix;
    use crate::AsLinearOperator; // For the test_compare_with_standard_matvec test
    use approx::assert_relative_eq;
    use ndarray::Array1;

    // Create a simple symmetric matrix for testing
    fn create_test_sym_csr() -> SymCsrMatrix<f64> {
        // Create a symmetric matrix:
        // [2 1 0]
        // [1 2 3]
        // [0 3 1]

        // Lower triangular part (which is stored):
        // [2 0 0]
        // [1 2 0]
        // [0 3 1]

        let data = vec![2.0, 1.0, 2.0, 3.0, 1.0];
        let indices = vec![0, 0, 1, 1, 2];
        let indptr = vec![0, 1, 3, 5];

        SymCsrMatrix::new(data, indptr, indices, (3, 3)).unwrap()
    }

    // Create a simple symmetric matrix in COO format for testing
    fn create_test_sym_coo() -> SymCooMatrix<f64> {
        // Create a symmetric matrix:
        // [2 1 0]
        // [1 2 3]
        // [0 3 1]

        // Lower triangular part (which is stored):
        // [2 0 0]
        // [1 2 0]
        // [0 3 1]

        let data = vec![2.0, 1.0, 2.0, 3.0, 1.0];
        let rows = vec![0, 1, 1, 2, 2];
        let cols = vec![0, 0, 1, 1, 2];

        SymCooMatrix::new(data, rows, cols, (3, 3)).unwrap()
    }

    #[test]
    fn test_sym_csr_matvec() {
        let matrix = create_test_sym_csr();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let y = sym_csr_matvec(&matrix, &x.view()).unwrap();

        // Expected result: [2*1 + 1*2 + 0*3, 1*1 + 2*2 + 3*3, 0*1 + 3*2 + 1*3] = [4, 14, 9]
        assert_eq!(y.len(), 3);
        assert_relative_eq!(y[0], 4.0);
        assert_relative_eq!(y[1], 14.0);
        assert_relative_eq!(y[2], 9.0);
    }

    #[test]
    fn test_sym_coo_matvec() {
        let matrix = create_test_sym_coo();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let y = sym_coo_matvec(&matrix, &x.view()).unwrap();

        // Expected result: [2*1 + 1*2 + 0*3, 1*1 + 2*2 + 3*3, 0*1 + 3*2 + 1*3] = [4, 14, 9]
        assert_eq!(y.len(), 3);
        assert_relative_eq!(y[0], 4.0);
        assert_relative_eq!(y[1], 14.0);
        assert_relative_eq!(y[2], 9.0);
    }

    #[test]
    fn test_sym_csr_rank1_update() {
        let mut matrix = create_test_sym_csr();
        let x = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let alpha = 3.0;

        // Original diagonal element at (0,0) is 2.0
        // After rank-1 update with [1,0,0] and alpha=3, it should be 2+3*1*1 = 5
        sym_csr_rank1_update(&mut matrix, &x.view(), alpha).unwrap();

        // Check the updated value
        assert_relative_eq!(matrix.get(0, 0), 5.0);

        // Other values should remain unchanged
        assert_relative_eq!(matrix.get(0, 1), 1.0);
        assert_relative_eq!(matrix.get(1, 1), 2.0);
        assert_relative_eq!(matrix.get(1, 2), 3.0);
        assert_relative_eq!(matrix.get(2, 2), 1.0);
    }

    #[test]
    fn test_sym_csr_quadratic_form() {
        let matrix = create_test_sym_csr();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = sym_csr_quadratic_form(&matrix, &x.view()).unwrap();

        // Expected result: [1,2,3] * [2,1,0; 1,2,3; 0,3,1] * [1;2;3]
        // = [1,2,3] * [4,14,9] = 1*4 + 2*14 + 3*9 = 4 + 28 + 27 = 59
        assert_relative_eq!(result, 59.0);
    }

    #[test]
    fn test_sym_csr_trace() {
        let matrix = create_test_sym_csr();

        let trace = sym_csr_trace(&matrix);

        // Expected: 2 + 2 + 1 = 5
        assert_relative_eq!(trace, 5.0);
    }

    #[test]
    fn test_compare_with_standard_matvec() {
        // Create matrices and vectors
        let sym_csr = create_test_sym_csr();
        let full_csr = sym_csr.to_csr().unwrap();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // Compute using the optimized function
        let y_optimized = sym_csr_matvec(&sym_csr, &x.view()).unwrap();

        // Compute using the standard function
        let linear_op = full_csr.as_linear_operator();
        let y_standard = linear_op.matvec(x.as_slice().unwrap()).unwrap();

        // Compare results
        for i in 0..y_optimized.len() {
            assert_relative_eq!(y_optimized[i], y_standard[i]);
        }
    }
}
