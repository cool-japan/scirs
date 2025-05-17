//! Linear operator interface for sparse matrices

use crate::error::{SparseError, SparseResult};
use num_traits::{Float, NumAssign};
use std::fmt::Debug;
use std::iter::Sum;
use std::marker::PhantomData;

/// Trait for representing a linear operator
///
/// This trait provides an abstract interface for linear operators,
/// allowing matrix-free implementations and compositions.
pub trait LinearOperator<F: Float> {
    /// The shape of the operator (rows, columns)
    fn shape(&self) -> (usize, usize);

    /// Apply the operator to a vector: y = A * x
    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>>;

    /// Apply the operator to a matrix: Y = A * X
    /// where X is column-major (each column is a vector)
    fn matmat(&self, x: &[Vec<F>]) -> SparseResult<Vec<Vec<F>>> {
        let mut result = Vec::new();
        for col in x {
            result.push(self.matvec(col)?);
        }
        Ok(result)
    }

    /// Apply the adjoint of the operator to a vector: y = A^H * x
    /// Default implementation returns an error
    fn rmatvec(&self, _x: &[F]) -> SparseResult<Vec<F>> {
        Err(crate::error::SparseError::OperationNotSupported(
            "adjoint not implemented for this operator".to_string(),
        ))
    }

    /// Apply the adjoint of the operator to a matrix: Y = A^H * X
    /// Default implementation calls rmatvec for each column
    fn rmatmat(&self, x: &[Vec<F>]) -> SparseResult<Vec<Vec<F>>> {
        let mut result = Vec::new();
        for col in x {
            result.push(self.rmatvec(col)?);
        }
        Ok(result)
    }

    /// Check if the operator supports adjoint operations
    fn has_adjoint(&self) -> bool {
        false
    }
}

/// Identity operator: I * x = x
pub struct IdentityOperator<F> {
    size: usize,
    _phantom: PhantomData<F>,
}

impl<F> IdentityOperator<F> {
    /// Create a new identity operator of given size
    pub fn new(size: usize) -> Self {
        Self {
            size,
            _phantom: PhantomData,
        }
    }
}

impl<F: Float> LinearOperator<F> for IdentityOperator<F> {
    fn shape(&self) -> (usize, usize) {
        (self.size, self.size)
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if x.len() != self.size {
            return Err(crate::error::SparseError::DimensionMismatch {
                expected: self.size,
                found: x.len(),
            });
        }
        Ok(x.to_vec())
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        self.matvec(x)
    }

    fn has_adjoint(&self) -> bool {
        true
    }
}

/// Scaled identity operator: (alpha * I) * x = alpha * x
pub struct ScaledIdentityOperator<F> {
    size: usize,
    scale: F,
}

impl<F: Float> ScaledIdentityOperator<F> {
    /// Create a new scaled identity operator
    pub fn new(size: usize, scale: F) -> Self {
        Self { size, scale }
    }
}

impl<F: Float + NumAssign> LinearOperator<F> for ScaledIdentityOperator<F> {
    fn shape(&self) -> (usize, usize) {
        (self.size, self.size)
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if x.len() != self.size {
            return Err(crate::error::SparseError::DimensionMismatch {
                expected: self.size,
                found: x.len(),
            });
        }
        Ok(x.iter().map(|&xi| xi * self.scale).collect())
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        // For real scalars, adjoint is the same
        self.matvec(x)
    }

    fn has_adjoint(&self) -> bool {
        true
    }
}

/// Diagonal operator: D * x where D is a diagonal matrix
pub struct DiagonalOperator<F> {
    diagonal: Vec<F>,
}

impl<F: Float> DiagonalOperator<F> {
    /// Create a new diagonal operator from diagonal values
    pub fn new(diagonal: Vec<F>) -> Self {
        Self { diagonal }
    }

    /// Get the diagonal values
    pub fn diagonal(&self) -> &[F] {
        &self.diagonal
    }
}

impl<F: Float + NumAssign> LinearOperator<F> for DiagonalOperator<F> {
    fn shape(&self) -> (usize, usize) {
        let n = self.diagonal.len();
        (n, n)
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if x.len() != self.diagonal.len() {
            return Err(crate::error::SparseError::DimensionMismatch {
                expected: self.diagonal.len(),
                found: x.len(),
            });
        }
        Ok(x.iter()
            .zip(&self.diagonal)
            .map(|(&xi, &di)| xi * di)
            .collect())
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        // For real diagonal matrices, adjoint is the same
        self.matvec(x)
    }

    fn has_adjoint(&self) -> bool {
        true
    }
}

/// Zero operator: 0 * x = 0
pub struct ZeroOperator<F> {
    shape: (usize, usize),
    _phantom: PhantomData<F>,
}

impl<F> ZeroOperator<F> {
    /// Create a new zero operator with given shape
    #[allow(dead_code)]
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            shape: (rows, cols),
            _phantom: PhantomData,
        }
    }
}

impl<F: Float> LinearOperator<F> for ZeroOperator<F> {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if x.len() != self.shape.1 {
            return Err(crate::error::SparseError::DimensionMismatch {
                expected: self.shape.1,
                found: x.len(),
            });
        }
        Ok(vec![F::zero(); self.shape.0])
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if x.len() != self.shape.0 {
            return Err(crate::error::SparseError::DimensionMismatch {
                expected: self.shape.0,
                found: x.len(),
            });
        }
        Ok(vec![F::zero(); self.shape.1])
    }

    fn has_adjoint(&self) -> bool {
        true
    }
}

/// Convert a sparse matrix to a linear operator
pub trait AsLinearOperator<F: Float> {
    /// Convert to a linear operator
    fn as_linear_operator(&self) -> Box<dyn LinearOperator<F>>;
}

/// Linear operator wrapper for sparse matrices
pub struct MatrixLinearOperator<F, M> {
    matrix: M,
    _phantom: PhantomData<F>,
}

impl<F, M> MatrixLinearOperator<F, M> {
    /// Create a new matrix linear operator
    pub fn new(matrix: M) -> Self {
        Self {
            matrix,
            _phantom: PhantomData,
        }
    }
}

// Implementation of LinearOperator for CSR matrices
use crate::csr::CsrMatrix;

impl<F: Float + NumAssign + Sum + 'static + Debug> LinearOperator<F>
    for MatrixLinearOperator<F, CsrMatrix<F>>
{
    fn shape(&self) -> (usize, usize) {
        (self.matrix.rows(), self.matrix.cols())
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if x.len() != self.matrix.cols() {
            return Err(SparseError::DimensionMismatch {
                expected: self.matrix.cols(),
                found: x.len(),
            });
        }

        // Manual implementation for generic types
        let mut result = vec![F::zero(); self.matrix.rows()];
        for (row, result_elem) in result.iter_mut().enumerate().take(self.matrix.rows()) {
            let row_range = self.matrix.row_range(row);
            let row_indices = &self.matrix.col_indices()[row_range.clone()];
            let row_data = &self.matrix.data[row_range];

            let mut sum = F::zero();
            for (col_idx, &col) in row_indices.iter().enumerate() {
                sum += row_data[col_idx] * x[col];
            }
            *result_elem = sum;
        }
        Ok(result)
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        // For CSR, we can compute A^T * x by transposing first
        let transposed = self.matrix.transpose();
        MatrixLinearOperator::new(transposed).matvec(x)
    }

    fn has_adjoint(&self) -> bool {
        true
    }
}

impl<F: Float + NumAssign + Sum + 'static + Debug> AsLinearOperator<F> for CsrMatrix<F> {
    fn as_linear_operator(&self) -> Box<dyn LinearOperator<F>> {
        Box::new(MatrixLinearOperator::new(self.clone()))
    }
}

// Composition operators for adding and multiplying operators
/// Sum of two linear operators: (A + B) * x = A * x + B * x
pub struct SumOperator<F> {
    a: Box<dyn LinearOperator<F>>,
    b: Box<dyn LinearOperator<F>>,
}

impl<F: Float + NumAssign> SumOperator<F> {
    /// Create a new sum operator
    #[allow(dead_code)]
    pub fn new(a: Box<dyn LinearOperator<F>>, b: Box<dyn LinearOperator<F>>) -> SparseResult<Self> {
        if a.shape() != b.shape() {
            return Err(crate::error::SparseError::ShapeMismatch {
                expected: a.shape(),
                found: b.shape(),
            });
        }
        Ok(Self { a, b })
    }
}

impl<F: Float + NumAssign> LinearOperator<F> for SumOperator<F> {
    fn shape(&self) -> (usize, usize) {
        self.a.shape()
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        let a_result = self.a.matvec(x)?;
        let b_result = self.b.matvec(x)?;
        Ok(a_result
            .iter()
            .zip(&b_result)
            .map(|(&a, &b)| a + b)
            .collect())
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if !self.a.has_adjoint() || !self.b.has_adjoint() {
            return Err(crate::error::SparseError::OperationNotSupported(
                "adjoint not supported for one or both operators".to_string(),
            ));
        }
        let a_result = self.a.rmatvec(x)?;
        let b_result = self.b.rmatvec(x)?;
        Ok(a_result
            .iter()
            .zip(&b_result)
            .map(|(&a, &b)| a + b)
            .collect())
    }

    fn has_adjoint(&self) -> bool {
        self.a.has_adjoint() && self.b.has_adjoint()
    }
}

/// Product of two linear operators: (A * B) * x = A * (B * x)
pub struct ProductOperator<F> {
    a: Box<dyn LinearOperator<F>>,
    b: Box<dyn LinearOperator<F>>,
}

impl<F: Float + NumAssign> ProductOperator<F> {
    /// Create a new product operator
    #[allow(dead_code)]
    pub fn new(a: Box<dyn LinearOperator<F>>, b: Box<dyn LinearOperator<F>>) -> SparseResult<Self> {
        let (_a_rows, a_cols) = a.shape();
        let (b_rows, _b_cols) = b.shape();
        if a_cols != b_rows {
            return Err(crate::error::SparseError::DimensionMismatch {
                expected: a_cols,
                found: b_rows,
            });
        }
        Ok(Self { a, b })
    }
}

impl<F: Float + NumAssign> LinearOperator<F> for ProductOperator<F> {
    fn shape(&self) -> (usize, usize) {
        let (a_rows, _) = self.a.shape();
        let (_, b_cols) = self.b.shape();
        (a_rows, b_cols)
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        let b_result = self.b.matvec(x)?;
        self.a.matvec(&b_result)
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if !self.a.has_adjoint() || !self.b.has_adjoint() {
            return Err(crate::error::SparseError::OperationNotSupported(
                "adjoint not supported for one or both operators".to_string(),
            ));
        }
        // (A * B)^H = B^H * A^H
        let a_result = self.a.rmatvec(x)?;
        self.b.rmatvec(&a_result)
    }

    fn has_adjoint(&self) -> bool {
        self.a.has_adjoint() && self.b.has_adjoint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_operator() {
        let op = IdentityOperator::<f64>::new(3);
        let x = vec![1.0, 2.0, 3.0];
        let y = op.matvec(&x).unwrap();
        assert_eq!(x, y);
    }

    #[test]
    fn test_scaled_identity_operator() {
        let op = ScaledIdentityOperator::new(3, 2.0);
        let x = vec![1.0, 2.0, 3.0];
        let y = op.matvec(&x).unwrap();
        assert_eq!(y, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_diagonal_operator() {
        let op = DiagonalOperator::new(vec![2.0, 3.0, 4.0]);
        let x = vec![1.0, 2.0, 3.0];
        let y = op.matvec(&x).unwrap();
        assert_eq!(y, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_zero_operator() {
        let op = ZeroOperator::<f64>::new(3, 3);
        let x = vec![1.0, 2.0, 3.0];
        let y = op.matvec(&x).unwrap();
        assert_eq!(y, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sum_operator() {
        let id = Box::new(IdentityOperator::<f64>::new(3));
        let scaled = Box::new(ScaledIdentityOperator::new(3, 2.0));
        let sum = SumOperator::new(id, scaled).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let y = sum.matvec(&x).unwrap();
        assert_eq!(y, vec![3.0, 6.0, 9.0]); // (I + 2I) * x = 3x
    }

    #[test]
    fn test_product_operator() {
        let id = Box::new(IdentityOperator::<f64>::new(3));
        let scaled = Box::new(ScaledIdentityOperator::new(3, 2.0));
        let product = ProductOperator::new(scaled, id).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let y = product.matvec(&x).unwrap();
        assert_eq!(y, vec![2.0, 4.0, 6.0]); // (2I * I) * x = 2x
    }
}
