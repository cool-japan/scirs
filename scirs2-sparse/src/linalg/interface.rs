//! Linear operator interface for sparse matrices

#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_mut)]

use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;
use std::iter::Sum;
use std::marker::PhantomData;

/// Type alias for matrix-vector function
type MatVecFn<F> = Box<dyn Fn(&[F]) -> SparseResult<Vec<F>> + Send + Sync>;

/// Type alias for solver function
type SolverFn<F> = Box<dyn Fn(&[F]) -> SparseResult<Vec<F>> + Send + Sync>;

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
    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
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
#[derive(Clone)]
pub struct IdentityOperator<F> {
    size: usize,
    phantom: PhantomData<F>,
}

impl<F> IdentityOperator<F> {
    /// Create a new identity operator of given size
    pub fn new(size: usize) -> Self {
        Self {
            size,
            phantom: PhantomData,
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
#[derive(Clone)]
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
#[derive(Clone)]
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
#[derive(Clone)]
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
    phantom: PhantomData<F>,
}

impl<F, M> MatrixLinearOperator<F, M> {
    /// Create a new matrix linear operator
    pub fn new(matrix: M) -> Self {
        Self {
            matrix,
            phantom: PhantomData,
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
            let row_indices = &self.matrix.colindices()[row_range.clone()];
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

// Implementation of LinearOperator for CsrArray
use crate::csr_array::CsrArray;

impl<F: Float + NumAssign + Sum + 'static + Debug> LinearOperator<F>
    for MatrixLinearOperator<F, CsrArray<F>>
{
    fn shape(&self) -> (usize, usize) {
        self.matrix.shape()
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if x.len() != self.matrix.shape().1 {
            return Err(SparseError::DimensionMismatch {
                expected: self.matrix.shape().1,
                found: x.len(),
            });
        }

        use scirs2_core::ndarray::Array1;
        let x_array = Array1::from_vec(x.to_vec());
        let result = self.matrix.dot_vector(&x_array.view())?;
        Ok(result.to_vec())
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        // For CSR A^T * x, we iterate through columns
        if x.len() != self.matrix.shape().0 {
            return Err(SparseError::DimensionMismatch {
                expected: self.matrix.shape().0,
                found: x.len(),
            });
        }

        let mut result = vec![F::zero(); self.matrix.shape().1];

        // Iterate through each row of the matrix
        for (row_idx, &x_val) in x.iter().enumerate() {
            if x_val != F::zero() {
                // Get row data for this row
                let row_start = self.matrix.get_indptr()[row_idx];
                let row_end = self.matrix.get_indptr()[row_idx + 1];

                for idx in row_start..row_end {
                    let col_idx = self.matrix.get_indices()[idx];
                    let data_val = self.matrix.get_data()[idx];
                    result[col_idx] += data_val * x_val;
                }
            }
        }

        Ok(result)
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

impl<F: Float + NumAssign + Sum + 'static + Debug> AsLinearOperator<F>
    for crate::csr_array::CsrArray<F>
{
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
        let (b_rows, b_cols) = b.shape();
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

/// Function-based linear operator for matrix-free implementations
pub struct FunctionOperator<F> {
    shape: (usize, usize),
    matvec_fn: MatVecFn<F>,
    rmatvec_fn: Option<MatVecFn<F>>,
}

impl<F: Float + 'static> FunctionOperator<F> {
    /// Create a new function-based operator
    #[allow(dead_code)]
    pub fn new<MV, RMV>(shape: (usize, usize), matvec_fn: MV, rmatvec_fn: Option<RMV>) -> Self
    where
        MV: Fn(&[F]) -> SparseResult<Vec<F>> + Send + Sync + 'static,
        RMV: Fn(&[F]) -> SparseResult<Vec<F>> + Send + Sync + 'static,
    {
        Self {
            shape,
            matvec_fn: Box::new(matvec_fn),
            rmatvec_fn: rmatvec_fn
                .map(|f| Box::new(f) as Box<dyn Fn(&[F]) -> SparseResult<Vec<F>> + Send + Sync>),
        }
    }

    /// Create a matrix-free operator from a function
    #[allow(dead_code)]
    pub fn from_function<FMv>(shape: (usize, usize), matvec_fn: FMv) -> Self
    where
        FMv: Fn(&[F]) -> SparseResult<Vec<F>> + Send + Sync + 'static,
    {
        Self::new(shape, matvec_fn, None::<fn(&[F]) -> SparseResult<Vec<F>>>)
    }
}

impl<F: Float> LinearOperator<F> for FunctionOperator<F> {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        (self.matvec_fn)(x)
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        match &self.rmatvec_fn {
            Some(f) => f(x),
            None => Err(SparseError::OperationNotSupported(
                "adjoint not implemented for this function operator".to_string(),
            )),
        }
    }

    fn has_adjoint(&self) -> bool {
        self.rmatvec_fn.is_some()
    }
}

/// Inverse operator: A^(-1)
/// Note: This is a conceptual operator, actual implementation depends on the specific matrix
pub struct InverseOperator<F> {
    original: Box<dyn LinearOperator<F>>,
    solver_fn: SolverFn<F>,
}

impl<F: Float> InverseOperator<F> {
    /// Create a new inverse operator with a custom solver function
    #[allow(dead_code)]
    pub fn new<S>(original: Box<dyn LinearOperator<F>>, solver_fn: S) -> SparseResult<Self>
    where
        S: Fn(&[F]) -> SparseResult<Vec<F>> + Send + Sync + 'static,
    {
        let (rows, cols) = original.shape();
        if rows != cols {
            return Err(SparseError::ValueError(
                "Cannot invert non-square operator".to_string(),
            ));
        }

        Ok(Self {
            original,
            solver_fn: Box::new(solver_fn),
        })
    }
}

impl<F: Float> LinearOperator<F> for InverseOperator<F> {
    fn shape(&self) -> (usize, usize) {
        self.original.shape()
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        // A^(-1) * x is equivalent to solving A * y = x for y
        (self.solver_fn)(x)
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        // (A^(-1))^H = (A^H)^(-1)
        // So we need to solve A^H * y = x for y
        if !self.original.has_adjoint() {
            return Err(SparseError::OperationNotSupported(
                "adjoint not supported for original operator".to_string(),
            ));
        }

        // This is a conceptual implementation - in practice, you'd need
        // a solver for the adjoint system
        Err(SparseError::OperationNotSupported(
            "adjoint of inverse operator not yet implemented".to_string(),
        ))
    }

    fn has_adjoint(&self) -> bool {
        false // Simplified for now
    }
}

/// Transpose operator: A^T
pub struct TransposeOperator<F> {
    original: Box<dyn LinearOperator<F>>,
}

impl<F: Float + NumAssign> TransposeOperator<F> {
    /// Create a new transpose operator
    pub fn new(original: Box<dyn LinearOperator<F>>) -> Self {
        Self { original }
    }
}

impl<F: Float + NumAssign> LinearOperator<F> for TransposeOperator<F> {
    fn shape(&self) -> (usize, usize) {
        let (rows, cols) = self.original.shape();
        (cols, rows) // Transpose dimensions
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        // A^T * x = (A^H * x) for real matrices
        self.original.rmatvec(x)
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        // (A^T)^H = A for real matrices
        self.original.matvec(x)
    }

    fn has_adjoint(&self) -> bool {
        true // Transpose always has adjoint
    }
}

/// Adjoint operator: A^H (Hermitian transpose)
pub struct AdjointOperator<F> {
    original: Box<dyn LinearOperator<F>>,
}

impl<F: Float + NumAssign> AdjointOperator<F> {
    /// Create a new adjoint operator
    pub fn new(original: Box<dyn LinearOperator<F>>) -> SparseResult<Self> {
        if !original.has_adjoint() {
            return Err(SparseError::OperationNotSupported(
                "Original operator does not support adjoint operations".to_string(),
            ));
        }
        Ok(Self { original })
    }
}

impl<F: Float + NumAssign> LinearOperator<F> for AdjointOperator<F> {
    fn shape(&self) -> (usize, usize) {
        let (rows, cols) = self.original.shape();
        (cols, rows) // Adjoint transposes dimensions
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        self.original.rmatvec(x)
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        self.original.matvec(x)
    }

    fn has_adjoint(&self) -> bool {
        true
    }
}

/// Difference operator: A - B
pub struct DifferenceOperator<F> {
    a: Box<dyn LinearOperator<F>>,
    b: Box<dyn LinearOperator<F>>,
}

impl<F: Float + NumAssign> DifferenceOperator<F> {
    /// Create a new difference operator
    pub fn new(a: Box<dyn LinearOperator<F>>, b: Box<dyn LinearOperator<F>>) -> SparseResult<Self> {
        if a.shape() != b.shape() {
            return Err(SparseError::ShapeMismatch {
                expected: a.shape(),
                found: b.shape(),
            });
        }
        Ok(Self { a, b })
    }
}

impl<F: Float + NumAssign> LinearOperator<F> for DifferenceOperator<F> {
    fn shape(&self) -> (usize, usize) {
        self.a.shape()
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        let a_result = self.a.matvec(x)?;
        let b_result = self.b.matvec(x)?;
        Ok(a_result
            .iter()
            .zip(&b_result)
            .map(|(&a, &b)| a - b)
            .collect())
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if !self.a.has_adjoint() || !self.b.has_adjoint() {
            return Err(SparseError::OperationNotSupported(
                "adjoint not supported for one or both operators".to_string(),
            ));
        }
        let a_result = self.a.rmatvec(x)?;
        let b_result = self.b.rmatvec(x)?;
        Ok(a_result
            .iter()
            .zip(&b_result)
            .map(|(&a, &b)| a - b)
            .collect())
    }

    fn has_adjoint(&self) -> bool {
        self.a.has_adjoint() && self.b.has_adjoint()
    }
}

/// Scaled operator: alpha * A
pub struct ScaledOperator<F> {
    alpha: F,
    operator: Box<dyn LinearOperator<F>>,
}

impl<F: Float + NumAssign> ScaledOperator<F> {
    /// Create a new scaled operator
    pub fn new(alpha: F, operator: Box<dyn LinearOperator<F>>) -> Self {
        Self { alpha, operator }
    }
}

impl<F: Float + NumAssign> LinearOperator<F> for ScaledOperator<F> {
    fn shape(&self) -> (usize, usize) {
        self.operator.shape()
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        let result = self.operator.matvec(x)?;
        Ok(result.iter().map(|&val| self.alpha * val).collect())
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if !self.operator.has_adjoint() {
            return Err(SparseError::OperationNotSupported(
                "adjoint not supported for underlying operator".to_string(),
            ));
        }
        let result = self.operator.rmatvec(x)?;
        Ok(result.iter().map(|&val| self.alpha * val).collect())
    }

    fn has_adjoint(&self) -> bool {
        self.operator.has_adjoint()
    }
}

/// Chain/composition of multiple operators: A_n * A_(n-1) * ... * A_1
pub struct ChainOperator<F> {
    operators: Vec<Box<dyn LinearOperator<F>>>,
    totalshape: (usize, usize),
}

impl<F: Float + NumAssign> ChainOperator<F> {
    /// Create a new chain operator from a list of operators
    /// Operators are applied from right to left (like function composition)
    #[allow(dead_code)]
    pub fn new(operators: Vec<Box<dyn LinearOperator<F>>>) -> SparseResult<Self> {
        if operators.is_empty() {
            return Err(SparseError::ValueError(
                "Cannot create chain with no operators".to_string(),
            ));
        }

        // Check dimension compatibility
        #[allow(clippy::needless_range_loop)]
        for i in 0..operators.len() - 1 {
            let (_, a_cols) = operators[i].shape();
            let (b_rows, _) = operators[i + 1].shape();
            if a_cols != b_rows {
                return Err(SparseError::DimensionMismatch {
                    expected: a_cols,
                    found: b_rows,
                });
            }
        }

        let (first_rows, _) = operators[0].shape();
        let (_, last_cols) = operators.last().unwrap().shape();
        let totalshape = (first_rows, last_cols);

        Ok(Self {
            operators,
            totalshape,
        })
    }
}

impl<F: Float + NumAssign> LinearOperator<F> for ChainOperator<F> {
    fn shape(&self) -> (usize, usize) {
        self.totalshape
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        let mut result = x.to_vec();
        // Apply operators from right to left
        for op in self.operators.iter().rev() {
            result = op.matvec(&result)?;
        }
        Ok(result)
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        // Check if all operators support adjoint
        for op in &self.operators {
            if !op.has_adjoint() {
                return Err(SparseError::OperationNotSupported(
                    "adjoint not supported for all operators in chain".to_string(),
                ));
            }
        }

        let mut result = x.to_vec();
        // Apply adjoints from left to right (reverse order)
        for op in &self.operators {
            result = op.rmatvec(&result)?;
        }
        Ok(result)
    }

    fn has_adjoint(&self) -> bool {
        self.operators.iter().all(|op| op.has_adjoint())
    }
}

/// Power operator: A^n (for positive integer n)
pub struct PowerOperator<F> {
    operator: Box<dyn LinearOperator<F>>,
    power: usize,
}

impl<F: Float + NumAssign> PowerOperator<F> {
    /// Create a new power operator
    pub fn new(operator: Box<dyn LinearOperator<F>>, power: usize) -> SparseResult<Self> {
        let (rows, cols) = operator.shape();
        if rows != cols {
            return Err(SparseError::ValueError(
                "Can only compute powers of square operators".to_string(),
            ));
        }
        if power == 0 {
            return Err(SparseError::ValueError(
                "Power must be positive".to_string(),
            ));
        }
        Ok(Self { operator, power })
    }
}

impl<F: Float + NumAssign> LinearOperator<F> for PowerOperator<F> {
    fn shape(&self) -> (usize, usize) {
        self.operator.shape()
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        let mut result = x.to_vec();
        for _ in 0..self.power {
            result = self.operator.matvec(&result)?;
        }
        Ok(result)
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if !self.operator.has_adjoint() {
            return Err(SparseError::OperationNotSupported(
                "adjoint not supported for underlying operator".to_string(),
            ));
        }
        let mut result = x.to_vec();
        for _ in 0..self.power {
            result = self.operator.rmatvec(&result)?;
        }
        Ok(result)
    }

    fn has_adjoint(&self) -> bool {
        self.operator.has_adjoint()
    }
}

/// Enhanced LinearOperator trait with composition methods
#[allow(dead_code)]
pub trait LinearOperatorExt<F: Float + NumAssign>: LinearOperator<F> {
    /// Add this operator with another
    fn add(&self, other: Box<dyn LinearOperator<F>>) -> SparseResult<Box<dyn LinearOperator<F>>>;

    /// Subtract another operator from this one
    fn sub(&self, other: Box<dyn LinearOperator<F>>) -> SparseResult<Box<dyn LinearOperator<F>>>;

    /// Multiply this operator with another (composition)
    fn mul(&self, other: Box<dyn LinearOperator<F>>) -> SparseResult<Box<dyn LinearOperator<F>>>;

    /// Scale this operator by a scalar
    fn scale(&self, alpha: F) -> Box<dyn LinearOperator<F>>;

    /// Transpose this operator
    fn transpose(&self) -> Box<dyn LinearOperator<F>>;

    /// Adjoint of this operator
    fn adjoint(&self) -> SparseResult<Box<dyn LinearOperator<F>>>;

    /// Power of this operator
    fn pow(&self, n: usize) -> SparseResult<Box<dyn LinearOperator<F>>>;
}

// Specific implementations for each cloneable operator type
macro_rules! impl_linear_operator_ext {
    ($typ:ty) => {
        impl<F: Float + NumAssign + Copy + 'static> LinearOperatorExt<F> for $typ {
            fn add(
                &self,
                other: Box<dyn LinearOperator<F>>,
            ) -> SparseResult<Box<dyn LinearOperator<F>>> {
                let self_box = Box::new(self.clone());
                Ok(Box::new(SumOperator::new(self_box, other)?))
            }

            fn sub(
                &self,
                other: Box<dyn LinearOperator<F>>,
            ) -> SparseResult<Box<dyn LinearOperator<F>>> {
                let self_box = Box::new(self.clone());
                Ok(Box::new(DifferenceOperator::new(self_box, other)?))
            }

            fn mul(
                &self,
                other: Box<dyn LinearOperator<F>>,
            ) -> SparseResult<Box<dyn LinearOperator<F>>> {
                let self_box = Box::new(self.clone());
                Ok(Box::new(ProductOperator::new(self_box, other)?))
            }

            fn scale(&self, alpha: F) -> Box<dyn LinearOperator<F>> {
                let self_box = Box::new(self.clone());
                Box::new(ScaledOperator::new(alpha, self_box))
            }

            fn transpose(&self) -> Box<dyn LinearOperator<F>> {
                let self_box = Box::new(self.clone());
                Box::new(TransposeOperator::new(self_box))
            }

            fn adjoint(&self) -> SparseResult<Box<dyn LinearOperator<F>>> {
                let self_box = Box::new(self.clone());
                Ok(Box::new(AdjointOperator::new(self_box)?))
            }

            fn pow(&self, n: usize) -> SparseResult<Box<dyn LinearOperator<F>>> {
                let self_box = Box::new(self.clone());
                Ok(Box::new(PowerOperator::new(self_box, n)?))
            }
        }
    };
}

// Apply the macro to all cloneable operator types
impl_linear_operator_ext!(IdentityOperator<F>);
impl_linear_operator_ext!(ScaledIdentityOperator<F>);
impl_linear_operator_ext!(DiagonalOperator<F>);
impl_linear_operator_ext!(ZeroOperator<F>);

/// Utility functions for operator composition
/// Add two operators: left + right
#[allow(dead_code)]
pub fn add_operators<F: Float + NumAssign + 'static>(
    left: Box<dyn LinearOperator<F>>,
    right: Box<dyn LinearOperator<F>>,
) -> SparseResult<Box<dyn LinearOperator<F>>> {
    Ok(Box::new(SumOperator::new(left, right)?))
}

/// Subtract two operators: left - right
#[allow(dead_code)]
pub fn subtract_operators<F: Float + NumAssign + 'static>(
    left: Box<dyn LinearOperator<F>>,
    right: Box<dyn LinearOperator<F>>,
) -> SparseResult<Box<dyn LinearOperator<F>>> {
    Ok(Box::new(DifferenceOperator::new(left, right)?))
}

/// Multiply two operators: left * right  
#[allow(dead_code)]
pub fn multiply_operators<F: Float + NumAssign + 'static>(
    left: Box<dyn LinearOperator<F>>,
    right: Box<dyn LinearOperator<F>>,
) -> SparseResult<Box<dyn LinearOperator<F>>> {
    Ok(Box::new(ProductOperator::new(left, right)?))
}

/// Scale an operator: alpha * operator
#[allow(dead_code)]
pub fn scale_operator<F: Float + NumAssign + 'static>(
    alpha: F,
    operator: Box<dyn LinearOperator<F>>,
) -> Box<dyn LinearOperator<F>> {
    Box::new(ScaledOperator::new(alpha, operator))
}

/// Transpose an operator: A^T
#[allow(dead_code)]
pub fn transpose_operator<F: Float + NumAssign + 'static>(
    operator: Box<dyn LinearOperator<F>>,
) -> Box<dyn LinearOperator<F>> {
    Box::new(TransposeOperator::new(operator))
}

/// Adjoint of an operator: A^H
#[allow(dead_code)]
pub fn adjoint_operator<F: Float + NumAssign + 'static>(
    operator: Box<dyn LinearOperator<F>>,
) -> SparseResult<Box<dyn LinearOperator<F>>> {
    Ok(Box::new(AdjointOperator::new(operator)?))
}

/// Compose multiple operators: A_n * A_(n-1) * ... * A_1
#[allow(dead_code)]
pub fn compose_operators<F: Float + NumAssign + 'static>(
    operators: Vec<Box<dyn LinearOperator<F>>>,
) -> SparseResult<Box<dyn LinearOperator<F>>> {
    Ok(Box::new(ChainOperator::new(operators)?))
}

/// Power of an operator: A^n
#[allow(dead_code)]
pub fn power_operator<F: Float + NumAssign + 'static>(
    operator: Box<dyn LinearOperator<F>>,
    n: usize,
) -> SparseResult<Box<dyn LinearOperator<F>>> {
    Ok(Box::new(PowerOperator::new(operator, n)?))
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

    #[test]
    fn test_difference_operator() {
        let scaled_3 = Box::new(ScaledIdentityOperator::new(3, 3.0));
        let scaled_2 = Box::new(ScaledIdentityOperator::new(3, 2.0));
        let diff = DifferenceOperator::new(scaled_3, scaled_2).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let y = diff.matvec(&x).unwrap();
        assert_eq!(y, vec![1.0, 2.0, 3.0]); // (3I - 2I) * x = I * x = x
    }

    #[test]
    fn test_scaled_operator() {
        let id = Box::new(IdentityOperator::<f64>::new(3));
        let scaled = ScaledOperator::new(5.0, id);
        let x = vec![1.0, 2.0, 3.0];
        let y = scaled.matvec(&x).unwrap();
        assert_eq!(y, vec![5.0, 10.0, 15.0]); // 5 * I * x = 5x
    }

    #[test]
    fn test_transpose_operator() {
        let diag = Box::new(DiagonalOperator::new(vec![2.0, 3.0, 4.0]));
        let transpose = TransposeOperator::new(diag);
        let x = vec![1.0, 2.0, 3.0];
        let y = transpose.matvec(&x).unwrap();
        // For diagonal matrices, transpose equals original
        assert_eq!(y, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_adjoint_operator() {
        let diag = Box::new(DiagonalOperator::new(vec![2.0, 3.0, 4.0]));
        let adjoint = AdjointOperator::new(diag).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let y = adjoint.matvec(&x).unwrap();
        // For real diagonal matrices, adjoint equals original
        assert_eq!(y, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_chain_operator() {
        let op1 = Box::new(ScaledIdentityOperator::new(3, 2.0));
        let op2 = Box::new(ScaledIdentityOperator::new(3, 3.0));
        let chain = ChainOperator::new(vec![op1, op2]).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let y = chain.matvec(&x).unwrap();
        // Chain applies from right to left: (2I) * (3I) * x = 6I * x = 6x
        assert_eq!(y, vec![6.0, 12.0, 18.0]);
    }

    #[test]
    fn test_power_operator() {
        let scaled = Box::new(ScaledIdentityOperator::new(3, 2.0));
        let power = PowerOperator::new(scaled, 3).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let y = power.matvec(&x).unwrap();
        // (2I)^3 * x = 8I * x = 8x
        assert_eq!(y, vec![8.0, 16.0, 24.0]);
    }

    #[test]
    fn test_composition_utility_functions() {
        let id = Box::new(IdentityOperator::<f64>::new(3));
        let scaled = Box::new(ScaledIdentityOperator::new(3, 2.0));

        // Test add_operators
        let sum = add_operators(id.clone(), scaled.clone()).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let y = sum.matvec(&x).unwrap();
        assert_eq!(y, vec![3.0, 6.0, 9.0]); // (I + 2I) * x = 3x

        // Test subtract_operators
        let diff = subtract_operators(scaled.clone(), id.clone()).unwrap();
        let y2 = diff.matvec(&x).unwrap();
        assert_eq!(y2, vec![1.0, 2.0, 3.0]); // (2I - I) * x = x

        // Test multiply_operators
        let product = multiply_operators(scaled.clone(), id.clone()).unwrap();
        let y3 = product.matvec(&x).unwrap();
        assert_eq!(y3, vec![2.0, 4.0, 6.0]); // (2I * I) * x = 2x

        // Test scale_operator
        let scaled_op = scale_operator(3.0, id.clone());
        let y4 = scaled_op.matvec(&x).unwrap();
        assert_eq!(y4, vec![3.0, 6.0, 9.0]); // 3 * I * x = 3x

        // Test transpose_operator
        let transpose = transpose_operator(scaled.clone());
        let y5 = transpose.matvec(&x).unwrap();
        assert_eq!(y5, vec![2.0, 4.0, 6.0]); // (2I)^T * x = 2I * x = 2x

        // Test compose_operators
        let ops: Vec<Box<dyn LinearOperator<f64>>> = vec![scaled.clone(), id.clone()];
        let composed = compose_operators(ops).unwrap();
        let y6 = composed.matvec(&x).unwrap();
        assert_eq!(y6, vec![2.0, 4.0, 6.0]); // (2I * I) * x = 2x

        // Test power_operator
        let power = power_operator(scaled.clone(), 2).unwrap();
        let y7 = power.matvec(&x).unwrap();
        assert_eq!(y7, vec![4.0, 8.0, 12.0]); // (2I)^2 * x = 4I * x = 4x
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let op1 = Box::new(IdentityOperator::<f64>::new(3));
        let op2 = Box::new(IdentityOperator::<f64>::new(4));

        // Test sum operator dimension mismatch
        assert!(SumOperator::new(op1.clone(), op2.clone()).is_err());

        // Test difference operator dimension mismatch
        assert!(DifferenceOperator::new(op1.clone(), op2.clone()).is_err());

        // Test product operator dimension mismatch (incompatible dimensions)
        let rect1 = Box::new(ZeroOperator::<f64>::new(3, 4));
        let rect2 = Box::new(ZeroOperator::<f64>::new(5, 3));
        assert!(ProductOperator::new(rect1, rect2).is_err());
    }

    #[test]
    fn test_adjoint_not_supported_error() {
        // Create a function operator without adjoint support
        let func_op = Box::new(FunctionOperator::from_function((3, 3), |x: &[f64]| {
            Ok(x.to_vec())
        }));

        // Attempting to create adjoint should fail
        assert!(AdjointOperator::new(func_op).is_err());
    }

    #[test]
    fn test_power_operator_errors() {
        let rect_op = Box::new(ZeroOperator::<f64>::new(3, 4));

        // Power of non-square operator should fail
        assert!(PowerOperator::new(rect_op, 2).is_err());

        let square_op = Box::new(IdentityOperator::<f64>::new(3));

        // Power of 0 should fail
        assert!(PowerOperator::new(square_op, 0).is_err());
    }
}
