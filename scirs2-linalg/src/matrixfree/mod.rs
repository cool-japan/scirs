//! Matrix-free operations for iterative solvers in large models
//!
//! This module provides abstractions for working with large matrices
//! without explicitly storing them, which is crucial for large-scale
//! machine learning and scientific computing applications.
//!
//! # Overview
//!
//! The key concept in matrix-free methods is to represent matrices implicitly
//! through their action on vectors (matrix-vector products), rather than
//! storing the entire matrix in memory. This approach is particularly useful for:
//!
//! * Large sparse matrices where most elements are zero
//! * Matrices with special structure (e.g., Toeplitz, circulant)
//! * Matrices that are too large to store in memory
//! * Algorithms that only require matrix-vector products
//!
//! # Examples
//!
//! ```
//! use scirs2_core::ndarray::{Array1, ArrayView1};
//! use scirs2_linalg::matrixfree::{LinearOperator, conjugate_gradient};
//!
//! // Define a linear operator that represents a 3x3 diagonal matrix
//! let diag_op = LinearOperator::new(
//!     3,                                  // dimension
//!     |v: &ArrayView1<f64>| -> Array1<f64> {
//!         let mut result = Array1::zeros(3);
//!         for i in 0..3 {
//!             result[i] = (i + 1) as f64 * v[i]; // Diagonal elements are 1, 2, 3
//!         }
//!         result
//!     }
//! );
//!
//! // Create a right-hand side vector
//! let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
//!
//! // Solve the system using matrix-free conjugate gradient
//! let x = conjugate_gradient(&diag_op, &b, 10, 1e-10).unwrap();
//!
//! // Expected solution: [1.0, 1.0, 1.0]
//! assert!((x[0] - 1.0).abs() < 1e-10);
//! assert!((x[1] - 1.0).abs() < 1e-10);
//! assert!((x[2] - 1.0).abs() < 1e-10);
//! ```

use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1};
use scirs2_core::numeric::{Float, NumAssign, One, Zero};
use std::{fmt::Debug, iter::Sum, sync::Arc};

use crate::error::{LinalgError, LinalgResult};
use crate::norm::vector_norm;

/// A trait for types that represent matrix-free linear operations
///
/// This trait abstracts the concept of a linear operator that maps
/// vectors from one space to another, without explicitly storing a matrix.
pub trait MatrixFreeOp<F>
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync,
{
    /// Apply the linear operator to a vector
    fn apply(&self, x: &ArrayView1<F>) -> LinalgResult<Array1<F>>;

    /// Get the number of rows in the linear operator
    fn nrows(&self) -> usize;

    /// Get the number of columns in the linear operator
    fn ncols(&self) -> usize;

    /// Check if the linear operator is symmetric
    fn is_symmetric(&self) -> bool {
        false // Default implementation assumes non-symmetric
    }

    /// Check if the linear operator is positive definite
    fn is_positive_definite(&self) -> bool {
        false // Default implementation assumes non-PD
    }
}

/// A concrete implementation of the `MatrixFreeOp` trait using closures
///
/// This struct provides a convenient way to create matrix-free operators
/// using closures that define the action of the operator on vectors.
/// Type alias for the linear operator function
pub type LinearOperatorFn<F> = Arc<dyn Fn(&ArrayView1<F>) -> Array1<F> + Send + Sync>;

pub struct LinearOperator<F>
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync,
{
    dim_rows: usize,
    dim_cols: usize,
    op: LinearOperatorFn<F>,
    symmetric: bool,
    positive_definite: bool,
}

impl<F> LinearOperator<F>
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync,
{
    /// Create a new square linear operator
    ///
    /// # Arguments
    ///
    /// * `dimension` - Dimension of the square operator
    /// * `op` - Closure that implements the matrix-vector product
    ///
    /// # Returns
    ///
    /// A new `LinearOperator` instance
    pub fn new<O>(dimension: usize, op: O) -> Self
    where
        O: Fn(&ArrayView1<F>) -> Array1<F> + Send + Sync + 'static,
    {
        LinearOperator {
            dim_rows: dimension,
            dim_cols: dimension,
            op: Arc::new(op),
            symmetric: false,
            positive_definite: false,
        }
    }

    /// Create a new rectangular linear operator
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// * `op` - Closure that implements the matrix-vector product
    ///
    /// # Returns
    ///
    /// A new `LinearOperator` instance
    pub fn new_rectangular<O>(rows: usize, cols: usize, op: O) -> Self
    where
        O: Fn(&ArrayView1<F>) -> Array1<F> + Send + Sync + 'static,
    {
        LinearOperator {
            dim_rows: rows,
            dim_cols: cols,
            op: Arc::new(op),
            symmetric: false,
            positive_definite: false,
        }
    }

    /// Mark the operator as symmetric
    ///
    /// # Returns
    ///
    /// Self with the symmetric flag set to true
    pub fn symmetric(mut self) -> Self {
        if self.dim_rows != self.dim_cols {
            panic!("Only square operators can be symmetric");
        }
        self.symmetric = true;
        self
    }

    /// Mark the operator as positive definite
    ///
    /// # Returns
    ///
    /// Self with the positive_definite flag set to true
    pub fn positive_definite(mut self) -> Self {
        if !self.symmetric {
            panic!("Only symmetric operators can be positive definite");
        }
        self.positive_definite = true;
        self
    }

    /// Create the transpose of this linear operator
    ///
    /// # Returns
    ///
    /// A new `LinearOperator` that represents the transpose of this operator
    pub fn transpose(&self) -> Self
    where
        F: 'static,
    {
        let op_arc = Arc::clone(&self.op);
        let rows = self.dim_rows;
        let cols = self.dim_cols;

        // If the operator is symmetric, just return a clone
        if self.symmetric && rows == cols {
            return LinearOperator {
                dim_rows: rows,
                dim_cols: cols,
                op: op_arc,
                symmetric: true,
                positive_definite: self.positive_definite,
            };
        }

        // For non-symmetric operators, we need to implement the transpose
        // using the adjoint trick: <A^T x, y> = <x, A y>
        // This is an approximation and will be less efficient
        LinearOperator {
            dim_rows: cols,
            dim_cols: rows,
            op: Arc::new(move |x: &ArrayView1<F>| {
                let mut result = Array1::zeros(rows);
                // This is a naive implementation and can be very inefficient
                // for large dimensions. In practice, the user should provide
                // a proper implementation of the transpose.
                for i in 0..cols {
                    let mut unit = Array1::zeros(cols);
                    unit[i] = F::one();
                    let col = (op_arc)(&unit.view());
                    for j in 0..rows {
                        result[j] += col[j] * x[i];
                    }
                }
                result
            }),
            symmetric: false,
            positive_definite: false,
        }
    }
}

impl<F> MatrixFreeOp<F> for LinearOperator<F>
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync,
{
    fn apply(&self, x: &ArrayView1<F>) -> LinalgResult<Array1<F>> {
        if x.len() != self.dim_cols {
            return Err(LinalgError::ShapeError(format!(
                "Input vector has wrong length: expected {}, got {}",
                self.dim_cols,
                x.len()
            )));
        }
        Ok((self.op)(x))
    }

    fn nrows(&self) -> usize {
        self.dim_rows
    }

    fn ncols(&self) -> usize {
        self.dim_cols
    }

    fn is_symmetric(&self) -> bool {
        self.symmetric
    }

    fn is_positive_definite(&self) -> bool {
        self.positive_definite
    }
}

/// Create a diagonal linear operator from a vector of diagonal elements
///
/// # Arguments
///
/// * `diag` - Vector of diagonal elements
///
/// # Returns
///
/// A `LinearOperator` implementing a diagonal matrix
#[allow(dead_code)]
pub fn diagonal_operator<F>(diag: &ArrayView1<F>) -> LinearOperator<F>
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Clone + Debug + Send + Sync + 'static,
{
    let diag_owned = diag.to_owned();
    let n = diag.len();

    LinearOperator {
        dim_rows: n,
        dim_cols: n,
        op: Arc::new(move |x: &ArrayView1<F>| {
            let mut result = Array1::zeros(n);
            for i in 0..n {
                result[i] = diag_owned[i] * x[i];
            }
            result
        }),
        symmetric: true,
        // A diagonal matrix is positive definite if all diagonal elements are positive
        positive_definite: diag.iter().all(|&d| d > F::zero()),
    }
}

/// Create a block diagonal linear operator from a vector of smaller operators
///
/// # Arguments
///
/// * `blocks` - Vector of smaller linear operators to place on the diagonal
///
/// # Returns
///
/// A `LinearOperator` implementing a block diagonal matrix
#[allow(dead_code)]
pub fn block_diagonal_operator<F>(blocks: Vec<LinearOperator<F>>) -> LinearOperator<F>
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Clone + Debug + Send + Sync + 'static,
{
    // Calculate dimensions
    let n_rows: usize = blocks.iter().map(|b| b.nrows()).sum();
    let n_cols: usize = blocks.iter().map(|b| b.ncols()).sum();

    // Check if all _blocks are symmetric/positive definite
    let all_symmetric = blocks.iter().all(|b| b.is_symmetric());
    let all_positive_definite = all_symmetric && blocks.iter().all(|b| b.is_positive_definite());

    // Create the operator
    let blocks_owned = blocks; // Move ownership
    LinearOperator {
        dim_rows: n_rows,
        dim_cols: n_cols,
        op: Arc::new(move |x: &ArrayView1<F>| {
            let mut result = Array1::zeros(n_rows);
            let mut row_offset = 0;
            let mut col_offset = 0;

            for block in &blocks_owned {
                let n_block_rows = block.nrows();
                let n_block_cols = block.ncols();

                // Extract the relevant part of the input vector
                let x_block = x.slice(s![col_offset..col_offset + n_block_cols]);

                // Apply the block operator
                let result_block = block.apply(&x_block.view()).unwrap();

                // Place the result in the output vector
                for (i, &val) in result_block.iter().enumerate() {
                    result[row_offset + i] = val;
                }

                row_offset += n_block_rows;
                col_offset += n_block_cols;
            }

            result
        }),
        symmetric: all_symmetric,
        positive_definite: all_positive_definite,
    }
}

/// Matrix-free implementation of the Conjugate Gradient method
///
/// Solves Ax = b using the Conjugate Gradient method, where A is represented
/// by a matrix-free operator. The operator must be symmetric positive definite.
///
/// # Arguments
///
/// * `a` - Matrix-free operator representing the coefficient matrix
/// * `b` - Right-hand side vector
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// * Solution vector x
#[allow(dead_code)]
pub fn conjugate_gradient<F, A>(
    a: &A,
    b: &Array1<F>,
    max_iter: usize,
    tol: F,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Debug + Send + Sync,
    A: MatrixFreeOp<F>,
{
    // Check that A is square
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Expected square operator, got shape {}x{}",
            a.nrows(),
            a.ncols()
        )));
    }

    // Check that dimensions are compatible
    if a.nrows() != b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Shape mismatch: operator shape {}x{}, vector shape {}",
            a.nrows(),
            a.ncols(),
            b.len()
        )));
    }

    // Verify that A is symmetric positive definite (if information is available)
    // This is just a hint, we can't actually verify it in the matrix-free setting
    if !a.is_symmetric() {
        eprintln!("Warning: Operator might not be symmetric");
    }
    if !a.is_positive_definite() {
        eprintln!("Warning: Operator might not be positive definite");
    }

    let n = a.nrows();

    // Initialize solution with zeros
    let mut x = Array1::zeros(n);

    // If b is zero, return zero solution
    let b_norm = vector_norm(&b.view(), 2)?;
    if b_norm < F::epsilon() {
        return Ok(x);
    }

    // Initial residual r = b - Ax
    let ax = a.apply(&x.view())?;
    let mut r = b.clone();
    r -= &ax;

    // Initial search direction p = r
    let mut p = r.clone();

    // Initial residual norm squared
    let mut rsold = r.dot(&r);

    // If initial guess is very close to solution
    if rsold.sqrt() < tol * b_norm {
        return Ok(x);
    }

    for _iter in 0..max_iter {
        // Compute A*p
        let ap = a.apply(&p.view())?;

        // Compute step size alpha
        let pap = p.dot(&ap);
        let alpha = rsold / pap;

        // Update solution x = x + alpha*p
        x = &x + &(&p * alpha);

        // Update residual r = r - alpha*A*p
        r = &r - &(&ap * alpha);

        // Compute new residual norm squared
        let rsnew = r.dot(&r);

        // Check convergence
        if rsnew.sqrt() < tol * b_norm {
            return Ok(x);
        }

        // Compute direction update beta
        let beta = rsnew / rsold;

        // Update search direction p = r + beta*p
        p = &r + &(&p * beta);

        // Update old residual norm
        rsold = rsnew;
    }

    // Return current solution if max iterations reached
    Ok(x)
}

/// Matrix-free implementation of GMRES (Generalized Minimal Residual) method
///
/// Solves Ax = b using the GMRES method, where A is represented by a matrix-free operator.
/// This method is suitable for non-symmetric matrices.
///
/// # Arguments
///
/// * `a` - Matrix-free operator representing the coefficient matrix
/// * `b` - Right-hand side vector
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
/// * `restart` - Number of iterations before restarting (optional)
///
/// # Returns
///
/// * Solution vector x
#[allow(dead_code)]
pub fn gmres<F, A>(
    a: &A,
    b: &Array1<F>,
    max_iter: usize,
    tol: F,
    restart: Option<usize>,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Debug + Send + Sync,
    A: MatrixFreeOp<F>,
{
    // Check that dimensions are compatible
    if a.nrows() != b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Shape mismatch: operator shape {}x{}, vector shape {}",
            a.nrows(),
            a.ncols(),
            b.len()
        )));
    }

    let n = a.nrows();
    let restart_iter = restart.unwrap_or(n);

    // Initialize solution with zeros
    let mut x = Array1::zeros(n);

    // If b is zero, return zero solution
    let b_norm = vector_norm(&b.view(), 2)?;
    if b_norm < F::epsilon() {
        return Ok(x);
    }

    // Outer iteration (restarts)
    for _outer in 0..max_iter {
        // Compute initial residual r = b - Ax
        let ax = a.apply(&x.view())?;
        let mut r = b.clone();
        r -= &ax;

        // Check if we've already converged
        let r_norm = vector_norm(&r.view(), 2)?;
        if r_norm < tol * b_norm {
            return Ok(x);
        }

        // Initialize the Krylov subspace with the normalized residual
        let beta = r_norm;
        let mut v = Array1::zeros(n);
        for i in 0..n {
            v[i] = r[i] / beta;
        }

        // Storage for the Hessenberg matrix
        let mut h = Array2::zeros((restart_iter + 1, restart_iter));

        // Storage for the orthogonal basis of the Krylov subspace
        let mut v_basis = Vec::with_capacity(restart_iter + 1);
        v_basis.push(v);

        // Storage for the rotations in the Hessenberg matrix
        let mut cs: Vec<F> = Vec::with_capacity(restart_iter);
        let mut sn: Vec<F> = Vec::with_capacity(restart_iter);

        // Storage for the right-hand side in the least squares problem
        let mut g = Array1::zeros(restart_iter + 1);
        g[0] = beta;

        // Inner iteration (Arnoldi process and least squares solve)
        let mut i = 0;
        while i < restart_iter {
            // Arnoldi process: Generate a new basis vector and orthogonalize
            let av = a.apply(&v_basis[i].view())?;
            let mut w = av;

            // Modified Gram-Schmidt orthogonalization
            for j in 0..=i {
                h[[j, i]] = w.dot(&v_basis[j]);
                w = &w - &(&v_basis[j] * h[[j, i]]);
            }

            // Compute the norm of the new basis vector
            h[[i + 1, i]] = vector_norm(&w.view(), 2)?;

            // If the norm is very small, we've reached a breakdown
            if h[[i + 1, i]] < F::epsilon() {
                // We've converged or encountered a breakdown
                i += 1;
                break;
            }

            // Normalize the new basis vector
            let mut new_v = Array1::zeros(n);
            for j in 0..n {
                new_v[j] = w[j] / h[[i + 1, i]];
            }
            v_basis.push(new_v);

            // Apply previous Givens rotations to the new column of the Hessenberg matrix
            for j in 0..i {
                let temp = h[[j, i]];
                h[[j, i]] = cs[j] * temp + sn[j] * h[[j + 1, i]];
                h[[j + 1, i]] = -sn[j] * temp + cs[j] * h[[j + 1, i]];
            }

            // Compute the new Givens rotation
            let (c, s) = givens_rotation(h[[i, i]], h[[i + 1, i]]);
            cs.push(c);
            sn.push(s);

            // Apply the new Givens rotation to the last element of the new column
            h[[i, i]] = c * h[[i, i]] + s * h[[i + 1, i]];
            h[[i + 1, i]] = F::zero();

            // Apply the new Givens rotation to the right-hand side
            let temp = g[i];
            g[i] = c * temp + s * g[i + 1];
            g[i + 1] = -s * temp + c * g[i + 1];

            // Check convergence
            let residual = g[i + 1].abs();
            if residual < tol * b_norm {
                // We've converged, solve the upper triangular system
                i += 1;
                break;
            }

            i += 1;
        }

        // Solve the upper triangular system H y = g
        let mut y = Array1::zeros(i);
        for j in (0..i).rev() {
            let mut sum = g[j];
            for k in (j + 1)..i {
                sum -= h[[j, k]] * y[k];
            }
            y[j] = sum / h[[j, j]];
        }

        // Update the solution x = x + V y
        for j in 0..i {
            x = &x + &(&v_basis[j] * y[j]);
        }

        // If we've converged or reached the maximum number of iterations, return
        let ax = a.apply(&x.view())?;
        let mut r = b.clone();
        r -= &ax;
        let r_norm = vector_norm(&r.view(), 2)?;
        if r_norm < tol * b_norm || i < restart_iter {
            return Ok(x);
        }
    }

    // Return the best solution we have
    Ok(x)
}

/// Helper function for GMRES: compute the Givens rotation matrix parameters
#[allow(dead_code)]
fn givens_rotation<F>(a: F, b: F) -> (F, F)
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync,
{
    if b == F::zero() {
        (F::one(), F::zero())
    } else if a.abs() < b.abs() {
        let t = a / b;
        let s = F::one() / (F::one() + t * t).sqrt();
        let c = s * t;
        (c, s)
    } else {
        let t = b / a;
        let c = F::one() / (F::one() + t * t).sqrt();
        let s = c * t;
        (c, s)
    }
}

/// Create a Jacobi preconditioner for matrix-free operators
///
/// The Jacobi preconditioner is the inverse of the diagonal of the matrix.
/// For matrix-free operators, we approximate the diagonal by applying the
/// operator to unit vectors.
///
/// # Arguments
///
/// * `a` - Matrix-free operator
///
/// # Returns
///
/// A linear operator representing the Jacobi preconditioner
#[allow(dead_code)]
pub fn jacobi_preconditioner<F, A>(a: &A) -> LinalgResult<LinearOperator<F>>
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Clone + Debug + Send + Sync + 'static,
    A: MatrixFreeOp<F>,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(
            "Jacobi preconditioner requires a square operator".to_string(),
        ));
    }

    let n = a.nrows();
    let mut diag = Array1::zeros(n);

    // Estimate the diagonal entries by applying A to unit vectors
    for i in 0..n {
        let mut e_i = Array1::zeros(n);
        e_i[i] = F::one();
        let a_e_i = a.apply(&e_i.view())?;
        diag[i] = a_e_i[i];
    }

    // Check for zeros on the diagonal
    for i in 0..n {
        if diag[i].abs() < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Jacobi preconditioner encountered zero on diagonal".to_string(),
            ));
        }
        // Invert the diagonal for the preconditioner
        diag[i] = F::one() / diag[i];
    }

    // Create and return the preconditioner
    Ok(diagonal_operator(&diag.view()))
}

/// Preconditioned Conjugate Gradient method for matrix-free operators
///
/// Solves Ax = b using the Preconditioned Conjugate Gradient method,
/// where A is represented by a matrix-free operator and M is a preconditioner.
///
/// # Arguments
///
/// * `a` - Matrix-free operator representing the coefficient matrix
/// * `m` - Matrix-free operator representing the preconditioner
/// * `b` - Right-hand side vector
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// * Solution vector x
#[allow(dead_code)]
pub fn preconditioned_conjugate_gradient<F, A, M>(
    a: &A,
    m: &M,
    b: &Array1<F>,
    max_iter: usize,
    tol: F,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Debug + Send + Sync,
    A: MatrixFreeOp<F>,
    M: MatrixFreeOp<F>,
{
    // Check that A is square
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Expected square operator, got shape {}x{}",
            a.nrows(),
            a.ncols()
        )));
    }

    // Check that dimensions are compatible
    if a.nrows() != b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Shape mismatch: operator shape {}x{}, vector shape {}",
            a.nrows(),
            a.ncols(),
            b.len()
        )));
    }

    // Check that the preconditioner is compatible
    if m.nrows() != a.nrows() || m.ncols() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Preconditioner shape {}x{} doesn't match operator shape {}x{}",
            m.nrows(),
            m.ncols(),
            a.nrows(),
            a.ncols()
        )));
    }

    let n = a.nrows();

    // Initialize solution with zeros
    let mut x = Array1::zeros(n);

    // If b is zero, return zero solution
    let b_norm = vector_norm(&b.view(), 2)?;
    if b_norm < F::epsilon() {
        return Ok(x);
    }

    // Initial residual r = b - Ax
    let ax = a.apply(&x.view())?;
    let mut r = b.clone();
    r -= &ax;

    // Initial preconditioned residual z = M^-1 r
    let mut z = m.apply(&r.view())?;

    // Initial search direction p = z
    let mut p = z.clone();

    // Initial residual inner product
    let mut rz_old = r.dot(&z);

    // If initial guess is very close to solution
    if vector_norm(&r.view(), 2)? < tol * b_norm {
        return Ok(x);
    }

    for _iter in 0..max_iter {
        // Compute A*p
        let ap = a.apply(&p.view())?;

        // Compute step size alpha
        let pap = p.dot(&ap);
        let alpha = rz_old / pap;

        // Update solution x = x + alpha*p
        x = &x + &(&p * alpha);

        // Update residual r = r - alpha*A*p
        r = &r - &(&ap * alpha);

        // Check convergence
        if vector_norm(&r.view(), 2)? < tol * b_norm {
            return Ok(x);
        }

        // Update preconditioned residual z = M^-1 r
        z = m.apply(&r.view())?;

        // Compute new residual inner product
        let rz_new = r.dot(&z);

        // Compute direction update beta
        let beta = rz_new / rz_old;

        // Update search direction p = z + beta*p
        p = &z + &(&p * beta);

        // Update old residual inner product
        rz_old = rz_new;
    }

    // Return current solution if max iterations reached
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    // Helper function to check solution
    fn check_solution<F, A>(a: &A, x: &ArrayView1<F>, b: &ArrayView1<F>, tol: F) -> bool
    where
        F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Debug + Send + Sync,
        A: MatrixFreeOp<F>,
    {
        let ax = a.apply(x).unwrap();
        let mut diff = Array1::zeros(x.len());
        for i in 0..x.len() {
            diff[i] = ax[i] - b[i];
        }

        let diff_norm = vector_norm(&diff.view(), 2).unwrap();
        let b_norm = vector_norm(b, 2).unwrap();

        diff_norm < tol * b_norm.max(F::one())
    }

    #[test]
    fn test_linear_operator_apply() {
        // Create a linear operator representing a 2x2 identity matrix
        let identity = LinearOperator::new(2, |v: &ArrayView1<f64>| v.to_owned());

        // Apply to a vector
        let x = array![1.0, 2.0];
        let y = identity.apply(&x.view()).unwrap();

        // Result should equal input for identity operator
        assert_relative_eq!(y[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diagonal_operator() {
        // Create a diagonal operator with elements [2.0, 3.0]
        let diag = array![2.0, 3.0];
        let diag_op = diagonal_operator(&diag.view());

        // Apply to a vector
        let x = array![1.0, 2.0];
        let y = diag_op.apply(&x.view()).unwrap();

        // Result should be [2.0*1.0, 3.0*2.0] = [2.0, 6.0]
        assert_relative_eq!(y[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_block_diagonal_operator() {
        // Create two diagonal operators
        let diag1 = array![2.0, 3.0];
        let diag_op1 = diagonal_operator(&diag1.view());

        let diag2 = array![4.0];
        let diag_op2 = diagonal_operator(&diag2.view());

        // Create a block diagonal operator
        let block_op = block_diagonal_operator(vec![diag_op1, diag_op2]);

        // Apply to a vector
        let x = array![1.0, 2.0, 3.0];
        let y = block_op.apply(&x.view()).unwrap();

        // Result should be [2.0*1.0, 3.0*2.0, 4.0*3.0] = [2.0, 6.0, 12.0]
        assert_relative_eq!(y[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 6.0, epsilon = 1e-10);
        assert_relative_eq!(y[2], 12.0, epsilon = 1e-10);
    }

    #[test]
    fn testmatrix_free_conjugate_gradient() {
        // Create a linear operator representing a symmetric positive definite matrix
        // [4.0, 1.0]
        // [1.0, 3.0]
        let spd_op = LinearOperator::new(2, |v: &ArrayView1<f64>| {
            let mut result = Array1::zeros(2);
            result[0] = 4.0 * v[0] + 1.0 * v[1];
            result[1] = 1.0 * v[0] + 3.0 * v[1];
            result
        })
        .symmetric()
        .positive_definite();

        // Define the right-hand side
        let b = array![1.0, 2.0];

        // Solve using matrix-free conjugate gradient
        let x = conjugate_gradient(&spd_op, &b, 10, 1e-10).unwrap();

        // Check solution
        assert!(check_solution(&spd_op, &x.view(), &b.view(), 1e-8));
    }

    #[test]
    fn testmatrix_free_gmres() {
        // Create a linear operator representing a non-symmetric matrix
        // [3.0, 1.0]
        // [1.0, 2.0]
        let op = LinearOperator::new_rectangular(2, 2, |v: &ArrayView1<f64>| {
            let mut result = Array1::zeros(2);
            result[0] = 3.0 * v[0] + 1.0 * v[1];
            result[1] = 1.0 * v[0] + 2.0 * v[1];
            result
        });

        // Define the right-hand side
        let b = array![4.0, 3.0];

        // Solve using matrix-free GMRES
        let x = gmres(&op, &b, 10, 1e-10, None).unwrap();

        // Check solution
        assert!(check_solution(&op, &x.view(), &b.view(), 1e-8));
    }

    #[test]
    fn test_jacobi_preconditioner() {
        // Create a matrix explicitly to compute the diagonal
        let a_mat = array![[4.0, 1.0], [1.0, 3.0]];

        // Create a linear operator representing this matrix
        let op = LinearOperator::new(2, move |v: &ArrayView1<f64>| {
            let mut result = Array1::zeros(2);
            for i in 0..2 {
                for j in 0..2 {
                    result[i] += a_mat[[i, j]] * v[j];
                }
            }
            result
        });

        // Create a Jacobi preconditioner
        let precond = jacobi_preconditioner(&op).unwrap();

        // Apply to a vector
        let x = array![1.0, 2.0];
        let y = precond.apply(&x.view()).unwrap();

        // The preconditioner should be the inverse of the diagonal of A
        // Diagonal of A: [4.0, 3.0]
        // Inverse: [1/4.0, 1/3.0]
        // Applied to [1.0, 2.0]: [1.0/4.0, 2.0/3.0] = [0.25, 0.6666...]
        assert_relative_eq!(y[0], 0.25, epsilon = 1e-10);
        assert_relative_eq!(y[1], 2.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_preconditioned_conjugate_gradient() {
        // Create a linear operator representing a symmetric positive definite matrix
        // [4.0, 1.0]
        // [1.0, 3.0]
        let spd_op = LinearOperator::new(2, |v: &ArrayView1<f64>| {
            let mut result = Array1::zeros(2);
            result[0] = 4.0 * v[0] + 1.0 * v[1];
            result[1] = 1.0 * v[0] + 3.0 * v[1];
            result
        })
        .symmetric()
        .positive_definite();

        // Create a Jacobi preconditioner (diagonal of A)
        let diag = array![1.0 / 4.0, 1.0 / 3.0];
        let precond = diagonal_operator(&diag.view());

        // Define the right-hand side
        let b = array![1.0, 2.0];

        // Solve using preconditioned matrix-free conjugate gradient
        let x = preconditioned_conjugate_gradient(&spd_op, &precond, &b, 10, 1e-10).unwrap();

        // Check solution
        assert!(check_solution(&spd_op, &x.view(), &b.view(), 1e-8));
    }
}
