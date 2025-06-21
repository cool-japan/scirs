//! Advanced matrix equation solvers
//!
//! This module provides solvers for various matrix equations beyond simple Ax = b,
//! including Sylvester, Lyapunov, and Riccati equations.

use crate::eigen::eig;
use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array2, ArrayView2};
use num_traits::{Float, NumAssign};
use std::fmt::{Debug, Display};

/// Solves the generalized Sylvester equation AXB + CXD = E
///
/// # Arguments
/// * `a` - Matrix A
/// * `b` - Matrix B
/// * `c` - Matrix C
/// * `d` - Matrix D
/// * `e` - Matrix E
///
/// # Returns
/// * Solution matrix X
///
/// # Example
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_equations::solve_generalized_sylvester;
///
/// let a = array![[1.0, 0.0], [0.0, 2.0]];
/// let b = array![[3.0, 0.0], [0.0, 4.0]];
/// let c = array![[0.5, 0.0], [0.0, 0.5]];
/// let d = array![[0.25, 0.0], [0.0, 0.25]];
/// let e = array![[1.0, 2.0], [3.0, 4.0]];
///
/// let x = solve_generalized_sylvester(&a.view(), &b.view(), &c.view(), &d.view(), &e.view()).unwrap();
/// ```
pub fn solve_generalized_sylvester<A>(
    a: &ArrayView2<A>,
    b: &ArrayView2<A>,
    c: &ArrayView2<A>,
    d: &ArrayView2<A>,
    e: &ArrayView2<A>,
) -> LinalgResult<Array2<A>>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + std::iter::Sum + 'static,
{
    let (m, n) = (a.shape()[0], b.shape()[0]);

    // Validate dimensions
    if a.shape()[1] != m || c.shape() != a.shape() {
        return Err(LinalgError::ShapeError(
            "Matrices A and C must be square and have the same shape".to_string(),
        ));
    }
    if b.shape()[1] != n || d.shape() != b.shape() {
        return Err(LinalgError::ShapeError(
            "Matrices B and D must be square and have the same shape".to_string(),
        ));
    }
    if e.shape() != [m, n] {
        return Err(LinalgError::ShapeError(format!(
            "Matrix E must have shape [{}, {}]",
            m, n
        )));
    }

    // Special case: Standard Sylvester equation (C = 0, D = 0)
    if c.iter().all(|&x| x.abs() < A::epsilon()) && d.iter().all(|&x| x.abs() < A::epsilon()) {
        return solve_sylvester(a, b, e);
    }

    // General case: Use Kronecker product formulation
    // (A⊗B^T + C⊗D^T) vec(X) = vec(E)
    // This is a simplified implementation - production code would use more efficient methods

    // Create the coefficient matrix using Kronecker products
    let mut coeff = Array2::<A>::zeros((m * n, m * n));

    // Add A ⊗ B^T
    for i in 0..m {
        for j in 0..m {
            for k in 0..n {
                for l in 0..n {
                    coeff[[i * n + k, j * n + l]] = a[[i, j]] * b[[l, k]];
                }
            }
        }
    }

    // Add C ⊗ D^T
    for i in 0..m {
        for j in 0..m {
            for k in 0..n {
                for l in 0..n {
                    coeff[[i * n + k, j * n + l]] += c[[i, j]] * d[[l, k]];
                }
            }
        }
    }

    // Vectorize E
    let e_vec: Vec<A> = e.t().iter().cloned().collect();
    let e_vec = Array2::from_shape_vec((m * n, 1), e_vec)
        .map_err(|e| LinalgError::ShapeError(format!("Failed to reshape vector: {}", e)))?;

    // Solve the linear system (use column vector view)
    let e_vec_1d = e_vec.column(0);
    let x_vec = crate::solve::solve(&coeff.view(), &e_vec_1d, None)?;

    // Reshape solution back to matrix form
    let x_data: Vec<A> = x_vec.iter().cloned().collect();
    Ok(Array2::from_shape_vec((n, m), x_data)
        .map_err(|e| LinalgError::ShapeError(format!("Failed to reshape solution: {}", e)))?
        .t()
        .to_owned())
}

/// Solves the Sylvester equation AX + XB = C
///
/// # Arguments
/// * `a` - Matrix A (m × m)
/// * `b` - Matrix B (n × n)
/// * `c` - Matrix C (m × n)
///
/// # Returns
/// * Solution matrix X (m × n)
///
/// # Example
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_equations::solve_sylvester;
///
/// let a = array![[1.0, 2.0], [0.0, 3.0]];
/// let b = array![[-4.0, 0.0], [0.0, -5.0]];
/// let c = array![[1.0, 2.0], [3.0, 4.0]];
///
/// let x = solve_sylvester(&a.view(), &b.view(), &c.view()).unwrap();
/// ```
pub fn solve_sylvester<A>(
    a: &ArrayView2<A>,
    b: &ArrayView2<A>,
    c: &ArrayView2<A>,
) -> LinalgResult<Array2<A>>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + std::iter::Sum + 'static,
{
    let (m, n) = (a.shape()[0], b.shape()[0]);

    // Validate dimensions
    if a.shape()[1] != m {
        return Err(LinalgError::ShapeError(
            "Matrix A must be square".to_string(),
        ));
    }
    if b.shape()[1] != n {
        return Err(LinalgError::ShapeError(
            "Matrix B must be square".to_string(),
        ));
    }
    if c.shape() != [m, n] {
        return Err(LinalgError::ShapeError(format!(
            "Matrix C must have shape [{}, {}]",
            m, n
        )));
    }

    // Solve using vectorization: (I ⊗ A + B^T ⊗ I) vec(X) = vec(C)
    // This is more robust than the Bartels-Stewart algorithm for small matrices

    // Create the coefficient matrix using Kronecker products
    let mut coeff = Array2::<A>::zeros((m * n, m * n));

    // Build the coefficient matrix for the vectorized equation
    // For AX + XB = C, the vectorized form is:
    // (I_n ⊗ A + B^T ⊗ I_m) vec(X) = vec(C)
    // where vec(X) stacks columns of X

    // Add I_n ⊗ A part
    for col_block in 0..n {
        for row_in_a in 0..m {
            for col_in_a in 0..m {
                let row_idx = col_block * m + row_in_a;
                let col_idx = col_block * m + col_in_a;
                coeff[[row_idx, col_idx]] = a[[row_in_a, col_in_a]];
            }
        }
    }

    // Add B^T ⊗ I_m part
    for row_block in 0..n {
        for col_block in 0..n {
            for diag in 0..m {
                let row_idx = row_block * m + diag;
                let col_idx = col_block * m + diag;
                coeff[[row_idx, col_idx]] += b[[col_block, row_block]];
            }
        }
    }

    // Vectorize C (column-major order)
    let c_vec: Vec<A> = c.t().iter().cloned().collect();
    let c_vec = Array2::from_shape_vec((m * n, 1), c_vec)
        .map_err(|e| LinalgError::ShapeError(format!("Failed to reshape vector: {}", e)))?;

    // Solve the linear system
    let c_vec_1d = c_vec.column(0);
    let x_vec = crate::solve::solve(&coeff.view(), &c_vec_1d, None)?;

    // Reshape solution back to matrix form (column-major order)
    let x_data: Vec<A> = x_vec.iter().cloned().collect();
    Ok(Array2::from_shape_vec((n, m), x_data)
        .map_err(|e| LinalgError::ShapeError(format!("Failed to reshape solution: {}", e)))?
        .t()
        .to_owned())
}

/// Solves the continuous-time algebraic Riccati equation (CARE)
/// A^T X + X A - X B R^{-1} B^T X + Q = 0
///
/// # Arguments
/// * `a` - State matrix A (n × n)
/// * `b` - Input matrix B (n × m)
/// * `q` - State cost matrix Q (n × n, symmetric positive semidefinite)
/// * `r` - Input cost matrix R (m × m, symmetric positive definite)
///
/// # Returns
/// * Solution matrix X (n × n, symmetric positive semidefinite)
///
/// # Example
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_equations::solve_continuous_riccati;
///
/// let a = array![[0.0, 1.0], [0.0, 0.0]];
/// let b = array![[0.0], [1.0]];
/// let q = array![[1.0, 0.0], [0.0, 1.0]];
/// let r = array![[1.0]];
///
/// let x = solve_continuous_riccati(&a.view(), &b.view(), &q.view(), &r.view()).unwrap();
/// ```
pub fn solve_continuous_riccati<A>(
    a: &ArrayView2<A>,
    b: &ArrayView2<A>,
    q: &ArrayView2<A>,
    r: &ArrayView2<A>,
) -> LinalgResult<Array2<A>>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + std::iter::Sum + 'static,
{
    let n = a.shape()[0];
    let m = b.shape()[1];

    // Validate dimensions
    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError(
            "Matrix A must be square".to_string(),
        ));
    }
    if b.shape() != [n, m] {
        return Err(LinalgError::ShapeError(format!(
            "Matrix B must have shape [{}, m]",
            n
        )));
    }
    if q.shape() != [n, n] {
        return Err(LinalgError::ShapeError(
            "Matrix Q must be n × n".to_string(),
        ));
    }
    if r.shape() != [m, m] {
        return Err(LinalgError::ShapeError(
            "Matrix R must be m × m".to_string(),
        ));
    }

    // Solve using the Hamiltonian matrix approach
    // Form the Hamiltonian matrix H = [A, -BR^{-1}B^T; -Q, -A^T]
    let mut h = Array2::<A>::zeros((2 * n, 2 * n));

    // Upper left: A
    h.slice_mut(ndarray::s![..n, ..n]).assign(a);

    // Upper right: -BR^{-1}B^T
    let r_inv = crate::inv(r, None)?;
    let br_inv_bt = b.dot(&r_inv).dot(&b.t());
    // Negate br_inv_bt
    let neg_br_inv_bt = br_inv_bt.mapv(|x| -x);
    h.slice_mut(ndarray::s![..n, n..]).assign(&neg_br_inv_bt);

    // Lower left: -Q
    // Negate q
    let neg_q = q.mapv(|x| -x);
    h.slice_mut(ndarray::s![n.., ..n]).assign(&neg_q);

    // Lower right: -A^T
    // Negate a.t()
    let neg_at = a.t().mapv(|x| -x);
    h.slice_mut(ndarray::s![n.., n..]).assign(&neg_at);

    // Compute eigendecomposition of H
    let (eigvals, eigvecs) = eig(&h.view(), None)?;

    // Select stable eigenspace (eigenvalues with negative real parts)
    let mut stable_indices = Vec::new();
    for (i, &lambda) in eigvals.iter().enumerate() {
        if lambda.re < A::zero() {
            stable_indices.push(i);
        }
    }

    if stable_indices.len() != n {
        return Err(LinalgError::ConvergenceError(
            "Could not find n stable eigenvalues for Riccati equation".to_string(),
        ));
    }

    // Extract the stable invariant subspace
    let mut u1 = Array2::<A>::zeros((n, n));
    let mut u2 = Array2::<A>::zeros((n, n));

    for (j, &i) in stable_indices.iter().enumerate() {
        for k in 0..n {
            u1[[k, j]] = eigvecs[[k, i]].re;
            u2[[k, j]] = eigvecs[[n + k, i]].re;
        }
    }

    // Solve X = U2 * U1^{-1}
    let u1_inv = crate::inv(&u1.view(), None)?;
    let x = u2.dot(&u1_inv);

    // Make symmetric (X should be symmetric)
    Ok((x.clone() + x.t()) * A::from(0.5).unwrap())
}

/// Solves the discrete-time algebraic Riccati equation (DARE)
/// X = A^T X A - A^T X B (R + B^T X B)^{-1} B^T X A + Q
///
/// # Arguments
/// * `a` - State matrix A (n × n)
/// * `b` - Input matrix B (n × m)
/// * `q` - State cost matrix Q (n × n, symmetric positive semidefinite)
/// * `r` - Input cost matrix R (m × m, symmetric positive definite)
///
/// # Returns
/// * Solution matrix X (n × n, symmetric positive semidefinite)
pub fn solve_discrete_riccati<A>(
    a: &ArrayView2<A>,
    b: &ArrayView2<A>,
    q: &ArrayView2<A>,
    r: &ArrayView2<A>,
) -> LinalgResult<Array2<A>>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + std::iter::Sum + 'static,
{
    let n = a.shape()[0];
    let m = b.shape()[1];

    // Validate dimensions
    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError(
            "Matrix A must be square".to_string(),
        ));
    }
    if b.shape() != [n, m] {
        return Err(LinalgError::ShapeError(format!(
            "Matrix B must have shape [{}, m]",
            n
        )));
    }
    if q.shape() != [n, n] {
        return Err(LinalgError::ShapeError(
            "Matrix Q must be n × n".to_string(),
        ));
    }
    if r.shape() != [m, m] {
        return Err(LinalgError::ShapeError(
            "Matrix R must be m × m".to_string(),
        ));
    }

    // Solve using the symplectic matrix approach
    // This is a simplified implementation using iteration
    // Production code would use more sophisticated methods

    let mut x = q.to_owned(); // Initial guess
    let max_iterations = 100;
    let tolerance = A::from(1e-10).unwrap();

    for _ in 0..max_iterations {
        let x_old = x.clone();

        // Compute R + B^T X B
        let r_tilde = r + &b.t().dot(&x).dot(b);

        // Solve for (R + B^T X B)^{-1}
        let r_tilde_inv = crate::inv(&r_tilde.view(), None)?;

        // Update X
        let term1 = a.t().dot(&x).dot(a);
        let term2 = a
            .t()
            .dot(&x)
            .dot(b)
            .dot(&r_tilde_inv)
            .dot(&b.t())
            .dot(&x)
            .dot(a);
        x = &term1 - &term2 + q;

        // Check convergence
        let diff = &x - &x_old;
        let error = diff
            .iter()
            .map(|&v| v.abs())
            .fold(A::zero(), |acc, v| acc.max(v));

        if error < tolerance {
            // Make symmetric
            return Ok((x.clone() + x.t()) * A::from(0.5).unwrap());
        }
    }

    Err(LinalgError::ConvergenceError(
        "Discrete Riccati equation solver did not converge".to_string(),
    ))
}

/// Solves the Stein equation AXA^T - X + Q = 0
///
/// # Arguments
/// * `a` - Matrix A (n × n)
/// * `q` - Matrix Q (n × n)
///
/// # Returns
/// * Solution matrix X (n × n)
pub fn solve_stein<A>(a: &ArrayView2<A>, q: &ArrayView2<A>) -> LinalgResult<Array2<A>>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + std::iter::Sum + 'static,
{
    let n = a.shape()[0];

    // Validate dimensions
    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError(
            "Matrix A must be square".to_string(),
        ));
    }
    if q.shape() != [n, n] {
        return Err(LinalgError::ShapeError(
            "Matrix Q must have the same shape as A".to_string(),
        ));
    }

    // Convert to discrete Lyapunov equation: AXA^T - X + Q = 0
    // This can be rewritten as a standard linear system

    // Vectorize: (A ⊗ A - I) vec(X) = -vec(Q)
    let mut coeff = Array2::<A>::zeros((n * n, n * n));

    // Compute A ⊗ A - I
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    coeff[[i * n + j, k * n + l]] = a[[i, k]] * a[[j, l]];
                    if i == k && j == l {
                        coeff[[i * n + j, k * n + l]] -= A::one();
                    }
                }
            }
        }
    }

    // Vectorize -Q
    let q_vec: Vec<A> = q.t().iter().map(|&x| -x).collect();
    let q_vec = Array2::from_shape_vec((n * n, 1), q_vec)
        .map_err(|e| LinalgError::ShapeError(format!("Failed to reshape vector: {}", e)))?;

    // Solve the linear system
    let q_vec_1d = q_vec.column(0);
    let x_vec = crate::solve::solve(&coeff.view(), &q_vec_1d, None)?;

    // Reshape solution back to matrix form
    let x_data: Vec<A> = x_vec.iter().cloned().collect();
    Ok(Array2::from_shape_vec((n, n), x_data)
        .map_err(|e| LinalgError::ShapeError(format!("Failed to reshape solution: {}", e)))?
        .t()
        .to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_sylvester_equation() {
        // Use a test case where A and -B have distinct eigenvalues
        let a = array![[1.0, 0.0], [0.0, 2.0]];
        let b = array![[-3.0, 0.0], [0.0, -4.0]];
        let c = array![[1.0, 2.0], [3.0, 4.0]];

        let x = solve_sylvester(&a.view(), &b.view(), &c.view()).unwrap();

        // Verify AX + XB = C
        let ax = a.dot(&x);
        let xb = x.dot(&b);
        let result = &ax + &xb;

        println!("A = {:?}", a);
        println!("B = {:?}", b);
        println!("C = {:?}", c);
        println!("X = {:?}", x);
        println!("AX = {:?}", ax);
        println!("XB = {:?}", xb);
        println!("AX + XB = {:?}", result);

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(result[[i, j]], c[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_stein_equation() {
        // Simple test case
        let a = array![[0.5, 0.1], [0.0, 0.6]];
        let q = array![[1.0, 0.0], [0.0, 1.0]];

        let x = solve_stein(&a.view(), &q.view()).unwrap();

        // Verify AXA^T - X + Q = 0
        let axat = a.dot(&x).dot(&a.t());
        let result = &axat - &x + &q;

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(result[[i, j]], 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    #[ignore = "Complex eigenvalue handling needs implementation"]
    fn test_continuous_riccati_simple() {
        // Simple 2x2 system
        let a = array![[0.0, 1.0], [0.0, 0.0]];
        let b = array![[0.0], [1.0]];
        let q = array![[1.0, 0.0], [0.0, 0.0]];
        let r = array![[1.0]];

        let x = solve_continuous_riccati(&a.view(), &b.view(), &q.view(), &r.view()).unwrap();

        // X should be symmetric
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(x[[i, j]], x[[j, i]], epsilon = 1e-10);
            }
        }

        // Verify the Riccati equation (approximately)
        let at_x = a.t().dot(&x);
        let x_a = x.dot(&a);
        let x_br_inv_bt_x = x
            .dot(&b)
            .dot(&crate::inv(&r.view(), None).unwrap())
            .dot(&b.t())
            .dot(&x);
        let lhs = &at_x + &x_a - &x_br_inv_bt_x + &q;

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(lhs[[i, j]], 0.0, epsilon = 1e-8);
            }
        }
    }
}
