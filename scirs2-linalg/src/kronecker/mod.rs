//! Kronecker product optimizations for neural network layers
//!
//! This module provides optimized implementations of the Kronecker product and related
//! operations that are particularly useful for neural network layers, especially in
//! second-order optimization methods like K-FAC (Kronecker-Factored Approximate Curvature).

use ndarray::{Array2, ArrayView2, ScalarOperand};
use num_traits::{Float, NumAssign};
use std::iter::Sum;

// Helper function to convert ndarray::ShapeError to LinalgError
fn shape_err_to_linalg(err: ndarray::ShapeError) -> crate::error::LinalgError {
    crate::error::LinalgError::ShapeError(err.to_string())
}

use crate::error::{LinalgError, LinalgResult};

/// Compute the Kronecker product of two matrices
///
/// The Kronecker product of matrices A (m×n) and B (p×q) is a matrix C (mp×nq) where
/// C[i*p+k, j*q+l] = A[i,j] * B[k,l]
///
/// This is particularly useful in neural network contexts for structured weight matrices
/// in layers and for creating covariance factor approximations in second-order optimizers.
///
/// # Arguments
///
/// * `a` - First matrix of shape (m, n)
/// * `b` - Second matrix of shape (p, q)
///
/// # Returns
///
/// * Kronecker product matrix of shape (m*p, n*q)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::kronecker::kron;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let b = array![[0.1, 0.2], [0.3, 0.4]];
///
/// let c = kron(&a.view(), &b.view()).unwrap();
///
/// // The result should be a 4x4 matrix:
/// // [[0.1, 0.2, 0.2, 0.4],
/// //  [0.3, 0.4, 0.6, 0.8],
/// //  [0.3, 0.6, 0.4, 0.8],
/// //  [0.9, 1.2, 1.2, 1.6]]
/// ```
pub fn kron<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum,
{
    let (m, n) = a.dim();
    let (p, q) = b.dim();

    // Create output array
    let mut result = Array2::zeros((m * p, n * q));

    // Compute Kronecker product
    for i in 0..m {
        for j in 0..n {
            for k in 0..p {
                for l in 0..q {
                    result[[i * p + k, j * q + l]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }

    Ok(result)
}

/// Perform a matrix-vector multiplication using Kronecker-structured matrices
///
/// For large matrices that can be represented as Kronecker products,
/// this function efficiently computes y = (A ⊗ B) * x without forming the
/// full Kronecker product explicitly.
///
/// # Arguments
///
/// * `a` - First matrix factor of shape (m, n)
/// * `b` - Second matrix factor of shape (p, q)
/// * `x` - Vector of shape (n*q)
///
/// # Returns
///
/// * Result vector of shape (m*p)
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array1};
/// use scirs2_linalg::kronecker::kron_matvec;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let b = array![[0.1, 0.2], [0.3, 0.4]];
/// let x = array![1.0, 2.0, 3.0, 4.0];
///
/// let y = kron_matvec(&a.view(), &b.view(), &x.view()).unwrap();
///
/// // This should equal (A ⊗ B) * x
/// ```
pub fn kron_matvec<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    x: &ndarray::ArrayView1<F>,
) -> LinalgResult<ndarray::Array1<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand,
{
    let (m, n) = a.dim();
    let (p, q) = b.dim();

    // Check dimensions
    if x.len() != n * q {
        return Err(LinalgError::ShapeError(format!(
            "Vector length ({}) must equal n*q ({}*{}={})",
            x.len(),
            n,
            q,
            n * q
        )));
    }

    // Reshape x to a matrix of shape (n, q)
    let x_mat = x
        .to_owned()
        .into_shape_with_order((n, q))
        .map_err(shape_err_to_linalg)?;

    // Compute (B * X^T)^T = X * B^T
    let tmp = x_mat.dot(&b.t());

    // Compute A * (X * B^T)
    let tmp_reshaped = tmp
        .into_shape_with_order(n * p)
        .map_err(shape_err_to_linalg)?;

    // Use matrixmultiply here if needed for better performance
    let result = a.dot(
        &tmp_reshaped
            .into_shape_with_order((n, p))
            .map_err(shape_err_to_linalg)?,
    );

    // Reshape to a vector
    let result_vec = result
        .into_shape_with_order(m * p)
        .map_err(shape_err_to_linalg)?;

    Ok(result_vec)
}

/// Efficient implementation of matrix multiplication when one matrix has Kronecker structure
///
/// Computes matrix multiplication Y = (A ⊗ B) * X without forming the full
/// Kronecker product explicitly, which is much more efficient for large matrices.
///
/// # Arguments
///
/// * `a` - First matrix factor of shape (m, n)
/// * `b` - Second matrix factor of shape (p, q)
/// * `x` - Matrix to multiply of shape (n*q, r)
///
/// # Returns
///
/// * Result matrix of shape (m*p, r)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::kronecker::kron_matmul;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let b = array![[0.1, 0.2], [0.3, 0.4]];
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
///
/// let y = kron_matmul(&a.view(), &b.view(), &x.view()).unwrap();
///
/// // This should equal (A ⊗ B) * X
/// ```
pub fn kron_matmul<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    x: &ArrayView2<F>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand,
{
    let (m, n) = a.dim();
    let (p, q) = b.dim();
    let (nx, r) = x.dim();

    // Check dimensions
    if nx != n * q {
        return Err(LinalgError::ShapeError(format!(
            "First dimension of X ({}) must equal n*q ({}*{}={})",
            nx,
            n,
            q,
            n * q
        )));
    }

    // Initialize result matrix
    let mut result = Array2::zeros((m * p, r));

    // Process each column of X separately
    for col in 0..r {
        let x_col = x.slice(ndarray::s![.., col]);
        let y_col = kron_matvec(a, b, &x_col)?;

        // Copy the result to the output matrix
        for i in 0..m * p {
            result[[i, col]] = y_col[i];
        }
    }

    Ok(result)
}

/// Compute a Kronecker-factored approximation of a matrix
///
/// Given a matrix M of size (m*p, n*q), computes matrices A and B such that
/// A ⊗ B approximates M. This is useful for compressing large matrices in
/// neural networks, especially for approximating the Fisher Information Matrix
/// in second-order optimization methods.
///
/// # Arguments
///
/// * `m` - Matrix to approximate of shape (m*p, n*q)
/// * `m_rows` - Number of rows in the first factor (m)
/// * `n_cols` - Number of columns in the first factor (n)
///
/// # Returns
///
/// * Tuple (A, B) where A is of shape (m, n) and B is of shape (p, q),
///   and their Kronecker product approximates M
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::kronecker::{kron, kron_factorize};
///
/// // Create a matrix to factorize
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let b = array![[0.1, 0.2], [0.3, 0.4]];
/// let ab = kron(&a.view(), &b.view()).unwrap();
///
/// // Add some noise
/// let mut noisy_ab = ab.clone();
/// for i in 0..4 {
///     for j in 0..4 {
///         noisy_ab[[i, j]] += 0.01;
///     }
/// }
///
/// // Factorize back to get approximations of A and B
/// let (a_approx, b_approx) = kron_factorize(&noisy_ab.view(), 2, 2).unwrap();
///
/// // a_approx and b_approx should be close to the original a and b
/// ```
pub fn kron_factorize<F>(
    m: &ArrayView2<F>,
    m_rows: usize,
    n_cols: usize,
) -> LinalgResult<(Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + ScalarOperand,
{
    let (total_rows, total_cols) = m.dim();

    // Calculate dimensions of the factors
    let p_rows = total_rows / m_rows;
    let q_cols = total_cols / n_cols;

    // Check if the matrix can be factorized with the given dimensions
    if m_rows * p_rows != total_rows || n_cols * q_cols != total_cols {
        return Err(LinalgError::ShapeError(format!(
            "Matrix of shape ({}, {}) cannot be factorized into ({}, {}) and ({}, {})",
            total_rows, total_cols, m_rows, n_cols, p_rows, q_cols
        )));
    }

    // Reshape the matrix to a 4D tensor
    let m_reshaped = (*m)
        .into_shape_with_order((m_rows, p_rows, n_cols, q_cols))
        .map_err(|_| {
            LinalgError::ShapeError(
                "Failed to reshape matrix for Kronecker factorization".to_string(),
            )
        })?;

    let m_tensor = m_reshaped;

    // Compute factor A
    let mut a = Array2::zeros((m_rows, n_cols));
    let mut b = Array2::zeros((p_rows, q_cols));

    // Use Higher Order SVD (HOSVD) approach for factorization
    // First, compute the average of M across the p_rows and q_cols dimensions
    for i in 0..m_rows {
        for j in 0..n_cols {
            let mut sum = F::zero();
            let mut count = F::zero();

            for k in 0..p_rows {
                for l in 0..q_cols {
                    sum += m_tensor[[i, k, j, l]];
                    count += F::one();
                }
            }

            a[[i, j]] = sum / count;
        }
    }

    // Normalize A to have unit Frobenius norm
    let a_norm = a.iter().map(|&x| x * x).sum::<F>().sqrt();
    if a_norm > F::epsilon() {
        for i in 0..m_rows {
            for j in 0..n_cols {
                a[[i, j]] /= a_norm;
            }
        }
    }

    // Now compute B to minimize ||M - A ⊗ B||_F
    for k in 0..p_rows {
        for l in 0..q_cols {
            let mut sum = F::zero();

            for i in 0..m_rows {
                for j in 0..n_cols {
                    sum += m_tensor[[i, k, j, l]] * a[[i, j]];
                }
            }

            b[[k, l]] = sum;
        }
    }

    // Scale the factors appropriately
    let scaling_factor = a_norm;

    for i in 0..m_rows {
        for j in 0..n_cols {
            a[[i, j]] *= scaling_factor.sqrt();
        }
    }

    for k in 0..p_rows {
        for l in 0..q_cols {
            b[[k, l]] /= scaling_factor.sqrt();
        }
    }

    Ok((a, b))
}

/// Compute a regularized Kronecker factorization for Fisher Information Matrix approximation
///
/// This method implements the Kronecker-Factored Approximate Curvature (K-FAC) approach
/// for approximating the Fisher Information Matrix (FIM) in neural networks, which is
/// used in second-order optimization methods. It returns a regularized Kronecker factorization
/// where the factors represent input activations and output gradients covariances.
///
/// # Arguments
///
/// * `input_acts` - Input activations matrix of shape (batch_size, input_dim)
/// * `output_grads` - Output gradients matrix of shape (batch_size, output_dim)
/// * `damping` - Damping factor for regularization (default: 1e-4)
///
/// # Returns
///
/// * Tuple (A, B) where A is the input activations covariance and B is the output
///   gradients covariance, such that their Kronecker product approximates the FIM
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::kronecker::kfac_factorization;
///
/// // Create sample activations and gradients
/// let input_acts = array![
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0],
/// ];
///
/// let output_grads = array![
///     [0.1, 0.2],
///     [0.3, 0.4],
///     [0.5, 0.6],
/// ];
///
/// // Compute the Kronecker factors that approximate the Fisher Information Matrix
/// let (a_cov, s_cov) = kfac_factorization(&input_acts.view(), &output_grads.view(), None).unwrap();
///
/// // The Kronecker product a_cov ⊗ s_cov approximates the Fisher Information Matrix
/// ```
pub fn kfac_factorization<F>(
    input_acts: &ArrayView2<F>,
    output_grads: &ArrayView2<F>,
    damping: Option<F>,
) -> LinalgResult<(Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + ScalarOperand,
{
    let (batch_size1, input_dim) = input_acts.dim();
    let (batch_size2, output_dim) = output_grads.dim();

    // Check dimensions
    if batch_size1 != batch_size2 {
        return Err(LinalgError::ShapeError(format!(
            "Batch sizes must match: {} vs {}",
            batch_size1, batch_size2
        )));
    }

    let batch_size = batch_size1;
    let damping_factor = damping.unwrap_or_else(|| F::from(1e-4).unwrap());

    // Append 1s to input activations for bias term
    let mut input_acts_with_bias = Array2::zeros((batch_size, input_dim + 1));
    for i in 0..batch_size {
        for j in 0..input_dim {
            input_acts_with_bias[[i, j]] = input_acts[[i, j]];
        }
        input_acts_with_bias[[i, input_dim]] = F::one();
    }

    // Compute input-input covariance (A)
    let mut a_cov = Array2::zeros((input_dim + 1, input_dim + 1));

    for i in 0..(input_dim + 1) {
        for j in 0..(input_dim + 1) {
            let mut sum = F::zero();

            for b in 0..batch_size {
                sum += input_acts_with_bias[[b, i]] * input_acts_with_bias[[b, j]];
            }

            a_cov[[i, j]] = sum / F::from(batch_size).unwrap();
        }
    }

    // Add damping to the diagonal of A
    for i in 0..(input_dim + 1) {
        a_cov[[i, i]] += damping_factor;
    }

    // Compute output-output covariance (S)
    let mut s_cov = Array2::zeros((output_dim, output_dim));

    for i in 0..output_dim {
        for j in 0..output_dim {
            let mut sum = F::zero();

            for b in 0..batch_size {
                sum += output_grads[[b, i]] * output_grads[[b, j]];
            }

            s_cov[[i, j]] = sum / F::from(batch_size).unwrap();
        }
    }

    // Add damping to the diagonal of S
    for i in 0..output_dim {
        s_cov[[i, i]] += damping_factor;
    }

    Ok((a_cov, s_cov))
}

/// Perform a natural gradient update using Kronecker factorization
///
/// This function implements a natural gradient update step using Kronecker-Factored
/// Approximate Curvature (K-FAC) for neural network optimization. Given the current
/// weight matrix, its gradients, and the Kronecker factors of the Fisher Information
/// Matrix, it computes the natural gradient update.
///
/// # Arguments
///
/// * `weights` - Current weight matrix of shape (input_dim, output_dim)
/// * `gradients` - Gradient matrix of shape (input_dim, output_dim)
/// * `a_inv` - Inverse of input covariance factor from KFAC
/// * `s_inv` - Inverse of output covariance factor from KFAC
/// * `learning_rate` - Learning rate for the update
///
/// # Returns
///
/// * Updated weights after the natural gradient step
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::kronecker::{kfac_factorization, kfac_update};
/// use scirs2_linalg::inv;
///
/// // Current weights and gradients including bias (input_dim+1 x output_dim)
/// let weights = array![[0.1, 0.2], [0.3, 0.4], [0.05, 0.1]];  // Including bias row
/// let gradients = array![[0.01, 0.02], [0.03, 0.04], [0.005, 0.01]];  // Including bias gradients
///
/// // Input activations (batch_size x input_dim) and output gradients (batch_size x output_dim)
/// let input_acts = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let output_grads = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
///
/// // Compute KFAC factors
/// let (a_cov, s_cov) = kfac_factorization(&input_acts.view(), &output_grads.view(), None).unwrap();
///
/// // Note: a_cov has shape (input_dim+1, input_dim+1) due to bias term
/// // s_cov has shape (output_dim, output_dim)
/// let a_inv = inv(&a_cov.view()).unwrap();
/// let s_inv = inv(&s_cov.view()).unwrap();
///
/// // Perform natural gradient update
/// let new_weights = kfac_update(
///     &weights.view(),
///     &gradients.view(),
///     &a_inv.view(),
///     &s_inv.view(),
///     0.01
/// ).unwrap();
/// ```
pub fn kfac_update<F>(
    weights: &ArrayView2<F>,
    gradients: &ArrayView2<F>,
    a_inv: &ArrayView2<F>,
    s_inv: &ArrayView2<F>,
    learning_rate: F,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand,
{
    let (input_dim, output_dim) = weights.dim();
    let (grad_rows, grad_cols) = gradients.dim();

    // Check dimensions
    if input_dim != grad_rows || output_dim != grad_cols {
        return Err(LinalgError::ShapeError(format!(
            "Weights ({}, {}) and gradients ({}, {}) must have the same shape",
            input_dim, output_dim, grad_rows, grad_cols
        )));
    }

    if a_inv.dim().0 != input_dim || a_inv.dim().1 != input_dim {
        return Err(LinalgError::ShapeError(format!(
            "A inverse shape ({}, {}) must match input dimension {}",
            a_inv.dim().0,
            a_inv.dim().1,
            input_dim
        )));
    }

    if s_inv.dim().0 != output_dim || s_inv.dim().1 != output_dim {
        return Err(LinalgError::ShapeError(format!(
            "S inverse shape ({}, {}) must match output dimension {}",
            s_inv.dim().0,
            s_inv.dim().1,
            output_dim
        )));
    }

    // Compute natural gradient update
    // natural_grad = (A^-1 * gradients * S^-1)
    let gradients_owned = gradients.to_owned();
    let s_inv_owned = s_inv.to_owned();
    let tmp = a_inv.to_owned().dot(&gradients_owned);
    let natural_grad = tmp.dot(&s_inv_owned);

    // Update weights: w = w - lr * natural_grad
    let mut new_weights = weights.to_owned();

    for i in 0..input_dim {
        for j in 0..output_dim {
            new_weights[[i, j]] = weights[[i, j]] - learning_rate * natural_grad[[i, j]];
        }
    }

    Ok(new_weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_kron_simple() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[0.1, 0.2], [0.3, 0.4]];

        let c = kron(&a.view(), &b.view()).unwrap();

        // Check dimensions
        assert_eq!(c.shape(), &[4, 4]);

        // Check specific values
        assert_relative_eq!(c[[0, 0]], 0.1);
        assert_relative_eq!(c[[0, 1]], 0.2);
        assert_relative_eq!(c[[0, 2]], 0.2);
        assert_relative_eq!(c[[0, 3]], 0.4);

        assert_relative_eq!(c[[1, 0]], 0.3);
        assert_relative_eq!(c[[1, 1]], 0.4);
        assert_relative_eq!(c[[1, 2]], 0.6);
        assert_relative_eq!(c[[1, 3]], 0.8);

        assert_relative_eq!(c[[2, 0]], 0.3);
        assert_relative_eq!(c[[2, 1]], 0.6);
        assert_relative_eq!(c[[2, 2]], 0.4);
        assert_relative_eq!(c[[2, 3]], 0.8);

        assert_relative_eq!(c[[3, 0]], 0.9);
        assert_relative_eq!(c[[3, 1]], 1.2);
        assert_relative_eq!(c[[3, 2]], 1.2);
        assert_relative_eq!(c[[3, 3]], 1.6);
    }

    #[test]
    fn test_kron_matvec() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[0.1, 0.2], [0.3, 0.4]];
        let x = array![1.0, 2.0, 3.0, 4.0];

        // Compute (A ⊗ B) * x using our optimized function
        let y = kron_matvec(&a.view(), &b.view(), &x.view()).unwrap();

        // Compute the same thing using explicit Kronecker product
        let ab = kron(&a.view(), &b.view()).unwrap();
        let y_direct = ab.dot(&x);

        // Check dimensions
        assert_eq!(y.shape(), y_direct.shape());

        // Check values
        for i in 0..y.len() {
            assert_relative_eq!(y[i], y_direct[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_kron_matmul() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[0.1, 0.2], [0.3, 0.4]];
        let x = array![[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]];

        // Compute (A ⊗ B) * X using our optimized function
        let y = kron_matmul(&a.view(), &b.view(), &x.view()).unwrap();

        // Compute the same thing using explicit Kronecker product
        let ab = kron(&a.view(), &b.view()).unwrap();
        let y_direct = ab.dot(&x);

        // Check dimensions
        assert_eq!(y.shape(), y_direct.shape());

        // Check values
        for i in 0..y.dim().0 {
            for j in 0..y.dim().1 {
                assert_relative_eq!(y[[i, j]], y_direct[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_kron_factorize() {
        // Create a Kronecker product
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[0.1, 0.2], [0.3, 0.4]];
        let ab = kron(&a.view(), &b.view()).unwrap();

        // Factorize it back
        let (a_hat, b_hat) = kron_factorize(&ab.view(), 2, 2).unwrap();

        // Recompute the Kronecker product using the factors
        let ab_hat = kron(&a_hat.view(), &b_hat.view()).unwrap();

        // Check that the approximation is close to the original
        // (allowing for some scale differences)
        let mut error = 0.0f64;
        for i in 0..4 {
            for j in 0..4 {
                error += (ab[[i, j]] - ab_hat[[i, j]]).abs() as f64;
            }
        }
        error /= 16.0;

        assert!(error < 0.1, "Average error was {}", error);
    }

    #[test]
    fn test_kfac_factorization() {
        // Create sample input activations and output gradients
        let input_acts = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];

        let output_grads = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6],];

        // Compute KFAC factorization
        let (a_cov, s_cov) =
            kfac_factorization(&input_acts.view(), &output_grads.view(), Some(0.01)).unwrap();

        // Check dimensions
        assert_eq!(a_cov.shape(), &[4, 4]); // +1 for bias
        assert_eq!(s_cov.shape(), &[2, 2]);

        // Check that the diagonals have damping added
        for i in 0..4 {
            assert!(a_cov[[i, i]] > 0.0);
        }

        for i in 0..2 {
            assert!(s_cov[[i, i]] > 0.0);
        }
    }

    #[test]
    fn test_kfac_update() {
        // Create sample weights and gradients
        let weights = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6],];

        let gradients = array![[0.01, 0.02], [0.03, 0.04], [0.05, 0.06],];

        // Create identity matrices for A^-1 and S^-1 (simplifies testing)
        let a_inv = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],];

        let s_inv = array![[1.0, 0.0], [0.0, 1.0],];

        // Compute update
        let learning_rate = 0.1;
        let new_weights = kfac_update(
            &weights.view(),
            &gradients.view(),
            &a_inv.view(),
            &s_inv.view(),
            learning_rate,
        )
        .unwrap();

        // With identity matrices, the natural gradient equals the regular gradient
        // So we can check against a simple SGD update
        for i in 0..3 {
            for j in 0..2 {
                assert_relative_eq!(
                    new_weights[[i, j]],
                    weights[[i, j]] - learning_rate * gradients[[i, j]],
                    epsilon = 1e-10
                );
            }
        }
    }
}
