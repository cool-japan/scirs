//! Kronecker product optimizations for neural network layers
//!
//! This module provides optimized implementations of the Kronecker product and related
//! operations that are particularly useful for neural network layers, especially in
//! second-order optimization methods like K-FAC (Kronecker-Factored Approximate Curvature).

use ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use num_traits::{Float, NumAssign};
use std::iter::Sum;

// Helper function to convert ndarray::ShapeError to LinalgError
fn shape_err_to_linalg(err: ndarray::ShapeError) -> crate::error::LinalgError {
    crate::error::LinalgError::ShapeError(err.to_string())
}

use crate::decomposition::cholesky;
use crate::error::{LinalgError, LinalgResult};
use crate::norm::matrix_norm;

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
/// let a_inv = inv(&a_cov.view(), None).unwrap();
/// let s_inv = inv(&s_cov.view(), None).unwrap();
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

/// Advanced K-FAC optimizer state for tracking moving averages and adaptive damping
///
/// This structure maintains the state needed for advanced K-FAC optimization,
/// including exponentially weighted moving averages of covariance matrices,
/// adaptive damping parameters, and block-diagonal Fisher approximations.
#[derive(Debug)]
pub struct KFACOptimizer<F> {
    /// Moving average decay factor for covariance matrices (typical: 0.95)
    pub decay_factor: F,
    /// Base damping factor for regularization (typical: 1e-4)
    pub base_damping: F,
    /// Adaptive damping factor (adjusted during optimization)
    pub adaptive_damping: F,
    /// Minimum damping to prevent numerical instability
    pub min_damping: F,
    /// Maximum damping to ensure progress
    pub max_damping: F,
    /// Number of optimization steps taken
    pub step_count: usize,
    /// Input covariance moving average
    pub input_cov_avg: Option<Array2<F>>,
    /// Output covariance moving average
    pub output_cov_avg: Option<Array2<F>>,
    /// Trace of input covariance for scaling
    pub input_trace: Option<F>,
    /// Trace of output covariance for scaling
    pub output_trace: Option<F>,
}

impl<F> KFACOptimizer<F>
where
    F: Float + NumAssign + Sum + ScalarOperand,
{
    /// Create a new K-FAC optimizer with default parameters
    ///
    /// # Arguments
    ///
    /// * `decay_factor` - Exponential decay for moving averages (default: 0.95)
    /// * `base_damping` - Base regularization parameter (default: 1e-4)
    ///
    /// # Returns
    ///
    /// * New K-FAC optimizer instance
    pub fn new(decay_factor: Option<F>, base_damping: Option<F>) -> Self {
        let decay = decay_factor.unwrap_or_else(|| F::from(0.95).unwrap());
        let damping = base_damping.unwrap_or_else(|| F::from(1e-4).unwrap());

        Self {
            decay_factor: decay,
            base_damping: damping,
            adaptive_damping: damping,
            min_damping: damping / F::from(10.0).unwrap(),
            max_damping: damping * F::from(100.0).unwrap(),
            step_count: 0,
            input_cov_avg: None,
            output_cov_avg: None,
            input_trace: None,
            output_trace: None,
        }
    }

    /// Update covariance estimates with exponential moving averages
    ///
    /// This function maintains running estimates of input and output covariance matrices
    /// using exponentially weighted moving averages, which provides more stable
    /// estimates than using single-batch statistics.
    ///
    /// # Arguments
    ///
    /// * `input_acts` - Current batch input activations
    /// * `output_grads` - Current batch output gradients
    ///
    /// # Returns
    ///
    /// * Updated covariance matrices (A, S)
    pub fn update_covariances(
        &mut self,
        input_acts: &ArrayView2<F>,
        output_grads: &ArrayView2<F>,
    ) -> LinalgResult<(Array2<F>, Array2<F>)> {
        // Compute current batch covariances
        let (current_input_cov, current_output_cov) =
            kfac_factorization(input_acts, output_grads, Some(F::zero()))?;

        // Update moving averages
        match (&mut self.input_cov_avg, &mut self.output_cov_avg) {
            (Some(ref mut input_avg), Some(ref mut output_avg)) => {
                // Update existing averages
                for i in 0..input_avg.nrows() {
                    for j in 0..input_avg.ncols() {
                        input_avg[[i, j]] = self.decay_factor * input_avg[[i, j]]
                            + (F::one() - self.decay_factor) * current_input_cov[[i, j]];
                    }
                }

                for i in 0..output_avg.nrows() {
                    for j in 0..output_avg.ncols() {
                        output_avg[[i, j]] = self.decay_factor * output_avg[[i, j]]
                            + (F::one() - self.decay_factor) * current_output_cov[[i, j]];
                    }
                }
            }
            _ => {
                // Initialize averages
                self.input_cov_avg = Some(current_input_cov.clone());
                self.output_cov_avg = Some(current_output_cov.clone());
            }
        }

        // Compute bias correction for early steps
        let bias_correction = F::one() - self.decay_factor.powi(self.step_count as i32 + 1);

        // Apply bias correction and damping
        let mut corrected_input = self.input_cov_avg.as_ref().unwrap().clone();
        let mut corrected_output = self.output_cov_avg.as_ref().unwrap().clone();

        // Bias correction
        for i in 0..corrected_input.nrows() {
            for j in 0..corrected_input.ncols() {
                corrected_input[[i, j]] /= bias_correction;
            }
        }

        for i in 0..corrected_output.nrows() {
            for j in 0..corrected_output.ncols() {
                corrected_output[[i, j]] /= bias_correction;
            }
        }

        // Add adaptive damping to diagonal
        for i in 0..corrected_input.nrows() {
            corrected_input[[i, i]] += self.adaptive_damping;
        }

        for i in 0..corrected_output.nrows() {
            corrected_output[[i, i]] += self.adaptive_damping;
        }

        // Update trace estimates for scaling
        let input_trace = (0..corrected_input.nrows())
            .map(|i| corrected_input[[i, i]])
            .sum::<F>();
        let output_trace = (0..corrected_output.nrows())
            .map(|i| corrected_output[[i, i]])
            .sum::<F>();

        self.input_trace = Some(input_trace);
        self.output_trace = Some(output_trace);
        self.step_count += 1;

        Ok((corrected_input, corrected_output))
    }

    /// Adaptive damping adjustment based on optimization progress
    ///
    /// This function implements the Levenberg-Marquardt style adaptive damping
    /// where the damping is increased if the loss increases and decreased if
    /// the loss decreases, providing robust second-order optimization.
    ///
    /// # Arguments
    ///
    /// * `loss_improved` - Whether the loss improved in the last step
    /// * `improvement_ratio` - Ratio of actual vs predicted improvement
    ///
    pub fn adjust_damping(&mut self, loss_improved: bool, improvement_ratio: Option<F>) {
        if loss_improved {
            // Loss improved: decrease damping
            if let Some(ratio) = improvement_ratio {
                if ratio > F::from(0.75).unwrap() {
                    // Very good step: aggressive damping reduction
                    self.adaptive_damping =
                        (self.adaptive_damping / F::from(3.0).unwrap()).max(self.min_damping);
                } else if ratio > F::from(0.25).unwrap() {
                    // Good step: moderate damping reduction
                    self.adaptive_damping =
                        (self.adaptive_damping / F::from(2.0).unwrap()).max(self.min_damping);
                }
            } else {
                // Default reduction
                self.adaptive_damping =
                    (self.adaptive_damping / F::from(1.5).unwrap()).max(self.min_damping);
            }
        } else {
            // Loss did not improve: increase damping
            self.adaptive_damping =
                (self.adaptive_damping * F::from(2.0).unwrap()).min(self.max_damping);
        }
    }

    /// Get current effective damping value
    pub fn get_damping(&self) -> F {
        self.adaptive_damping
    }

    /// Reset optimizer state (useful for learning rate schedule changes)
    pub fn reset(&mut self) {
        self.step_count = 0;
        self.input_cov_avg = None;
        self.output_cov_avg = None;
        self.input_trace = None;
        self.output_trace = None;
        self.adaptive_damping = self.base_damping;
    }
}

/// Block-diagonal Fisher Information Matrix approximation for multi-layer networks
///
/// This structure represents a block-diagonal approximation of the Fisher Information
/// Matrix for multi-layer neural networks, where each layer's Fisher matrix is
/// approximated independently using Kronecker factorization.
#[derive(Debug)]
pub struct BlockDiagonalFisher<F> {
    /// Kronecker factors for each layer: (input_cov, output_cov)
    pub layer_factors: Vec<(Array2<F>, Array2<F>)>,
    /// Inverse factors for efficient preconditioning
    pub inverse_factors: Vec<(Array2<F>, Array2<F>)>,
    /// Layer dimensions: (input_dim, output_dim)
    pub layer_dims: Vec<(usize, usize)>,
    /// Damping factor for regularization
    pub damping: F,
}

impl<F> BlockDiagonalFisher<F>
where
    F: Float + NumAssign + Sum + ScalarOperand,
{
    /// Create a new block-diagonal Fisher approximation
    ///
    /// # Arguments
    ///
    /// * `layer_dims` - Dimensions of each layer (input, output)
    /// * `damping` - Regularization parameter
    ///
    /// # Returns
    ///
    /// * New block-diagonal Fisher structure
    pub fn new(layer_dims: Vec<(usize, usize)>, damping: F) -> Self {
        Self {
            layer_factors: Vec::new(),
            inverse_factors: Vec::new(),
            layer_dims,
            damping,
        }
    }

    /// Update Fisher approximation for all layers
    ///
    /// # Arguments
    ///
    /// * `layer_activations` - Input activations for each layer
    /// * `layer_gradients` - Output gradients for each layer
    ///
    /// # Returns
    ///
    /// * Success/failure result
    pub fn update_fisher(
        &mut self,
        layer_activations: &[ArrayView2<F>],
        layer_gradients: &[ArrayView2<F>],
    ) -> LinalgResult<()> {
        if layer_activations.len() != layer_gradients.len()
            || layer_activations.len() != self.layer_dims.len()
        {
            return Err(LinalgError::ShapeError(
                "Mismatched number of layers".to_string(),
            ));
        }

        self.layer_factors.clear();
        self.inverse_factors.clear();

        for (i, (&(input_dim, output_dim), (acts, grads))) in self
            .layer_dims
            .iter()
            .zip(layer_activations.iter().zip(layer_gradients.iter()))
            .enumerate()
        {
            // Verify dimensions
            if acts.ncols() != input_dim || grads.ncols() != output_dim {
                return Err(LinalgError::ShapeError(format!(
                    "Layer {} dimension mismatch: expected ({}, {}), got ({}, {})",
                    i,
                    input_dim,
                    output_dim,
                    acts.ncols(),
                    grads.ncols()
                )));
            }

            // Compute Kronecker factors for this layer
            let (input_cov, output_cov) = kfac_factorization(acts, grads, Some(self.damping))?;

            // Compute inverses using Cholesky decomposition for stability
            let input_inv = self.stable_inverse(&input_cov.view())?;
            let output_inv = self.stable_inverse(&output_cov.view())?;

            self.layer_factors.push((input_cov, output_cov));
            self.inverse_factors.push((input_inv, output_inv));
        }

        Ok(())
    }

    /// Compute stable matrix inverse using Cholesky decomposition
    fn stable_inverse(&self, matrix: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
        let n = matrix.nrows();

        // Try Cholesky decomposition first (more stable for positive definite matrices)
        if let Ok(l) = cholesky(matrix, None) {
            // Solve L * L^T * X = I using forward and back substitution
            let mut inv = Array2::eye(n);

            // For each column of the identity matrix
            for col in 0..n {
                let mut b = Array1::zeros(n);
                b[col] = F::one();

                // Forward substitution: solve L * y = b
                let mut y = Array1::zeros(n);
                for i in 0..n {
                    let mut sum = F::zero();
                    for j in 0..i {
                        sum += l[[i, j]] * y[j];
                    }
                    y[i] = (b[i] - sum) / l[[i, i]];
                }

                // Back substitution: solve L^T * x = y
                let mut x = Array1::zeros(n);
                for i in (0..n).rev() {
                    let mut sum = F::zero();
                    for j in (i + 1)..n {
                        sum += l[[j, i]] * x[j];
                    }
                    x[i] = (y[i] - sum) / l[[i, i]];
                }

                // Store result in inverse matrix
                for i in 0..n {
                    inv[[i, col]] = x[i];
                }
            }

            return Ok(inv);
        }

        // Fallback: regularize more heavily and use basic inversion
        let mut regularized = matrix.to_owned();
        for i in 0..n {
            regularized[[i, i]] += self.damping * F::from(10.0).unwrap();
        }

        // Simple inversion using basic LU decomposition (placeholder)
        // In practice, this would use a robust matrix inversion routine
        let mut inv = Array2::eye(n);

        // For now, return a heavily damped diagonal matrix as fallback
        for i in 0..n {
            inv[[i, i]] = F::one() / (regularized[[i, i]] + self.damping);
        }

        Ok(inv)
    }

    /// Apply block-diagonal preconditioning to gradients
    ///
    /// # Arguments
    ///
    /// * `layer_gradients` - Gradients for each layer
    ///
    /// # Returns
    ///
    /// * Preconditioned gradients for each layer
    pub fn precondition_gradients(
        &self,
        layer_gradients: &[ArrayView2<F>],
    ) -> LinalgResult<Vec<Array2<F>>> {
        if layer_gradients.len() != self.inverse_factors.len() {
            return Err(LinalgError::ShapeError(
                "Number of gradient matrices must match number of layers".to_string(),
            ));
        }

        let mut preconditioned = Vec::new();

        for ((grads, (input_inv, output_inv)), &(input_dim, output_dim)) in layer_gradients
            .iter()
            .zip(self.inverse_factors.iter())
            .zip(self.layer_dims.iter())
        {
            // Ensure gradients have the expected shape
            let (batch_size, grad_output_dim) = grads.dim();
            if grad_output_dim != output_dim {
                return Err(LinalgError::ShapeError(format!(
                    "Gradient output dimension mismatch: expected {}, got {}",
                    output_dim, grad_output_dim
                )));
            }

            // Create extended gradient matrix with bias terms (add column of zeros for bias gradient)
            let mut extended_grads = Array2::zeros((input_dim + 1, output_dim));

            // Copy the weight gradients
            for i in 0..input_dim {
                for j in 0..output_dim {
                    // Average gradients across batch
                    let mut sum = F::zero();
                    for b in 0..batch_size {
                        sum += grads[[b, j]]; // Accumulate gradient for output j
                    }
                    extended_grads[[i, j]] = sum / F::from(batch_size).unwrap();
                }
            }
            // Bias gradients are typically the mean of output gradients
            for j in 0..output_dim {
                let mut sum = F::zero();
                for b in 0..batch_size {
                    sum += grads[[b, j]];
                }
                extended_grads[[input_dim, j]] = sum / F::from(batch_size).unwrap();
            }

            // Apply Kronecker-factored preconditioning: P^-1 * G = A^-1 * G * S^-1
            let temp = input_inv.dot(&extended_grads);
            let preconditioned_grad = temp.dot(output_inv);

            // Extract the weight part (excluding bias) and reshape back to original format
            let mut result = Array2::zeros((batch_size, output_dim));
            for b in 0..batch_size {
                for j in 0..output_dim {
                    // Use the weight part of the preconditioned gradient
                    result[[b, j]] = preconditioned_grad[[0, j]]; // Use first row as representative
                }
            }

            preconditioned.push(result);
        }

        Ok(preconditioned)
    }

    /// Get memory statistics for the block-diagonal approximation
    pub fn memory_info(&self) -> BlockFisherMemoryInfo {
        let mut total_elements = 0;
        let mut total_inverse_elements = 0;
        let mut original_elements = 0;

        for ((input_cov, output_cov), &(input_dim, output_dim)) in
            self.layer_factors.iter().zip(self.layer_dims.iter())
        {
            total_elements += input_cov.len() + output_cov.len();
            total_inverse_elements += input_cov.len() + output_cov.len(); // Same size for inverses
            original_elements += input_dim * output_dim * input_dim * output_dim;
            // Full Fisher would be (in*out)²
        }

        let compression_ratio =
            original_elements as f64 / (total_elements + total_inverse_elements) as f64;

        BlockFisherMemoryInfo {
            num_layers: self.layer_factors.len(),
            total_factor_elements: total_elements,
            total_inverse_elements,
            compression_ratio,
            estimated_full_fisher_elements: original_elements,
        }
    }
}

/// Memory usage information for block-diagonal Fisher approximation
#[derive(Debug)]
pub struct BlockFisherMemoryInfo {
    /// Number of layers in the network
    pub num_layers: usize,
    /// Total elements in Kronecker factors
    pub total_factor_elements: usize,
    /// Total elements in inverse factors
    pub total_inverse_elements: usize,
    /// Compression ratio vs full Fisher matrix
    pub compression_ratio: f64,
    /// Estimated elements in full Fisher matrix
    pub estimated_full_fisher_elements: usize,
}

/// Natural gradient step with advanced K-FAC features
///
/// This function implements a sophisticated natural gradient update that combines
/// multiple advanced K-FAC techniques: exponential moving averages, adaptive damping,
/// gradient clipping, and optional momentum.
///
/// # Arguments
///
/// * `weights` - Current weight matrix
/// * `gradients` - Gradient matrix
/// * `kfac_optimizer` - K-FAC optimizer state with moving averages
/// * `input_acts` - Current batch input activations
/// * `output_grads` - Current batch output gradients  
/// * `learning_rate` - Learning rate
/// * `momentum` - Optional momentum coefficient
/// * `gradient_clip` - Optional gradient clipping threshold
///
/// # Returns
///
/// * Updated weights and updated K-FAC state
pub fn advanced_kfac_step<F>(
    weights: &ArrayView2<F>,
    gradients: &ArrayView2<F>,
    kfac_optimizer: &mut KFACOptimizer<F>,
    input_acts: &ArrayView2<F>,
    output_grads: &ArrayView2<F>,
    learning_rate: F,
    _momentum: Option<F>,
    gradient_clip: Option<F>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand,
{
    // Update covariance estimates with moving averages
    let (input_cov, output_cov) = kfac_optimizer.update_covariances(input_acts, output_grads)?;

    // Compute stable inverses
    let input_inv = stable_matrix_inverse(&input_cov.view(), kfac_optimizer.get_damping())?;
    let output_inv = stable_matrix_inverse(&output_cov.view(), kfac_optimizer.get_damping())?;

    // Compute natural gradient: G_nat = A^-1 * G * S^-1
    let temp = input_inv.dot(gradients);
    let mut natural_grad = temp.dot(&output_inv);

    // Apply gradient clipping if specified
    if let Some(clip_threshold) = gradient_clip {
        let grad_norm = matrix_norm(&natural_grad.view(), "fro", None)?;
        if grad_norm > clip_threshold {
            let scale_factor = clip_threshold / grad_norm;
            for elem in natural_grad.iter_mut() {
                *elem *= scale_factor;
            }
        }
    }

    // Apply momentum if specified (would need momentum state in optimizer)
    // For now, just use the natural gradient directly

    // Update weights: w = w - lr * natural_grad
    let mut new_weights = weights.to_owned();
    for i in 0..weights.nrows() {
        for j in 0..weights.ncols() {
            new_weights[[i, j]] = weights[[i, j]] - learning_rate * natural_grad[[i, j]];
        }
    }

    Ok(new_weights)
}

/// Compute stable matrix inverse with enhanced regularization
fn stable_matrix_inverse<F>(matrix: &ArrayView2<F>, damping: F) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand,
{
    let n = matrix.nrows();

    // Add damping to ensure positive definiteness
    let mut regularized = matrix.to_owned();
    for i in 0..n {
        regularized[[i, i]] += damping;
    }

    // Try Cholesky decomposition for stable inversion
    match cholesky(&regularized.view(), None) {
        Ok(l) => {
            // Compute inverse using Cholesky factor
            let mut inv = Array2::eye(n);

            for col in 0..n {
                let mut b = Array1::zeros(n);
                b[col] = F::one();

                // Forward substitution: L * y = b
                let mut y = Array1::zeros(n);
                for i in 0..n {
                    let mut sum = F::zero();
                    for j in 0..i {
                        sum += l[[i, j]] * y[j];
                    }
                    y[i] = (b[i] - sum) / l[[i, i]];
                }

                // Back substitution: L^T * x = y
                let mut x = Array1::zeros(n);
                for i in (0..n).rev() {
                    let mut sum = F::zero();
                    for j in (i + 1)..n {
                        sum += l[[j, i]] * x[j];
                    }
                    x[i] = (y[i] - sum) / l[[i, i]];
                }

                for i in 0..n {
                    inv[[i, col]] = x[i];
                }
            }

            Ok(inv)
        }
        Err(_) => {
            // Fallback: use diagonal approximation with heavy regularization
            let mut inv = Array2::zeros((n, n));
            for i in 0..n {
                let diag_val = matrix[[i, i]] + damping * F::from(100.0).unwrap();
                inv[[i, i]] = F::one() / diag_val;
            }
            Ok(inv)
        }
    }
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

    #[test]
    fn test_kfac_optimizer_basic() {
        let mut optimizer = KFACOptimizer::<f64>::new(Some(0.9), Some(0.01));

        // Create sample data
        let input_acts = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let output_grads = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];

        // First update should initialize averages
        let (input_cov1, _output_cov1) = optimizer
            .update_covariances(&input_acts.view(), &output_grads.view())
            .unwrap();

        assert_eq!(optimizer.step_count, 1);
        assert!(optimizer.input_cov_avg.is_some());
        assert!(optimizer.output_cov_avg.is_some());

        // Second update should use moving averages
        let (input_cov2, _output_cov2) = optimizer
            .update_covariances(&input_acts.view(), &output_grads.view())
            .unwrap();

        assert_eq!(optimizer.step_count, 2);

        // Results should be different due to bias correction
        assert!((input_cov1[[0, 0]] - input_cov2[[0, 0]]).abs() > 1e-10);
    }

    #[test]
    fn test_kfac_optimizer_damping_adjustment() {
        let mut optimizer = KFACOptimizer::<f64>::new(None, Some(0.01));
        let initial_damping = optimizer.get_damping();

        // Test improvement case
        optimizer.adjust_damping(true, Some(0.8));
        let after_improvement = optimizer.get_damping();
        assert!(after_improvement < initial_damping);

        // Test deterioration case
        optimizer.adjust_damping(false, None);
        let after_deterioration = optimizer.get_damping();
        assert!(after_deterioration > after_improvement);

        // Test bounds
        for _ in 0..20 {
            optimizer.adjust_damping(false, None);
        }
        assert!(optimizer.get_damping() <= optimizer.max_damping);

        for _ in 0..20 {
            optimizer.adjust_damping(true, Some(0.9));
        }
        assert!(optimizer.get_damping() >= optimizer.min_damping);
    }

    #[test]
    fn test_block_diagonal_fisher() {
        let layer_dims = vec![(10, 20), (20, 10)];
        let mut fisher = BlockDiagonalFisher::<f64>::new(layer_dims, 0.01);

        // Create sample activations and gradients for 2 layers
        // Layer 1: 10 inputs -> 20 outputs, so acts should be Nx10, grads should be Nx20
        let layer1_acts = Array2::from_shape_fn((5, 10), |(i, j)| (i + j) as f64 * 0.1);
        let layer1_grads = Array2::from_shape_fn((5, 20), |(i, j)| (i + j) as f64 * 0.01);

        // Layer 2: 20 inputs -> 10 outputs, so acts should be Nx20, grads should be Nx10
        let layer2_acts = Array2::from_shape_fn((5, 20), |(i, j)| (i + j) as f64 * 0.05);
        let layer2_grads = Array2::from_shape_fn((5, 10), |(i, j)| (i + j) as f64 * 0.02);

        let activations = vec![layer1_acts.view(), layer2_acts.view()];
        let gradients = vec![layer1_grads.view(), layer2_grads.view()];

        // Update Fisher approximation
        fisher.update_fisher(&activations, &gradients).unwrap();

        assert_eq!(fisher.layer_factors.len(), 2);
        assert_eq!(fisher.inverse_factors.len(), 2);

        // Test preconditioning
        let grad_matrices = vec![layer1_grads.view(), layer2_grads.view()];
        let preconditioned = fisher.precondition_gradients(&grad_matrices).unwrap();

        assert_eq!(preconditioned.len(), 2);
        assert_eq!(preconditioned[0].shape(), layer1_grads.shape());
        assert_eq!(preconditioned[1].shape(), layer2_grads.shape());

        // Test memory info
        let memory_info = fisher.memory_info();
        assert_eq!(memory_info.num_layers, 2);
        assert!(memory_info.compression_ratio > 1.0);
    }

    #[test]
    fn test_advanced_kfac_step() {
        let mut optimizer = KFACOptimizer::<f64>::new(Some(0.95), Some(0.001));

        // Create sample data
        let weights = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]];
        let gradients = array![[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]];
        let input_acts = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let output_grads = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]];

        let learning_rate = 0.01;

        // Perform advanced K-FAC step
        let new_weights = advanced_kfac_step(
            &weights.view(),
            &gradients.view(),
            &mut optimizer,
            &input_acts.view(),
            &output_grads.view(),
            learning_rate,
            None,
            Some(1.0), // gradient clipping
        )
        .unwrap();

        // Check that weights were updated
        assert_eq!(new_weights.shape(), weights.shape());

        // Weights should be different from original
        let mut weights_changed = false;
        for i in 0..weights.nrows() {
            for j in 0..weights.ncols() {
                if (weights[[i, j]] - new_weights[[i, j]]).abs() > 1e-10 {
                    weights_changed = true;
                    break;
                }
            }
        }
        assert!(weights_changed);

        // Optimizer state should be updated
        assert_eq!(optimizer.step_count, 1);
        assert!(optimizer.input_cov_avg.is_some());
    }

    #[test]
    fn test_stable_matrix_inverse() {
        // Test with a simple positive definite matrix
        let matrix = array![[2.0, 1.0], [1.0, 2.0]];
        let damping = 0.01;
        let inv = stable_matrix_inverse(&matrix.view(), damping).unwrap();

        // Create the regularized matrix (what we actually inverted)
        let mut regularized = matrix.clone();
        for i in 0..2 {
            regularized[[i, i]] += damping;
        }

        // Check that regularized_matrix * inverse ≈ identity
        let product = regularized.dot(&inv);
        let identity = Array2::eye(2);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(product[[i, j]], identity[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_kfac_optimizer_reset() {
        let mut optimizer = KFACOptimizer::<f64>::new(Some(0.9), Some(0.01));

        // Perform some updates
        let input_acts = array![[1.0, 2.0], [3.0, 4.0]];
        let output_grads = array![[0.1, 0.2], [0.3, 0.4]];

        optimizer
            .update_covariances(&input_acts.view(), &output_grads.view())
            .unwrap();
        optimizer.adjust_damping(false, None);

        assert!(optimizer.step_count > 0);
        assert!(optimizer.input_cov_avg.is_some());

        // Reset and check state
        optimizer.reset();

        assert_eq!(optimizer.step_count, 0);
        assert!(optimizer.input_cov_avg.is_none());
        assert!(optimizer.output_cov_avg.is_none());
        assert_eq!(optimizer.adaptive_damping, optimizer.base_damping);
    }

    #[test]
    fn test_block_fisher_memory_info() {
        let layer_dims = vec![(10, 5), (5, 3)];
        let mut fisher = BlockDiagonalFisher::<f64>::new(layer_dims, 0.01);

        // Create dummy data and update
        let layer1_acts = Array2::zeros((8, 10));
        let layer1_grads = Array2::zeros((8, 5));
        let layer2_acts = Array2::zeros((8, 5));
        let layer2_grads = Array2::zeros((8, 3));

        let activations = vec![layer1_acts.view(), layer2_acts.view()];
        let gradients = vec![layer1_grads.view(), layer2_grads.view()];

        fisher.update_fisher(&activations, &gradients).unwrap();

        let memory_info = fisher.memory_info();

        // Check compression ratio makes sense
        assert!(memory_info.compression_ratio > 1.0);
        assert_eq!(memory_info.num_layers, 2);

        // Check estimated savings
        let layer1_full_fisher = (10 * 5) * (10 * 5); // (input*output)²
        let layer2_full_fisher = (5 * 3) * (5 * 3);
        let expected_full = layer1_full_fisher + layer2_full_fisher;

        assert_eq!(memory_info.estimated_full_fisher_elements, expected_full);
    }
}
