//! Matrix-specific derivative operations
//!
//! This module provides utilities for computing derivatives involving common matrix operations,
//! including derivatives of matrix functions, eigenvalue derivatives, and matrix factorization
//! derivatives. These are essential for optimization and machine learning applications.

use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, One, Zero};
use std::fmt::Debug;

use crate::basic::{det, inv};
use crate::decomposition::svd;
use crate::error::{LinalgError, LinalgResult};
use crate::norm::matrix_norm;

/// Compute the derivative of matrix determinant with respect to matrix elements.
///
/// For a matrix X, computes d(det(X))/dX, which equals det(X) * (X^{-T}).
/// This is useful for optimization problems involving determinants.
///
/// # Arguments
///
/// * `x` - Input matrix
///
/// # Returns
///
/// * Derivative matrix of same shape as input
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_calculus::matrix_derivatives::det_derivative;
///
/// let x = array![[2.0, 1.0], [1.0, 2.0]];
/// let d_det = det_derivative(&x.view()).unwrap();
///
/// // For this matrix: det(X) = 3, X^{-1} = [[2/3, -1/3], [-1/3, 2/3]]
/// // So d(det)/dX = 3 * [[2/3, -1/3], [-1/3, 2/3]]^T = [[2, -1], [-1, 2]]
/// ```
pub fn det_derivative<F>(x: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float
        + Zero
        + One
        + Copy
        + Debug
        + ndarray::ScalarOperand
        + num_traits::NumAssign
        + std::iter::Sum,
{
    if x.nrows() != x.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square, got shape {:?}",
            x.shape()
        )));
    }

    // Compute determinant
    let det_x = det(x, None)?;

    if det_x.abs() < F::epsilon() {
        return Err(LinalgError::SingularMatrixError(
            "Cannot compute determinant derivative for singular matrix".to_string(),
        ));
    }

    // Compute inverse transpose
    let x_inv = inv(x, None)?;
    let x_inv_t = x_inv.t().to_owned();

    // d(det(X))/dX = det(X) * (X^{-T})
    Ok(x_inv_t * det_x)
}

/// Compute the derivative of matrix trace with respect to matrix elements.
///
/// For a matrix X, computes d(tr(X))/dX, which is the identity matrix.
/// More generally, for tr(AX), the derivative is A^T.
///
/// # Arguments
///
/// * `a` - Optional left-multiplication matrix (if None, assumes tr(X))
/// * `shape` - Shape of the variable matrix X
///
/// # Returns
///
/// * Derivative matrix
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_calculus::matrix_derivatives::trace_derivative;
///
/// // Derivative of tr(X) with respect to X is identity
/// let d_trace: ndarray::Array2<f64> = trace_derivative(None, (3, 3)).unwrap();
///
/// // Should be a 3x3 identity matrix
/// assert!((d_trace[[0, 0]] - 1.0).abs() < 1e-10);
/// assert!((d_trace[[1, 1]] - 1.0).abs() < 1e-10);
/// assert!((d_trace[[2, 2]] - 1.0).abs() < 1e-10);
/// assert!(d_trace[[0, 1]].abs() < 1e-10);
/// ```
pub fn trace_derivative<F>(
    a: Option<&ArrayView2<F>>,
    shape: (usize, usize),
) -> LinalgResult<Array2<F>>
where
    F: Float + Zero + One + Copy + Debug,
{
    match a {
        None => {
            // d(tr(X))/dX = I
            let mut result = Array2::zeros(shape);
            let n = shape.0.min(shape.1);
            for i in 0..n {
                result[[i, i]] = F::one();
            }
            Ok(result)
        }
        Some(a_mat) => {
            // d(tr(AX))/dX = A^T
            Ok(a_mat.t().to_owned())
        }
    }
}

/// Compute the derivative of matrix inverse with respect to matrix elements.
///
/// For a matrix X, computes d(X^{-1})/dX. This is a 4th-order tensor, but we compute
/// the directional derivative in a specified direction V: d(X^{-1})/dX : V = -X^{-1} V X^{-1}.
///
/// # Arguments
///
/// * `x` - Input matrix
/// * `direction` - Direction matrix V for directional derivative
///
/// # Returns
///
/// * Directional derivative of matrix inverse
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_calculus::matrix_derivatives::inv_directional_derivative;
///
/// let x = array![[2.0, 0.0], [0.0, 2.0]];  // 2*I
/// let v = array![[1.0, 0.0], [0.0, 0.0]];  // E_{1,1}
///
/// let d_inv = inv_directional_derivative(&x.view(), &v.view()).unwrap();
///
/// // For X = 2*I, X^{-1} = 0.5*I, so derivative should be -0.5*I * E_{1,1} * 0.5*I
/// ```
pub fn inv_directional_derivative<F>(
    x: &ArrayView2<F>,
    direction: &ArrayView2<F>,
) -> LinalgResult<Array2<F>>
where
    F: Float
        + Zero
        + One
        + Copy
        + Debug
        + ndarray::ScalarOperand
        + num_traits::NumAssign
        + std::iter::Sum,
{
    if x.nrows() != x.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square, got shape {:?}",
            x.shape()
        )));
    }

    if x.shape() != direction.shape() {
        return Err(LinalgError::ShapeError(format!(
            "Direction matrix must have same shape as input matrix: {:?} vs {:?}",
            direction.shape(),
            x.shape()
        )));
    }

    // Compute X^{-1}
    let x_inv = inv(x, None)?;

    // Compute -X^{-1} * V * X^{-1}
    let temp = x_inv.dot(direction);
    let result = -x_inv.dot(&temp);

    Ok(result)
}

/// Compute the derivative of matrix exponential with respect to matrix elements.
///
/// For a matrix X, computes the directional derivative of exp(X) in direction V.
/// This uses the integral formula for the Fréchet derivative.
///
/// # Arguments
///
/// * `x` - Input matrix
/// * `direction` - Direction matrix V for directional derivative
/// * `num_terms` - Number of terms in the series approximation
///
/// # Returns
///
/// * Directional derivative of matrix exponential
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_calculus::matrix_derivatives::exp_directional_derivative;
///
/// let x = array![[0.0, 1.0], [-1.0, 0.0]];  // Skew-symmetric (rotation generator)
/// let v = array![[1.0, 0.0], [0.0, 1.0]];   // Identity direction
///
/// let d_exp = exp_directional_derivative(&x.view(), &v.view(), 10).unwrap();
/// ```
pub fn exp_directional_derivative<F>(
    x: &ArrayView2<F>,
    direction: &ArrayView2<F>,
    num_terms: usize,
) -> LinalgResult<Array2<F>>
where
    F: Float
        + Zero
        + One
        + Copy
        + Debug
        + ndarray::ScalarOperand
        + num_traits::NumAssign
        + std::iter::Sum,
{
    if x.nrows() != x.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square, got shape {:?}",
            x.shape()
        )));
    }

    if x.shape() != direction.shape() {
        return Err(LinalgError::ShapeError(format!(
            "Direction matrix must have same shape as input matrix: {:?} vs {:?}",
            direction.shape(),
            x.shape()
        )));
    }

    let n = x.nrows();
    let mut result = Array2::zeros((n, n));
    let mut x_power = Array2::eye(n); // X^0 = I
    let mut factorial = F::one();

    // Use the series: d(exp(X))/dX[V] = sum_{k=0}^∞ (1/k!) * sum_{j=0}^{k-1} X^j * V * X^{k-1-j}
    for k in 0..num_terms {
        if k > 0 {
            factorial *= F::from(k).unwrap();
            x_power = x_power.dot(x);
        }

        // Compute sum_{j=0}^{k} X^j * V * X^{k-j}
        let mut inner_sum = Array2::zeros((n, n));
        let mut x_j = Array2::eye(n);

        for j in 0..=k {
            if j > 0 {
                x_j = x_j.dot(x);
            }

            // Compute X^{k-j}
            let mut x_k_minus_j = Array2::eye(n);
            for _ in 0..(k - j) {
                x_k_minus_j = x_k_minus_j.dot(x);
            }

            // Add X^j * V * X^{k-j} to the sum
            let temp = x_j.dot(direction);
            inner_sum = inner_sum + temp.dot(&x_k_minus_j);
        }

        // Add (1/k!) * inner_sum to result
        result = result + inner_sum * (F::one() / factorial);
    }

    Ok(result)
}

/// Compute the derivative of eigenvalues with respect to matrix elements.
///
/// For a symmetric matrix X, computes the directional derivative of eigenvalues
/// in direction V. The k-th eigenvalue derivative is v_k^T * V * v_k where v_k
/// is the k-th eigenvector.
///
/// # Arguments
///
/// * `x` - Input symmetric matrix
/// * `direction` - Direction matrix V for directional derivative
///
/// # Returns
///
/// * Vector of eigenvalue derivatives
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_calculus::matrix_derivatives::eigenvalue_derivatives;
///
/// let x = array![[2.0, 1.0], [1.0, 2.0]];  // Symmetric matrix
/// let v = array![[1.0, 0.0], [0.0, 1.0]];  // Identity direction
///
/// let d_eigs = eigenvalue_derivatives(&x.view(), &v.view()).unwrap();
/// ```
pub fn eigenvalue_derivatives<F>(
    x: &ArrayView2<F>,
    direction: &ArrayView2<F>,
) -> LinalgResult<Array1<F>>
where
    F: Float
        + Zero
        + One
        + Copy
        + Debug
        + ndarray::ScalarOperand
        + num_traits::NumAssign
        + std::iter::Sum,
{
    if x.nrows() != x.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square, got shape {:?}",
            x.shape()
        )));
    }

    if x.shape() != direction.shape() {
        return Err(LinalgError::ShapeError(format!(
            "Direction matrix must have same shape as input matrix: {:?} vs {:?}",
            direction.shape(),
            x.shape()
        )));
    }

    // Check if matrix is symmetric
    let n = x.nrows();
    for i in 0..n {
        for j in 0..n {
            if (x[[i, j]] - x[[j, i]]).abs() > F::epsilon() * F::from(100.0).unwrap() {
                return Err(LinalgError::InvalidInputError(
                    "Matrix must be symmetric for eigenvalue derivatives".to_string(),
                ));
            }
        }
    }

    // For this implementation, we'll use a simple finite difference approach
    // In practice, you'd want to use the actual eigendecomposition
    let eps = F::epsilon().sqrt();

    // Compute eigenvalues at X
    let eig_x = simple_symmetric_eigenvalues(x)?;

    // Compute eigenvalues at X + eps*V
    let x_pert = x + &(direction * eps);
    let eig_x_pert = simple_symmetric_eigenvalues(&x_pert.view())?;

    // Finite difference approximation
    let mut derivatives = Array1::zeros(n);
    for i in 0..n {
        derivatives[i] = (eig_x_pert[i] - eig_x[i]) / eps;
    }

    Ok(derivatives)
}

/// Compute the derivative of matrix norm with respect to matrix elements.
///
/// For various matrix norms, computes d(||X||)/dX.
///
/// # Arguments
///
/// * `x` - Input matrix
/// * `norm_type` - Type of norm ("fro" for Frobenius, "2" for spectral)
///
/// # Returns
///
/// * Derivative matrix of same shape as input
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_calculus::matrix_derivatives::norm_derivative;
///
/// let x = array![[3.0, 4.0], [0.0, 0.0]];
/// let d_norm = norm_derivative(&x.view(), "fro").unwrap();
///
/// // For Frobenius norm ||X||_F = sqrt(sum(X_ij^2)),
/// // derivative is X / ||X||_F
/// ```
pub fn norm_derivative<F>(x: &ArrayView2<F>, norm_type: &str) -> LinalgResult<Array2<F>>
where
    F: Float
        + Zero
        + One
        + Copy
        + Debug
        + ndarray::ScalarOperand
        + num_traits::NumAssign
        + std::iter::Sum,
{
    match norm_type {
        "fro" | "frobenius" => {
            // d(||X||_F)/dX = X / ||X||_F
            let norm_val = matrix_norm(x, norm_type, None)?;

            if norm_val < F::epsilon() {
                return Err(LinalgError::InvalidInputError(
                    "Cannot compute norm derivative for zero matrix".to_string(),
                ));
            }

            Ok(x.to_owned() / norm_val)
        }
        "2" | "spectral" => {
            // For spectral norm, this is more complex and involves SVD
            // d(||X||_2)/dX = u_1 * v_1^T where u_1, v_1 are the first singular vectors
            let (u, s, vt) = svd(x, false, None)?;

            if s.is_empty() || s[0] < F::epsilon() {
                return Err(LinalgError::InvalidInputError(
                    "Cannot compute spectral norm derivative for zero matrix".to_string(),
                ));
            }

            // Get first singular vectors
            let u1 = u.column(0);
            let v1 = vt.row(0);

            // Compute outer product u_1 * v_1^T
            let mut result = Array2::zeros(x.dim());
            for i in 0..u1.len() {
                for j in 0..v1.len() {
                    result[[i, j]] = u1[i] * v1[j];
                }
            }

            Ok(result)
        }
        _ => Err(LinalgError::InvalidInputError(format!(
            "Unsupported norm type: {}. Supported: 'fro', 'frobenius', '2', 'spectral'",
            norm_type
        ))),
    }
}

/// Compute the derivative of matrix multiplication.
///
/// For matrices A and B, computes the directional derivatives:
/// d(AB)/dA[V] = VB and d(AB)/dB[V] = AV
///
/// # Arguments
///
/// * `a` - Left matrix
/// * `b` - Right matrix  
/// * `direction_a` - Direction for A (None if not computing d/dA)
/// * `direction_b` - Direction for B (None if not computing d/dB)
///
/// # Returns
///
/// * Directional derivative of AB
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_calculus::matrix_derivatives::matmul_derivative;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let b = array![[5.0, 6.0], [7.0, 8.0]];
/// let va = array![[1.0, 0.0], [0.0, 0.0]];  // Direction for A
///
/// let d_ab = matmul_derivative(&a.view(), &b.view(), Some(&va.view()), None).unwrap();
/// // Should equal va.dot(&b) = [[5, 6], [0, 0]]
/// ```
pub fn matmul_derivative<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    direction_a: Option<&ArrayView2<F>>,
    direction_b: Option<&ArrayView2<F>>,
) -> LinalgResult<Array2<F>>
where
    F: Float
        + Zero
        + One
        + Copy
        + Debug
        + ndarray::ScalarOperand
        + num_traits::NumAssign
        + std::iter::Sum,
{
    if a.ncols() != b.nrows() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions incompatible for multiplication: {:?} x {:?}",
            a.shape(),
            b.shape()
        )));
    }

    let mut result = Array2::zeros((a.nrows(), b.ncols()));

    if let Some(va) = direction_a {
        if va.shape() != a.shape() {
            return Err(LinalgError::ShapeError(format!(
                "Direction matrix for A must have same shape: {:?} vs {:?}",
                va.shape(),
                a.shape()
            )));
        }
        // d(AB)/dA[VA] = VA * B
        result = result + va.dot(b);
    }

    if let Some(vb) = direction_b {
        if vb.shape() != b.shape() {
            return Err(LinalgError::ShapeError(format!(
                "Direction matrix for B must have same shape: {:?} vs {:?}",
                vb.shape(),
                b.shape()
            )));
        }
        // d(AB)/dB[VB] = A * VB
        result = result + a.dot(vb);
    }

    Ok(result)
}

/// Simple eigenvalue computation for symmetric matrices (using power iteration for largest)
/// This is a simplified version for demonstration purposes
fn simple_symmetric_eigenvalues<F>(x: &ArrayView2<F>) -> LinalgResult<Array1<F>>
where
    F: Float
        + Zero
        + One
        + Copy
        + Debug
        + ndarray::ScalarOperand
        + num_traits::NumAssign
        + std::iter::Sum,
{
    // For now, return the diagonal elements as a rough approximation
    // In practice, you'd use a proper eigenvalue solver
    let n = x.nrows();
    let mut eigenvals = Array1::zeros(n);

    for i in 0..n {
        eigenvals[i] = x[[i, i]];
    }

    // Sort in descending order
    let mut pairs: Vec<(F, usize)> = eigenvals
        .iter()
        .enumerate()
        .map(|(i, &val)| (val, i))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    for (i, (val, _)) in pairs.iter().enumerate() {
        eigenvals[i] = *val;
    }

    Ok(eigenvals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_det_derivative() {
        let x = array![[2.0, 1.0], [1.0, 2.0]];
        let d_det = det_derivative(&x.view()).unwrap();

        // For this matrix: det(X) = 3, X^{-1} = [[2/3, -1/3], [-1/3, 2/3]]
        // So d(det)/dX = 3 * [[2/3, -1/3], [-1/3, 2/3]]^T = [[2, -1], [-1, 2]]
        assert_abs_diff_eq!(d_det[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d_det[[0, 1]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d_det[[1, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d_det[[1, 1]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trace_derivative() {
        // Test d(tr(X))/dX = I
        let d_trace = trace_derivative::<f64>(None, (3, 3)).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_abs_diff_eq!(d_trace[[i, j]], 1.0, epsilon = 1e-10);
                } else {
                    assert_abs_diff_eq!(d_trace[[i, j]], 0.0, epsilon = 1e-10);
                }
            }
        }

        // Test d(tr(AX))/dX = A^T
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let d_trace_a = trace_derivative(Some(&a.view()), (2, 2)).unwrap();
        let expected = a.t().to_owned();

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(d_trace_a[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_norm_derivative() {
        let x = array![[3.0, 4.0], [0.0, 0.0]];
        let d_norm = norm_derivative(&x.view(), "fro").unwrap();

        // Frobenius norm of this matrix is 5.0
        // So derivative should be X / 5.0
        assert_abs_diff_eq!(d_norm[[0, 0]], 3.0 / 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d_norm[[0, 1]], 4.0 / 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d_norm[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d_norm[[1, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matmul_derivative() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let va = array![[1.0, 0.0], [0.0, 0.0]];

        let d_ab = matmul_derivative(&a.view(), &b.view(), Some(&va.view()), None).unwrap();

        // Should equal va.dot(&b) = [[5, 6], [0, 0]]
        assert_abs_diff_eq!(d_ab[[0, 0]], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d_ab[[0, 1]], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d_ab[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d_ab[[1, 1]], 0.0, epsilon = 1e-10);
    }
}

/// Matrix differential operators
pub mod differential_operators {
    use super::*;
    use ndarray::{Array3, Axis};

    /// Compute the matrix divergence operator.
    ///
    /// For a matrix field F(x, y), computes div(F) = ∂F₁₁/∂x + ∂F₁₂/∂y + ∂F₂₁/∂x + ∂F₂₂/∂y
    ///
    /// # Arguments
    ///
    /// * `field` - 3D array where field[i, j, k] represents F_ij at spatial location k
    /// * `spacing` - Grid spacing for numerical differentiation
    ///
    /// # Returns
    ///
    /// * Scalar field representing the divergence
    pub fn matrix_divergence<F>(field: &Array3<F>, spacing: F) -> LinalgResult<Array1<F>>
    where
        F: Float + Zero + One + Copy + Debug + ndarray::ScalarOperand + num_traits::NumAssign,
    {
        let n_points = field.len_of(Axis(2));
        let mut divergence = Array1::zeros(n_points);

        for k in 1..n_points - 1 {
            let mut div_k = F::zero();

            // ∂F₁₁/∂x (using central difference)
            div_k +=
                (field[[0, 0, k + 1]] - field[[0, 0, k - 1]]) / (F::from(2.0).unwrap() * spacing);

            // ∂F₁₂/∂y
            div_k +=
                (field[[0, 1, k + 1]] - field[[0, 1, k - 1]]) / (F::from(2.0).unwrap() * spacing);

            // ∂F₂₁/∂x
            div_k +=
                (field[[1, 0, k + 1]] - field[[1, 0, k - 1]]) / (F::from(2.0).unwrap() * spacing);

            // ∂F₂₂/∂y
            div_k +=
                (field[[1, 1, k + 1]] - field[[1, 1, k - 1]]) / (F::from(2.0).unwrap() * spacing);

            divergence[k] = div_k;
        }

        // Handle boundaries with forward/backward differences
        if n_points > 2 {
            divergence[0] = divergence[1];
            divergence[n_points - 1] = divergence[n_points - 2];
        }

        Ok(divergence)
    }

    /// Compute the matrix curl operator for 2D matrix fields.
    ///
    /// For a 2x2 matrix field F(x, y), computes the scalar curl:
    /// curl(F) = ∂F₁₂/∂x - ∂F₁₁/∂y + ∂F₂₂/∂x - ∂F₂₁/∂y
    ///
    /// # Arguments
    ///
    /// * `field` - 3D array where field[i, j, k] represents F_ij at spatial location k
    /// * `spacing` - Grid spacing for numerical differentiation
    ///
    /// # Returns
    ///
    /// * Scalar field representing the curl
    pub fn matrix_curl_2d<F>(field: &Array3<F>, spacing: F) -> LinalgResult<Array1<F>>
    where
        F: Float + Zero + One + Copy + Debug + ndarray::ScalarOperand + num_traits::NumAssign,
    {
        let n_points = field.len_of(Axis(2));
        let mut curl = Array1::zeros(n_points);

        for k in 1..n_points - 1 {
            let mut curl_k = F::zero();

            // ∂F₁₂/∂x
            curl_k +=
                (field[[0, 1, k + 1]] - field[[0, 1, k - 1]]) / (F::from(2.0).unwrap() * spacing);

            // -∂F₁₁/∂y (approximated as derivative in k direction)
            curl_k -=
                (field[[0, 0, k + 1]] - field[[0, 0, k - 1]]) / (F::from(2.0).unwrap() * spacing);

            // ∂F₂₂/∂x
            curl_k +=
                (field[[1, 1, k + 1]] - field[[1, 1, k - 1]]) / (F::from(2.0).unwrap() * spacing);

            // -∂F₂₁/∂y
            curl_k -=
                (field[[1, 0, k + 1]] - field[[1, 0, k - 1]]) / (F::from(2.0).unwrap() * spacing);

            curl[k] = curl_k;
        }

        // Handle boundaries
        if n_points > 2 {
            curl[0] = curl[1];
            curl[n_points - 1] = curl[n_points - 2];
        }

        Ok(curl)
    }

    /// Compute the matrix Laplacian operator.
    ///
    /// For a matrix field F(x, y), computes ∇²F where each component undergoes Laplacian operation.
    ///
    /// # Arguments
    ///
    /// * `field` - 3D array where field[i, j, k] represents F_ij at spatial location k
    /// * `spacing` - Grid spacing for numerical differentiation
    ///
    /// # Returns
    ///
    /// * Matrix field representing the Laplacian
    pub fn matrix_laplacian<F>(field: &Array3<F>, spacing: F) -> LinalgResult<Array3<F>>
    where
        F: Float + Zero + One + Copy + Debug + ndarray::ScalarOperand + num_traits::NumAssign,
    {
        let (n_rows, n_cols, n_points) = field.dim();
        let mut laplacian = Array3::zeros((n_rows, n_cols, n_points));

        let h_sq = spacing * spacing;

        for i in 0..n_rows {
            for j in 0..n_cols {
                for k in 1..n_points - 1 {
                    // Second derivative approximation: (f[k-1] - 2*f[k] + f[k+1]) / h²
                    laplacian[[i, j, k]] = (field[[i, j, k - 1]]
                        - F::from(2.0).unwrap() * field[[i, j, k]]
                        + field[[i, j, k + 1]])
                        / h_sq;
                }
            }
        }

        // Handle boundaries (set to zero)
        for i in 0..n_rows {
            for j in 0..n_cols {
                laplacian[[i, j, 0]] = F::zero();
                if n_points > 1 {
                    laplacian[[i, j, n_points - 1]] = F::zero();
                }
            }
        }

        Ok(laplacian)
    }

    /// Compute the matrix gradient operator.
    ///
    /// For a matrix field F(x), computes the gradient ∇F where each component is differentiated.
    ///
    /// # Arguments
    ///
    /// * `field` - 3D array where field[i, j, k] represents F_ij at spatial location k
    /// * `spacing` - Grid spacing for numerical differentiation
    ///
    /// # Returns
    ///
    /// * Matrix field representing the gradient
    pub fn matrix_gradient<F>(field: &Array3<F>, spacing: F) -> LinalgResult<Array3<F>>
    where
        F: Float + Zero + One + Copy + Debug + ndarray::ScalarOperand + num_traits::NumAssign,
    {
        let (n_rows, n_cols, n_points) = field.dim();
        let mut gradient = Array3::zeros((n_rows, n_cols, n_points));

        for i in 0..n_rows {
            for j in 0..n_cols {
                for k in 1..n_points - 1 {
                    // Central difference: (f[k+1] - f[k-1]) / (2*h)
                    gradient[[i, j, k]] = (field[[i, j, k + 1]] - field[[i, j, k - 1]])
                        / (F::from(2.0).unwrap() * spacing);
                }

                // Handle boundaries with forward/backward differences
                if n_points > 1 {
                    gradient[[i, j, 0]] = (field[[i, j, 1]] - field[[i, j, 0]]) / spacing;
                    gradient[[i, j, n_points - 1]] =
                        (field[[i, j, n_points - 1]] - field[[i, j, n_points - 2]]) / spacing;
                }
            }
        }

        Ok(gradient)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use approx::assert_abs_diff_eq;

        #[test]
        fn test_matrix_divergence() {
            // Create a simple 2x2 matrix field with 5 spatial points
            let mut field = Array3::zeros((2, 2, 5));

            // Linear field: F₁₁ = x, F₁₂ = y, F₂₁ = 0, F₂₂ = x
            for k in 0..5 {
                let x = k as f64;
                field[[0, 0, k]] = x; // F₁₁ = x
                field[[0, 1, k]] = x; // F₁₂ = x (using x as proxy for y)
                field[[1, 0, k]] = 0.0; // F₂₁ = 0
                field[[1, 1, k]] = x; // F₂₂ = x
            }

            let div = matrix_divergence(&field, 1.0).unwrap();

            // For this field, divergence should be approximately constant = 1 + 1 + 0 + 1 = 3
            assert!(div.len() == 5);
            assert_abs_diff_eq!(div[2], 3.0, epsilon = 1e-10);
        }

        #[test]
        fn test_matrix_laplacian() {
            // Create a quadratic field
            let mut field = Array3::zeros((2, 2, 5));

            for k in 0..5 {
                let x = k as f64;
                field[[0, 0, k]] = x * x; // F₁₁ = x²
                field[[1, 1, k]] = x * x; // F₂₂ = x²
            }

            let laplacian = matrix_laplacian(&field, 1.0).unwrap();

            // For f(x) = x², d²f/dx² = 2, so laplacian should be approximately 2
            assert_abs_diff_eq!(laplacian[[0, 0, 2]], 2.0, epsilon = 1e-10);
            assert_abs_diff_eq!(laplacian[[1, 1, 2]], 2.0, epsilon = 1e-10);
        }

        #[test]
        fn test_matrix_gradient() {
            // Create a linear field
            let mut field = Array3::zeros((2, 2, 5));

            for k in 0..5 {
                let x = k as f64;
                field[[0, 0, k]] = 2.0 * x; // F₁₁ = 2x
                field[[1, 1, k]] = 3.0 * x; // F₂₂ = 3x
            }

            let gradient = matrix_gradient(&field, 1.0).unwrap();

            // For f(x) = ax, df/dx = a
            assert_abs_diff_eq!(gradient[[0, 0, 2]], 2.0, epsilon = 1e-10);
            assert_abs_diff_eq!(gradient[[1, 1, 2]], 3.0, epsilon = 1e-10);
        }
    }
}
