//! Special matrix functions with automatic differentiation suppor
//!
//! This module provides implementations of special matrix operations and
//! functions with gradient tracking.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, One, Zero};
use std::fmt::Debug;

use scirs2_autograd::error::Result as AutogradResult;
use scirs2_autograd::graph::Node;
use scirs2_autograd::tensor::Tensor;
use scirs2_autograd::variable::Variable;

/// Compute the pseudo-inverse of a matrix using SVD with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Input matrix tensor
/// * `rcond` - Cutoff for small singular values
///
/// # Returns
///
/// A new tensor containing the pseudo-inverse with gradient tracking.
#[allow(dead_code)]
pub fn pinv<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
    rcond: Option<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure input is a 2D tensor
    if a.data.ndim() != 2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Pseudo-inverse requires a 2D tensor".to_string(),
        ));
    }

    let ashape = a.shape();
    let m = ashape[0];
    let n = ashape[1];

    // For simplicity, only implement for small matrices
    if m > 2 || n > 2 {
        return Err(scirs2_autograd::error::AutogradError::OperationError(
            "Pseudo-inverse for matrices larger than 2x2 not yet implemented in autodiff"
                .to_string(),
        ));
    }

    // Compute SVD: A = U * Σ * V^T
    // Simplified implementation for small matrices
    let a_t_a = Array2::<F>::from_shape_fn((n, n), |ij| {
        let (i, j) = (ij.0, ij.1);
        let mut sum = F::zero();
        for k in 0..m {
            sum = sum + a.data[[k, i]] * a.data[[k, j]];
        }
        sum
    });

    let a_a_t = Array2::<F>::from_shape_fn((m, m), |ij| {
        let (i, j) = (ij.0, ij.1);
        let mut sum = F::zero();
        for k in 0..n {
            sum = sum + a.data[[i, k]] * a.data[[j, k]];
        }
        sum
    });

    // Compute eigendecomposition of a_t_a to get V and singular values
    let mut s_squared = Array1::<F>::zeros(std::cmp::min(m, n));
    let mut v = Array2::<F>::zeros((n, n));

    // Simplified eigendecomposition for small matrices
    if n == 1 {
        s_squared[0] = a_t_a[[0, 0]];
        v[[0, 0]] = F::one();
    } else if n == 2 {
        let a11 = a_t_a[[0, 0]];
        let a12 = a_t_a[[0, 1]];
        let a21 = a_t_a[[1, 0]];
        let a22 = a_t_a[[1, 1]];

        let trace = a11 + a22;
        let det = a11 * a22 - a12 * a21;

        let discriminant = trace * trace - F::from(4.0).unwrap() * det;

        if discriminant < F::zero() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Complex eigenvalues encountered in SVD".to_string(),
            ));
        }

        let sqrt_disc = discriminant.sqrt();
        s_squared[0] = (trace + sqrt_disc) / F::from(2.0).unwrap();
        if s_squared.len() > 1 {
            s_squared[1] = (trace - sqrt_disc) / F::from(2.0).unwrap();
        }

        // Compute eigenvectors
        if a12.abs() > F::epsilon() {
            v[[0, 0]] = s_squared[0] - a22;
            v[[1, 0]] = a21;
            if n > 1 {
                v[[0, 1]] = s_squared[1] - a22;
                v[[1, 1]] = a21;
            }
        } else if a21.abs() > F::epsilon() {
            v[[0, 0]] = a12;
            v[[1, 0]] = s_squared[0] - a11;
            if n > 1 {
                v[[0, 1]] = a12;
                v[[1, 1]] = s_squared[1] - a11;
            }
        } else {
            // Diagonal matrix
            v[[0, 0]] = F::one();
            v[[1, 0]] = F::zero();
            if n > 1 {
                v[[0, 1]] = F::zero();
                v[[1, 1]] = F::one();
            }
        }

        // Normalize eigenvectors
        for j in 0..n {
            let mut norm_sq = F::zero();
            for i in 0..n {
                norm_sq = norm_sq + v[[i, j]] * v[[i, j]];
            }
            let norm = norm_sq.sqrt();

            if norm > F::epsilon() {
                for i in 0..n {
                    v[[i, j]] = v[[i, j]] / norm;
                }
            }
        }
    }

    // Compute singular values
    let mut s = Array1::<F>::zeros(std::cmp::min(m, n));
    for i in 0..s.len() {
        s[i] = s_squared[i].sqrt();
    }

    // Compute U = A * V * Σ^(-1)
    let mut u = Array2::<F>::zeros((m, std::cmp::min(m, n)));

    for j in 0..std::cmp::min(m, n) {
        if s[j] > F::epsilon() {
            for i in 0..m {
                let mut sum = F::zero();
                for k in 0..n {
                    sum = sum + a.data[[i, k]] * v[[k, j]];
                }
                u[[i, j]] = sum / s[j];
            }
        } else {
            // Handle zero singular values
            if j < m {
                u[[j, j]] = F::one();
            }
        }
    }

    // Apply cutoff for small singular values
    let default_rcond = F::from(1e-15).unwrap().sqrt();
    let rcond_val = rcond.unwrap_or(default_rcond);
    let max_s = s.fold(F::zero(), |a, &b| if a > b { a } else { b });
    let cutoff = max_s * rcond_val;

    // Compute the pseudo-inverse: A^+ = V * Σ^+ * U^T
    let mut s_inv = Array1::<F>::zeros(s.len());
    for i in 0..s.len() {
        if s[i] > cutoff {
            s_inv[i] = F::one() / s[i];
        } else {
            s_inv[i] = F::zero();
        }
    }

    // Compute A^+ = V * Σ^+ * U^T
    let mut result = Array2::<F>::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            let mut sum = F::zero();
            for k in 0..std::cmp::min(m, n) {
                sum = sum + v[[i, k]] * s_inv[k] * u[[j, k]];
            }
            result[[i, j]] = sum;
        }
    }

    let result_data = result.into_dyn();
    let requires_grad = a.requires_grad;

    if requires_grad {
        let a_data = a.data.clone();
        let pinv_data = result_data.clone();

        // Backward function for gradient computation
        // This is a simplified approximation of the true gradient of pseudo-inverse
        let backward = if requires_grad {
            Some(
                Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    // Simplified gradient approximation
                    // For a proper implementation, see the paper:
                    // "Matrix Backpropagation for Deep Networks with Structured Layers"
                    let grad_2d = grad.clone().intoshape((n, m)).unwrap();
                    let pinv_2d = pinv_data.clone().intoshape((n, m)).unwrap();

                    // Approximate gradient: -A^+ * dL/dA^+ * A^+
                    let mut result = Array2::<F>::zeros((m, n));

                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = F::zero();
                            for k in 0..n {
                                for l in 0..m {
                                    sum = sum + (-pinv_2d[[k, i]] * grad_2d[[k, l]] * pinv_2d[[j, l]]);
                                }
                            }
                            result[[i, j]] = sum;
                        }
                    }

                    Ok(result.into_dyn())
                })
                    as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("pinv".to_string()),
            vec![a],
            vec![backward],
        );

        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Compute matrix square root with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Input square matrix tensor
///
/// # Returns
///
/// A new tensor containing the matrix square root with gradient tracking.
#[allow(dead_code)]
pub fn sqrtm<F: Float + Debug + Send + Sync + 'static>(a: &Tensor<F>) -> AutogradResult<Tensor<F>> {
    // Ensure input is a square 2D tensor
    if a.data.ndim() != 2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Matrix square root requires a 2D tensor".to_string(),
        ));
    }

    let ashape = a.shape();
    if ashape[0] != ashape[1] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Matrix square root requires a square matrix".to_string(),
        ));
    }

    let n = ashape[0];

    // For simplicity, only implement for small matrices
    if n > 2 {
        return Err(scirs2_autograd::error::AutogradError::OperationError(
            "Matrix square root for matrices larger than 2x2 not yet implemented in autodiff"
                .to_string(),
        ));
    }

    let mut result = Array2::<F>::zeros((n, n));

    if n == 1 {
        // For 1x1 matrices, simply take the square root of the elemen
        if a.data[[0, 0]] < F::zero() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Cannot compute square root of negative value".to_string(),
            ));
        }

        result[[0, 0]] = a.data[[0, 0]].sqrt();
    } else if n == 2 {
        // For 2x2 matrices, use the eigendecomposition approach: A^(1/2) = V * D^(1/2) * V^(-1)
        // where D is the diagonal matrix of eigenvalues and V is the matrix of eigenvectors

        // Compute eigendecomposition
        let a11 = a.data[[0, 0]];
        let a12 = a.data[[0, 1]];
        let a21 = a.data[[1, 0]];
        let a22 = a.data[[1, 1]];

        let trace = a11 + a22;
        let det = a11 * a22 - a12 * a21;

        let discriminant = trace * trace - F::from(4.0).unwrap() * det;

        if discriminant < F::zero() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Complex eigenvalues encountered".to_string(),
            ));
        }

        let sqrt_disc = discriminant.sqrt();
        let lambda1 = (trace + sqrt_disc) / F::from(2.0).unwrap();
        let lambda2 = (trace - sqrt_disc) / F::from(2.0).unwrap();

        // Check for negative eigenvalues
        if lambda1 < F::zero() || lambda2 < F::zero() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Matrix square root not defined for matrices with negative eigenvalues".to_string(),
            ));
        }

        // Compute eigenvectors
        let mut v = Array2::<F>::zeros((n, n));

        if a12.abs() > F::epsilon() {
            v[[0, 0]] = lambda1 - a22;
            v[[1, 0]] = a21;
            v[[0, 1]] = lambda2 - a22;
            v[[1, 1]] = a21;
        } else if a21.abs() > F::epsilon() {
            v[[0, 0]] = a12;
            v[[1, 0]] = lambda1 - a11;
            v[[0, 1]] = a12;
            v[[1, 1]] = lambda2 - a11;
        } else {
            // Diagonal matrix
            v[[0, 0]] = F::one();
            v[[1, 0]] = F::zero();
            v[[0, 1]] = F::zero();
            v[[1, 1]] = F::one();
        }

        // Normalize eigenvectors
        let norm1 = (v[[0, 0]] * v[[0, 0]] + v[[1, 0]] * v[[1, 0]]).sqrt();
        let norm2 = (v[[0, 1]] * v[[0, 1]] + v[[1, 1]] * v[[1, 1]]).sqrt();

        if norm1 > F::epsilon() {
            v[[0, 0]] = v[[0, 0]] / norm1;
            v[[1, 0]] = v[[1, 0]] / norm1;
        }

        if norm2 > F::epsilon() {
            v[[0, 1]] = v[[0, 1]] / norm2;
            v[[1, 1]] = v[[1, 1]] / norm2;
        }

        // Compute V^(-1)
        let det_v = v[[0, 0]] * v[[1, 1]] - v[[0, 1]] * v[[1, 0]];
        if det_v.abs() < F::epsilon() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Eigenvector matrix is singular".to_string(),
            ));
        }

        let mut v_inv = Array2::<F>::zeros((n, n));
        let inv_det_v = F::one() / det_v;

        v_inv[[0, 0]] = v[[1, 1]] * inv_det_v;
        v_inv[[0, 1]] = -v[[0, 1]] * inv_det_v;
        v_inv[[1, 0]] = -v[[1, 0]] * inv_det_v;
        v_inv[[1, 1]] = v[[0, 0]] * inv_det_v;

        // Construct diagonal matrix D^(1/2)
        let mut d_sqrt = Array2::<F>::zeros((n, n));
        d_sqrt[[0, 0]] = lambda1.sqrt();
        d_sqrt[[1, 1]] = lambda2.sqrt();

        // Compute A^(1/2) = V * D^(1/2) * V^(-1)
        let v_d_sqrt = Array2::<F>::from_shape_fn((n, n), |ij| {
            let (i, j) = (ij.0, ij.1);
            let mut sum = F::zero();
            for k in 0..n {
                sum = sum + v[[i, k]] * d_sqrt[[k, j]];
            }
            sum
        });

        result = Array2::<F>::from_shape_fn((n, n), |ij| {
            let (i, j) = (ij.0, ij.1);
            let mut sum = F::zero();
            for k in 0..n {
                sum = sum + v_d_sqrt[[i, k]] * v_inv[[k, j]];
            }
            sum
        });
    }

    let result_data = result.into_dyn();
    let requires_grad = a.requires_grad;

    if requires_grad {
        let a_data = a.data.clone();
        let sqrtm_data = result_data.clone();

        // Backward function for gradient computation
        let backward = if requires_grad {
            Some(
                Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    // For sqrtm, the gradient involves solving a Sylvester equation
                    // For simplicity, we'll use a crude approximation
                    let grad_2d = grad.clone().intoshape((n, n)).unwrap();
                    let sqrtm_2d = sqrtm_data.clone().intoshape((n, n)).unwrap();

                    // Approximate solution: Q = grad * sqrtm^(-1) / 2
                    let sqrtm_inv = if n == 1 {
                        let mut inv = Array2::<F>::zeros((1, 1));
                        inv[[0, 0]] = F::one() / sqrtm_2d[[0, 0]];
                        inv
                    } else {
                        // For 2x2, compute inverse directly
                        let det = sqrtm_2d[[0, 0]] * sqrtm_2d[[1, 1]] - sqrtm_2d[[0, 1]] * sqrtm_2d[[1, 0]];
                        if det.abs() < F::epsilon() {
                            // Return a zero gradient if sqrtm is singular
                            return Ok(Array2::<F>::zeros((n, n)).into_dyn());
                        }

                        let mut inv = Array2::<F>::zeros((2, 2));
                        let inv_det = F::one() / det;

                        inv[[0, 0]] = sqrtm_2d[[1, 1]] * inv_det;
                        inv[[0, 1]] = -sqrtm_2d[[0, 1]] * inv_det;
                        inv[[1, 0]] = -sqrtm_2d[[1, 0]] * inv_det;
                        inv[[1, 1]] = sqrtm_2d[[0, 0]] * inv_det;
                        inv
                    };

                    // Q = grad * sqrtm^(-1) / 2
                    let mut q = Array2::<F>::zeros((n, n));
                    for i in 0..n {
                        for j in 0..n {
                            let mut sum = F::zero();
                            for k in 0..n {
                                sum = sum + grad_2d[[i, k]] * sqrtm_inv[[k, j]];
                            }
                            q[[i, j]] = sum / F::from(2.0).unwrap();
                        }
                    }

                    // Approximate gradient: sqrtm^(-T) * Q^T + Q * sqrtm^(-1)
                    let sqrtm_inv_t = sqrtm_inv.t().to_owned();
                    let q_t = q.t().to_owned();

                    let mut result = Array2::<F>::zeros((n, n));

                    for i in 0..n {
                        for j in 0..n {
                            let mut sum1 = F::zero();
                            let mut sum2 = F::zero();

                            for k in 0..n {
                                sum1 = sum1 + sqrtm_inv_t[[i, k]] * q_t[[k, j]];
                                sum2 = sum2 + q[[i, k]] * sqrtm_inv[[k, j]];
                            }

                            result[[i, j]] = sum1 + sum2;
                        }
                    }

                    Ok(result.into_dyn())
                })
                    as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("sqrtm".to_string()),
            vec![a],
            vec![backward],
        );

        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Compute matrix logarithm with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Input square matrix tensor
///
/// # Returns
///
/// A new tensor containing the matrix logarithm with gradient tracking.
#[allow(dead_code)]
pub fn logm<F: Float + Debug + Send + Sync + 'static>(a: &Tensor<F>) -> AutogradResult<Tensor<F>> {
    // Ensure input is a square 2D tensor
    if a.data.ndim() != 2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Matrix logarithm requires a 2D tensor".to_string(),
        ));
    }

    let ashape = a.shape();
    if ashape[0] != ashape[1] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Matrix logarithm requires a square matrix".to_string(),
        ));
    }

    let n = ashape[0];

    // For simplicity, only implement for small matrices
    if n > 2 {
        return Err(scirs2_autograd::error::AutogradError::OperationError(
            "Matrix logarithm for matrices larger than 2x2 not yet implemented in autodiff"
                .to_string(),
        ));
    }

    let mut result = Array2::<F>::zeros((n, n));

    if n == 1 {
        // For 1x1 matrices, simply take the logarithm of the elemen
        if a.data[[0, 0]] <= F::zero() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Cannot compute logarithm of non-positive value".to_string(),
            ));
        }

        result[[0, 0]] = a.data[[0, 0]].ln();
    } else if n == 2 {
        // For 2x2 matrices, use the eigendecomposition approach: log(A) = V * log(D) * V^(-1)
        // where D is the diagonal matrix of eigenvalues and V is the matrix of eigenvectors

        // Compute eigendecomposition
        let a11 = a.data[[0, 0]];
        let a12 = a.data[[0, 1]];
        let a21 = a.data[[1, 0]];
        let a22 = a.data[[1, 1]];

        let trace = a11 + a22;
        let det = a11 * a22 - a12 * a21;

        let discriminant = trace * trace - F::from(4.0).unwrap() * det;

        if discriminant < F::zero() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Complex eigenvalues encountered".to_string(),
            ));
        }

        let sqrt_disc = discriminant.sqrt();
        let lambda1 = (trace + sqrt_disc) / F::from(2.0).unwrap();
        let lambda2 = (trace - sqrt_disc) / F::from(2.0).unwrap();

        // Check for non-positive eigenvalues
        if lambda1 <= F::zero() || lambda2 <= F::zero() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Matrix logarithm not defined for matrices with non-positive eigenvalues"
                    .to_string(),
            ));
        }

        // Compute eigenvectors
        let mut v = Array2::<F>::zeros((n, n));

        if a12.abs() > F::epsilon() {
            v[[0, 0]] = lambda1 - a22;
            v[[1, 0]] = a21;
            v[[0, 1]] = lambda2 - a22;
            v[[1, 1]] = a21;
        } else if a21.abs() > F::epsilon() {
            v[[0, 0]] = a12;
            v[[1, 0]] = lambda1 - a11;
            v[[0, 1]] = a12;
            v[[1, 1]] = lambda2 - a11;
        } else {
            // Diagonal matrix
            v[[0, 0]] = F::one();
            v[[1, 0]] = F::zero();
            v[[0, 1]] = F::zero();
            v[[1, 1]] = F::one();
        }

        // Normalize eigenvectors
        let norm1 = (v[[0, 0]] * v[[0, 0]] + v[[1, 0]] * v[[1, 0]]).sqrt();
        let norm2 = (v[[0, 1]] * v[[0, 1]] + v[[1, 1]] * v[[1, 1]]).sqrt();

        if norm1 > F::epsilon() {
            v[[0, 0]] = v[[0, 0]] / norm1;
            v[[1, 0]] = v[[1, 0]] / norm1;
        }

        if norm2 > F::epsilon() {
            v[[0, 1]] = v[[0, 1]] / norm2;
            v[[1, 1]] = v[[1, 1]] / norm2;
        }

        // Compute V^(-1)
        let det_v = v[[0, 0]] * v[[1, 1]] - v[[0, 1]] * v[[1, 0]];
        if det_v.abs() < F::epsilon() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Eigenvector matrix is singular".to_string(),
            ));
        }

        let mut v_inv = Array2::<F>::zeros((n, n));
        let inv_det_v = F::one() / det_v;

        v_inv[[0, 0]] = v[[1, 1]] * inv_det_v;
        v_inv[[0, 1]] = -v[[0, 1]] * inv_det_v;
        v_inv[[1, 0]] = -v[[1, 0]] * inv_det_v;
        v_inv[[1, 1]] = v[[0, 0]] * inv_det_v;

        // Construct diagonal matrix log(D)
        let mut d_log = Array2::<F>::zeros((n, n));
        d_log[[0, 0]] = lambda1.ln();
        d_log[[1, 1]] = lambda2.ln();

        // Compute log(A) = V * log(D) * V^(-1)
        let v_d_log = Array2::<F>::from_shape_fn((n, n), |ij| {
            let (i, j) = (ij.0, ij.1);
            let mut sum = F::zero();
            for k in 0..n {
                sum = sum + v[[i, k]] * d_log[[k, j]];
            }
            sum
        });

        result = Array2::<F>::from_shape_fn((n, n), |ij| {
            let (i, j) = (ij.0, ij.1);
            let mut sum = F::zero();
            for k in 0..n {
                sum = sum + v_d_log[[i, k]] * v_inv[[k, j]];
            }
            sum
        });
    }

    let result_data = result.into_dyn();
    let requires_grad = a.requires_grad;

    if requires_grad {
        let a_data = a.data.clone();
        let logm_data = result_data.clone();

        // Backward function for gradient computation
        // The gradient of matrix logarithm is complex and involves the Fréchet derivative
        let backward = if requires_grad {
            Some(
                Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    // For simplicity, we'll use a crude approximation for small matrices
                    let grad_2d = grad.clone().intoshape((n, n)).unwrap();

                    // Crude approximation: grad_a ≈ A^(-1) * grad
                    let a_inv = if n == 1 {
                        let mut inv = Array2::<F>::zeros((1, 1));
                        inv[[0, 0]] = F::one() / a_data[[0, 0]];
                        inv
                    } else {
                        // For 2x2, compute inverse directly
                        let det = a_data[[0, 0]] * a_data[[1, 1]] - a_data[[0, 1]] * a_data[[1, 0]];
                        if det.abs() < F::epsilon() {
                            // Return a zero gradient if matrix is singular
                            return Ok(Array2::<F>::zeros((n, n)).into_dyn());
                        }

                        let mut inv = Array2::<F>::zeros((2, 2));
                        let inv_det = F::one() / det;

                        inv[[0, 0]] = a_data[[1, 1]] * inv_det;
                        inv[[0, 1]] = -a_data[[0, 1]] * inv_det;
                        inv[[1, 0]] = -a_data[[1, 0]] * inv_det;
                        inv[[1, 1]] = a_data[[0, 0]] * inv_det;
                        inv
                    };

                    // Compute A^(-1) * grad
                    let mut result = Array2::<F>::zeros((n, n));

                    for i in 0..n {
                        for j in 0..n {
                            let mut sum = F::zero();
                            for k in 0..n {
                                sum = sum + a_inv[[i, k]] * grad_2d[[k, j]];
                            }
                            result[[i, j]] = sum;
                        }
                    }

                    Ok(result.into_dyn())
                })
                    as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("logm".to_string()),
            vec![a],
            vec![backward],
        );

        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// High-level interface for special matrix functions with autodiff suppor
pub mod variable {
    use super::*;
    use scirs2_autograd::variable::Variable;

    /// Pseudo-inverse for Variables
    pub fn pinv<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
        rcond: Option<F>,
    ) -> AutogradResult<Variable<F>> {
        let result_tensor = super::pinv(&a.tensor, rcond)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    /// Matrix square root for Variables
    pub fn sqrtm<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
    ) -> AutogradResult<Variable<F>> {
        let result_tensor = super::sqrtm(&a.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    /// Matrix logarithm for Variables
    pub fn logm<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
    ) -> AutogradResult<Variable<F>> {
        let result_tensor = super::logm(&a.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }
}
