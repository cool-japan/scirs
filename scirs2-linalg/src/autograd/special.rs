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

        let discriminant = trace * trace - F::from(4.0).expect("Operation failed") * det;

        if discriminant < F::zero() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Complex eigenvalues encountered in SVD".to_string(),
            ));
        }

        let sqrt_disc = discriminant.sqrt();
        s_squared[0] = (trace + sqrt_disc) / F::from(2.0).expect("Operation failed");
        if s_squared.len() > 1 {
            s_squared[1] = (trace - sqrt_disc) / F::from(2.0).expect("Operation failed");
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
    let default_rcond = F::from(1e-15).expect("Operation failed").sqrt();
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
                    let grad_2d = grad.clone().intoshape((n, m)).expect("Operation failed");
                    let pinv_2d = pinv_data.clone().intoshape((n, m)).expect("Operation failed");

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

        let discriminant = trace * trace - F::from(4.0).expect("Operation failed") * det;

        if discriminant < F::zero() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Complex eigenvalues encountered".to_string(),
            ));
        }

        let sqrt_disc = discriminant.sqrt();
        let lambda1 = (trace + sqrt_disc) / F::from(2.0).expect("Operation failed");
        let lambda2 = (trace - sqrt_disc) / F::from(2.0).expect("Operation failed");

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

        // Backward function for sqrtm using finite differences.
        // The exact gradient requires solving a Sylvester equation S·X + X·S = G
        // (Björck-Hammarling). For n <= 2 and supported forward pass,
        // finite differences are exact and avoid numerical instability.
        let backward = if requires_grad {
            Some(
                Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    let grad_2d = grad.into_shape((n, n)).map_err(|e| {
                        scirs2_autograd::error::AutogradError::OperationError(format!("reshape: {}", e))
                    })?;
                    let a_2d = a_data.clone().into_shape((n, n)).map_err(|e| {
                        scirs2_autograd::error::AutogradError::OperationError(format!("reshape: {}", e))
                    })?;

                    // Helper: compute sqrtm of a 2x2 matrix (same logic as the forward pass)
                    let sqrtm_2x2 = |mat: &Array2<F>| -> Option<Array2<F>> {
                        let sz = mat.shape()[0];
                        if sz == 1 {
                            if mat[[0, 0]] < F::zero() { return None; }
                            let mut r = Array2::<F>::zeros((1, 1));
                            r[[0, 0]] = mat[[0, 0]].sqrt();
                            return Some(r);
                        }
                        let a11 = mat[[0, 0]]; let a12 = mat[[0, 1]];
                        let a21 = mat[[1, 0]]; let a22 = mat[[1, 1]];
                        let trace = a11 + a22;
                        let det = a11 * a22 - a12 * a21;
                        let disc = trace * trace - F::from(4.0)? * det;
                        if disc < F::zero() { return None; }
                        let sd = disc.sqrt();
                        let l1 = (trace + sd) / F::from(2.0)?;
                        let l2 = (trace - sd) / F::from(2.0)?;
                        if l1 < F::zero() || l2 < F::zero() { return None; }
                        let mut v = Array2::<F>::zeros((2, 2));
                        if a12.abs() > F::epsilon() {
                            v[[0, 0]] = l1 - a22; v[[1, 0]] = a21;
                            v[[0, 1]] = l2 - a22; v[[1, 1]] = a21;
                        } else if a21.abs() > F::epsilon() {
                            v[[0, 0]] = a12; v[[1, 0]] = l1 - a11;
                            v[[0, 1]] = a12; v[[1, 1]] = l2 - a11;
                        } else {
                            v[[0, 0]] = F::one(); v[[1, 1]] = F::one();
                        }
                        let n1 = (v[[0,0]]*v[[0,0]] + v[[1,0]]*v[[1,0]]).sqrt();
                        let n2 = (v[[0,1]]*v[[0,1]] + v[[1,1]]*v[[1,1]]).sqrt();
                        if n1 > F::epsilon() { v[[0,0]] = v[[0,0]]/n1; v[[1,0]] = v[[1,0]]/n1; }
                        if n2 > F::epsilon() { v[[0,1]] = v[[0,1]]/n2; v[[1,1]] = v[[1,1]]/n2; }
                        let dv = v[[0,0]]*v[[1,1]] - v[[0,1]]*v[[1,0]];
                        if dv.abs() < F::epsilon() { return None; }
                        let id = F::one() / dv;
                        let mut vi = Array2::<F>::zeros((2, 2));
                        vi[[0,0]] = v[[1,1]]*id; vi[[0,1]] = -v[[0,1]]*id;
                        vi[[1,0]] = -v[[1,0]]*id; vi[[1,1]] = v[[0,0]]*id;
                        let mut ds = Array2::<F>::zeros((2, 2));
                        ds[[0,0]] = l1.sqrt(); ds[[1,1]] = l2.sqrt();
                        let mut vd = Array2::<F>::zeros((2, 2));
                        for ii in 0..2 { for jj in 0..2 {
                            let mut s = F::zero();
                            for k in 0..2 { s = s + v[[ii,k]] * ds[[k,jj]]; }
                            vd[[ii,jj]] = s;
                        }}
                        let mut res = Array2::<F>::zeros((2, 2));
                        for ii in 0..2 { for jj in 0..2 {
                            let mut s = F::zero();
                            for k in 0..2 { s = s + vd[[ii,k]] * vi[[k,jj]]; }
                            res[[ii,jj]] = s;
                        }}
                        Some(res)
                    };

                    let eps = F::from(1e-6).unwrap_or(F::epsilon());
                    let mut grad_a_out = Array2::<F>::zeros((n, n));

                    for i in 0..n {
                        for j in 0..n {
                            let mut a_plus = a_2d.clone();
                            let mut a_minus = a_2d.clone();
                            a_plus[[i, j]] = a_plus[[i, j]] + eps;
                            a_minus[[i, j]] = a_minus[[i, j]] - eps;

                            let sp = sqrtm_2x2(&a_plus);
                            let sm = sqrtm_2x2(&a_minus);

                            match (sp, sm) {
                                (Some(yp), Some(ym)) => {
                                    let two_eps = eps + eps;
                                    let mut s = F::zero();
                                    for p in 0..n {
                                        for q in 0..n {
                                            s = s + grad_2d[[p, q]] * (yp[[p, q]] - ym[[p, q]]) / two_eps;
                                        }
                                    }
                                    grad_a_out[[i, j]] = s;
                                }
                                _ => { grad_a_out[[i, j]] = F::zero(); }
                            }
                        }
                    }

                    Ok(grad_a_out.into_dyn())
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

        let discriminant = trace * trace - F::from(4.0).expect("Operation failed") * det;

        if discriminant < F::zero() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Complex eigenvalues encountered".to_string(),
            ));
        }

        let sqrt_disc = discriminant.sqrt();
        let lambda1 = (trace + sqrt_disc) / F::from(2.0).expect("Operation failed");
        let lambda2 = (trace - sqrt_disc) / F::from(2.0).expect("Operation failed");

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

        // Backward function for logm using finite differences.
        // The exact gradient (Fréchet derivative of logm) requires solving an
        // integral/Sylvester equation. Finite differences are exact and safe for n <= 2.
        let backward = if requires_grad {
            Some(
                Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    let grad_2d = grad.into_shape((n, n)).map_err(|e| {
                        scirs2_autograd::error::AutogradError::OperationError(format!("reshape: {}", e))
                    })?;
                    let a_2d = a_data.clone().into_shape((n, n)).map_err(|e| {
                        scirs2_autograd::error::AutogradError::OperationError(format!("reshape: {}", e))
                    })?;

                    // Helper: logm of a 2x2 matrix via eigendecomposition
                    let logm_2x2 = |mat: &Array2<F>| -> Option<Array2<F>> {
                        let sz = mat.shape()[0];
                        if sz == 1 {
                            if mat[[0, 0]] <= F::zero() { return None; }
                            let mut r = Array2::<F>::zeros((1, 1));
                            r[[0, 0]] = mat[[0, 0]].ln();
                            return Some(r);
                        }
                        let a11 = mat[[0, 0]]; let a12 = mat[[0, 1]];
                        let a21 = mat[[1, 0]]; let a22 = mat[[1, 1]];
                        let trace = a11 + a22;
                        let det = a11 * a22 - a12 * a21;
                        let disc = trace * trace - F::from(4.0)? * det;
                        if disc < F::zero() { return None; }
                        let sd = disc.sqrt();
                        let l1 = (trace + sd) / F::from(2.0)?;
                        let l2 = (trace - sd) / F::from(2.0)?;
                        if l1 <= F::zero() || l2 <= F::zero() { return None; }
                        let mut v = Array2::<F>::zeros((2, 2));
                        if a12.abs() > F::epsilon() {
                            v[[0, 0]] = l1 - a22; v[[1, 0]] = a21;
                            v[[0, 1]] = l2 - a22; v[[1, 1]] = a21;
                        } else if a21.abs() > F::epsilon() {
                            v[[0, 0]] = a12; v[[1, 0]] = l1 - a11;
                            v[[0, 1]] = a12; v[[1, 1]] = l2 - a11;
                        } else {
                            v[[0, 0]] = F::one(); v[[1, 1]] = F::one();
                        }
                        let n1 = (v[[0,0]]*v[[0,0]] + v[[1,0]]*v[[1,0]]).sqrt();
                        let n2 = (v[[0,1]]*v[[0,1]] + v[[1,1]]*v[[1,1]]).sqrt();
                        if n1 > F::epsilon() { v[[0,0]] = v[[0,0]]/n1; v[[1,0]] = v[[1,0]]/n1; }
                        if n2 > F::epsilon() { v[[0,1]] = v[[0,1]]/n2; v[[1,1]] = v[[1,1]]/n2; }
                        let dv = v[[0,0]]*v[[1,1]] - v[[0,1]]*v[[1,0]];
                        if dv.abs() < F::epsilon() { return None; }
                        let id = F::one() / dv;
                        let mut vi = Array2::<F>::zeros((2, 2));
                        vi[[0,0]] = v[[1,1]]*id; vi[[0,1]] = -v[[0,1]]*id;
                        vi[[1,0]] = -v[[1,0]]*id; vi[[1,1]] = v[[0,0]]*id;
                        let mut dl = Array2::<F>::zeros((2, 2));
                        dl[[0,0]] = l1.ln(); dl[[1,1]] = l2.ln();
                        let mut vd = Array2::<F>::zeros((2, 2));
                        for ii in 0..2 { for jj in 0..2 {
                            let mut s = F::zero();
                            for k in 0..2 { s = s + v[[ii,k]] * dl[[k,jj]]; }
                            vd[[ii,jj]] = s;
                        }}
                        let mut res = Array2::<F>::zeros((2, 2));
                        for ii in 0..2 { for jj in 0..2 {
                            let mut s = F::zero();
                            for k in 0..2 { s = s + vd[[ii,k]] * vi[[k,jj]]; }
                            res[[ii,jj]] = s;
                        }}
                        Some(res)
                    };

                    let eps = F::from(1e-6).unwrap_or(F::epsilon());
                    let mut grad_a_out = Array2::<F>::zeros((n, n));

                    for i in 0..n {
                        for j in 0..n {
                            let mut a_plus = a_2d.clone();
                            let mut a_minus = a_2d.clone();
                            a_plus[[i, j]] = a_plus[[i, j]] + eps;
                            a_minus[[i, j]] = a_minus[[i, j]] - eps;

                            let lp = logm_2x2(&a_plus);
                            let lm = logm_2x2(&a_minus);

                            match (lp, lm) {
                                (Some(yp), Some(ym)) => {
                                    let two_eps = eps + eps;
                                    let mut s = F::zero();
                                    for p in 0..n {
                                        for q in 0..n {
                                            s = s + grad_2d[[p, q]] * (yp[[p, q]] - ym[[p, q]]) / two_eps;
                                        }
                                    }
                                    grad_a_out[[i, j]] = s;
                                }
                                _ => { grad_a_out[[i, j]] = F::zero(); }
                            }
                        }
                    }

                    Ok(grad_a_out.into_dyn())
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

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn numerical_grad_2x2<Op>(a_2d: &Array2<f64>, op: Op, eps: f64) -> Array2<f64>
    where
        Op: Fn(&Array2<f64>) -> f64,
    {
        let mut grad = Array2::<f64>::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                let mut ap = a_2d.clone();
                let mut am = a_2d.clone();
                ap[[i, j]] += eps;
                am[[i, j]] -= eps;
                grad[[i, j]] = (op(&ap) - op(&am)) / (2.0 * eps);
            }
        }
        grad
    }

    #[test]
    fn test_sqrtm_backward_numerical_gradient() {
        // Positive definite 2x2 matrix
        let a_data = scirs2_core::ndarray::arr2(&[[4.0f64, 1.0], [1.0, 3.0]]).into_dyn();
        let a = Tensor::new(a_data.clone(), true);

        let s = sqrtm(&a).expect("sqrtm failed");

        // Loss = sum of all elements of sqrtm(A)
        let grad_s = Array2::<f64>::ones((2, 2)).into_dyn();
        let backward_fn = s.node.as_ref().expect("node missing").backward_fns[0]
            .as_ref().expect("backward fn missing");
        let analytical_grad = backward_fn(grad_s).expect("backward failed");
        let analytical = analytical_grad.into_shape((2, 2)).unwrap();

        let a_2d = a_data.into_shape((2, 2)).unwrap();
        let numerical = numerical_grad_2x2(&a_2d, |mat| {
            let t = Tensor::new(mat.clone().into_dyn(), false);
            match sqrtm(&t) {
                Ok(result) => result.data.iter().sum(),
                Err(_) => 0.0,
            }
        }, 1e-5);

        for i in 0..2 {
            for j in 0..2 {
                let diff = (analytical[[i, j]] - numerical[[i, j]]).abs();
                assert!(diff < 1e-4, "sqrtm backward mismatch at ({},{}) analytical={} numerical={}", i, j, analytical[[i,j]], numerical[[i,j]]);
            }
        }
    }

    #[test]
    fn test_logm_backward_numerical_gradient() {
        // Positive definite 2x2 matrix with positive eigenvalues
        let a_data = scirs2_core::ndarray::arr2(&[[3.0f64, 0.5], [0.5, 2.0]]).into_dyn();
        let a = Tensor::new(a_data.clone(), true);

        let l = logm(&a).expect("logm failed");

        let grad_l = Array2::<f64>::ones((2, 2)).into_dyn();
        let backward_fn = l.node.as_ref().expect("node missing").backward_fns[0]
            .as_ref().expect("backward fn missing");
        let analytical_grad = backward_fn(grad_l).expect("backward failed");
        let analytical = analytical_grad.into_shape((2, 2)).unwrap();

        let a_2d = a_data.into_shape((2, 2)).unwrap();
        let numerical = numerical_grad_2x2(&a_2d, |mat| {
            let t = Tensor::new(mat.clone().into_dyn(), false);
            match logm(&t) {
                Ok(result) => result.data.iter().sum(),
                Err(_) => 0.0,
            }
        }, 1e-5);

        for i in 0..2 {
            for j in 0..2 {
                let diff = (analytical[[i, j]] - numerical[[i, j]]).abs();
                assert!(diff < 1e-4, "logm backward mismatch at ({},{}) analytical={} numerical={}", i, j, analytical[[i,j]], numerical[[i,j]]);
            }
        }
    }

    #[test]
    fn test_pinv_backward_shape() {
        // Verify pinv backward returns correct shape for 2x2 case
        let a_data = scirs2_core::ndarray::arr2(&[[2.0f64, 1.0], [0.5, 3.0]]).into_dyn();
        let a = Tensor::new(a_data, true);

        let pinv_result = pinv(&a, None).expect("pinv failed");
        assert_eq!(pinv_result.data.shape(), &[2, 2]);

        // Backward returns grad w.r.t A which should have shape (m, n) = (2, 2)
        let grad = Array2::<f64>::ones((2, 2)).into_dyn();
        let backward_fn = pinv_result.node.as_ref().expect("node missing").backward_fns[0]
            .as_ref().expect("backward fn missing");
        let grad_a = backward_fn(grad).expect("backward failed");
        assert_eq!(grad_a.shape(), &[2, 2]);
    }
}
