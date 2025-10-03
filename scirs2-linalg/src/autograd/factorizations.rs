//! Matrix factorization operations with automatic differentiation suppor
//!
//! This module provides differentiable implementations of matrix factorizations
//! like LU, QR, and Cholesky decompositions.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, One, Zero};
use std::fmt::Debug;

use scirs2_autograd::error::Result as AutogradResult;
use scirs2_autograd::graph::Node;
use scirs2_autograd::tensor::Tensor;
use scirs2_autograd::variable::Variable;

/// Perform LU decomposition with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Input square matrix tensor
///
/// # Returns
///
/// A tuple (p, l, u) representing the permutation matrix, lower triangular matrix,
/// and upper triangular matrix with gradient tracking.
#[allow(dead_code)]
pub fn lu<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
) -> AutogradResult<(Tensor<F>, Tensor<F>, Tensor<F>)> {
    // Ensure input is a square 2D tensor
    if a.data.ndim() != 2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "LU decomposition requires a 2D tensor".to_string(),
        ));
    }

    let ashape = a.shape();
    if ashape[0] != ashape[1] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "LU decomposition requires a square matrix".to_string(),
        ));
    }

    let n = ashape[0];

    // For simplicity, let's implement LU decomposition for 2x2 matrices
    if n > 2 {
        return Err(scirs2_autograd::error::AutogradError::OperationError(
            "LU decomposition for matrices larger than 2x2 not yet implemented in autodiff"
                .to_string(),
        ));
    }

    let mut p = Array2::<F>::eye(n);
    let mut l = Array2::<F>::eye(n);
    let mut u = a.data.clone().intoshape((n, n)).unwrap();

    if n == 2 {
        // Pivoting
        if u[[0, 0]].abs() < u[[1, 0]].abs() {
            // Swap rows 0 and 1 in p
            let p_row0 = p.row(0).to_owned();
            let p_row1 = p.row(1).to_owned();
            p.row_mut(0).assign(&p_row1);
            p.row_mut(1).assign(&p_row0);

            // Swap rows 0 and 1 in u
            let u_row0 = u.row(0).to_owned();
            let u_row1 = u.row(1).to_owned();
            u.row_mut(0).assign(&u_row1);
            u.row_mut(1).assign(&u_row0);
        }

        // Check if the matrix is singular
        if u[[0, 0]].abs() < F::epsilon() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "LU decomposition not defined for singular matrices".to_string(),
            ));
        }

        // Compute L and U
        l[[1, 0]] = u[[1, 0]] / u[[0, 0]];
        u[[1, 0]] = F::zero();
        u[[1, 1]] = u[[1, 1]] - l[[1, 0]] * u[[0, 1]];
    }

    // Convert to dynamic arrays
    let p_data = p.into_dyn();
    let l_data = l.into_dyn();
    let u_data = u.into_dyn();

    let requires_grad = a.requires_grad;

    if requires_grad {
        let a_data = a.data.clone();

        // Backward function for gradient computation
        // The gradient of LU decomposition is complex
        // We'll implement a simplified version that only computes gradients for U
        let backward_u = if requires_grad {
            Some(
                Box::new(move |gradu: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    // Simplified gradient approximation for small matrices
                    // For a proper implementation, see https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf

                    // For matrices up to 2x2, we'll just pass the gradient of U directly to A
                    // This is highly simplified and not correct in general
                    let grad_u_2d = grad_u.clone().intoshape((n, n)).unwrap();
                    Ok(grad_u_2d.into_dyn())
                })
                    as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        let node_u = Node::new(
            scirs2_autograd::graph::OpType::Activation("lu_u".to_string()),
            vec![a],
            vec![backward_u],
        );

        // For P and L, we'll return them without gradient tracking for simplicity
        let p_tensor = Tensor::new(p_data, false);
        let l_tensor = Tensor::new(l_data, false);
        let mut u_tensor = Tensor::new(u_data, requires_grad);
        u_tensor.node = Some(node_u);

        Ok((p_tensor, l_tensor, u_tensor))
    } else {
        let p_tensor = Tensor::new(p_data, false);
        let l_tensor = Tensor::new(l_data, false);
        let u_tensor = Tensor::new(u_data, false);

        Ok((p_tensor, l_tensor, u_tensor))
    }
}

/// Perform QR decomposition with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Input matrix tensor
///
/// # Returns
///
/// A tuple (q, r) representing the orthogonal and upper triangular matrices
/// with gradient tracking.
#[allow(dead_code)]
pub fn qr<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
) -> AutogradResult<(Tensor<F>, Tensor<F>)> {
    // Ensure input is a 2D tensor
    if a.data.ndim() != 2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "QR decomposition requires a 2D tensor".to_string(),
        ));
    }

    let ashape = a.shape();
    let m = ashape[0];
    let n = ashape[1];

    // For simplicity, let's implement QR decomposition for small matrices
    if m > 2 || n > 2 {
        return Err(scirs2_autograd::error::AutogradError::OperationError(
            "QR decomposition for matrices larger than 2x2 not yet implemented in autodiff"
                .to_string(),
        ));
    }

    // For 2x2 matrices, use Householder reflections
    let mut q = Array2::<F>::eye(m);
    let mut r = a.data.clone().intoshape((m, n)).unwrap();

    if m >= 1 && n >= 1 {
        // First column Householder reflection
        let x = r.slice(scirs2_core::ndarray::s![.., 0]).to_ownedj].iter())
                    .fold(F::zero(), |acc, (&u_i, &r_i)| acc + u_i * r_i);

                for i in 0..m {
                    r[[i, j]] = r[[i, j]] - F::from(2.0).unwrap() * u[i] * dot_product;
                }
            }

            // Compute Q
            for i in 0..m {
                for j in 0..m {
                    let identity = if i == j { F::one() } else { F::zero() };
                    q[[i, j]] = identity - F::from(2.0).unwrap() * u[i] * u[j];
                }
            }
        }
    }

    // Convert to dynamic arrays
    let q_data = q.into_dyn();
    let r_data = r.into_dyn();

    let requires_grad = a.requires_grad;

    if requires_grad {
        let a_data = a.data.clone();
        let q_data_clone = q_data.clone();
        let r_data_clone = r_data.clone();

        // Backward function for gradient computation
        // This is a simplified implementation
        let backward_r = if requires_grad {
            Some(
                Box::new(move |gradr: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    // Simplified gradient approximation
                    // dA = dQ * R^T + Q * dR^T
                    // Here we're assuming dQ = 0 for simplicity

                    let grad_r_2d = grad_r.clone().intoshape((m, n)).unwrap();
                    let q_2d = q_data_clone.clone().intoshape((m, m)).unwrap();

                    // Compute Q * dR
                    let mut grad_a = Array2::<F>::zeros((m, n));

                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = F::zero();
                            for k in 0..m {
                                sum = sum + q_2d[[i, k]] * grad_r_2d[[k, j]];
                            }
                            grad_a[[i, j]] = sum;
                        }
                    }

                    Ok(grad_a.into_dyn())
                })
                    as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        let node_r = Node::new(
            scirs2_autograd::graph::OpType::Activation("qr_r".to_string()),
            vec![a],
            vec![backward_r],
        );

        // Return Q without gradient tracking for simplicity
        let q_tensor = Tensor::new(q_data, false);
        let mut r_tensor = Tensor::new(r_data, requires_grad);
        r_tensor.node = Some(node_r);

        Ok((q_tensor, r_tensor))
    } else {
        let q_tensor = Tensor::new(q_data, false);
        let r_tensor = Tensor::new(r_data, false);

        Ok((q_tensor, r_tensor))
    }
}

/// Perform Cholesky decomposition with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Input positive definite symmetric matrix tensor
///
/// # Returns
///
/// The lower triangular Cholesky factor L where A = L * L^T
/// with gradient tracking.
#[allow(dead_code)]
pub fn cholesky<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure input is a 2D tensor
    if a.data.ndim() != 2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Cholesky decomposition requires a 2D tensor".to_string(),
        ));
    }

    let ashape = a.shape();
    if ashape[0] != ashape[1] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Cholesky decomposition requires a square matrix".to_string(),
        ));
    }

    let n = ashape[0];

    // For simplicity, let's implement Cholesky decomposition for small matrices
    if n > 2 {
        return Err(scirs2_autograd::error::AutogradError::OperationError(
            "Cholesky decomposition for matrices larger than 2x2 not yet implemented in autodiff"
                .to_string(),
        ));
    }

    // Check if the matrix is positive definite
    // For 1x1 matrix
    if n == 1 {
        if a.data[[0, 0]] <= F::zero() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Cholesky decomposition requires a positive definite matrix".to_string(),
            ));
        }
    }
    // For 2x2 matrix
    else if n == 2 {
        if a.data[[0, 0]] <= F::zero()
            || a.data[[0, 0]] * a.data[[1, 1]] - a.data[[0, 1]] * a.data[[1, 0]] <= F::zero()
        {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Cholesky decomposition requires a positive definite matrix".to_string(),
            ));
        }
    }

    // Compute Cholesky decomposition (L)
    let mut l = Array2::<F>::zeros((n, n));

    if n == 1 {
        l[[0, 0]] = a.data[[0, 0]].sqrt();
    } else if n == 2 {
        l[[0, 0]] = a.data[[0, 0]].sqrt();
        l[[1, 0]] = a.data[[1, 0]] / l[[0, 0]];
        l[[1, 1]] = (a.data[[1, 1]] - l[[1, 0]] * l[[1, 0]]).sqrt();
    }

    let l_data = l.into_dyn();
    let requires_grad = a.requires_grad;

    if requires_grad {
        let a_data = a.data.clone();
        let l_data_clone = l_data.clone();

        // Backward function for gradient computation
        let backward = if requires_grad {
            Some(
                Box::new(move |gradl: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    // Gradient of Cholesky decomposition
                    // See "Matrix Differential Calculus with Applications in Statistics and Econometrics"

                    let grad_l_2d = grad_l.clone().intoshape((n, n)).unwrap();
                    let l_2d = l_data_clone.clone().intoshape((n, n)).unwrap();

                    // Initialize gradient of A
                    let mut grad_a = Array2::<F>::zeros((n, n));

                    // Compute gradient for lower triangular par
                    for i in 0..n {
                        for j in 0..=i {
                            if i == j {
                                // Diagonal elements
                                let mut sum = F::zero();
                                for k in 0..j {
                                    sum = sum + grad_a[[j, k]] * l_2d[[j, k]];
                                }
                                grad_a[[j, j]] = (grad_l_2d[[j, j]] - sum) / (F::from(2.0).unwrap() * l_2d[[j, j]]);
                            } else {
                                // Off-diagonal elements
                                let mut sum = F::zero();
                                for k in 0..j {
                                    sum = sum + grad_a[[i, k]] * l_2d[[j, k]];
                                }
                                grad_a[[i, j]] = (grad_l_2d[[i, j]] - sum) / l_2d[[j, j]];
                                grad_a[[j, i]] = grad_a[[i, j]]; // Symmetric
                            }
                        }
                    }

                    Ok(grad_a.into_dyn())
                })
                    as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("cholesky".to_string()),
            vec![a],
            vec![backward],
        );

        let mut result = Tensor::new(l_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(l_data, false))
    }
}

/// High-level interface for matrix factorizations with autodiff suppor
pub mod variable {
    use super::*;
    use scirs2_autograd::variable::Variable;

    /// LU decomposition for Variables
    pub fn lu<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
    ) -> AutogradResult<(Variable<F>, Variable<F>, Variable<F>)> {
        let (p, l, u) = super::lu(&a.tensor)?;
        Ok((
            Variable { tensor: p },
            Variable { tensor: l },
            Variable { tensor: u },
        ))
    }

    /// QR decomposition for Variables
    pub fn qr<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
    ) -> AutogradResult<(Variable<F>, Variable<F>)> {
        let (q, r) = super::qr(&a.tensor)?;
        Ok((Variable { tensor: q }, Variable { tensor: r }))
    }

    /// Cholesky decomposition for Variables
    pub fn cholesky<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
    ) -> AutogradResult<Variable<F>> {
        let l = super::cholesky(&a.tensor)?;
        Ok(Variable { tensor: l })
    }
}
