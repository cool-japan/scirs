//! Batch matrix operations with automatic differentiation support
//!
//! This module provides batch operations on tensors that support gradient tracking.
//! Batch operations apply the same operation to multiple matrices or vectors at once.

use scirs2_core::ndarray::{Array, ArrayView4, Axis, IxDyn};
use scirs2_core::numeric::{Float, One, Zero};
use std::fmt::Debug;

use scirs2_autograd::error::Result as AutogradResult;
use scirs2_autograd::graph::Node;
use scirs2_autograd::tensor::Tensor;
use scirs2_autograd::variable::Variable;

/// Perform batch matrix multiplication with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - First tensor with batch dimensions, shape (..., n, m)
/// * `b` - Second tensor with batch dimensions, shape (..., m, p)
///
/// # Returns
///
/// A new tensor of shape (..., n, p) containing the batch matrix products.
#[allow(dead_code)]
pub fn batch_matmul<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
    b: &Tensor<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure input tensors have at least 3 dimensions (batch dims + matrix dims)
    if a.data.ndim() < 3 || b.data.ndim() < 3 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Batch matrix multiplication requires at least 3D tensors (batch dim + 2D matrices)"
                .to_string(),
        ));
    }

    let ashape = a.shape();
    let bshape = b.shape();

    // Check that the matrix dimensions are compatible for matmul
    if ashape[ashape.len() - 1] != bshape[bshape.len() - 2] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            format!(
                "Matrix multiplication dimension mismatch: {:?} and {:?}",
                ashape, bshape
            ),
        ));
    }

    // Check that batch dimensions match
    let a_batch_dims = &ashape[..ashape.len() - 2];
    let b_batch_dims = &bshape[..bshape.len() - 2];

    if a_batch_dims != b_batch_dims {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            format!(
                "Batch dimensions mismatch: {:?} and {:?}",
                a_batch_dims, b_batch_dims
            ),
        ));
    }

    // For simplicity, let's implement a special case for 3D tensors (batch of matrices)
    // A complete implementation would handle arbitrary batch dimensions
    if a.data.ndim() == 3 && b.data.ndim() == 3 {
        let batchsize = ashape[0];
        let n = ashape[1];
        let m = ashape[2];
        let p = bshape[2];

        // Compute batch matmul
        let mut result_data = Array::zeros((batchsize, n, p));

        for batch_idx in 0..batchsize {
            for i in 0..n {
                for j in 0..p {
                    let mut sum = F::zero();
                    for k in 0..m {
                        sum = sum + a.data[[batch_idx, i, k]] * b.data[[batch_idx, k, j]];
                    }
                    result_data[[batch_idx, i, j]] = sum;
                }
            }
        }

        let result_data = result_data.into_dyn();
        let requires_grad = a.requires_grad || b.requires_grad;

        if requires_grad {
            let a_data = a.data.clone();
            let b_data = b.data.clone();

            // Backward function for the first tensor
            let backward_a = if a.requires_grad {
                Some(Box::new(
                    move |grad: Array<F, IxDyn>| -> AutogradResult<Array<F, IxDyn>> {
                        // For 3D tensors: dL/dA[b,i,k] = sum_j dL/dC[b,i,j] * B[b,k,j]
                        let grad_3d = grad.clone().intoshape((batchsize, n, p))
                            .map_err(|_| scirs2_autograd::error::AutogradError::ShapeMismatch(
                                format!("Failed to reshape gradient to ({}, {}, {})", batchsize, n, p)
                            ))?;
                        let b_3d = b_data.clone().intoshape((batchsize, m, p))
                            .map_err(|_| scirs2_autograd::error::AutogradError::ShapeMismatch(
                                format!("Failed to reshape B to ({}, {}, {})", batchsize, m, p)
                            ))?;

                        let mut grad_a = Array::zeros((batchsize, n, m));

                        for batch_idx in 0..batchsize {
                            for i in 0..n {
                                for k in 0..m {
                                    let mut sum = F::zero();
                                    for j in 0..p {
                                        sum = sum
                                            + grad_3d[[batch_idx, i, j]] * b_3d[[batch_idx, k, j]];
                                    }
                                    grad_a[[batch_idx, i, k]] = sum;
                                }
                            }
                        }

                        Ok(grad_a.into_dyn())
                    },
                )
                    as Box<
                        dyn Fn(Array<F, IxDyn>) -> AutogradResult<Array<F, IxDyn>> + Send + Sync,
                    >)
            } else {
                None
            };

            // Backward function for the second tensor
            let backward_b = if b.requires_grad {
                Some(Box::new(
                    move |grad: Array<F, IxDyn>| -> AutogradResult<Array<F, IxDyn>> {
                        // For 3D tensors: dL/dB[b,k,j] = sum_i dL/dC[b,i,j] * A[b,i,k]
                        let grad_3d = grad.clone().intoshape((batchsize, n, p))
                            .map_err(|_| scirs2_autograd::error::AutogradError::ShapeMismatch(
                                format!("Failed to reshape gradient to ({}, {}, {})", batchsize, n, p)
                            ))?;
                        let a_3d = a_data.clone().intoshape((batchsize, n, m))
                            .map_err(|_| scirs2_autograd::error::AutogradError::ShapeMismatch(
                                format!("Failed to reshape A to ({}, {}, {})", batchsize, n, m)
                            ))?;

                        let mut grad_b = Array::zeros((batchsize, m, p));

                        for batch_idx in 0..batchsize {
                            for k in 0..m {
                                for j in 0..p {
                                    let mut sum = F::zero();
                                    for i in 0..n {
                                        sum = sum
                                            + grad_3d[[batch_idx, i, j]] * a_3d[[batch_idx, i, k]];
                                    }
                                    grad_b[[batch_idx, k, j]] = sum;
                                }
                            }
                        }

                        Ok(grad_b.into_dyn())
                    },
                )
                    as Box<
                        dyn Fn(Array<F, IxDyn>) -> AutogradResult<Array<F, IxDyn>> + Send + Sync,
                    >)
            } else {
                None
            };

            let node = Node::new(
                scirs2_autograd::graph::OpType::Activation("batch_matmul".to_string()),
                vec![a, b],
                vec![backward_a, backward_b],
            );

            let mut result = Tensor::new(result_data, requires_grad);
            result.node = Some(node);
            Ok(result)
        } else {
            Ok(Tensor::new(result_data, false))
        }
    } else {
        // General case for arbitrary batch dimensions (>3D tensors)
        let ashape = a.shape();
        let bshape = b.shape();
        
        // Get matrix dimensions (last 2 dimensions)
        let n = ashape[ashape.len() - 2];
        let m = ashape[ashape.len() - 1];
        let p = bshape[bshape.len() - 1];
        
        // Calculate total batch size by multiplying all batch dimensions
        let batch_dims = &ashape[..ashape.len() - 2];
        let total_batchsize: usize = batch_dims.iter().product();
        
        // Reshape tensors to 3D for easier processing
        let a_reshaped = a.data.clone().intoshape((total_batchsize, n, m))
            .map_err(|_| scirs2_autograd::error::AutogradError::ShapeMismatch(
                format!("Failed to reshape A to ({}, {}, {})", total_batchsize, n, m)
            ))?;
        let b_reshaped = b.data.clone().intoshape((total_batchsize, m, p))
            .map_err(|_| scirs2_autograd::error::AutogradError::ShapeMismatch(
                format!("Failed to reshape B to ({}, {}, {})", total_batchsize, m, p)
            ))?;
        
        // Perform batch matrix multiplication
        let mut result_reshaped = Array::zeros((total_batchsize, n, p));
        
        for batch_idx in 0..total_batchsize {
            for i in 0..n {
                for j in 0..p {
                    let mut sum = F::zero();
                    for k in 0..m {
                        sum = sum + a_reshaped[[batch_idx, i, k]] * b_reshaped[[batch_idx, k, j]];
                    }
                    result_reshaped[[batch_idx, i, j]] = sum;
                }
            }
        }
        
        // Calculate result shape by combining batch dimensions with matrix result dimensions
        let mut resultshape = batch_dims.to_vec();
        resultshape.push(n);
        resultshape.push(p);
        
        // Reshape result back to original batch dimensions
        let result_data = result_reshaped.intoshape(resultshape.as_slice())
            .map_err(|_| scirs2_autograd::error::AutogradError::ShapeMismatch(
                format!("Failed to reshape result to {:?}", resultshape)
            ))?.into_dyn();
        
        let requires_grad = a.requires_grad || b.requires_grad;
        
        if requires_grad {
            // Store data for gradient computation
            let a_data = a.data.clone();
            let b_data = b.data.clone();
            let ashape_clone = ashape.clone();
            let bshape_clone = bshape.clone();
            let resultshape_clone = resultshape.clone();
            
            // Backward function for the first tensor
            let backward_a = if a.requires_grad {
                Some(Box::new(
                    move |grad: Array<F, IxDyn>| -> AutogradResult<Array<F, IxDyn>> {
                        // Reshape gradient and b for computation
                        let grad_reshaped = grad.clone().intoshape((total_batchsize, n, p))
                            .map_err(|_| scirs2_autograd::error::AutogradError::ShapeMismatch(
                                "Failed to reshape gradient for backward_a".to_string()
                            ))?;
                        let b_reshaped = b_data.clone().intoshape((total_batchsize, m, p))
                            .map_err(|_| scirs2_autograd::error::AutogradError::ShapeMismatch(
                                "Failed to reshape B for backward_a".to_string()
                            ))?;
                        
                        let mut grad_a_reshaped = Array::zeros((total_batchsize, n, m));
                        
                        // Compute gradient: dL/dA = dL/dC @ B^T
                        for batch_idx in 0..total_batchsize {
                            for i in 0..n {
                                for k in 0..m {
                                    let mut sum = F::zero();
                                    for j in 0..p {
                                        sum = sum + grad_reshaped[[batch_idx, i, j]] * b_reshaped[[batch_idx, k, j]];
                                    }
                                    grad_a_reshaped[[batch_idx, i, k]] = sum;
                                }
                            }
                        }
                        
                        // Reshape back to original A shape
                        let grad_a = grad_a_reshaped.intoshape(ashape_clone.as_slice())
                            .map_err(|_| scirs2_autograd::error::AutogradError::ShapeMismatch(
                                "Failed to reshape gradient A back to original shape".to_string()
                            ))?;
                        
                        Ok(grad_a.into_dyn())
                    },
                )
                    as Box<
                        dyn Fn(Array<F, IxDyn>) -> AutogradResult<Array<F, IxDyn>> + Send + Sync,
                    >)
            } else {
                None
            };
            
            // Backward function for the second tensor
            let backward_b = if b.requires_grad {
                Some(Box::new(
                    move |grad: Array<F, IxDyn>| -> AutogradResult<Array<F, IxDyn>> {
                        // Reshape gradient and a for computation
                        let grad_reshaped = grad.clone().intoshape((total_batchsize, n, p))
                            .map_err(|_| scirs2_autograd::error::AutogradError::ShapeMismatch(
                                "Failed to reshape gradient for backward_b".to_string()
                            ))?;
                        let a_reshaped = a_data.clone().intoshape((total_batchsize, n, m))
                            .map_err(|_| scirs2_autograd::error::AutogradError::ShapeMismatch(
                                "Failed to reshape A for backward_b".to_string()
                            ))?;
                        
                        let mut grad_b_reshaped = Array::zeros((total_batchsize, m, p));
                        
                        // Compute gradient: dL/dB = A^T @ dL/dC
                        for batch_idx in 0..total_batchsize {
                            for k in 0..m {
                                for j in 0..p {
                                    let mut sum = F::zero();
                                    for i in 0..n {
                                        sum = sum + a_reshaped[[batch_idx, i, k]] * grad_reshaped[[batch_idx, i, j]];
                                    }
                                    grad_b_reshaped[[batch_idx, k, j]] = sum;
                                }
                            }
                        }
                        
                        // Reshape back to original B shape
                        let grad_b = grad_b_reshaped.intoshape(bshape_clone.as_slice())
                            .map_err(|_| scirs2_autograd::error::AutogradError::ShapeMismatch(
                                "Failed to reshape gradient B back to original shape".to_string()
                            ))?;
                        
                        Ok(grad_b.into_dyn())
                    },
                )
                    as Box<
                        dyn Fn(Array<F, IxDyn>) -> AutogradResult<Array<F, IxDyn>> + Send + Sync,
                    >)
            } else {
                None
            };
            
            let node = Node::new(
                scirs2_autograd::graph::OpType::Activation("batch_matmul_nd".to_string()),
                vec![a, b],
                vec![backward_a, backward_b],
            );
            
            let mut result = Tensor::new(result_data, requires_grad);
            result.node = Some(node);
            Ok(result)
        } else {
            Ok(Tensor::new(result_data, false))
        }
    }
}

/// Perform batch matrix-vector multiplication with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Batch of matrices, shape (batchsize, n, m)
/// * `x` - Batch of vectors, shape (batchsize, m)
///
/// # Returns
///
/// A new tensor of shape (batchsize, n) containing the batch matrix-vector products.
#[allow(dead_code)]
pub fn batch_matvec<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
    x: &Tensor<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure a is a 3D tensor (batch of matrices)
    if a.data.ndim() != 3 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Batch matrix-vector multiplication requires a 3D tensor (batch of matrices)"
                .to_string(),
        ));
    }

    // Ensure x is a 2D tensor (batch of vectors)
    if x.data.ndim() != 2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Batch matrix-vector multiplication requires a 2D tensor (batch of vectors)"
                .to_string(),
        ));
    }

    let ashape = a.shape();
    let xshape = x.shape();

    // Check batch dimensions match
    if ashape[0] != xshape[0] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            format!(
                "Batch dimensions mismatch: {} and {}",
                ashape[0], xshape[0]
            ),
        ));
    }

    // Check that matrix and vector dimensions are compatible
    if ashape[2] != xshape[1] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            format!(
                "Matrix-vector multiplication dimension mismatch: ({},{}) and {}",
                ashape[1], ashape[2], xshape[1]
            ),
        ));
    }

    let batchsize = ashape[0];
    let n = ashape[1];
    let m = ashape[2];

    // Compute batch matvec
    let mut result_data = Array::zeros((batchsize, n));

    for batch_idx in 0..batchsize {
        for i in 0..n {
            let mut sum = F::zero();
            for j in 0..m {
                sum = sum + a.data[[batch_idx, i, j]] * x.data[[batch_idx, j]];
            }
            result_data[[batch_idx, i]] = sum;
        }
    }

    let result_data = result_data.into_dyn();
    let requires_grad = a.requires_grad || x.requires_grad;

    if requires_grad {
        let a_data = a.data.clone();
        let x_data = x.data.clone();

        // Backward function for the matrices
        let backward_a = if a.requires_grad {
            Some(Box::new(
                move |grad: Array<F, IxDyn>| -> AutogradResult<Array<F, IxDyn>> {
                    // For batch matvec: dL/dA[b,i,j] = dL/dY[b,i] * X[b,j]
                    let grad_2d = grad.clone().intoshape((batchsize, n))
                        .map_err(|_| scirs2_autograd::error::AutogradError::ShapeMismatch(
                            format!("Failed to reshape gradient to ({}, {})", batchsize, n)
                        ))?;
                    let x_2d = x_data.clone().intoshape((batchsize, m))
                        .map_err(|_| scirs2_autograd::error::AutogradError::ShapeMismatch(
                            format!("Failed to reshape x to ({}, {})", batchsize, m)
                        ))?;

                    let mut grad_a = Array::zeros((batchsize, n, m));

                    for batch_idx in 0..batchsize {
                        for i in 0..n {
                            for j in 0..m {
                                grad_a[[batch_idx, i, j]] =
                                    grad_2d[[batch_idx, i]] * x_2d[[batch_idx, j]];
                            }
                        }
                    }

                    Ok(grad_a.into_dyn())
                },
            )
                as Box<
                    dyn Fn(Array<F, IxDyn>) -> AutogradResult<Array<F, IxDyn>> + Send + Sync,
                >)
        } else {
            None
        };

        // Backward function for the vectors
        let backward_x = if x.requires_grad {
            Some(Box::new(
                move |grad: Array<F, IxDyn>| -> AutogradResult<Array<F, IxDyn>> {
                    // For batch matvec: dL/dX[b,j] = sum_i dL/dY[b,i] * A[b,i,j]
                    let grad_2d = grad.clone().intoshape((batchsize, n))
                        .map_err(|_| scirs2_autograd::error::AutogradError::ShapeMismatch(
                            format!("Failed to reshape gradient to ({}, {})", batchsize, n)
                        ))?;
                    let a_3d = a_data.clone().intoshape((batchsize, n, m))
                        .map_err(|_| scirs2_autograd::error::AutogradError::ShapeMismatch(
                            format!("Failed to reshape A to ({}, {}, {})", batchsize, n, m)
                        ))?;

                    let mut grad_x = Array::zeros((batchsize, m));

                    for batch_idx in 0..batchsize {
                        for j in 0..m {
                            let mut sum = F::zero();
                            for i in 0..n {
                                sum = sum + grad_2d[[batch_idx, i]] * a_3d[[batch_idx, i, j]];
                            }
                            grad_x[[batch_idx, j]] = sum;
                        }
                    }

                    Ok(grad_x.into_dyn())
                },
            )
                as Box<
                    dyn Fn(Array<F, IxDyn>) -> AutogradResult<Array<F, IxDyn>> + Send + Sync,
                >)
        } else {
            None
        };

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("batch_matvec".to_string()),
            vec![a, x],
            vec![backward_a, backward_x],
        );

        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Compute batch matrix inverse with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Batch of square matrices, shape (batchsize, n, n)
///
/// # Returns
///
/// A new tensor of shape (batchsize, n, n) containing the batch matrix inverses.
#[allow(dead_code)]
pub fn batch_inv<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure a is a 3D tensor (batch of matrices)
    if a.data.ndim() != 3 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Batch matrix inverse requires a 3D tensor (batch of matrices)".to_string(),
        ));
    }

    let ashape = a.shape();

    // Check matrices are square
    if ashape[1] != ashape[2] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Batch matrix inverse requires square matrices".to_string(),
        ));
    }

    let batchsize = ashape[0];
    let n = ashape[1];

    // For simplicity, only implement 2x2 batch inverse
    if n > 2 {
        return Err(scirs2_autograd::error::AutogradError::OperationError(
            "Batch matrix inverse for matrices larger than 2x2 not yet implemented in autodiff"
                .to_string(),
        ));
    }

    let mut result_data = Array::zeros((batchsize, n, n));

    for batch_idx in 0..batchsize {
        // Extract individual matrix
        let mut matrix = Array::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                matrix[[i, j]] = a.data[[batch_idx, i, j]];
            }
        }

        // Compute determinant
        let det_val = if n == 1 {
            matrix[[0, 0]]
        } else {
            matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]]
        };

        // Check if matrix is singular
        if det_val.abs() < F::epsilon() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                format!(
                    "Cannot compute inverse of singular matrix at batch index {}",
                    batch_idx
                ),
            ));
        }

        // Compute inverse
        if det_val == F::zero() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                format!("Singular matrix encountered in batch inverse at index {}", batch_idx)
            ));
        }
        let inv_det = F::one() / det_val;

        if n == 1 {
            result_data[[batch_idx, 0, 0]] = F::one() / matrix[[0, 0]];
        } else {
            result_data[[batch_idx, 0, 0]] = matrix[[1, 1]] * inv_det;
            result_data[[batch_idx, 0, 1]] = -matrix[[0, 1]] * inv_det;
            result_data[[batch_idx, 1, 0]] = -matrix[[1, 0]] * inv_det;
            result_data[[batch_idx, 1, 1]] = matrix[[0, 0]] * inv_det;
        }
    }

    let result_data = result_data.into_dyn();
    let requires_grad = a.requires_grad;

    if requires_grad {
        let a_data = a.data.clone();
        let inv_data = result_data.clone();

        // Backward function for gradient computation
        let backward = if requires_grad {
            Some(Box::new(
                move |grad: Array<F, IxDyn>| -> AutogradResult<Array<F, IxDyn>> {
                    // Gradient of matrix inverse: dL/dA = -A^(-1) * dL/dA^(-1) * A^(-1)
                    let grad_3d = grad.clone().intoshape((batchsize, n, n)).unwrap();
                    let inv_3d = inv_data.clone().intoshape((batchsize, n, n)).unwrap();

                    let mut grad_a = Array::zeros((batchsize, n, n));

                    for batch_idx in 0..batchsize {
                        for i in 0..n {
                            for j in 0..n {
                                let mut sum = F::zero();
                                for k in 0..n {
                                    for l in 0..n {
                                        sum = sum
                                            + (-inv_3d[[batch_idx, i, k]]
                                                * grad_3d[[batch_idx, k, l]]
                                                * inv_3d[[batch_idx, l, j]]);
                                    }
                                }
                                grad_a[[batch_idx, i, j]] = sum;
                            }
                        }
                    }

                    Ok(grad_a.into_dyn())
                },
            )
                as Box<
                    dyn Fn(Array<F, IxDyn>) -> AutogradResult<Array<F, IxDyn>> + Send + Sync,
                >)
        } else {
            None
        };

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("batch_inv".to_string()),
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

/// Compute batch matrix determinant with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Batch of square matrices, shape (batchsize, n, n)
///
/// # Returns
///
/// A new tensor of shape (batchsize, 1) containing the batch matrix determinants.
#[allow(dead_code)]
pub fn batch_det<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure a is a 3D tensor (batch of matrices)
    if a.data.ndim() != 3 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Batch matrix determinant requires a 3D tensor (batch of matrices)".to_string(),
        ));
    }

    let ashape = a.shape();

    // Check matrices are square
    if ashape[1] != ashape[2] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Batch matrix determinant requires square matrices".to_string(),
        ));
    }

    let batchsize = ashape[0];
    let n = ashape[1];

    // For simplicity, only implement up to 3x3 matrix determinants
    if n > 3 {
        return Err(scirs2_autograd::error::AutogradError::OperationError(
            "Batch matrix determinant for matrices larger than 3x3 not yet implemented in autodiff"
                .to_string(),
        ));
    }

    let mut result_data = Array::zeros((batchsize, 1));

    for batch_idx in 0..batchsize {
        let det_val = match n {
            0 => F::one(),
            1 => a.data[[batch_idx, 0, 0]],
            2 => {
                a.data[[batch_idx, 0, 0]] * a.data[[batch_idx, 1, 1]]
                    - a.data[[batch_idx, 0, 1]] * a.data[[batch_idx, 1, 0]]
            }
            3 => {
                a.data[[batch_idx, 0, 0]]
                    * (a.data[[batch_idx, 1, 1]] * a.data[[batch_idx, 2, 2]]
                        - a.data[[batch_idx, 1, 2]] * a.data[[batch_idx, 2, 1]])
                    - a.data[[batch_idx, 0, 1]]
                        * (a.data[[batch_idx, 1, 0]] * a.data[[batch_idx, 2, 2]]
                            - a.data[[batch_idx, 1, 2]] * a.data[[batch_idx, 2, 0]])
                    + a.data[[batch_idx, 0, 2]]
                        * (a.data[[batch_idx, 1, 0]] * a.data[[batch_idx, 2, 1]]
                            - a.data[[batch_idx, 1, 1]] * a.data[[batch_idx, 2, 0]])
            }
            _ => unreachable!(),
        };

        result_data[[batch_idx, 0]] = det_val;
    }

    let result_data = result_data.into_dyn();
    let requires_grad = a.requires_grad;

    if requires_grad {
        let a_data = a.data.clone();

        // Backward function for gradient computation
        let backward = if requires_grad {
            Some(Box::new(
                move |grad: Array<F, IxDyn>| -> AutogradResult<Array<F, IxDyn>> {
                    // Gradient of determinant is adj(A)^T * grad
                    let grad_2d = grad.clone().intoshape((batchsize, 1)).unwrap();

                    let mut grad_a = Array::zeros((batchsize, n, n));

                    for batch_idx in 0..batchsize {
                        let grad_scalar = grad_2d[[batch_idx, 0]];

                        match n {
                            1 => {
                                grad_a[[batch_idx, 0, 0]] = grad_scalar;
                            }
                            2 => {
                                // adjugate of a 2x2 matrix
                                grad_a[[batch_idx, 0, 0]] = grad_scalar * a_data[[batch_idx, 1, 1]];
                                grad_a[[batch_idx, 0, 1]] =
                                    grad_scalar * (-a_data[[batch_idx, 1, 0]]);
                                grad_a[[batch_idx, 1, 0]] =
                                    grad_scalar * (-a_data[[batch_idx, 0, 1]]);
                                grad_a[[batch_idx, 1, 1]] = grad_scalar * a_data[[batch_idx, 0, 0]];
                            }
                            3 => {
                                // adjugate of a 3x3 matrix - simplified implementation
                                // First cofactor
                                grad_a[[batch_idx, 0, 0]] = grad_scalar
                                    * (a_data[[batch_idx, 1, 1]] * a_data[[batch_idx, 2, 2]]
                                        - a_data[[batch_idx, 1, 2]] * a_data[[batch_idx, 2, 1]]);
                                // and so on for other elements...
                                // This is a simplified placeholder that only computes one element
                                // A full implementation would compute all cofactors
                            }
                            _ => {}
                        }
                    }

                    Ok(grad_a.into_dyn())
                },
            )
                as Box<
                    dyn Fn(Array<F, IxDyn>) -> AutogradResult<Array<F, IxDyn>> + Send + Sync,
                >)
        } else {
            None
        };

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("batch_det".to_string()),
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

/// High-level interface for batch matrix operations with autodiff support
pub mod variable {
    use super::*;
    use scirs2_autograd::variable::Variable;

    /// Batch matrix multiplication for Variables
    pub fn batch_matmul<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
        b: &Variable<F>,
    ) -> AutogradResult<Variable<F>> {
        let result_tensor = super::batch_matmul(&a.tensor, &b.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    /// Batch matrix-vector multiplication for Variables
    pub fn batch_matvec<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
        x: &Variable<F>,
    ) -> AutogradResult<Variable<F>> {
        let result_tensor = super::batch_matvec(&a.tensor, &x.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    /// Batch matrix inverse for Variables
    pub fn batch_inv<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
    ) -> AutogradResult<Variable<F>> {
        let result_tensor = super::batch_inv(&a.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    /// Batch matrix determinant for Variables
    pub fn batch_det<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
    ) -> AutogradResult<Variable<F>> {
        let result_tensor = super::batch_det(&a.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }
}
