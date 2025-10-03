//! Tensor algebra operations with automatic differentiation suppor
//!
//! This module provides differentiable implementations of tensor operations
//! like contraction, outer product, and tensor-vector product.

use scirs2_core::ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis, Dimension, IxDyn};
use scirs2_core::numeric::{Float, One, Zero};
use std::fmt::Debug;

use scirs2_autograd::error::Result as AutogradResult;
use scirs2_autograd::graph::Node;
use scirs2_autograd::tensor::Tensor;
use scirs2_autograd::variable::Variable;

/// Tensor contraction along specified dimensions with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
/// * `dims_a` - Dimensions of a to contrac
/// * `dims_b` - Dimensions of b to contrac
///
/// # Returns
///
/// The contracted tensor with gradient tracking.
#[allow(dead_code)]
pub fn contract<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
    b: &Tensor<F>,
    dims_a: &[usize],
    dims_b: &[usize],
) -> AutogradResult<Tensor<F>> {
    // Validate inputs
    if dims_a.len() != dims_b.len() {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            format!(
                "Number of contracted dimensions must match, got {} and {}",
                dims_a.len(),
                dims_b.len()
            ),
        ));
    }

    let ashape = a.shape();
    let bshape = b.shape();

    // Check contracted dimensions have the same size
    for (&dim_a, &dim_b) in dims_a.iter().zip(dims_b.iter()) {
        if dim_a >= ashape.len() {
            return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
                format!(
                    "Dimension {} out of bounds for first tensor with {} dimensions",
                    dim_a,
                    ashape.len()
                ),
            ));
        }
        if dim_b >= bshape.len() {
            return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
                format!(
                    "Dimension {} out of bounds for second tensor with {} dimensions",
                    dim_b,
                    bshape.len()
                ),
            ));
        }
        if ashape[dim_a] != bshape[dim_b] {
            return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
                format!(
                    "Contracted dimensions must have the same size, got {} and {}",
                    ashape[dim_a], bshape[dim_b]
                ),
            ));
        }
    }

    // For simplicity, implement for simple cases firs
    // This implementation handles matrix-matrix contraction (like matmul)
    // and vector-vector contraction (like dot product)

    if a.data.ndim() == 2 && b.data.ndim() == 2 && dims_a == &[1] && dims_b == &[0] {
        // This is matrix multiplication: A[i,j] * B[j,k] -> C[i,k]
        let m = ashape[0];
        let n = ashape[1];
        let p = bshape[1];

        // Create result matrix
        let mut result_data = Array2::<F>::zeros((m, p));

        // Compute the contraction (matrix multiplication)
        for i in 0..m {
            for k in 0..p {
                let mut sum = F::zero();
                for j in 0..n {
                    sum = sum + a.data[[i, j]] * b.data[[j, k]];
                }
                result_data[[i, k]] = sum;
            }
        }

        let result_data = result_data.into_dyn();
        let requires_grad = a.requires_grad || b.requires_grad;

        if requires_grad {
            let a_data = a.data.clone();
            let b_data = b.data.clone();

            // Backward function for the first tensor
            let backward_a = if a.requires_grad {
                Some(
                    Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                        // dA[i,j] = sum_k dC[i,k] * B[j,k]
                        let grad_2d = grad.clone().intoshape((m, p)).unwrap();
                        let b_2d = b_data.clone().intoshape((n, p)).unwrap();

                        let mut grad_a = Array2::<F>::zeros((m, n));

                        for i in 0..m {
                            for j in 0..n {
                                let mut sum = F::zero();
                                for k in 0..p {
                                    sum = sum + grad_2d[[i, k]] * b_2d[[j, k]];
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

            // Backward function for the second tensor
            let backward_b = if b.requires_grad {
                Some(
                    Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                        // dB[j,k] = sum_i dC[i,k] * A[i,j]
                        let grad_2d = grad.clone().intoshape((m, p)).unwrap();
                        let a_2d = a_data.clone().intoshape((m, n)).unwrap();

                        let mut grad_b = Array2::<F>::zeros((n, p));

                        for j in 0..n {
                            for k in 0..p {
                                let mut sum = F::zero();
                                for i in 0..m {
                                    sum = sum + grad_2d[[i, k]] * a_2d[[i, j]];
                                }
                                grad_b[[j, k]] = sum;
                            }
                        }

                        Ok(grad_b.into_dyn())
                    })
                        as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
                )
            } else {
                None
            };

            let node = Node::new(
                scirs2_autograd::graph::OpType::Activation("contract".to_string()),
                vec![_a_b],
                vec![backward_a, backward_b],
            );

            let mut result = Tensor::new(result_data, requires_grad);
            result.node = Some(node);
            Ok(result)
        } else {
            Ok(Tensor::new(result_data, false))
        }
    } else if a.data.ndim() == 1 && b.data.ndim() == 1 && dims_a == &[0] && dims_b == &[0] {
        // This is dot product: A[i] * B[i] -> C[]
        let n = ashape[0];

        // Compute the dot produc
        let mut dot_product = F::zero();
        for i in 0..n {
            dot_product = dot_product + a.data[i] * b.data[i];
        }

        let result_data = Array::from_elem(IxDyn(&[1]), dot_product);
        let requires_grad = a.requires_grad || b.requires_grad;

        if requires_grad {
            let a_data = a.data.clone();
            let b_data = b.data.clone();

            // Backward function for the first tensor
            let backward_a = if a.requires_grad {
                Some(
                    Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                        // dA[i] = dC * B[i]
                        let grad_scalar = grad[[0]];
                        let mut grad_a = Array1::<F>::zeros(n);

                        for i in 0..n {
                            grad_a[i] = grad_scalar * b_data[i];
                        }

                        Ok(grad_a.into_dyn())
                    })
                        as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
                )
            } else {
                None
            };

            // Backward function for the second tensor
            let backward_b = if b.requires_grad {
                Some(
                    Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                        // dB[i] = dC * A[i]
                        let grad_scalar = grad[[0]];
                        let mut grad_b = Array1::<F>::zeros(n);

                        for i in 0..n {
                            grad_b[i] = grad_scalar * a_data[i];
                        }

                        Ok(grad_b.into_dyn())
                    })
                        as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
                )
            } else {
                None
            };

            let node = Node::new(
                scirs2_autograd::graph::OpType::Activation("contract_dot".to_string()),
                vec![_a_b],
                vec![backward_a, backward_b],
            );

            let mut result = Tensor::new(result_data, requires_grad);
            result.node = Some(node);
            Ok(result)
        } else {
            Ok(Tensor::new(result_data, false))
        }
    } else {
        // For more complex tensor contractions, return an error for now
        Err(scirs2_autograd::error::AutogradError::OperationError(
            "General tensor contraction not yet implemented in autodiff".to_string(),
        ))
    }
}

/// Compute the outer product of two vectors with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - First vector tensor
/// * `b` - Second vector tensor
///
/// # Returns
///
/// The outer product tensor with gradient tracking.
#[allow(dead_code)]
pub fn outer<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
    b: &Tensor<F>,
) -> AutogradResult<Tensor<F>> {
    // Outer product of two vectors: A[i] * B[j] -> C[i,j]

    // Validate inputs
    if a.data.ndim() != 1 || b.data.ndim() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            format!(
                "Outer product requires two 1D vectors, got shapes {:?} and {:?}",
                a.shape(),
                b.shape()
            ),
        ));
    }

    let ashape = a.shape();
    let bshape = b.shape();

    let m = ashape[0];
    let n = bshape[0];

    // Create the result tensor
    let mut result_data = Array2::<F>::zeros((m, n));

    // Compute the outer produc
    for i in 0..m {
        for j in 0..n {
            result_data[[i, j]] = a.data[i] * b.data[j];
        }
    }

    let result_data = result_data.into_dyn();
    let requires_grad = a.requires_grad || b.requires_grad;

    if requires_grad {
        let a_data = a.data.clone();
        let b_data = b.data.clone();

        // Backward function for the first vector
        let backward_a = if a.requires_grad {
            Some(
                Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    // Convert gradient to 2D shape
                    let grad_2d = grad.clone().intoshape((m, n)).unwrap();

                    // dA[i] = sum_j dC[i,j] * B[j]
                    let mut grad_a = Array1::<F>::zeros(m);

                    for i in 0..m {
                        let mut sum = F::zero();
                        for j in 0..n {
                            sum = sum + grad_2d[[i, j]] * b_data[j];
                        }
                        grad_a[i] = sum;
                    }

                    Ok(grad_a.into_dyn())
                })
                    as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        // Backward function for the second vector
        let backward_b = if b.requires_grad {
            Some(
                Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    // Convert gradient to 2D shape
                    let grad_2d = grad.clone().intoshape((m, n)).unwrap();

                    // dB[j] = sum_i dC[i,j] * A[i]
                    let mut grad_b = Array1::<F>::zeros(n);

                    for j in 0..n {
                        let mut sum = F::zero();
                        for i in 0..m {
                            sum = sum + grad_2d[[i, j]] * a_data[i];
                        }
                        grad_b[j] = sum;
                    }

                    Ok(grad_b.into_dyn())
                })
                    as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("outer".to_string()),
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

/// Compute tensor-vector contraction (tensor-vector product) with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Tensor
/// * `v` - Vector
/// * `axis` - Axis of the tensor to contract with the vector
///
/// # Returns
///
/// The contracted tensor with gradient tracking.
#[allow(dead_code)]
pub fn tensor_vector_product<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
    v: &Tensor<F>,
    axis: usize,
) -> AutogradResult<Tensor<F>> {
    // Validate inputs
    let ashape = a.shape();
    let vshape = v.shape();

    if v.data.ndim() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            format!("Vector must be a 1D tensor, got shape {:?}", vshape),
        ));
    }

    if axis >= ashape.len() {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                ashape.len()
            ),
        ));
    }

    if ashape[axis] != vshape[0] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            format!(
                "Tensor dimension {} must match vector dimension, got {} and {}",
                axis, ashape[axis], vshape[0]
            ),
        ));
    }

    // Create result shape by removing the contracted axis
    let mut resultshape = Vec::with_capacity(ashape.len() - 1);
    for (i, &dim) in ashape.iter().enumerate() {
        if i != axis {
            resultshape.push(dim);
        }
    }

    // Special case for simple matrix-vector multiplication (matrix * vector)
    if ashape.len() == 2 && axis == 1 {
        // This is just a standard matrix-vector product A[i,j] * v[j] -> w[i]
        let m = ashape[0];
        let n = ashape[1];

        let mut result_data = Array1::<F>::zeros(m);

        for i in 0..m {
            let mut sum = F::zero();
            for j in 0..n {
                sum = sum + a.data[[i, j]] * v.data[j];
            }
            result_data[i] = sum;
        }

        let result_data = result_data.into_dyn();
        let requires_grad = a.requires_grad || v.requires_grad;

        if requires_grad {
            let a_data = a.data.clone();
            let v_data = v.data.clone();

            // Backward function for the tensor
            let backward_a = if a.requires_grad {
                Some(
                    Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                        // dA[i,j] = dw[i] * v[j]
                        let grad_1d = grad.clone().intoshape(m).unwrap();
                        let mut grad_a = Array2::<F>::zeros((m, n));

                        for i in 0..m {
                            for j in 0..n {
                                grad_a[[i, j]] = grad_1d[i] * v_data[j];
                            }
                        }

                        Ok(grad_a.into_dyn())
                    })
                        as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
                )
            } else {
                None
            };

            // Backward function for the vector
            let backward_v = if v.requires_grad {
                Some(
                    Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                        // dv[j] = sum_i dw[i] * A[i,j]
                        let grad_1d = grad.clone().intoshape(m).unwrap();
                        let mut grad_v = Array1::<F>::zeros(n);

                        for j in 0..n {
                            let mut sum = F::zero();
                            for i in 0..m {
                                sum = sum + grad_1d[i] * a_data[[i, j]];
                            }
                            grad_v[j] = sum;
                        }

                        Ok(grad_v.into_dyn())
                    })
                        as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
                )
            } else {
                None
            };

            let node = Node::new(
                scirs2_autograd::graph::OpType::Activation("tensor_vector_product".to_string()),
                vec![a, v],
                vec![backward_a, backward_v],
            );

            let mut result = Tensor::new(result_data, requires_grad);
            result.node = Some(node);
            Ok(result)
        } else {
            Ok(Tensor::new(result_data, false))
        }
    } else {
        // For more complex tensor-vector products, return an error for now
        Err(scirs2_autograd::error::AutogradError::OperationError(
            "General tensor-vector product not yet implemented in autodiff".to_string(),
        ))
    }
}

/// High-level interface for tensor algebra operations with autodiff suppor
pub mod variable {
    use super::*;
    use scirs2_autograd::variable::Variable;

    /// Tensor contraction for Variables
    pub fn contract<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
        b: &Variable<F>,
        dims_a: &[usize],
        dims_b: &[usize],
    ) -> AutogradResult<Variable<F>> {
        let result_tensor = super::contract(&_a.tensor, &_b.tensor, dims_a, dims_b)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    /// Outer product for Variables
    pub fn outer<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
        b: &Variable<F>,
    ) -> AutogradResult<Variable<F>> {
        let result_tensor = super::outer(&a.tensor, &b.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    /// Tensor-vector product for Variables
    pub fn tensor_vector_product<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
        v: &Variable<F>,
        axis: usize,
    ) -> AutogradResult<Variable<F>> {
        let result_tensor = super::tensor_vector_product(&a.tensor, &v.tensor, axis)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }
}
