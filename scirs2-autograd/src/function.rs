//! Functional interface for autograd operations.
//!
//! This module provides a functional interface to autograd operations,
//! making it easier to use the computational graph with a Pythonic style.

use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{AutogradError, Result};
use crate::graph::Node;
use crate::tensor::Tensor;

/// Perform element-wise addition of two tensors.
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
///
/// A new tensor containing the element-wise sum.
#[allow(dead_code)]
pub fn add<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
    b: &Tensor<F>,
) -> Result<Tensor<F>> {
    // Check shape compatibility
    if a.shape() != b.shape() {
        return Err(AutogradError::ShapeMismatch(format!(
            "Addition requires tensors of the same shape: {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }

    Ok(a + b)
}

/// Perform element-wise subtraction of two tensors.
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
///
/// A new tensor containing the element-wise difference.
#[allow(dead_code)]
pub fn sub<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
    b: &Tensor<F>,
) -> Result<Tensor<F>> {
    // Check shape compatibility
    if a.shape() != b.shape() {
        return Err(AutogradError::ShapeMismatch(format!(
            "Subtraction requires tensors of the same shape: {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }

    Ok(a - b)
}

/// Perform element-wise multiplication of two tensors.
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
///
/// A new tensor containing the element-wise product.
#[allow(dead_code)]
pub fn mul<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
    b: &Tensor<F>,
) -> Result<Tensor<F>> {
    // Check shape compatibility
    if a.shape() != b.shape() {
        return Err(AutogradError::ShapeMismatch(format!(
            "Multiplication requires tensors of the same shape: {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }

    Ok(a * b)
}

/// Perform element-wise division of two tensors.
///
/// # Arguments
///
/// * `a` - First tensor (numerator)
/// * `b` - Second tensor (denominator)
///
/// # Returns
///
/// A new tensor containing the element-wise quotient.
#[allow(dead_code)]
pub fn div<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
    b: &Tensor<F>,
) -> Result<Tensor<F>> {
    // Check shape compatibility
    if a.shape() != b.shape() {
        return Err(AutogradError::ShapeMismatch(format!(
            "Division requires tensors of the same shape: {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }

    Ok(a / b)
}

/// Perform matrix multiplication of two tensors.
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
///
/// A new tensor containing the matrix product.
#[allow(dead_code)]
pub fn matmul<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
    b: &Tensor<F>,
) -> Result<Tensor<F>> {
    // Ensure dimensions are compatible
    if a.data.ndim() < 2 || b.data.ndim() < 2 {
        return Err(AutogradError::ShapeMismatch(
            "Matrix multiplication requires at least 2D tensors".to_string(),
        ));
    }

    let ashape = a.shape();
    let bshape = b.shape();

    if ashape[ashape.len() - 1] != bshape[bshape.len() - 2] {
        return Err(AutogradError::ShapeMismatch(format!(
            "Matrix multiplication dimension mismatch: {:?} and {:?}",
            ashape, bshape
        )));
    }

    // Compute the result
    // Implement matrix multiplication manually to avoid recursion issues
    let a_rows = a.data.shape()[0];
    let a_cols = a.data.shape()[1];
    let b_cols = b.data.shape()[1];

    // Create result matrix
    let mut result_data_2d = Array::<F>::zeros((a_rows, b_cols));

    // Manually compute matrix multiplication
    for i in 0..a_rows {
        for j in 0..b_cols {
            let mut sum = F::zero();
            for k in 0..a_cols {
                sum = sum + a.data[[i, k]] * b.data[[k, j]];
            }
            result_data_2d[[i, j]] = sum;
        }
    }

    // Convert to dynamic dimension
    let result_data = result_data_2d.into_dyn();
    let requires_grad = a.requires_grad || b.requires_grad;

    if requires_grad {
        let node = Node::matmul(a, b)?;
        Ok(Tensor::from_operation(result_data, node, requires_grad))
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Apply the ReLU activation function element-wise.
///
/// # Arguments
///
/// * `x` - Input tensor
///
/// # Returns
///
/// A new tensor with ReLU applied element-wise.
#[allow(dead_code)]
pub fn relu<F: Float + Debug + Send + Sync + 'static>(x: &Tensor<F>) -> Result<Tensor<F>> {
    let result_data = x.data.mapv(|v| if v > F::zero() { v } else { F::zero() });

    if x.requires_grad {
        let node = Node::relu(x);
        Ok(Tensor::from_operation(result_data, node, true))
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Apply the sigmoid activation function element-wise.
///
/// # Arguments
///
/// * `x` - Input tensor
///
/// # Returns
///
/// A new tensor with sigmoid applied element-wise.
#[allow(dead_code)]
pub fn sigmoid<F: Float + Debug + Send + Sync + 'static>(x: &Tensor<F>) -> Result<Tensor<F>> {
    let result_data = x.data.mapv(|v| F::one() / (F::one() + (-v).exp()));

    if x.requires_grad {
        let node = Node::sigmoid(x);
        Ok(Tensor::from_operation(result_data, node, true))
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Apply the tanh activation function element-wise.
///
/// # Arguments
///
/// * `x` - Input tensor
///
/// # Returns
///
/// A new tensor with tanh applied element-wise.
#[allow(dead_code)]
pub fn tanh<F: Float + Debug + Send + Sync + 'static>(x: &Tensor<F>) -> Result<Tensor<F>> {
    let result_data = x.data.mapv(|v| v.tanh());

    if x.requires_grad {
        let node = Node::tanh(x);
        Ok(Tensor::from_operation(result_data, node, true))
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Compute the sum of all elements or along a specified axis.
///
/// # Arguments
///
/// * `x` - Input tensor
/// * `axis` - Optional axis along which to sum
///
/// # Returns
///
/// A new tensor with the sum results.
#[allow(dead_code)]
pub fn sum<F: Float + Debug + Send + Sync + 'static>(
    x: &Tensor<F>,
    axis: Option<usize>,
) -> Result<Tensor<F>> {
    let result_data = if let Some(axis) = axis {
        // Sum along a specific axis
        if axis >= x.data.ndim() {
            return Err(AutogradError::ShapeMismatch(format!(
                "Sum axis {} out of bounds for tensor with {} dimensions",
                axis,
                x.data.ndim()
            )));
        }

        x.data.sum_axis(scirs2_core::ndarray::Axis(axis)).into_dyn()
    } else {
        // Sum all elements
        let sum_val = x.data.sum();
        Array::from_elem(IxDyn(&[1]), sum_val)
    };

    if x.requires_grad {
        let node = Node::sum(x, axis);
        Ok(Tensor::from_operation(result_data, node, true))
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Compute the mean of all elements or along a specified axis.
///
/// # Arguments
///
/// * `x` - Input tensor
/// * `axis` - Optional axis along which to take the mean
///
/// # Returns
///
/// A new tensor with the mean results.
#[allow(dead_code)]
pub fn mean<F: Float + Debug + Send + Sync + 'static>(
    x: &Tensor<F>,
    axis: Option<usize>,
) -> Result<Tensor<F>> {
    let result_data = if let Some(axis) = axis {
        // Mean along a specific axis
        if axis >= x.data.ndim() {
            return Err(AutogradError::ShapeMismatch(format!(
                "Mean axis {} out of bounds for tensor with {} dimensions",
                axis,
                x.data.ndim()
            )));
        }

        let sum = x.data.sum_axis(scirs2_core::ndarray::Axis(axis));
        let count = F::from(x.shape()[axis]).unwrap();
        let mean = sum.mapv(|v| v / count);
        mean.into_dyn()
    } else {
        // Mean of all elements
        let sum_val = x.data.sum();
        let count = F::from(x.size()).unwrap();
        let mean_val = sum_val / count;
        Array::from_elem(IxDyn(&[1]), mean_val)
    };

    if x.requires_grad {
        let node = Node::mean(x, axis);
        Ok(Tensor::from_operation(result_data, node, true))
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Reshape a tensor to a new shape.
///
/// # Arguments
///
/// * `x` - Input tensor
/// * `shape` - New shape for the tensor
///
/// # Returns
///
/// A new tensor with the same data but reshaped.
#[allow(dead_code)]
pub fn reshape<F: Float + Debug + Send + Sync + 'static>(
    x: &Tensor<F>,
    shape: &[usize],
) -> Result<Tensor<F>> {
    // Verify that the new shape has the same number of elements
    let old_size = x.size();
    let new_size: usize = shape.iter().product();

    if old_size != new_size {
        return Err(AutogradError::ShapeMismatch(format!(
            "Cannot reshape tensor of size {} to shape {:?} with size {}",
            old_size, shape, new_size
        )));
    }

    // Perform the reshape
    let result_data = match x.data.clone().into_shape_with_order(shape) {
        Ok(reshaped) => reshaped.into_dyn(),
        Err(e) => {
            return Err(AutogradError::OperationError(format!(
                "Reshape error: {}",
                e
            )))
        }
    };

    // In PyTorch/TensorFlow, reshape maintains gradient tracking
    // since it's just a view of the same data
    if x.requires_grad {
        // Create a backward function that reshapes gradients back to original shape
        let originalshape = x.shape().to_vec();
        let backward = Box::new(move |grad: Array<F, IxDyn>| -> Result<Array<F, IxDyn>> {
            // Use toshape instead of intoshape to avoid FnOnce issues
            match grad.clone().into_shape_with_order(originalshape.clone()) {
                Ok(reshaped_grad) => Ok(reshaped_grad),
                Err(e) => Err(AutogradError::OperationError(format!(
                    "Gradient reshape error: {}",
                    e
                ))),
            }
        })
            as Box<dyn Fn(Array<F, IxDyn>) -> Result<Array<F, IxDyn>> + Send + Sync>;

        let node = Node::new(super::graph::OpType::Reshape, vec![x], vec![Some(backward)]);
        Ok(Tensor::from_operation(result_data, node, true))
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Transpose a tensor by swapping two dimensions.
///
/// # Arguments
///
/// * `x` - Input tensor
/// * `dim0` - First dimension to swap
/// * `dim1` - Second dimension to swap
///
/// # Returns
///
/// A new tensor with the dimensions swapped.
#[allow(dead_code)]
pub fn transpose<F: Float + Debug + Send + Sync + 'static>(
    x: &Tensor<F>,
    dim0: usize,
    dim1: usize,
) -> Result<Tensor<F>> {
    let ndim = x.data.ndim();

    // Check that dimensions are valid
    if dim0 >= ndim || dim1 >= ndim {
        return Err(AutogradError::ShapeMismatch(format!(
            "Transpose dimensions {}, {} out of bounds for tensor with {} dimensions",
            dim0, dim1, ndim
        )));
    }

    // Perform the transpose
    let result_data = x.data.clone().permuted_axes(
        (0..ndim)
            .map(|i| {
                if i == dim0 {
                    dim1
                } else if i == dim1 {
                    dim0
                } else {
                    i
                }
            })
            .collect::<Vec<_>>(),
    );

    if x.requires_grad {
        // Create a backward function that transposes gradients back
        let backward = Box::new(move |grad: Array<F, IxDyn>| -> Result<Array<F, IxDyn>> {
            Ok(grad.permuted_axes(
                (0..ndim)
                    .map(|i| {
                        if i == dim0 {
                            dim1
                        } else if i == dim1 {
                            dim0
                        } else {
                            i
                        }
                    })
                    .collect::<Vec<_>>(),
            ))
        })
            as Box<dyn Fn(Array<F, IxDyn>) -> Result<Array<F, IxDyn>> + Send + Sync>;

        let node = Node::new(
            super::graph::OpType::Transpose,
            vec![x],
            vec![Some(backward)],
        );
        Ok(Tensor::from_operation(result_data, node, true))
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Compute the log (natural logarithm) of a tensor element-wise.
///
/// # Arguments
///
/// * `x` - Input tensor
///
/// # Returns
///
/// A new tensor with the natural logarithm applied element-wise.
#[allow(dead_code)]
pub fn log<F: Float + Debug + Send + Sync + 'static>(x: &Tensor<F>) -> Result<Tensor<F>> {
    let result_data = x.data.mapv(|v| v.ln());

    if x.requires_grad {
        // Backward pass: d(log(x))/dx = 1/x
        let x_data = x.data.clone();
        let backward = Box::new(move |grad: Array<F, IxDyn>| -> Result<Array<F, IxDyn>> {
            Ok(&grad / &x_data)
        })
            as Box<dyn Fn(Array<F, IxDyn>) -> Result<Array<F, IxDyn>> + Send + Sync>;

        let node = Node::new(super::graph::OpType::Log, vec![x], vec![Some(backward)]);
        Ok(Tensor::from_operation(result_data, node, true))
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Compute the exponential of a tensor element-wise.
///
/// # Arguments
///
/// * `x` - Input tensor
///
/// # Returns
///
/// A new tensor with the exponential applied element-wise.
#[allow(dead_code)]
pub fn exp<F: Float + Debug + Send + Sync + 'static>(x: &Tensor<F>) -> Result<Tensor<F>> {
    let result_data = x.data.mapv(|v| v.exp());

    if x.requires_grad {
        // Backward pass: d(exp(x))/dx = exp(x)
        let result_data_clone = result_data.clone();
        let backward = Box::new(move |grad: Array<F, IxDyn>| -> Result<Array<F, IxDyn>> {
            Ok(&grad * &result_data_clone)
        })
            as Box<dyn Fn(Array<F, IxDyn>) -> Result<Array<F, IxDyn>> + Send + Sync>;

        let node = Node::new(super::graph::OpType::Exp, vec![x], vec![Some(backward)]);
        Ok(Tensor::from_operation(result_data, node, true))
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Compute the softmax of a tensor along the specified dimension.
///
/// # Arguments
///
/// * `x` - Input tensor
/// * `dim` - Dimension along which to compute softmax
///
/// # Returns
///
/// A new tensor with softmax applied along the specified dimension.
#[allow(dead_code)]
pub fn softmax<F: Float + Debug + Send + Sync + 'static>(
    x: &Tensor<F>,
    dim: usize,
) -> Result<Tensor<F>> {
    let ndim = x.data.ndim();

    // Check that dimension is valid
    if dim >= ndim {
        return Err(AutogradError::ShapeMismatch(format!(
            "Softmax dimension {} out of bounds for tensor with {} dimensions",
            dim, ndim
        )));
    }

    // Compute max for numerical stability
    let max_vals = x.data.map_axis(scirs2_core::ndarray::Axis(dim), |view| {
        view.fold(F::neg_infinity(), |a, &b| if a > b { a } else { b })
    });

    // Subtract max and compute exp
    let mut exp_vals = x.data.clone();
    for (mut row, &max) in exp_vals
        .lanes_mut(scirs2_core::ndarray::Axis(dim))
        .into_iter()
        .zip(max_vals.iter())
    {
        row.mapv_inplace(|v| (v - max).exp());
    }

    // Compute sum of exps
    let sum_vals = exp_vals.map_axis(scirs2_core::ndarray::Axis(dim), |view| view.sum());

    // Normalize by sum
    let mut result_data = exp_vals.clone();
    for (mut row, &sum) in result_data
        .lanes_mut(scirs2_core::ndarray::Axis(dim))
        .into_iter()
        .zip(sum_vals.iter())
    {
        row.mapv_inplace(|v| v / sum);
    }

    if x.requires_grad {
        // Backward pass for softmax is complex:
        // d(softmax(x)_i)/d(x_j) = softmax(x)_i * (δ_{ij} - softmax(x)_j)
        let result_data_clone = result_data.clone();
        let _dim_clone = dim;

        let backward = Box::new(move |grad: Array<F, IxDyn>| -> Result<Array<F, IxDyn>> {
            // For simplicity, we'll implement a less efficient but correct version
            // This would need to be optimized for a production autograd system
            let mut dx = grad.clone();

            // Proper softmax backward implementation
            // For softmax, the Jacobian is:
            // dS_i/dx_j = S_i * (delta_ij - S_j)
            // where delta_ij is 1 when i=j and 0 otherwise
            // Instead of forming the full Jacobian, we use the fact that:
            // dy/dx = S * (I - S_transpose)
            // where * is matrix multiplication and S is the softmax output
            // when applied to incoming gradient gy, we get:
            // gy * dy/dx = gy * S * (I - S_transpose)
            // which simplifies to S * (gy - dot(S, gy))
            
            // Compute S * gy (element-wise multiplication)
            let s_times_gy = &dx * &result_data_clone;
            
            // If we're not working with a batch dimension, handle as a simple vector
            if result_data_clone.ndim() == 1 {
                // Compute dot(S, gy) for a vector (single example)
                let dot_s_gy = result_data_clone.dot(&dx);
                
                // Subtract from each element of s_times_gy
                dx = &s_times_gy - &(&result_data_clone * dot_s_gy);
            } else {
                // Handle batch dimensions (assume last dim is features)
                // This is a simplification for common cases and may need to be extended
                // for more complex scenarios
                let last_dim = result_data_clone.ndim() - 1;
                
                // Compute dot product along last dimension for each item in the batch
                // For each batch item, compute S_i * (gy_i - sum(S_i * gy_i))
                let mut result = s_times_gy.clone();
                
                // For each batch element
                for idx in scirs2_core::ndarray::indices(result_data_clone.shape()[..last_dim].iter().cloned()) {
                    // Get the batch item's softmax outputs and gradients
                    let s_batch = result_data_clone.index_axis(scirs2_core::ndarray::Axis(last_dim-1), idx[last_dim-1]);
                    let gy_batch = dx.index_axis(scirs2_core::ndarray::Axis(last_dim-1), idx[last_dim-1]);
                    
                    // Compute sum(S_i * gy_i) for this batch item
                    let dot_s_gy = (&s_batch * &gy_batch).sum();
                    
                    // Update the result for this batch item
                    let mut view = result.index_axis_mut(scirs2_core::ndarray::Axis(last_dim-1), idx[last_dim-1]);
                    view -= &(&s_batch * dot_s_gy);
                }
                
                dx = result;
            }

            Ok(dx)
        })
            as Box<dyn Fn(Array<F, IxDyn>) -> Result<Array<F, IxDyn>> + Send + Sync>;

        let node = Node::new(
            super::graph::OpType::Activation("softmax".to_string()),
            vec![x],
            vec![Some(backward)],
        );
        Ok(Tensor::from_operation(result_data, node, true))
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Concatenate tensors along a specified dimension.
///
/// # Arguments
///
/// * `tensors` - List of tensors to concatenate
/// * `dim` - Dimension along which to concatenate
///
/// # Returns
///
/// A new tensor that is the concatenation of the input tensors.
#[allow(dead_code)]
pub fn cat<F: Float + Debug + Send + Sync + 'static>(
    tensors: &[&Tensor<F>],
    dim: usize,
) -> Result<Tensor<F>> {
    if tensors.is_empty() {
        return Err(AutogradError::OperationError(
            "Cannot concatenate empty list of tensors".to_string(),
        ));
    }

    // Check dimension compatibility
    let refshape = tensors[0].shape();
    let ndim = refshape.len();

    if dim >= ndim {
        return Err(AutogradError::ShapeMismatch(format!(
            "Concatenation dimension {} out of bounds for tensor with {} dimensions",
            dim, ndim
        )));
    }

    for (i, tensor) in tensors.iter().enumerate().skip(1) {
        let shape = tensor.shape();

        if shape.len() != ndim {
            return Err(AutogradError::ShapeMismatch(format!(
                "All tensors must have the same number of dimensions, but tensor 0 has {} and tensor {} has {}",
                ndim, i, shape.len()
            )));
        }

        for (j, (&s1, &s2)) in refshape.iter().zip(shape.iter()).enumerate() {
            if j != dim && s1 != s2 {
                return Err(AutogradError::ShapeMismatch(format!(
                    "Incompatible shapes for concatenation: {:?} and {:?} at dimension {}",
                    refshape, shape, j
                )));
            }
        }
    }

    // Implement concatenation along the specified dimension
    // First, we'll create a new shape for the result tensor
    let mut resultshape = refshape.to_vec();
    resultshape[dim] = tensors.iter().map(|t| t.shape()[dim]).sum();
    
    // Create an empty array with the new shape
    let mut result_data = Array::<F, IxDyn>::zeros(resultshape.clone());
    
    // Now fill the result by copying each tensor into the appropriate slice
    let mut offset = 0;
    for tensor in tensors {
        // Calculate start and end indices for this tensor along the concat dimension
        let size_along_dim = tensor.shape()[dim];
        
        // Create a view into the result array where this tensor's data will go
        let mut indices: Vec<scirs2_core::ndarray::SliceInfo<_, scirs2_core::ndarray::SliceArg>> = 
            (0..ndim).map(|d| if d == dim {
                scirs2_core::ndarray::s![offset..offset + size_along_dim]
            } else {
                scirs2_core::ndarray::s![..]
            }).collect();
        
        // Get a mutable slice view and assign tensor data
        let mut slice = result_data.slice_each_axis_mut(|ax| indices[ax.axis.index()]);
        slice.assign(&tensor.data);
        
        // Update offset for the next tensor
        offset += size_along_dim;
    }
    

    // Check if any tensor requires gradients
    let requires_grad = tensors.iter().any(|t| t.requires_grad);

    if requires_grad {
        // Create backward functions for each input tensor
        let mut backward_fns = Vec::with_capacity(tensors.len());
        let mut offsets = Vec::with_capacity(tensors.len());
        let mut total_dim_size = 0;

        for tensor in tensors {
            let offset = total_dim_size;
            offsets.push(offset);
            total_dim_size += tensor.shape()[dim];

            if tensor.requires_grad {
                let _tensorshape = tensor.shape();
                let _dim_clone = dim;

                backward_fns.push(Some(Box::new(
                    move |grad: Array<F, IxDyn>| -> Result<Array<F, IxDyn>> {
                        // Extract the appropriate slice of the gradient for this tensor
                        let tensorshape = tensorshape.clone();
                        let dim = dim_clone;
                        let offset = offset; // Capture from outer scope
                        
                        // Get the size of this tensor along the concatenation dimension
                        let size_along_dim = tensorshape[dim];
                        
                        // Create a slice of the gradient corresponding to this tensor
                        let mut indices: Vec<scirs2_core::ndarray::SliceInfo<_, scirs2_core::ndarray::SliceArg>> = 
                            (0..grad.ndim()).map(|d| if d == dim {
                                scirs2_core::ndarray::s![offset..offset + size_along_dim]
                            } else {
                                scirs2_core::ndarray::s![..]
                            }).collect();
                        
                        // Extract the slice view
                        let grad_slice = grad.slice_each_axis(|ax| indices[ax.axis.index()]).to_owned();
                        
                        Ok(grad_slice)
                    },
                )
                    as Box<dyn Fn(Array<F, IxDyn>) -> Result<Array<F, IxDyn>> + Send + Sync>));
            } else {
                backward_fns.push(None);
            }
        }

        let node = Node::new(super::graph::OpType::Concat, tensors.to_vec(), backward_fns);
        Ok(Tensor::from_operation(result_data, node, requires_grad))
    } else {
        Ok(Tensor::new(result_data, false))
    }
}
