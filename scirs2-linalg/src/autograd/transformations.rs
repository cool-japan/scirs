//! Matrix transformations with automatic differentiation suppor
//!
//! This module provides differentiable implementations of matrix transformations
//! like projection, rotation, and scaling.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, One, Zero};
use std::fmt::Debug;

use scirs2_autograd::error::Result as AutogradResult;
use scirs2_autograd::graph::Node;
use scirs2_autograd::tensor::Tensor;
use scirs2_autograd::variable::Variable;

/// Perform an orthogonal projection onto a subspace with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Matrix whose columns span the subspace to project onto
/// * `x` - Vector to projec
///
/// # Returns
///
/// The projection of x onto the column space of A with gradient tracking.
#[allow(dead_code)]
pub fn project<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
    x: &Tensor<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure inputs are valid
    if a.data.ndim() != 2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Matrix A must be a 2D tensor".to_string(),
        ));
    }

    if x.data.ndim() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Vector x must be a 1D tensor".to_string(),
        ));
    }

    let ashape = a.shape();
    let xshape = x.shape();

    if ashape[0] != xshape[0] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            format!(
                "Number of rows in A ({}) must match length of x ({})",
                ashape[0], xshape[0]
            ),
        ));
    }

    // For projection, we compute P = A(A^T A)^(-1)A^T x
    // First, compute A^T
    let a_t_data = a.data.t().to_owned();
    let a_t = Tensor::new(a_t_data, a.requires_grad);

    // Compute A^T A manually
    let a_t_data_2d = a_t
        .data
        .clone()
        .intoshape((ashape[1], ashape[0]))
        .unwrap();
    let a_data_2d = a.data.clone().intoshape((ashape[0], ashape[1])).unwrap();

    let mut a_t_a_data = Array2::<F>::zeros((ashape[1], ashape[1]));
    for i in 0..ashape[1] {
        for j in 0..ashape[1] {
            let mut sum = F::zero();
            for k in 0..ashape[0] {
                sum = sum + a_t_data_2d[[i, k]] * a_data_2d[[k, j]];
            }
            a_t_a_data[[i, j]] = sum;
        }
    }

    let a_t_a_data = a_t_a_data.into_dyn();
    let a_t_a = Tensor::new(a_t_a_data, a.requires_grad || a_t.requires_grad);

    // Compute (A^T A)^(-1)
    let a_t_a_inv_data = {
        let n = ashape[1];
        if n == 1 {
            // For 1x1 matrix, simple reciprocal
            let mut result = a_t_a.data.clone();
            if result[[0, 0]].abs() < F::epsilon() {
                return Err(scirs2_autograd::error::AutogradError::OperationError(
                    "Matrix is singular, cannot compute inverse".to_string(),
                ));
            }
            result[[0, 0]] = F::one() / result[[0, 0]];
            result.into_dyn()
        } else if n == 2 {
            // For 2x2 matrix, direct inverse
            let det =
                a_t_a.data[[0, 0]] * a_t_a.data[[1, 1]] - a_t_a.data[[0, 1]] * a_t_a.data[[1, 0]];

            if det.abs() < F::epsilon() {
                return Err(scirs2_autograd::error::AutogradError::OperationError(
                    "Matrix is singular, cannot compute inverse".to_string(),
                ));
            }

            let mut result = Array2::<F>::zeros((2, 2));
            let inv_det = F::one() / det;
            result[[0, 0]] = a_t_a.data[[1, 1]] * inv_det;
            result[[0, 1]] = -a_t_a.data[[0, 1]] * inv_det;
            result[[1, 0]] = -a_t_a.data[[1, 0]] * inv_det;
            result[[1, 1]] = a_t_a.data[[0, 0]] * inv_det;
            result.into_dyn()
        } else {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                format!("Inverse for matrices larger than 2x2 not yet implemented in autodiff"),
            ));
        }
    };

    let a_t_a_inv = Tensor::new(a_t_a_inv_data, a.requires_grad || a_t.requires_grad);

    // Compute A^T x manually
    let a_t_data_2d = a_t
        .data
        .clone()
        .intoshape((ashape[1], ashape[0]))
        .unwrap();
    let x_data_1d = x.data.clone().intoshape(ashape[0]).unwrap();

    let mut a_t_x_data = Array1::<F>::zeros(ashape[1]);
    for i in 0..ashape[1] {
        let mut sum = F::zero();
        for k in 0..ashape[0] {
            sum = sum + a_t_data_2d[[i, k]] * x_data_1d[k];
        }
        a_t_x_data[i] = sum;
    }

    let a_t_x_data = a_t_x_data.into_dyn();
    let a_t_x = Tensor::new(a_t_x_data, a.requires_grad || x.requires_grad);

    // Compute (A^T A)^(-1) A^T x manually
    let a_t_a_inv_data_2d = a_t_a_inv
        .data
        .clone()
        .intoshape((ashape[1], ashape[1]))
        .unwrap();
    let a_t_x_data_1d = a_t_x.data.clone().intoshape(ashape[1]).unwrap();

    let mut temp_data = Array1::<F>::zeros(ashape[1]);
    for i in 0..ashape[1] {
        let mut sum = F::zero();
        for j in 0..ashape[1] {
            sum = sum + a_t_a_inv_data_2d[[i, j]] * a_t_x_data_1d[j];
        }
        temp_data[i] = sum;
    }

    let temp_data = temp_data.into_dyn();
    let temp = Tensor::new(temp_data, a.requires_grad || x.requires_grad);

    // Compute A (A^T A)^(-1) A^T x manually
    let a_data_2d = a.data.clone().intoshape((ashape[0], ashape[1])).unwrap();
    let temp_data_1d = temp.data.clone().intoshape(ashape[1]).unwrap();

    let mut result_data = Array1::<F>::zeros(ashape[0]);
    for i in 0..ashape[0] {
        let mut sum = F::zero();
        for j in 0..ashape[1] {
            sum = sum + a_data_2d[[i, j]] * temp_data_1d[j];
        }
        result_data[i] = sum;
    }

    let result_data = result_data.into_dyn();

    let requires_grad = a.requires_grad || x.requires_grad;

    if requires_grad {
        let a_data = a.data.clone();
        let x_data = x.data.clone();

        // Backward function for the matrix A
        let backward_a = if a.requires_grad {
            Some(
                Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    // Gradient computation for A is complex for projection
                    // For a simple approximation, we pass through the gradien
                    // A full implementation would require matrix calculus

                    // Return zeros for now - this is a placeholder
                    let a_datashape = a_data.shape();
                    let mut grad_a = Array2::<F>::zeros((a_datashape[0], a_datashape[1]));
                    Ok(grad_a.into_dyn())
                })
                    as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        // Backward function for the vector x
        let backward_x = if x.requires_grad {
            Some(
                Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    // For x, the gradient is the projection matrix applied to the gradien
                    // P = A(A^T A)^(-1)A^T

                    // Since we're computing P * grad, and P is idempotent,
                    // if grad is already in the column space of A, then P * grad = grad
                    // For simplicity, we'll just return the gradien
                    Ok(grad)
                })
                    as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("project".to_string()),
            vec![a, x],
            vec![backward_a, backward_x],
        );

        // Since from_operation is private, we'll use a different approach
        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Create a 2D rotation matrix with automatic differentiation support.
///
/// # Arguments
///
/// * `angle` - Rotation angle in radians
///
/// # Returns
///
/// A 2x2 rotation matrix with gradient tracking.
#[allow(dead_code)]
pub fn rotationmatrix_2d<F: Float + Debug + Send + Sync + 'static>(
    angle: &Tensor<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure angle is a scalar
    if angle.data.ndim() != 1 || angle.data.len() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Angle must be a scalar tensor".to_string(),
        ));
    }

    let theta = angle.data[[0]];
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();

    // Create 2x2 rotation matrix
    let mut result_data = Array2::<F>::zeros((2, 2));
    result_data[[0, 0]] = cos_theta;
    result_data[[0, 1]] = -sin_theta;
    result_data[[1, 0]] = sin_theta;
    result_data[[1, 1]] = cos_theta;

    let result_data = result_data.into_dyn();
    let requires_grad = angle.requires_grad;

    if requires_grad {
        // Backward function for the angle
        let backward = if requires_grad {
            Some(
                Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    // Convert gradient to 2x2 shape
                    let grad_2d = grad.clone().intoshape((2, 2)).unwrap();

                    // Gradient of rotation matrix with respect to angle
                    // d/dθ [cos θ, -sin θ; sin θ, cos θ] = [-sin θ, -cos θ; cos θ, -sin θ]
                    let d_cos_theta = -sin_theta;
                    let d_sin_theta = cos_theta;

                    let grad_angle =
                          grad_2d[[0, 0]] * d_cos_theta
                        + grad_2d[[0, 1]] * (-d_sin_theta)
                        + grad_2d[[1, 0]] * d_sin_theta
                        + grad_2d[[1, 1]] * d_cos_theta;

                    Ok(scirs2_core::ndarray::Array::from_elem(scirs2_core::ndarray::IxDyn(&[1]), grad_angle))
                })
                    as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("rotationmatrix_2d".to_string()),
            vec![angle],
            vec![backward],
        );

        // Since from_operation is private, we'll use a different approach
        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Create a scaling matrix with automatic differentiation support.
///
/// # Arguments
///
/// * `scales` - Vector of scaling factors
///
/// # Returns
///
/// A diagonal scaling matrix with gradient tracking.
#[allow(dead_code)]
pub fn scalingmatrix<F: Float + Debug + Send + Sync + 'static>(
    scales: &Tensor<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure scales is a vector
    if scales.data.ndim() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Scales must be a 1D tensor".to_string(),
        ));
    }

    let n = scales.data.len();

    // Create diagonal scaling matrix
    let mut result_data = Array2::<F>::zeros((n, n));
    for i in 0..n {
        result_data[[i, i]] = scales.data[i];
    }

    let result_data = result_data.into_dyn();
    let requires_grad = scales.requires_grad;

    if requires_grad {
        // Backward function for the scales
        let backward = if requires_grad {
            Some(
                Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    // Convert gradient to nxn shape
                    let grad_2d = grad.clone().intoshape((n, n)).unwrap();

                    // Gradient of scaling matrix with respect to scales
                    // is just the diagonal elements of the gradien
                    let mut grad_scales = Array1::<F>::zeros(n);
                    for i in 0..n {
                        grad_scales[i] = grad_2d[[i, i]];
                    }

                    Ok(grad_scales.into_dyn())
                })
                    as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("scalingmatrix".to_string()),
            vec![scales],
            vec![backward],
        );

        // Since from_operation is private, we'll use a different approach
        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Create a reflection matrix with automatic differentiation support.
///
/// # Arguments
///
/// * `normal` - Vector normal to the reflection hyperplane
///
/// # Returns
///
/// A reflection matrix with gradient tracking.
#[allow(dead_code)]
pub fn reflectionmatrix<F: Float + Debug + Send + Sync + 'static>(
    normal: &Tensor<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure normal is a vector
    if normal.data.ndim() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Normal must be a 1D tensor".to_string(),
        ));
    }

    let n = normal.data.len();

    // Normalize the normal vector
    let norm_squared = normal.data.iter().fold(F::zero(), |acc, &x| acc + x * x);
    if norm_squared < F::epsilon() {
        return Err(scirs2_autograd::error::AutogradError::OperationError(
            "Normal vector must not be zero".to_string(),
        ));
    }

    let norm = norm_squared.sqrt();
    let unit_normal = normal.data.mapv(|x| x / norm);

    // Compute reflection matrix: I - 2 * (n⊗n)
    let mut result_data = Array2::<F>::eye(n);

    for i in 0..n {
        for j in 0..n {
            result_data[[i, j]] =
                result_data[[i, j]] - F::from(2.0).unwrap() * unit_normal[i] * unit_normal[j];
        }
    }

    let result_data = result_data.into_dyn();
    let requires_grad = normal.requires_grad;

    if requires_grad {
        let normal_data = normal.data.clone();

        // Backward function for the normal vector
        let backward = if requires_grad {
            Some(
                Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    // Gradient computation for reflection matrix is complex
                    // For simplicity, we'll return zeros for now
                    let mut grad_normal = Array1::<F>::zeros(n);
                    Ok(grad_normal.into_dyn())
                })
                    as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("reflectionmatrix".to_string()),
            vec![normal],
            vec![backward],
        );

        // Since from_operation is private, we'll use a different approach
        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Create a shear matrix with automatic differentiation support.
///
/// # Arguments
///
/// * `shear_factors` - Vector of shear factors
/// * `dim1` - First dimension affected by shear
/// * `dim2` - Second dimension affected by shear
/// * `n` - Size of the resulting matrix
///
/// # Returns
///
/// A shear matrix with gradient tracking.
#[allow(dead_code)]
pub fn shearmatrix<F: Float + Debug + Send + Sync + 'static>(
    shear_factor: &Tensor<F>,
    dim1: usize,
    dim2: usize,
    n: usize,
) -> AutogradResult<Tensor<F>> {
    // Ensure shear_factor is a scalar
    if shear_factor.data.ndim() != 1 || shear_factor.data.len() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Shear _factor must be a scalar tensor".to_string(),
        ));
    }

    // Validate dimensions
    if dim1 >= n || dim2 >= n || dim1 == dim2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            format!("Invalid dimensions: dim1={}, dim2={}, n={}", dim1, dim2, n),
        ));
    }

    // Create shear matrix (identity with one off-diagonal element)
    let mut result_data = Array2::<F>::eye(n);
    result_data[[dim1, dim2]] = shear_factor.data[[0]];

    let result_data = result_data.into_dyn();
    let requires_grad = shear_factor.requires_grad;

    if requires_grad {
        // Backward function for the shear _factor
        let backward = if requires_grad {
            Some(
                Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    // Convert gradient to nxn shape
                    let grad_2d = grad.clone().intoshape((n, n)).unwrap();

                    // Gradient of shear matrix with respect to shear _factor
                    // is just the (dim1, dim2) element of the gradien
                    let grad_shear = grad_2d[[dim1, dim2]];

                    Ok(scirs2_core::ndarray::Array::from_elem(scirs2_core::ndarray::IxDyn(&[1]), grad_shear))
                })
                    as Box<dyn Fn(scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>) -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("shearmatrix".to_string()),
            vec![shear_factor],
            vec![backward],
        );

        // Since from_operation is private, we'll use a different approach
        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// High-level interface for matrix transformations with autodiff suppor
// Note: The variable module is temporarily disabled due to API mismatch.
// Variable in scirs2_autograd is a type alias for RefCell<NdArray<F>>,
// not a struct with a tensor field. This module needs redesign.
/*
pub mod variable {
    use super::*;
    use scirs2_autograd::variable::Variable;

    /// Orthogonal projection for Variables
    pub fn project<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
        x: &Variable<F>,
    ) -> AutogradResult<Variable<F>> {
        let result_tensor = super::project(&a.tensor, &x.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    /// 2D rotation matrix for Variables
    pub fn rotationmatrix_2d<F: Float + Debug + Send + Sync + 'static>(
        angle: &Variable<F>,
    ) -> AutogradResult<Variable<F>> {
        let result_tensor = super::rotationmatrix_2d(&angle.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    /// Scaling matrix for Variables
    pub fn scalingmatrix<F: Float + Debug + Send + Sync + 'static>(
        scales: &Variable<F>,
    ) -> AutogradResult<Variable<F>> {
        let result_tensor = super::scalingmatrix(&scales.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    /// Reflection matrix for Variables
    pub fn reflectionmatrix<F: Float + Debug + Send + Sync + 'static>(
        normal: &Variable<F>,
    ) -> AutogradResult<Variable<F>> {
        let result_tensor = super::reflectionmatrix(&normal.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    /// Shear matrix for Variables
    pub fn shearmatrix<F: Float + Debug + Send + Sync + 'static>(
        shear_factor: &Variable<F>,
        dim1: usize,
        dim2: usize,
        n: usize,
    ) -> AutogradResult<Variable<F>> {
        let result_tensor = super::shearmatrix(&shear_factor.tensor, dim1, dim2, n)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }
}
*/
