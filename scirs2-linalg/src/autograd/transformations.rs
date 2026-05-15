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
        .expect("Operation failed");
    let a_data_2d = a.data.clone().intoshape((ashape[0], ashape[1])).expect("Operation failed");

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
        .expect("Operation failed");
    let x_data_1d = x.data.clone().intoshape(ashape[0]).expect("Operation failed");

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
        .expect("Operation failed");
    let a_t_x_data_1d = a_t_x.data.clone().intoshape(ashape[1]).expect("Operation failed");

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
    let a_data_2d = a.data.clone().intoshape((ashape[0], ashape[1])).expect("Operation failed");
    let temp_data_1d = temp.data.clone().intoshape(ashape[1]).expect("Operation failed");

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

        // Backward function for the matrix A using finite-difference
        // The gradient of projection w.r.t. A is complex (matrix calculus with
        // A^T A inversion). We use a numerically-stable finite-difference approach
        // for n <= 2, which is the regime supported by the forward pass.
        let backward_a = if a.requires_grad {
            Some(
                Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    let a_shape = a_data.shape();
                    let m = a_shape[0];
                    let n_cols = a_shape[1];
                    let a_2d = a_data.clone().into_shape((m, n_cols)).map_err(|e| {
                        scirs2_autograd::error::AutogradError::OperationError(format!("reshape failed: {}", e))
                    })?;
                    let x_1d_opt = x_data.clone().into_shape(m);
                    let x_1d = x_1d_opt.map_err(|e| {
                        scirs2_autograd::error::AutogradError::OperationError(format!("reshape failed: {}", e))
                    })?;
                    let grad_1d_opt = grad.clone().into_shape(m);
                    let grad_1d = grad_1d_opt.map_err(|e| {
                        scirs2_autograd::error::AutogradError::OperationError(format!("reshape failed: {}", e))
                    })?;

                    // Finite-difference helper: compute projection y = P_A x for matrix A
                    let project_fd = |a_mat: &Array2<F>, x_vec: &scirs2_core::ndarray::Array1<F>| -> Option<scirs2_core::ndarray::Array1<F>> {
                        let nc = n_cols;
                        let mr = m;
                        // A^T A
                        let mut ata = Array2::<F>::zeros((nc, nc));
                        for i in 0..nc {
                            for j in 0..nc {
                                let mut s = F::zero();
                                for k in 0..mr { s = s + a_mat[[k, i]] * a_mat[[k, j]]; }
                                ata[[i, j]] = s;
                            }
                        }
                        // (A^T A)^{-1} for n_cols <= 2
                        let ata_inv = if nc == 1 {
                            if ata[[0, 0]].abs() < F::epsilon() { return None; }
                            let mut inv = Array2::<F>::zeros((1, 1));
                            inv[[0, 0]] = F::one() / ata[[0, 0]];
                            inv
                        } else {
                            let det = ata[[0, 0]] * ata[[1, 1]] - ata[[0, 1]] * ata[[1, 0]];
                            if det.abs() < F::epsilon() { return None; }
                            let id = F::one() / det;
                            let mut inv = Array2::<F>::zeros((2, 2));
                            inv[[0, 0]] = ata[[1, 1]] * id; inv[[0, 1]] = -ata[[0, 1]] * id;
                            inv[[1, 0]] = -ata[[1, 0]] * id; inv[[1, 1]] = ata[[0, 0]] * id;
                            inv
                        };
                        // A^T x
                        let mut atx = scirs2_core::ndarray::Array1::<F>::zeros(nc);
                        for i in 0..nc {
                            let mut s = F::zero();
                            for k in 0..mr { s = s + a_mat[[k, i]] * x_vec[k]; }
                            atx[i] = s;
                        }
                        // (A^T A)^{-1} A^T x
                        let mut ata_inv_atx = scirs2_core::ndarray::Array1::<F>::zeros(nc);
                        for i in 0..nc {
                            let mut s = F::zero();
                            for k in 0..nc { s = s + ata_inv[[i, k]] * atx[k]; }
                            ata_inv_atx[i] = s;
                        }
                        // A (A^T A)^{-1} A^T x
                        let mut y = scirs2_core::ndarray::Array1::<F>::zeros(mr);
                        for i in 0..mr {
                            let mut s = F::zero();
                            for k in 0..nc { s = s + a_mat[[i, k]] * ata_inv_atx[k]; }
                            y[i] = s;
                        }
                        Some(y)
                    };

                    let eps = F::from(1e-6).unwrap_or(F::epsilon());
                    let mut grad_a_out = Array2::<F>::zeros((m, n_cols));

                    for i in 0..m {
                        for j in 0..n_cols {
                            let mut a_plus = a_2d.clone();
                            let mut a_minus = a_2d.clone();
                            a_plus[[i, j]] = a_plus[[i, j]] + eps;
                            a_minus[[i, j]] = a_minus[[i, j]] - eps;

                            let y_plus = project_fd(&a_plus, &x_1d);
                            let y_minus = project_fd(&a_minus, &x_1d);

                            match (y_plus, y_minus) {
                                (Some(yp), Some(ym)) => {
                                    // dL/dA[i,j] = sum_k grad_1d[k] * (yp[k] - ym[k]) / (2*eps)
                                    let mut s = F::zero();
                                    let two_eps = eps + eps;
                                    for k in 0..m {
                                        s = s + grad_1d[k] * (yp[k] - ym[k]) / two_eps;
                                    }
                                    grad_a_out[[i, j]] = s;
                                }
                                _ => {
                                    grad_a_out[[i, j]] = F::zero();
                                }
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

        // Backward function for the vector x
        // grad w.r.t x is P^T grad = P grad (projection is self-adjoint: P^T = P)
        let backward_x = if x.requires_grad {
            Some(
                Box::new(move |grad: scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>| -> AutogradResult<scirs2_core::ndarray::Array<F, scirs2_core::ndarray::IxDyn>> {
                    // gradient w.r.t x = P^T * grad_y = P * grad_y (P is symmetric)
                    // For n_cols <= 2, compute P * grad_y directly
                    let a_shape = a_data.shape();
                    let m_x = a_shape[0];
                    let nc = a_shape[1];
                    let a_2d = a_data.clone().into_shape((m_x, nc)).map_err(|e| {
                        scirs2_autograd::error::AutogradError::OperationError(format!("reshape failed: {}", e))
                    })?;
                    let grad_1d_opt = grad.into_shape(m_x);
                    let grad_1d = grad_1d_opt.map_err(|e| {
                        scirs2_autograd::error::AutogradError::OperationError(format!("reshape failed: {}", e))
                    })?;

                    // A^T A
                    let mut ata = Array2::<F>::zeros((nc, nc));
                    for i in 0..nc {
                        for j in 0..nc {
                            let mut s = F::zero();
                            for k in 0..m_x { s = s + a_2d[[k, i]] * a_2d[[k, j]]; }
                            ata[[i, j]] = s;
                        }
                    }
                    // (A^T A)^{-1}
                    let ata_inv = if nc == 1 {
                        if ata[[0, 0]].abs() < F::epsilon() {
                            return Ok(scirs2_core::ndarray::Array1::<F>::zeros(m_x).into_dyn());
                        }
                        let mut inv = Array2::<F>::zeros((1, 1));
                        inv[[0, 0]] = F::one() / ata[[0, 0]];
                        inv
                    } else {
                        let det = ata[[0, 0]] * ata[[1, 1]] - ata[[0, 1]] * ata[[1, 0]];
                        if det.abs() < F::epsilon() {
                            return Ok(scirs2_core::ndarray::Array1::<F>::zeros(m_x).into_dyn());
                        }
                        let id = F::one() / det;
                        let mut inv = Array2::<F>::zeros((2, 2));
                        inv[[0, 0]] = ata[[1, 1]] * id; inv[[0, 1]] = -ata[[0, 1]] * id;
                        inv[[1, 0]] = -ata[[1, 0]] * id; inv[[1, 1]] = ata[[0, 0]] * id;
                        inv
                    };
                    // A^T grad
                    let mut at_grad = scirs2_core::ndarray::Array1::<F>::zeros(nc);
                    for i in 0..nc {
                        let mut s = F::zero();
                        for k in 0..m_x { s = s + a_2d[[k, i]] * grad_1d[k]; }
                        at_grad[i] = s;
                    }
                    // (A^T A)^{-1} A^T grad
                    let mut tmp = scirs2_core::ndarray::Array1::<F>::zeros(nc);
                    for i in 0..nc {
                        let mut s = F::zero();
                        for k in 0..nc { s = s + ata_inv[[i, k]] * at_grad[k]; }
                        tmp[i] = s;
                    }
                    // A (A^T A)^{-1} A^T grad = P grad
                    let mut p_grad = scirs2_core::ndarray::Array1::<F>::zeros(m_x);
                    for i in 0..m_x {
                        let mut s = F::zero();
                        for k in 0..nc { s = s + a_2d[[i, k]] * tmp[k]; }
                        p_grad[i] = s;
                    }
                    Ok(p_grad.into_dyn())
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
                    let grad_2d = grad.clone().intoshape((2, 2)).expect("Operation failed");

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
                    let grad_2d = grad.clone().intoshape((n, n)).expect("Operation failed");

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
                result_data[[i, j]] - F::from(2.0).expect("Operation failed") * unit_normal[i] * unit_normal[j];
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
                    let grad_2d = grad.clone().intoshape((n, n)).expect("Operation failed");

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

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_project_backward_a_numerical_gradient() {
        // A = 2x1 matrix (column vector spanning a 1D subspace)
        // x = 2D vector
        // Loss = sum(project(A, x))
        let a_data = scirs2_core::ndarray::arr2(&[[1.0f64], [1.0]]).into_dyn();
        let x_data = scirs2_core::ndarray::arr1(&[3.0f64, 1.0]).into_dyn();

        let a = Tensor::new(a_data.clone(), true);
        let x = Tensor::new(x_data.clone(), false);

        let y = project(&a, &x).expect("project failed");

        // grad_y = ones (loss = sum(y))
        let grad = Array1::<f64>::ones(2).into_dyn();
        let backward_a = y.node.as_ref().expect("node").backward_fns[0]
            .as_ref().expect("backward_a fn");
        let analytical_a = backward_a(grad).expect("backward_a failed");
        let analytical = analytical_a.into_shape((2, 1)).unwrap();

        // Numerical gradient
        let a_2d = a_data.into_shape((2, 1)).unwrap();
        let x_1d = x_data.into_shape(2).unwrap();
        let eps = 1e-5;
        for i in 0..2 {
            for j in 0..1 {
                let mut ap = a_2d.clone();
                let mut am = a_2d.clone();
                ap[[i, j]] += eps;
                am[[i, j]] -= eps;

                let proj_sum = |a_mat: Array2<f64>| -> f64 {
                    let at = Tensor::new(a_mat.into_dyn(), false);
                    let xt = Tensor::new(x_1d.clone().into_dyn(), false);
                    match project(&at, &xt) {
                        Ok(r) => r.data.iter().sum(),
                        Err(_) => 0.0,
                    }
                };

                let num = (proj_sum(ap) - proj_sum(am)) / (2.0 * eps);
                let diff = (analytical[[i, j]] - num).abs();
                assert!(diff < 1e-4, "project backward_a mismatch at ({},{}) analytical={} numerical={}", i, j, analytical[[i,j]], num);
            }
        }
    }

    #[test]
    fn test_project_backward_x_is_p_times_grad() {
        // backward_x should return P * grad where P = A(A^T A)^{-1} A^T
        // For a column vector A = [1, 0]^T, P projects onto e1.
        // P = [[1,0],[0,0]], so P * [1,1]^T = [1, 0]^T
        let a_data = scirs2_core::ndarray::arr2(&[[1.0f64], [0.0]]).into_dyn();
        let x_data = scirs2_core::ndarray::arr1(&[0.5f64, 0.5]).into_dyn();

        let a = Tensor::new(a_data, false);
        let x = Tensor::new(x_data, true);

        let y = project(&a, &x).expect("project failed");
        let grad = Array1::<f64>::ones(2).into_dyn();
        let backward_x = y.node.as_ref().expect("node").backward_fns[1]
            .as_ref().expect("backward_x fn");
        let grad_x = backward_x(grad).expect("backward_x failed");
        let gx = grad_x.into_shape(2).unwrap();
        // P * [1,1] = [1, 0]
        assert!((gx[0] - 1.0).abs() < 1e-10, "backward_x[0]={}", gx[0]);
        assert!((gx[1] - 0.0).abs() < 1e-10, "backward_x[1]={}", gx[1]);
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
