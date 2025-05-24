//! Matrix calculus utilities
//!
//! This module provides utilities for matrix calculus operations, including:
//! * Computing gradients of scalar-valued functions
//! * Computing Jacobians of vector-valued functions
//! * Computing Hessians of scalar-valued functions
//! * Computing directional derivatives
//! * Jacobian-vector and vector-Jacobian products
//! * Matrix-valued function derivatives
//! * Taylor series approximations
//! * Critical point finding
//! * Additional advanced matrix calculus operations
//!
//! ## Basic Operations
//!
//! The core functionality includes computing derivatives of functions:
//! * `gradient` - Compute the gradient of a scalar-valued function
//! * `jacobian` - Compute the Jacobian of a vector-valued function
//! * `hessian` - Compute the Hessian of a scalar-valued function
//! * `directional_derivative` - Compute the derivative along a specified direction
//!
//! ## Enhanced Operations
//!
//! Advanced operations are available in the `enhanced` submodule:
//! * Jacobian-vector and vector-Jacobian products for efficient computation
//! * Matrix-valued function gradient and Jacobian computation
//! * Taylor series approximation of functions
//! * Numerical critical point finding for optimization
//! * Hessian-vector products for large-scale optimization
//!
//! ## Examples
//!
//! ```
//! use ndarray::{array, ArrayView1};
//! use scirs2_linalg::matrix_calculus::gradient;
//! use scirs2_linalg::error::LinalgResult;
//!
//! // Define a simple quadratic function
//! let f = |x: &ArrayView1<f64>| -> LinalgResult<f64> {
//!     Ok(x[0] * x[0] + 2.0_f64 * x[1] * x[1])
//! };
//!
//! // Compute the gradient at point [1.0, 1.0]
//! let x = array![1.0_f64, 1.0_f64];
//! let grad = gradient(f, &x.view(), None).unwrap();
//!
//! // The gradient should be [2*x[0], 4*x[1]] = [2.0, 4.0]
//! assert!((grad[0] - 2.0_f64).abs() < 1e-6_f64);
//! assert!((grad[1] - 4.0_f64).abs() < 1e-6_f64);
//! ```

use ndarray::{Array1, Array2, ArrayView1};
use num_traits::{Float, One, Zero};
use std::fmt::Debug;

use crate::error::{LinalgError, LinalgResult};

// Export the enhanced matrix calculus operations
pub mod enhanced;

/// Compute the Jacobian matrix of a vector-valued function.
///
/// For a function f: R^n -> R^m, the Jacobian is an m×n matrix where
/// J[i,j] = df_i/dx_j.
///
/// # Arguments
///
/// * `f` - Function mapping an n-dimensional vector to an m-dimensional vector
/// * `x` - Point at which to evaluate the Jacobian
/// * `epsilon` - Step size for finite difference approximation
///
/// # Returns
///
/// * Jacobian matrix of size m×n
pub fn jacobian<F>(
    f: impl Fn(&ArrayView1<F>) -> LinalgResult<Array1<F>>,
    x: &ArrayView1<F>,
    epsilon: Option<F>,
) -> LinalgResult<Array2<F>>
where
    F: Float + Zero + One + Copy + Debug,
{
    let n = x.len();
    let f_x = f(x)?;
    let m = f_x.len();
    let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt());

    let mut jac = Array2::zeros((m, n));

    for j in 0..n {
        // Perturb the j-th component
        let mut x_plus = x.to_owned();
        x_plus[j] = x_plus[j] + eps;

        // Evaluate f at perturbed point
        let f_x_plus = f(&x_plus.view())?;

        // Compute finite difference approximation
        for i in 0..m {
            jac[[i, j]] = (f_x_plus[i] - f_x[i]) / eps;
        }
    }

    Ok(jac)
}

/// Compute the gradient of a scalar-valued function.
///
/// For a function f: R^n -> R, the gradient is an n-dimensional vector where
/// grad[i] = df/dx_i.
///
/// # Arguments
///
/// * `f` - Function mapping an n-dimensional vector to a scalar
/// * `x` - Point at which to evaluate the gradient
/// * `epsilon` - Step size for finite difference approximation
///
/// # Returns
///
/// * Gradient vector of size n
pub fn gradient<F>(
    f: impl Fn(&ArrayView1<F>) -> LinalgResult<F>,
    x: &ArrayView1<F>,
    epsilon: Option<F>,
) -> LinalgResult<Array1<F>>
where
    F: Float + Zero + One + Copy + Debug,
{
    let n = x.len();
    let f_x = f(x)?;
    let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt());

    let mut grad = Array1::zeros(n);

    for i in 0..n {
        // Perturb the i-th component
        let mut x_plus = x.to_owned();
        x_plus[i] = x_plus[i] + eps;

        // Evaluate f at perturbed point
        let f_x_plus = f(&x_plus.view())?;

        // Compute finite difference approximation
        grad[i] = (f_x_plus - f_x) / eps;
    }

    Ok(grad)
}

/// Compute the Hessian matrix of a scalar-valued function.
///
/// For a function f: R^n -> R, the Hessian is an n×n matrix where
/// H[i,j] = d²f/(dx_i dx_j).
///
/// # Arguments
///
/// * `f` - Function mapping an n-dimensional vector to a scalar
/// * `x` - Point at which to evaluate the Hessian
/// * `epsilon` - Step size for finite difference approximation
///
/// # Returns
///
/// * Hessian matrix of size n×n
pub fn hessian<F>(
    f: impl Fn(&ArrayView1<F>) -> LinalgResult<F>,
    x: &ArrayView1<F>,
    epsilon: Option<F>,
) -> LinalgResult<Array2<F>>
where
    F: Float + Zero + One + Copy + Debug,
{
    let n = x.len();
    let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt().sqrt());

    let mut hess = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            // Use central difference for better accuracy
            let mut x_pp = x.to_owned();
            let mut x_pm = x.to_owned();
            let mut x_mp = x.to_owned();
            let mut x_mm = x.to_owned();

            x_pp[i] = x_pp[i] + eps;
            x_pp[j] = x_pp[j] + eps;

            x_pm[i] = x_pm[i] + eps;
            x_pm[j] = x_pm[j] - eps;

            x_mp[i] = x_mp[i] - eps;
            x_mp[j] = x_mp[j] + eps;

            x_mm[i] = x_mm[i] - eps;
            x_mm[j] = x_mm[j] - eps;

            // Evaluate f at perturbed points
            let f_pp = f(&x_pp.view())?;
            let f_pm = f(&x_pm.view())?;
            let f_mp = f(&x_mp.view())?;
            let f_mm = f(&x_mm.view())?;

            // Compute the mixed partial derivative using central difference
            let h_ij = (f_pp - f_pm - f_mp + f_mm) / (F::from(4.0).unwrap() * eps * eps);
            hess[[i, j]] = h_ij;
        }
    }

    Ok(hess)
}

/// Compute the directional derivative of a scalar-valued function.
///
/// For a function f: R^n -> R and a direction vector v,
/// computes ∇f(x) · v, which represents the rate of change of f in the direction v.
///
/// # Arguments
///
/// * `f` - Function mapping an n-dimensional vector to a scalar
/// * `x` - Point at which to evaluate the directional derivative
/// * `v` - Direction vector (will be normalized if not already a unit vector)
/// * `epsilon` - Step size for finite difference approximation
///
/// # Returns
///
/// * Directional derivative (scalar)
pub fn directional_derivative<F>(
    f: impl Fn(&ArrayView1<F>) -> LinalgResult<F>,
    x: &ArrayView1<F>,
    v: &ArrayView1<F>,
    epsilon: Option<F>,
) -> LinalgResult<F>
where
    F: Float + Zero + One + Copy + Debug,
{
    if x.len() != v.len() {
        return Err(LinalgError::ShapeError(format!(
            "Direction vector must have the same dimension as input: {:?} vs {:?}",
            v.shape(),
            x.shape()
        )));
    }

    // Normalize the direction vector
    let v_norm = v.iter().fold(F::zero(), |acc, &val| acc + val * val).sqrt();

    if v_norm < F::epsilon() {
        return Err(LinalgError::InvalidInputError(
            "Direction vector must not be zero".to_string(),
        ));
    }

    let unit_v = Array1::from_iter(v.iter().map(|&val| val / v_norm));

    // Compute the gradient
    let grad = gradient(f, x, epsilon)?;

    // Compute the dot product of gradient and direction
    let dir_deriv = grad
        .iter()
        .zip(unit_v.iter())
        .fold(F::zero(), |acc, (&g, &v)| acc + g * v);

    Ok(dir_deriv)
}
