//! Gradient calculation utilities for neural networks
//!
//! This module provides utilities for calculating gradients in the context of
//! neural network training, focusing on efficiency and numerical stability.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

/// Calculate the gradient of mean squared error with respect to predictions
///
/// Computes the gradient of MSE loss function with respect to predictions.
/// This is a common gradient calculation in regression tasks.
///
/// # Arguments
///
/// * `predictions` - Predicted values
/// * `targets` - Target values (ground truth)
///
/// # Returns
///
/// * The gradient of MSE with respect to predictions
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::gradient::mse_gradient;
/// use approx::assert_relative_eq;
///
/// let predictions = array![3.0, 1.0, 2.0];
/// let targets = array![2.5, 0.5, 2.0];
///
/// let gradient = mse_gradient(&predictions.view(), &targets.view()).unwrap();
///
/// // gradients = 2 * (predictions - targets) / n
/// // = 2 * ([3.0, 1.0, 2.0] - [2.5, 0.5, 2.0]) / 3
/// // = 2 * [0.5, 0.5, 0.0] / 3
/// // = [0.333..., 0.333..., 0.0]
/// assert_relative_eq!(gradient[0], 1.0/3.0, epsilon = 1e-10);
/// assert_relative_eq!(gradient[1], 1.0/3.0, epsilon = 1e-10);
/// assert_relative_eq!(gradient[2], 0.0, epsilon = 1e-10);
/// ```
#[allow(dead_code)]
pub fn mse_gradient<F>(
    predictions: &ArrayView1<F>,
    targets: &ArrayView1<F>,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand,
{
    // Check dimensions compatibility
    if predictions.shape() != targets.shape() {
        return Err(LinalgError::ShapeError(format!(
            "Shape mismatch for mse_gradient: predictions has shape {:?} but targets has shape {:?}",
            predictions.shape(),
            targets.shape()
        )));
    }

    let n = F::from(predictions.len()).unwrap();
    let two = F::from(2.0).unwrap();

    // Compute (predictions - targets) * 2/n
    // This is the gradient of MSE with respect to predictions
    let scale = two / n;
    let gradient = predictions - targets;
    let gradient = &gradient * scale;

    Ok(gradient.to_owned())
}

/// Calculate the gradient of binary cross-entropy with respect to predictions
///
/// Computes the gradient of binary cross-entropy loss function with respect to predictions.
/// This is a common gradient calculation in binary classification tasks.
///
/// # Arguments
///
/// * `predictions` - Predicted probabilities (must be between 0 and 1)
/// * `targets` - Target values (ground truth, must be 0 or 1)
///
/// # Returns
///
/// * The gradient of binary cross-entropy with respect to predictions
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::gradient::binary_crossentropy_gradient;
/// use approx::assert_relative_eq;
///
/// let predictions = array![0.7, 0.3, 0.9];
/// let targets = array![1.0, 0.0, 1.0];
///
/// let gradient = binary_crossentropy_gradient(&predictions.view(), &targets.view()).unwrap();
///
/// // gradients = -targets/predictions + (1-targets)/(1-predictions)
/// // = -[1.0, 0.0, 1.0]/[0.7, 0.3, 0.9] + [0.0, 1.0, 0.0]/[0.3, 0.7, 0.1]
/// // = [-1.428..., 0.0, -1.111...] + [0.0, 1.428..., 0.0]
/// // = [-1.428..., 1.428..., -1.111...]
///
/// assert_relative_eq!(gradient[0], -1.428571, epsilon = 1e-6);
/// assert_relative_eq!(gradient[1], 1.428571, epsilon = 1e-6);
/// assert_relative_eq!(gradient[2], -1.111111, epsilon = 1e-6);
/// ```
#[allow(dead_code)]
pub fn binary_crossentropy_gradient<F>(
    predictions: &ArrayView1<F>,
    targets: &ArrayView1<F>,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand,
{
    // Check dimensions compatibility
    if predictions.shape() != targets.shape() {
        return Err(LinalgError::ShapeError(format!(
            "Shape mismatch for binary_crossentropy_gradient: predictions has shape {:?} but targets has shape {:?}",
            predictions.shape(),
            targets.shape()
        )));
    }

    // Check that predictions are between 0 and 1
    for &p in predictions.iter() {
        if p <= F::zero() || p >= F::one() {
            return Err(LinalgError::InvalidInputError(
                "Predictions must be between 0 and 1 for binary cross-entropy".to_string(),
            ));
        }
    }

    // Check that targets are either 0 or 1
    for &t in targets.iter() {
        if (t - F::zero()).abs() > F::epsilon() && (t - F::one()).abs() > F::epsilon() {
            return Err(LinalgError::InvalidInputError(
                "Targets must be 0 or 1 for binary cross-entropy".to_string(),
            ));
        }
    }

    let one = F::one();
    let eps = F::from(1e-15).unwrap(); // Small epsilon to prevent division by zero

    // Compute -targets/(predictions+eps) + (1-targets)/(1-predictions+eps)
    // This is the gradient of binary cross-entropy with respect to predictions
    let mut gradient = Array1::zeros(predictions.len());
    for i in 0..predictions.len() {
        let p = predictions[i];
        let t = targets[i];
        let term1 = if t > F::epsilon() {
            -t / (p + eps)
        } else {
            F::zero()
        };
        let term2 = if (one - t) > F::epsilon() {
            (one - t) / (one - p + eps)
        } else {
            F::zero()
        };
        gradient[i] = term1 + term2;
    }

    Ok(gradient)
}

/// Calculate the gradient of softmax cross-entropy with respect to logits
///
/// Computes the gradient of softmax + cross-entropy loss with respect to logits (pre-softmax).
/// This is a common gradient calculation in multi-class classification tasks.
///
/// # Arguments
///
/// * `softmax_output` - The output of the softmax function (probabilities that sum to 1)
/// * `targets` - Target one-hot encoded vectors
///
/// # Returns
///
/// * The gradient of softmax cross-entropy with respect to logits
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::gradient::softmax_crossentropy_gradient;
/// use approx::assert_relative_eq;
///
/// // Softmax outputs (probabilities)
/// let softmax_output = array![[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]];
/// // One-hot encoded targets
/// let targets = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
///
/// let gradient = softmax_crossentropy_gradient(&softmax_output.view(), &targets.view()).unwrap();
///
/// // For each example, gradient = (softmax_output - targets) / batchsize
/// // = ([0.7, 0.2, 0.1] - [1.0, 0.0, 0.0]) / 2
/// // = [-0.15, 0.1, 0.05]
/// // For the second example:
/// // = ([0.3, 0.6, 0.1] - [0.0, 1.0, 0.0]) / 2
/// // = [0.15, -0.2, 0.05]
///
/// assert_relative_eq!(gradient[[0, 0]], -0.15, epsilon = 1e-10);
/// assert_relative_eq!(gradient[[0, 1]], 0.1, epsilon = 1e-10);
/// assert_relative_eq!(gradient[[0, 2]], 0.05, epsilon = 1e-10);
/// assert_relative_eq!(gradient[[1, 0]], 0.15, epsilon = 1e-10);
/// assert_relative_eq!(gradient[[1, 1]], -0.2, epsilon = 1e-10);
/// assert_relative_eq!(gradient[[1, 2]], 0.05, epsilon = 1e-10);
/// ```
#[allow(dead_code)]
pub fn softmax_crossentropy_gradient<F>(
    softmax_output: &ArrayView2<F>,
    targets: &ArrayView2<F>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + std::fmt::Display,
{
    // Check dimensions compatibility
    if softmax_output.shape() != targets.shape() {
        return Err(LinalgError::ShapeError(format!(
            "Shape mismatch for softmax_crossentropy_gradient: softmax_output has shape {:?} but targets has shape {:?}",
            softmax_output.shape(),
            targets.shape()
        )));
    }

    // Check that softmax outputs sum to 1 for each example
    let (batchsize, _num_classes) = softmax_output.dim();
    for i in 0..batchsize {
        let row_sum = softmax_output.slice(s![i, ..]).sum();
        if (row_sum - F::one()).abs() > F::from(1e-5).unwrap() {
            return Err(LinalgError::InvalidInputError(format!(
                "softmax_output row {i} does not sum to 1: sum is {row_sum}"
            )));
        }
    }

    // Check that targets are valid one-hot vectors
    for i in 0..batchsize {
        let row_sum = targets.slice(s![i, ..]).sum();
        if (row_sum - F::one()).abs() > F::from(1e-6).unwrap() {
            return Err(LinalgError::InvalidInputError(format!(
                "targets row {i} is not a valid one-hot vector: sum is {row_sum}"
            )));
        }

        // Check that only one element is (close to) 1, rest are 0
        let mut has_one = false;
        for val in targets.slice(s![i, ..]).iter() {
            if (*val - F::one()).abs() < F::from(1e-6).unwrap() {
                if has_one {
                    // More than one value close to 1
                    return Err(LinalgError::InvalidInputError(format!(
                        "targets row {i} is not a valid one-hot vector: multiple entries close to 1"
                    )));
                }
                has_one = true;
            } else if *val > F::from(1e-6).unwrap() {
                // Value is not close to 0 or 1
                return Err(LinalgError::InvalidInputError(format!(
                    "targets row {i} is not a valid one-hot vector: contains value {} not close to 0 or 1", *val
                )));
            }
        }

        if !has_one {
            return Err(LinalgError::InvalidInputError(format!(
                "targets row {i} is not a valid one-hot vector: no entry close to 1"
            )));
        }
    }

    let batchsize_f = F::from(batchsize).unwrap();

    // Compute softmax_output - targets
    let mut gradient = softmax_output.to_owned() - targets;

    // Scale by 1/batchsize
    gradient /= batchsize_f;

    Ok(gradient)
}

/// Calculate the Jacobian matrix for a function that maps from R^n to R^m
///
/// Computes a numerical approximation of the Jacobian matrix for a function
/// that takes an n-dimensional input and produces an m-dimensional output.
///
/// # Arguments
///
/// * `f` - A function that maps from R^n to R^m
/// * `x` - The point at which to evaluate the Jacobian
/// * `epsilon` - The step size for the finite difference approximation
///
/// # Returns
///
/// * The Jacobian matrix of shape (m, n)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_linalg::gradient::jacobian;
///
/// // Define a simple function R^2 -> R^3
/// // f(x,y) = [x^2 + y, 2*x + y^2, x*y]
/// let f = |v: &Array1<f64>| -> Array1<f64> {
///     let x = v[0];
///     let y = v[1];
///     array![x*x + y, 2.0*x + y*y, x*y]
/// };
///
/// let x = array![2.0, 3.0];  // Point at which to evaluate the Jacobian
/// let epsilon = 1e-5;
///
/// let jac = jacobian(&f, &x, epsilon).unwrap();
///
/// // Analytical Jacobian at (2,3) is:
/// // [2x, 1]     [4, 1]
/// // [2, 2y]  =  [2, 6]
/// // [y, x]      [3, 2]
///
/// assert!((jac[[0, 0]] - 4.0).abs() < 1e-4);
/// assert!((jac[[0, 1]] - 1.0).abs() < 1e-4);
/// assert!((jac[[1, 0]] - 2.0).abs() < 1e-4);
/// assert!((jac[[1, 1]] - 6.0).abs() < 1e-4);
/// assert!((jac[[2, 0]] - 3.0).abs() < 1e-4);
/// assert!((jac[[2, 1]] - 2.0).abs() < 1e-4);
/// ```
#[allow(dead_code)]
pub fn jacobian<F, G>(f: &G, x: &Array1<F>, epsilon: F) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand,
    G: Fn(&Array1<F>) -> Array1<F>,
{
    let n = x.len();

    // Evaluate function at the given point
    let f_x = f(x);
    let m = f_x.len();

    let mut jacobian = Array2::zeros((m, n));
    let two_epsilon = F::from(2.0).unwrap() * epsilon;

    // Compute each column of the Jacobian by central finite differences
    for j in 0..n {
        // Create forward and backward perturbations for central differences
        let mut x_forward = x.clone();
        let mut x_backward = x.clone();

        // Perturb jth component in both directions
        x_forward[j] = x[j] + epsilon;
        x_backward[j] = x[j] - epsilon;

        // Evaluate function at perturbed points
        let f_forward = f(&x_forward);
        let f_backward = f(&x_backward);

        // Compute jth column of Jacobian by central difference formula
        // This is more accurate than forward or backward differences
        for i in 0..m {
            jacobian[[i, j]] = (f_forward[i] - f_backward[i]) / two_epsilon;
        }
    }

    Ok(jacobian)
}

/// Calculate the Hessian matrix for a scalar-valued function
///
/// Computes a numerical approximation of the Hessian matrix (second derivatives)
/// for a function that takes an n-dimensional input and produces a scalar output.
///
/// # Arguments
///
/// * `f` - A function that maps from R^n to R
/// * `x` - The point at which to evaluate the Hessian
/// * `epsilon` - The step size for the finite difference approximation
///
/// # Returns
///
/// * The Hessian matrix of shape (n, n)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_linalg::gradient::hessian;
///
/// // Define a simple quadratic function: f(x,y) = x^2 + xy + 2y^2
/// let f = |v: &Array1<f64>| -> f64 {
///     let x = v[0];
///     let y = v[1];
///     x*x + x*y + 2.0*y*y
/// };
///
/// let x = array![1.0, 2.0];  // Point at which to evaluate the Hessian
/// let epsilon = 1e-5;
///
/// let hess = hessian(&f, &x, epsilon).unwrap();
///
/// // Analytical Hessian is:
/// // [∂²f/∂x², ∂²f/∂x∂y]   [2, 1]
/// // [∂²f/∂y∂x, ∂²f/∂y²] = [1, 4]
///
/// assert!((hess[[0, 0]] - 2.0).abs() < 1e-4);
/// assert!((hess[[0, 1]] - 1.0).abs() < 1e-4);
/// assert!((hess[[1, 0]] - 1.0).abs() < 1e-4);
/// assert!((hess[[1, 1]] - 4.0).abs() < 1e-4);
/// ```
#[allow(dead_code)]
pub fn hessian<F, G>(f: &G, x: &Array1<F>, epsilon: F) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand,
    G: Fn(&Array1<F>) -> F,
{
    let n = x.len();
    let mut hessian = Array2::zeros((n, n));

    let two = F::from(2.0).unwrap();
    let epsilon_squared = epsilon * epsilon;

    // Use central difference method for better accuracy
    let f_x = f(x);

    // Create arrays for the perturbed points
    for i in 0..n {
        for j in 0..=i {
            // Use symmetry: compute only lower triangle
            if i == j {
                // Diagonal elements: use central difference formula for second derivative
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();

                x_plus[i] = x[i] + epsilon;
                x_minus[i] = x[i] - epsilon;

                let f_plus = f(&x_plus);
                let f_minus = f(&x_minus);

                // Central difference formula for second derivative:
                // f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
                let h_ii = (f_plus - two * f_x + f_minus) / epsilon_squared;
                hessian[[i, i]] = h_ii;
            } else {
                // Off-diagonal elements (mixed partial derivatives): use central difference
                let mut x_plus_plus = x.clone();
                let mut x_plus_minus = x.clone();
                let mut x_minus_plus = x.clone();
                let mut x_minus_minus = x.clone();

                // (i+,j+): Both variables increased by epsilon
                x_plus_plus[i] = x[i] + epsilon;
                x_plus_plus[j] = x[j] + epsilon;

                // (i+,j-): First variable increased, second decreased
                x_plus_minus[i] = x[i] + epsilon;
                x_plus_minus[j] = x[j] - epsilon;

                // (i-,j+): First variable decreased, second increased
                x_minus_plus[i] = x[i] - epsilon;
                x_minus_plus[j] = x[j] + epsilon;

                // (i-,j-): Both variables decreased by epsilon
                x_minus_minus[i] = x[i] - epsilon;
                x_minus_minus[j] = x[j] - epsilon;

                // Evaluate function at all these points
                let f_plus_plus = f(&x_plus_plus);
                let f_plus_minus = f(&x_plus_minus);
                let f_minus_plus = f(&x_minus_plus);
                let f_minus_minus = f(&x_minus_minus);

                // Mixed partial derivative using central difference:
                // ∂²f/∂x∂y ≈ (f(x+h,y+h) - f(x+h,y-h) - f(x-h,y+h) + f(x-h,y-h)) / (4h²)
                let four = F::from(4.0).unwrap();
                let h_ij = (f_plus_plus - f_plus_minus - f_minus_plus + f_minus_minus)
                    / (four * epsilon_squared);

                hessian[[i, j]] = h_ij;
                hessian[[j, i]] = h_ij; // Hessian is symmetric
            }
        }
    }

    Ok(hessian)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mse_gradient() {
        // Test with a simple case
        let predictions = array![3.0, 1.0, 2.0];
        let targets = array![2.5, 0.5, 2.0];

        let gradient = mse_gradient(&predictions.view(), &targets.view()).unwrap();

        // gradients = 2 * (predictions - targets) / n
        // = 2 * ([3.0, 1.0, 2.0] - [2.5, 0.5, 2.0]) / 3
        // = 2 * [0.5, 0.5, 0.0] / 3
        // = [0.333..., 0.333..., 0.0]
        assert_relative_eq!(gradient[0], 1.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(gradient[1], 1.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(gradient[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_binary_crossentropy_gradient() {
        // Test with a simple case
        let predictions = array![0.7, 0.3, 0.9];
        let targets = array![1.0, 0.0, 1.0];

        let gradient = binary_crossentropy_gradient(&predictions.view(), &targets.view()).unwrap();

        // gradients = -targets/predictions + (1-targets)/(1-predictions)
        // = -[1.0, 0.0, 1.0]/[0.7, 0.3, 0.9] + [0.0, 1.0, 0.0]/[0.3, 0.7, 0.1]
        // = [-1.428..., 0.0, -1.111...] + [0.0, 1.428..., 0.0]
        // = [-1.428..., 1.428..., -1.111...]
        assert_relative_eq!(gradient[0], -1.428571, epsilon = 1e-6);
        assert_relative_eq!(gradient[1], 1.428571, epsilon = 1e-6);
        assert_relative_eq!(gradient[2], -1.111111, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax_crossentropy_gradient() {
        // Test with a simple case
        let softmax_output = array![[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]];
        let targets = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        let gradient =
            softmax_crossentropy_gradient(&softmax_output.view(), &targets.view()).unwrap();

        // For each example, gradient = (softmax_output - targets) / batchsize
        // = ([0.7, 0.2, 0.1] - [1.0, 0.0, 0.0]) / 2
        // = [-0.15, 0.1, 0.05]
        // For the second example:
        // = ([0.3, 0.6, 0.1] - [0.0, 1.0, 0.0]) / 2
        // = [0.15, -0.2, 0.05]
        assert_relative_eq!(gradient[[0, 0]], -0.15, epsilon = 1e-10);
        assert_relative_eq!(gradient[[0, 1]], 0.1, epsilon = 1e-10);
        assert_relative_eq!(gradient[[0, 2]], 0.05, epsilon = 1e-10);
        assert_relative_eq!(gradient[[1, 0]], 0.15, epsilon = 1e-10);
        assert_relative_eq!(gradient[[1, 1]], -0.2, epsilon = 1e-10);
        assert_relative_eq!(gradient[[1, 2]], 0.05, epsilon = 1e-10);
    }

    #[test]
    fn test_jacobian() {
        // Define a simple function R^2 -> R^3
        // f(x,y) = [x^2 + y, 2*x + y^2, x*y]
        let f = |v: &Array1<f64>| -> Array1<f64> {
            let x = v[0];
            let y = v[1];
            array![x * x + y, 2.0 * x + y * y, x * y]
        };

        let x = array![2.0, 3.0]; // Point at which to evaluate the Jacobian
        let epsilon = 1e-5;

        let jac = jacobian(&f, &x, epsilon).unwrap();

        // Analytical Jacobian at (2,3) is:
        // [2x, 1]     [4, 1]
        // [2, 2y]  =  [2, 6]
        // [y, x]      [3, 2]

        assert_relative_eq!(jac[[0, 0]], 4.0, epsilon = 1e-4);
        assert_relative_eq!(jac[[0, 1]], 1.0, epsilon = 1e-4);
        assert_relative_eq!(jac[[1, 0]], 2.0, epsilon = 1e-4);
        assert_relative_eq!(jac[[1, 1]], 6.0, epsilon = 1e-4);
        assert_relative_eq!(jac[[2, 0]], 3.0, epsilon = 1e-4);
        assert_relative_eq!(jac[[2, 1]], 2.0, epsilon = 1e-4);
    }

    #[test]
    fn test_hessian() {
        // A very simple quadratic function: f(x) = 2x²
        // Has constant second derivative: f''(x) = 4
        let f = |v: &Array1<f64>| -> f64 {
            let x = v[0];
            2.0 * x * x
        };

        let x = array![0.5];
        let epsilon = 1e-4;

        let hess = hessian(&f, &x, epsilon).unwrap();

        // The Hessian (second derivative) of f(x) = 2x² is 4
        assert_relative_eq!(hess[[0, 0]], 4.0, epsilon = 1e-2);
    }

    #[test]
    fn test_hessian_multidimensional() {
        // Multivariable function: f(x,y,z) = x²y + y²z + z²x
        let f = |v: &Array1<f64>| -> f64 {
            let x = v[0];
            let y = v[1];
            let z = v[2];
            x * x * y + y * y * z + z * z * x
        };

        let x = array![1.0, 1.0, 1.0];
        let epsilon = 1e-4;

        let hess = hessian(&f, &x, epsilon).unwrap();

        // Analytical Hessian at (1,1,1) for f(x,y,z) = x²y + y²z + z²x:
        // First-order derivatives:
        // ∂f/∂x = 2xy + z²
        // ∂f/∂y = x² + 2yz
        // ∂f/∂z = y² + 2zx
        //
        // Second-order derivatives:
        // ∂²f/∂x² = 2y = 2 (at point [1,1,1])
        // ∂²f/∂y² = 2z = 2 (at point [1,1,1])
        // ∂²f/∂z² = 2x = 2 (at point [1,1,1])
        // ∂²f/∂x∂y = ∂²f/∂y∂x = 2x = 2 (at point [1,1,1])
        // ∂²f/∂y∂z = ∂²f/∂z∂y = 2y = 2 (at point [1,1,1])
        // ∂²f/∂z∂x = ∂²f/∂x∂z = 2z = 2 (at point [1,1,1])

        // Diagonal elements
        assert_relative_eq!(hess[[0, 0]], 2.0, epsilon = 1e-2);
        assert_relative_eq!(hess[[1, 1]], 2.0, epsilon = 1e-2);
        assert_relative_eq!(hess[[2, 2]], 2.0, epsilon = 1e-2);

        // Off-diagonal elements
        assert_relative_eq!(hess[[0, 1]], 2.0, epsilon = 1e-2);
        assert_relative_eq!(hess[[1, 0]], 2.0, epsilon = 1e-2);
        assert_relative_eq!(hess[[1, 2]], 2.0, epsilon = 1e-2);
        assert_relative_eq!(hess[[2, 1]], 2.0, epsilon = 1e-2);
        assert_relative_eq!(hess[[0, 2]], 2.0, epsilon = 1e-2);
        assert_relative_eq!(hess[[2, 0]], 2.0, epsilon = 1e-2);
    }

    #[test]
    fn test_hessian_quadratic_form() {
        // Quadratic form: f(x,y) = x² + xy + 2y²
        let f = |v: &Array1<f64>| -> f64 {
            let x = v[0];
            let y = v[1];
            x * x + x * y + 2.0 * y * y
        };

        let x = array![1.0, 2.0];
        let epsilon = 1e-5;

        let hess = hessian(&f, &x, epsilon).unwrap();

        // Analytical Hessian is:
        // [∂²f/∂x², ∂²f/∂x∂y]   [2, 1]
        // [∂²f/∂y∂x, ∂²f/∂y²] = [1, 4]
        assert_relative_eq!(hess[[0, 0]], 2.0, epsilon = 1e-4);
        assert_relative_eq!(hess[[0, 1]], 1.0, epsilon = 1e-4);
        assert_relative_eq!(hess[[1, 0]], 1.0, epsilon = 1e-4);
        assert_relative_eq!(hess[[1, 1]], 4.0, epsilon = 1e-4);
    }

    #[test]
    fn test_mse_gradient_dimension_error() {
        let predictions = array![1.0, 2.0, 3.0];
        let targets = array![1.0, 2.0];

        let result = mse_gradient(&predictions.view(), &targets.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_crossentropy_gradient_invalid_predictions() {
        let predictions = array![0.5, 1.2, 0.3]; // Contains value > 1
        let targets = array![1.0, 0.0, 1.0];

        let result = binary_crossentropy_gradient(&predictions.view(), &targets.view());
        assert!(result.is_err());

        let predictions = array![0.5, -0.1, 0.3]; // Contains value < 0
        let targets = array![1.0, 0.0, 1.0];

        let result = binary_crossentropy_gradient(&predictions.view(), &targets.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_crossentropy_gradient_invalid_targets() {
        let predictions = array![0.5, 0.7, 0.3];
        let targets = array![1.0, 0.5, 1.0]; // Contains value neither 0 nor 1

        let result = binary_crossentropy_gradient(&predictions.view(), &targets.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_softmax_crossentropy_gradient_invalid_softmax() {
        let softmax_output = array![[0.7, 0.2, 0.2], [0.3, 0.6, 0.1]]; // First row sums to 1.1
        let targets = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        let result = softmax_crossentropy_gradient(&softmax_output.view(), &targets.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_softmax_crossentropy_gradient_invalid_targets() {
        let softmax_output = array![[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]];
        let targets = array![[1.0, 0.0, 0.0], [0.3, 0.3, 0.4]]; // Second row is definitely not one-hot (sum = 1 but not one-hot)

        let result = softmax_crossentropy_gradient(&softmax_output.view(), &targets.view());
        assert!(result.is_err());
    }
}
