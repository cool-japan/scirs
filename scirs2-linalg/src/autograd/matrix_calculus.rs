//! Matrix calculus operations with automatic differentiation support
//!
//! This module provides differentiable matrix calculus operations like
//! gradients, Jacobians, and Hessians that integrate with the autograd system.

use num_traits::{Float, NumAssign};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

/// Derivatives of specific matrix operations
pub mod matrix_derivatives {
    use super::*;
    use ndarray::{Array2, ArrayView2};
    use num_traits::{Float, One, Zero};

    /// Compute the derivative of the determinant of a matrix
    ///
    /// For a matrix X, d/dX det(X) = det(X) * (X^-1)^T
    ///
    /// # Arguments
    /// * `x` - Input matrix
    ///
    /// # Returns
    /// * Matrix containing the derivative of the determinant
    pub fn det_derivative<F>(x: &ArrayView2<F>) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + Sum + Send + Sync + 'static + ndarray::ScalarOperand,
    {
        if x.nrows() != x.ncols() {
            return Err(LinalgError::DimensionError(
                "Matrix must be square for determinant derivative".to_string(),
            ));
        }

        // Compute determinant
        let det_val = crate::basic::det(x, None)?;

        // Compute inverse
        let inv_x = crate::basic::inv(x, None)?;

        // Return det(X) * (X^-1)^T
        let result = inv_x.t().to_owned() * det_val;
        Ok(result)
    }

    /// Compute the derivative of the matrix inverse
    ///
    /// For a matrix X, d/dX inv(X) = -inv(X) ⊗ inv(X)^T
    /// This returns a function that, given a direction dX, computes d/dX inv(X)[dX]
    ///
    /// # Arguments
    /// * `x` - Input matrix
    /// * `dx` - Direction matrix for the derivative
    ///
    /// # Returns
    /// * Matrix containing the directional derivative of the inverse
    pub fn inv_derivative<F>(x: &ArrayView2<F>, dx: &ArrayView2<F>) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + Sum + Send + Sync + 'static + ndarray::ScalarOperand,
    {
        if x.nrows() != x.ncols() {
            return Err(LinalgError::DimensionError(
                "Matrix must be square for inverse derivative".to_string(),
            ));
        }

        if x.shape() != dx.shape() {
            return Err(LinalgError::DimensionError(
                "Direction matrix must have same shape as input matrix".to_string(),
            ));
        }

        // Compute inverse
        let inv_x = crate::basic::inv(x, None)?;

        // Compute -inv(X) * dX * inv(X)
        let temp = inv_x.dot(dx);
        let result = inv_x.dot(&temp) * (-F::one());
        Ok(result)
    }

    /// Compute the derivative of the matrix trace
    ///
    /// For a matrix X, d/dX tr(X) = I (identity matrix)
    ///
    /// # Arguments
    /// * `x` - Input matrix
    ///
    /// # Returns
    /// * Identity matrix (derivative of trace)
    pub fn trace_derivative<F>(x: &ArrayView2<F>) -> LinalgResult<Array2<F>>
    where
        F: Float + Zero + One,
    {
        if x.nrows() != x.ncols() {
            return Err(LinalgError::DimensionError(
                "Matrix must be square for trace derivative".to_string(),
            ));
        }

        let n = x.nrows();
        let mut result = Array2::zeros((n, n));
        for i in 0..n {
            result[[i, i]] = F::one();
        }
        Ok(result)
    }

    /// Compute the derivative of matrix multiplication X*Y with respect to X
    ///
    /// For matrices X and Y, d/dX (X*Y) = Y^T ⊗ I
    /// This function computes the directional derivative given dX
    ///
    /// # Arguments
    /// * `_x` - Left matrix (unused for this directional derivative)
    /// * `y` - Right matrix  
    /// * `dx` - Direction for derivative with respect to X
    ///
    /// # Returns
    /// * Directional derivative dX * Y
    pub fn matmul_derivative_left<F>(
        _x: &ArrayView2<F>,
        y: &ArrayView2<F>,
        dx: &ArrayView2<F>,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + 'static,
    {
        // d/dX (X*Y) applied to direction dX gives dX * Y
        Ok(dx.dot(y))
    }

    /// Compute the derivative of matrix multiplication X*Y with respect to Y
    ///
    /// For matrices X and Y, d/dY (X*Y) = I ⊗ X^T
    /// This function computes the directional derivative given dY
    ///
    /// # Arguments
    /// * `x` - Left matrix
    /// * `_y` - Right matrix (unused for this directional derivative)
    /// * `dy` - Direction for derivative with respect to Y
    ///
    /// # Returns
    /// * Directional derivative X * dY
    pub fn matmul_derivative_right<F>(
        x: &ArrayView2<F>,
        _y: &ArrayView2<F>,
        dy: &ArrayView2<F>,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + 'static,
    {
        // d/dY (X*Y) applied to direction dY gives X * dY
        Ok(x.dot(dy))
    }

    /// Compute the derivative of the Frobenius norm
    ///
    /// For a matrix X, d/dX ||X||_F = X / ||X||_F
    ///
    /// # Arguments
    /// * `x` - Input matrix
    ///
    /// # Returns
    /// * Matrix containing the derivative of the Frobenius norm
    pub fn frobenius_norm_derivative<F>(x: &ArrayView2<F>) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + Sum + Send + Sync + 'static + ndarray::ScalarOperand,
    {
        let norm_val = crate::norm::matrix_norm(x, "frobenius", None)?;

        if norm_val.abs() < F::epsilon() {
            // If norm is zero, derivative is undefined - return zeros
            return Ok(Array2::zeros(x.dim()));
        }

        let result = x.mapv(|elem| elem / norm_val);
        Ok(result.to_owned())
    }

    /// Compute the derivative of matrix power X^n for integer n
    ///
    /// For a matrix X and integer n, d/dX X^n = sum_{k=0}^{n-1} X^k * dX * X^{n-1-k}
    /// This function computes the directional derivative given dX
    ///
    /// # Arguments
    /// * `x` - Input matrix
    /// * `n` - Power (must be positive)
    /// * `dx` - Direction for derivative
    ///
    /// # Returns
    /// * Directional derivative of matrix power
    pub fn matrix_power_derivative<F>(
        x: &ArrayView2<F>,
        n: i32,
        dx: &ArrayView2<F>,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + Sum + Send + Sync + 'static + ndarray::ScalarOperand,
    {
        if x.nrows() != x.ncols() {
            return Err(LinalgError::DimensionError(
                "Matrix must be square for power derivative".to_string(),
            ));
        }

        if n <= 0 {
            return Err(LinalgError::InvalidInputError(
                "Power must be positive for this derivative computation".to_string(),
            ));
        }

        if n == 1 {
            return Ok(dx.to_owned());
        }

        // For n=2: d/dX X^2 = dX*X + X*dX
        if n == 2 {
            let term1 = dx.dot(x);
            let term2 = x.dot(dx);
            return Ok(term1 + term2);
        }

        // For higher powers, we use the general formula
        // This is a simplified implementation - a full implementation would
        // compute the sum more efficiently
        let mut result = Array2::zeros(x.dim());

        // Compute powers of X up to n-1
        let mut x_powers = Vec::new();
        x_powers.push(Array2::eye(x.nrows())); // X^0 = I

        let mut current_power = x.to_owned();
        x_powers.push(current_power.clone()); // X^1

        for _ in 2..n {
            current_power = current_power.dot(x);
            x_powers.push(current_power.clone());
        }

        // Sum over k: X^k * dX * X^{n-1-k}
        for k in 0..n {
            let left_power = &x_powers[k as usize];
            let right_power = &x_powers[(n - 1 - k) as usize];

            let term = left_power.dot(dx).dot(right_power);
            result = result + term;
        }

        Ok(result)
    }

    /// Compute the derivative of matrix exponential
    ///
    /// For a matrix X, d/dX exp(X)[dX] ≈ exp(X) * dX + dX * exp(X) (first-order approximation)
    /// The exact formula involves an integral: ∫₀¹ exp(sX) * dX * exp((1-s)X) ds
    ///
    /// # Arguments
    /// * `x` - Input matrix
    /// * `dx` - Direction for derivative
    ///
    /// # Returns
    /// * Directional derivative of matrix exponential
    pub fn matrix_exp_derivative<F>(
        x: &ArrayView2<F>,
        dx: &ArrayView2<F>,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + Sum + Send + Sync + 'static + ndarray::ScalarOperand,
    {
        if x.nrows() != x.ncols() {
            return Err(LinalgError::DimensionError(
                "Matrix must be square for exponential derivative".to_string(),
            ));
        }

        if x.shape() != dx.shape() {
            return Err(LinalgError::DimensionError(
                "Direction matrix must have same shape as input matrix".to_string(),
            ));
        }

        // Compute matrix exponential
        let exp_x = crate::matrix_functions::expm(x, None)?;

        // First-order approximation: (exp(X) * dX + dX * exp(X)) / 2
        let term1 = exp_x.dot(dx);
        let term2 = dx.dot(&exp_x);
        let result = (term1 + term2) * F::from(0.5).unwrap();

        Ok(result)
    }

    /// Compute the derivative of matrix logarithm
    ///
    /// For a matrix X, d/dX log(X)[dX] = X^{-1} * dX
    ///
    /// # Arguments
    /// * `x` - Input matrix
    /// * `dx` - Direction for derivative
    ///
    /// # Returns
    /// * Directional derivative of matrix logarithm
    pub fn matrix_log_derivative<F>(
        x: &ArrayView2<F>,
        dx: &ArrayView2<F>,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + Sum + Send + Sync + 'static + ndarray::ScalarOperand,
    {
        if x.nrows() != x.ncols() {
            return Err(LinalgError::DimensionError(
                "Matrix must be square for logarithm derivative".to_string(),
            ));
        }

        if x.shape() != dx.shape() {
            return Err(LinalgError::DimensionError(
                "Direction matrix must have same shape as input matrix".to_string(),
            ));
        }

        // Compute inverse
        let inv_x = crate::basic::inv(x, None)?;

        // Return X^{-1} * dX
        Ok(inv_x.dot(dx))
    }
}

/// Matrix differential operators for matrix-valued functions
pub mod differential_operators {
    use super::*;
    use ndarray::{Array2, Array3};

    /// Compute the divergence of a matrix field
    ///
    /// For a matrix field F(x,y) = [F_ij(x,y)], the divergence is
    /// div(F) = sum_i ∂F_ii/∂x_i (trace of the spatial gradient)
    ///
    /// # Arguments
    /// * `field` - 3D array where field[i][j] contains the (i,j) component of the matrix field
    /// * `spacing` - Grid spacing for finite differences
    ///
    /// # Returns
    /// * Scalar field containing the divergence
    pub fn matrix_divergence<F>(field: &Array3<F>, spacing: F) -> LinalgResult<Array2<F>>
    where
        F: Float + Copy,
    {
        let (nx, ny, _) = field.dim();
        let mut result = Array2::zeros((nx, ny));

        // Compute divergence using finite differences
        for i in 1..(nx - 1) {
            for j in 1..(ny - 1) {
                // ∂F_xx/∂x + ∂F_yy/∂y (diagonal terms)
                let df_xx_dx = (field[[i + 1, j, 0]] - field[[i - 1, j, 0]])
                    / (F::from(2.0).unwrap() * spacing);
                let df_yy_dy = (field[[i, j + 1, 3]] - field[[i, j - 1, 3]])
                    / (F::from(2.0).unwrap() * spacing);

                result[[i, j]] = df_xx_dx + df_yy_dy;
            }
        }

        Ok(result)
    }

    /// Compute the curl of a matrix field (generalized to matrix-valued functions)
    ///
    /// For a 2x2 matrix field F = [[F11, F12], [F21, F22]], the curl is
    /// curl(F) = [[∂F12/∂x - ∂F11/∂y, ∂F22/∂x - ∂F21/∂y],
    ///           [∂F11/∂x - ∂F12/∂y, ∂F21/∂x - ∂F22/∂y]]
    ///
    /// # Arguments
    /// * `field` - 3D array where field[i][j] contains matrix components
    /// * `spacing` - Grid spacing for finite differences
    ///
    /// # Returns
    /// * Matrix field containing the curl
    pub fn matrix_curl<F>(field: &Array3<F>, spacing: F) -> LinalgResult<Array3<F>>
    where
        F: Float + Copy,
    {
        let (nx, ny, ncomp) = field.dim();

        if ncomp != 4 {
            return Err(LinalgError::DimensionError(
                "Matrix field must have 4 components for 2x2 matrices".to_string(),
            ));
        }

        let mut result = Array3::zeros((nx, ny, 4));

        // Compute curl using finite differences
        for i in 1..(nx - 1) {
            for j in 1..(ny - 1) {
                // Component (0,0): ∂F12/∂x - ∂F11/∂y
                let df12_dx = (field[[i + 1, j, 1]] - field[[i - 1, j, 1]])
                    / (F::from(2.0).unwrap() * spacing);
                let df11_dy = (field[[i, j + 1, 0]] - field[[i, j - 1, 0]])
                    / (F::from(2.0).unwrap() * spacing);
                result[[i, j, 0]] = df12_dx - df11_dy;

                // Component (0,1): ∂F22/∂x - ∂F21/∂y
                let df22_dx = (field[[i + 1, j, 3]] - field[[i - 1, j, 3]])
                    / (F::from(2.0).unwrap() * spacing);
                let df21_dy = (field[[i, j + 1, 2]] - field[[i, j - 1, 2]])
                    / (F::from(2.0).unwrap() * spacing);
                result[[i, j, 1]] = df22_dx - df21_dy;

                // Component (1,0): ∂F11/∂x - ∂F12/∂y
                let df11_dx = (field[[i + 1, j, 0]] - field[[i - 1, j, 0]])
                    / (F::from(2.0).unwrap() * spacing);
                let df12_dy = (field[[i, j + 1, 1]] - field[[i, j - 1, 1]])
                    / (F::from(2.0).unwrap() * spacing);
                result[[i, j, 2]] = df11_dx - df12_dy;

                // Component (1,1): ∂F21/∂x - ∂F22/∂y
                let df21_dx = (field[[i + 1, j, 2]] - field[[i - 1, j, 2]])
                    / (F::from(2.0).unwrap() * spacing);
                let df22_dy = (field[[i, j + 1, 3]] - field[[i, j - 1, 3]])
                    / (F::from(2.0).unwrap() * spacing);
                result[[i, j, 3]] = df21_dx - df22_dy;
            }
        }

        Ok(result)
    }

    /// Compute the Laplacian of a matrix field
    ///
    /// For a matrix field F, the Laplacian is computed component-wise:
    /// (∇²F)_ij = ∇²F_ij = ∂²F_ij/∂x² + ∂²F_ij/∂y²
    ///
    /// # Arguments
    /// * `field` - 3D array containing matrix field components
    /// * `spacing` - Grid spacing for finite differences
    ///
    /// # Returns
    /// * Matrix field containing the Laplacian
    pub fn matrix_laplacian<F>(field: &Array3<F>, spacing: F) -> LinalgResult<Array3<F>>
    where
        F: Float + Copy,
    {
        let (nx, ny, ncomp) = field.dim();
        let mut result = Array3::zeros((nx, ny, ncomp));

        let spacing_sq = spacing * spacing;

        // Compute Laplacian for each component
        for comp in 0..ncomp {
            for i in 1..(nx - 1) {
                for j in 1..(ny - 1) {
                    // ∂²F/∂x² + ∂²F/∂y² using finite differences
                    let d2f_dx2 = (field[[i + 1, j, comp]]
                        - F::from(2.0).unwrap() * field[[i, j, comp]]
                        + field[[i - 1, j, comp]])
                        / spacing_sq;
                    let d2f_dy2 = (field[[i, j + 1, comp]]
                        - F::from(2.0).unwrap() * field[[i, j, comp]]
                        + field[[i, j - 1, comp]])
                        / spacing_sq;

                    result[[i, j, comp]] = d2f_dx2 + d2f_dy2;
                }
            }
        }

        Ok(result)
    }

    /// Compute the gradient of a matrix field
    ///
    /// For a matrix field F(x,y), the gradient is a 3D array where
    /// gradient[i][j][k] contains ∂F_k/∂x_i at position (i,j)
    ///
    /// # Arguments
    /// * `field` - 3D array containing matrix field components
    /// * `spacing` - Grid spacing for finite differences
    ///
    /// # Returns
    /// * 4D array containing the gradient (gradient[x][y][component][direction])
    pub fn matrix_gradient<F>(field: &Array3<F>, spacing: F) -> LinalgResult<ndarray::Array4<F>>
    where
        F: Float + Copy,
    {
        let (nx, ny, ncomp) = field.dim();
        let mut result = ndarray::Array4::zeros((nx, ny, ncomp, 2)); // 2 for x and y directions

        // Compute gradient using finite differences
        for comp in 0..ncomp {
            for i in 1..(nx - 1) {
                for j in 1..(ny - 1) {
                    // ∂F/∂x
                    let df_dx = (field[[i + 1, j, comp]] - field[[i - 1, j, comp]])
                        / (F::from(2.0).unwrap() * spacing);
                    result[[i, j, comp, 0]] = df_dx;

                    // ∂F/∂y
                    let df_dy = (field[[i, j + 1, comp]] - field[[i, j - 1, comp]])
                        / (F::from(2.0).unwrap() * spacing);
                    result[[i, j, comp, 1]] = df_dy;
                }
            }
        }

        Ok(result)
    }
}

/// Support for matrix-valued functions with enhanced derivative tracking
pub mod matrix_functions {
    use super::*;
    use ndarray::{Array2, ArrayView2};

    /// A trait for matrix-valued functions that support differentiation
    pub trait DifferentiableMatrixFunction<F: Float> {
        /// Evaluate the function at a given matrix
        fn evaluate(&self, x: &ArrayView2<F>) -> LinalgResult<Array2<F>>;

        /// Compute the directional derivative at x in direction dx
        fn directional_derivative(
            &self,
            x: &ArrayView2<F>,
            dx: &ArrayView2<F>,
        ) -> LinalgResult<Array2<F>>;

        /// Compute the gradient (if the function is scalar-valued)
        fn gradient(&self, _x: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
            Err(LinalgError::NotImplementedError(
                "Gradient not implemented for this matrix function".to_string(),
            ))
        }
    }

    /// Matrix exponential function with derivatives
    pub struct MatrixExp;

    impl<F: Float + NumAssign + Sum + Send + Sync + 'static + ndarray::ScalarOperand>
        DifferentiableMatrixFunction<F> for MatrixExp
    {
        fn evaluate(&self, x: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
            crate::matrix_functions::expm(x, None)
        }

        fn directional_derivative(
            &self,
            x: &ArrayView2<F>,
            dx: &ArrayView2<F>,
        ) -> LinalgResult<Array2<F>> {
            super::matrix_derivatives::matrix_exp_derivative(x, dx)
        }
    }

    /// Matrix logarithm function with derivatives  
    pub struct MatrixLog;

    impl<F: Float + NumAssign + Sum + Send + Sync + 'static + ndarray::ScalarOperand>
        DifferentiableMatrixFunction<F> for MatrixLog
    {
        fn evaluate(&self, x: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
            crate::matrix_functions::logm(x)
        }

        fn directional_derivative(
            &self,
            x: &ArrayView2<F>,
            dx: &ArrayView2<F>,
        ) -> LinalgResult<Array2<F>> {
            super::matrix_derivatives::matrix_log_derivative(x, dx)
        }
    }

    /// Matrix power function with derivatives
    pub struct MatrixPower {
        pub power: i32,
    }

    impl<F: Float + NumAssign + Sum + Send + Sync + 'static + ndarray::ScalarOperand>
        DifferentiableMatrixFunction<F> for MatrixPower
    {
        fn evaluate(&self, x: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
            crate::basic::matrix_power(x, self.power, None)
        }

        fn directional_derivative(
            &self,
            x: &ArrayView2<F>,
            dx: &ArrayView2<F>,
        ) -> LinalgResult<Array2<F>> {
            super::matrix_derivatives::matrix_power_derivative(x, self.power, dx)
        }
    }

    /// Compose matrix functions with proper derivative tracking
    pub fn compose_functions<
        F: Float + NumAssign + Sum + Send + Sync + 'static + ndarray::ScalarOperand,
    >(
        f: &dyn DifferentiableMatrixFunction<F>,
        g: &dyn DifferentiableMatrixFunction<F>,
        x: &ArrayView2<F>,
        dx: &ArrayView2<F>,
    ) -> LinalgResult<Array2<F>> {
        // Chain rule: d/dX f(g(X))[dX] = df/dY|_{Y=g(X)} [dg/dX|_X [dX]]
        let g_x = g.evaluate(x)?;
        let dg_dx = g.directional_derivative(x, dx)?;
        let df_dy = f.directional_derivative(&g_x.view(), &dg_dx.view())?;
        Ok(df_dy)
    }
}

/// Finite difference utilities for gradient computation
pub mod finite_difference {
    use super::*;
    use ndarray::{Array2, ArrayView2};

    /// Compute gradient using finite differences
    ///
    /// # Arguments
    /// * `f` - Function that takes a matrix and returns a scalar
    /// * `x` - Input matrix at which to evaluate the gradient
    /// * `epsilon` - Step size for finite difference approximation
    ///
    /// # Returns
    /// * Matrix containing the gradient
    pub fn gradient_finite_diff<F>(
        f: impl Fn(&ArrayView2<F>) -> LinalgResult<F>,
        x: &ArrayView2<F>,
        epsilon: Option<F>,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + Copy,
    {
        let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt());
        let (m, n) = x.dim();
        let mut grad = Array2::zeros((m, n));

        // Compute gradient using finite differences
        for i in 0..m {
            for j in 0..n {
                // Create perturbed matrices x + eps*e_ij and x - eps*e_ij
                let mut x_plus = x.to_owned();
                x_plus[[i, j]] = x_plus[[i, j]] + eps;

                let mut x_minus = x.to_owned();
                x_minus[[i, j]] = x_minus[[i, j]] - eps;

                // Evaluate function at perturbed points
                let f_plus = f(&x_plus.view())?;
                let f_minus = f(&x_minus.view())?;

                // Compute central difference approximation
                grad[[i, j]] = (f_plus - f_minus) / (F::from(2.0).unwrap() * eps);
            }
        }

        Ok(grad)
    }

    /// Compute Jacobian using finite differences for matrix-valued functions
    ///
    /// # Arguments
    /// * `f` - Function that takes a matrix and returns a matrix
    /// * `x` - Input matrix at which to evaluate the Jacobian
    /// * `epsilon` - Step size for finite difference approximation
    ///
    /// # Returns
    /// * 4D array containing the Jacobian
    pub fn jacobian_finite_diff<F>(
        f: impl Fn(&ArrayView2<F>) -> LinalgResult<Array2<F>>,
        x: &ArrayView2<F>,
        epsilon: Option<F>,
    ) -> LinalgResult<ndarray::Array4<F>>
    where
        F: Float + Copy,
    {
        let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt());
        let (m, n) = x.dim();

        // Evaluate function at base point to get output dimensions
        let f_x = f(x)?;
        let (p, q) = f_x.dim();

        let mut jac = ndarray::Array4::zeros((p, q, m, n));

        // Compute Jacobian using finite differences
        for i in 0..m {
            for j in 0..n {
                // Create perturbed matrix x + eps*e_ij
                let mut x_plus = x.to_owned();
                x_plus[[i, j]] = x_plus[[i, j]] + eps;

                // Evaluate function at perturbed point
                let f_plus = f(&x_plus.view())?;

                // Compute forward difference approximation
                for p_idx in 0..p {
                    for q_idx in 0..q {
                        jac[[p_idx, q_idx, i, j]] =
                            (f_plus[[p_idx, q_idx]] - f_x[[p_idx, q_idx]]) / eps;
                    }
                }
            }
        }

        Ok(jac)
    }
}

// Re-export key functionality
pub use differential_operators::*;
pub use finite_difference::*;
pub use matrix_derivatives::*;
pub use matrix_functions::*;
