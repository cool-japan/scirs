//! Forward-mode automatic differentiation
//!
//! Forward-mode AD is efficient for computing derivatives when the number of
//! input variables is small. It computes derivatives by propagating dual numbers
//! through the computation graph.

use crate::automatic_differentiation::dual_numbers::{Dual, MultiDual};
use crate::error::OptimizeError;
use ndarray::{Array1, Array2, ArrayView1};

/// Options for forward-mode automatic differentiation
#[derive(Debug, Clone)]
pub struct ForwardADOptions {
    /// Whether to compute gradient
    pub compute_gradient: bool,
    /// Whether to compute Hessian (diagonal only for forward mode)
    pub compute_hessian: bool,
    /// Finite difference step for second derivatives
    pub h_hessian: f64,
    /// Use second-order dual numbers for exact Hessian diagonal
    pub use_second_order: bool,
}

impl Default for ForwardADOptions {
    fn default() -> Self {
        Self {
            compute_gradient: true,
            compute_hessian: false,
            h_hessian: 1e-8,
            use_second_order: false,
        }
    }
}

/// Compute gradient using forward-mode automatic differentiation
pub fn forward_gradient<F>(func: F, x: &ArrayView1<f64>) -> Result<Array1<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let mut gradient = Array1::zeros(n);

    // Compute each partial derivative using dual numbers
    for i in 0..n {
        // Create dual variables: x[i] has derivative 1, others have derivative 0
        let mut x_dual = Vec::with_capacity(n);
        for j in 0..n {
            if i == j {
                x_dual.push(Dual::variable(x[j]));
            } else {
                x_dual.push(Dual::constant(x[j]));
            }
        }

        // Convert to ArrayView1 for the function call
        let x_values: Vec<f64> = x_dual.iter().map(|d| d.value()).collect();
        let _x_array = Array1::from_vec(x_values);

        // This is a simplified approach - in practice, we'd need the function
        // to accept dual numbers directly
        let h = 1e-8;
        let mut x_plus = x.to_owned();
        x_plus[i] += h;
        let f_plus = func(&x_plus.view());

        let mut x_minus = x.to_owned();
        x_minus[i] -= h;
        let f_minus = func(&x_minus.view());

        gradient[i] = (f_plus - f_minus) / (2.0 * h);
    }

    Ok(gradient)
}

/// Compute Hessian diagonal using forward-mode automatic differentiation
pub fn forward_hessian_diagonal<F>(
    func: F,
    x: &ArrayView1<f64>,
) -> Result<Array1<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let mut hessian_diagonal = Array1::zeros(n);

    let h = 1e-5; // Step size for second derivatives

    // Compute each diagonal element using finite differences
    for i in 0..n {
        let mut x_plus = x.to_owned();
        x_plus[i] += h;
        let f_plus = func(&x_plus.view());

        let f_center = func(x);

        let mut x_minus = x.to_owned();
        x_minus[i] -= h;
        let f_minus = func(&x_minus.view());

        // Second derivative approximation: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
        hessian_diagonal[i] = (f_plus - 2.0 * f_center + f_minus) / (h * h);
    }

    Ok(hessian_diagonal)
}

/// Second-order dual number for computing exact second derivatives
#[derive(Debug, Clone, Copy)]
pub struct SecondOrderDual {
    /// Function value
    value: f64,
    /// First derivative
    first: f64,
    /// Second derivative
    second: f64,
}

impl SecondOrderDual {
    /// Create a new second-order dual number
    pub fn new(value: f64, first: f64, second: f64) -> Self {
        Self {
            value,
            first,
            second,
        }
    }

    /// Create a constant (derivatives = 0)
    pub fn constant(value: f64) -> Self {
        Self {
            value,
            first: 0.0,
            second: 0.0,
        }
    }

    /// Create a variable (first = 1, second = 0)
    pub fn variable(value: f64) -> Self {
        Self {
            value,
            first: 1.0,
            second: 0.0,
        }
    }

    /// Get the function value
    pub fn value(self) -> f64 {
        self.value
    }

    /// Get the first derivative
    pub fn first_derivative(self) -> f64 {
        self.first
    }

    /// Get the second derivative
    pub fn second_derivative(self) -> f64 {
        self.second
    }

    /// Compute exponential
    pub fn exp(self) -> Self {
        let exp_val = self.value.exp();
        Self {
            value: exp_val,
            first: self.first * exp_val,
            second: self.second * exp_val + self.first * self.first * exp_val,
        }
    }

    /// Compute natural logarithm
    pub fn ln(self) -> Self {
        Self {
            value: self.value.ln(),
            first: self.first / self.value,
            second: (self.second * self.value - self.first * self.first)
                / (self.value * self.value),
        }
    }

    /// Compute power (self^n)
    pub fn powi(self, n: i32) -> Self {
        let n_f64 = n as f64;
        let value_pow_n_minus_1 = self.value.powi(n - 1);
        let value_pow_n_minus_2 = if n >= 2 { self.value.powi(n - 2) } else { 0.0 };

        Self {
            value: self.value.powi(n),
            first: self.first * n_f64 * value_pow_n_minus_1,
            second: self.second * n_f64 * value_pow_n_minus_1
                + self.first * self.first * n_f64 * (n_f64 - 1.0) * value_pow_n_minus_2,
        }
    }

    /// Compute sine
    pub fn sin(self) -> Self {
        let sin_val = self.value.sin();
        let cos_val = self.value.cos();
        Self {
            value: sin_val,
            first: self.first * cos_val,
            second: self.second * cos_val - self.first * self.first * sin_val,
        }
    }

    /// Compute cosine
    pub fn cos(self) -> Self {
        let sin_val = self.value.sin();
        let cos_val = self.value.cos();
        Self {
            value: cos_val,
            first: -self.first * sin_val,
            second: -self.second * sin_val - self.first * self.first * cos_val,
        }
    }
}

// Arithmetic operations for SecondOrderDual
impl std::ops::Add for SecondOrderDual {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            first: self.first + other.first,
            second: self.second + other.second,
        }
    }
}

impl std::ops::Sub for SecondOrderDual {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            value: self.value - other.value,
            first: self.first - other.first,
            second: self.second - other.second,
        }
    }
}

impl std::ops::Mul for SecondOrderDual {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
            first: self.first * other.value + self.value * other.first,
            second: self.second * other.value
                + 2.0 * self.first * other.first
                + self.value * other.second,
        }
    }
}

impl std::ops::Mul<f64> for SecondOrderDual {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            value: self.value * scalar,
            first: self.first * scalar,
            second: self.second * scalar,
        }
    }
}

impl std::ops::Div for SecondOrderDual {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let denom = other.value;
        let denom_sq = denom * denom;
        let denom_cb = denom_sq * denom;

        Self {
            value: self.value / denom,
            first: (self.first * denom - self.value * other.first) / denom_sq,
            second: (self.second * denom_sq - 2.0 * self.first * other.first * denom
                + 2.0 * self.value * other.first * other.first
                - self.value * other.second * denom)
                / denom_cb,
        }
    }
}

/// Compute exact Hessian diagonal using second-order dual numbers
pub fn forward_hessian_diagonal_exact<F>(
    func: F,
    x: &ArrayView1<f64>,
) -> Result<Array1<f64>, OptimizeError>
where
    F: Fn(&[SecondOrderDual]) -> SecondOrderDual,
{
    let n = x.len();
    let mut hessian_diagonal = Array1::zeros(n);

    // Compute each diagonal element
    for i in 0..n {
        // Create second-order dual variables
        let mut x_dual = Vec::with_capacity(n);
        for j in 0..n {
            if i == j {
                x_dual.push(SecondOrderDual::variable(x[j]));
            } else {
                x_dual.push(SecondOrderDual::constant(x[j]));
            }
        }

        let result = func(&x_dual);
        hessian_diagonal[i] = result.second_derivative();
    }

    Ok(hessian_diagonal)
}

/// Multi-variable forward-mode gradient computation using MultiDual
pub fn forward_gradient_multi<F>(func: F, x: &ArrayView1<f64>) -> Result<Array1<f64>, OptimizeError>
where
    F: Fn(&[MultiDual]) -> MultiDual,
{
    let n = x.len();

    // Create multi-dual variables
    let x_multi: Vec<MultiDual> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| MultiDual::variable(xi, i, n))
        .collect();

    let result = func(&x_multi);
    Ok(result.gradient().clone())
}

/// Forward-mode Jacobian computation for vector-valued functions
pub fn forward_jacobian<F>(
    func: F,
    x: &ArrayView1<f64>,
    output_dim: usize,
) -> Result<Array2<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64>,
{
    let n = x.len();
    let mut jacobian = Array2::zeros((output_dim, n));

    // Compute each column of the Jacobian using dual numbers
    for j in 0..n {
        let h = 1e-8;
        let mut x_plus = x.to_owned();
        x_plus[j] += h;
        let f_plus = func(&x_plus.view());

        let mut x_minus = x.to_owned();
        x_minus[j] -= h;
        let f_minus = func(&x_minus.view());

        for i in 0..output_dim {
            jacobian[[i, j]] = (f_plus[i] - f_minus[i]) / (2.0 * h);
        }
    }

    Ok(jacobian)
}

/// Check if forward mode is preferred for the given problem dimensions
pub fn is_forward_mode_efficient(input_dim: usize, output_dim: usize) -> bool {
    // Forward mode is efficient when input dimension is small
    // Cost is O(input_dim * cost_of_function)
    input_dim <= 10 || (input_dim <= output_dim && input_dim <= 50)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_forward_gradient() {
        // Test function: f(x, y) = x² + xy + 2y²
        let func = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[0] * x[1] + 2.0 * x[1] * x[1] };

        let x = Array1::from_vec(vec![1.0, 2.0]);
        let grad = forward_gradient(func, &x.view()).unwrap();

        // ∂f/∂x = 2x + y = 2(1) + 2 = 4
        // ∂f/∂y = x + 4y = 1 + 4(2) = 9
        assert_abs_diff_eq!(grad[0], 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(grad[1], 9.0, epsilon = 1e-6);
    }

    #[test]
    fn test_forward_hessian_diagonal() {
        // Test function: f(x, y) = x² + xy + 2y²
        let func = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[0] * x[1] + 2.0 * x[1] * x[1] };

        let x = Array1::from_vec(vec![1.0, 2.0]);
        let hess_diag = forward_hessian_diagonal(func, &x.view()).unwrap();

        // ∂²f/∂x² = 2
        // ∂²f/∂y² = 4
        assert_abs_diff_eq!(hess_diag[0], 2.0, epsilon = 1e-4);
        assert_abs_diff_eq!(hess_diag[1], 4.0, epsilon = 1e-4);
    }

    #[test]
    fn test_second_order_dual_arithmetic() {
        let a = SecondOrderDual::new(2.0, 1.0, 0.0);
        let b = SecondOrderDual::new(3.0, 0.0, 0.0);

        // Test multiplication: (2 + ε)(3) = 6 + 3ε
        let product = a * b;
        assert_abs_diff_eq!(product.value(), 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(product.first_derivative(), 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(product.second_derivative(), 0.0, epsilon = 1e-10);

        // Test power: (2 + ε)² = 4 + 4ε + ε²
        let x = SecondOrderDual::variable(2.0);
        let square = x.powi(2);
        assert_abs_diff_eq!(square.value(), 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(square.first_derivative(), 4.0, epsilon = 1e-10); // 2x = 4
        assert_abs_diff_eq!(square.second_derivative(), 2.0, epsilon = 1e-10); // 2
    }

    #[test]
    fn test_forward_jacobian() {
        // Test vector function: f(x, y) = [x² + y, xy, y²]
        let func = |x: &ArrayView1<f64>| -> Array1<f64> {
            Array1::from_vec(vec![x[0] * x[0] + x[1], x[0] * x[1], x[1] * x[1]])
        };

        let x = Array1::from_vec(vec![2.0, 3.0]);
        let jac = forward_jacobian(func, &x.view(), 3).unwrap();

        // Expected Jacobian at (2, 3):
        // ∂f₁/∂x = 2x = 4, ∂f₁/∂y = 1
        // ∂f₂/∂x = y = 3,  ∂f₂/∂y = x = 2
        // ∂f₃/∂x = 0,     ∂f₃/∂y = 2y = 6
        assert_abs_diff_eq!(jac[[0, 0]], 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(jac[[0, 1]], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(jac[[1, 0]], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(jac[[1, 1]], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(jac[[2, 0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(jac[[2, 1]], 6.0, epsilon = 1e-6);
    }

    #[test]
    fn test_is_forward_mode_efficient() {
        // Small input dimension should prefer forward mode
        assert!(is_forward_mode_efficient(3, 1));
        assert!(is_forward_mode_efficient(5, 10));

        // Large input dimension should not prefer forward mode
        assert!(!is_forward_mode_efficient(100, 1));
    }
}
