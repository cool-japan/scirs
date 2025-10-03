//! Dual numbers for forward-mode automatic differentiation
//!
//! This module implements dual numbers, which are used for forward-mode automatic
//! differentiation. Dual numbers extend real numbers with an infinitesimal part
//! that tracks derivatives.

use scirs2_core::ndarray::{Array1, ArrayView1};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Dual number for forward-mode automatic differentiation
///
/// A dual number is of the form a + b*ε where ε² = 0.
/// This allows us to compute both function values and derivatives simultaneously.
#[derive(Debug, Clone, Copy)]
pub struct Dual {
    /// Real part (function value)
    value: f64,
    /// Dual part (derivative)
    derivative: f64,
}

impl Dual {
    /// Create a new dual number
    pub fn new(value: f64, derivative: f64) -> Self {
        Self { value, derivative }
    }

    /// Create a dual number representing a constant (derivative = 0)
    pub fn constant(value: f64) -> Self {
        Self {
            value,
            derivative: 0.0,
        }
    }

    /// Create a dual number representing a variable (derivative = 1)
    pub fn variable(value: f64) -> Self {
        Self {
            value,
            derivative: 1.0,
        }
    }

    /// Get the real part (function value)
    pub fn value(self) -> f64 {
        self.value
    }

    /// Get the dual part (derivative)
    pub fn derivative(self) -> f64 {
        self.derivative
    }

    /// Compute sine of dual number
    pub fn sin(self) -> Self {
        Self {
            value: self.value.sin(),
            derivative: self.derivative * self.value.cos(),
        }
    }

    /// Compute cosine of dual number
    pub fn cos(self) -> Self {
        Self {
            value: self.value.cos(),
            derivative: -self.derivative * self.value.sin(),
        }
    }

    /// Compute tangent of dual number
    pub fn tan(self) -> Self {
        let cos_val = self.value.cos();
        Self {
            value: self.value.tan(),
            derivative: self.derivative / (cos_val * cos_val),
        }
    }

    /// Compute exponential of dual number
    pub fn exp(self) -> Self {
        let exp_val = self.value.exp();
        Self {
            value: exp_val,
            derivative: self.derivative * exp_val,
        }
    }

    /// Compute natural logarithm of dual number
    pub fn ln(self) -> Self {
        Self {
            value: self.value.ln(),
            derivative: self.derivative / self.value,
        }
    }

    /// Compute power of dual number (self^n)
    pub fn powi(self, n: i32) -> Self {
        let n_f64 = n as f64;
        Self {
            value: self.value.powi(n),
            derivative: self.derivative * n_f64 * self.value.powi(n - 1),
        }
    }

    /// Compute power of dual number (self^p) where p is real
    pub fn powf(self, p: f64) -> Self {
        Self {
            value: self.value.powf(p),
            derivative: self.derivative * p * self.value.powf(p - 1.0),
        }
    }

    /// Compute square root of dual number
    pub fn sqrt(self) -> Self {
        let sqrt_val = self.value.sqrt();
        Self {
            value: sqrt_val,
            derivative: self.derivative / (2.0 * sqrt_val),
        }
    }

    /// Compute absolute value of dual number
    pub fn abs(self) -> Self {
        Self {
            value: self.value.abs(),
            derivative: if self.value >= 0.0 {
                self.derivative
            } else {
                -self.derivative
            },
        }
    }

    /// Compute maximum of two dual numbers
    pub fn max(self, other: Self) -> Self {
        if self.value >= other.value {
            self
        } else {
            other
        }
    }

    /// Compute minimum of two dual numbers
    pub fn min(self, other: Self) -> Self {
        if self.value <= other.value {
            self
        } else {
            other
        }
    }
}

// Arithmetic operations for dual numbers

impl Add for Dual {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            derivative: self.derivative + other.derivative,
        }
    }
}

impl Add<f64> for Dual {
    type Output = Self;

    fn add(self, scalar: f64) -> Self {
        Self {
            value: self.value + scalar,
            derivative: self.derivative,
        }
    }
}

impl Add<Dual> for f64 {
    type Output = Dual;

    fn add(self, dual: Dual) -> Dual {
        dual + self
    }
}

impl Sub for Dual {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            value: self.value - other.value,
            derivative: self.derivative - other.derivative,
        }
    }
}

impl Sub<f64> for Dual {
    type Output = Self;

    fn sub(self, scalar: f64) -> Self {
        Self {
            value: self.value - scalar,
            derivative: self.derivative,
        }
    }
}

impl Sub<Dual> for f64 {
    type Output = Dual;

    fn sub(self, dual: Dual) -> Dual {
        Dual {
            value: self - dual.value,
            derivative: -dual.derivative,
        }
    }
}

impl Mul for Dual {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
            derivative: self.derivative * other.value + self.value * other.derivative,
        }
    }
}

impl Mul<f64> for Dual {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            value: self.value * scalar,
            derivative: self.derivative * scalar,
        }
    }
}

impl Mul<Dual> for f64 {
    type Output = Dual;

    fn mul(self, dual: Dual) -> Dual {
        dual * self
    }
}

impl Div for Dual {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let denom = other.value * other.value;

        // Protect against division by zero
        let value = if other.value == 0.0 {
            if self.value == 0.0 {
                f64::NAN
            }
            // 0/0 is undefined
            else if self.value > 0.0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            }
        } else {
            self.value / other.value
        };

        let derivative = if denom == 0.0 {
            // Handle derivative at division by zero
            if other.value == 0.0 && self.derivative == 0.0 && other.derivative == 0.0 {
                f64::NAN
            } else {
                f64::INFINITY
            }
        } else {
            (self.derivative * other.value - self.value * other.derivative) / denom
        };

        Self { value, derivative }
    }
}

impl Div<f64> for Dual {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        if scalar == 0.0 {
            // Division by zero
            Self {
                value: if self.value == 0.0 {
                    f64::NAN
                } else if self.value > 0.0 {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                },
                derivative: if self.derivative == 0.0 {
                    f64::NAN
                } else {
                    f64::INFINITY
                },
            }
        } else {
            Self {
                value: self.value / scalar,
                derivative: self.derivative / scalar,
            }
        }
    }
}

impl Div<Dual> for f64 {
    type Output = Dual;

    fn div(self, dual: Dual) -> Dual {
        Dual::constant(self) / dual
    }
}

impl Neg for Dual {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            value: -self.value,
            derivative: -self.derivative,
        }
    }
}

// Conversion traits
impl From<f64> for Dual {
    fn from(value: f64) -> Self {
        Self::constant(value)
    }
}

impl From<Dual> for f64 {
    fn from(dual: Dual) -> Self {
        dual.value
    }
}

// Partial ordering for optimization algorithms
impl PartialEq for Dual {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl PartialOrd for Dual {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

/// Trait for dual number operations
pub trait DualNumber: Clone + Copy {
    /// Get the value part
    fn value(self) -> f64;

    /// Get the derivative part
    fn derivative(self) -> f64;

    /// Create from value and derivative
    fn new(value: f64, derivative: f64) -> Self;

    /// Create constant (derivative = 0)
    fn constant(value: f64) -> Self;

    /// Create variable (derivative = 1)
    fn variable(value: f64) -> Self;
}

impl DualNumber for Dual {
    fn value(self) -> f64 {
        self.value
    }

    fn derivative(self) -> f64 {
        self.derivative
    }

    fn new(value: f64, derivative: f64) -> Self {
        Self::new(value, derivative)
    }

    fn constant(value: f64) -> Self {
        Self::constant(value)
    }

    fn variable(value: f64) -> Self {
        Self::variable(value)
    }
}

/// Multi-dimensional dual number for computing gradients
#[derive(Debug, Clone)]
pub struct MultiDual {
    /// Function value
    value: f64,
    /// Partial derivatives (gradient components)
    derivatives: Array1<f64>,
}

impl MultiDual {
    /// Create a new multi-dimensional dual number
    pub fn new(value: f64, derivatives: Array1<f64>) -> Self {
        Self { value, derivatives }
    }

    /// Create a constant multi-dual (all derivatives = 0)
    pub fn constant(value: f64, nvars: usize) -> Self {
        Self {
            value,
            derivatives: Array1::zeros(nvars),
        }
    }

    /// Create a variable multi-dual (one derivative = 1, others = 0)
    pub fn variable(value: f64, var_index: usize, nvars: usize) -> Self {
        let mut derivatives = Array1::zeros(nvars);
        derivatives[var_index] = 1.0;
        Self { value, derivatives }
    }

    /// Get the function value
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Get the gradient
    pub fn gradient(&self) -> &Array1<f64> {
        &self.derivatives
    }

    /// Get a specific partial derivative
    pub fn partial(&self, index: usize) -> f64 {
        self.derivatives[index]
    }
}

// Arithmetic operations for MultiDual
impl Add for MultiDual {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            derivatives: &self.derivatives + &other.derivatives,
        }
    }
}

impl Mul for MultiDual {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
            derivatives: &self.derivatives * other.value + &other.derivatives * self.value,
        }
    }
}

impl Mul<f64> for MultiDual {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            value: self.value * scalar,
            derivatives: &self.derivatives * scalar,
        }
    }
}

/// Create an array of dual numbers for gradient computation
#[allow(dead_code)]
pub fn create_dual_variables(x: &ArrayView1<f64>) -> Vec<Dual> {
    x.iter().map(|&xi| Dual::variable(xi)).collect()
}

/// Create multi-dual variables for a given point
#[allow(dead_code)]
pub fn create_multi_dual_variables(x: &ArrayView1<f64>) -> Vec<MultiDual> {
    let n = x.len();
    x.iter()
        .enumerate()
        .map(|(i, &xi)| MultiDual::variable(xi, i, n))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dual_arithmetic() {
        let a = Dual::new(2.0, 1.0);
        let b = Dual::new(3.0, 0.5);

        // Test addition
        let sum = a + b;
        assert_abs_diff_eq!(sum.value(), 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sum.derivative(), 1.5, epsilon = 1e-10);

        // Test multiplication
        let product = a * b;
        assert_abs_diff_eq!(product.value(), 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(product.derivative(), 4.0, epsilon = 1e-10); // 1*3 + 2*0.5

        // Test division
        let quotient = a / b;
        assert_abs_diff_eq!(quotient.value(), 2.0 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(
            quotient.derivative(),
            (1.0 * 3.0 - 2.0 * 0.5) / (3.0 * 3.0),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_dual_functions() {
        let x = Dual::variable(1.0);

        // Test exp(x) at x=1
        let exp_x = x.exp();
        assert_abs_diff_eq!(exp_x.value(), std::f64::consts::E, epsilon = 1e-10);
        assert_abs_diff_eq!(exp_x.derivative(), std::f64::consts::E, epsilon = 1e-10);

        // Test sin(x) at x=0
        let x0 = Dual::variable(0.0);
        let sin_x = x0.sin();
        assert_abs_diff_eq!(sin_x.value(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sin_x.derivative(), 1.0, epsilon = 1e-10); // cos(0) = 1

        // Test x² at x=3
        let x3 = Dual::variable(3.0);
        let x_squared = x3.powi(2);
        assert_abs_diff_eq!(x_squared.value(), 9.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x_squared.derivative(), 6.0, epsilon = 1e-10); // 2*3
    }

    #[test]
    fn test_multi_dual() {
        let x = MultiDual::variable(2.0, 0, 2);
        let y = MultiDual::variable(3.0, 1, 2);

        // Test x * y
        let product = x * y;
        assert_abs_diff_eq!(product.value(), 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(product.partial(0), 3.0, epsilon = 1e-10); // ∂(xy)/∂x = y
        assert_abs_diff_eq!(product.partial(1), 2.0, epsilon = 1e-10); // ∂(xy)/∂y = x
    }

    #[test]
    fn test_create_dual_variables() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let duals = create_dual_variables(&x.view());

        assert_eq!(duals.len(), 3);
        assert_abs_diff_eq!(duals[0].value(), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(duals[1].value(), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(duals[2].value(), 3.0, epsilon = 1e-10);

        // All should have derivative = 1 (variables)
        for dual in &duals {
            assert_abs_diff_eq!(dual.derivative(), 1.0, epsilon = 1e-10);
        }
    }
}
