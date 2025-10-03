//! Dual number implementation for automatic differentiation
//!
//! Dual numbers are the foundation of forward-mode automatic differentiation.
//! A dual number has the form a + b*ε where ε² = 0.

use crate::common::IntegrateFloat;
use scirs2_core::ndarray::{Array1, ArrayView1};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Dual number for forward-mode automatic differentiation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual<F: IntegrateFloat> {
    /// The value part
    pub val: F,
    /// The derivative part
    pub der: F,
}

impl<F: IntegrateFloat> Dual<F> {
    /// Create a new dual number
    pub fn new(val: F, der: F) -> Self {
        Dual { val, der }
    }

    /// Create a constant dual number (zero derivative)
    pub fn constant(val: F) -> Self {
        Dual {
            val,
            der: F::zero(),
        }
    }

    /// Create a variable dual number (unit derivative)
    pub fn variable(val: F) -> Self {
        Dual { val, der: F::one() }
    }

    /// Extract the value
    pub fn value(&self) -> F {
        self.val
    }

    /// Extract the derivative
    pub fn derivative(&self) -> F {
        self.der
    }

    /// Compute sin(x)
    pub fn sin(&self) -> Self {
        Dual {
            val: self.val.sin(),
            der: self.der * self.val.cos(),
        }
    }

    /// Compute cos(x)
    pub fn cos(&self) -> Self {
        Dual {
            val: self.val.cos(),
            der: -self.der * self.val.sin(),
        }
    }

    /// Compute exp(x)
    pub fn exp(&self) -> Self {
        let exp_val = self.val.exp();
        Dual {
            val: exp_val,
            der: self.der * exp_val,
        }
    }

    /// Compute ln(x)
    pub fn ln(&self) -> Self {
        Dual {
            val: self.val.ln(),
            der: self.der / self.val,
        }
    }

    /// Compute sqrt(x)
    pub fn sqrt(&self) -> Self {
        let sqrt_val = self.val.sqrt();
        Dual {
            val: sqrt_val,
            der: self.der / (F::from(2.0).unwrap() * sqrt_val),
        }
    }

    /// Compute x^y for constant y
    pub fn powf(&self, n: F) -> Self {
        Dual {
            val: self.val.powf(n),
            der: self.der * n * self.val.powf(n - F::one()),
        }
    }

    /// Compute abs(x)
    pub fn abs(&self) -> Self {
        if self.val >= F::zero() {
            *self
        } else {
            -*self
        }
    }

    /// Compute tan(x)
    pub fn tan(&self) -> Self {
        let cos_val = self.val.cos();
        Dual {
            val: self.val.tan(),
            der: self.der / (cos_val * cos_val),
        }
    }

    /// Compute tanh(x)
    pub fn tanh(&self) -> Self {
        let tanh_val = self.val.tanh();
        Dual {
            val: tanh_val,
            der: self.der * (F::one() - tanh_val * tanh_val),
        }
    }

    /// Compute sinh(x)
    pub fn sinh(&self) -> Self {
        Dual {
            val: self.val.sinh(),
            der: self.der * self.val.cosh(),
        }
    }

    /// Compute cosh(x)
    pub fn cosh(&self) -> Self {
        Dual {
            val: self.val.cosh(),
            der: self.der * self.val.sinh(),
        }
    }

    /// Compute atan(x)
    pub fn atan(&self) -> Self {
        Dual {
            val: self.val.atan(),
            der: self.der / (F::one() + self.val * self.val),
        }
    }

    /// Compute asin(x)
    pub fn asin(&self) -> Self {
        Dual {
            val: self.val.asin(),
            der: self.der / (F::one() - self.val * self.val).sqrt(),
        }
    }

    /// Compute acos(x)
    pub fn acos(&self) -> Self {
        Dual {
            val: self.val.acos(),
            der: -self.der / (F::one() - self.val * self.val).sqrt(),
        }
    }

    /// Compute atan2(y, x)
    pub fn atan2(&self, x: Self) -> Self {
        let r2 = self.val * self.val + x.val * x.val;
        Dual {
            val: self.val.atan2(x.val),
            der: (self.der * x.val - self.val * x.der) / r2,
        }
    }

    /// Compute max(self, other)
    pub fn max(&self, other: Self) -> Self {
        if self.val > other.val {
            *self
        } else if self.val < other.val {
            other
        } else {
            // When values are equal, average the derivatives
            Dual {
                val: self.val,
                der: (self.der + other.der) / F::from(2.0).unwrap(),
            }
        }
    }

    /// Compute min(self, other)
    pub fn min(&self, other: Self) -> Self {
        if self.val < other.val {
            *self
        } else if self.val > other.val {
            other
        } else {
            // When values are equal, average the derivatives
            Dual {
                val: self.val,
                der: (self.der + other.der) / F::from(2.0).unwrap(),
            }
        }
    }

    /// Compute x^y where both x and y are dual numbers
    pub fn pow(&self, other: Self) -> Self {
        let val = self.val.powf(other.val);
        let der = if self.val > F::zero() {
            val * (other.der * self.val.ln() + other.val * self.der / self.val)
        } else {
            F::zero() // Handle edge case
        };
        Dual { val, der }
    }
}

// Arithmetic operations for dual numbers
impl<F: IntegrateFloat> Add for Dual<F> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Dual {
            val: self.val + other.val,
            der: self.der + other.der,
        }
    }
}

impl<F: IntegrateFloat> Sub for Dual<F> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Dual {
            val: self.val - other.val,
            der: self.der - other.der,
        }
    }
}

impl<F: IntegrateFloat> Mul for Dual<F> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Dual {
            val: self.val * other.val,
            der: self.der * other.val + self.val * other.der,
        }
    }
}

impl<F: IntegrateFloat> Div for Dual<F> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let inv_val = F::one() / other.val;
        Dual {
            val: self.val * inv_val,
            der: (self.der * other.val - self.val * other.der) * inv_val * inv_val,
        }
    }
}

impl<F: IntegrateFloat> Neg for Dual<F> {
    type Output = Self;

    fn neg(self) -> Self {
        Dual {
            val: -self.val,
            der: -self.der,
        }
    }
}

// Operations with scalars
impl<F: IntegrateFloat> Add<F> for Dual<F> {
    type Output = Self;

    fn add(self, scalar: F) -> Self {
        Dual {
            val: self.val + scalar,
            der: self.der,
        }
    }
}

impl<F: IntegrateFloat> Sub<F> for Dual<F> {
    type Output = Self;

    fn sub(self, scalar: F) -> Self {
        Dual {
            val: self.val - scalar,
            der: self.der,
        }
    }
}

impl<F: IntegrateFloat> Mul<F> for Dual<F> {
    type Output = Self;

    fn mul(self, scalar: F) -> Self {
        Dual {
            val: self.val * scalar,
            der: self.der * scalar,
        }
    }
}

impl<F: IntegrateFloat> Div<F> for Dual<F> {
    type Output = Self;

    fn div(self, scalar: F) -> Self {
        Dual {
            val: self.val / scalar,
            der: self.der / scalar,
        }
    }
}

impl<F: IntegrateFloat> fmt::Display for Dual<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}ε", self.val, self.der)
    }
}

/// Vector of dual numbers for multi-dimensional forward AD
pub struct DualVector<F: IntegrateFloat> {
    /// Values
    pub values: Array1<F>,
    /// Jacobian matrix (derivatives)
    pub jacobian: Array1<Array1<F>>,
}

impl<F: IntegrateFloat> DualVector<F> {
    /// Create a new dual vector
    pub fn new(values: Array1<F>, jacobian: Array1<Array1<F>>) -> Self {
        DualVector { values, jacobian }
    }

    /// Create from a regular vector with specified active variable
    pub fn from_vector(_values: ArrayView1<F>, activevar: usize) -> Self {
        let n = _values.len();
        let mut jacobian = Array1::from_elem(n, Array1::zeros(n));

        // Set the derivative of the active variable to 1
        jacobian[activevar][activevar] = F::one();

        DualVector {
            values: _values.to_owned(),
            jacobian,
        }
    }

    /// Create a constant dual vector (zero derivatives)
    pub fn constant(values: Array1<F>) -> Self {
        let n = values.len();
        let jacobian = Array1::from_elem(n, Array1::zeros(n));
        DualVector { values, jacobian }
    }

    /// Get the dimension
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Extract values
    pub fn values(&self) -> &Array1<F> {
        &self.values
    }

    /// Extract Jacobian
    pub fn jacobian(&self) -> &Array1<Array1<F>> {
        &self.jacobian
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_arithmetic() {
        let x = Dual::new(2.0, 1.0);
        let y = Dual::new(3.0, 0.0);

        // Test addition
        let sum = x + y;
        assert_eq!(sum.val, 5.0);
        assert_eq!(sum.der, 1.0);

        // Test multiplication
        let prod = x * y;
        assert_eq!(prod.val, 6.0);
        assert_eq!(prod.der, 3.0); // d/dx(x*3) = 3

        // Test chain rule: d/dx(x^2) = 2x
        let square = x * x;
        assert_eq!(square.val, 4.0);
        assert_eq!(square.der, 4.0);
    }

    #[test]
    fn test_dual_functions() {
        let x = Dual::variable(0.0);

        // Test sin/cos derivatives
        let sin_x = x.sin();
        assert!((sin_x.val - 0.0_f64).abs() < 1e-10_f64);
        assert!((sin_x.der - 1.0_f64).abs() < 1e-10_f64); // cos(0) = 1

        let cos_x = x.cos();
        assert!((cos_x.val - 1.0).abs() < 1e-10);
        assert!((cos_x.der - 0.0).abs() < 1e-10); // -sin(0) = 0

        // Test exp derivative
        let exp_x = x.exp();
        assert!((exp_x.val - 1.0).abs() < 1e-10);
        assert!((exp_x.der - 1.0).abs() < 1e-10); // exp(0) = 1
    }
}
