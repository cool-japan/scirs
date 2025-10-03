//! Error handling wrappers for special functions
//!
//! This module provides consistent error-handling wrappers for all special functions,
//! ensuring proper validation, error context, and recovery strategies.

use crate::error::SpecialResult;
use crate::error_context::{ErrorContext, ErrorContextExt, RecoveryStrategy};
use crate::special_error;
use crate::validation;
use scirs2_core::ndarray::{Array1, ArrayBase, ArrayView1};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

/// Configuration for error handling behavior
#[derive(Debug, Clone)]
pub struct ErrorConfig {
    /// Whether to use recovery strategies
    pub enable_recovery: bool,
    /// Default recovery strategy
    pub default_recovery: RecoveryStrategy,
    /// Whether to log errors
    pub log_errors: bool,
    /// Maximum iterations before convergence error
    pub max_iterations: usize,
    /// Tolerance for convergence
    pub tolerance: f64,
}

impl Default for ErrorConfig {
    fn default() -> Self {
        Self {
            enable_recovery: false,
            default_recovery: RecoveryStrategy::PropagateError,
            log_errors: false,
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

/// Wrapper for single-argument special functions
pub struct SingleArgWrapper<F, T> {
    pub name: &'static str,
    pub func: F,
    pub config: ErrorConfig,
    _phantom: std::marker::PhantomData<T>,
}

impl<F, T> SingleArgWrapper<F, T>
where
    F: Fn(T) -> T,
    T: Float + Display + Debug + FromPrimitive,
{
    pub fn new(name: &'static str, func: F) -> Self {
        Self {
            name,
            func,
            config: ErrorConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn with_config(mut self, config: ErrorConfig) -> Self {
        self.config = config;
        self
    }

    /// Evaluate the function with full error handling
    pub fn evaluate(&self, x: T) -> SpecialResult<T> {
        // Check for special cases that might cause issues
        if x.is_nan() {
            return Ok(T::nan());
        }
        if x.is_infinite() {
            return Ok(T::infinity()); // Return positive infinity for gamma(∞)
        }

        // Validate input (after handling NaN and infinity)
        validation::check_finite(x, "x")
            .with_context(|| ErrorContext::new(self.name, "input validation").with_param("x", x))?;

        // Compute the result
        let result = (self.func)(x);

        // Validate output
        if result.is_nan() && !x.is_nan() {
            if self.config.enable_recovery {
                // Try recovery strategies
                if let Some(recovered) = self.try_recover(x) {
                    return Ok(recovered);
                }
            }

            return Err(special_error!(
                computation: self.name, "evaluation",
                "x" => x
            ));
        }

        if result.is_infinite() && !x.is_infinite() {
            // Check if this is expected (e.g., gamma(0) = inf)
            if !self.is_expected_infinity(x) {
                return Err(special_error!(
                    computation: self.name, "overflow",
                    "x" => x
                ));
            }
        }

        Ok(result)
    }

    /// Check if infinity is expected for this input
    fn is_expected_infinity(&self, x: T) -> bool {
        // This would be customized per function
        match self.name {
            "gamma" => x == T::zero(),
            "digamma" => x == T::zero() || (x < T::zero() && x.fract() == T::zero()),
            _ => false,
        }
    }

    /// Try to recover from an error
    fn try_recover(&self, _x: T) -> Option<T> {
        match self.config.default_recovery {
            RecoveryStrategy::ReturnDefault => Some(T::zero()),
            RecoveryStrategy::ClampToRange => {
                // Function-specific clamping logic
                None
            }
            RecoveryStrategy::UseApproximation => {
                // Function-specific approximation
                None
            }
            RecoveryStrategy::PropagateError => None,
        }
    }
}

/// Wrapper for two-argument special functions
pub struct TwoArgWrapper<F, T> {
    pub name: &'static str,
    pub func: F,
    pub config: ErrorConfig,
    _phantom: std::marker::PhantomData<T>,
}

impl<F, T> TwoArgWrapper<F, T>
where
    F: Fn(T, T) -> T,
    T: Float + Display + Debug + FromPrimitive,
{
    pub fn new(name: &'static str, func: F) -> Self {
        Self {
            name,
            func,
            config: ErrorConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn with_config(mut self, config: ErrorConfig) -> Self {
        self.config = config;
        self
    }

    /// Evaluate the function with full error handling
    pub fn evaluate(&self, a: T, b: T) -> SpecialResult<T> {
        // Validate inputs
        validation::check_finite(a, "a").with_context(|| {
            ErrorContext::new(self.name, "input validation")
                .with_param("a", a)
                .with_param("b", b)
        })?;

        validation::check_finite(b, "b").with_context(|| {
            ErrorContext::new(self.name, "input validation")
                .with_param("a", a)
                .with_param("b", b)
        })?;

        // Additional function-specific validation
        self.validate_specific(a, b)?;

        // Compute the result
        let result = (self.func)(a, b);

        // Validate output
        if result.is_nan() && !a.is_nan() && !b.is_nan() {
            return Err(special_error!(
                computation: self.name, "evaluation",
                "a" => a,
                "b" => b
            ));
        }

        Ok(result)
    }

    /// Function-specific validation
    fn validate_specific(&self, a: T, b: T) -> SpecialResult<()> {
        match self.name {
            "beta" => {
                // Beta function requires positive arguments
                validation::check_positive(a, "a")?;
                validation::check_positive(b, "b")?;
            }
            "bessel_jn" => {
                // Bessel functions might have order restrictions
                // This would be more specific based on the actual function
            }
            _ => {}
        }
        Ok(())
    }
}

/// Wrapper for array operations with error handling
pub struct ArrayWrapper<F, T> {
    pub name: &'static str,
    pub func: F,
    pub config: ErrorConfig,
    _phantom: std::marker::PhantomData<T>,
}

impl<F, T> ArrayWrapper<F, T>
where
    F: Fn(&ArrayView1<T>) -> Array1<T>,
    T: Float + Display + Debug + FromPrimitive,
{
    pub fn new(name: &'static str, func: F) -> Self {
        Self {
            name,
            func,
            config: ErrorConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Evaluate the function on an array with full error handling
    pub fn evaluate<S>(
        &self,
        input: &ArrayBase<S, scirs2_core::ndarray::Ix1>,
    ) -> SpecialResult<Array1<T>>
    where
        S: scirs2_core::ndarray::Data<Elem = T>,
    {
        // Validate array
        validation::check_array_finite(input, "input").with_context(|| {
            ErrorContext::new(self.name, "array validation")
                .with_param("shape", format!("{:?}", input.shape()))
        })?;

        validation::check_not_empty(input, "input")?;

        // Compute the result
        let result = (self.func)(&input.view());

        // Validate output
        let nan_count = result.iter().filter(|&&x| x.is_nan()).count();
        if nan_count > 0 {
            let total = result.len();
            return Err(special_error!(
                computation: self.name, "array evaluation",
                "nan_count" => nan_count,
                "total_elements" => total
            ));
        }

        Ok(result)
    }
}

/// Create error-wrapped versions of functions
pub mod wrapped {
    use super::*;
    use crate::{beta, digamma, erf, erfc, gamma};

    /// Create a wrapped gamma function with error handling
    pub fn gamma_wrapped() -> SingleArgWrapper<fn(f64) -> f64, f64> {
        SingleArgWrapper::new("gamma", gamma::<f64>)
    }

    /// Create a wrapped digamma function with error handling
    pub fn digamma_wrapped() -> SingleArgWrapper<fn(f64) -> f64, f64> {
        SingleArgWrapper::new("digamma", digamma::<f64>)
    }

    /// Create a wrapped beta function with error handling
    pub fn beta_wrapped() -> TwoArgWrapper<fn(f64, f64) -> f64, f64> {
        TwoArgWrapper::new("beta", beta::<f64>)
    }

    /// Create a wrapped erf function with error handling
    pub fn erf_wrapped() -> SingleArgWrapper<fn(f64) -> f64, f64> {
        SingleArgWrapper::new("erf", erf)
    }

    /// Create a wrapped erfc function with error handling
    pub fn erfc_wrapped() -> SingleArgWrapper<fn(f64) -> f64, f64> {
        SingleArgWrapper::new("erfc", erfc)
    }
}

#[cfg(test)]
mod tests {
    use super::wrapped::*;
    use super::*;

    #[test]
    fn test_gamma_wrapped() {
        let gamma = gamma_wrapped();

        // Valid input
        let result = gamma.evaluate(5.0);
        assert!(result.is_ok());
        assert!((result.unwrap() - 24.0).abs() < 1e-10);

        // Invalid input (NaN)
        let result = gamma.evaluate(f64::NAN);
        assert!(result.is_ok()); // NaN input returns NaN output
        assert!(result.unwrap().is_nan());

        // Invalid input (infinity)
        let result = gamma.evaluate(f64::INFINITY);
        assert!(result.is_ok());
        assert!(result.unwrap().is_infinite());
    }

    #[test]
    fn test_beta_wrapped() {
        let beta = beta_wrapped();

        // Valid inputs
        let result = beta.evaluate(2.0, 3.0);
        assert!(result.is_ok());

        // Invalid inputs (negative)
        let result = beta.evaluate(-1.0, 2.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_array_wrapper() {
        use scirs2_core::ndarray::arr1;

        let arr_gamma = ArrayWrapper::new("gamma_array", |x: &ArrayView1<f64>| {
            x.mapv(crate::gamma::gamma::<f64>)
        });

        // Valid array
        let input = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let result = arr_gamma.evaluate(&input);
        assert!(result.is_ok());

        // Array with NaN
        let input = arr1(&[1.0, f64::NAN, 3.0]);
        let result = arr_gamma.evaluate(&input);
        assert!(result.is_err());
    }
}
