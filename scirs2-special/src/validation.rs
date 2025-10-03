//! Validation utilities for special functions module
//!
//! This module provides consistent validation helpers that leverage scirs2-core::validation
//! utilities and provide domain-specific error messages for special functions.

use crate::error::{SpecialError, SpecialResult};
use scirs2_core::ndarray::{ArrayBase, Dimension};
use scirs2_core::numeric::{Float, Zero};
use scirs2_core::validation;

/// Check if a value is positive (> 0)
#[allow(dead_code)]
pub fn check_positive<T>(value: T, name: &str) -> SpecialResult<T>
where
    T: Float + std::fmt::Display + Copy + Zero,
{
    validation::check_positive(value, name)
        .map_err(|_| SpecialError::DomainError(format!("{name} must be positive, got {value}")))
}

/// Check if a value is non-negative (>= 0)
#[allow(dead_code)]
pub fn check_non_negative<T>(value: T, name: &str) -> SpecialResult<T>
where
    T: Float + std::fmt::Display + Copy + Zero,
{
    validation::check_non_negative(value, name)
        .map_err(|_| SpecialError::DomainError(format!("{name} must be non-negative, got {value}")))
}

/// Check if a value is finite
#[allow(dead_code)]
pub fn check_finite<T>(value: T, name: &str) -> SpecialResult<T>
where
    T: Float + std::fmt::Display + Copy,
{
    validation::check_finite(value, name)
        .map_err(|_| SpecialError::DomainError(format!("{name} must be finite, got {value}")))
}

/// Check if a value is within bounds (inclusive)
#[allow(dead_code)]
pub fn check_in_bounds<T>(value: T, min: T, max: T, name: &str) -> SpecialResult<T>
where
    T: PartialOrd + std::fmt::Display + Copy,
{
    validation::check_in_bounds(value, min, max, name).map_err(|_| {
        SpecialError::DomainError(format!("{name} must be in [{min}, {max}], got {value}"))
    })
}

/// Check if a probability value is valid (0 <= p <= 1)
#[allow(dead_code)]
pub fn check_probability<T>(value: T, name: &str) -> SpecialResult<T>
where
    T: Float + std::fmt::Display + Copy,
{
    validation::check_probability(value, name)
        .map_err(|_| SpecialError::DomainError(format!("{name} must be in [0, 1], got {value}")))
}

/// Check if all values in an array are finite
#[allow(dead_code)]
pub fn check_array_finite<S, D>(array: &ArrayBase<S, D>, name: &str) -> SpecialResult<()>
where
    S: scirs2_core::ndarray::Data,
    D: Dimension,
    S::Elem: Float + std::fmt::Display,
{
    validation::checkarray_finite(array, name)
        .map_err(|_| SpecialError::DomainError(format!("{name} must contain only finite values")))
}

/// Check if an array is not empty
#[allow(dead_code)]
pub fn check_not_empty<S, D>(array: &ArrayBase<S, D>, name: &str) -> SpecialResult<()>
where
    S: scirs2_core::ndarray::Data,
    D: Dimension,
{
    validation::check_not_empty(array, name)
        .map_err(|_| SpecialError::ValueError(format!("{name} cannot be empty")))
}

/// Check if two arrays have the same shape
#[allow(dead_code)]
pub fn check_sameshape<S1, S2, D1, D2>(
    a: &ArrayBase<S1, D1>,
    a_name: &str,
    b: &ArrayBase<S2, D2>,
    b_name: &str,
) -> SpecialResult<()>
where
    S1: scirs2_core::ndarray::Data,
    S2: scirs2_core::ndarray::Data,
    D1: Dimension,
    D2: Dimension,
{
    validation::check_sameshape(a, a_name, b, b_name).map_err(|_| {
        SpecialError::ValueError(format!(
            "{} and {} must have the same shape, got {:?} and {:?}",
            a_name,
            b_name,
            a.shape(),
            b.shape()
        ))
    })
}

// Special function specific validations

/// Check if order n is valid for special functions (non-negative integer or real)
#[allow(dead_code)]
pub fn check_order<T>(n: T, name: &str) -> SpecialResult<T>
where
    T: Float + std::fmt::Display + Copy,
{
    check_finite(n, name)
}

/// Check if degree l is valid (non-negative integer)
#[allow(dead_code)]
pub fn check_degree(l: i32, name: &str) -> SpecialResult<i32> {
    if l < 0 {
        return Err(SpecialError::DomainError(format!(
            "{name} must be non-negative, got {l}"
        )));
    }
    Ok(l)
}

/// Check if order m is valid for associated functions (|m| <= l)
#[allow(dead_code)]
pub fn check_order_m(l: i32, m: i32) -> SpecialResult<i32> {
    if m.abs() > l {
        return Err(SpecialError::DomainError(format!(
            "|m| must be <= l, got |{m}| > {l}"
        )));
    }
    Ok(m)
}

/// Check convergence parameters
#[allow(dead_code)]
pub fn check_convergence_params(maxiter: usize, tolerance: f64) -> SpecialResult<()> {
    if maxiter == 0 {
        return Err(SpecialError::ValueError("maxiter must be > 0".to_string()));
    }
    check_positive(tolerance, "tolerance")?;
    Ok(())
}

/// Helper to convert convergence failures to ConvergenceError
#[allow(dead_code)]
pub fn convergence_error(function: &str, iterations: usize) -> SpecialError {
    SpecialError::ConvergenceError(format!(
        "{function} did not converge after {iterations} iterations"
    ))
}

/// Helper to convert not implemented features to NotImplementedError
#[allow(dead_code)]
pub fn not_implemented(feature: &str) -> SpecialError {
    SpecialError::NotImplementedError(format!("{feature} is not yet implemented"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, arr2};

    #[test]
    fn test_check_positive() {
        assert!(check_positive(1.0, "x").is_ok());
        assert!(check_positive(0.0, "x").is_err());
        assert!(check_positive(-1.0, "x").is_err());
    }

    #[test]
    fn test_check_non_negative() {
        assert!(check_non_negative(1.0, "x").is_ok());
        assert!(check_non_negative(0.0, "x").is_ok());
        assert!(check_non_negative(-1.0, "x").is_err());
    }

    #[test]
    fn test_check_finite() {
        assert!(check_finite(1.0, "x").is_ok());
        assert!(check_finite(f64::INFINITY, "x").is_err());
        assert!(check_finite(f64::NAN, "x").is_err());
    }

    #[test]
    fn test_check_in_bounds() {
        assert!(check_in_bounds(0.5, 0.0, 1.0, "x").is_ok());
        assert!(check_in_bounds(0.0, 0.0, 1.0, "x").is_ok());
        assert!(check_in_bounds(1.0, 0.0, 1.0, "x").is_ok());
        assert!(check_in_bounds(-0.1, 0.0, 1.0, "x").is_err());
        assert!(check_in_bounds(1.1, 0.0, 1.0, "x").is_err());
    }

    #[test]
    fn test_check_probability() {
        assert!(check_probability(0.5, "p").is_ok());
        assert!(check_probability(0.0, "p").is_ok());
        assert!(check_probability(1.0, "p").is_ok());
        assert!(check_probability(-0.1, "p").is_err());
        assert!(check_probability(1.1, "p").is_err());
    }

    #[test]
    fn test_check_array_finite() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        assert!(check_array_finite(&a, "array").is_ok());

        let b = arr1(&[1.0, f64::INFINITY, 3.0]);
        assert!(check_array_finite(&b, "array").is_err());
    }

    #[test]
    fn test_check_not_empty() {
        let a = arr1(&[1.0]);
        assert!(check_not_empty(&a, "array").is_ok());

        let b = arr1(&[] as &[f64]);
        assert!(check_not_empty(&b, "array").is_err());
    }

    #[test]
    fn test_check_sameshape() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        assert!(check_sameshape(&a, "a", &b, "b").is_ok());

        let c = arr2(&[[1.0], [2.0]]);
        assert!(check_sameshape(&a, "a", &c, "c").is_err());
    }

    #[test]
    fn test_check_degree() {
        assert!(check_degree(0, "l").is_ok());
        assert!(check_degree(5, "l").is_ok());
        assert!(check_degree(-1, "l").is_err());
    }

    #[test]
    fn test_check_order_m() {
        assert!(check_order_m(5, 3).is_ok());
        assert!(check_order_m(5, -3).is_ok());
        assert!(check_order_m(5, 5).is_ok());
        assert!(check_order_m(5, 6).is_err());
        assert!(check_order_m(5, -6).is_err());
    }

    #[test]
    fn test_check_convergence_params() {
        assert!(check_convergence_params(100, 1e-10).is_ok());
        assert!(check_convergence_params(0, 1e-10).is_err());
        assert!(check_convergence_params(100, 0.0).is_err());
    }

    #[test]
    fn test_convergence_error() {
        let err = convergence_error("bessel_j", 100);
        match err {
            SpecialError::ConvergenceError(msg) => {
                assert!(msg.contains("bessel_j"));
                assert!(msg.contains("100"));
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_not_implemented() {
        let err = not_implemented("complex spheroidal functions");
        match err {
            SpecialError::NotImplementedError(msg) => {
                assert!(msg.contains("complex spheroidal functions"));
            }
            _ => panic!("Wrong error type"),
        }
    }
}
