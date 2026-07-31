//! Standardized error message templates for consistent error reporting
//!
//! This module provides template functions for generating consistent error
//! messages across the entire codebase.

use std::fmt::Display;

/// Generate an error message for dimension mismatch
#[allow(dead_code)]
pub fn dimension_mismatch(operation: &str, expected: impl Display, actual: impl Display) -> String {
    format!("{operation}: dimension mismatch (expected: {expected}, actual: {actual})")
}

/// Generate an error message for shape mismatch in array operations
#[allow(dead_code)]
pub fn shape_mismatch(operation: &str, shape1: &[usize], shape2: &[usize]) -> String {
    format!("{operation}: incompatible shapes ({shape1:?} vs {shape2:?})")
}

/// Generate an error message for invalid parameter values
#[allow(dead_code)]
pub fn value(param_name: &str, constraint: &str, actualvalue: impl Display) -> String {
    format!("Parameter '{param_name}': {constraint} (got: {actualvalue})")
}

/// Generate an error message for out of bounds access
#[allow(dead_code)]
pub fn index(index: usize, length: usize) -> String {
    format!("Index out of bounds: index {index} is invalid for length {length}")
}

/// Generate an error message for empty input
#[allow(dead_code)]
pub fn empty_input(operation: &str) -> String {
    format!("{operation}: input array/collection is empty")
}

/// Generate an error message for numerical computation errors
#[allow(dead_code)]
pub fn numericalerror(operation: &str, issue: &str) -> String {
    format!("{operation}: {issue} - check input values for numerical issues")
}

/// Generate an error message for convergence failures
#[allow(dead_code)]
pub fn algorithm(algorithm: &str, iterations: usize, tolerance: impl Display) -> String {
    format!(
        "{algorithm}: failed to converge after {iterations} iterations (tolerance: {tolerance})"
    )
}

/// Generate an error message for not implemented features
#[allow(dead_code)]
pub fn feature(feature: &str) -> String {
    format!("Feature not implemented: {feature}")
}

/// Generate an error message for invalid array dimensions
#[allow(dead_code)]
pub fn dims(operation: &str, requirement: &str, actualdims: &[usize]) -> String {
    format!("{operation}: {requirement} (got: {actualdims:?})")
}

/// Generate an error message for domain errors
#[allow(dead_code)]
pub fn desc(valuedesc: &str, constraint: &str, value: impl Display) -> String {
    format!("{valuedesc} must be {constraint} (got: {value})")
}

/// Generate an error message for allocation failures
#[allow(dead_code)]
pub fn allocationerror(size: usize, elementtype: &str) -> String {
    format!("Failed to allocate memory for {size} elements of type {elementtype}")
}

/// Generate an error message for file I/O errors
#[allow(dead_code)]
pub fn ioerror(operation: &str, path: &str, details: &str) -> String {
    format!("{operation} failed for '{path}': {details}")
}

/// Generate an error message for parse errors
#[allow(dead_code)]
pub fn parseerror(typename: &str, input: &str, reason: &str) -> String {
    format!("Failed to parse '{input}' as {typename}: {reason}")
}

/// Generate an error message for invalid state
#[allow(dead_code)]
pub fn state(object: &str, expected_state: &str, actualstate: &str) -> String {
    format!("{object} is in invalid state: expected {expected_state}, but was {actualstate}")
}

/// Generate an error message with recovery suggestion
#[allow(dead_code)]
pub fn msg(errormsg: &str, suggestion: &str) -> String {
    format!("{errormsg}\nSuggestion: {suggestion}")
}

/// Common parameter constraints
pub mod constraints {
    /// Generate constraint message for positive values
    pub fn positive() -> &'static str {
        "must be positive (> 0)"
    }

    /// Generate constraint message for non-negative values
    pub fn non_negative() -> &'static str {
        "must be non-negative (>= 0)"
    }

    /// Generate constraint message for probability values
    pub fn probability() -> &'static str {
        "must be a valid probability in [0, 1]"
    }

    /// Generate constraint message for finite values
    pub fn finite() -> &'static str {
        "must be finite (not NaN or infinite)"
    }

    /// Generate constraint message for non-empty
    pub fn non_empty() -> &'static str {
        "must not be empty"
    }

    /// Generate constraint message for square matrix
    pub fn squarematrix() -> &'static str {
        "must be a square matrix"
    }

    /// Generate constraint message for positive definite matrix
    pub fn positive_definite() -> &'static str {
        "must be a positive definite matrix"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_mismatch() {
        let msg = dimension_mismatch("matrix multiplication", "3x4", "5x3");
        assert_eq!(
            msg,
            "matrix multiplication: dimension mismatch (expected: 3x4, actual: 5x3)"
        );
    }

    #[test]
    fn testshape_mismatch() {
        let msg = shape_mismatch("element-wise operation", &[3, 4], &[3, 5]);
        assert_eq!(
            msg,
            "element-wise operation: incompatible shapes ([3, 4] vs [3, 5])"
        );
    }

    // Renamed since these tests were first written: invalid_parameter -> value,
    // domainerror -> desc, with_suggestion -> msg, convergence_failed -> algorithm.
    #[test]
    fn test_invalid_parameter() {
        let result = value("alpha", constraints::positive(), -0.5);
        assert_eq!(
            result,
            "Parameter 'alpha': must be positive (> 0) (got: -0.5)"
        );
    }

    #[test]
    fn test_with_suggestion() {
        let error = desc("input", constraints::positive(), -1.0);
        let result = msg(&error, "use absolute value or check input data");
        assert!(result.contains("input must be must be positive (> 0) (got: -1)"));
        assert!(result.contains("Suggestion: use absolute value or check input data"));
    }

    #[test]
    fn test_convergence_failed() {
        let result = algorithm("Newton-Raphson", 100, 1e-6);
        assert_eq!(
            result,
            "Newton-Raphson: failed to converge after 100 iterations (tolerance: 0.000001)"
        );
    }
}
