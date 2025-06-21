//! Validation utilities for ``SciRS2``
//!
//! This module provides utilities for validating data and parameters, including
//! production-level security hardening and comprehensive input validation.

use ndarray::{ArrayBase, Dimension, ScalarOperand};
use num_traits::{Float, One, Zero};

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};

/// Checks if a value is within bounds (inclusive)
///
/// # Arguments
///
/// * `value` - The value to check
/// * `min` - The minimum allowed value (inclusive)
/// * `max` - The maximum allowed value (inclusive)
/// * `name` - The name of the parameter being checked
///
/// # Returns
///
/// * `Ok(value)` if the value is within bounds
/// * `Err(CoreError::ValueError)` if the value is out of bounds
///
/// # Errors
///
/// Returns `CoreError::ValueError` if the value is outside the specified bounds.
pub fn check_in_bounds<T, S>(value: T, min: T, max: T, name: S) -> CoreResult<T>
where
    T: PartialOrd + std::fmt::Display + Copy,
    S: Into<String>,
{
    if value < min || value > max {
        return Err(CoreError::ValueError(
            ErrorContext::new(format!(
                "{} must be between {min} and {max}, got {value}",
                name.into()
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    Ok(value)
}

/// Checks if a value is positive
///
/// # Arguments
///
/// * `value` - The value to check
/// * `name` - The name of the parameter being checked
///
/// # Returns
///
/// * `Ok(value)` if the value is positive
/// * `Err(CoreError::ValueError)` if the value is not positive
///
/// # Errors
///
/// Returns `CoreError::ValueError` if the value is not positive.
pub fn check_positive<T, S>(value: T, name: S) -> CoreResult<T>
where
    T: PartialOrd + std::fmt::Display + Copy + Zero,
    S: Into<String>,
{
    if value <= T::zero() {
        return Err(CoreError::ValueError(
            ErrorContext::new(format!("{} must be positive, got {value}", name.into()))
                .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    Ok(value)
}

/// Checks if a value is non-negative
///
/// # Arguments
///
/// * `value` - The value to check
/// * `name` - The name of the parameter being checked
///
/// # Returns
///
/// * `Ok(value)` if the value is non-negative
/// * `Err(CoreError::ValueError)` if the value is negative
///
/// # Errors
///
/// Returns `CoreError::ValueError` if the value is negative.
pub fn check_non_negative<T, S>(value: T, name: S) -> CoreResult<T>
where
    T: PartialOrd + std::fmt::Display + Copy + Zero,
    S: Into<String>,
{
    if value < T::zero() {
        return Err(CoreError::ValueError(
            ErrorContext::new(format!("{} must be non-negative, got {value}", name.into()))
                .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    Ok(value)
}

/// Checks if a floating-point value is finite
///
/// # Arguments
///
/// * `value` - The value to check
/// * `name` - The name of the parameter being checked
///
/// # Returns
///
/// * `Ok(value)` if the value is finite
/// * `Err(CoreError::ValueError)` if the value is not finite
///
/// # Errors
///
/// Returns `CoreError::ValueError` if the value is not finite.
pub fn check_finite<T, S>(value: T, name: S) -> CoreResult<T>
where
    T: Float + std::fmt::Display + Copy,
    S: Into<String>,
{
    if !value.is_finite() {
        return Err(CoreError::ValueError(
            ErrorContext::new(format!("{} must be finite, got {value}", name.into()))
                .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    Ok(value)
}

/// Checks if all values in an array are finite
///
/// # Arguments
///
/// * `array` - The array to check
/// * `name` - The name of the array being checked
///
/// # Returns
///
/// * `Ok(())` if all values are finite
/// * `Err(CoreError::ValueError)` if any value is not finite
///
/// # Errors
///
/// Returns `CoreError::ValueError` if any value in the array is not finite.
pub fn check_array_finite<S, A, D>(array: &ArrayBase<S, D>, name: A) -> CoreResult<()>
where
    S: ndarray::Data,
    D: Dimension,
    S::Elem: Float + std::fmt::Display,
    A: Into<String>,
{
    let name = name.into();
    for (idx, &value) in array.indexed_iter() {
        if !value.is_finite() {
            return Err(CoreError::ValueError(
                ErrorContext::new(format!(
                    "{name} must contain only finite values, got {value} at {idx:?}"
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
    }
    Ok(())
}

/// Checks if an array has the expected shape
///
/// # Arguments
///
/// * `array` - The array to check
/// * `expected_shape` - The expected shape
/// * `name` - The name of the array being checked
///
/// # Returns
///
/// * `Ok(())` if the array has the expected shape
/// * `Err(CoreError::ShapeError)` if the array does not have the expected shape
///
/// # Errors
///
/// Returns `CoreError::ShapeError` if the array does not have the expected shape.
pub fn check_shape<S, D, A>(
    array: &ArrayBase<S, D>,
    expected_shape: &[usize],
    name: A,
) -> CoreResult<()>
where
    S: ndarray::Data,
    D: Dimension,
    A: Into<String>,
{
    let actual_shape = array.shape();
    if actual_shape != expected_shape {
        return Err(CoreError::ShapeError(
            ErrorContext::new(format!(
                "{} has incorrect shape: expected {expected_shape:?}, got {actual_shape:?}",
                name.into()
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    Ok(())
}

/// Checks if an array is 1D
///
/// # Arguments
///
/// * `array` - The array to check
/// * `name` - The name of the array being checked
///
/// # Returns
///
/// * `Ok(())` if the array is 1D
/// * `Err(CoreError::ShapeError)` if the array is not 1D
///
/// # Errors
///
/// Returns `CoreError::ShapeError` if the array is not 1D.
pub fn check_1d<S, D, A>(array: &ArrayBase<S, D>, name: A) -> CoreResult<()>
where
    S: ndarray::Data,
    D: Dimension,
    A: Into<String>,
{
    if array.ndim() != 1 {
        return Err(CoreError::ShapeError(
            ErrorContext::new(format!("{} must be 1D, got {}D", name.into(), array.ndim()))
                .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    Ok(())
}

/// Checks if an array is 2D
///
/// # Arguments
///
/// * `array` - The array to check
/// * `name` - The name of the array being checked
///
/// # Returns
///
/// * `Ok(())` if the array is 2D
/// * `Err(CoreError::ShapeError)` if the array is not 2D
///
/// # Errors
///
/// Returns `CoreError::ShapeError` if the array is not 2D.
pub fn check_2d<S, D, A>(array: &ArrayBase<S, D>, name: A) -> CoreResult<()>
where
    S: ndarray::Data,
    D: Dimension,
    A: Into<String>,
{
    if array.ndim() != 2 {
        return Err(CoreError::ShapeError(
            ErrorContext::new(format!("{} must be 2D, got {}D", name.into(), array.ndim()))
                .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    Ok(())
}

/// Checks if two arrays have the same shape
///
/// # Arguments
///
/// * `a` - The first array
/// * `a_name` - The name of the first array
/// * `b` - The second array
/// * `b_name` - The name of the second array
///
/// # Returns
///
/// * `Ok(())` if the arrays have the same shape
/// * `Err(CoreError::ShapeError)` if the arrays have different shapes
///
/// # Errors
///
/// Returns `CoreError::ShapeError` if the arrays have different shapes.
pub fn check_same_shape<S1, S2, D1, D2, A, B>(
    a: &ArrayBase<S1, D1>,
    a_name: A,
    b: &ArrayBase<S2, D2>,
    b_name: B,
) -> CoreResult<()>
where
    S1: ndarray::Data,
    S2: ndarray::Data,
    D1: Dimension,
    D2: Dimension,
    A: Into<String>,
    B: Into<String>,
{
    let a_shape = a.shape();
    let b_shape = b.shape();
    if a_shape != b_shape {
        return Err(CoreError::ShapeError(
            ErrorContext::new(format!(
                "{} and {} must have the same shape, got {:?} and {:?}",
                a_name.into(),
                b_name.into(),
                a_shape,
                b_shape
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    Ok(())
}

/// Checks if a matrix is square
///
/// # Arguments
///
/// * `matrix` - The matrix to check
/// * `name` - The name of the matrix being checked
///
/// # Returns
///
/// * `Ok(())` if the matrix is square
/// * `Err(CoreError::ShapeError)` if the matrix is not square
///
/// # Errors
///
/// Returns `CoreError::ShapeError` if the matrix is not square.
pub fn check_square<S, D, A>(matrix: &ArrayBase<S, D>, name: A) -> CoreResult<()>
where
    S: ndarray::Data,
    D: Dimension,
    A: Into<String> + std::string::ToString,
{
    check_2d(matrix, name.to_string())?;
    let shape = matrix.shape();
    if shape[0] != shape[1] {
        return Err(CoreError::ShapeError(
            ErrorContext::new(format!(
                "{} must be square, got shape {:?}",
                name.into(),
                shape
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    Ok(())
}

/// Checks if a probability value is valid (between 0 and 1, inclusive)
///
/// # Arguments
///
/// * `p` - The probability value to check
/// * `name` - The name of the parameter being checked
///
/// # Returns
///
/// * `Ok(p)` if the probability is valid
/// * `Err(CoreError::ValueError)` if the probability is not valid
///
/// # Errors
///
/// Returns `CoreError::ValueError` if the probability is not between 0 and 1.
pub fn check_probability<T, S>(p: T, name: S) -> CoreResult<T>
where
    T: Float + std::fmt::Display + Copy,
    S: Into<String>,
{
    if p < T::zero() || p > T::one() {
        return Err(CoreError::ValueError(
            ErrorContext::new(format!(
                "{} must be between 0 and 1, got {}",
                name.into(),
                p
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    Ok(p)
}

/// Checks if an array contains only probabilities (between 0 and 1, inclusive)
///
/// # Arguments
///
/// * `probs` - The array of probabilities to check
/// * `name` - The name of the array being checked
///
/// # Returns
///
/// * `Ok(())` if all values are valid probabilities
/// * `Err(CoreError::ValueError)` if any value is not a valid probability
///
/// # Errors
///
/// Returns `CoreError::ValueError` if any value is not a valid probability.
pub fn check_probabilities<S, D, A>(probs: &ArrayBase<S, D>, name: A) -> CoreResult<()>
where
    S: ndarray::Data,
    D: Dimension,
    S::Elem: Float + std::fmt::Display,
    A: Into<String>,
{
    let name = name.into();
    for (idx, &p) in probs.indexed_iter() {
        if p < S::Elem::zero() || p > S::Elem::one() {
            return Err(CoreError::ValueError(
                ErrorContext::new(format!(
                    "{} must contain only values between 0 and 1, got {} at {:?}",
                    name, p, idx
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
    }
    Ok(())
}

/// Checks if probability values sum to 1
///
/// # Arguments
///
/// * `probs` - The array of probabilities to check
/// * `name` - The name of the array being checked
/// * `tol` - Tolerance for the sum (default: 1e-10)
///
/// # Returns
///
/// * `Ok(())` if the probabilities sum to 1 (within tolerance)
/// * `Err(CoreError::ValueError)` if the sum is not 1 (within tolerance)
pub fn check_probabilities_sum_to_one<S, D, A>(
    probs: &ArrayBase<S, D>,
    name: A,
    tol: Option<S::Elem>,
) -> CoreResult<()>
where
    S: ndarray::Data,
    D: Dimension,
    S::Elem: Float + std::fmt::Display + ScalarOperand,
    A: Into<String> + std::string::ToString,
{
    let tol = tol.unwrap_or_else(|| {
        let eps: f64 = 1e-10;
        num_traits::cast(eps).unwrap()
    });

    check_probabilities(probs, name.to_string())?;

    let sum = probs.sum();
    let one = S::Elem::one();

    if (sum - one).abs() > tol {
        return Err(CoreError::ValueError(
            ErrorContext::new(format!("{} must sum to 1, got sum = {}", name.into(), sum))
                .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }

    Ok(())
}

/// Checks if an array is not empty
///
/// # Arguments
///
/// * `array` - The array to check
/// * `name` - The name of the array being checked
///
/// # Returns
///
/// * `Ok(())` if the array is not empty
/// * `Err(CoreError::ValueError)` if the array is empty
pub fn check_not_empty<S, D, A>(array: &ArrayBase<S, D>, name: A) -> CoreResult<()>
where
    S: ndarray::Data,
    D: Dimension,
    A: Into<String>,
{
    if array.is_empty() {
        return Err(CoreError::ValueError(
            ErrorContext::new(format!("{} cannot be empty", name.into()))
                .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    Ok(())
}

/// Checks if an array has at least the minimum number of samples
///
/// # Arguments
///
/// * `array` - The array to check
/// * `min_samples` - The minimum required number of samples
/// * `name` - The name of the array being checked
///
/// # Returns
///
/// * `Ok(())` if the array has sufficient samples
/// * `Err(CoreError::ValueError)` if the array has too few samples
pub fn check_min_samples<S, D, A>(
    array: &ArrayBase<S, D>,
    min_samples: usize,
    name: A,
) -> CoreResult<()>
where
    S: ndarray::Data,
    D: Dimension,
    A: Into<String>,
{
    let n_samples = array.shape()[0];
    if n_samples < min_samples {
        return Err(CoreError::ValueError(
            ErrorContext::new(format!(
                "{} must have at least {} samples, got {}",
                name.into(),
                min_samples,
                n_samples
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    Ok(())
}

/// Clustering-specific validation utilities
pub mod clustering {
    use super::*;

    /// Validate number of clusters relative to data size
    ///
    /// # Arguments
    ///
    /// * `data` - Input data array
    /// * `n_clusters` - Number of clusters
    /// * `operation` - Name of the operation for error messages
    ///
    /// # Returns
    ///
    /// * `Ok(())` if n_clusters is valid
    /// * `Err(CoreError::ValueError)` if n_clusters is invalid
    pub fn check_n_clusters_bounds<S, D>(
        data: &ArrayBase<S, D>,
        n_clusters: usize,
        operation: &str,
    ) -> CoreResult<()>
    where
        S: ndarray::Data,
        D: Dimension,
    {
        let n_samples = data.shape()[0];

        if n_clusters == 0 {
            return Err(CoreError::ValueError(
                ErrorContext::new(format!(
                    "{}: number of clusters must be > 0, got {}",
                    operation, n_clusters
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        if n_clusters > n_samples {
            return Err(CoreError::ValueError(
                ErrorContext::new(format!(
                    "{}: number of clusters ({}) cannot exceed number of samples ({})",
                    operation, n_clusters, n_samples
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        Ok(())
    }

    /// Comprehensive data validation for clustering algorithms
    ///
    /// # Arguments
    ///
    /// * `data` - Input data array
    /// * `operation` - Name of the operation for error messages
    /// * `check_finite` - Whether to check for finite values
    /// * `min_samples` - Optional minimum number of samples required
    ///
    /// # Returns
    ///
    /// * `Ok(())` if data is valid
    /// * `Err(CoreError)` if data validation fails
    pub fn validate_clustering_data<S, D>(
        data: &ArrayBase<S, D>,
        _operation: &str,
        check_finite: bool,
        min_samples: Option<usize>,
    ) -> CoreResult<()>
    where
        S: ndarray::Data,
        D: Dimension,
        S::Elem: Float + std::fmt::Display,
    {
        // Check not empty
        check_not_empty(data, "data")?;

        // Check 2D for most clustering algorithms
        check_2d(data, "data")?;

        // Check minimum samples if specified
        if let Some(min) = min_samples {
            check_min_samples(data, min, "data")?;
        }

        // Check finite if requested
        if check_finite {
            check_array_finite(data, "data")?;
        }

        Ok(())
    }
}

/// Parameter validation utilities
pub mod parameters {
    use super::*;

    /// Validate algorithm iteration parameters
    ///
    /// # Arguments
    ///
    /// * `max_iter` - Maximum number of iterations
    /// * `tolerance` - Convergence tolerance
    /// * `operation` - Name of the operation for error messages
    ///
    /// # Returns
    ///
    /// * `Ok(())` if parameters are valid
    /// * `Err(CoreError::ValueError)` if parameters are invalid
    pub fn check_iteration_params<T>(
        max_iter: usize,
        tolerance: T,
        operation: &str,
    ) -> CoreResult<()>
    where
        T: Float + std::fmt::Display + Copy,
    {
        if max_iter == 0 {
            return Err(CoreError::ValueError(
                ErrorContext::new(format!(
                    "{}: max_iter must be > 0, got {}",
                    operation, max_iter
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        check_positive(tolerance, format!("{} tolerance", operation))?;

        Ok(())
    }

    /// Validate probability-like parameters (0 <= p <= 1)
    ///
    /// # Arguments
    ///
    /// * `value` - Value to check
    /// * `name` - Parameter name for error messages
    /// * `operation` - Operation name for error messages
    ///
    /// # Returns
    ///
    /// * `Ok(value)` if value is in [0, 1]
    /// * `Err(CoreError::ValueError)` if value is out of range
    pub fn check_unit_interval<T>(value: T, name: &str, operation: &str) -> CoreResult<T>
    where
        T: Float + std::fmt::Display + Copy,
    {
        if value < T::zero() || value > T::one() {
            return Err(CoreError::ValueError(
                ErrorContext::new(format!(
                    "{}: {} must be in [0, 1], got {}",
                    operation, name, value
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        Ok(value)
    }

    /// Validate bandwidth parameter for density-based clustering
    ///
    /// # Arguments
    ///
    /// * `bandwidth` - Bandwidth value
    /// * `operation` - Operation name for error messages
    ///
    /// # Returns
    ///
    /// * `Ok(bandwidth)` if bandwidth is valid
    /// * `Err(CoreError::ValueError)` if bandwidth is invalid
    pub fn check_bandwidth<T>(bandwidth: T, operation: &str) -> CoreResult<T>
    where
        T: Float + std::fmt::Display + Copy,
    {
        check_positive(bandwidth, format!("{} bandwidth", operation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_check_in_bounds() {
        assert!(check_in_bounds(5, 0, 10, "param").is_ok());
        assert!(check_in_bounds(0, 0, 10, "param").is_ok());
        assert!(check_in_bounds(10, 0, 10, "param").is_ok());
        assert!(check_in_bounds(-1, 0, 10, "param").is_err());
        assert!(check_in_bounds(11, 0, 10, "param").is_err());
    }

    #[test]
    fn test_check_positive() {
        assert!(check_positive(5, "param").is_ok());
        assert!(check_positive(0.1, "param").is_ok());
        assert!(check_positive(0, "param").is_err());
        assert!(check_positive(-1, "param").is_err());
    }

    #[test]
    fn test_check_non_negative() {
        assert!(check_non_negative(5, "param").is_ok());
        assert!(check_non_negative(0, "param").is_ok());
        assert!(check_non_negative(-0.1, "param").is_err());
        assert!(check_non_negative(-1, "param").is_err());
    }

    #[test]
    fn test_check_finite() {
        assert!(check_finite(5.0, "param").is_ok());
        assert!(check_finite(0.0, "param").is_ok());
        assert!(check_finite(-1.0, "param").is_ok());
        assert!(check_finite(f64::INFINITY, "param").is_err());
        assert!(check_finite(f64::NEG_INFINITY, "param").is_err());
        assert!(check_finite(f64::NAN, "param").is_err());
    }

    #[test]
    fn test_check_array_finite() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        assert!(check_array_finite(&a, "array").is_ok());

        let b = arr1(&[1.0, f64::INFINITY, 3.0]);
        assert!(check_array_finite(&b, "array").is_err());

        let c = arr1(&[1.0, f64::NAN, 3.0]);
        assert!(check_array_finite(&c, "array").is_err());
    }

    #[test]
    fn test_check_shape() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        assert!(check_shape(&a, &[2, 2], "array").is_ok());
        assert!(check_shape(&a, &[2, 3], "array").is_err());
    }

    #[test]
    fn test_check_1d() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        assert!(check_1d(&a, "array").is_ok());

        let b = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        assert!(check_1d(&b, "array").is_err());
    }

    #[test]
    fn test_check_2d() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        assert!(check_2d(&a, "array").is_ok());

        let b = arr1(&[1.0, 2.0, 3.0]);
        assert!(check_2d(&b, "array").is_err());
    }

    #[test]
    fn test_check_same_shape() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        assert!(check_same_shape(&a, "a", &b, "b").is_ok());

        let c = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert!(check_same_shape(&a, "a", &c, "c").is_err());
    }

    #[test]
    fn test_check_square() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        assert!(check_square(&a, "matrix").is_ok());

        let b = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert!(check_square(&b, "matrix").is_err());

        let c = arr1(&[1.0, 2.0, 3.0]);
        assert!(check_square(&c, "matrix").is_err());
    }

    #[test]
    fn test_check_probability() {
        assert!(check_probability(0.0, "p").is_ok());
        assert!(check_probability(0.5, "p").is_ok());
        assert!(check_probability(1.0, "p").is_ok());
        assert!(check_probability(-0.1, "p").is_err());
        assert!(check_probability(1.1, "p").is_err());
    }

    #[test]
    fn test_check_probabilities() {
        let a = arr1(&[0.0, 0.5, 1.0]);
        assert!(check_probabilities(&a, "probs").is_ok());

        let b = arr1(&[0.0, 0.5, 1.1]);
        assert!(check_probabilities(&b, "probs").is_err());

        let c = arr1(&[-0.1, 0.5, 1.0]);
        assert!(check_probabilities(&c, "probs").is_err());
    }

    #[test]
    fn test_check_probabilities_sum_to_one() {
        let a = arr1(&[0.3, 0.2, 0.5]);
        assert!(check_probabilities_sum_to_one(&a, "probs", None).is_ok());

        let b = arr1(&[0.3, 0.2, 0.6]);
        assert!(check_probabilities_sum_to_one(&b, "probs", None).is_err());

        // Test with custom tolerance
        let c = arr1(&[0.3, 0.2, 0.501]);
        assert!(check_probabilities_sum_to_one(&c, "probs", Some(0.01)).is_ok());
        assert!(check_probabilities_sum_to_one(&c, "probs", Some(0.0001)).is_err());
    }

    #[test]
    fn test_check_not_empty() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        assert!(check_not_empty(&a, "array").is_ok());

        let b = arr1(&[] as &[f64]);
        assert!(check_not_empty(&b, "array").is_err());
    }

    #[test]
    fn test_check_min_samples() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        assert!(check_min_samples(&a, 2, "array").is_ok());
        assert!(check_min_samples(&a, 3, "array").is_ok());
        assert!(check_min_samples(&a, 4, "array").is_err());
    }

    mod clustering_tests {
        use super::*;
        use crate::validation::clustering::*;

        #[test]
        fn test_check_n_clusters_bounds() {
            let data = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);

            assert!(check_n_clusters_bounds(&data, 1, "test").is_ok());
            assert!(check_n_clusters_bounds(&data, 2, "test").is_ok());
            assert!(check_n_clusters_bounds(&data, 3, "test").is_ok());
            assert!(check_n_clusters_bounds(&data, 0, "test").is_err());
            assert!(check_n_clusters_bounds(&data, 4, "test").is_err());
        }

        #[test]
        fn test_validate_clustering_data() {
            let data = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
            assert!(validate_clustering_data(&data, "test", true, Some(2)).is_ok());
            assert!(validate_clustering_data(&data, "test", true, Some(4)).is_err());

            let empty_data = arr2(&[] as &[[f64; 2]; 0]);
            assert!(validate_clustering_data(&empty_data, "test", true, None).is_err());

            let inf_data = arr2(&[[1.0, f64::INFINITY], [3.0, 4.0]]);
            assert!(validate_clustering_data(&inf_data, "test", true, None).is_err());
            assert!(validate_clustering_data(&inf_data, "test", false, None).is_ok());
        }
    }

    mod parameters_tests {
        use crate::validation::parameters::*;

        #[test]
        fn test_check_iteration_params() {
            assert!(check_iteration_params(100, 1e-6, "test").is_ok());
            assert!(check_iteration_params(0, 1e-6, "test").is_err());
            assert!(check_iteration_params(100, 0.0, "test").is_err());
            assert!(check_iteration_params(100, -1e-6, "test").is_err());
        }

        #[test]
        fn test_check_unit_interval() {
            assert!(check_unit_interval(0.0, "param", "test").is_ok());
            assert!(check_unit_interval(0.5, "param", "test").is_ok());
            assert!(check_unit_interval(1.0, "param", "test").is_ok());
            assert!(check_unit_interval(-0.1, "param", "test").is_err());
            assert!(check_unit_interval(1.1, "param", "test").is_err());
        }

        #[test]
        fn test_check_bandwidth() {
            assert!(check_bandwidth(1.0, "test").is_ok());
            assert!(check_bandwidth(0.1, "test").is_ok());
            assert!(check_bandwidth(0.0, "test").is_err());
            assert!(check_bandwidth(-1.0, "test").is_err());
        }
    }
}

// Production-level validation with comprehensive security and performance features
pub mod production;

/// Comprehensive data validation system with schema validation and constraint enforcement
#[cfg(feature = "data_validation")]
pub mod data;
