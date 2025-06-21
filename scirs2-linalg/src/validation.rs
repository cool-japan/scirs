//! Parameter validation utilities for linear algebra operations
//!
//! This module provides consistent parameter validation functions that are used
//! throughout the linear algebra library to ensure robust error handling and
//! user-friendly error messages.

use crate::error::{LinalgError, LinalgResult};
use ndarray::{ArrayView1, ArrayView2};
use num_traits::Float;

/// Validates that a matrix is not empty
///
/// # Arguments
///
/// * `matrix` - The matrix to validate
/// * `operation` - Name of the operation being performed (for error messages)
///
/// # Returns
///
/// * `Ok(())` if the matrix is not empty
/// * `Err(LinalgError)` if the matrix is empty
pub fn validate_not_empty_matrix<F>(matrix: &ArrayView2<F>, operation: &str) -> LinalgResult<()>
where
    F: Float,
{
    if matrix.is_empty() {
        return Err(LinalgError::ShapeError(format!(
            "{} failed: Input matrix cannot be empty",
            operation
        )));
    }
    Ok(())
}

/// Validates that a vector is not empty
///
/// # Arguments
///
/// * `vector` - The vector to validate
/// * `operation` - Name of the operation being performed (for error messages)
///
/// # Returns
///
/// * `Ok(())` if the vector is not empty
/// * `Err(LinalgError)` if the vector is empty
pub fn validate_not_empty_vector<F>(vector: &ArrayView1<F>, operation: &str) -> LinalgResult<()>
where
    F: Float,
{
    if vector.is_empty() {
        return Err(LinalgError::ShapeError(format!(
            "{} failed: Input vector cannot be empty",
            operation
        )));
    }
    Ok(())
}

/// Validates that a matrix is square
///
/// # Arguments
///
/// * `matrix` - The matrix to validate
/// * `operation` - Name of the operation being performed (for error messages)
///
/// # Returns
///
/// * `Ok(())` if the matrix is square
/// * `Err(LinalgError)` if the matrix is not square
pub fn validate_square_matrix<F>(matrix: &ArrayView2<F>, operation: &str) -> LinalgResult<()>
where
    F: Float,
{
    if matrix.nrows() != matrix.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "{} failed: Matrix must be square\\nMatrix shape: {}×{}\\nExpected: Square matrix (n×n)",
            operation, matrix.nrows(), matrix.ncols()
        )));
    }
    Ok(())
}

/// Validates that matrix dimensions are compatible for matrix-vector operations
///
/// # Arguments
///
/// * `matrix` - The matrix
/// * `vector` - The vector
/// * `operation` - Name of the operation being performed (for error messages)
///
/// # Returns
///
/// * `Ok(())` if dimensions are compatible
/// * `Err(LinalgError)` if dimensions are incompatible
pub fn validate_matrix_vector_dimensions<F>(
    matrix: &ArrayView2<F>,
    vector: &ArrayView1<F>,
    operation: &str,
) -> LinalgResult<()>
where
    F: Float,
{
    if matrix.nrows() != vector.len() {
        return Err(LinalgError::ShapeError(format!(
            "{} failed: Matrix and vector dimensions must match\\nMatrix shape: {}×{}\\nVector shape: {}\\nExpected: Vector length = {}",
            operation, matrix.nrows(), matrix.ncols(), vector.len(), matrix.nrows()
        )));
    }
    Ok(())
}

/// Validates that matrix dimensions are compatible for matrix-matrix operations
///
/// # Arguments
///
/// * `matrix_a` - The first matrix
/// * `matrix_b` - The second matrix
/// * `operation` - Name of the operation being performed (for error messages)
/// * `require_same_rows` - Whether the matrices must have the same number of rows
///
/// # Returns
///
/// * `Ok(())` if dimensions are compatible
/// * `Err(LinalgError)` if dimensions are incompatible
pub fn validate_matrix_matrix_dimensions<F>(
    matrix_a: &ArrayView2<F>,
    matrix_b: &ArrayView2<F>,
    operation: &str,
    require_same_rows: bool,
) -> LinalgResult<()>
where
    F: Float,
{
    if require_same_rows && matrix_a.nrows() != matrix_b.nrows() {
        return Err(LinalgError::ShapeError(format!(
            "{} failed: Matrix rows must match\\nFirst matrix shape: {}×{}\\nSecond matrix shape: {}×{}\\nExpected: Same number of rows",
            operation, matrix_a.nrows(), matrix_a.ncols(), matrix_b.nrows(), matrix_b.ncols()
        )));
    }
    Ok(())
}

/// Validates that all values in a matrix are finite
///
/// # Arguments
///
/// * `matrix` - The matrix to validate
/// * `operation` - Name of the operation being performed (for error messages)
///
/// # Returns
///
/// * `Ok(())` if all values are finite
/// * `Err(LinalgError)` if any value is non-finite
pub fn validate_finite_matrix<F>(matrix: &ArrayView2<F>, operation: &str) -> LinalgResult<()>
where
    F: Float,
{
    for &val in matrix.iter() {
        if !val.is_finite() {
            return Err(LinalgError::InvalidInputError(format!(
                "{} failed: Matrix contains non-finite values",
                operation
            )));
        }
    }
    Ok(())
}

/// Validates that all values in a vector are finite
///
/// # Arguments
///
/// * `vector` - The vector to validate
/// * `operation` - Name of the operation being performed (for error messages)
///
/// # Returns
///
/// * `Ok(())` if all values are finite
/// * `Err(LinalgError)` if any value is non-finite
pub fn validate_finite_vector<F>(vector: &ArrayView1<F>, operation: &str) -> LinalgResult<()>
where
    F: Float,
{
    for &val in vector.iter() {
        if !val.is_finite() {
            return Err(LinalgError::InvalidInputError(format!(
                "{} failed: Vector contains non-finite values",
                operation
            )));
        }
    }
    Ok(())
}

/// Comprehensive validation for linear system operations (Ax = b)
///
/// Performs all necessary validations for solving linear systems:
/// - Matrix and vector are not empty
/// - Matrix is square
/// - Matrix and vector dimensions are compatible
/// - All values are finite
///
/// # Arguments
///
/// * `matrix` - The coefficient matrix A
/// * `vector` - The right-hand side vector b
/// * `operation` - Name of the operation being performed (for error messages)
///
/// # Returns
///
/// * `Ok(())` if all validations pass
/// * `Err(LinalgError)` if any validation fails
pub fn validate_linear_system<F>(
    matrix: &ArrayView2<F>,
    vector: &ArrayView1<F>,
    operation: &str,
) -> LinalgResult<()>
where
    F: Float,
{
    validate_not_empty_matrix(matrix, operation)?;
    validate_not_empty_vector(vector, operation)?;
    validate_square_matrix(matrix, operation)?;
    validate_matrix_vector_dimensions(matrix, vector, operation)?;
    validate_finite_matrix(matrix, operation)?;
    validate_finite_vector(vector, operation)?;
    Ok(())
}

/// Comprehensive validation for least squares operations (Ax = b with A not necessarily square)
///
/// Performs all necessary validations for least squares problems:
/// - Matrix and vector are not empty
/// - Matrix rows and vector length are compatible
/// - All values are finite
///
/// # Arguments
///
/// * `matrix` - The coefficient matrix A
/// * `vector` - The right-hand side vector b
/// * `operation` - Name of the operation being performed (for error messages)
///
/// # Returns
///
/// * `Ok(())` if all validations pass
/// * `Err(LinalgError)` if any validation fails
pub fn validate_least_squares<F>(
    matrix: &ArrayView2<F>,
    vector: &ArrayView1<F>,
    operation: &str,
) -> LinalgResult<()>
where
    F: Float,
{
    validate_not_empty_matrix(matrix, operation)?;
    validate_not_empty_vector(vector, operation)?;
    validate_matrix_vector_dimensions(matrix, vector, operation)?;
    validate_finite_matrix(matrix, operation)?;
    validate_finite_vector(vector, operation)?;
    Ok(())
}

/// Comprehensive validation for matrix decomposition operations
///
/// Performs all necessary validations for matrix decompositions:
/// - Matrix is not empty
/// - Matrix is square (for decompositions that require it)
/// - All values are finite
///
/// # Arguments
///
/// * `matrix` - The matrix to decompose
/// * `operation` - Name of the operation being performed (for error messages)
/// * `require_square` - Whether the decomposition requires a square matrix
///
/// # Returns
///
/// * `Ok(())` if all validations pass
/// * `Err(LinalgError)` if any validation fails
pub fn validate_decomposition<F>(
    matrix: &ArrayView2<F>,
    operation: &str,
    require_square: bool,
) -> LinalgResult<()>
where
    F: Float,
{
    validate_not_empty_matrix(matrix, operation)?;
    if require_square {
        validate_square_matrix(matrix, operation)?;
    }
    validate_finite_matrix(matrix, operation)?;
    Ok(())
}

/// Comprehensive validation for multiple linear systems (AX = B)
///
/// Performs all necessary validations for solving multiple linear systems:
/// - Both matrices are not empty
/// - Coefficient matrix is square
/// - Matrix dimensions are compatible
/// - All values are finite
///
/// # Arguments
///
/// * `coeff_matrix` - The coefficient matrix A
/// * `rhs_matrix` - The right-hand sides matrix B
/// * `operation` - Name of the operation being performed (for error messages)
///
/// # Returns
///
/// * `Ok(())` if all validations pass
/// * `Err(LinalgError)` if any validation fails
pub fn validate_multiple_linear_systems<F>(
    coeff_matrix: &ArrayView2<F>,
    rhs_matrix: &ArrayView2<F>,
    operation: &str,
) -> LinalgResult<()>
where
    F: Float,
{
    validate_not_empty_matrix(coeff_matrix, operation)?;
    validate_not_empty_matrix(rhs_matrix, operation)?;
    validate_square_matrix(coeff_matrix, operation)?;
    validate_matrix_matrix_dimensions(coeff_matrix, rhs_matrix, operation, true)?;
    validate_finite_matrix(coeff_matrix, operation)?;
    validate_finite_matrix(rhs_matrix, operation)?;
    Ok(())
}

/// Enhanced validation for parameter ranges and values
///
/// This function provides validation for common parameter constraints
/// that appear frequently in linear algebra operations.
///
/// # Arguments
///
/// * `value` - The value to validate
/// * `min` - Minimum allowed value (inclusive, None = no minimum)
/// * `max` - Maximum allowed value (inclusive, None = no maximum)
/// * `operation` - Name of the operation being performed (for error messages)
/// * `parameter_name` - Name of the parameter being validated
///
/// # Returns
///
/// * `Ok(())` if all validations pass
/// * `Err(LinalgError)` if any validation fails
#[allow(dead_code)]
pub fn validate_parameter_range<F>(
    value: F,
    min: Option<F>,
    max: Option<F>,
    operation: &str,
    parameter_name: &str,
) -> LinalgResult<()>
where
    F: Float + PartialOrd + std::fmt::Debug,
{
    if !value.is_finite() {
        return Err(LinalgError::InvalidInputError(format!(
            "{} failed: Parameter '{}' must be finite, got {:?}",
            operation, parameter_name, value
        )));
    }

    if let Some(min_val) = min {
        if value < min_val {
            return Err(LinalgError::InvalidInputError(format!(
                "{} failed: Parameter '{}' must be >= {:?}, got {:?}",
                operation, parameter_name, min_val, value
            )));
        }
    }

    if let Some(max_val) = max {
        if value > max_val {
            return Err(LinalgError::InvalidInputError(format!(
                "{} failed: Parameter '{}' must be <= {:?}, got {:?}",
                operation, parameter_name, max_val, value
            )));
        }
    }

    Ok(())
}

/// Validate iteration parameters commonly used in iterative algorithms
///
/// This function validates parameters that appear frequently in iterative
/// algorithms like solvers and eigenvalue methods.
///
/// # Arguments
///
/// * `max_iterations` - Maximum number of iterations (must be > 0)
/// * `tolerance` - Convergence tolerance (must be > 0)
/// * `operation` - Name of the operation being performed (for error messages)
///
/// # Returns
///
/// * `Ok(())` if all validations pass
/// * `Err(LinalgError)` if any validation fails
#[allow(dead_code)]
pub fn validate_iteration_parameters<F>(
    max_iterations: usize,
    tolerance: F,
    operation: &str,
) -> LinalgResult<()>
where
    F: Float + std::fmt::Debug,
{
    if max_iterations == 0 {
        return Err(LinalgError::InvalidInputError(format!(
            "{} failed: Maximum iterations must be > 0, got {}",
            operation, max_iterations
        )));
    }

    validate_parameter_range(tolerance, Some(F::zero()), None, operation, "tolerance")?;

    if tolerance == F::zero() {
        return Err(LinalgError::InvalidInputError(format!(
            "{} failed: Tolerance must be > 0 for convergence, got {:?}",
            operation, tolerance
        )));
    }

    Ok(())
}

/// Validate matrix dimensions for specific algorithm requirements
///
/// This function provides enhanced validation for matrices that need
/// to meet specific size requirements for certain algorithms.
///
/// # Arguments
///
/// * `matrix` - The matrix to validate
/// * `min_size` - Minimum required size (None = no minimum)
/// * `max_size` - Maximum allowed size (None = no maximum)  
/// * `required_square` - Whether the matrix must be square
/// * `operation` - Name of the operation being performed (for error messages)
///
/// # Returns
///
/// * `Ok(())` if all validations pass
/// * `Err(LinalgError)` if any validation fails
#[allow(dead_code)]
pub fn validate_matrix_size_requirements<F>(
    matrix: &ArrayView2<F>,
    min_size: Option<usize>,
    max_size: Option<usize>,
    required_square: bool,
    operation: &str,
) -> LinalgResult<()>
where
    F: Float,
{
    validate_not_empty_matrix(matrix, operation)?;

    if required_square {
        validate_square_matrix(matrix, operation)?;
    }

    let (rows, cols) = matrix.dim();
    let size = if required_square {
        rows
    } else {
        std::cmp::max(rows, cols)
    };

    if let Some(min) = min_size {
        if size < min {
            return Err(LinalgError::InvalidInputError(format!(
                "{} failed: Matrix size {} is below minimum required size {}",
                operation, size, min
            )));
        }
    }

    if let Some(max) = max_size {
        if size > max {
            return Err(LinalgError::InvalidInputError(format!(
                "{} failed: Matrix size {} exceeds maximum allowed size {}",
                operation, size, max
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_validate_not_empty_matrix() {
        let empty_matrix = Array2::<f64>::zeros((0, 0));
        let result = validate_not_empty_matrix(&empty_matrix.view(), "test operation");
        assert!(result.is_err());

        let valid_matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let result = validate_not_empty_matrix(&valid_matrix.view(), "test operation");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_square_matrix() {
        let non_square = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = validate_square_matrix(&non_square.view(), "test operation");
        assert!(result.is_err());

        let square = array![[1.0, 2.0], [3.0, 4.0]];
        let result = validate_square_matrix(&square.view(), "test operation");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_finite_matrix() {
        let invalid_matrix = array![[1.0, f64::NAN], [3.0, 4.0]];
        let result = validate_finite_matrix(&invalid_matrix.view(), "test operation");
        assert!(result.is_err());

        let valid_matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let result = validate_finite_matrix(&valid_matrix.view(), "test operation");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_linear_system() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let vector = array![5.0, 6.0];
        let result = validate_linear_system(&matrix.view(), &vector.view(), "test solve");
        assert!(result.is_ok());

        // Test dimension mismatch
        let wrong_vector = array![5.0, 6.0, 7.0];
        let result = validate_linear_system(&matrix.view(), &wrong_vector.view(), "test solve");
        assert!(result.is_err());
    }
}
