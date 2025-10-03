//! Parameter validation utilities for linear algebra operations
//!
//! This module provides consistent parameter validation functions that are used
//! throughout the linear algebra library to ensure robust error handling and
//! user-friendly error messages.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;

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
#[allow(dead_code)]
pub fn validate_not_emptymatrix<F>(matrix: &ArrayView2<F>, operation: &str) -> LinalgResult<()>
where
    F: Float,
{
    if matrix.is_empty() {
        return Err(LinalgError::ShapeError(format!(
            "{operation} failed: Input matrix cannot be empty"
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
#[allow(dead_code)]
pub fn validate_not_empty_vector<F>(vector: &ArrayView1<F>, operation: &str) -> LinalgResult<()>
where
    F: Float,
{
    if vector.is_empty() {
        return Err(LinalgError::ShapeError(format!(
            "{operation} failed: Input _vector cannot be empty"
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
#[allow(dead_code)]
pub fn validate_squarematrix<F>(matrix: &ArrayView2<F>, operation: &str) -> LinalgResult<()>
where
    F: Float,
{
    if matrix.nrows() != matrix.ncols() {
        let rows = matrix.nrows();
        let cols = matrix.ncols();
        return Err(LinalgError::ShapeError(format!(
            "{operation} failed: Matrix must be square\\nMatrix shape: {rows}×{cols}\\nExpected: Square matrix (n×n)"
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
#[allow(dead_code)]
pub fn validatematrix_vector_dimensions<F>(
    matrix: &ArrayView2<F>,
    vector: &ArrayView1<F>,
    operation: &str,
) -> LinalgResult<()>
where
    F: Float,
{
    if matrix.nrows() != vector.len() {
        let matrix_rows = matrix.nrows();
        let matrix_cols = matrix.ncols();
        let vector_len = vector.len();
        return Err(LinalgError::ShapeError(format!(
            "{operation} failed: Matrix and vector dimensions must match\\nMatrix shape: {matrix_rows}×{matrix_cols}\\nVector shape: {vector_len}\\nExpected: Vector length = {matrix_rows}"
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
#[allow(dead_code)]
pub fn validatematrixmatrix_dimensions<F>(
    matrix_a: &ArrayView2<F>,
    matrix_b: &ArrayView2<F>,
    operation: &str,
    require_same_rows: bool,
) -> LinalgResult<()>
where
    F: Float,
{
    if require_same_rows && matrix_a.nrows() != matrix_b.nrows() {
        let a_rows = matrix_a.nrows();
        let a_cols = matrix_a.ncols();
        let b_rows = matrix_b.nrows();
        let b_cols = matrix_b.ncols();
        return Err(LinalgError::ShapeError(format!(
            "{operation} failed: Matrix _rows must match\\nFirst matrix shape: {a_rows}×{a_cols}\\nSecond matrix shape: {b_rows}×{b_cols}\\nExpected: Same number of _rows"
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
#[allow(dead_code)]
pub fn validate_finitematrix<F>(matrix: &ArrayView2<F>, operation: &str) -> LinalgResult<()>
where
    F: Float,
{
    for &val in matrix.iter() {
        if !val.is_finite() {
            return Err(LinalgError::InvalidInputError(format!(
                "{operation} failed: Matrix contains non-finite values"
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
#[allow(dead_code)]
pub fn validate_finite_vector<F>(vector: &ArrayView1<F>, operation: &str) -> LinalgResult<()>
where
    F: Float,
{
    for &val in vector.iter() {
        if !val.is_finite() {
            return Err(LinalgError::InvalidInputError(format!(
                "{operation} failed: Vector contains non-finite values"
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
#[allow(dead_code)]
pub fn validate_linear_system<F>(
    matrix: &ArrayView2<F>,
    vector: &ArrayView1<F>,
    operation: &str,
) -> LinalgResult<()>
where
    F: Float,
{
    validate_not_emptymatrix(matrix, operation)?;
    validate_not_empty_vector(vector, operation)?;
    validate_squarematrix(matrix, operation)?;
    validatematrix_vector_dimensions(matrix, vector, operation)?;
    validate_finitematrix(matrix, operation)?;
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
#[allow(dead_code)]
pub fn validate_least_squares<F>(
    matrix: &ArrayView2<F>,
    vector: &ArrayView1<F>,
    operation: &str,
) -> LinalgResult<()>
where
    F: Float,
{
    validate_not_emptymatrix(matrix, operation)?;
    validate_not_empty_vector(vector, operation)?;
    validatematrix_vector_dimensions(matrix, vector, operation)?;
    validate_finitematrix(matrix, operation)?;
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
#[allow(dead_code)]
pub fn validate_decomposition<F>(
    matrix: &ArrayView2<F>,
    operation: &str,
    require_square: bool,
) -> LinalgResult<()>
where
    F: Float,
{
    validate_not_emptymatrix(matrix, operation)?;
    if require_square {
        validate_squarematrix(matrix, operation)?;
    }
    validate_finitematrix(matrix, operation)?;
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
/// * `coeffmatrix` - The coefficient matrix A
/// * `rhsmatrix` - The right-hand sides matrix B
/// * `operation` - Name of the operation being performed (for error messages)
///
/// # Returns
///
/// * `Ok(())` if all validations pass
/// * `Err(LinalgError)` if any validation fails
#[allow(dead_code)]
pub fn validate_multiple_linear_systems<F>(
    coeffmatrix: &ArrayView2<F>,
    rhsmatrix: &ArrayView2<F>,
    operation: &str,
) -> LinalgResult<()>
where
    F: Float,
{
    validate_not_emptymatrix(coeffmatrix, operation)?;
    validate_not_emptymatrix(rhsmatrix, operation)?;
    validate_squarematrix(coeffmatrix, operation)?;
    validatematrixmatrix_dimensions(coeffmatrix, rhsmatrix, operation, true)?;
    validate_finitematrix(coeffmatrix, operation)?;
    validate_finitematrix(rhsmatrix, operation)?;
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
            "{operation} failed: Parameter '{parameter_name}' must be finite, got {value:?}"
        )));
    }

    if let Some(min_val) = min {
        if value < min_val {
            return Err(LinalgError::InvalidInputError(format!(
                "{operation} failed: Parameter '{parameter_name}' must be >= {min_val:?}, got {value:?}"
            )));
        }
    }

    if let Some(max_val) = max {
        if value > max_val {
            return Err(LinalgError::InvalidInputError(format!(
                "{operation} failed: Parameter '{parameter_name}' must be <= {max_val:?}, got {value:?}"
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
            "{operation} failed: Maximum _iterations must be > 0, got {max_iterations}"
        )));
    }

    validate_parameter_range(tolerance, Some(F::zero()), None, operation, "tolerance")?;

    if tolerance == F::zero() {
        return Err(LinalgError::InvalidInputError(format!(
            "{operation} failed: Tolerance must be > 0 for convergence, got {tolerance:?}"
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
/// * `minsize` - Minimum required size (None = no minimum)
/// * `maxsize` - Maximum allowed size (None = no maximum)  
/// * `required_square` - Whether the matrix must be square
/// * `operation` - Name of the operation being performed (for error messages)
///
/// # Returns
///
/// * `Ok(())` if all validations pass
/// * `Err(LinalgError)` if any validation fails
#[allow(dead_code)]
pub fn validatematrixsize_requirements<F>(
    matrix: &ArrayView2<F>,
    minsize: Option<usize>,
    maxsize: Option<usize>,
    required_square: bool,
    operation: &str,
) -> LinalgResult<()>
where
    F: Float,
{
    validate_not_emptymatrix(matrix, operation)?;

    if required_square {
        validate_squarematrix(matrix, operation)?;
    }

    let (rows, cols) = matrix.dim();
    let size = if required_square {
        rows
    } else {
        std::cmp::max(rows, cols)
    };

    if let Some(min) = minsize {
        if size < min {
            return Err(LinalgError::InvalidInputError(format!(
                "{operation} failed: Matrix size {size} is below minimum required size {min}"
            )));
        }
    }

    if let Some(max) = maxsize {
        if size > max {
            return Err(LinalgError::InvalidInputError(format!(
                "{operation} failed: Matrix size {size} exceeds maximum allowed size {max}"
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_validate_not_emptymatrix() {
        let emptymatrix = Array2::<f64>::zeros((0, 0));
        let result = validate_not_emptymatrix(&emptymatrix.view(), "test operation");
        assert!(result.is_err());

        let validmatrix = array![[1.0, 2.0], [3.0, 4.0]];
        let result = validate_not_emptymatrix(&validmatrix.view(), "test operation");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_squarematrix() {
        let non_square = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = validate_squarematrix(&non_square.view(), "test operation");
        assert!(result.is_err());

        let square = array![[1.0, 2.0], [3.0, 4.0]];
        let result = validate_squarematrix(&square.view(), "test operation");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_finitematrix() {
        let invalidmatrix = array![[1.0, f64::NAN], [3.0, 4.0]];
        let result = validate_finitematrix(&invalidmatrix.view(), "test operation");
        assert!(result.is_err());

        let validmatrix = array![[1.0, 2.0], [3.0, 4.0]];
        let result = validate_finitematrix(&validmatrix.view(), "test operation");
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
