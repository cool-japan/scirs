//! Error types for the SciRS2 linear algebra module

use scirs2_core::error::CoreError;
use thiserror::Error;

/// Linear algebra error type
#[derive(Error, Debug)]
pub enum LinalgError {
    /// Core error
    #[error(transparent)]
    CoreError(#[from] CoreError),

    /// Computation error (generic error)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Domain error (input outside valid domain)
    #[error("Domain error: {0}")]
    DomainError(String),

    /// Convergence error (algorithm did not converge)
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Dimension mismatch error
    #[error("Dimension mismatch error: {0}")]
    DimensionError(String),

    /// Shape error (matrices/arrays have incompatible shapes)
    #[error("Shape error: {0}")]
    ShapeError(String),

    /// Out of bounds error
    #[error("Index out of bounds: {0}")]
    IndexError(String),

    /// Singular matrix error
    #[error("Singular matrix error: {0}")]
    SingularMatrixError(String),

    /// Non-positive definite matrix error
    #[error("Non-positive definite matrix error: {0}")]
    NonPositiveDefiniteError(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplementedError(String),

    /// Implementation error (method exists but not fully implemented yet)
    #[error("Implementation error: {0}")]
    ImplementationError(String),

    /// Value error (invalid value)
    #[error("Value error: {0}")]
    ValueError(String),

    /// Invalid input error
    #[error("Invalid input error: {0}")]
    InvalidInputError(String),

    /// Invalid input error (SciPy-compatible alias)
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Numerical error (e.g., overflow, underflow, division by zero)
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Enhanced error types with regularization suggestions
impl LinalgError {
    /// Create a singular matrix error with regularization suggestions
    pub fn singular_matrix_with_suggestions(
        operation: &str,
        matrix_shape: (usize, usize),
        condition_number: Option<f64>,
    ) -> Self {
        let base_msg = format!("Matrix is singular during {} operation", operation);
        let shape_info = format!("Matrix shape: {}×{}", matrix_shape.0, matrix_shape.1);

        let mut suggestions = vec![
            "Consider the following regularization approaches:".to_string(),
            "1. Ridge regularization: Add λI to the matrix (small positive λ)".to_string(),
            "2. Pseudo-inverse: Use SVD-based pseudo-inverse for rank-deficient matrices"
                .to_string(),
            "3. Truncated SVD: Remove small singular values below a threshold".to_string(),
        ];

        if let Some(cond) = condition_number {
            suggestions.push(format!(
                "4. Condition number: {:.2e} (>1e12 indicates ill-conditioning)",
                cond
            ));
            if cond > 1e12 {
                suggestions.push(
                    "5. Use extended precision arithmetic for better numerical stability"
                        .to_string(),
                );
            }
        }

        suggestions.extend_from_slice(&[
            "6. Check input data for linear dependencies or scaling issues".to_string(),
            "7. Use iterative refinement for improved accuracy".to_string(),
        ]);

        let full_msg = format!("{}\n{}\n{}", base_msg, shape_info, suggestions.join("\n"));
        LinalgError::SingularMatrixError(full_msg)
    }

    /// Create a non-positive definite error with regularization suggestions
    pub fn non_positive_definite_with_suggestions(
        operation: &str,
        matrix_shape: (usize, usize),
        negative_eigenvalues: Option<usize>,
    ) -> Self {
        let base_msg = format!(
            "Matrix is not positive definite during {} operation",
            operation
        );
        let shape_info = format!("Matrix shape: {}×{}", matrix_shape.0, matrix_shape.1);

        let mut suggestions = vec![
            "Consider the following regularization approaches:".to_string(),
            "1. Diagonal regularization: Add λI where λ > |most negative eigenvalue|".to_string(),
            "2. Modified Cholesky: Use algorithms that ensure positive definiteness".to_string(),
            "3. Eigenvalue clipping: Replace negative eigenvalues with small positive values"
                .to_string(),
            "4. Use LDL decomposition instead of Cholesky for indefinite matrices".to_string(),
        ];

        if let Some(neg_count) = negative_eigenvalues {
            suggestions.push(format!(
                "5. Found {} negative eigenvalue(s) - consider spectral regularization",
                neg_count
            ));
        }

        suggestions.extend_from_slice(&[
            "6. Check if matrix is symmetric (required for Cholesky)".to_string(),
            "7. Use pivoted Cholesky for rank-deficient positive semidefinite matrices".to_string(),
            "8. Consider using QR or LU decomposition for non-symmetric matrices".to_string(),
        ]);

        let full_msg = format!("{}\n{}\n{}", base_msg, shape_info, suggestions.join("\n"));
        LinalgError::NonPositiveDefiniteError(full_msg)
    }

    /// Create a convergence error with algorithm suggestions
    pub fn convergence_with_suggestions(
        algorithm: &str,
        iterations: usize,
        tolerance: f64,
        current_residual: Option<f64>,
    ) -> Self {
        let base_msg = format!(
            "{} failed to converge after {} iterations",
            algorithm, iterations
        );
        let tolerance_info = format!("Target tolerance: {:.2e}", tolerance);

        let mut suggestions = vec![
            "Consider the following approaches to improve convergence:".to_string(),
            "1. Increase maximum iterations limit".to_string(),
            "2. Relax convergence tolerance".to_string(),
            "3. Use preconditioning to improve condition number".to_string(),
            "4. Try different initial guess or starting point".to_string(),
        ];

        if let Some(residual) = current_residual {
            suggestions.push(format!(
                "5. Current residual: {:.2e} (target: {:.2e})",
                residual, tolerance
            ));
            if residual / tolerance < 10.0 {
                suggestions.push(
                    "6. Close to convergence - try increasing iterations slightly".to_string(),
                );
            } else {
                suggestions
                    .push("6. Far from convergence - consider algorithm changes".to_string());
            }
        }

        suggestions.extend_from_slice(&[
            "7. Use mixed precision arithmetic for better numerical stability".to_string(),
            "8. Check matrix conditioning - use regularization if poorly conditioned".to_string(),
            "9. Consider switching to direct methods for smaller problems".to_string(),
        ]);

        let full_msg = format!(
            "{}\n{}\n{}",
            base_msg,
            tolerance_info,
            suggestions.join("\n")
        );
        LinalgError::ConvergenceError(full_msg)
    }
}

/// Result type for linear algebra operations
pub type LinalgResult<T> = Result<T, LinalgError>;

/// Checks if a condition is true, otherwise returns a domain error
///
/// # Arguments
///
/// * `condition` - The condition to check
/// * `message` - The error message if the condition is false
///
/// # Returns
///
/// * `Ok(())` if the condition is true
/// * `Err(LinalgError::DomainError)` if the condition is false
pub fn check_domain<S: AsRef<str>>(condition: bool, message: S) -> LinalgResult<()> {
    if condition {
        Ok(())
    } else {
        Err(LinalgError::DomainError(message.as_ref().to_string()))
    }
}

/// Checks matrix dimensions
///
/// # Arguments
///
/// * `condition` - The condition to check
/// * `message` - The error message if the condition is false
///
/// # Returns
///
/// * `Ok(())` if the condition is true
/// * `Err(LinalgError::DimensionError)` if the condition is false
///
/// # Note
///
/// This is a linalg-specific wrapper around scirs2_core::validation functions.
/// For new code, consider using scirs2_core::validation functions directly when possible.
pub fn check_dimensions<S: AsRef<str>>(condition: bool, message: S) -> LinalgResult<()> {
    if condition {
        Ok(())
    } else {
        Err(LinalgError::DimensionError(message.as_ref().to_string()))
    }
}

/// Checks if a value is valid
///
/// # Arguments
///
/// * `condition` - The condition to check
/// * `message` - The error message if the condition is false
///
/// # Returns
///
/// * `Ok(())` if the condition is true
/// * `Err(LinalgError::ValueError)` if the condition is false
///
/// # Note
///
/// This is a linalg-specific wrapper around scirs2_core::validation functions.
/// For new code, consider using scirs2_core::validation functions directly when possible.
pub fn check_value<S: AsRef<str>>(condition: bool, message: S) -> LinalgResult<()> {
    if condition {
        Ok(())
    } else {
        Err(LinalgError::ValueError(message.as_ref().to_string()))
    }
}
