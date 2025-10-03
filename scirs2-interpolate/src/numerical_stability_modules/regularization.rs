//! Regularization and numerical stabilization methods
//!
//! This module provides various regularization techniques and safe arithmetic
//! operations to improve numerical stability in interpolation algorithms.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, SubAssign};

use super::types::{machine_epsilon, StabilityDiagnostics};
use crate::error::{InterpolateError, InterpolateResult};

/// Check if a division operation is numerically safe
pub fn check_safe_division<F>(numerator: F, denominator: F) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + std::fmt::LowerExp,
{
    let eps = machine_epsilon::<F>();
    let min_denominator = eps * F::from(1000.0).unwrap_or_else(|| F::from(1000.0).unwrap());

    if denominator.abs() < min_denominator {
        Err(InterpolateError::NumericalInstability {
            message: format!(
                "Division by very small number: {:.2e} < {:.2e}",
                denominator.abs(),
                min_denominator
            ),
        })
    } else {
        Ok(numerator / denominator)
    }
}

/// Compute safe reciprocal with underflow protection
pub fn safe_reciprocal<F>(value: F) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + std::fmt::LowerExp,
{
    let eps = machine_epsilon::<F>();
    let min_value = eps * F::from(1000.0).unwrap_or_else(|| F::from(1000.0).unwrap());

    if value.abs() < min_value {
        Err(InterpolateError::NumericalInstability {
            message: format!(
                "Cannot compute reciprocal of very small value: {:.2e}",
                value.abs()
            ),
        })
    } else {
        Ok(F::one() / value)
    }
}

/// Apply Tikhonov regularization to a matrix
pub fn apply_tikhonov_regularization<F>(
    matrix: &ArrayView2<F>,
    regularization_parameter: F,
) -> InterpolateResult<Array2<F>>
where
    F: Float + FromPrimitive + Clone + AddAssign,
{
    if matrix.nrows() != matrix.ncols() {
        return Err(InterpolateError::ShapeMismatch {
            expected: "square matrix".to_string(),
            actual: format!("{}x{}", matrix.nrows(), matrix.ncols()),
            object: "Tikhonov regularization".to_string(),
        });
    }

    let n = matrix.nrows();
    let mut regularized = matrix.to_owned();

    // Add regularization to diagonal elements
    for i in 0..n {
        regularized[(i, i)] += regularization_parameter;
    }

    Ok(regularized)
}

/// Apply adaptive regularization based on matrix properties
pub fn apply_adaptive_regularization<F>(
    matrix: &ArrayView2<F>,
    diagnostics: &StabilityDiagnostics<F>,
) -> InterpolateResult<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + Clone,
{
    // Calculate adaptive regularization parameter
    let base_regularization = diagnostics.machine_epsilon;

    let adaptive_factor = if let Some(min_sv) = diagnostics.min_singular_value {
        if let Some(max_sv) = diagnostics.max_singular_value {
            // Use condition number to determine regularization strength
            let condition_ratio = max_sv / min_sv.max(diagnostics.machine_epsilon);
            let log_condition = condition_ratio.ln().max(F::one());
            log_condition.sqrt()
        } else {
            F::from(10.0).unwrap_or_else(|| F::from(10.0).unwrap())
        }
    } else {
        F::from(10.0).unwrap_or_else(|| F::from(10.0).unwrap())
    };

    let regularization_param = base_regularization * adaptive_factor;

    apply_tikhonov_regularization(matrix, regularization_param)
}

/// Detect various numerical edge cases in a matrix
pub fn detect_edge_cases<F>(
    matrix: &ArrayView2<F>,
) -> InterpolateResult<super::types::EdgeCaseReport<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + std::ops::MulAssign,
{
    let n = matrix.nrows();
    if n == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "Cannot analyze empty matrix".to_string(),
        });
    }

    let mut report = super::types::EdgeCaseReport::default();

    // Check for near-singularity
    report.is_nearly_singular = is_nearly_singular(matrix);

    // Check diagonal dominance
    report.has_diagonal_dominance = super::condition::check_diagonal_dominance(matrix);

    // Count zero diagonal elements
    report.zero_diagonal_count = super::condition::count_zero_diagonal_elements(matrix);

    // Estimate numerical rank
    report.numerical_rank = Some(estimate_numerical_rank(matrix));

    // Determine recommended treatment
    report.recommended_treatment = determine_treatment(&report);

    // Generate issue description
    report.issue_description = generate_issue_description(&report);

    Ok(report)
}

/// Check if matrix is nearly singular
fn is_nearly_singular<F>(matrix: &ArrayView2<F>) -> bool
where
    F: Float + FromPrimitive + std::ops::MulAssign,
{
    let n = matrix.nrows();
    let eps = machine_epsilon::<F>();
    let threshold =
        eps * F::from(n).unwrap() * F::from(1000.0).unwrap_or_else(|| F::from(1000.0).unwrap());

    // For 2x2 matrices, compute actual determinant
    if n == 2 {
        let det = matrix[(0, 0)] * matrix[(1, 1)] - matrix[(0, 1)] * matrix[(1, 0)];
        return det.abs() < F::from(1e-10).unwrap_or_else(|| F::from(1e-10).unwrap());
    }

    // For larger matrices, check diagonal product as approximation
    let mut diag_product = F::one();
    for i in 0..n {
        diag_product *= matrix[(i, i)].abs();
        if diag_product < threshold {
            return true;
        }
    }

    false
}

/// Estimate numerical rank of matrix
fn estimate_numerical_rank<F>(matrix: &ArrayView2<F>) -> usize
where
    F: Float + FromPrimitive,
{
    let n = matrix.nrows();
    let eps = machine_epsilon::<F>();

    // Simple rank estimation based on diagonal elements
    // In practice, this would use SVD or QR decomposition
    let threshold =
        eps * F::from(n).unwrap() * F::from(100.0).unwrap_or_else(|| F::from(100.0).unwrap());

    let mut rank = 0;
    for i in 0..n {
        if matrix[(i, i)].abs() > threshold {
            rank += 1;
        }
    }

    rank.max(1) // Ensure at least rank 1
}

/// Determine appropriate treatment based on edge case analysis
fn determine_treatment<F>(
    report: &super::types::EdgeCaseReport<F>,
) -> super::types::EdgeCaseTreatment
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    use super::types::EdgeCaseTreatment;

    if report.is_nearly_singular {
        if report.zero_diagonal_count > 0 {
            EdgeCaseTreatment::Pivoting
        } else {
            EdgeCaseTreatment::TikhonovRegularization
        }
    } else if !report.has_diagonal_dominance {
        EdgeCaseTreatment::Preconditioning
    } else if report.zero_diagonal_count > 0 {
        EdgeCaseTreatment::Pivoting
    } else {
        EdgeCaseTreatment::None
    }
}

/// Generate human-readable description of detected issues
fn generate_issue_description<F>(report: &super::types::EdgeCaseReport<F>) -> String
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let mut issues = Vec::new();

    if report.is_nearly_singular {
        issues.push("Matrix is nearly singular".to_string());
    }

    if !report.has_diagonal_dominance {
        issues.push("Matrix lacks diagonal dominance".to_string());
    }

    if report.zero_diagonal_count > 0 {
        issues.push(format!(
            "{} zero diagonal elements detected",
            report.zero_diagonal_count
        ));
    }

    if let Some(rank) = report.numerical_rank {
        if rank < report.zero_diagonal_count + report.zero_diagonal_count {
            issues.push(format!(
                "Matrix appears rank-deficient (estimated rank: {})",
                rank
            ));
        }
    }

    if issues.is_empty() {
        "No numerical issues detected".to_string()
    } else {
        issues.join("; ")
    }
}

/// Apply iterative refinement to improve solution accuracy
pub fn iterative_refinement<F>(
    original_matrix: &ArrayView2<F>,
    lu_factors: &ArrayView2<F>,
    original_rhs: &ArrayView1<F>,
    initial_solution: &ArrayView1<F>,
    max_iterations: usize,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + Clone,
{
    let n = original_matrix.nrows();
    let tolerance = machine_epsilon::<F>()
        * F::from(n).unwrap()
        * F::from(100.0).unwrap_or_else(|| F::from(100.0).unwrap());

    let mut solution = initial_solution.to_owned();
    let mut residual = Array1::zeros(n);

    for iteration in 0..max_iterations {
        // Compute residual: r = b - Ax
        for i in 0..n {
            let mut ax_i = F::zero();
            for j in 0..n {
                ax_i += original_matrix[(i, j)] * solution[j];
            }
            residual[i] = original_rhs[i] - ax_i;
        }

        // Check convergence
        let residual_norm = residual
            .iter()
            .map(|x| x.abs())
            .fold(F::zero(), |acc, x| acc.max(x));
        if residual_norm < tolerance {
            break;
        }

        // Solve for correction: LU * delta = residual
        let correction = solve_with_lu_factors(lu_factors, &residual.view())?;

        // Update solution
        for i in 0..n {
            solution[i] += correction[i];
        }

        // Prevent infinite refinement
        if iteration > max_iterations / 2 {
            let improvement_ratio = residual_norm / (residual_norm + tolerance);
            if improvement_ratio > F::from(0.9).unwrap_or_else(|| F::from(0.9).unwrap()) {
                break; // Not converging well enough
            }
        }
    }

    Ok(solution)
}

/// Solve linear system using precomputed LU factors
fn solve_with_lu_factors<F>(
    lu_factors: &ArrayView2<F>,
    rhs: &ArrayView1<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + Clone,
{
    let n = lu_factors.nrows();

    // Forward substitution (L * y = b)
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= lu_factors[(i, j)] * y[j];
        }
        y[i] = sum; // L has unit diagonal
    }

    // Back substitution (U * x = y)
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= lu_factors[(i, j)] * x[j];
        }

        let diagonal = lu_factors[(i, i)];
        if diagonal.abs() < machine_epsilon::<F>() {
            return Err(InterpolateError::NumericalInstability {
                message: format!("Zero diagonal element at position {}", i),
            });
        }

        x[i] = sum / diagonal;
    }

    Ok(x)
}

/// Apply preconditioning to improve matrix condition
pub fn apply_preconditioning<F>(
    matrix: &ArrayView2<F>,
    preconditioner_type: PreconditionerType,
) -> InterpolateResult<(Array2<F>, Array2<F>)>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + Clone,
{
    match preconditioner_type {
        PreconditionerType::Diagonal => diagonal_preconditioning(matrix),
        PreconditionerType::IncompleteCholesky => incomplete_cholesky_preconditioning(matrix),
        PreconditionerType::Jacobi => jacobi_preconditioning(matrix),
    }
}

/// Types of preconditioners available
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreconditionerType {
    /// Simple diagonal preconditioning
    Diagonal,
    /// Incomplete Cholesky factorization
    IncompleteCholesky,
    /// Jacobi preconditioning
    Jacobi,
}

/// Apply diagonal preconditioning
fn diagonal_preconditioning<F>(matrix: &ArrayView2<F>) -> InterpolateResult<(Array2<F>, Array2<F>)>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + Clone,
{
    let n = matrix.nrows();
    let mut preconditioner = Array2::zeros((n, n));
    let mut inv_preconditioner = Array2::zeros((n, n));

    let eps = machine_epsilon::<F>();

    for i in 0..n {
        let diag_val = matrix[(i, i)];
        if diag_val.abs() < eps {
            return Err(InterpolateError::NumericalInstability {
                message: format!("Zero diagonal element at position {}", i),
            });
        }

        let sqrt_diag = diag_val.abs().sqrt();
        preconditioner[(i, i)] = sqrt_diag;
        inv_preconditioner[(i, i)] = F::one() / sqrt_diag;
    }

    Ok((preconditioner, inv_preconditioner))
}

/// Apply incomplete Cholesky preconditioning (simplified version)
fn incomplete_cholesky_preconditioning<F>(
    matrix: &ArrayView2<F>,
) -> InterpolateResult<(Array2<F>, Array2<F>)>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + Clone,
{
    // Simplified IC(0) - only diagonal and original non-zero pattern
    let n = matrix.nrows();
    let mut l = Array2::zeros((n, n));

    // Incomplete Cholesky factorization
    for i in 0..n {
        // Compute diagonal element
        let mut sum = F::zero();
        for k in 0..i {
            sum += l[(i, k)] * l[(i, k)];
        }

        let diag_candidate = matrix[(i, i)] - sum;
        if diag_candidate <= F::zero() {
            return Err(InterpolateError::NumericalInstability {
                message: format!("Non-positive definite matrix at position {}", i),
            });
        }

        l[(i, i)] = diag_candidate.sqrt();

        // Compute lower triangular elements
        for j in (i + 1)..n {
            if matrix[(j, i)] != F::zero() {
                // Only fill existing non-zeros
                let mut sum = F::zero();
                for k in 0..i {
                    sum += l[(i, k)] * l[(j, k)];
                }
                l[(j, i)] = (matrix[(j, i)] - sum) / l[(i, i)];
            }
        }
    }

    let l_transpose = transpose_matrix(&l.view());
    Ok((l, l_transpose))
}

/// Apply Jacobi preconditioning
fn jacobi_preconditioning<F>(matrix: &ArrayView2<F>) -> InterpolateResult<(Array2<F>, Array2<F>)>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + Clone,
{
    diagonal_preconditioning(matrix) // Jacobi is same as diagonal for preconditioning
}

/// Transpose a matrix
fn transpose_matrix<F>(matrix: &ArrayView2<F>) -> Array2<F>
where
    F: Clone + scirs2_core::numeric::Zero,
{
    let (rows, cols) = matrix.dim();
    let mut result = Array2::zeros((cols, rows));

    for i in 0..rows {
        for j in 0..cols {
            result[(j, i)] = matrix[(i, j)].clone();
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_safe_division() {
        assert!(check_safe_division(1.0, 2.0).is_ok());
        assert!(check_safe_division(1.0, 1e-20).is_err());
    }

    #[test]
    fn test_safe_reciprocal() {
        assert!(safe_reciprocal(2.0).is_ok());
        assert_eq!(safe_reciprocal(2.0).unwrap(), 0.5);
        assert!(safe_reciprocal(1e-20).is_err());
    }

    #[test]
    fn test_tikhonov_regularization() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 0.5, 0.5, 1.0]).unwrap();
        let regularized = apply_tikhonov_regularization(&matrix.view(), 0.1).unwrap();

        assert_eq!(regularized[(0, 0)], 1.1);
        assert_eq!(regularized[(1, 1)], 1.1);
        assert_eq!(regularized[(0, 1)], 0.5);
        assert_eq!(regularized[(1, 0)], 0.5);
    }

    #[test]
    fn test_edge_case_detection() {
        let singular_matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        let report = detect_edge_cases(&singular_matrix.view()).unwrap();

        assert!(report.is_nearly_singular);
    }

    #[test]
    fn test_diagonal_preconditioning() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 9.0]).unwrap();
        let (precond, inv_precond) = diagonal_preconditioning(&matrix.view()).unwrap();

        assert!((precond[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((precond[(1, 1)] - 3.0).abs() < 1e-10);
        assert!((inv_precond[(0, 0)] - 0.5).abs() < 1e-10);
        assert!((inv_precond[(1, 1)] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_numerical_rank_estimation() {
        let full_rank = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let rank_deficient = Array2::from_shape_vec((2, 2), vec![1e-20, 0.0, 0.0, 1.0]).unwrap();

        assert_eq!(estimate_numerical_rank(&full_rank.view()), 2);
        assert_eq!(estimate_numerical_rank(&rank_deficient.view()), 1);
    }
}
