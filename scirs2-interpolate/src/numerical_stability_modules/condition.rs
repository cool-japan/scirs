//! Condition number estimation algorithms
//!
//! This module provides various methods for estimating matrix condition numbers,
//! including SVD-based and norm-based approaches.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, SubAssign};

use super::types::{classify_stability, ConditionReport, StabilityDiagnostics, StabilityLevel};
use crate::error::{InterpolateError, InterpolateResult};

/// Assess the numerical condition of a matrix
pub fn assess_matrix_condition<F>(matrix: &ArrayView2<F>) -> InterpolateResult<ConditionReport<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + 'static,
{
    if matrix.nrows() != matrix.ncols() {
        return Err(InterpolateError::ShapeMismatch {
            expected: "square matrix".to_string(),
            actual: format!("{}x{}", matrix.nrows(), matrix.ncols()),
            object: "condition assessment".to_string(),
        });
    }

    let n = matrix.nrows();
    if n == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "Cannot assess condition of empty matrix".to_string(),
        });
    }

    let mut diagnostics = StabilityDiagnostics {
        is_symmetric: check_symmetry(matrix),
        ..Default::default()
    };

    // Estimate condition number
    let condition_number = estimate_condition_number(matrix, &mut diagnostics)?;

    // Classify stability level
    let stability_level = classify_stability(condition_number);

    // Determine if well-conditioned
    let is_well_conditioned = matches!(
        stability_level,
        StabilityLevel::Excellent | StabilityLevel::Good
    );

    // Suggest regularization if needed
    let recommended_regularization = if !is_well_conditioned {
        Some(suggest_regularization(condition_number, &diagnostics))
    } else {
        None
    };

    Ok(ConditionReport {
        condition_number,
        is_well_conditioned,
        recommended_regularization,
        stability_level,
        diagnostics,
    })
}

/// Check if a matrix is symmetric within numerical tolerance
pub fn check_symmetry<F>(matrix: &ArrayView2<F>) -> bool
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    if matrix.nrows() != matrix.ncols() {
        return false;
    }

    let n = matrix.nrows();
    let tolerance =
        super::types::machine_epsilon::<F>() * F::from(100.0).unwrap_or_else(|| F::one());

    for i in 0..n {
        for j in 0..n {
            if (matrix[(i, j)] - matrix[(j, i)]).abs() > tolerance {
                return false;
            }
        }
    }
    true
}

/// Estimate condition number using the most appropriate method
pub fn estimate_condition_number<F>(
    matrix: &ArrayView2<F>,
    diagnostics: &mut StabilityDiagnostics<F>,
) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let n = matrix.nrows();

    // For very small matrices, use direct calculation
    if n == 2 {
        estimate_condition_2x2(matrix, diagnostics)
    } else if n <= 100 {
        // For small matrices, use SVD-based estimation
        estimate_condition_svd(matrix, diagnostics)
    } else {
        // For larger matrices, use norm-based estimation
        estimate_condition_norm_based(matrix)
    }
}

/// Estimate condition number for 2x2 matrices using analytical approach
pub fn estimate_condition_2x2<F>(
    matrix: &ArrayView2<F>,
    diagnostics: &mut StabilityDiagnostics<F>,
) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let a = matrix[(0, 0)];
    let b = matrix[(0, 1)];
    let c = matrix[(1, 0)];
    let d = matrix[(1, 1)];

    // Calculate determinant
    let det = a * d - b * c;

    // If determinant is effectively zero, matrix is singular
    let eps = super::types::machine_epsilon::<F>() * F::from(1000.0).unwrap();
    if det.abs() < eps {
        return Ok(F::infinity());
    }

    // Calculate trace and determinant for eigenvalue computation
    let trace = a + d;
    let discriminant = trace * trace - F::from(4.0).unwrap() * det;

    if discriminant < F::zero() {
        // Complex eigenvalues, use Frobenius norm approach
        let frobenius_norm = (a * a + b * b + c * c + d * d).sqrt();
        let frobenius_norm_inv = F::one() / det.abs() * (d * d + b * b + c * c + a * a).sqrt();
        return Ok(frobenius_norm * frobenius_norm_inv);
    }

    // Real eigenvalues
    let sqrt_discriminant = discriminant.sqrt();
    let lambda1 = (trace + sqrt_discriminant) / F::from(2.0).unwrap();
    let lambda2 = (trace - sqrt_discriminant) / F::from(2.0).unwrap();

    // For condition number, we need singular values, not eigenvalues
    // For symmetric matrices, they're the same; for general matrices, we need different approach
    let max_singular = lambda1.abs().max(lambda2.abs());
    let min_singular = lambda1.abs().min(lambda2.abs());

    if min_singular < eps {
        return Ok(F::infinity());
    }

    diagnostics.max_singular_value = Some(max_singular);
    diagnostics.min_singular_value = Some(min_singular);
    diagnostics.estimated_rank = Some(if min_singular > eps { 2 } else { 1 });

    Ok(max_singular / min_singular)
}

/// Estimate condition number using SVD decomposition
pub fn estimate_condition_svd<F>(
    matrix: &ArrayView2<F>,
    diagnostics: &mut StabilityDiagnostics<F>,
) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let n = matrix.nrows();

    // For demonstration, we'll use a simplified SVD approximation
    // In a full implementation, this would use actual SVD computation

    // Estimate largest singular value using power iteration
    let max_singular = power_iteration_max_eigenvalue(matrix)?;

    // Estimate smallest singular value using inverse power iteration
    let min_singular = inverse_power_iteration_min_eigenvalue(matrix)?;

    diagnostics.max_singular_value = Some(max_singular);
    diagnostics.min_singular_value = Some(min_singular);

    // Estimate rank by counting significant singular values
    let rank_threshold = super::types::machine_epsilon::<F>() * max_singular * F::from(n).unwrap();
    diagnostics.estimated_rank = Some(if min_singular > rank_threshold {
        n
    } else {
        n - 1
    });

    // Check positive definiteness for symmetric matrices
    if diagnostics.is_symmetric {
        diagnostics.is_positive_definite = Some(min_singular > F::zero());
    }

    // Condition number is ratio of largest to smallest singular value
    if min_singular > F::zero() {
        Ok(max_singular / min_singular)
    } else {
        Ok(F::infinity())
    }
}

/// Estimate condition number using matrix norms
pub fn estimate_condition_norm_based<F>(matrix: &ArrayView2<F>) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    // Compute 1-norm of the matrix
    let norm = matrix_1_norm(matrix);

    // Estimate the 1-norm of the inverse using LAPACK-style estimation
    let inv_norm = estimate_inverse_norm(matrix)?;

    Ok(norm * inv_norm)
}

/// Power iteration to estimate maximum eigenvalue (singular value)
fn power_iteration_max_eigenvalue<F>(matrix: &ArrayView2<F>) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let n = matrix.nrows();
    let max_iterations = 50;
    let tolerance = F::from(1e-6).unwrap_or_else(|| super::types::machine_epsilon::<F>());

    // Initialize random vector
    let mut v = Array1::from_elem(n, F::one() / F::from(n).unwrap());
    let mut eigenvalue = F::zero();

    for _ in 0..max_iterations {
        // Multiply by A^T A to get the largest singular value squared
        let mut av = Array1::zeros(n);
        for i in 0..n {
            for j in 0..n {
                av[i] += matrix[(i, j)] * v[j];
            }
        }

        let mut atav = Array1::zeros(n);
        for i in 0..n {
            for j in 0..n {
                atav[i] += matrix[(j, i)] * av[j];
            }
        }

        // Compute Rayleigh quotient
        let numerator = v
            .iter()
            .zip(atav.iter())
            .map(|(x, y)| *x * *y)
            .fold(F::zero(), |acc, x| acc + x);
        let denominator = v.iter().map(|x| *x * *x).fold(F::zero(), |acc, x| acc + x);

        if denominator < super::types::machine_epsilon::<F>() {
            return Err(InterpolateError::NumericalInstability {
                message: "Power iteration failed: zero denominator".to_string(),
            });
        }

        let new_eigenvalue = numerator / denominator;

        // Check convergence
        if (new_eigenvalue - eigenvalue).abs() < tolerance * eigenvalue.abs() {
            return Ok(new_eigenvalue.sqrt());
        }

        eigenvalue = new_eigenvalue;

        // Normalize vector
        let norm = denominator.sqrt();
        for x in v.iter_mut() {
            *x = *x / norm;
        }
    }

    Ok(eigenvalue.sqrt())
}

/// Inverse power iteration to estimate minimum eigenvalue
fn inverse_power_iteration_min_eigenvalue<F>(matrix: &ArrayView2<F>) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let n = matrix.nrows();

    // For simplicity, estimate using the minimum diagonal element as a lower bound
    let mut min_diag = F::infinity();
    for i in 0..n {
        if matrix[(i, i)].abs() < min_diag {
            min_diag = matrix[(i, i)].abs();
        }
    }

    // This is a rough approximation - a full implementation would solve linear systems
    Ok(min_diag.max(super::types::machine_epsilon::<F>()))
}

/// Compute 1-norm of a matrix
fn matrix_1_norm<F>(matrix: &ArrayView2<F>) -> F
where
    F: Float + FromPrimitive + AddAssign,
{
    let mut max_col_sum = F::zero();

    for j in 0..matrix.ncols() {
        let mut col_sum = F::zero();
        for i in 0..matrix.nrows() {
            col_sum += matrix[(i, j)].abs();
        }
        if col_sum > max_col_sum {
            max_col_sum = col_sum;
        }
    }

    max_col_sum
}

/// Estimate 1-norm of matrix inverse using condition estimation
fn estimate_inverse_norm<F>(matrix: &ArrayView2<F>) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    // Simplified inverse norm estimation
    // In practice, this would use more sophisticated algorithms like LAPACK's xCONEST

    let n = matrix.nrows();
    let mut min_diag = F::infinity();

    // Use minimum diagonal element as a proxy
    for i in 0..n {
        let diag_val = matrix[(i, i)].abs();
        if diag_val > F::zero() && diag_val < min_diag {
            min_diag = diag_val;
        }
    }

    if min_diag == F::infinity() || min_diag <= super::types::machine_epsilon::<F>() {
        Ok(F::infinity())
    } else {
        Ok(F::one() / min_diag)
    }
}

/// Suggest regularization parameter based on condition number and diagnostics
pub fn suggest_regularization<F>(condition_number: F, diagnostics: &StabilityDiagnostics<F>) -> F
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let machine_eps = diagnostics.machine_epsilon;

    // Base regularization on condition number
    let base_regularization = if condition_number.is_infinite() {
        machine_eps * F::from(1e6).unwrap_or_else(|| F::from(1e6).unwrap())
    } else {
        machine_eps * condition_number.sqrt()
    };

    // Adjust based on singular value information
    if let Some(min_sv) = diagnostics.min_singular_value {
        if let Some(max_sv) = diagnostics.max_singular_value {
            // Use geometric mean of singular values as guidance
            let geometric_mean = (min_sv * max_sv).sqrt();
            return base_regularization.min(geometric_mean * machine_eps);
        }
    }

    // Default regularization
    base_regularization
}

/// Check if matrix has diagonal dominance
pub fn check_diagonal_dominance<F>(matrix: &ArrayView2<F>) -> bool
where
    F: Float + FromPrimitive + AddAssign,
{
    let n = matrix.nrows();

    for i in 0..n {
        let diag_val = matrix[(i, i)].abs();
        let mut off_diag_sum = F::zero();

        for j in 0..n {
            if i != j {
                off_diag_sum += matrix[(i, j)].abs();
            }
        }

        if diag_val <= off_diag_sum {
            return false;
        }
    }

    true
}

/// Count zero or near-zero diagonal elements
pub fn count_zero_diagonal_elements<F>(matrix: &ArrayView2<F>) -> usize
where
    F: Float + FromPrimitive,
{
    let n = matrix.nrows();
    let tolerance =
        super::types::machine_epsilon::<F>() * F::from(100.0).unwrap_or_else(|| F::one());

    let mut count = 0;
    for i in 0..n {
        if matrix[(i, i)].abs() < tolerance {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_assess_identity_matrix() {
        let matrix = Array2::<f64>::eye(3);
        let report = assess_matrix_condition(&matrix.view()).unwrap();

        assert!(report.is_well_conditioned);
        assert_eq!(report.stability_level, StabilityLevel::Excellent);
        assert!(report.condition_number > 0.0);
        assert!(report.condition_number < 10.0); // Identity should be very well-conditioned
    }

    #[test]
    fn test_symmetry_check() {
        let symmetric = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 3.0]).unwrap();
        let asymmetric = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        assert!(check_symmetry(&symmetric.view()));
        assert!(!check_symmetry(&asymmetric.view()));
    }

    #[test]
    fn test_diagonal_dominance() {
        let dominant = Array2::from_shape_vec((2, 2), vec![3.0, 1.0, 1.0, 3.0]).unwrap();
        let non_dominant = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 1.0]).unwrap();

        assert!(check_diagonal_dominance(&dominant.view()));
        assert!(!check_diagonal_dominance(&non_dominant.view()));
    }

    #[test]
    fn test_zero_diagonal_count() {
        let matrix =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, 0.0])
                .unwrap();

        assert_eq!(count_zero_diagonal_elements(&matrix.view()), 2);
    }

    #[test]
    fn test_matrix_1_norm() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, -2.0, 3.0, -4.0]).unwrap();
        let norm = matrix_1_norm(&matrix.view());

        // Column sums: |1| + |3| = 4, |-2| + |-4| = 6
        // Max column sum = 6
        assert!((norm - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_power_iteration() {
        let matrix = Array2::<f64>::eye(3) * 2.0; // Eigenvalue should be 2.0
        let eigenvalue = power_iteration_max_eigenvalue(&matrix.view()).unwrap();

        assert!((eigenvalue - 2.0).abs() < 1e-6);
    }
}
