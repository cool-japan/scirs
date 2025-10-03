//! Enhanced linear solvers with stability monitoring
//!
//! This module provides robust linear system solvers with comprehensive
//! stability monitoring, iterative refinement, and adaptive solver selection.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, SubAssign};

use super::condition::assess_matrix_condition;
use super::regularization::{detect_edge_cases, iterative_refinement};
use super::types::{
    ConditionReport, ConvergenceInfo, EdgeCaseReport, EnhancedStabilityReport, SolveStrategy,
    StabilityLevel,
};
use crate::error::{InterpolateError, InterpolateResult};

/// Solve linear system with comprehensive stability monitoring
pub fn solve_with_enhanced_monitoring<F>(
    matrix: &ArrayView2<F>,
    rhs: &ArrayView1<F>,
) -> InterpolateResult<(Array1<F>, EnhancedStabilityReport<F>)>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + Clone
        + 'static
        + std::ops::MulAssign
        + std::ops::DivAssign,
{
    if matrix.nrows() != matrix.ncols() {
        return Err(InterpolateError::ShapeMismatch {
            expected: "square matrix".to_string(),
            actual: format!("{}x{}", matrix.nrows(), matrix.ncols()),
            object: "linear system solve".to_string(),
        });
    }

    if matrix.nrows() != rhs.len() {
        return Err(InterpolateError::ShapeMismatch {
            expected: format!("{} elements", matrix.nrows()),
            actual: format!("{} elements", rhs.len()),
            object: "right-hand side vector".to_string(),
        });
    }

    // Comprehensive stability analysis
    let condition_report = assess_matrix_condition(matrix)?;
    let edge_case_report = detect_edge_cases(matrix)?;

    // Determine optimal solving strategy
    let recommended_strategy = determine_solve_strategy(&condition_report, &edge_case_report);

    // Create convergence info
    let convergence_info = create_convergence_info(&condition_report, recommended_strategy);

    // Determine if iterative refinement is needed
    let needs_iterative_refinement = condition_report.stability_level == StabilityLevel::Marginal
        || condition_report.stability_level == StabilityLevel::Poor;

    let enhanced_report = EnhancedStabilityReport {
        condition_report,
        edge_case_report,
        recommended_strategy,
        convergence_info,
        needs_iterative_refinement,
    };

    // Solve using the recommended strategy
    let solution = solve_with_strategy(matrix, rhs, &enhanced_report)?;

    Ok((solution, enhanced_report))
}

/// Solve linear system with basic stability monitoring
pub fn solve_with_stability_monitoring<F>(
    matrix: &ArrayView2<F>,
    rhs: &ArrayView1<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + Clone
        + 'static
        + std::ops::MulAssign
        + std::ops::DivAssign,
{
    let (solution, _report) = solve_with_enhanced_monitoring(matrix, rhs)?;
    Ok(solution)
}

/// Solve using the recommended strategy from stability analysis
fn solve_with_strategy<F>(
    matrix: &ArrayView2<F>,
    rhs: &ArrayView1<F>,
    report: &EnhancedStabilityReport<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + Clone
        + std::ops::DivAssign,
{
    match report.recommended_strategy {
        SolveStrategy::DirectLU => solve_direct_lu(matrix, rhs, report),
        SolveStrategy::DirectQR => solve_direct_qr(matrix, rhs, report),
        SolveStrategy::IterativeCG => solve_iterative_cg(matrix, rhs, report),
        SolveStrategy::IterativeGMRES => solve_iterative_gmres(matrix, rhs, report),
        SolveStrategy::Regularized => solve_regularized(matrix, rhs, report),
    }
}

/// Solve using direct LU decomposition with optional iterative refinement
fn solve_direct_lu<F>(
    matrix: &ArrayView2<F>,
    rhs: &ArrayView1<F>,
    report: &EnhancedStabilityReport<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + Clone,
{
    // Perform LU decomposition
    let lu_result = lu_decomposition_with_pivoting(matrix)?;
    let (lu_factors, permutation) = lu_result;

    // Solve using LU factors
    let mut solution = solve_with_lu_factors(&lu_factors.view(), &permutation, rhs)?;

    // Apply iterative refinement if recommended
    if report.needs_iterative_refinement {
        solution = iterative_refinement(matrix, &lu_factors.view(), rhs, &solution.view(), 5)?;
    }

    Ok(solution)
}

/// Solve using direct QR decomposition
fn solve_direct_qr<F>(
    matrix: &ArrayView2<F>,
    rhs: &ArrayView1<F>,
    _report: &EnhancedStabilityReport<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + Clone
        + std::ops::DivAssign,
{
    // Simplified QR solve - in practice would use Householder reflections
    let (q, r) = qr_decomposition(matrix)?;

    // Solve Q^T * Q * R * x = Q^T * b
    let qt_b = multiply_qt_vector(&q.view(), rhs)?;
    let solution = solve_upper_triangular(&r.view(), &qt_b.view())?;

    Ok(solution)
}

/// Solve using iterative Conjugate Gradient method
fn solve_iterative_cg<F>(
    matrix: &ArrayView2<F>,
    rhs: &ArrayView1<F>,
    report: &EnhancedStabilityReport<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + Clone,
{
    let n = matrix.nrows();
    let max_iterations = report.convergence_info.expected_iterations;
    let tolerance = report.convergence_info.recommended_tolerance;

    // Initialize solution
    let mut x = Array1::zeros(n);
    let mut r = rhs.to_owned(); // r = b - A*x (x starts at 0)
    let mut p = r.clone();
    let mut rsold = dot_product(&r.view(), &r.view());

    for _iteration in 0..max_iterations {
        // Check convergence
        if rsold.sqrt() < tolerance {
            break;
        }

        // Compute A*p
        let ap = matrix_vector_multiply(matrix, &p.view())?;

        // Compute step size
        let pap = dot_product(&p.view(), &ap.view());
        if pap.abs() < super::types::machine_epsilon::<F>() {
            break; // Avoid division by zero
        }
        let alpha = rsold / pap;

        // Update solution and residual
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rsnew = dot_product(&r.view(), &r.view());

        // Update search direction
        let beta = rsnew / rsold;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
    }

    Ok(x)
}

/// Solve using iterative GMRES method (simplified version)
fn solve_iterative_gmres<F>(
    matrix: &ArrayView2<F>,
    rhs: &ArrayView1<F>,
    report: &EnhancedStabilityReport<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + Clone,
{
    // Simplified GMRES implementation
    // In practice, this would use Arnoldi iteration and Givens rotations

    let n = matrix.nrows();
    let max_iterations = report.convergence_info.expected_iterations.min(50); // Limit restart
    let tolerance = report.convergence_info.recommended_tolerance;

    let mut x = Array1::zeros(n);
    let mut r = rhs.to_owned();

    // Simple Richardson iteration as GMRES placeholder
    for _iteration in 0..max_iterations {
        let residual_norm = vector_norm(&r.view());
        if residual_norm < tolerance {
            break;
        }

        // Simple step: x = x + alpha * r
        let alpha = F::from(0.1).unwrap_or_else(|| F::from(0.1).unwrap());
        for i in 0..n {
            x[i] += alpha * r[i];
        }

        // Update residual: r = b - A*x
        let ax = matrix_vector_multiply(matrix, &x.view())?;
        for i in 0..n {
            r[i] = rhs[i] - ax[i];
        }
    }

    Ok(x)
}

/// Solve using regularization
fn solve_regularized<F>(
    matrix: &ArrayView2<F>,
    rhs: &ArrayView1<F>,
    report: &EnhancedStabilityReport<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + Clone,
{
    // Apply Tikhonov regularization
    let regularization_param = report
        .condition_report
        .recommended_regularization
        .unwrap_or_else(|| {
            super::types::machine_epsilon::<F>()
                * F::from(1000.0).unwrap_or_else(|| F::from(1000.0).unwrap())
        });

    let regularized_matrix =
        super::regularization::apply_tikhonov_regularization(matrix, regularization_param)?;

    // Solve regularized system
    solve_direct_lu(&regularized_matrix.view(), rhs, report)
}

/// Determine optimal solving strategy based on matrix properties
fn determine_solve_strategy<F>(
    condition_report: &ConditionReport<F>,
    edge_case_report: &EdgeCaseReport<F>,
) -> SolveStrategy
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    // If regularization is recommended, use regularized solve
    if condition_report.recommended_regularization.is_some() {
        return SolveStrategy::Regularized;
    }

    // If nearly singular, use regularized approach
    if edge_case_report.is_nearly_singular {
        return SolveStrategy::Regularized;
    }

    // For well-conditioned symmetric positive definite matrices, use CG
    if condition_report.stability_level == StabilityLevel::Excellent
        && condition_report.diagnostics.is_symmetric
        && condition_report.diagnostics.is_positive_definite == Some(true)
    {
        return SolveStrategy::IterativeCG;
    }

    // For good stability, use direct LU
    if matches!(
        condition_report.stability_level,
        StabilityLevel::Excellent | StabilityLevel::Good
    ) {
        return SolveStrategy::DirectLU;
    }

    // For marginal stability, use QR (more stable than LU)
    if condition_report.stability_level == StabilityLevel::Marginal {
        return SolveStrategy::DirectQR;
    }

    // For poor stability, use iterative methods
    SolveStrategy::IterativeGMRES
}

/// Create convergence information based on matrix properties
fn create_convergence_info<F>(
    condition_report: &ConditionReport<F>,
    strategy: SolveStrategy,
) -> ConvergenceInfo<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let condition_number = condition_report.condition_number;

    // Estimate iterations based on condition number and method
    let base_iterations = match strategy {
        SolveStrategy::IterativeCG => {
            // CG convergence depends on sqrt(condition number)
            let sqrt_cond = condition_number.sqrt();
            (sqrt_cond.ln() * F::from(10.0).unwrap_or_else(|| F::from(10.0).unwrap()))
                .ceil()
                .to_usize()
                .unwrap_or(50)
        }
        SolveStrategy::IterativeGMRES => {
            // GMRES may need more iterations
            (condition_number.ln() * F::from(5.0).unwrap_or_else(|| F::from(5.0).unwrap()))
                .ceil()
                .to_usize()
                .unwrap_or(100)
        }
        _ => 1, // Direct methods
    };

    let expected_iterations = base_iterations.min(1000).max(1);

    // Set tolerance based on condition number
    let recommended_tolerance = condition_report.diagnostics.machine_epsilon
        * condition_number.sqrt()
        * F::from(100.0).unwrap_or_else(|| F::from(100.0).unwrap());

    // Recommend preconditioning for iterative methods with poor conditioning
    let needs_preconditioning = matches!(
        strategy,
        SolveStrategy::IterativeCG | SolveStrategy::IterativeGMRES
    ) && condition_number
        > F::from(1e10).unwrap_or_else(|| F::from(1e10).unwrap());

    ConvergenceInfo {
        expected_iterations,
        recommended_tolerance,
        needs_preconditioning,
    }
}

/// LU decomposition with partial pivoting
fn lu_decomposition_with_pivoting<F>(
    matrix: &ArrayView2<F>,
) -> InterpolateResult<(Array2<F>, Vec<usize>)>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + Clone,
{
    let n = matrix.nrows();
    let mut lu = matrix.to_owned();
    let mut permutation = (0..n).collect::<Vec<_>>();

    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        let mut max_val = lu[(k, k)].abs();
        for i in (k + 1)..n {
            let val = lu[(i, k)].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        // Check for singular matrix
        if max_val < super::types::machine_epsilon::<F>() {
            return Err(InterpolateError::NumericalInstability {
                message: "Matrix is singular or nearly singular".to_string(),
            });
        }

        // Swap rows if needed
        if max_row != k {
            for j in 0..n {
                let temp = lu[(k, j)];
                lu[(k, j)] = lu[(max_row, j)];
                lu[(max_row, j)] = temp;
            }
            permutation.swap(k, max_row);
        }

        // Elimination
        for i in (k + 1)..n {
            let factor = lu[(i, k)] / lu[(k, k)];
            lu[(i, k)] = factor; // Store L factor

            for j in (k + 1)..n {
                let kj_value = lu[(k, j)];
                lu[(i, j)] -= factor * kj_value;
            }
        }
    }

    Ok((lu, permutation))
}

/// Solve using LU factors and permutation
fn solve_with_lu_factors<F>(
    lu_factors: &ArrayView2<F>,
    permutation: &[usize],
    rhs: &ArrayView1<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + Clone,
{
    let n = lu_factors.nrows();

    // Apply permutation to RHS
    let mut pb = Array1::zeros(n);
    for i in 0..n {
        pb[i] = rhs[permutation[i]];
    }

    // Forward substitution (solve Ly = Pb)
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut sum = pb[i];
        for j in 0..i {
            sum -= lu_factors[(i, j)] * y[j];
        }
        y[i] = sum; // L has unit diagonal
    }

    // Back substitution (solve Ux = y)
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= lu_factors[(i, j)] * x[j];
        }

        let diagonal = lu_factors[(i, i)];
        if diagonal.abs() < super::types::machine_epsilon::<F>() {
            return Err(InterpolateError::NumericalInstability {
                message: format!("Zero diagonal element at position {}", i),
            });
        }

        x[i] = sum / diagonal;
    }

    Ok(x)
}

/// Simplified QR decomposition
fn qr_decomposition<F>(matrix: &ArrayView2<F>) -> InterpolateResult<(Array2<F>, Array2<F>)>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + Clone
        + std::ops::DivAssign,
{
    let (m, n) = matrix.dim();
    let mut q = Array2::zeros((m, n));
    let mut r = Array2::zeros((n, n));

    // Modified Gram-Schmidt
    for j in 0..n {
        // Copy column j of A to column j of Q
        for i in 0..m {
            q[(i, j)] = matrix[(i, j)];
        }

        // Orthogonalize against previous columns
        for i in 0..j {
            // Compute R[i,j] = Q[:,i]^T * Q[:,j]
            let mut dot = F::zero();
            for k in 0..m {
                dot += q[(k, i)] * q[(k, j)];
            }
            r[(i, j)] = dot;

            // Q[:,j] = Q[:,j] - R[i,j] * Q[:,i]
            for k in 0..m {
                let q_ki = q[(k, i)];
                q[(k, j)] -= r[(i, j)] * q_ki;
            }
        }

        // Normalize column j
        let mut norm = F::zero();
        for k in 0..m {
            norm += q[(k, j)] * q[(k, j)];
        }
        norm = norm.sqrt();

        if norm < super::types::machine_epsilon::<F>() {
            return Err(InterpolateError::NumericalInstability {
                message: format!("Zero norm in QR decomposition at column {}", j),
            });
        }

        r[(j, j)] = norm;
        for k in 0..m {
            q[(k, j)] /= norm;
        }
    }

    Ok((q, r))
}

/// Multiply Q^T with a vector
fn multiply_qt_vector<F>(q: &ArrayView2<F>, vector: &ArrayView1<F>) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + AddAssign,
{
    let (m, n) = q.dim();
    if m != vector.len() {
        return Err(InterpolateError::ShapeMismatch {
            expected: format!("{} elements", m),
            actual: format!("{} elements", vector.len()),
            object: "Q^T vector multiplication".to_string(),
        });
    }

    let mut result = Array1::zeros(n);
    for j in 0..n {
        for i in 0..m {
            result[j] += q[(i, j)] * vector[i];
        }
    }

    Ok(result)
}

/// Solve upper triangular system
fn solve_upper_triangular<F>(
    upper: &ArrayView2<F>,
    rhs: &ArrayView1<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let n = upper.nrows();
    let mut x = Array1::zeros(n);

    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum -= upper[(i, j)] * x[j];
        }

        let diagonal = upper[(i, i)];
        if diagonal.abs() < super::types::machine_epsilon::<F>() {
            return Err(InterpolateError::NumericalInstability {
                message: format!("Zero diagonal element at position {}", i),
            });
        }

        x[i] = sum / diagonal;
    }

    Ok(x)
}

/// Matrix-vector multiplication
fn matrix_vector_multiply<F>(
    matrix: &ArrayView2<F>,
    vector: &ArrayView1<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + AddAssign,
{
    let (m, n) = matrix.dim();
    if n != vector.len() {
        return Err(InterpolateError::ShapeMismatch {
            expected: format!("{} elements", n),
            actual: format!("{} elements", vector.len()),
            object: "matrix-vector multiplication".to_string(),
        });
    }

    let mut result = Array1::zeros(m);
    for i in 0..m {
        for j in 0..n {
            result[i] += matrix[(i, j)] * vector[j];
        }
    }

    Ok(result)
}

/// Compute dot product of two vectors
fn dot_product<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> F
where
    F: Float + AddAssign,
{
    let mut result = F::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        result += *x * *y;
    }
    result
}

/// Compute vector norm
fn vector_norm<F>(vector: &ArrayView1<F>) -> F
where
    F: Float + AddAssign,
{
    dot_product(vector, vector).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical_stability_modules::{EdgeCaseReport, StabilityDiagnostics};
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_solve_well_conditioned_system() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 3.0]).unwrap();
        let rhs = Array1::from_vec(vec![1.0, 2.0]);

        let (solution, report) =
            solve_with_enhanced_monitoring(&matrix.view(), &rhs.view()).unwrap();

        // Verify solution: Ax should equal b
        let verification = matrix_vector_multiply(&matrix.view(), &solution.view()).unwrap();
        for i in 0..rhs.len() {
            assert!((verification[i] - rhs[i]).abs() < 1e-10);
        }

        assert!(report.condition_report.is_well_conditioned);
        assert_eq!(report.recommended_strategy, SolveStrategy::DirectLU);
    }

    #[test]
    fn test_lu_decomposition() {
        let matrix =
            Array2::from_shape_vec((3, 3), vec![2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 0.0])
                .unwrap();

        let (lu, perm) = lu_decomposition_with_pivoting(&matrix.view()).unwrap();

        // Verify dimensions
        assert_eq!(lu.nrows(), 3);
        assert_eq!(lu.ncols(), 3);
        assert_eq!(perm.len(), 3);
    }

    #[test]
    fn test_qr_decomposition() {
        let matrix =
            Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
                .unwrap();

        let (q, r) = qr_decomposition(&matrix.view()).unwrap();

        // For identity matrix, Q should be identity and R should be identity
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!((q[(i, j)] - 1.0).abs() < 1e-10);
                    assert!((r[(i, j)] - 1.0).abs() < 1e-10);
                } else {
                    assert!(q[(i, j)].abs() < 1e-10);
                    assert!(r[(i, j)].abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_iterative_cg() {
        // Test CG on a simple SPD system
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).unwrap();
        let rhs = Array1::from_vec(vec![4.0, 6.0]);

        let mut report = EnhancedStabilityReport {
            condition_report: ConditionReport {
                condition_number: 1.0,
                is_well_conditioned: true,
                recommended_regularization: None,
                stability_level: StabilityLevel::Excellent,
                diagnostics: StabilityDiagnostics::default(),
            },
            edge_case_report: EdgeCaseReport::default(),
            recommended_strategy: SolveStrategy::IterativeCG,
            convergence_info: ConvergenceInfo {
                expected_iterations: 10,
                recommended_tolerance: 1e-10,
                needs_preconditioning: false,
            },
            needs_iterative_refinement: false,
        };

        let solution = solve_iterative_cg(&matrix.view(), &rhs.view(), &report).unwrap();

        // Expected solution: [2.0, 3.0]
        assert!((solution[0] - 2.0).abs() < 1e-6);
        assert!((solution[1] - 3.0).abs() < 1e-6);
    }
}
