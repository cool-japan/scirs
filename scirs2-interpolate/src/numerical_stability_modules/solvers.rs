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

/// Minimum Krylov subspace (restart) dimension used by GMRES, regardless of
/// the condition-number-based iteration estimate. Keeps small-to-moderate
/// systems from restarting more often than necessary.
const GMRES_MIN_RESTART_DIM: usize = 5;

/// Maximum Krylov subspace (restart) dimension. Bounds the O(restart_dim * n)
/// memory and O(restart_dim^2 * n) per-cycle Arnoldi cost for large systems.
const GMRES_MAX_RESTART_DIM: usize = 50;

/// Minimum number of restart cycles attempted before giving up, even when
/// the condition-number-based iteration estimate would suggest fewer.
const GMRES_MIN_RESTART_CYCLES: usize = 5;

/// Hard safety cap on the total number of Krylov (Arnoldi) iterations across
/// all restart cycles combined, bounding worst-case runtime.
const GMRES_MAX_TOTAL_ITERATIONS: usize = 2000;

/// Solve using the restarted GMRES(m) (Generalized Minimal RESidual) method.
///
/// This is the classical Saad-Schultz algorithm:
/// 1. An Arnoldi process builds an orthonormal Krylov basis together with
///    the associated upper Hessenberg matrix.
/// 2. Givens rotations incrementally reduce the Hessenberg matrix to upper
///    triangular form, which both solves the least-squares problem for the
///    current cycle and gives the residual norm at every step for free.
/// 3. Once the Krylov subspace reaches its restart dimension (or the
///    process breaks down / converges early), the small triangular system
///    is solved and the approximate solution is updated; the cycle then
///    restarts from the new residual.
/// 4. Convergence is checked against the true (recomputed) residual norm,
///    both between cycles and once the iteration budget is exhausted.
fn solve_iterative_gmres<F>(
    matrix: &ArrayView2<F>,
    rhs: &ArrayView1<F>,
    report: &EnhancedStabilityReport<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + Clone,
{
    let n = matrix.nrows();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let tolerance = report.convergence_info.recommended_tolerance;
    let eps = super::types::machine_epsilon::<F>();

    // Krylov subspace dimension for each restart cycle.
    let restart_dim = report
        .convergence_info
        .expected_iterations
        .clamp(GMRES_MIN_RESTART_DIM, GMRES_MAX_RESTART_DIM)
        .min(n);

    // Number of restart cycles: enough to cover the condition-number-based
    // iteration estimate (and at least a handful of full cycles), capped by
    // an absolute safety ceiling on total Krylov iterations.
    let max_restarts = report
        .convergence_info
        .expected_iterations
        .max(n)
        .div_ceil(restart_dim)
        .max(GMRES_MIN_RESTART_CYCLES)
        .min(GMRES_MAX_TOTAL_ITERATIONS.div_ceil(restart_dim));

    let mut x = Array1::<F>::zeros(n);
    let mut total_iterations = 0usize;

    for _cycle in 0..max_restarts {
        // True residual r0 = b - A*x
        let ax = matrix_vector_multiply(matrix, &x.view())?;
        let mut residual = Array1::<F>::zeros(n);
        for i in 0..n {
            residual[i] = rhs[i] - ax[i];
        }
        let beta = vector_norm(&residual.view());

        if beta < tolerance {
            return Ok(x);
        }

        // Orthonormal Krylov basis q_0, q_1, ... built by the Arnoldi process.
        let mut basis: Vec<Array1<F>> = Vec::with_capacity(restart_dim + 1);
        basis.push(residual.mapv(|v| v / beta));

        // Upper Hessenberg matrix produced by the Arnoldi process.
        let mut hessenberg = Array2::<F>::zeros((restart_dim + 1, restart_dim));
        // Accumulated Givens rotation coefficients (one pair per column).
        let mut cos_rot = vec![F::zero(); restart_dim];
        let mut sin_rot = vec![F::zero(); restart_dim];
        // Right-hand side of the incrementally rotated least-squares problem.
        let mut g = Array1::<F>::zeros(restart_dim + 1);
        g[0] = beta;

        let mut steps_used = 0usize;

        for k in 0..restart_dim {
            total_iterations += 1;

            // Arnoldi step: w = A * q_k, orthogonalized (modified
            // Gram-Schmidt) against every previously built basis vector.
            let mut w = matrix_vector_multiply(matrix, &basis[k].view())?;
            for i in 0..=k {
                let h_ik = dot_product(&w.view(), &basis[i].view());
                hessenberg[(i, k)] = h_ik;
                for idx in 0..n {
                    w[idx] -= h_ik * basis[i][idx];
                }
            }
            let h_next = vector_norm(&w.view());
            hessenberg[(k + 1, k)] = h_next;

            // Apply the previously accumulated Givens rotations to the new
            // Hessenberg column.
            for i in 0..k {
                let temp = cos_rot[i] * hessenberg[(i, k)] + sin_rot[i] * hessenberg[(i + 1, k)];
                hessenberg[(i + 1, k)] =
                    -sin_rot[i] * hessenberg[(i, k)] + cos_rot[i] * hessenberg[(i + 1, k)];
                hessenberg[(i, k)] = temp;
            }

            // New Givens rotation eliminating the sub-diagonal entry.
            let (c_k, s_k) = givens_rotation(hessenberg[(k, k)], hessenberg[(k + 1, k)]);
            cos_rot[k] = c_k;
            sin_rot[k] = s_k;
            hessenberg[(k, k)] = c_k * hessenberg[(k, k)] + s_k * hessenberg[(k + 1, k)];
            hessenberg[(k + 1, k)] = F::zero();

            let g_k = g[k];
            g[k] = c_k * g_k;
            g[k + 1] = -s_k * g_k;

            steps_used = k + 1;

            // Lucky breakdown: the Krylov subspace already spans the exact
            // solution direction, so there is no orthogonal component left
            // to explore (this is a successful termination, not a failure).
            let breakdown = h_next <= eps * (F::one() + beta);
            let residual_estimate = g[k + 1].abs();

            if breakdown || residual_estimate < tolerance {
                break;
            }

            if k + 1 < restart_dim {
                basis.push(w.mapv(|v| v / h_next));
            }
        }

        // Solve the small upper-triangular least-squares system and fold the
        // correction into the current approximate solution.
        let y = solve_hessenberg_triangular(&hessenberg, &g, steps_used)?;
        for (j, y_j) in y.iter().enumerate() {
            for i in 0..n {
                x[i] += *y_j * basis[j][i];
            }
        }
    }

    // GMRES is a numerical method: restarted GMRES(m) with m < n is not
    // guaranteed to converge for every nonsingular matrix (unlike full,
    // unrestricted GMRES). Verify the true residual instead of silently
    // returning a solution that never actually met the requested tolerance.
    let ax = matrix_vector_multiply(matrix, &x.view())?;
    let mut final_residual_vec = Array1::<F>::zeros(n);
    for i in 0..n {
        final_residual_vec[i] = rhs[i] - ax[i];
    }
    let final_residual = vector_norm(&final_residual_vec.view());

    if final_residual < tolerance {
        Ok(x)
    } else {
        Err(InterpolateError::NumericalInstability {
            message: format!(
                "GMRES({restart_dim}) failed to converge within {max_restarts} restart cycle(s) \
                 ({total_iterations} total Krylov iterations): residual norm {final_residual} \
                 exceeds tolerance {tolerance}"
            ),
        })
    }
}

/// Compute a numerically stable Givens rotation `(c, s)` such that
/// `c * a + s * b = sqrt(a^2 + b^2)` and `-s * a + c * b = 0`.
fn givens_rotation<F>(a: F, b: F) -> (F, F)
where
    F: Float,
{
    if b == F::zero() {
        (F::one(), F::zero())
    } else if b.abs() > a.abs() {
        let tau = a / b;
        let s = F::one() / (F::one() + tau * tau).sqrt();
        let c = s * tau;
        (c, s)
    } else {
        let tau = b / a;
        let c = F::one() / (F::one() + tau * tau).sqrt();
        let s = c * tau;
        (c, s)
    }
}

/// Back-substitute the upper-triangular system formed by the first `k`
/// rows/columns of a (possibly larger) Hessenberg-derived matrix against the
/// first `k` entries of `g`, as produced by [`solve_iterative_gmres`].
fn solve_hessenberg_triangular<F>(
    hessenberg: &Array2<F>,
    g: &Array1<F>,
    k: usize,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let mut y = Array1::<F>::zeros(k);
    for i in (0..k).rev() {
        let mut sum = g[i];
        for j in (i + 1)..k {
            sum -= hessenberg[(i, j)] * y[j];
        }

        let diagonal = hessenberg[(i, i)];
        if diagonal.abs() < super::types::machine_epsilon::<F>() {
            return Err(InterpolateError::NumericalInstability {
                message: format!("GMRES: zero pivot in Hessenberg system at position {i}"),
            });
        }

        y[i] = sum / diagonal;
    }

    Ok(y)
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
                * F::from(1000.0).unwrap_or_else(|| {
                    F::from(1000.0).expect("Failed to convert constant to float")
                })
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
            (sqrt_cond.ln()
                * F::from(10.0)
                    .unwrap_or_else(|| F::from(10.0).expect("Failed to convert constant to float")))
            .ceil()
            .to_usize()
            .unwrap_or(50)
        }
        SolveStrategy::IterativeGMRES => {
            // GMRES may need more iterations
            (condition_number.ln()
                * F::from(5.0)
                    .unwrap_or_else(|| F::from(5.0).expect("Failed to convert constant to float")))
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
        * F::from(100.0)
            .unwrap_or_else(|| F::from(100.0).expect("Failed to convert constant to float"));

    // Recommend preconditioning for iterative methods with poor conditioning
    let needs_preconditioning = matches!(
        strategy,
        SolveStrategy::IterativeCG | SolveStrategy::IterativeGMRES
    ) && condition_number
        > F::from(1e10)
            .unwrap_or_else(|| F::from(1e10).expect("Failed to convert constant to float"));

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
        let matrix =
            Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 3.0]).expect("Operation failed");
        let rhs = Array1::from_vec(vec![1.0, 2.0]);

        let (solution, report) =
            solve_with_enhanced_monitoring(&matrix.view(), &rhs.view()).expect("Operation failed");

        // Verify solution: Ax should equal b
        let verification =
            matrix_vector_multiply(&matrix.view(), &solution.view()).expect("Operation failed");
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
                .expect("Operation failed");

        let (lu, perm) = lu_decomposition_with_pivoting(&matrix.view()).expect("Operation failed");

        // Verify dimensions
        assert_eq!(lu.nrows(), 3);
        assert_eq!(lu.ncols(), 3);
        assert_eq!(perm.len(), 3);
    }

    #[test]
    fn test_qr_decomposition() {
        let matrix =
            Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
                .expect("Operation failed");

        let (q, r) = qr_decomposition(&matrix.view()).expect("Operation failed");

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
        let matrix =
            Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).expect("Operation failed");
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

        let solution =
            solve_iterative_cg(&matrix.view(), &rhs.view(), &report).expect("Operation failed");

        // Expected solution: [2.0, 3.0]
        assert!((solution[0] - 2.0).abs() < 1e-6);
        assert!((solution[1] - 3.0).abs() < 1e-6);
    }

    /// Build the 6x6 nonsymmetric, strongly diagonally-dominant (hence
    /// well-conditioned) tridiagonal test matrix shared by the GMRES tests
    /// below: diagonal 30, superdiagonal 3, subdiagonal -2.
    ///
    /// This matrix is deliberately adversarial for a *fixed step-size*
    /// Richardson iteration with alpha = 0.1 (which is what the previous
    /// "GMRES placeholder" actually computed): its eigenvalues cluster
    /// around 30, so `1 - alpha * eig ~ 1 - 3 = -2`, giving a spectral
    /// radius of about 2 for the Richardson iteration matrix and causing it
    /// to diverge to the order of 1e17-1e18 within 50 iterations instead of
    /// converging. Real GMRES, in contrast, must converge to machine
    /// precision within at most n = 6 Krylov steps (in exact arithmetic).
    fn gmres_test_matrix() -> Array2<f64> {
        let n = 6;
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 30.0;
            if i + 1 < n {
                data[i * n + (i + 1)] = 3.0;
            }
            if i > 0 {
                data[i * n + (i - 1)] = -2.0;
            }
        }
        Array2::from_shape_vec((n, n), data).expect("Operation failed")
    }

    fn gmres_test_report(expected_iterations: usize) -> EnhancedStabilityReport<f64> {
        EnhancedStabilityReport {
            condition_report: ConditionReport {
                condition_number: 10.0,
                is_well_conditioned: true,
                recommended_regularization: None,
                stability_level: StabilityLevel::Poor,
                diagnostics: StabilityDiagnostics::default(),
            },
            edge_case_report: EdgeCaseReport::default(),
            recommended_strategy: SolveStrategy::IterativeGMRES,
            convergence_info: ConvergenceInfo {
                expected_iterations,
                recommended_tolerance: 1e-10,
                needs_preconditioning: false,
            },
            needs_iterative_refinement: false,
        }
    }

    #[test]
    fn test_gmres_nonsymmetric_well_conditioned_converges() {
        // Non-constant right-hand side / solution (not all-ones), so the
        // test can actually distinguish a real solve from a fabricated one.
        let matrix = gmres_test_matrix();
        let x_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = matrix_vector_multiply(&matrix.view(), &x_true.view()).expect("Operation failed");

        // expected_iterations = 50 clamps the restart dimension to n = 6,
        // i.e. a single, unrestarted GMRES cycle.
        let report = gmres_test_report(50);

        let solution = solve_iterative_gmres(&matrix.view(), &rhs.view(), &report)
            .expect("real GMRES must converge on a well-conditioned nonsymmetric system");

        // The old placeholder (fixed alpha = 0.1 Richardson iteration)
        // diverges on this matrix to residuals on the order of 1e17-1e18
        // within its 50-iteration cap; real GMRES must land far below the
        // requested tolerance instead.
        let verification =
            matrix_vector_multiply(&matrix.view(), &solution.view()).expect("Operation failed");
        for i in 0..rhs.len() {
            assert!(
                (verification[i] - rhs[i]).abs() < 1e-8,
                "residual too large at index {i}: {} vs {}",
                verification[i],
                rhs[i]
            );
        }
        for i in 0..x_true.len() {
            assert!(
                (solution[i] - x_true[i]).abs() < 1e-6,
                "component {i}: got {}, expected {}",
                solution[i],
                x_true[i]
            );
        }
    }

    #[test]
    fn test_gmres_restart_cycle_reaches_convergence() {
        // Same well-conditioned nonsymmetric system, but expected_iterations
        // = 5 clamps the Krylov restart dimension to 5 < n = 6, forcing at
        // least one actual restart cycle (not just a single Arnoldi pass)
        // to reach the requested tolerance. This exercises the restart
        // logic itself, which the old placeholder never had at all.
        let matrix = gmres_test_matrix();
        let x_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = matrix_vector_multiply(&matrix.view(), &x_true.view()).expect("Operation failed");

        let report = gmres_test_report(5);

        let solution = solve_iterative_gmres(&matrix.view(), &rhs.view(), &report)
            .expect("restarted GMRES(5) must still converge on this well-conditioned system");

        let verification =
            matrix_vector_multiply(&matrix.view(), &solution.view()).expect("Operation failed");
        for i in 0..rhs.len() {
            assert!(
                (verification[i] - rhs[i]).abs() < 1e-8,
                "residual too large at index {i}: {} vs {}",
                verification[i],
                rhs[i]
            );
        }
    }

    #[test]
    fn test_gmres_via_stability_monitoring_dispatch() {
        // Exercise the same `solve_with_strategy` dispatch that
        // `solve_with_enhanced_monitoring` / `solve_with_stability_monitoring`
        // (the crate-root-exported entry point) use internally, with
        // `recommended_strategy` explicitly set to `IterativeGMRES`. This
        // confirms the dispatch match arm actually reaches the real GMRES
        // implementation and returns an accurate solution end to end.
        let matrix = gmres_test_matrix();
        let x_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = matrix_vector_multiply(&matrix.view(), &x_true.view()).expect("Operation failed");

        let report = gmres_test_report(50);
        let solution = solve_with_strategy(&matrix.view(), &rhs.view(), &report)
            .expect("dispatch to IterativeGMRES must succeed and converge");

        let verification =
            matrix_vector_multiply(&matrix.view(), &solution.view()).expect("Operation failed");
        for i in 0..rhs.len() {
            assert!((verification[i] - rhs[i]).abs() < 1e-8);
        }
    }

    #[test]
    fn test_givens_rotation_zeroes_second_component() {
        let (c, s) = givens_rotation(3.0_f64, 4.0);
        // Applying [c s; -s c] to [a, b] must zero the second component and
        // produce the Euclidean norm in the first, with (c, s) on the unit
        // circle.
        let r = c * 3.0 + s * 4.0;
        let zeroed = -s * 3.0 + c * 4.0;
        assert!((r - 5.0).abs() < 1e-12);
        assert!(zeroed.abs() < 1e-12);
        assert!((c * c + s * s - 1.0).abs() < 1e-12);

        // Degenerate case: b already zero.
        let (c0, s0) = givens_rotation(7.0_f64, 0.0);
        assert!((c0 - 1.0).abs() < 1e-12);
        assert!(s0.abs() < 1e-12);
    }
}
