//! Advanced iterative eigenvalue methods with extended precision
//!
//! This module contains advanced iterative algorithms for eigenvalue computation
//! including Rayleigh quotient iteration, Newton corrections, and other
//! high-precision numerical methods.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, One, Zero};

use super::super::{DemotableTo, PromotableTo};
use super::standard_eigen::extended_eigh;
use crate::error::LinalgResult;

/// Advanced-precision eigenvalue solver targeting 1e-12+ accuracy (advanced mode)
///
/// This function implements state-of-the-art numerical techniques for achieving
/// advanced-high precision eigenvalue computation, including:
/// - Kahan summation for compensated arithmetic
/// - Multiple-stage Rayleigh quotient iteration
/// - Newton's method eigenvalue correction
/// - Advanced-aggressive adaptive tolerance based on matrix conditioning
/// - Enhanced Gram-Schmidt orthogonalization
/// - Automatic advanced-precision activation for high precision targets
///
/// # Parameters
///
/// * `a` - Input symmetric matrix
/// * `max_iter` - Maximum number of iterations (default: 500)
/// * `target_precision` - Target precision (default: 1e-12, advanced mode enhancement)
/// * `auto_detect` - Automatically activate advanced-precision for challenging matrices
///
/// # Returns
///
/// * Tuple containing (eigenvalues, eigenvectors) with advanced-high precision
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::extended_precision::eigen::advanced_precision_eigh;
///
/// let a = array![[2.0f32, 1.0], [1.0, 2.0]];
/// let (eigvals, eigvecs) = advanced_precision_eigh::<_, f64>(&a.view(), None, None, true).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn advanced_precision_eigh<A, I>(
    a: &ArrayView2<A>,
    max_iter: Option<usize>,
    target_precision: Option<A>,
    auto_detect: bool,
) -> LinalgResult<(Array1<A>, Array2<A>)>
where
    A: Float
        + Zero
        + One
        + PromotableTo<I>
        + DemotableTo<A>
        + Copy
        + std::fmt::Debug
        + std::ops::AddAssign,
    I: Float
        + Zero
        + One
        + DemotableTo<A>
        + Copy
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::DivAssign
        + 'static,
{
    if a.nrows() != a.ncols() {
        return Err(crate::error::LinalgError::ShapeError(format!(
            "Expected square matrix, got shape {:?}",
            a.shape()
        )));
    }

    let _n = a.nrows();
    let max_iter = max_iter.unwrap_or(500);
    let target_precision = target_precision.unwrap_or(A::from(1e-12).expect("Operation failed"));

    // Compute matrix condition number for adaptive tolerance selection
    let condition_number = estimate_condition_number(a)?;

    // Advanced-aggressive adaptive tolerance selection for 1e-12+ accuracy
    let adaptive_tolerance = if condition_number > A::from(1e12).expect("Operation failed") {
        target_precision * A::from(100.0).expect("Operation failed") // Relax tolerance for ill-conditioned matrices
    } else if condition_number < A::from(1e3).expect("Operation failed") {
        target_precision * A::from(0.01).expect("Operation failed") // Advanced-tight tolerance for extremely well-conditioned matrices
    } else if condition_number < A::from(1e6).expect("Operation failed") {
        target_precision * A::from(0.1).expect("Operation failed") // Tighter tolerance for well-conditioned matrices
    } else {
        target_precision
    };

    // Auto-detect if advanced-precision mode should be activated (more aggressive in advanced mode)
    let use_advanced_precision = auto_detect
        && (
            condition_number > A::from(1e12).expect("Operation failed")
                || target_precision <= A::from(1e-11).expect("Operation failed")
            // Activate for high precision targets
        );

    if use_advanced_precision {
        advanced_precision_solver_internal::<A, I>(a, max_iter, adaptive_tolerance)
    } else {
        // Use standard extended precision for well-conditioned matrices
        extended_eigh(a, Some(max_iter), Some(adaptive_tolerance))
    }
}

/// Internal advanced-precision solver with advanced numerical techniques.
///
/// The base eigendecomposition is delegated to [`extended_eigh`] (Householder
/// tridiagonalization plus implicit-shift QL with deflation, already proven
/// correct and convergence-checked) rather than a from-scratch
/// Rayleigh-quotient-iteration pipeline: the previous from-scratch pipeline's
/// QR step (`apply_qr_step_with_shift`) never updated the off-diagonal array
/// or the eigenvector accumulator (only nudged two diagonal entries by a
/// tiny fraction of the shift), so it never actually converged the
/// tridiagonal form for non-trivial matrices. On top of that proven base,
/// this function applies genuine additional advanced-precision refinement:
/// Newton's method eigenvalue correction against the real characteristic
/// polynomial (evaluated via a genuine LU-based determinant, not the
/// previous placeholder that always returned `matrix[[0, 0]]` for `n > 2`
/// and collapsed every eigenvalue estimate onto `A[[0, 0]]`), followed by
/// re-orthogonalization and a genuine inverse-iteration eigenvector
/// refinement pass.
#[allow(dead_code)]
fn advanced_precision_solver_internal<A, I>(
    a: &ArrayView2<A>,
    max_iter: usize,
    tolerance: A,
) -> LinalgResult<(Array1<A>, Array2<A>)>
where
    A: Float
        + Zero
        + One
        + PromotableTo<I>
        + DemotableTo<A>
        + Copy
        + std::fmt::Debug
        + std::ops::AddAssign,
    I: Float
        + Zero
        + One
        + DemotableTo<A>
        + Copy
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::DivAssign
        + 'static,
{
    let a_work = a.to_owned();

    // Real, proven, convergence-checked base solver.
    let (mut d, mut q) = extended_eigh::<A, I>(a, Some(max_iter), Some(tolerance))?;

    // Newton's method eigenvalue correction using the real characteristic
    // polynomial as an additional advanced-precision refinement pass.
    newton_eigenvalue_correction(&mut d, &a_work, tolerance)?;

    // Re-orthogonalize after the (small) Newton-driven eigenvalue nudge.
    enhanced_gram_schmidt_orthogonalization(&mut q, 3)?;

    // Final residual verification and (real) inverse-iteration refinement.
    final_residual_verification(&mut d, &mut q, &a_work, tolerance)?;

    Ok((d, q))
}

/// Newton's method eigenvalue correction for final accuracy verification
#[allow(dead_code)]
fn newton_eigenvalue_correction<A>(
    eigenvalues: &mut Array1<A>,
    originalmatrix: &Array2<A>,
    tolerance: A,
) -> LinalgResult<()>
where
    A: Float + Zero + One + Copy,
{
    let n = eigenvalues.len();

    for i in 0..n {
        let mut lambda = eigenvalues[i];

        for _ in 0..10 {
            // Maximum 10 Newton iterations
            // Compute f(lambda) = det(A - lambda*I) and f'(lambda)
            let f_val = compute_characteristic_polynomial_value(originalmatrix, lambda)?;
            let f_prime = compute_characteristic_polynomial_derivative(originalmatrix, lambda)?;

            if f_prime.abs() < A::epsilon() {
                break; // Avoid division by zero
            }

            let delta = f_val / f_prime;
            lambda = lambda - delta;

            if delta.abs() < tolerance {
                break;
            }
        }

        eigenvalues[i] = lambda;
    }

    Ok(())
}

/// Compute characteristic polynomial value at lambda
#[allow(dead_code)]
fn compute_characteristic_polynomial_value<A>(matrix: &Array2<A>, lambda: A) -> LinalgResult<A>
where
    A: Float + Zero + One + Copy,
{
    let n = matrix.nrows();
    let mut a_shifted = matrix.clone();

    // Compute A - lambda*I
    for i in 0..n {
        a_shifted[[i, i]] = a_shifted[[i, i]] - lambda;
    }

    Ok(compute_determinant(&a_shifted))
}

/// Compute characteristic polynomial derivative at lambda
#[allow(dead_code)]
fn compute_characteristic_polynomial_derivative<A>(matrix: &Array2<A>, lambda: A) -> LinalgResult<A>
where
    A: Float + Zero + One + Copy,
{
    // Numerical derivative approximation
    let h = A::from(1e-8).expect("Operation failed");
    let f_plus = compute_characteristic_polynomial_value(matrix, lambda + h)?;
    let f_minus = compute_characteristic_polynomial_value(matrix, lambda - h)?;

    Ok((f_plus - f_minus) / (A::from(2.0).expect("Operation failed") * h))
}

/// Determinant of a general square matrix via Gaussian elimination with
/// partial pivoting (`O(n^3)`, real for every `n`, unlike the previous
/// placeholder which returned `matrix[[0, 0]]` for any `n > 2`).
// `A` only has `Copy` here (not `NumAssign`), so `x = x op y` (rather than
// `x op= y`) is used deliberately throughout this function.
#[allow(dead_code, clippy::assign_op_pattern)]
fn compute_determinant<A>(matrix: &Array2<A>) -> A
where
    A: Float + Zero + One + Copy,
{
    let n = matrix.nrows();
    let mut m = matrix.clone();
    let mut det = A::one();

    for col in 0..n {
        // Partial pivoting: find the largest-magnitude entry in this column.
        let mut pivot_row = col;
        let mut pivot_val = m[[col, col]].abs();
        for row in (col + 1)..n {
            let val = m[[row, col]].abs();
            if val > pivot_val {
                pivot_val = val;
                pivot_row = row;
            }
        }

        if pivot_val <= A::epsilon() {
            // Singular (or numerically singular): determinant is zero.
            return A::zero();
        }

        if pivot_row != col {
            for k in 0..n {
                let tmp = m[[col, k]];
                m[[col, k]] = m[[pivot_row, k]];
                m[[pivot_row, k]] = tmp;
            }
            det = -det; // A row swap flips the sign of the determinant.
        }

        let pivot = m[[col, col]];
        det = det * pivot;

        for row in (col + 1)..n {
            let factor = m[[row, col]] / pivot;
            for k in col..n {
                m[[row, k]] = m[[row, k]] - factor * m[[col, k]];
            }
        }
    }

    det
}

/// Solve `a x = b` via Gaussian elimination with partial pivoting. Returns
/// `None` if `a` is (numerically) singular.
// `A` only has `Copy` here (not `NumAssign`), so `x = x op y` is used
// deliberately throughout this function.
#[allow(clippy::assign_op_pattern)]
fn solve_linear_system_partial_pivot<A>(a: &Array2<A>, b: &Array1<A>) -> Option<Array1<A>>
where
    A: Float + Zero + One + Copy,
{
    let n = a.nrows();
    let mut m = a.clone();
    let mut rhs = b.clone();

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_val = m[[col, col]].abs();
        for row in (col + 1)..n {
            let val = m[[row, col]].abs();
            if val > pivot_val {
                pivot_val = val;
                pivot_row = row;
            }
        }

        if pivot_val <= A::epsilon() {
            return None;
        }

        if pivot_row != col {
            for k in 0..n {
                let tmp = m[[col, k]];
                m[[col, k]] = m[[pivot_row, k]];
                m[[pivot_row, k]] = tmp;
            }
            let tmp = rhs[col];
            rhs[col] = rhs[pivot_row];
            rhs[pivot_row] = tmp;
        }

        let pivot = m[[col, col]];
        for row in (col + 1)..n {
            let factor = m[[row, col]] / pivot;
            for k in col..n {
                m[[row, k]] = m[[row, k]] - factor * m[[col, k]];
            }
            rhs[row] = rhs[row] - factor * rhs[col];
        }
    }

    let mut x = Array1::<A>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum = sum - m[[i, j]] * x[j];
        }
        x[i] = sum / m[[i, i]];
    }

    Some(x)
}

/// Enhanced Gram-Schmidt orthogonalization with multiple passes
#[allow(dead_code)]
fn enhanced_gram_schmidt_orthogonalization<A>(
    q: &mut Array2<A>,
    num_passes: usize,
) -> LinalgResult<()>
where
    A: Float + Zero + One + Copy + std::ops::AddAssign,
{
    let n = q.nrows();

    for _pass in 0..num_passes {
        for j in 0..n {
            // Normalize column j
            let mut norm_sq = A::zero();
            for i in 0..n {
                norm_sq += q[[i, j]] * q[[i, j]];
            }
            let norm = norm_sq.sqrt();

            if norm > A::epsilon() {
                for i in 0..n {
                    q[[i, j]] = q[[i, j]] / norm;
                }
            }

            // Orthogonalize against previous columns
            for k in 0..j {
                let mut dot_product = A::zero();
                for i in 0..n {
                    dot_product += q[[i, j]] * q[[i, k]];
                }

                for i in 0..n {
                    q[[i, j]] = q[[i, j]] - dot_product * q[[i, k]];
                }
            }
        }
    }

    Ok(())
}

/// Final residual verification and eigenvalue correction
#[allow(dead_code)]
fn final_residual_verification<A>(
    eigenvalues: &mut Array1<A>,
    eigenvectors: &mut Array2<A>,
    originalmatrix: &Array2<A>,
    tolerance: A,
) -> LinalgResult<()>
where
    A: Float + Zero + One + Copy + std::ops::AddAssign,
{
    let n = eigenvalues.len();

    for j in 0..n {
        let lambda = eigenvalues[j];
        let v = eigenvectors.column(j);

        // Compute residual: ||A*v - lambda*v||
        let mut residual = Array1::zeros(n);
        for i in 0..n {
            let mut av_i = A::zero();
            for k in 0..n {
                av_i += originalmatrix[[i, k]] * v[k];
            }
            residual[i] = av_i - lambda * v[i];
        }

        // Compute residual norm with Kahan summation
        let mut residual_norm_sq = A::zero();
        let mut c = A::zero();
        for &val in residual.iter() {
            let y = val * val - c;
            let t = residual_norm_sq + y;
            c = (t - residual_norm_sq) - y;
            residual_norm_sq = t;
        }

        let residual_norm = residual_norm_sq.sqrt();

        // If residual is too large, apply correction
        if residual_norm > tolerance {
            // Apply inverse iteration for eigenvector refinement
            inverse_iteration_refinement(eigenvectors, originalmatrix, eigenvalues[j], j)?;
        }
    }

    Ok(())
}

/// Inverse iteration for eigenvector refinement: solves
/// `(A - (lambda + eps) I) v_new = v_old` (via Gaussian elimination with
/// partial pivoting) and normalizes, which is the standard technique for
/// polishing an eigenvector once its eigenvalue is already an accurate
/// estimate. The tiny `eps` shift keeps the system solvable in floating
/// point while still concentrating the solve onto the eigenvector
/// direction. Leaves `eigenvectors` untouched if the shifted system is
/// (numerically) exactly singular, rather than corrupting the previous
/// estimate.
// `A` only has `Copy` here (not `NumAssign`), so `x = x op y` is used
// deliberately throughout this function.
#[allow(dead_code, clippy::assign_op_pattern)]
fn inverse_iteration_refinement<A>(
    eigenvectors: &mut Array2<A>,
    matrix: &Array2<A>,
    eigenvalue: A,
    col_index: usize,
) -> LinalgResult<()>
where
    A: Float + Zero + One + Copy,
{
    let n = matrix.nrows();
    let eps = A::epsilon() * (A::one() + eigenvalue.abs());
    let shift = eigenvalue + eps;

    let mut shifted = matrix.clone();
    for i in 0..n {
        shifted[[i, i]] = shifted[[i, i]] - shift;
    }

    let v_old: Array1<A> = eigenvectors.column(col_index).to_owned();

    if let Some(mut v_new) = solve_linear_system_partial_pivot(&shifted, &v_old) {
        let mut norm_sq = A::zero();
        for &val in v_new.iter() {
            norm_sq = norm_sq + val * val;
        }
        let norm = norm_sq.sqrt();
        if norm > A::epsilon() {
            for val in v_new.iter_mut() {
                *val = *val / norm;
            }

            // Keep the orientation closest to the previous estimate so
            // repeated refinement passes don't flip sign back and forth.
            let mut dot = A::zero();
            for i in 0..n {
                dot = dot + v_new[i] * v_old[i];
            }
            if dot < A::zero() {
                for val in v_new.iter_mut() {
                    *val = -*val;
                }
            }

            eigenvectors.column_mut(col_index).assign(&v_new);
        }
    }

    Ok(())
}

/// Estimate matrix condition number for adaptive tolerance selection
#[allow(dead_code)]
pub(super) fn estimate_condition_number<A>(matrix: &ArrayView2<A>) -> LinalgResult<A>
where
    A: Float + Zero + One + Copy + std::ops::AddAssign,
{
    // Simplified condition number estimation using matrix norm ratio
    // In practice, would use more sophisticated methods like SVD
    let n = matrix.nrows();

    // Estimate largest eigenvalue (matrix norm)
    let mut max_row_sum = A::zero();
    for i in 0..n {
        let mut row_sum = A::zero();
        for j in 0..n {
            row_sum += matrix[[i, j]].abs();
        }
        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }

    // Estimate smallest eigenvalue (simplified)
    let mut min_diagonal = matrix[[0, 0]].abs();
    for i in 1..n {
        let diag_val = matrix[[i, i]].abs();
        if diag_val < min_diagonal && diag_val > A::epsilon() {
            min_diagonal = diag_val;
        }
    }

    if min_diagonal > A::epsilon() {
        Ok(max_row_sum / min_diagonal)
    } else {
        Ok(A::from(1e15).expect("Operation failed")) // Large condition number for near-singular matrices
    }
}

/// Compute eigenvector using inverse iteration in extended precision
#[allow(dead_code)]
pub(super) fn compute_eigenvector_inverse_iteration<I>(
    shiftedmatrix: &Array2<scirs2_core::numeric::Complex<I>>,
    _lambda: scirs2_core::numeric::Complex<I>,
    max_iter: usize,
    tol: I,
) -> Array1<scirs2_core::numeric::Complex<I>>
where
    I: Float
        + Zero
        + One
        + Copy
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::DivAssign,
{
    let n = shiftedmatrix.nrows();

    // Start with a random vector
    let mut v = Array1::zeros(n);
    v[0] = scirs2_core::numeric::Complex::new(I::one(), I::zero());

    for _ in 0..max_iter {
        // Solve (A - λI)u = v for u using LU decomposition
        let mut u = solve_linear_system_complex(shiftedmatrix, &v);

        // Normalize u
        let norm = compute_complex_norm(&u);
        if norm > I::epsilon() {
            let norm_complex = scirs2_core::numeric::Complex::new(norm, I::zero());
            for i in 0..n {
                u[i] = u[i] / norm_complex;
            }
        }

        // Check convergence
        let mut diff = I::zero();
        for i in 0..n {
            let delta = (u[i] - v[i]).norm();
            diff += delta;
        }

        if diff < tol {
            return u;
        }

        v = u;
    }

    v
}

/// Solve complex linear system using simplified Gaussian elimination
#[allow(dead_code)]
fn solve_linear_system_complex<I>(
    a: &Array2<scirs2_core::numeric::Complex<I>>,
    b: &Array1<scirs2_core::numeric::Complex<I>>,
) -> Array1<scirs2_core::numeric::Complex<I>>
where
    I: Float + Zero + One + Copy + std::fmt::Debug,
{
    let n = a.nrows();
    let mut aug = Array2::zeros((n, n + 1));

    // Create augmented matrix
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        for i in k + 1..n {
            if aug[[i, k]].norm() > aug[[max_row, k]].norm() {
                max_row = i;
            }
        }

        // Swap rows
        if max_row != k {
            for j in 0..n + 1 {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Make diagonal elements 1
        let pivot = aug[[k, k]];
        if pivot.norm() > I::epsilon() {
            for j in k..n + 1 {
                aug[[k, j]] = aug[[k, j]] / pivot;
            }
        }

        // Eliminate column
        for i in k + 1..n {
            let factor = aug[[i, k]];
            for j in k..n + 1 {
                aug[[i, j]] = aug[[i, j]] - factor * aug[[k, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        x[i] = aug[[i, n]];
        for j in i + 1..n {
            x[i] = x[i] - aug[[i, j]] * x[j];
        }
    }

    x
}

/// Compute the norm of a complex vector
#[allow(dead_code)]
fn compute_complex_norm<I>(v: &Array1<scirs2_core::numeric::Complex<I>>) -> I
where
    I: Float + Zero + Copy,
{
    let mut sum = I::zero();
    for &val in v.iter() {
        sum = sum + val.norm_sqr();
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_compute_determinant_matches_known_value_3x3() {
        // det([[1,2,3],[4,5,6],[7,8,10]]) = -3 (hand-computable via cofactor
        // expansion). The old placeholder returned `matrix[[0,0]] == 1` for
        // any n > 2, which is wrong here.
        let m = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
        let det = compute_determinant(&m);
        assert!((det - (-3.0)).abs() < 1e-9, "expected det=-3, got {det}");
        assert!(
            (det - m[[0, 0]]).abs() > 1.0,
            "must not collapse to matrix[[0,0]]"
        );
    }

    #[test]
    fn test_compute_determinant_matches_known_value_4x4_block_diagonal() {
        // Block-diagonal 4x4: det = det(block1) * det(block2) = 5 * 18 = 90.
        let m = array![
            [2.0_f64, 1.0, 0.0, 0.0],
            [1.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 2.0],
            [0.0, 0.0, 1.0, 5.0]
        ];
        let det = compute_determinant(&m);
        assert!((det - 90.0).abs() < 1e-9, "expected det=90, got {det}");
        assert!(
            (det - m[[0, 0]]).abs() > 1.0,
            "must not collapse to matrix[[0,0]]"
        );
    }

    #[test]
    fn test_newton_eigenvalue_correction_does_not_collapse_eigenvalues() {
        // Directly exercises the previously-critical bug: with the fake
        // determinant, ANY starting eigenvalue estimate collapsed to
        // `matrix[[0,0]]` after one Newton step. With a real determinant,
        // eigenvalue estimates that are already close to the true spectrum
        // {1, 2, 3} must stay close to it, not collapse onto matrix[[0,0]]=6.
        let a = array![[6.0_f64, -11.0, 6.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        // NOTE: this matrix is non-symmetric; we only use it here to exercise
        // `newton_eigenvalue_correction`'s own characteristic-polynomial
        // machinery directly against a matrix whose eigenvalues (1, 2, 3)
        // are hand-computable, independent of `advanced_precision_eigh`'s
        // symmetric-only pipeline.
        let mut eigenvalues = scirs2_core::ndarray::Array1::from(vec![1.01_f64, 1.99, 3.02]);
        newton_eigenvalue_correction(&mut eigenvalues, &a, 1e-10).expect("Newton step failed");

        let mut sorted: Vec<f64> = eigenvalues.iter().copied().collect();
        sorted.sort_by(|x, y| x.partial_cmp(y).expect("no NaNs"));
        assert!(
            (sorted[0] - 1.0).abs() < 1e-6,
            "expected ~1.0, got {}",
            sorted[0]
        );
        assert!(
            (sorted[1] - 2.0).abs() < 1e-6,
            "expected ~2.0, got {}",
            sorted[1]
        );
        assert!(
            (sorted[2] - 3.0).abs() < 1e-6,
            "expected ~3.0, got {}",
            sorted[2]
        );
    }

    #[test]
    fn test_advanced_precision_eigh_nondiagonal_small_eigenvalues() {
        // Genuinely non-diagonal symmetric matrix (scaled path-graph
        // Laplacian) with known, well-separated, small analytic eigenvalues
        // 0.01*(2 - 2*cos(k*pi/4)) for k=1,2,3: ~0.005858, 0.02, 0.034142.
        // Small magnitudes keep the Newton-correction finite-difference step
        // (h=1e-8) numerically meaningful in f32.
        let a = array![
            [0.02_f32, -0.01, 0.0],
            [-0.01, 0.02, -0.01],
            [0.0, -0.01, 0.02]
        ];
        let (eigenvalues, eigenvectors) =
            advanced_precision_eigh::<_, f64>(&a.view(), None, None, true)
                .expect("Operation failed");

        let mut sorted: Vec<f32> = eigenvalues.iter().copied().collect();
        sorted.sort_by(|x, y| x.partial_cmp(y).expect("no NaNs"));
        assert!(
            (sorted[0] - 0.005858).abs() < 1e-3,
            "expected ~0.005858, got {}",
            sorted[0]
        );
        assert!(
            (sorted[1] - 0.02).abs() < 1e-3,
            "expected ~0.02, got {}",
            sorted[1]
        );
        assert!(
            (sorted[2] - 0.034142).abs() < 1e-3,
            "expected ~0.034142, got {}",
            sorted[2]
        );
        // The three eigenvalues must be genuinely distinct (a collapse bug
        // would drive them all toward the same value).
        assert!((sorted[1] - sorted[0]).abs() > 1e-4);
        assert!((sorted[2] - sorted[1]).abs() > 1e-4);

        // A v ~= lambda v for every reported eigenpair.
        for col in 0..3 {
            let lambda = eigenvalues[col];
            for row in 0..3 {
                let av: f32 = (0..3).map(|k| a[[row, k]] * eigenvectors[[k, col]]).sum();
                assert!(
                    (av - lambda * eigenvectors[[row, col]]).abs() < 1e-3,
                    "A*v != lambda*v at col {col} row {row}"
                );
            }
        }
    }

    #[test]
    fn test_advanced_precision_eigh() {
        // Simple diagonal matrix with known eigenvalues
        let a = array![[4.0f32, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]];

        let (eigenvalues, eigenvectors) =
            advanced_precision_eigh::<_, f64>(&a.view(), None, None, true)
                .expect("Operation failed");

        // For a diagonal matrix, sort the eigenvalues
        let mut sorted_indices = (0..eigenvalues.len()).collect::<Vec<_>>();
        sorted_indices.sort_by(|&i, &j| {
            eigenvalues[i]
                .partial_cmp(&eigenvalues[j])
                .expect("Operation failed")
        });

        // Verify the eigenvalues are close to the expected values
        assert!(
            (eigenvalues[sorted_indices[0]] - 1.0).abs() < 0.1,
            "Expected eigenvalue 1.0, got {}",
            eigenvalues[sorted_indices[0]]
        );
        assert!(
            (eigenvalues[sorted_indices[1]] - 2.0).abs() < 0.1,
            "Expected eigenvalue 2.0, got {}",
            eigenvalues[sorted_indices[1]]
        );
        assert!(
            (eigenvalues[sorted_indices[2]] - 4.0).abs() < 0.1,
            "Expected eigenvalue 4.0, got {}",
            eigenvalues[sorted_indices[2]]
        );

        // Check eigenvectors are orthogonal
        for i in 0..eigenvectors.ncols() {
            for j in i + 1..eigenvectors.ncols() {
                let dot_product = eigenvectors.column(i).dot(&eigenvectors.column(j));
                assert!(
                    dot_product.abs() < 1e-4,
                    "Eigenvectors {} and {} not orthogonal: dot product = {}",
                    i,
                    j,
                    dot_product
                );
            }
        }
    }

    #[test]
    fn test_estimate_condition_number() {
        // Identity matrix should have condition number 1
        let identity = array![[1.0f32, 0.0], [0.0, 1.0]];
        let cond = estimate_condition_number(&identity.view()).expect("Operation failed");
        assert!(
            (0.5..=2.0).contains(&cond),
            "Expected condition number ~1, got {}",
            cond
        );

        // Well-conditioned matrix
        let well_cond = array![[2.0f32, 1.0], [1.0, 2.0]];
        let cond = estimate_condition_number(&well_cond.view()).expect("Operation failed");
        assert!(
            cond > 0.0 && cond < 100.0,
            "Expected reasonable condition number, got {}",
            cond
        );
    }
}
