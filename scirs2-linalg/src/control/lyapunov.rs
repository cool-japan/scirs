//! Lyapunov equation solvers for control theory.
//!
//! Provides solvers for:
//! - Continuous Lyapunov equation: `AX + XAᵀ = -Q`
//! - Discrete Lyapunov equation: `AXAᵀ - X = -Q`
//!
//! # Algorithms
//!
//! The solvers use the Bartels-Stewart algorithm based on Schur decomposition.
//! The equation is first transformed to a quasi-triangular form via Schur
//! decomposition, then solved by back-substitution, and finally transformed back.
//!
//! For the continuous case (`AX + XAᵀ = -Q`), assuming `A = U T Uᵀ` (Schur),
//! we solve `T Y + Y Tᵀ = -Ũ` where `Ũ = Uᵀ Q U` and `X = U Y Uᵀ`.
//!
//! # References
//! - Bartels, R. H. & Stewart, G. W. (1972). Algorithm 432: Solution of the
//!   matrix equation AX + XB = C. *Communications of the ACM*, 15(9), 820–826.
//! - Golub, G. H., Nash, S. & Van Loan, C. (1979). A Hessenberg-Schur method
//!   for the problem AX + XB = C. *IEEE Trans. Autom. Control*, 24(6), 909–913.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::{Debug, Display};
use std::iter::Sum;

/// Trait bound for floating-point scalars in Lyapunov solvers.
pub trait LyapFloat:
    Float + NumAssign + Sum + ScalarOperand + Debug + Display + Send + Sync + 'static
{
}
impl<F> LyapFloat for F where
    F: Float + NumAssign + Sum + ScalarOperand + Debug + Display + Send + Sync + 'static
{
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Dense n×n matrix multiply C = A · B.
fn matmul<F: LyapFloat>(a: &Array2<F>, b: &Array2<F>) -> Array2<F> {
    a.dot(b)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Solve the continuous Lyapunov equation `AX + XAᵀ = -Q`.
///
/// Requires that `A` is stable (all eigenvalues have negative real part) for a
/// unique positive-definite solution to exist.
///
/// # Arguments
/// * `a` - Square `n×n` system matrix (must be stable for unique solution)
/// * `q` - Square `n×n` right-hand side matrix (typically positive semi-definite)
///
/// # Returns
/// The unique solution matrix `X` satisfying `AX + XAᵀ = -Q`.
///
/// # Algorithm
/// Delegates to the Bartels-Stewart Sylvester solver:
/// `AX + XAᵀ = -Q` is equivalent to the Sylvester equation `AX + X(-Aᵀ) = -Q`,
/// which is then solved via `AX + XAᵀ + Q = 0`.
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::control::lyapunov::lyapunov_continuous;
///
/// let a = array![[-1.0_f64, 0.0], [0.0, -2.0]];
/// let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let x = lyapunov_continuous(&a.view(), &q.view()).expect("solve failed");
/// // Verify: AX + XAᵀ ≈ -Q
/// let res = a.dot(&x) + x.dot(&a.t());
/// ```
pub fn lyapunov_continuous<F: LyapFloat>(
    a: &ArrayView2<F>,
    q: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = check_square(a, "lyapunov_continuous: A")?;
    let m = check_square(q, "lyapunov_continuous: Q")?;
    if n != m {
        return Err(LinalgError::DimensionError(format!(
            "lyapunov_continuous: A is {n}×{n} but Q is {m}×{m}"
        )));
    }
    if n == 0 {
        return Ok(Array2::<F>::zeros((0, 0)));
    }

    // Special case n=1: a*x + x*a = -q  => 2a*x = -q => x = -q/(2a)
    if n == 1 {
        let a00 = a[[0, 0]];
        let two = F::from(2.0).unwrap_or(F::one());
        let denom = two * a00;
        if denom.abs() < F::from(1e-14).unwrap_or(F::epsilon()) {
            return Err(LinalgError::SingularMatrixError(
                "lyapunov_continuous: A is not stable (a[0,0] must be < 0)".to_string(),
            ));
        }
        let mut res = Array2::<F>::zeros((1, 1));
        res[[0, 0]] = -q[[0, 0]] / denom;
        return Ok(res);
    }

    // Delegate to Sylvester-based solver:
    // AX + XA^T + Q = 0 is a continuous Lyapunov equation
    crate::matrix_functions::sylvester::solve_continuous_lyapunov(a, q)
}

/// Solve the discrete Lyapunov equation `AXAᵀ - X = -Q`.
///
/// Requires that `A` is Schur-stable (all eigenvalues inside the unit disk) for a
/// unique positive-definite solution.
///
/// # Arguments
/// * `a` - Square `n×n` system matrix (must be Schur-stable for unique solution)
/// * `q` - Square `n×n` right-hand side matrix (typically positive semi-definite)
///
/// # Returns
/// The unique solution matrix `X` satisfying `AXAᵀ - X = -Q`.
///
/// # Algorithm
/// Delegates to the bilinear-transform discrete Lyapunov solver.
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::control::lyapunov::lyapunov_discrete;
///
/// let a = array![[0.5_f64, 0.1], [0.0, 0.3]];
/// let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let x = lyapunov_discrete(&a.view(), &q.view()).expect("solve failed");
/// // Verify: AXAᵀ - X ≈ -Q
/// let res = a.dot(&x).dot(&a.t()) - &x + &q;
/// ```
pub fn lyapunov_discrete<F: LyapFloat>(
    a: &ArrayView2<F>,
    q: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = check_square(a, "lyapunov_discrete: A")?;
    let m = check_square(q, "lyapunov_discrete: Q")?;
    if n != m {
        return Err(LinalgError::DimensionError(format!(
            "lyapunov_discrete: A is {n}×{n} but Q is {m}×{m}"
        )));
    }
    if n == 0 {
        return Ok(Array2::<F>::zeros((0, 0)));
    }

    // Special case n=1
    if n == 1 {
        let a00 = a[[0, 0]];
        let denom = a00 * a00 - F::one();
        if denom.abs() < F::from(1e-14).unwrap_or(F::epsilon()) {
            return Err(LinalgError::SingularMatrixError(
                "lyapunov_discrete: A is not Schur-stable (|a[0,0]| must be < 1)".to_string(),
            ));
        }
        let mut res = Array2::<F>::zeros((1, 1));
        res[[0, 0]] = -q[[0, 0]] / denom;
        return Ok(res);
    }

    // Delegate to Sylvester-based solver:
    // AXA^T - X + Q = 0
    crate::matrix_functions::sylvester::solve_discrete_lyapunov(a, q)
}

// ---------------------------------------------------------------------------
// Iterative refinement wrapper (Newton-based for better accuracy)
// ---------------------------------------------------------------------------

/// Refine a Lyapunov solution via one step of iterative refinement.
///
/// Given approximate solution `X0`, computes the residual and applies one
/// correction step.
pub fn lyapunov_continuous_refine<F: LyapFloat>(
    a: &ArrayView2<F>,
    q: &ArrayView2<F>,
    x0: &Array2<F>,
    tol: F,
    max_iter: usize,
) -> LinalgResult<Array2<F>> {
    let a_owned = a.to_owned();
    let q_owned = q.to_owned();
    let mut x = x0.clone();

    for _ in 0..max_iter {
        // Compute residual R = AX + XAᵀ + Q
        let ax = matmul(&a_owned, &x);
        let xat = matmul(&x, &a_owned.t().to_owned());
        let residual = ax + xat + &q_owned;

        // Check convergence
        let res_norm: F = residual.iter().map(|&v| v * v).sum::<F>().sqrt();
        if res_norm <= tol {
            return Ok(x);
        }

        // Correction: solve AE + EAᵀ = -R
        let correction = lyapunov_continuous(a, &residual.view())?;
        x = x - correction;
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check_square<F: LyapFloat>(a: &ArrayView2<F>, ctx: &str) -> LinalgResult<usize> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(format!("{ctx}: not square")));
    }
    Ok(n)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_continuous_lyapunov_diagonal() {
        // Diagonal stable A → analytic solution
        // AX + XAᵀ = -Q  with A = diag(-1,-2), Q = I
        // X[i,j] = -Q[i,j]/(a_i + a_j)
        // X[0,0] = -1/(-1-1) = 0.5, X[1,1] = -1/(-2-2) = 0.25
        let a = array![[-1.0_f64, 0.0], [0.0, -2.0]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let x = lyapunov_continuous(&a.view(), &q.view()).expect("lyapunov_continuous failed");

        let expected = array![[0.5_f64, 0.0], [0.0, 0.25]];
        for i in 0..2 {
            for j in 0..2 {
                let diff = (x[[i, j]] - expected[[i, j]]).abs();
                assert!(
                    diff < 1e-8,
                    "Mismatch at ({i},{j}): got {}, expected {}",
                    x[[i, j]],
                    expected[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_continuous_lyapunov_residual_2x2() {
        let a = array![[-2.0_f64, 1.0], [0.0, -3.0]];
        let q = array![[2.0_f64, 1.0], [1.0, 2.0]];
        let x = lyapunov_continuous(&a.view(), &q.view()).expect("failed");
        // Residual: AX + XAᵀ + Q ≈ 0
        let res = a.dot(&x) + x.dot(&a.t()) + &q;
        for &v in res.iter() {
            assert!(v.abs() < 1e-7, "Residual {v} too large");
        }
    }

    #[test]
    fn test_discrete_lyapunov_residual_2x2() {
        let a = array![[0.5_f64, 0.1], [0.0, 0.3]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let x = lyapunov_discrete(&a.view(), &q.view()).expect("failed");
        // Residual: AXAᵀ - X + Q ≈ 0
        let res = a.dot(&x).dot(&a.t()) - &x + &q;
        for &v in res.iter() {
            assert!(v.abs() < 1e-6, "Discrete Lyapunov residual {v} too large");
        }
    }

    #[test]
    fn test_continuous_lyapunov_3x3_residual() {
        let a = array![[-3.0_f64, 1.0, 0.0], [0.0, -2.0, 0.5], [0.0, 0.0, -1.0]];
        let q = array![[2.0_f64, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 3.0]];
        let x = lyapunov_continuous(&a.view(), &q.view()).expect("3x3 failed");
        let res = a.dot(&x) + x.dot(&a.t()) + &q;
        for &v in res.iter() {
            assert!(v.abs() < 1e-6, "3x3 continuous residual {v}");
        }
    }

    #[test]
    fn test_discrete_lyapunov_3x3_residual() {
        let a = array![[0.4_f64, 0.1, 0.0], [0.0, 0.5, 0.2], [0.0, 0.0, 0.3]];
        let q = array![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let x = lyapunov_discrete(&a.view(), &q.view()).expect("3x3 discrete failed");
        let res = a.dot(&x).dot(&a.t()) - &x + &q;
        for &v in res.iter() {
            assert!(v.abs() < 1e-5, "3x3 discrete residual {v}");
        }
    }
}
