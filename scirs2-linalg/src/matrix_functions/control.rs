//! Control theory matrix functions
//!
//! This module provides solvers and utilities for control theory, including:
//!
//! - Continuous and discrete Lyapunov equations
//! - Continuous and discrete algebraic Riccati equations (CARE/DARE)
//! - LQR (Linear Quadratic Regulator) gain computation
//! - Controllability and observability analysis

use crate::error::{LinalgError, LinalgResult};
use crate::matrix_equations::{solve_continuous_lyapunov, solve_discrete_lyapunov};
use scirs2_core::ndarray::{s, Array2, ArrayView2};

// -----------------------------------------------------------------------
// Lyapunov equations (thin wrappers exposing f64-specialised API)
// -----------------------------------------------------------------------

/// Solve the continuous Lyapunov equation AX + XA^T + Q = 0.
///
/// Equivalently solves AX + XA^T = -Q (passed as `q`).
/// Uses the Kronecker-product (vectorization) approach via
/// `matrix_equations::solve_continuous_lyapunov`.
///
/// # Arguments
/// * `a` - Square state matrix A (n×n)
/// * `q` - Symmetric matrix Q (n×n). The convention used here is the common
///   sign convention where Q appears on the *right-hand side with a
///   negative sign*: the caller provides Q and the function solves for X
///   satisfying AX + XA^T = -Q.
///
/// # Returns
/// Solution matrix X such that AX + XA^T + Q = 0
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::control::lyapunov;
///
/// let a = array![[-1.0_f64, 0.0], [0.0, -2.0]];
/// let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let x = lyapunov(&a.view(), &q.view()).expect("lyapunov failed");
///
/// // Verify AX + XA^T + Q ≈ 0
/// let residual = a.dot(&x) + x.dot(&a.t()) + &q;
/// for &v in residual.iter() {
///     assert!(v.abs() < 1e-8, "residual too large: {v}");
/// }
/// ```
pub fn lyapunov(a: &ArrayView2<f64>, q: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError("Matrix A must be square".into()));
    }
    if q.shape() != [n, n] {
        return Err(LinalgError::ShapeError(
            "Matrix Q must have the same shape as A".into(),
        ));
    }
    // AX + XA^T = -Q  =>  solve_continuous_lyapunov with rhs = -Q
    let neg_q = q.mapv(|v| -v);
    solve_continuous_lyapunov(&a, &neg_q.view())
}

/// Solve the discrete Lyapunov equation AXA^T - X + Q = 0.
///
/// Uses the vectorization / bilinear-transform approach from
/// `matrix_equations::solve_discrete_lyapunov`.
///
/// # Arguments
/// * `a` - Square state matrix A (n×n)
/// * `q` - Symmetric matrix Q (n×n)
///
/// # Returns
/// Solution matrix X such that AXA^T - X + Q = 0
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::control::discrete_lyapunov;
///
/// let a = array![[0.5_f64, 0.1], [0.0, 0.6]];
/// let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let x = discrete_lyapunov(&a.view(), &q.view()).expect("discrete_lyapunov failed");
///
/// // Verify AXA^T - X + Q ≈ 0
/// let residual = a.dot(&x).dot(&a.t()) - &x + &q;
/// for &v in residual.iter() {
///     assert!(v.abs() < 1e-7, "residual too large: {v}");
/// }
/// ```
pub fn discrete_lyapunov(a: &ArrayView2<f64>, q: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    solve_discrete_lyapunov(a, q)
}

// -----------------------------------------------------------------------
// Continuous Algebraic Riccati Equation (CARE)
// -----------------------------------------------------------------------

/// Solve the continuous algebraic Riccati equation (CARE):
///   A^T X + X A - X B R^{-1} B^T X + Q = 0
///
/// Uses the Hamiltonian matrix eigenvalue decomposition approach.  The 2n×2n
/// Hamiltonian is formed and its stable invariant subspace (eigenvalues with
/// negative real part) is extracted to compute X.
///
/// # Arguments
/// * `a` - State matrix A (n×n)
/// * `b` - Input matrix B (n×m)
/// * `q` - State-cost matrix Q (n×n, symmetric positive semidefinite)
/// * `r` - Input-cost matrix R (m×m, symmetric positive definite)
///
/// # Returns
/// Solution X (n×n, symmetric positive semidefinite)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::control::care;
///
/// // Double integrator
/// let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
/// let b = array![[0.0_f64], [1.0]];
/// let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let r = array![[1.0_f64]];
/// let x = care(&a.view(), &b.view(), &q.view(), &r.view()).expect("CARE failed");
/// assert!(x[[0, 0]] > 0.0);
/// ```
pub fn care(
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
    q: &ArrayView2<f64>,
    r: &ArrayView2<f64>,
) -> LinalgResult<Array2<f64>> {
    let n = a.shape()[0];
    let m = b.shape()[1];

    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError("Matrix A must be square".into()));
    }
    if b.shape()[0] != n {
        return Err(LinalgError::ShapeError(format!(
            "Matrix B must have {n} rows"
        )));
    }
    if q.shape() != [n, n] {
        return Err(LinalgError::ShapeError("Matrix Q must be n×n".into()));
    }
    if r.shape() != [m, m] {
        return Err(LinalgError::ShapeError("Matrix R must be m×m".into()));
    }

    // Form the Hamiltonian H = [[A, -S], [-Q, -A^T]] where S = B R^{-1} B^T
    let r_inv = crate::inv(r, None)?;
    let s = b.dot(&r_inv).dot(&b.t()); // n×n

    let mut h = Array2::<f64>::zeros((2 * n, 2 * n));
    h.slice_mut(s![..n, ..n]).assign(a);
    h.slice_mut(s![..n, n..]).assign(&s.mapv(|v| -v));
    h.slice_mut(s![n.., ..n]).assign(&q.mapv(|v| -v));
    h.slice_mut(s![n.., n..]).assign(&a.t().mapv(|v| -v));

    // Compute the stable invariant subspace via the matrix sign function.
    //
    // The matrix sign function sign(H) satisfies:
    //   sign(H)^2 = I,  and the stable/unstable invariant subspaces are the
    //   eigenspaces of sign(H) corresponding to ±1.
    //
    // Iteration (Roberts/Kenney-Laub):
    //   S_{k+1} = (S_k + S_k^{-1}) / 2,  S_0 = H
    // This converges quadratically to sign(H).
    //
    // The stable projector is P = (I - sign(H)) / 2 (rank n).
    let frob_norm: f64 = h.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut sign_h = h.clone();

    for _ in 0..50 {
        let sign_inv = crate::inv(&sign_h.view(), None)?;
        let sign_new = (&sign_h + &sign_inv) * 0.5;
        let diff: f64 = (&sign_new - &sign_h)
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        sign_h = sign_new;
        if diff < 1e-12 * (1.0 + frob_norm) {
            break;
        }
    }

    // Stable projector P = (I - sign(H)) / 2
    let eye_2n = Array2::<f64>::eye(2 * n);
    let p_stable = (&eye_2n - &sign_h) * 0.5;

    // The stable invariant subspace = column space of p_stable (rank n).
    // Extract an orthonormal basis via SVD.
    let (u_svd, _sv, _vt) = crate::decomposition::svd(&p_stable.view(), true, None)?;

    // First n left singular vectors span the stable subspace
    let u_stable = u_svd.slice(s![.., ..n]).to_owned();
    let u1 = u_stable.slice(s![..n, ..]).to_owned();
    let u2 = u_stable.slice(s![n.., ..]).to_owned();

    let u1_inv = crate::inv(&u1.view(), None)?;
    let x = u2.dot(&u1_inv);

    // Symmetrize
    Ok((&x + &x.t()) * 0.5)
}

// -----------------------------------------------------------------------
// Discrete Algebraic Riccati Equation (DARE)
// -----------------------------------------------------------------------

/// Solve the discrete algebraic Riccati equation (DARE):
///   X = A^T X A - A^T X B (R + B^T X B)^{-1} B^T X A + Q
///
/// Uses a Structure-Preserving Doubling Algorithm (SDA) followed by a
/// Newton refinement step if the doubling algorithm does not converge.
///
/// # Arguments
/// * `a` - State matrix A (n×n)
/// * `b` - Input matrix B (n×m)
/// * `q` - State-cost matrix Q (n×n, symmetric positive semidefinite)
/// * `r` - Input-cost matrix R (m×m, symmetric positive definite)
///
/// # Returns
/// Solution X (n×n, symmetric positive semidefinite)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::control::dare;
///
/// let a = array![[1.0_f64, 0.1], [0.0, 1.0]];
/// let b = array![[0.0_f64], [0.1]];
/// let q = array![[1.0_f64, 0.0], [0.0, 0.0]];
/// let r = array![[1.0_f64]];
/// let x = dare(&a.view(), &b.view(), &q.view(), &r.view()).expect("DARE failed");
/// assert!(x[[0, 0]] > 0.0);
/// ```
pub fn dare(
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
    q: &ArrayView2<f64>,
    r: &ArrayView2<f64>,
) -> LinalgResult<Array2<f64>> {
    crate::matrix_equations::solve_discrete_riccati(a, b, q, r)
}

// -----------------------------------------------------------------------
// LQR Gain
// -----------------------------------------------------------------------

/// Compute the LQR state-feedback gain K = R^{-1} B^T X.
///
/// X is the solution of the CARE A^T X + X A - X B R^{-1} B^T X + Q = 0.
/// The optimal LQR control law is u = -K x.
///
/// # Arguments
/// * `a` - State matrix A (n×n)
/// * `b` - Input matrix B (n×m)
/// * `q` - State-cost matrix Q (n×n)
/// * `r` - Input-cost matrix R (m×m)
///
/// # Returns
/// Gain matrix K (m×n)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::control::lqr_gain;
///
/// // Double integrator
/// let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
/// let b = array![[0.0_f64], [1.0]];
/// let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let r = array![[1.0_f64]];
/// let k = lqr_gain(&a.view(), &b.view(), &q.view(), &r.view()).expect("LQR failed");
/// assert_eq!(k.shape(), &[1, 2]);
/// ```
pub fn lqr_gain(
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
    q: &ArrayView2<f64>,
    r: &ArrayView2<f64>,
) -> LinalgResult<Array2<f64>> {
    let x = care(a, b, q, r)?;
    let r_inv = crate::inv(r, None)?;
    // K = R^{-1} B^T X
    Ok(r_inv.dot(&b.t()).dot(&x))
}

// -----------------------------------------------------------------------
// Controllability and Observability
// -----------------------------------------------------------------------

/// Construct the controllability matrix C = [B, AB, A²B, ..., A^{n-1}B].
///
/// # Arguments
/// * `a` - State matrix A (n×n)
/// * `b` - Input matrix B (n×m)
///
/// # Returns
/// Controllability matrix (n × n*m)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::control::controllability_matrix;
///
/// let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
/// let b = array![[0.0_f64], [1.0]];
/// let c = controllability_matrix(&a.view(), &b.view());
/// assert_eq!(c.shape(), &[2, 2]);
/// ```
pub fn controllability_matrix(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Array2<f64> {
    let n = a.shape()[0];
    let m = b.shape()[1];
    let total_cols = n * m;

    let mut result = Array2::<f64>::zeros((n, total_cols));
    let mut a_pow_b = b.to_owned(); // A^0 B = B

    for k in 0..n {
        let col_start = k * m;
        let col_end = col_start + m;
        result
            .slice_mut(s![.., col_start..col_end])
            .assign(&a_pow_b);
        if k + 1 < n {
            a_pow_b = a.dot(&a_pow_b);
        }
    }
    result
}

/// Check whether the system (A, B) is controllable.
///
/// The system is controllable iff the controllability matrix has rank n.
///
/// # Arguments
/// * `a` - State matrix A (n×n)
/// * `b` - Input matrix B (n×m)
///
/// # Returns
/// `true` if the system is controllable
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::control::is_controllable;
///
/// // Controllable double integrator
/// let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
/// let b = array![[0.0_f64], [1.0]];
/// assert!(is_controllable(&a.view(), &b.view()));
///
/// // Uncontrollable: B doesn't affect x1
/// let a2 = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let b2 = array![[0.0_f64], [0.0]];
/// assert!(!is_controllable(&a2.view(), &b2.view()));
/// ```
pub fn is_controllable(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> bool {
    let n = a.shape()[0];
    let ctrl = controllability_matrix(a, b);
    numerical_rank_matrix(&ctrl.view(), None) >= n
}

/// Construct the observability matrix O = [C; CA; CA²; ...; CA^{n-1}].
///
/// # Arguments
/// * `a` - State matrix A (n×n)
/// * `c` - Output matrix C (p×n)
///
/// # Returns
/// Observability matrix (n*p × n)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::control::observability_matrix;
///
/// let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
/// let c = array![[1.0_f64, 0.0]];
/// let o = observability_matrix(&a.view(), &c.view());
/// assert_eq!(o.shape(), &[2, 2]);
/// ```
pub fn observability_matrix(a: &ArrayView2<f64>, c: &ArrayView2<f64>) -> Array2<f64> {
    let n = a.shape()[0];
    let p = c.shape()[0];
    let total_rows = n * p;

    let mut result = Array2::<f64>::zeros((total_rows, n));
    let mut c_a_pow = c.to_owned(); // C A^0 = C

    for k in 0..n {
        let row_start = k * p;
        let row_end = row_start + p;
        result
            .slice_mut(s![row_start..row_end, ..])
            .assign(&c_a_pow);
        if k + 1 < n {
            c_a_pow = c_a_pow.dot(a);
        }
    }
    result
}

/// Check whether the system (A, C) is observable.
///
/// The system is observable iff the observability matrix has rank n.
///
/// # Arguments
/// * `a` - State matrix A (n×n)
/// * `c` - Output matrix C (p×n)
///
/// # Returns
/// `true` if the system is observable
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::control::is_observable;
///
/// let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
/// let c = array![[1.0_f64, 0.0]];
/// assert!(is_observable(&a.view(), &c.view()));
/// ```
pub fn is_observable(a: &ArrayView2<f64>, c: &ArrayView2<f64>) -> bool {
    let n = a.shape()[0];
    let obs = observability_matrix(a, c);
    numerical_rank_matrix(&obs.view(), None) >= n
}

// -----------------------------------------------------------------------
// Internal helper: numerical rank via SVD
// -----------------------------------------------------------------------

fn numerical_rank_matrix(a: &ArrayView2<f64>, tol: Option<f64>) -> usize {
    match crate::decomposition::svd(a, false, None) {
        Ok((_, s, _)) => {
            let max_sv = s.iter().cloned().fold(0.0_f64, f64::max);
            let threshold = tol.unwrap_or_else(|| {
                let (m, n) = (a.nrows(), a.ncols());
                let eps = f64::EPSILON;
                eps * (m.max(n) as f64) * max_sv
            });
            s.iter().filter(|&&sv| sv > threshold).count()
        }
        Err(_) => 0,
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    const TOL: f64 = 1e-7;

    #[test]
    fn test_lyapunov_diagonal() {
        // A = diag(-1, -2), Q = I  => X = diag(1/2, 1/4)
        let a = array![[-1.0_f64, 0.0], [0.0, -2.0]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let x = lyapunov(&a.view(), &q.view()).expect("lyapunov failed");

        // AX + XA^T + Q = 0
        let residual = a.dot(&x) + x.dot(&a.t()) + &q;
        for &v in residual.iter() {
            assert!(v.abs() < TOL, "lyapunov residual {v}");
        }
    }

    #[test]
    fn test_lyapunov_full() {
        let a = array![[-2.0_f64, 1.0], [-1.0, -3.0]];
        let q = array![[2.0_f64, 0.5], [0.5, 3.0]];
        let x = lyapunov(&a.view(), &q.view()).expect("lyapunov failed");
        let residual = a.dot(&x) + x.dot(&a.t()) + &q;
        for &v in residual.iter() {
            assert!(v.abs() < TOL, "lyapunov full residual {v}");
        }
    }

    #[test]
    fn test_discrete_lyapunov_basic() {
        let a = array![[0.5_f64, 0.1], [0.0, 0.6]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let x = discrete_lyapunov(&a.view(), &q.view()).expect("discrete_lyapunov failed");
        // AXA^T - X + Q = 0
        let residual = a.dot(&x).dot(&a.t()) - &x + &q;
        for &v in residual.iter() {
            assert!(v.abs() < TOL, "discrete_lyapunov residual {v}");
        }
    }

    #[test]
    fn test_care_double_integrator() {
        let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
        let b = array![[0.0_f64], [1.0]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let r = array![[1.0_f64]];
        let x = care(&a.view(), &b.view(), &q.view(), &r.view()).expect("CARE failed");
        // X should be positive definite
        assert!(x[[0, 0]] > 0.0);
        assert!(x[[1, 1]] > 0.0);
        // Verify CARE residual: A^T X + X A - X B R^{-1} B^T X + Q ≈ 0
        let r_inv = crate::inv(&r.view(), None).expect("inv failed");
        let residual = a.t().dot(&x) + x.dot(&a) - x.dot(&b).dot(&r_inv).dot(&b.t()).dot(&x) + &q;
        for &v in residual.iter() {
            assert!(v.abs() < 1e-6, "CARE residual {v}");
        }
    }

    #[test]
    fn test_dare_basic() {
        let a = array![[1.0_f64, 0.1], [0.0, 1.0]];
        let b = array![[0.0_f64], [0.1]];
        let q = array![[1.0_f64, 0.0], [0.0, 0.0]];
        let r = array![[1.0_f64]];
        let x = dare(&a.view(), &b.view(), &q.view(), &r.view()).expect("DARE failed");
        assert!(x[[0, 0]] > 0.0);
    }

    #[test]
    fn test_lqr_gain_shape() {
        let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
        let b = array![[0.0_f64], [1.0]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let r = array![[1.0_f64]];
        let k = lqr_gain(&a.view(), &b.view(), &q.view(), &r.view()).expect("LQR failed");
        assert_eq!(k.shape(), &[1, 2]);
    }

    #[test]
    fn test_controllability_matrix_shape() {
        // n=3, m=2  =>  shape (3, 6)
        let a = Array2::<f64>::eye(3);
        let b = Array2::<f64>::zeros((3, 2));
        let c = controllability_matrix(&a.view(), &b.view());
        assert_eq!(c.shape(), &[3, 6]);
    }

    #[test]
    fn test_is_controllable_true() {
        let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
        let b = array![[0.0_f64], [1.0]];
        assert!(is_controllable(&a.view(), &b.view()));
    }

    #[test]
    fn test_is_controllable_false() {
        // Zero B → uncontrollable
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let b = Array2::<f64>::zeros((2, 1));
        assert!(!is_controllable(&a.view(), &b.view()));
    }

    #[test]
    fn test_observability_matrix_shape() {
        // n=3, p=1  =>  shape (3, 3)
        let a = Array2::<f64>::eye(3);
        let c = Array2::<f64>::zeros((1, 3));
        let o = observability_matrix(&a.view(), &c.view());
        assert_eq!(o.shape(), &[3, 3]);
    }

    #[test]
    fn test_is_observable_true() {
        let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
        let c = array![[1.0_f64, 0.0]];
        assert!(is_observable(&a.view(), &c.view()));
    }

    #[test]
    fn test_is_observable_false() {
        // C = 0  =>  not observable
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let c = Array2::<f64>::zeros((1, 2));
        assert!(!is_observable(&a.view(), &c.view()));
    }
}
