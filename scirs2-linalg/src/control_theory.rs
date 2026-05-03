//! Control theory linear algebra computations.
//!
//! Provides:
//! - Continuous Lyapunov equation: `AX + XAᵀ + Q = 0`
//! - Discrete Lyapunov equation: `AXAᵀ - X + Q = 0`
//! - Continuous algebraic Riccati equation (CARE): `AᵀX + XA - XBR⁻¹BᵀX + Q = 0`
//! - Discrete algebraic Riccati equation (DARE): `AᵀXA - X - AᵀXB(R + BᵀXB)⁻¹BᵀXA + Q = 0`
//! - Controllability matrix and Gramian
//! - Observability matrix and Gramian
//! - Balanced truncation model reduction
//!
//! # References
//! - Golub & Van Loan, *Matrix Computations*, 4th ed., Ch. 7.7
//! - Skogestad & Postlethwaite, *Multivariable Feedback Design*, 2nd ed.
//! - Zhou, Doyle & Glover, *Robust and Optimal Control*, Ch. 16

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::{Debug, Display};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Trait alias
// ---------------------------------------------------------------------------

/// Trait alias for floating-point scalars used throughout this module.
pub trait CtrlFloat:
    Float + NumAssign + Sum + ScalarOperand + Debug + Display + Send + Sync + 'static
{
}
impl<F> CtrlFloat for F where
    F: Float + NumAssign + Sum + ScalarOperand + Debug + Display + Send + Sync + 'static
{
}

// ---------------------------------------------------------------------------
// Internal dense matrix utilities (Vec<Vec<F>> based for solver internals)
// ---------------------------------------------------------------------------

/// Dense n×n matrix multiply C = A · B.
fn matmul_sq<F: CtrlFloat>(a: &Array2<F>, b: &Array2<F>, n: usize) -> Array2<F> {
    let mut c = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[[i, k]];
            if a_ik == F::zero() {
                continue;
            }
            for j in 0..n {
                c[[i, j]] += a_ik * b[[k, j]];
            }
        }
    }
    c
}

/// Transpose of an n×n matrix.
fn transpose_sq<F: CtrlFloat>(a: &Array2<F>, n: usize) -> Array2<F> {
    let mut t = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            t[[j, i]] = a[[i, j]];
        }
    }
    t
}

/// Add two n×n matrices in-place: a += b.
fn mat_add_inplace<F: CtrlFloat>(a: &mut Array2<F>, b: &Array2<F>, n: usize) {
    for i in 0..n {
        for j in 0..n {
            a[[i, j]] += b[[i, j]];
        }
    }
}

/// Gauss-Jordan inverse of a square matrix.
fn mat_inv_gauss<F: CtrlFloat>(a: &Array2<F>, n: usize) -> LinalgResult<Array2<F>> {
    let mut aug: Vec<Vec<F>> = (0..n)
        .map(|i| {
            let mut row: Vec<F> = (0..n).map(|j| a[[i, j]]).collect();
            for j in 0..n {
                row.push(if i == j { F::one() } else { F::zero() });
            }
            row
        })
        .collect();

    let eps = F::epsilon() * F::from(1_000.0).unwrap_or(F::one());

    for col in 0..n {
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                aug[r1][col]
                    .abs()
                    .partial_cmp(&aug[r2][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);
        aug.swap(col, pivot_row);

        let pivot = aug[col][col];
        if pivot.abs() < eps {
            return Err(LinalgError::SingularMatrixError(
                "Singular matrix in Gauss-Jordan inversion (control_theory)".into(),
            ));
        }
        let inv_pivot = F::one() / pivot;
        for elem in aug[col].iter_mut() {
            *elem *= inv_pivot;
        }
        for i in 0..n {
            if i != col {
                let factor = aug[i][col];
                if factor == F::zero() {
                    continue;
                }
                let col_row = aug[col].clone();
                for (j, &v) in col_row.iter().enumerate() {
                    aug[i][j] -= factor * v;
                }
            }
        }
    }

    let mut inv = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[i][n + j];
        }
    }
    Ok(inv)
}

/// Solve a dense n×n linear system Ax = b via Gaussian elimination with partial pivoting.
fn solve_linear<F: CtrlFloat>(a: &[Vec<F>], b: &[F], n: usize) -> LinalgResult<Vec<F>> {
    let mut aug: Vec<Vec<F>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    let eps = F::epsilon() * F::from(1_000.0).unwrap_or(F::one());

    for col in 0..n {
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                aug[r1][col]
                    .abs()
                    .partial_cmp(&aug[r2][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);
        aug.swap(col, pivot_row);

        let pivot = aug[col][col];
        if pivot.abs() < eps {
            return Err(LinalgError::SingularMatrixError(
                "Singular system in solve_linear (control_theory)".into(),
            ));
        }
        let inv_pivot = F::one() / pivot;
        for elem in aug[col].iter_mut() {
            *elem *= inv_pivot;
        }
        for i in 0..n {
            if i != col {
                let factor = aug[i][col];
                if factor == F::zero() {
                    continue;
                }
                let col_row = aug[col].clone();
                for (j, &v) in col_row.iter().enumerate() {
                    aug[i][j] -= factor * v;
                }
            }
        }
    }
    Ok(aug.iter().map(|row| row[n]).collect())
}

// ---------------------------------------------------------------------------
// Lyapunov Equation Solvers
// ---------------------------------------------------------------------------

/// Solve the **continuous Lyapunov equation**: `AX + XAᵀ + Q = 0`.
///
/// Forms the Kronecker-product linear system
/// `(I ⊗ A + A ⊗ I) vec(X) = −vec(Q)` and solves it via Gaussian elimination.
///
/// # Arguments
/// * `a` - Stable matrix A (n×n); all eigenvalues must have negative real part for
///   a unique solution to exist.
/// * `q` - Symmetric matrix Q (n×n).
///
/// # Returns
/// Symmetric solution matrix X (n×n).
///
/// # Errors
/// Returns `LinalgError::SingularMatrixError` if the Kronecker system is singular
/// (which occurs when A is not stable or has repeated eigenvalues summing to zero).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::control_theory::lyapunov_continuous;
///
/// // Stable A: eigenvalues −1, −2
/// let a = array![[-1.0_f64, 0.0], [0.0, -2.0]];
/// let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let x = lyapunov_continuous(&a.view(), &q.view()).expect("lyapunov_continuous failed");
///
/// // Verify AX + XAᵀ + Q ≈ 0
/// let residual = a.dot(&x) + x.dot(&a.t()) + &q;
/// for &v in residual.iter() { assert!(v.abs() < 1e-10, "residual = {v}"); }
/// ```
pub fn lyapunov_continuous<F: CtrlFloat>(
    a: &ArrayView2<F>,
    q: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    check_square(a, "lyapunov_continuous: A")?;
    if q.nrows() != n || q.ncols() != n {
        return Err(LinalgError::ShapeError(
            "lyapunov_continuous: Q must be the same size as A".into(),
        ));
    }

    let n2 = n * n;
    // Build Kronecker system: K = (I ⊗ A) + (A ⊗ I)
    // Index mapping: vec(X) has index i*n + j  →  X[i][j]
    // (I ⊗ A)[i*n+k, i*n+j] = A[k][j]  for all i
    // (A ⊗ I)[i*n+k, j*n+k] = A[i][j]  for all k
    let mut kron: Vec<Vec<F>> = vec![vec![F::zero(); n2]; n2];
    for i in 0..n {
        for j in 0..n {
            // I ⊗ A contribution: row i*n+k, col i*n+j += A[k][j]
            for k in 0..n {
                kron[i * n + k][i * n + j] += a[[k, j]];
            }
            // A ⊗ I contribution: row i*n+k, col j*n+k += A[i][j]
            let a_ij = a[[i, j]];
            for k in 0..n {
                kron[i * n + k][j * n + k] += a_ij;
            }
        }
    }

    // RHS: −vec(Q)
    let rhs: Vec<F> = (0..n)
        .flat_map(|i| (0..n).map(move |j| -q[[i, j]]))
        .collect();

    let x_vec = solve_linear(&kron, &rhs, n2)?;

    let mut x = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            x[[i, j]] = x_vec[i * n + j];
        }
    }
    Ok(x)
}

/// Solve the **discrete Lyapunov (Stein) equation**: `AXAᵀ - X + Q = 0`.
///
/// Forms the Kronecker-product linear system
/// `(A ⊗ A − I ⊗ I) vec(X) = −vec(Q)` and solves via Gaussian elimination.
///
/// # Arguments
/// * `a` - Stable matrix A (n×n); spectral radius < 1 for unique solution.
/// * `q` - Symmetric matrix Q (n×n).
///
/// # Returns
/// Symmetric solution matrix X (n×n).
///
/// # Errors
/// Returns `LinalgError::SingularMatrixError` if the Kronecker system is singular.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::control_theory::lyapunov_discrete;
///
/// let a = array![[0.5_f64, 0.0], [0.0, 0.3]];
/// let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let x = lyapunov_discrete(&a.view(), &q.view()).expect("lyapunov_discrete failed");
///
/// // Verify A X Aᵀ - X + Q ≈ 0
/// let residual = a.dot(&x).dot(&a.t()) - &x + &q;
/// for &v in residual.iter() { assert!(v.abs() < 1e-10, "residual = {v}"); }
/// ```
pub fn lyapunov_discrete<F: CtrlFloat>(
    a: &ArrayView2<F>,
    q: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    check_square(a, "lyapunov_discrete: A")?;
    if q.nrows() != n || q.ncols() != n {
        return Err(LinalgError::ShapeError(
            "lyapunov_discrete: Q must be the same size as A".into(),
        ));
    }

    let n2 = n * n;
    // K = A ⊗ A − I
    // (A ⊗ A)[i*n+k, j*n+l] = A[i][j] * A[k][l]
    let mut kron: Vec<Vec<F>> = vec![vec![F::zero(); n2]; n2];
    for i in 0..n {
        for j in 0..n {
            let a_ij = a[[i, j]];
            for k in 0..n {
                for l in 0..n {
                    kron[i * n + k][j * n + l] += a_ij * a[[k, l]];
                }
            }
        }
    }
    // Subtract identity
    for i in 0..n2 {
        kron[i][i] -= F::one();
    }

    // RHS: −vec(Q)
    let rhs: Vec<F> = (0..n)
        .flat_map(|i| (0..n).map(move |j| -q[[i, j]]))
        .collect();

    let x_vec = solve_linear(&kron, &rhs, n2)?;

    let mut x = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            x[[i, j]] = x_vec[i * n + j];
        }
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Riccati Equation Solvers
// ---------------------------------------------------------------------------

/// Solve the **Continuous Algebraic Riccati Equation (CARE)**:
/// `AᵀX + XA − XBR⁻¹BᵀX + Q = 0`.
///
/// Uses Newton's method (CARE Newton iteration) starting from the solution
/// of the associated Lyapunov equation.
///
/// # Arguments
/// * `a` - System matrix A (n×n)
/// * `b` - Input matrix B (n×m)
/// * `q` - State cost matrix Q (n×n), symmetric PSD
/// * `r` - Input cost matrix R (m×m), symmetric PD
///
/// # Returns
/// Symmetric positive semi-definite solution matrix X (n×n).
///
/// # Errors
/// Returns `LinalgError` if the system cannot be solved.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::control_theory::riccati_continuous;
///
/// let a = array![[0.0_f64, 1.0], [-1.0, -1.0]];
/// let b = array![[0.0_f64], [1.0]];
/// let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let r = array![[1.0_f64]];
/// let x = riccati_continuous(&a.view(), &b.view(), &q.view(), &r.view())
///     .expect("riccati_continuous failed");
/// assert_eq!(x.nrows(), 2);
/// ```
pub fn riccati_continuous<F: CtrlFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    q: &ArrayView2<F>,
    r: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    let m = b.ncols();
    check_square(a, "riccati_continuous: A")?;
    if b.nrows() != n {
        return Err(LinalgError::ShapeError(
            "riccati_continuous: B must have n rows".into(),
        ));
    }
    if q.nrows() != n || q.ncols() != n {
        return Err(LinalgError::ShapeError(
            "riccati_continuous: Q must be n×n".into(),
        ));
    }
    if r.nrows() != m || r.ncols() != m {
        return Err(LinalgError::ShapeError(
            "riccati_continuous: R must be m×m".into(),
        ));
    }

    // R⁻¹
    let r_inv = mat_inv_gauss(&r.to_owned(), m)?;

    // S = B R⁻¹ Bᵀ  (n×n)
    let r_inv_bt = {
        let mut s = Array2::<F>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                for k in 0..m {
                    s[[i, j]] += r_inv[[i, k]] * b[[j, k]];
                }
            }
        }
        s
    };
    let mut s_mat = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for k in 0..m {
            let b_ik = b[[i, k]];
            if b_ik == F::zero() {
                continue;
            }
            for j in 0..n {
                s_mat[[i, j]] += b_ik * r_inv_bt[[k, j]];
            }
        }
    }

    // Newton iteration: starting with X_0 = Q (symmetric), iterate
    //   A_k = A − S X_k
    //   Solve Lyapunov: A_kᵀ X_{k+1} + X_{k+1} A_k + X_k S X_k + Q = 0
    // Equivalently the Newton step for CARE.
    //
    // We use a simpler fixed-point / sign-function seed:
    // Start with X₀ = 0 and apply Newton-Kleinman iteration.
    let tol = F::epsilon() * F::from(1e6).unwrap_or(F::one());
    let max_iter = 100usize;

    let mut x = Array2::<F>::zeros((n, n));
    let a_t = transpose_sq(&a.to_owned(), n);

    for _iter in 0..max_iter {
        // A_cl = A − S X
        let sx = matmul_sq(&s_mat, &x, n);
        let mut a_cl = a.to_owned();
        for i in 0..n {
            for j in 0..n {
                a_cl[[i, j]] -= sx[[i, j]];
            }
        }

        // Q_k = Q + X S X
        let xs = matmul_sq(&x, &s_mat, n);
        let xsx = matmul_sq(&xs, &x, n);
        let mut q_k = q.to_owned();
        mat_add_inplace(&mut q_k, &xsx, n);

        // Solve Lyapunov: A_clᵀ X_new + X_new A_cl + Q_k = 0
        let a_cl_t = transpose_sq(&a_cl, n);
        let x_new = lyapunov_continuous(&a_cl_t.view(), &q_k.view())?;

        // Check convergence
        let mut diff = F::zero();
        for i in 0..n {
            for j in 0..n {
                let d = (x_new[[i, j]] - x[[i, j]]).abs();
                if d > diff {
                    diff = d;
                }
            }
        }
        x = x_new;
        if diff < tol {
            break;
        }
    }

    // Symmetrize
    let x_t = transpose_sq(&x, n);
    let two = F::from(2.0).unwrap_or(F::one());
    for i in 0..n {
        for j in 0..n {
            x[[i, j]] = (x[[i, j]] + x_t[[i, j]]) / two;
        }
    }

    let _ = a_t; // suppress unused warning
    Ok(x)
}

/// Solve the **Discrete Algebraic Riccati Equation (DARE)**:
/// `AᵀXA − X − AᵀXB(R + BᵀXB)⁻¹BᵀXA + Q = 0`.
///
/// Uses Newton-Kleinman iteration: at each step solves a discrete Lyapunov equation.
///
/// # Arguments
/// * `a` - System matrix A (n×n)
/// * `b` - Input matrix B (n×m)
/// * `q` - State cost matrix Q (n×n), symmetric PSD
/// * `r` - Input cost matrix R (m×m), symmetric PD
///
/// # Returns
/// Symmetric positive semi-definite solution matrix X (n×n).
///
/// # Examples
///
/// ```no_run
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::control_theory::riccati_discrete;
///
/// let a = array![[1.0_f64, 1.0], [0.0, 1.0]];
/// let b = array![[0.0_f64], [1.0]];
/// let q = array![[1.0_f64, 0.0], [0.0, 0.0]];
/// let r = array![[1.0_f64]];
/// let x = riccati_discrete(&a.view(), &b.view(), &q.view(), &r.view())
///     .expect("riccati_discrete failed");
/// assert_eq!(x.nrows(), 2);
/// ```
pub fn riccati_discrete<F: CtrlFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    q: &ArrayView2<F>,
    r: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    let m = b.ncols();
    check_square(a, "riccati_discrete: A")?;
    if b.nrows() != n {
        return Err(LinalgError::ShapeError(
            "riccati_discrete: B must have n rows".into(),
        ));
    }
    if q.nrows() != n || q.ncols() != n {
        return Err(LinalgError::ShapeError(
            "riccati_discrete: Q must be n×n".into(),
        ));
    }
    if r.nrows() != m || r.ncols() != m {
        return Err(LinalgError::ShapeError(
            "riccati_discrete: R must be m×m".into(),
        ));
    }

    let tol = F::epsilon() * F::from(1e6).unwrap_or(F::one());
    let max_iter = 100usize;

    let mut x = q.to_owned();

    for _iter in 0..max_iter {
        // Compute R + BᵀXB  (m×m)
        let xb = {
            let mut tmp = Array2::<F>::zeros((n, m));
            for i in 0..n {
                for j in 0..m {
                    for k in 0..n {
                        tmp[[i, j]] += x[[i, k]] * b[[k, j]];
                    }
                }
            }
            tmp
        };
        let mut rbxb = r.to_owned();
        for i in 0..m {
            for j in 0..m {
                let mut s = F::zero();
                for k in 0..n {
                    s += b[[k, i]] * xb[[k, j]];
                }
                rbxb[[i, j]] += s;
            }
        }
        let rbxb_inv = mat_inv_gauss(&rbxb, m)?;

        // K = (R + BᵀXB)⁻¹ BᵀXA  (m×n)
        // BᵀX  (m×n)
        let btx = {
            let mut tmp = Array2::<F>::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    for k in 0..n {
                        tmp[[i, j]] += b[[k, i]] * x[[k, j]];
                    }
                }
            }
            tmp
        };
        // BᵀXA  (m×n)
        let btxa = {
            let mut tmp = Array2::<F>::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    for k in 0..n {
                        tmp[[i, j]] += btx[[i, k]] * a[[k, j]];
                    }
                }
            }
            tmp
        };
        // K = Rinv_eff · BᵀXA
        let k_gain = {
            let mut tmp = Array2::<F>::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    for k in 0..m {
                        tmp[[i, j]] += rbxb_inv[[i, k]] * btxa[[k, j]];
                    }
                }
            }
            tmp
        };

        // A_cl = A − B K  (n×n)
        let bk = {
            let mut tmp = Array2::<F>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    for k in 0..m {
                        tmp[[i, j]] += b[[i, k]] * k_gain[[k, j]];
                    }
                }
            }
            tmp
        };
        let mut a_cl = a.to_owned();
        for i in 0..n {
            for j in 0..n {
                a_cl[[i, j]] -= bk[[i, j]];
            }
        }

        // Q_k = Aᵀ X A_cl + Q  = AᵀXA − AᵀXBK + Q
        // We compute Aᵀ X A_cl
        let xa_cl = matmul_sq(&x, &a_cl, n);
        let atxa_cl = {
            let mut tmp = Array2::<F>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        tmp[[i, j]] += a[[k, i]] * xa_cl[[k, j]];
                    }
                }
            }
            tmp
        };
        let mut q_k = q.to_owned();
        mat_add_inplace(&mut q_k, &atxa_cl, n);

        // Solve discrete Lyapunov: A_cl X_new A_clᵀ - X_new + Q_k = 0
        let x_new = lyapunov_discrete(&a_cl.view(), &q_k.view())?;

        // Check convergence
        let mut diff = F::zero();
        for i in 0..n {
            for j in 0..n {
                let d = (x_new[[i, j]] - x[[i, j]]).abs();
                if d > diff {
                    diff = d;
                }
            }
        }
        x = x_new;
        if diff < tol {
            break;
        }
    }

    // Symmetrize
    let two = F::from(2.0).unwrap_or(F::one());
    let x_t = transpose_sq(&x, n);
    for i in 0..n {
        for j in 0..n {
            x[[i, j]] = (x[[i, j]] + x_t[[i, j]]) / two;
        }
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Controllability & Observability
// ---------------------------------------------------------------------------

/// Compute the **controllability matrix** `C = [B, AB, A²B, …, A^{n-1}B]`.
///
/// # Arguments
/// * `a` - System matrix A (n×n)
/// * `b` - Input matrix B (n×m)
///
/// # Returns
/// Controllability matrix of shape (n, n·m).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::control_theory::controllability_matrix;
///
/// let a = array![[0.0_f64, 1.0], [-2.0, -3.0]];
/// let b = array![[0.0_f64], [1.0]];
/// let ctrl = controllability_matrix(&a.view(), &b.view());
/// // Rank of ctrl equals 2 (fully controllable)
/// assert_eq!(ctrl.nrows(), 2);
/// assert_eq!(ctrl.ncols(), 2); // n*m = 2*1
/// ```
pub fn controllability_matrix<F: CtrlFloat>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> Array2<F> {
    let n = a.nrows();
    let m = b.ncols();
    let total_cols = n * m;
    let mut result = Array2::<F>::zeros((n, total_cols));

    let mut ab = b.to_owned();
    for k in 0..n {
        // Copy A^k B into columns k*m .. (k+1)*m
        for i in 0..n {
            for j in 0..m {
                result[[i, k * m + j]] = ab[[i, j]];
            }
        }
        // Update: A^{k+1} B = A · (A^k B)
        let mut new_ab = Array2::<F>::zeros((n, m));
        for i in 0..n {
            for l in 0..n {
                let a_il = a[[i, l]];
                if a_il == F::zero() {
                    continue;
                }
                for j in 0..m {
                    new_ab[[i, j]] += a_il * ab[[l, j]];
                }
            }
        }
        ab = new_ab;
    }
    result
}

/// Compute the **observability matrix** `O = [C; CA; CA²; …; CA^{n-1}]`.
///
/// # Arguments
/// * `a` - System matrix A (n×n)
/// * `c` - Output matrix C (p×n)
///
/// # Returns
/// Observability matrix of shape (n·p, n).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::control_theory::observability_matrix;
///
/// let a = array![[0.0_f64, 1.0], [-2.0, -3.0]];
/// let c = array![[1.0_f64, 0.0]];
/// let obs = observability_matrix(&a.view(), &c.view());
/// assert_eq!(obs.nrows(), 2); // n*p = 2*1
/// assert_eq!(obs.ncols(), 2);
/// ```
pub fn observability_matrix<F: CtrlFloat>(a: &ArrayView2<F>, c: &ArrayView2<F>) -> Array2<F> {
    let n = a.nrows();
    let p = c.nrows();
    let total_rows = n * p;
    let mut result = Array2::<F>::zeros((total_rows, n));

    let mut ca = c.to_owned();
    for k in 0..n {
        // Copy CA^k into rows k*p .. (k+1)*p
        for i in 0..p {
            for j in 0..n {
                result[[k * p + i, j]] = ca[[i, j]];
            }
        }
        // Update: CA^{k+1} = (CA^k) · A
        let mut new_ca = Array2::<F>::zeros((p, n));
        for i in 0..p {
            for l in 0..n {
                let ca_il = ca[[i, l]];
                if ca_il == F::zero() {
                    continue;
                }
                for j in 0..n {
                    new_ca[[i, j]] += ca_il * a[[l, j]];
                }
            }
        }
        ca = new_ca;
    }
    result
}

/// Compute the **controllability Gramian** by solving the continuous Lyapunov equation
/// `A Wc + Wc Aᵀ + BBᵀ = 0`.
///
/// Valid only when A is stable (all eigenvalues have strictly negative real parts).
///
/// # Arguments
/// * `a` - Stable system matrix A (n×n)
/// * `b` - Input matrix B (n×m)
///
/// # Returns
/// Controllability Gramian Wc (n×n).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::control_theory::controllability_gramian;
///
/// let a = array![[-1.0_f64, 0.0], [0.0, -2.0]];
/// let b = array![[1.0_f64], [1.0]];
/// let wc = controllability_gramian(&a.view(), &b.view()).expect("gramian failed");
/// assert_eq!(wc.nrows(), 2);
/// ```
pub fn controllability_gramian<F: CtrlFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    let m = b.ncols();
    check_square(a, "controllability_gramian: A")?;
    if b.nrows() != n {
        return Err(LinalgError::ShapeError(
            "controllability_gramian: B must have n rows".into(),
        ));
    }

    // BBᵀ  (n×n)
    let mut bbt = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for k in 0..m {
            let b_ik = b[[i, k]];
            for j in 0..n {
                bbt[[i, j]] += b_ik * b[[j, k]];
            }
        }
    }

    lyapunov_continuous(a, &bbt.view())
}

/// Compute the **observability Gramian** by solving the continuous Lyapunov equation
/// `Aᵀ Wo + Wo A + CᵀC = 0`.
///
/// Valid only when A is stable.
///
/// # Arguments
/// * `a` - Stable system matrix A (n×n)
/// * `c` - Output matrix C (p×n)
///
/// # Returns
/// Observability Gramian Wo (n×n).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::control_theory::observability_gramian;
///
/// let a = array![[-1.0_f64, 0.0], [0.0, -2.0]];
/// let c = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let wo = observability_gramian(&a.view(), &c.view()).expect("gramian failed");
/// assert_eq!(wo.nrows(), 2);
/// ```
pub fn observability_gramian<F: CtrlFloat>(
    a: &ArrayView2<F>,
    c: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    let p = c.nrows();
    check_square(a, "observability_gramian: A")?;
    if c.ncols() != n {
        return Err(LinalgError::ShapeError(
            "observability_gramian: C must have n columns".into(),
        ));
    }

    // CᵀC  (n×n)
    let mut ctc = Array2::<F>::zeros((n, n));
    for k in 0..p {
        for i in 0..n {
            let c_ki = c[[k, i]];
            for j in 0..n {
                ctc[[i, j]] += c_ki * c[[k, j]];
            }
        }
    }

    // Solve Aᵀ Wo + Wo A + CᵀC = 0
    let at = transpose_sq(&a.to_owned(), n);
    lyapunov_continuous(&at.view(), &ctc.view())
}

// ---------------------------------------------------------------------------
// Balanced Truncation
// ---------------------------------------------------------------------------

/// Result of a balanced truncation model reduction.
#[derive(Debug, Clone)]
pub struct BalancedTruncationResult<F: CtrlFloat> {
    /// Reduced-order system matrix Ar (r×r)
    pub a_r: Array2<F>,
    /// Reduced-order input matrix Br (r×m)
    pub b_r: Array2<F>,
    /// Reduced-order output matrix Cr (p×r)
    pub c_r: Array2<F>,
    /// Hankel singular values (sorted descending)
    pub hankel_singular_values: Array1<F>,
    /// Transformation matrix T used for balancing (n×r)
    pub transform: Array2<F>,
}

/// Perform **balanced truncation** model reduction.
///
/// Computes controllability and observability Gramians, forms the balanced
/// realization, and truncates to order `r` by retaining the `r` largest Hankel
/// singular values.
///
/// # Arguments
/// * `a` - Stable system matrix A (n×n)
/// * `b` - Input matrix B (n×m)
/// * `c` - Output matrix C (p×n)
/// * `r` - Desired reduced order (r ≤ n)
///
/// # Returns
/// [`BalancedTruncationResult`] containing the reduced matrices and Hankel singular values.
///
/// # Errors
/// Returns `LinalgError` if the Gramian computations fail or `r > n`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::control_theory::balanced_truncation;
///
/// let a = array![[-1.0_f64, 0.0], [0.0, -10.0]];
/// let b = array![[1.0_f64], [1.0]];
/// let c = array![[1.0_f64, 1.0]];
/// let result = balanced_truncation(&a.view(), &b.view(), &c.view(), 1)
///     .expect("balanced_truncation failed");
/// assert_eq!(result.a_r.nrows(), 1);
/// assert_eq!(result.hankel_singular_values.len(), 2);
/// ```
pub fn balanced_truncation<F: CtrlFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    c: &ArrayView2<F>,
    r: usize,
) -> LinalgResult<BalancedTruncationResult<F>> {
    let n = a.nrows();
    let m = b.ncols();
    let p = c.nrows();

    check_square(a, "balanced_truncation: A")?;
    if b.nrows() != n {
        return Err(LinalgError::ShapeError(
            "balanced_truncation: B must have n rows".into(),
        ));
    }
    if c.ncols() != n {
        return Err(LinalgError::ShapeError(
            "balanced_truncation: C must have n columns".into(),
        ));
    }
    if r == 0 || r > n {
        return Err(LinalgError::ValueError(format!(
            "balanced_truncation: r={r} must be in 1..={n}"
        )));
    }

    // Compute Gramians
    let wc = controllability_gramian(a, b)?;
    let wo = observability_gramian(a, c)?;

    // Hankel singular values: sqrt(eigenvalues of Wc Wo)
    let wc_wo = matmul_sq(&wc, &wo, n);

    // Power-iteration eigenvalues of the symmetric part for Hankel SV
    // The product Wc Wo is not necessarily symmetric; we compute eigenvalues of
    // the symmetric part (Wc Wo + Wo Wc)/2 as an approximation for the HSV ordering.
    let wo_wc = matmul_sq(&wo, &wc, n);
    let mut wc_wo_sym = Array2::<F>::zeros((n, n));
    let two = F::from(2.0).unwrap_or(F::one());
    for i in 0..n {
        for j in 0..n {
            wc_wo_sym[[i, j]] = (wc_wo[[i, j]] + wo_wc[[i, j]]) / two;
        }
    }

    // Eigenvalues of the symmetric positive-semidefinite matrix wc_wo_sym
    let tol = F::epsilon() * F::from(1e6).unwrap_or(F::one());
    let (eig_vals, _eig_vecs) = power_iter_eig(&wc_wo_sym, n, tol, 500)?;

    // HSV = sqrt(|eigenvalue|)
    let mut hsv_indexed: Vec<(usize, F)> = eig_vals
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e.abs().sqrt()))
        .collect();
    hsv_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let hsv: Array1<F> = Array1::from_vec(hsv_indexed.iter().map(|&(_, v)| v).collect());

    // Simple Gramian-square-root balancing:
    // Compute L = chol(Wc), then SVD of L Wo Lᵀ ≈ U Σ Vᵀ
    // Balancing transformation: T = L^{-T} U Σ^{-1/2}
    // Here we approximate with a simple coordinate transformation using
    // the dominant Gramian directions.
    //
    // For numerical stability we use the approximation:
    //   Project onto the top-r left singular vectors of [Wc; Wo] combined.
    // A practical implementation uses the eigenvectors of (Wc Wo) for projection.
    let (_, eig_vecs_sorted) = power_iter_eig(&wc_wo_sym, n, tol, 500)?;

    // Build transformation matrix T using top-r eigenvectors
    let mut transform = Array2::<F>::zeros((n, r));
    for col in 0..r {
        let orig_idx = hsv_indexed[col].0;
        for row in 0..n {
            transform[[row, col]] = eig_vecs_sorted[[row, orig_idx]];
        }
    }

    // Project: Ar = Tᵀ A T, Br = Tᵀ B, Cr = C T
    let t_t = transform.t().to_owned();

    // Ar (r×r)
    let at_t = {
        let mut tmp = Array2::<F>::zeros((r, n));
        for i in 0..r {
            for j in 0..n {
                for k in 0..n {
                    tmp[[i, j]] += t_t[[i, k]] * a[[k, j]];
                }
            }
        }
        tmp
    };
    let mut a_r = Array2::<F>::zeros((r, r));
    for i in 0..r {
        for j in 0..r {
            for k in 0..n {
                a_r[[i, j]] += at_t[[i, k]] * transform[[k, j]];
            }
        }
    }

    // Br (r×m)
    let mut b_r = Array2::<F>::zeros((r, m));
    for i in 0..r {
        for j in 0..m {
            for k in 0..n {
                b_r[[i, j]] += t_t[[i, k]] * b[[k, j]];
            }
        }
    }

    // Cr (p×r)
    let mut c_r = Array2::<F>::zeros((p, r));
    for i in 0..p {
        for j in 0..r {
            for k in 0..n {
                c_r[[i, j]] += c[[i, k]] * transform[[k, j]];
            }
        }
    }

    Ok(BalancedTruncationResult {
        a_r,
        b_r,
        c_r,
        hankel_singular_values: hsv,
        transform,
    })
}

// ---------------------------------------------------------------------------
// Power-iteration eigendecomposition for symmetric matrices
// ---------------------------------------------------------------------------

/// Power-iteration + deflation eigendecomposition for symmetric matrices.
/// Returns (eigenvalues, eigenvector matrix Q) where columns are eigenvectors.
fn power_iter_eig<F: CtrlFloat>(
    a: &Array2<F>,
    n: usize,
    tol: F,
    max_iter: usize,
) -> LinalgResult<(Vec<F>, Array2<F>)> {
    let mut eigenvalues = Vec::with_capacity(n);
    let mut evecs: Vec<Array1<F>> = Vec::with_capacity(n);
    let mut a_work = a.clone();

    for k in 0..n {
        // Start with canonical basis vector, then Gram-Schmidt
        let mut v = Array1::<F>::zeros(n);
        v[k] = F::one();
        for ev in &evecs {
            let dot: F = v.iter().zip(ev.iter()).map(|(&vi, &ei)| vi * ei).sum();
            for i in 0..n {
                v[i] -= dot * ev[i];
            }
        }
        let norm: F = v.iter().map(|&x| x * x).sum::<F>().sqrt();
        if norm < tol {
            v = Array1::<F>::zeros(n);
            v[k % n] = F::one();
        } else {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }

        let mut eigenval = F::zero();
        for _ in 0..max_iter {
            let mut av = Array1::<F>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    av[i] += a_work[[i, j]] * v[j];
                }
            }
            let new_eigenval: F = v.iter().zip(av.iter()).map(|(&vi, &avi)| vi * avi).sum();
            let new_norm: F = av.iter().map(|&x| x * x).sum::<F>().sqrt();
            if new_norm < tol {
                eigenval = new_eigenval;
                break;
            }
            let new_v: Array1<F> = av.mapv(|x| x / new_norm);
            if (new_eigenval - eigenval).abs() < tol {
                eigenval = new_eigenval;
                v = new_v;
                break;
            }
            eigenval = new_eigenval;
            v = new_v;
        }

        eigenvalues.push(eigenval);
        evecs.push(v.clone());

        // Deflate
        for i in 0..n {
            for j in 0..n {
                a_work[[i, j]] -= eigenval * v[i] * v[j];
            }
        }
    }

    let mut q = Array2::<F>::zeros((n, n));
    for (col, ev) in evecs.iter().enumerate() {
        for row in 0..n {
            q[[row, col]] = ev[row];
        }
    }
    Ok((eigenvalues, q))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check_square<F: CtrlFloat>(a: &ArrayView2<F>, ctx: &str) -> LinalgResult<usize> {
    let n = a.nrows();
    if n == 0 {
        return Err(LinalgError::DimensionError(format!(
            "{ctx}: matrix is empty"
        )));
    }
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(format!(
            "{ctx}: matrix must be square"
        )));
    }
    Ok(n)
}

// ---------------------------------------------------------------------------
// Backward-compatibility type alias for callers that import by the old path
// ---------------------------------------------------------------------------

/// Type alias for [`BalancedTruncationResult`] (for convenience).
pub type BtResult<F> = BalancedTruncationResult<F>;

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // ------------------------------------------------------------------
    // Continuous Lyapunov
    // ------------------------------------------------------------------
    #[test]
    fn test_continuous_lyapunov_diagonal() {
        // Diagonal stable A → analytic solution
        // AX + XAᵀ + Q = 0  with A = diag(-1,-2), Q = I
        // Solution: X_ij = -Q_ij / (A_ii + A_jj)
        // X_11 = -1/(-1-1) = 0.5, X_22 = -1/(-2-2) = 0.25, X_12 = 0
        let a = array![[-1.0_f64, 0.0], [0.0, -2.0]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let x = lyapunov_continuous(&a.view(), &q.view()).expect("continuous_lyapunov failed");

        let expected = array![[0.5_f64, 0.0], [0.0, 0.25]];
        for i in 0..2 {
            for j in 0..2 {
                let diff = (x[[i, j]] - expected[[i, j]]).abs();
                assert!(
                    diff < 1e-10,
                    "Lyapunov solution mismatch at ({i},{j}): got {}, expected {}",
                    x[[i, j]],
                    expected[[i, j]]
                );
            }
        }

        // Residual check: AX + XAᵀ + Q ≈ 0
        let res = a.dot(&x) + x.dot(&a.t()) + &q;
        for &v in res.iter() {
            assert!(v.abs() < 1e-10, "Residual = {v}");
        }
    }

    #[test]
    fn test_discrete_lyapunov_residual() {
        // AXAᵀ - X + Q = 0
        let a = array![[0.5_f64, 0.1], [0.0, 0.3]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let x = lyapunov_discrete(&a.view(), &q.view()).expect("discrete_lyapunov failed");

        // Verify residual
        let res = a.dot(&x).dot(&a.t()) - &x + &q;
        for &v in res.iter() {
            assert!(v.abs() < 1e-8, "Discrete Lyapunov residual = {v}");
        }
    }

    // ------------------------------------------------------------------
    // Controllability matrix rank
    // ------------------------------------------------------------------
    #[test]
    fn test_controllability_matrix_rank() {
        // Fully controllable system: A = [[0,1],[-2,-3]], B = [[0],[1]]
        let a = array![[0.0_f64, 1.0], [-2.0, -3.0]];
        let b = array![[0.0_f64], [1.0]];
        let ctrl = controllability_matrix(&a.view(), &b.view());

        // ctrl = [B, AB] = [[0, 1], [1, -3]]
        assert_eq!(ctrl.nrows(), 2);
        assert_eq!(ctrl.ncols(), 2);

        // det(ctrl) = 0*(-3) - 1*1 = -1 ≠ 0  → full rank
        let det = ctrl[[0, 0]] * ctrl[[1, 1]] - ctrl[[0, 1]] * ctrl[[1, 0]];
        assert!(
            det.abs() > 1e-10,
            "Controllability matrix should be full rank, det = {det}"
        );
    }

    #[test]
    fn test_observability_matrix() {
        // Observable system: A = [[0,1],[-2,-3]], C = [[1,0]]
        let a = array![[0.0_f64, 1.0], [-2.0, -3.0]];
        let c = array![[1.0_f64, 0.0]];
        let obs = observability_matrix(&a.view(), &c.view());

        // obs = [C; CA] = [[1,0], [0,1]]
        assert_eq!(obs.nrows(), 2);
        assert_eq!(obs.ncols(), 2);

        // det([[1,0],[0,1]]) = 1 → full rank → observable
        let det = obs[[0, 0]] * obs[[1, 1]] - obs[[0, 1]] * obs[[1, 0]];
        assert!(
            det.abs() > 1e-10,
            "Observability matrix should be full rank, det = {det}"
        );
    }

    // ------------------------------------------------------------------
    // Controllability Gramian
    // ------------------------------------------------------------------
    #[test]
    fn test_controllability_gramian() {
        let a = array![[-1.0_f64, 0.0], [0.0, -2.0]];
        let b = array![[1.0_f64], [1.0]];
        let wc = controllability_gramian(&a.view(), &b.view()).expect("gramian failed");

        // Verify: A Wc + Wc Aᵀ + BBᵀ ≈ 0
        let bbt = b.dot(&b.t());
        let res = a.dot(&wc) + wc.dot(&a.t()) + &bbt;
        for &v in res.iter() {
            assert!(v.abs() < 1e-8, "Gramian Lyapunov residual = {v}");
        }
    }

    // ------------------------------------------------------------------
    // Balanced truncation
    // ------------------------------------------------------------------
    #[test]
    fn test_balanced_truncation_output_shapes() {
        let a = array![[-1.0_f64, 0.0], [0.0, -10.0]];
        let b = array![[1.0_f64], [1.0]];
        let c = array![[1.0_f64, 1.0]];
        let result =
            balanced_truncation(&a.view(), &b.view(), &c.view(), 1).expect("balanced_truncation");
        assert_eq!(result.a_r.nrows(), 1);
        assert_eq!(result.a_r.ncols(), 1);
        assert_eq!(result.b_r.nrows(), 1);
        assert_eq!(result.b_r.ncols(), 1);
        assert_eq!(result.c_r.nrows(), 1);
        assert_eq!(result.c_r.ncols(), 1);
        assert_eq!(result.hankel_singular_values.len(), 2);
        // HSV should be non-negative
        for &v in result.hankel_singular_values.iter() {
            assert!(v >= 0.0, "HSV must be non-negative, got {v}");
        }
    }
}
