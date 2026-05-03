//! Krylov subspace methods for dense linear systems
//!
//! This module implements classical Krylov subspace iterative solvers for
//! dense linear systems Ax = b. These complement sparse-specific solvers by
//! providing preconditioned and restarted variants for dense problems.
//!
//! ## Algorithms implemented
//!
//! - **CG (Conjugate Gradient)**: For symmetric positive definite (SPD) systems,
//!   based on the Hestenes-Stiefel 1952 three-term recurrence. Supports
//!   optional preconditioning via an arbitrary operator.
//! - **GMRES (restarted)**: For general non-symmetric systems. Uses the Arnoldi
//!   process with modified Gram-Schmidt orthogonalisation and Givens rotations
//!   to solve the projected least-squares problem.
//! - **BiCGSTAB**: For non-symmetric systems. Uses Van der Vorst's stabilization
//!   to prevent the irregular convergence behaviour of BiCG.
//! - **MINRES**: For symmetric (possibly indefinite) systems. Based on the
//!   Paige-Saunders 1975 Lanczos-based algorithm.
//!
//! Convergence criterion for all solvers: `‖r‖ / ‖b‖ < tol`.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};
use crate::validation::{validate_iteration_parameters, validate_linear_system};

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result returned by every Krylov solver.
#[derive(Debug, Clone)]
pub struct IterativeSolveResult<T> {
    /// Approximate solution vector.
    pub x: Array1<T>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Euclidean norm of the final residual.
    pub residual_norm: T,
    /// Whether the tolerance was reached before `max_iter`.
    pub converged: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix-vector product  y = A · x.
fn matvec<F>(a: &Array2<F>, x: &Array1<F>) -> Array1<F>
where
    F: Float + NumAssign + Sum + ScalarOperand,
{
    let n = a.nrows();
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut acc = F::zero();
        for j in 0..a.ncols() {
            acc += a[[i, j]] * x[j];
        }
        y[i] = acc;
    }
    y
}

/// Euclidean norm of a vector.
#[inline]
fn norm2<F>(v: &Array1<F>) -> F
where
    F: Float + NumAssign + Sum + ScalarOperand,
{
    let sq: F = v.iter().map(|&x| x * x).fold(F::zero(), |a, b| a + b);
    sq.sqrt()
}

/// Dot product of two vectors.
#[inline]
fn dot<F>(a: &Array1<F>, b: &Array1<F>) -> F
where
    F: Float + NumAssign + Sum + ScalarOperand,
{
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .fold(F::zero(), |acc, v| acc + v)
}

// ─────────────────────────────────────────────────────────────────────────────
// Conjugate Gradient
// ─────────────────────────────────────────────────────────────────────────────

/// Conjugate Gradient solver for symmetric positive definite systems.
///
/// Implements the classic Hestenes-Stiefel (1952) three-term recurrence with
/// optional left preconditioning. For a preconditioned solve the system solved
/// is conceptually `M⁻¹ A x = M⁻¹ b` where `preconditioner(v)` approximates
/// `M⁻¹ v`.
///
/// # Arguments
///
/// * `a` – Square symmetric positive definite matrix.
/// * `b` – Right-hand side vector.
/// * `x0` – Optional initial guess. Defaults to zero.
/// * `tol` – Relative residual tolerance: `‖r‖/‖b‖ < tol`.
/// * `max_iter` – Maximum iteration count.
/// * `preconditioner` – Optional preconditioning operator `M⁻¹`.
///
/// # Returns
///
/// An [`IterativeSolveResult`] containing the solution and convergence info.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::iterative::conjugate_gradient;
///
/// let a = array![[4.0_f64, 1.0], [1.0, 3.0]];
/// let b = array![1.0_f64, 2.0];
/// let result = conjugate_gradient(&a, &b.view(), None, 1e-12, 100, None)
///     .expect("CG failed");
/// assert!(result.converged);
/// ```
#[allow(clippy::type_complexity)]
pub fn conjugate_gradient<F>(
    a: &Array2<F>,
    b: &ArrayView1<F>,
    x0: Option<&Array1<F>>,
    tol: F,
    max_iter: usize,
    preconditioner: Option<&dyn Fn(&Array1<F>) -> Array1<F>>,
) -> LinalgResult<IterativeSolveResult<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static + std::fmt::Debug,
{
    validate_linear_system(&a.view(), b, "Conjugate Gradient")?;
    validate_iteration_parameters(max_iter, tol, "Conjugate Gradient")?;

    let n = a.nrows();

    // Initial guess
    let mut x: Array1<F> = match x0 {
        Some(x_init) => {
            if x_init.len() != n {
                return Err(LinalgError::DimensionError(format!(
                    "Initial guess length {} does not match system dimension {}",
                    x_init.len(),
                    n
                )));
            }
            x_init.clone()
        }
        None => Array1::zeros(n),
    };

    // Compute ‖b‖ for relative tolerance
    let b_norm = norm2(&b.to_owned());
    if b_norm <= F::epsilon() {
        // b is zero → x = 0 is the solution
        return Ok(IterativeSolveResult {
            x: Array1::zeros(n),
            iterations: 0,
            residual_norm: F::zero(),
            converged: true,
        });
    }

    let abs_tol = tol * b_norm;

    // r = b - A·x
    let ax = matvec(a, &x);
    let mut r: Array1<F> = Array1::from_iter(b.iter().zip(ax.iter()).map(|(&bi, &axi)| bi - axi));

    // Preconditioned direction z = M⁻¹ r
    let mut z: Array1<F> = match &preconditioner {
        Some(m_inv) => m_inv(&r),
        None => r.clone(),
    };

    let mut p = z.clone();
    let mut rz_old = dot(&r, &z);

    let mut residual_norm = norm2(&r);

    if residual_norm <= abs_tol {
        return Ok(IterativeSolveResult {
            x,
            iterations: 0,
            residual_norm,
            converged: true,
        });
    }

    let mut iter_count = 0usize;

    for iter in 0..max_iter {
        iter_count = iter + 1;

        let ap = matvec(a, &p);
        let pap = dot(&p, &ap);

        if pap.abs() < F::epsilon() {
            // Breakdown: p is in the null space of A
            break;
        }

        let alpha = rz_old / pap;

        // x = x + α·p
        for i in 0..n {
            x[i] += alpha * p[i];
        }

        // r = r - α·A·p
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }

        residual_norm = norm2(&r);

        if residual_norm <= abs_tol {
            return Ok(IterativeSolveResult {
                x,
                iterations: iter_count,
                residual_norm,
                converged: true,
            });
        }

        // z = M⁻¹ r
        z = match &preconditioner {
            Some(m_inv) => m_inv(&r),
            None => r.clone(),
        };

        let rz_new = dot(&r, &z);
        let beta = rz_new / rz_old;

        // p = z + β·p
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }

        rz_old = rz_new;
    }

    Ok(IterativeSolveResult {
        x,
        iterations: iter_count,
        residual_norm,
        converged: false,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// GMRES (restarted)
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a single Givens rotation in-place: `[c, s; -s, c] · [a; b]`.
fn apply_givens_rotation<F>(h: &mut [F], cs: &[F], sn: &[F], k: usize)
where
    F: Float + NumAssign,
{
    for i in 0..k {
        let h0 = h[i];
        let h1 = h[i + 1];
        h[i] = cs[i] * h0 + sn[i] * h1;
        h[i + 1] = -sn[i] * h0 + cs[i] * h1;
    }
}

/// Compute Givens coefficients to zero out the sub-diagonal of h[k+1, k].
fn compute_givens<F>(h_k: F, h_k1: F) -> (F, F)
where
    F: Float,
{
    let t = (h_k * h_k + h_k1 * h_k1).sqrt();
    if t < F::epsilon() {
        (F::one(), F::zero())
    } else {
        (h_k / t, h_k1 / t)
    }
}

/// Restarted GMRES solver for general (non-symmetric) linear systems.
///
/// Uses the Arnoldi process with modified Gram-Schmidt orthogonalization and
/// Givens rotations to solve the projected least-squares problem. The outer
/// loop restarts the Krylov basis every `restart` steps.
///
/// # Arguments
///
/// * `a` – Square matrix (need not be symmetric).
/// * `b` – Right-hand side vector.
/// * `x0` – Optional initial guess. Defaults to zero.
/// * `tol` – Relative residual tolerance: `‖r‖/‖b‖ < tol`.
/// * `max_iter` – Maximum *outer* restart iterations.
/// * `restart` – Krylov subspace dimension before restart (Arnoldi restart).
///
/// # Returns
///
/// An [`IterativeSolveResult`] containing the solution and convergence info.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::iterative::gmres;
///
/// let a = array![[3.0_f64, 1.0], [1.0, 4.0]];
/// let b = array![5.0_f64, 6.0];
/// let result = gmres(&a, &b.view(), None, 1e-12, 50, 10)
///     .expect("GMRES failed");
/// assert!(result.converged);
/// ```
pub fn gmres<F>(
    a: &Array2<F>,
    b: &ArrayView1<F>,
    x0: Option<&Array1<F>>,
    tol: F,
    max_iter: usize,
    restart: usize,
) -> LinalgResult<IterativeSolveResult<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static + std::fmt::Debug,
{
    validate_linear_system(&a.view(), b, "GMRES")?;
    validate_iteration_parameters(max_iter, tol, "GMRES")?;

    if restart == 0 {
        return Err(LinalgError::InvalidInputError(
            "GMRES restart parameter must be positive".to_string(),
        ));
    }

    let n = a.nrows();

    let mut x: Array1<F> = match x0 {
        Some(x_init) => {
            if x_init.len() != n {
                return Err(LinalgError::DimensionError(format!(
                    "Initial guess length {} does not match system dimension {}",
                    x_init.len(),
                    n
                )));
            }
            x_init.clone()
        }
        None => Array1::zeros(n),
    };

    let b_norm = norm2(&b.to_owned());
    if b_norm <= F::epsilon() {
        return Ok(IterativeSolveResult {
            x: Array1::zeros(n),
            iterations: 0,
            residual_norm: F::zero(),
            converged: true,
        });
    }

    let abs_tol = tol * b_norm;
    let mut total_iters = 0usize;
    let mut residual_norm = F::zero();

    'outer: for outer in 0..max_iter {
        let _ = outer; // suppress unused warning

        // r = b - A·x
        let ax = matvec(a, &x);
        let r: Array1<F> = Array1::from_iter(b.iter().zip(ax.iter()).map(|(&bi, &axi)| bi - axi));
        let beta = norm2(&r);

        residual_norm = beta;
        if beta <= abs_tol {
            return Ok(IterativeSolveResult {
                x,
                iterations: total_iters,
                residual_norm,
                converged: true,
            });
        }

        // Krylov basis Q (columns are basis vectors)
        let m = restart;
        let mut q: Vec<Array1<F>> = Vec::with_capacity(m + 1);

        // q₀ = r / ‖r‖
        let q0: Array1<F> = r.mapv(|v| v / beta);
        q.push(q0);

        // Upper Hessenberg matrix H (stored as column-major vectors of length m+1)
        let mut hess: Vec<Vec<F>> = Vec::with_capacity(m);

        // Givens rotation sines / cosines
        let mut cs: Vec<F> = Vec::with_capacity(m);
        let mut sn: Vec<F> = Vec::with_capacity(m);

        // Right-hand side of least-squares: e₁ · β
        let mut g: Vec<F> = Vec::with_capacity(m + 1);
        g.push(beta);

        let mut inner_iters = 0usize;

        for j in 0..m {
            total_iters += 1;

            // w = A · q[j]
            let mut w = matvec(a, &q[j]);

            // Modified Gram-Schmidt orthogonalisation
            let mut h_col: Vec<F> = vec![F::zero(); j + 2];
            for (i, qi) in q.iter().enumerate().take(j + 1) {
                let h_ij = dot(&w, qi);
                h_col[i] = h_ij;
                for k in 0..n {
                    w[k] -= h_ij * qi[k];
                }
            }
            let h_j1j = norm2(&w);
            h_col[j + 1] = h_j1j;

            // Apply previous Givens rotations to h_col[0..j+1]
            apply_givens_rotation(&mut h_col, &cs, &sn, j);

            // Compute new Givens rotation to zero h_col[j+1]
            let (c, s) = compute_givens(h_col[j], h_col[j + 1]);
            cs.push(c);
            sn.push(s);

            // Apply new rotation to h_col and g
            h_col[j] = c * h_col[j] + s * h_col[j + 1];
            h_col[j + 1] = F::zero();

            // Extend g
            let g_j = g[j];
            g.push(-sn[j] * g_j);
            g[j] = cs[j] * g_j;

            hess.push(h_col);

            residual_norm = g[j + 1].abs();
            inner_iters = j + 1;

            // Add next Krylov vector if not the last step and h_j1j > 0
            if h_j1j > F::epsilon() && j < m - 1 {
                let q_next: Array1<F> = w.mapv(|v| v / h_j1j);
                q.push(q_next);
            } else if h_j1j <= F::epsilon() {
                // Lucky breakdown – exact solution found in Krylov space
                break;
            } else {
                // Last iteration of this cycle
                let q_next: Array1<F> = w.mapv(|v| v / h_j1j);
                q.push(q_next);
            }

            if residual_norm <= abs_tol {
                break;
            }
        }

        // Back-substitution to compute y from upper triangular R (stored in hess)
        // R y = g[0..inner_iters]
        let mut y: Vec<F> = vec![F::zero(); inner_iters];
        for i in (0..inner_iters).rev() {
            let mut sum = g[i];
            for k in (i + 1)..inner_iters {
                sum -= hess[k][i] * y[k];
            }
            let diag = hess[i][i];
            if diag.abs() < F::epsilon() {
                y[i] = F::zero();
            } else {
                y[i] = sum / diag;
            }
        }

        // Update solution x += Q[:, 0..m] · y
        for (i, yi) in y.iter().enumerate() {
            for k in 0..n {
                x[k] += *yi * q[i][k];
            }
        }

        if residual_norm <= abs_tol {
            return Ok(IterativeSolveResult {
                x,
                iterations: total_iters,
                residual_norm,
                converged: true,
            });
        }

        // Check outer iteration limit
        if total_iters >= max_iter * restart {
            break 'outer;
        }
    }

    Ok(IterativeSolveResult {
        x,
        iterations: total_iters,
        residual_norm,
        converged: false,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// BiCGSTAB
// ─────────────────────────────────────────────────────────────────────────────

/// BiCGSTAB solver for general non-symmetric linear systems.
///
/// Implements Van der Vorst's (1992) BiConjugate Gradient Stabilized algorithm.
/// Compared to BiCG it avoids the erratic convergence behaviour by applying a
/// second stabilization polynomial. Works well for mildly non-symmetric and
/// non-Hermitian systems.
///
/// # Arguments
///
/// * `a` – Square matrix.
/// * `b` – Right-hand side vector.
/// * `x0` – Optional initial guess. Defaults to zero.
/// * `tol` – Relative residual tolerance: `‖r‖/‖b‖ < tol`.
/// * `max_iter` – Maximum number of BiCGSTAB steps.
///
/// # Returns
///
/// An [`IterativeSolveResult`] containing the solution and convergence info.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::iterative::bicgstab;
///
/// let a = array![[5.0_f64, 1.0], [1.0, 4.0]];
/// let b = array![6.0_f64, 5.0];
/// let result = bicgstab(&a, &b.view(), None, 1e-12, 100)
///     .expect("BiCGSTAB failed");
/// assert!(result.converged);
/// ```
pub fn bicgstab<F>(
    a: &Array2<F>,
    b: &ArrayView1<F>,
    x0: Option<&Array1<F>>,
    tol: F,
    max_iter: usize,
) -> LinalgResult<IterativeSolveResult<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static + std::fmt::Debug,
{
    validate_linear_system(&a.view(), b, "BiCGSTAB")?;
    validate_iteration_parameters(max_iter, tol, "BiCGSTAB")?;

    let n = a.nrows();

    let mut x: Array1<F> = match x0 {
        Some(x_init) => {
            if x_init.len() != n {
                return Err(LinalgError::DimensionError(format!(
                    "Initial guess length {} does not match system dimension {}",
                    x_init.len(),
                    n
                )));
            }
            x_init.clone()
        }
        None => Array1::zeros(n),
    };

    let b_norm = norm2(&b.to_owned());
    if b_norm <= F::epsilon() {
        return Ok(IterativeSolveResult {
            x: Array1::zeros(n),
            iterations: 0,
            residual_norm: F::zero(),
            converged: true,
        });
    }

    let abs_tol = tol * b_norm;

    // r = b - A·x
    let ax = matvec(a, &x);
    let r: Array1<F> = Array1::from_iter(b.iter().zip(ax.iter()).map(|(&bi, &axi)| bi - axi));

    let mut residual_norm = norm2(&r);
    if residual_norm <= abs_tol {
        return Ok(IterativeSolveResult {
            x,
            iterations: 0,
            residual_norm,
            converged: true,
        });
    }

    // r̂ = r (arbitrary, conventionally the initial residual)
    let r_hat = r.clone();
    let mut r = r;

    let mut p: Array1<F> = r.clone();
    let mut rho_old = F::one();
    let mut alpha = F::one();
    let mut omega = F::one();
    let mut v: Array1<F> = Array1::zeros(n);

    let mut iter_count = 0usize;

    for iter in 0..max_iter {
        iter_count = iter + 1;

        let rho_new = dot(&r_hat, &r);

        if rho_new.abs() < F::epsilon() {
            // Breakdown: r̂ and r are orthogonal, restart with new r̂
            // For simplicity, report non-convergence at this point
            break;
        }

        let beta = (rho_new / rho_old) * (alpha / omega);

        // p = r + β·(p - ω·v)
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        v = matvec(a, &p);
        let r_hat_v = dot(&r_hat, &v);

        if r_hat_v.abs() < F::epsilon() {
            break;
        }

        alpha = rho_new / r_hat_v;

        // s = r - α·v
        let mut s: Array1<F> = Array1::zeros(n);
        for i in 0..n {
            s[i] = r[i] - alpha * v[i];
        }

        residual_norm = norm2(&s);
        if residual_norm <= abs_tol {
            // x = x + α·p
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            return Ok(IterativeSolveResult {
                x,
                iterations: iter_count,
                residual_norm,
                converged: true,
            });
        }

        let t = matvec(a, &s);
        let t_norm_sq: F = t
            .iter()
            .map(|&ti| ti * ti)
            .fold(F::zero(), |acc, v| acc + v);

        omega = if t_norm_sq < F::epsilon() {
            F::zero()
        } else {
            dot(&t, &s) / t_norm_sq
        };

        // x = x + α·p + ω·s
        for i in 0..n {
            x[i] += alpha * p[i] + omega * s[i];
        }

        // r = s - ω·t
        for i in 0..n {
            r[i] = s[i] - omega * t[i];
        }

        residual_norm = norm2(&r);

        if residual_norm <= abs_tol {
            return Ok(IterativeSolveResult {
                x,
                iterations: iter_count,
                residual_norm,
                converged: true,
            });
        }

        if omega.abs() < F::epsilon() {
            break;
        }

        rho_old = rho_new;
    }

    Ok(IterativeSolveResult {
        x,
        iterations: iter_count,
        residual_norm,
        converged: false,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// MINRES
// ─────────────────────────────────────────────────────────────────────────────

/// MINRES solver for symmetric (possibly indefinite) linear systems.
///
/// Based on the Paige-Saunders (1975) Lanczos-based MINRES algorithm.
/// Unlike CG, MINRES does not require positive definiteness and minimises
/// ‖r_k‖₂ over the Krylov subspace at each step.
///
/// # Arguments
///
/// * `a` – Square symmetric matrix (may be indefinite).
/// * `b` – Right-hand side vector.
/// * `x0` – Optional initial guess. Defaults to zero.
/// * `tol` – Relative residual tolerance: `‖r‖/‖b‖ < tol`.
/// * `max_iter` – Maximum number of MINRES steps.
///
/// # Returns
///
/// An [`IterativeSolveResult`] containing the solution and convergence info.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::iterative::minres;
///
/// // Symmetric indefinite system
/// let a = array![[2.0_f64, 1.0], [1.0, -1.0]];
/// let b = array![3.0_f64, 0.0];
/// let result = minres(&a, &b.view(), None, 1e-10, 100)
///     .expect("MINRES failed");
/// assert!(result.converged);
/// ```
pub fn minres<F>(
    a: &Array2<F>,
    b: &ArrayView1<F>,
    x0: Option<&Array1<F>>,
    tol: F,
    max_iter: usize,
) -> LinalgResult<IterativeSolveResult<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static + std::fmt::Debug,
{
    validate_linear_system(&a.view(), b, "MINRES")?;
    validate_iteration_parameters(max_iter, tol, "MINRES")?;

    let n = a.nrows();

    let x_init: Array1<F> = match x0 {
        Some(xi) => {
            if xi.len() != n {
                return Err(LinalgError::DimensionError(format!(
                    "Initial guess length {} does not match system dimension {}",
                    xi.len(),
                    n
                )));
            }
            xi.clone()
        }
        None => Array1::zeros(n),
    };

    let b_norm = norm2(&b.to_owned());
    if b_norm <= F::epsilon() {
        return Ok(IterativeSolveResult {
            x: Array1::zeros(n),
            iterations: 0,
            residual_norm: F::zero(),
            converged: true,
        });
    }

    let abs_tol = tol * b_norm;

    // Initialise Lanczos recurrence
    // r = b - A·x0
    let ax0 = matvec(a, &x_init);
    let r1: Array1<F> = Array1::from_iter(b.iter().zip(ax0.iter()).map(|(&bi, &axi)| bi - axi));

    let mut beta1 = norm2(&r1);
    let mut residual_norm = beta1;

    if residual_norm <= abs_tol {
        return Ok(IterativeSolveResult {
            x: x_init,
            iterations: 0,
            residual_norm,
            converged: true,
        });
    }

    // Normalise first Lanczos vector
    let mut v_old: Array1<F> = Array1::zeros(n);
    let mut v: Array1<F> = r1.mapv(|vi| vi / beta1);

    // Solution update vectors: MINRES needs w_{k}, w_{k-1}, w_{k-2}
    // w_old = w_{k-2}, w_cur = w_{k-1}, w_new = w_k (computed each iteration)
    let mut w_old: Array1<F> = Array1::zeros(n);
    let mut w_cur: Array1<F> = Array1::zeros(n);

    let mut x = x_init;

    // Lanczos scalars
    let mut beta = beta1;
    // Givens rotation history: we maintain two previous rotation pairs
    let mut c_old = F::one(); // c_{k-2}
    let mut s_old = F::zero(); // s_{k-2}
    let mut c = F::one(); // c_{k-1}
    let mut s = F::zero(); // s_{k-1}

    // QR factorization tracking
    let mut rho_bar: F;
    let mut phi_bar = beta1;
    let mut delta_bar: F;

    let mut iter_count = 0usize;

    for iter in 0..max_iter {
        iter_count = iter + 1;

        // ── Lanczos step ────────────────────────────────────────────────────
        let av = matvec(a, &v);
        let alpha = dot(&v, &av);

        // v_new = A·v - alpha·v - beta·v_old
        let mut v_new: Array1<F> = Array1::zeros(n);
        for i in 0..n {
            v_new[i] = av[i] - alpha * v[i] - beta * v_old[i];
        }

        let beta_new = norm2(&v_new);

        // ── Apply previous two Givens rotations to new Hessenberg column ───
        // The 3-element column from the Lanczos recurrence for step k is:
        //   [beta_{k-1}, alpha_k, beta_k]  (tridiagonal)
        // We need to apply rotations G_{k-2} and G_{k-1} before computing G_k.

        // Apply G_{k-2}
        let eps1 = s_old * beta;
        let delta1 = c_old * beta;

        // Apply G_{k-1}
        let eps2 = s * delta1;
        delta_bar = -c * delta1;
        rho_bar = c * alpha - s * eps2;
        let _rho_bar_orig = rho_bar; // for debugging

        // Now compute G_k to zero out beta_new
        let gamma = (rho_bar * rho_bar + beta_new * beta_new).sqrt();
        let gamma_safe = if gamma < F::epsilon() {
            F::epsilon()
        } else {
            gamma
        };

        let c_new = rho_bar / gamma_safe;
        let s_new = beta_new / gamma_safe;

        // ── Solution update ─────────────────────────────────────────────────
        let phi = c_new * phi_bar;
        phi_bar = s_new * phi_bar;

        // w_new = (v - eps1 * w_old - delta_bar * w_cur) / gamma_safe
        // This is the proper three-term recurrence for the w vectors
        let mut w_new: Array1<F> = Array1::zeros(n);
        for i in 0..n {
            w_new[i] = (v[i] - eps1 * w_old[i] - delta_bar * w_cur[i]) / gamma_safe;
        }

        // x = x + phi * w_new
        for i in 0..n {
            x[i] += phi * w_new[i];
        }

        residual_norm = phi_bar.abs();

        if residual_norm <= abs_tol {
            return Ok(IterativeSolveResult {
                x,
                iterations: iter_count,
                residual_norm,
                converged: true,
            });
        }

        // ── Breakdown check ─────────────────────────────────────────────────
        if beta_new < F::epsilon() {
            // Lanczos breakdown: A-invariant subspace found.
            // The current iterate is the best we can get.
            break;
        }

        // ── Advance recurrence ──────────────────────────────────────────────
        v_old = v;
        v = v_new.mapv(|vi| vi / beta_new);

        w_old = w_cur;
        w_cur = w_new;

        beta = beta_new;
        c_old = c;
        s_old = s;
        c = c_new;
        s = s_new;
    }

    Ok(IterativeSolveResult {
        x,
        iterations: iter_count,
        residual_norm,
        converged: false,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    // ── helper: compute ‖A·x - b‖ ──
    fn residual_norm_check(a: &Array2<f64>, x: &Array1<f64>, b: &[f64]) -> f64 {
        let n = a.nrows();
        let mut norm = 0.0_f64;
        for i in 0..n {
            let mut ai_x = 0.0_f64;
            for j in 0..n {
                ai_x += a[[i, j]] * x[j];
            }
            let diff = ai_x - b[i];
            norm += diff * diff;
        }
        norm.sqrt()
    }

    // ════════════════════════════════════════════════════
    // CG tests
    // ════════════════════════════════════════════════════

    #[test]
    fn test_cg_simple_2x2() {
        // [[4,1],[1,3]] x = [1,2] → x = [1/11, 7/11]
        let a = array![[4.0_f64, 1.0], [1.0, 3.0]];
        let b = array![1.0_f64, 2.0];
        let result =
            conjugate_gradient(&a, &b.view(), None, 1e-12, 100, None).expect("CG must succeed");
        assert!(result.converged, "CG should converge on SPD 2×2");
        assert!(
            residual_norm_check(&a, &result.x, &[1.0, 2.0]) < 1e-10,
            "CG residual too large"
        );
    }

    #[test]
    fn test_cg_identity() {
        // I·x = b → x = b
        let a = Array2::eye(4);
        let b = array![1.0_f64, 2.0, 3.0, 4.0];
        let result = conjugate_gradient(&a, &b.view(), None, 1e-12, 100, None)
            .expect("CG on identity must succeed");
        assert!(result.converged);
        for i in 0..4 {
            assert_relative_eq!(result.x[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cg_diagonal_spd() {
        // Diagonal SPD system
        let mut a = Array2::zeros((5, 5));
        for i in 0..5 {
            a[[i, i]] = (i as f64 + 1.0) * 2.0;
        }
        let b = array![2.0_f64, 4.0, 6.0, 8.0, 10.0];
        let result = conjugate_gradient(&a, &b.view(), None, 1e-12, 100, None)
            .expect("CG on diagonal SPD must succeed");
        assert!(result.converged);
        for i in 0..5 {
            assert_relative_eq!(result.x[i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cg_with_initial_guess() {
        let a = array![[4.0_f64, 1.0], [1.0, 3.0]];
        let b = array![1.0_f64, 2.0];
        let x0 = array![0.5_f64, 0.5];
        let result = conjugate_gradient(&a, &b.view(), Some(&x0), 1e-12, 100, None)
            .expect("CG with initial guess must succeed");
        assert!(result.converged);
        assert!(residual_norm_check(&a, &result.x, &[1.0, 2.0]) < 1e-10);
    }

    #[test]
    fn test_cg_with_diagonal_preconditioner() {
        // Jacobi preconditioner: M⁻¹ v = v / diag(A)
        let a = array![[10.0_f64, 1.0], [1.0, 5.0]];
        let b = array![11.0_f64, 6.0];
        let diag = [10.0_f64, 5.0];
        let precond = move |v: &Array1<f64>| -> Array1<f64> {
            Array1::from_iter(v.iter().zip(diag.iter()).map(|(&vi, &di)| vi / di))
        };
        let result = conjugate_gradient(&a, &b.view(), None, 1e-12, 100, Some(&precond))
            .expect("Preconditioned CG must succeed");
        assert!(result.converged);
        for i in 0..2 {
            assert_relative_eq!(result.x[i], 1.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_cg_zero_rhs() {
        let a = array![[4.0_f64, 1.0], [1.0, 3.0]];
        let b = array![0.0_f64, 0.0];
        let result = conjugate_gradient(&a, &b.view(), None, 1e-12, 100, None)
            .expect("CG zero rhs must succeed");
        assert!(result.converged);
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-14);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_cg_3x3_spd() {
        // Tridiagonal SPD: [[4,-1,0],[-1,4,-1],[0,-1,4]]
        let a = array![[4.0_f64, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 4.0]];
        let b = array![1.0_f64, 0.0, 1.0];
        let result =
            conjugate_gradient(&a, &b.view(), None, 1e-12, 100, None).expect("CG 3×3 must succeed");
        assert!(result.converged);
        assert!(residual_norm_check(&a, &result.x, &[1.0, 0.0, 1.0]) < 1e-10);
    }

    #[test]
    fn test_cg_convergence_iterations() {
        // CG converges in at most n steps for n-dim SPD (exact arithmetic)
        let n = 5usize;
        let mut a = Array2::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = 4.0;
            if i > 0 {
                a[[i, i - 1]] = -1.0;
                a[[i - 1, i]] = -1.0;
            }
        }
        let b = Array1::from_iter((0..n).map(|i| i as f64 + 1.0));
        let result =
            conjugate_gradient(&a, &b.view(), None, 1e-10, 50, None).expect("CG must succeed");
        assert!(result.converged);
        assert!(
            result.iterations <= n + 1,
            "CG should converge in ≤ n steps"
        );
    }

    // ════════════════════════════════════════════════════
    // GMRES tests
    // ════════════════════════════════════════════════════

    #[test]
    fn test_gmres_symmetric() {
        let a = array![[3.0_f64, 1.0], [1.0, 4.0]];
        let b = array![5.0_f64, 6.0];
        let result = gmres(&a, &b.view(), None, 1e-12, 50, 10).expect("GMRES must succeed");
        assert!(result.converged);
        assert!(residual_norm_check(&a, &result.x, &[5.0, 6.0]) < 1e-9);
    }

    #[test]
    fn test_gmres_nonsymmetric() {
        // Non-symmetric matrix
        let a = array![[4.0_f64, 2.0], [1.0, 3.0]];
        let b = array![8.0_f64, 5.0];
        let result =
            gmres(&a, &b.view(), None, 1e-12, 50, 10).expect("GMRES non-symmetric must succeed");
        assert!(result.converged);
        assert!(residual_norm_check(&a, &result.x, &[8.0, 5.0]) < 1e-9);
    }

    #[test]
    fn test_gmres_identity() {
        let a: Array2<f64> = Array2::eye(3);
        let b = array![1.0_f64, 2.0, 3.0];
        let result = gmres(&a, &b.view(), None, 1e-12, 50, 5).expect("GMRES identity must succeed");
        assert!(result.converged);
        for i in 0..3 {
            assert_relative_eq!(result.x[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gmres_restart_parameter() {
        // Test with small restart = 2 (forces multiple outer iterations)
        let a = array![[4.0_f64, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 5.0]];
        let b = array![5.0_f64, 5.0, 6.0];
        let result = gmres(&a, &b.view(), None, 1e-10, 50, 2)
            .expect("GMRES with small restart must succeed");
        assert!(result.converged || result.residual_norm < 1e-8);
    }

    #[test]
    fn test_gmres_zero_rhs() {
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
        let b = array![0.0_f64, 0.0];
        let result = gmres(&a, &b.view(), None, 1e-12, 50, 5).expect("GMRES zero rhs must succeed");
        assert!(result.converged);
    }

    #[test]
    fn test_gmres_upper_triangular() {
        // Upper triangular non-symmetric
        let a = array![[2.0_f64, 3.0], [0.0, 4.0]];
        let b = array![11.0_f64, 8.0];
        let result =
            gmres(&a, &b.view(), None, 1e-12, 50, 5).expect("GMRES upper-triangular must succeed");
        assert!(result.converged);
        assert!(residual_norm_check(&a, &result.x, &[11.0, 8.0]) < 1e-9);
    }

    // ════════════════════════════════════════════════════
    // BiCGSTAB tests
    // ════════════════════════════════════════════════════

    #[test]
    fn test_bicgstab_symmetric() {
        let a = array![[4.0_f64, 1.0], [1.0, 3.0]];
        let b = array![1.0_f64, 2.0];
        let result = bicgstab(&a, &b.view(), None, 1e-12, 100).expect("BiCGSTAB must succeed");
        assert!(result.converged);
        assert!(residual_norm_check(&a, &result.x, &[1.0, 2.0]) < 1e-10);
    }

    #[test]
    fn test_bicgstab_nonsymmetric() {
        let a = array![[5.0_f64, 2.0], [1.0, 4.0]];
        let b = array![12.0_f64, 9.0];
        let result =
            bicgstab(&a, &b.view(), None, 1e-12, 100).expect("BiCGSTAB non-symmetric must succeed");
        assert!(result.converged);
        assert!(residual_norm_check(&a, &result.x, &[12.0, 9.0]) < 1e-9);
    }

    #[test]
    fn test_bicgstab_3x3() {
        let a = array![[6.0_f64, 2.0, 1.0], [2.0, 5.0, 1.0], [1.0, 1.0, 4.0]];
        let b = array![9.0_f64, 8.0, 6.0];
        let result = bicgstab(&a, &b.view(), None, 1e-12, 100).expect("BiCGSTAB 3×3 must succeed");
        assert!(result.converged);
        assert!(residual_norm_check(&a, &result.x, &[9.0, 8.0, 6.0]) < 1e-9);
    }

    #[test]
    fn test_bicgstab_identity() {
        let a: Array2<f64> = Array2::eye(4);
        let b = array![1.0_f64, 2.0, 3.0, 4.0];
        let result =
            bicgstab(&a, &b.view(), None, 1e-12, 100).expect("BiCGSTAB identity must succeed");
        assert!(result.converged);
        for i in 0..4 {
            assert_relative_eq!(result.x[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bicgstab_with_initial_guess() {
        let a = array![[4.0_f64, 1.0], [1.0, 3.0]];
        let b = array![1.0_f64, 2.0];
        let x0 = array![0.2_f64, 0.6];
        let result = bicgstab(&a, &b.view(), Some(&x0), 1e-12, 100)
            .expect("BiCGSTAB with initial guess must succeed");
        assert!(result.converged);
        assert!(residual_norm_check(&a, &result.x, &[1.0, 2.0]) < 1e-10);
    }

    #[test]
    fn test_bicgstab_zero_rhs() {
        let a = array![[3.0_f64, 1.0], [1.0, 2.0]];
        let b = array![0.0_f64, 0.0];
        let result =
            bicgstab(&a, &b.view(), None, 1e-12, 100).expect("BiCGSTAB zero rhs must succeed");
        assert!(result.converged);
    }

    // ════════════════════════════════════════════════════
    // MINRES tests
    // ════════════════════════════════════════════════════

    #[test]
    fn test_minres_spd() {
        // MINRES should also work on SPD systems
        let a = array![[4.0_f64, 1.0], [1.0, 3.0]];
        let b = array![1.0_f64, 2.0];
        let result = minres(&a, &b.view(), None, 1e-10, 200).expect("MINRES on SPD must succeed");
        assert!(
            result.converged || result.residual_norm < 1e-8,
            "MINRES residual {}",
            result.residual_norm
        );
    }

    #[test]
    fn test_minres_symmetric_indefinite() {
        // Symmetric indefinite 2×2
        let a = array![[2.0_f64, 1.0], [1.0, -1.0]];
        let b = array![3.0_f64, 0.0];
        let result = minres(&a, &b.view(), None, 1e-10, 200)
            .expect("MINRES symmetric indefinite must succeed");
        // Verify residual quality
        let res = residual_norm_check(&a, &result.x, &[3.0, 0.0]);
        assert!(res < 1e-8 || result.converged, "MINRES residual {res}");
    }

    #[test]
    fn test_minres_identity() {
        let a: Array2<f64> = Array2::eye(3);
        let b = array![2.0_f64, 4.0, 6.0];
        let result = minres(&a, &b.view(), None, 1e-12, 100).expect("MINRES identity must succeed");
        assert!(result.converged);
        for i in 0..3 {
            assert_relative_eq!(result.x[i], b[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_minres_zero_rhs() {
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
        let b = array![0.0_f64, 0.0];
        let result = minres(&a, &b.view(), None, 1e-12, 100).expect("MINRES zero rhs must succeed");
        assert!(result.converged);
    }

    #[test]
    fn test_minres_diagonal() {
        let mut a = Array2::<f64>::zeros((4, 4));
        a[[0, 0]] = 2.0;
        a[[1, 1]] = -3.0;
        a[[2, 2]] = 4.0;
        a[[3, 3]] = -1.0;
        let b = array![2.0_f64, -3.0, 4.0, -1.0];
        let result = minres(&a, &b.view(), None, 1e-10, 100)
            .expect("MINRES diagonal indefinite must succeed");
        let res = residual_norm_check(&a, &result.x, &[2.0, -3.0, 4.0, -1.0]);
        assert!(res < 1e-8 || result.converged, "MINRES residual {res}");
    }

    // ════════════════════════════════════════════════════
    // Cross-solver comparison tests
    // ════════════════════════════════════════════════════

    #[test]
    fn test_all_solvers_agree_on_spd() {
        // All solvers should agree on an SPD system
        let a = array![[5.0_f64, 2.0], [2.0, 4.0]];
        let b = array![9.0_f64, 8.0];

        let cg = conjugate_gradient(&a, &b.view(), None, 1e-12, 100, None).expect("CG failed");
        let gm = gmres(&a, &b.view(), None, 1e-12, 50, 10).expect("GMRES failed");
        let bi = bicgstab(&a, &b.view(), None, 1e-12, 100).expect("BiCGSTAB failed");

        assert!(
            cg.converged && gm.converged && bi.converged,
            "All solvers should converge: CG={} GMRES={} BiCGSTAB={}",
            cg.converged,
            gm.converged,
            bi.converged
        );

        for i in 0..2 {
            assert_relative_eq!(cg.x[i], gm.x[i], epsilon = 1e-8);
            assert_relative_eq!(cg.x[i], bi.x[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_invalid_initial_guess_dimension() {
        let a = array![[4.0_f64, 1.0], [1.0, 3.0]];
        let b = array![1.0_f64, 2.0];
        let x0 = array![0.5_f64, 0.5, 0.5]; // wrong dimension
        let result = conjugate_gradient(&a, &b.view(), Some(&x0), 1e-12, 100, None);
        assert!(result.is_err(), "Should error on wrong x0 dimension");
    }

    #[test]
    fn test_minres_lanczos_breakdown() {
        // A system where Lanczos might break down early
        // (identity matrix: Lanczos generates exact solution in 1 step)
        let a: Array2<f64> = Array2::eye(3);
        let b = array![1.0_f64, 2.0, 3.0];
        let result =
            minres(&a, &b.view(), None, 1e-12, 100).expect("MINRES on identity must succeed");
        // Should converge very quickly
        assert!(
            result.converged || result.iterations <= 3,
            "MINRES on identity: iters={} converged={}",
            result.iterations,
            result.converged
        );
    }

    #[test]
    fn test_minres_3x3_symmetric_indefinite() {
        // Larger symmetric indefinite system
        let a = array![[3.0_f64, 1.0, 0.0], [1.0, -2.0, 1.0], [0.0, 1.0, 4.0]];
        let b = array![4.0_f64, 0.0, 5.0];
        let result =
            minres(&a, &b.view(), None, 1e-10, 200).expect("MINRES 3x3 indefinite must succeed");
        let res = residual_norm_check(&a, &result.x, &[4.0, 0.0, 5.0]);
        assert!(
            res < 1e-6 || result.converged,
            "MINRES 3x3 indefinite residual {res}"
        );
    }
}
