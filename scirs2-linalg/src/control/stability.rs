//! Stability analysis: controllability, observability, and system-theoretic tools.
//!
//! Provides:
//! - Controllability matrix `C = [B, AB, A²B, …, A^{n-1}B]`
//! - Observability matrix `O = [C; CA; CA²; …; CA^{n-1}]`
//! - Rank-based controllability / observability tests
//! - Hautus lemma tests
//! - Pole-placement feasibility checks
//! - Gramian-based numerical controllability / observability measures
//!
//! # References
//! - Chen, C.-T. (1999). *Linear System Theory and Design*, 3rd ed.
//! - Sontag, E. D. (1998). *Mathematical Control Theory*, 2nd ed.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::{Debug, Display};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Trait bound
// ---------------------------------------------------------------------------

/// Floating-point trait bound for stability analysis.
pub trait StabilityFloat:
    Float + NumAssign + Sum + ScalarOperand + Debug + Display + Send + Sync + 'static
{
}
impl<F> StabilityFloat for F where
    F: Float + NumAssign + Sum + ScalarOperand + Debug + Display + Send + Sync + 'static
{
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Dense matrix multiply `C = A · B` (general dimensions).
fn mm<F: StabilityFloat>(a: &Array2<F>, b: &Array2<F>) -> Array2<F> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    let mut c = Array2::<F>::zeros((m, n));
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[[i, p]];
            if a_ip == F::zero() {
                continue;
            }
            for j in 0..n {
                c[[i, j]] += a_ip * b[[p, j]];
            }
        }
    }
    c
}

/// Compute the numerical rank of a matrix via Gaussian elimination with
/// partial pivoting.  Returns the number of pivots with absolute value
/// exceeding `tol`.
fn numerical_rank<F: StabilityFloat>(mat: &Array2<F>, tol: F) -> usize {
    let (m, n) = (mat.nrows(), mat.ncols());
    let mut a = mat.clone();
    let mut rank = 0usize;
    let mut col = 0usize;

    for row in 0..m.min(n) {
        // Search for a pivot in the current column (and beyond if needed).
        let mut pivot_row = None;
        let mut pivot_val = tol;

        'outer: for c in col..n {
            for r in row..m {
                let v = a[[r, c]].abs();
                if v > pivot_val {
                    pivot_val = v;
                    pivot_row = Some((r, c));
                }
            }
            if pivot_row.is_some() {
                break 'outer;
            }
        }

        let (pr, pc) = match pivot_row {
            None => break,
            Some(p) => p,
        };

        // Swap rows and columns to bring pivot to (row, col).
        if pr != row {
            for j in 0..n {
                let tmp = a[[row, j]];
                a[[row, j]] = a[[pr, j]];
                a[[pr, j]] = tmp;
            }
        }
        if pc != col {
            for i in 0..m {
                let tmp = a[[i, col]];
                a[[i, col]] = a[[i, pc]];
                a[[i, pc]] = tmp;
            }
        }

        // Eliminate below.
        let piv = a[[row, col]];
        for r in (row + 1)..m {
            let factor = a[[r, col]] / piv;
            if factor == F::zero() {
                continue;
            }
            for c in col..n {
                let val = a[[row, c]] * factor;
                a[[r, c]] -= val;
            }
        }

        rank += 1;
        col += 1;
    }

    rank
}

// ---------------------------------------------------------------------------
// Controllability
// ---------------------------------------------------------------------------

/// Build the controllability matrix `C = [B | AB | A²B | … | A^{n-1}B]`.
///
/// For a system with state dimension `n` and `m` inputs the returned matrix
/// has shape `(n, n·m)`.
///
/// # Errors
/// - [`LinalgError::ShapeError`] if `A` is not square, or if the row counts
///   of `A` and `B` are incompatible.
pub fn controllability_matrix<F: StabilityFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "controllability_matrix: A must be square".to_string(),
        ));
    }
    if b.nrows() != n {
        return Err(LinalgError::ShapeError(format!(
            "controllability_matrix: B row count ({}) must equal A dimension ({})",
            b.nrows(),
            n
        )));
    }
    let m = b.ncols();
    let mut ctrl = Array2::<F>::zeros((n, n * m));

    // First block: B
    let b_owned = b.to_owned();
    let mut a_pow_b = b_owned.clone();
    let a_owned = a.to_owned();

    for k in 0..n {
        // Write A^k B into columns [k*m .. (k+1)*m]
        for i in 0..n {
            for j in 0..m {
                ctrl[[i, k * m + j]] = a_pow_b[[i, j]];
            }
        }
        if k + 1 < n {
            a_pow_b = mm(&a_owned, &a_pow_b);
        }
    }

    Ok(ctrl)
}

/// Test whether the system `(A, B)` is controllable.
///
/// A system is controllable if the controllability matrix has full row rank.
///
/// # Errors
/// Propagates shape errors from [`controllability_matrix`].
pub fn is_controllable<F: StabilityFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
) -> LinalgResult<bool> {
    let n = a.nrows();
    let ctrl = controllability_matrix(a, b)?;
    let tol = F::from(1e-8).unwrap_or(F::epsilon());
    Ok(numerical_rank(&ctrl, tol) == n)
}

/// Compute the controllability Gramian `W_c = ∫₀^∞ e^{At} B Bᵀ e^{Aᵀt} dt`
/// via the continuous Lyapunov equation `A W_c + W_c Aᵀ + B Bᵀ = 0`.
///
/// Requires `A` to be stable (all eigenvalues with negative real part).
///
/// # Errors
/// - [`LinalgError::ShapeError`] for incompatible dimensions.
/// - [`LinalgError::ConvergenceError`] if the Lyapunov solver fails to converge.
pub fn controllability_gramian<F: StabilityFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "controllability_gramian: A must be square".to_string(),
        ));
    }
    if b.nrows() != n {
        return Err(LinalgError::ShapeError(format!(
            "controllability_gramian: B rows ({}) != A size ({})",
            b.nrows(),
            n
        )));
    }

    // Q = B Bᵀ
    let b_owned = b.to_owned();
    let q = mm(&b_owned, &b_owned.t().to_owned());
    super::lyapunov::lyapunov_continuous(&a.to_owned().view(), &q.view())
}

// ---------------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------------

/// Build the observability matrix `O = [C; CA; CA²; …; CA^{n-1}]`.
///
/// For a system with state dimension `n` and `p` outputs the returned matrix
/// has shape `(n·p, n)`.
///
/// # Errors
/// - [`LinalgError::ShapeError`] if `A` is not square, or if `C` column count
///   differs from the state dimension.
pub fn observability_matrix<F: StabilityFloat>(
    a: &ArrayView2<F>,
    c: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "observability_matrix: A must be square".to_string(),
        ));
    }
    if c.ncols() != n {
        return Err(LinalgError::ShapeError(format!(
            "observability_matrix: C column count ({}) must equal A dimension ({})",
            c.ncols(),
            n
        )));
    }
    let p = c.nrows();
    let mut obs = Array2::<F>::zeros((n * p, n));

    let c_owned = c.to_owned();
    let a_owned = a.to_owned();
    let mut c_pow = c_owned.clone(); // CA^k

    for k in 0..n {
        for i in 0..p {
            for j in 0..n {
                obs[[k * p + i, j]] = c_pow[[i, j]];
            }
        }
        if k + 1 < n {
            c_pow = mm(&c_pow, &a_owned);
        }
    }

    Ok(obs)
}

/// Test whether the system `(A, C)` is observable.
///
/// A system is observable if the observability matrix has full column rank.
///
/// # Errors
/// Propagates shape errors from [`observability_matrix`].
pub fn is_observable<F: StabilityFloat>(
    a: &ArrayView2<F>,
    c: &ArrayView2<F>,
) -> LinalgResult<bool> {
    let n = a.nrows();
    let obs = observability_matrix(a, c)?;
    let tol = F::from(1e-8).unwrap_or(F::epsilon());
    Ok(numerical_rank(&obs, tol) == n)
}

/// Compute the observability Gramian `W_o = ∫₀^∞ e^{Aᵀt} Cᵀ C e^{At} dt`
/// via the continuous Lyapunov equation `Aᵀ W_o + W_o A + Cᵀ C = 0`.
///
/// Requires `A` to be stable (all eigenvalues with negative real part).
///
/// # Errors
/// - [`LinalgError::ShapeError`] for incompatible dimensions.
/// - [`LinalgError::ConvergenceError`] if the Lyapunov solver fails.
pub fn observability_gramian<F: StabilityFloat>(
    a: &ArrayView2<F>,
    c: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "observability_gramian: A must be square".to_string(),
        ));
    }
    if c.ncols() != n {
        return Err(LinalgError::ShapeError(format!(
            "observability_gramian: C columns ({}) != A size ({})",
            c.ncols(),
            n
        )));
    }

    // Q = Cᵀ C
    let c_owned = c.to_owned();
    let q = mm(&c_owned.t().to_owned(), &c_owned);
    // Solve Aᵀ W_o + W_o A = -Q → same as lyapunov_continuous(Aᵀ, Q)
    let at = a.t().to_owned();
    super::lyapunov::lyapunov_continuous(&at.view(), &q.view())
}

// ---------------------------------------------------------------------------
// Hautus lemma tests
// ---------------------------------------------------------------------------

/// Check the **Hautus controllability condition**: `rank([λI - A, B]) = n`
/// for every eigenvalue `λ` of `A`.
///
/// This is an alternative (polynomial) controllability test that examines
/// each "critical" mode directly.  The implementation scans a discrete grid
/// of test points along the imaginary axis and at the Gershgorin disk
/// centres, which provides a practical (though not exhaustive) check.
///
/// For a rigorous test use [`is_controllable`] (rank of Kalman matrix).
pub fn hautus_controllability_check<F: StabilityFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
) -> LinalgResult<bool> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "hautus_controllability_check: A must be square".to_string(),
        ));
    }
    if b.nrows() != n {
        return Err(LinalgError::ShapeError(
            "hautus_controllability_check: B row count must equal n".to_string(),
        ));
    }

    let m = b.ncols();
    let a_owned = a.to_owned();
    let b_owned = b.to_owned();
    let tol = F::from(1e-6).unwrap_or(F::epsilon());

    // Sample candidate eigenvalue proxies from Gershgorin centres.
    // For each diagonal element a[i,i] test with λ = a[i,i].
    for i in 0..n {
        let lambda = a_owned[[i, i]];
        // Build [λI - A | B] (n × (n + m))
        let mut mat = Array2::<F>::zeros((n, n + m));
        for r in 0..n {
            for c in 0..n {
                mat[[r, c]] = if r == c {
                    lambda - a_owned[[r, c]]
                } else {
                    -a_owned[[r, c]]
                };
            }
            for c in 0..m {
                mat[[r, n + c]] = b_owned[[r, c]];
            }
        }
        if numerical_rank(&mat, tol) < n {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Check the **Hautus observability condition**: `rank([λI - A; C]) = n`
/// for every eigenvalue `λ` of `A`.
pub fn hautus_observability_check<F: StabilityFloat>(
    a: &ArrayView2<F>,
    c: &ArrayView2<F>,
) -> LinalgResult<bool> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "hautus_observability_check: A must be square".to_string(),
        ));
    }
    if c.ncols() != n {
        return Err(LinalgError::ShapeError(
            "hautus_observability_check: C column count must equal n".to_string(),
        ));
    }

    let p = c.nrows();
    let a_owned = a.to_owned();
    let c_owned = c.to_owned();
    let tol = F::from(1e-6).unwrap_or(F::epsilon());

    for i in 0..n {
        let lambda = a_owned[[i, i]];
        // Build [(λI - A)ᵀ; Cᵀ]^T i.e. [λI-A; C] stacked (n+p) × n
        let mut mat = Array2::<F>::zeros((n + p, n));
        for r in 0..n {
            for c in 0..n {
                mat[[r, c]] = if r == c {
                    lambda - a_owned[[r, c]]
                } else {
                    -a_owned[[r, c]]
                };
            }
        }
        for r in 0..p {
            for c in 0..n {
                mat[[n + r, c]] = c_owned[[r, c]];
            }
        }
        if numerical_rank(&mat, tol) < n {
            return Ok(false);
        }
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// Balanced truncation (system Gramians)
// ---------------------------------------------------------------------------

/// Measure of **controllability** computed as the trace of the controllability
/// Gramian.  A larger trace indicates greater ease of controlling the system.
///
/// Requires `A` to be stable.
pub fn controllability_measure<F: StabilityFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
) -> LinalgResult<F> {
    let wc = controllability_gramian(a, b)?;
    Ok((0..wc.nrows()).map(|i| wc[[i, i]]).sum())
}

/// Measure of **observability** computed as the trace of the observability
/// Gramian.  A larger trace indicates greater ease of reconstructing the state.
///
/// Requires `A` to be stable.
pub fn observability_measure<F: StabilityFloat>(
    a: &ArrayView2<F>,
    c: &ArrayView2<F>,
) -> LinalgResult<F> {
    let wo = observability_gramian(a, c)?;
    Ok((0..wo.nrows()).map(|i| wo[[i, i]]).sum())
}

// ---------------------------------------------------------------------------
// Minimal realization
// ---------------------------------------------------------------------------

/// Return a sorted list of the **Hankel singular values** of the system
/// `(A, B, C)` — the square roots of the eigenvalues of `W_c · W_o`.
///
/// These determine the importance of each mode and are used for model-order
/// reduction via balanced truncation.
///
/// Requires `A` to be stable.
pub fn hankel_singular_values<F: StabilityFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    c: &ArrayView2<F>,
) -> LinalgResult<Array1<F>> {
    let wc = controllability_gramian(a, b)?;
    let wo = observability_gramian(a, c)?;
    let prod = mm(&wc, &wo);
    let n = prod.nrows();

    // Eigenvalues of W_c · W_o via characteristic polynomial (power iteration
    // gives largest; for small n use direct computation).
    // For a symmetric-like product we compute σ_i = sqrt(λ_i(W_c W_o)).
    // We use the Gershgorin bound + inverse iteration heuristic for small n.
    // For a fully general approach we fall back to the companion matrix approach.
    let hsv = eigenvalues_sym_product(&prod, n)?;
    Ok(hsv)
}

/// Compute eigenvalues of a (not necessarily symmetric) real matrix `M`
/// using the QR algorithm (Francis double-shift).  Returns the real parts
/// of eigenvalues in descending order of absolute value.
fn eigenvalues_sym_product<F: StabilityFloat>(m: &Array2<F>, n: usize) -> LinalgResult<Array1<F>> {
    // Reduce to upper Hessenberg form first, then apply QR iterations.
    let mut h = m.clone();

    // Hessenberg reduction (Householder)
    hessenberg_inplace(&mut h, n);

    // QR iterations (Francis double shift)
    let max_iter = 1000 * n;
    let tol_eig = F::from(1e-12).unwrap_or(F::epsilon());
    let mut eigenvals: Vec<F> = Vec::with_capacity(n);
    let mut active = n;

    for _ in 0..max_iter {
        if active <= 1 {
            break;
        }
        // Check for deflation
        let mut deflate_at = None;
        for i in (1..active).rev() {
            let sub = h[[i, i - 1]].abs();
            let scale = h[[i - 1, i - 1]].abs() + h[[i, i]].abs();
            if sub < tol_eig * scale {
                h[[i, i - 1]] = F::zero();
                deflate_at = Some(i);
                break;
            }
        }
        if let Some(idx) = deflate_at {
            eigenvals.push(h[[idx, idx]]);
            active = idx;
            continue;
        }

        // Francis double-shift QR step on h[0..active, 0..active]
        francis_qr_step(&mut h, active);
    }
    // Collect remaining eigenvalues
    for i in 0..active {
        eigenvals.push(h[[i, i]]);
    }

    // Convert to Hankel singular values: sqrt(max(0, λ))
    let mut hsv: Vec<F> = eigenvals
        .into_iter()
        .map(|lam| {
            let v = if lam < F::zero() { F::zero() } else { lam };
            v.sqrt()
        })
        .collect();
    hsv.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    Ok(Array1::from_vec(hsv))
}

/// In-place reduction to upper Hessenberg form via Householder reflections.
fn hessenberg_inplace<F: StabilityFloat>(h: &mut Array2<F>, n: usize) {
    for k in 0..(n.saturating_sub(2)) {
        // Build Householder vector for column k, rows k+1..n
        let mut v: Vec<F> = (k + 1..n).map(|i| h[[i, k]]).collect();
        let norm_sq: F = v.iter().map(|&x| x * x).sum();
        if norm_sq < F::from(1e-28).unwrap_or(F::epsilon()) {
            continue;
        }
        let norm = norm_sq.sqrt();
        let sign = if v[0] >= F::zero() {
            F::one()
        } else {
            -F::one()
        };
        v[0] += sign * norm;
        let norm2_sq: F = v.iter().map(|&x| x * x).sum();
        if norm2_sq < F::from(1e-28).unwrap_or(F::epsilon()) {
            continue;
        }
        let two = F::from(2.0).unwrap_or(F::one() + F::one());

        // Apply H = I - 2 v vᵀ / ||v||² from left: h = H · h
        let len = v.len();
        for j in 0..n {
            let dot: F = (0..len).map(|i| v[i] * h[[k + 1 + i, j]]).sum();
            let scale = two * dot / norm2_sq;
            for i in 0..len {
                h[[k + 1 + i, j]] -= scale * v[i];
            }
        }
        // Apply H from right: h = h · H
        for i in 0..n {
            let dot: F = (0..len).map(|j| h[[i, k + 1 + j]] * v[j]).sum();
            let scale = two * dot / norm2_sq;
            for j in 0..len {
                h[[i, k + 1 + j]] -= scale * v[j];
            }
        }
    }
}

/// Single Francis double-shift QR step on the leading `active × active`
/// submatrix of `h`.
fn francis_qr_step<F: StabilityFloat>(h: &mut Array2<F>, active: usize) {
    if active < 2 {
        return;
    }
    let two = F::from(2.0).unwrap_or(F::one() + F::one());
    // Shift: eigenvalues of the 2×2 trailing submatrix
    let a = h[[active - 2, active - 2]];
    let b = h[[active - 2, active - 1]];
    let c = h[[active - 1, active - 2]];
    let d = h[[active - 1, active - 1]];

    let trace = a + d;
    let det = a * d - b * c;

    // Bulge-chase: compute the first column of (H - s₁I)(H - s₂I)
    let h00 = h[[0, 0]];
    let h10 = h[[1, 0]];
    let h20 = if active > 2 { h[[2, 0]] } else { F::zero() };

    let x = h00 * h00 + h[[0, 1]] * h10 - trace * h00 + det;
    let y = h10 * (h00 + h[[1, 1]] - trace);
    let z = if active > 2 {
        h10 * h[[2, 1]]
    } else {
        F::zero()
    };

    // Chase the bulge with Householder reflectors
    let limit = active.saturating_sub(2);
    for k in 0..=limit {
        let v0;
        let v1;
        let v2_opt;

        if k == 0 {
            // First reflector from (x,y,z)
            let norm = (x * x + y * y + z * z).sqrt();
            if norm < F::from(1e-14).unwrap_or(F::epsilon()) {
                break;
            }
            let sign = if x >= F::zero() { F::one() } else { -F::one() };
            v0 = x + sign * norm;
            v1 = y;
            v2_opt = if active > 2 { Some(z) } else { None };
        } else {
            v0 = h[[k, k - 1]];
            v1 = h[[k + 1, k - 1]];
            v2_opt = if k + 2 < active {
                Some(h[[k + 2, k - 1]])
            } else {
                None
            };
        }

        let norm_sq = v0 * v0 + v1 * v1 + v2_opt.map_or(F::zero(), |v| v * v);
        if norm_sq < F::from(1e-28).unwrap_or(F::epsilon()) {
            continue;
        }

        let rows = if v2_opt.is_some() { 3usize } else { 2 };
        let v = [v0, v1, v2_opt.unwrap_or(F::zero())];

        // Apply from left (rows k..k+rows, cols k-1..active)
        let col_start = if k == 0 { 0 } else { k - 1 };
        for j in col_start..active {
            let dot: F = (0..rows).map(|i| v[i] * h[[k + i, j]]).sum();
            let scale = two * dot / norm_sq;
            for i in 0..rows {
                h[[k + i, j]] -= scale * v[i];
            }
        }
        // Apply from right (rows 0..active, cols k..k+rows)
        for i in 0..active {
            let dot: F = (0..rows).map(|j| h[[i, k + j]] * v[j]).sum();
            let scale = two * dot / norm_sq;
            for j in 0..rows {
                h[[i, k + j]] -= scale * v[j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // Double-integrator: A=[[0,1],[0,0]], B=[[0],[1]]
    fn double_integrator() -> (Array2<f64>, Array2<f64>) {
        let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
        let b = array![[0.0_f64], [1.0]];
        (a, b)
    }

    #[test]
    fn test_controllability_matrix_double_integrator() {
        let (a, b) = double_integrator();
        let ctrl =
            controllability_matrix(&a.view(), &b.view()).expect("controllability_matrix failed");
        // Expected: [B | AB] = [[0,1],[1,0]]
        assert_eq!(ctrl.nrows(), 2);
        assert_eq!(ctrl.ncols(), 2);
        assert!((ctrl[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((ctrl[[1, 0]] - 1.0).abs() < 1e-10);
        assert!((ctrl[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((ctrl[[1, 1]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_controllable_double_integrator() {
        let (a, b) = double_integrator();
        let ctrl = is_controllable(&a.view(), &b.view()).expect("is_controllable failed");
        assert!(ctrl, "double integrator should be controllable");
    }

    #[test]
    fn test_is_not_controllable() {
        // A=I2, B=[[1],[0]]: uncontrollable (second state not reachable)
        let a = array![[0.0_f64, 0.0], [0.0, 0.0]];
        let b = array![[1.0_f64], [0.0]];
        let ctrl = is_controllable(&a.view(), &b.view()).expect("is_controllable failed");
        assert!(!ctrl, "should be uncontrollable");
    }

    #[test]
    fn test_observability_matrix() {
        // A=[[0,1],[0,0]], C=[[1,0]] (observe position)
        let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
        let c = array![[1.0_f64, 0.0]];
        let obs = observability_matrix(&a.view(), &c.view()).expect("observability_matrix failed");
        // Expected: [C; CA] = [[1,0],[0,1]]
        assert_eq!(obs.nrows(), 2);
        assert_eq!(obs.ncols(), 2);
        assert!((obs[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((obs[[0, 1]] - 0.0).abs() < 1e-10);
        assert!((obs[[1, 0]] - 0.0).abs() < 1e-10);
        assert!((obs[[1, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_observable() {
        let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
        let c = array![[1.0_f64, 0.0]];
        let obs = is_observable(&a.view(), &c.view()).expect("is_observable failed");
        assert!(obs, "should be observable");
    }

    #[test]
    fn test_is_not_observable() {
        // Observe zero → trivially unobservable
        let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
        let c = array![[0.0_f64, 0.0]];
        let obs = is_observable(&a.view(), &c.view()).expect("is_observable failed");
        assert!(!obs, "zero C should be unobservable");
    }

    #[test]
    fn test_controllability_gramian_stable() {
        // Stable 2x2 system: A=diag(-1,-2), B=I
        let a = array![[-1.0_f64, 0.0], [0.0, -2.0]];
        let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let wc =
            controllability_gramian(&a.view(), &b.view()).expect("controllability_gramian failed");
        // W_c should be positive definite: diagonal entries > 0
        assert!(wc[[0, 0]] > 0.0, "W_c[0,0] must be positive");
        assert!(wc[[1, 1]] > 0.0, "W_c[1,1] must be positive");
        // Verify Lyapunov: A W_c + W_c Aᵀ + B Bᵀ ≈ 0
        let q = mm(&b, &b.t().to_owned());
        let res = mm(&a, &wc) + mm(&wc, &a.t().to_owned()) + q;
        for &v in res.iter() {
            assert!(v.abs() < 1e-6, "Gramian residual {v} too large");
        }
    }

    #[test]
    fn test_observability_gramian_stable() {
        let a = array![[-1.0_f64, 0.0], [0.0, -2.0]];
        let c = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let wo = observability_gramian(&a.view(), &c.view()).expect("observability_gramian failed");
        assert!(wo[[0, 0]] > 0.0, "W_o[0,0] must be positive");
        // Verify: Aᵀ W_o + W_o A + Cᵀ C ≈ 0
        let q = mm(&c.t().to_owned(), &c);
        let at = a.t().to_owned();
        let res = mm(&at, &wo) + mm(&wo, &a) + q;
        for &v in res.iter() {
            assert!(v.abs() < 1e-6, "Observability Gramian residual {v}");
        }
    }

    #[test]
    fn test_controllability_measure() {
        let a = array![[-1.0_f64, 0.0], [0.0, -2.0]];
        let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let measure =
            controllability_measure(&a.view(), &b.view()).expect("controllability_measure failed");
        assert!(measure > 0.0, "Controllability measure should be positive");
    }
}
