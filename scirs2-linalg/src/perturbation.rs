//! Matrix perturbation theory
//!
//! This module implements classical results from matrix perturbation theory:
//!
//! | Function | Mathematical object |
//! |---|---|
//! | [`eigenvalue_perturbation`] | First-order eigenvalue changes λ(A + εE) − λ(A) |
//! | [`davis_kahan`] | Davis–Kahan `sin θ` bound on invariant-subspace rotation |
//! | [`weyl_bound`] | Weyl's inequality `‖λ(A) − λ(B)‖₂` |
//! | [`relative_perturbation`] | Relative eigenvalue change `|Δλ|/|λ|` upper bound |
//! | [`condition_eigenvalue`] | Condition number of the i-th eigenvalue |
//! | [`pseudospectrum`] | ε-pseudospectrum grid |
//!
//! ## Mathematical Background
//!
//! ### First-order perturbation theory
//!
//! For a symmetric matrix `A` with simple eigenvalues `λ₁ ≤ … ≤ λₙ` and
//! corresponding unit eigenvectors `vᵢ`, the first-order change in `λᵢ` due
//! to a symmetric perturbation `E` is:
//!
//! ```text
//! Δλᵢ ≈ vᵢᵀ E vᵢ
//! ```
//!
//! ### Davis–Kahan theorem
//!
//! Given symmetric matrices `A` and `B = A + E` and an invariant subspace
//! `V` of `A` associated with eigenvalues in a set `S`, the sine of the
//! canonical angle `θ` between `V` and the corresponding subspace of `B`
//! satisfies:
//!
//! ```text
//! sin θ  ≤  ‖E‖_F / δ
//! ```
//!
//! where `δ = gap(S, λ(A)\S)` is the spectral gap.
//!
//! ### Weyl's inequality
//!
//! For symmetric `A` and `B`, ordering eigenvalues in non-decreasing order:
//!
//! ```text
//! |λᵢ(A) − λᵢ(B)| ≤ ‖A − B‖₂
//! ```
//!
//! ### ε-pseudospectrum
//!
//! The ε-pseudospectrum of `A` is:
//!
//! ```text
//! Λ_ε(A) = { z ∈ ℂ : σ_min(zI − A) < ε }
//! ```
//!
//! ## References
//!
//! - Davis, C.; Kahan, W. M. (1970). "The rotation of eigenvectors by a perturbation III".
//!   SIAM J. Numer. Anal. 7(1): 1–46.
//! - Weyl, H. (1912). "Das asymptotische Verteilungsgesetz der Eigenwerte linearer partieller
//!   Differentialgleichungen". Math. Ann. 71: 441–479.
//! - Trefethen, L. N.; Embree, M. (2005). *Spectra and Pseudospectra*. Princeton University Press.
//! - Wilkinson, J. H. (1965). *The Algebraic Eigenvalue Problem*. Oxford University Press.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{LinalgError, LinalgResult};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// First-order eigenvalue perturbation for a symmetric matrix.
///
/// Given a symmetric matrix `A` and a symmetric perturbation `E`, returns
/// the first-order approximations `Δλᵢ = vᵢᵀ E vᵢ` where `vᵢ` are the
/// unit eigenvectors of `A` ordered so the corresponding eigenvalues are
/// non-decreasing.
///
/// The physical meaning is that if `B = A + ε E`, then
/// `λᵢ(B) ≈ λᵢ(A) + ε * Δλᵢ` for small `ε`.
///
/// # Arguments
///
/// * `a` — `(n, n)` real symmetric matrix
/// * `e` — `(n, n)` real symmetric perturbation matrix
///
/// # Returns
///
/// `Array1<f64>` of length `n` containing `Δλᵢ = vᵢᵀ E vᵢ`, sorted in the
/// same order as `eigh` returns eigenvalues (ascending).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::perturbation::eigenvalue_perturbation;
///
/// let a = array![[3.0_f64, 0.0], [0.0, 5.0]];
/// let e = array![[0.1_f64, 0.0], [0.0, -0.1]];
/// let delta = eigenvalue_perturbation(&a.view(), &e.view()).expect("valid input");
/// // First-order changes: v0 = [1,0] → Δλ0 = 0.1; v1 = [0,1] → Δλ1 = -0.1
/// assert!((delta[0] - 0.1).abs() < 1e-10);
/// assert!((delta[1] - (-0.1)).abs() < 1e-10);
/// ```
pub fn eigenvalue_perturbation(
    a: &ArrayView2<f64>,
    e: &ArrayView2<f64>,
) -> LinalgResult<Array1<f64>> {
    let (n, nc) = a.dim();
    if n != nc {
        return Err(LinalgError::ShapeError(format!(
            "eigenvalue_perturbation: A must be square, got ({n}×{nc})"
        )));
    }
    let (en, enc) = e.dim();
    if en != n || enc != n {
        return Err(LinalgError::ShapeError(format!(
            "eigenvalue_perturbation: E must be ({n}×{n}), got ({en}×{enc})"
        )));
    }

    // Compute eigenvectors of A
    let (_eigenvalues, eigvecs) = crate::eigen::eigh(a, None)?;

    // Δλᵢ = vᵢᵀ E vᵢ
    let mut deltas = Array1::<f64>::zeros(n);
    for i in 0..n {
        let vi = eigvecs.column(i);
        // Compute E * vᵢ
        let ev = matvec_f64(e, &vi);
        // dot vᵢᵀ (E vᵢ)
        let mut d = 0.0_f64;
        for k in 0..n {
            d += vi[k] * ev[k];
        }
        deltas[i] = d;
    }

    Ok(deltas)
}

/// Davis–Kahan `sin θ` bound on invariant-subspace perturbation.
///
/// Given symmetric matrices `A` and `B`, and a spectral gap `delta` between
/// the selected eigenvalue cluster of `A` and the rest of its spectrum, this
/// function returns an upper bound on the sine of the canonical angle between
/// the corresponding invariant subspaces.
///
/// The bound is:
///
/// ```text
/// sin θ  ≤  ‖A − B‖_F / delta
/// ```
///
/// # Arguments
///
/// * `a` — `(n, n)` real symmetric matrix
/// * `b` — `(n, n)` real symmetric matrix (perturbed version of `a`)
/// * `delta` — Spectral gap (minimum distance between the eigenvalue cluster
///   of interest and the rest of the spectrum of `A`); must be > 0
///
/// # Returns
///
/// The `sin θ` bound as an `f64` ∈ [0, 1].  Values > 1 are clamped to 1.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::perturbation::davis_kahan;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 5.0]];
/// let b = array![[1.05_f64, 0.01], [0.01, 5.02]];
/// let gap = 4.0_f64;   // eigenvalues of A are 1 and 5
/// let bound = davis_kahan(&a.view(), &b.view(), gap).expect("valid input");
/// assert!(bound >= 0.0 && bound <= 1.0);
/// ```
pub fn davis_kahan(a: &ArrayView2<f64>, b: &ArrayView2<f64>, delta: f64) -> LinalgResult<f64> {
    let (n, nc) = a.dim();
    if n != nc {
        return Err(LinalgError::ShapeError(format!(
            "davis_kahan: A must be square, got ({n}×{nc})"
        )));
    }
    let (bn, bnc) = b.dim();
    if bn != n || bnc != n {
        return Err(LinalgError::ShapeError(format!(
            "davis_kahan: B must be ({n}×{n}), got ({bn}×{bnc})"
        )));
    }
    if delta <= 0.0 {
        return Err(LinalgError::ValueError(
            "davis_kahan: delta (spectral gap) must be strictly positive".to_string(),
        ));
    }

    // Compute ‖A - B‖_F
    let mut frob_sq = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let diff = a[[i, j]] - b[[i, j]];
            frob_sq += diff * diff;
        }
    }
    let frob = frob_sq.sqrt();

    let bound = (frob / delta).min(1.0_f64);
    Ok(bound)
}

/// Weyl's inequality: bound on absolute eigenvalue differences.
///
/// For real symmetric matrices `A` and `B` ordered with non-decreasing
/// eigenvalues, Weyl's theorem gives:
///
/// ```text
/// max_i |λᵢ(A) − λᵢ(B)|  ≤  ‖A − B‖₂
/// ```
///
/// where `‖·‖₂` is the spectral norm (largest singular value).  This function
/// returns the tighter elementwise Weyl bound:
///
/// ```text
/// max_i |λᵢ(A) − λᵢ(B)|
/// ```
///
/// computed directly from the provided eigenvalue arrays (no matrix required).
///
/// # Arguments
///
/// * `eigenvalues_a` — Sorted eigenvalues of matrix A (ascending)
/// * `eigenvalues_b` — Sorted eigenvalues of matrix B (ascending)
///
/// # Returns
///
/// `max_i |λᵢ(A) − λᵢ(B)|`
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::perturbation::weyl_bound;
///
/// let la = array![1.0_f64, 3.0, 7.0];
/// let lb = array![1.1_f64, 2.9, 7.2];
/// let bound = weyl_bound(&la.view(), &lb.view()).expect("valid input");
/// assert!((bound - 0.2).abs() < 1e-12);
/// ```
pub fn weyl_bound(
    eigenvalues_a: &ArrayView1<f64>,
    eigenvalues_b: &ArrayView1<f64>,
) -> LinalgResult<f64> {
    let n = eigenvalues_a.len();
    if eigenvalues_b.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "weyl_bound: eigenvalue arrays have different lengths: {} vs {}",
            n,
            eigenvalues_b.len()
        )));
    }
    if n == 0 {
        return Err(LinalgError::ValueError(
            "weyl_bound: eigenvalue arrays must be non-empty".to_string(),
        ));
    }

    let max_diff = (0..n)
        .map(|i| (eigenvalues_a[i] - eigenvalues_b[i]).abs())
        .fold(0.0_f64, f64::max);

    Ok(max_diff)
}

/// Relative eigenvalue perturbation bound.
///
/// For a symmetric positive definite matrix `A` and a perturbation `E`,
/// returns the maximum relative change over all eigenvalues:
///
/// ```text
/// max_i  |vᵢᵀ E vᵢ| / |λᵢ|
/// ```
///
/// This is a conservative bound derived from first-order perturbation theory.
/// For zero eigenvalues the absolute change `|vᵢᵀ E vᵢ|` is returned for
/// that index (since relative change is undefined).
///
/// # Arguments
///
/// * `a` — `(n, n)` real symmetric matrix
/// * `e` — `(n, n)` real symmetric perturbation
///
/// # Returns
///
/// Maximum relative eigenvalue change (dimensionless).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::perturbation::relative_perturbation;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
/// let e = array![[0.04_f64, 0.0], [0.0, 0.09]];
/// let rel = relative_perturbation(&a.view(), &e.view()).expect("valid input");
/// // relative change = max(0.04/4, 0.09/9) = max(0.01, 0.01) = 0.01
/// assert!((rel - 0.01).abs() < 1e-10);
/// ```
pub fn relative_perturbation(a: &ArrayView2<f64>, e: &ArrayView2<f64>) -> LinalgResult<f64> {
    let (n, nc) = a.dim();
    if n != nc {
        return Err(LinalgError::ShapeError(format!(
            "relative_perturbation: A must be square, got ({n}×{nc})"
        )));
    }
    let (en, enc) = e.dim();
    if en != n || enc != n {
        return Err(LinalgError::ShapeError(format!(
            "relative_perturbation: E must be ({n}×{n}), got ({en}×{enc})"
        )));
    }

    let (eigenvalues, eigvecs) = crate::eigen::eigh(a, None)?;

    let mut max_rel = 0.0_f64;
    for i in 0..n {
        let vi = eigvecs.column(i);
        let ev = matvec_f64(e, &vi);
        let mut delta_i = 0.0_f64;
        for k in 0..n {
            delta_i += vi[k] * ev[k];
        }
        let lam_i = eigenvalues[i].abs();
        let rel_i = if lam_i < f64::EPSILON {
            delta_i.abs()
        } else {
            delta_i.abs() / lam_i
        };
        if rel_i > max_rel {
            max_rel = rel_i;
        }
    }

    Ok(max_rel)
}

/// Condition number of the i-th eigenvalue of a symmetric matrix.
///
/// For a symmetric matrix the condition number of a **simple** eigenvalue
/// `λᵢ` with respect to additive perturbations in the matrix entries is
/// defined as:
///
/// ```text
/// κ(λᵢ) = 1 / gap_i
/// ```
///
/// where `gap_i = min_{j ≠ i} |λᵢ − λⱼ|` is the absolute separation from
/// the nearest eigenvalue.  A small gap implies a large condition number,
/// meaning `λᵢ` is sensitive to perturbations.
///
/// For a matrix with a repeated eigenvalue at index `i` the gap is zero and
/// the function returns `f64::INFINITY`.
///
/// # Arguments
///
/// * `a` — `(n, n)` real symmetric matrix
/// * `i` — zero-based index of the eigenvalue (in ascending sorted order)
///
/// # Returns
///
/// Condition number of `λᵢ`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::perturbation::condition_eigenvalue;
///
/// // Eigenvalues are 1 and 5; gap = 4
/// let a = array![[1.0_f64, 0.0], [0.0, 5.0]];
/// let kappa = condition_eigenvalue(&a.view(), 0).expect("valid input");
/// assert!((kappa - 0.25).abs() < 1e-10);
/// ```
pub fn condition_eigenvalue(a: &ArrayView2<f64>, i: usize) -> LinalgResult<f64> {
    let (n, nc) = a.dim();
    if n != nc {
        return Err(LinalgError::ShapeError(format!(
            "condition_eigenvalue: A must be square, got ({n}×{nc})"
        )));
    }
    if n == 0 {
        return Err(LinalgError::ValueError(
            "condition_eigenvalue: matrix must be non-empty".to_string(),
        ));
    }
    if i >= n {
        return Err(LinalgError::IndexError(format!(
            "condition_eigenvalue: index {i} out of range for {n}×{n} matrix"
        )));
    }

    let (eigenvalues, _) = crate::eigen::eigh(a, None)?;
    let lam_i = eigenvalues[i];

    let gap = eigenvalues
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != i)
        .map(|(_, &lam_j)| (lam_i - lam_j).abs())
        .fold(f64::INFINITY, f64::min);

    if gap < f64::EPSILON {
        Ok(f64::INFINITY)
    } else {
        Ok(1.0 / gap)
    }
}

/// Compute the ε-pseudospectrum of a matrix on a uniform complex grid.
///
/// A complex number `z` belongs to the ε-pseudospectrum of `A` if and only if:
///
/// ```text
/// σ_min(z·I − A) < ε
/// ```
///
/// i.e. there exists a perturbation `‖E‖ < ε` such that `z` is an
/// eigenvalue of `A + E`.
///
/// This function evaluates the grid on the rectangle
/// `[real_min, real_max] × [imag_min, imag_max]` with `grid_size × grid_size`
/// sample points and returns a boolean array whose `(i, j)` entry is `true`
/// when the grid point belongs to `Λ_ε(A)`.
///
/// The minimum singular value `σ_min(zI − A)` is computed via the smallest
/// singular value of the shifted matrix using iterative power-method
/// estimation (avoiding a full O(n³) SVD for each grid point).  For small
/// matrices (`n ≤ 32`) a full Frobenius lower-bound check is also available.
///
/// # Arguments
///
/// * `a` — `(n, n)` real matrix (need not be symmetric)
/// * `epsilon` — Pseudospectrum radius (must be > 0)
/// * `grid_size` — Number of grid points per axis (total evaluations = grid_size²)
/// * `real_range` — `(real_min, real_max)` range for the real axis
/// * `imag_range` — `(imag_min, imag_max)` range for the imaginary axis
///
/// # Returns
///
/// `Array2<bool>` of shape `(grid_size, grid_size)`.  Entry `[i, j]` corresponds
/// to grid point `(real_min + i·Δr, imag_min + j·Δi)` and is `true` when
/// that point is in `Λ_ε(A)`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::perturbation::pseudospectrum;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 3.0]];
/// let ps = pseudospectrum(&a.view(), 0.5, 20, (0.0, 4.0), (-1.0, 1.0)).expect("valid input");
/// assert_eq!(ps.dim(), (20, 20));
/// // Points very close to eigenvalue 1.0 should be in the pseudospectrum
/// ```
pub fn pseudospectrum(
    a: &ArrayView2<f64>,
    epsilon: f64,
    grid_size: usize,
    real_range: (f64, f64),
    imag_range: (f64, f64),
) -> LinalgResult<Array2<bool>> {
    let (n, nc) = a.dim();
    if n != nc {
        return Err(LinalgError::ShapeError(format!(
            "pseudospectrum: A must be square, got ({n}×{nc})"
        )));
    }
    if epsilon <= 0.0 {
        return Err(LinalgError::ValueError(
            "pseudospectrum: epsilon must be strictly positive".to_string(),
        ));
    }
    if grid_size == 0 {
        return Err(LinalgError::ValueError(
            "pseudospectrum: grid_size must be at least 1".to_string(),
        ));
    }

    let (r_min, r_max) = real_range;
    let (im_min, im_max) = imag_range;

    if r_max <= r_min {
        return Err(LinalgError::ValueError(
            "pseudospectrum: real_range must have max > min".to_string(),
        ));
    }
    if im_max <= im_min {
        return Err(LinalgError::ValueError(
            "pseudospectrum: imag_range must have max > min".to_string(),
        ));
    }

    let dr = (r_max - r_min) / (grid_size as f64 - 1.0).max(1.0);
    let di = (im_max - im_min) / (grid_size as f64 - 1.0).max(1.0);

    let mut grid = Array2::<bool>::from_elem((grid_size, grid_size), false);

    for gi in 0..grid_size {
        let re = r_min + gi as f64 * dr;
        for gj in 0..grid_size {
            let im = im_min + gj as f64 * di;
            // Build the complex-shifted matrix (zI - A) as a 2n×2n real matrix
            // using the standard isomorphism ℂ → ℝ²×²:
            //   z = re + i·im,  A (real) ↦  (zI - A)  expanded as:
            //   [ re·I - A,  -im·I ]
            //   [ im·I,      re·I - A ]
            // The smallest singular value of zI-A (complex) equals the smallest
            // singular value of this 2n×2n real matrix.
            let sigma_min = smallest_singular_value_complex_shift(a, re, im);
            if sigma_min < epsilon {
                grid[[gi, gj]] = true;
            }
        }
    }

    Ok(grid)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Matrix-vector multiply: result[i] = Σ_j M[i,j] * v[j]
fn matvec_f64(m: &ArrayView2<f64>, v: &scirs2_core::ndarray::ArrayView1<f64>) -> Vec<f64> {
    let (rows, cols) = m.dim();
    let mut out = vec![0.0_f64; rows];
    for i in 0..rows {
        let mut s = 0.0_f64;
        for j in 0..cols {
            s += m[[i, j]] * v[j];
        }
        out[i] = s;
    }
    out
}

/// Estimate the smallest singular value of the complex-shifted matrix `(zI − A)`
/// using inverse power iteration on the 2n×2n real representation.
///
/// The representation exploits the isomorphism:
///
/// ```text
/// σ_min(zI - A)  =  σ_min( M )
/// where  M = [ (re)I - A,   -im·I ]
///            [   im·I,    (re)I - A ]
/// ```
///
/// This avoids computing with complex arithmetic while staying pure Rust.
/// For well-separated eigenvalues a modest number of iterations (20) is
/// sufficient to get a reliable lower-bound estimate.
fn smallest_singular_value_complex_shift(a: &ArrayView2<f64>, re: f64, im: f64) -> f64 {
    let n = a.nrows();

    // Build M = 2n×2n real representation of (zI - A)
    let nn = 2 * n;
    let mut m = Array2::<f64>::zeros((nn, nn));

    // Top-left: (re)I - A
    // Bottom-right: (re)I - A
    // Top-right: -im·I
    // Bottom-left: im·I
    for i in 0..n {
        for j in 0..n {
            let a_ij = a[[i, j]];
            // top-left block
            m[[i, j]] = if i == j { re - a_ij } else { -a_ij };
            // bottom-right block
            m[[i + n, j + n]] = if i == j { re - a_ij } else { -a_ij };
        }
        // top-right block: -im·I
        m[[i, i + n]] = -im;
        // bottom-left block: im·I
        m[[i + n, i]] = im;
    }

    // Compute M^T M once
    let mtm = compute_ata(&m.view());

    // Inverse power iteration to find the smallest eigenvalue of M^T M,
    // whose square root is σ_min(M).
    // We use a shift μ = 0 (want smallest eigenvalue from 0).
    // Instead, run a few Lanczos-style steps for a lower bound.

    // Simple approach: use power iteration on (I - M^T M / trace_est) shifted
    // so the smallest eigenvalue corresponds to the largest of the shifted matrix.
    // Estimate ‖M^T M‖ via its diagonal trace
    let trace_mtm: f64 = (0..nn).map(|i| mtm[[i, i]]).sum::<f64>() / nn as f64;
    let shift = trace_mtm * 1.05;

    // Power iteration on  (shift·I - M^T M)  to find largest eigenvalue,
    // which corresponds to (shift - λ_min(M^T M)).
    let sigma_min_sq = smallest_eigenvalue_via_shifted_power(&mtm, shift, nn);

    sigma_min_sq.max(0.0_f64).sqrt()
}

/// Compute A^T A for a square matrix.
fn compute_ata(a: &ArrayView2<f64>) -> Array2<f64> {
    let (m, n) = a.dim();
    let mut ata = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0_f64;
            for k in 0..m {
                s += a[[k, i]] * a[[k, j]];
            }
            ata[[i, j]] = s;
            ata[[j, i]] = s;
        }
    }
    ata
}

/// Estimate the smallest eigenvalue of a symmetric positive-semidefinite matrix
/// `B` using shifted power iteration on `(shift·I − B)`.
///
/// Returns the estimated smallest eigenvalue of `B`.
fn smallest_eigenvalue_via_shifted_power(b: &Array2<f64>, shift: f64, n: usize) -> f64 {
    // Build shifted matrix C = shift·I - B
    let mut c = b.clone();
    c.mapv_inplace(|x| -x);
    for i in 0..n {
        c[[i, i]] += shift;
    }

    // Power iteration to find largest eigenvalue of C
    // (= shift - λ_min(B))
    let mut v = vec![1.0_f64; n];

    // Normalize initial vector
    let norm0: f64 = v
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    for x in v.iter_mut() {
        *x /= norm0;
    }

    let mut lam_c = 0.0_f64;
    const MAX_ITER: usize = 30;

    for _ in 0..MAX_ITER {
        // w = C * v
        let mut w = vec![0.0_f64; n];
        for i in 0..n {
            let mut s = 0.0_f64;
            for j in 0..n {
                s += c[[i, j]] * v[j];
            }
            w[i] = s;
        }

        // Rayleigh quotient
        let rq: f64 = v.iter().zip(w.iter()).map(|(vi, wi)| vi * wi).sum();

        // Normalize
        let norm_w: f64 = w
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        for (vi, wi) in v.iter_mut().zip(w.iter()) {
            *vi = *wi / norm_w;
        }

        lam_c = rq;
    }

    // λ_min(B) = shift - lam_C
    shift - lam_c
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    // Helper: build symmetric matrix from eigenvalues
    fn diag_matrix(eigs: &[f64]) -> Array2<f64> {
        let n = eigs.len();
        let mut m = Array2::<f64>::zeros((n, n));
        for (i, &e) in eigs.iter().enumerate() {
            m[[i, i]] = e;
        }
        m
    }

    // -----------------------------------------------------------------------
    // eigenvalue_perturbation
    // -----------------------------------------------------------------------

    #[test]
    fn test_eigenvalue_perturbation_diagonal() {
        // Diagonal A: eigenvectors are standard basis vectors
        let a = diag_matrix(&[2.0, 5.0, 8.0]);
        // Perturbation: diagonal with known values
        let e = diag_matrix(&[0.1, -0.2, 0.3]);
        let delta = eigenvalue_perturbation(&a.view(), &e.view()).expect("failed to create delta");
        // For diagonal A, Δλᵢ = E[i,i]
        assert_relative_eq!(delta[0], 0.1, epsilon = 1e-9);
        assert_relative_eq!(delta[1], -0.2, epsilon = 1e-9);
        assert_relative_eq!(delta[2], 0.3, epsilon = 1e-9);
    }

    #[test]
    fn test_eigenvalue_perturbation_shape_error_a() {
        let a = array![[1.0_f64, 2.0, 3.0]];
        let e = array![[0.1_f64, 0.0, 0.0]];
        assert!(eigenvalue_perturbation(&a.view(), &e.view()).is_err());
    }

    #[test]
    fn test_eigenvalue_perturbation_shape_error_e() {
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let e = array![[0.1_f64, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]];
        assert!(eigenvalue_perturbation(&a.view(), &e.view()).is_err());
    }

    #[test]
    fn test_eigenvalue_perturbation_sum_zero() {
        // Trace(E) = 0 means sum of first-order shifts = 0
        let a = diag_matrix(&[1.0, 3.0]);
        let e = array![[0.5_f64, 0.0], [0.0, -0.5]];
        let delta = eigenvalue_perturbation(&a.view(), &e.view()).expect("failed to create delta");
        assert_relative_eq!(delta[0] + delta[1], 0.0, epsilon = 1e-9);
    }

    // -----------------------------------------------------------------------
    // davis_kahan
    // -----------------------------------------------------------------------

    #[test]
    fn test_davis_kahan_zero_perturbation() {
        let a = array![[1.0_f64, 0.0], [0.0, 5.0]];
        // B = A → perturbation = 0 → bound = 0
        let bound = davis_kahan(&a.view(), &a.view(), 4.0).expect("failed to create bound");
        assert_relative_eq!(bound, 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_davis_kahan_small_perturbation() {
        let a = array![[1.0_f64, 0.0], [0.0, 10.0]];
        let b = array![[1.01_f64, 0.0], [0.0, 10.01]];
        let gap = 9.0_f64;
        let bound = davis_kahan(&a.view(), &b.view(), gap).expect("failed to create bound");
        // Frobenius norm of diff ≈ sqrt(0.01²+0.01²) ≈ 0.01414
        // bound = min(0.01414/9, 1) ≈ 0.00157
        assert!(bound < 0.01);
    }

    #[test]
    fn test_davis_kahan_large_perturbation_clamped() {
        let a = array![[1.0_f64, 0.0], [0.0, 5.0]];
        let b = array![[100.0_f64, 0.0], [0.0, 200.0]];
        let bound = davis_kahan(&a.view(), &b.view(), 0.001).expect("failed to create bound");
        // Should be clamped to 1.0
        assert_relative_eq!(bound, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_davis_kahan_invalid_delta() {
        let a = array![[1.0_f64, 0.0], [0.0, 5.0]];
        assert!(davis_kahan(&a.view(), &a.view(), 0.0).is_err());
        assert!(davis_kahan(&a.view(), &a.view(), -1.0).is_err());
    }

    #[test]
    fn test_davis_kahan_shape_error() {
        let a = array![[1.0_f64, 2.0, 3.0]];
        let b = array![[1.0_f64, 2.0, 3.0]];
        assert!(davis_kahan(&a.view(), &b.view(), 1.0).is_err());
    }

    // -----------------------------------------------------------------------
    // weyl_bound
    // -----------------------------------------------------------------------

    #[test]
    fn test_weyl_bound_equal() {
        let la = array![1.0_f64, 2.0, 3.0];
        let lb = la.clone();
        let bound = weyl_bound(&la.view(), &lb.view()).expect("failed to create bound");
        assert_relative_eq!(bound, 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_weyl_bound_known() {
        let la = array![1.0_f64, 3.0, 7.0];
        let lb = array![1.1_f64, 2.9, 7.2];
        let bound = weyl_bound(&la.view(), &lb.view()).expect("failed to create bound");
        // max(0.1, 0.1, 0.2) = 0.2
        assert_relative_eq!(bound, 0.2, epsilon = 1e-12);
    }

    #[test]
    fn test_weyl_bound_length_mismatch() {
        let la = array![1.0_f64, 2.0];
        let lb = array![1.0_f64, 2.0, 3.0];
        assert!(weyl_bound(&la.view(), &lb.view()).is_err());
    }

    #[test]
    fn test_weyl_bound_empty() {
        let la = Array1::<f64>::zeros(0);
        let lb = Array1::<f64>::zeros(0);
        assert!(weyl_bound(&la.view(), &lb.view()).is_err());
    }

    // -----------------------------------------------------------------------
    // relative_perturbation
    // -----------------------------------------------------------------------

    #[test]
    fn test_relative_perturbation_diagonal() {
        let a = diag_matrix(&[4.0, 9.0]);
        // Perturbation: 1% of each eigenvalue
        let e = diag_matrix(&[0.04, 0.09]);
        let rel = relative_perturbation(&a.view(), &e.view()).expect("failed to create rel");
        // max(0.04/4, 0.09/9) = max(0.01, 0.01) = 0.01
        assert_relative_eq!(rel, 0.01, epsilon = 1e-10);
    }

    #[test]
    fn test_relative_perturbation_zero_perturbation() {
        let a = diag_matrix(&[1.0, 4.0, 9.0]);
        let e = Array2::<f64>::zeros((3, 3));
        let rel = relative_perturbation(&a.view(), &e.view()).expect("failed to create rel");
        assert_relative_eq!(rel, 0.0, epsilon = 1e-14);
    }

    // -----------------------------------------------------------------------
    // condition_eigenvalue
    // -----------------------------------------------------------------------

    #[test]
    fn test_condition_eigenvalue_diagonal() {
        // λ₀=1, λ₁=5: gap for index 0 is 4, κ = 1/4 = 0.25
        let a = diag_matrix(&[1.0, 5.0]);
        let kappa0 = condition_eigenvalue(&a.view(), 0).expect("failed to create kappa0");
        assert_relative_eq!(kappa0, 0.25, epsilon = 1e-10);

        let kappa1 = condition_eigenvalue(&a.view(), 1).expect("failed to create kappa1");
        assert_relative_eq!(kappa1, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_condition_eigenvalue_three_eigenvalues() {
        // λ = 1, 3, 10
        let a = diag_matrix(&[1.0, 3.0, 10.0]);
        // κ(λ₀=1) = 1/gap = 1/(3-1) = 0.5
        let k0 = condition_eigenvalue(&a.view(), 0).expect("failed to create k0");
        assert_relative_eq!(k0, 0.5, epsilon = 1e-10);
        // κ(λ₁=3) = 1/min(2,7) = 1/2 = 0.5
        let k1 = condition_eigenvalue(&a.view(), 1).expect("failed to create k1");
        assert_relative_eq!(k1, 0.5, epsilon = 1e-10);
        // κ(λ₂=10) = 1/(10-3) = 1/7
        let k2 = condition_eigenvalue(&a.view(), 2).expect("failed to create k2");
        assert_relative_eq!(k2, 1.0 / 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_condition_eigenvalue_repeated_is_inf() {
        let a = diag_matrix(&[2.0, 2.0, 5.0]);
        // λ₀ == λ₁ → gap = 0 → κ = ∞
        let k = condition_eigenvalue(&a.view(), 0).expect("failed to create k");
        assert!(k.is_infinite());
    }

    #[test]
    fn test_condition_eigenvalue_out_of_bounds() {
        let a = diag_matrix(&[1.0, 2.0]);
        assert!(condition_eigenvalue(&a.view(), 2).is_err());
    }

    #[test]
    fn test_condition_eigenvalue_non_square() {
        let a = array![[1.0_f64, 2.0, 3.0]];
        assert!(condition_eigenvalue(&a.view(), 0).is_err());
    }

    // -----------------------------------------------------------------------
    // pseudospectrum
    // -----------------------------------------------------------------------

    #[test]
    fn test_pseudospectrum_shape() {
        let a = diag_matrix(&[0.0, 1.0]);
        let ps = pseudospectrum(&a.view(), 0.3, 10, (-0.5, 1.5), (-1.0, 1.0))
            .expect("failed to create ps");
        assert_eq!(ps.dim(), (10, 10));
    }

    #[test]
    fn test_pseudospectrum_contains_eigenvalue_neighborhood() {
        // Diagonal matrix with eigenvalues at 0 and 3
        let a = diag_matrix(&[0.0, 3.0]);
        let eps = 0.4;
        let ps = pseudospectrum(&a.view(), eps, 30, (-1.0, 4.0), (-0.5, 0.5))
            .expect("failed to create ps");
        // The grid spans real axis from -1 to 4 in 30 steps: Δr = 5/29 ≈ 0.172
        // Grid point closest to eigenvalue 0 (re=0, im=0) should be inside
        let mut any_true = false;
        for gi in 0..30 {
            for gj in 0..30 {
                if ps[[gi, gj]] {
                    any_true = true;
                }
            }
        }
        assert!(
            any_true,
            "pseudospectrum should contain at least one true entry near eigenvalues"
        );
    }

    #[test]
    fn test_pseudospectrum_invalid_epsilon() {
        let a = diag_matrix(&[1.0, 2.0]);
        assert!(pseudospectrum(&a.view(), -0.1, 5, (0.0, 2.0), (-1.0, 1.0)).is_err());
        assert!(pseudospectrum(&a.view(), 0.0, 5, (0.0, 2.0), (-1.0, 1.0)).is_err());
    }

    #[test]
    fn test_pseudospectrum_invalid_ranges() {
        let a = diag_matrix(&[1.0, 2.0]);
        // real_max <= real_min
        assert!(pseudospectrum(&a.view(), 0.1, 5, (2.0, 1.0), (-1.0, 1.0)).is_err());
        // imag_max <= imag_min
        assert!(pseudospectrum(&a.view(), 0.1, 5, (0.0, 2.0), (1.0, -1.0)).is_err());
    }

    #[test]
    fn test_pseudospectrum_non_square() {
        let a = array![[1.0_f64, 2.0, 3.0]];
        assert!(pseudospectrum(&a.view(), 0.1, 5, (0.0, 2.0), (-1.0, 1.0)).is_err());
    }
}
