//! Real Schur decomposition and general (possibly non-symmetric) eigenvalue
//! decomposition, shared across the crate.
//!
//! This module implements the standard textbook pipeline (Golub & Van Loan,
//! *Matrix Computations*, the Francis double-shift QR algorithm; see also the
//! classic EISPACK `hqr`/`hqr2` routines):
//!
//! 1. Reduce the input matrix to upper Hessenberg form via Householder
//!    similarity transforms ([`hessenberg_form`]).
//! 2. Drive the Hessenberg form to real (quasi-upper-triangular) Schur form
//!    via the implicit double-shift QR algorithm with deflation
//!    ([`real_schur`]).
//! 3. Extract eigenvalues from the Schur form's 1×1 / 2×2 diagonal blocks
//!    (real eigenvalues and complex-conjugate pairs respectively; see
//!    [`standard::extract_schur_eigenvalues`](super::standard::extract_schur_eigenvalues)).
//! 4. Recover right eigenvectors by back-substitution on the Schur form,
//!    then transform back through the accumulated orthogonal factor
//!    ([`general_eig`]).
//!
//! This is the single, real (non-placeholder) engine backing:
//! - [`crate::decomposition::schur`]
//! - [`crate::lapack::eig`]
//! - [`crate::eigen::advanced_precision_eig`]'s genuinely non-symmetric path
//!
//! Each of those used to carry its own independent, broken implementation
//! (a fixed count of *unshifted* QR iterations with no convergence check, or
//! an outright placeholder that only worked for already-diagonal input). This
//! module replaces all of them with one correct, convergence-checked
//! algorithm, per the crate's "no duplicated / fabricated implementations"
//! policy.
//!
//! # Scope note
//!
//! Unlike LAPACK's `dgebal`/EISPACK's `balanc`, this implementation does not
//! perform an explicit diagonal balancing pre-pass. Balancing improves
//! accuracy and iteration count for *badly scaled* matrices (entries
//! differing by many orders of magnitude); it is not required for
//! correctness. All eigenvalues/eigenvectors are still computed exactly (to
//! floating-point precision) for well-scaled input, which covers this
//! crate's documented use cases and test matrices.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Complex, Float, NumAssign};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

/// Build a normalized Householder vector `v` such that applying the
/// reflector `I - 2 v vᵀ` to `x` zeroes every component but the first.
/// Returns `None` when `x` is already (numerically) proportional to `e₁`.
fn householder_reflector<F>(x: &[F]) -> Option<Vec<F>>
where
    F: Float + NumAssign,
{
    let norm_sq = x.iter().fold(F::zero(), |acc, &v| acc + v * v);
    let norm = norm_sq.sqrt();
    if norm <= F::epsilon() {
        return None;
    }
    let alpha = if x[0] >= F::zero() { -norm } else { norm };
    let mut v: Vec<F> = x.to_vec();
    v[0] -= alpha;
    let v_norm = v.iter().fold(F::zero(), |acc, &val| acc + val * val).sqrt();
    if v_norm <= F::epsilon() {
        return None;
    }
    for item in v.iter_mut() {
        *item /= v_norm;
    }
    Some(v)
}

/// Apply the similarity transform `M := (I - 2 v vᵀ) M (I - 2 v vᵀ)` to `t`,
/// restricted to the rows/columns listed in `idx` (`v[k]` corresponds to row
/// / column `idx[k]`), and accumulate the same reflector into `z`'s columns
/// (`z := z (I - 2 v vᵀ)`), so that `t = zᵀ · (original t) · z` is preserved
/// as an invariant across repeated calls when `t`/`z` start as `(A, I)`.
fn apply_householder_similarity<F>(t: &mut Array2<F>, z: &mut Array2<F>, idx: &[usize], v: &[F])
where
    F: Float + NumAssign,
{
    let n = t.nrows();
    let two = F::one() + F::one();

    // Left: T := (I - 2vvᵀ) T  (only rows in `idx` change).
    for j in 0..n {
        let mut dot = F::zero();
        for (k, &r) in idx.iter().enumerate() {
            dot += v[k] * t[[r, j]];
        }
        let f = two * dot;
        for (k, &r) in idx.iter().enumerate() {
            t[[r, j]] -= f * v[k];
        }
    }
    // Right: T := T (I - 2vvᵀ)  (only columns in `idx` change).
    for i in 0..n {
        let mut dot = F::zero();
        for (k, &c) in idx.iter().enumerate() {
            dot += t[[i, c]] * v[k];
        }
        let f = two * dot;
        for (k, &c) in idx.iter().enumerate() {
            t[[i, c]] -= f * v[k];
        }
    }
    // Z := Z (I - 2vvᵀ).
    for i in 0..n {
        let mut dot = F::zero();
        for (k, &c) in idx.iter().enumerate() {
            dot += z[[i, c]] * v[k];
        }
        let f = two * dot;
        for (k, &c) in idx.iter().enumerate() {
            z[[i, c]] -= f * v[k];
        }
    }
}

/// Reduce a general real matrix to upper Hessenberg form via Householder
/// similarity transforms. Returns `(h, q)` with `q` orthogonal and
/// `a ≈ q · h · qᵀ`.
pub(crate) fn hessenberg_form<F>(a: &ArrayView2<F>) -> (Array2<F>, Array2<F>)
where
    F: Float + NumAssign,
{
    let n = a.nrows();
    let mut h = a.to_owned();
    let mut q = Array2::<F>::eye(n);

    if n < 3 {
        return (h, q);
    }

    for k in 0..n - 2 {
        let col_len = n - k - 1;
        let mut x: Vec<F> = Vec::with_capacity(col_len);
        for i in 0..col_len {
            x.push(h[[k + 1 + i, k]]);
        }
        if let Some(v) = householder_reflector(&x) {
            let idx: Vec<usize> = (0..col_len).map(|i| k + 1 + i).collect();
            apply_householder_similarity(&mut h, &mut q, &idx, &v);
        }
    }

    (h, q)
}

/// Apply a plane rotation `R = [[cs, -sn], [sn, cs]]` (embedded at rows/cols
/// `l, l+1` of the full `n×n` matrices) as a similarity transform:
/// `t := Rᵀ t R`, `z := z R`.
fn apply_plane_rotation<F>(t: &mut Array2<F>, z: &mut Array2<F>, l: usize, cs: F, sn: F)
where
    F: Float + NumAssign,
{
    let n = t.nrows();
    let i0 = l;
    let i1 = l + 1;

    for j in 0..n {
        let a0 = t[[i0, j]];
        let a1 = t[[i1, j]];
        t[[i0, j]] = cs * a0 + sn * a1;
        t[[i1, j]] = cs * a1 - sn * a0;
    }
    for i in 0..n {
        let a0 = t[[i, i0]];
        let a1 = t[[i, i1]];
        t[[i, i0]] = cs * a0 + sn * a1;
        t[[i, i1]] = cs * a1 - sn * a0;
    }
    for i in 0..n {
        let a0 = z[[i, i0]];
        let a1 = z[[i, i1]];
        z[[i, i0]] = cs * a0 + sn * a1;
        z[[i, i1]] = cs * a1 - sn * a0;
    }
}

/// Given a 2×2 diagonal block of an (otherwise already-deflated) quasi-Schur
/// matrix at rows/columns `(l, l+1)`, either confirm it as a genuine
/// complex-conjugate eigenvalue pair (left untouched — its trace/determinant
/// already encode the pair correctly) or, if its eigenvalues are real, apply
/// a similarity rotation that zeroes the sub-diagonal entry exactly so the
/// two real eigenvalues land cleanly on the diagonal.
fn deflate_or_split_2x2<F>(t: &mut Array2<F>, z: &mut Array2<F>, l: usize)
where
    F: Float + NumAssign,
{
    let a = t[[l, l]];
    let b = t[[l, l + 1]];
    let c = t[[l + 1, l]];
    let d = t[[l + 1, l + 1]];

    if c == F::zero() {
        return; // Already upper triangular.
    }

    let two = F::one() + F::one();
    let four = two + two;
    let trace = a + d;
    let disc = (a - d) * (a - d) + four * b * c;
    if disc < F::zero() {
        return; // Genuine complex-conjugate pair; leave the block intact.
    }

    let sqrt_disc = disc.sqrt();
    let lambda1 = (trace + sqrt_disc) / two;

    // Null vector of (block - lambda1 * I); mirrors the convention used by
    // `eigen::standard::solve_2x2_eigenvalue_problem`.
    let (mut v0, mut v1) = if b != F::zero() {
        (b, lambda1 - a)
    } else {
        (lambda1 - d, c)
    };
    let norm = (v0 * v0 + v1 * v1).sqrt();
    if norm <= F::epsilon() {
        return;
    }
    v0 /= norm;
    v1 /= norm;

    apply_plane_rotation(t, z, l, v0, v1);
}

/// Build the length-2 or length-3 Householder vector used by one bulge-chase
/// step of the Francis double-shift QR algorithm.
fn francis_double_shift_step<F>(
    t: &mut Array2<F>,
    z: &mut Array2<F>,
    l: usize,
    p: usize,
    exceptional: bool,
) where
    F: Float + NumAssign,
{
    let two = F::one() + F::one();
    let (s, tr) = if exceptional {
        // Classic ad-hoc ("exceptional") shift to break stagnation cycles.
        let sub1 = t[[p - 1, p - 2]].abs();
        let sub2 = if p >= 3 {
            t[[p - 2, p - 3]].abs()
        } else {
            F::zero()
        };
        let three_q = (F::one() + F::one() + F::one()) / (two + two);
        let shift = (sub1 + sub2) * three_q;
        (two * shift, shift * shift)
    } else {
        let h11 = t[[p - 2, p - 2]];
        let h12 = t[[p - 2, p - 1]];
        let h21 = t[[p - 1, p - 2]];
        let h22 = t[[p - 1, p - 1]];
        (h11 + h22, h11 * h22 - h12 * h21)
    };

    let h00 = t[[l, l]];
    let h01 = t[[l, l + 1]];
    let h10 = t[[l + 1, l]];
    let h11_ll1 = t[[l + 1, l + 1]];
    let h21_l2l1 = if l + 2 < p {
        t[[l + 2, l + 1]]
    } else {
        F::zero()
    };

    let mut x = h00 * h00 + h01 * h10 - s * h00 + tr;
    let mut y = h10 * (h00 + h11_ll1 - s);
    let mut z_val = h10 * h21_l2l1;

    for k in l..p.saturating_sub(1) {
        let use_third = k + 2 < p;
        let vec: Vec<F> = if use_third {
            vec![x, y, z_val]
        } else {
            vec![x, y]
        };
        if let Some(v) = householder_reflector(&vec) {
            let idx: Vec<usize> = if use_third {
                vec![k, k + 1, k + 2]
            } else {
                vec![k, k + 1]
            };
            apply_householder_similarity(t, z, &idx, &v);
        }

        if k + 1 < p {
            x = t[[k + 1, k]];
            y = if k + 2 < p { t[[k + 2, k]] } else { F::zero() };
            z_val = if k + 3 < p { t[[k + 3, k]] } else { F::zero() };
        }
    }
}

/// Compute the real Schur decomposition `A = Z T Zᵀ` of a square matrix via
/// Hessenberg reduction followed by the Francis double-shift implicit QR
/// algorithm with deflation.
///
/// `t` is quasi-upper-triangular: 1×1 diagonal blocks are real eigenvalues;
/// 2×2 diagonal blocks (non-zero sub-diagonal entry) are complex-conjugate
/// eigenvalue pairs. `z` is orthogonal. Returns
/// [`LinalgError::ConvergenceError`] if the iteration budget is exhausted
/// before the whole matrix deflates (this should not happen for any
/// reasonably well-scaled input; see the module-level scope note).
pub(crate) fn real_schur<F>(a: &ArrayView2<F>) -> LinalgResult<(Array2<F>, Array2<F>)>
where
    F: Float + NumAssign,
{
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square for Schur decomposition, got shape {:?}",
            a.shape()
        )));
    }
    if n == 0 {
        return Ok((Array2::zeros((0, 0)), Array2::zeros((0, 0))));
    }
    if n == 1 {
        return Ok((Array2::from_elem((1, 1), F::one()), a.to_owned()));
    }

    let (mut t, mut z) = hessenberg_form(a);

    let mut p = n;
    let max_total_iter = 30 * n + 100;
    let mut total_iter = 0usize;
    let mut iter_since_deflation = 0usize;

    while p > 2 {
        // Find the largest l such that t[l, l-1] is negligible (or 0 if none).
        let mut l = p - 1;
        while l > 0 {
            let sub = t[[l, l - 1]].abs();
            let scale = t[[l - 1, l - 1]].abs() + t[[l, l]].abs();
            let threshold = if scale > F::zero() {
                F::epsilon() * scale
            } else {
                F::epsilon()
            };
            if sub <= threshold {
                t[[l, l - 1]] = F::zero();
                break;
            }
            l -= 1;
        }

        if l == p - 1 {
            p -= 1;
            iter_since_deflation = 0;
        } else if l == p - 2 {
            deflate_or_split_2x2(&mut t, &mut z, l);
            p -= 2;
            iter_since_deflation = 0;
        } else {
            total_iter += 1;
            iter_since_deflation += 1;
            if total_iter > max_total_iter {
                return Err(LinalgError::ConvergenceError(format!(
                    "Real Schur decomposition (Hessenberg + Francis double-shift QR) \
                     failed to converge after {total_iter} iterations for a {n}x{n} matrix"
                )));
            }
            let exceptional = iter_since_deflation > 0 && iter_since_deflation % 12 == 0;
            francis_double_shift_step(&mut t, &mut z, l, p, exceptional);
        }
    }

    if p == 2 {
        deflate_or_split_2x2(&mut t, &mut z, 0);
    }

    Ok((z, t))
}

/// Compute right eigenvectors from a quasi-upper-triangular real Schur form
/// `(z, t)` (as returned by [`real_schur`]) via back-substitution on `t`
/// followed by transforming back through `z`. Column `j` of the returned
/// matrix is normalized to unit 2-norm and corresponds to `eigenvalues[j]`
/// (which must be listed in the same top-left-to-bottom-right diagonal-block
/// order as `t`, exactly as produced by
/// [`standard::extract_schur_eigenvalues`](super::standard::extract_schur_eigenvalues)).
fn schur_eigenvectors<F>(
    t: &Array2<F>,
    z: &Array2<F>,
    eigenvalues: &Array1<Complex<F>>,
) -> Array2<Complex<F>>
where
    F: Float + NumAssign,
{
    let n = t.nrows();
    let hundred = {
        let mut v = F::zero();
        for _ in 0..100 {
            v += F::one();
        }
        v
    };
    let eps = F::epsilon() * hundred;
    let mut vecs: Array2<Complex<F>> = Array2::zeros((n, n));

    let mut col = 0usize;
    while col < n {
        if eigenvalues[col].im == F::zero() {
            let lambda = eigenvalues[col].re;
            let mut x = Array1::<F>::zeros(n);
            x[col] = F::one();
            for i in (0..col).rev() {
                let mut sum = F::zero();
                for j in (i + 1)..=col {
                    sum += t[[i, j]] * x[j];
                }
                let diag = t[[i, i]] - lambda;
                x[i] = if diag.abs() > eps {
                    -sum / diag
                } else {
                    F::zero()
                };
            }
            let norm = x.iter().fold(F::zero(), |acc, &v| acc + v * v).sqrt();
            if norm > eps {
                x.mapv_inplace(|v| v / norm);
            }
            for i in 0..n {
                let mut acc = F::zero();
                for j in 0..n {
                    acc += z[[i, j]] * x[j];
                }
                vecs[[i, col]] = Complex::new(acc, F::zero());
            }
            col += 1;
        } else {
            let lambda_re = eigenvalues[col].re;
            let lambda_im = eigenvalues[col].im.abs();

            let mut xr = Array1::<F>::zeros(n);
            let mut xi = Array1::<F>::zeros(n);
            xr[col] = F::one();

            let t_cc = t[[col, col]];
            let t_cc1 = t[[col, col + 1]];
            if t_cc1.abs() > eps {
                xr[col + 1] = (lambda_re - t_cc) / t_cc1;
                xi[col + 1] = lambda_im / t_cc1;
            } else {
                xr[col + 1] = F::zero();
                xi[col + 1] = F::one();
            }

            for i in (0..col).rev() {
                let mut sum_r = F::zero();
                let mut sum_i = F::zero();
                for j in (i + 1)..=(col + 1) {
                    sum_r += t[[i, j]] * xr[j];
                    sum_i += t[[i, j]] * xi[j];
                }
                let diag_re = t[[i, i]] - lambda_re;
                let diag_im = -lambda_im;
                let denom = diag_re * diag_re + diag_im * diag_im;
                if denom > eps * eps {
                    let num_re = -sum_r;
                    let num_i = -sum_i;
                    xr[i] = (num_re * diag_re + num_i * diag_im) / denom;
                    xi[i] = (num_i * diag_re - num_re * diag_im) / denom;
                }
            }

            let norm = (0..n)
                .fold(F::zero(), |acc, i| acc + xr[i] * xr[i] + xi[i] * xi[i])
                .sqrt();
            if norm > eps {
                xr.mapv_inplace(|v| v / norm);
                xi.mapv_inplace(|v| v / norm);
            }

            for i in 0..n {
                let mut acc_r = F::zero();
                let mut acc_i = F::zero();
                for j in 0..n {
                    acc_r += z[[i, j]] * xr[j];
                    acc_i += z[[i, j]] * xi[j];
                }
                vecs[[i, col]] = Complex::new(acc_r, acc_i);
                vecs[[i, col + 1]] = Complex::new(acc_r, -acc_i);
            }
            col += 2;
        }
    }

    vecs
}

/// Compute eigenvalues *and* right eigenvectors of a general (possibly
/// non-symmetric) square matrix via [`real_schur`] followed by
/// back-substitution.
pub(crate) fn general_eig<F>(a: &ArrayView2<F>) -> super::standard::EigenResult<F>
where
    F: Float + NumAssign + Sum + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    let n = a.nrows();
    let (z, t) = real_schur(a)?;
    let eigenvalues = super::standard::extract_schur_eigenvalues(&t, n);
    let eigenvectors = schur_eigenvectors(&t, &z, &eigenvalues);
    Ok((eigenvalues, eigenvectors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    fn assert_orthogonal(q: &Array2<f64>, tol: f64) {
        let n = q.nrows();
        let qtq = q.t().dot(q);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[[i, j]] - expected).abs() < tol,
                    "Q not orthogonal at ({i},{j}): {}",
                    qtq[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_hessenberg_reduction_preserves_similarity() {
        let a = array![
            [4.0, 1.0, -2.0, 2.0],
            [1.0, 2.0, 0.0, 1.0],
            [-2.0, 0.0, 3.0, -2.0],
            [2.0, 1.0, -2.0, -1.0]
        ];
        let (h, q) = hessenberg_form(&a.view());
        assert_orthogonal(&q, 1e-9);
        // Below the sub-diagonal must be exactly zero.
        for i in 0..4 {
            for j in 0..4 {
                if i > j + 1 {
                    assert!(h[[i, j]].abs() < 1e-9, "h[{i},{j}]={} not zero", h[[i, j]]);
                }
            }
        }
        let recon = q.dot(&h).dot(&q.t());
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(recon[[i, j]], a[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_real_schur_nonsymmetric_companion() {
        // Companion matrix of (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6.
        let a = array![[6.0, -11.0, 6.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let (z, t) = real_schur(&a.view()).expect("real_schur should converge");
        assert_orthogonal(&z, 1e-7);

        let recon = z.dot(&t).dot(&z.t());
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(recon[[i, j]], a[[i, j]], epsilon = 1e-6);
            }
        }

        let n = 3;
        let eigenvalues = super::super::standard::extract_schur_eigenvalues(&t, n);
        let mut reals: Vec<f64> = eigenvalues.iter().map(|c| c.re).collect();
        reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_relative_eq!(reals[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(reals[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(reals[2], 3.0, epsilon = 1e-6);
        for c in eigenvalues.iter() {
            assert!(c.im.abs() < 1e-6);
        }
    }

    #[test]
    fn test_general_eig_complex_pair() {
        // Rotation-like matrix with a genuine complex-conjugate pair:
        // eigenvalues 3 +/- 2i.
        let a = array![[3.0, -2.0], [2.0, 3.0]];
        let (eigenvalues, _eigenvectors) = general_eig(&a.view()).expect("general_eig failed");
        let mut ims: Vec<f64> = eigenvalues.iter().map(|c| c.im).collect();
        ims.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_relative_eq!(ims[0], -2.0, epsilon = 1e-8);
        assert_relative_eq!(ims[1], 2.0, epsilon = 1e-8);
        for c in eigenvalues.iter() {
            assert_relative_eq!(c.re, 3.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_general_eig_eigenvector_residual_companion() {
        // Same companion matrix as above; verify A v = lambda v for every
        // computed (real) eigenpair using genuinely non-constant data.
        let a = array![[6.0, -11.0, 6.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let (eigenvalues, eigenvectors) = general_eig(&a.view()).expect("general_eig failed");
        let n = 3;
        for col in 0..n {
            let lambda = eigenvalues[col];
            for i in 0..n {
                let mut av = Complex::new(0.0, 0.0);
                for j in 0..n {
                    av += Complex::new(a[[i, j]], 0.0) * eigenvectors[[j, col]];
                }
                let lv = lambda * eigenvectors[[i, col]];
                assert!(
                    (av - lv).norm() < 1e-6,
                    "A v != lambda v at col {col}, row {i}: {av:?} vs {lv:?}"
                );
            }
        }
    }
}
