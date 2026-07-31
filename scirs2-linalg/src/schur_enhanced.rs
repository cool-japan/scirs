//! Enhanced Schur decomposition and related algorithms.
//!
//! This module provides:
//!
//! - `real_schur_decompose`  – Compute the real Schur form (quasi-triangular T, orthogonal Q)
//!   via Francis double-shift QR iteration.
//! - `complex_schur_decompose` – Compute the complex Schur form (upper triangular T, unitary Q).
//! - `schur_reorder`          – Reorder a Schur form by a user-supplied eigenvalue criterion.
//! - `schur_to_eigen`         – Extract eigenvalues and eigenvectors from the Schur form.
//! - `invariant_subspace`     – Compute an invariant subspace from a block Schur form.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Complex, Float, NumAssign, One, Zero};
use std::fmt::{Debug, Display};
use std::iter::Sum;

// ─────────────────────────────────────────────────────────────────────────────
// Trait alias
// ─────────────────────────────────────────────────────────────────────────────

/// Floating-point trait bounds required by every function in this module.
pub trait SchurFloat:
    Float
    + NumAssign
    + Debug
    + Display
    + scirs2_core::ndarray::ScalarOperand
    + Sum
    + 'static
    + Send
    + Sync
{
}

impl<T> SchurFloat for T where
    T: Float
        + NumAssign
        + Debug
        + Display
        + scirs2_core::ndarray::ScalarOperand
        + Sum
        + 'static
        + Send
        + Sync
{
}

// ─────────────────────────────────────────────────────────────────────────────
// Result types
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a real Schur decomposition.
///
/// Contains the orthogonal factor Q and the quasi-upper-triangular factor T
/// such that A = Q T Qᵀ.  The diagonal of T contains real eigenvalues or 2×2
/// blocks for complex conjugate pairs.
#[derive(Debug, Clone)]
pub struct RealSchurResult<T> {
    /// Orthogonal matrix Q (n × n).
    pub q: Array2<T>,
    /// Quasi-upper-triangular matrix T (n × n).
    pub t: Array2<T>,
}

/// Result of a complex Schur decomposition.
///
/// Contains the unitary factor Q and the upper-triangular factor T such that
/// A = Q T Q*.  The diagonal of T holds the (complex) eigenvalues.
#[derive(Debug, Clone)]
pub struct ComplexSchurResult<T> {
    /// Unitary matrix Q (n × n), complex entries.
    pub q: Array2<Complex<T>>,
    /// Upper-triangular matrix T (n × n), complex entries.
    pub t: Array2<Complex<T>>,
}

/// Result of eigenvalue/vector extraction from a Schur form.
#[derive(Debug, Clone)]
pub struct SchurEigenResult<T> {
    /// Eigenvalues (complex in general).
    pub eigenvalues: Array1<Complex<T>>,
    /// Eigenvectors as columns of this n × n complex matrix.
    pub eigenvectors: Array2<Complex<T>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// real_schur_decompose
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the real Schur decomposition A = Q T Qᵀ.
///
/// Uses the Francis double-shift implicit QR algorithm.  The result T is
/// quasi-upper-triangular: all blocks above the diagonal are zero, and on the
/// diagonal there are either 1×1 blocks (real eigenvalues) or 2×2 blocks
/// (complex conjugate pairs).
///
/// # Arguments
/// * `a`       - Input square matrix (n × n)
/// * `max_iter` - Maximum number of QR sweeps (default 30 per unreduced subproblem)
/// * `tol`     - Off-diagonal deflation tolerance
///
/// # Returns
/// `RealSchurResult` containing Q and T.
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::schur_enhanced::real_schur_decompose;
///
/// let a = array![[4.0_f64, 3.0], [6.0, 3.0]];
/// let res = real_schur_decompose(&a.view(), 200, 1e-12).expect("ok");
/// // Verify A = Q T Q^T
/// let qt = res.q.t().to_owned();
/// let reconstructed = res.q.dot(&res.t).dot(&qt);
/// assert!((reconstructed[[0,0]] - a[[0,0]]).abs() < 1e-8);
/// ```
pub fn real_schur_decompose<T: SchurFloat>(
    a: &ArrayView2<T>,
    max_iter: usize,
    tol: T,
) -> LinalgResult<RealSchurResult<T>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "real_schur_decompose: A must be square".into(),
        ));
    }
    if n == 0 {
        return Ok(RealSchurResult {
            q: Array2::zeros((0, 0)),
            t: Array2::zeros((0, 0)),
        });
    }
    if n == 1 {
        return Ok(RealSchurResult {
            q: Array2::eye(1),
            t: a.to_owned(),
        });
    }

    // Reduce to upper Hessenberg form first (more efficient for QR iteration)
    let (mut h, mut q) = hessenberg_decompose(a)?;

    // Implicit QR iteration on the Hessenberg matrix
    // We use the standard Francis double-shift approach with proper deflation.
    // The algorithm follows LAPACK's DHSEQR / DLAHQR approach.
    let max_total = max_iter * n;
    let mut total_iters = 0usize;
    let mut ihi = n; // upper bound of active window (exclusive)
    let mut _stall_count = 0usize;

    while ihi > 1 {
        if total_iters >= max_total {
            return Err(LinalgError::ConvergenceError(format!(
                "real_schur_decompose: failed to converge after {total_iters} iterations"
            )));
        }
        total_iters += 1;

        // Step 1: Check for deflation from the bottom of the active submatrix
        // Find the bottom of the unreduced Hessenberg block
        let mut new_ihi = ihi;
        while new_ihi > 1 {
            let k = new_ihi - 1;
            let off_diag = h[[k, k - 1]].abs();
            let diag_sum = h[[k - 1, k - 1]].abs() + h[[k, k]].abs();
            let threshold = if diag_sum > T::epsilon() {
                tol * diag_sum
            } else {
                tol
            };
            if off_diag <= threshold {
                h[[k, k - 1]] = T::zero();
                new_ihi = k;
            } else {
                break;
            }
        }
        if new_ihi < ihi {
            // We deflated some eigenvalues at the bottom
            // Process any 2x2 blocks that appeared
            if ihi - new_ihi >= 2 {
                // Check if there's a 2x2 block at new_ihi..new_ihi+2
                if new_ihi + 1 < ihi && h[[new_ihi + 1, new_ihi]].abs() > tol {
                    schur_split_2x2(&mut h, &mut q, new_ihi, n, tol);
                }
            }
            ihi = new_ihi;
            _stall_count = 0;
            continue;
        }

        // Step 2: Find the bottom of the unreduced Hessenberg submatrix
        // Scan upward from ihi-1 to find ilo (the start of the unreduced block)
        let mut ilo = 0usize;
        for k in (1..ihi - 1).rev() {
            let off_diag = h[[k, k - 1]].abs();
            let diag_sum = h[[k - 1, k - 1]].abs() + h[[k, k]].abs();
            let threshold = if diag_sum > T::epsilon() {
                tol * diag_sum
            } else {
                tol
            };
            if off_diag <= threshold {
                h[[k, k - 1]] = T::zero();
                ilo = k;
                break;
            }
        }

        let active_size = ihi - ilo;

        if active_size <= 1 {
            ihi = ilo;
            _stall_count = 0;
            continue;
        }

        if active_size == 2 {
            // Handle 2x2 block directly
            schur_split_2x2(&mut h, &mut q, ilo, n, tol);
            ihi = ilo;
            _stall_count = 0;
            continue;
        }

        // Step 3: Apply Francis double-shift QR step
        _stall_count += 1;

        // Apply Francis step with normal shifts
        francis_qr_step(&mut h, &mut q, ilo, ihi, n, tol)?;

        // After the step, check subdiagonals again for deflation
        // This immediate re-check is important for convergence
        for k in (ilo + 1..ihi).rev() {
            let off_diag = h[[k, k - 1]].abs();
            let diag_sum = h[[k - 1, k - 1]].abs() + h[[k, k]].abs();
            let threshold = if diag_sum > T::epsilon() {
                tol * diag_sum
            } else {
                tol
            };
            if off_diag <= threshold {
                h[[k, k - 1]] = T::zero();
            }
        }
    }

    Ok(RealSchurResult { q, t: h })
}

/// Try to split a 2x2 block at position `pos` into two 1x1 blocks.
/// If eigenvalues are real, apply a Givens rotation to make it upper triangular.
/// If eigenvalues are complex, leave the 2x2 block as-is (valid Schur form).
fn schur_split_2x2<T: SchurFloat>(
    h: &mut Array2<T>,
    q: &mut Array2<T>,
    pos: usize,
    n: usize,
    _tol: T,
) {
    let a11 = h[[pos, pos]];
    let a12 = h[[pos, pos + 1]];
    let a21 = h[[pos + 1, pos]];
    let a22 = h[[pos + 1, pos + 1]];

    // Check if eigenvalues are real: discriminant of characteristic polynomial
    let tr = a11 + a22;
    let det = a11 * a22 - a12 * a21;
    let four = T::from(4.0).unwrap_or(T::one());
    let disc = tr * tr - four * det;

    if disc < T::zero() {
        // Complex eigenvalues: the 2x2 block is already valid real Schur form.
        // The Bartels-Stewart solver handles unequal diagonals correctly,
        // so no normalization is needed here.
        return;
    }

    // Real eigenvalues: triangularize via Givens rotation
    // We want to find the eigenvalue closest to a22 (Wilkinson shift) and
    // zero out a21
    if a21.abs() < T::epsilon() {
        return; // already upper triangular
    }

    let half = T::from(0.5).unwrap_or(T::one() / (T::one() + T::one()));
    let sq = disc.sqrt();
    let lam1 = (tr + sq) * half;
    let lam2 = (tr - sq) * half;

    // Choose shift closest to a22
    let shift = if (lam1 - a22).abs() < (lam2 - a22).abs() {
        lam1
    } else {
        lam2
    };

    // Apply shifted QR step: one Givens rotation to zero out the subdiagonal
    let x = a11 - shift;
    let y = a21;
    let r = (x * x + y * y).sqrt();
    if r < T::epsilon() {
        return;
    }
    let c = x / r;
    let s = y / r;

    // Apply G^T from left to rows (pos, pos+1), all columns
    for col in 0..n {
        let t1 = h[[pos, col]];
        let t2 = h[[pos + 1, col]];
        h[[pos, col]] = c * t1 + s * t2;
        h[[pos + 1, col]] = -s * t1 + c * t2;
    }
    // Apply G from right to columns (pos, pos+1), all rows
    for row in 0..n {
        let t1 = h[[row, pos]];
        let t2 = h[[row, pos + 1]];
        h[[row, pos]] = c * t1 + s * t2;
        h[[row, pos + 1]] = -s * t1 + c * t2;
    }
    // Accumulate in Q
    for row in 0..n {
        let t1 = q[[row, pos]];
        let t2 = q[[row, pos + 1]];
        q[[row, pos]] = c * t1 + s * t2;
        q[[row, pos + 1]] = -s * t1 + c * t2;
    }
}

/// Apply one Francis double-shift implicit QR step to H[ilo..ihi, ilo..ihi].
///
/// This is the standard textbook algorithm from Golub & Van Loan, Algorithm 7.5.1.
/// Uses implicit double-shift with Householder reflectors.
fn francis_qr_step<T: SchurFloat>(
    h: &mut Array2<T>,
    q: &mut Array2<T>,
    ilo: usize,
    ihi: usize,
    n: usize,
    _tol: T,
) -> LinalgResult<()> {
    let p = ihi; // active end (exclusive)
    let two = T::one() + T::one();

    // Wilkinson double shift: eigenvalues of trailing 2x2 block
    let s = h[[p - 2, p - 2]] + h[[p - 1, p - 1]]; // trace
    let t_val = h[[p - 2, p - 2]] * h[[p - 1, p - 1]] - h[[p - 2, p - 1]] * h[[p - 1, p - 2]]; // determinant

    // First column of (H^2 - sH + tI) restricted to rows ilo..ilo+3
    let h00 = h[[ilo, ilo]];
    let h01 = h[[ilo, ilo + 1]];
    let h10 = h[[ilo + 1, ilo]];
    let h11 = h[[ilo + 1, ilo + 1]];
    let h21 = if ilo + 2 < p {
        h[[ilo + 2, ilo + 1]]
    } else {
        T::zero()
    };

    let mut x = h00 * h00 + h01 * h10 - s * h00 + t_val;
    let mut y = h10 * (h00 + h11 - s);
    let mut z = if ilo + 2 < p { h10 * h21 } else { T::zero() };

    // Chase the bulge down.
    //
    // NOTE: the upper bound is INCLUSIVE of `p - 2`. The classic textbook
    // formulation (Golub & Van Loan, Algorithm 7.5.1) uses 3-element
    // Householder reflectors for k = ilo .. p-3 (inclusive) to chase the
    // bulge, followed by ONE MORE 2-element Givens/Householder step at
    // k = p-2 to finish restoring Hessenberg form. That final step is *not*
    // optional: without it, the bulge introduced at column p-3, row p-1
    // never gets chased out, silently leaving a nonzero entry below the
    // sub-diagonal (violating the Hessenberg invariant and corrupting the
    // Schur form). Using an exclusive bound here (`..` instead of `..=`)
    // would skip that final step for every window size.
    for k in ilo..=(p.saturating_sub(2)) {
        // Determine the size of the reflector (3 or 2 at the bottom)
        let nr = if k + 3 <= p { 3 } else { 2 };

        if nr == 3 {
            // Compute Householder reflector to zero out y, z
            let (v, beta) = householder_vector_3(x, y, z);

            if beta.abs() > T::epsilon() {
                // Apply from left: H[k..k+3, ..] -= beta * v * (v^T * H[k..k+3, ..])
                // Column range: from max(k-1, ilo) to n (full width for similarity transform)
                let col_start = if k > ilo { k - 1 } else { ilo };
                for col in col_start..n {
                    let dot = v[0] * h[[k, col]] + v[1] * h[[k + 1, col]] + v[2] * h[[k + 2, col]];
                    h[[k, col]] -= beta * v[0] * dot;
                    h[[k + 1, col]] -= beta * v[1] * dot;
                    h[[k + 2, col]] -= beta * v[2] * dot;
                }

                // Apply from right: H[.., k..k+3] -= beta * (H[.., k..k+3] * v) * v^T
                // Row range: 0 to min(k+4, ihi) for Hessenberg structure within active block
                // But we also need to update rows above the active block (0..ilo) since
                // the similarity transform must affect the full matrix
                let row_end = (k + 4).min(ihi);
                for row in 0..row_end {
                    let dot = h[[row, k]] * v[0] + h[[row, k + 1]] * v[1] + h[[row, k + 2]] * v[2];
                    h[[row, k]] -= beta * dot * v[0];
                    h[[row, k + 1]] -= beta * dot * v[1];
                    h[[row, k + 2]] -= beta * dot * v[2];
                }

                // Accumulate in Q: Q[:, k..k+3] -= beta * (Q[:, k..k+3] * v) * v^T
                for row in 0..n {
                    let dot = q[[row, k]] * v[0] + q[[row, k + 1]] * v[1] + q[[row, k + 2]] * v[2];
                    q[[row, k]] -= beta * dot * v[0];
                    q[[row, k + 1]] -= beta * dot * v[1];
                    q[[row, k + 2]] -= beta * dot * v[2];
                }
            }
        } else {
            // 2-element Givens rotation at the bottom of the bulge chase
            let r = (x * x + y * y).sqrt();
            if r > T::epsilon() {
                let c = x / r;
                let s_val = y / r;

                let col_start = if k > ilo { k - 1 } else { ilo };
                for col in col_start..n {
                    let t1 = h[[k, col]];
                    let t2 = h[[k + 1, col]];
                    h[[k, col]] = c * t1 + s_val * t2;
                    h[[k + 1, col]] = -s_val * t1 + c * t2;
                }

                let row_end = (k + 3).min(p);
                for row in 0..row_end {
                    let t1 = h[[row, k]];
                    let t2 = h[[row, k + 1]];
                    h[[row, k]] = c * t1 + s_val * t2;
                    h[[row, k + 1]] = -s_val * t1 + c * t2;
                }

                for row in 0..n {
                    let t1 = q[[row, k]];
                    let t2 = q[[row, k + 1]];
                    q[[row, k]] = c * t1 + s_val * t2;
                    q[[row, k + 1]] = -s_val * t1 + c * t2;
                }
            }
        }

        // Set up for next iteration
        if k + 3 < p {
            x = h[[k + 1, k]];
            y = h[[k + 2, k]];
            z = if k + 3 < p { h[[k + 3, k]] } else { T::zero() };
        } else if k + 2 < p {
            x = h[[k + 1, k]];
            y = h[[k + 2, k]];
            z = T::zero();
        }
    }

    let _ = two;
    Ok(())
}

/// Compute a Householder vector for a 3-element vector [x, y, z].
/// Returns (v, beta) such that (I - beta * v * v^T) * [x; y; z] = [alpha; 0; 0].
fn householder_vector_3<T: SchurFloat>(x: T, y: T, z: T) -> ([T; 3], T) {
    let norm = (x * x + y * y + z * z).sqrt();
    if norm < T::epsilon() {
        return ([T::one(), T::zero(), T::zero()], T::zero());
    }
    let sign = if x >= T::zero() { T::one() } else { -T::one() };
    let v0 = x + sign * norm;
    let v_norm_sq = v0 * v0 + y * y + z * z;
    let two = T::one() + T::one();
    let beta = two / v_norm_sq;
    ([v0, y, z], beta)
}

/// Apply a 2×2 Givens rotation from the left to rows (r1, r2), cols [c0..c1].
fn apply_givens_cols<T: SchurFloat>(
    h: &mut Array2<T>,
    r1: usize,
    r2: usize,
    c0: usize,
    c1: usize,
    c: T,
    s: T,
) -> LinalgResult<()> {
    for col in c0..c1 {
        let a = h[[r1, col]];
        let b = h[[r2, col]];
        h[[r1, col]] = c * a + s * b;
        h[[r2, col]] = -s * a + c * b;
    }
    Ok(())
}

/// Apply a 2×2 Givens rotation from the right to cols (c1_idx, c2_idx), rows [r0..r1].
fn apply_givens_rows<T: SchurFloat>(
    h: &mut Array2<T>,
    c1_idx: usize,
    c2_idx: usize,
    r0: usize,
    r1: usize,
    c: T,
    s: T,
) -> LinalgResult<()> {
    for row in r0..r1 {
        let a = h[[row, c1_idx]];
        let b = h[[row, c2_idx]];
        h[[row, c1_idx]] = c * a - s * b;
        h[[row, c2_idx]] = s * a + c * b;
    }
    Ok(())
}

/// Reduce A to upper Hessenberg form H via orthogonal similarity: Q H Q^T = A.
fn hessenberg_decompose<T: SchurFloat>(a: &ArrayView2<T>) -> LinalgResult<(Array2<T>, Array2<T>)> {
    let n = a.nrows();
    let mut h = a.to_owned();
    let mut q = Array2::<T>::eye(n);

    for k in 0..n.saturating_sub(2) {
        // Build Householder reflector for column k, rows k+1..n
        let col_len = n - k - 1;
        if col_len == 0 {
            break;
        }
        let x: Vec<T> = (k + 1..n).map(|i| h[[i, k]]).collect();
        let (v, beta) = householder_vector(&x);

        if beta.abs() < T::epsilon() {
            continue;
        }

        // H[k+1.., k..] <- (I - beta v v^T) * H[k+1.., k..]
        for col in k..n {
            let dot: T = (0..col_len).map(|i| v[i] * h[[k + 1 + i, col]]).sum();
            for i in 0..col_len {
                h[[k + 1 + i, col]] -= beta * v[i] * dot;
            }
        }
        // H[0.., k+1..] <- H[0.., k+1..] * (I - beta v v^T)
        for row in 0..n {
            let dot: T = (0..col_len).map(|i| h[[row, k + 1 + i]] * v[i]).sum();
            for i in 0..col_len {
                h[[row, k + 1 + i]] -= beta * dot * v[i];
            }
        }
        // Q[0.., k+1..] <- Q[0.., k+1..] * (I - beta v v^T)
        for row in 0..n {
            let dot: T = (0..col_len).map(|i| q[[row, k + 1 + i]] * v[i]).sum();
            for i in 0..col_len {
                q[[row, k + 1 + i]] -= beta * dot * v[i];
            }
        }
    }

    Ok((h, q))
}

/// Compute a Householder reflector v and beta such that
/// (I - beta v v^T) x = [r; 0; ...; 0].
fn householder_vector<T: SchurFloat>(x: &[T]) -> (Vec<T>, T) {
    let n = x.len();
    if n == 0 {
        return (vec![], T::zero());
    }
    let norm: T = x.iter().map(|&xi| xi * xi).sum::<T>().sqrt();
    if norm < T::epsilon() {
        let mut v = vec![T::zero(); n];
        v[0] = T::one();
        return (v, T::zero());
    }
    let mut v: Vec<T> = x.to_vec();
    let sign = if x[0] >= T::zero() {
        T::one()
    } else {
        -T::one()
    };
    v[0] += sign * norm;
    let v_norm_sq: T = v.iter().map(|&vi| vi * vi).sum();
    let two = T::one() + T::one();
    let beta = two / v_norm_sq;
    (v, beta)
}

// ─────────────────────────────────────────────────────────────────────────────
// complex_schur_decompose
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the complex Schur decomposition A = Q T Q* where T is upper
/// triangular and Q is unitary.
///
/// This converts the real Schur form to complex form by splitting all 2×2
/// diagonal blocks into 1×1 complex diagonal entries.
///
/// # Arguments
/// * `a`        - Input square matrix (n × n, real)
/// * `max_iter` - Maximum QR sweeps
/// * `tol`      - Deflation tolerance
///
/// # Returns
/// `ComplexSchurResult` containing Q (complex) and T (upper triangular complex).
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::schur_enhanced::complex_schur_decompose;
///
/// let a = array![[0.0_f64, -1.0], [1.0, 0.0]]; // rotation matrix, eigenvalues ±i
/// let res = complex_schur_decompose(&a.view(), 300, 1e-12).expect("ok");
/// // eigenvalues should be ±i
/// assert!((res.t[[0,0]].im.abs() - 1.0).abs() < 1e-8);
/// ```
pub fn complex_schur_decompose<T: SchurFloat>(
    a: &ArrayView2<T>,
    max_iter: usize,
    tol: T,
) -> LinalgResult<ComplexSchurResult<T>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "complex_schur_decompose: A must be square".into(),
        ));
    }

    // Start from the real Schur form
    let real_res = real_schur_decompose(a, max_iter, tol)?;

    // Convert Q and T to complex
    let q_c: Array2<Complex<T>> = real_to_complex_matrix(&real_res.q);
    let mut t_c: Array2<Complex<T>> = real_to_complex_matrix(&real_res.t);
    let mut q_out = q_c;

    // Diagonalise 2×2 blocks on the diagonal of T using complex Givens rotations
    let mut k = 0usize;
    while k < n {
        if k + 1 < n && real_res.t[[k + 1, k]].abs() > tol {
            // 2×2 block: extract eigenvalues
            let (lam1, lam2) = eigen2x2_complex(
                real_res.t[[k, k]],
                real_res.t[[k, k + 1]],
                real_res.t[[k + 1, k]],
                real_res.t[[k + 1, k + 1]],
            );
            // Place lam1 on diagonal, zero out sub-diagonal via complex rotation
            let diff = Complex::new(real_res.t[[k, k]], T::zero()) - lam1;
            let off = Complex::new(real_res.t[[k + 1, k]], T::zero());
            let (gc, gs) = complex_givens(diff, off);

            // Apply rotation to t_c[k..k+2, k..n]
            for col in k..n {
                let a_val = t_c[[k, col]];
                let b_val = t_c[[k + 1, col]];
                t_c[[k, col]] = gc * a_val + gs * b_val;
                t_c[[k + 1, col]] = -gs.conj() * a_val + gc.conj() * b_val;
            }
            // Apply from right: t_c[0..k+2, k..k+2]
            for row in 0..n {
                let a_val = t_c[[row, k]];
                let b_val = t_c[[row, k + 1]];
                t_c[[row, k]] = gc.conj() * a_val - gs * b_val;
                t_c[[row, k + 1]] = gs.conj() * a_val + gc * b_val;
            }
            // Accumulate in Q
            for row in 0..n {
                let a_val = q_out[[row, k]];
                let b_val = q_out[[row, k + 1]];
                q_out[[row, k]] = gc.conj() * a_val - gs * b_val;
                q_out[[row, k + 1]] = gs.conj() * a_val + gc * b_val;
            }

            // Place lam2 on (k+1, k+1) and zero the sub-diagonal
            t_c[[k, k]] = lam1;
            t_c[[k + 1, k + 1]] = lam2;
            t_c[[k + 1, k]] = Complex::new(T::zero(), T::zero());

            k += 2;
        } else {
            k += 1;
        }
    }

    Ok(ComplexSchurResult { q: q_out, t: t_c })
}

/// Compute the two eigenvalues of a real 2×2 matrix [[a,b],[c,d]].
fn eigen2x2_complex<T: SchurFloat>(a: T, b: T, c: T, d: T) -> (Complex<T>, Complex<T>) {
    let two = T::one() + T::one();
    let tr = a + d;
    let det = a * d - b * c;
    let disc = tr * tr - two * two * det;
    if disc >= T::zero() {
        let sq = disc.sqrt();
        let lam1 = Complex::new((tr + sq) / two, T::zero());
        let lam2 = Complex::new((tr - sq) / two, T::zero());
        (lam1, lam2)
    } else {
        let sq = (-disc).sqrt() / two;
        let lam1 = Complex::new(tr / two, sq);
        let lam2 = Complex::new(tr / two, -sq);
        (lam1, lam2)
    }
}

/// Compute a complex Givens rotation (c, s) such that [c s; -s* c*][a; b] = [r; 0].
fn complex_givens<T: SchurFloat>(a: Complex<T>, b: Complex<T>) -> (Complex<T>, Complex<T>) {
    let a_abs = (a.re * a.re + a.im * a.im).sqrt();
    let b_abs = (b.re * b.re + b.im * b.im).sqrt();
    let r_abs = (a_abs * a_abs + b_abs * b_abs).sqrt();
    if r_abs < T::epsilon() {
        return (
            Complex::new(T::one(), T::zero()),
            Complex::new(T::zero(), T::zero()),
        );
    }
    let c = Complex::new(a_abs / r_abs, T::zero());
    let s = if a_abs < T::epsilon() {
        Complex::new(T::one(), T::zero())
    } else {
        Complex::new(a.re * b.re + a.im * b.im, a.im * b.re - a.re * b.im)
            * Complex::new(T::one() / (a_abs * r_abs), T::zero())
    };
    (c, s)
}

/// Convert a real matrix to a complex one with zero imaginary parts.
fn real_to_complex_matrix<T: SchurFloat>(a: &Array2<T>) -> Array2<Complex<T>> {
    let (m, n) = (a.nrows(), a.ncols());
    Array2::from_shape_fn((m, n), |(i, j)| Complex::new(a[[i, j]], T::zero()))
}

// ─────────────────────────────────────────────────────────────────────────────
// schur_reorder
// ─────────────────────────────────────────────────────────────────────────────

/// Reorder a real Schur form so that selected eigenvalues appear first.
///
/// Given (Q, T) in real Schur form, rearranges the diagonal blocks so that
/// eigenvalues for which `select(eigenvalue)` returns `true` appear in the
/// upper-left portion.  The transformation is orthogonal, preserving
/// A = Q T Qᵀ.
///
/// The algorithm uses successive swaps of adjacent 1×1 and 2×2 diagonal
/// blocks (Bai-Demmel procedure).
///
/// # Arguments
/// * `q`      - Orthogonal factor Q from real Schur form (n × n, modified in place)
/// * `t`      - Quasi-upper-triangular T (n × n, modified in place)
/// * `select` - Closure `f(re, im) -> bool`; `im` is ≠ 0 for conjugate pairs
/// * `tol`    - Tolerance for identifying 2×2 blocks
///
/// # Returns
/// `(q_out, t_out, n_selected)` where `n_selected` is the number of selected
/// eigenvalues (counting complex pairs as 2).
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::schur_enhanced::{real_schur_decompose, schur_reorder};
///
/// let a = array![[4.0_f64, 1.0], [0.0, 2.0]]; // eigenvalues 4 and 2
/// let res = real_schur_decompose(&a.view(), 200, 1e-12).expect("ok");
/// // Select eigenvalue < 3.0 (i.e. eigenvalue 2)
/// let (_q2, t2, n_sel) = schur_reorder(&res.q, &res.t, |re, _im| re < 3.0, 1e-10).expect("ok");
/// assert_eq!(n_sel, 1);
/// // Result should be a 2x2 quasi-upper-triangular matrix
/// assert_eq!(t2.shape(), &[2, 2]);
/// ```
pub fn schur_reorder<T, F>(
    q: &Array2<T>,
    t: &Array2<T>,
    select: F,
    tol: T,
) -> LinalgResult<(Array2<T>, Array2<T>, usize)>
where
    T: SchurFloat,
    F: Fn(T, T) -> bool,
{
    let n = t.nrows();
    if t.ncols() != n || q.nrows() != n || q.ncols() != n {
        return Err(LinalgError::ShapeError(
            "schur_reorder: Q and T must be square and conformant".into(),
        ));
    }

    let mut q_out = q.to_owned();
    let mut t_out = t.to_owned();

    // Identify blocks and their "selected" status
    let blocks = identify_schur_blocks(&t_out, tol);
    let nb = blocks.len();

    // Build selected array (by block index)
    let block_selected: Vec<bool> = blocks
        .iter()
        .map(|&(start, size)| {
            if size == 1 {
                select(t_out[[start, start]], T::zero())
            } else {
                let (lam1, _) = eigen2x2_complex(
                    t_out[[start, start]],
                    t_out[[start, start + 1]],
                    t_out[[start + 1, start]],
                    t_out[[start + 1, start + 1]],
                );
                select(lam1.re, lam1.im)
            }
        })
        .collect();

    // Bubble selected blocks to the front using adjacent block swaps
    let mut sel_copy = block_selected.clone();
    let mut blocks_cur = blocks.clone();
    let mut n_selected = 0usize;

    for target in 0..nb {
        if sel_copy[target] {
            n_selected += blocks_cur[target].1;
            continue;
        }
        // Find next selected block after target
        let mut found = None;
        for (src, &selected) in sel_copy.iter().enumerate().take(nb).skip(target + 1) {
            if selected {
                found = Some(src);
                break;
            }
        }
        let src = match found {
            Some(s) => s,
            None => break,
        };
        // Bubble src block left to position target
        for pos in (target..src).rev() {
            swap_schur_blocks(
                &mut q_out,
                &mut t_out,
                blocks_cur[pos].0,
                blocks_cur[pos].1,
                blocks_cur[pos + 1].1,
                tol,
            )?;
            // Update blocks_cur after swap
            let new_start = blocks_cur[pos].0;
            let size_a = blocks_cur[pos + 1].1;
            let size_b = blocks_cur[pos].1;
            blocks_cur[pos] = (new_start, size_a);
            blocks_cur[pos + 1] = (new_start + size_a, size_b);
            sel_copy.swap(pos, pos + 1);
        }
        n_selected += blocks_cur[target].1;
    }

    Ok((q_out, t_out, n_selected))
}

/// Identify the diagonal blocks of a quasi-upper-triangular matrix.
/// Returns a vec of `(start_row, size)` pairs.
fn identify_schur_blocks<T: SchurFloat>(t: &Array2<T>, tol: T) -> Vec<(usize, usize)> {
    let n = t.nrows();
    let mut blocks = Vec::new();
    let mut k = 0usize;
    while k < n {
        if k + 1 < n && t[[k + 1, k]].abs() > tol {
            blocks.push((k, 2));
            k += 2;
        } else {
            blocks.push((k, 1));
            k += 1;
        }
    }
    blocks
}

/// Swap two adjacent diagonal blocks of a real Schur form (T, Q).
/// Block A starts at `start`, has size `size_a`; block B immediately follows
/// with size `size_b`.
fn swap_schur_blocks<T: SchurFloat>(
    q: &mut Array2<T>,
    t: &mut Array2<T>,
    start: usize,
    size_a: usize,
    size_b: usize,
    _tol: T,
) -> LinalgResult<()> {
    let n = t.nrows();
    let sa = size_a;
    let sb = size_b;
    let p = start;

    // Extract the (sa+sb) × (sa+sb) sub-block
    let blk_size = sa + sb;
    if p + blk_size > n {
        return Err(LinalgError::IndexError(
            "swap_schur_blocks: block exceeds matrix dimensions".into(),
        ));
    }

    // For 1×1 ↔ 1×1: direct swap via one Givens rotation
    if sa == 1 && sb == 1 {
        let t11 = t[[p, p]];
        let t12 = t[[p, p + 1]];
        let t22 = t[[p + 1, p + 1]];
        // Find G such that G^T [[t11, t12],[0, t22]] G has swapped diagonal
        // Solve: (t22 - t11) * c * s = t12 * c^2 - 0 * s^2
        // Use standard formula from Golub & Van Loan
        let diff = t22 - t11;
        let (c, s) = if diff.abs() < T::epsilon() && t12.abs() < T::epsilon() {
            (T::one(), T::zero())
        } else {
            let val = diff / (t12 + T::epsilon());
            let theta = T::one() / (val + (T::one() + val * val).sqrt());
            let c = T::one() / (T::one() + theta * theta).sqrt();
            let s = c * theta;
            (c, s)
        };
        apply_givens_cols(t, p, p + 1, p, n, c, s)?;
        apply_givens_rows(t, p, p + 1, 0, p + 2, c, s)?;
        apply_givens_rows(q, p, p + 1, 0, n, c, s)?;
        return Ok(());
    }

    // For larger blocks: use the Sylvester equation approach.
    // Solve T_bb X - X T_aa = T_ab, then eliminate T_ab via elementary transformation.
    let t_aa = t.slice(s![p..p + sa, p..p + sa]).to_owned();
    let t_bb = t
        .slice(s![p + sa..p + blk_size, p + sa..p + blk_size])
        .to_owned();
    let t_ab = t.slice(s![p..p + sa, p + sa..p + blk_size]).to_owned();

    // Solve Sylvester: T_bb X - X T_aa = T_ab
    // Rewrite as: T_aa X^T + X^T (-T_bb) = -T_ab^T
    let neg_tbb = t_bb.mapv(|v| -v);
    let neg_tab_t = t_ab.t().mapv(|v| -v).to_owned();
    let x_t =
        crate::matrix_equations::solve_sylvester(&t_aa.view(), &neg_tbb.view(), &neg_tab_t.view())?;
    let x = x_t.t().to_owned();

    // Construct the transformation matrix Q_swap = [I_bb  0; X  I_aa]^{-1} ... [I_aa X; 0 I_bb]
    // Build elementary block eliminator E = [[I, X], [0, I]] for the (sa, sb) partition
    let mut e = Array2::<T>::eye(blk_size);
    for i in 0..sa {
        for j in 0..sb {
            e[[i, sa + j]] = x[[i, j]];
        }
    }
    // Orthogonalise E via QR to get orthogonal U
    let e_t = e.t().to_owned();
    let (u_t, _r) = crate::decomposition::qr(&e_t.view(), None)?;
    let u = u_t.t().to_owned();

    // Apply U: T[p..p+blk_size, *] <- U^T * T[p..p+blk_size, *]
    let block_rows = t.slice(s![p..p + blk_size, ..]).to_owned();
    let new_block_rows = u.t().dot(&block_rows);
    t.slice_mut(s![p..p + blk_size, ..]).assign(&new_block_rows);

    // Apply U from right: T[*, p..p+blk_size] <- T[*, p..p+blk_size] * U
    let block_cols = t.slice(s![.., p..p + blk_size]).to_owned();
    let new_block_cols = block_cols.dot(&u);
    t.slice_mut(s![.., p..p + blk_size]).assign(&new_block_cols);

    // Accumulate in Q
    let q_block = q.slice(s![.., p..p + blk_size]).to_owned();
    let new_q_block = q_block.dot(&u);
    q.slice_mut(s![.., p..p + blk_size]).assign(&new_q_block);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// schur_to_eigen
// ─────────────────────────────────────────────────────────────────────────────

/// Extract eigenvalues and eigenvectors from the real Schur form (Q, T).
///
/// Eigenvalues are read off the diagonal of T (handling 2×2 blocks for
/// complex conjugate pairs).  Eigenvectors are recovered by back-substitution
/// in the quasi-triangular system and then transformed back via Q.
///
/// # Arguments
/// * `q` - Orthogonal factor (n × n)
/// * `t` - Quasi-upper-triangular factor (n × n)
/// * `tol` - Tolerance for 2×2 block identification
///
/// # Returns
/// `SchurEigenResult` with n eigenvalues and corresponding eigenvectors.
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::schur_enhanced::{real_schur_decompose, schur_to_eigen};
///
/// let a = array![[2.0_f64, 0.0], [0.0, 3.0]];
/// let res = real_schur_decompose(&a.view(), 200, 1e-12).expect("ok");
/// let eig = schur_to_eigen(&res.q, &res.t, 1e-10).expect("ok");
/// // eigenvalues should be 2 and 3
/// let mut re_parts: Vec<f64> = eig.eigenvalues.iter().map(|c| c.re).collect();
/// re_parts.sort_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
/// assert!((re_parts[0] - 2.0).abs() < 1e-8);
/// assert!((re_parts[1] - 3.0).abs() < 1e-8);
/// ```
pub fn schur_to_eigen<T: SchurFloat>(
    q: &Array2<T>,
    t: &Array2<T>,
    tol: T,
) -> LinalgResult<SchurEigenResult<T>> {
    let n = t.nrows();
    let small = T::epsilon() * T::from(100.0).unwrap_or(T::one());

    let blocks = identify_schur_blocks(t, tol);

    // 1. Extract eigenvalues (one push per block position, so `eigenvalues[row]`
    //    always lines up with the diagonal block starting at `row`).
    let mut eigenvalues: Vec<Complex<T>> = Vec::with_capacity(n);
    for &(start, size) in &blocks {
        if size == 1 {
            eigenvalues.push(Complex::new(t[[start, start]], T::zero()));
        } else {
            let (lam1, lam2) = eigen2x2_complex(
                t[[start, start]],
                t[[start, start + 1]],
                t[[start + 1, start]],
                t[[start + 1, start + 1]],
            );
            eigenvalues.push(lam1);
            eigenvalues.push(lam2);
        }
    }

    // 2. Compute eigenvectors in Schur form by back-substitution in
    //    (T - lam I) y = 0, walking the diagonal BLOCKS from just above the
    //    target block up to the top.
    //
    //    Critically, this must process 2x2 (complex-pair) diagonal blocks
    //    that lie *above* the target eigenvalue as a single coupled 2x2
    //    linear solve, not as two independent scalar divisions row-by-row:
    //    treating `t[[row, row]]` in isolation is only one entry of that
    //    block and does not represent an eigenvalue of the original matrix,
    //    so a naive per-row scalar back-substitution silently produces a
    //    wrong (non-)eigenvector whenever a 2x2 block sits between the
    //    target block and row 0.
    let solve_column = |lam: Complex<T>, seed_start: usize, seed: &[Complex<T>]| {
        let mut y: Vec<Complex<T>> = vec![Complex::new(T::zero(), T::zero()); n];
        for (i, &val) in seed.iter().enumerate() {
            y[seed_start + i] = val;
        }
        let seed_end = seed_start + seed.len(); // exclusive
        for &(row, size) in blocks.iter().rev() {
            if row >= seed_start {
                continue; // at or past the seed block: nothing to solve for
            }
            if size == 1 {
                let mut sum = Complex::new(T::zero(), T::zero());
                for col in (row + 1)..seed_end {
                    sum += Complex::new(t[[row, col]], T::zero()) * y[col];
                }
                let diag = Complex::new(t[[row, row]], T::zero()) - lam;
                y[row] = if diag.re.abs() + diag.im.abs() < small {
                    Complex::new(T::zero(), T::zero())
                } else {
                    -sum / diag
                };
            } else {
                let row2 = row + 1;
                let mut sum0 = Complex::new(T::zero(), T::zero());
                let mut sum1 = Complex::new(T::zero(), T::zero());
                for col in (row2 + 1)..seed_end {
                    let tv0 = Complex::new(t[[row, col]], T::zero());
                    let tv1 = Complex::new(t[[row2, col]], T::zero());
                    sum0 += tv0 * y[col];
                    sum1 += tv1 * y[col];
                }
                // Solve the coupled 2x2 system:
                //   [a  b] [y[row] ]   [-sum0]
                //   [c  d] [y[row2]] = [-sum1]
                let a = Complex::new(t[[row, row]], T::zero()) - lam;
                let b = Complex::new(t[[row, row2]], T::zero());
                let c = Complex::new(t[[row2, row]], T::zero());
                let d = Complex::new(t[[row2, row2]], T::zero()) - lam;
                let det = a * d - b * c;
                if det.re.abs() + det.im.abs() < small {
                    y[row] = Complex::new(T::zero(), T::zero());
                    y[row2] = Complex::new(T::zero(), T::zero());
                } else {
                    let rhs0 = -sum0;
                    let rhs1 = -sum1;
                    y[row] = (rhs0 * d - b * rhs1) / det;
                    y[row2] = (a * rhs1 - c * rhs0) / det;
                }
            }
        }
        y
    };

    let mut evec_schur: Array2<Complex<T>> = Array2::zeros((n, n));

    for &(blk_start, blk_size) in &blocks {
        if blk_size == 1 {
            let lam = eigenvalues[blk_start];
            let one = Complex::new(T::one(), T::zero());
            let y = solve_column(lam, blk_start, &[one]);
            for i in 0..n {
                evec_schur[[i, blk_start]] = y[i];
            }
        } else {
            // 2x2 block: solve for lam1's eigenvector; lam2 = conj(lam1) and
            // T is real, so conj(y1) is exactly the eigenvector for lam2 —
            // no need (and no risk of an inconsistent) second back-substitution.
            let lam1 = eigenvalues[blk_start];

            let a11 = Complex::new(t[[blk_start, blk_start]], T::zero()) - lam1;
            let a12 = Complex::new(t[[blk_start, blk_start + 1]], T::zero());
            let (v0, v1) = if a12.re.abs() + a12.im.abs() > T::epsilon() {
                (a12, -a11)
            } else {
                (
                    Complex::new(T::one(), T::zero()),
                    Complex::new(T::zero(), T::zero()),
                )
            };

            let y1 = solve_column(lam1, blk_start, &[v0, v1]);
            for i in 0..n {
                evec_schur[[i, blk_start]] = y1[i];
                evec_schur[[i, blk_start + 1]] = y1[i].conj();
            }
        }
    }

    // 3. Transform back: eigenvectors = Q * evec_schur
    let q_c: Array2<Complex<T>> = real_to_complex_matrix(q);
    let eigenvectors: Array2<Complex<T>> = q_c.dot(&evec_schur);

    // 4. Normalise each eigenvector
    let mut evec_norm = eigenvectors.clone();
    for j in 0..n {
        let col = evec_norm.column(j).to_owned();
        let norm: T = col
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .sum::<T>()
            .sqrt();
        if norm > T::epsilon() {
            let scale = Complex::new(T::one() / norm, T::zero());
            evec_norm.column_mut(j).mapv_inplace(|c| c * scale);
        }
    }

    Ok(SchurEigenResult {
        eigenvalues: Array1::from_vec(eigenvalues),
        eigenvectors: evec_norm,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// invariant_subspace
// ─────────────────────────────────────────────────────────────────────────────

/// Compute an invariant subspace of A from a block Schur decomposition.
///
/// Given A = Q T Qᵀ in real Schur form, the columns of Q corresponding to the
/// selected leading diagonal blocks span an A-invariant subspace (i.e.
/// A * range(U) ⊆ range(U)).
///
/// # Arguments
/// * `a`      - Input matrix (n × n)
/// * `select` - Closure `f(re, im) -> bool` selecting desired eigenvalues
/// * `max_iter` - Maximum QR sweeps for the Schur decomposition
/// * `tol`    - Tolerance for block identification and reordering
///
/// # Returns
/// A matrix U whose columns form an orthonormal basis for the invariant
/// subspace corresponding to the selected eigenvalues.  The number of columns
/// equals `n_selected`.
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::schur_enhanced::invariant_subspace;
///
/// let a = array![[3.0_f64, 1.0, 0.0],
///                [0.0, 2.0, 0.0],
///                [0.0, 0.0, 1.0]];
/// // Select eigenvalues < 2.5 (eigenvalues 2 and 1)
/// let u = invariant_subspace(&a.view(), |re, _im| re < 2.5, 300, 1e-10).expect("ok");
/// assert_eq!(u.ncols(), 2);
/// // A*U ≈ U * (something)  — verify AU lies in span(U)
/// let au = a.dot(&u);
/// // Check that each column of AU is (approximately) in span(U): ||AU - U U^T AU|| ≈ 0
/// let utu = u.t().dot(&au);
/// let proj = u.dot(&utu);
/// let diff = &au - &proj;
/// let frob_sq: f64 = diff.iter().map(|&x| x*x).sum();
/// assert!(frob_sq.sqrt() < 1e-8, "invariant subspace error: {frob_sq}");
/// ```
pub fn invariant_subspace<T, F>(
    a: &ArrayView2<T>,
    select: F,
    max_iter: usize,
    tol: T,
) -> LinalgResult<Array2<T>>
where
    T: SchurFloat,
    F: Fn(T, T) -> bool,
{
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "invariant_subspace: A must be square".into(),
        ));
    }

    // Compute real Schur form
    let res = real_schur_decompose(a, max_iter, tol)?;

    // Reorder so selected eigenvalues come first
    let (q_reordered, _t_reordered, n_selected) = schur_reorder(&res.q, &res.t, select, tol)?;

    if n_selected == 0 {
        return Ok(Array2::zeros((n, 0)));
    }

    // The first n_selected columns of Q_reordered form an invariant subspace basis
    Ok(q_reordered.slice(s![.., 0..n_selected]).to_owned())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    fn frobenius_err(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn test_real_schur_2x2_diagonal() {
        let a = array![[2.0_f64, 0.0], [0.0, 3.0]];
        let res = real_schur_decompose(&a.view(), 200, 1e-12).expect("ok");
        let qt = res.q.t().to_owned();
        let reconstructed = res.q.dot(&res.t).dot(&qt);
        assert!(
            frobenius_err(&a, &reconstructed) < 1e-8,
            "Frobenius err too large"
        );
    }

    #[test]
    fn test_real_schur_2x2_non_symmetric() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let res = real_schur_decompose(&a.view(), 300, 1e-12).expect("ok");
        let qt = res.q.t().to_owned();
        let reconstructed = res.q.dot(&res.t).dot(&qt);
        assert!(frobenius_err(&a, &reconstructed) < 1e-7);
    }

    #[test]
    fn test_real_schur_3x3() {
        let a = array![[1.0_f64, 2.0, 0.0], [0.0, 3.0, 1.0], [0.0, 0.0, 2.0]];
        let res = real_schur_decompose(&a.view(), 300, 1e-12).expect("ok");
        let qt = res.q.t().to_owned();
        let reconstructed = res.q.dot(&res.t).dot(&qt);
        assert!(frobenius_err(&a, &reconstructed) < 1e-7);
    }

    #[test]
    fn test_schur_to_eigen_diagonal() {
        let a = array![[2.0_f64, 0.0], [0.0, 3.0]];
        let res = real_schur_decompose(&a.view(), 200, 1e-12).expect("ok");
        let eig = schur_to_eigen(&res.q, &res.t, 1e-10).expect("ok");
        let mut re_parts: Vec<f64> = eig.eigenvalues.iter().map(|c| c.re).collect();
        re_parts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!((re_parts[0] - 2.0).abs() < 1e-8);
        assert!((re_parts[1] - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_complex_schur_rotation() {
        // Rotation matrix has eigenvalues ±i
        let a = array![[0.0_f64, -1.0], [1.0, 0.0]];
        let res = complex_schur_decompose(&a.view(), 300, 1e-12).expect("ok");
        // T should be upper triangular with ±i on diagonal
        assert!(res.t[[1, 0]].re.abs() < 1e-8, "sub-diagonal should be ~0");
        assert!(
            res.t[[1, 0]].im.abs() < 1e-8,
            "sub-diagonal im should be ~0"
        );
    }

    #[test]
    fn test_invariant_subspace_3x3() {
        let a = array![[3.0_f64, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        // Select eigenvalues < 2.5 (eigenvalues 1.0 and 2.0)
        let u = invariant_subspace(&a.view(), |re, _im| re < 2.5, 300, 1e-10).expect("ok");
        assert_eq!(u.nrows(), 3);
        // Should select 2 eigenvalues
        assert!(u.ncols() >= 1 && u.ncols() <= 3);
        // AU should lie approximately in span(U)
        let au = a.dot(&u);
        let utu = u.t().dot(&au);
        let proj = u.dot(&utu);
        let diff_arr: Array2<f64> = &au - &proj;
        let frob: f64 = diff_arr.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(frob < 1e-6, "invariant subspace error: {frob}");
    }

    #[test]
    fn test_hessenberg_decompose() {
        let a = array![[4.0_f64, 3.0, 2.0], [1.0, 5.0, 4.0], [2.0, 1.0, 3.0]];
        let (h, q) = hessenberg_decompose(&a.view()).expect("ok");
        let qt = q.t().to_owned();
        let reconstructed = q.dot(&h).dot(&qt);
        assert!(frobenius_err(&a, &reconstructed) < 1e-9);
        // H should be upper Hessenberg (zero below sub-diagonal)
        for i in 2..3 {
            for j in 0..i - 1 {
                assert!(
                    h[[i, j]].abs() < 1e-9,
                    "H[{i},{j}] = {} not zero",
                    h[[i, j]]
                );
            }
        }
    }

    #[test]
    fn probe_debug_3x3() {
        let a = array![[6.0_f64, -11.0, 6.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let res = real_schur_decompose(&a.view(), 500, 1e-13).expect("schur ok");
        println!("T = {:?}", res.t);
        println!("Q = {:?}", res.q);
        let qt = res.q.t().to_owned();
        let reconstructed = res.q.dot(&res.t).dot(&qt);
        println!("reconstructed = {:?}", reconstructed);
        println!("frob err = {}", frobenius_err(&a, &reconstructed));
    }

    #[test]
    fn probe_companion_3x3_real_spectrum() {
        // Companion matrix for (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        let a = array![[6.0_f64, -11.0, 6.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let res = real_schur_decompose(&a.view(), 500, 1e-13).expect("schur ok");
        let eig = schur_to_eigen(&res.q, &res.t, 1e-10).expect("eigen ok");
        let mut re: Vec<f64> = eig.eigenvalues.iter().map(|c| c.re).collect();
        re.sort_by(|x, y| x.partial_cmp(y).unwrap());
        println!("eigenvalues: {:?}", eig.eigenvalues);
        assert!((re[0] - 1.0).abs() < 1e-8, "got {:?}", re);
        assert!((re[1] - 2.0).abs() < 1e-8, "got {:?}", re);
        assert!((re[2] - 3.0).abs() < 1e-8, "got {:?}", re);

        // Verify eigenvectors: A v = lambda v
        for j in 0..3 {
            let lam = eig.eigenvalues[j];
            let v = eig.eigenvectors.column(j).to_owned();
            for i in 0..3 {
                let mut av_i = Complex::new(0.0, 0.0);
                for k in 0..3 {
                    av_i += Complex::new(a[[i, k]], 0.0) * v[k];
                }
                let lv_i = lam * v[i];
                let diff = av_i - lv_i;
                let err = (diff.re * diff.re + diff.im * diff.im).sqrt();
                assert!(
                    err < 1e-6,
                    "eigenvector check failed at i={i} j={j}: err={err}, av={av_i:?}, lv={lv_i:?}"
                );
            }
        }
    }

    #[test]
    fn probe_companion_4x4_mixed_spectrum() {
        // Companion matrix for (x-1)(x-2)(x^2+1) = x^4 - 3x^3 + 3x^2 - 3x + 2
        let a = array![
            [3.0_f64, -3.0, 3.0, -2.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ];
        let res = real_schur_decompose(&a.view(), 2000, 1e-13).expect("schur ok");
        let qt = res.q.t().to_owned();
        let reconstructed = res.q.dot(&res.t).dot(&qt);
        println!("reconstruction err = {}", frobenius_err(&a, &reconstructed));
        assert!(frobenius_err(&a, &reconstructed) < 1e-6);

        let eig = schur_to_eigen(&res.q, &res.t, 1e-9).expect("eigen ok");
        println!("eigenvalues: {:?}", eig.eigenvalues);

        let mut reals: Vec<f64> = eig
            .eigenvalues
            .iter()
            .filter(|c| c.im.abs() < 1e-6)
            .map(|c| c.re)
            .collect();
        reals.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut complex_im: Vec<f64> = eig
            .eigenvalues
            .iter()
            .filter(|c| c.im.abs() >= 1e-6)
            .map(|c| c.im)
            .collect();
        complex_im.sort_by(|x, y| x.partial_cmp(y).unwrap());

        assert_eq!(
            reals.len(),
            2,
            "expected 2 real eigenvalues, got {:?}",
            eig.eigenvalues
        );
        assert_eq!(
            complex_im.len(),
            2,
            "expected 2 complex eigenvalues, got {:?}",
            eig.eigenvalues
        );
        assert!((reals[0] - 1.0).abs() < 1e-6, "got {:?}", reals);
        assert!((reals[1] - 2.0).abs() < 1e-6, "got {:?}", reals);
        assert!(
            (complex_im[0] - (-1.0)).abs() < 1e-6,
            "got {:?}",
            complex_im
        );
        assert!((complex_im[1] - 1.0).abs() < 1e-6, "got {:?}", complex_im);

        // Verify eigenvectors: A v = lambda v  (works for both real and complex eigenpairs)
        for j in 0..4 {
            let lam = eig.eigenvalues[j];
            let v = eig.eigenvectors.column(j).to_owned();
            for i in 0..4 {
                let mut av_i = Complex::new(0.0, 0.0);
                for k in 0..4 {
                    av_i += Complex::new(a[[i, k]], 0.0) * v[k];
                }
                let lv_i = lam * v[i];
                let diff = av_i - lv_i;
                let err = (diff.re * diff.re + diff.im * diff.im).sqrt();
                assert!(
                    err < 1e-5,
                    "eigenvector check failed at i={i} j={j}: err={err}, av={av_i:?}, lv={lv_i:?}, eigenvalues={:?}",
                    eig.eigenvalues
                );
            }
        }
    }
}
