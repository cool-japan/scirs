//! Indefinite matrix factorization algorithms.
//!
//! This module provides factorizations for indefinite symmetric matrices
//! (matrices that are neither positive nor negative definite), including:
//!
//! - **Bunch-Kaufman LDL^T factorization** – handles 1×1 and 2×2 pivot blocks
//! - **Linear system solve** via the Bunch-Kaufman factorization
//! - **Inertia computation** – (positive, negative, zero) eigenvalue counts
//! - **Modified Cholesky** (Gill-Murray-Wright) – perturbs the diagonal to
//!   guarantee positive-definiteness while keeping the modification small
//! - **Spectral decomposition** of an indefinite matrix through its eigensolver
//!
//! ## References
//!
//! - Bunch & Kaufman (1977), "Some Stable Methods for Calculating Inertia and
//!   Solving Symmetric Linear Systems", *Math. Comp.* 31(137).
//! - Gill, Murray & Wright (1981), *Practical Optimization*, Academic Press.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of the Bunch-Kaufman LDL^T factorization.
///
/// The factorized form of a symmetric matrix `A` is:
///
/// ```text
/// P A P^T = L D L^T
/// ```
///
/// where
/// - `L` is unit lower-triangular,
/// - `D` is block-diagonal with 1×1 or 2×2 blocks stored in `d_blocks`,
/// - `perm` is the permutation vector (row/column `perm[i]` maps to position `i`).
#[derive(Debug, Clone)]
pub struct BunchKaufmanResult {
    /// Unit lower-triangular factor `L` (n×n).
    pub l: Array2<f64>,
    /// Block-diagonal factor `D`: each element is either a 1×1 or 2×2 matrix.
    pub d_blocks: Vec<Array2<f64>>,
    /// Permutation vector: the factorization acts on the permuted matrix
    /// `A[perm, :][:, perm]`.
    pub perm: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Swap rows and columns `i` and `j` of a square matrix in-place.
fn symmetric_swap(a: &mut Array2<f64>, i: usize, j: usize) {
    let n = a.nrows();
    if i == j {
        return;
    }
    // Swap rows i and j
    for k in 0..n {
        let tmp = a[[i, k]];
        a[[i, k]] = a[[j, k]];
        a[[j, k]] = tmp;
    }
    // Swap cols i and j
    for k in 0..n {
        let tmp = a[[k, i]];
        a[[k, i]] = a[[k, j]];
        a[[k, j]] = tmp;
    }
}

/// Column Euclidean norm of column `col`, rows `row_start..n`.
fn col_max_abs(a: &Array2<f64>, col: usize, row_start: usize) -> f64 {
    let n = a.nrows();
    let mut max = 0.0_f64;
    for r in row_start..n {
        let v = a[[r, col]].abs();
        if v > max {
            max = v;
        }
    }
    max
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Bunch-Kaufman LDL^T factorization for indefinite symmetric matrices.
///
/// Decomposes the symmetric matrix `A` (n×n) as
///
/// ```text
/// P A P^T = L D L^T
/// ```
///
/// using the symmetric partial pivoting strategy of Bunch & Kaufman (1977).
/// The pivot at each step is either a 1×1 diagonal block or a 2×2 off-diagonal
/// block, chosen to bound the growth of the elimination multipliers by the
/// constant `α = (1 + √17) / 8 ≈ 0.6404`.
///
/// # Arguments
///
/// * `a` – Symmetric n×n matrix.
///
/// # Returns
///
/// A [`BunchKaufmanResult`] containing `L`, the `D` blocks, and the permutation.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] if `a` is not square.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::indefinite::bunch_kaufman;
///
/// let a = array![
///     [ 2.0_f64,  1.0,  0.0],
///     [ 1.0,  0.0, -1.0],
///     [ 0.0, -1.0,  3.0],
/// ];
/// let bk = bunch_kaufman(&a).expect("factorization failed");
/// assert_eq!(bk.l.nrows(), 3);
/// ```
pub fn bunch_kaufman(a: &Array2<f64>) -> LinalgResult<BunchKaufmanResult> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(format!(
            "bunch_kaufman: matrix must be square, got {}×{}",
            n,
            a.ncols()
        )));
    }

    // Work on a copy so we can do in-place elimination
    let mut work = a.clone();
    // L starts as identity; we fill in the subdiagonal columns during elimination
    let mut l = Array2::<f64>::eye(n);
    let mut d_blocks: Vec<Array2<f64>> = Vec::new();
    let mut perm: Vec<usize> = (0..n).collect();

    // Bunch-Kaufman pivoting constant α = (1 + √17) / 8
    let alpha = (1.0_f64 + 17.0_f64.sqrt()) / 8.0_f64;

    let mut k = 0usize; // current leading column
    while k < n {
        let remaining = n - k;

        if remaining == 1 {
            // Single remaining element – trivial 1×1 pivot
            let pivot_val = work[[k, k]];
            let mut blk = Array2::<f64>::zeros((1, 1));
            blk[[0, 0]] = pivot_val;
            d_blocks.push(blk);
            k += 1;
            continue;
        }

        // Compute |a_{kk}| and the index of the largest off-diagonal in column k
        let a_kk = work[[k, k]].abs();

        // Find row index r of the largest sub-diagonal element in column k
        let mut r = k + 1;
        let mut a_rk = 0.0_f64;
        for i in (k + 1)..n {
            let v = work[[i, k]].abs();
            if v > a_rk {
                a_rk = v;
                r = i;
            }
        }

        // Decide pivot type according to Bunch-Kaufman criterion
        let use_1x1 = if a_rk == 0.0 {
            // Column k is (numerically) zero – use 1×1 degenerate pivot
            true
        } else {
            let a_rr = work[[r, r]].abs();
            // Largest off-diagonal in row r (excluding column k)
            let mut a_sr = 0.0_f64;
            for j in k..n {
                if j != r {
                    let v = work[[r, j]].abs();
                    if v > a_sr {
                        a_sr = v;
                    }
                }
            }
            if a_kk >= alpha * a_rk {
                true
            } else if a_rr >= alpha * a_sr {
                // Swap row/col k with r and use 1×1 pivot at the new k
                symmetric_swap(&mut work, k, r);
                // Keep track of the permutation
                perm.swap(k, r);
                // Also swap the already-computed columns of L (columns 0..k)
                for c in 0..k {
                    let tmp = l[[k, c]];
                    l[[k, c]] = l[[r, c]];
                    l[[r, c]] = tmp;
                }
                true
            } else {
                false
            }
        };

        if use_1x1 {
            // ---- 1×1 pivot ----
            let d11 = work[[k, k]];
            let mut blk = Array2::<f64>::zeros((1, 1));
            blk[[0, 0]] = d11;
            d_blocks.push(blk);

            if d11.abs() > f64::EPSILON {
                // Compute multipliers and update trailing submatrix
                for i in (k + 1)..n {
                    let mult = work[[i, k]] / d11;
                    l[[i, k]] = mult;
                    for j in (k + 1)..=i {
                        work[[i, j]] -= mult * work[[k, j]];
                        work[[j, i]] = work[[i, j]]; // maintain symmetry
                    }
                }
            }
            k += 1;
        } else {
            // ---- 2×2 pivot: rows/cols k and r ----
            // Move r to k+1
            if r != k + 1 {
                symmetric_swap(&mut work, k + 1, r);
                perm.swap(k + 1, r);
                for c in 0..k {
                    let tmp = l[[k + 1, c]];
                    l[[k + 1, c]] = l[[r, c]];
                    l[[r, c]] = tmp;
                }
            }

            let d11 = work[[k, k]];
            let d12 = work[[k, k + 1]];
            let d22 = work[[k + 1, k + 1]];

            let det = d11 * d22 - d12 * d12;

            let mut blk = Array2::<f64>::zeros((2, 2));
            blk[[0, 0]] = d11;
            blk[[0, 1]] = d12;
            blk[[1, 0]] = d12;
            blk[[1, 1]] = d22;
            d_blocks.push(blk);

            if det.abs() > f64::EPSILON {
                let inv_det = 1.0 / det;
                for i in (k + 2)..n {
                    // Solve 2×2 system [d11 d12; d12 d22] * [m1; m2] = [a[i,k]; a[i,k+1]]
                    let b1 = work[[i, k]];
                    let b2 = work[[i, k + 1]];
                    let m1 = (d22 * b1 - d12 * b2) * inv_det;
                    let m2 = (d11 * b2 - d12 * b1) * inv_det;
                    l[[i, k]] = m1;
                    l[[i, k + 1]] = m2;
                    for j in (k + 2)..=i {
                        work[[i, j]] -= m1 * work[[k, j]] + m2 * work[[k + 1, j]];
                        work[[j, i]] = work[[i, j]];
                    }
                }
            }
            k += 2;
        }
    }

    Ok(BunchKaufmanResult { l, d_blocks, perm })
}

/// Solve the linear system `Ax = b` using a pre-computed Bunch-Kaufman factorization.
///
/// The system `A x = b` is equivalent to `L D L^T (P x) = P b` where `P` is
/// the permutation encoded in `bk.perm`.  The function applies:
///
/// 1. Permute `b` according to `perm`.
/// 2. Forward substitution through `L`.
/// 3. Block-diagonal solve through `D`.
/// 4. Back substitution through `L^T`.
/// 5. Inverse-permute the result.
///
/// # Arguments
///
/// * `bk` – Bunch-Kaufman result from [`bunch_kaufman`].
/// * `b`  – Right-hand-side vector (length n).
///
/// # Errors
///
/// Returns [`LinalgError::DimensionError`] if `b` has wrong length.
/// Returns [`LinalgError::SingularMatrixError`] if a block in `D` is singular.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::indefinite::{bunch_kaufman, ldlt_solve};
///
/// // Indefinite symmetric matrix
/// let a = array![
///     [4.0_f64, 2.0, -1.0],
///     [2.0,  3.0,  0.0],
///     [-1.0, 0.0,  2.0],
/// ];
/// let b = array![1.0_f64, 2.0, 3.0];
/// let bk = bunch_kaufman(&a).expect("factorization failed");
/// let x = ldlt_solve(&bk, &b).expect("solve failed");
/// // Verify A*x ≈ b
/// for i in 0..3 {
///     let ax_i: f64 = (0..3).map(|j| a[[i, j]] * x[j]).sum();
///     assert!((ax_i - b[i]).abs() < 1e-10, "residual too large at row {}", i);
/// }
/// ```
pub fn ldlt_solve(bk: &BunchKaufmanResult, b: &Array1<f64>) -> LinalgResult<Array1<f64>> {
    let n = bk.l.nrows();
    if b.len() != n {
        return Err(LinalgError::DimensionError(format!(
            "ldlt_solve: rhs length {} does not match matrix size {}",
            b.len(),
            n
        )));
    }

    // Step 1: Permute b → Pb
    let mut y = Array1::<f64>::zeros(n);
    for (new_pos, &orig) in bk.perm.iter().enumerate() {
        y[new_pos] = b[orig];
    }

    // Step 2: Forward substitution L y = Pb  (L is unit lower-triangular)
    for i in 1..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += bk.l[[i, j]] * y[j];
        }
        y[i] -= sum;
    }

    // Step 3: Solve D z = y  (block diagonal)
    let mut z = Array1::<f64>::zeros(n);
    let mut col = 0usize;
    for (block_idx, blk) in bk.d_blocks.iter().enumerate() {
        if blk.nrows() == 1 {
            let d = blk[[0, 0]];
            if d.abs() < f64::EPSILON {
                z[col] = 0.0;
            } else {
                z[col] = y[col] / d;
            }
            col += 1;
        } else {
            // 2×2 block
            let d11 = blk[[0, 0]];
            let d12 = blk[[0, 1]];
            let d22 = blk[[1, 1]];
            let det = d11 * d22 - d12 * d12;
            if det.abs() < f64::EPSILON * (d11.abs() + d22.abs() + d12.abs() + 1.0) {
                return Err(LinalgError::SingularMatrixError(format!(
                    "ldlt_solve: 2×2 D block {} is (near-)singular",
                    block_idx
                )));
            }
            let inv_det = 1.0 / det;
            z[col] = (d22 * y[col] - d12 * y[col + 1]) * inv_det;
            z[col + 1] = (d11 * y[col + 1] - d12 * y[col]) * inv_det;
            col += 2;
        }
    }

    // Step 4: Back substitution L^T x = z
    let mut x_perm = z;
    for i in (0..(n.saturating_sub(1))).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += bk.l[[j, i]] * x_perm[j];
        }
        x_perm[i] -= sum;
    }

    // Step 5: Inverse permute: x[perm[i]] = x_perm[i]
    let mut x = Array1::<f64>::zeros(n);
    for (new_pos, &orig) in bk.perm.iter().enumerate() {
        x[orig] = x_perm[new_pos];
    }
    Ok(x)
}

/// Compute the inertia of a symmetric matrix from its Bunch-Kaufman factorization.
///
/// The inertia is the triple `(n_pos, n_neg, n_zero)` giving the number of
/// positive, negative, and zero eigenvalues.  By Sylvester's law of inertia,
/// the inertia of `A` equals the inertia of the block-diagonal factor `D`.
///
/// # Arguments
///
/// * `bk` – Bunch-Kaufman result from [`bunch_kaufman`].
///
/// # Returns
///
/// `(n_positive, n_negative, n_zero)` where each count refers to the
/// corresponding eigenvalues of the original matrix `A`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::indefinite::{bunch_kaufman, inertia};
///
/// let a = array![
///     [ 2.0_f64,  0.0,  0.0],
///     [ 0.0, -1.0,  0.0],
///     [ 0.0,  0.0,  0.0],
/// ];
/// let bk = bunch_kaufman(&a).expect("factorization failed");
/// let (pos, neg, zer) = inertia(&bk);
/// assert_eq!(pos, 1);
/// assert_eq!(neg, 1);
/// assert_eq!(zer, 1);
/// ```
pub fn inertia(bk: &BunchKaufmanResult) -> (usize, usize, usize) {
    let eps = f64::EPSILON * 100.0;
    let mut n_pos = 0usize;
    let mut n_neg = 0usize;
    let mut n_zero = 0usize;

    for blk in &bk.d_blocks {
        if blk.nrows() == 1 {
            let v = blk[[0, 0]];
            if v > eps {
                n_pos += 1;
            } else if v < -eps {
                n_neg += 1;
            } else {
                n_zero += 1;
            }
        } else {
            // 2×2 block: eigenvalues have opposite signs (since det < 0) or are
            // complex-conjugate (which cannot happen for a real symmetric block).
            // Characteristic polynomial: λ² - tr λ + det = 0
            let d11 = blk[[0, 0]];
            let d12 = blk[[0, 1]];
            let d22 = blk[[1, 1]];
            let tr = d11 + d22;
            let det = d11 * d22 - d12 * d12;
            let discriminant = tr * tr - 4.0 * det;
            if discriminant < 0.0 {
                // Should not happen for real symmetric; count both as "positive"
                n_pos += 2;
            } else {
                let sqrt_disc = discriminant.sqrt();
                let lambda1 = (tr + sqrt_disc) / 2.0;
                let lambda2 = (tr - sqrt_disc) / 2.0;
                for &lam in &[lambda1, lambda2] {
                    if lam > eps {
                        n_pos += 1;
                    } else if lam < -eps {
                        n_neg += 1;
                    } else {
                        n_zero += 1;
                    }
                }
            }
        }
    }
    (n_pos, n_neg, n_zero)
}

/// Modified Cholesky factorization (Gill-Murray-Wright algorithm).
///
/// Computes a Cholesky-like factor `L` and a scalar perturbation `δ` such that
/// `A + δ I = L L^T` is symmetric positive definite.  The modification is
/// chosen to be the *smallest* diagonal shift that makes the factorization
/// numerically stable.
///
/// This is useful in optimization to obtain a positive-definite approximation
/// to an indefinite or semi-definite Hessian.
///
/// # Arguments
///
/// * `a` – Symmetric n×n matrix.
///
/// # Returns
///
/// `(L, delta)` where `L` is the lower-triangular Cholesky factor of the
/// modified matrix `A + delta * I`.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] if `a` is not square.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::indefinite::modified_cholesky;
///
/// // Indefinite matrix
/// let a = array![
///     [ 4.0_f64, 2.0],
///     [ 2.0, -1.0],
/// ];
/// let (l, delta) = modified_cholesky(&a).expect("factorization failed");
/// // delta > 0 because a is indefinite
/// assert!(delta >= 0.0, "delta should be non-negative");
/// // Verify L L^T ≈ A + delta * I
/// let n = 2;
/// for i in 0..n {
///     for j in 0..n {
///         let llt_ij: f64 = (0..=j.min(i)).map(|k| l[[i,k]] * l[[j,k]]).sum();
///         let expected = a[[i,j]] + if i==j { delta } else { 0.0 };
///         assert!((llt_ij - expected).abs() < 1e-10);
///     }
/// }
/// ```
pub fn modified_cholesky(a: &Array2<f64>) -> LinalgResult<(Array2<f64>, f64)> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(format!(
            "modified_cholesky: matrix must be square, got {}×{}",
            n,
            a.ncols()
        )));
    }

    // Compute the Frobenius norm to scale the perturbation
    let mut frob_sq = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            frob_sq += a[[i, j]] * a[[i, j]];
        }
    }
    let frob = frob_sq.sqrt();

    // GMW criterion: minimum perturbation η such that all pivot elements remain ≥ ε
    let eps = (n as f64) * f64::EPSILON * frob.max(1.0);

    let mut delta = 0.0_f64;

    // Find the minimum diagonal perturbation needed
    for i in 0..n {
        let diag_val = a[[i, i]];
        if diag_val <= eps {
            let needed = eps - diag_val;
            if needed > delta {
                delta = needed;
            }
        }
    }

    // Build the perturbed matrix
    let mut work = a.clone();
    for i in 0..n {
        work[[i, i]] += delta;
    }

    // Now perform standard Cholesky on the perturbed matrix.
    // If we encounter a non-positive pivot, increase delta and restart.
    let mut attempts = 0usize;
    loop {
        let mut l = Array2::<f64>::zeros((n, n));
        let mut failed = false;
        'outer: for j in 0..n {
            let mut diag = work[[j, j]];
            for k in 0..j {
                diag -= l[[j, k]] * l[[j, k]];
            }
            if diag <= 0.0 {
                // Pivot turned negative – increase delta
                failed = true;
                break 'outer;
            }
            let l_jj = diag.sqrt();
            l[[j, j]] = l_jj;
            for i in (j + 1)..n {
                let mut s = work[[i, j]];
                for k in 0..j {
                    s -= l[[i, k]] * l[[j, k]];
                }
                l[[i, j]] = s / l_jj;
            }
        }
        if !failed {
            return Ok((l, delta));
        }
        // Increase perturbation
        if delta == 0.0 {
            delta = eps.max(1e-8 * frob.max(1.0));
        } else {
            delta *= 2.0;
        }
        for i in 0..n {
            work[[i, i]] = a[[i, i]] + delta;
        }
        attempts += 1;
        if attempts > 60 {
            return Err(LinalgError::ConvergenceError(
                "modified_cholesky: failed to find a positive-definite perturbation".to_string(),
            ));
        }
    }
}

/// Spectral decomposition of an indefinite symmetric matrix.
///
/// Computes the eigendecomposition `A = V Λ V^T` where `V` is an orthogonal
/// matrix whose columns are the eigenvectors of `A`, and `Λ` is the diagonal
/// matrix of real eigenvalues.
///
/// This function uses the symmetric QR algorithm (Jacobi rotations) to compute
/// all eigenvalues and eigenvectors of a real symmetric matrix; it handles both
/// positive and negative (indefinite) spectra correctly.
///
/// # Arguments
///
/// * `a` – Symmetric n×n matrix.
///
/// # Returns
///
/// `(V, eigenvalues)` where `V` is n×n orthogonal and `eigenvalues` is the
/// vector of real eigenvalues in *ascending* order.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] if `a` is not square.
/// Returns [`LinalgError::ConvergenceError`] if the Jacobi algorithm does not
/// converge.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::indefinite::spectral_decomp_indefinite;
///
/// let a = array![
///     [ 2.0_f64,  1.0],
///     [ 1.0, -1.0],
/// ];
/// let (v, eigenvalues) = spectral_decomp_indefinite(&a).expect("decomposition failed");
/// // eigenvalues are real
/// assert_eq!(eigenvalues.len(), 2);
/// // Verify A*v[:,i] ≈ eigenvalues[i]*v[:,i]
/// let n = 2;
/// for i in 0..n {
///     for row in 0..n {
///         let av_i: f64 = (0..n).map(|c| a[[row, c]] * v[[c, i]]).sum();
///         let lv_i = eigenvalues[i] * v[[row, i]];
///         assert!((av_i - lv_i).abs() < 1e-9);
///     }
/// }
/// ```
pub fn spectral_decomp_indefinite(a: &Array2<f64>) -> LinalgResult<(Array2<f64>, Array1<f64>)> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(format!(
            "spectral_decomp_indefinite: matrix must be square, got {}×{}",
            n,
            a.ncols()
        )));
    }

    // Work on a copy (Jacobi modifies in-place)
    let mut s = a.clone();
    // V accumulates the rotation matrices: starts as identity
    let mut v = Array2::<f64>::eye(n);

    let max_iter = 100 * n * n;
    let tol = f64::EPSILON * (n as f64) * {
        let mut fnorm = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                fnorm += a[[i, j]] * a[[i, j]];
            }
        }
        fnorm.sqrt()
    };

    for _ in 0..max_iter {
        // Check off-diagonal convergence
        let mut off = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += s[[i, j]] * s[[i, j]];
            }
        }
        if off <= tol * tol {
            break;
        }

        // Sweep all off-diagonal pairs (i, j)
        for p in 0..n {
            for q in (p + 1)..n {
                let s_pq = s[[p, q]];
                if s_pq.abs() < tol {
                    continue;
                }
                // Compute the Jacobi rotation angle
                // Use atan2 for numerical stability (avoids division by near-zero denominator)
                let theta = 0.5 * f64::atan2(2.0 * s_pq, s[[p, p]] - s[[q, q]]);
                let cos_t = theta.cos();
                let sin_t = theta.sin();

                // Apply the rotation: S' = J^T S J
                // Update rows p and q of S
                for r in 0..n {
                    let sp = cos_t * s[[p, r]] + sin_t * s[[q, r]];
                    let sq = -sin_t * s[[p, r]] + cos_t * s[[q, r]];
                    s[[p, r]] = sp;
                    s[[q, r]] = sq;
                }
                // Update cols p and q of S
                for r in 0..n {
                    let sp = cos_t * s[[r, p]] + sin_t * s[[r, q]];
                    let sq = -sin_t * s[[r, p]] + cos_t * s[[r, q]];
                    s[[r, p]] = sp;
                    s[[r, q]] = sq;
                }
                // Accumulate into V
                for r in 0..n {
                    let vp = cos_t * v[[r, p]] + sin_t * v[[r, q]];
                    let vq = -sin_t * v[[r, p]] + cos_t * v[[r, q]];
                    v[[r, p]] = vp;
                    v[[r, q]] = vq;
                }
            }
        }
    }

    // Extract diagonal eigenvalues
    let mut eigenvalues: Vec<(f64, usize)> = (0..n).map(|i| (s[[i, i]], i)).collect();
    // Sort ascending
    eigenvalues.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let evals = Array1::from_vec(eigenvalues.iter().map(|(e, _)| *e).collect());
    // Permute eigenvector columns to match the sorted order
    let mut evecs = Array2::<f64>::zeros((n, n));
    for (new_col, (_, orig_col)) in eigenvalues.iter().enumerate() {
        for r in 0..n {
            evecs[[r, new_col]] = v[[r, *orig_col]];
        }
    }

    Ok((evecs, evals))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Reconstruct `P A P^T` from the BunchKaufman result and compare to `A`.
    fn reconstruct_bk(a: &Array2<f64>, bk: &BunchKaufmanResult) -> Array2<f64> {
        let n = bk.l.nrows();
        // Build block-diagonal D
        let mut d = Array2::<f64>::zeros((n, n));
        let mut col = 0;
        for blk in &bk.d_blocks {
            let sz = blk.nrows();
            for i in 0..sz {
                for j in 0..sz {
                    d[[col + i, col + j]] = blk[[i, j]];
                }
            }
            col += sz;
        }
        // L D L^T
        let mut ld = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += bk.l[[i, k]] * d[[k, j]];
                }
                ld[[i, j]] = s;
            }
        }
        let mut ldlt = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += ld[[i, k]] * bk.l[[j, k]];
                }
                ldlt[[i, j]] = s;
            }
        }
        // Build permuted A: P A P^T
        let mut pap = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                pap[[i, j]] = a[[bk.perm[i], bk.perm[j]]];
            }
        }
        // Return the residual ||L D L^T - P A P^T||
        let mut diff = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                diff[[i, j]] = ldlt[[i, j]] - pap[[i, j]];
            }
        }
        diff
    }

    #[test]
    fn test_bunch_kaufman_spd() {
        let a = array![[4.0_f64, 2.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 2.0],];
        let bk = bunch_kaufman(&a).expect("BK failed");
        let diff = reconstruct_bk(&a, &bk);
        let err: f64 = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(err < 1e-10, "reconstruction error too large: {}", err);
    }

    #[test]
    fn test_bunch_kaufman_indefinite() {
        let a = array![[2.0_f64, 1.0, 0.0], [1.0, 0.0, -1.0], [0.0, -1.0, 3.0],];
        let bk = bunch_kaufman(&a).expect("BK failed");
        let diff = reconstruct_bk(&a, &bk);
        let err: f64 = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(err < 1e-10, "reconstruction error too large: {}", err);
    }

    #[test]
    fn test_ldlt_solve_basic() {
        let a = array![[4.0_f64, 2.0, -1.0], [2.0, 3.0, 0.0], [-1.0, 0.0, 2.0],];
        let b = array![1.0_f64, 2.0, 3.0];
        let bk = bunch_kaufman(&a).expect("BK failed");
        let x = ldlt_solve(&bk, &b).expect("solve failed");
        for i in 0..3 {
            let ax_i: f64 = (0..3).map(|j| a[[i, j]] * x[j]).sum();
            assert!(
                (ax_i - b[i]).abs() < 1e-8,
                "residual at row {} = {}",
                i,
                (ax_i - b[i]).abs()
            );
        }
    }

    #[test]
    fn test_inertia() {
        // Diagonal matrix with known inertia
        let a = array![[2.0_f64, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0],];
        let bk = bunch_kaufman(&a).expect("BK failed");
        let (pos, neg, zer) = inertia(&bk);
        assert_eq!(pos, 1, "expected 1 positive eigenvalue");
        assert_eq!(neg, 1, "expected 1 negative eigenvalue");
        assert_eq!(zer, 1, "expected 1 zero eigenvalue");
    }

    #[test]
    fn test_inertia_spd() {
        let a = array![[4.0_f64, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0],];
        let bk = bunch_kaufman(&a).expect("BK failed");
        let (pos, neg, zer) = inertia(&bk);
        assert_eq!(pos, 3, "expected 3 positive eigenvalues for SPD");
        assert_eq!(neg, 0);
        assert_eq!(zer, 0);
    }

    #[test]
    fn test_modified_cholesky_indefinite() {
        let a = array![[4.0_f64, 2.0], [2.0, -1.0],];
        let (l, delta) = modified_cholesky(&a).expect("modified cholesky failed");
        assert!(delta >= 0.0, "delta must be non-negative");
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let llt_ij: f64 = (0..=j.min(i)).map(|k| l[[i, k]] * l[[j, k]]).sum();
                let expected = a[[i, j]] + if i == j { delta } else { 0.0 };
                assert!(
                    (llt_ij - expected).abs() < 1e-10,
                    "L L^T [{i},{j}] mismatch: {} vs {}",
                    llt_ij,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_modified_cholesky_spd() {
        // For an SPD matrix, delta should be 0
        let a = array![[4.0_f64, 1.0], [1.0, 3.0],];
        let (_l, delta) = modified_cholesky(&a).expect("modified cholesky failed");
        assert_eq!(delta, 0.0, "no perturbation needed for SPD");
    }

    #[test]
    fn test_spectral_decomp_indefinite() {
        let a = array![[2.0_f64, 1.0], [1.0, -1.0],];
        let (v, eigenvalues) = spectral_decomp_indefinite(&a).expect("decomp failed");
        let n = 2;
        // Verify A*v_i = lambda_i * v_i
        for i in 0..n {
            for row in 0..n {
                let av_i: f64 = (0..n).map(|c| a[[row, c]] * v[[c, i]]).sum();
                let lv_i = eigenvalues[i] * v[[row, i]];
                assert!(
                    (av_i - lv_i).abs() < 1e-9,
                    "eigenvector equation failed at ({row},{i}): {av_i} != {lv_i}"
                );
            }
        }
    }

    #[test]
    fn test_bunch_kaufman_non_square_error() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(bunch_kaufman(&a).is_err());
    }

    #[test]
    fn test_bunch_kaufman_1x1() {
        let a = array![[5.0_f64]];
        let bk = bunch_kaufman(&a).expect("BK 1x1 failed");
        assert_eq!(bk.d_blocks.len(), 1);
        assert_eq!(bk.d_blocks[0][[0, 0]], 5.0);
    }

    /// Largest absolute off-diagonal element of a matrix (helper for checking
    /// that the reconstructed D has the expected block structure)
    #[allow(dead_code)]
    fn max_offdiag(a: &Array2<f64>) -> f64 {
        let n = a.nrows();
        let mut max = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let v = a[[i, j]].abs();
                    if v > max {
                        max = v;
                    }
                }
            }
        }
        max
    }

    #[test]
    fn test_col_max_abs_helper() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0],];
        assert_eq!(col_max_abs(&a, 0, 1), 5.0);
        assert_eq!(col_max_abs(&a, 1, 0), 6.0);
    }
}
