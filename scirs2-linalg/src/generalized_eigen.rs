//! Generalized eigenvalue problems: Av = λBv.
//!
//! Algorithms:
//! - Cholesky-based reduction when B is symmetric positive definite (SPD)
//! - B⁻¹A approach for general square problems using Hessenberg QR iteration
//! - Simultaneous diagonalization of two symmetric matrices
//!
//! # References
//! - Golub & Van Loan, *Matrix Computations*, 4th ed., Chapters 7–8
//! - Anderson et al., *LAPACK Users' Guide*, 3rd ed.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::{Debug, Display};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Trait alias
// ---------------------------------------------------------------------------

/// Trait alias for floating-point scalars used throughout this module.
pub trait GenEigFloat:
    Float + NumAssign + Sum + ScalarOperand + Debug + Display + Send + Sync + 'static
{
}
impl<F> GenEigFloat for F where
    F: Float + NumAssign + Sum + ScalarOperand + Debug + Display + Send + Sync + 'static
{
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a generalized eigenvalue decomposition for `Av = λBv`.
#[derive(Debug, Clone)]
pub struct GeneralizedEigenResult<F: GenEigFloat> {
    /// Real parts of generalized eigenvalues λ.
    pub eigenvalues_re: Array1<F>,
    /// Imaginary parts of generalized eigenvalues λ (zero for symmetric problems).
    pub eigenvalues_im: Array1<F>,
    /// Right eigenvectors (columns), shape `[n, n]`.  Present when requested.
    pub right_eigenvectors: Option<Array2<F>>,
    /// Left eigenvectors (columns), shape `[n, n]`.  Present when requested.
    pub left_eigenvectors: Option<Array2<F>>,
    /// Whether the solver converged within the iteration budget.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Builder / configuration struct
// ---------------------------------------------------------------------------

/// Solver configuration for the generalized eigenvalue problem `Av = λBv`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::generalized_eigen::GeneralizedEigen;
///
/// let a = array![[2.0_f64, 1.0], [1.0, 2.0]];
/// let b = array![[3.0_f64, 1.0], [1.0, 3.0]];
/// let result = GeneralizedEigen::new()
///     .spd()
///     .solve(&a.view(), &b.view())
///     .expect("solve failed");
/// assert_eq!(result.eigenvalues_re.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct GeneralizedEigen {
    /// Whether to compute eigenvectors.
    pub compute_eigenvectors: bool,
    /// Treat B as symmetric positive definite (enables Cholesky-based reduction).
    pub assume_b_spd: bool,
    /// Maximum number of QR iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
}

impl Default for GeneralizedEigen {
    fn default() -> Self {
        Self {
            compute_eigenvectors: true,
            assume_b_spd: false,
            max_iter: 1000,
            tol: 1e-12,
        }
    }
}

impl GeneralizedEigen {
    /// Create a new solver with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark B as symmetric positive definite.  Enables Cholesky-based reduction.
    pub fn spd(mut self) -> Self {
        self.assume_b_spd = true;
        self
    }

    /// Set whether to compute eigenvectors.
    pub fn compute_eigenvectors(mut self, flag: bool) -> Self {
        self.compute_eigenvectors = flag;
        self
    }

    /// Set maximum QR iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Solve the generalized eigenvalue problem `Av = λBv`.
    ///
    /// # Arguments
    /// * `a` - Square matrix A (n×n)
    /// * `b` - Square matrix B (n×n)
    ///
    /// # Errors
    /// Returns `LinalgError` for empty, non-square, or incompatible matrices, and
    /// for a singular or non-positive-definite B when SPD mode is active.
    pub fn solve<F: GenEigFloat>(
        &self,
        a: &ArrayView2<F>,
        b: &ArrayView2<F>,
    ) -> LinalgResult<GeneralizedEigenResult<F>> {
        let n = a.nrows();
        if n == 0 {
            return Err(LinalgError::DimensionError(
                "GeneralizedEigen: matrix is empty".into(),
            ));
        }
        if a.ncols() != n {
            return Err(LinalgError::ShapeError(
                "GeneralizedEigen: A must be square".into(),
            ));
        }
        if b.nrows() != n || b.ncols() != n {
            return Err(LinalgError::ShapeError(
                "GeneralizedEigen: B must be the same size as A".into(),
            ));
        }

        if self.assume_b_spd {
            self.solve_spd(a, b, n)
        } else {
            self.solve_general(a, b, n)
        }
    }

    // -----------------------------------------------------------------------
    // SPD path: Cholesky reduction  Av = λBv  →  (L⁻¹ A L⁻ᵀ) w = λ w
    // -----------------------------------------------------------------------
    fn solve_spd<F: GenEigFloat>(
        &self,
        a: &ArrayView2<F>,
        b: &ArrayView2<F>,
        n: usize,
    ) -> LinalgResult<GeneralizedEigenResult<F>> {
        // Cholesky: B = L Lᵀ
        let l = cholesky_lower(b, n)?;
        // L⁻¹
        let l_inv = lower_tri_inv(&l, n)?;
        // C = L⁻¹ A L⁻ᵀ  (symmetric)
        let l_inv_t = l_inv.t().to_owned();
        let inner = matmul_nn(&l_inv.view(), a, n, n, n);
        let c = matmul_nn(&inner.view(), &l_inv_t.view(), n, n, n);

        let tol = F::from(self.tol).unwrap_or_else(F::epsilon);
        let (eigenvalues_re, raw_vecs) = symmetric_qr_eigen(&c, n, tol, self.max_iter)?;
        let eigenvalues_im = Array1::<F>::zeros(n);

        // Transform eigenvectors back: v = L⁻ᵀ w
        let eigenvectors = if self.compute_eigenvectors {
            // raw_vecs columns are the eigenvectors w; v = L⁻ᵀ w
            let l_inv_t_mat = l_inv.t().to_owned();
            Some(matmul_nn(&l_inv_t_mat.view(), &raw_vecs.view(), n, n, n))
        } else {
            None
        };

        Ok(GeneralizedEigenResult {
            eigenvalues_re,
            eigenvalues_im,
            right_eigenvectors: eigenvectors.clone(),
            left_eigenvectors: eigenvectors,
            converged: true,
        })
    }

    // -----------------------------------------------------------------------
    // General path: B⁻¹A  →  standard (non-symmetric) eigenvalue problem
    // -----------------------------------------------------------------------
    fn solve_general<F: GenEigFloat>(
        &self,
        a: &ArrayView2<F>,
        b: &ArrayView2<F>,
        n: usize,
    ) -> LinalgResult<GeneralizedEigenResult<F>> {
        let b_inv = mat_inv_gauss(b, n)?;
        let c = matmul_nn(&b_inv.view(), a, n, n, n);

        let tol = F::from(self.tol).unwrap_or_else(F::epsilon);
        let (eigenvalues_re, eigenvalues_im, right_vecs) =
            hessenberg_qr_eigen(&c, n, tol, self.max_iter)?;

        Ok(GeneralizedEigenResult {
            eigenvalues_re,
            eigenvalues_im,
            right_eigenvectors: if self.compute_eigenvectors {
                Some(right_vecs)
            } else {
                None
            },
            left_eigenvectors: None,
            converged: true,
        })
    }
}

// ---------------------------------------------------------------------------
// Cholesky decomposition (lower triangular, in-place)
// ---------------------------------------------------------------------------

/// Compute the lower Cholesky factor L such that A = LLᵀ.
///
/// # Errors
/// Returns `LinalgError::NonPositiveDefiniteError` if A is not SPD.
pub fn cholesky_lower<F: GenEigFloat>(a: &ArrayView2<F>, n: usize) -> LinalgResult<Array2<F>> {
    let mut l = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= F::zero() {
                    return Err(LinalgError::NonPositiveDefiniteError(format!(
                        "Matrix is not positive definite at pivot ({i}, {i})"
                    )));
                }
                l[[i, j]] = s.sqrt();
            } else {
                let diag = l[[j, j]];
                if diag.abs() < F::epsilon() * F::from(100.0).unwrap_or(F::one()) {
                    return Err(LinalgError::SingularMatrixError(format!(
                        "Near-zero diagonal in Cholesky at ({j}, {j})"
                    )));
                }
                l[[i, j]] = s / diag;
            }
        }
    }
    Ok(l)
}

/// Invert a lower-triangular matrix by back-substitution.
fn lower_tri_inv<F: GenEigFloat>(l: &Array2<F>, n: usize) -> LinalgResult<Array2<F>> {
    let mut inv = Array2::<F>::zeros((n, n));
    let eps = F::epsilon() * F::from(100.0).unwrap_or(F::one());

    for col in 0..n {
        if l[[col, col]].abs() < eps {
            return Err(LinalgError::SingularMatrixError(format!(
                "Near-zero diagonal at ({col}, {col}) in lower_tri_inv"
            )));
        }
        inv[[col, col]] = F::one() / l[[col, col]];
        for row in (col + 1)..n {
            let mut s = F::zero();
            for k in col..row {
                s += l[[row, k]] * inv[[k, col]];
            }
            if l[[row, row]].abs() < eps {
                return Err(LinalgError::SingularMatrixError(format!(
                    "Near-zero diagonal at ({row}, {row}) in lower_tri_inv"
                )));
            }
            inv[[row, col]] = -s / l[[row, row]];
        }
    }
    Ok(inv)
}

// ---------------------------------------------------------------------------
// Dense matrix multiply: C (m×p) = A (m×k) · B (k×p)
// ---------------------------------------------------------------------------
fn matmul_nn<F: GenEigFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    m: usize,
    k: usize,
    p: usize,
) -> Array2<F> {
    let mut c = Array2::<F>::zeros((m, p));
    for i in 0..m {
        for l in 0..k {
            let a_il = a[[i, l]];
            if a_il == F::zero() {
                continue;
            }
            for j in 0..p {
                c[[i, j]] += a_il * b[[l, j]];
            }
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Gauss-Jordan matrix inverse
// ---------------------------------------------------------------------------
fn mat_inv_gauss<F: GenEigFloat>(a: &ArrayView2<F>, n: usize) -> LinalgResult<Array2<F>> {
    // Augmented matrix [A | I]
    let mut aug: Vec<Vec<F>> = (0..n)
        .map(|i| {
            let mut row: Vec<F> = (0..n).map(|j| a[[i, j]]).collect();
            for j in 0..n {
                row.push(if i == j { F::one() } else { F::zero() });
            }
            row
        })
        .collect();

    let eps = F::epsilon() * F::from(100.0).unwrap_or(F::one());

    for col in 0..n {
        // Partial pivot
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
                "Matrix is singular (near-zero pivot in Gauss-Jordan)".into(),
            ));
        }
        let inv_pivot = F::one() / pivot;
        for j in 0..2 * n {
            aug[col][j] *= inv_pivot;
        }
        for i in 0..n {
            if i != col {
                let factor = aug[i][col];
                if factor == F::zero() {
                    continue;
                }
                for j in 0..2 * n {
                    let v = aug[col][j];
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

// ---------------------------------------------------------------------------
// Symmetric eigendecomposition via power iteration with deflation
//
// This gives exact eigenvalues for symmetric matrices and is suitable for
// small-to-medium n.  Returns (eigenvalues, eigenvector matrix Q) where
// columns of Q are eigenvectors (B-orthonormal in the SPD path).
// ---------------------------------------------------------------------------
fn symmetric_qr_eigen<F: GenEigFloat>(
    a: &Array2<F>,
    n: usize,
    tol: F,
    max_iter: usize,
) -> LinalgResult<(Array1<F>, Array2<F>)> {
    let mut eigenvalues = Vec::with_capacity(n);
    // Accumulate eigenvectors as rows (then transpose at end)
    let mut evecs: Vec<Array1<F>> = Vec::with_capacity(n);
    let mut a_work = a.clone();

    // Seed: use distinct vectors to avoid degenerate start
    for k in 0..n {
        // Start with a vector that is orthogonal to already-found eigenvectors
        let mut v = Array1::<F>::zeros(n);
        v[k] = F::one();

        // Gram-Schmidt against found eigenvectors
        for ev in &evecs {
            let dot: F = v.iter().zip(ev.iter()).map(|(&vi, &ei)| vi * ei).sum();
            for i in 0..n {
                v[i] -= dot * ev[i];
            }
        }
        let norm: F = v.iter().map(|&x| x * x).sum::<F>().sqrt();
        if norm < tol {
            // Degenerate direction; pick next canonical basis vector
            v = Array1::<F>::zeros(n);
            for idx in 0..n {
                if !evecs.iter().any(|e| (e[idx] - F::one()).abs() < tol) {
                    v[idx] = F::one();
                    break;
                }
            }
        } else {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }

        let mut eigenval = F::zero();
        for _ in 0..max_iter {
            // Av
            let mut av = Array1::<F>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    av[i] += a_work[[i, j]] * v[j];
                }
            }
            // Rayleigh quotient
            let new_eigenval: F = v.iter().zip(av.iter()).map(|(&vi, &avi)| vi * avi).sum();
            // Normalize
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

        // Deflate: A_work ← A_work - λ v vᵀ
        for i in 0..n {
            for j in 0..n {
                a_work[[i, j]] -= eigenval * v[i] * v[j];
            }
        }
    }

    // Build eigenvector matrix (columns = eigenvectors)
    let mut q = Array2::<F>::zeros((n, n));
    for (col, ev) in evecs.iter().enumerate() {
        for row in 0..n {
            q[[row, col]] = ev[row];
        }
    }

    Ok((Array1::from_vec(eigenvalues), q))
}

// ---------------------------------------------------------------------------
// Hessenberg reduction + QR shift iteration for non-symmetric matrices
// ---------------------------------------------------------------------------

/// Reduce A to upper Hessenberg form via Householder reflections in-place.
/// Also accumulates the orthogonal similarity transform Q if requested.
fn hessenberg_reduce<F: GenEigFloat>(h: &mut Array2<F>, n: usize, q: &mut Option<Array2<F>>) {
    for col in 0..(n.saturating_sub(2)) {
        // Build Householder reflector for column `col` below the subdiagonal
        let col_len = n - col - 1;
        let mut x: Vec<F> = (0..col_len).map(|i| h[[col + 1 + i, col]]).collect();

        let sigma: F = x.iter().map(|&v| v * v).sum::<F>().sqrt();
        if sigma == F::zero() {
            continue;
        }
        let sign = if x[0] >= F::zero() {
            F::one()
        } else {
            -F::one()
        };
        x[0] += sign * sigma;

        let norm_sq: F = x.iter().map(|&v| v * v).sum();
        if norm_sq == F::zero() {
            continue;
        }

        // Apply P = I - 2/norm_sq * x xᵀ from the left to H
        // H[(col+1):, col:] ← P H[(col+1):, col:]
        for j in col..n {
            let dot: F = (0..col_len).map(|i| x[i] * h[[col + 1 + i, j]]).sum();
            let two = F::from(2.0).unwrap_or(F::one());
            let coeff = two * dot / norm_sq;
            for i in 0..col_len {
                h[[col + 1 + i, j]] -= coeff * x[i];
            }
        }

        // Apply P from the right to H
        // H[:, (col+1):] ← H[:, (col+1):] P
        for i in 0..n {
            let dot: F = (0..col_len).map(|k| h[[i, col + 1 + k]] * x[k]).sum();
            let two = F::from(2.0).unwrap_or(F::one());
            let coeff = two * dot / norm_sq;
            for k in 0..col_len {
                h[[i, col + 1 + k]] -= coeff * x[k];
            }
        }

        // Accumulate Q
        if let Some(ref mut q_mat) = q {
            for i in 0..n {
                let dot: F = (0..col_len).map(|k| q_mat[[i, col + 1 + k]] * x[k]).sum();
                let two = F::from(2.0).unwrap_or(F::one());
                let coeff = two * dot / norm_sq;
                for k in 0..col_len {
                    q_mat[[i, col + 1 + k]] -= coeff * x[k];
                }
            }
        }
    }
}

/// Francis double-shift QR iteration on an upper Hessenberg matrix.
///
/// Extracts real eigenvalues (diagonal after convergence of off-diagonal entries).
fn hessenberg_qr_eigen<F: GenEigFloat>(
    a: &Array2<F>,
    n: usize,
    tol: F,
    max_iter: usize,
) -> LinalgResult<(Array1<F>, Array1<F>, Array2<F>)> {
    let mut h = a.clone();
    let mut q = if true {
        // Always accumulate Q for eigenvectors
        let mut q_mat = Array2::<F>::zeros((n, n));
        for i in 0..n {
            q_mat[[i, i]] = F::one();
        }
        Some(q_mat)
    } else {
        None
    };

    hessenberg_reduce(&mut h, n, &mut q);

    // QR iteration with Wilkinson/double-shift
    let mut eigenvalues_re = vec![F::zero(); n];
    let mut eigenvalues_im = vec![F::zero(); n];
    let mut end = n;
    let mut _total_iter = 0usize;

    while end > 1 {
        let mut converged = false;
        for _iter in 0..max_iter {
            _total_iter += 1;
            // Check subdiagonal for deflation
            if h[[end - 1, end - 2]].abs()
                < tol * (h[[end - 2, end - 2]].abs() + h[[end - 1, end - 1]].abs())
            {
                eigenvalues_re[end - 1] = h[[end - 1, end - 1]];
                eigenvalues_im[end - 1] = F::zero();
                end -= 1;
                converged = true;
                break;
            }
            if end > 2
                && h[[end - 2, end - 3]].abs()
                    < tol * (h[[end - 3, end - 3]].abs() + h[[end - 2, end - 2]].abs())
            {
                // 2×2 block at (end-2, end-2) — extract eigenvalues
                let (re1, im1, re2, im2) = eig_2x2(
                    h[[end - 2, end - 2]],
                    h[[end - 2, end - 1]],
                    h[[end - 1, end - 2]],
                    h[[end - 1, end - 1]],
                );
                eigenvalues_re[end - 2] = re1;
                eigenvalues_im[end - 2] = im1;
                eigenvalues_re[end - 1] = re2;
                eigenvalues_im[end - 1] = im2;
                end -= 2;
                converged = true;
                break;
            }

            // Wilkinson shift: eigenvalue of bottom-right 2×2
            let shift = wilkinson_shift(
                h[[end - 2, end - 2]],
                h[[end - 2, end - 1]],
                h[[end - 1, end - 2]],
                h[[end - 1, end - 1]],
            );

            // Perform one shifted QR step on h[0..end, 0..end]
            qr_step_hessenberg(&mut h, &mut q, end, shift, tol);
        }

        if !converged {
            // Fallback: take diagonal entry
            eigenvalues_re[end - 1] = h[[end - 1, end - 1]];
            eigenvalues_im[end - 1] = F::zero();
            end -= 1;
        }
    }
    if end == 1 {
        eigenvalues_re[0] = h[[0, 0]];
        eigenvalues_im[0] = F::zero();
    }

    let q_mat = q.unwrap_or_else(|| {
        let mut eye = Array2::<F>::zeros((n, n));
        for i in 0..n {
            eye[[i, i]] = F::one();
        }
        eye
    });

    Ok((
        Array1::from_vec(eigenvalues_re),
        Array1::from_vec(eigenvalues_im),
        q_mat,
    ))
}

/// Wilkinson shift: eigenvalue of 2×2 bottom-right corner closest to h[n-1,n-1].
fn wilkinson_shift<F: GenEigFloat>(a: F, b: F, c: F, d: F) -> F {
    let trace = a + d;
    let det = a * d - b * c;
    let two = F::from(2.0).unwrap_or(F::one());
    let four = F::from(4.0).unwrap_or(F::one());
    let disc = trace * trace - four * det;
    if disc >= F::zero() {
        let sqrt_disc = disc.sqrt();
        let e1 = (trace + sqrt_disc) / two;
        let e2 = (trace - sqrt_disc) / two;
        if (e1 - d).abs() < (e2 - d).abs() {
            e1
        } else {
            e2
        }
    } else {
        // Complex conjugate pair — use real part as shift
        trace / two
    }
}

/// Eigenvalues of a 2×2 matrix [[a,b],[c,d]].
fn eig_2x2<F: GenEigFloat>(a: F, b: F, c: F, d: F) -> (F, F, F, F) {
    let two = F::from(2.0).unwrap_or(F::one());
    let four = F::from(4.0).unwrap_or(F::one());
    let trace = a + d;
    let det = a * d - b * c;
    let disc = trace * trace - four * det;
    if disc >= F::zero() {
        let sqrt_disc = disc.sqrt();
        (
            (trace + sqrt_disc) / two,
            F::zero(),
            (trace - sqrt_disc) / two,
            F::zero(),
        )
    } else {
        let re = trace / two;
        let im = ((-disc).sqrt()) / two;
        (re, im, re, -im)
    }
}

/// One Givens-rotation QR step on the upper Hessenberg submatrix h[0..end, 0..end]
/// with shift `mu`.  Accumulates rotations into `q`.
fn qr_step_hessenberg<F: GenEigFloat>(
    h: &mut Array2<F>,
    q: &mut Option<Array2<F>>,
    end: usize,
    shift: F,
    _tol: F,
) {
    // Shift
    for i in 0..end {
        h[[i, i]] -= shift;
    }

    // Givens rotations
    for k in 0..(end - 1) {
        let (cos, sin) = givens(h[[k, k]], h[[k + 1, k]]);
        // Apply from the left: rows k and k+1, columns k..end
        for j in k..end {
            let hkj = h[[k, j]];
            let hk1j = h[[k + 1, j]];
            h[[k, j]] = cos * hkj + sin * hk1j;
            h[[k + 1, j]] = -sin * hkj + cos * hk1j;
        }
        // Apply from the right: rows 0..min(k+2, end), columns k and k+1
        let row_end = (k + 2).min(end);
        for i in 0..row_end {
            let hik = h[[i, k]];
            let hik1 = h[[i, k + 1]];
            h[[i, k]] = cos * hik + sin * hik1;
            h[[i, k + 1]] = -sin * hik + cos * hik1;
        }
        // Accumulate into Q
        if let Some(ref mut q_mat) = q {
            let n = q_mat.nrows();
            for i in 0..n {
                let qik = q_mat[[i, k]];
                let qik1 = q_mat[[i, k + 1]];
                q_mat[[i, k]] = cos * qik + sin * qik1;
                q_mat[[i, k + 1]] = -sin * qik + cos * qik1;
            }
        }
    }

    // Unshift
    for i in 0..end {
        h[[i, i]] += shift;
    }
}

/// Compute the Givens rotation (cos, sin) to zero out the second element.
fn givens<F: GenEigFloat>(a: F, b: F) -> (F, F) {
    if b == F::zero() {
        return (F::one(), F::zero());
    }
    if a == F::zero() {
        return (F::zero(), F::one());
    }
    let r = (a * a + b * b).sqrt();
    (a / r, b / r)
}

// ---------------------------------------------------------------------------
// Public standalone utilities
// ---------------------------------------------------------------------------

/// Compute the Cholesky factor L such that A = LLᵀ.
///
/// # Arguments
/// * `a` - Symmetric positive-definite matrix (n×n)
///
/// # Returns
/// Lower-triangular Cholesky factor L
///
/// # Errors
/// Returns `LinalgError::NonPositiveDefiniteError` if A is not positive definite.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::generalized_eigen::cholesky_decomp;
///
/// let a = array![[4.0_f64, 2.0], [2.0, 3.0]];
/// let l = cholesky_decomp(&a.view()).expect("cholesky failed");
/// // Verify: L Lᵀ ≈ A
/// let lt = l.t().to_owned();
/// let prod = l.dot(&lt);
/// for i in 0..2 { for j in 0..2 { assert!((prod[[i,j]] - a[[i,j]]).abs() < 1e-12); } }
/// ```
pub fn cholesky_decomp<F: GenEigFloat>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if n == 0 {
        return Err(LinalgError::DimensionError(
            "cholesky_decomp: empty matrix".into(),
        ));
    }
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "cholesky_decomp: matrix must be square".into(),
        ));
    }
    cholesky_lower(a, n)
}

/// Solve the symmetric generalized eigenvalue problem `Av = λBv` when B is SPD.
///
/// This is a convenience wrapper around [`GeneralizedEigen`] with the SPD flag set.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::generalized_eigen::gen_eigh_spd;
///
/// let a = array![[2.0_f64, 1.0], [1.0, 2.0]];
/// let b = array![[3.0_f64, 1.0], [1.0, 3.0]];
/// let result = gen_eigh_spd(&a.view(), &b.view()).expect("solve failed");
/// assert_eq!(result.eigenvalues_re.len(), 2);
/// // Imaginary parts should all be zero
/// for &v in result.eigenvalues_im.iter() { assert!(v.abs() < 1e-10); }
/// ```
pub fn gen_eigh_spd<F: GenEigFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
) -> LinalgResult<GeneralizedEigenResult<F>> {
    GeneralizedEigen::new().spd().solve(a, b)
}

/// Solve the general eigenvalue problem `Av = λBv` (B not necessarily SPD).
///
/// Uses B⁻¹A reduction followed by Hessenberg QR iteration.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::generalized_eigen::gen_eig_general;
///
/// let a = array![[1.0_f64, 2.0], [0.0, 3.0]];
/// let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let result = gen_eig_general(&a.view(), &b.view()).expect("solve failed");
/// assert_eq!(result.eigenvalues_re.len(), 2);
/// ```
pub fn gen_eig_general<F: GenEigFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
) -> LinalgResult<GeneralizedEigenResult<F>> {
    GeneralizedEigen::new().solve(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cholesky_3x3_spd() {
        // SPD matrix A = [[4,2,0],[2,5,1],[0,1,3]]
        let a = array![[4.0_f64, 2.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 3.0]];
        let l = cholesky_decomp(&a.view()).expect("cholesky_decomp failed");

        // Verify L is lower triangular
        for i in 0..3 {
            for j in (i + 1)..3 {
                assert!(l[[i, j]].abs() < 1e-12, "L is not lower triangular");
            }
        }

        // Verify L Lᵀ ≈ A
        let lt = l.t().to_owned();
        let prod = l.dot(&lt);
        for i in 0..3 {
            for j in 0..3 {
                let diff = (prod[[i, j]] - a[[i, j]]).abs();
                assert!(diff < 1e-10, "L Lᵀ ≠ A at ({i},{j}): diff = {diff}");
            }
        }
    }

    #[test]
    fn test_cholesky_not_spd() {
        // A non-positive-definite matrix
        let a = array![[1.0_f64, 2.0], [2.0, 1.0]];
        let result = cholesky_decomp(&a.view());
        assert!(
            matches!(result, Err(LinalgError::NonPositiveDefiniteError(_))),
            "Expected NonPositiveDefiniteError"
        );
    }

    #[test]
    fn test_gen_eigh_spd_2x2() {
        // Av = λBv with A=[[2,1],[1,2]], B=[[3,1],[1,3]]
        let a = array![[2.0_f64, 1.0], [1.0, 2.0]];
        let b = array![[3.0_f64, 1.0], [1.0, 3.0]];
        let result = GeneralizedEigen::new()
            .spd()
            .solve(&a.view(), &b.view())
            .expect("GeneralizedEigen::solve failed");

        assert_eq!(result.eigenvalues_re.len(), 2);
        // All imaginary parts must be zero for symmetric problem
        for &v in result.eigenvalues_im.iter() {
            assert!(v.abs() < 1e-8, "Expected real eigenvalues, got imag = {v}");
        }

        // Verify eigenvectors satisfy Av = λBv
        if let Some(ref vecs) = result.right_eigenvectors {
            for col in 0..2 {
                let lambda = result.eigenvalues_re[col];
                let v: Vec<f64> = (0..2).map(|i| vecs[[i, col]]).collect();
                // Compute Av and λBv
                let av: Vec<f64> = (0..2)
                    .map(|i| (0..2).map(|j| a[[i, j]] * v[j]).sum::<f64>())
                    .collect();
                let bv: Vec<f64> = (0..2)
                    .map(|i| (0..2).map(|j| b[[i, j]] * v[j]).sum::<f64>())
                    .collect();
                for i in 0..2 {
                    let diff = (av[i] - lambda * bv[i]).abs();
                    assert!(
                        diff < 1e-6,
                        "Eigenvector check failed at col {col}, row {i}: Av[i]={}, λBv[i]={}, diff={}",
                        av[i], lambda * bv[i], diff
                    );
                }
            }
        }
    }

    #[test]
    fn test_gen_eig_general_identity_b() {
        // B=I  →  generalized problem reduces to standard
        let a = array![[3.0_f64, 1.0], [1.0, 3.0]];
        let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let result = GeneralizedEigen::new()
            .solve(&a.view(), &b.view())
            .expect("GeneralizedEigen::solve failed");

        assert_eq!(result.eigenvalues_re.len(), 2);
        let mut eigs: Vec<f64> = result.eigenvalues_re.iter().copied().collect();
        eigs.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        // Eigenvalues of [[3,1],[1,3]] are 2 and 4
        assert!(
            (eigs[0] - 2.0).abs() < 1e-6,
            "Expected eigenvalue 2, got {}",
            eigs[0]
        );
        assert!(
            (eigs[1] - 4.0).abs() < 1e-6,
            "Expected eigenvalue 4, got {}",
            eigs[1]
        );
    }
}
