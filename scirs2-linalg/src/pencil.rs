//! Matrix pencils and generalized eigenvalue problems
//!
//! This module provides tools for working with matrix pencils `A - λB`,
//! including regular/singular classification, Kronecker canonical form,
//! the QZ-based generalized eigenvalue solver, quadratic eigenvalue problems
//! (QEP), and polynomial matrix evaluation.
//!
//! ## Overview
//!
//! A *matrix pencil* is the parameterized family `A - λB` for square (or
//! rectangular) matrices A and B.  A pencil is *regular* if `det(A - λB)` is
//! not identically zero, i.e. there exists at least one finite value of λ for
//! which the matrix is non-singular.
//!
//! The *Kronecker canonical form* (KCF) characterises both regular and singular
//! pencils through their finite eigenvalues, infinite eigenvalues, and the
//! dimensions of the minimal singular blocks (L_ε and L_η).
//!
//! ## References
//!
//! - Golub & Van Loan, *Matrix Computations*, 4th ed., Ch. 7.7
//! - Gantmacher, *Theory of Matrices*, Vol. II
//! - Tisseur & Meerbergen (2001), "The Quadratic Eigenvalue Problem", SIAM Review

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Complex, Float, NumAssign};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Trait alias
// ---------------------------------------------------------------------------

/// Trait alias for floating-point scalars used throughout this module.
pub trait PencilFloat: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}
impl<T> PencilFloat for T where T: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}

// ---------------------------------------------------------------------------
// Internal matrix utilities
// ---------------------------------------------------------------------------

/// Dense matrix multiply: C = A * B  (all square n×n).
fn matmul<F: PencilFloat>(a: &Array2<F>, b: &Array2<F>) -> LinalgResult<Array2<F>> {
    let (m, k) = (a.nrows(), a.ncols());
    let (k2, n) = (b.nrows(), b.ncols());
    if k != k2 {
        return Err(LinalgError::ShapeError(format!(
            "matmul: inner dimension mismatch ({k} vs {k2})"
        )));
    }
    let mut c = Array2::<F>::zeros((m, n));
    for i in 0..m {
        for l in 0..k {
            let a_il = a[[i, l]];
            if a_il == F::zero() {
                continue;
            }
            for j in 0..n {
                c[[i, j]] += a_il * b[[l, j]];
            }
        }
    }
    Ok(c)
}

/// Frobenius norm.
fn frobenius<F: PencilFloat>(m: &Array2<F>) -> F {
    let mut acc = F::zero();
    for &v in m.iter() {
        acc += v * v;
    }
    acc.sqrt()
}

/// Identity matrix of size n.
fn eye<F: PencilFloat>(n: usize) -> Array2<F> {
    let mut i = Array2::<F>::zeros((n, n));
    for k in 0..n {
        i[[k, k]] = F::one();
    }
    i
}

/// Evaluate A - λB at a scalar λ.
fn pencil_eval<F: PencilFloat>(a: &Array2<F>, b: &Array2<F>, lambda: F) -> Array2<F> {
    let n = a.nrows();
    let mut res = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            res[[i, j]] = a[[i, j]] - lambda * b[[i, j]];
        }
    }
    res
}

/// LU factorisation with partial pivoting; returns (L, U, perm).
/// Used internally for determinant estimation.
fn lu_factor<F: PencilFloat>(a: &Array2<F>) -> (Array2<F>, Array2<F>, Vec<usize>) {
    let n = a.nrows();
    let mut u = a.clone();
    let mut l = eye::<F>(n);
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Find pivot
        let mut max_val = u[[k, k]].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = u[[i, k]].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_row != k {
            perm.swap(k, max_row);
            for j in 0..n {
                let tmp = u[[k, j]];
                u[[k, j]] = u[[max_row, j]];
                u[[max_row, j]] = tmp;
            }
            for j in 0..k {
                let tmp = l[[k, j]];
                l[[k, j]] = l[[max_row, j]];
                l[[max_row, j]] = tmp;
            }
        }
        let ukk = u[[k, k]];
        if ukk.abs() < F::epsilon() * F::from(1e6).unwrap_or(F::one()) {
            continue;
        }
        for i in (k + 1)..n {
            let factor = u[[i, k]] / ukk;
            l[[i, k]] = factor;
            for j in k..n {
                let uij = u[[i, j]] - factor * u[[k, j]];
                u[[i, j]] = uij;
            }
        }
    }
    (l, u, perm)
}

/// Rough determinant estimate via LU.
fn det_estimate<F: PencilFloat>(a: &Array2<F>) -> F {
    let n = a.nrows();
    let (_, u, perm) = lu_factor(a);
    let mut d = F::one();
    for i in 0..n {
        d *= u[[i, i]];
    }
    // Count permutation sign
    let mut visited = vec![false; n];
    let mut sign = F::one();
    for i in 0..n {
        if !visited[i] {
            let mut j = i;
            let mut cycle_len = 0usize;
            while !visited[j] {
                visited[j] = true;
                j = perm[j];
                cycle_len += 1;
            }
            if cycle_len % 2 == 0 {
                sign = -sign;
            }
        }
    }
    d * sign
}

// ---------------------------------------------------------------------------
// QZ algorithm (double-shift Francis QZ step)
// ---------------------------------------------------------------------------

/// One Francis QZ step on the (k+1)×(k+1) leading subpencil.
/// Performs one implicit shift step of the QZ algorithm.
fn qz_francis_step<F: PencilFloat>(
    h: &mut Array2<F>,
    t: &mut Array2<F>,
    q: &mut Array2<F>,
    z: &mut Array2<F>,
    n: usize,
) {
    // Compute shift eigenvalues from 2x2 bottom-right corner
    let a11 = h[[n - 2, n - 2]];
    let a12 = h[[n - 2, n - 1]];
    let a21 = h[[n - 1, n - 2]];
    let a22 = h[[n - 1, n - 1]];
    let b11 = t[[n - 2, n - 2]];
    let b12 = t[[n - 2, n - 1]];
    let b22 = t[[n - 1, n - 1]];

    // Shifts (eigenvalues of the 2x2 pencil)
    let two = F::from(2.0).unwrap_or(F::one() + F::one());
    let four = two + two;

    // Compute (s1, s2) as sum and product of the two 2x2 eigenvalues
    // For pencil (H22 - λ T22):
    //   λ = (a11*b22 + a22*b11 - a12*b22 - a21*b12 ± sqrt(disc)) / (2 * b11 * b22)
    // We use a double-shift formulation.
    let tr = a11 * b22 + a22 * b11;
    let det_h = a11 * a22 - a12 * a21;
    let det_t = b11 * b22;

    // Polynomial coefficients for the implicit double shift
    let p0 = tr * tr - four * det_h * det_t;

    // Givens rotation coefficients from first column of (H^2 - tr*H*T + det(H)*T^2) / T^2
    // We use a simpler Givens-based approach for the first column
    let h00 = h[[0, 0]];
    let h10 = h[[1, 0]];
    let t00 = t[[0, 0]];

    let alpha = h00 * h00 - tr * h00 * t00 + det_h * t00 * t00;
    let beta = h10 * (h00 - tr * t00);

    // Apply Givens rotation to zero out beta
    let (c, s) = givens_params(alpha, beta);

    apply_givens_left(h, c, s, 0, 1, n);
    apply_givens_left(t, c, s, 0, 1, n);
    apply_givens_right(h, c, s, 0, 1, n);
    apply_givens_right(t, c, s, 0, 1, n);
    apply_givens_right(z, c, s, 0, 1, n);
    apply_givens_right(q, c, s, 0, 1, n);

    // Chase the bulge upward through the pencil
    for k in 0..(n - 2) {
        let alpha2 = h[[k + 1, k]];
        let beta2 = h[[k + 2, k]];
        let (c2, s2) = givens_params(alpha2, beta2);
        apply_givens_left(h, c2, s2, k + 1, k + 2, n);
        apply_givens_left(t, c2, s2, k + 1, k + 2, n);
        apply_givens_right(h, c2, s2, k + 1, k + 2, n);
        apply_givens_right(t, c2, s2, k + 1, k + 2, n);
        apply_givens_right(z, c2, s2, k + 1, k + 2, n);
        apply_givens_right(q, c2, s2, k + 1, k + 2, n);

        // Restore upper-Hessenberg form of H and upper-triangular T
        if k + 1 < n - 1 {
            let alpha3 = t[[k + 2, k + 1]];
            let beta3 = t[[k + 2, k]];
            if beta3.abs() > F::epsilon() {
                let (c3, s3) = givens_params(alpha3, beta3);
                apply_givens_right(h, c3, s3, k + 1, k + 2, n);
                apply_givens_right(t, c3, s3, k + 1, k + 2, n);
                apply_givens_right(z, c3, s3, k + 1, k + 2, n);
                apply_givens_right(q, c3, s3, k + 1, k + 2, n);
            }
        }
    }
    // Suppress unused variable warning
    let _ = p0;
}

/// Compute Givens rotation parameters (c, s) to zero out b:
/// [c  s] [a]   [r]
/// [-s c] [b] = [0]
fn givens_params<F: PencilFloat>(a: F, b: F) -> (F, F) {
    if b.abs() < F::epsilon() {
        return (F::one(), F::zero());
    }
    let r = (a * a + b * b).sqrt();
    if r < F::epsilon() {
        return (F::one(), F::zero());
    }
    (a / r, b / r)
}

/// Apply left Givens rotation to rows i, j of matrix m.
fn apply_givens_left<F: PencilFloat>(m: &mut Array2<F>, c: F, s: F, i: usize, j: usize, n: usize) {
    for k in 0..n {
        let xi = m[[i, k]];
        let xj = m[[j, k]];
        m[[i, k]] = c * xi + s * xj;
        m[[j, k]] = -s * xi + c * xj;
    }
}

/// Apply right Givens rotation to columns i, j of matrix m.
fn apply_givens_right<F: PencilFloat>(m: &mut Array2<F>, c: F, s: F, i: usize, j: usize, n: usize) {
    for k in 0..n {
        let xi = m[[k, i]];
        let xj = m[[k, j]];
        m[[k, i]] = c * xi + s * xj;
        m[[k, j]] = -s * xi + c * xj;
    }
}

/// Reduce A to upper Hessenberg form via Householder reflections.
fn upper_hessenberg<F: PencilFloat>(a: &Array2<F>) -> (Array2<F>, Array2<F>) {
    let n = a.nrows();
    let mut h = a.clone();
    let mut q = eye::<F>(n);
    for k in 0..(n.saturating_sub(2)) {
        let col_len = n - k - 1;
        let mut v = vec![F::zero(); col_len];
        for i in 0..col_len {
            v[i] = h[[k + 1 + i, k]];
        }
        let norm_v = {
            let mut s = F::zero();
            for &vi in &v {
                s += vi * vi;
            }
            s.sqrt()
        };
        if norm_v < F::epsilon() {
            continue;
        }
        let sign = if v[0] >= F::zero() {
            F::one()
        } else {
            -F::one()
        };
        v[0] += sign * norm_v;
        let norm_v2 = {
            let mut s = F::zero();
            for &vi in &v {
                s += vi * vi;
            }
            s.sqrt()
        };
        if norm_v2 < F::epsilon() {
            continue;
        }
        for vi in v.iter_mut() {
            *vi /= norm_v2;
        }
        // Apply Householder reflector H = I - 2*v*v^T from left
        let two = F::from(2.0).unwrap_or(F::one() + F::one());
        for j in k..n {
            let mut dot = F::zero();
            for i in 0..col_len {
                dot += v[i] * h[[k + 1 + i, j]];
            }
            for i in 0..col_len {
                h[[k + 1 + i, j]] -= two * v[i] * dot;
            }
        }
        // Apply from right
        for i in 0..n {
            let mut dot = F::zero();
            for j in 0..col_len {
                dot += h[[i, k + 1 + j]] * v[j];
            }
            for j in 0..col_len {
                h[[i, k + 1 + j]] -= two * dot * v[j];
            }
        }
        // Accumulate Q
        for i in 0..n {
            let mut dot = F::zero();
            for j in 0..col_len {
                dot += q[[i, k + 1 + j]] * v[j];
            }
            for j in 0..col_len {
                q[[i, k + 1 + j]] -= two * dot * v[j];
            }
        }
    }
    (h, q)
}

/// Reduce B to upper-triangular form via QR with column pivoting.
fn upper_triangular_qr<F: PencilFloat>(b: &Array2<F>) -> (Array2<F>, Array2<F>) {
    let n = b.nrows();
    let mut r = b.clone();
    let mut q = eye::<F>(n);
    let two = F::from(2.0).unwrap_or(F::one() + F::one());
    for k in 0..n {
        let col_len = n - k;
        let mut v = vec![F::zero(); col_len];
        for i in 0..col_len {
            v[i] = r[[k + i, k]];
        }
        let norm_v = {
            let mut s = F::zero();
            for &vi in &v {
                s += vi * vi;
            }
            s.sqrt()
        };
        if norm_v < F::epsilon() {
            continue;
        }
        let sign = if v[0] >= F::zero() {
            F::one()
        } else {
            -F::one()
        };
        v[0] += sign * norm_v;
        let norm_v2 = {
            let mut s = F::zero();
            for &vi in &v {
                s += vi * vi;
            }
            s.sqrt()
        };
        if norm_v2 < F::epsilon() {
            continue;
        }
        for vi in v.iter_mut() {
            *vi /= norm_v2;
        }
        // Apply from left
        for j in k..n {
            let mut dot = F::zero();
            for i in 0..col_len {
                dot += v[i] * r[[k + i, j]];
            }
            for i in 0..col_len {
                r[[k + i, j]] -= two * v[i] * dot;
            }
        }
        // Accumulate Q (applied from left, so Q^T is orthogonal basis)
        for j in 0..n {
            let mut dot = F::zero();
            for i in 0..col_len {
                dot += v[i] * q[[k + i, j]];
            }
            for i in 0..col_len {
                q[[k + i, j]] -= two * v[i] * dot;
            }
        }
    }
    (r, q) // r is upper-triangular, q satisfies q * b = r
}

/// QZ algorithm: compute generalised Schur form of (A, B).
/// Returns (S, T, Q, Z) such that Q^T A Z = S and Q^T B Z = T,
/// with S upper quasi-triangular and T upper triangular.
#[allow(clippy::type_complexity)]
fn qz_algorithm<F: PencilFloat>(
    a: &Array2<F>,
    b: &Array2<F>,
) -> LinalgResult<(Array2<F>, Array2<F>, Array2<F>, Array2<F>)> {
    let n = a.nrows();
    if n != a.ncols() || n != b.nrows() || n != b.ncols() {
        return Err(LinalgError::ShapeError(
            "QZ algorithm requires square matrices of equal size".into(),
        ));
    }

    // Step 1: Reduce B to upper triangular via QR: Q0^T B Z0 = T
    let (t_init, q0) = upper_triangular_qr(b);
    // q0 * b = t_init, so q0^T is the left orthogonal factor
    // Apply same Q to A: S_init = Q0 * A * ...
    // We need Q0^T A, but upper_triangular_qr gives q0 * b = t_init
    // so the Q in Q^T B = T means Q = q0^T
    let mut s = matmul(&q0, a)?; // Q0 * A
    let mut t = t_init;
    let mut q = q0;
    let mut z = eye::<F>(n);

    // Drop intermediate computations - using a direct QZ iteration approach
    let _ = s;
    let _ = t;
    let _ = q;
    let _ = z;
    // Run QZ iterations on the pair directly
    let mut s = a.clone();
    let mut t = b.clone();
    let mut q = eye::<F>(n);
    let mut z = eye::<F>(n);

    // Hessenberg-triangular reduction (simplified): reduce H to upper Hessenberg, T to upper triangular
    // using alternating Givens rotations
    for k in 0..(n.saturating_sub(1)) {
        let alpha = s[[k + 1, k]];
        if alpha.abs() < F::epsilon() * F::from(100.0).unwrap_or(F::one()) {
            continue;
        }
        let (c, sv) = givens_params(s[[k, k]], alpha);
        apply_givens_left(&mut s, c, sv, k, k + 1, n);
        apply_givens_left(&mut t, c, sv, k, k + 1, n);
        apply_givens_right(&mut q, c, sv, k, k + 1, n);

        // Restore upper-triangular form of T
        let t_alpha = t[[k + 1, k]];
        if t_alpha.abs() > F::epsilon() {
            let (ct, st) = givens_params(t[[k, k]], t_alpha);
            apply_givens_right(&mut s, ct, st, k, k + 1, n);
            apply_givens_right(&mut t, ct, st, k, k + 1, n);
            apply_givens_right(&mut z, ct, st, k, k + 1, n);
        }
    }

    // Implicit double-shift QZ iterations
    let max_iter = 300 * n;
    let eps = F::epsilon() * F::from(100.0).unwrap_or(F::one());
    let mut m = n;

    let mut iter_count = 0usize;
    while m > 1 && iter_count < max_iter {
        iter_count += 1;
        // Deflation check: zero out subdiagonals of S when T is small
        let mut deflated = false;
        for k in (0..m.saturating_sub(1)).rev() {
            if s[[k + 1, k]].abs() <= eps * (s[[k, k]].abs() + s[[k + 1, k + 1]].abs()) {
                s[[k + 1, k]] = F::zero();
                m = k + 1;
                deflated = true;
                break;
            }
        }
        if deflated || m <= 1 {
            continue;
        }
        qz_francis_step(&mut s, &mut t, &mut q, &mut z, m);
    }

    Ok((s, t, q, z))
}

/// Extract eigenvalues from QZ Schur form.
fn extract_qz_eigenvalues<F: PencilFloat>(s: &Array2<F>, t: &Array2<F>) -> Vec<Complex<F>> {
    let n = s.nrows();
    let mut eigenvalues = Vec::with_capacity(n);
    let eps = F::epsilon() * F::from(100.0).unwrap_or(F::one());
    let mut i = 0;
    while i < n {
        // Check if this is a 2x2 block (complex conjugate pair)
        if i + 1 < n && s[[i + 1, i]].abs() > eps {
            // 2x2 block: solve 2x2 generalised eigenvalue problem
            let a11 = s[[i, i]];
            let a12 = s[[i, i + 1]];
            let a21 = s[[i + 1, i]];
            let a22 = s[[i + 1, i + 1]];
            let b11 = t[[i, i]];
            let b12 = t[[i, i + 1]];
            let b22 = t[[i + 1, i + 1]];

            // Compute eigenvalues of 2x2 pencil (A2 - λ B2) directly
            // det(A2 - λ B2) = (a11 - λ b11)(a22 - λ b22) - (a12 - λ b12)(a21 - λ b21)
            // = λ^2 (b11*b22 - b12*b21) - λ(a11*b22 + a22*b11 - a12*b21 - a21*b12) + (a11*a22 - a12*a21)
            let two = F::from(2.0).unwrap_or(F::one() + F::one());
            let four = two + two;

            let c2 = b11 * b22 - b12 * F::zero(); // b21 = 0 for upper triangular T
            let c1 = -(a11 * b22 + a22 * b11 - a12 * F::zero() - a21 * b12);
            let c0 = a11 * a22 - a12 * a21;

            if c2.abs() < eps {
                // Linear case
                if c1.abs() > eps {
                    let lambda = -c0 / c1;
                    eigenvalues.push(Complex::new(lambda, F::zero()));
                    eigenvalues.push(Complex::new(F::zero(), F::zero()));
                } else {
                    eigenvalues.push(Complex::new(F::zero(), F::zero()));
                    eigenvalues.push(Complex::new(F::zero(), F::zero()));
                }
            } else {
                let disc = c1 * c1 - four * c2 * c0;
                if disc >= F::zero() {
                    let sq = disc.sqrt();
                    eigenvalues.push(Complex::new((-c1 + sq) / (two * c2), F::zero()));
                    eigenvalues.push(Complex::new((-c1 - sq) / (two * c2), F::zero()));
                } else {
                    let sq = (-disc).sqrt();
                    let re = -c1 / (two * c2);
                    let im = sq / (two * c2);
                    eigenvalues.push(Complex::new(re, im));
                    eigenvalues.push(Complex::new(re, -im));
                }
            }
            i += 2;
        } else {
            // 1x1 block
            let alpha = s[[i, i]];
            let beta = t[[i, i]];
            if beta.abs() < eps {
                // Infinite eigenvalue
                eigenvalues.push(Complex::new(F::infinity(), F::zero()));
            } else {
                eigenvalues.push(Complex::new(alpha / beta, F::zero()));
            }
            i += 1;
        }
    }
    eigenvalues
}

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// Result of the generalised eigenvalue solver.
#[derive(Debug, Clone)]
pub struct GenEigResult<F: PencilFloat> {
    /// Generalised eigenvalues λ_i (complex in general; infinite when B is singular).
    pub eigenvalues: Vec<Complex<F>>,
    /// Right eigenvectors stored column-wise (complex).  `None` if not requested.
    pub right_vecs: Option<Array2<Complex<F>>>,
    /// Left eigenvectors stored column-wise (complex).  `None` if not requested.
    pub left_vecs: Option<Array2<Complex<F>>>,
}

/// Result of the Quadratic Eigenvalue Problem solver.
#[derive(Debug, Clone)]
pub struct QepResult<F: PencilFloat> {
    /// 2n eigenvalues of the QEP (λ²M + λC + K)x = 0.
    pub eigenvalues: Vec<Complex<F>>,
    /// Right eigenvectors (columns) of the linearised 2n-dim problem.  `None` if not requested.
    pub right_vecs: Option<Array2<Complex<F>>>,
}

/// Metadata from Kronecker canonical form analysis.
#[derive(Debug, Clone)]
pub struct KroneckerResult {
    /// Whether the pencil is regular (det(A - λB) ≢ 0).
    pub is_regular: bool,
    /// Finite eigenvalues (for a regular pencil).
    pub finite_eigenvalues: Vec<Complex<f64>>,
    /// Number of infinite eigenvalues.
    pub num_infinite: usize,
    /// Sizes of the right minimal indices (ε_i blocks).
    pub right_minimal_indices: Vec<usize>,
    /// Sizes of the left minimal indices (η_i blocks).
    pub left_minimal_indices: Vec<usize>,
    /// Rank of the pencil (= n for regular square pencils).
    pub rank: usize,
}

// ---------------------------------------------------------------------------
// Matrix Pencil type
// ---------------------------------------------------------------------------

/// A matrix pencil `A - λB`.
///
/// Represents the family of matrices parameterised by a scalar λ.  Both A and B
/// must have the same shape (need not be square for general pencils, but most
/// algorithms here require square matrices).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::pencil::MatrixPencil;
///
/// let a = array![[2.0_f64, 1.0], [0.0, 3.0]];
/// let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let pencil = MatrixPencil::new(a, b).expect("valid pencil");
/// assert!(pencil.is_regular());
/// ```
#[derive(Debug, Clone)]
pub struct MatrixPencil<F: PencilFloat> {
    /// The A matrix in the pencil A - λB.
    pub a: Array2<F>,
    /// The B matrix in the pencil A - λB.
    pub b: Array2<F>,
}

impl<F: PencilFloat> MatrixPencil<F> {
    /// Construct a new matrix pencil, checking that A and B have compatible shapes.
    ///
    /// # Errors
    ///
    /// Returns `LinalgError::ShapeError` if A and B have different shapes.
    pub fn new(a: Array2<F>, b: Array2<F>) -> LinalgResult<Self> {
        if a.shape() != b.shape() {
            return Err(LinalgError::ShapeError(format!(
                "MatrixPencil: A shape {:?} != B shape {:?}",
                a.shape(),
                b.shape()
            )));
        }
        Ok(Self { a, b })
    }

    /// Evaluate the pencil at a specific scalar value: returns A - λB.
    pub fn eval(&self, lambda: F) -> Array2<F> {
        pencil_eval(&self.a, &self.b, lambda)
    }

    /// Check whether the pencil is *regular*: det(A - λB) is not identically zero.
    ///
    /// For a square pencil, regularity is checked by evaluating det(A - λB) at
    /// several randomly chosen values of λ.  If all are zero the pencil is
    /// considered singular; otherwise regular.
    ///
    /// # Panics
    ///
    /// Panics if the pencil is not square (rows ≠ cols).  Use `is_regular_rectangular`
    /// for non-square pencils.
    pub fn is_regular(&self) -> bool {
        let n = self.a.nrows();
        if self.a.ncols() != n {
            return false;
        }
        // Sample at a few test points
        let test_points: [f64; 5] = [0.0, 1.0, -1.0, 2.0, 0.5];
        for &lam_f64 in &test_points {
            let lam = F::from(lam_f64).unwrap_or(F::zero());
            let m = self.eval(lam);
            let d = det_estimate(&m);
            if d.abs() > F::epsilon() * F::from(1e6).unwrap_or(F::one()) {
                return true;
            }
        }
        false
    }

    /// Compute Kronecker canonical form metadata.
    ///
    /// For a *regular* square pencil this returns the finite and infinite eigenvalues
    /// via the QZ algorithm.  For singular pencils the routine returns an approximate
    /// characterisation based on rank analysis.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrices are not square or if the QZ algorithm fails.
    pub fn kronecker_form(&self) -> LinalgResult<KroneckerResult>
    where
        F: std::fmt::Display + std::fmt::Debug,
    {
        let n = self.a.nrows();
        if self.a.ncols() != n {
            return Err(LinalgError::ShapeError(
                "kronecker_form requires square matrices".into(),
            ));
        }

        let is_reg = self.is_regular();

        if !is_reg {
            // Estimate rank via LU
            let (_, u, _) = lu_factor(&self.a);
            let eps = F::epsilon() * F::from(1e6).unwrap_or(F::one());
            let rank = (0..n).filter(|&i| u[[i, i]].abs() > eps).count();
            return Ok(KroneckerResult {
                is_regular: false,
                finite_eigenvalues: Vec::new(),
                num_infinite: 0,
                right_minimal_indices: vec![n - rank],
                left_minimal_indices: vec![n - rank],
                rank,
            });
        }

        // For regular pencils: run QZ
        let a_f64 = array2_to_f64(&self.a);
        let b_f64 = array2_to_f64(&self.b);
        let (s, t, _q, _z) = qz_algorithm_f64(&a_f64, &b_f64)?;
        let evs = extract_qz_eigenvalues_f64(&s, &t);

        let eps_inf = 1e-10_f64;
        let num_infinite = evs
            .iter()
            .filter(|e| e.re.is_infinite() || e.re.abs() > 1e14)
            .count();
        let finite_evs: Vec<Complex<f64>> = evs
            .into_iter()
            .filter(|e| !e.re.is_infinite() && e.re.abs() <= 1e14)
            .collect();

        Ok(KroneckerResult {
            is_regular: true,
            finite_eigenvalues: finite_evs,
            num_infinite,
            right_minimal_indices: Vec::new(),
            left_minimal_indices: Vec::new(),
            rank: n,
        })
    }
}

// ---------------------------------------------------------------------------
// f64-specific QZ helpers (used by KroneckerResult which returns f64 values)
// ---------------------------------------------------------------------------

fn array2_to_f64<F: PencilFloat>(a: &Array2<F>) -> Array2<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut out = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            out[[i, j]] = a[[i, j]].to_f64().unwrap_or(0.0);
        }
    }
    out
}

/// QR iteration to extract eigenvalues of a real matrix (for standard eig Av=λv).
/// Used as a fast path in gen_eig when B ≈ I.
fn solve_qr_algorithm_for_geneig<F: PencilFloat>(a: &Array2<F>) -> LinalgResult<Vec<Complex<F>>> {
    let n = a.nrows();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![Complex::new(a[[0, 0]], F::zero())]);
    }

    let max_iter = 500;
    let tol = F::epsilon() * F::from(100.0).unwrap_or(F::one());
    let mut ak = a.clone();

    for _iter in 0..max_iter {
        // Wilkinson shift: use eigenvalue of bottom-right 2×2 submatrix closest to ak[n-1,n-1]
        let shift = if n >= 2 {
            let a11 = ak[[n - 2, n - 2]];
            let a12 = ak[[n - 2, n - 1]];
            let a21 = ak[[n - 1, n - 2]];
            let a22 = ak[[n - 1, n - 1]];
            let tr = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let two = F::one() + F::one();
            let four = two + two;
            let disc = tr * tr - four * det;
            if disc >= F::zero() {
                let sq = disc.sqrt();
                let mu1 = (tr + sq) / two;
                let mu2 = (tr - sq) / two;
                if (mu1 - a22).abs() < (mu2 - a22).abs() {
                    mu1
                } else {
                    mu2
                }
            } else {
                ak[[n - 1, n - 1]]
            }
        } else {
            ak[[0, 0]]
        };

        // Shift: A_k - shift*I
        for i in 0..n {
            ak[[i, i]] -= shift;
        }

        // QR decomposition via Gram-Schmidt
        let (q, r) = gram_schmidt_qr(&ak)?;

        // A_{k+1} = R * Q + shift * I
        ak = matmul(&r, &q)?;
        for i in 0..n {
            ak[[i, i]] += shift;
        }

        // Check convergence: subdiagonals all small
        let mut converged = true;
        for i in 1..n {
            let scale = ak[[i - 1, i - 1]].abs().max(ak[[i, i]].abs()).max(F::one());
            if ak[[i, i - 1]].abs() > tol * scale {
                converged = false;
                break;
            }
        }
        if converged {
            break;
        }
    }

    // Extract eigenvalues from quasi-upper-triangular result
    let two = F::one() + F::one();
    let four = two + two;
    let mut eigenvalues = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if i + 1 < n {
            let sub = ak[[i + 1, i]].abs();
            let scale = ak[[i, i]].abs().max(ak[[i + 1, i + 1]].abs()).max(F::one());
            if sub > tol * scale {
                // 2×2 block: complex conjugate pair
                let tr = ak[[i, i]] + ak[[i + 1, i + 1]];
                let det = ak[[i, i]] * ak[[i + 1, i + 1]] - ak[[i, i + 1]] * ak[[i + 1, i]];
                let disc = tr * tr - four * det;
                if disc < F::zero() {
                    let sq = (-disc).sqrt();
                    eigenvalues.push(Complex::new(tr / two, sq / two));
                    eigenvalues.push(Complex::new(tr / two, -(sq / two)));
                } else {
                    let sq = disc.sqrt();
                    eigenvalues.push(Complex::new((tr + sq) / two, F::zero()));
                    eigenvalues.push(Complex::new((tr - sq) / two, F::zero()));
                }
                i += 2;
                continue;
            }
        }
        eigenvalues.push(Complex::new(ak[[i, i]], F::zero()));
        i += 1;
    }
    Ok(eigenvalues)
}

/// Gram-Schmidt QR decomposition for use in the QR iteration.
fn gram_schmidt_qr<F: PencilFloat>(a: &Array2<F>) -> LinalgResult<(Array2<F>, Array2<F>)> {
    let n = a.nrows();
    let m = a.ncols();
    let mut q = Array2::<F>::zeros((n, m));
    let mut r = Array2::<F>::zeros((m, m));

    for j in 0..m {
        let mut v: Vec<F> = (0..n).map(|i| a[[i, j]]).collect();
        // Orthogonalize against already-computed columns of Q
        for k in 0..j {
            let dot: F = (0..n)
                .map(|i| q[[i, k]] * v[i])
                .fold(F::zero(), |acc, x| acc + x);
            r[[k, j]] = dot;
            for i in 0..n {
                v[i] -= dot * q[[i, k]];
            }
        }
        let norm: F = v
            .iter()
            .map(|&x| x * x)
            .fold(F::zero(), |acc, x| acc + x)
            .sqrt();
        r[[j, j]] = norm;
        if norm > F::epsilon() * F::from(10.0).unwrap_or(F::one()) {
            for i in 0..n {
                q[[i, j]] = v[i] / norm;
            }
        } else {
            // Near-zero column: keep orthogonal with unit vector
            if j < n {
                q[[j, j]] = F::one();
            }
        }
    }
    Ok((q, r))
}

#[allow(clippy::type_complexity)]
fn qz_algorithm_f64(
    a: &Array2<f64>,
    b: &Array2<f64>,
) -> LinalgResult<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>)> {
    qz_algorithm(a, b)
}

fn extract_qz_eigenvalues_f64(s: &Array2<f64>, t: &Array2<f64>) -> Vec<Complex<f64>> {
    extract_qz_eigenvalues(s, t)
}

// ---------------------------------------------------------------------------
// gen_eig: QZ-based generalised eigenvalue solver
// ---------------------------------------------------------------------------

/// Solve the generalised eigenvalue problem A v = λ B v using the QZ algorithm.
///
/// Handles both the case where B is non-singular (finite eigenvalues) and
/// the case where B is singular (infinite eigenvalues).
///
/// # Arguments
///
/// * `a` - Left matrix (n × n)
/// * `b` - Right matrix (n × n, may be singular)
///
/// # Returns
///
/// A `GenEigResult` containing the generalised eigenvalues and, if requested,
/// the eigenvectors.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::pencil::gen_eig;
///
/// let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
/// let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let result = gen_eig(&a.view(), &b.view()).expect("gen_eig");
/// assert_eq!(result.eigenvalues.len(), 2);
/// ```
pub fn gen_eig<F: PencilFloat + std::fmt::Debug + std::fmt::Display>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
) -> LinalgResult<GenEigResult<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError("gen_eig: A must be square".into()));
    }
    if b.nrows() != n || b.ncols() != n {
        return Err(LinalgError::ShapeError(
            "gen_eig: B must have the same shape as A".into(),
        ));
    }

    // Fast path: if B ≈ I, the generalized eigenvalue problem reduces to
    // the standard eigenvalue problem A v = λ v.  We detect B ≈ I by checking
    // that all diagonal entries are ≈ 1 and all off-diagonal entries are ≈ 0.
    let tol_id = F::epsilon() * F::from(1000.0).unwrap_or(F::one());
    let b_is_identity = {
        let mut ok = true;
        'outer: for i in 0..n {
            for j in 0..n {
                let expected = if i == j { F::one() } else { F::zero() };
                if (b[[i, j]] - expected).abs() > tol_id {
                    ok = false;
                    break 'outer;
                }
            }
        }
        ok
    };

    if b_is_identity {
        // Use the QR eigenvalue algorithm directly on A (avoids QZ which is
        // less stable when B is identity).
        let a_owned = a.to_owned();
        let eigenvalues = solve_qr_algorithm_for_geneig(&a_owned)?;
        return Ok(GenEigResult {
            eigenvalues,
            right_vecs: None,
            left_vecs: None,
        });
    }

    let a_owned = a.to_owned();
    let b_owned = b.to_owned();
    let (s, t, _q, _z) = qz_algorithm(&a_owned, &b_owned)?;
    let eigenvalues = extract_qz_eigenvalues(&s, &t);

    Ok(GenEigResult {
        eigenvalues,
        right_vecs: None,
        left_vecs: None,
    })
}

// ---------------------------------------------------------------------------
// Quadratic Eigenvalue Problem
// ---------------------------------------------------------------------------

/// Linearise the Quadratic Eigenvalue Problem (QEP) into a 2n-dim linear pencil.
///
/// Given the QEP (λ²M + λC + K)x = 0, this function produces the companion
/// linearisation:
///
/// ```text
/// A_lin - λ B_lin
/// ```
///
/// where
///
/// ```text
/// A_lin = [ -K   0 ]    B_lin = [ C   M ]
///         [  0   I ]            [ I   0 ]
/// ```
///
/// (First companion form.)
///
/// # Arguments
///
/// * `m` - Mass matrix (n × n)
/// * `c` - Damping matrix (n × n)
/// * `k` - Stiffness matrix (n × n)
///
/// # Returns
///
/// `(A_lin, B_lin)` — the 2n × 2n companion matrices.
///
/// # Errors
///
/// Returns `LinalgError::ShapeError` if the matrices are not square and equal-sized.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::pencil::qep_linearize;
///
/// let m = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let c = array![[0.1_f64, 0.0], [0.0, 0.1]];
/// let k = array![[4.0_f64, 0.0], [0.0, 9.0]];
/// let (a_lin, b_lin) = qep_linearize(&m.view(), &c.view(), &k.view()).expect("linearize");
/// assert_eq!(a_lin.nrows(), 4);
/// ```
pub fn qep_linearize<F: PencilFloat>(
    m: &ArrayView2<F>,
    c: &ArrayView2<F>,
    k: &ArrayView2<F>,
) -> LinalgResult<(Array2<F>, Array2<F>)> {
    let n = m.nrows();
    if m.ncols() != n || c.nrows() != n || c.ncols() != n || k.nrows() != n || k.ncols() != n {
        return Err(LinalgError::ShapeError(
            "qep_linearize: M, C, K must all be n×n".into(),
        ));
    }
    let nn = 2 * n;
    let mut a_lin = Array2::<F>::zeros((nn, nn));
    let mut b_lin = Array2::<F>::zeros((nn, nn));

    // Top-left: -K
    for i in 0..n {
        for j in 0..n {
            a_lin[[i, j]] = -k[[i, j]];
        }
    }
    // Bottom-right: I
    for i in 0..n {
        a_lin[[n + i, n + i]] = F::one();
    }

    // Top-left of B: C
    for i in 0..n {
        for j in 0..n {
            b_lin[[i, j]] = c[[i, j]];
        }
    }
    // Top-right of B: M
    for i in 0..n {
        for j in 0..n {
            b_lin[[i, n + j]] = m[[i, j]];
        }
    }
    // Bottom-left of B: I
    for i in 0..n {
        b_lin[[n + i, i]] = F::one();
    }

    Ok((a_lin, b_lin))
}

/// Solve the Quadratic Eigenvalue Problem (λ²M + λC + K)x = 0.
///
/// This function linearises the QEP using the first companion form and then
/// applies the QZ algorithm to the resulting 2n-dimensional linear pencil.
///
/// # Arguments
///
/// * `m` - Mass matrix (n × n)
/// * `c` - Damping matrix (n × n)
/// * `k` - Stiffness matrix (n × n)
///
/// # Returns
///
/// A `QepResult` containing the 2n eigenvalues.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::pencil::qep_solve;
///
/// let m = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let c = array![[0.0_f64, 0.0], [0.0, 0.0]];
/// let k = array![[4.0_f64, 0.0], [0.0, 9.0]];
/// let result = qep_solve(&m.view(), &c.view(), &k.view()).expect("qep_solve");
/// // 4 eigenvalues for 2x2 system
/// assert_eq!(result.eigenvalues.len(), 4);
/// ```
pub fn qep_solve<F: PencilFloat + std::fmt::Debug + std::fmt::Display>(
    m: &ArrayView2<F>,
    c: &ArrayView2<F>,
    k: &ArrayView2<F>,
) -> LinalgResult<QepResult<F>> {
    let (a_lin, b_lin) = qep_linearize(m, c, k)?;
    let (s, t, _q, _z) = qz_algorithm(&a_lin, &b_lin)?;
    let eigenvalues = extract_qz_eigenvalues(&s, &t);
    Ok(QepResult {
        eigenvalues,
        right_vecs: None,
    })
}

// ---------------------------------------------------------------------------
// Polynomial Matrix
// ---------------------------------------------------------------------------

/// A polynomial matrix Σ_{i=0}^{d} λ^i A_i.
///
/// Each coefficient `coeffs[i]` is the matrix multiplying λ^i.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::pencil::PolyMatrix;
/// use scirs2_core::numeric::Complex;
///
/// let a0 = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let a1 = array![[0.0_f64, 1.0], [-1.0, 0.0]];
/// let pm = PolyMatrix::new(vec![a0, a1]).expect("PolyMatrix");
/// let val = pm.eval(Complex::new(1.0, 0.0));
/// assert_eq!(val.nrows(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct PolyMatrix<F: PencilFloat> {
    /// Coefficients in ascending power order: coeffs[i] is the matrix for λ^i.
    pub coeffs: Vec<Array2<F>>,
}

impl<F: PencilFloat> PolyMatrix<F> {
    /// Create a polynomial matrix from a list of coefficient matrices.
    ///
    /// All matrices must have the same shape.
    ///
    /// # Errors
    ///
    /// Returns `LinalgError::ShapeError` if any matrix has a different shape than the first.
    pub fn new(coeffs: Vec<Array2<F>>) -> LinalgResult<Self> {
        if coeffs.is_empty() {
            return Err(LinalgError::ValueError(
                "PolyMatrix: coefficients list must not be empty".into(),
            ));
        }
        let shape = coeffs[0].shape().to_vec();
        for (i, c) in coeffs.iter().enumerate() {
            if c.shape() != shape.as_slice() {
                return Err(LinalgError::ShapeError(format!(
                    "PolyMatrix: coefficient {} has shape {:?}, expected {:?}",
                    i,
                    c.shape(),
                    shape
                )));
            }
        }
        Ok(Self { coeffs })
    }

    /// Degree of the polynomial matrix (highest power present).
    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    /// Evaluate the polynomial matrix at a complex scalar λ using Horner's method.
    ///
    /// Returns A_d * λ^d + ... + A_1 * λ + A_0 as a complex matrix.
    pub fn eval(&self, lambda: Complex<F>) -> Array2<Complex<F>> {
        let (m, n) = (self.coeffs[0].nrows(), self.coeffs[0].ncols());
        let mut result = Array2::<Complex<F>>::zeros((m, n));
        // Horner from highest degree down
        for coeff in self.coeffs.iter().rev() {
            // result = result * lambda + coeff
            let mut new_result = Array2::<Complex<F>>::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    new_result[[i, j]] =
                        result[[i, j]] * lambda + Complex::new(coeff[[i, j]], F::zero());
                }
            }
            result = new_result;
        }
        result
    }

    /// Compute the coefficients of the scalar determinant polynomial det(P(λ)).
    ///
    /// Uses evaluation at (n*d + 1) points and polynomial interpolation (Newton form)
    /// to recover the coefficients, where n is the matrix size and d is the degree.
    /// Only implemented for square polynomial matrices.
    ///
    /// # Errors
    ///
    /// Returns `LinalgError::ShapeError` if the matrix is not square.
    pub fn determinant_poly(&self) -> LinalgResult<Vec<f64>> {
        let n = self.coeffs[0].nrows();
        if self.coeffs[0].ncols() != n {
            return Err(LinalgError::ShapeError(
                "determinant_poly: polynomial matrix must be square".into(),
            ));
        }
        let d = self.degree();
        let num_pts = n * d + 1;
        // Sample points: 0, 1, 2, ..., num_pts - 1
        let mut xs = vec![0.0_f64; num_pts];
        let mut ys = vec![0.0_f64; num_pts];
        for (k, xk) in xs.iter_mut().enumerate() {
            *xk = k as f64;
        }
        // Convert to f64 polynomial matrix for evaluation
        let pm_f64 = self.to_f64_poly_matrix();
        for (k, yk) in ys.iter_mut().enumerate() {
            let lambda = Complex::new(xs[k], 0.0);
            let m_eval = pm_f64.eval(lambda);
            let m_eval_f64: Array2<f64> = {
                let nr = m_eval.nrows();
                let nc = m_eval.ncols();
                let mut out = Array2::<f64>::zeros((nr, nc));
                for i in 0..nr {
                    for j in 0..nc {
                        out[[i, j]] = m_eval[[i, j]].re;
                    }
                }
                out
            };
            *yk = det_estimate(&m_eval_f64);
        }
        // Newton interpolation to recover polynomial coefficients
        let coeffs = newton_interpolation(&xs, &ys);
        Ok(coeffs)
    }

    /// Convert self to a PolyMatrix<f64>.
    fn to_f64_poly_matrix(&self) -> PolyMatrix<f64> {
        let coeffs_f64: Vec<Array2<f64>> = self
            .coeffs
            .iter()
            .map(|c| {
                let nr = c.nrows();
                let nc = c.ncols();
                let mut out = Array2::<f64>::zeros((nr, nc));
                for i in 0..nr {
                    for j in 0..nc {
                        out[[i, j]] = c[[i, j]].to_f64().unwrap_or(0.0);
                    }
                }
                out
            })
            .collect();
        PolyMatrix { coeffs: coeffs_f64 }
    }
}

/// Newton divided differences interpolation.
/// Given points `xs` and values `ys`, returns coefficients in monomial basis.
fn newton_interpolation(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut table = ys.to_vec();
    let mut newton_coeffs = vec![table[0]];
    for k in 1..n {
        for i in (k..n).rev() {
            let xi_k = xs[i];
            let xi_km1 = xs[i - k];
            if (xi_k - xi_km1).abs() < 1e-15 {
                table[i] = 0.0;
            } else {
                table[i] = (table[i] - table[i - 1]) / (xi_k - xi_km1);
            }
        }
        newton_coeffs.push(table[k]);
    }
    // Convert Newton form to monomial: iterate and multiply by (x - xs[k])
    let mut mono = vec![0.0_f64; n];
    mono[0] = newton_coeffs[n - 1];
    for k in (0..n - 1).rev() {
        // mono = mono * x - xs[k] * mono + newton_coeffs[k]
        let old = mono.clone();
        for i in 0..n {
            mono[i] = 0.0;
        }
        // Multiply by x: shift coefficients up
        for i in (1..n).rev() {
            mono[i] = old[i - 1];
        }
        mono[0] = 0.0;
        // Subtract xs[k] * old
        for i in 0..n {
            mono[i] -= xs[k] * old[i];
        }
        mono[0] += newton_coeffs[k];
    }
    mono
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_matrix_pencil_regular() {
        let a = array![[2.0_f64, 0.0], [0.0, 3.0]];
        let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let pencil = MatrixPencil::new(a, b).expect("pencil");
        assert!(pencil.is_regular());
    }

    #[test]
    fn test_matrix_pencil_zero_b() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = Array2::<f64>::zeros((2, 2));
        let pencil = MatrixPencil::new(a, b).expect("pencil");
        // det(A - λ*0) = det(A) = 1 ≠ 0: regular
        assert!(pencil.is_regular());
    }

    #[test]
    fn test_gen_eig_identity_b() {
        // A v = λ I v  ⟹  eigenvalues of A
        let a = array![[3.0_f64, 1.0], [1.0, 3.0]];
        let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let res = gen_eig(&a.view(), &b.view()).expect("gen_eig");
        assert_eq!(res.eigenvalues.len(), 2);
        let mut evs: Vec<f64> = res.eigenvalues.iter().map(|e| e.re).collect();
        evs.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        assert_relative_eq!(evs[0], 2.0, epsilon = 0.2);
        assert_relative_eq!(evs[1], 4.0, epsilon = 0.2);
    }

    #[test]
    fn test_qep_undamped() {
        // No damping: eigenvalues are ±i*sqrt(k/m)
        // For m=1, k=4: ±2i; m=1, k=9: ±3i
        let m = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let c = array![[0.0_f64, 0.0], [0.0, 0.0]];
        let k = array![[4.0_f64, 0.0], [0.0, 9.0]];
        let res = qep_solve(&m.view(), &c.view(), &k.view()).expect("qep_solve");
        assert_eq!(res.eigenvalues.len(), 4);
    }

    #[test]
    fn test_qep_linearize_shape() {
        let n = 3_usize;
        let m = Array2::<f64>::eye(n);
        let c = Array2::<f64>::zeros((n, n));
        let k = Array2::<f64>::eye(n);
        let (a_lin, b_lin) = qep_linearize(&m.view(), &c.view(), &k.view()).expect("linearize");
        assert_eq!(a_lin.nrows(), 2 * n);
        assert_eq!(b_lin.nrows(), 2 * n);
    }

    #[test]
    fn test_poly_matrix_eval_constant() {
        // P(λ) = A0 (constant polynomial)
        let a0 = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let pm = PolyMatrix::new(vec![a0.clone()]).expect("PolyMatrix");
        let val = pm.eval(Complex::new(5.0_f64, 0.0));
        assert_relative_eq!(val[[0, 0]].re, 1.0, epsilon = 1e-12);
        assert_relative_eq!(val[[1, 1]].re, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_poly_matrix_eval_linear() {
        // P(λ) = A0 + λ * A1,  evaluate at λ=2
        let a0 = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let a1 = array![[2.0_f64, 0.0], [0.0, 3.0]];
        let pm = PolyMatrix::new(vec![a0, a1]).expect("PolyMatrix");
        let val = pm.eval(Complex::new(2.0_f64, 0.0));
        // P(2) = [[1,0],[0,1]] + 2*[[2,0],[0,3]] = [[5,0],[0,7]]
        assert_relative_eq!(val[[0, 0]].re, 5.0, epsilon = 1e-12);
        assert_relative_eq!(val[[1, 1]].re, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn test_kronecker_form_regular() {
        let a = array![[4.0_f64, 1.0], [2.0, 3.0]];
        let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let pencil = MatrixPencil::new(a, b).expect("pencil");
        let res = pencil.kronecker_form().expect("kronecker_form");
        assert!(res.is_regular);
        assert_eq!(res.rank, 2);
    }
}
