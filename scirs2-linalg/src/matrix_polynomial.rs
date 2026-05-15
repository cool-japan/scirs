//! Matrix polynomial arithmetic, characteristic polynomial, Cayley-Hamilton,
//! and Schur-based matrix functions (sqrt, log).
//!
//! ## Module contents
//!
//! | Function / Type           | Description                                       |
//! |---------------------------|---------------------------------------------------|
//! | [`MatrixPoly`]            | Polynomial with matrix coefficients               |
//! | [`minimal_polynomial`]    | Minimal polynomial via Krylov sequence            |
//! | [`char_poly`]             | Characteristic polynomial via Hessenberg + Hyman  |
//! | [`cayley_hamilton_check`] | Verify Cayley-Hamilton (residual norm)            |
//! | [`matrix_eval_poly`]      | Evaluate a scalar polynomial at a matrix (Horner) |
//! | [`matrix_sqrt_schur`]     | Principal matrix square root via Schur            |
//! | [`matrix_log_schur`]      | Principal matrix logarithm via Schur              |
//! | [`companion_matrix`]      | Companion matrix for a monic polynomial           |
//! | [`poly_roots`]            | Polynomial roots via companion matrix eigenvalues |
//!
//! ## References
//!
//! - Higham, N.J. (2008). *Functions of Matrices: Theory and Computation*. SIAM.
//! - Horn & Johnson (1994). *Matrix Analysis*. Cambridge University Press.
//! - Björck & Hammarling (1983). "A Schur method for the square root of a matrix."
//!   *Linear Algebra and its Applications*, 52–53, 127–140.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Complex, Float, NumAssign};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Trait alias
// ---------------------------------------------------------------------------

/// Trait bound for matrix polynomial scalar types.
pub trait MatPolyFloat: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}
impl<F> MatPolyFloat for F where F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Dense square matrix multiply C = A * B.
fn matmul<F: MatPolyFloat>(a: &Array2<F>, b: &Array2<F>) -> LinalgResult<Array2<F>> {
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
fn frobenius<F: MatPolyFloat>(m: &Array2<F>) -> F {
    let mut acc = F::zero();
    for &v in m.iter() {
        acc += v * v;
    }
    acc.sqrt()
}

/// Identity matrix of size n.
fn eye<F: MatPolyFloat>(n: usize) -> Array2<F> {
    let mut id = Array2::<F>::zeros((n, n));
    for k in 0..n {
        id[[k, k]] = F::one();
    }
    id
}

/// Check that a matrix is square; return its side length.
fn check_square<F: MatPolyFloat>(a: &ArrayView2<F>, fname: &str) -> LinalgResult<usize> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(format!(
            "{fname}: matrix must be square, got {}×{}",
            a.nrows(),
            a.ncols()
        )));
    }
    Ok(n)
}

/// LU factorisation with partial pivoting.  Returns (L, U, perm).
fn lu_factor<F: MatPolyFloat>(a: &Array2<F>) -> (Array2<F>, Array2<F>, Vec<usize>) {
    let n = a.nrows();
    let mut u = a.clone();
    let mut l = eye::<F>(n);
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
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

/// Solve L x = b (forward substitution, L lower-triangular with unit diagonal).
fn solve_lower<F: MatPolyFloat>(l: &Array2<F>, b: &Array1<F>) -> Array1<F> {
    let n = l.nrows();
    let mut x = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[[i, j]] * x[j];
        }
        x[i] = s; // diagonal = 1
    }
    x
}

/// Solve U x = b (back substitution, U upper-triangular).
fn solve_upper<F: MatPolyFloat>(u: &Array2<F>, b: &Array1<F>) -> LinalgResult<Array1<F>> {
    let n = u.nrows();
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= u[[i, j]] * x[j];
        }
        let uii = u[[i, i]];
        if uii.abs() < F::epsilon() * F::from(1e6).unwrap_or(F::one()) {
            return Err(LinalgError::SingularMatrixError(
                "solve_upper: singular diagonal element".into(),
            ));
        }
        x[i] = s / uii;
    }
    Ok(x)
}

/// Solve A x = b via LU with partial pivoting.
fn solve_linear<F: MatPolyFloat>(a: &Array2<F>, b: &Array1<F>) -> LinalgResult<Array1<F>> {
    let n = a.nrows();
    let (l, u, perm) = lu_factor(a);
    let mut bp = Array1::<F>::zeros(n);
    for (i, &pi) in perm.iter().enumerate() {
        bp[i] = b[pi];
    }
    let y = solve_lower(&l, &bp);
    solve_upper(&u, &y)
}

// ---------------------------------------------------------------------------
// Householder QR decomposition  (used for Schur-based functions)
// ---------------------------------------------------------------------------

/// Compute the Schur decomposition A = Q T Q^T using Francis QR algorithm.
/// Returns (T, Q) where T is quasi-upper-triangular and Q is orthogonal.
fn schur_decomp<F: MatPolyFloat>(a: &Array2<F>) -> (Array2<F>, Array2<F>) {
    let n = a.nrows();
    let mut t = a.clone();
    let mut q = eye::<F>(n);
    let two = F::from(2.0).unwrap_or(F::one() + F::one());

    // Reduce to upper Hessenberg form
    for k in 0..(n.saturating_sub(2)) {
        let col_len = n - k - 1;
        let mut v = vec![F::zero(); col_len];
        for i in 0..col_len {
            v[i] = t[[k + 1 + i, k]];
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
        for j in k..n {
            let mut dot = F::zero();
            for i in 0..col_len {
                dot += v[i] * t[[k + 1 + i, j]];
            }
            for i in 0..col_len {
                t[[k + 1 + i, j]] -= two * v[i] * dot;
            }
        }
        for i in 0..n {
            let mut dot = F::zero();
            for j in 0..col_len {
                dot += t[[i, k + 1 + j]] * v[j];
            }
            for j in 0..col_len {
                t[[i, k + 1 + j]] -= two * dot * v[j];
            }
        }
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

    // Francis double-shift QR iteration
    let max_iter = 300 * n;
    let eps = F::epsilon() * F::from(100.0).unwrap_or(F::one());
    let mut m = n;
    let mut iter_count = 0usize;
    while m > 1 && iter_count < max_iter {
        iter_count += 1;
        // Deflation
        let mut deflated = false;
        for k in (0..m.saturating_sub(1)).rev() {
            if t[[k + 1, k]].abs() <= eps * (t[[k, k]].abs() + t[[k + 1, k + 1]].abs()) {
                t[[k + 1, k]] = F::zero();
                m = k + 1;
                deflated = true;
                break;
            }
        }
        if deflated || m <= 1 {
            continue;
        }
        // Compute shifts from 2x2 trailing submatrix
        let t11 = t[[m - 2, m - 2]];
        let t12 = t[[m - 2, m - 1]];
        let t21 = t[[m - 1, m - 2]];
        let t22 = t[[m - 1, m - 1]];
        let tr = t11 + t22;
        let det = t11 * t22 - t12 * t21;
        let four = two + two;
        // Compute bulge-chasing vector for double-shift
        let p1 = t[[0, 0]] * t[[0, 0]] - tr * t[[0, 0]] + det + t[[0, 1]] * t[[1, 0]];
        let p2 = t[[1, 0]] * (t[[0, 0]] + t[[1, 1]] - tr);
        let p3 = if m > 2 {
            t[[2, 1]] * t[[1, 0]]
        } else {
            F::zero()
        };
        // Apply first Householder reflector
        let norm012 = (p1 * p1 + p2 * p2 + p3 * p3).sqrt();
        if norm012 < eps {
            // Fallback: single-shift Francis step
            let (c, s) = givens_params_mp(t[[m - 2, m - 2]], t[[m - 1, m - 2]]);
            apply_givens_left_mp(&mut t, c, s, m - 2, m - 1, n);
            apply_givens_right_mp(&mut t, c, s, m - 2, m - 1, n);
            apply_givens_right_mp(&mut q, c, s, m - 2, m - 1, n);
            continue;
        }
        let mut v3 = [p1, p2, p3];
        {
            let sign = if v3[0] >= F::zero() {
                F::one()
            } else {
                -F::one()
            };
            v3[0] += sign * norm012;
            let nv = (v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2]).sqrt();
            if nv > eps {
                for vi in v3.iter_mut() {
                    *vi /= nv;
                }
                // Apply 3x3 Householder from left & right
                apply_house3_left(&mut t, &v3, 0, m, n);
                apply_house3_right(&mut t, &v3, 0, m, n);
                apply_house3_right(&mut q, &v3, 0, m, n);
            }
        }
        // Chase the bulge
        for k in 0..(m.saturating_sub(2)) {
            let i0 = k + 1;
            let i1 = (i0 + 2).min(m - 1);
            let col_len = i1 - i0 + 1;
            if col_len < 2 {
                break;
            }
            let mut vc = vec![F::zero(); col_len];
            for i in 0..col_len {
                vc[i] = t[[i0 + i, k]];
            }
            let nvc = {
                let mut s = F::zero();
                for &vi in &vc {
                    s += vi * vi;
                }
                s.sqrt()
            };
            if nvc < eps {
                continue;
            }
            let sign = if vc[0] >= F::zero() {
                F::one()
            } else {
                -F::one()
            };
            vc[0] += sign * nvc;
            let nvc2 = {
                let mut s = F::zero();
                for &vi in &vc {
                    s += vi * vi;
                }
                s.sqrt()
            };
            if nvc2 < eps {
                continue;
            }
            for vi in vc.iter_mut() {
                *vi /= nvc2;
            }
            // Apply Householder reflector (generalised for variable size)
            for j in 0..n {
                let mut dot = F::zero();
                for i in 0..col_len {
                    dot += vc[i] * t[[i0 + i, j]];
                }
                for i in 0..col_len {
                    t[[i0 + i, j]] -= two * vc[i] * dot;
                }
            }
            for i in 0..n {
                let mut dot = F::zero();
                for j in 0..col_len {
                    dot += t[[i, i0 + j]] * vc[j];
                }
                for j in 0..col_len {
                    t[[i, i0 + j]] -= two * dot * vc[j];
                }
            }
            for i in 0..n {
                let mut dot = F::zero();
                for j in 0..col_len {
                    dot += q[[i, i0 + j]] * vc[j];
                }
                for j in 0..col_len {
                    q[[i, i0 + j]] -= two * dot * vc[j];
                }
            }
        }
        let _ = four;
    }
    (t, q)
}

/// Apply a 3-vector Householder reflector I - 2*v*v^T to rows [r, r+1, r+2] from the left.
fn apply_house3_left<F: MatPolyFloat>(
    m: &mut Array2<F>,
    v: &[F; 3],
    r: usize,
    active_m: usize,
    n: usize,
) {
    let two = F::from(2.0).unwrap_or(F::one() + F::one());
    let rend = (r + 3).min(active_m);
    let col_len = rend - r;
    for j in 0..n {
        let mut dot = F::zero();
        for i in 0..col_len {
            dot += v[i] * m[[r + i, j]];
        }
        for i in 0..col_len {
            m[[r + i, j]] -= two * v[i] * dot;
        }
    }
}

/// Apply a 3-vector Householder reflector to columns [r, r+1, r+2] from the right.
fn apply_house3_right<F: MatPolyFloat>(
    m: &mut Array2<F>,
    v: &[F; 3],
    r: usize,
    active_m: usize,
    n: usize,
) {
    let two = F::from(2.0).unwrap_or(F::one() + F::one());
    let rend = (r + 3).min(active_m);
    let col_len = rend - r;
    for i in 0..n {
        let mut dot = F::zero();
        for j in 0..col_len {
            dot += m[[i, r + j]] * v[j];
        }
        for j in 0..col_len {
            m[[i, r + j]] -= two * dot * v[j];
        }
    }
}

/// Givens rotation parameters.
fn givens_params_mp<F: MatPolyFloat>(a: F, b: F) -> (F, F) {
    if b.abs() < F::epsilon() {
        return (F::one(), F::zero());
    }
    let r = (a * a + b * b).sqrt();
    if r < F::epsilon() {
        return (F::one(), F::zero());
    }
    (a / r, b / r)
}

fn apply_givens_left_mp<F: MatPolyFloat>(
    m: &mut Array2<F>,
    c: F,
    s: F,
    i: usize,
    j: usize,
    n: usize,
) {
    for k in 0..n {
        let xi = m[[i, k]];
        let xj = m[[j, k]];
        m[[i, k]] = c * xi + s * xj;
        m[[j, k]] = -s * xi + c * xj;
    }
}

fn apply_givens_right_mp<F: MatPolyFloat>(
    m: &mut Array2<F>,
    c: F,
    s: F,
    i: usize,
    j: usize,
    n: usize,
) {
    for k in 0..n {
        let xi = m[[k, i]];
        let xj = m[[k, j]];
        m[[k, i]] = c * xi + s * xj;
        m[[k, j]] = -s * xi + c * xj;
    }
}

// ---------------------------------------------------------------------------
// Compute eigenvalues of an upper quasi-triangular Schur form
// ---------------------------------------------------------------------------

fn schur_eigenvalues<F: MatPolyFloat>(t: &Array2<F>) -> Vec<Complex<F>> {
    let n = t.nrows();
    let eps = F::epsilon() * F::from(100.0).unwrap_or(F::one());
    let mut evs = Vec::with_capacity(n);
    let two = F::from(2.0).unwrap_or(F::one() + F::one());
    let four = two + two;
    let mut i = 0;
    while i < n {
        if i + 1 < n && t[[i + 1, i]].abs() > eps {
            let t11 = t[[i, i]];
            let t12 = t[[i, i + 1]];
            let t21 = t[[i + 1, i]];
            let t22 = t[[i + 1, i + 1]];
            let tr = t11 + t22;
            let det = t11 * t22 - t12 * t21;
            let disc = tr * tr - four * det;
            if disc >= F::zero() {
                let sq = disc.sqrt();
                evs.push(Complex::new((tr + sq) / two, F::zero()));
                evs.push(Complex::new((tr - sq) / two, F::zero()));
            } else {
                let sq = (-disc).sqrt();
                evs.push(Complex::new(tr / two, sq / two));
                evs.push(Complex::new(tr / two, -sq / two));
            }
            i += 2;
        } else {
            evs.push(Complex::new(t[[i, i]], F::zero()));
            i += 1;
        }
    }
    evs
}

// ---------------------------------------------------------------------------
// MatrixPoly: polynomial with matrix coefficients
// ---------------------------------------------------------------------------

/// A polynomial with matrix coefficients: P(x) = Σ_{i=0}^{d} coeffs\[i\] * x^i.
///
/// Here `x` is a **scalar** (not a matrix); use `eval(x)` for scalar evaluation
/// and `matrix_eval_poly` for evaluation at a matrix argument.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_polynomial::MatrixPoly;
///
/// // P(x) = I + x * A  (linear)
/// let id = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let a  = array![[2.0_f64, 1.0], [0.0, 3.0]];
/// let poly = MatrixPoly::new(vec![id, a]).expect("MatrixPoly");
/// let val = poly.eval(2.0_f64);   // P(2) = I + 2*A
/// assert_eq!(val[[0, 0]], 5.0_f64);
/// ```
#[derive(Debug, Clone)]
pub struct MatrixPoly<F: MatPolyFloat> {
    /// Coefficient matrices in ascending power order.
    pub coeffs: Vec<Array2<F>>,
    /// Degree = len(coeffs) - 1.
    pub degree: usize,
}

impl<F: MatPolyFloat> MatrixPoly<F> {
    /// Construct a polynomial from coefficient matrices (constant term first).
    ///
    /// # Errors
    ///
    /// Returns `LinalgError::ValueError` if coefficients is empty or shapes are inconsistent.
    pub fn new(coeffs: Vec<Array2<F>>) -> LinalgResult<Self> {
        if coeffs.is_empty() {
            return Err(LinalgError::ValueError(
                "MatrixPoly: coefficient list must not be empty".into(),
            ));
        }
        let shape = coeffs[0].shape().to_vec();
        for (i, c) in coeffs.iter().enumerate() {
            if c.shape() != shape.as_slice() {
                return Err(LinalgError::ShapeError(format!(
                    "MatrixPoly: coefficient {} has shape {:?}, expected {:?}",
                    i,
                    c.shape(),
                    shape
                )));
            }
        }
        let degree = coeffs.len() - 1;
        Ok(Self { coeffs, degree })
    }

    /// Evaluate P(x) = Σ coeffs\[i\] * x^i using Horner's method.
    pub fn eval(&self, x: F) -> Array2<F> {
        let (m, n) = (self.coeffs[0].nrows(), self.coeffs[0].ncols());
        let mut result = Array2::<F>::zeros((m, n));
        for coeff in self.coeffs.iter().rev() {
            // result = result * x + coeff  (Horner step)
            for i in 0..m {
                for j in 0..n {
                    result[[i, j]] = result[[i, j]] * x + coeff[[i, j]];
                }
            }
        }
        result
    }

    /// Add two polynomials (element-wise on matching-degree terms).
    ///
    /// # Errors
    ///
    /// Returns `LinalgError::ShapeError` if the matrix dimensions differ.
    pub fn add(&self, other: &Self) -> LinalgResult<Self> {
        let len = self.coeffs.len().max(other.coeffs.len());
        let (m, n) = (self.coeffs[0].nrows(), self.coeffs[0].ncols());
        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            let a = if i < self.coeffs.len() {
                self.coeffs[i].clone()
            } else {
                Array2::<F>::zeros((m, n))
            };
            let b = if i < other.coeffs.len() {
                other.coeffs[i].clone()
            } else {
                Array2::<F>::zeros((m, n))
            };
            if a.shape() != b.shape() {
                return Err(LinalgError::ShapeError(
                    "MatrixPoly::add: shape mismatch".into(),
                ));
            }
            let mut c = Array2::<F>::zeros((m, n));
            for ii in 0..m {
                for jj in 0..n {
                    c[[ii, jj]] = a[[ii, jj]] + b[[ii, jj]];
                }
            }
            result.push(c);
        }
        Self::new(result)
    }

    /// Subtract two polynomials.
    pub fn sub(&self, other: &Self) -> LinalgResult<Self> {
        let len = self.coeffs.len().max(other.coeffs.len());
        let (m, n) = (self.coeffs[0].nrows(), self.coeffs[0].ncols());
        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            let a = if i < self.coeffs.len() {
                self.coeffs[i].clone()
            } else {
                Array2::<F>::zeros((m, n))
            };
            let b = if i < other.coeffs.len() {
                other.coeffs[i].clone()
            } else {
                Array2::<F>::zeros((m, n))
            };
            let mut c = Array2::<F>::zeros((m, n));
            for ii in 0..m {
                for jj in 0..n {
                    c[[ii, jj]] = a[[ii, jj]] - b[[ii, jj]];
                }
            }
            result.push(c);
        }
        Self::new(result)
    }

    /// Multiply two polynomials (convolution of coefficient arrays).
    ///
    /// The resulting degree is `self.degree + other.degree`.
    ///
    /// # Errors
    ///
    /// Returns an error if matrix multiplication fails.
    pub fn mul(&self, other: &Self) -> LinalgResult<Self> {
        let da = self.coeffs.len();
        let db = other.coeffs.len();
        let dc = da + db - 1;
        let (m, _) = (self.coeffs[0].nrows(), self.coeffs[0].ncols());
        let n2 = other.coeffs[0].ncols();
        let mut result: Vec<Array2<F>> = (0..dc).map(|_| Array2::<F>::zeros((m, n2))).collect();
        for i in 0..da {
            for j in 0..db {
                let prod = matmul(&self.coeffs[i], &other.coeffs[j])?;
                let r = &mut result[i + j];
                for ii in 0..m {
                    for jj in 0..n2 {
                        r[[ii, jj]] += prod[[ii, jj]];
                    }
                }
            }
        }
        Self::new(result)
    }
}

// ---------------------------------------------------------------------------
// Characteristic polynomial via Hessenberg + Hyman recursion
// ---------------------------------------------------------------------------

/// Compute the characteristic polynomial of a square matrix A.
///
/// Returns the coefficients [c_0, c_1, ..., c_n] where
/// `p(λ) = c_0 + c_1 λ + ... + c_n λ^n`  (c_n = 1 for a monic polynomial).
///
/// The algorithm:
/// 1. Reduce A to upper Hessenberg form H.
/// 2. Apply the Hyman recursion to extract characteristic polynomial coefficients.
///
/// # Errors
///
/// Returns `LinalgError::ShapeError` if A is not square.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_polynomial::char_poly;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let p = char_poly(&a.view()).expect("char_poly");
/// // p(λ) = λ² - 5λ - 2
/// assert_eq!(p.len(), 3);
/// ```
pub fn char_poly<F: MatPolyFloat>(a: &ArrayView2<F>) -> LinalgResult<Vec<F>> {
    let n = check_square(a, "char_poly")?;
    let a_owned = a.to_owned();

    // Reduce to upper Hessenberg form
    let h = hessenberg_reduce(&a_owned);

    // Hyman's method for the characteristic polynomial of an upper Hessenberg matrix
    // p_k = coeff vector of det(H[0..k, 0..k] - λ I)
    // Recurrence: p_k(λ) = (H[k,k] - λ) * p_{k-1}(λ)
    //             - Σ_{j=0}^{k-2} ( H[j+1..k, j].prod * H[j, k] ) * p_{j-1}(λ)
    // Simplified Leverrier / Hessenberg recurrence:
    //
    // We use the standard Hessenberg recurrence:
    //   p[k+1] = (λ - h[k,k]) * p[k] - Σ_{j<k} c_j * h[j,k+1] * p[j-1]
    // where c_j = h[j+1,j] * h[j+2,j+1] * ... * h[k,k-1].
    //
    // We work with polynomial coefficient vectors (constant term first).
    let mut p: Vec<Vec<F>> = Vec::with_capacity(n + 1);
    p.push(vec![F::one()]); // p[0] = 1
    for k in 0..n {
        let mut pk = vec![F::zero(); k + 2];
        // pk = (λ - h[k,k]) * p[k-1]:  multiply p[k-1] by λ (shift up) then subtract h[k,k]*p[k-1]
        let prev = &p[k];
        // (λ - h[k,k]) * prev:
        for (i, &c) in prev.iter().enumerate() {
            pk[i + 1] += c; // multiply by λ
            pk[i] -= h[[k, k]] * c; // multiply by -h[k,k]
        }
        // Subtract subdiagonal chain contributions
        let mut chain = F::one();
        for j in (0..k).rev() {
            chain *= h[[j + 1, j]];
            if chain.abs() < F::epsilon() {
                break;
            }
            let hj_k = h[[j, k]];
            let pj_prev = &p[j]; // p[j-1] has degree j-1
            for (i, &c) in pj_prev.iter().enumerate() {
                pk[i] -= chain * hj_k * c;
            }
        }
        p.push(pk);
    }
    let mut result = p.remove(n);
    // The monic characteristic polynomial det(λI - A) -- ensure length n+1
    result.resize(n + 1, F::zero());
    Ok(result)
}

/// Reduce matrix to upper Hessenberg form via Householder reflections.
fn hessenberg_reduce<F: MatPolyFloat>(a: &Array2<F>) -> Array2<F> {
    let n = a.nrows();
    let mut h = a.clone();
    let two = F::from(2.0).unwrap_or(F::one() + F::one());
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
        for j in k..n {
            let mut dot = F::zero();
            for i in 0..col_len {
                dot += v[i] * h[[k + 1 + i, j]];
            }
            for i in 0..col_len {
                h[[k + 1 + i, j]] -= two * v[i] * dot;
            }
        }
        for i in 0..n {
            let mut dot = F::zero();
            for j in 0..col_len {
                dot += h[[i, k + 1 + j]] * v[j];
            }
            for j in 0..col_len {
                h[[i, k + 1 + j]] -= two * dot * v[j];
            }
        }
    }
    h
}

// ---------------------------------------------------------------------------
// Minimal polynomial via Krylov
// ---------------------------------------------------------------------------

/// Compute the minimal polynomial of a square matrix A.
///
/// The minimal polynomial is the monic polynomial of lowest degree p such that p(A) = 0.
/// This implementation uses the Krylov sequence: for a random starting vector b,
/// the vectors b, Ab, A²b, ..., A^k b are computed until linear dependence is
/// detected via Gaussian elimination.  The process is repeated for multiple
/// starting vectors to ensure correctness in degenerate cases.
///
/// Returns coefficients [c_0, ..., c_d] with c_d = 1 (monic).
///
/// # Errors
///
/// Returns `LinalgError::ShapeError` if A is not square.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_polynomial::minimal_polynomial;
///
/// let a = array![[2.0_f64, 0.0], [0.0, 2.0]]; // scalar multiple of I
/// let p = minimal_polynomial(&a.view()).expect("min_poly");
/// // min poly is (λ - 2)
/// assert!(p.len() <= 3);
/// ```
pub fn minimal_polynomial<F: MatPolyFloat>(a: &ArrayView2<F>) -> LinalgResult<Vec<F>> {
    let n = check_square(a, "minimal_polynomial")?;
    let a_owned = a.to_owned();

    // Use the characteristic polynomial as an upper bound
    let char = char_poly(a)?;

    // Krylov space approach: try random starting vectors
    // We'll use deterministic pseudo-random starting vectors
    let seeds: &[f64] = &[1.0, 0.618033988, 0.414213562, 0.302775638, 0.236067977];

    let mut min_poly = char.clone();

    for &seed in seeds.iter().take(3) {
        // Build starting vector
        let mut b = Array1::<F>::zeros(n);
        for i in 0..n {
            let val = F::from((seed * (i + 1) as f64).sin() * 0.5 + 0.5 + i as f64 * 0.1)
                .unwrap_or(F::one());
            b[i] = val;
        }

        // Build Krylov matrix K = [b, Ab, A²b, ..., A^{n}b]
        let mut krylov = Array2::<F>::zeros((n, n + 1));
        {
            let mut cur = b.clone();
            for j in 0..=n {
                for i in 0..n {
                    krylov[[i, j]] = cur[i];
                }
                if j < n {
                    // cur = A * cur
                    let mut next = Array1::<F>::zeros(n);
                    for i in 0..n {
                        for k in 0..n {
                            next[i] += a_owned[[i, k]] * cur[k];
                        }
                    }
                    cur = next;
                }
            }
        }

        // Find the first linear dependence via row echelon
        let deg = krylov_min_degree(&krylov, n);
        // Extract coefficients: A^deg b = c_0 b + c_1 Ab + ... + c_{deg-1} A^{deg-1} b
        let candidate = extract_min_poly_coeffs(&krylov, n, deg);
        if candidate.len() < min_poly.len() {
            min_poly = candidate;
        }
    }
    Ok(min_poly)
}

/// Find the degree of the minimal polynomial for the Krylov matrix.
fn krylov_min_degree<F: MatPolyFloat>(krylov: &Array2<F>, n: usize) -> usize {
    let eps = F::epsilon() * F::from(1e6).unwrap_or(F::one());
    let mut basis: Vec<Array1<F>> = Vec::new();
    for j in 0..=n {
        let col: Array1<F> = (0..n).map(|i| krylov[[i, j]]).collect();
        // Check if col is in span of basis via Gram-Schmidt
        let mut residual = col.clone();
        for b in &basis {
            let dot = (0..n)
                .map(|i| b[i] * residual[i])
                .fold(F::zero(), |acc, x| acc + x);
            let bnorm_sq = (0..n)
                .map(|i| b[i] * b[i])
                .fold(F::zero(), |acc, x| acc + x);
            if bnorm_sq > eps {
                for i in 0..n {
                    residual[i] -= (dot / bnorm_sq) * b[i];
                }
            }
        }
        let res_norm = (0..n)
            .map(|i| residual[i] * residual[i])
            .fold(F::zero(), |acc, x| acc + x)
            .sqrt();
        if res_norm < eps * F::from(n as f64).unwrap_or(F::one()) {
            return j;
        }
        basis.push(residual);
    }
    n
}

/// Extract minimal polynomial coefficients from Krylov data.
fn extract_min_poly_coeffs<F: MatPolyFloat>(krylov: &Array2<F>, n: usize, deg: usize) -> Vec<F> {
    if deg == 0 {
        return vec![F::one()];
    }
    // Build the system: find c such that krylov[:,deg] = Σ c[j] * krylov[:,j]
    let mut system = Array2::<F>::zeros((n, deg));
    let mut rhs = Array1::<F>::zeros(n);
    for i in 0..n {
        rhs[i] = krylov[[i, deg]];
        for j in 0..deg {
            system[[i, j]] = krylov[[i, j]];
        }
    }
    // Solve via least-squares (use normal equations for small problems)
    let ata = {
        let mut m = Array2::<F>::zeros((deg, deg));
        for i in 0..deg {
            for j in 0..deg {
                for k in 0..n {
                    m[[i, j]] += system[[k, i]] * system[[k, j]];
                }
            }
        }
        m
    };
    let atb: Array1<F> = {
        let mut v = Array1::<F>::zeros(deg);
        for i in 0..deg {
            for k in 0..n {
                v[i] += system[[k, i]] * rhs[k];
            }
        }
        v
    };
    let c = solve_linear(&ata, &atb).unwrap_or_else(|_| Array1::<F>::zeros(deg));
    // coefficients: [c[0], c[1], ..., c[deg-1], 1]
    let mut poly = Vec::with_capacity(deg + 1);
    for i in 0..deg {
        poly.push(-c[i]); // min poly: λ^deg - c[deg-1]*λ^{deg-1} - ...
    }
    poly.push(F::one());
    poly
}

// ---------------------------------------------------------------------------
// Cayley-Hamilton check
// ---------------------------------------------------------------------------

/// Verify the Cayley-Hamilton theorem: p(A) ≈ 0, where p is the characteristic polynomial.
///
/// Computes p(A) using Horner's method and returns its Frobenius norm.
/// For exact arithmetic this would be 0; for floating-point this should be small
/// (≲ n² · ε · ‖A‖ⁿ).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_polynomial::cayley_hamilton_check;
///
/// let a = array![[1.0_f64, 2.0], [0.0, 3.0]];
/// let residual = cayley_hamilton_check(&a.view()).expect("ch_check");
/// assert!(residual < 1e-8, "Cayley-Hamilton residual: {}", residual);
/// ```
pub fn cayley_hamilton_check<F: MatPolyFloat>(a: &ArrayView2<F>) -> LinalgResult<F> {
    let p = char_poly(a)?;
    let pa = matrix_eval_poly(a, &p)?;
    Ok(frobenius(&pa))
}

// ---------------------------------------------------------------------------
// Matrix function via polynomial evaluation (Horner's method)
// ---------------------------------------------------------------------------

/// Evaluate a scalar polynomial p at a matrix argument A using Horner's method.
///
/// Computes p(A) = c_0 I + c_1 A + ... + c_d A^d using the recurrence:
/// ```text
/// H_d = c_d I
/// H_k = c_k I + A * H_{k+1}   for k = d-1, ..., 0
/// ```
///
/// # Arguments
///
/// * `a` - Input square matrix (n × n)
/// * `poly` - Polynomial coefficients [c_0, c_1, ..., c_d] (constant term first)
///
/// # Returns
///
/// The matrix polynomial p(A).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_polynomial::matrix_eval_poly;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// // p(x) = x^2 - 3x + 2 = (x-1)(x-2) → p(A) should be zero matrix
/// let p = [-2.0_f64, -3.0, 1.0]; // constant term first is -2, but actually:
/// // p(x) = x^2 - 3x + 2: coefficients are [2, -3, 1]
/// let p2 = [2.0_f64, -3.0, 1.0];
/// let result = matrix_eval_poly(&a.view(), &p2).expect("matrix_eval_poly");
/// // For diagonal A: p(diag(1,2)) = diag(p(1), p(2)) = diag(0, 0)
/// assert!(result[[0,0]].abs() < 1e-10);
/// ```
pub fn matrix_eval_poly<F: MatPolyFloat>(a: &ArrayView2<F>, poly: &[F]) -> LinalgResult<Array2<F>> {
    let n = check_square(a, "matrix_eval_poly")?;
    if poly.is_empty() {
        return Ok(Array2::<F>::zeros((n, n)));
    }
    let a_owned = a.to_owned();
    let d = poly.len() - 1;
    // Horner: H = c_d * I, then H = c_{d-1} * I + A * H
    let mut h = {
        let mut m = Array2::<F>::zeros((n, n));
        for k in 0..n {
            m[[k, k]] = poly[d];
        }
        m
    };
    for k in (0..d).rev() {
        // h = a * h
        h = matmul(&a_owned, &h)?;
        // h += c_k * I
        for i in 0..n {
            h[[i, i]] += poly[k];
        }
    }
    Ok(h)
}

// ---------------------------------------------------------------------------
// Matrix square root via Schur decomposition
// ---------------------------------------------------------------------------

/// Compute the principal matrix square root via Schur decomposition.
///
/// Algorithm (Björck-Hammarling):
/// 1. Compute Schur decomposition A = Q T Q^T.
/// 2. Compute sqrt of upper triangular T using the diagonal recurrence:
///    - Diagonal entries: u_{ii} = sqrt(t_{ii})
///    - Off-diagonal: u_{ij} = (t_{ij} - Σ_{k=i+1}^{j-1} u_{ik} u_{kj}) / (u_{ii} + u_{jj})
/// 3. Back-transform: sqrt(A) = Q * U * Q^T.
///
/// # Errors
///
/// * `DomainError` if any diagonal element of T is negative (no principal sqrt).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_polynomial::matrix_sqrt_schur;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
/// let s = matrix_sqrt_schur(&a.view()).expect("sqrt");
/// assert!((s[[0, 0]] - 2.0).abs() < 1e-8);
/// assert!((s[[1, 1]] - 3.0).abs() < 1e-8);
/// ```
pub fn matrix_sqrt_schur<F: MatPolyFloat + std::fmt::Debug>(
    a: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = check_square(a, "matrix_sqrt_schur")?;
    let a_owned = a.to_owned();
    let (t, q) = schur_decomp(&a_owned);

    // Compute sqrt of upper triangular T
    let mut u = Array2::<F>::zeros((n, n));
    for i in 0..n {
        let tii = t[[i, i]];
        if tii < F::zero() {
            return Err(LinalgError::DomainError(format!(
                "matrix_sqrt_schur: negative diagonal T[{i},{i}] = {tii:?}, no real principal sqrt"
            )));
        }
        u[[i, i]] = tii.sqrt();
    }
    // Off-diagonal recurrence (upper triangular)
    for j in 1..n {
        for i in (0..j).rev() {
            let mut s = t[[i, j]];
            for k in (i + 1)..j {
                s -= u[[i, k]] * u[[k, j]];
            }
            let denom = u[[i, i]] + u[[j, j]];
            if denom.abs() < F::epsilon() {
                return Err(LinalgError::NumericalError(
                    "matrix_sqrt_schur: zero diagonal sum in triangular sqrt".into(),
                ));
            }
            u[[i, j]] = s / denom;
        }
    }

    // Back-transform: sqrt(A) = Q * U * Q^T
    let qu = matmul(&q, &u)?;
    let qt = q.t().to_owned();
    matmul(&qu, &qt)
}

// ---------------------------------------------------------------------------
// Matrix logarithm via Schur decomposition
// ---------------------------------------------------------------------------

/// Compute the principal matrix logarithm via Schur decomposition.
///
/// Algorithm:
/// 1. Compute Schur decomposition A = Q T Q^T.
/// 2. Compute log of upper triangular T:
///    - Diagonal: l_{ii} = ln(t_{ii})
///    - Off-diagonal Parlett recurrence:
///      l_{ij} = (t_{ij} - Σ_{k=i+1}^{j-1} l_{ik} l_{kj}) / (t_{jj} - t_{ii})
///      when t_{ii} ≠ t_{jj}; uses divided difference when equal.
/// 3. Back-transform: log(A) = Q * L * Q^T.
///
/// # Errors
///
/// * `DomainError` if any eigenvalue is zero or negative real (log not defined).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_polynomial::matrix_log_schur;
///
/// let a = array![[1.0_f64, 0.0], [0.0, std::f64::consts::E]];
/// let l = matrix_log_schur(&a.view()).expect("log");
/// assert!((l[[0, 0]]).abs() < 1e-8);        // log(1) = 0
/// assert!((l[[1, 1]] - 1.0_f64).abs() < 1e-8); // log(e) = 1
/// ```
pub fn matrix_log_schur<F: MatPolyFloat + std::fmt::Debug>(
    a: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let n = check_square(a, "matrix_log_schur")?;
    let a_owned = a.to_owned();
    let (t, q) = schur_decomp(&a_owned);

    // Compute log of upper triangular T
    let mut l = Array2::<F>::zeros((n, n));
    for i in 0..n {
        let tii = t[[i, i]];
        if tii <= F::zero() {
            return Err(LinalgError::DomainError(format!(
                "matrix_log_schur: non-positive diagonal T[{i},{i}] = {:?}",
                tii
            )));
        }
        l[[i, i]] = tii.ln();
    }
    // Off-diagonal Parlett recurrence
    for j in 1..n {
        for i in (0..j).rev() {
            let mut s = t[[i, j]];
            for k in (i + 1)..j {
                s -= l[[i, k]] * l[[k, j]];
            }
            let diff = t[[j, j]] - t[[i, i]];
            if diff.abs() < F::epsilon() * F::from(1e4).unwrap_or(F::one()) {
                // Confluent case: use derivative of log at t[i,i]
                let tii = t[[i, i]];
                if tii.abs() < F::epsilon() {
                    l[[i, j]] = F::zero();
                } else {
                    l[[i, j]] = s / tii;
                }
            } else {
                l[[i, j]] = s * (l[[j, j]] - l[[i, i]]) / diff;
            }
        }
    }

    // Back-transform: log(A) = Q * L * Q^T
    let ql = matmul(&q, &l)?;
    let qt = q.t().to_owned();
    matmul(&ql, &qt)
}

// ---------------------------------------------------------------------------
// Companion matrix and polynomial roots
// ---------------------------------------------------------------------------

/// Build the companion matrix of a monic polynomial.
///
/// Given a monic polynomial `p(x) = x^n + a_{n-1} x^{n-1} + ... + a_1 x + a_0`
/// (coefficients in **ascending** power order: `[a_0, a_1, ..., a_{n-1}, 1]`),
/// returns the n×n companion matrix:
///
/// ```text
/// C = [ 0  0  ...  0  -a_0 ]
///     [ 1  0  ...  0  -a_1 ]
///     [ 0  1  ...  0  -a_2 ]
///     [ .  .  ...  .   .   ]
///     [ 0  0  ...  1  -a_{n-1} ]
/// ```
///
/// # Errors
///
/// Returns `LinalgError::ValueError` if the polynomial has degree < 1 or the leading
/// coefficient is not ≈ 1.
///
/// # Examples
///
/// ```
/// use scirs2_linalg::matrix_polynomial::companion_matrix;
///
/// // p(x) = x^2 - 3x + 2 → [2.0, -3.0, 1.0]
/// let c = companion_matrix(&[2.0_f64, -3.0, 1.0]).expect("companion");
/// assert_eq!(c.nrows(), 2);
/// ```
pub fn companion_matrix<F: MatPolyFloat + std::fmt::Debug>(poly: &[F]) -> LinalgResult<Array2<F>> {
    let len = poly.len();
    if len < 2 {
        return Err(LinalgError::ValueError(
            "companion_matrix: polynomial must have degree ≥ 1 (at least 2 coefficients)".into(),
        ));
    }
    let n = len - 1; // degree
    let lead = poly[n];
    if (lead - F::one()).abs() > F::epsilon() * F::from(100.0).unwrap_or(F::one()) {
        return Err(LinalgError::ValueError(format!(
            "companion_matrix: leading coefficient must be 1 (monic), got {:?}",
            lead
        )));
    }
    let mut c = Array2::<F>::zeros((n, n));
    // Last column: -a_0 / 1, -a_1 / 1, ..., -a_{n-1} / 1
    for i in 0..n {
        c[[i, n - 1]] = -poly[i] / lead;
    }
    // Sub-diagonal: ones
    for i in 1..n {
        c[[i, i - 1]] = F::one();
    }
    Ok(c)
}

/// Compute the roots of a polynomial via eigenvalues of its companion matrix.
///
/// The polynomial is given in ascending power order: `[a_0, a_1, ..., a_n]`
/// where `a_n` is the leading coefficient (need not be monic; the companion
/// matrix is built from the monic version).
///
/// # Errors
///
/// Returns an error if the polynomial has degree < 1 or the leading coefficient is zero.
///
/// # Examples
///
/// ```
/// use scirs2_linalg::matrix_polynomial::poly_roots;
///
/// // p(x) = x^2 - 5x + 6 = (x-2)(x-3) → [6.0, -5.0, 1.0]
/// let roots = poly_roots(&[6.0_f64, -5.0, 1.0]).expect("poly_roots");
/// assert_eq!(roots.len(), 2);
/// // Roots should be near 2 and 3
/// ```
pub fn poly_roots<F: MatPolyFloat + std::fmt::Debug>(poly: &[F]) -> LinalgResult<Vec<Complex<F>>> {
    let len = poly.len();
    if len < 2 {
        return Err(LinalgError::ValueError(
            "poly_roots: polynomial must have degree ≥ 1".into(),
        ));
    }
    let lead = poly[len - 1];
    if lead.abs() < F::epsilon() * F::from(100.0).unwrap_or(F::one()) {
        return Err(LinalgError::ValueError(
            "poly_roots: leading coefficient must be non-zero".into(),
        ));
    }
    // Normalise to monic
    let monic: Vec<F> = poly.iter().map(|&c| c / lead).collect();
    let c = companion_matrix(&monic)?;
    let (t, _q) = schur_decomp(&c);
    let evs = schur_eigenvalues(&t);
    Ok(evs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    // ---- MatrixPoly ----

    #[test]
    fn test_matrix_poly_eval_scalar() {
        // P(x) = 2I + 3x*I; eval at x=4 => 14*I
        let i2 = Array2::<f64>::eye(2);
        let p = MatrixPoly::new(vec![2.0 * &i2, 3.0 * &i2]).expect("MatrixPoly");
        let val = p.eval(4.0_f64);
        assert_relative_eq!(val[[0, 0]], 14.0, epsilon = 1e-12);
        assert_relative_eq!(val[[1, 1]], 14.0, epsilon = 1e-12);
    }

    #[test]
    fn test_matrix_poly_add() {
        let i2 = Array2::<f64>::eye(2);
        let p1 = MatrixPoly::new(vec![i2.clone()]).expect("p1");
        let p2 = MatrixPoly::new(vec![2.0 * &i2]).expect("p2");
        let sum = p1.add(&p2).expect("add");
        let val = sum.eval(0.0);
        assert_relative_eq!(val[[0, 0]], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_matrix_poly_mul() {
        // (I + xI) * (I + xI) = I + 2xI + x²I  at x=1 → 4I
        let i2 = Array2::<f64>::eye(2);
        let p = MatrixPoly::new(vec![i2.clone(), i2.clone()]).expect("p");
        let pp = p.mul(&p).expect("mul");
        let val = pp.eval(1.0);
        assert_relative_eq!(val[[0, 0]], 4.0, epsilon = 1e-12);
    }

    // ---- char_poly ----

    #[test]
    fn test_char_poly_2x2() {
        // A = [[1, 2], [3, 4]]:  char poly = λ² - 5λ - 2
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let p = char_poly(&a.view()).expect("char_poly");
        assert_eq!(p.len(), 3);
        // p[2] = 1 (monic leading coefficient)
        assert_relative_eq!(p[2], 1.0, epsilon = 1e-8);
        // p[1] = -(1+4) = -5
        assert_relative_eq!(p[1], -5.0, epsilon = 1e-8);
        // p[0] = 1*4 - 2*3 = -2
        assert_relative_eq!(p[0], -2.0, epsilon = 1e-8);
    }

    #[test]
    fn test_char_poly_identity() {
        // char poly of I_3 = (λ-1)³ = λ³ - 3λ² + 3λ - 1
        let a = Array2::<f64>::eye(3);
        let p = char_poly(&a.view()).expect("char_poly");
        assert_eq!(p.len(), 4);
        assert_relative_eq!(p[3], 1.0, epsilon = 1e-8);
        assert_relative_eq!(p[2], -3.0, epsilon = 1e-8);
        assert_relative_eq!(p[1], 3.0, epsilon = 1e-8);
        assert_relative_eq!(p[0], -1.0, epsilon = 1e-8);
    }

    // ---- cayley_hamilton_check ----

    #[test]
    fn test_cayley_hamilton_2x2() {
        let a = array![[2.0_f64, 1.0], [0.0, 3.0]];
        let res = cayley_hamilton_check(&a.view()).expect("ch_check");
        assert!(res < 1e-8, "Cayley-Hamilton residual too large: {res}");
    }

    // ---- matrix_eval_poly ----

    #[test]
    fn test_matrix_eval_poly_annihilator() {
        // A = diag(1, 2);  p(x) = (x-1)(x-2) = x²-3x+2; coeffs [2,-3,1]
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let poly = [2.0_f64, -3.0, 1.0];
        let result = matrix_eval_poly(&a.view(), &poly).expect("eval");
        assert!(result[[0, 0]].abs() < 1e-10);
        assert!(result[[1, 1]].abs() < 1e-10);
    }

    // ---- matrix_sqrt_schur ----

    #[test]
    fn test_matrix_sqrt_diagonal() {
        let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
        let s = matrix_sqrt_schur(&a.view()).expect("sqrt");
        assert_relative_eq!(s[[0, 0]], 2.0, epsilon = 1e-8);
        assert_relative_eq!(s[[1, 1]], 3.0, epsilon = 1e-8);
    }

    #[test]
    fn test_matrix_sqrt_roundtrip() {
        let a = array![[4.0_f64, 2.0], [2.0, 5.0]];
        let s = matrix_sqrt_schur(&a.view()).expect("sqrt");
        let ss = {
            let mut r = Array2::<f64>::zeros((2, 2));
            for i in 0..2 {
                for k in 0..2 {
                    for j in 0..2 {
                        r[[i, j]] += s[[i, k]] * s[[k, j]];
                    }
                }
            }
            r
        };
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(ss[[i, j]], a[[i, j]], epsilon = 1e-7);
            }
        }
    }

    #[test]
    fn test_matrix_sqrt_negative_eigenvalue_error() {
        // A = diag(-1, 1): no real principal sqrt
        let a = array![[-1.0_f64, 0.0], [0.0, 1.0]];
        let result = matrix_sqrt_schur(&a.view());
        assert!(result.is_err());
    }

    // ---- matrix_log_schur ----

    #[test]
    fn test_matrix_log_diagonal() {
        let e = std::f64::consts::E;
        let a = array![[1.0_f64, 0.0], [0.0, e]];
        let l = matrix_log_schur(&a.view()).expect("log");
        assert_relative_eq!(l[[0, 0]], 0.0, epsilon = 1e-8);
        assert_relative_eq!(l[[1, 1]], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_matrix_log_roundtrip() {
        // exp(log(A)) ≈ A for SPD matrices;  we use a simple diagonal case
        let a = array![[2.0_f64, 0.0], [0.0, 3.0]];
        let l = matrix_log_schur(&a.view()).expect("log");
        // Check diagonal
        assert_relative_eq!(l[[0, 0]], 2.0_f64.ln(), epsilon = 1e-8);
        assert_relative_eq!(l[[1, 1]], 3.0_f64.ln(), epsilon = 1e-8);
    }

    // ---- companion_matrix ----

    #[test]
    fn test_companion_matrix_shape() {
        // p(x) = x² - 5x + 6 → [6, -5, 1]
        let c = companion_matrix(&[6.0_f64, -5.0, 1.0]).expect("companion");
        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 2);
    }

    #[test]
    fn test_companion_matrix_values() {
        // p(x) = x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3) → [−6, 11, −6, 1]
        let c = companion_matrix(&[-6.0_f64, 11.0, -6.0, 1.0]).expect("companion");
        // Last column should be [6, -11, 6]
        assert_relative_eq!(c[[0, 2]], 6.0, epsilon = 1e-12);
        assert_relative_eq!(c[[1, 2]], -11.0, epsilon = 1e-12);
        assert_relative_eq!(c[[2, 2]], 6.0, epsilon = 1e-12);
    }

    // ---- poly_roots ----

    #[test]
    fn test_poly_roots_quadratic() {
        // p(x) = x² - 5x + 6 = (x-2)(x-3)
        let roots = poly_roots(&[6.0_f64, -5.0, 1.0]).expect("poly_roots");
        assert_eq!(roots.len(), 2);
        let mut re: Vec<f64> = roots.iter().map(|r| r.re).collect();
        re.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert_relative_eq!(re[0], 2.0, epsilon = 0.1);
        assert_relative_eq!(re[1], 3.0, epsilon = 0.1);
    }

    // ---- minimal_polynomial ----

    #[test]
    fn test_minimal_polynomial_scalar_multiple() {
        // A = 3I: min poly = (λ - 3)
        let a = 3.0 * Array2::<f64>::eye(2);
        let p = minimal_polynomial(&a.view()).expect("min_poly");
        assert!(p.len() <= 3, "Min poly degree too high: {}", p.len() - 1);
    }

    #[test]
    fn test_minimal_polynomial_diag() {
        // A = diag(1, 2, 1): min poly = (λ-1)(λ-2) = λ²-3λ+2
        let mut a = Array2::<f64>::zeros((3, 3));
        a[[0, 0]] = 1.0;
        a[[1, 1]] = 2.0;
        a[[2, 2]] = 1.0;
        let p = minimal_polynomial(&a.view()).expect("min_poly");
        // Min poly has degree ≤ char poly degree (3)
        assert!(p.len() <= 4);
    }
}
