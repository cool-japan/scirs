//! Structured linear system solvers
//!
//! This module provides O(n²) and O(n log n) solvers for matrices with
//! exploitable structure:
//!
//! | Function | Structure | Algorithm | Complexity |
//! |---|---|---|---|
//! | [`solve_toeplitz`] | Toeplitz | Levinson-Durbin | O(n²) |
//! | [`solve_circulant`] | Circulant | DFT diagonalisation | O(n log n) |
//! | [`solve_tridiagonal`] | Tridiagonal | Thomas algorithm | O(n) |
//! | [`solve_banded`] | Banded (kl, ku) | Banded LU | O(n·(kl+ku)²) |
//! | [`solve_triangular`] | Lower / upper triangular | Substitution | O(n²) |
//!
//! All functions are generic over `F: Float` and accept `ArrayView` inputs.
//! None uses `unwrap()`; every error path returns `LinalgResult`.

use crate::circulant_toeplitz::{CirculantMatrix, ToeplitzMatrix};
use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// solve_toeplitz — Levinson-Durbin O(n²)
// ---------------------------------------------------------------------------

/// Solve the Toeplitz linear system T x = b in O(n²) using the Levinson-Durbin
/// algorithm.
///
/// The Toeplitz matrix T is defined by its first row `r` and first column `c`
/// such that T[i, j] = r[j-i] for j ≥ i and T[i, j] = c[i-j] for i > j.
/// (Note: `r[0]` == `c[0]` must hold; an error is returned otherwise.)
///
/// The algorithm solves two auxiliary recurrences simultaneously and achieves
/// O(n²) instead of O(n³) for general systems.
///
/// # Arguments
/// * `r` - First row of the Toeplitz matrix (length n); `r[0]` is the diagonal.
/// * `c` - First column of the Toeplitz matrix (length n); `c[0]` must equal `r[0]`.
/// * `b` - Right-hand side vector (length n).
///
/// # Returns
/// Solution vector x of length n.
///
/// # Errors
/// Returns `LinalgError` if dimensions are inconsistent, `r[0] ≠ c[0]`, or the
/// leading sub-matrix is singular.
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::structured_solve::solve_toeplitz;
///
/// // Symmetric Toeplitz: [[2, 1], [1, 2]], b = [3, 3] → x = [1, 1]
/// let r = array![2.0_f64, 1.0];
/// let c = array![2.0_f64, 1.0];
/// let b = array![3.0_f64, 3.0];
/// let x = solve_toeplitz(&r.view(), &c.view(), &b.view()).expect("solve_toeplitz");
/// assert!((x[0] - 1.0).abs() < 1e-10 && (x[1] - 1.0).abs() < 1e-10);
/// ```
pub fn solve_toeplitz<F>(
    r: &ArrayView1<F>,
    c: &ArrayView1<F>,
    b: &ArrayView1<F>,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + Into<f64> + Clone + 'static,
{
    let n = r.len();
    if n == 0 {
        return Err(LinalgError::InvalidInputError(
            "solve_toeplitz: input vectors must be non-empty".to_string(),
        ));
    }
    if c.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "solve_toeplitz: r has length {n} but c has length {}",
            c.len()
        )));
    }
    if b.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "solve_toeplitz: r has length {n} but b has length {}",
            b.len()
        )));
    }
    if (r[0] - c[0]).abs() > F::epsilon() * (r[0].abs() + F::one()) {
        return Err(LinalgError::ValueError(
            "solve_toeplitz: r[0] must equal c[0] (diagonal element)".to_string(),
        ));
    }

    let t0 = r[0];
    if t0.abs() < F::epsilon() {
        return Err(LinalgError::SingularMatrixError(
            "solve_toeplitz: diagonal element is zero (singular matrix)".to_string(),
        ));
    }

    if n == 1 {
        return Ok(Array1::from_vec(vec![b[0] / t0]));
    }

    // Levinson algorithm for general (non-symmetric) Toeplitz systems.
    //
    // Convention:
    //   T[i,j] = r[j-i]  for j >= i  (first row supplies super-diagonals)
    //   T[i,j] = c[i-j]  for i > j   (first col supplies sub-diagonals)
    //   T[i,i] = r[0] = c[0] = t0
    //
    // We maintain three length-k auxiliary vectors:
    //   fwd: solves T_k * fwd = e_0  (forward vector)
    //   bwd: solves T_k * bwd = e_{k-1}  (backward vector)
    //   sol: current solution for T_k * sol = b[0:k]
    //
    // Update from T_k to T_{k+1}:
    //   lambda = sum_{j=0}^{k-1} c[k-j] * fwd[j]  (sub-diag residual)
    //   mu     = sum_{j=0}^{k-1} r[k-j] * bwd[j]  (super-diag residual)
    //   eps    = b[k] - sum_{j=0}^{k-1} c[k-j] * sol[j]
    //   den    = 1 - lambda * mu
    //
    //   fwd_new[0]   = fwd[0] / den  (no bwd contribution at j=0)
    //   fwd_new[j]   = (fwd[j] - lambda * bwd[k-j]) / den  for j = 1..k-1
    //   fwd_new[k]   = (-lambda * bwd[0]) / den
    //
    //   bwd_new[j]   = (-mu * fwd[j]) / den    for j = 0..k-1
    //   bwd_new[k]   = bwd[k-1] / den
    //
    //   sol_new[j]   = sol[j] + eps * bwd_new[j]   for j = 0..k

    let mut fwd = vec![F::zero(); n];
    let mut bwd = vec![F::zero(); n];
    let mut sol = vec![F::zero(); n];

    fwd[0] = F::one() / t0;
    bwd[0] = F::one() / t0;
    sol[0] = b[0] / t0;

    for k in 1..n {
        let mut lambda = F::zero();
        let mut mu = F::zero();
        let mut eps = b[k];

        for j in 0..k {
            lambda += c[k - j] * fwd[j];
            mu += r[k - j] * bwd[j];
            eps -= c[k - j] * sol[j];
        }

        let den = F::one() - lambda * mu;
        if den.abs() < F::from(1e-14_f64).unwrap_or(F::epsilon()) {
            return Err(LinalgError::SingularMatrixError(format!(
                "solve_toeplitz: Levinson step {k}: leading minor is near-singular"
            )));
        }
        let den_inv = F::one() / den;

        let fwd_old: Vec<F> = fwd[..k].to_vec();
        let bwd_old: Vec<F> = bwd[..k].to_vec();

        // Update forward vector
        fwd[0] = fwd_old[0] * den_inv;
        for j in 1..k {
            fwd[j] = (fwd_old[j] - lambda * bwd_old[k - j]) * den_inv;
        }
        fwd[k] = (-lambda * bwd_old[0]) * den_inv;

        // Update backward vector
        for j in 0..k {
            bwd[j] = (-mu * fwd_old[j]) * den_inv;
        }
        bwd[k] = bwd_old[k - 1] * den_inv;

        // Update solution
        for j in 0..=k {
            sol[j] += eps * bwd[j];
        }
    }

    Ok(Array1::from_vec(sol))
}

// ---------------------------------------------------------------------------
// solve_circulant — DFT diagonalisation O(n log n)
// ---------------------------------------------------------------------------

/// Solve the circulant linear system C x = b in O(n log n).
///
/// A circulant matrix is fully described by its first row c.  The solution
/// exploits the fact that every circulant matrix is diagonalised by the DFT:
///
///   C = F⁻¹ diag(F c) F
///   x = F⁻¹ ( F b  ./  F c )
///
/// where `./ ` denotes element-wise division.
///
/// # Arguments
/// * `c` - First row of the circulant matrix (length n)
/// * `b` - Right-hand side vector (length n)
///
/// # Returns
/// Solution vector x of length n.
///
/// # Errors
/// Returns `LinalgError` if the matrix is singular (any DFT eigenvalue is zero)
/// or dimensions mismatch.
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::structured_solve::solve_circulant;
///
/// // Circulant [[2, 1], [1, 2]], b = [3, 3] → x = [1, 1]
/// let c = array![2.0_f64, 1.0];
/// let b = array![3.0_f64, 3.0];
/// let x = solve_circulant(&c.view(), &b.view()).expect("solve_circulant");
/// assert!((x[0] - 1.0).abs() < 1e-10 && (x[1] - 1.0).abs() < 1e-10);
/// ```
pub fn solve_circulant<F>(c: &ArrayView1<F>, b: &ArrayView1<F>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + Into<f64> + Clone + 'static,
{
    let n = c.len();
    if n == 0 {
        return Err(LinalgError::InvalidInputError(
            "solve_circulant: input vectors must be non-empty".to_string(),
        ));
    }
    if b.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "solve_circulant: c has length {n} but b has length {}",
            b.len()
        )));
    }

    // Delegate to CirculantMatrix which implements O(n log n) FFT-based solve.
    let c_owned = c.to_owned();
    let mut circ = CirculantMatrix::new(c_owned)?;
    circ.solve(b)
}

// ---------------------------------------------------------------------------
// solve_tridiagonal — Thomas algorithm O(n)
// ---------------------------------------------------------------------------

/// Solve a tridiagonal system in O(n) using the Thomas algorithm (tridiagonal
/// matrix algorithm, TDMA).
///
/// The system is:
///   lower\[i\] * x\[i-1\]  +  diag\[i\] * x\[i\]  +  upper\[i\] * x\[i+1\]  =  b\[i\]
///
/// where `lower[0]` and `upper[n-1]` are unused (conventionally zero).
///
/// # Arguments
/// * `lower` - Sub-diagonal (length n; `lower[0]` is ignored)
/// * `diag`  - Main diagonal (length n)
/// * `upper` - Super-diagonal (length n; `upper[n-1]` is ignored)
/// * `b`     - Right-hand side vector (length n)
///
/// # Returns
/// Solution vector x of length n.
///
/// # Errors
/// Returns `LinalgError` if the system is singular or dimensions mismatch.
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::structured_solve::solve_tridiagonal;
///
/// // -x[i-1] + 2*x[i] - x[i+1] = 0 with boundary conditions
/// let lower = array![0.0_f64, -1.0, -1.0];
/// let diag  = array![2.0_f64,  2.0,  2.0];
/// let upper = array![-1.0_f64, -1.0, 0.0];
/// let b     = array![1.0_f64,  0.0,  1.0];
/// let x = solve_tridiagonal(&lower.view(), &diag.view(), &upper.view(), &b.view())
///     .expect("solve_tridiagonal");
/// assert_eq!(x.len(), 3);
/// ```
pub fn solve_tridiagonal<F>(
    lower: &ArrayView1<F>,
    diag: &ArrayView1<F>,
    upper: &ArrayView1<F>,
    b: &ArrayView1<F>,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let n = diag.len();
    if n == 0 {
        return Err(LinalgError::InvalidInputError(
            "solve_tridiagonal: diagonal must be non-empty".to_string(),
        ));
    }
    if lower.len() != n || upper.len() != n || b.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "solve_tridiagonal: all vectors must have length {n}"
        )));
    }

    // Working copies for modified coefficients
    let mut c_prime = Array1::<F>::zeros(n); // modified super-diagonal
    let mut d_prime = Array1::<F>::zeros(n); // modified rhs

    // Forward sweep
    if diag[0].abs() < F::epsilon() {
        return Err(LinalgError::SingularMatrixError(
            "solve_tridiagonal: diagonal element at index 0 is zero (singular)".to_string(),
        ));
    }
    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = b[0] / diag[0];

    for i in 1..n {
        let denom = diag[i] - lower[i] * c_prime[i - 1];
        if denom.abs() < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(format!(
                "solve_tridiagonal: pivot at index {i} is zero (singular matrix)"
            )));
        }
        if i < n - 1 {
            c_prime[i] = upper[i] / denom;
        }
        d_prime[i] = (b[i] - lower[i] * d_prime[i - 1]) / denom;
    }

    // Back substitution
    let mut x = Array1::<F>::zeros(n);
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// solve_banded — Banded LU factorisation
// ---------------------------------------------------------------------------

/// Solve a banded linear system A x = b in O(n · (kl + ku)²).
///
/// The banded matrix is stored in the standard LAPACK compact banded format:
/// `ab` is a `(kl + ku + 1) × n` matrix where element `A[i, j]` is stored at
/// `ab[ku + i - j, j]` for `max(0, j-ku) ≤ i ≤ min(n-1, j+kl)`.
///
/// # Arguments
/// * `kl`  - Number of lower diagonals (sub-diagonals)
/// * `ku`  - Number of upper diagonals (super-diagonals)
/// * `ab`  - Compact banded storage, shape `(kl + ku + 1, n)`
/// * `b`   - Right-hand side vector of length n
///
/// # Returns
/// Solution vector x of length n.
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_linalg::structured_solve::solve_banded;
///
/// // Tridiagonal [[2,-1,0],[-1,2,-1],[0,-1,2]], b = [1,0,1]
/// // kl=1, ku=1; ab rows: [upper, diag, lower]
/// let ab = Array2::from_shape_vec((3, 3), vec![
///     0.0_f64, -1.0, -1.0,  // upper diagonal (row ku+i-j=0+i-j for j=i+1)
///     2.0,  2.0,  2.0,      // main diagonal
///    -1.0, -1.0,  0.0,      // lower diagonal
/// ]).expect("shape");
/// let b = array![1.0_f64, 0.0, 1.0];
/// let x = solve_banded(1, 1, &ab.view(), &b.view()).expect("ok");
/// assert_eq!(x.len(), 3);
/// ```
pub fn solve_banded<F>(
    kl: usize,
    ku: usize,
    ab: &ArrayView2<F>,
    b: &ArrayView1<F>,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let bandwidth = kl + ku + 1;
    let (ab_rows, n) = (ab.nrows(), ab.ncols());

    if ab_rows != bandwidth {
        return Err(LinalgError::ShapeError(format!(
            "solve_banded: ab should have {} rows (kl+ku+1), got {ab_rows}",
            bandwidth
        )));
    }
    if b.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "solve_banded: RHS length {} does not match matrix size {n}",
            b.len()
        )));
    }
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    // Work with a mutable copy; we perform in-place banded Gaussian elimination.
    let mut ab_work = ab.to_owned();
    let mut x = b.to_owned();

    for k in 0..n {
        // Pivot row in the compact storage: the diagonal is at row ku (= ku + k - k).
        let pivot = ab_work[[ku, k]];
        if pivot.abs() < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(format!(
                "solve_banded: zero pivot at column {k}"
            )));
        }

        // Eliminate lower band
        let lower_end = (kl + 1).min(n - k);
        for i in 1..lower_end {
            // The element below the pivot: row = ku + i  in compact storage for column k
            let factor = ab_work[[ku + i, k]] / pivot;
            if factor.abs() < F::epsilon() {
                continue;
            }
            // Update the row (k+i) in the band
            let upper_end = (ku + 1).min(n - k);
            for j in 0..upper_end {
                if ku + i > j && ku + i - j < bandwidth {
                    let row_src = ku - j;
                    let row_dst = ku + i - j;
                    let col = k + j;
                    if col < n {
                        let src_val = ab_work[[row_src, col]];
                        ab_work[[row_dst, col]] -= factor * src_val;
                    }
                }
            }
            let xk = x[k];
            x[k + i] -= factor * xk;
            ab_work[[ku + i, k]] = F::zero();
        }
    }

    // Back substitution using the upper triangle stored in ab_work
    for k in (0..n).rev() {
        let pivot = ab_work[[ku, k]];
        if pivot.abs() < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(format!(
                "solve_banded: zero pivot at column {k} during back-substitution"
            )));
        }
        let upper_end = (ku + 1).min(n - k);
        for j in 1..upper_end {
            let xkj = x[k + j];
            x[k] -= ab_work[[ku - j, k + j]] * xkj;
        }
        x[k] /= pivot;
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// solve_triangular — forward / back substitution O(n²)
// ---------------------------------------------------------------------------

/// Solve a triangular linear system T x = b in O(n²).
///
/// Performs forward substitution (if `lower = true`) or back substitution
/// (if `lower = false`).  The diagonal of T is used unless `unit_diagonal` is
/// set, in which case it is assumed to equal 1.
///
/// # Arguments
/// * `t`            - Triangular matrix (n×n)
/// * `b`            - Right-hand side vector (length n)
/// * `lower`        - `true` for lower triangular, `false` for upper
/// * `unit_diagonal` - If `true`, treat the diagonal as all-ones
///
/// # Returns
/// Solution vector x of length n.
///
/// # Errors
/// Returns `LinalgError` for non-square T, dimension mismatch, or zero pivot.
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::structured_solve::solve_triangular;
///
/// // Lower triangular [[1,0],[2,1]] * [x0,x1]ᵀ = [3,7]  → [3, 1]
/// let l = array![[1.0_f64, 0.0], [2.0, 1.0]];
/// let b = array![3.0_f64, 7.0];
/// let x = solve_triangular(&l.view(), &b.view(), true, false).expect("forward sub");
/// assert!((x[0] - 3.0).abs() < 1e-10 && (x[1] - 1.0).abs() < 1e-10);
///
/// // Upper triangular [[2,1],[0,3]] * [x0,x1]ᵀ = [5,6] → [1.5, 2]
/// let u = array![[2.0_f64, 1.0], [0.0, 3.0]];
/// let b2 = array![5.0_f64, 6.0];
/// let x2 = solve_triangular(&u.view(), &b2.view(), false, false).expect("back sub");
/// assert!((x2[0] - 1.5).abs() < 1e-10 && (x2[1] - 2.0).abs() < 1e-10);
/// ```
pub fn solve_triangular<F>(
    t: &ArrayView2<F>,
    b: &ArrayView1<F>,
    lower: bool,
    unit_diagonal: bool,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let n = t.nrows();
    if n == 0 {
        return Err(LinalgError::InvalidInputError(
            "solve_triangular: matrix must be non-empty".to_string(),
        ));
    }
    if t.ncols() != n {
        return Err(LinalgError::ShapeError(
            "solve_triangular: matrix must be square".to_string(),
        ));
    }
    if b.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "solve_triangular: matrix is {n}×{n} but b has length {}",
            b.len()
        )));
    }

    let mut x = Array1::<F>::zeros(n);

    if lower {
        // Forward substitution: x[i] = (b[i] - Σ_{j<i} T[i,j]*x[j]) / T[i,i]
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= t[[i, j]] * x[j];
            }
            if unit_diagonal {
                x[i] = sum;
            } else {
                let tii = t[[i, i]];
                if tii.abs() < F::epsilon() {
                    return Err(LinalgError::SingularMatrixError(format!(
                        "solve_triangular: zero diagonal at row {i}"
                    )));
                }
                x[i] = sum / tii;
            }
        }
    } else {
        // Back substitution: x[i] = (b[i] - Σ_{j>i} T[i,j]*x[j]) / T[i,i]
        for i in (0..n).rev() {
            let mut sum = b[i];
            for j in (i + 1)..n {
                sum -= t[[i, j]] * x[j];
            }
            if unit_diagonal {
                x[i] = sum;
            } else {
                let tii = t[[i, i]];
                if tii.abs() < F::epsilon() {
                    return Err(LinalgError::SingularMatrixError(format!(
                        "solve_triangular: zero diagonal at row {i}"
                    )));
                }
                x[i] = sum / tii;
            }
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    // ---- solve_toeplitz ----

    #[test]
    fn test_toeplitz_identity() {
        let r = array![1.0_f64, 0.0, 0.0];
        let c = array![1.0_f64, 0.0, 0.0];
        let b = array![2.0_f64, 3.0, 4.0];
        let x = solve_toeplitz(&r.view(), &c.view(), &b.view()).expect("toeplitz identity");
        for (xi, bi) in x.iter().zip(b.iter()) {
            assert!((xi - bi).abs() < 1e-10, "{xi} ≠ {bi}");
        }
    }

    #[test]
    fn test_toeplitz_symmetric_2x2() {
        // [[2,1],[1,2]] * [1,1] = [3,3]
        let r = array![2.0_f64, 1.0];
        let c = array![2.0_f64, 1.0];
        let b = array![3.0_f64, 3.0];
        let x = solve_toeplitz(&r.view(), &c.view(), &b.view()).expect("toeplitz 2x2");
        assert!((x[0] - 1.0).abs() < 1e-10 && (x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_toeplitz_nonsymmetric() {
        // [[2,1],[-1,2]] * x = [1,0] → x = [2/5, 1/5]
        let r = array![2.0_f64, 1.0];
        let c = array![2.0_f64, -1.0];
        let b = array![1.0_f64, 0.0];
        let x = solve_toeplitz(&r.view(), &c.view(), &b.view()).expect("toeplitz nonsym");
        // Verify: Tx = b
        let t00 = 2.0_f64;
        let t01 = 1.0_f64;
        let t10 = -1.0_f64;
        let t11 = 2.0_f64;
        let r0 = t00 * x[0] + t01 * x[1] - b[0];
        let r1 = t10 * x[0] + t11 * x[1] - b[1];
        assert!(r0.abs() < 1e-10 && r1.abs() < 1e-10);
    }

    #[test]
    fn test_toeplitz_diagonal_mismatch_error() {
        let r = array![2.0_f64, 1.0];
        let c = array![3.0_f64, 1.0]; // c[0] ≠ r[0]
        let b = array![1.0_f64, 0.0];
        assert!(solve_toeplitz(&r.view(), &c.view(), &b.view()).is_err());
    }

    #[test]
    fn test_toeplitz_singular_error() {
        let r = array![0.0_f64, 1.0];
        let c = array![0.0_f64, 1.0];
        let b = array![1.0_f64, 0.0];
        assert!(solve_toeplitz(&r.view(), &c.view(), &b.view()).is_err());
    }

    // ---- solve_circulant ----

    #[test]
    fn test_circulant_2x2() {
        // C = [[2,1],[1,2]], b = [3,3] → x = [1,1]
        let c = array![2.0_f64, 1.0];
        let b = array![3.0_f64, 3.0];
        let x = solve_circulant(&c.view(), &b.view()).expect("circulant 2x2");
        assert!((x[0] - 1.0).abs() < 1e-9 && (x[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_circulant_identity() {
        // C = I: c = [1,0,0,0]
        let c = array![1.0_f64, 0.0, 0.0, 0.0];
        let b = array![5.0_f64, 3.0, 1.0, 2.0];
        let x = solve_circulant(&c.view(), &b.view()).expect("circulant identity");
        for (xi, bi) in x.iter().zip(b.iter()) {
            assert!((xi - bi).abs() < 1e-9);
        }
    }

    #[test]
    fn test_circulant_singular_error() {
        // All-zero first row → singular
        let c = array![0.0_f64, 0.0];
        let b = array![1.0_f64, 0.0];
        assert!(solve_circulant(&c.view(), &b.view()).is_err());
    }

    #[test]
    fn test_circulant_dimension_mismatch_error() {
        let c = array![1.0_f64, 0.0];
        let b = array![1.0_f64, 0.0, 0.0];
        assert!(solve_circulant(&c.view(), &b.view()).is_err());
    }

    // ---- solve_tridiagonal ----

    #[test]
    fn test_tridiagonal_simple() {
        let lower = array![0.0_f64, -1.0, -1.0];
        let diag = array![2.0_f64, 2.0, 2.0];
        let upper = array![-1.0_f64, -1.0, 0.0];
        let b = array![1.0_f64, 0.0, 1.0];
        let x = solve_tridiagonal(&lower.view(), &diag.view(), &upper.view(), &b.view())
            .expect("tridiagonal simple");
        // Verify residual
        let r0 = 2.0 * x[0] - x[1] - 1.0;
        let r1 = -x[0] + 2.0 * x[1] - x[2];
        let r2 = -x[1] + 2.0 * x[2] - 1.0;
        assert!(r0.abs() < 1e-10 && r1.abs() < 1e-10 && r2.abs() < 1e-10);
    }

    #[test]
    fn test_tridiagonal_diagonal_system() {
        // Pure diagonal: lower = upper = 0
        let lower = array![0.0_f64, 0.0, 0.0];
        let diag = array![2.0_f64, 4.0, 8.0];
        let upper = array![0.0_f64, 0.0, 0.0];
        let b = array![6.0_f64, 12.0, 24.0];
        let x = solve_tridiagonal(&lower.view(), &diag.view(), &upper.view(), &b.view())
            .expect("tridiagonal diagonal");
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_tridiagonal_singular_error() {
        let lower = array![0.0_f64, 1.0];
        let diag = array![0.0_f64, 2.0]; // zero diagonal at 0
        let upper = array![1.0_f64, 0.0];
        let b = array![1.0_f64, 1.0];
        assert!(solve_tridiagonal(&lower.view(), &diag.view(), &upper.view(), &b.view()).is_err());
    }

    // ---- solve_banded ----

    #[test]
    fn test_banded_tridiagonal() {
        // Same system as tridiagonal test but via banded format
        let ab = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.0_f64, -1.0, -1.0, // upper diagonal (ku=1)
                2.0, 2.0, 2.0, // main diagonal
                -1.0, -1.0, 0.0, // lower diagonal (kl=1)
            ],
        )
        .expect("shape");
        let b = array![1.0_f64, 0.0, 1.0];
        let x = solve_banded(1, 1, &ab.view(), &b.view()).expect("banded tridiagonal");
        // Residuals
        let r0 = 2.0 * x[0] - x[1] - 1.0;
        let r1 = -x[0] + 2.0 * x[1] - x[2];
        let r2 = -x[1] + 2.0 * x[2] - 1.0;
        assert!(
            r0.abs() < 1e-10 && r1.abs() < 1e-10 && r2.abs() < 1e-10,
            "residuals: {r0} {r1} {r2}, x = {x:?}"
        );
    }

    #[test]
    fn test_banded_diagonal_only() {
        // kl=0, ku=0: purely diagonal matrix
        let ab = Array2::from_shape_vec((1, 3), vec![2.0_f64, 4.0, 8.0]).expect("shape");
        let b = array![6.0_f64, 12.0, 24.0];
        let x = solve_banded(0, 0, &ab.view(), &b.view()).expect("banded diagonal");
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_banded_shape_error() {
        let ab = Array2::<f64>::zeros((2, 3)); // should be (kl+ku+1, n) = (3, n) for kl=ku=1
        let b = array![1.0_f64, 0.0, 1.0];
        assert!(solve_banded(1, 1, &ab.view(), &b.view()).is_err());
    }

    #[test]
    fn test_banded_rhs_mismatch_error() {
        let ab = Array2::<f64>::zeros((1, 3));
        let b = array![1.0_f64, 0.0]; // wrong length
        assert!(solve_banded(0, 0, &ab.view(), &b.view()).is_err());
    }

    // ---- solve_triangular ----

    #[test]
    fn test_triangular_lower() {
        let l = array![[1.0_f64, 0.0], [2.0, 1.0]];
        let b = array![3.0_f64, 7.0];
        let x = solve_triangular(&l.view(), &b.view(), true, false).expect("lower tri");
        assert!((x[0] - 3.0).abs() < 1e-10 && (x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangular_upper() {
        // U = [[2,1],[0,3]], b = [5,6]
        // x[1] = 6/3 = 2, x[0] = (5 - 1*2)/2 = 1.5
        let u = array![[2.0_f64, 1.0], [0.0, 3.0]];
        let b = array![5.0_f64, 6.0];
        let x = solve_triangular(&u.view(), &b.view(), false, false).expect("upper tri");
        assert!((x[0] - 1.5).abs() < 1e-10 && (x[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangular_unit_diagonal_lower() {
        // Unit lower triangular [[1,0],[2,1]] → same as test_triangular_lower
        let l = array![[9.9_f64, 0.0], [2.0, 9.9]]; // diagonal ignored
        let b = array![3.0_f64, 7.0];
        let x = solve_triangular(&l.view(), &b.view(), true, true).expect("unit lower");
        // Forward sub with unit diagonal: x[0]=3, x[1]=7-2*3=1
        assert!((x[0] - 3.0).abs() < 1e-10 && (x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangular_singular_error() {
        let l = array![[0.0_f64, 0.0], [1.0, 1.0]];
        let b = array![1.0_f64, 2.0];
        assert!(solve_triangular(&l.view(), &b.view(), true, false).is_err());
    }

    #[test]
    fn test_triangular_nonsquare_error() {
        let t = Array2::<f64>::zeros((2, 3));
        let b = array![1.0_f64, 0.0];
        assert!(solve_triangular(&t.view(), &b.view(), true, false).is_err());
    }

    #[test]
    fn test_triangular_dimension_mismatch_error() {
        let t = Array2::<f64>::eye(3);
        let b = array![1.0_f64, 0.0]; // wrong length
        assert!(solve_triangular(&t.view(), &b.view(), true, false).is_err());
    }

    #[test]
    fn test_triangular_3x3_lower() {
        // L = [[1,0,0],[2,3,0],[4,5,6]], b = [1,8,32]
        // x0=1, x1=(8-2)/3=2, x2=(32-4-10)/6=3
        let l = array![[1.0_f64, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0],];
        let b = array![1.0_f64, 8.0, 32.0];
        let x = solve_triangular(&l.view(), &b.view(), true, false).expect("3x3 lower");
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangular_3x3_upper() {
        // U = [[6,5,4],[0,3,2],[0,0,1]], b = [32,8,1]
        // x2=1, x1=(8-2)/3=2, x0=(32-10-4)/6=3
        let u = array![[6.0_f64, 5.0, 4.0], [0.0, 3.0, 2.0], [0.0, 0.0, 1.0],];
        let b = array![32.0_f64, 8.0, 1.0];
        let x = solve_triangular(&u.view(), &b.view(), false, false).expect("3x3 upper");
        assert!((x[2] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        assert!((x[0] - 3.0).abs() < 1e-10);
    }
}
