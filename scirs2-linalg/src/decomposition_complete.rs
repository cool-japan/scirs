//! Complete Orthogonal Decomposition, Rank-Revealing QR, and ULV Decomposition
//!
//! This module provides rank-revealing and complete orthogonal factorizations:
//!
//! - **COD** (Complete Orthogonal Decomposition): A = U T Z^T where T is upper
//!   trapezoidal and U, Z are orthogonal.  Ideal for rank-deficient least-squares.
//! - **RRQR** (Rank-Revealing QR): A P = Q R with column pivoting so that
//!   trailing diagonal of R decays rapidly.
//! - **ULV** Decomposition: A = U L V^T where L is lower triangular.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};

// -----------------------------------------------------------------------
// Result types
// -----------------------------------------------------------------------

/// Result of the Complete Orthogonal Decomposition A = U T Z^T.
pub struct CodResult {
    /// Left orthogonal factor (m×m)
    pub u: Array2<f64>,
    /// Upper trapezoidal core factor (m×n)
    pub t: Array2<f64>,
    /// Right orthogonal factor (n×n)
    pub z: Array2<f64>,
    /// Numerical rank
    pub rank: usize,
}

/// Result of the Rank-Revealing QR decomposition.
pub struct RrqrResult {
    /// Left orthogonal factor Q (m×m)
    pub q: Array2<f64>,
    /// Upper trapezoidal factor R (m×n)
    pub r: Array2<f64>,
    /// Column permutation vector: A[:, perm] = Q R
    pub perm: Vec<usize>,
    /// Numerical rank
    pub rank: usize,
}

/// Result of the ULV Decomposition A = U L V^T.
pub struct UlvResult {
    /// Left orthogonal factor U (m×m)
    pub u: Array2<f64>,
    /// Lower trapezoidal factor L (m×n)
    pub l: Array2<f64>,
    /// Right orthogonal factor V (n×n)
    pub v: Array2<f64>,
    /// Numerical rank
    pub rank: usize,
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/// Compute the Euclidean norm of a 1-D slice.
fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Apply a Householder reflector H = I - 2 v v^T / (v^T v) to the columns
/// of `a` (in-place) from the left.  `v` is the Householder vector.
fn apply_householder_left(a: &mut Array2<f64>, v: &[f64], row_start: usize) {
    let m = a.nrows();
    let n = a.ncols();
    let vn2: f64 = v.iter().map(|&x| x * x).sum();
    if vn2 < f64::EPSILON {
        return;
    }
    let scale = 2.0 / vn2;
    for j in 0..n {
        let dot: f64 = (row_start..m).map(|i| v[i - row_start] * a[[i, j]]).sum();
        let coeff = scale * dot;
        for i in row_start..m {
            a[[i, j]] -= coeff * v[i - row_start];
        }
    }
}

/// Apply a Householder reflector from the right (to rows of `a`).
fn apply_householder_right(a: &mut Array2<f64>, v: &[f64], col_start: usize) {
    let m = a.nrows();
    let n = a.ncols();
    let vn2: f64 = v.iter().map(|&x| x * x).sum();
    if vn2 < f64::EPSILON {
        return;
    }
    let scale = 2.0 / vn2;
    for i in 0..m {
        let dot: f64 = (col_start..n).map(|j| v[j - col_start] * a[[i, j]]).sum();
        let coeff = scale * dot;
        for j in col_start..n {
            a[[i, j]] -= coeff * v[j - col_start];
        }
    }
}

/// Build the full Householder product Q stored as a sequence of reflectors.
/// Each entry is (start_index, householder_vector).
fn build_q_from_reflectors(m: usize, reflectors: &[(usize, Vec<f64>)]) -> Array2<f64> {
    let mut q = Array2::<f64>::eye(m);
    // Apply in reverse order
    for &(start, ref v) in reflectors.iter().rev() {
        apply_householder_left(&mut q, v, start);
    }
    q
}

/// Build the full right-side Householder product Z from reflectors.
fn build_z_from_reflectors(n: usize, reflectors: &[(usize, Vec<f64>)]) -> Array2<f64> {
    let mut z = Array2::<f64>::eye(n);
    for &(start, ref v) in reflectors.iter().rev() {
        apply_householder_right(&mut z, v, start);
    }
    z
}

// -----------------------------------------------------------------------
// Rank-Revealing QR (column-pivoted QR)
// -----------------------------------------------------------------------

/// Compute the Rank-Revealing QR decomposition with column pivoting:
///   A P = Q R
///
/// At each step the column with the largest 2-norm is brought to the pivot
/// position, so the diagonal of R is non-increasing.  The rank is estimated
/// from the diagonal of R using `tol`.
///
/// # Arguments
/// * `a`   - Input matrix (m×n)
/// * `tol` - Rank threshold (default: max(m,n) * eps * |R\[0,0\]|)
///
/// # Returns
/// [`RrqrResult`] containing Q, R, column permutation, and numerical rank
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::decomposition_complete::rrqr;
///
/// let a = array![
///     [1.0_f64, 2.0, 3.0],
///     [4.0,     5.0, 6.0],
///     [7.0,     8.0, 9.0],
/// ];
/// let res = rrqr(&a.view(), None).expect("rrqr failed");
/// // Reconstruct: A * P_matrix = Q * R
/// let mut ap = a.clone();
/// // A permuted
/// let ap2: Vec<_> = res.perm.iter().map(|&c| a.column(c).to_owned()).collect();
/// let rows = a.nrows();
/// let cols = a.ncols();
/// let mut a_perm = scirs2_core::ndarray::Array2::<f64>::zeros((rows, cols));
/// for (j, col) in ap2.iter().enumerate() {
///     a_perm.column_mut(j).assign(col);
/// }
/// let recon = res.q.dot(&res.r);
/// let err: f64 = (&recon - &a_perm).iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
/// assert!(err < 1e-10, "RRQR reconstruction error: {err}");
/// ```
pub fn rrqr(a: &ArrayView2<f64>, tol: Option<f64>) -> LinalgResult<RrqrResult> {
    let m = a.nrows();
    let n = a.ncols();
    let mut work = a.to_owned();
    let mut perm: Vec<usize> = (0..n).collect();
    let k = m.min(n);

    // Track column norms for efficient pivoting.
    // Do NOT use as_slice() on strided column views — it returns None for non-contiguous
    // memory layouts and would yield an empty slice (giving norm 0 for all columns).
    // Explicitly iterate over column elements instead.
    let mut col_norms: Vec<f64> = (0..n)
        .map(|j| {
            let norm_sq: f64 = (0..m).map(|i| work[[i, j]] * work[[i, j]]).sum();
            norm_sq.sqrt()
        })
        .collect();

    let mut left_reflectors: Vec<(usize, Vec<f64>)> = Vec::with_capacity(k);

    for step in 0..k {
        // Find pivot column (max norm among remaining columns)
        let pivot_rel = (step..n)
            .max_by(|&a, &b| col_norms[a].partial_cmp(&col_norms[b]).expect("nan norm"))
            .unwrap_or(step);

        if col_norms[pivot_rel] < f64::EPSILON {
            break;
        }

        // Swap columns
        if pivot_rel != step {
            for i in 0..m {
                let tmp = work[[i, step]];
                work[[i, step]] = work[[i, pivot_rel]];
                work[[i, pivot_rel]] = tmp;
            }
            perm.swap(step, pivot_rel);
            col_norms.swap(step, pivot_rel);
        }

        // Householder reflector for column `step` starting at row `step`
        let col_len = m - step;
        let mut v: Vec<f64> = (step..m).map(|i| work[[i, step]]).collect();
        let sigma = norm2(&v);
        if sigma < f64::EPSILON {
            continue;
        }
        let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
        v[0] += sign * sigma;
        apply_householder_left(&mut work, &v, step);

        // Update column norms (Rank-1 downdate for numerical stability)
        for j in (step + 1)..n {
            let r_jj = work[[step, j]];
            let old_sq = col_norms[j] * col_norms[j];
            let new_sq = (old_sq - r_jj * r_jj).max(0.0);
            col_norms[j] = new_sq.sqrt();
        }

        left_reflectors.push((step, v));
        let _ = col_len; // suppress unused warning
    }

    // Build Q
    let q = build_q_from_reflectors(m, &left_reflectors);

    // R is the upper triangular part of work
    let r = work;

    // Determine rank
    let r00 = r[[0, 0]].abs();
    let threshold = tol.unwrap_or_else(|| {
        let eps = f64::EPSILON;
        eps * (m.max(n) as f64) * r00
    });
    let rank = (0..k).take_while(|&i| r[[i, i]].abs() > threshold).count();

    Ok(RrqrResult { q, r, perm, rank })
}

// -----------------------------------------------------------------------
// Complete Orthogonal Decomposition (COD): A = U T Z^T
// -----------------------------------------------------------------------

/// Compute the Complete Orthogonal Decomposition A = U T Z^T.
///
/// T is upper trapezoidal (rank × n), U is left orthogonal (m×m), Z is right
/// orthogonal (n×n).  The decomposition is computed by first performing a
/// rank-revealing QR, then applying additional Householder reflectors from the
/// right to zero out the trailing block of R, leaving an upper-left triangular T.
///
/// # Arguments
/// * `a`   - Input matrix (m×n)
/// * `tol` - Rank threshold
///
/// # Returns
/// [`CodResult`] with U, T, Z, rank
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::decomposition_complete::cod;
///
/// let a = array![
///     [1.0_f64, 2.0, 3.0],
///     [2.0,     4.0, 6.0],
/// ];
/// let res = cod(&a.view(), None).expect("COD failed");
/// // Reconstruct A permuted = U * T * Z^T
/// let recon = res.u.dot(&res.t).dot(&res.z.t());
/// // recon should approximate the column-permuted A (handled by z)
/// let mut err = 0.0_f64;
/// for i in 0..a.nrows() {
///     for j in 0..a.ncols() {
///         err = err.max((recon[[i,j]] - a[[i,j]]).abs());
///     }
/// }
/// assert!(err < 1e-10, "COD reconstruction error: {err}");
/// ```
pub fn cod(a: &ArrayView2<f64>, tol: Option<f64>) -> LinalgResult<CodResult> {
    let m = a.nrows();
    let n = a.ncols();

    // Step 1: RRQR to get Q R P^T  (A P = Q R)
    let rrqr_res = rrqr(a, tol)?;
    let rank = rrqr_res.rank;

    // U starts as Q (m×m)
    let u = rrqr_res.q.clone();

    // Work with R, then apply right Householder to zero out R[0..rank, rank..n]
    let mut t = rrqr_res.r; // m × n

    // Build the permutation matrix (applied on right: A * P_mat = Q * R)
    // We'll incorporate it into Z at the end.
    let mut z = Array2::<f64>::eye(n);
    // Apply column permutation to z: z = I * P_mat (columns reordered)
    let perm = rrqr_res.perm;
    let z_perm: Vec<_> = perm.iter().map(|&c| z.column(c).to_owned()).collect();
    for (j, col) in z_perm.iter().enumerate() {
        z.column_mut(j).assign(col);
    }

    // Step 2: Apply right Householder reflectors to zero out T[0..rank, rank..n]
    // This brings T to pure upper triangular form within the rank×rank block.
    let mut right_reflectors: Vec<(usize, Vec<f64>)> = Vec::new();
    for i in 0..rank {
        if rank >= n {
            break; // Already triangular
        }
        // Reflect row i of T from column i to column n-1 to zero out T[i, rank..n]
        // We use a Householder applied from the right targeting columns rank..n for row i
        // Actually: zero out T[i, rank..n] by a Householder on columns [i, rank..n]
        // Build vector from T[i, i..n]
        let row_len = n - i;
        if row_len <= 1 {
            break;
        }
        let v_raw: Vec<f64> = (i..n).map(|j| t[[i, j]]).collect();
        // We want to zero out v_raw[1..] (indices rank-i .. row_len) that are beyond rank
        // Standard: make T[i, i] = sigma, zero out T[i, i+1..n]
        let sigma = norm2(&v_raw);
        if sigma < f64::EPSILON {
            continue;
        }
        let sign = if v_raw[0] >= 0.0 { 1.0 } else { -1.0 };
        let mut v = v_raw;
        v[0] += sign * sigma;
        apply_householder_right(&mut t, &v, i);
        apply_householder_right(&mut z, &v, i);
        right_reflectors.push((i, v));
    }

    Ok(CodResult { u, t, z, rank })
}

// -----------------------------------------------------------------------
// ULV Decomposition: A = U L V^T
// -----------------------------------------------------------------------

/// Compute the ULV Decomposition A = U L V^T.
///
/// L is lower trapezoidal (m×n), U is left orthogonal (m×m), V is right
/// orthogonal (n×n).  The decomposition is obtained by first computing SVD and
/// then constructing the factors from the SVD structure (since any ULV can be
/// derived from SVD via Q-factors).  The rank is estimated from singular values.
///
/// # Arguments
/// * `a` - Input matrix (m×n)
///
/// # Returns
/// [`UlvResult`] with U, L, V, rank
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::decomposition_complete::ulv;
///
/// let a = array![[3.0_f64, 2.0, 1.0], [1.0, 2.0, 3.0]];
/// let res = ulv(&a.view()).expect("ULV failed");
/// let recon = res.u.dot(&res.l).dot(&res.v.t());
/// let err: f64 = (&recon - a).iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
/// assert!(err < 1e-10, "ULV error: {err}");
/// ```
pub fn ulv(a: &ArrayView2<f64>) -> LinalgResult<UlvResult> {
    let m = a.nrows();
    let n = a.ncols();

    // Use SVD: A = U_s * Sigma * V_s^T
    // Then U = U_s, L = Sigma * V_s^T permuted to be lower triangular form,
    // V = I permutation to arrange columns.
    //
    // More precisely: A = U_s diag(sigma) V_s^T
    // We want lower triangular, so flip: reverse singular values + vectors.
    //
    // A = U_s * flip(Sigma) * flip(V_s)^T
    // where flip reverses the column/row ordering so smallest SV is first,
    // giving L = flip(Sigma) * ...
    //
    // Practical approach: from SVD, construct L via QR of (V_s * Sigma)^T
    // which gives an exact ULV with L lower triangular.

    let (u_s, s_vec, vt_s) = crate::decomposition::svd(a, true, None)?;

    let k = m.min(n);

    // Form L = U_s^T * A * V_s  (diagonal * permutation to make it lower tri)
    // After SVD: A = U_s * S * Vt_s, so U_s^T A V_s = S (diagonal, not lower-tri yet).
    // To get *lower* trapezoidal L we can use: L = (S * Vt_s * V_s)^T rearranged.
    // Simplest: reverse column order of U and V so the first column has the smallest SV.
    // Then S reversed on diagonal is still diagonal but anti-sorted.
    // To truly get lower triangular: permute so L[i,j]=0 for j>i.
    // We use: swap the ordering and take the transpose of the ULV to get LVU-like factoring,
    // then reinterpret.
    //
    // Cleanest numerically correct approach:
    // - Build "reversed" SVD: flip U columns, S entries, V^T rows.
    // - Then A = U_rev * S_rev * Vt_rev where S_rev is anti-diagonal.
    // - To get lower triangular L from S_rev * Vt_rev, perform QR-like step.
    //
    // Actually the simplest exact approach:
    //   Set U = U_s, V = V_s, L = diag(s) * (identity extension to m×n)
    //   This is already "valid" as ULV with L diagonal (subset of lower triangular).
    //   For strict lower triangular: we need one more right QR step.

    // Build diagonal sigma matrix (m×n)
    let mut l_mat = Array2::<f64>::zeros((m, n));
    for i in 0..k {
        l_mat[[i, i]] = s_vec[i];
    }

    // Apply Vt_s: L_full = diag(s) * Vt_s = U_s^T * A
    // This is a (m×n) matrix.  We need it to be lower triangular.
    // Perform a QR of L_full^T = (Vt_s^T * diag(s)) to get L_full^T = Q' R'
    // => L_full = R'^T Q'^T  (lower tri × orthogonal)
    // Then ULV: A = U_s * L_full * I = U_s * R'^T * Q'^T * I
    //   U = U_s * (identity), L = R'^T, V = Q'  (V^T = Q'^T)

    // L_full = diag(s) * Vt_s  (m × n)
    let sigma_mat = l_mat; // this is diag(s) embedded in m×n
    let l_full = sigma_mat.dot(&vt_s); // m × n

    // To get a lower trapezoidal L, we perform QR of l_full^T.
    // QR requires rows >= columns, so this only works when n >= m (wide or square A).
    // For tall matrices (m > n), l_full is already lower trapezoidal:
    //   rows 0..n contain the diagonal × Vt_s, rows n..m are zero.
    // In that case use A = U_s * l_full * I directly (V = I_n).
    let (l, u, v) = if n >= m {
        // Wide or square A (n >= m): QR of l_full^T is (n×m), and n >= m so QR is valid.
        let lt = l_full.t().to_owned(); // n × m
        let (q_prime, r_prime) = crate::decomposition::qr(&lt.view(), None)?;
        // l_full^T = q_prime * r_prime → l_full = r_prime^T * q_prime^T
        // A = U_s * l_full * I = U_s * r_prime^T * q_prime^T * I
        // => U = U_s, L = r_prime^T (m×m lower tri), V = q_prime (n×m)
        (r_prime.t().to_owned(), u_s, q_prime)
    } else {
        // Tall matrix (m > n): l_full = diag(s) * Vt_s is m×n.
        // QR of l_full^T = (n×m) would fail since n < m.
        // Instead use A = U_s * diag(s) * Vt_s directly:
        //   U = U_s, L = diag(s) (m×n lower trapezoidal), V = Vt_s^T (n×n orthogonal)
        let mut l_diag = Array2::<f64>::zeros((m, n));
        for i in 0..k {
            l_diag[[i, i]] = s_vec[i];
        }
        let v_full = vt_s.t().to_owned(); // n×n orthogonal
        (l_diag, u_s, v_full)
    };

    // Estimate rank from singular values
    let s_max = s_vec.iter().cloned().fold(0.0_f64, f64::max);
    let eps = f64::EPSILON;
    let thresh = eps * (m.max(n) as f64) * s_max;
    let rank = s_vec.iter().filter(|&&sv| sv > thresh).count();

    Ok(UlvResult { u, l, v, rank })
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    fn max_abs_error(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        (a - b).iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
    }

    // ---- RRQR ----

    #[test]
    fn test_rrqr_full_rank() {
        let a = array![
            [1.0_f64, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0], // slightly changed to make it full rank
        ];
        let res = rrqr(&a.view(), None).expect("rrqr failed");

        // Reconstruct: A[:, perm] = Q R
        let mut a_perm = Array2::<f64>::zeros((3, 3));
        for (j, &c) in res.perm.iter().enumerate() {
            a_perm.column_mut(j).assign(&a.column(c).to_owned());
        }
        let recon = res.q.dot(&res.r);
        assert!(
            max_abs_error(&recon, &a_perm) < 1e-10,
            "RRQR reconstruction error"
        );
    }

    #[test]
    fn test_rrqr_rank_deficient() {
        // Rank-1 matrix: rows are multiples of [1,2,3]
        let a = array![[1.0_f64, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0],];
        let res = rrqr(&a.view(), Some(1e-8)).expect("rrqr failed");
        assert_eq!(res.rank, 1, "Rank-1 matrix should have rank 1");
    }

    #[test]
    fn test_rrqr_tall_matrix() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];
        let res = rrqr(&a.view(), None).expect("rrqr failed");
        let mut a_perm = Array2::<f64>::zeros((4, 2));
        for (j, &c) in res.perm.iter().enumerate() {
            a_perm.column_mut(j).assign(&a.column(c).to_owned());
        }
        let recon = res.q.dot(&res.r);
        // Take only first min(m,n) columns of Q * R (R is m×n, Q is m×m)
        assert!(
            max_abs_error(&recon.slice(s![.., ..2]).to_owned(), &a_perm) < 1e-10,
            "RRQR tall reconstruction"
        );
    }

    #[test]
    fn test_rrqr_permutation_valid() {
        let a = array![[1.0_f64, 5.0, 2.0], [0.0, 0.0, 3.0],];
        let res = rrqr(&a.view(), None).expect("rrqr failed");
        // perm must be a permutation of 0..n
        let mut sorted = res.perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    // ---- COD ----

    #[test]
    fn test_cod_full_rank_reconstruction() {
        let a = array![[1.0_f64, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0],];
        let res = cod(&a.view(), None).expect("COD failed");
        let recon = res.u.dot(&res.t).dot(&res.z.t());
        assert!(max_abs_error(&recon, &a) < 1e-8, "COD reconstruction error");
    }

    #[test]
    fn test_cod_rank_deficient() {
        let a = array![[1.0_f64, 2.0, 3.0], [2.0, 4.0, 6.0],];
        let res = cod(&a.view(), Some(1e-8)).expect("COD failed");
        assert_eq!(res.rank, 1, "rank-deficient COD rank");
    }

    #[test]
    fn test_cod_reconstruction_rank_deficient() {
        let a = array![[1.0_f64, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0],];
        let res = cod(&a.view(), Some(1e-8)).expect("COD failed");
        let recon = res.u.dot(&res.t).dot(&res.z.t());
        // Reconstruction should match original (U T Z^T = Q R P^T rearranged)
        // The error should be small relative to the Frobenius norm of a
        let err = max_abs_error(&recon, &a);
        assert!(err < 1e-7, "COD rank-deficient reconstruction error: {err}");
    }

    // ---- ULV ----

    #[test]
    fn test_ulv_reconstruction() {
        let a = array![[3.0_f64, 2.0, 1.0], [1.0, 2.0, 3.0],];
        let res = ulv(&a.view()).expect("ULV failed");
        let recon = res.u.dot(&res.l).dot(&res.v.t());
        assert!(
            max_abs_error(&recon, &a) < 1e-10,
            "ULV reconstruction error"
        );
    }

    #[test]
    fn test_ulv_square() {
        let a = array![[4.0_f64, 3.0], [6.0, 3.0],];
        let res = ulv(&a.view()).expect("ULV square failed");
        let recon = res.u.dot(&res.l).dot(&res.v.t());
        assert!(
            max_abs_error(&recon, &a) < 1e-10,
            "ULV square reconstruction"
        );
    }

    #[test]
    fn test_ulv_rank() {
        let a = array![[1.0_f64, 2.0, 3.0], [2.0, 4.0, 6.0],];
        let res = ulv(&a.view()).expect("ULV rank failed");
        assert_eq!(res.rank, 1, "Rank-1 ULV rank estimate");
    }

    #[test]
    fn test_ulv_orthogonality_u() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0],];
        let res = ulv(&a.view()).expect("ULV orth U");
        let utu = res.u.t().dot(&res.u);
        let eye3 = Array2::<f64>::eye(3);
        assert!(max_abs_error(&utu, &eye3) < 1e-10, "U not orthogonal");
    }

    #[test]
    fn test_ulv_orthogonality_v() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0],];
        let res = ulv(&a.view()).expect("ULV orth V");
        let vtv = res.v.t().dot(&res.v);
        let eye3 = Array2::<f64>::eye(3);
        assert!(max_abs_error(&vtv, &eye3) < 1e-10, "V not orthogonal");
    }
}
