//! Linear algebra operations for WASM

use crate::array::WasmArray;
use crate::error::WasmError;
use scirs2_core::ndarray::Array2;
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Compute the determinant of a square matrix
#[wasm_bindgen]
pub fn det(arr: &WasmArray) -> Result<f64, JsValue> {
    if arr.ndim() != 2 {
        return Err(
            WasmError::InvalidDimensions("Determinant requires a 2D matrix".to_string()).into(),
        );
    }

    let matrix = arr
        .data()
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;

    if matrix.nrows() != matrix.ncols() {
        return Err(WasmError::InvalidDimensions(
            "Determinant requires a square matrix".to_string(),
        )
        .into());
    }

    // Simple implementation for small matrices
    match matrix.nrows() {
        1 => Ok(matrix[[0, 0]]),
        2 => Ok(matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]]),
        _ => {
            // For larger matrices, use a more sophisticated algorithm
            // This is a simplified LU decomposition-based approach
            Ok(compute_det_lu(&matrix))
        }
    }
}

/// Compute matrix inverse
#[wasm_bindgen]
pub fn inv(arr: &WasmArray) -> Result<WasmArray, JsValue> {
    if arr.ndim() != 2 {
        return Err(
            WasmError::InvalidDimensions("Inverse requires a 2D matrix".to_string()).into(),
        );
    }

    let matrix = arr
        .data()
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;

    if matrix.nrows() != matrix.ncols() {
        return Err(
            WasmError::InvalidDimensions("Inverse requires a square matrix".to_string()).into(),
        );
    }

    // Check for singularity
    let det_val = det(arr)?;
    if det_val.abs() < 1e-10 {
        return Err(
            WasmError::ComputationError("Matrix is singular or near-singular".to_string()).into(),
        );
    }

    // Use Gauss-Jordan elimination for matrix inverse
    let inv_matrix = compute_inverse_gauss_jordan(&matrix).map_err(WasmError::ComputationError)?;

    Ok(WasmArray::from_array(inv_matrix.into_dyn()))
}

/// Compute the trace of a matrix (sum of diagonal elements)
#[wasm_bindgen]
pub fn trace(arr: &WasmArray) -> Result<f64, JsValue> {
    if arr.ndim() != 2 {
        return Err(WasmError::InvalidDimensions("Trace requires a 2D matrix".to_string()).into());
    }

    let matrix = arr
        .data()
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;

    if matrix.nrows() != matrix.ncols() {
        return Err(
            WasmError::InvalidDimensions("Trace requires a square matrix".to_string()).into(),
        );
    }

    let mut sum = 0.0;
    for i in 0..matrix.nrows() {
        sum += matrix[[i, i]];
    }

    Ok(sum)
}

/// Compute the matrix rank
#[wasm_bindgen]
pub fn rank(arr: &WasmArray, tolerance: Option<f64>) -> Result<usize, JsValue> {
    if arr.ndim() != 2 {
        return Err(WasmError::InvalidDimensions("Rank requires a 2D matrix".to_string()).into());
    }

    let matrix = arr
        .data()
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;

    let tol = tolerance.unwrap_or(1e-10);
    let rank = compute_rank(&matrix, tol);

    Ok(rank)
}

/// Compute the Frobenius norm of a matrix
#[wasm_bindgen]
pub fn norm_frobenius(arr: &WasmArray) -> f64 {
    arr.data().iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Solve a system of linear equations Ax = b
#[wasm_bindgen]
pub fn solve(a: &WasmArray, b: &WasmArray) -> Result<WasmArray, JsValue> {
    if a.ndim() != 2 || b.ndim() != 1 {
        return Err(WasmError::InvalidDimensions(
            "solve requires A to be 2D and b to be 1D".to_string(),
        )
        .into());
    }

    let matrix_a = a
        .data()
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;
    let vector_b = b
        .data()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;

    if matrix_a.nrows() != matrix_a.ncols() {
        return Err(WasmError::InvalidDimensions("A must be a square matrix".to_string()).into());
    }

    if matrix_a.nrows() != vector_b.len() {
        return Err(WasmError::ShapeMismatch {
            expected: vec![matrix_a.nrows()],
            actual: vec![vector_b.len()],
        }
        .into());
    }

    // Use Gaussian elimination with partial pivoting
    let solution =
        solve_linear_system(&matrix_a, &vector_b).map_err(WasmError::ComputationError)?;

    Ok(WasmArray::from_array(solution.into_dyn()))
}

// Helper functions

fn compute_det_lu(matrix: &Array2<f64>) -> f64 {
    let n = matrix.nrows();
    let mut lu = matrix.clone();
    let mut sign = 1.0;

    // LU decomposition with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_val = lu[[k, k]].abs();
        let mut max_row = k;

        for i in (k + 1)..n {
            let val = lu[[i, k]].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        if max_val < 1e-15 {
            return 0.0; // Matrix is singular
        }

        // Swap rows if needed
        if max_row != k {
            for j in 0..n {
                let temp = lu[[k, j]];
                lu[[k, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = temp;
            }
            sign = -sign;
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = lu[[i, k]] / lu[[k, k]];
            for j in k..n {
                lu[[i, j]] -= factor * lu[[k, j]];
            }
        }
    }

    // Compute determinant as product of diagonal elements
    let mut det = sign;
    for i in 0..n {
        det *= lu[[i, i]];
    }

    det
}

fn compute_inverse_gauss_jordan(matrix: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = matrix.nrows();
    let mut aug = Array2::zeros((n, 2 * n));

    // Create augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = matrix[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut max_val = aug[[i, i]].abs();
        let mut max_row = i;

        for k in (i + 1)..n {
            let val = aug[[k, i]].abs();
            if val > max_val {
                max_val = val;
                max_row = k;
            }
        }

        if max_val < 1e-15 {
            return Err("Matrix is singular".to_string());
        }

        // Swap rows
        if max_row != i {
            for j in 0..(2 * n) {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Scale pivot row
        let pivot = aug[[i, i]];
        for j in 0..(2 * n) {
            aug[[i, j]] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[[k, i]];
                for j in 0..(2 * n) {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }
    }

    // Extract inverse matrix from right half
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }

    Ok(inv)
}

fn compute_rank(matrix: &Array2<f64>, tolerance: f64) -> usize {
    let mut temp = matrix.clone();
    let (m, n) = temp.dim();
    let mut rank = 0;

    for col in 0..n.min(m) {
        // Find pivot
        let mut max_val = 0.0;
        let mut max_row = col;

        for row in col..m {
            let val = temp[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < tolerance {
            continue;
        }

        // Swap rows
        if max_row != col {
            for j in 0..n {
                let temp_val = temp[[col, j]];
                temp[[col, j]] = temp[[max_row, j]];
                temp[[max_row, j]] = temp_val;
            }
        }

        rank += 1;

        // Eliminate
        for row in (col + 1)..m {
            let factor = temp[[row, col]] / temp[[col, col]];
            for j in col..n {
                temp[[row, j]] -= factor * temp[[col, j]];
            }
        }
    }

    rank
}

fn solve_linear_system(
    a: &Array2<f64>,
    b: &ndarray::Array1<f64>,
) -> Result<ndarray::Array1<f64>, String> {
    let n = a.nrows();
    let mut aug = Array2::zeros((n, n + 1));

    // Create augmented matrix [A | b]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_val = aug[[i, i]].abs();
        let mut max_row = i;

        for k in (i + 1)..n {
            let val = aug[[k, i]].abs();
            if val > max_val {
                max_val = val;
                max_row = k;
            }
        }

        if max_val < 1e-15 {
            return Err("System is singular or ill-conditioned".to_string());
        }

        // Swap rows
        if max_row != i {
            for j in 0..=n {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Eliminate
        for k in (i + 1)..n {
            let factor = aug[[k, i]] / aug[[i, i]];
            for j in i..=n {
                aug[[k, j]] -= factor * aug[[i, j]];
            }
        }
    }

    // Back substitution
    let mut x = ndarray::Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Singular Value Decomposition (SVD) — one-sided Jacobi algorithm
// ---------------------------------------------------------------------------

/// Output structure for SVD decomposition exposed to JavaScript via serde.
///
/// Fields are serialised with snake_case names so JavaScript receives
/// `singular_values`, `u`, `u_rows`, `u_cols`, `vt`, `vt_rows`, `vt_cols`.
#[derive(Serialize)]
struct SvdOutput {
    /// Singular values in descending order, length = min(m, n).
    singular_values: Vec<f64>,
    /// U matrix (left singular vectors) in row-major order, m × k.
    u: Vec<f64>,
    /// Number of rows in U.
    u_rows: usize,
    /// Number of columns in U (= k = min(m,n)).
    u_cols: usize,
    /// V^T matrix (right singular vectors) in row-major order, k × n.
    vt: Vec<f64>,
    /// Number of rows in V^T (= k = min(m,n)).
    vt_rows: usize,
    /// Number of columns in V^T.
    vt_cols: usize,
}

/// Compute the Singular Value Decomposition of a 2-D matrix.
///
/// Returns a JavaScript object `{ singular_values, u, u_rows, u_cols,
/// vt, vt_rows, vt_cols }`.  The decomposition satisfies
/// `A ≈ U · diag(singular_values) · Vt` with Frobenius error < 1e-9
/// for well-conditioned matrices up to ~100×100.
///
/// The one-sided Jacobi algorithm used here is pure-Rust and compiles
/// directly to `wasm32-unknown-unknown` without any BLAS/LAPACK dependency.
#[wasm_bindgen]
pub fn svd(arr: &WasmArray) -> Result<JsValue, JsValue> {
    if arr.ndim() != 2 {
        return Err(WasmError::InvalidDimensions("SVD requires a 2D matrix".to_string()).into());
    }

    let matrix = arr
        .data()
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e: ndarray::ShapeError| {
            Into::<JsValue>::into(WasmError::ComputationError(e.to_string()))
        })?;

    // Validate: no non-finite entries
    for &v in matrix.iter() {
        if !v.is_finite() {
            return Err(WasmError::InvalidParameter(
                "SVD: matrix contains non-finite values".to_string(),
            )
            .into());
        }
    }

    let output =
        jacobi_svd(&matrix).map_err(|e| Into::<JsValue>::into(WasmError::ComputationError(e)))?;

    serde_wasm_bindgen::to_value(&output).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// One-sided Jacobi SVD.
///
/// Works on `A` (m × n).  Applies Jacobi rotations directly to the columns
/// of `A` until all column-pairs are orthogonal, accumulating the rotations
/// into `V` (n × n).  After convergence:
///   - σ_j = ‖A[:,j]‖₂   (singular values)
///   - U[:,j] = A[:,j] / σ_j  (left singular vectors; zero for σ_j ≈ 0)
///   - Vt = V^T            (right singular vectors transposed)
///
/// Complexity: O(n² · sweeps · m).  Typically 10–25 sweeps for small matrices.
fn jacobi_svd(a: &Array2<f64>) -> Result<SvdOutput, String> {
    let m = a.nrows();
    let n = a.ncols();
    let k = m.min(n); // number of singular values

    if m == 0 || n == 0 {
        return Err("SVD: matrix must have non-zero dimensions".to_string());
    }

    // Work on a flat column-major copy: columns[j] is column j, length m.
    // We store columns as Vec<Vec<f64>> for clarity.
    let mut cols: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..m).map(|i| a[[i, j]]).collect())
        .collect();

    // V: n × n identity — accumulate column Jacobi rotations here.
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0_f64; n];
            row[i] = 1.0;
            row
        })
        .collect();

    // Sweep until convergence (max 50 sweeps).
    const MAX_SWEEPS: usize = 50;
    const TOL: f64 = 1e-14;

    for _sweep in 0..MAX_SWEEPS {
        let mut converged = true;

        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                // Compute dot products: alpha = col_p · col_p, beta = col_q · col_q, gamma = col_p · col_q
                let alpha: f64 = cols[p].iter().map(|&x| x * x).sum();
                let beta: f64 = cols[q].iter().map(|&x| x * x).sum();
                let gamma: f64 = cols[p]
                    .iter()
                    .zip(cols[q].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();

                // Skip if already orthogonal.
                let norm_pq = (alpha * beta).sqrt();
                if norm_pq < TOL || gamma.abs() <= TOL * norm_pq {
                    continue;
                }

                converged = false;

                // Compute Jacobi rotation angle.
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta >= 0.0 {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                } else {
                    -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                // Apply rotation to columns p and q of A.
                // Use split_at_mut so we can mutably borrow two disjoint slices.
                let (lo, hi) = if p < q {
                    let (left, right) = cols.split_at_mut(q);
                    (&mut left[p], &mut right[0])
                } else {
                    let (left, right) = cols.split_at_mut(p);
                    (&mut right[0], &mut left[q])
                };
                for (ap_ref, aq_ref) in lo.iter_mut().zip(hi.iter_mut()) {
                    let ap = *ap_ref;
                    let aq = *aq_ref;
                    *ap_ref = c * ap - s * aq;
                    *aq_ref = s * ap + c * aq;
                }

                // Apply rotation to rows p and q of V (i.e., columns of V^T).
                let (vlo, vhi) = if p < q {
                    let (left, right) = v.split_at_mut(q);
                    (&mut left[p], &mut right[0])
                } else {
                    let (left, right) = v.split_at_mut(p);
                    (&mut right[0], &mut left[q])
                };
                for (vp_ref, vq_ref) in vlo.iter_mut().zip(vhi.iter_mut()) {
                    let vp = *vp_ref;
                    let vq = *vq_ref;
                    *vp_ref = c * vp - s * vq;
                    *vq_ref = s * vp + c * vq;
                }
            }
        }

        if converged {
            break;
        }
    }

    // Extract singular values as column norms.
    let mut sigma: Vec<f64> = (0..n)
        .map(|j| cols[j].iter().map(|&x| x * x).sum::<f64>().sqrt())
        .collect();

    // Sort descending by σ; track permutation to reorder U columns and V rows.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        sigma[b]
            .partial_cmp(&sigma[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sigma_sorted: Vec<f64> = order.iter().map(|&i| sigma[i]).collect();
    let cols_sorted: Vec<Vec<f64>> = order.iter().map(|&i| cols[i].clone()).collect();
    let v_sorted: Vec<Vec<f64>> = order.iter().map(|&i| v[i].clone()).collect();

    // Re-assign for clarity.
    sigma = sigma_sorted;
    let cols = cols_sorted;
    let v = v_sorted;

    // Build U (m × k) in row-major order: U[i][j] = cols[j][i] / sigma[j].
    // Only keep k = min(m, n) singular values.
    let sv: Vec<f64> = sigma[..k].to_vec();
    let mut u_flat: Vec<f64> = vec![0.0_f64; m * k];
    for j in 0..k {
        let s = sigma[j];
        if s > TOL {
            for i in 0..m {
                // Row-major: U[i][j] = u_flat[i*k + j]
                u_flat[i * k + j] = cols[j][i] / s;
            }
        }
        // else: leave as 0.0 (rank-deficient column)
    }

    // Build Vt (k × n) in row-major order: Vt[j][i] = V[i][j] = v[j][i].
    // v[j] is column j of V stored as a row vector of length n, so v[j][i] = V[i][j] = Vt[j][i].
    let mut vt_flat: Vec<f64> = Vec::with_capacity(k * n);
    for row in v.iter().take(k) {
        // row has length n; this is one row of V^T.
        vt_flat.extend_from_slice(row);
    }

    Ok(SvdOutput {
        singular_values: sv,
        u: u_flat,
        u_rows: m,
        u_cols: k,
        vt: vt_flat,
        vt_rows: k,
        vt_cols: n,
    })
}

// ---------------------------------------------------------------------------
// SVD tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod svd_tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn frobenius_err(
        a: &Array2<f64>,
        u: &[f64],
        s: &[f64],
        vt: &[f64],
        m: usize,
        k: usize,
        n: usize,
    ) -> f64 {
        // Reconstruct A from U · diag(s) · Vt and measure ‖A - Â‖_F.
        let mut err_sq = 0.0_f64;
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for r in 0..k {
                    val += u[i * k + r] * s[r] * vt[r * n + j];
                }
                let orig = a[[i, j]];
                err_sq += (orig - val).powi(2);
            }
        }
        err_sq.sqrt()
    }

    #[test]
    fn test_svd_identity_3x3() {
        let a = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let result = jacobi_svd(&a).expect("jacobi_svd identity 3x3");
        assert_eq!(result.singular_values.len(), 3);
        let mut sv = result.singular_values.clone();
        sv.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        for &s in &sv {
            assert!((s - 1.0).abs() < 1e-9, "singular value {s} should be 1.0");
        }
    }

    #[test]
    fn test_svd_diagonal_3x3() {
        let a = array![[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        let result = jacobi_svd(&a).expect("jacobi_svd diagonal 3x3");
        let sv = &result.singular_values;
        assert_eq!(sv.len(), 3);
        // Descending order
        assert!((sv[0] - 3.0).abs() < 1e-9, "sv[0]={}", sv[0]);
        assert!((sv[1] - 2.0).abs() < 1e-9, "sv[1]={}", sv[1]);
        assert!((sv[2] - 1.0).abs() < 1e-9, "sv[2]={}", sv[2]);
    }

    #[test]
    fn test_svd_reconstruction_4x3() {
        // Non-square tall matrix
        let a = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];
        let m = 4;
        let k = 3;
        let n = 3;
        let result = jacobi_svd(&a).expect("jacobi_svd 4x3");
        assert_eq!(result.u_rows, m);
        assert_eq!(result.u_cols, k);
        assert_eq!(result.vt_rows, k);
        assert_eq!(result.vt_cols, n);
        let err = frobenius_err(&a, &result.u, &result.singular_values, &result.vt, m, k, n);
        assert!(err < 1e-9, "Frobenius reconstruction error = {err}");
    }

    #[test]
    fn test_svd_reconstruction_2x4() {
        // Non-square wide matrix
        let a = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let m = 2;
        let k = 2;
        let n = 4;
        let result = jacobi_svd(&a).expect("jacobi_svd 2x4");
        assert_eq!(result.u_rows, m);
        assert_eq!(result.u_cols, k);
        assert_eq!(result.vt_rows, k);
        assert_eq!(result.vt_cols, n);
        let err = frobenius_err(&a, &result.u, &result.singular_values, &result.vt, m, k, n);
        assert!(err < 1e-9, "Frobenius reconstruction error = {err}");
    }

    #[test]
    fn test_svd_rank_deficient_2x2() {
        // Rank-1 matrix: [[1,2],[2,4]]
        let a = array![[1.0, 2.0], [2.0, 4.0]];
        let result = jacobi_svd(&a).expect("jacobi_svd rank-deficient 2x2");
        let sv = &result.singular_values;
        // One non-zero singular value, one near-zero.
        let max_sv = sv.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_sv = sv.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            max_sv > 0.1,
            "max singular value should be significant: {max_sv}"
        );
        assert!(
            min_sv < 1e-10,
            "min singular value should be near-zero: {min_sv}"
        );
    }

    #[test]
    fn test_svd_non_finite_rejected() {
        let a = array![[1.0, f64::NAN], [2.0, 3.0]];
        let wa = WasmArray::from_array(a.into_dyn());
        let result = svd(&wa);
        assert!(result.is_err(), "SVD with NaN should return Err");
    }
}
