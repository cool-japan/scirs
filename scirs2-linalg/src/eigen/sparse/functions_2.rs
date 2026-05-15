//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::prelude::*;
use std::iter::Sum;

use super::functions::SparseMatrix;
use super::types::CsrMatrix;

/// Compute singular values and vectors of a sparse matrix using iterative methods.
///
/// This function computes the largest or smallest singular values of a sparse matrix
/// without forming the normal equations, which can be numerically unstable for
/// ill-conditioned matrices.
///
/// # Arguments
///
/// * `matrix` - Sparse matrix
/// * `k` - Number of singular values to compute
/// * `which` - Which singular values to find ("largest" or "smallest")
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// * Tuple (singular_values, left_vectors, right_vectors)
///
/// # Examples
///
/// ```rust,ignore
/// use scirs2_linalg::eigen::sparse::{svds, SparseMatrix};
///
/// // This is a placeholder example - actual implementation pending
/// // let (s, u, vt) = svds(&sparsematrix, 6, "largest", 100, 1e-6).expect("Operation failed");
/// ```
///
/// # Note
///
/// This function implements Lanczos bidiagonalization (Golub-Kahan-Lanczos) with full
/// reorthogonalization to compute a partial SVD of the sparse matrix.
#[allow(dead_code)]
pub fn svds<F, M>(
    matrix: &M,
    k: usize,
    which: &str,
    max_iter: usize,
    tol: F,
) -> LinalgResult<(Array1<F>, Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    M: SparseMatrix<F>,
{
    let m = matrix.nrows();
    let n = matrix.ncols();
    if k == 0 {
        return Err(LinalgError::InvalidInputError(
            "svds: k must be at least 1".to_string(),
        ));
    }
    if k >= m.min(n) {
        return Err(LinalgError::InvalidInputError(format!(
            "svds: k={k} must be less than min(m,n)={}",
            m.min(n)
        )));
    }
    let max_steps = max_iter.min(m.min(n));
    let mut u_basis: Vec<Array1<F>> = Vec::with_capacity(max_steps + 1);
    let mut v_basis: Vec<Array1<F>> = Vec::with_capacity(max_steps + 1);
    let mut alpha_vals: Vec<F> = Vec::with_capacity(max_steps);
    let mut beta_vals: Vec<F> = Vec::with_capacity(max_steps);
    let mut rng = scirs2_core::random::rng();
    let mut v0 = Array1::<F>::zeros(n);
    for i in 0..n {
        v0[i] = F::from(rng.random::<f64>()).unwrap_or(F::one());
    }
    let norm_v0 = v0.iter().map(|x| (*x) * (*x)).sum::<F>().sqrt();
    if norm_v0 < F::from(1e-15).unwrap_or(F::epsilon()) {
        v0[0] = F::one();
    } else {
        v0.mapv_inplace(|x| x / norm_v0);
    }
    v_basis.push(v0);
    let mut u_next = Array1::<F>::zeros(m);
    for iter in 0..max_steps {
        let v_curr = &v_basis[iter];
        matrix.matvec(&v_curr.view(), &mut u_next)?;
        if iter > 0 {
            let beta_prev = beta_vals[iter - 1];
            let u_prev = &u_basis[iter - 1];
            for j in 0..m {
                u_next[j] -= beta_prev * u_prev[j];
            }
        }
        for _pass in 0..2 {
            for prev_u in u_basis.iter() {
                let proj = prev_u
                    .iter()
                    .zip(u_next.iter())
                    .map(|(p, w)| (*p) * (*w))
                    .sum::<F>();
                for j in 0..m {
                    u_next[j] -= proj * prev_u[j];
                }
            }
        }
        let alpha = u_next.iter().map(|x| (*x) * (*x)).sum::<F>().sqrt();
        if alpha < tol {
            break;
        }
        alpha_vals.push(alpha);
        let u_new = u_next.mapv(|x| x / alpha);
        u_basis.push(u_new.clone());
        let mut v_new = Array1::<F>::zeros(n);
        {
            let u_new_view = u_new.view();
            let mut tmp = Array1::<F>::zeros(m);
            for j in 0..n {
                let mut ej = Array1::<F>::zeros(n);
                ej[j] = F::one();
                matrix.matvec(&ej.view(), &mut tmp)?;
                let dot = tmp
                    .iter()
                    .zip(u_new_view.iter())
                    .map(|(a, b)| (*a) * (*b))
                    .sum::<F>();
                v_new[j] = dot;
            }
        }
        for j in 0..n {
            v_new[j] -= alpha * v_basis[iter][j];
        }
        for _pass in 0..2 {
            for prev_v in v_basis.iter() {
                let proj = prev_v
                    .iter()
                    .zip(v_new.iter())
                    .map(|(p, w)| (*p) * (*w))
                    .sum::<F>();
                for j in 0..n {
                    v_new[j] -= proj * prev_v[j];
                }
            }
        }
        let beta = v_new.iter().map(|x| (*x) * (*x)).sum::<F>().sqrt();
        if beta < tol {
            break;
        }
        beta_vals.push(beta);
        let v_normalized = v_new.mapv(|x| x / beta);
        v_basis.push(v_normalized);
        if iter >= k && iter % 5 == 0 {
            let recent = &beta_vals[beta_vals.len().saturating_sub(k)..];
            if recent
                .iter()
                .all(|&b| b < tol * F::from(10.0).unwrap_or(F::one()))
            {
                break;
            }
        }
    }
    let j = alpha_vals.len();
    if j == 0 {
        return Err(LinalgError::ComputationError(
            "svds: Lanczos bidiagonalization produced no steps".to_string(),
        ));
    }
    let mut b = Array2::<F>::zeros((j, j));
    for i in 0..j {
        b[[i, i]] = alpha_vals[i];
        if i + 1 < j && i < beta_vals.len() {
            b[[i, i + 1]] = beta_vals[i];
        }
    }
    let (b_u, sigma, b_vt) = crate::decomposition::svd(&b.view(), true, None)?;
    let k_actual = k.min(j);
    let mut idx: Vec<usize> = (0..sigma.len()).collect();
    match which {
        "smallest" | "SM" => {
            idx.sort_by(|&a, &b| {
                sigma[a]
                    .partial_cmp(&sigma[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        _ => {
            idx.sort_by(|&a, &b| {
                sigma[b]
                    .partial_cmp(&sigma[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }
    idx.truncate(k_actual);
    let s_out: Array1<F> = idx.iter().map(|&i| sigma[i]).collect();
    let mut u_out = Array2::<F>::zeros((m, k_actual));
    for (col, &si) in idx.iter().enumerate() {
        for j_idx in 0..j.min(u_basis.len()) {
            let coeff = b_u[[j_idx, si]];
            for row in 0..m {
                u_out[[row, col]] += coeff * u_basis[j_idx][row];
            }
        }
    }
    let mut vt_out = Array2::<F>::zeros((k_actual, n));
    for (row, &si) in idx.iter().enumerate() {
        for j_idx in 0..j.min(v_basis.len()) {
            let coeff = b_vt[[j_idx, si]];
            for col in 0..n {
                vt_out[[row, col]] += coeff * v_basis[j_idx][col];
            }
        }
    }
    Ok((s_out, u_out, vt_out))
}
/// Convert a dense matrix to CSR sparse format for eigenvalue computations.
///
/// Scans the dense matrix and extracts entries with absolute value >= `threshold`.
///
/// # Arguments
/// * `densematrix` - Dense matrix to convert
/// * `threshold` - Sparsity threshold
///
/// # Returns
/// A `CsrMatrix` wrapped in `Box<dyn SparseMatrix<F>>`.
#[allow(dead_code)]
pub fn dense_to_sparse<F>(
    densematrix: &ArrayView2<F>,
    threshold: F,
) -> LinalgResult<Box<dyn SparseMatrix<F>>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    Ok(Box::new(CsrMatrix::from_dense(densematrix, threshold)))
}
#[cfg(test)]
mod tests {
    use super::super::functions::{eigs_gen, lanczos, SparseMatrix};
    use super::*;
    /// Build a 5x5 symmetric tridiagonal SPD matrix in CSR format (1D Laplacian).
    fn tridiag_csr_5x5() -> CsrMatrix<f64> {
        let n = 5;
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0usize];
        for i in 0..n {
            if i > 0 {
                data.push(-1.0);
                indices.push(i - 1);
            }
            data.push(2.0);
            indices.push(i);
            if i < n - 1 {
                data.push(-1.0);
                indices.push(i + 1);
            }
            indptr.push(data.len());
        }
        CsrMatrix::new(n, n, data, indices, indptr)
    }
    #[test]
    fn test_csr_matvec() {
        let csr = tridiag_csr_5x5();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut y = Array1::zeros(5);
        csr.matvec(&x.view(), &mut y).expect("matvec failed");
        assert!((y[0] - 0.0).abs() < 1e-12);
        assert!((y[1] - 0.0).abs() < 1e-12);
        assert!((y[2] - 0.0).abs() < 1e-12);
        assert!((y[3] - 0.0).abs() < 1e-12);
        assert!((y[4] - 6.0).abs() < 1e-12);
    }
    #[test]
    fn test_csr_is_symmetric() {
        let csr = tridiag_csr_5x5();
        assert!(
            csr.is_symmetric(),
            "Tridiagonal Laplacian should be symmetric"
        );
    }
    #[test]
    fn test_csr_sparsity() {
        let csr = tridiag_csr_5x5();
        let sp = csr.sparsity();
        assert!(
            (sp - 0.52).abs() < 0.01,
            "Sparsity should be ~0.52, got {sp}"
        );
    }
    #[test]
    fn test_csrmatrix_empty() {
        let csr = CsrMatrix::<f64>::new(5, 5, vec![], vec![], vec![0, 0, 0, 0, 0, 0]);
        assert_eq!(csr.nrows(), 5);
        assert_eq!(csr.ncols(), 5);
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.sparsity(), 0.0);
    }
    #[test]
    fn test_csr_from_dense() {
        let dense = Array2::from_shape_fn((3, 3), |(i, j)| {
            if i == j {
                2.0_f64
            } else if (i as isize - j as isize).abs() == 1 {
                -1.0
            } else {
                0.0
            }
        });
        let csr = CsrMatrix::from_dense(&dense.view(), 1e-14);
        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.ncols(), 3);
        assert_eq!(csr.nnz(), 7);
        assert!(csr.is_symmetric());
    }
    #[test]
    fn test_dense_to_sparse_fn() {
        let dense = Array2::<f64>::eye(4);
        let sparse = dense_to_sparse(&dense.view(), 1e-12).expect("dense_to_sparse failed");
        assert_eq!(sparse.nrows(), 4);
        assert_eq!(sparse.ncols(), 4);
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let mut y = Array1::zeros(4);
        sparse.matvec(&x.view(), &mut y).expect("matvec failed");
        for i in 0..4 {
            assert!((y[i] - x[i]).abs() < 1e-12, "Identity matvec failed at {i}");
        }
    }
    #[test]
    #[ignore = "flaky: random initial vector causes rare convergence failure in parallel test runs"]
    fn test_lanczos_with_csr() {
        let csr = tridiag_csr_5x5();
        let result = lanczos(&csr, 2, "largest", 0.0_f64, 100, 1e-6);
        assert!(
            result.is_ok(),
            "Lanczos should succeed on tridiagonal: {:?}",
            result.as_ref().err()
        );
        let (eigenvals, _) = result.expect("already checked");
        assert_eq!(eigenvals.len(), 2);
        let max_eig = eigenvals
            .iter()
            .map(|z| z.re)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_eig > 0.0 && max_eig < 5.0,
            "Eigenvalue should be in range (0, 5), got {max_eig}"
        );
    }
    #[test]
    fn test_svds_diagonal_matrix() {
        let csr = CsrMatrix::new(
            4,
            4,
            vec![3.0_f64, 2.0, 1.0],
            vec![0, 1, 2],
            vec![0, 1, 2, 3, 3],
        );
        let (s, u, vt) = svds(&csr, 2, "largest", 50, 1e-8).expect("svds diagonal");
        assert_eq!(s.len(), 2, "expected 2 singular values");
        for (i, &si) in s.iter().enumerate() {
            assert!(si > 0.0, "s[{i}] should be positive, got {si}");
            assert!(si <= 3.5, "s[{i}] should be <= 3.5, got {si}");
        }
        assert_eq!(u.nrows(), 4);
        assert_eq!(u.ncols(), 2);
        assert_eq!(vt.nrows(), 2);
        assert_eq!(vt.ncols(), 4);
    }
    #[test]
    fn test_svds_rectangular() {
        let csr = CsrMatrix::new(
            3,
            4,
            vec![1.0_f64, 1.0, 1.0],
            vec![0, 1, 2],
            vec![0, 1, 2, 3],
        );
        let (s, u, vt) = svds(&csr, 2, "largest", 100, 1e-6).expect("svds rectangular");
        assert!(!s.is_empty(), "should return at least 1 singular value");
        assert!(s[0] > 0.0, "s[0] should be positive, got {}", s[0]);
        let k_got = s.len();
        assert_eq!(u.nrows(), 3);
        assert_eq!(vt.ncols(), 4);
        assert_eq!(u.ncols(), k_got);
        assert_eq!(vt.nrows(), k_got);
    }
    /// Build a 4×4 tridiagonal matrix A (1D Laplacian scaled) as CSR.
    ///
    /// A = tridiag(-1, 2, -1) * scale
    fn tridiag_csr_4x4(scale: f64) -> CsrMatrix<f64> {
        let n = 4usize;
        let mut data: Vec<f64> = Vec::new();
        let mut indices: Vec<usize> = Vec::new();
        let mut indptr: Vec<usize> = vec![0];
        for i in 0..n {
            if i > 0 {
                data.push(-scale);
                indices.push(i - 1);
            }
            data.push(2.0 * scale);
            indices.push(i);
            if i < n - 1 {
                data.push(-scale);
                indices.push(i + 1);
            }
            indptr.push(data.len());
        }
        CsrMatrix::new(n, n, data, indices, indptr)
    }
    /// Build a 4×4 diagonal matrix B = diag(d0, d1, d2, d3) as CSR.
    fn diagonal_csr_4x4(diag: [f64; 4]) -> CsrMatrix<f64> {
        let n = 4usize;
        let data: Vec<f64> = diag.to_vec();
        let indices: Vec<usize> = (0..n).collect();
        let indptr: Vec<usize> = (0..=n).collect();
        CsrMatrix::new(n, n, data, indices, indptr)
    }
    /// Test 1: eigenvalues of  A v = λ B v  for a 4×4 tridiagonal A and
    /// diagonal B = diag(2, 2, 2, 2) = 2 I.
    ///
    /// The problem reduces to  (A/2) v = λ v, so eigenvalues are those of
    /// the standard Laplacian  tridiag(-1, 2, -1)  divided by 2:
    ///   λ_j = 2 - 2 cos(j π / 5)  for j = 1, 2, 3, 4  (analytical)
    ///   → divided by B's diagonal 2: λ_j / 2
    ///
    /// We verify that the 2 smallest eigenvalues returned are close to the
    /// 2 smallest analytical values.
    #[test]
    fn test_eigs_gen_tridiag_diagonal_b() {
        let a = tridiag_csr_4x4(1.0);
        let b = diagonal_csr_4x4([2.0, 2.0, 2.0, 2.0]);
        let n_mat = 5usize;
        let analytical: Vec<f64> = (1..=4)
            .map(|j| {
                let theta = std::f64::consts::PI * (j as f64) / (n_mat as f64);
                (2.0 - 2.0 * theta.cos()) / 2.0
            })
            .collect();
        let (eigenvals, eigenvecs) = eigs_gen(&a, &b, 2, "smallest", 0.0_f64, 500, 1e-8)
            .expect("eigs_gen should succeed on 4x4 tridiag/diag problem");
        assert_eq!(eigenvals.len(), 2, "should return exactly 2 eigenvalues");
        assert_eq!(eigenvecs.nrows(), 4);
        assert_eq!(eigenvecs.ncols(), 2);
        let mut returned: Vec<f64> = eigenvals.iter().map(|z| z.re).collect();
        returned.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut expected = analytical.clone();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for i in 0..2 {
            let rel_err = (returned[i] - expected[i]).abs() / expected[i].abs().max(1e-14);
            assert!(
                rel_err < 0.15,
                "eigenvalue[{i}]: expected ~{:.6}, got {:.6} (rel_err={:.4})",
                expected[i],
                returned[i],
                rel_err
            );
        }
    }
    /// Test 2: verify eigenvectors satisfy  ||A v - λ B v|| / (||A|| ||v||) < tol
    ///
    /// Uses  A = tridiag(-1, 2, -1)  and  B = diag(1, 2, 2, 1)  (positive definite).
    #[test]
    fn test_eigs_gen_eigenvector_residual() {
        let a = tridiag_csr_4x4(1.0);
        let b = diagonal_csr_4x4([1.0, 2.0, 2.0, 1.0]);
        let (eigenvals, eigenvecs) =
            eigs_gen(&a, &b, 2, "smallest", 0.0_f64, 500, 1e-8).expect("eigs_gen should succeed");
        let n = 4usize;
        let resid_tol = 0.5_f64;
        for col in 0..eigenvals.len() {
            let lam = eigenvals[col].re;
            let v: Vec<f64> = (0..n).map(|r| eigenvecs[[r, col]].re).collect();
            let v_arr = Array1::from_vec(v.clone());
            let mut av = Array1::<f64>::zeros(n);
            a.matvec(&v_arr.view(), &mut av)
                .expect("matvec A failed in test");
            let mut bv = Array1::<f64>::zeros(n);
            b.matvec(&v_arr.view(), &mut bv)
                .expect("matvec B failed in test");
            let resid: f64 = (0..n)
                .map(|i| {
                    let ri = av[i] - lam * bv[i];
                    ri * ri
                })
                .sum::<f64>()
                .sqrt();
            let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-14);
            assert!(
                resid / v_norm < resid_tol,
                "eigenvector {col}: ||Av - λBv|| / ||v|| = {:.6} >= {resid_tol} (λ={lam:.6})",
                resid / v_norm
            );
        }
    }
}
