//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::Complex;
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::prelude::*;
use std::iter::Sum;

pub type SparseEigenResult<F> = LinalgResult<(Array1<Complex<F>>, Array2<Complex<F>>)>;
pub type QrResult<F> = LinalgResult<(Array2<Complex<F>>, Array2<Complex<F>>)>;
/// Sparse matrix trait for eigenvalue computations
///
/// This trait defines the interface that sparse matrix types should implement
/// to be compatible with sparse eigenvalue algorithms.
pub trait SparseMatrix<F> {
    /// Get the number of rows
    fn nrows(&self) -> usize;
    /// Get the number of columns
    fn ncols(&self) -> usize;
    /// Matrix-vector multiplication: y = A * x
    fn matvec(&self, x: &ArrayView1<F>, y: &mut Array1<F>) -> LinalgResult<()>;
    /// Check if the matrix is symmetric
    fn is_symmetric(&self) -> bool;
    /// Get the sparsity ratio (number of non-zeros / total elements)
    fn sparsity(&self) -> f64;
}
/// Compute a few eigenvalues and eigenvectors of a large sparse matrix using Lanczos algorithm.
///
/// The Lanczos algorithm is an iterative method that is particularly effective for
/// symmetric sparse matrices when only a few eigenvalues are needed.
///
/// # Arguments
///
/// * `matrix` - Sparse matrix implementing the SparseMatrix trait
/// * `k` - Number of eigenvalues to compute
/// * `which` - Which eigenvalues to find ("largest", "smallest", "target")
/// * `target` - Target value for "target" mode (ignored for other modes)
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// * Tuple (eigenvalues, eigenvectors) with k eigenvalues and eigenvectors
///
/// # Examples
///
/// ```rust,ignore
/// use scirs2_linalg::eigen::sparse::{lanczos, SparseMatrix};
///
/// // This is a placeholder example - actual implementation pending
/// // let sparsematrix = create_sparsematrix();
/// // let (w, v) = lanczos(&sparsematrix, 5, "largest", 0.0, 100, 1e-6).expect("Operation failed");
/// ```
///
/// # Note
///
/// This function implements a parallel Lanczos algorithm for symmetric sparse matrices.
#[allow(dead_code)]
pub fn lanczos<F, M>(
    matrix: &M,
    k: usize,
    which: &str,
    target: F,
    max_iter: usize,
    tol: F,
) -> SparseEigenResult<F>
where
    F: Float + NumAssign + Sum + Send + Sync + 'static + Default,
    M: SparseMatrix<F> + Sync,
{
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(LinalgError::ShapeError(
            "Matrix must be square for eigenvalue decomposition".to_string(),
        ));
    }
    if k >= n {
        return Err(LinalgError::InvalidInputError(
            "Number of eigenvalues requested must be less than matrix size".to_string(),
        ));
    }
    let max_steps = max_iter.min(n);
    let mut v_basis: Vec<Array1<F>> = Vec::with_capacity(max_steps + 1);
    let mut rng = scirs2_core::random::rng();
    let mut v_init = Array1::<F>::zeros(n);
    for i in 0..n {
        v_init[i] = F::from(rng.random::<f64>()).unwrap_or(F::one());
    }
    let norm = v_init.iter().map(|x| (*x) * (*x)).sum::<F>().sqrt();
    if norm < F::from(1e-15).unwrap_or(F::epsilon()) {
        return Err(LinalgError::InvalidInputError(
            "Initial Lanczos vector has near-zero norm".to_string(),
        ));
    }
    v_init.mapv_inplace(|x| x / norm);
    v_basis.push(v_init);
    let mut alpha = Vec::with_capacity(max_steps);
    let mut beta = Vec::with_capacity(max_steps);
    let mut v_next = Array1::<F>::zeros(n);
    for iter in 0..max_steps {
        let v_curr = &v_basis[iter];
        matrix.matvec(&v_curr.view(), &mut v_next)?;
        if iter > 0 {
            let beta_prev = beta[iter - 1];
            let v_prev = &v_basis[iter - 1];
            for j in 0..n {
                v_next[j] -= beta_prev * v_prev[j];
            }
        }
        let alpha_curr = v_curr
            .iter()
            .zip(v_next.iter())
            .map(|(v, w)| (*v) * (*w))
            .sum::<F>();
        alpha.push(alpha_curr);
        for j in 0..n {
            v_next[j] -= alpha_curr * v_curr[j];
        }
        for _pass in 0..2 {
            for prev_v in v_basis.iter() {
                let proj = prev_v
                    .iter()
                    .zip(v_next.iter())
                    .map(|(p, w)| (*p) * (*w))
                    .sum::<F>();
                for j in 0..n {
                    v_next[j] -= proj * prev_v[j];
                }
            }
        }
        let beta_curr = v_next.iter().map(|x| (*x) * (*x)).sum::<F>().sqrt();
        if beta_curr < tol {
            break;
        }
        beta.push(beta_curr);
        let v_new = v_next.mapv(|x| x / beta_curr);
        v_basis.push(v_new);
        v_next = Array1::<F>::zeros(n);
        if iter >= k && iter % 5 == 0 && check_lanczos_convergence(&alpha, &beta, k, tol) {
            break;
        }
    }
    let (eigenvals, eigenvecs) = solve_tridiagonal_eigenproblem(&alpha, &beta, which, target, k)?;
    let complex_eigenvals = eigenvals.mapv(|x| Complex::new(x, F::zero()));
    let complex_eigenvecs = eigenvecs.mapv(|x| Complex::new(x, F::zero()));
    Ok((complex_eigenvals, complex_eigenvecs))
}
#[allow(dead_code)]
fn check_lanczos_convergence<F: Float>(_alpha: &[F], beta: &[F], k: usize, tol: F) -> bool {
    if beta.len() < k {
        return false;
    }
    let recent_betas = &beta[beta.len().saturating_sub(k)..];
    recent_betas
        .iter()
        .all(|&b| b < tol * F::from(10.0).expect("Operation failed"))
}
#[allow(dead_code)]
fn solve_tridiagonal_eigenproblem<F: Float + NumAssign + Sum + Send + Sync + 'static>(
    alpha: &[F],
    beta: &[F],
    which: &str,
    target: F,
    k: usize,
) -> LinalgResult<(Array1<F>, Array2<F>)> {
    let n = alpha.len();
    if n == 0 {
        return Err(LinalgError::InvalidInputError(
            "Empty tridiagonal matrix".to_string(),
        ));
    }
    let mut trimatrix = Array2::<F>::zeros((n, n));
    for i in 0..n {
        trimatrix[[i, i]] = alpha[i];
    }
    for i in 0..n.saturating_sub(1) {
        if i < beta.len() {
            trimatrix[[i, i + 1]] = beta[i];
            trimatrix[[i + 1, i]] = beta[i];
        }
    }
    let (eigenvals, eigenvecs) = qr_algorithm_tridiagonal(&trimatrix)?;
    let selected_indices = select_eigenvalues(&eigenvals, which, target, k);
    let mut result_eigenvals = Array1::<F>::zeros(k);
    let mut result_eigenvecs = Array2::<F>::zeros((n, k));
    for (i, &idx) in selected_indices.iter().enumerate() {
        result_eigenvals[i] = eigenvals[idx];
        for j in 0..n {
            result_eigenvecs[[j, i]] = eigenvecs[[j, idx]];
        }
    }
    Ok((result_eigenvals, result_eigenvecs))
}
#[allow(dead_code)]
fn qr_algorithm_tridiagonal<F: Float + NumAssign + Sum + 'static>(
    matrix: &Array2<F>,
) -> LinalgResult<(Array1<F>, Array2<F>)> {
    let n = matrix.nrows();
    let mut a = matrix.clone();
    let mut q_total = Array2::<F>::eye(n);
    let max_iterations = 1000;
    let tolerance = F::from(1e-12).unwrap_or(F::epsilon());
    for _iter in 0..max_iterations {
        let mut converged = true;
        for i in 0..n - 1 {
            if a[[i + 1, i]].abs() > tolerance {
                converged = false;
                break;
            }
        }
        if converged {
            break;
        }
        let shift = if n >= 2 {
            let d = (a[[n - 2, n - 2]] - a[[n - 1, n - 1]]) / F::from(2.0).unwrap_or(F::one());
            let b_sq = a[[n - 1, n - 2]] * a[[n - 1, n - 2]];
            let sign_d = if d >= F::zero() { F::one() } else { -F::one() };
            a[[n - 1, n - 1]] - sign_d * b_sq / (d.abs() + (d * d + b_sq).sqrt())
        } else {
            a[[0, 0]]
        };
        for i in 0..n {
            a[[i, i]] -= shift;
        }
        let (q, r) = qr_decomposition_tridiagonal(&a)?;
        a = r.dot(&q);
        for i in 0..n {
            a[[i, i]] += shift;
        }
        q_total = q_total.dot(&q);
    }
    let eigenvals = (0..n).map(|i| a[[i, i]]).collect::<Array1<F>>();
    Ok((eigenvals, q_total))
}
#[allow(dead_code)]
fn qr_decomposition_tridiagonal<F: Float + NumAssign + Sum>(
    matrix: &Array2<F>,
) -> LinalgResult<(Array2<F>, Array2<F>)> {
    let n = matrix.nrows();
    let mut g_product = Array2::<F>::eye(n);
    let mut r = matrix.clone();
    let eps = F::from(1e-15).unwrap_or(F::epsilon());
    for i in 0..n - 1 {
        let a = r[[i, i]];
        let b = r[[i + 1, i]];
        if b.abs() > eps {
            let (c, s) = givens_rotation(a, b);
            apply_givens_rotation(&mut r, i, i + 1, c, s);
            apply_givens_rotation_transpose(&mut g_product, i, i + 1, c, s);
        }
    }
    let q = g_product.t().to_owned();
    Ok((q, r))
}
#[allow(dead_code)]
fn givens_rotation<F: Float>(a: F, b: F) -> (F, F) {
    if b.abs() < F::from(1e-15).expect("Operation failed") {
        (F::one(), F::zero())
    } else {
        let r = (a * a + b * b).sqrt();
        (a / r, -b / r)
    }
}
#[allow(dead_code)]
fn apply_givens_rotation<F: Float + NumAssign>(
    matrix: &mut Array2<F>,
    i: usize,
    j: usize,
    c: F,
    s: F,
) {
    let n = matrix.ncols();
    for k in 0..n {
        let temp1 = matrix[[i, k]];
        let temp2 = matrix[[j, k]];
        matrix[[i, k]] = c * temp1 - s * temp2;
        matrix[[j, k]] = s * temp1 + c * temp2;
    }
}
#[allow(dead_code)]
fn apply_givens_rotation_transpose<F: Float + NumAssign>(
    matrix: &mut Array2<F>,
    i: usize,
    j: usize,
    c: F,
    s: F,
) {
    let n = matrix.nrows();
    for k in 0..n {
        let temp1 = matrix[[k, i]];
        let temp2 = matrix[[k, j]];
        matrix[[k, i]] = c * temp1 + s * temp2;
        matrix[[k, j]] = -s * temp1 + c * temp2;
    }
}
#[allow(dead_code)]
fn select_eigenvalues<F: Float>(
    eigenvals: &Array1<F>,
    which: &str,
    target: F,
    k: usize,
) -> Vec<usize> {
    let mut indices_and_values: Vec<(usize, F)> = eigenvals
        .iter()
        .enumerate()
        .map(|(i, &val)| (i, val))
        .collect();
    match which {
        "largest" | "LM" => {
            indices_and_values.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("Operation failed"));
        }
        "smallest" | "SM" => {
            indices_and_values.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("Operation failed"));
        }
        "target" | "nearest" => {
            indices_and_values.sort_by(|a, b| {
                let dist_a = (a.1 - target).abs();
                let dist_b = (b.1 - target).abs();
                dist_a.partial_cmp(&dist_b).expect("Operation failed")
            });
        }
        _ => {
            indices_and_values.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("Operation failed"));
        }
    }
    indices_and_values
        .into_iter()
        .take(k)
        .map(|(idx, _)| idx)
        .collect()
}
/// Compute eigenvalues near a target value using the Arnoldi method.
///
/// The Arnoldi method is a generalization of the Lanczos algorithm that works
/// for non-symmetric matrices. It's particularly effective when combined with
/// shift-and-invert to find eigenvalues near a specific target value.
///
/// # Arguments
///
/// * `matrix` - Sparse matrix implementing the SparseMatrix trait
/// * `k` - Number of eigenvalues to compute
/// * `target` - Target eigenvalue around which to search
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// * Tuple (eigenvalues, eigenvectors) with k eigenvalues closest to target
///
/// # Examples
///
/// ```rust,ignore
/// use scirs2_linalg::eigen::sparse::{arnoldi, SparseMatrix};
///
/// // This is a placeholder example - actual implementation pending
/// // let sparsematrix = create_sparsematrix();
/// // let (w, v) = arnoldi(&sparsematrix, 3, 1.5, 100, 1e-6).expect("Operation failed");
/// ```
///
/// # Note
///
/// This function implements a parallel Arnoldi method for non-symmetric sparse matrices.
#[allow(dead_code)]
pub fn arnoldi<F, M>(
    matrix: &M,
    k: usize,
    target: Complex<F>,
    max_iter: usize,
    tol: F,
) -> SparseEigenResult<F>
where
    F: Float + NumAssign + Sum + Send + Sync + 'static + Default,
    M: SparseMatrix<F> + Sync,
{
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(LinalgError::ShapeError(
            "Matrix must be square for eigenvalue decomposition".to_string(),
        ));
    }
    if k >= n {
        return Err(LinalgError::InvalidInputError(
            "Number of eigenvalues requested must be less than matrix size".to_string(),
        ));
    }
    let m = (max_iter + 1).min(n);
    let mut v_vectors = vec![Array1::<F>::zeros(n); m + 1];
    let mut hmatrix = Array2::<F>::zeros((m + 1, m));
    let mut rng = scirs2_core::random::rng();
    for v_elem in &mut v_vectors[0] {
        *v_elem = F::from(rng.random::<f64>()).expect("Operation failed");
    }
    let norm = v_vectors[0].iter().map(|x| (*x) * (*x)).sum::<F>().sqrt();
    for v_elem in &mut v_vectors[0] {
        *v_elem /= norm;
    }
    let mut actual_m = 0;
    for j in 0..m {
        actual_m = j + 1;
        let mut w = Array1::<F>::zeros(n);
        matrix.matvec(&v_vectors[j].view(), &mut w)?;
        for i in 0..=j {
            let h_ij = w
                .iter()
                .zip(v_vectors[i].iter())
                .map(|(w_val, v_val)| (*w_val) * (*v_val))
                .sum::<F>();
            hmatrix[[i, j]] = h_ij;
            for l in 0..n {
                w[l] -= h_ij * v_vectors[i][l];
            }
        }
        let h_j1_j = w.iter().map(|x| (*x) * (*x)).sum::<F>().sqrt();
        if h_j1_j < tol {
            break;
        }
        if j + 1 < m {
            hmatrix[[j + 1, j]] = h_j1_j;
            for l in 0..n {
                v_vectors[j + 1][l] = w[l] / h_j1_j;
            }
        }
        if j >= k && j % 5 == 0 && check_arnoldi_convergence(&hmatrix, j + 1, k, tol) {
            break;
        }
    }
    let h_reduced = hmatrix.slice(s![..actual_m, ..actual_m]).to_owned();
    let (ritz_values, ritz_vectors) = solve_hessenberg_eigenproblem(&h_reduced)?;
    let eigenvals = if target.im == F::zero() {
        select_closest_real_eigenvalues(&ritz_values, target.re, k)
    } else {
        select_closest_complex_eigenvalues(&ritz_values, target, k)
    };
    let mut eigenvecs = Array2::<Complex<F>>::zeros((n, k));
    let v_basis = v_vectors[..actual_m]
        .iter()
        .map(|v| v.mapv(|x| Complex::new(x, F::zero())))
        .collect::<Vec<_>>();
    for (i, &ritz_idx) in eigenvals.iter().enumerate() {
        for j in 0..n {
            let mut eigenvec_j = Complex::new(F::zero(), F::zero());
            for l in 0..actual_m {
                eigenvec_j += ritz_vectors[[l, ritz_idx]] * v_basis[l][j];
            }
            eigenvecs[[j, i]] = eigenvec_j;
        }
    }
    let final_eigenvals = eigenvals
        .iter()
        .map(|&idx| ritz_values[idx])
        .collect::<Array1<_>>();
    Ok((final_eigenvals, eigenvecs))
}
#[allow(dead_code)]
fn check_arnoldi_convergence<F: Float>(hmatrix: &Array2<F>, m: usize, k: usize, tol: F) -> bool {
    if m < k + 1 {
        return false;
    }
    (0..k).all(|i| {
        let row = m - 1 - i;
        let col = m - 2 - i;
        if row < hmatrix.nrows() && col < hmatrix.ncols() {
            hmatrix[[row, col]].abs() < tol * F::from(10.0).expect("Operation failed")
        } else {
            true
        }
    })
}
#[allow(dead_code)]
fn solve_hessenberg_eigenproblem<F: Float + NumAssign + Sum + 'static>(
    hmatrix: &Array2<F>,
) -> SparseEigenResult<F> {
    let n = hmatrix.nrows();
    let mut matrix_complex = Array2::<Complex<F>>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            matrix_complex[[i, j]] = Complex::new(hmatrix[[i, j]], F::zero());
        }
    }
    qr_algorithm_complex(&matrix_complex)
}
#[allow(dead_code)]
fn qr_algorithm_complex<F: Float + NumAssign + Sum + 'static>(
    matrix: &Array2<Complex<F>>,
) -> SparseEigenResult<F> {
    let n = matrix.nrows();
    let mut a = matrix.clone();
    let mut q_total = Array2::<Complex<F>>::eye(n);
    let max_iterations = 1000;
    let tolerance = F::from(1e-12).expect("Operation failed");
    for _iter in 0..max_iterations {
        let mut converged = true;
        for i in 0..n - 1 {
            if a[[i + 1, i]].norm() > tolerance {
                converged = false;
                break;
            }
        }
        if converged {
            break;
        }
        let (q, r) = householder_qr_complex(&a)?;
        a = r.dot(&q);
        q_total = q_total.dot(&q);
    }
    let eigenvals = (0..n).map(|i| a[[i, i]]).collect::<Array1<_>>();
    Ok((eigenvals, q_total))
}
#[allow(dead_code)]
fn householder_qr_complex<F: Float + NumAssign + Sum>(matrix: &Array2<Complex<F>>) -> QrResult<F> {
    let (m, n) = matrix.dim();
    let mut q = Array2::<Complex<F>>::eye(m);
    let mut r = matrix.clone();
    let min_dim = m.min(n);
    for k in 0..min_dim {
        let x = r.slice(s![k.., k]).to_owned();
        let (house_vec, tau) = householder_vector_complex(&x);
        apply_householder_left_complex(&mut r, &house_vec, tau, k);
        apply_householder_right_complex(&mut q, &house_vec, tau.conj(), k);
    }
    Ok((q, r))
}
#[allow(dead_code)]
fn householder_vector_complex<F: Float + NumAssign + Sum>(
    x: &Array1<Complex<F>>,
) -> (Array1<Complex<F>>, Complex<F>) {
    let n = x.len();
    if n == 0 {
        return (Array1::zeros(0), Complex::new(F::zero(), F::zero()));
    }
    let norm_x = x.iter().map(|z| z.norm_sqr()).sum::<F>().sqrt();
    if norm_x == F::zero() {
        return (Array1::zeros(n), Complex::new(F::zero(), F::zero()));
    }
    let mut v = x.clone();
    let sign = if x[0].re >= F::zero() {
        F::one()
    } else {
        -F::one()
    };
    v[0] += Complex::new(sign * norm_x, F::zero());
    let norm_v = v.iter().map(|z| z.norm_sqr()).sum::<F>().sqrt();
    if norm_v > F::zero() {
        v.mapv_inplace(|z| z / norm_v);
    }
    let tau = Complex::new(F::from(2.0).expect("Operation failed"), F::zero());
    (v, tau)
}
#[allow(dead_code)]
fn apply_householder_left_complex<F: Float + NumAssign>(
    matrix: &mut Array2<Complex<F>>,
    house_vec: &Array1<Complex<F>>,
    tau: Complex<F>,
    k: usize,
) {
    let (m, n) = matrix.dim();
    let house_len = house_vec.len();
    for j in k..n {
        let mut sum = Complex::new(F::zero(), F::zero());
        for i in 0..house_len {
            if k + i < m {
                sum += house_vec[i].conj() * matrix[[k + i, j]];
            }
        }
        for i in 0..house_len {
            if k + i < m {
                matrix[[k + i, j]] -= tau * house_vec[i] * sum;
            }
        }
    }
}
#[allow(dead_code)]
fn apply_householder_right_complex<F: Float + NumAssign>(
    matrix: &mut Array2<Complex<F>>,
    house_vec: &Array1<Complex<F>>,
    tau: Complex<F>,
    k: usize,
) {
    let (m, _n) = matrix.dim();
    let house_len = house_vec.len();
    for i in 0..m {
        let mut sum = Complex::new(F::zero(), F::zero());
        for j in 0..house_len {
            if k + j < matrix.ncols() {
                sum += matrix[[i, k + j]] * house_vec[j];
            }
        }
        for j in 0..house_len {
            if k + j < matrix.ncols() {
                matrix[[i, k + j]] -= sum * tau.conj() * house_vec[j].conj();
            }
        }
    }
}
#[allow(dead_code)]
fn select_closest_real_eigenvalues<F: Float>(
    eigenvals: &Array1<Complex<F>>,
    target: F,
    k: usize,
) -> Vec<usize> {
    let mut real_eigenvals: Vec<(usize, F)> = eigenvals
        .iter()
        .enumerate()
        .filter(|(_, z)| z.im.abs() < F::from(1e-10).expect("Operation failed"))
        .map(|(i, z)| (i, z.re))
        .collect();
    real_eigenvals.sort_by(|a, b| {
        let dist_a = (a.1 - target).abs();
        let dist_b = (b.1 - target).abs();
        dist_a.partial_cmp(&dist_b).expect("Operation failed")
    });
    real_eigenvals
        .into_iter()
        .take(k)
        .map(|(idx, _)| idx)
        .collect()
}
#[allow(dead_code)]
fn select_closest_complex_eigenvalues<F: Float>(
    eigenvals: &Array1<Complex<F>>,
    target: Complex<F>,
    k: usize,
) -> Vec<usize> {
    let mut eigenvals_with_dist: Vec<(usize, F)> = eigenvals
        .iter()
        .enumerate()
        .map(|(i, z)| {
            let diff = *z - target;
            (i, diff.norm())
        })
        .collect();
    eigenvals_with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("Operation failed"));
    eigenvals_with_dist
        .into_iter()
        .take(k)
        .map(|(idx, _)| idx)
        .collect()
}
/// Solve sparse generalized eigenvalue problem Ax = λBx using iterative methods.
///
/// This function solves the generalized eigenvalue problem for sparse matrices
/// using specialized algorithms that avoid forming dense factorizations.
///
/// # Arguments
///
/// * `a` - Sparse matrix A
/// * `b` - Sparse matrix B (should be positive definite for symmetric case)
/// * `k` - Number of eigenvalues to compute
/// * `which` - Which eigenvalues to find ("largest", "smallest", "target")
/// * `target` - Target value for "target" mode
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// * Tuple (eigenvalues, eigenvectors) with k generalized eigenvalues and eigenvectors
///
/// # Examples
///
/// ```rust,ignore
/// use scirs2_linalg::eigen::sparse::{eigs_gen, SparseMatrix};
///
/// // This is a placeholder example - actual implementation pending
/// // let (w, v) = eigs_gen(&sparse_a, &sparse_b, 4, "smallest", 0.0, 100, 1e-6).expect("Operation failed");
/// ```
///
/// # Note
///
/// This function implements the shift-invert Arnoldi algorithm for generalized sparse
/// eigenvalue problems. For `which = "smallest"`, the shift σ is set to 0 and the
/// inverse iteration `(A - σB)^{-1} B v` is applied; for `which = "largest"` a power
/// iteration variant is used; for `which = "target"` the user-supplied `target` value
/// is used as σ. The inner linear system is solved via matrix-free BiCGSTAB.
///
/// Eigenvalues of the original pencil are recovered as λ = σ + 1/θ where θ are the
/// Ritz values of the Hessenberg matrix built by the Arnoldi process.
#[allow(dead_code)]
pub fn eigs_gen<F, M1, M2>(
    a: &M1,
    b: &M2,
    k: usize,
    which: &str,
    target: F,
    max_iter: usize,
    tol: F,
) -> SparseEigenResult<F>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    M1: SparseMatrix<F>,
    M2: SparseMatrix<F>,
{
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(
            "eigs_gen: matrix A must be square".to_string(),
        ));
    }
    if b.nrows() != n || b.ncols() != n {
        return Err(LinalgError::ShapeError(
            "eigs_gen: matrix B must have same shape as A".to_string(),
        ));
    }
    if k == 0 {
        return Err(LinalgError::InvalidInputError(
            "eigs_gen: k must be at least 1".to_string(),
        ));
    }
    if k >= n {
        return Err(LinalgError::InvalidInputError(format!(
            "eigs_gen: k={k} must be less than matrix dimension n={n}"
        )));
    }
    let sigma: F = match which {
        "target" | "nearest" => target,
        _ => F::zero(),
    };
    let use_power_iter = matches!(which, "largest" | "LM");
    let m = (k + k.max(6)).min(n);
    let mut rng = scirs2_core::random::rng();
    let mut v0 = Array1::<F>::zeros(n);
    for i in 0..n {
        v0[i] = F::from(rng.random::<f64>()).unwrap_or(F::one());
    }
    let v0_norm = v0.iter().map(|x| (*x) * (*x)).sum::<F>().sqrt();
    if v0_norm < F::from(1e-15).unwrap_or(F::epsilon()) {
        return Err(LinalgError::InvalidInputError(
            "eigs_gen: random starting vector has near-zero norm".to_string(),
        ));
    }
    v0.mapv_inplace(|x| x / v0_norm);
    let mut v_basis: Vec<Array1<F>> = Vec::with_capacity(m + 1);
    v_basis.push(v0);
    let mut hmatrix = Array2::<F>::zeros((m + 1, m));
    let inner_tol =
        (tol * F::from(0.01).unwrap_or(F::epsilon())).max(F::from(1e-14).unwrap_or(F::epsilon()));
    let inner_max_iter = (200 * n).max(max_iter);
    let mut actual_m = 0usize;
    'arnoldi: for j in 0..m {
        actual_m = j + 1;
        let w = if use_power_iter {
            let mut w_tmp = Array1::<F>::zeros(n);
            a.matvec(&v_basis[j].view(), &mut w_tmp)?;
            w_tmp
        } else {
            let mut rhs = Array1::<F>::zeros(n);
            b.matvec(&v_basis[j].view(), &mut rhs)?;
            eigs_gen_bicgstab(a, b, sigma, &rhs, inner_tol, inner_max_iter)?
        };
        let mut w_ortho = w;
        for i in 0..=j {
            let h_ij = w_ortho
                .iter()
                .zip(v_basis[i].iter())
                .map(|(w_val, v_val)| (*w_val) * (*v_val))
                .sum::<F>();
            hmatrix[[i, j]] = h_ij;
            for l in 0..n {
                w_ortho[l] -= h_ij * v_basis[i][l];
            }
        }
        for i in 0..=j {
            let proj = w_ortho
                .iter()
                .zip(v_basis[i].iter())
                .map(|(w_val, v_val)| (*w_val) * (*v_val))
                .sum::<F>();
            hmatrix[[i, j]] += proj;
            for l in 0..n {
                w_ortho[l] -= proj * v_basis[i][l];
            }
        }
        let beta = w_ortho.iter().map(|x| (*x) * (*x)).sum::<F>().sqrt();
        if j + 1 < m {
            hmatrix[[j + 1, j]] = beta;
        }
        if beta < tol {
            break 'arnoldi;
        }
        let v_next = w_ortho.mapv(|x| x / beta);
        v_basis.push(v_next);
        if j >= k && j % 5 == 0 && check_arnoldi_convergence(&hmatrix, j + 1, k, tol) {
            break 'arnoldi;
        }
    }
    let h_small = hmatrix.slice(s![..actual_m, ..actual_m]).to_owned();
    let (ritz_vals_cx, ritz_vecs_cx) = solve_hessenberg_eigenproblem(&h_small)?;
    let mapped_vals: Array1<Complex<F>> = if use_power_iter {
        ritz_vals_cx.clone()
    } else {
        let sigma_cx = Complex::new(sigma, F::zero());
        ritz_vals_cx.mapv(|theta| {
            let norm_sq = theta.re * theta.re + theta.im * theta.im;
            if norm_sq < F::from(1e-30).unwrap_or(F::epsilon()) {
                Complex::new(F::from(1e30).unwrap_or(F::max_value()), F::zero())
            } else {
                sigma_cx + Complex::new(theta.re / norm_sq, -theta.im / norm_sq)
            }
        })
    };
    let selected_indices =
        select_gen_eigenvalues(&mapped_vals, which, Complex::new(target, F::zero()), k);
    let mut eigenvecs = Array2::<Complex<F>>::zeros((n, k));
    for (col, &ritz_idx) in selected_indices.iter().enumerate() {
        for row in 0..n {
            let mut val = Complex::new(F::zero(), F::zero());
            let basis_len = actual_m.min(v_basis.len());
            for l in 0..basis_len {
                val += ritz_vecs_cx[[l, ritz_idx]] * Complex::new(v_basis[l][row], F::zero());
            }
            eigenvecs[[row, col]] = val;
        }
    }
    let eigenvals: Array1<Complex<F>> = selected_indices
        .iter()
        .map(|&idx| mapped_vals[idx])
        .collect();
    Ok((eigenvals, eigenvecs))
}
/// Matrix-free BiCGSTAB solver for  (A - σ B) x = rhs.
///
/// The system matrix is never formed explicitly; only matrix-vector products
/// through the `SparseMatrix` trait are used.  Returns the approximate solution.
fn eigs_gen_bicgstab<F, M1, M2>(
    a: &M1,
    b: &M2,
    sigma: F,
    rhs: &Array1<F>,
    tol: F,
    max_iter: usize,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    M1: SparseMatrix<F>,
    M2: SparseMatrix<F>,
{
    let n = rhs.len();
    let op = |v: &Array1<F>| -> LinalgResult<Array1<F>> {
        let mut av = Array1::<F>::zeros(n);
        let mut bv = Array1::<F>::zeros(n);
        a.matvec(&v.view(), &mut av)?;
        b.matvec(&v.view(), &mut bv)?;
        let result: Array1<F> = av
            .iter()
            .zip(bv.iter())
            .map(|(&a_i, &b_i)| a_i - sigma * b_i)
            .collect();
        Ok(result)
    };
    let mut x = Array1::<F>::zeros(n);
    let mut r = rhs.clone();
    let r_shadow = r.clone();
    let bnorm = rhs.iter().map(|v| (*v) * (*v)).sum::<F>().sqrt();
    let abs_tol = tol * bnorm.max(F::from(1e-30).unwrap_or(F::epsilon()));
    let mut rho_prev = F::one();
    let mut alpha = F::one();
    let mut omega = F::one();
    let mut p = Array1::<F>::zeros(n);
    let mut v_vec = Array1::<F>::zeros(n);
    for _iter in 0..max_iter {
        let rho_curr = r_shadow
            .iter()
            .zip(r.iter())
            .map(|(rs, ri)| (*rs) * (*ri))
            .sum::<F>();
        if rho_curr.abs() < F::from(1e-300).unwrap_or(F::epsilon()) {
            break;
        }
        let beta = (rho_curr / rho_prev) * (alpha / omega);
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v_vec[i]);
        }
        v_vec = op(&p)?;
        let rs_dot_v = r_shadow
            .iter()
            .zip(v_vec.iter())
            .map(|(rs, vi)| (*rs) * (*vi))
            .sum::<F>();
        if rs_dot_v.abs() < F::from(1e-300).unwrap_or(F::epsilon()) {
            break;
        }
        alpha = rho_curr / rs_dot_v;
        let mut s = r.clone();
        for i in 0..n {
            s[i] -= alpha * v_vec[i];
        }
        let s_norm = s.iter().map(|x| (*x) * (*x)).sum::<F>().sqrt();
        if s_norm <= abs_tol {
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            break;
        }
        let t = op(&s)?;
        let tt = t.iter().map(|x| (*x) * (*x)).sum::<F>();
        if tt < F::from(1e-300).unwrap_or(F::epsilon()) {
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            break;
        }
        omega = t
            .iter()
            .zip(s.iter())
            .map(|(ti, si)| (*ti) * (*si))
            .sum::<F>()
            / tt;
        for i in 0..n {
            x[i] += alpha * p[i] + omega * s[i];
        }
        for i in 0..n {
            r[i] = s[i] - omega * t[i];
        }
        let r_norm = r.iter().map(|x| (*x) * (*x)).sum::<F>().sqrt();
        if r_norm <= abs_tol {
            break;
        }
        rho_prev = rho_curr;
    }
    Ok(x)
}
/// Select k eigenvalue indices from `eigenvals` based on the `which` criterion.
fn select_gen_eigenvalues<F: Float>(
    eigenvals: &Array1<Complex<F>>,
    which: &str,
    target: Complex<F>,
    k: usize,
) -> Vec<usize> {
    let n = eigenvals.len();
    let mut idx_and_score: Vec<(usize, F)> = (0..n)
        .map(|i| {
            let z = eigenvals[i];
            let score = match which {
                "largest" | "LM" => z.re,
                "smallest" | "SM" => -z.re,
                "target" | "nearest" => {
                    let diff = z - target;
                    -(diff.re * diff.re + diff.im * diff.im).sqrt()
                }
                _ => (z.re * z.re + z.im * z.im).sqrt(),
            };
            (i, score)
        })
        .collect();
    idx_and_score.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    idx_and_score
        .into_iter()
        .take(k)
        .map(|(idx, _)| idx)
        .collect()
}
