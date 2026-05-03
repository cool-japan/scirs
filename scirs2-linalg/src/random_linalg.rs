//! Randomized Numerical Linear Algebra – `Vec<Vec<f64>>` flat API.
//!
//! Provides high-level, easy-to-use functions for large-scale linear algebra
//! problems using randomized algorithms:
//!
//! - [`randomized_svd`] – Halko-Martinsson-Tropp randomized SVD
//! - [`randomized_range_finder`] – randomized orthonormal basis for range of A
//! - [`nystrom_approximation`] – Nyström approximation for PSD matrices
//! - [`sketched_lstsq`] – sketched least squares via random projection
//! - [`lsrn_solve`] – LSRN (Randomized Normal Equations) solver
//!
//! All heavy random number generation uses `scirs2_core::random`.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_linalg::random_linalg::{randomized_svd, RsvdResult};
//!
//! // Build a rank-2 matrix
//! let a = vec![
//!     vec![1.0_f64, 0.0, 0.0, 0.0],
//!     vec![0.0, 2.0, 0.0, 0.0],
//!     vec![0.0, 0.0, 0.5, 0.0],
//! ];
//! let result = randomized_svd(&a, 3, 4, 2, 2, 5, 42);
//! assert_eq!(result.s.len(), 2);
//! assert!(result.s[0] >= result.s[1]);
//! ```

use crate::error::{LinalgError, LinalgResult};

// ============================================================================
// Internal helpers
// ============================================================================

fn matmul(a: &[Vec<f64>], b: &[Vec<f64>], m: usize, k: usize, n: usize) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for l in 0..k {
            if a[i][l] == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
    c
}

/// A^T * B (A is m×k, B is m×n, result is k×n)
fn matmul_ata(a: &[Vec<f64>], b: &[Vec<f64>], m: usize, k: usize, n: usize) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0; n]; k];
    for l in 0..m {
        for i in 0..k {
            if a[l][i] == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i][j] += a[l][i] * b[l][j];
            }
        }
    }
    c
}

/// Transpose m×n → n×m
fn transpose(a: &[Vec<f64>], m: usize, n: usize) -> Vec<Vec<f64>> {
    let mut t = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            t[j][i] = a[i][j];
        }
    }
    t
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn vec_norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn frobenius_norm(a: &[Vec<f64>], m: usize, n: usize) -> f64 {
    let mut s = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            s += a[i][j] * a[i][j];
        }
    }
    s.sqrt()
}

/// QR decomposition via Gram-Schmidt (thin: A is m×k, returns Q m×k, R k×k).
fn qr_gram_schmidt(a: &[Vec<f64>], m: usize, k: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut q: Vec<Vec<f64>> = vec![vec![0.0; m]; k]; // stored as columns
    let mut r: Vec<Vec<f64>> = vec![vec![0.0; k]; k];

    for j in 0..k {
        // Start with column j of a
        let mut v: Vec<f64> = (0..m).map(|i| a[i][j]).collect();
        // Orthogonalize against previous columns
        for i in 0..j {
            let qi: Vec<f64> = q[i].clone();
            let proj = dot(&qi, &v);
            r[i][j] = proj;
            for l in 0..m {
                v[l] -= proj * qi[l];
            }
        }
        let n_v = vec_norm(&v);
        r[j][j] = n_v;
        if n_v > 1e-14 {
            for l in 0..m {
                q[j][l] = v[l] / n_v;
            }
        } else {
            // Column nearly zero — use canonical basis vector (pivoting)
            for l in 0..m {
                q[j][l] = 0.0;
            }
            if j < m {
                q[j][j % m] = 1.0;
            }
        }
    }
    // Convert q from row-of-columns to matrix form (m×k)
    let mut q_mat = vec![vec![0.0; k]; m];
    for j in 0..k {
        for i in 0..m {
            q_mat[i][j] = q[j][i];
        }
    }
    (q_mat, r)
}

/// Thin SVD of a small k×k (or small) matrix via Jacobi sweeps.
/// Returns (U k×k, S k, Vt k×k).
fn svd_small(a: &[Vec<f64>], rows: usize, cols: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    // For small matrices use Golub-Reinsch bidiagonalization + QR
    // Here we implement a simple Jacobi SVD for correctness
    let k = rows.min(cols);
    // Work on A^T A for right singular vectors, then recover left
    // Build A^T A (cols × cols)
    let at = transpose(a, rows, cols);
    let ata = matmul(&at, a, cols, rows, cols);
    // Jacobi eigen-decomposition of A^T A (symmetric)
    let (eigenvals, eigenvecs) = jacobi_eigen_sym(&ata, cols);
    // Sort descending
    let mut order: Vec<usize> = (0..cols).collect();
    order.sort_by(|&i, &j| {
        eigenvals[j]
            .partial_cmp(&eigenvals[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut s = vec![0.0f64; k];
    let mut vt = vec![vec![0.0f64; cols]; k];
    for (idx, &orig) in order.iter().enumerate().take(k) {
        let sigma = eigenvals[orig].max(0.0).sqrt();
        s[idx] = sigma;
        for j in 0..cols {
            vt[idx][j] = eigenvecs[j][orig];
        }
    }

    // Compute U = A * V * diag(1/sigma) for non-zero singular values
    let mut u = vec![vec![0.0f64; rows]; k];
    for idx in 0..k {
        if s[idx] > 1e-14 {
            // u_i = A * v_i / s_i
            for r in 0..rows {
                let mut sum = 0.0;
                for c in 0..cols {
                    sum += a[r][c] * vt[idx][c];
                }
                u[idx][r] = sum / s[idx];
            }
        } else {
            // Use Gram-Schmidt extension (just leave zero for now)
        }
    }
    // Transpose U so it's rows×k (u_mat[i][j] = ith row, jth singular vector component)
    let mut u_mat = vec![vec![0.0f64; k]; rows];
    for i in 0..k {
        for r in 0..rows {
            u_mat[r][i] = u[i][r];
        }
    }
    // Convert flat vt to 2D
    (
        u_mat,
        s,
        vt.iter()
            .flat_map(|r| r.iter().copied())
            .collect::<Vec<_>>()
            .chunks(cols)
            .map(|c| c.to_vec())
            .collect(),
    )
}

/// Symmetric Jacobi eigen-decomposition. Returns (eigenvalues, eigenvectors as columns).
fn jacobi_eigen_sym(a: &[Vec<f64>], n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut mat = a.to_vec();
    let mut vecs: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row
        })
        .collect();

    for _ in 0..100 {
        // Find off-diagonal element with largest absolute value
        let mut max_val = 0.0f64;
        let mut p = 0usize;
        let mut q = 1usize;
        for i in 0..n {
            for j in i + 1..n {
                if mat[i][j].abs() > max_val {
                    max_val = mat[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-12 {
            break;
        }
        // Jacobi rotation
        let theta = (mat[q][q] - mat[p][p]) / (2.0 * mat[p][q]);
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            1.0 / (theta - (1.0 + theta * theta).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let tau = s / (1.0 + c);

        // Update matrix
        let a_pp = mat[p][p];
        let a_qq = mat[q][q];
        let a_pq = mat[p][q];
        mat[p][p] = a_pp - t * a_pq;
        mat[q][q] = a_qq + t * a_pq;
        mat[p][q] = 0.0;
        mat[q][p] = 0.0;
        for r in 0..n {
            if r != p && r != q {
                let a_rp = mat[r][p];
                let a_rq = mat[r][q];
                mat[r][p] = a_rp - s * (a_rq + tau * a_rp);
                mat[p][r] = mat[r][p];
                mat[r][q] = a_rq + s * (a_rp - tau * a_rq);
                mat[q][r] = mat[r][q];
            }
        }
        // Update eigenvectors
        for r in 0..n {
            let v_rp = vecs[r][p];
            let v_rq = vecs[r][q];
            vecs[r][p] = v_rp - s * (v_rq + tau * v_rp);
            vecs[r][q] = v_rq + s * (v_rp - tau * v_rq);
        }
    }
    let eigenvals: Vec<f64> = (0..n).map(|i| mat[i][i]).collect();
    (eigenvals, vecs)
}

/// Solve A x = b (m×n, m≥n) via normal equations.
fn solve_lstsq_small(a: &[Vec<f64>], b: &[f64], m: usize, n: usize) -> LinalgResult<Vec<f64>> {
    // Normal equations: A^T A x = A^T b
    let at = transpose(a, m, n);
    let ata = matmul(&at, a, n, m, n);
    let atb: Vec<f64> = (0..n).map(|i| dot(&at[i], b)).collect();
    solve_linear_square(&ata, &atb, n)
}

/// Gaussian elimination with partial pivoting (square system).
fn solve_linear_square(a: &[Vec<f64>], b: &[f64], n: usize) -> LinalgResult<Vec<f64>> {
    let mut mat = a.to_vec();
    let mut rhs = b.to_vec();
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = mat[col][col].abs();
        for row in col + 1..n {
            if mat[row][col].abs() > max_val {
                max_val = mat[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-300 {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular".to_string(),
            ));
        }
        mat.swap(col, max_row);
        rhs.swap(col, max_row);
        let pivot = mat[col][col];
        for row in col + 1..n {
            let factor = mat[row][col] / pivot;
            rhs[row] -= factor * rhs[col];
            for j in col..n {
                let v = mat[col][j];
                mat[row][j] -= factor * v;
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for j in i + 1..n {
            s -= mat[i][j] * x[j];
        }
        if mat[i][i].abs() < 1e-300 {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular".to_string(),
            ));
        }
        x[i] = s / mat[i][i];
    }
    Ok(x)
}

/// Generate m×k standard Gaussian random matrix using seed.
fn random_gaussian_matrix(m: usize, k: usize, seed: u64) -> Vec<Vec<f64>> {
    use scirs2_core::random::prelude::*;
    use scirs2_core::random::rngs::SmallRng;
    use scirs2_core::random::{Distribution, Normal};

    let normal = Normal::new(0.0f64, 1.0)
        .unwrap_or_else(|_| Normal::new(0.0, 1.0 - f64::EPSILON).expect("normal distribution"));
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..m)
        .map(|_| (0..k).map(|_| normal.sample(&mut rng)).collect())
        .collect()
}

// ============================================================================
// Public API
// ============================================================================

/// Result of a randomized SVD computation.
#[derive(Debug, Clone)]
pub struct RsvdResult {
    /// Left singular vectors (m × rank).
    pub u: Vec<Vec<f64>>,
    /// Singular values in descending order (rank,).
    pub s: Vec<f64>,
    /// Right singular vectors transposed (rank × n).
    pub vt: Vec<Vec<f64>>,
}

/// Randomized SVD using the Halko-Martinsson-Tropp (2011) algorithm.
///
/// Computes a rank-`k` approximation `A ≈ U * diag(S) * Vt` using random
/// projections and power iteration for accuracy.
///
/// # Arguments
///
/// * `a` - Input matrix (m × n)
/// * `m`, `n` - Dimensions
/// * `k` - Target rank
/// * `n_power_iter` - Number of power iterations (1–4; more = more accurate, slower)
/// * `n_oversampling` - Oversampling parameter (typically 10)
/// * `seed` - Random seed
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::random_linalg::randomized_svd;
///
/// let a = vec![
///     vec![3.0_f64, 0.0],
///     vec![0.0, 2.0],
///     vec![0.0, 0.0],
/// ];
/// let r = randomized_svd(&a, 3, 2, 2, 2, 5, 0);
/// assert_eq!(r.s.len(), 2);
/// assert!(r.s[0] >= r.s[1]);
/// ```
pub fn randomized_svd(
    a: &[Vec<f64>],
    m: usize,
    n: usize,
    k: usize,
    n_power_iter: usize,
    n_oversampling: usize,
    seed: u64,
) -> RsvdResult {
    let l = (k + n_oversampling).min(m.min(n));

    // Step 1: Find randomized range
    let q = randomized_range_finder(a, m, n, l, n_power_iter, seed);
    // q is m × l

    // Step 2: Project A into the range: B = Q^T A  (l × n)
    let qt = transpose(&q, m, l);
    let b = matmul(&qt, a, l, m, n);

    // Step 3: SVD of small matrix B (l × n)
    let (ub, sb, vt) = svd_small(&b, l, n);
    // ub is l × l, sb is min(l,n), vt is min(l,n) × n

    let rank = k.min(sb.len());

    // Step 4: U = Q * Ub[:, :rank]
    let mut u = vec![vec![0.0; rank]; m];
    for i in 0..m {
        for j in 0..rank {
            for t in 0..l {
                u[i][j] += q[i][t] * ub[t][j];
            }
        }
    }

    RsvdResult {
        u,
        s: sb[..rank].to_vec(),
        vt: vt[..rank].to_vec(),
    }
}

/// Randomized range finder: returns an orthonormal basis Q (m × k) for the range of A.
///
/// Uses random Gaussian projections + power iteration to improve accuracy.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::random_linalg::randomized_range_finder;
///
/// let a = vec![vec![1.0_f64, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
/// let q = randomized_range_finder(&a, 3, 2, 2, 1, 42);
/// assert_eq!(q.len(), 3);
/// assert_eq!(q[0].len(), 2);
/// ```
pub fn randomized_range_finder(
    a: &[Vec<f64>],
    m: usize,
    n: usize,
    k: usize,
    n_power_iter: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let l = k.min(m.min(n));

    // Random Gaussian test matrix (n × l)
    let omega = random_gaussian_matrix(n, l, seed);

    // Initial sketch Y = A * Omega (m × l)
    let mut y = matmul(a, &omega, m, n, l);

    // Power iteration: Y = (A A^T)^q A Omega
    let at = transpose(a, m, n);
    for _ in 0..n_power_iter {
        // Y = A^T Y  (n × l)
        let aty = matmul(&at, &y, n, m, l);
        // Y = A * aty (m × l)
        y = matmul(a, &aty, m, n, l);
        // Re-orthogonalize to prevent numerical issues
        let (q, _r) = qr_gram_schmidt(&y, m, l);
        y = q;
    }

    // QR decomposition to get orthonormal basis
    let (q, _r) = qr_gram_schmidt(&y, m, l);
    q
}

/// Nyström approximation for a symmetric positive semi-definite (PSD) matrix.
///
/// Computes an approximate factorization `K ≈ U * diag(lambda) * V^T` of rank `k`.
///
/// # Arguments
///
/// * `kernel_mat` - n×n PSD matrix
/// * `n` - dimension
/// * `k` - rank
/// * `seed` - random seed
///
/// # Returns
///
/// `(U, V, lambda)` where `K ≈ U * diag(lambda) * V^T` (U, V are n×k).
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::random_linalg::nystrom_approximation;
///
/// let k = vec![
///     vec![4.0_f64, 2.0, 0.0],
///     vec![2.0, 3.0, 1.0],
///     vec![0.0, 1.0, 2.0],
/// ];
/// let (u, v, lam) = nystrom_approximation(&k, 3, 2, 42);
/// assert_eq!(u.len(), 3); // n rows
/// assert_eq!(u[0].len(), 2); // k cols
/// ```
pub fn nystrom_approximation(
    kernel_mat: &[Vec<f64>],
    n: usize,
    k: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
    use scirs2_core::random::prelude::*;
    use scirs2_core::random::rngs::SmallRng;

    let l = k.min(n);
    let mut rng = SmallRng::seed_from_u64(seed);

    // Sample l column indices uniformly at random (without replacement)
    let mut indices: Vec<usize> = (0..n).collect();
    // Partial Fisher-Yates shuffle for l elements
    for i in 0..l {
        let j = i + (rng.next_u64() as usize % (n - i));
        indices.swap(i, j);
    }
    let col_indices: Vec<usize> = indices[..l].to_vec();

    // Build C = K[:, col_indices] (n × l)
    let c: Vec<Vec<f64>> = (0..n)
        .map(|i| col_indices.iter().map(|&j| kernel_mat[i][j]).collect())
        .collect();

    // Build W = K[col_indices, col_indices] (l × l)
    let w: Vec<Vec<f64>> = col_indices
        .iter()
        .map(|&i| col_indices.iter().map(|&j| kernel_mat[i][j]).collect())
        .collect();

    // Eigen-decompose W (symmetric)
    let (eigenvals_w, eigenvecs_w) = jacobi_eigen_sym(&w, l);
    let mut order: Vec<usize> = (0..l).collect();
    order.sort_by(|&i, &j| {
        eigenvals_w[j]
            .partial_cmp(&eigenvals_w[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // U = C * V_w * diag(1/sqrt(lambda_w)) (Nyström formula)
    let k_rank = k.min(l);
    let mut u = vec![vec![0.0; k_rank]; n];
    let mut lambda = vec![0.0; k_rank];

    for (idx, &orig) in order.iter().enumerate().take(k_rank) {
        let lam_val = eigenvals_w[orig].max(0.0);
        lambda[idx] = lam_val;
        let sqrt_lam = lam_val.max(1e-300).sqrt();
        for r in 0..n {
            let mut sum = 0.0;
            for s in 0..l {
                sum += c[r][s] * eigenvecs_w[s][orig];
            }
            u[r][idx] = sum / sqrt_lam;
        }
    }

    // V = U in the PSD case (K ≈ U * diag(lambda) * U^T)
    let v = u.clone();
    (u, v, lambda)
}

/// Sketched least squares: solve `min ||Ax - b||` approximately using a random projection.
///
/// Builds a sketch `S*A` of size `sketch_size × n` and solves `(SA)x = Sb`.
/// This is fast when `m >> n` and `sketch_size` is large enough (typically 2n–4n).
///
/// # Arguments
///
/// * `a` - m × n coefficient matrix (m >> n)
/// * `b` - right-hand side of length m
/// * `m`, `n` - dimensions
/// * `sketch_size` - number of rows in the sketch (>= n)
/// * `seed` - random seed
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::random_linalg::sketched_lstsq;
///
/// // Overdetermined system: y = 2x + noise
/// let a: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
/// let b: Vec<f64> = (0..10).map(|i| 2.0 * i as f64).collect();
/// let x = sketched_lstsq(&a, &b, 10, 1, 5, 42);
/// assert!((x[0] - 2.0).abs() < 0.5);
/// ```
pub fn sketched_lstsq(
    a: &[Vec<f64>],
    b: &[f64],
    m: usize,
    n: usize,
    sketch_size: usize,
    seed: u64,
) -> Vec<f64> {
    let s = sketch_size.max(n);

    // Build Gaussian sketch matrix S (s × m)
    let sketch = random_gaussian_matrix(s, m, seed);

    // SA (s × n)
    let sa = matmul(&sketch, a, s, m, n);

    // Sb (s,)
    let sb: Vec<f64> = (0..s).map(|i| dot(&sketch[i], b)).collect();

    // Solve normal equations of (SA)x = Sb
    solve_lstsq_small(&sa, &sb, s, n).unwrap_or_else(|_| vec![0.0; n])
}

/// LSRN (Least Squares via Random Normal equations) solver.
///
/// Randomized preconditioned iterative solver for overdetermined systems
/// `min ||Ax - b||_2` when `m >= n`. Uses a random sketch to build a
/// preconditioner `N = A*Omega` and then runs LSQR-style iterations.
///
/// # Arguments
///
/// * `a` - m × n matrix (m >= n)
/// * `b` - right-hand side (length m)
/// * `m`, `n` - dimensions
/// * `oversampling` - sketch rows = n + oversampling (typically 5–20)
/// * `seed` - random seed
/// * `max_iter` - maximum CG iterations
/// * `tol` - convergence tolerance
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::random_linalg::lsrn_solve;
///
/// // Small well-conditioned system
/// let a = vec![
///     vec![1.0_f64, 0.0],
///     vec![0.0, 1.0],
///     vec![1.0, 1.0],
/// ];
/// let b = vec![1.0_f64, 2.0, 3.5];
/// let x = lsrn_solve(&a, &b, 3, 2, 5, 42, 100, 1e-8);
/// // Least-squares solution approximately [1, 2]
/// assert!((x[0] - 1.0).abs() < 0.5);
/// assert!((x[1] - 2.0).abs() < 0.5);
/// ```
pub fn lsrn_solve(
    a: &[Vec<f64>],
    b: &[f64],
    m: usize,
    n: usize,
    oversampling: usize,
    seed: u64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let s = (n + oversampling).min(m);

    // Build sketch: Omega is n × s Gaussian, N = A * Omega^T won't work for LSRN
    // Correct LSRN: build S is s × m Gaussian, SA is s × n; SVD(SA) gives preconditioner
    let sketch = random_gaussian_matrix(s, m, seed);
    let sa = matmul(&sketch, a, s, m, n);

    // SVD of SA to get right preconditioner N (sa = U Sigma V^T, N = V)
    let (_u_sa, sigma, vt_sa) = svd_small(&sa, s, n);

    let rank = sigma.iter().filter(|&&sv| sv > 1e-12).count().max(1);
    let vt_r = &vt_sa[..rank]; // rank × n

    // Preconditioned normal equations: solve min ||A (V R^{-1}) y - b||
    // where x = V R^{-1} y, and R = diag(sigma[:rank])

    // Compute preconditioned matrix: A_p = A * V^T[:rank]^T  (m × rank)
    let vr: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..rank).map(|k| vt_r[k][j]).collect())
        .collect();
    let ap = matmul(a, &vr, m, n, rank);

    // LSQR on A_p * y = b
    let y = lsqr_solve(&ap, b, m, rank, max_iter, tol);

    // Recover x = Vr * y
    // The preconditioner Vr contains the right singular vectors of the sketch SA.
    // We solved min ||A*Vr*y - b|| for y, so x = Vr * y (no sigma scaling needed).
    let mut x = vec![0.0; n];
    for j in 0..rank {
        for i in 0..n {
            x[i] += vr[i][j] * y[j];
        }
    }
    x
}

/// LSQR algorithm for min ||Ax - b|| (m × n system).
fn lsqr_solve(
    a: &[Vec<f64>],
    b: &[f64],
    m: usize,
    n: usize,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let at = transpose(a, m, n);

    let mut x = vec![0.0f64; n];
    let mut r: Vec<f64> = b.to_vec();
    // r = b - A x = b (since x = 0)

    // u = r / ||r||
    let mut beta = vec_norm(&r);
    if beta < 1e-300 {
        return x;
    }
    let mut u: Vec<f64> = r.iter().map(|&v| v / beta).collect();

    // v = A^T u
    let mut atv: Vec<f64> = (0..n).map(|j| dot(&at[j], &u)).collect();
    let mut alpha = vec_norm(&atv);
    let mut v: Vec<f64> = if alpha > 1e-300 {
        atv.iter().map(|&x| x / alpha).collect()
    } else {
        return x;
    };

    let mut w = v.clone();
    let mut phi_bar = beta;
    let mut rho_bar = alpha;

    for _ in 0..max_iter {
        // Bidiagonalization
        let av: Vec<f64> = (0..m).map(|i| dot(&a[i], &v)).collect();
        beta = {
            let mut u_new: Vec<f64> = av
                .iter()
                .enumerate()
                .map(|(i, &v)| v - alpha * u[i])
                .collect();
            let b_new = vec_norm(&u_new);
            if b_new > 1e-300 {
                for x in u_new.iter_mut() {
                    *x /= b_new;
                }
                u = u_new;
            }
            b_new
        };

        atv = (0..n).map(|j| dot(&at[j], &u)).collect();
        alpha = {
            let mut v_new: Vec<f64> = atv
                .iter()
                .enumerate()
                .map(|(i, &x)| x - beta * v[i])
                .collect();
            let a_new = vec_norm(&v_new);
            if a_new > 1e-300 {
                for x in v_new.iter_mut() {
                    *x /= a_new;
                }
                v = v_new;
            }
            a_new
        };

        // Plane rotation
        let rho = (rho_bar * rho_bar + beta * beta).sqrt();
        let c = rho_bar / rho;
        let s = beta / rho;
        let theta = s * alpha;
        rho_bar = -c * alpha;
        let phi = c * phi_bar;
        phi_bar *= s;

        // Update x and w
        let phi_over_rho = if rho > 1e-300 { phi / rho } else { 0.0 };
        for i in 0..n {
            x[i] += phi_over_rho * w[i];
        }
        let theta_over_rho = if rho > 1e-300 { theta / rho } else { 0.0 };
        for i in 0..n {
            w[i] = v[i] - theta_over_rho * w[i];
        }

        if phi_bar.abs() < tol {
            break;
        }
    }
    x
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_error(a: &[Vec<f64>], b: &[Vec<f64>], m: usize, n: usize) -> f64 {
        let diff: Vec<Vec<f64>> = a
            .iter()
            .enumerate()
            .map(|(i, row)| row.iter().enumerate().map(|(j, &v)| v - b[i][j]).collect())
            .collect();
        let fn_diff = frobenius_norm(&diff, m, n);
        let fn_b = frobenius_norm(b, m, n);
        if fn_b > 1e-300 {
            fn_diff / fn_b
        } else {
            fn_diff
        }
    }

    #[test]
    fn test_randomized_range_finder_rank() {
        // Rank-2 matrix: A = u v^T + u2 v2^T
        let a = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 4.0, 6.0, 8.0],
            vec![0.5, 1.0, 1.5, 2.0],
        ];
        let q = randomized_range_finder(&a, 3, 4, 2, 2, 42);
        assert_eq!(q.len(), 3);
        assert_eq!(q[0].len(), 2);
        // Q should be orthonormal
        for j in 0..2 {
            let col: Vec<f64> = (0..3).map(|i| q[i][j]).collect();
            let n = vec_norm(&col);
            assert!((n - 1.0).abs() < 1e-8, "column {} norm = {}", j, n);
        }
    }

    #[test]
    fn test_randomized_svd_diagonal() {
        // Diagonal matrix with known singular values
        let a = vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let r = randomized_svd(&a, 3, 3, 3, 2, 5, 0);
        assert_eq!(r.s.len(), 3);
        assert!(r.s[0] >= r.s[1]);
        assert!(r.s[1] >= r.s[2]);
        // Check that singular values are close to 3, 2, 1
        let mut sv = r.s.clone();
        sv.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
        assert!((sv[0] - 3.0).abs() < 0.5, "sv[0]={}", sv[0]);
        assert!((sv[1] - 2.0).abs() < 0.5, "sv[1]={}", sv[1]);
    }

    #[test]
    fn test_nystrom_approximation() {
        // 3×3 PSD matrix
        let k = vec![
            vec![4.0, 2.0, 0.0],
            vec![2.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let (u, v, lam) = nystrom_approximation(&k, 3, 2, 42);
        assert_eq!(u.len(), 3);
        assert_eq!(u[0].len(), 2);
        assert_eq!(v.len(), 3);
        assert_eq!(lam.len(), 2);
        // Eigenvalues should be positive
        for &l in &lam {
            assert!(l >= 0.0);
        }
    }

    #[test]
    fn test_sketched_lstsq_simple() {
        // y = 2x: a = [[0],[1],[2],...], b = [0,2,4,...]
        let a: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let b: Vec<f64> = (0..20).map(|i| 2.0 * i as f64).collect();
        let x = sketched_lstsq(&a, &b, 20, 1, 40, 42);
        assert!((x[0] - 2.0).abs() < 0.2, "x[0]={}", x[0]);
    }

    #[test]
    fn test_lsrn_solve_simple() {
        // Overdetermined: min ||[I; I] x - [1; 2]||, solution x = 1.5
        let a = vec![
            vec![1.0_f64, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let b = vec![1.0_f64, 1.0, 2.0, 3.0];
        let x = lsrn_solve(&a, &b, 4, 2, 5, 42, 50, 1e-8);
        // Normal solution: x = [1.5, 2.0]
        assert!((x[0] - 1.5).abs() < 0.5, "x[0]={}", x[0]);
        assert!((x[1] - 2.0).abs() < 0.5, "x[1]={}", x[1]);
    }
}
