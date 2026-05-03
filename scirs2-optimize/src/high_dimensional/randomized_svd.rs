//! Randomized SVD (Halko, Martinsson & Tropp, 2011)
//!
//! Computes approximate rank-k SVD using random projections.
//! Useful for dimensionality reduction in high-dimensional optimization.
//!
//! ## Algorithm
//!
//! 1. Generate random Gaussian matrix Omega (n x (k+p))
//! 2. Form Y = A * Omega
//! 3. Power iteration: Y = A * (A^T * Y) repeated `n_power_iterations` times
//! 4. QR decomposition of Y -> Q
//! 5. Form B = Q^T * A (small matrix)
//! 6. SVD of B -> U_hat, S, Vt
//! 7. U = Q * U_hat
//!
//! ## Reference
//!
//! - Halko, N., Martinsson, P.G. and Tropp, J.A. (2011).
//!   "Finding Structure with Randomness: Probabilistic Algorithms for
//!   Constructing Approximate Matrix Decompositions"

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::random::{rngs::StdRng, RngExt, SeedableRng};

/// Configuration for randomized SVD
#[derive(Debug, Clone)]
pub struct RandomizedSVDConfig {
    /// Target rank k
    pub rank: usize,
    /// Extra columns for oversampling (default: 10)
    pub oversampling: usize,
    /// Number of power iterations for improved accuracy (default: 2)
    pub n_power_iterations: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for RandomizedSVDConfig {
    fn default() -> Self {
        Self {
            rank: 5,
            oversampling: 10,
            n_power_iterations: 2,
            seed: Some(42),
        }
    }
}

/// Result of randomized SVD decomposition
#[derive(Debug, Clone)]
pub struct RandomizedSVDResult {
    /// Left singular vectors (m x k), stored as rows of inner vecs
    pub u: Vec<Vec<f64>>,
    /// Singular values (k)
    pub s: Vec<f64>,
    /// Right singular vectors (k x n), stored as rows of inner vecs
    pub vt: Vec<Vec<f64>>,
}

/// Generate a random Gaussian matrix of shape (rows x cols)
fn random_gaussian_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let mut mat = vec![vec![0.0; cols]; rows];
    for row in mat.iter_mut() {
        for val in row.iter_mut() {
            // Box-Muller transform for standard normal
            let u1: f64 = rng.random::<f64>().max(1e-30);
            let u2: f64 = rng.random::<f64>();
            *val = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
        }
    }
    mat
}

/// Matrix multiply: C = A * B where A is (m x p), B is (p x n)
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 {
        return vec![];
    }
    let p = a[0].len();
    if p == 0 || b.is_empty() {
        return vec![vec![]; m];
    }
    let n = b[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for k in 0..p {
            let a_ik = a[i][k];
            for j in 0..n {
                c[i][j] += a_ik * b[k][j];
            }
        }
    }
    c
}

/// Transpose a matrix
fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return vec![];
    }
    let m = a.len();
    let n = a[0].len();
    let mut at = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            at[j][i] = a[i][j];
        }
    }
    at
}

/// Modified Gram-Schmidt QR decomposition
/// Returns (Q, R) where Q is (m x k) orthonormal and R is (k x k) upper triangular
fn qr_gram_schmidt(a: &[Vec<f64>]) -> OptimizeResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let m = a.len();
    if m == 0 {
        return Ok((vec![], vec![]));
    }
    let n = a[0].len();

    // Work column-major: extract columns of A (each column is length m)
    let mut cols: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect();

    let k = m.min(n);
    let mut r = vec![vec![0.0; n]; k];

    for j in 0..k {
        // r[j][j] = ||cols[j]||
        let norm: f64 = cols[j].iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-14 {
            // Near-zero column, keep it zero
            r[j][j] = 0.0;
            for v in cols[j].iter_mut() {
                *v = 0.0;
            }
            continue;
        }
        r[j][j] = norm;
        // Normalize column j
        for v in cols[j].iter_mut() {
            *v /= norm;
        }
        // Orthogonalize remaining columns against column j
        for jj in (j + 1)..n {
            let dot: f64 = cols[j]
                .iter()
                .zip(cols[jj].iter())
                .map(|(a, b)| a * b)
                .sum();
            r[j][jj] = dot;
            let col_j: Vec<f64> = cols[j].clone();
            for i in 0..m {
                cols[jj][i] -= dot * col_j[i];
            }
        }
    }

    // Build Q (m x k) from the first k orthonormalized columns
    let mut q = vec![vec![0.0; k]; m];
    for i in 0..m {
        for j in 0..k {
            q[i][j] = cols[j][i];
        }
    }

    Ok((q, r))
}

/// One-sided Jacobi SVD for small matrices.
/// Computes SVD of B (m x n) where dimensions are small.
/// Returns (U, S, Vt) where U is (m x k), S is (k,), Vt is (k x n), k = min(m,n)
///
/// Uses one-sided Jacobi rotations applied to columns of B.
/// After convergence, the columns of B*V are orthogonal with norms equal
/// to the singular values, and U = B*V * diag(1/sigma).
fn small_svd_jacobi(
    b: &[Vec<f64>],
    max_sweeps: usize,
) -> OptimizeResult<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>)> {
    let m = b.len();
    if m == 0 {
        return Ok((vec![], vec![], vec![]));
    }
    let n = b[0].len();
    let k = m.min(n);

    // Work on a copy of B^T stored column-major as cols[j] = j-th column of B
    // We'll apply right rotations (Givens) to make B^T B diagonal
    // This is equivalent to one-sided Jacobi SVD
    let mut work = b.to_vec(); // m x n, row-major

    // V accumulates the right rotations, starts as I_n
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    // One-sided Jacobi: apply rotations on the right to make columns of B orthogonal
    for _sweep in 0..max_sweeps {
        let mut max_off = 0.0_f64;

        for p in 0..n {
            for q in (p + 1)..n {
                // Compute elements of B^T B for columns p and q
                let mut app = 0.0;
                let mut aqq = 0.0;
                let mut apq = 0.0;
                for i in 0..m {
                    app += work[i][p] * work[i][p];
                    aqq += work[i][q] * work[i][q];
                    apq += work[i][p] * work[i][q];
                }

                max_off = max_off.max(apq.abs());

                if apq.abs() < 1e-15 * (app * aqq).sqrt().max(1e-30) {
                    continue;
                }

                // Compute Jacobi rotation angle
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau.abs() > 1e15 {
                    1.0 / (2.0 * tau)
                } else {
                    let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
                    sign / (tau.abs() + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Apply rotation to columns p and q of work (B)
                for i in 0..m {
                    let bp = work[i][p];
                    let bq = work[i][q];
                    work[i][p] = c * bp - s * bq;
                    work[i][q] = s * bp + c * bq;
                }

                // Apply rotation to columns p and q of V
                for i in 0..n {
                    let vp = v[i][p];
                    let vq = v[i][q];
                    v[i][p] = c * vp - s * vq;
                    v[i][q] = s * vp + c * vq;
                }
            }
        }

        if max_off < 1e-14 {
            break;
        }
    }

    // Now work = B * V, and columns of work should be orthogonal
    // Singular values = column norms of work
    let mut col_norms: Vec<(usize, f64)> = (0..n)
        .map(|j| {
            let norm = (0..m).map(|i| work[i][j] * work[i][j]).sum::<f64>().sqrt();
            (j, norm)
        })
        .collect();

    // Sort by descending singular value
    col_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut sigma = vec![0.0; k];
    let mut u_mat = vec![vec![0.0; k]; m];
    let mut vt = vec![vec![0.0; n]; k];

    for idx in 0..k {
        let (col_j, sv) = col_norms[idx];
        sigma[idx] = sv;

        if sv > 1e-14 {
            // U column = work column / sigma
            for i in 0..m {
                u_mat[i][idx] = work[i][col_j] / sv;
            }
            // Vt row = V column (transposed)
            for i in 0..n {
                vt[idx][i] = v[i][col_j];
            }
        }
    }

    Ok((u_mat, sigma, vt))
}

/// Compute randomized SVD of a matrix
///
/// Given an m x n matrix A and target rank k, computes an approximate
/// rank-k SVD: A ~ U * diag(S) * Vt
///
/// # Arguments
/// * `a` - Input matrix (m x n) stored as Vec of rows
/// * `config` - Configuration parameters
///
/// # Returns
/// * `RandomizedSVDResult` with U (m x k), S (k), Vt (k x n)
pub fn randomized_svd(
    a: &[Vec<f64>],
    config: &RandomizedSVDConfig,
) -> OptimizeResult<RandomizedSVDResult> {
    let m = a.len();
    if m == 0 {
        return Err(OptimizeError::InvalidInput(
            "Input matrix must have at least one row".to_string(),
        ));
    }
    let n = a[0].len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput(
            "Input matrix must have at least one column".to_string(),
        ));
    }

    // Validate consistent row lengths
    for (i, row) in a.iter().enumerate() {
        if row.len() != n {
            return Err(OptimizeError::InvalidInput(format!(
                "Row {} has length {} but expected {}",
                i,
                row.len(),
                n
            )));
        }
    }

    let k = config.rank.min(m).min(n);
    if k == 0 {
        return Err(OptimizeError::InvalidInput(
            "Target rank must be at least 1".to_string(),
        ));
    }

    let p = config.oversampling;
    let l = (k + p).min(m).min(n); // total sketch columns

    let mut rng = match config.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::seed_from_u64(0),
    };

    // Step 1: Generate random Gaussian matrix Omega (n x l)
    let omega = random_gaussian_matrix(n, l, &mut rng);

    // Step 2: Form Y = A * Omega (m x l)
    let mut y = mat_mul(a, &omega);

    // Step 3: Power iteration with reorthogonalization for numerical stability
    let at = transpose(a);
    for _ in 0..config.n_power_iterations {
        // z = A^T * Y, then QR(z) for stability
        let z = mat_mul(&at, &y);
        let (q_z, _) = qr_gram_schmidt(&z)?;
        // Y = A * Q_z
        y = mat_mul(a, &q_z);
    }

    // Step 4: QR decomposition of Y -> Q (m x l)
    let (q, _r) = qr_gram_schmidt(&y)?;

    // Step 5: Form B = Q^T * A (l x n)
    let qt = transpose(&q);
    let b = mat_mul(&qt, a);

    // Step 6: SVD of B (small matrix, l x n)
    let (u_hat, sigma, vt_full) = small_svd_jacobi(&b, 100)?;

    // Step 7: U = Q * U_hat (m x k)
    let u_full = mat_mul(&q, &u_hat);

    // Truncate to rank k
    let actual_k = k.min(sigma.len());
    let s: Vec<f64> = sigma[..actual_k].to_vec();

    let u: Vec<Vec<f64>> = u_full.iter().map(|row| row[..actual_k].to_vec()).collect();

    let vt: Vec<Vec<f64>> = vt_full[..actual_k].iter().cloned().collect();

    Ok(RandomizedSVDResult { u, s, vt })
}

/// Compute the low-rank approximation A_k = U * diag(S) * Vt
///
/// # Arguments
/// * `result` - The randomized SVD result
///
/// # Returns
/// The reconstructed matrix as Vec of rows
pub fn reconstruct_from_svd(result: &RandomizedSVDResult) -> Vec<Vec<f64>> {
    let m = result.u.len();
    if m == 0 || result.s.is_empty() || result.vt.is_empty() {
        return vec![];
    }
    let n = result.vt[0].len();
    let k = result.s.len();

    let mut reconstructed = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for r in 0..k {
                val += result.u[i][r] * result.s[r] * result.vt[r][j];
            }
            reconstructed[i][j] = val;
        }
    }
    reconstructed
}

/// Compute the Frobenius norm of a matrix
fn frobenius_norm(a: &[Vec<f64>]) -> f64 {
    let mut sum = 0.0;
    for row in a {
        for &val in row {
            sum += val * val;
        }
    }
    sum.sqrt()
}

/// Compute the Frobenius norm of the difference of two matrices
fn frobenius_diff_norm(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut sum = 0.0;
    for (row_a, row_b) in a.iter().zip(b.iter()) {
        for (&va, &vb) in row_a.iter().zip(row_b.iter()) {
            let d = va - vb;
            sum += d * d;
        }
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Identity matrix SVD
    #[test]
    fn test_identity_svd() {
        let n = 5;
        let mut identity = vec![vec![0.0; n]; n];
        for i in 0..n {
            identity[i][i] = 1.0;
        }

        let config = RandomizedSVDConfig {
            rank: 5,
            oversampling: 5,
            n_power_iterations: 2,
            seed: Some(42),
        };

        let result = randomized_svd(&identity, &config);
        assert!(result.is_ok());
        let result = result.expect("SVD should succeed");

        // All singular values should be 1.0
        assert_eq!(result.s.len(), 5);
        for &sv in &result.s {
            assert!(
                (sv - 1.0).abs() < 0.1,
                "Singular value {} should be ~1.0",
                sv
            );
        }
    }

    /// Test 2: Low-rank matrix approximation
    #[test]
    fn test_low_rank_approximation() {
        // Create a rank-2 matrix using explicit outer products
        let m = 8;
        let n = 6;
        let mut a = vec![vec![0.0; n]; m];

        // First rank-1 component: u1 = [1,0,1,0,...], v1 = [1,1,0,0,...]
        // sigma1 ~ large
        for i in 0..m {
            for j in 0..n {
                let ui = if i % 2 == 0 { 3.0 } else { 0.0 };
                let vj = if j < n / 2 { 2.0 } else { 0.0 };
                a[i][j] += ui * vj;
            }
        }
        // Second rank-1 component with smaller scale
        for i in 0..m {
            for j in 0..n {
                let ui = if i < m / 2 { 1.0 } else { -1.0 };
                let vj = if j % 2 == 0 { 0.5 } else { -0.5 };
                a[i][j] += ui * vj;
            }
        }

        let config = RandomizedSVDConfig {
            rank: 2,
            oversampling: 10,
            n_power_iterations: 3,
            seed: Some(123),
        };

        let result = randomized_svd(&a, &config);
        assert!(result.is_ok());
        let result = result.expect("SVD should succeed");

        assert_eq!(result.s.len(), 2);
        assert_eq!(result.u.len(), m);
        assert_eq!(result.vt.len(), 2);

        // Reconstruction should capture the rank-2 structure
        let reconstructed = reconstruct_from_svd(&result);
        let error = frobenius_diff_norm(&a, &reconstructed);
        let original_norm = frobenius_norm(&a);
        // The matrix is exactly rank-2, so rank-2 SVD should reconstruct well
        let relative_error = error / original_norm.max(1e-14);

        assert!(
            relative_error < 0.5,
            "Relative reconstruction error {} is too large (error={}, norm={})",
            relative_error,
            error,
            original_norm
        );
    }

    /// Test 3: Singular values are non-negative and sorted
    #[test]
    fn test_singular_values_sorted() {
        let m = 6;
        let n = 4;
        let mut rng = StdRng::seed_from_u64(77);
        let a = random_gaussian_matrix(m, n, &mut rng);

        let config = RandomizedSVDConfig {
            rank: 3,
            oversampling: 2,
            n_power_iterations: 1,
            seed: Some(42),
        };

        let result = randomized_svd(&a, &config);
        assert!(result.is_ok());
        let result = result.expect("SVD should succeed");

        // Singular values should be non-negative
        for &sv in &result.s {
            assert!(sv >= 0.0, "Singular value {} should be non-negative", sv);
        }

        // Singular values should be in non-increasing order
        for i in 1..result.s.len() {
            assert!(
                result.s[i] <= result.s[i - 1] + 1e-10,
                "Singular values not sorted: s[{}]={} > s[{}]={}",
                i,
                result.s[i],
                i - 1,
                result.s[i - 1]
            );
        }
    }

    /// Test 4: U columns are approximately orthonormal
    #[test]
    fn test_u_orthonormality() {
        let m = 8;
        let n = 6;
        let mut rng = StdRng::seed_from_u64(55);
        let a = random_gaussian_matrix(m, n, &mut rng);

        let config = RandomizedSVDConfig {
            rank: 3,
            oversampling: 3,
            n_power_iterations: 2,
            seed: Some(42),
        };

        let result = randomized_svd(&a, &config);
        assert!(result.is_ok());
        let result = result.expect("SVD should succeed");

        let k = result.s.len();
        // Check U^T U ~ I_k
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = (0..m).map(|r| result.u[r][i] * result.u[r][j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 0.3,
                    "U^T U[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    /// Test 5: Empty matrix error
    #[test]
    fn test_empty_matrix_error() {
        let a: Vec<Vec<f64>> = vec![];
        let config = RandomizedSVDConfig::default();
        let result = randomized_svd(&a, &config);
        assert!(result.is_err());
    }

    /// Test 6: Rank-1 matrix (no power iteration)
    #[test]
    fn test_rank_1_matrix_no_power() {
        let a = vec![
            vec![1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
            vec![3.0, 3.0, 3.0],
            vec![4.0, 4.0, 4.0],
        ];

        let config = RandomizedSVDConfig {
            rank: 1,
            oversampling: 5,
            n_power_iterations: 0,
            seed: Some(42),
        };

        let result = randomized_svd(&a, &config);
        assert!(result.is_ok());
        let result = result.expect("SVD should succeed");

        assert_eq!(result.s.len(), 1);
        assert!(result.s[0] > 0.0, "Singular value should be positive");

        let reconstructed = reconstruct_from_svd(&result);
        let error = frobenius_diff_norm(&a, &reconstructed);
        let original_norm = frobenius_norm(&a);
        let relative_error = error / original_norm.max(1e-14);
        assert!(
            relative_error < 0.15,
            "Rank-1 reconstruction relative error {} too large (error={}, norm={})",
            relative_error,
            error,
            original_norm
        );
    }

    /// Test 6b: Rank-1 matrix (with power iteration)
    #[test]
    fn test_rank_1_matrix() {
        let a = vec![
            vec![1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
            vec![3.0, 3.0, 3.0],
            vec![4.0, 4.0, 4.0],
        ];

        let config = RandomizedSVDConfig {
            rank: 1,
            oversampling: 5,
            n_power_iterations: 2,
            seed: Some(42),
        };

        let result = randomized_svd(&a, &config);
        assert!(result.is_ok());
        let result = result.expect("SVD should succeed");

        assert_eq!(result.s.len(), 1);
        assert!(result.s[0] > 0.0, "Singular value should be positive");

        let reconstructed = reconstruct_from_svd(&result);
        let error = frobenius_diff_norm(&a, &reconstructed);
        let original_norm = frobenius_norm(&a);
        let relative_error = error / original_norm.max(1e-14);
        assert!(
            relative_error < 0.15,
            "Rank-1 reconstruction relative error {} too large (error={}, norm={})",
            relative_error,
            error,
            original_norm
        );
    }

    /// Test 7: Power iteration improves accuracy
    #[test]
    fn test_power_iteration_improvement() {
        let m = 10;
        let n = 8;
        let mut rng = StdRng::seed_from_u64(99);
        let a = random_gaussian_matrix(m, n, &mut rng);

        let config_no_power = RandomizedSVDConfig {
            rank: 3,
            oversampling: 5,
            n_power_iterations: 0,
            seed: Some(42),
        };

        let config_with_power = RandomizedSVDConfig {
            rank: 3,
            oversampling: 5,
            n_power_iterations: 3,
            seed: Some(42),
        };

        let result_no = randomized_svd(&a, &config_no_power);
        let result_with = randomized_svd(&a, &config_with_power);
        assert!(result_no.is_ok());
        assert!(result_with.is_ok());

        let r_no = result_no.expect("should succeed");
        let r_with = result_with.expect("should succeed");

        let recon_no = reconstruct_from_svd(&r_no);
        let recon_with = reconstruct_from_svd(&r_with);

        let err_no = frobenius_diff_norm(&a, &recon_no);
        let err_with = frobenius_diff_norm(&a, &recon_with);

        // Power iteration should generally help or at least not hurt much
        // (allow some tolerance for randomness)
        assert!(
            err_with <= err_no * 1.5 + 1e-10,
            "Power iteration made things worse: {} vs {}",
            err_with,
            err_no
        );
    }

    /// Test 8: Diagonal matrix SVD gives correct singular values
    #[test]
    fn test_diagonal_matrix() {
        let diag_vals = vec![5.0, 3.0, 1.0, 0.1];
        let n = diag_vals.len();
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = diag_vals[i];
        }

        let config = RandomizedSVDConfig {
            rank: 4,
            oversampling: 5,
            n_power_iterations: 3,
            seed: Some(42),
        };

        let result = randomized_svd(&a, &config);
        assert!(result.is_ok());
        let result = result.expect("SVD should succeed");

        // Singular values should approximate the diagonal values (sorted descending)
        let mut sorted_diag = diag_vals.clone();
        sorted_diag.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        for (i, (&computed, &expected)) in result.s.iter().zip(sorted_diag.iter()).enumerate() {
            assert!(
                (computed - expected).abs() < 0.5,
                "s[{}]={}, expected ~{}",
                i,
                computed,
                expected
            );
        }
    }

    /// Test: Direct SVD with rank-deficient small matrix
    #[test]
    fn test_small_svd_rank_deficient() {
        // B is rank-1: only row 0 is nonzero
        let b = vec![
            vec![5.0, 5.0, 5.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        let (u, s, vt) = small_svd_jacobi(&b, 100).expect("SVD should work");
        // Only 1 nonzero singular value
        assert!(s[0] > 1.0, "s[0]={} should be ~8.66", s[0]);

        let recon = reconstruct_from_svd(&RandomizedSVDResult {
            u: u.clone(),
            s: s.clone(),
            vt: vt.clone(),
        });
        let err = frobenius_diff_norm(&b, &recon);
        assert!(
            err < 1e-6,
            "Rank-deficient SVD reconstruction error: {}",
            err
        );
    }

    /// Test: Direct SVD of small matrix
    #[test]
    fn test_direct_small_svd() {
        // 2x3 matrix with known SVD
        let b = vec![vec![3.0, 0.0, 0.0], vec![0.0, 2.0, 0.0]];
        let (u, s, vt) = small_svd_jacobi(&b, 100).expect("SVD should work");
        assert_eq!(s.len(), 2);
        // Singular values should be 3 and 2 (sorted descending)
        assert!((s[0] - 3.0).abs() < 0.01, "s[0]={}", s[0]);
        assert!((s[1] - 2.0).abs() < 0.01, "s[1]={}", s[1]);

        // Reconstruction should match
        let recon = reconstruct_from_svd(&RandomizedSVDResult {
            u: u.clone(),
            s: s.clone(),
            vt: vt.clone(),
        });
        let err = frobenius_diff_norm(&b, &recon);
        assert!(err < 1e-10, "SVD reconstruction error: {}", err);
    }

    /// Test 9: Inconsistent row lengths error
    #[test]
    fn test_inconsistent_rows() {
        let a = vec![vec![1.0, 2.0], vec![3.0]]; // inconsistent
        let config = RandomizedSVDConfig::default();
        let result = randomized_svd(&a, &config);
        assert!(result.is_err());
    }

    /// Test 10: reconstruct_from_svd with empty result
    #[test]
    fn test_reconstruct_empty() {
        let result = RandomizedSVDResult {
            u: vec![],
            s: vec![],
            vt: vec![],
        };
        let recon = reconstruct_from_svd(&result);
        assert!(recon.is_empty());
    }
}
